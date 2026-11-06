"""Phase 2, step 1: fill MDS 2.0 items on MIMIC-IV with the per-admission graph, and measure cost and delay per arm.

Rules set before any run (docs/EXPERIMENT_PLAN.md). Data: the rebuilt 500-admission proxy sample; this step uses the first N_ADM
(note keys x000 onward, already in random order). Entity layer = model NER + verifier (chosen in E1: D1 held, D2 failed).
Arms (all from the same calls, so they differ only in which graph edges count):
  notes        note mentions without the verifier
  notes+v      note mentions kept by the verifier
  notes+v+t    notes+v with time updating (a later current mention of the same concept supersedes earlier ones)
  records      record edges only
  graph        notes+v plus records (the full admission graph)
Reported per arm: items filled per admission by section, cost per admission (dollars, tokens), delay per admission (the admission's
calls run in parallel: the slowest extraction call + the slowest verifier call + the slowest membership call for strings not yet
cached, i.e. a cold start), and agreement between note-based arms and records per item (both yes, note only, records only).
Agreement is not accuracy: records are incomplete and some share sources with the note. Outputs hold counts only; nothing that
identifies an admission and no text leaves /home/user/mimic.
"""
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import admission_graph as A  # noqa: E402
from sweetspot.llm import LLM, PRICES, DEFAULT_MODEL  # noqa: E402

PRIVATE = Path("/home/user/mimic/pilot_private")
WORK = Path("/home/user/mimic/sweetspot")
OUT = ROOT / "results/sweetspot/p2_mimic_graph.json"
N_ADM = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def usd(rows):
    pin, pout = PRICES[DEFAULT_MODEL]
    return sum(r["input_tokens"] * pin / 1e6 + r["output_tokens"] * pout / 1e6 for r in rows if r)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    sents = pd.read_parquet(PRIVATE / "proxy_sentences.parquet")
    ids = pd.read_parquet(PRIVATE / "proxy_id_map.parquet").astype(str)
    keys = sorted(sents["note_key"].unique())[:N_ADM]
    hadm = dict(zip(ids["note_key"], ids["hadm_id"]))
    rec_path = WORK / "records_proxy.pkl"
    if rec_path.exists():
        records, discharge = pickle.loads(rec_path.read_bytes())
    else:
        records, discharge = A.load_records(ids["hadm_id"].tolist())
        rec_path.write_bytes(pickle.dumps((records, discharge)))

    # 1. mentions, one call per chunk
    chunk_of = {}
    reqs = []
    for k in keys:
        rows = sents[sents["note_key"] == k].sort_values("sent_id")
        triples = list(zip(rows["sent_id"], rows["section"], rows["text"]))
        for c in A.chunks(triples):
            chunk_of[len(reqs)] = (k, c)
            reqs.append(A.mention_request(c))
    ext = LLM("p2_mentions")
    replies = ext.run(reqs)
    raw = defaultdict(list)
    per_adm_calls = defaultdict(lambda: {"extract": [], "verify": [], "member": []})
    for n, r in enumerate(replies):
        k, c = chunk_of[n]
        per_adm_calls[k]["extract"].append(r)
        if r:
            valid = {sid for sid, _, _ in c}
            raw[(k, n)] = [m for m in r["parsed"]["mentions"] if m["sentence"] in valid and m["text"].strip()]
    # 2. verifier, one call per chunk with mentions
    vreq, vkey = [], []
    for (k, n), ms in raw.items():
        if ms:
            vreq.append(A.verify_request(chunk_of[n][1], ms))
            vkey.append((k, n))
    ver = LLM("p2_verify")
    vrep = ver.run(vreq)
    kept = {}
    for (k, n), r in zip(vkey, vrep):
        per_adm_calls[k]["verify"].append(r)
        kept[(k, n)] = A.apply_verdicts(raw[(k, n)], r["parsed"] if r else None)
    mentions = {"notes": defaultdict(list), "notes+v": defaultdict(list)}
    for (k, n), ms in raw.items():
        mentions["notes"][k] += ms
        mentions["notes+v"][k] += kept.get((k, n), ms)
    # 3. membership for every distinct concept string (notes and records)
    concepts = set()
    for arm in mentions:
        for ms in mentions[arm].values():
            concepts |= {(m["type"], A.norm(m["text"])) for m in ms}
    for k in keys:
        concepts |= {(r["type"], A.norm(r["text"])) for r in records.get(hadm[k], [])}
    mem = LLM("p2_membership")
    mreqs = A.member_requests(sorted(concepts))
    mrep = mem.run([q for q, _, _ in mreqs])
    valid_codes = {i["code"] for i in A.ITEMS["items"]}
    membership, member_row = {}, {}
    for (q, t, batch), r in zip(mreqs, mrep):
        for n, text in enumerate(batch):
            member_row[(t, text)] = r
        if not r:
            continue
        for a in r["parsed"]["answers"]:
            if 0 <= a["concept"] < len(batch):
                membership[(t, batch[a["concept"]])] = [c for c in a["items"] if c in valid_codes]
    # 4. graphs per arm
    arms = {}
    for k in keys:
        h = hadm[k]
        recs = records.get(h, [])
        d = discharge.get(h)
        d = d.to_pydatetime() if hasattr(d, "to_pydatetime") else d
        arms.setdefault("notes", {})[k] = A.build_graph(h, mentions["notes"][k], [], membership, d)
        arms.setdefault("notes+v", {})[k] = A.build_graph(h, mentions["notes+v"][k], [], membership, d)
        arms.setdefault("notes+v+t", {})[k] = A.build_graph(h, mentions["notes+v"][k], [], membership, d, time_update=True)
        arms.setdefault("records", {})[k] = A.build_graph(h, [], recs, membership, d)
        arms.setdefault("graph", {})[k] = A.build_graph(h, mentions["notes+v"][k], recs, membership, d)
        own = {(m["type"], A.norm(m["text"])) for m in mentions["notes+v"][k]} | {(r["type"], A.norm(r["text"])) for r in recs}
        per_adm_calls[k]["member"] = list({id(member_row[c]): member_row[c] for c in own if member_row.get(c)}.values())
    (WORK / "p2_graphs.pkl").write_bytes(pickle.dumps({"arms": arms, "mentions": mentions, "membership": membership}))

    codes = [i["code"] for i in A.ITEMS["items"]]
    section = {i["code"]: i["section"] for i in A.ITEMS["items"]}
    summary = {}
    for arm, g in arms.items():
        filled = Counter(c for k in keys for c in g[k])
        per = [len(g[k]) for k in keys]
        summary[arm] = {"items_per_admission_mean": round(sum(per) / len(per), 2),
                        "by_section_mean": {s: round(sum(sum(1 for c in g[k] if section[c] == s) for k in keys) / len(keys), 2) for s in "IOP"},
                        "prevalence": {c: round(filled[c] / len(keys), 3) for c in codes if filled[c]}}
    agree = {}
    for arm in ("notes+v", "notes+v+t"):
        agree[arm] = {}
        for c in codes:
            both = sum(c in arms[arm][k] and c in arms["records"][k] for k in keys)
            note_only = sum(c in arms[arm][k] and c not in arms["records"][k] for k in keys)
            rec_only = sum(c not in arms[arm][k] and c in arms["records"][k] for k in keys)
            if both + note_only + rec_only:
                agree[arm][c] = {"both": both, "note_only": note_only, "records_only": rec_only}
    cost_per = []
    delay = []
    for k in keys:
        calls = per_adm_calls[k]
        cost_per.append({"extract": usd(calls["extract"]), "verify": usd(calls["verify"]),
                         "member_cold": usd(calls["member"]), "calls": len(calls["extract"]) + len(calls["verify"])})
        mx = lambda rows: max([r["seconds"] for r in rows if r] or [0])
        delay.append({"notes": mx(calls["extract"]), "notes+v": mx(calls["extract"]) + mx(calls["verify"]),
                      "graph_cold": mx(calls["extract"]) + mx(calls["verify"]) + mx(calls["member"])})
    mean = lambda xs: round(sum(xs) / len(xs), 4)
    q = lambda xs, p: round(sorted(xs)[int(p * (len(xs) - 1))], 2)
    total_member = usd(mrep)
    out = {"admissions": len(keys), "chunk_sentences": A.CHUNK, "mentions_per_admission": {
               a: round(sum(len(v) for v in mentions[a].values()) / len(keys), 1) for a in mentions},
           "distinct_concepts": len(concepts), "membership_calls": len(mreqs),
           "arms": summary, "note_vs_records_agreement": agree,
           "cost_usd_per_admission": {"extract": mean([c["extract"] for c in cost_per]), "verify": mean([c["verify"] for c in cost_per]),
                                      "membership_amortised": round(total_member / len(keys), 4),
                                      "membership_cold_start": mean([c["member_cold"] for c in cost_per])},
           "delay_seconds_per_admission": {a: {"median": q([d[a] for d in delay], .5), "p95": q([d[a] for d in delay], .95)}
                                           for a in ("notes", "notes+v", "graph_cold")},
           "failures": {"extract": ext.last_failures, "verify": ver.last_failures, "membership": mem.last_failures}}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "note_vs_records_agreement"}, indent=1))


if __name__ == "__main__":
    main()
