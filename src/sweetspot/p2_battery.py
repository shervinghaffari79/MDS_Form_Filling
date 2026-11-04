"""Phase 2, step 2: label-free robustness battery on MIMIC for the note side of the admission graph.

Rules set before any run (docs/EXPERIMENT_PLAN.md; the tests validated on NegEx in E2, D4 held):
Sample: 30 admissions from the pilot (seed 31). Per admission, up to 3 items the notes+v arm fills from the note ("yes cases") and up
  to 2 items it does not fill ("no cases"), each drawn at random (seed 31).
Perturbations of the discharge summary, then the same pipeline (extraction and verifier per chunk; chunks the change does not touch
  hit the cache; new concept strings get membership calls):
  mask         remove every cited sentence of the item                         expected: item turns off (comprehensiveness)
  sufficiency  keep only the cited sentences, with their sections              expected: item stays on (sufficiency)
  negate       each cited sentence rewritten to deny the cited concept (kept only if NLI says contradiction)   expected: off
  paraphrase   each cited sentence reworded (kept only if NLI says two-way entailment)                       expected: on
  reorder      sections in reverse order, sentences inside each section unchanged                            expected: on
  family       no cases: one model-written sentence saying a family member has the item's condition, added to
               the end of the note                                              expected: stays off (attribution)
  ruled_out    no cases: one model-written sentence saying the item's condition was considered and ruled out expected: stays off
Arms scored: notes, notes+v, notes+v+t (time updating). Score = share of cases where the expected outcome holds, with a 95% CI from
  resampling admissions. Cost and calls are reported. No labels are used; outputs hold counts only.
"""
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import admission_graph as A  # noqa: E402
from sweetspot import perturb  # noqa: E402
from sweetspot.llm import LLM, cost  # noqa: E402

PRIVATE = Path("/home/user/mimic/pilot_private")
WORK = Path("/home/user/mimic/sweetspot")
OUT = ROOT / "results/sweetspot/p2_battery.json"
ARMS = ("notes", "notes+v", "notes+v+t")
EXPECT = {"mask": False, "sufficiency": True, "negate": False, "paraphrase": True, "reorder": True, "family": False, "ruled_out": False}
ADD_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["sentence"], "properties": {"sentence": {"type": "string"}}}


def run_many(cases, llms, membership):
    """Batched version of run_note for many perturbed notes: every stage is sent as one parallel batch (same requests, same results)."""
    ext, ver, mem = llms
    flat = [(n, c) for n, (chunks_list, _) in enumerate(cases) for c in chunks_list]
    rows = ext.run([A.mention_request(c) for _, c in flat])
    raw = []
    for (n, c), r in zip(flat, rows):
        valid = {sid for sid, _, _ in c}
        raw.append([m for m in (r["parsed"]["mentions"] if r else []) if m["sentence"] in valid and m["text"].strip()])
    vq = [(i, A.verify_request(flat[i][1], ms)) for i, ms in enumerate(raw) if ms]
    vr = ver.run([q for _, q in vq])
    kept = list(raw)
    for (i, _), r in zip(vq, vr):
        kept[i] = A.apply_verdicts(raw[i], r["parsed"] if r else None)
    new = sorted({(m["type"], A.norm(m["text"])) for ms in raw + kept for m in ms} - set(membership))
    mrows = []
    if new:
        valid_codes = {i["code"] for i in A.ITEMS["items"]}
        mreqs = A.member_requests(new)
        mrows = mem.run([q for q, _, _ in mreqs])
        for (q, t, batch), r in zip(mreqs, mrows):
            for a in (r["parsed"]["answers"] if r else []):
                if 0 <= a["concept"] < len(batch):
                    membership[(t, batch[a["concept"]])] = [c for c in a["items"] if c in valid_codes]
    per_raw, per_kept = defaultdict(list), defaultdict(list)
    for i, (n, _) in enumerate(flat):
        per_raw[n] += raw[i]
        per_kept[n] += kept[i]
    out = []
    for n, (_, code) in enumerate(cases):
        g = {"notes": A.build_graph("", per_raw[n], [], membership, None),
             "notes+v": A.build_graph("", per_kept[n], [], membership, None),
             "notes+v+t": A.build_graph("", per_kept[n], [], membership, None, time_update=True)}
        out.append({arm: code in gg for arm, gg in g.items()})
    return out, list(rows) + list(vr) + list(mrows)


def main():
    state = pickle.loads((WORK / "p2_graphs.pkl").read_bytes())
    arms, membership = state["arms"], state["membership"]
    sents = pd.read_parquet(PRIVATE / "proxy_sentences.parquet")
    rng = random.Random(31)
    keys = sorted(arms["notes+v"])
    sample = rng.sample(keys, min(30, len(keys)))
    items = {i["code"]: i for i in A.ITEMS["items"]}
    llms = (LLM("p2_mentions"), LLM("p2_verify"), LLM("p2_membership"))
    gen = LLM("p2_battery_rewrites")
    nli = perturb.NLI()
    results = defaultdict(list)  # kind -> list of (admission, {arm: holds})
    spend = []
    specs = []          # (admission, code, kind, builder data)
    rewrite_reqs, rewrite_meta = [], []
    add_reqs, add_meta = [], []
    notes = {}
    for k in sample:
        rows = sents[sents["note_key"] == k].sort_values("sent_id")
        triples = list(zip(rows["sent_id"], rows["section"], rows["text"]))
        notes[k] = (triples, {sid: (sec, t) for sid, sec, t in triples}, list(A.chunks(triples)))
        filled = arms["notes+v"][k]
        yes = [c for c in filled if any(p["source"] == "note" for p in filled[c])]
        yes = rng.sample(yes, min(3, len(yes)))
        no = rng.sample([c for c in items if c not in filled and items[c]["section"] == "I"], 2)
        for code in yes:
            cited = sorted({p["sentence"] for p in filled[code] if p["source"] == "note"})
            concept = filled[code][0]["concept"]
            specs.append((k, code, cited))
            for kind in ("negate", "paraphrase"):
                its = [{"id": sid, "sentence": notes[k][1][sid][1], "concept": concept} for sid in cited]
                for it, q in zip(its, perturb.requests_for(its, kind)):
                    rewrite_reqs.append(q)
                    rewrite_meta.append((k, code, kind, it))
        for code in no:
            label = items[code]["label"]
            for kind, ask in (("family", f"Write one sentence for a discharge summary stating that the patient's mother has a history of {label}."),
                              ("ruled_out", f"Write one sentence for a discharge summary stating that {label} was considered in this patient and ruled out.")):
                add_reqs.append({"name": "add", "system": "You write single sentences for a robustness test of clinical NLP.", "prompt": ask,
                                 "schema": ADD_SCHEMA})
                add_meta.append((k, code, kind))
    rew = gen.run(rewrite_reqs)
    adds = gen.run(add_reqs)
    spend += rew + adds
    cand = [(m, r["parsed"]["text"].strip()) for m, r in zip(rewrite_meta, rew) if r and r["parsed"]["text"].strip() != m[3]["sentence"]]
    by_kind = defaultdict(list)
    for (m, t) in cand:
        by_kind[m[2]].append((m, t))
    new_text = defaultdict(dict)
    for kind, pairs in by_kind.items():
        ok = perturb.accept(kind, nli, [m[3]["sentence"] for m, _ in pairs], [t for _, t in pairs])
        for (m, t), good in zip(pairs, ok):
            if good:
                new_text[(m[0], m[1], kind)][m[3]["id"]] = t
    cases, meta = [], []
    for k, code, cited in specs:
        triples, text_of, orig_chunks = notes[k]
        cases.append(([c for c in ([x for x in c if x[0] not in cited] for c in orig_chunks) if c], code)); meta.append((k, "mask"))
        cases.append(([[(sid, *text_of[sid]) for sid in cited]], code)); meta.append((k, "sufficiency"))
        for kind in ("negate", "paraphrase"):
            nt = new_text.get((k, code, kind), {})
            if len(nt) == len(cited):  # every cited sentence must be changed, or the case is not used
                cases.append(([[(sid, sec, nt.get(sid, t)) for sid, sec, t in c] for c in orig_chunks], code)); meta.append((k, kind))
        order = list(dict.fromkeys(sec for _, sec, _ in triples))[::-1]
        cases.append((list(A.chunks([x for sec in order for x in triples if x[1] == sec])), code)); meta.append((k, "reorder"))
    for (k, code, kind), r in zip(add_meta, adds):
        if not r:
            continue
        triples, _, orig_chunks = notes[k]
        section = "Family History" if kind == "family" else "Brief Hospital Course"
        extra = (triples[-1][0] + 1, section, r["parsed"]["sentence"].strip())
        cases.append((orig_chunks[:-1] + [orig_chunks[-1] + [extra]], code)); meta.append((k, kind))
    got, used = run_many(cases, llms, membership)
    spend += used
    for (k, kind), g in zip(meta, got):
        results[kind].append((k, {arm: g[arm] == EXPECT[kind] for arm in ARMS}))
    generation = {kind: {"cases_with_all_cited_rewritten": sum(1 for _, kd in meta if kd == kind)} for kind in ("negate", "paraphrase")}
    boot = np.random.default_rng(0)
    summary = {}
    for kind, rows in results.items():
        summary[kind] = {"cases": len(rows), "expected": "on" if EXPECT[kind] else "off"}
        adms = sorted({a for a, _ in rows})
        for arm in ARMS:
            by = defaultdict(list)
            for a, h in rows:
                by[a].append(h[arm])
            point = float(np.mean([h[arm] for _, h in rows]))
            vals = []
            for _ in range(2000):
                pick = boot.choice(adms, len(adms))
                xs = [x for a in pick for x in by[a]]
                vals.append(np.mean(xs))
            summary[kind][arm] = {"holds": round(point, 3), "ci": [round(float(np.quantile(vals, .025)), 3), round(float(np.quantile(vals, .975)), 3)]}
    out = {"admissions": len(sample), "yes_cases": len(specs), "battery": summary, "rewrites_accepted": generation, "cost": cost(spend)}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
