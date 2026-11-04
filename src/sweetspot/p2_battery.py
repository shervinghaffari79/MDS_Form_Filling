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


def run_note(chunks_list, llms, membership, item_codes, records=None):
    """Extraction, verifier and membership for one perturbed note; returns {arm: set of filled codes among item_codes}."""
    ext, ver, mem = llms
    rows = ext.run([A.mention_request(c) for c in chunks_list])
    raw = []
    for c, r in zip(chunks_list, rows):
        valid = {sid for sid, _, _ in c}
        raw.append([m for m in (r["parsed"]["mentions"] if r else []) if m["sentence"] in valid and m["text"].strip()])
    vq = [(i, A.verify_request(c, ms)) for i, (c, ms) in enumerate(zip(chunks_list, raw)) if ms]
    vr = ver.run([q for _, q in vq])
    kept = list(raw)
    for (i, _), r in zip(vq, vr):
        kept[i] = A.apply_verdicts(raw[i], r["parsed"] if r else None)
    flat_raw = [m for ms in raw for m in ms]
    flat_kept = [m for ms in kept for m in ms]
    new = sorted({(m["type"], A.norm(m["text"])) for m in flat_raw + flat_kept} - set(membership))
    if new:
        valid_codes = {i["code"] for i in A.ITEMS["items"]}
        mreqs = A.member_requests(new)
        for (q, t, batch), r in zip(mreqs, mem.run([q for q, _, _ in mreqs])):
            for a in (r["parsed"]["answers"] if r else []):
                if 0 <= a["concept"] < len(batch):
                    membership[(t, batch[a["concept"]])] = [c for c in a["items"] if c in valid_codes]
    out = {"notes": A.build_graph("", flat_raw, [], membership, None),
           "notes+v": A.build_graph("", flat_kept, [], membership, None),
           "notes+v+t": A.build_graph("", flat_kept, [], membership, None, time_update=True)}
    return {arm: {c for c in g if c in item_codes} for arm, g in out.items()}, rows + vr


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
    for k in sample:
        rows = sents[sents["note_key"] == k].sort_values("sent_id")
        triples = list(zip(rows["sent_id"], rows["section"], rows["text"]))
        text_of = {sid: (sec, t) for sid, sec, t in triples}
        orig_chunks = list(A.chunks(triples))
        filled = arms["notes+v"][k]
        yes = [c for c in filled if any(p["source"] == "note" for p in filled[c])]
        yes = rng.sample(yes, min(3, len(yes)))
        no = rng.sample([c for c in items if c not in filled and items[c]["section"] == "I"], 2)
        for code in yes:
            cited = sorted({p["sentence"] for p in filled[code] if p["source"] == "note"})
            concept = filled[code][0]["concept"]
            variants = {
                "mask": [[x for x in c if x[0] not in cited] for c in orig_chunks],
                "sufficiency": [[(sid, *text_of[sid]) for sid in cited]],
            }
            for kind in ("negate", "paraphrase"):
                items_in = [{"id": sid, "sentence": text_of[sid][1], "concept": concept} for sid in cited]
                reps = gen.run(perturb.requests_for(items_in, kind))
                spend += reps
                new_text = {}
                for it, r in zip(items_in, reps):
                    t = r["parsed"]["text"].strip() if r else ""
                    if t and t != it["sentence"] and perturb.accept(kind, nli, [it["sentence"]], [t])[0]:
                        new_text[it["id"]] = t
                if len(new_text) == len(cited):  # every cited sentence must be changed, or the case is not used
                    variants[kind] = [[(sid, sec, new_text.get(sid, t)) for sid, sec, t in c] for c in orig_chunks]
            order = list(dict.fromkeys(sec for _, sec, _ in triples))[::-1]
            variants["reorder"] = list(A.chunks([x for sec in order for x in triples if x[1] == sec]))
            for kind, chunks_list in variants.items():
                got, used = run_note([c for c in chunks_list if c], llms, membership, {code})
                spend += used
                results[kind].append((k, {arm: (code in got[arm]) == EXPECT[kind] for arm in ARMS}))
        for code in no:
            label = items[code]["label"]
            for kind, ask in (("family", f"Write one sentence for a discharge summary stating that the patient's mother has a history of {label}."),
                              ("ruled_out", f"Write one sentence for a discharge summary stating that {label} was considered in this patient and ruled out.")):
                r = gen.run([{"name": "add", "system": "You write single sentences for a robustness test of clinical NLP.", "prompt": ask,
                              "schema": ADD_SCHEMA}])[0]
                spend.append(r)
                if not r:
                    continue
                last_sid = triples[-1][0] + 1
                section = "Family History" if kind == "family" else "Brief Hospital Course"
                chunks_list = orig_chunks[:-1] + [orig_chunks[-1] + [(last_sid, section, r["parsed"]["sentence"].strip())]]
                got, used = run_note(chunks_list, llms, membership, {code})
                spend += used
                results[kind].append((k, {arm: (code in got[arm]) == EXPECT[kind] for arm in ARMS}))
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
    out = {"admissions": len(sample), "battery": summary, "cost": cost(spend)}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
