"""Phase 2, step 4: the same robustness battery for the item-by-item reader, on its own evidence.

Rules set before any run, mirroring p2_battery.py so the two methods are compared on equal terms:
Admissions: the 20 reader-baseline admissions. Per admission, up to 3 items the reader answers yes (cases use the sentences the
  reader cited) and 2 Section I items it answers no, drawn with seed 31. Yes cases without any cited sentence are skipped.
Perturbations and expected outcomes as in p2_battery.py: mask, sufficiency, negate, paraphrase, reorder (on), family, ruled_out (off).
  Rewrites are generated and NLI-checked the same way; a negate or paraphrase case is used only if every cited sentence was changed.
After each perturbation the reader is rerun for that item: retrieval of the 20 sentences most similar to the item wording on the
  perturbed note, then the unchanged prompt. Score: share of cases where the expected outcome holds, 95% CI resampling admissions.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mds_pilot.reader import PLAIN_SCHEMA, SYSTEM, build_prompt  # noqa: E402
from sweetspot import admission_graph as A  # noqa: E402
from sweetspot import perturb  # noqa: E402
from sweetspot.llm import LLM, cost  # noqa: E402
from sweetspot.p2_battery import ADD_SCHEMA, EXPECT  # noqa: E402

PRIVATE = Path("/home/user/mimic/pilot_private")
OUT = ROOT / "results/sweetspot/p2_reader_battery.json"
TOP = 20


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    items = {i["code"]: i for i in A.ITEMS["items"]}
    qv = {c: v for c, v in zip(items, enc.encode(["Represent this sentence for searching relevant passages: " + i["wording"]
                                                  for i in items.values()], normalize_embeddings=True))}
    sents = pd.read_parquet(PRIVATE / "proxy_sentences.parquet")
    keys = sorted(sents["note_key"].unique())[:20]
    llm, gen = LLM("p2_reader"), LLM("p2_reader_battery_rewrites")
    nli = perturb.NLI()

    emb_cache = {}  # sentence text -> embedding; each distinct sentence is encoded once (the encoder is deterministic)

    def ask(triples, code):
        texts = [t for _, _, t in triples]
        missing = [t for t in dict.fromkeys(texts) if t not in emb_cache]
        if missing:
            emb_cache.update(zip(missing, enc.encode(missing, normalize_embeddings=True, batch_size=128)))
        e = np.stack([emb_cache[t] for t in texts])
        top = np.sort(np.argsort(-(e @ qv[code]))[:TOP])
        block = [triples[t] for t in top]
        return {"name": "plain", "system": SYSTEM, "prompt": build_prompt("plain", items[code]["wording"], block), "schema": PLAIN_SCHEMA}

    notes, base_reqs, base_where = {}, [], []
    for k in keys:
        rows = sents[sents["note_key"] == k].sort_values("sent_id")
        triples = [(int(a), b, c) for a, b, c in zip(rows["sent_id"], rows["section"], rows["text"])]
        notes[k] = triples
        for code in items:
            base_reqs.append(ask(triples, code))
            base_where.append((k, code))
    base = dict(zip(base_where, llm.run(base_reqs)))  # cache hits from the baseline run
    rng = random.Random(31)
    specs, rewrite_reqs, rewrite_meta, add_reqs, add_meta = [], [], [], [], []
    for k in keys:
        triples = notes[k]
        text_of = {sid: (sec, t) for sid, sec, t in triples}
        yes = [c for c in items if base[(k, c)] and base[(k, c)]["parsed"]["answer"] == "yes"
               and any(s in text_of for s in base[(k, c)]["parsed"]["evidence"])]
        yes = rng.sample(yes, min(3, len(yes)))
        no = rng.sample([c for c in items if items[c]["section"] == "I" and base[(k, c)] and base[(k, c)]["parsed"]["answer"] == "no"], 2)
        for code in yes:
            cited = sorted({s for s in base[(k, code)]["parsed"]["evidence"] if s in text_of})
            specs.append((k, code, cited))
            for kind in ("negate", "paraphrase"):
                its = [{"id": s, "sentence": text_of[s][1], "concept": items[code]["label"]} for s in cited]
                for it, q in zip(its, perturb.requests_for(its, kind)):
                    rewrite_reqs.append(q)
                    rewrite_meta.append((k, code, kind, it))
        for code in no:
            label = items[code]["label"]
            for kind, prompt in (("family", f"Write one sentence for a discharge summary stating that the patient's mother has a history of {label}."),
                                 ("ruled_out", f"Write one sentence for a discharge summary stating that {label} was considered in this patient and ruled out.")):
                add_reqs.append({"name": "add", "system": "You write single sentences for a robustness test of clinical NLP.", "prompt": prompt,
                                 "schema": ADD_SCHEMA})
                add_meta.append((k, code, kind))
    rew, adds = gen.run(rewrite_reqs), gen.run(add_reqs)
    spend = list(rew) + list(adds)
    by_kind = defaultdict(list)
    for m, r in zip(rewrite_meta, rew):
        if r and r["parsed"]["text"].strip() != m[3]["sentence"]:
            by_kind[m[2]].append((m, r["parsed"]["text"].strip()))
    new_text = defaultdict(dict)
    for kind, pairs in by_kind.items():
        for (m, t), good in zip(pairs, perturb.accept(kind, nli, [m[3]["sentence"] for m, _ in pairs], [t for _, t in pairs])):
            if good:
                new_text[(m[0], m[1], kind)][m[3]["id"]] = t
    cases, meta = [], []
    for k, code, cited in specs:
        triples = notes[k]
        cases.append(([x for x in triples if x[0] not in cited], code)); meta.append((k, "mask"))
        cases.append(([x for x in triples if x[0] in cited], code)); meta.append((k, "sufficiency"))
        for kind in ("negate", "paraphrase"):
            nt = new_text.get((k, code, kind), {})
            if len(nt) == len(cited):
                cases.append(([(s, sec, nt.get(s, t)) for s, sec, t in triples], code)); meta.append((k, kind))
        order = list(dict.fromkeys(sec for _, sec, _ in triples))[::-1]
        cases.append(([x for sec in order for x in triples if x[1] == sec], code)); meta.append((k, "reorder"))
    for (k, code, kind), r in zip(add_meta, adds):
        if r:
            triples = notes[k]
            section = "Family History" if kind == "family" else "Brief Hospital Course"
            cases.append((triples + [(triples[-1][0] + 1, section, r["parsed"]["sentence"].strip())], code)); meta.append((k, kind))
    replies = llm.run([ask(tr, code) for tr, code in cases])
    spend += replies
    results = defaultdict(list)
    for (k, kind), r in zip(meta, replies):
        if r:
            results[kind].append((k, (r["parsed"]["answer"] == "yes") == EXPECT[kind]))
    boot = np.random.default_rng(0)
    summary = {}
    for kind, rows in results.items():
        adms = sorted({a for a, _ in rows})
        by = defaultdict(list)
        for a, h in rows:
            by[a].append(h)
        vals = [np.mean([x for a in boot.choice(adms, len(adms)) for x in by[a]]) for _ in range(2000)]
        summary[kind] = {"cases": len(rows), "expected": "on" if EXPECT[kind] else "off",
                         "reader": {"holds": round(float(np.mean([h for _, h in rows])), 3),
                                    "ci": [round(float(np.quantile(vals, .025)), 3), round(float(np.quantile(vals, .975)), 3)]}}
    out = {"admissions": len(keys), "yes_cases": len(specs), "battery": summary, "cost": cost(spend)}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
