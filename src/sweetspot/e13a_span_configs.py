"""E13a: NER and LLM configurations on the labelled span sets (glaucoma lines, MACCROBAT reports): the same units, prompts and cached
model replies as E1, so everything reused here is free and comparable. No new models are searched or added: NER is GLiNER-BioMed,
the LLM is gpt-6-luna. Rules set before the run.

Base arms (from E1, rebuilt from the caches): ner (GLiNER alone), ner+verify, llm (zero-shot), llm+verify.
Combinations (a span matches another when they have the same type and overlap):
  llm AND ner             LLM spans that some NER span supports (precision gate; no calls)
  llm+verify AND ner      the same on the verified LLM spans
  llm OR ner+verify       LLM spans plus the verified NER spans the LLM did not find (recall union)
  llm+verify OR ner+verify
  two-path: llm, keep if verifier OK or NER supports      (a span survives if either path vouches for it)
  two-path: llm, keep if verifier OK and NER supports     (both paths must vouch)
Scores: lenient and strict micro precision, recall, F1 on the core types, with 95% CI by resampling units, paired differences against
llm+verify (E1's best arm). A configuration is called stable when its core F1 ranks above llm+verify on both datasets or below on both;
otherwise dataset-dependent. Output: results/sweetspot/e13a_span_configs.json (no text).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import modal_ner_bench as G  # noqa: E402
from sweetspot import e1_entity_layer as E1  # noqa: E402
from sweetspot.llm import LLM, spent_usd  # noqa: E402

OUT = ROOT / os.environ.get("E13A_OUT", "results/sweetspot/e13a_span_configs.json")


def overlap(a, b):
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


def supported(spans, other):
    return [s for s in spans if any(overlap(s, o) for o in other)]


def union(a, b):
    return list(a) + [s for s in b if not any(overlap(s, x) for x in a)]


def run(name, gliner):
    cfg = E1.DATASETS[name]
    units = E1.load(name)
    llm = LLM(f"e1_{name}_ner")
    rows = llm.run([E1.ner_request(cfg, u["text"]) for u in units])
    llm_spans = [G.align_mentions(u["text"], r["parsed"]["mentions"]) if r else [] for u, r in zip(units, rows)]
    inv = {v: k for k, v in cfg["gliner_labels"].items()}
    ner = [[(inv[e["label"]], e["start"], e["end"]) for e in gliner.predict_entities(u["text"], list(cfg["gliner_labels"].values()), threshold=0.5)] for u in units]
    ver = LLM(f"e1_{name}_verify")
    v = {}
    for key, base in (("llm", llm_spans), ("ner", ner)):
        reqs = [E1.verify_request(cfg, u["text"], p) if p else None for u, p in zip(units, base)]
        sent = ver.run([r for r in reqs if r])
        it = iter(sent)
        replies = [next(it) if r else None for r in reqs]
        v[key] = [E1.apply_verdicts(p, r["parsed"] if r else None) for p, r in zip(base, replies)]
    llm_v, ner_v = v["llm"], v["ner"]
    ok_set = [set(map(tuple, x)) for x in llm_v]
    arms = {
        "ner": ner, "ner+verify": ner_v, "llm": llm_spans, "llm+verify": llm_v,
        "llm AND ner": [supported(a, b) for a, b in zip(llm_spans, ner)],
        "llm+verify AND ner": [supported(a, b) for a, b in zip(llm_v, ner)],
        "llm OR ner+verify": [union(a, b) for a, b in zip(llm_spans, ner_v)],
        "llm+verify OR ner+verify": [union(a, b) for a, b in zip(llm_v, ner_v)],
        "two-path OR (verifier ok or NER supports)": [[s for s in a if tuple(s) in ok or any(overlap(s, o) for o in b)] for a, ok, b in zip(llm_spans, ok_set, ner)],
        "two-path AND (verifier ok and NER supports)": [[s for s in a if tuple(s) in ok and any(overlap(s, o) for o in b)] for a, ok, b in zip(llm_spans, ok_set, ner)],
    }
    per = {a: {m: E1.counts(units, p, cfg["types"], m) for m in ("lenient", "strict")} for a, p in arms.items()}
    out = {}
    for a in arms:
        out[a] = {}
        for m in ("lenient", "strict"):
            p, r, f = E1.pooled(per[a][m], cfg["core"])
            out[a][m] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
        out[a]["minus_llm+verify_f1"] = E1.paired(per[a]["lenient"], per["llm+verify"]["lenient"], cfg["core"], 2)
    return {"units": len(units), "arms": out}


def main():
    from gliner import GLiNER
    gliner = GLiNER.from_pretrained("Ihor/gliner-biomed-base-v1.0")
    res = {n: run(n, gliner) for n in E1.DATASETS}
    ref = {n: res[n]["arms"]["llm+verify"]["lenient"]["f1"] for n in res}
    stable = {}
    for a in res["glaucoma"]["arms"]:
        d = [res[n]["arms"][a]["lenient"]["f1"] - ref[n] for n in res]
        stable[a] = "same sign on both" if (d[0] > 0) == (d[1] > 0) else "dataset-dependent"
    out = {"api_spent_usd": spent_usd(), "datasets": res, "rank_vs_llm_verify": stable}
    OUT.write_text(json.dumps(out, indent=1))
    for n in res:
        print("==", n, res[n]["units"], "units")
        for a, v in sorted(res[n]["arms"].items(), key=lambda kv: -kv[1]["lenient"]["f1"]):
            d = v["minus_llm+verify_f1"]
            print(f"  {a:46s} P {v['lenient']['precision']:.3f} R {v['lenient']['recall']:.3f} F1 {v['lenient']['f1']:.3f}  strict F1 {v['strict']['f1']:.3f}  vs llm+verify {d[0]:+.3f} {d[1]}")
    print("api spent", out["api_spent_usd"])


if __name__ == "__main__":
    main()
