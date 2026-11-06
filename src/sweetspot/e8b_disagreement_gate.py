"""E8b (free, no API): abstention checks applied only where methods disagree, on the CRF dev split (80 notes, cached outputs).

Rules set before any run. Hard gates in E8 removed many more correct values than wrong ones, so here a value is checked only when
another method does not give the same value; agreed values are kept as they are.
  agree(base, other)             keep only values both give (reference: pure agreement)
  base, check disagreements Z2   keep agreed values; keep a disagreed value only if the NLI verifier entails it (measured values:
                                 only if the value occurs in the note)
  base, check disagreements Z4   the same with GLiNER detection as the check
Pairs: (graph+audit, direct), (direct, graph+audit), (direct, reader), (graph+audit, reader). Scores as in E8.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e4_crf_benchmark as E  # noqa: E402
from sweetspot.e8_smoke import Local, table  # noqa: E402
from sweetspot.llm import LLM  # noqa: E402

OUT = ROOT / "results/sweetspot/e8b_disagreement_gate.json"


def main():
    from sentence_transformers import SentenceTransformer
    opts = E.options()
    docs = E.load("development")
    g, _, _, _, _ = E.graph_arms(docs, opts, "development", SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"))
    dr = LLM("e4_direct").run(E.direct_requests(docs, opts))
    rl = LLM("e4_reader", workers=32)
    reqs, where = E.reader_requests(docs, opts)
    reader = {d["id"]: {} for d in docs}
    for (k, i), r in zip(where, rl.run(reqs)):
        if r and E.valid(i, r["parsed"]["value"], opts) != E.UNK:
            reader[k][i] = E.valid(i, r["parsed"]["value"], opts)
    base = {"direct": {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, dr)}, "graph+audit": g["graph+audit"], "reader": reader}
    notes = {d["id"]: d["note"] for d in docs}
    loc = Local(docs, opts)
    ref = {m: E.official_score(docs, p) for m, p in base.items()}
    out = {m: table(docs, p, {"direct": ref["direct"]})[0] for m, p in base.items()}
    for a, b in (("graph+audit", "direct"), ("direct", "graph+audit"), ("direct", "reader"), ("graph+audit", "reader")):
        A, B = base[a], base[b]
        agreed = {k: {i: v for i, v in A[k].items() if B[k].get(i) == v} for k in A}
        disagreed = {k: {i: v for i, v in A[k].items() if B[k].get(i) != v} for k in A}
        z2 = loc.z2(disagreed, notes)
        z4 = loc.z4(disagreed, notes)
        variants = {f"agree({a},{b})": agreed,
                    f"{a}, disagreements checked by Z2 vs {b}": {k: {**agreed[k], **z2[k]} for k in A},
                    f"{a}, disagreements checked by Z4 vs {b}": {k: {**agreed[k], **z4[k]} for k in A}}
        for name, p in variants.items():
            out[name] = table(docs, p, {"direct": ref["direct"], a: ref[a]})[0]
    OUT.write_text(json.dumps({"split": "development", "notes": len(docs), "api_calls": 0, "arms": out}, indent=1))
    for k, v in out.items():
        d = v.get("minus_direct", {})
        print(f"{k:58s} F1 {v['macro_f1']:.4f} FP {v['false_positives']:3d} docacc {v['accuracy_on_documented_items']:.3f} vs direct {d.get('difference', '')} {d.get('ci', '')}")


if __name__ == "__main__":
    main()
