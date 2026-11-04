"""NegEx/ConText hybrid: rules propose, the status record checks only where a rule fired. Built from stored outputs, no new calls.
Rules set before scoring.

Gate: the ConText rules (medspaCy) output a label for each of 2,376 mentions. A mention "fired" if ConText gave any non-default value
(Negated, Historical or Not particular, or experiencer Other). Fired mentions take the status-record answer (already computed for every
mention); the rest keep ConText's default (Affirmed, Recent, Patient). The model reads only the fired share, so its cost is that share of the
status arm's tokens (proportional estimate, since replies are not stored per mention).
Metrics as in the first NegEx run: negation F1 (Negated), temporality macro F1 over Recent, Historical and Not particular, experiencer accuracy.
Decisions, 95% CI from resampling mentions:
  X1 the hybrid beats ConText: negation F1 and temporality macro F1 differences both with CI above 0.
  X2 the hybrid matches the status record: lower CI bound of (hybrid minus status) above -0.015 on both negation F1 and temporality macro F1.
  X3 cost: the hybrid reads at most 50% of the mentions the status arm read.
Also reported: gold non-default mentions that ConText never fired on (the gate cannot recover them), and fired mentions whose gold is default.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "benchmarks_open/negex/rsAnnotations-1-120-random.txt"
OUT = ROOT / "results/open_bench"
BOOT = 2000
TIMING = ("Recent", "Historical", "Not particular")


def gold_rows():
    rows = {}
    for line in GOLD.read_text(errors="replace").splitlines():
        p = line.split("\t")
        if len(p) == 7:
            rows[p[0]] = {"negation": p[4].strip(), "temporality": p[5].strip(), "experiencer": p[6].strip()}
    return rows


def f1(g, p, label):
    g, p = np.asarray(g) == label, np.asarray(p) == label
    tp = (g & p).sum()
    return 2 * tp / (g.sum() + p.sum()) if g.sum() + p.sum() else float("nan")


def metrics(g, p):
    return {"negation_f1": f1(g["negation"], p["negation"], "Negated"),
            "temporality_macro_f1": float(np.mean([f1(g["temporality"], p["temporality"], c) for c in TIMING])),
            "experiencer_accuracy": float((np.asarray(g["experiencer"]) == np.asarray(p["experiencer"])).mean())}


def main():
    gold = gold_rows()
    preds = {a: {r["id"]: r for r in json.loads((OUT / f"negex_{a}_predictions.json").read_text())["predictions"]} for a in ("context", "status")}
    status_tokens = json.loads((OUT / "negex_status_predictions.json").read_text())["new_tokens"]
    ids = sorted(i for i in gold if i in preds["context"] and i in preds["status"])
    fired = {i: (preds["context"][i]["negation"] != "Affirmed" or preds["context"][i]["temporality"] != "Recent"
                 or preds["context"][i]["experiencer"] != "Patient") for i in ids}
    preds["hybrid"] = {i: (preds["status"][i] if fired[i] else preds["context"][i]) for i in ids}
    cols = lambda a, sel: {k: [(preds[a][i] if a != "gold" else gold[i])[k] for i in sel] for k in ("negation", "temporality", "experiencer")}
    gcols = {k: [gold[i][k] for i in ids] for k in ("negation", "temporality", "experiencer")}
    arms = ("context", "status", "hybrid")
    result = {"mentions": len(ids), "fired": int(sum(fired.values())), "fired_share": round(sum(fired.values()) / len(ids), 4),
              "approx_tokens_hybrid": int(status_tokens * sum(fired.values()) / len(ids)), "status_tokens": status_tokens,
              "scores": {a: {k: round(v, 4) for k, v in metrics(gcols, cols(a, ids)).items()} for a in arms}}
    default = lambda i: gold[i]["negation"] == "Affirmed" and gold[i]["temporality"] == "Recent" and gold[i]["experiencer"] == "Patient"
    result["gate"] = {"gold_nondefault": int(sum(not default(i) for i in ids)), "gold_nondefault_never_fired": int(sum(not default(i) and not fired[i] for i in ids)),
                      "fired_but_gold_default": int(sum(fired[i] and default(i) for i in ids))}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), size=(BOOT, len(ids)))
    arrays = {a: {k: np.array(v, dtype=object) for k, v in cols(a, ids).items()} for a in arms}
    garr = {k: np.array(v, dtype=object) for k, v in gcols.items()}

    def stat(a, rows, key):
        g = {k: v[rows] for k, v in garr.items()}
        p = {k: v[rows] for k, v in arrays[a].items()}
        return metrics(g, p)[key]

    def diff(a, b, key):
        point = stat(a, np.arange(len(ids)), key) - stat(b, np.arange(len(ids)), key)
        vals = [stat(a, r, key) - stat(b, r, key) for r in idx]
        return {"difference": round(float(point), 4), "ci": [round(float(np.quantile(vals, .025)), 4), round(float(np.quantile(vals, .975)), 4)]}

    d = {f"{a}_minus_{b}": {k: diff(a, b, k) for k in ("negation_f1", "temporality_macro_f1")} for a, b in (("hybrid", "context"), ("hybrid", "status"))}
    result["contrasts"] = d
    result["decisions"] = {
        "X1_hybrid_beats_context": bool(all(d["hybrid_minus_context"][k]["ci"][0] > 0 for k in d["hybrid_minus_context"])),
        "X2_hybrid_matches_status": bool(all(d["hybrid_minus_status"][k]["ci"][0] > -0.015 for k in d["hybrid_minus_status"])),
        "X3_cost_at_most_half": bool(result["fired_share"] <= .5)}
    (OUT / "negex_gated.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
