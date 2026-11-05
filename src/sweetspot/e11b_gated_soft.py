"""E11b: soft scoring applied only where the methods disagree (the structure that worked as a tie-break), on the same cached dev outputs.

Rule set before the run, after E11 showed an equal-weight mean over all signals everywhere loses to direct (0.6245 vs 0.7118):
  accept a candidate value when  votes >= v0   (enough of the four methods give exactly this value)
                          or     mean of the soft signals (enc, nli, ner, status; those available) >= tau
  per item the highest-scoring accepted value is kept.  Grid: v0 in {0.25, 0.5, 0.75, 1.0}, tau in 0.20..0.80 step 0.05.
Designs: all four soft signals; leave out each soft signal in turn (nothing is dropped from the design without being tested);
  references: direct, agree(direct, graph+audit), votes-only (v0 alone, no soft part).
Same stability rule as E11: 5-fold CV over notes (seed 0) choosing (v0, tau) on four folds; selected design = highest CV macro-F1 among
those beating direct in at least 4 of 5 folds whose (v0, tau) plateau (grid points within 0.005 of the best) has at least 5 points.
Free: cached outputs, local models, SWEETSPOT_MAX_USD=0, no test look. Output: results/sweetspot/e11b_gated_soft.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e11_soft_design as D  # noqa: E402
from sweetspot import e4_crf_benchmark as E  # noqa: E402
from sweetspot import e8_smoke as S  # noqa: E402
from sweetspot.llm import LLM, spent_usd  # noqa: E402

OUT = ROOT / "results/sweetspot/e11b_gated_soft.json"
V0S = [0.25, 0.5, 0.75, 1.0]
SOFT = ["enc", "nli", "ner", "status"]


def decide_gated(cands, soft, v0, tau):
    out = {}
    for k, rows in cands.items():
        best = {}
        for (i, v), sig in rows.items():
            vals = [sig[s] for s in soft if sig.get(s) is not None]
            score = float(np.mean(vals)) if vals else 0.0
            if sig["votes"] >= v0 or (soft and score >= tau):
                rank = sig["votes"] + score  # agreement first, soft evidence breaks ties between competing values
                if i not in best or rank > best[i][0]:
                    best[i] = (rank, v)
        out[k] = {i: v for i, (r, v) in best.items()}
    return out


def main():
    from sentence_transformers import SentenceTransformer
    opts = E.options()
    docs = E.load("development")
    g, _, _, _, extra = E.graph_arms(docs, opts, "development", SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"))
    direct = {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, LLM("e4_direct").run(E.direct_requests(docs, opts)))}
    reqs, where = E.reader_requests(docs, opts)
    reader = {d["id"]: {} for d in docs}
    for (k, i), r in zip(where, LLM("e4_reader", workers=32).run(reqs)):
        if r and E.valid(i, r["parsed"]["value"], opts) != E.UNK:
            reader[k][i] = E.valid(i, r["parsed"]["value"], opts)
    methods = {"direct": direct, "graph": g["graph"], "graph+audit": g["graph+audit"], "reader": reader}
    loc = S.Local(docs, opts)
    ner = D.gliner_scores({d["id"]: d["note"] for d in docs}, list(opts))
    cands = D.build_candidates(docs, opts, loc, methods, extra["evidence"]["graph"], ner)
    n = len(docs)
    base = np.array(E.official_score(docs, direct))
    folds = np.array_split(np.random.default_rng(0).permutation(n), 5)
    designs = {"all four soft signals": SOFT, "votes only (no soft part)": []}
    for s in SOFT:
        designs[f"leave out {s}"] = [x for x in SOFT if x != s]
    results = {}
    for name, soft in designs.items():
        grid = {}
        for v0 in V0S:
            for tau in (D.TAUS if soft else [1.1]):
                p = decide_gated(cands, soft, v0, tau)
                per, fp, acc = D.evaluate(docs, p)
                grid[(v0, tau)] = (np.array(per), fp, acc)
        means = {k: float(v[0].mean()) for k, v in grid.items()}
        best = max(means, key=means.get)
        plateau = [k for k in means if means[k] >= means[best] - .005]
        cv, gains, chosen = [], [], []
        for f in folds:
            train = np.setdiff1d(np.arange(n), f)
            k = max(grid, key=lambda x: grid[x][0][train].mean())
            chosen.append(list(k))
            cv.append(float(grid[k][0][f].mean()))
            gains.append(float(grid[k][0][f].mean() - base[f].mean()))
        per, fp, acc = grid[best]
        results[name] = {"soft_signals": soft, "best_v0_tau": list(best), "dev_macro_f1": round(means[best], 4), "plateau_points": len(plateau),
                         "cv_macro_f1": round(float(np.mean(cv)), 4), "cv_fold_sd": round(float(np.std(cv)), 4),
                         "cv_gain_over_direct_by_fold": [round(x, 4) for x in gains], "folds_beating_direct": int(sum(x > 0 for x in gains)),
                         "chosen_per_fold": chosen, "false_positives": fp, "accuracy_on_documented_items": round(acc, 4),
                         "minus_direct_in_sample": {"difference": round(float(per.mean() - base.mean()), 4), "ci": E.ci(per, base)}}
    eligible = {k: v for k, v in results.items() if v["folds_beating_direct"] >= 4 and v["plateau_points"] >= 5 and k != "votes only (no soft part)"}
    pick = max(eligible, key=lambda k: eligible[k]["cv_macro_f1"]) if eligible else None
    out = {"split": "development", "notes": n, "api_spent_usd": spent_usd(), "direct_macro_f1": round(float(base.mean()), 4),
           "designs": results, "selected_by_rule": pick, "eligible": sorted(eligible)}
    OUT.write_text(json.dumps(out, indent=1))
    print("api spent", out["api_spent_usd"], "| direct", out["direct_macro_f1"])
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["cv_macro_f1"]):
        print(f"{k:28s} dev {v['dev_macro_f1']:.4f}@{v['best_v0_tau']} plateau {v['plateau_points']} CV {v['cv_macro_f1']:.4f}±{v['cv_fold_sd']:.3f} "
              f"folds>direct {v['folds_beating_direct']}/5 FP {v['false_positives']} docacc {v['accuracy_on_documented_items']:.3f} vs direct {v['minus_direct_in_sample']['difference']} {v['minus_direct_in_sample']['ci']}")
    print("selected by rule:", pick)


if __name__ == "__main__":
    main()
