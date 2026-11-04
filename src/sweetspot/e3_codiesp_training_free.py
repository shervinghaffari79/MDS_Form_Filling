"""E3: a training-free decision over the same CodiEsp graph edges, against the saved logistic-regression graph.

Rules set before any run. Edges per candidate code (modal_open_bench.codiesp_graph_features, unchanged): reader chose it (and rank), free
coding chose it (and rank), index-E retrieval rank, training evidence phrases found in the case text, co-occurrence support from the
reader's other codes, training frequency.
Ranking (MAP), no fitted weights:
  T1 reciprocal rank fusion (k=60) over all six edge kinds, each kind ranking the case's candidates (a code missing from a kind adds 0)
  T2 the same over the four direct-evidence kinds (reader, free, retrieval, phrases), leaving out the two dataset priors
  T3 the same over the two language-model kinds only (reader, free): uses no CodiEsp training data at all (generalisation reference)
Acceptance (F1), no thresholds learned:
  A1 at least two independent evidence paths among {reader chose, free coding chose, a training phrase is in the text}
  A2 reader or free coding chose it
  A3 both chose it
Dev picks one ranking by MAP and one acceptance by F1 (ties within 0.005 go to the simpler: T3 < T2 < T1, A2 < A3 < A1).
D3 (amended before the run): the logistic graph's dev score is in-sample (it was fit on dev), so the comparison is made on test, where
  both are out of sample: the training-free choice is kept if its test MAP is within 0.05 of the logistic graph (lower 95% CI bound of
  the difference above -0.05). This is a further look at the CodiEsp test split (the 7th).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import modal_open_bench as M  # noqa: E402

OUT = ROOT / "results/sweetspot/e3_codiesp_training_free.json"
KINDS = {"T1": ["reader", "free", "e", "phrase", "cooccur", "freq"], "T2": ["reader", "free", "e", "phrase"], "T3": ["reader", "free"]}


def kind_scores(f):
    """Higher is better; 0 or less means the kind gives no evidence for this code."""
    return {"reader": f["reader_inv_rank"], "free": f["free_inv_rank"], "e": f["e_inv_rank"], "phrase": f["phrase_hits"],
            "cooccur": f["cooccur"], "freq": f["log_freq"]}


def rrf(codes, kinds, k=60):
    total = {c: 0.0 for c in codes}
    for kind in kinds:
        vals = {c: kind_scores(f)[kind] for c, f in codes.items()}
        ranked = sorted((c for c in vals if vals[c] > 0), key=lambda c: -vals[c])
        for r, c in enumerate(ranked, 1):
            total[c] += 1.0 / (k + r)
    return sorted(total, key=lambda c: (-total[c], c))


def accept(codes, rule):
    out = []
    for c, f in codes.items():
        r, fr, ph = f["in_reader"] > 0, f["in_free"] > 0, f["phrase_hits"] > 0
        if (rule == "A1" and r + fr + ph >= 2) or (rule == "A2" and (r or fr)) or (rule == "A3" and r and fr):
            out.append(c)
    return out


def main():
    load = lambda arm, sp: json.loads((M.LOCAL_OUT / f"codiesp_{arm}_{sp}_predictions.json").read_text())["predictions"]
    feats, gold = {}, {}
    for sp in ("dev", "test"):
        res = M.codiesp_graph_features.local(sp, load("two_stage_e", sp), load("free", sp))
        feats[sp], gold[sp] = res["features"], res["gold"]

    def run(sp, rank_rule, acc_rule):
        ranked = {d: rrf(codes, KINDS[rank_rule]) for d, codes in feats[sp].items()}
        accepted = {d: [c for c in ranked[d] if c in set(accept(codes, acc_rule))] for d, codes in feats[sp].items()}
        s_rank, aps = M.codiesp_scores(gold[sp], ranked)
        s_acc, _ = M.codiesp_scores(gold[sp], accepted)
        return s_rank, s_acc, aps, accepted

    dev = {}
    for t in KINDS:
        for a in ("A1", "A2", "A3"):
            s_rank, s_acc, _, _ = run("dev", t, a)
            dev[f"{t}+{a}"] = {"map": round(s_rank["map"], 4), "f1": round(s_acc["f1"], 4)}
    simpler = {"T3": 0, "T2": 1, "T1": 2, "A2": 0, "A3": 1, "A1": 2}
    maps = {t: dev[f"{t}+A1"]["map"] for t in KINDS}
    best = max(maps.values())
    pick_t = min((t for t in KINDS if maps[t] >= best - .005), key=simpler.get)
    f1s = {a: dev[f"{pick_t}+{a}"]["f1"] for a in ("A1", "A2", "A3")}
    best_f = max(f1s.values())
    pick_a = min((a for a in f1s if f1s[a] >= best_f - .005), key=simpler.get)

    s_rank, s_acc, aps, accepted = run("test", pick_t, pick_a)
    # logistic graph, refit exactly as saved (fit on dev, applied to test), for paired comparison
    from sklearn.linear_model import LogisticRegression
    dv, ts = M.graph_frame(feats["dev"]), M.graph_frame(feats["test"])
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(dv[M.GRAPH_EDGES].to_numpy(), dv["gold"].to_numpy())
    ts["p"] = lr.predict_proba(ts[M.GRAPH_EDGES].to_numpy())[:, 1]
    lr_ranked = {d: g.sort_values("p", ascending=False)["code"].tolist() for d, g in ts.groupby("doc")}
    lr_acc = {d: g.loc[g["p"] >= .5].sort_values("p", ascending=False)["code"].tolist() for d, g in ts.groupby("doc")}
    lr_rank_s, lr_aps = M.codiesp_scores(gold["test"], lr_ranked)
    lr_acc_s, _ = M.codiesp_scores(gold["test"], lr_acc)
    diff, ci = M.paired_ci(aps, lr_aps)
    t3_rank, t3_acc, _, _ = run("test", "T3", pick_a)
    out = {"dev_grid": dev, "picked": {"ranking": pick_t, "acceptance": pick_a},
           "test": {"training_free": {"map": round(s_rank["map"], 4), "f1": round(s_acc["f1"], 4),
                                      "precision": round(s_acc["precision"], 4), "recall": round(s_acc["recall"], 4)},
                    "logistic_graph": {"map": round(lr_rank_s["map"], 4), "f1": round(lr_acc_s["f1"], 4),
                                       "precision": round(lr_acc_s["precision"], 4), "recall": round(lr_acc_s["recall"], 4)},
                    "llm_paths_only_T3": {"map": round(t3_rank["map"], 4), "f1": round(t3_acc["f1"], 4)},
                    "map_training_free_minus_logistic": {"difference": diff, "ci": ci}},
           "leaderboard": {"by_map": M.place(s_rank["map"], 1), "by_f1": M.place(s_acc["f1"], 2)},
           "D3_holds": bool(ci[0] > -0.05), "look_at_test": 7}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
