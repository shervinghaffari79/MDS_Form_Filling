"""Score every method against the blinded MIMIC review labels (make_review_tool.py). Rules set before any label exists.

Input: /home/user/mimic/review/method_outputs.json and every labels file in /home/user/mimic/review/labels/*.json.
Truth: Present = applies, Absent = does not; Can't tell is left out and counted. With two reviewers on the same item, their agreement
  (Cohen's kappa) is reported and the item is scored only where they agree.
Per method (notes, notes+v, records, graph, reader), on Section I, O, P and overall:
  precision = labelled-present among the items the method filled;
  recall_candidates = filled-and-present / all present candidates (every method's fills are candidates);
  recall_estimated also counts the misses shared by all methods: the present rate among the 5 random unfilled items per admission
    times the number of unfilled items, added to the denominator;
  F1 from precision and recall_estimated. 95% CI by resampling admissions (2,000 draws).
Outputs results/sweetspot/p2_review_scores.json with counts and scores only.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import admission_graph as A  # noqa: E402

REVIEW = Path("/home/user/mimic/review")
OUT = ROOT / "results/sweetspot/p2_review_scores.json"
METHODS = ("notes", "notes+v", "records", "graph", "reader")
N_ITEMS = len(A.ITEMS["items"])
SECTION = {i["code"]: i["section"] for i in A.ITEMS["items"]}


def load_truth():
    files = sorted((REVIEW / "labels").glob("*.json"))
    by_item = defaultdict(dict)
    for f in files:
        j = json.loads(f.read_text())
        for r in j["labels"]:
            by_item[(r["note_key"], r["item"])][j["reviewer"]] = r["label"]
    truth, unclear, pairs = {}, 0, []
    for k, votes in by_item.items():
        vals = [v for v in votes.values() if v != "unclear"]
        if len(votes) >= 2:
            a, b = list(votes.values())[:2]
            pairs.append((a, b))
        if not vals or len(set(vals)) > 1:
            unclear += 1
            continue
        truth[k] = vals[0] == "present"
    return truth, unclear, pairs, [f.name for f in files]


def kappa(pairs):
    pairs = [(a, b) for a, b in pairs if "unclear" not in (a, b)]
    if not pairs:
        return None
    po = np.mean([a == b for a, b in pairs])
    pa = np.mean([a == "present" for a, _ in pairs])
    pb = np.mean([b == "present" for _, b in pairs])
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round(float((po - pe) / (1 - pe)), 3) if pe < 1 else None


def scores(rows, truth, method, section, adms):
    def one(adm_set):
        draws = defaultdict(int)
        for a in adm_set:  # an admission drawn twice in a resample counts twice everywhere
            draws[a] += 1
        rs = [r for r in rows for _ in range(draws.get(r["note_key"], 0)) if (section == "all" or SECTION[r["item"]] == section)
              and (r["note_key"], r["item"]) in truth]
        filled = [r for r in rs if r[method]]
        tp = sum(truth[(r["note_key"], r["item"])] for r in filled)
        cand_present = sum(truth[(r["note_key"], r["item"])] for r in rs if not r["sampled_negative"])
        neg = [r for r in rs if r["sampled_negative"]]
        neg_rate = (sum(truth[(r["note_key"], r["item"])] for r in neg) / len(neg)) if neg else 0.0
        n_items = N_ITEMS if section == "all" else sum(1 for s in SECTION.values() if s == section)
        unfilled = sum(n_items - len({r["item"] for r in rows if r["note_key"] == a and not r["sampled_negative"]
                                       and (section == "all" or SECTION[r["item"]] == section)}) for a in adm_set)
        missed = neg_rate * unfilled
        p = tp / len(filled) if filled else float("nan")
        rc = tp / cand_present if cand_present else float("nan")
        re = tp / (cand_present + missed) if cand_present + missed else float("nan")
        f = 2 * p * re / (p + re) if p == p and re == re and p + re else float("nan")
        return p, rc, re, f, len(filled), tp

    point = one(adms)
    rng = np.random.default_rng(0)
    boots = [one(list(rng.choice(adms, len(adms)))) for _ in range(2000)]
    ci = lambda i: [round(float(np.nanquantile([b[i] for b in boots], q)), 3) for q in (.025, .975)]
    return {"precision": round(point[0], 3), "precision_ci": ci(0), "recall_candidates": round(point[1], 3),
            "recall_estimated": round(point[2], 3), "recall_estimated_ci": ci(2), "f1": round(point[3], 3), "f1_ci": ci(3),
            "filled": point[4], "true_positive": point[5]}


def main():
    rows = json.loads((REVIEW / "method_outputs.json").read_text())
    truth, unclear, pairs, files = load_truth()
    adms = sorted({r["note_key"] for r in rows})
    out = {"label_files": len(files), "labelled_items": len(truth), "unclear_or_disagreed": unclear, "candidates": len(rows),
           "inter_reviewer_kappa": kappa(pairs), "double_labelled": len(pairs),
           "present_rate_among_random_unfilled": round(float(np.mean([truth[(r["note_key"], r["item"])] for r in rows
                                                                   if r["sampled_negative"] and (r["note_key"], r["item"]) in truth] or [0])), 3),
           "methods": {m: {s: scores(rows, truth, m, s, adms) for s in ("all", "I", "O", "P")} for m in METHODS}}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "methods"}, indent=1))
    for m in METHODS:
        a = out["methods"][m]["all"]
        print(f"{m:8s} P {a['precision']} {a['precision_ci']}  R(est) {a['recall_estimated']} {a['recall_estimated_ci']}  F1 {a['f1']} {a['f1_ci']}")


if __name__ == "__main__":
    main()
