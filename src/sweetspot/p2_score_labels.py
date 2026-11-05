"""Score the systems against complete labels from the web UI (webui/). Rules set before any label exists.

Input: the labels database written by webui/server.py (default /home/user/mimic/review/labels.sqlite) and the pilot outputs.
Admissions used: those marked done by at least one reviewer (all 78 MDS 2.0 items labelled).
Reference: Present = 1, Absent = 0; Can't tell is left out. With several reviewers on an admission, an item is used only where every
  reviewer who labelled it (other than Can't tell) agrees; Cohen's kappa is reported for each pair of reviewers with overlap.
Systems: graph arms from the pilot (notes, notes+v, records, graph) and the reader where it was run. Precision, recall, F1 per
  section (I, O, P) and overall, 95% CI by resampling admissions. Because suggestions can anchor reviewers, scores are also
  reported separately for labels given with system answers hidden, shown after answering, and shown.
Output: results/sweetspot/p2_label_scores.json (counts and scores only).
"""
import argparse
import itertools
import json
import pickle
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import admission_graph as A  # noqa: E402

WORK = Path("/home/user/mimic/sweetspot")
OUT = ROOT / "results/sweetspot/p2_label_scores.json"
SECTION = {i["code"]: i["section"] for i in A.ITEMS["items"]}


def load(db, data):
    con = sqlite3.connect(db)
    labels = con.execute("SELECT reviewer, note_key, item, label, suggestions_mode FROM labels").fetchall()
    done = {r[0] for r in con.execute("SELECT DISTINCT note_key FROM done")}
    review = json.loads(Path(data).read_text())
    reader = {(a["key"], i["code"]): i["reader"]["answer"] for a in review["admissions"] for i in a["items"] if i["reader"] is not None}
    return labels, done, reader


def reference(labels, done):
    votes, modes = defaultdict(dict), defaultdict(set)
    for rev, key, item, lab, mode in labels:
        if key in done:
            votes[(key, item)][rev] = lab
            modes[(key, item)].add(mode)
    truth = {}
    for k, v in votes.items():
        vals = {x for x in v.values() if x != "unclear"}
        if len(vals) == 1:
            truth[k] = vals.pop() == "present"
    revs = sorted({r for v in votes.values() for r in v})
    kappas = {}
    for a, b in itertools.combinations(revs, 2):
        pairs = [(v[a], v[b]) for v in votes.values() if a in v and b in v and "unclear" not in (v[a], v[b])]
        if len(pairs) >= 20:
            po = np.mean([x == y for x, y in pairs])
            pa, pb = np.mean([x == "present" for x, _ in pairs]), np.mean([y == "present" for _, y in pairs])
            pe = pa * pb + (1 - pa) * (1 - pb)
            kappas[f"{a}~{b}"] = {"items": len(pairs), "kappa": round(float((po - pe) / (1 - pe)), 3) if pe < 1 else None}
    return truth, modes, kappas


def prf(truth, pred, keys):
    tp = sum(truth[k] and pred(k) for k in keys)
    fp = sum((not truth[k]) and pred(k) for k in keys)
    fn = sum(truth[k] and not pred(k) for k in keys)
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return p, r, (2 * p * r / (p + r) if p == p and r == r and p + r else float("nan"))


def score(truth, pred, keys):
    adms = sorted({k[0] for k in keys})
    by = defaultdict(list)
    for k in keys:
        by[k[0]].append(k)
    point = prf(truth, pred, keys)
    rng = np.random.default_rng(0)
    boots = [prf(truth, pred, [k for a in rng.choice(adms, len(adms)) for k in by[a]]) for _ in range(2000)]
    ci = lambda i: [round(float(np.nanquantile([b[i] for b in boots], q)), 3) for q in (.025, .975)]
    return {"precision": round(point[0], 3), "precision_ci": ci(0), "recall": round(point[1], 3), "recall_ci": ci(1),
            "f1": round(point[2], 3), "f1_ci": ci(2), "items": len(keys), "present": int(sum(truth[k] for k in keys))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/home/user/mimic/review/labels.sqlite")
    ap.add_argument("--data", default="/home/user/mimic/review/review_data.json")
    a = ap.parse_args()
    labels, done, reader = load(a.db, a.data)
    truth, modes, kappas = reference(labels, done)
    arms = pickle.loads((WORK / "p2_graphs.pkl").read_bytes())["arms"]
    systems = {m: (lambda k, m=m: k[1] in arms[m].get(k[0], {})) for m in ("notes", "notes+v", "records", "graph")}
    systems["reader"] = lambda k: reader.get(k, False)
    out = {"admissions_done": len(done), "items_with_reference": len(truth), "reviewer_kappa": kappas, "systems": {}}
    for name, pred in systems.items():
        keys = [k for k in truth if name != "reader" or k in reader]
        out["systems"][name] = {s: score(truth, pred, [k for k in keys if s == "all" or SECTION[k[1]] == s]) for s in ("all", "I", "O", "P")}
        out["systems"][name]["by_suggestion_mode"] = {
            m: score(truth, pred, [k for k in keys if modes[k] == {m}]) for m in ("hidden", "after", "shown")
            if any(modes[k] == {m} for k in keys)}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "systems"}, indent=1))
    for name, s in out["systems"].items():
        x = s["all"]
        print(f"{name:8s} P {x['precision']} {x['precision_ci']}  R {x['recall']} {x['recall_ci']}  F1 {x['f1']} {x['f1_ci']}  (n={x['items']})")


if __name__ == "__main__":
    main()
