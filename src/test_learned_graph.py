"""Learned evidence graph, phase 1: source weights learned from data and abstention on conflict, against the earlier approaches. No model
calls: it reuses the stored reader answers, status tags and records. Rules set before any scoring.

Edges per admission and item (the same graph as test_kg_rules.py): reader (plain yes), note_window (status record: a patient, affirmed
mention in the task's window), note_negated (a patient, negated mention) and record (the task's record edge in its window).
Two learned variants:
  LG-U  unsupervised: a Dawid-Skene latent model per task over reader, note_window and record (note_negated enters as the complement of a
        negative signal: a source that says yes unless a negated mention exists) learns each source's sensitivity and specificity from
        agreement alone. Answer: the posterior of the item being present. No truth is used anywhere.
  LG-S  supervised by a hospital source on other admissions: logistic regression over the four edges and the item, fit on the development
        admissions with their held-out truth as the label, then applied unchanged to the evaluation admissions. This asks how much a hospital
        could gain by learning weights from its own administration data.
  Answer yes if the probability is at least 0.5. Abstain if it lies strictly between 0.2 and 0.8 (fixed before scoring), which in practice
  is where sources disagree; a pair with no edges at all gets a probability near the item's base rate and is answered, not abstained.
Splits: development = the 50 pilot notes plus a seeded half of the eligible proxy admissions; evaluation = the other half, scored once.
Comparators on the evaluation half: reader, records alone, kg_rules (fixed OR) and the timeline LLM (N only).
Decisions per task, 95% CI above 0 resampling admissions: M1 LG beats records alone on balanced accuracy; M2 LG beats kg_rules;
M3 abstention helps: the error rate of the forced answer among abstained pairs exceeds that among answered pairs. Also reported: accuracy,
precision, recall, F1, wrong share of yes, coverage, and the risk at 80/90/95% coverage.
Phase 2 (not here): learn the entity-to-item links that replace the fixed ontology.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.kg import balanced_accuracy, dawid_skene
from mds_pilot.reader import RULES_VERSION
from run_reader import request_key
import test_kg_rules as KG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/learned_graph"
SEED = 20261103
LOW, HIGH = .2, .8
EDGES = ["reader", "note_window", "note_negated", "record"]


def edge_rows():
    frame = KG.pairs()
    status, reader = KG.stored_status(), KG.reader_answers()
    timeline = {}
    path = ROOT / "pilot_private/runs/proxy500/robustness.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        r = json.loads(line)
        if r.get("stage") == "timeline":
            timeline[(r["note"], r["item"])] = r["parsed"]["answer"] == "yes"
    rows = []
    for t, a, item, truth, record in frame.itertuples(index=False):
        mentions = status.get((a, item))
        if mentions is None:
            continue
        patient = [m for m in mentions if m["subject"] == "patient"]
        rows.append({"task": t, "admission": a, "item": item, "truth": bool(truth), "record": bool(record),
                     "reader": reader.get((a, item), (False, False))[0],
                     "note_window": any(m["assertion"] == "affirmed" and m["timing"] in KG.WINDOWS[t] for m in patient),
                     "note_negated": any(m["assertion"] == "negated" for m in patient),
                     "kg_rules": KG.decide(mentions, record, KG.WINDOWS[t])[0] == "yes", "timeline": timeline.get((a, item))})
    return pd.DataFrame(rows)


def split(df):
    admissions = sorted(df["admission"].unique())
    rng = np.random.default_rng(SEED)
    dev = set(rng.choice(admissions, size=len(admissions) // 2, replace=False))
    return df[df["admission"].isin(dev)], df[~df["admission"].isin(dev)]


def lg_unsupervised(task_frame):
    x = np.column_stack([task_frame["reader"], task_frame["note_window"], ~task_frame["note_negated"].to_numpy(), task_frame["record"]]).astype(float)
    post, sens, spec, prev = dawid_skene(x)
    return post, {"sensitivity": dict(zip(["reader", "note_window", "not_negated", "record"], np.round(sens, 3).tolist())),
                  "specificity": dict(zip(["reader", "note_window", "not_negated", "record"], np.round(spec, 3).tolist())), "prevalence": round(prev, 3)}


def design(frame, items):
    x = frame[EDGES].astype(float)
    for item in items:
        x[f"item={item}"] = (frame["item"] == item).astype(float)
    return x.to_numpy()


def risk_at(truth, prob, coverages=(.8, .9, .95)):
    confidence = np.abs(prob - .5)
    order = np.argsort(-confidence)
    wrong = ((prob >= .5) != truth)[order]
    return {f"{int(c * 100)}%": round(float(wrong[:max(1, int(round(c * len(wrong))))].mean()), 4) for c in coverages}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = edge_rows()
    dev, ev = split(df)
    result = {"split": {"dev_admissions": int(dev["admission"].nunique()), "eval_admissions": int(ev["admission"].nunique())}}
    for task in sorted(df["task"].unique()):
        d, e = dev[dev["task"] == task].copy(), ev[ev["task"] == task].copy()
        items = sorted(df.loc[df["task"] == task, "item"].unique())
        # LG-U uses no labels, so it is fit on the evaluation pairs themselves (only agreement between sources is used).
        e["p_u"], weights_u = lg_unsupervised(e)
        model = LogisticRegression(max_iter=1000, C=1.0).fit(design(d, items), d["truth"])
        e["p_s"] = model.predict_proba(design(e, items))[:, 1]
        res = {"pairs": len(e), "admissions": int(e["admission"].nunique()), "true": int(e["truth"].sum()), "lg_u_source_estimates": weights_u,
               "lg_s_edge_coefficients": dict(zip(EDGES, np.round(model.coef_[0][:len(EDGES)], 3).tolist()))}
        for name, col in (("LG-U", "p_u"), ("LG-S", "p_s")):
            e[name] = e[col] >= .5
            answered = e[(e[col] <= LOW) | (e[col] >= HIGH)]
            abstained = e[(e[col] > LOW) & (e[col] < HIGH)]
            res[name] = {"all": KG.score(e["truth"], e[name]), "answered": KG.score(answered["truth"], answered[name]) if len(answered) else None,
                         "coverage": round(len(answered) / len(e), 4), "risk_at_coverage": risk_at(e["truth"].to_numpy(), e[col].to_numpy())}
            res[name]["M1_minus_records"] = KG.boot_ci(e, lambda x, n=name: balanced_accuracy(x["truth"], x[n]) - balanced_accuracy(x["truth"], x["record"]))
            res[name]["M2_minus_kg_rules"] = KG.boot_ci(e, lambda x, n=name: balanced_accuracy(x["truth"], x[n]) - balanced_accuracy(x["truth"], x["kg_rules"]))
            if len(abstained) and len(answered):
                err = lambda x, n=name: float((x[n] != x["truth"]).mean()) if len(x) else float("nan")
                res[name]["M3_abstained_minus_answered_error"] = KG.boot_ci(
                    e, lambda x, n=name, c=col: err(x[(x[c] > LOW) & (x[c] < HIGH)], n) - err(x[(x[c] <= LOW) | (x[c] >= HIGH)], n))
        res["comparators"] = {m: KG.score(e["truth"], e[m]) for m in ("reader", "record", "kg_rules")}
        if task == "N_window" and e["timeline"].notna().all():
            res["comparators"]["timeline_llm"] = KG.score(e["truth"], e["timeline"].astype(bool))
        result[task] = res
    (OUT / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
