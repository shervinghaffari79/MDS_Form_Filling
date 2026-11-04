"""Benchmark the knowledge graph against held-out hospital sources. Writes aggregates to results/kg_benchmark.

MIMIC has no MDS forms, so each task holds out a source that is never a graph input and checks answers against it:
  O_icu        transfusion, ventilation, non-invasive ventilation, dialysis   inputs: note, order, billing      check: ICU events     who: ICU stays
  N_emar       11 N0415 drug classes                                         inputs: note, prescription        check: administration who: eMAR coverage
  I_diabetes   diabetes                                                      inputs: note, billing, insulin or
                                                                                     oral drug (cross-item link)    check: HbA1c >= 6.5   who: HbA1c tested
  I_uti        urinary tract infection                                       inputs: note, billing             check: urine organism who: culture sent
Stage 1, --develop: on the 50 pilot notes, pick each task's fusion method by balanced accuracy and freeze it. Ties within 0.005 go to
the simpler method; a task with fewer than 10 true pilot cases defaults to OR. Stage 2, --evaluate: score every method on the
500 proxy notes. Decision set before stage 2: the graph adds value on a task only if its frozen method minus note alone has a CI above
0 (resampling admissions). The latent model's label-free source estimates are called useful if their mean absolute error against the
held-out check is at most 0.10.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS
from mds_pilot.kg import FUSIONS, balanced_accuracy, dawid_skene, fuse, source_matrix
from build_kg import note_answers

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private/kg"
OUT = ROOT / "results/kg_benchmark"
FROZEN = OUT / "frozen_choice.json"
BOOT = 2000
TIE = .005
MIN_TRUE = 10
N_ITEMS = [i.id for i in ITEMS if i.section == "N"]
TASKS = {"O_icu": {"items": ["O_transfusion", "O_invasive_vent", "O_noninvasive_vent", "O_dialysis"], "inputs": ["note", "order", "billing"],
                   "truth": "icu_event", "eligible": "icu_stay"},
         "N_emar": {"items": N_ITEMS, "inputs": ["note", "prescription"], "truth": "administration", "eligible": "emar_coverage"},
         "I_diabetes": {"items": ["I_diabetes"], "inputs": ["note", "billing", "prescription_for_diagnosis"], "truth": "lab_positive", "eligible": "lab_tested"},
         "I_uti": {"items": ["I_uti"], "inputs": ["note", "billing"], "truth": "culture_positive", "eligible": "culture_sent"}}


def task_frame(edges, answers, sample_keys, task):
    spec = TASKS[task]
    eligible_edges = edges[edges["source"] == spec["eligible"]]
    if eligible_edges["item"].isna().all():
        eligible = set(eligible_edges["subject"])
        pairs = [(a, i) for a in sorted(eligible & sample_keys) for i in spec["items"]]
    else:
        pairs = sorted({(a, i) for a, i in zip(eligible_edges["subject"], eligible_edges["item"]) if a in sample_keys and i in spec["items"]})
    frame = source_matrix(edges, answers, pairs, spec["inputs"] + [spec["truth"]])
    return frame.rename(columns={spec["truth"]: "truth"})


def methods_for(task):
    n = len(TASKS[task]["inputs"])
    return [m for m in FUSIONS if n >= 3 or m not in ("majority", "latent")]


def score(frame, task):
    rows = {}
    for method in methods_for(task):
        answer = fuse(frame, TASKS[task]["inputs"], method)
        truth = frame["truth"].astype(bool)
        rows[method] = {"balanced": balanced_accuracy(truth, answer), "sensitivity": float(answer[truth].mean()), "specificity": float((~answer[~truth]).mean()),
                        "yes_rate": float(answer.mean())}
    return rows


def contrast_ci(frame, task, a, b, seed=0):
    inputs = TASKS[task]["inputs"]
    answers = {m: fuse(frame, inputs, m).to_numpy() for m in (a, b)}
    truth = frame["truth"].astype(bool).to_numpy()
    groups = [np.flatnonzero(frame["admission"].to_numpy() == s) for s in frame["admission"].unique()]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        values.append(balanced_accuracy(truth[idx], answers[a][idx]) - balanced_accuracy(truth[idx], answers[b][idx]))
    low, high = np.nanquantile(values, [.025, .975])
    return round(balanced_accuracy(truth, answers[a]) - balanced_accuracy(truth, answers[b]), 3), round(float(low), 3), round(float(high), 3)


def choose(scores, true_cases):
    if true_cases < MIN_TRUE:
        return "or", f"fewer than {MIN_TRUE} true pilot cases"
    best = max(v["balanced"] for v in scores.values())
    for method in scores:  # FUSIONS order = simplest first
        if scores[method]["balanced"] >= best - TIE:
            return method, "best pilot balanced accuracy"


def reliability(frame, task):
    """Label-free sensitivity and specificity per source from the latent model, against the same measured on the held-out check."""
    inputs = TASKS[task]["inputs"]
    rows = []
    for item_id, group in frame.groupby("item"):
        truth = group["truth"].astype(bool).to_numpy()
        if truth.sum() < 5 or (~truth).sum() < 5:
            continue
        _, sens, spec, prev = dawid_skene(group[inputs].to_numpy())
        for j, source in enumerate(inputs):
            x = group[source].astype(bool).to_numpy()
            rows.append({"task": task, "item": item_id, "source": source, "estimated_sensitivity": round(float(sens[j]), 3),
                         "measured_sensitivity": round(float(x[truth].mean()), 3), "estimated_specificity": round(float(spec[j]), 3),
                         "measured_specificity": round(float((~x[~truth]).mean()), 3), "estimated_prevalence": round(prev, 3),
                         "measured_prevalence": round(float(truth.mean()), 3)})
    return rows


def load():
    edges = pd.read_parquet(PRIVATE / "edges.parquet")
    admissions = pd.read_parquet(PRIVATE / "admissions.parquet")
    keys = {s: set(admissions.loc[admissions["sample"] == s, "note_key"]) for s in ("pilot", "proxy")}
    return edges, note_answers(), keys


def develop():
    OUT.mkdir(parents=True, exist_ok=True)
    edges, answers, keys = load()
    frozen = {}
    for task in TASKS:
        frame = task_frame(edges, answers, keys["pilot"], task)
        scores = score(frame, task)
        method, why = choose(scores, int(frame["truth"].sum()))
        frozen[task] = {"method": method, "reason": why, "pilot_pairs": len(frame), "pilot_true": int(frame["truth"].sum()),
                        "pilot_scores": {m: round(v["balanced"], 3) for m, v in scores.items()}}
    FROZEN.write_text(json.dumps(frozen, indent=1))
    print(json.dumps(frozen, indent=1))


def evaluate():
    if not FROZEN.exists():
        sys.exit("Run --develop first: the method for each task must be frozen on the pilot.")
    frozen = json.loads(FROZEN.read_text())
    edges, answers, keys = load()
    table, contrasts, reliable = [], [], []
    for task in TASKS:
        frame = task_frame(edges, answers, keys["proxy"], task)
        chosen = frozen[task]["method"]
        for method, v in score(frame, task).items():
            table.append({"task": task, "method": method, "frozen": method == chosen, "admissions": int(frame["admission"].nunique()), "pairs": len(frame),
                          "true_cases": int(frame["truth"].sum()), **{k: round(x, 3) for k, x in v.items()}})
        for other in [m for m in methods_for(task) if m != chosen]:
            d, low, high = contrast_ci(frame, task, chosen, other)
            contrasts.append({"task": task, "frozen": chosen, "versus": other, "difference": d, "ci_low": low, "ci_high": high})
        if len(TASKS[task]["inputs"]) >= 3:
            reliable += reliability(frame, task)
        for item_id, group in frame.groupby("item"):
            for method in methods_for(task):
                answer = fuse(group, TASKS[task]["inputs"], method)
                table.append({"task": task, "item": item_id, "method": method, "frozen": method == chosen, "pairs": len(group), "true_cases": int(group["truth"].sum()),
                              "balanced": round(balanced_accuracy(group["truth"], answer), 3)})
    table, contrasts, reliable = pd.DataFrame(table), pd.DataFrame(contrasts), pd.DataFrame(reliable)
    pooled = table[table["item"].isna()].drop(columns="item")
    by_item = table[table["item"].notna()]
    decisions = {}
    for task in TASKS:
        row = contrasts[(contrasts["task"] == task) & (contrasts["versus"] == "note")]
        decisions[task] = {"frozen": frozen[task]["method"], "minus_note": None if row.empty else row.iloc[0][["difference", "ci_low", "ci_high"]].tolist(),
                           "adds_value": bool(not row.empty and row.iloc[0]["ci_low"] > 0)}
    errors = []
    if not reliable.empty:
        errors = pd.concat([(reliable["estimated_sensitivity"] - reliable["measured_sensitivity"]).abs(),
                            (reliable["estimated_specificity"] - reliable["measured_specificity"]).abs()])
    summary = {"decisions": decisions, "latent_reliability_mae": round(float(np.mean(errors)), 3) if len(errors) else None,
               "latent_reliability_useful": bool(len(errors) and np.mean(errors) <= .10)}
    pooled.to_csv(OUT / "benchmark_pooled.csv", index=False)
    by_item.to_csv(OUT / "benchmark_by_item.csv", index=False)
    contrasts.to_csv(OUT / "contrasts.csv", index=False)
    reliable.to_csv(OUT / "source_reliability.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(pooled.to_string(index=False)); print(contrasts.to_string(index=False)); print(reliable.to_string(index=False))
    print(by_item.pivot_table(index=["task", "item"], columns="method", values="balanced").round(3).to_string())
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--develop", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()
    if args.develop:
        develop()
    if args.evaluate:
        evaluate()
