"""Port the learned graph from CodiEsp to MIMIC (A) and fix the label-free version (B). No model calls. Rules set before scoring.

A. Same model and clue roles as the CodiEsp learned graph (logistic regression over clue edges, fit on development labels, applied once):
   first AI reading (reader yes), second AI reading (status record in window; for N also the timeline LLM), evidence location (a cited
   sentence in Discharge Medications or Discharge Instructions), record in the window, record during the stay, negated patient mention,
   related items (sum over the admission's other items with a record edge of P(this item's record | that item's record), learned from
   development records only, no truth), and the item itself. Same seeded halves as test_learned_graph.py; fit on development, scored
   once on evaluation.
   G1: the port beats phase 1 (LG-S) on balanced accuracy; G2: it beats records alone; each with a 95% CI above 0, resampling admissions.
   Also F1, precision, recall, wrong share of yes, and the deferral check (precision of yes at probability >= 0.8 against 0.2-0.8).
B. Label-free latent model (Dawid-Skene) with the two AI readings merged into one note source (reader or status), since they read the same
   sentences. It needs at least three conditionally independent sources, so it runs where three exist:
   O_icu: note, order record, billing record. N and UTI have only note and records and are reported as not identifiable.
   U1: on O_icu it beats records alone (order or billing) on balanced accuracy, CI above 0. Fit on the evaluation half without labels.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.kg import balanced_accuracy, dawid_skene, drug_pattern
import test_kg_rules as KG
import test_learned_graph as LG
from test_window_graph import note_answers, records, sections

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/graph_port"
DISCHARGE = {"Discharge Medications", "Discharge Instructions"}


def extra_clues(df):
    """Adds evidence location, stay-level records, and separate order/billing edges for O."""
    answers, section_of = note_answers(), sections()
    df["evidence_discharge"] = [any(section_of.get((a, s)) in DISCHARGE for s in answers.get((a, i), (False, []))[1])
                                for a, i in zip(df["admission"], df["item"])]
    edges = pd.read_parquet(ROOT / "pilot_private/kg/edges.parquet")
    has = lambda src: {(s, i) for s, i in zip(edges.loc[edges["source"] == src, "subject"], edges.loc[edges["source"] == src, "item"])}
    order, billing, rx_any = has("order"), has("billing"), has("prescription")
    df["order"] = [(a, i) in order for a, i in zip(df["admission"], df["item"])]
    df["billing"] = [(a, i) in billing for a, i in zip(df["admission"], df["item"])]
    df["record_stay"] = [((a, i) in rx_any) if t == "N_window" else r for t, a, i, r in zip(df["task"], df["admission"], df["item"], df["record"])]
    df["timeline"] = df["timeline"].fillna(False).astype(bool)
    return df


def related_items(dev, frame):
    """P(record for item | record for other item), from development records; summed over the admission's other recorded items."""
    rec = dev.pivot_table(index="admission", columns="item", values="record", aggfunc="max").fillna(False).astype(bool)
    items = rec.columns
    cond = {(j, i): (rec[j] & rec[i]).sum() / rec[j].sum() for j in items for i in items if i != j and rec[j].sum() >= 3}
    by_adm = frame.groupby("admission").apply(lambda g: set(g.loc[g["record"], "item"]))
    return [sum(cond.get((j, i), 0.0) for j in by_adm.get(a, set()) if j != i) for a, i in zip(frame["admission"], frame["item"])]


def features(frame, items, task):
    cols = ["reader", "note_window", "evidence_discharge", "record", "record_stay", "note_negated", "related"]
    if task == "N_window":
        cols.append("timeline")
    x = frame[cols].astype(float)
    for item in items:
        x[f"item={item}"] = (frame["item"] == item).astype(float)
    return x.to_numpy(), cols


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = extra_clues(LG.edge_rows())
    dev, ev = LG.split(df)
    out = {}
    for task in sorted(df["task"].unique()):
        d, e = dev[dev["task"] == task].copy(), ev[ev["task"] == task].copy()
        d["related"], e["related"] = related_items(d, d), related_items(d, e)
        items = sorted(df.loc[df["task"] == task, "item"].unique())
        xd, cols = features(d, items, task)
        xe, _ = features(e, items, task)
        port = LogisticRegression(max_iter=2000).fit(xd, d["truth"])
        e["p"] = port.predict_proba(xe)[:, 1]
        e["port"] = e["p"] >= .5
        phase1 = LogisticRegression(max_iter=1000).fit(LG.design(d, items), d["truth"])
        e["phase1"] = phase1.predict_proba(LG.design(e, items))[:, 1] >= .5
        sure, unsure = e[(e["p"] >= .8)], e[(e["p"] > .2) & (e["p"] < .8)]
        res = {"pairs": len(e), "admissions": int(e["admission"].nunique()),
               "scores": {m: KG.score(e["truth"], e[m]) for m in ("reader", "record", "phase1", "port")},
               "weights": dict(zip(cols, np.round(port.coef_[0][:len(cols)], 3).tolist())),
               "G1_port_minus_phase1": KG.boot_ci(e, lambda x: balanced_accuracy(x["truth"], x["port"]) - balanced_accuracy(x["truth"], x["phase1"])),
               "G2_port_minus_records": KG.boot_ci(e, lambda x: balanced_accuracy(x["truth"], x["port"]) - balanced_accuracy(x["truth"], x["record"])),
               "deferral": {"share_deferred": round(len(unsure) / len(e), 4), "precision_yes_p80": round(float(sure["truth"].mean()), 4) if len(sure) else None,
                            "precision_deferred_if_yes": round(float(unsure["truth"].mean()), 4) if len(unsure) else None}}
        if task == "O_icu":
            note = (e["reader"] | e["note_window"]).astype(float)
            x = np.column_stack([note, e["order"].astype(float), e["billing"].astype(float)])
            post, sens, spec, prev = dawid_skene(x)
            e["label_free"] = post >= .5
            e["records_or"] = e["order"] | e["billing"]
            res["B_label_free"] = {"score": KG.score(e["truth"], e["label_free"]), "records_or": KG.score(e["truth"], e["records_or"]),
                                   "estimated_sensitivity": dict(zip(["note", "order", "billing"], np.round(sens, 3).tolist())),
                                   "estimated_specificity": dict(zip(["note", "order", "billing"], np.round(spec, 3).tolist())),
                                   "U1_minus_records": KG.boot_ci(e, lambda x: balanced_accuracy(x["truth"], x["label_free"]) - balanced_accuracy(x["truth"], x["records_or"]))}
        else:
            res["B_label_free"] = "not identifiable: only two independent sources (note, records)"
        out[task] = res
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
