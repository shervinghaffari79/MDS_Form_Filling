"""Weighted estimates, real conflicting text and group effects, from stored results. Writes aggregates to results/offline_checks."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS
from mds_pilot.metrics import cohen_kappa
from mds_pilot.reader import answer_is_yes
from mds_pilot.retrieval import bm25_scores, candidate_rows, lexicon_scores, note_index_from

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private"
OUT = ROOT / "results/offline_checks"
SEED = 20261101
BOOT = 2000
CUES = re.compile(r"\b(held|hold|holding|discontinu\w*|stopp\w*|resum\w*|restart\w*|ruled? out|no evidence|cancel\w*|prior|history of|h/o|hx of|"
                  r"allerg\w*|not (given|started|taking|continued)|was on|previously)\b", re.I)
GROUP_SIZES = (3, 10, 25)
GROUPS_PER_SIZE = 5


def weighted_rate(values, weights):
    return float(np.average(values, weights=weights))


def stratified_ci(frame, value, seed=0):
    """Resample notes within each stratum, as the pilot was drawn."""
    rng = np.random.default_rng(seed)
    by_note = frame.groupby("note").agg(value=(value, "mean"), weight=("weight", "first"), stratum=("stratum", "first"))
    out = []
    groups = [g for _, g in by_note.groupby("stratum")]
    for _ in range(BOOT):
        sample = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])
        out.append(weighted_rate(sample["value"], sample["weight"]))
    low, high = np.quantile(out, [.025, .975])
    return round(float(low), 3), round(float(high), 3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(x) for x in (PRIVATE / "reader_runs.jsonl").read_text().splitlines()]
    labels = pd.read_parquet(PRIVATE / "pilot_labels.parquet").set_index("note_key")
    decisions = pd.read_csv(ROOT / "results/mds_output/decisions_no_text.csv")
    decisions["weight"] = decisions["note"].map(labels["weight"])
    decisions["stratum"] = decisions["note"].map(labels["stratum"])
    decisions["yes"] = decisions["answer"] == "yes"
    decisions["section"] = decisions["item"].str[0]
    has_table = decisions[decisions["table_status"] != "no_table_value"].copy()
    has_table["flag"] = has_table["table_status"] != "agrees"

    weighted = {"yes_rate": {}, "flag_rate": {}}
    for section, group in decisions.groupby("section"):
        weighted["yes_rate"][section] = {"unweighted": round(float(group["yes"].mean()), 3), "weighted": round(weighted_rate(group["yes"], group["weight"]), 3),
                                         "ci": stratified_ci(group, "yes")}
    for section, group in has_table.groupby("section"):
        weighted["flag_rate"][section] = {"unweighted": round(float(group["flag"].mean()), 3), "weighted": round(weighted_rate(group["flag"], group["weight"]), 3),
                                          "ci": stratified_ci(group, "flag")}
    by_field = decisions.groupby("item", sort=False).apply(lambda g: pd.Series({"unweighted_yes": g["yes"].mean(), "weighted_yes": weighted_rate(g["yes"], g["weight"])}), include_groups=False).round(3)

    sent = pd.read_parquet(PRIVATE / "pilot_sentences.parquet").sort_values(["note_key", "sent_id"]).reset_index(drop=True)
    lex = lexicon_scores(sent["text"].tolist()) > 0
    position = {(n, s): i for i, (n, s) in enumerate(zip(sent["note_key"], sent["sent_id"]))}
    cue = np.array([bool(CUES.search(t)) for t in sent["text"]])
    cands = pd.read_parquet(ROOT / "results/modal_pilot_ids/candidates_pilot.parquet")
    picks = {key: sorted(g["sent_id"]) for key, g in cands.groupby(["note_key", "item"])}
    answers = {(r["arm"], r["note"], r["item"], r["repeat"]): answer_is_yes(r["arm"], r["parsed"]) for r in runs if r["stage"] == "real"}
    rows = []
    for j, item in enumerate(ITEMS):
        for note in labels.index:
            at = [position[(note, s)] for s in picks[(note, item.id)]]
            mention = lex[at, j]
            cued = mention & cue[at]
            kind = "no_mention" if not mention.any() else "conflict_like" if cued.any() and (mention & ~cue[at]).any() else "cue_only" if cued.any() else "plain_mention"
            plain, status = answers[("plain", note, item.id, 0)], answers[("status", note, item.id, 0)]
            again = answers.get(("plain", note, item.id, 1))
            rows.append({"note": note, "item": item.id, "kind": kind, "plain": plain, "status": status, "arms_differ": plain != status,
                         "rerun_changed": None if again is None else again != plain, "table": int(labels.loc[note, item.id]) if item.has_proxy else None})
    conflict = pd.DataFrame(rows)
    table_rows = []
    for kind, group in conflict.groupby("kind"):
        has = group.dropna(subset=["table"])
        rerun = group.dropna(subset=["rerun_changed"])
        table_rows.append({"kind": kind, "pairs": len(group), "plain_yes": round(float(group["plain"].mean()), 3), "arms_differ": round(float(group["arms_differ"].mean()), 3),
                           "rerun_pairs": len(rerun), "rerun_changed": round(float(rerun["rerun_changed"].astype(bool).mean()), 3) if len(rerun) else None,
                           "table_agree": round(float((has["plain"].astype(int) == has["table"].astype(int)).mean()), 3),
                           "kappa": round(float(cohen_kappa(has["table"].astype(int).to_numpy(), has["plain"].astype(int).to_numpy())), 3)})
    conflict_table = pd.DataFrame(table_rows)
    conflict[["note", "item", "kind"]].to_parquet(OUT / "pair_kinds_no_text.parquet")

    cached = sorted((PRIVATE / "runs/verify_pilot50").glob("encoder_scores_*.npz"))[0]
    with np.load(cached) as saved:
        enc = [saved["bge"], saved["biolord"]]
    keys, ids = sent["note_key"].to_numpy(), sent["sent_id"].to_numpy()
    texts = sent["text"].tolist()
    base = {}
    for r in candidate_rows(keys, ids, [bm25_scores(texts, "syn"), *enc], note_index_from(keys), 20):
        base.setdefault((r["note_key"], r["item"]), set()).add(r["sent_id"])
    rng = np.random.default_rng(SEED)
    notes = sorted(labels.index)
    group_rows = []
    for size in GROUP_SIZES:
        for _ in range(GROUPS_PER_SIZE):
            chosen = set(rng.choice(notes, size, replace=False))
            mask = np.array([k in chosen for k in keys])
            subset_texts = [t for t, m in zip(texts, mask) if m]
            arrays = [bm25_scores(subset_texts, "syn"), enc[0][mask], enc[1][mask]]
            found = {}
            for r in candidate_rows(keys[mask], ids[mask], arrays, note_index_from(keys[mask]), 20):
                found.setdefault((r["note_key"], r["item"]), set()).add(r["sent_id"])
            overlap = [len(found[k] & base[k]) / len(base[k]) for k in found]
            group_rows.append({"group_size": size, "pairs": len(found), "identical": float(np.mean([found[k] == base[k] for k in found])),
                               "mean_overlap": float(np.mean(overlap)), "min_overlap": float(np.min(overlap))})
    group_table = pd.DataFrame(group_rows).groupby("group_size").agg(pairs=("pairs", "sum"), identical=("identical", "mean"), mean_overlap=("mean_overlap", "mean"),
                                                                     min_overlap=("min_overlap", "min")).round(3)

    by_field.to_csv(OUT / "weighted_by_field.csv")
    conflict_table.to_csv(OUT / "real_conflict_kinds.csv", index=False)
    group_table.to_csv(OUT / "group_effect.csv")
    (OUT / "weighted.json").write_text(json.dumps(weighted, indent=1))
    pd.set_option("display.width", 180, "display.max_rows", 50)
    print(json.dumps(weighted, indent=1)); print(by_field.to_string()); print(conflict_table.to_string()); print(group_table.to_string())


if __name__ == "__main__":
    main()
