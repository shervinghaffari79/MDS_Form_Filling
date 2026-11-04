"""Test the sentence graph and draft-and-verify without new reader calls. Writes aggregates to results/graph_gate.

Decisions set before the run:
Graph: invented two-sentence cases are added to real notes. Keep the graph only if, for deciding sentences that do not name the
item, having both sentences in context improves by at least 5 points with a CI above 0 (resampling items), and it adds at most
5 sentences per note-item pair on real notes.
Draft gate: skip the reader and answer no when no retrieved sentence names the item and the encoder score is below a threshold
set on the other half of the notes. Keep it only if it loses at most 2% of held-out yes answers, saves at least 30% of calls, and
at least 95% of invented paraphrases still pass.
Verifier: drop yes answers that fail the stored read-alone check. Keep it only if kappa with the tables rises by at least 0.01.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.graph import expand_with_followers
from mds_pilot.items import ITEMS
from mds_pilot.known_facts import FACTS, TIERS
from mds_pilot.metrics import cohen_kappa
from mds_pilot.reader import hard_cases
from mds_pilot.retrieval import bm25_scores, lexicon_scores, note_index_from, rrf, within_note_ranks
from run_pipeline import score_modal

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private"
OUT = ROOT / "results/graph_gate"
SEED = 20261101
K = 20
NOTES_PER_ITEM = 3
GAPS = (0, 1, 2)
WINDOWS = (1, 2)
BOOT = 2000
SECTION = "Brief Hospital Course"


def top_k_with_added(note_scores, added_scores, position_a, position_b):
    """Rank one item inside one note after adding two sentences; returns (mask in new reading order, new positions)."""
    n = note_scores.shape[1]
    order = list(range(n))
    order.insert(position_b, n + 1)
    order.insert(position_a, n)
    full = np.column_stack([note_scores, added_scores])[:, order]
    ranks = np.argsort(np.argsort(-full, axis=1, kind="stable"), axis=1, kind="stable") + 1
    fused = sum((full[c] > 0) / (60 + ranks[c]) for c in range(full.shape[0]))
    top = np.zeros(len(order), dtype=bool)
    top[np.argsort(-fused, kind="stable")[:K]] = True
    return top, order.index(n), order.index(n + 1), order


def ci(values, seed=0):
    values = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(values), (BOOT, len(values)))
    low, high = np.quantile(values[idx].mean(axis=1), [.025, .975])
    return round(float(low), 3), round(float(high), 3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sent = pd.read_parquet(PRIVATE / "pilot_sentences.parquet").sort_values(["note_key", "sent_id"]).reset_index(drop=True)
    texts = sent["text"].tolist()
    keys, sections = sent["note_key"].to_numpy(), sent["section"].to_numpy()
    index = note_index_from(keys)
    cached = sorted((PRIVATE / "runs/verify_pilot50").glob("encoder_scores_*.npz"))[0]
    with np.load(cached) as saved:
        real_enc = [saved["bge"], saved["biolord"]]

    invented = sorted({s for item in ITEMS for case in hard_cases(item).values() for s in case[:2]}
                      | {FACTS[item.id][TIERS.index("paraphrase")] for item in ITEMS})
    position = {s: i for i, s in enumerate(invented)}
    print(f"[progress] scoring {len(invented)} invented sentences", flush=True)
    inv_enc = score_modal(invented)
    bm = bm25_scores(texts + invented, "syn")
    real_bm, inv_bm = bm[:len(texts)], bm[len(texts):]
    real_lex, inv_lex = lexicon_scores(texts) > 0, lexicon_scores(invented) > 0
    print("[progress] scored; running graph test", flush=True)

    labels = pd.read_parquet(PRIVATE / "pilot_labels.parquet").set_index("note_key")
    runs = [json.loads(x) for x in (PRIVATE / "reader_runs.jsonl").read_text().splitlines()]
    plain = {(r["note"], r["item"]): r for r in runs if r["stage"] == "real" and r["arm"] == "plain" and r["repeat"] == 0}
    rng = np.random.default_rng(SEED)
    note_rows = {keys[i[0]]: i for i in index}
    trials = []
    for j, item in enumerate(ITEMS):
        silent = [n for n, rows in note_rows.items() if not real_lex[rows, j].any() and plain[(n, item.id)]["parsed"]["answer"] == "no"
                  and (sections[rows] == SECTION).sum() >= 4 and not (item.has_proxy and labels.loc[n, item.id] == 1)]
        for note in list(rng.permutation(silent)[:NOTES_PER_ITEM]):
            rows = note_rows[note]
            note_scores = np.vstack([real_bm[rows, j], real_enc[0][rows, j], real_enc[1][rows, j]])
            span = np.flatnonzero(sections[rows] == SECTION)
            for case, (opening, decider, _) in hard_cases(item).items():
                for gap in GAPS:
                    start = int(rng.integers(0, len(span) - gap))
                    a, b = int(span[start]), int(span[start + gap])  # gap real sentences end up between the two
                    added = np.array([[s[position[opening], j], s[position[decider], j]] for s in (inv_bm, inv_enc[0], inv_enc[1])])
                    top, pa, pb, order = top_k_with_added(note_scores, added, a, b)
                    new_sections = np.array([sections[rows][o] if o < len(rows) else SECTION for o in order])
                    new_lex = np.array([real_lex[rows[o], j] if o < len(rows) else inv_lex[position[opening if o == len(rows) else decider], j] for o in order])
                    row = {"item": item.id, "case": case, "gap": gap, "decider_names_item": bool(inv_lex[position[decider], j]),
                           "opening_top": bool(top[pa]), "decider_top": bool(top[pb]), "both_base": bool(top[pa] and top[pb])}
                    for w in WINDOWS:
                        grown = expand_with_followers(top, new_lex, new_sections, w)
                        row[f"both_w{w}"] = bool(grown[pa] and grown[pb])
                    trials.append(row)
    trials = pd.DataFrame(trials)
    by_group = trials.groupby(["decider_names_item", "gap"])[["opening_top", "decider_top", "both_base", "both_w1", "both_w2"]].mean().round(3)
    pooled = trials.groupby("decider_names_item")[["opening_top", "decider_top", "both_base", "both_w1", "both_w2"]].mean().round(3)
    unnamed = trials[~trials["decider_names_item"]]

    arrays = [real_bm, *real_enc]
    ranks = within_note_ranks(rrf([within_note_ranks(a, index) for a in arrays], score_arrays=arrays), index)
    added_sentences = {w: [] for w in WINDOWS}
    gate_rows = []
    mean_enc = (real_enc[0] + real_enc[1]) / 2
    for rows in index:
        for j, item in enumerate(ITEMS):
            top = ranks[rows, j] <= K
            for w in WINDOWS:
                added_sentences[w].append(int(expand_with_followers(top, real_lex[rows, j], sections[rows], w).sum() - top.sum()))
            gate_rows.append({"note": keys[rows[0]], "item": item.id, "lex": bool(real_lex[rows, j][top].any()),
                              "score": float(mean_enc[rows, j][top].max()), "yes": plain[(keys[rows[0]], item.id)]["parsed"]["answer"] == "yes"})
    gate = pd.DataFrame(gate_rows)
    graph = {}
    for w in WINDOWS:
        per_item = unnamed.groupby("item")[[f"both_w{w}", "both_base"]].mean()
        diff = per_item[f"both_w{w}"] - per_item["both_base"]
        low, high = ci(diff)
        extra = float(np.mean(added_sentences[w]))
        graph[f"window_{w}"] = {"gain": round(float(diff.mean()), 3), "ci": [low, high], "mean_added_sentences": round(extra, 2),
                                "keep": bool(diff.mean() >= .05 and low > 0 and extra <= 5)}

    notes = sorted(gate["note"].unique())
    folds = [set(notes[0::2]), set(notes[1::2])]
    fold_rows = []
    paraphrase = np.array([(inv_enc[0][position[FACTS[i.id][TIERS.index("paraphrase")]], j] + inv_enc[1][position[FACTS[i.id][TIERS.index("paraphrase")]], j]) / 2 for j, i in enumerate(ITEMS)])
    for tune, test in (folds, folds[::-1]):
        a, b = gate[gate["note"].isin(tune)], gate[gate["note"].isin(test)]
        threshold = float(np.quantile(a.loc[a["yes"] & ~a["lex"], "score"], .02)) if (a["yes"] & ~a["lex"]).any() else float(a["score"].min())
        read = b["lex"] | (b["score"] >= threshold)
        fold_rows.append({"threshold": round(threshold, 3), "calls_saved": round(float(1 - read.mean()), 3), "test_yes": int(b["yes"].sum()),
                          "yes_lost": int((b["yes"] & ~read).sum()), "paraphrase_pass": round(float((paraphrase >= threshold).mean()), 3)})
    gate_table = pd.DataFrame(fold_rows)
    lost, saved, para = gate_table["yes_lost"].sum() / gate_table["test_yes"].sum(), gate_table["calls_saved"].mean(), gate_table["paraphrase_pass"].mean()

    supported = {(r["note"], r["item"]): r["parsed"]["supported"] for r in runs if r["kind"] == "support" and r["arm"] == "plain"}
    proxied = [i for i in ITEMS if i.has_proxy]
    y, before, after = [], [], []
    for note in notes:
        for item in proxied:
            answer = plain[(note, item.id)]["parsed"]["answer"] == "yes"
            y.append(int(labels.loc[note, item.id])); before.append(int(answer))
            after.append(int(answer and supported.get((note, item.id), True)))
    kappa_before, kappa_after = cohen_kappa(np.array(y), np.array(before)), cohen_kappa(np.array(y), np.array(after))
    dropped = [t for t, b, a in zip(y, before, after) if b and not a]

    summary = {"trials": len(trials), "graph": graph,
               "gate": {"yes_lost_share": round(float(lost), 3), "calls_saved": round(float(saved), 3), "paraphrase_pass": round(float(para), 3),
                        "keep": bool(lost <= .02 and saved >= .3 and para >= .95)},
               "verifier": {"dropped_yes": len(dropped), "dropped_table_yes": int(sum(dropped)), "kappa_before": round(float(kappa_before), 3),
                            "kappa_after": round(float(kappa_after), 3), "keep": bool(kappa_after - kappa_before >= .01)}}
    by_group.to_csv(OUT / "graph_by_gap.csv")
    pooled.to_csv(OUT / "graph_pooled.csv")
    gate_table.to_csv(OUT / "gate_folds.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 180)
    print(by_group.to_string()); print(pooled.to_string()); print(gate_table.to_string()); print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
