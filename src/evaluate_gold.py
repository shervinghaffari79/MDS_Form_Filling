"""Score retrieval components against human labels. Writes aggregate results only.

Evidence means any sentence a reviewer needs for the item, including negated or past mentions.
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS, ITEM_IDS
from mds_pilot.metrics import cohen_kappa, stratified_bootstrap_recall, weighted_recall
from mds_pilot.retrieval import (bm25_scores, embed_sentences, encoder_scores, lexicon_scores, load_encoder,
                                 note_index_from, rrf, within_note_ranks)

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private"
RESULTS = ROOT / "results"
KS = [1, 3, 5, 10, 20, 30]
GATE_K = 20
GATE_RECALL = 0.95
MIN_NOTES_FOR_DECISION = 15


def load_labels(directory):
    files = sorted(Path(directory).glob("*.json"))
    if not files:
        raise SystemExit(f"No label files in {directory}. Export them from the labelling tool first.")
    return [json.loads(f.read_text()) for f in files]


def gold_table(annotation):
    done = set(annotation["done"])
    rows = [t for t in annotation["tags"] if t["note_key"] in done]
    return pd.DataFrame(rows, columns=["note_key", "sent_id", "item", "status"]), done


def agreement(annotations, sentences):
    rows = []
    for a, b in combinations(annotations, 2):
        shared = set(a["done"]) & set(b["done"])
        if not shared:
            continue
        ga, gb = gold_table(a)[0], gold_table(b)[0]
        sub = sentences[sentences["note_key"].isin(shared)]
        keys = [(n, s) for n, s in zip(sub["note_key"], sub["sent_id"])]
        for item in ITEM_IDS:
            sa = set(zip(*[ga.loc[ga["item"] == item, c] for c in ("note_key", "sent_id")]))
            sb = set(zip(*[gb.loc[gb["item"] == item, c] for c in ("note_key", "sent_id")]))
            if not sa and not sb:
                continue
            xa = [k in sa for k in keys]
            xb = [k in sb for k in keys]
            na = {k[0] for k in sa}
            nb = {k[0] for k in sb}
            notes = sorted(shared)
            rows.append({"pair": f"{a['annotator']}|{b['annotator']}", "notes": len(shared), "item": item,
                         "sentence_kappa": round(cohen_kappa(xa, xb), 3),
                         "note_kappa": round(cohen_kappa([n in na for n in notes], [n in nb for n in notes]), 3)})
    return pd.DataFrame(rows)


def window(mask, index, width):
    """Expand retrieved sentences by a reading-order neighbour window within the note."""
    out = mask.copy()
    for idx in index:
        block = mask[idx]
        grown = block.copy()
        for w in range(1, width + 1):
            grown[w:] |= block[:-w]
            grown[:-w] |= block[w:]
        out[idx] = grown
    return out


def evaluate(masks, gold_mask, sent_notes, index, notes_meta):
    """Note-item recall, sentence recall and reading budget for each retrieved mask."""
    rows, detail = [], {}
    for name, mask in masks.items():
        records = []
        for idx in index:
            note = sent_notes[idx[0]]
            has = gold_mask[idx].any(axis=0)
            found = (mask[idx] & gold_mask[idx]).any(axis=0)
            for j in np.flatnonzero(has):
                records.append({"note_key": note, "item": ITEM_IDS[j], "stratum": notes_meta.loc[note, "stratum"],
                                "weight": notes_meta.loc[note, "weight"], "hit": int(found[j])})
        table = pd.DataFrame(records)
        detail[name] = table
        if table.empty:
            continue
        low, high = stratified_bootstrap_recall(table, n=1000)
        section_recall = table.assign(sec=table["item"].str[0]).groupby("sec").apply(
            lambda g: weighted_recall(g["hit"], g["weight"]), include_groups=False)
        rows.append({"method": name, "note_item_recall_weighted": round(weighted_recall(table["hit"], table["weight"]), 4),
                     "ci_low": round(low, 4), "ci_high": round(high, 4),
                     "note_item_recall_unweighted": round(float(table["hit"].mean()), 4),
                     **{f"recall_{s}": round(float(section_recall.get(s, np.nan)), 4) for s in "INO"},
                     "sentence_recall": round(float((mask & gold_mask).sum() / max(gold_mask.sum(), 1)), 4),
                     "mean_sentences_read_per_item_note": round(float(mask.sum() / (len(index) * mask.shape[1])), 2),
                     "share_of_note_read": round(float(mask.mean()), 4), "positive_pairs": len(table)})
    return pd.DataFrame(rows), detail


def paired(detail, a, b, n=2000, seed=0):
    ta = detail[a].rename(columns={"hit": "a"})
    tb = detail[b][["note_key", "item", "hit"]].rename(columns={"hit": "b"})
    m = ta.merge(tb, on=["note_key", "item"])
    rng = np.random.default_rng(seed)
    strata = {s: g["note_key"].unique() for s, g in m.groupby("stratum")}
    groups = {k: g for k, g in m.groupby("note_key")}
    point = weighted_recall(m["a"], m["weight"]) - weighted_recall(m["b"], m["weight"])
    vals = []
    for _ in range(n):
        ha = hb = w = 0.0
        for notes in strata.values():
            for note in rng.choice(notes, len(notes)):
                g = groups[note]
                ha += (g["a"].values * g["weight"].values).sum()
                hb += (g["b"].values * g["weight"].values).sum()
                w += g["weight"].values.sum()
        vals.append((ha - hb) / w)
    low, high = np.quantile(vals, [.025, .975])
    return {"test": f"{a} minus {b}", "recall_diff": round(point, 4), "ci_low": round(float(low), 4),
            "ci_high": round(float(high), 4),
            "verdict": "improves" if low > 0 else "worse" if high < 0 else "no clear difference"}


def rerank(fused_ranks, sentences, top=30):
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    out = np.full(fused_ranks.shape, -1e6, dtype=np.float32)
    texts = sentences["text"].tolist()
    for j, item in enumerate(ITEMS):
        rows = np.flatnonzero(fused_ranks[:, j] <= top)
        query = f"{item.definition}: {', '.join(item.synonyms[:8])}"
        out[rows, j] = model.predict([(query, texts[r]) for r in rows], batch_size=32, show_progress_bar=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(PRIVATE / "annotations"))
    ap.add_argument("--annotator", default=None, help="Reference annotator; defaults to the one with most done notes")
    ap.add_argument("--encoders", default="bge,mpnet")
    ap.add_argument("--with-reranker", action="store_true")
    ap.add_argument("--sentences", default=str(PRIVATE / "pilot_sentences.parquet"))
    ap.add_argument("--meta", default=str(PRIVATE / "pilot_labels.parquet"))
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    annotations = load_labels(args.labels)
    ref = next((a for a in annotations if a["annotator"] == args.annotator), None) if args.annotator else \
        max(annotations, key=lambda a: len(a["done"]))
    if ref is None:
        raise SystemExit("Reference annotator not found.")
    gold, done = gold_table(ref)
    sentences = pd.read_parquet(args.sentences)
    meta = pd.read_parquet(args.meta).set_index("note_key")
    iaa = agreement(annotations, sentences)
    sent = sentences[sentences["note_key"].isin(done)].sort_values(["note_key", "sent_id"]).reset_index(drop=True)
    texts = sent["text"].tolist()
    index = note_index_from(sent["note_key"].values)
    position = {(n, s): i for i, (n, s) in enumerate(zip(sent["note_key"], sent["sent_id"]))}
    item_pos = {i: j for j, i in enumerate(ITEM_IDS)}
    gold_mask = np.zeros((len(sent), len(ITEMS)), dtype=bool)
    for r in gold.itertuples():
        gold_mask[position[(r.note_key, r.sent_id)], item_pos[r.item]] = True

    encoders = [e for e in args.encoders.split(",") if e]
    scores = {"bm25_def": bm25_scores(texts, "def"), "bm25_syn": bm25_scores(texts, "syn")}
    for name in encoders:
        emb, model = embed_sentences(texts, name)
        scores[f"{name}_def"] = encoder_scores(emb, model, name, "def")
        scores[f"{name}_syn"] = encoder_scores(emb, model, name, "syn")
        del emb, model
    ranks = {m: within_note_ranks(s, index) for m, s in scores.items()}
    syn = [f"{e}_syn" for e in encoders]
    fused = {"fusion_all": ["bm25_syn", *syn]}
    for e in encoders:
        fused[f"fusion_bm25+{e}"] = ["bm25_syn", f"{e}_syn"]
    for name, members in fused.items():
        ranks[name] = within_note_ranks(rrf([ranks[m] for m in members], score_arrays=[scores[m] for m in members]), index)
    if args.with_reranker:
        ranks["fusion_all+reranker"] = within_note_ranks(rerank(ranks["fusion_all"], sent), index)

    lexicon = lexicon_scores(texts) > 0
    masks = {"lexicon_only": lexicon, "whole_note": np.ones_like(gold_mask)}
    for m, r in ranks.items():
        for k in KS:
            masks[f"{m}@{k}"] = r <= k
    best_key = f"fusion_all@{GATE_K}"
    masks[f"lexicon_or_{best_key}"] = lexicon | masks[best_key]
    masks[f"{best_key}+window1"] = window(masks[best_key], index, 1)
    masks["fusion_all@5+window1"] = window(masks["fusion_all@5"], index, 1)
    results, detail = evaluate(masks, gold_mask, sent["note_key"].values, index, meta)

    tests = []
    pairs = [(f"bm25_syn@{GATE_K}", f"bm25_def@{GATE_K}")]
    for e in encoders:
        pairs += [(f"{e}_syn@{GATE_K}", f"{e}_def@{GATE_K}"), (f"{e}_syn@{GATE_K}", f"bm25_syn@{GATE_K}"),
                  (f"fusion_bm25+{e}@{GATE_K}", f"bm25_syn@{GATE_K}")]
    pairs += [(best_key, f"fusion_bm25+{e}@{GATE_K}") for e in encoders]
    pairs += [(best_key, "lexicon_only"), (f"lexicon_or_{best_key}", best_key), (f"{best_key}+window1", best_key)]
    if args.with_reranker:
        pairs.append((f"fusion_all+reranker@5", "fusion_all@5"))
    tests = pd.DataFrame([paired(detail, a, b) for a, b in pairs if a in detail and b in detail])

    by_item = []
    for name in (best_key, "lexicon_only"):
        table = detail[name]
        if table.empty:
            continue
        g = table.groupby("item")
        by_item.append(pd.DataFrame({"positive_notes": g["hit"].size(), f"recall_{name}": g["hit"].mean().round(3)}))
    by_item = pd.concat(by_item, axis=1) if by_item else pd.DataFrame()
    by_item = by_item.loc[:, ~by_item.columns.duplicated()]
    gate = detail[best_key].groupby("item")["hit"].agg(["size", "mean"]) if not detail[best_key].empty else pd.DataFrame()
    decision = "insufficient notes: provisional only" if len(done) < MIN_NOTES_FOR_DECISION else \
        ("all items pass" if (gate["mean"] >= GATE_RECALL).all() else "some items fail; use whole-note reading for them")

    results.to_csv(out_dir / "gold_retrieval_results.csv", index=False)
    tests.to_csv(out_dir / "gold_component_tests.csv", index=False)
    by_item.to_csv(out_dir / "gold_recall_by_item.csv")
    iaa.to_csv(out_dir / "gold_agreement.csv", index=False)
    status_counts = gold["status"].value_counts().to_dict()
    summary = {"reference_annotator_notes": len(done), "gold_tags": len(gold), "status_counts": status_counts,
               "gate": f"note-item recall >= {GATE_RECALL} at k={GATE_K} per item", "gate_decision": decision,
               "items_below_gate": gate.index[gate["mean"] < GATE_RECALL].tolist() if len(gate) else []}
    (out_dir / "gold_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200, "display.max_colwidth", 70)
    print(json.dumps(summary, indent=2)); print(results.to_string()); print(tests.to_string())


if __name__ == "__main__":
    main()
