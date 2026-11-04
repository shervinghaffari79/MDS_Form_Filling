"""Component tests that need no human labels. Writes aggregate results only.

These compare components against noisy table proxies and a lexical silver standard.
They rank components against each other; they do not measure true accuracy.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS, ITEM_IDS
from mds_pilot.metrics import auroc
from mds_pilot.retrieval import (align_cached_embeddings, bm25_scores, embed_sentences, encoder_scores, lexicon_scores,
                                 load_encoder, note_index_from, rrf, term_pattern, within_note_ranks)

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private"
RESULTS = ROOT / "results"
BOOT = 500


def segmentation_audit(sent, labels):
    per_note = sent.groupby("note_key").size()
    lengths = sent["text"].str.len()
    sections = sent.groupby("section")["note_key"].nunique() / sent["note_key"].nunique()
    return {
        "notes": int(sent["note_key"].nunique()),
        "sentences": int(len(sent)),
        "sentences_per_note_p10_p50_p90": [float(x) for x in per_note.quantile([.1, .5, .9])],
        "chars_per_sentence_p50_p90_p99": [float(x) for x in lengths.quantile([.5, .9, .99])],
        "share_sentences_over_300_chars": round(float((lengths > 300).mean()), 4),
        "share_sentences_under_15_chars": round(float((lengths < 15).mean()), 4),
        "share_sentences_in_preamble": round(float((sent["section"] == "Preamble").mean()), 4),
        "note_share_with_section": {k: round(float(v), 3) for k, v in sections.sort_values().items()},
    }


def note_max(scores, index):
    return np.vstack([scores[idx].max(axis=0) for idx in index])


def percentile(matrix):
    return pd.DataFrame(matrix).rank(pct=True).to_numpy(dtype=np.float32)


def macro_auroc(y, s, cols):
    vals = [auroc(y[:, j], s[:, j]) for j in cols]
    return float(np.nanmean(vals)), vals


def silver_recall(ranks, lex, index, ks):
    """Share of (note, item) pairs with a lexicon hit whose top-k contains a hit sentence."""
    out, per_item = {}, {}
    for k in ks:
        hit, total = np.zeros(lex.shape[1]), np.zeros(lex.shape[1])
        for idx in index:
            has = lex[idx].max(axis=0) > 0
            found = ((ranks[idx] <= k) & (lex[idx] > 0)).any(axis=0)
            hit += found & has
            total += has
        out[k] = float(hit.sum() / total.sum())
        per_item[k] = hit / np.maximum(total, 1)
    return out, per_item


def load_or_embed(sent, texts, name):
    """Reuse embeddings from run_self_supervised after confirming they line up with these rows."""
    path = PRIVATE / f"emb_proxy_{name}.npy"
    if not path.exists():
        return embed_sentences(texts, name)
    emb = align_cached_embeddings(np.load(path), sent)
    model = load_encoder(name)
    picks = np.random.default_rng(0).choice(len(texts), size=64, replace=False)
    fresh = model.encode([texts[i] for i in picks], normalize_embeddings=True, show_progress_bar=False,
                         convert_to_numpy=True)
    if float((fresh * emb[picks]).sum(axis=1).min()) < 0.999:
        raise ValueError(f"cached {name} embeddings do not match sentence order")
    return emb, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="bge,mpnet")
    args = ap.parse_args()
    encoders = [e for e in args.encoders.split(",") if e]

    sent = pd.read_parquet(PRIVATE / "proxy_sentences.parquet")
    labels = pd.read_parquet(PRIVATE / "proxy_labels.parquet")
    sent["note_key"] = pd.Categorical(sent["note_key"], categories=labels["note_key"], ordered=True)
    sent = sent.sort_values(["note_key", "sent_id"]).reset_index(drop=True)
    sent["note_key"] = sent["note_key"].astype(str)
    texts = sent["text"].tolist()
    index = note_index_from(sent["note_key"].values)
    keys = [sent["note_key"].iloc[i[0]] for i in index]
    y = labels.set_index("note_key").loc[keys, ITEM_IDS].to_numpy(dtype=np.int8)
    audit = segmentation_audit(sent, labels)
    print(json.dumps(audit, indent=1), flush=True)

    cols = [j for j, it in enumerate(ITEMS) if it.has_proxy and 5 <= y[:, j].sum() <= len(y) - 5]
    excluded = [ITEMS[j].id for j in range(len(ITEMS)) if j not in cols]

    lex = lexicon_scores(texts)
    scores = {"lexicon_any": lex, "bm25_def": bm25_scores(texts, "def"), "bm25_syn": bm25_scores(texts, "syn")}
    timing = {}
    for name in encoders:
        t0 = time.time()
        emb, model = load_or_embed(sent, texts, name)
        timing[name] = round(time.time() - t0, 1)
        scores[f"{name}_def"] = encoder_scores(emb, model, name, "def")
        scores[f"{name}_syn"] = encoder_scores(emb, model, name, "syn")
        del emb, model
        print(f"[progress] encoded {name} in {timing[name]}s", flush=True)

    note_scores = {m: note_max(s, index) for m, s in scores.items()}
    base = [m for m in note_scores if m.endswith("_syn")]
    fusions = {}
    if len(base) >= 2:
        fusions["+".join(base)] = base
        for enc in encoders:
            fusions[f"bm25_syn+{enc}_syn"] = ["bm25_syn", f"{enc}_syn"]
    for name, members in fusions.items():
        note_scores[name] = np.mean([percentile(note_scores[m]) for m in members], axis=0)

    rows, per_item = [], {}
    for m, s in note_scores.items():
        mean, vals = macro_auroc(y, s, cols)
        per_item[m] = vals
        rows.append({"method": m, "macro_auroc": round(mean, 4)})
    table = pd.DataFrame(rows).sort_values("macro_auroc", ascending=False).reset_index(drop=True)

    rng = np.random.default_rng(0)
    samples = rng.integers(0, len(y), size=(BOOT, len(y)))
    boot = {m: [] for m in note_scores}
    for idx in samples:
        for m, s in note_scores.items():
            boot[m].append(macro_auroc(y[idx], s[idx], cols)[0])
    boot = {m: np.array(v) for m, v in boot.items()}
    table["ci_low"] = [round(float(np.quantile(boot[m], .025)), 4) for m in table["method"]]
    table["ci_high"] = [round(float(np.quantile(boot[m], .975)), 4) for m in table["method"]]

    def contrast(a, b):
        d = boot[a] - boot[b]
        return {"component_test": f"{a} minus {b}", "diff": round(float(d.mean()), 4),
                "ci_low": round(float(np.quantile(d, .025)), 4), "ci_high": round(float(np.quantile(d, .975)), 4)}

    contrasts = [contrast("bm25_syn", "bm25_def"), contrast("bm25_syn", "lexicon_any")]
    for enc in encoders:
        contrasts += [contrast(f"{enc}_syn", f"{enc}_def"), contrast(f"{enc}_syn", "bm25_syn"),
                      contrast(f"bm25_syn+{enc}_syn", "bm25_syn")]
    if len(encoders) >= 2:
        contrasts += [contrast("+".join(base), f"bm25_syn+{e}_syn") for e in encoders]
    contrasts = pd.DataFrame(contrasts)

    item_table = pd.DataFrame({m: per_item[m] for m in note_scores},
                              index=[ITEMS[j].id for j in cols]).round(3)
    item_table.insert(0, "proxy_prevalence", np.round(y[:, cols].mean(axis=0), 3))

    has_lex = note_max(lex, index) > 0
    gap = pd.DataFrame({
        "item": ITEM_IDS,
        "proxy_positive_notes": y.sum(axis=0),
        "proxy_pos_no_lexicon_hit": ((y == 1) & ~has_lex).sum(axis=0),
        "lexicon_hit_proxy_negative": ((y == 0) & has_lex).sum(axis=0),
        "mean_lexicon_sentences_when_present": [
            round(float(np.sum(lex[:, j]) / max(has_lex[:, j].sum(), 1)), 2) for j in range(len(ITEMS))],
    })
    gap.loc[[not i.has_proxy for i in ITEMS], ["proxy_positive_notes", "proxy_pos_no_lexicon_hit", "lexicon_hit_proxy_negative"]] = -1

    note_of = np.zeros(len(texts), dtype=np.int32)
    for n, idx in enumerate(index):
        note_of[idx] = n
    term_rows = []
    for j, item in enumerate(ITEMS):
        for term in item.synonyms:
            pat = term_pattern(term)
            hits = np.fromiter((1 if pat.search(t) else 0 for t in texts), dtype=np.int8, count=len(texts))
            notes = np.unique(note_of[hits == 1])
            row = {"item": item.id, "term": term, "notes_with_term": int(len(notes))}
            if item.has_proxy and len(notes):
                row["share_of_those_proxy_positive"] = round(float(y[notes, j].mean()), 3)
                row["item_prevalence"] = round(float(y[:, j].mean()), 3)
            term_rows.append(row)
    terms = pd.DataFrame(term_rows)

    ks = [1, 3, 5, 10, 20]
    silver = {}
    silver_methods = ["bm25_def"] + [f"{e}_{a}" for e in encoders for a in ("def", "syn")]
    ranks = {m: within_note_ranks(scores[m], index) for m in silver_methods}
    for m in silver_methods:
        silver[m] = silver_recall(ranks[m], lex, index, ks)[0]
    for enc in encoders:
        name = f"bm25_def+{enc}_def"
        silver[name] = silver_recall(
            within_note_ranks(rrf([ranks["bm25_def"], ranks[f"{enc}_def"]]), index), lex, index, ks)[0]
    silver = pd.DataFrame(silver).T.rename(columns=lambda k: f"recall@{k}").round(4)

    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "proxy_macro_auroc.csv", index=False)
    contrasts.to_csv(RESULTS / "proxy_component_contrasts.csv", index=False)
    item_table.to_csv(RESULTS / "proxy_auroc_by_item.csv")
    gap.to_csv(RESULTS / "lexicon_vs_tables_by_item.csv", index=False)
    terms.to_csv(RESULTS / "lexicon_term_audit.csv", index=False)
    silver.to_csv(RESULTS / "lexical_silver_recall.csv")
    summary = {"segmentation": audit, "encode_seconds": timing, "items_scored": [ITEMS[j].id for j in cols],
               "items_excluded_from_auroc": excluded, "n_notes": len(y)}
    (RESULTS / "proxy_test_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200, "display.max_colwidth", 60)
    print(table.to_string()); print(contrasts.to_string()); print(silver.to_string()); print(item_table.to_string())
    print(gap.to_string())


if __name__ == "__main__":
    main()
