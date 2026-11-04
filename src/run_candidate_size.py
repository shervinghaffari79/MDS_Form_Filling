"""Measure how much note text the chosen retrieval would pass to the LLM. Writes aggregate results only.

Retrieval is reciprocal rank fusion of BM25, BGE and BioLORD, each using definition plus synonyms.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS
from mds_pilot.retrieval import bm25_scores, encoder_scores, load_encoder, rrf, within_note_ranks
from run_self_supervised import cached_embeddings, load_sample

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
KS = (5, 10, 20)
CHARS_PER_TOKEN = 4  # rough estimate for English prose, not a tokenizer count


def candidate_mask(ranks, idx, k, neighbours=0):
    """Sentences in a note retrieved for at least one item in the given columns."""
    chosen = (ranks[idx] <= k).any(axis=1)
    if neighbours:
        chosen = np.convolve(chosen, np.ones(2 * neighbours + 1), mode="same") > 0
    return chosen


def describe(values):
    return {"p10": round(float(np.quantile(values, .1)), 1), "median": round(float(np.median(values)), 1),
            "p90": round(float(np.quantile(values, .9)), 1), "max": round(float(np.max(values)), 1)}


def main():
    sent, _, index, _ = load_sample("proxy")
    texts = sent["text"].tolist()
    chars = sent["text"].str.len().to_numpy() + 1
    scores = {"bm25": bm25_scores(texts, "syn")}
    for name in ("bge", "biolord"):
        emb = cached_embeddings("proxy", texts, name)
        print(f"[progress] encoded {name}", flush=True)
        scores[name] = encoder_scores(emb, load_encoder(name), name, "syn")
    arrays = list(scores.values())
    ranks = within_note_ranks(rrf([within_note_ranks(a, index) for a in arrays], score_arrays=arrays), index)
    print("[progress] fused", flush=True)

    sections = np.array([i.section for i in ITEMS])
    note_chars = np.array([chars[idx].sum() for idx in index])
    note_sents = np.array([len(idx) for idx in index])
    rows, section_rows, share_rows = [], [], []
    for k in KS:
        for neighbours in (0, 1):
            union_n, union_chars, per_item_chars = [], [], []
            for idx in index:
                mask = candidate_mask(ranks, idx, k, neighbours)
                union_n.append(mask.sum())
                union_chars.append(chars[idx][mask].sum())
                per_item_chars.append(sum(chars[idx][candidate_mask(ranks[:, [j]], idx, k, neighbours)].sum()
                                          for j in range(len(ITEMS))))
            union_n, union_chars, per_item_chars = map(np.array, (union_n, union_chars, per_item_chars))
            rows.append({"k": k, "neighbours": neighbours,
                         "union_sentences": describe(union_n),
                         "union_share_of_sentences": describe(100 * union_n / note_sents),
                         "union_share_of_chars": describe(100 * union_chars / note_chars),
                         "one_prompt_per_note_tokens": describe(union_chars / CHARS_PER_TOKEN),
                         "one_prompt_per_item_note_total_tokens": describe(per_item_chars / CHARS_PER_TOKEN)})
        for sec in ("I", "N", "O"):
            cols = np.flatnonzero(sections == sec)
            union = [candidate_mask(ranks[:, cols], idx, k).sum() for idx in index]
            section_rows.append({"k": k, "mds_section": sec, "items": len(cols), **describe(np.array(union))})
        hit = (ranks <= k).any(axis=1)
        by_section = sent.assign(hit=hit).groupby("section")["hit"].agg(["sum", "size"])
        for name, row in by_section.iterrows():
            share_rows.append({"k": k, "note_section": name, "share_of_candidates": round(100 * row["sum"] / hit.sum(), 1),
                               "share_of_section_selected": round(100 * row["sum"] / row["size"], 1)})

    summary = {"notes": len(index), "note_sentences": describe(note_sents),
               "note_tokens_estimate": describe(note_chars / CHARS_PER_TOKEN), "chars_per_token": CHARS_PER_TOKEN,
               "items": len(ITEMS), "by_k": rows}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "candidate_size.json").write_text(json.dumps(summary, indent=1))
    pd.DataFrame(section_rows).to_csv(RESULTS / "candidate_size_by_mds_section.csv", index=False)
    pd.DataFrame(share_rows).to_csv(RESULTS / "candidate_source_sections.csv", index=False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
