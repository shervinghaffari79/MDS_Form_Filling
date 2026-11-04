"""Save per-item candidate sentence IDs for the chosen retrieval. Output holds IDs and ranks, never note text."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.retrieval import bm25_scores, candidate_rows, encoder_scores, load_encoder
from run_self_supervised import cached_embeddings, load_sample

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="pilot")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    sent, _, index, _ = load_sample(args.sample)
    texts = sent["text"].tolist()
    arrays = [bm25_scores(texts, "syn")]
    for name in ("bge", "biolord"):
        emb = cached_embeddings(args.sample, texts, name)
        print(f"[progress] encoded {name}", flush=True)
        arrays.append(encoder_scores(emb, load_encoder(name), name, "syn"))
    rows = candidate_rows(sent["note_key"].to_numpy(), sent["sent_id"].to_numpy(), arrays, index, args.k)
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(RESULTS / f"candidates_{args.sample}.parquet")
    print(f"[progress] wrote {len(rows)} candidate rows", flush=True)


if __name__ == "__main__":
    main()
