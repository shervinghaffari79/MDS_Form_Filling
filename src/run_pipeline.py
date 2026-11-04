"""Run the whole pipeline with one command: notes in, MDS output out.

Steps: split notes into sentences, retrieve the top sentences per item, read each with the plain yes/no reader through the
OpenAI API, compare with structured tables if available, and write the output.

  run_pipeline.py --name demo --mimic-sample pilot --limit 3 --encode modal --dry-run
  run_pipeline.py --name demo --notes notes.parquet --hosp mimic-iv-2.2/hosp

--notes takes a CSV or parquet file with note_key and text, plus hadm_id if --hosp is used. Use note_key values that are not
patient or admission IDs, because they appear in the text-free results.
Text-bearing files go to pilot_private/runs/<name>; text-free files go to results/runs/<name>. Printed output holds no note text.
A rerun with the same name resumes without repeating API calls. Output is written only when every call has succeeded.
Note: BM25 statistics come from the notes in the run, so the sentences retrieved for a note can vary with the other notes in it.
"""
import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_mds_output import write_outputs
from mds_pilot.items import ITEMS
from mds_pilot.mds_fields import SCOPE_NOTE
from mds_pilot.reader import RULES_VERSION, SYSTEM, build_prompt
from mds_pilot.retrieval import bm25_scores, candidate_rows, embed_sentences, encoder_scores, note_index_from
from mds_pilot.segment import segment_note
from mds_pilot.tables import load_flags
from run_reader import Store, load_env, request_key, run_requests

ROOT = Path(__file__).resolve().parents[1]
ENCODER_NAMES = ("bge", "biolord")
DEFAULT_K = 20
DEFAULT_MAX_TOKENS = 500_000
OUTPUT_TOKENS_PER_CALL = 60


def load_notes(args):
    if args.mimic_sample:
        ids = pd.read_parquet(ROOT / f"pilot_private/{args.mimic_sample}_id_map.parquet")
        texts = pd.read_parquet(ROOT / "pilot_private/cohort_notes.parquet", columns=["hadm_id", "text"])
        notes = ids[["note_key", "hadm_id"]].merge(texts, on="hadm_id")
    elif args.notes.suffix == ".parquet":
        notes = pd.read_parquet(args.notes)
    else:
        notes = pd.read_csv(args.notes, dtype=str)
    missing = {"note_key", "text"} - set(notes.columns)
    if missing:
        sys.exit(f"Notes file lacks required columns: {sorted(missing)}")
    if notes["note_key"].duplicated().any():
        sys.exit("Notes file has duplicate note_key values")
    notes = notes.sort_values("note_key").reset_index(drop=True)
    return notes.head(args.limit) if args.limit else notes


def split_notes(notes):
    rows = []
    for key, text in zip(notes["note_key"], notes["text"]):
        rows.extend({"note_key": key, **r} for r in segment_note(text))
    frame = pd.DataFrame(rows, columns=["note_key", "sent_id", "section", "text"])
    return frame.sort_values(["note_key", "sent_id"]).reset_index(drop=True)


def score_local(texts):
    out = []
    for name in ENCODER_NAMES:
        emb, model = embed_sentences(texts, name)
        out.append(encoder_scores(emb, model, name, "syn"))
    return out


def score_modal(texts):
    from modal_run import app, score_items
    with app.run():
        payloads = score_items.remote(texts, list(ENCODER_NAMES))
    return [np.load(io.BytesIO(p)) for p in payloads]


def cached_scores(texts, scorer, private):
    """Encoding is the slow step, so its scores are kept for reruns of the same sentences."""
    digest = hashlib.sha256(json.dumps([list(ENCODER_NAMES), texts]).encode()).hexdigest()[:16]
    path = private / f"encoder_scores_{digest}.npz"
    if path.exists():
        with np.load(path) as saved:
            return [saved[n] for n in ENCODER_NAMES]
    scores = scorer(texts)
    np.savez(path, **dict(zip(ENCODER_NAMES, scores)))
    return scores


def build_requests(sent, picks, model):
    lookup = {(n, s): (sec, t) for n, s, sec, t in zip(sent["note_key"], sent["sent_id"], sent["section"], sent["text"])}
    requests = []
    for (note, item_id), sent_ids in picks.items():
        rows = [(i + 1, *lookup[(note, s)]) for i, s in enumerate(sent_ids)]
        prompt = build_prompt("plain", ITEM_BY_ID[item_id].definition, rows)
        requests.append({"key": request_key("pipeline", RULES_VERSION, model, hashlib.sha256(prompt.encode()).hexdigest()),
                         "stage": "pipeline", "arm": "plain", "kind": "plain", "note": note, "item": item_id, "prompt": prompt})
    return requests


def estimate_tokens(requests):
    return sum((len(r["prompt"]) + len(SYSTEM)) // 4 + OUTPUT_TOKENS_PER_CALL for r in requests)


ITEM_BY_ID = {i.id: i for i in ITEMS}


def table_labels(notes, hosp):
    if hosp is None:
        return None
    if "hadm_id" not in notes.columns:
        sys.exit("--hosp needs a hadm_id column in the notes file")
    flags = load_flags(hosp, notes["hadm_id"].tolist()).set_index("hadm_id")
    labels = flags.loc[[str(h) for h in notes["hadm_id"]]].reset_index(drop=True)
    labels.index = notes["note_key"]
    return labels


def run(notes, name, k, scorer, client, model, max_tokens, dry_run, labels=None, root=ROOT):
    """client is only needed when dry_run is False. Returns an exit code."""
    private, results = root / "pilot_private/runs" / name, root / "results/runs" / name
    private.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    sent = split_notes(notes)
    print(f"[progress] {len(notes)} notes, {len(sent)} sentences", flush=True)
    picks = {}
    if len(sent):
        texts = sent["text"].tolist()
        arrays = [bm25_scores(texts, "syn"), *cached_scores(texts, scorer, private)]
        keys = sent["note_key"].to_numpy()
        rows = candidate_rows(keys, sent["sent_id"].to_numpy(), arrays, note_index_from(keys), k)
        candidates = pd.DataFrame(rows)
        candidates.to_parquet(results / "candidates.parquet")
        picks = {key: sorted(g["sent_id"]) for key, g in candidates.groupby(["note_key", "item"])}
    print(f"[progress] retrieved top {k} for {len(picks)} note-item pairs", flush=True)

    requests = build_requests(sent, picks, model)
    store = Store(private / "replies.jsonl")
    todo = [r for r in requests if r["key"] not in store.done]
    estimate = estimate_tokens(todo)
    print(f"[progress] {len(requests)} calls, {len(todo)} to send, about {estimate} tokens, limit {max_tokens}", flush=True)
    if dry_run:
        print("Dry run: nothing was sent to the API and no output was written.")
        return 0
    if store.tokens + estimate > max_tokens:
        print(f"Stopped before sending: estimate exceeds the limit of {max_tokens}. Raise --max-tokens to continue.")
        return 2
    failures = run_requests(requests, store, client, model, budget=max_tokens)
    if failures:
        print("Some calls failed, so no output was written. Rerun the same command to resume.")
        return 1

    answers = {(r["note"], r["item"]): store.done[r["key"]]["parsed"] for r in requests}
    provenance = {"retrieval": f"BM25 + BGE-small + BioLORD, RRF, top {k}", "reader": "plain yes/no", "reader_rules": RULES_VERSION,
                  "model": model, "scope": SCOPE_NOTE}
    write_outputs(notes["note_key"].tolist(), answers, sent, picks, labels, private / "mds_output.jsonl", results, provenance)
    print(f"Tokens this run: {store.tokens}\nText-free results: {results}\nOutput with evidence text, keep private: {private / 'mds_output.jsonl'}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name; reuse it to resume")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--notes", type=Path, help="CSV or parquet with note_key and text, plus hadm_id for --hosp")
    source.add_argument("--mimic-sample", choices=["pilot", "proxy"], help="use a prepared sample from pilot_private")
    ap.add_argument("--hosp", type=Path, help="MIMIC-IV hosp folder for table flags; a prepared sample defaults to mimic-iv-2.2/hosp")
    ap.add_argument("--no-tables", action="store_true", help="skip table flags")
    ap.add_argument("--limit", type=int, help="use only the first N notes")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--encode", choices=["local", "modal"], default="local", help="where sentence encoders run")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="stop before this many API tokens in this run")
    ap.add_argument("--dry-run", action="store_true", help="retrieve and estimate cost, with no API calls")
    args = ap.parse_args()

    load_env()
    model = os.environ.get("MDS_LLM_MODEL")
    if not model:
        sys.exit("Set MDS_LLM_MODEL in the environment or .env")
    notes = load_notes(args)
    hosp = None if args.no_tables else args.hosp or (ROOT / "mimic-iv-2.2/hosp" if args.mimic_sample else None)
    labels = None if args.dry_run else table_labels(notes, hosp)
    client = None
    if not args.dry_run:
        from openai import OpenAI
        client = OpenAI(max_retries=5, timeout=180)
    scorer = score_modal if args.encode == "modal" else score_local
    return run(notes, args.name, args.k, scorer, client, model, args.max_tokens, args.dry_run, labels)


if __name__ == "__main__":
    sys.exit(main())
