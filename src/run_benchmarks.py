"""Test the methods on three labelled public benchmarks, one per MDS section, then place them on the published results.

Data, downloaded by the user after signing each agreement, kept owner-only and never printed:
  benchmarks_private/i2b2_2008/   the i2b2 2008 Obesity XML files (records and annotations, training and test)
  benchmarks_private/n2c2_2018/   n2c2 2018 Track 2 brat files (.txt and .ann) in folders whose names contain 'train' and 'test'
  benchmarks_private/snomed/      the SNOMED CT Entity Linking annotations CSV from PhysioNet; a SNOMED RF2 snapshot in snomed/release
                                  maps concepts to O items through the hierarchy. Note text comes from the local MIMIC-IV-Note copy.
Arms, written before any data was seen:
  i2b2_2008  textual:   t_retrieval (two-stage: ensemble retrieval + judgment), t_status (status record: subject, assertion, timing,
                        turned into a judgment by a fixed rule), full (whole note, all 16 diseases in one call)
             intuitive: i_retrieval, i_kg (knowledge graph: the textual answer plus treatment-to-disease links, no extra calls), full
  n2c2_2018  per N class and IV / chemotherapy: n_retrieval, n_status, n_full; spans for the leaderboard: span
  snomed     per O item: o_retrieval, o_status, o_full; plus knowledge graph fusion with the MIMIC-IV records (no extra calls)
  Retrieval is the encoder ensemble: BM25 + BGE-small + BioLORD, reciprocal rank fusion, top 20 sentences.
  --retrieval checks the ensemble against each encoder alone with the gold spans, with no API calls (n2c2 and SNOMED).
Decision rules, set before any run:
  1. Development notes (i2b2, n2c2: a seeded sample of the training split; SNOMED: a seeded quarter of the notes) choose one arm per task
     by the primary metric; a tie within 0.005 goes to the cheaper arm. --freeze writes the choice before the test notes are scored, and
     only the frozen arm is placed on the leaderboard.
  2. Primary metrics: i2b2 official macro F, textual and intuitive. n2c2 macro balanced accuracy over the note labels, and lenient micro F
     for spans. SNOMED macro balanced accuracy over the O items.
  3. One arm beats another if the paired 95% CI of the difference, resampling notes, is above 0.
  4. The knowledge graph keeps a disease's treatment links only if they raise its intuitive F on the development notes.
  5. A finding transfers to the MIMIC-IV proxy notes only if it holds on the test notes by rule 3.
Published comparisons: i2b2 2008 has the top 10 of 30 teams; n2c2 2018 only the best score per task; the SNOMED challenge ranks entity
linking, a different task, so it gets no ranking. Results hold only scores and counts, never text or note IDs.
Usage: run_benchmarks.py --dataset D (--check | --retrieval | --plan | --run ARM... | --freeze | --score) [--split dev|test]
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
from mds_pilot import benchmarks as B
from mds_pilot.items import ITEMS, Item
from mds_pilot.kg import FUSIONS, balanced_accuracy, fuse, source_matrix
from mds_pilot.reader import CRITERION, STATUS_SCHEMA, SYSTEM, build_prompt, derive_state, format_sentences
from mds_pilot.retrieval import bm25_scores, candidate_rows, encoder_scores, embed_sentences, note_index_from, rrf, within_note_ranks
from run_pipeline import split_notes
from run_reader import SCHEMAS, Store, load_env, run_requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks_private"
RESULTS = ROOT / "results/benchmarks"
NOTES_CSV = ROOT / "mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv"
ENCODERS = ("bge", "biolord")
K = 20
SEED = 20261102
DEV_NOTES = {"i2b2_2008": 100, "n2c2_2018": 40}
SNOMED_DEV_SHARE = .25
TIE = .005
BOOT = 2000
OUTPUT_TOKENS = {"textual": 40, "intuitive": 40, "plain": 40, "status": 120, "full_i2b2": 500, "full_n": 150, "full_o": 100, "span": 900}
N_ITEMS = [i for i in ITEMS if i.section == "N"] + [i for i in ITEMS if i.id in ("O_iv_medication", "O_chemotherapy")]
O_ITEMS = [i for i in ITEMS if i.id in B.O_SNOMED_SEEDS]
ARMS = {"i2b2_2008": ("t_retrieval", "t_status", "i_retrieval", "full"), "n2c2_2018": ("n_retrieval", "n_status", "n_full", "span"),
        "snomed": ("o_retrieval", "o_status", "o_full")}
# Cost order for rule 1 ties: fewer tokens first.
CHEAPER = ["i_kg", "full", "n_full", "o_full", "t_retrieval", "i_retrieval", "n_retrieval", "o_retrieval", "t_status", "n_status", "o_status"]
SCHEMAS.update(B.BENCH_SCHEMAS)
SCHEMAS.update({"full_n": B.yes_no_full_schema(N_ITEMS), "full_o": B.yes_no_full_schema(O_ITEMS), "status": STATUS_SCHEMA})


def private_dir(dataset):
    path = DATA / dataset / "work"
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(DATA, 0o700)
    return path


def key_of(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


# ---------------------------------------------------------------- data

def dev_sample(keys, n):
    rng = np.random.default_rng(SEED)
    return set(rng.choice(sorted(keys), size=min(n, len(keys)), replace=False))


def snomed_concepts(require_release):
    children, names = B.load_snomed_release(DATA / "snomed/release")
    if children is None and require_release:
        sys.exit("No SNOMED RF2 snapshot in benchmarks_private/snomed/release, so O items would use seed concepts only. "
                 "Add the release, or pass --allow-seed-only to accept that.")
    return {item: B.expand(seeds, children) for item, seeds in B.O_SNOMED_SEEDS.items()}, names


def load(dataset, split, allow_seed_only=False):
    """Returns notes (note_key, text), gold, items. gold is dataset specific; for SNOMED it also carries hadm and subject IDs."""
    folder = DATA / dataset
    if not folder.exists():
        sys.exit(f"Missing {folder}. Download the dataset after signing its agreement and place it there (see this script's docstring).")
    if dataset == "i2b2_2008":
        notes, gold = B.load_i2b2_2008(folder, "train" if split == "dev" else "test")
        if split == "dev":
            keep = dev_sample(notes["note_key"], DEV_NOTES[dataset])
            notes, gold = notes[notes["note_key"].isin(keep)], gold[gold["note_key"].isin(keep)]
        return notes.reset_index(drop=True), gold, B.I2B2_DISEASES
    if dataset == "n2c2_2018":
        notes, spans = B.load_n2c2_2018(folder, "train" if split == "dev" else "test")
        if split == "dev":
            keep = dev_sample(notes["note_key"], DEV_NOTES[dataset])
            notes, spans = notes[notes["note_key"].isin(keep)], spans[spans["note_key"].isin(keep)]
        return notes.reset_index(drop=True), {"spans": spans, "labels": B.n2c2_note_labels(notes, spans)}, N_ITEMS
    annotations = B.load_snomed_annotations(folder)
    if annotations is None:
        sys.exit(f"No annotations CSV found under {folder}.")
    ids = sorted(annotations["note_id"].unique())
    dev = dev_sample(ids, int(round(len(ids) * SNOMED_DEV_SHARE)))
    keep = [i for i in ids if (i in dev) == (split == "dev")]
    import duckdb
    con = duckdb.connect()
    con.register("wanted", pd.DataFrame({"note_id": keep}))
    notes = con.execute(f"SELECT note_id AS note_key, CAST(hadm_id AS VARCHAR) AS hadm_id, CAST(subject_id AS VARCHAR) AS subject_id, text "
                        f"FROM read_csv('{NOTES_CSV}', header=true, all_varchar=true) WHERE note_id IN (SELECT note_id FROM wanted)").df()
    concepts, _ = snomed_concepts(not allow_seed_only)
    mine = annotations[annotations["note_id"].isin(keep)]
    labels = B.snomed_note_labels(mine, concepts)
    return notes.sort_values("note_key").reset_index(drop=True), {"annotations": mine, "labels": labels, "concepts": concepts}, O_ITEMS


# ---------------------------------------------------------------- retrieval

def retrieval_items(dataset, intuitive=False):
    items = load_items(dataset)
    if intuitive:  # intuitive retrieval also looks for the linked treatments
        items = [Item(i.id, i.section, i.definition, i.synonyms + B.DISEASE_TREATMENTS.get(i.id, ())) for i in items]
    return items


def load_items(dataset):
    return {"i2b2_2008": B.I2B2_DISEASES, "n2c2_2018": N_ITEMS, "snomed": O_ITEMS}[dataset]


def encoder_arrays(texts, items, encode, work):
    digest = key_of(ENCODERS, [(i.id, i.definition, i.synonyms) for i in items], texts)
    path = work / f"encoder_scores_{digest[:16]}.npz"
    if path.exists():
        with np.load(path) as saved:
            return [saved[n] for n in ENCODERS]
    if encode == "modal":
        from modal_run import app, score_items
        with app.run():
            payloads = score_items.remote(texts, list(ENCODERS), [(i.id, i.definition, list(i.synonyms)) for i in items])
        arrays = [np.load(io.BytesIO(p)) for p in payloads]
    else:
        arrays = []
        for name in ENCODERS:
            emb, model = embed_sentences(texts, name)
            arrays.append(encoder_scores(emb, model, name, "syn", items))
    np.savez(path, **dict(zip(ENCODERS, arrays)))
    return arrays


def sentences(notes, work, split):
    path = work / f"sentences_{split}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    sent = split_notes(notes)
    sent.to_parquet(path)
    os.chmod(path, 0o600)
    return sent


def score_arrays(sent, items, encode, work):
    texts = sent["text"].tolist()
    return {"bm25": bm25_scores(texts, "syn", items), **dict(zip(ENCODERS, encoder_arrays(texts, items, encode, work)))}


def _group(rows):
    out = {}
    for r in rows:
        out.setdefault((r["note_key"], r["item"]), []).append(r["sent_id"])
    return {k: sorted(v) for k, v in out.items()}


# ---------------------------------------------------------------- requests

def rows_for(sent_lookup, note, ids):
    return [(n + 1, *sent_lookup[(note, s)]) for n, s in enumerate(ids)]


def build_requests(dataset, arm, notes, sent, picks, model):
    lookup = {(n, s): (sec, t) for n, s, sec, t in zip(sent["note_key"], sent["sent_id"], sent["section"], sent["text"])}
    items, out = load_items(dataset), []

    def add(kind, note, item, prompt, system=SYSTEM, **extra):
        out.append({"key": key_of(B.BENCH_VERSION, dataset, arm, model, note, item, prompt), "dataset": dataset, "arm": arm, "kind": kind,
                    "note": note, "item": item, "prompt": prompt, "system": system, **extra})

    if arm in ("full", "n_full", "o_full"):
        for note, text in zip(notes["note_key"], notes["text"]):
            if arm == "full":
                add("full_i2b2", note, "all", B.full_i2b2_prompt(text), B.SYSTEM_FULL)
            elif arm == "n_full":
                add("full_n", note, "all", B.full_yes_no_prompt(N_ITEMS, B.MENTION_RULE, text), B.SYSTEM_FULL)
            else:
                add("full_o", note, "all", B.full_yes_no_prompt(O_ITEMS, CRITERION, text), B.SYSTEM_FULL)
        return out
    if arm == "span":
        for note, text in zip(notes["note_key"], notes["text"]):
            for b, block in enumerate(B.line_blocks(text)):
                add("span", note, f"block{b}", B.span_prompt(block), B.SYSTEM_FULL, block=b)
        return out
    for note in notes["note_key"]:
        for item in items:
            ids = picks.get((note, item.id), [])
            rows = rows_for(lookup, note, ids)
            block = format_sentences(rows)
            if arm == "t_retrieval":
                add("textual", note, item.id, B.judgment_prompt(B.TEXTUAL_RULE, item.definition, block))
            elif arm == "i_retrieval":
                add("intuitive", note, item.id, B.judgment_prompt(B.INTUITIVE_RULE, item.definition, block))
            elif arm in ("t_status", "n_status", "o_status"):
                add("status", note, item.id, build_prompt("status", item.definition, rows))
            elif arm == "n_retrieval":
                add("plain", note, item.id, f"Item: {item.definition}\n{B.MENTION_RULE}\n\nAnswer yes or no: does the item apply? List the numbers "
                    f"of the sentences that support a yes; use an empty list for no.\n\nSentences:\n{block}")
            elif arm == "o_retrieval":
                add("plain", note, item.id, build_prompt("plain", item.definition, rows))
    return out


def estimate(requests):
    return sum((len(r["prompt"]) + len(r["system"])) // 4 + OUTPUT_TOKENS[r["kind"]] for r in requests)


# ---------------------------------------------------------------- answers per arm

def answers(dataset, arm, requests, store, notes=None):
    """{(note, item): answer}. i2b2: (textual, intuitive) judgments; others: bool. span: frame of predicted spans."""
    done = {r["key"]: r for r in requests if r["key"] in store.done}
    got = {r["key"]: store.done[r["key"]]["parsed"] for r in done.values()}
    out = {}
    if arm == "span":
        texts = dict(zip(notes["note_key"], notes["text"]))
        rows = []
        for r in requests:
            if r["key"] in got:
                block = B.line_blocks(texts[r["note"]])[r["block"]]
                rows += [{"note_key": r["note"], **s} for s in B.align_mentions(block, got[r["key"]]["mentions"])]
        return pd.DataFrame(rows, columns=["note_key", "type", "start", "end"])
    for r in requests:
        if r["key"] not in got:
            continue
        p = got[r["key"]]
        if arm == "full":
            for item, v in p.items():
                out[(r["note"], item)] = (v["textual"], v["intuitive"])
        elif arm in ("n_full", "o_full"):
            for item, v in p.items():
                out[(r["note"], item)] = v == "yes"
        elif arm == "t_retrieval":
            out[(r["note"], r["item"])] = (p["judgment"], None)
        elif arm == "i_retrieval":
            out[(r["note"], r["item"])] = (None, p["judgment"])
        elif arm == "t_status":
            out[(r["note"], r["item"])] = (B.textual_from_status(p["mentions"]), None)
        elif arm == "n_status":
            out[(r["note"], r["item"])] = B.mention_from_status(p["mentions"])
        elif arm == "o_status":
            out[(r["note"], r["item"])] = derive_state(p["mentions"])[0] == "yes"
        else:
            out[(r["note"], r["item"])] = p["answer"] == "yes"
    return out


# ---------------------------------------------------------------- scoring

def i2b2_frame(gold, source):
    g = gold[gold["source"] == source]
    return pd.Series(g["judgment"].to_numpy(), index=pd.MultiIndex.from_arrays([g["note_key"], g["item"]]))


def i2b2_predictions(arm_answers, which):
    position = 0 if which == "textual" else 1
    return {k: v[position] for k, v in arm_answers.items() if v[position] is not None}


def kg_intuitive(textual_pred, notes, links):
    """Knowledge graph arm: the textual answer as a clinician's judgment, upgraded to Y when the note names a linked treatment."""
    out = {}
    for note, text in zip(notes["note_key"], notes["text"]):
        cues = B.treatment_cues([text], links)
        for item in B.I2B2_DISEASES:
            base = B.intuitive_from_textual(textual_pred.get((note, item.id), "U"))
            out[(note, item.id)] = "Y" if base == "N" and cues.get(item.id) else base
    return out


def macro_f_boot(gold, preds, classes, seed=0):
    """Bootstrap the official macro F by resampling notes. gold: Series indexed by (note, item). preds: list of dicts.
    Returns point values and the resampled values for each prediction set."""
    notes = gold.index.get_level_values(0).to_numpy()
    uniq, inv = np.unique(notes, return_inverse=True)
    g = gold.astype(str).to_numpy()
    ps = [np.array([p.get(k, "missing") for k in gold.index], dtype=object) for p in preds]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=BOOT)[:, inv].astype(float)  # BOOT x rows
    out = []
    for p in ps:
        fs = []
        for c in classes:
            gc, pc = g == c, p == c
            if not gc.any():
                continue
            tp, fp, fn = weights @ (gc & pc), weights @ (~gc & pc), weights @ (gc & ~pc)
            prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
            rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp + fn > 0)
            fs.append(np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=prec + rec > 0))
        out.append(np.mean(fs, axis=0))
    return out


def note_ba(labels, pred, items):
    """Macro balanced accuracy over items with both classes present. labels: frame note_key, item, gold."""
    scores = {}
    for item in items:
        g = labels[labels["item"] == item.id]
        if g["gold"].all() or not g["gold"].any():
            continue
        p = [bool(pred.get((n, item.id), False)) for n in g["note_key"]]
        scores[item.id] = balanced_accuracy(g["gold"].to_numpy(), p)
    return float(np.mean(list(scores.values()))) if scores else float("nan"), scores


def ba_boot(labels, preds, items, seed=0):
    notes = labels["note_key"].unique()
    index = {n: np.flatnonzero(labels["note_key"].to_numpy() == n) for n in notes}
    rng = np.random.default_rng(seed)
    out = [[] for _ in preds]
    gold = labels["gold"].to_numpy().astype(bool)
    item_col = labels["item"].to_numpy()
    arrays = [np.array([bool(p.get((n, i), False)) for n, i in zip(labels["note_key"], item_col)]) for p in preds]
    for _ in range(BOOT):
        idx = np.concatenate([index[n] for n in rng.choice(notes, len(notes))])
        for j, a in enumerate(arrays):
            vals = [balanced_accuracy(gold[idx][item_col[idx] == it.id], a[idx][item_col[idx] == it.id]) for it in items]
            out[j].append(np.nanmean(vals))
    return [np.array(v) for v in out]


def ci(values):
    low, high = np.nanquantile(values, [.025, .975])
    return round(float(low), 4), round(float(high), 4)


# ---------------------------------------------------------------- commands

def check(dataset, args):
    notes, gold, items = load(dataset, args.split, args.allow_seed_only)
    report = {"dataset": dataset, "split": args.split, "notes": len(notes)}
    if dataset == "i2b2_2008":
        report["judgments"] = gold.groupby(["source", "judgment"]).size().to_dict()
        report["diseases"] = int(gold["item"].nunique())
    elif dataset == "n2c2_2018":
        report["spans_by_type"] = gold["spans"]["type"].value_counts().to_dict()
        report["positive_notes_by_item"] = gold["labels"].groupby("item")["gold"].sum().astype(int).to_dict()
    else:
        _, names = snomed_concepts(False)
        report["seed_names"] = {item: {c: (names or {}).get(c, "NOT FOUND - check this ID") for c in seeds}
                                for item, seeds in B.O_SNOMED_SEEDS.items()}
        report["concepts_per_item"] = {k: len(v) for k, v in gold["concepts"].items()}
        report["positive_notes_by_item"] = gold["labels"].groupby("item")["gold"].sum().astype(int).to_dict()
        report["notes_with_hadm"] = int(notes["hadm_id"].notna().sum())
    print(json.dumps(report, indent=1, default=str))


def retrieval(dataset, args):
    """Zero-token test of the encoder ensemble: does each note's top 20 hold a sentence with a gold span for that item?"""
    if dataset == "i2b2_2008":
        sys.exit("i2b2 2008 has no gold spans, so retrieval cannot be checked on it.")
    notes, gold, items = load(dataset, args.split, args.allow_seed_only)
    work = private_dir(dataset)
    sent = sentences(notes, work, args.split)
    arrays = score_arrays(sent, items, args.encode, work)
    if dataset == "n2c2_2018":
        spans = gold["spans"]
        rows = []
        for item in items:
            if item.id == "O_iv_medication":
                hit = spans[(spans["type"] == "Route") & spans["text"].str.contains(B.IV_ROUTE)]
            else:
                terms = B.CHEMO_DRUGS if item.id == "O_chemotherapy" else item.rx
                hit = spans[(spans["type"] == "Drug") & spans["text"].apply(lambda t: B._matches(t, terms))]
            rows.append(hit.assign(item=item.id)[["note_key", "item", "text"]])
        evidence_spans = pd.concat(rows)
    else:
        texts = dict(zip(notes["note_key"], notes["text"]))
        ann = gold["annotations"]
        rows = []
        for item, ids in gold["concepts"].items():
            a = ann[ann["concept_id"].isin(ids)]
            rows += [{"note_key": n, "item": item, "text": texts[n][int(s):int(e)]} for n, s, e in zip(a["note_id"], a["start"], a["end"]) if n in texts]
        evidence_spans = pd.DataFrame(rows, columns=["note_key", "item", "text"])
    evidence, placed = B.evidence_sentences(sent, evidence_spans)
    methods = {"bm25": ["bm25"], "bge": ["bge"], "biolord": ["biolord"], "ensemble": ["bm25", *ENCODERS]}
    hits = {}
    for name, parts in methods.items():
        picks = _group(candidate_rows(sent["note_key"].to_numpy(), sent["sent_id"].to_numpy(), [arrays[p] for p in parts],
                                      note_index_from(sent["note_key"].to_numpy()), K, items))
        hits[name] = np.array([bool(evidence[k] & set(picks.get(k, []))) for k in sorted(evidence)])
    pairs = sorted(evidence)
    notes_of = np.array([k[0] for k in pairs])
    uniq = np.unique(notes_of)
    rng = np.random.default_rng(0)
    idx_by = {n: np.flatnonzero(notes_of == n) for n in uniq}
    result = {"dataset": dataset, "split": args.split, "pairs_with_gold_evidence": len(pairs), "spans_placed_in_a_sentence": round(placed, 3),
              "recall_at_20": {m: round(float(h.mean()), 4) for m, h in hits.items()}}
    best_single = max(("bm25", "bge", "biolord"), key=lambda m: hits[m].mean())
    diffs = []
    for _ in range(BOOT):
        idx = np.concatenate([idx_by[n] for n in rng.choice(uniq, len(uniq))])
        diffs.append(hits["ensemble"][idx].mean() - hits[best_single][idx].mean())
    result["ensemble_minus_best_single"] = {"versus": best_single, "difference": round(float(hits["ensemble"].mean() - hits[best_single].mean()), 4),
                                            "ci": ci(np.array(diffs))}
    out = RESULTS / dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / f"retrieval_{args.split}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def prepared(dataset, args):
    notes, gold, items = load(dataset, args.split, args.allow_seed_only)
    work = private_dir(dataset)
    sent = sentences(notes, work, args.split)
    picks = {}
    needs = set(args.run or ARMS[dataset]) if args.command != "score" else set(ARMS[dataset])
    if needs & {"t_retrieval", "t_status", "n_retrieval", "n_status", "o_retrieval", "o_status"}:
        picks["plain"] = _group(candidate_rows(sent["note_key"].to_numpy(), sent["sent_id"].to_numpy(),
                                               list(score_arrays(sent, items, args.encode, work).values()),
                                               note_index_from(sent["note_key"].to_numpy()), K, items))
    if "i_retrieval" in needs:
        intuitive_items = retrieval_items(dataset, intuitive=True)
        picks["intuitive"] = _group(candidate_rows(sent["note_key"].to_numpy(), sent["sent_id"].to_numpy(),
                                                   list(score_arrays(sent, intuitive_items, args.encode, work).values()),
                                                   note_index_from(sent["note_key"].to_numpy()), K, intuitive_items))
    return notes, gold, items, work, sent, picks


def all_requests(dataset, arms, notes, sent, picks, model):
    return {arm: build_requests(dataset, arm, notes, sent, picks.get("intuitive" if arm == "i_retrieval" else "plain", {}), model) for arm in arms}


def plan_or_run(dataset, args):
    load_env()
    model = os.environ.get("MDS_LLM_MODEL", "unset")
    arms = args.run or ARMS[dataset]
    notes, gold, items, work, sent, picks = prepared(dataset, args)
    store = Store(work / "replies.jsonl")
    requests = all_requests(dataset, arms, notes, sent, picks, model)
    total = 0
    for arm, reqs in requests.items():
        todo = [r for r in reqs if r["key"] not in store.done]
        total += estimate(todo)
        print(f"{arm}: {len(reqs)} calls, {len(todo)} to send, about {estimate(todo):,} tokens")
    print(f"Total to send: about {total:,} tokens. Stored so far for {dataset}: {store.tokens:,}.")
    if args.command == "plan":
        return
    if total > args.max_tokens:
        sys.exit(f"Stopped: the estimate exceeds --max-tokens {args.max_tokens:,}.")
    from openai import OpenAI
    client = OpenAI()
    for arm, reqs in requests.items():
        failures = run_requests(reqs, store, client, model, budget=store.tokens + args.max_tokens)
        if failures:
            print(f"{arm}: some calls failed ({failures}); rerun to resume.")
    os.chmod(work / "replies.jsonl", 0o600)


def evaluate(dataset, args):
    """Scores every arm on the split. On dev it also applies rule 1 and 4 when --freeze is given."""
    load_env()
    model = os.environ.get("MDS_LLM_MODEL", "unset")
    notes, gold, items, work, sent, picks = prepared(dataset, args)
    store = Store(work / "replies.jsonl")
    requests = all_requests(dataset, ARMS[dataset], notes, sent, picks, model)
    got = {arm: answers(dataset, arm, reqs, store, notes) for arm, reqs in requests.items()}
    complete = {arm: all(r["key"] in store.done for r in reqs) for arm, reqs in requests.items()}
    frozen_path = RESULTS / dataset / "frozen_choice.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    if args.split == "test" and not frozen:
        sys.exit("Run --freeze on the dev split first: rule 1 fixes the arm before the test notes are scored.")
    out = {"dataset": dataset, "split": args.split, "notes": len(notes), "complete_arms": complete, "model": model}

    if dataset == "i2b2_2008":
        textual = {a: i2b2_predictions(got[a], "textual") for a in ("t_retrieval", "t_status", "full") if complete[a]}
        intuitive = {a: i2b2_predictions(got[a], "intuitive") for a in ("i_retrieval", "full") if complete[a]}
        links = frozen["kg_links"] if frozen else B.DISEASE_TREATMENTS
        base = (frozen or {}).get("textual") or (max(textual, key=lambda a: B.i2b2_scores(i2b2_frame(gold, "textual"), pd.Series(textual[a]), B.TEXTUAL)["macro_f1"]) if textual else None)
        if base in textual:
            intuitive["i_kg"] = kg_intuitive(textual[base], notes, links)
        scores = {}
        for source, preds, classes in (("textual", textual, B.TEXTUAL), ("intuitive", intuitive, B.INTUITIVE)):
            g = i2b2_frame(gold, source)
            names = list(preds)
            boots = macro_f_boot(g, [preds[n] for n in names], classes) if names else []
            scores[source] = {}
            for n, b in zip(names, boots):
                s = B.i2b2_scores(g, pd.Series(preds[n]), classes)
                scores[source][n] = {"macro_f1": round(s["macro_f1"], 4), "macro_f1_ci": ci(b), "micro_f1": round(s["micro_f1"], 4),
                                     "per_class_f1": s["per_class"],
                                     "per_disease_macro_f1": B.i2b2_per_disease(gold[gold["source"] == source], preds[n], classes)}
            scores[source]["contrasts"] = [{"a": a, "b": c, "difference": round(float(boots[i].mean() - boots[j].mean()), 4),
                                            "ci": ci(boots[i] - boots[j])} for i, a in enumerate(names) for j, c in enumerate(names) if i < j]
        out["scores"] = scores
        out["mds_overlap"] = {d: B.I2B2_TO_MDS[d] for d in B.I2B2_TO_MDS}
        if args.freeze:
            choice = {}
            for source in ("textual", "intuitive"):
                ranked = sorted(((v["macro_f1"], a) for a, v in scores[source].items() if a != "contrasts"), reverse=True)
                best = ranked[0][0]
                choice[source] = min((a for f, a in ranked if f >= best - TIE), key=CHEAPER.index)
            kept = {}
            g = gold[gold["source"] == "intuitive"]
            for disease, terms in B.DISEASE_TREATMENTS.items():
                with_link = kg_intuitive(textual[choice["textual"]], notes, {disease: terms})
                without = kg_intuitive(textual[choice["textual"]], notes, {})
                sub = g[g["item"] == disease]
                f = lambda p: B.i2b2_per_disease(sub, p, B.INTUITIVE).get(disease, 0)
                if f(with_link) > f(without):
                    kept[disease] = terms
            choice["kg_links"] = kept
            write(dataset, "frozen_choice.json", choice)
            out["frozen"] = choice
        if args.split == "test":
            out["leaderboard"] = {}
            for source in ("textual", "intuitive"):
                arm = frozen[source]
                if arm in scores[source]:
                    position = B.leaderboard_position(scores[source][arm]["macro_f1"], B.I2B2_2008_TOP10[source], B.I2B2_2008_TEAMS)
                    low, high = scores[source][arm]["macro_f1_ci"]
                    position["frozen_arm"] = arm
                    position["place_range_from_ci"] = [B.leaderboard_position(high, B.I2B2_2008_TOP10[source], 30)["place"],
                                                       B.leaderboard_position(low, B.I2B2_2008_TOP10[source], 30)["place"]]
                    out["leaderboard"][source] = position
    else:
        labels = gold["labels"]
        doc_arms = [a for a in ARMS[dataset] if a != "span" and complete[a]]
        preds = {a: got[a] for a in doc_arms}
        if dataset == "snomed" and "o_retrieval" in preds:
            preds.update(kg_fusions(notes, preds["o_retrieval"], labels))
        names = list(preds)
        boots = ba_boot(labels, [preds[n] for n in names], items) if names else []
        out["note_level"] = {}
        for n, b in zip(names, boots):
            macro, per_item = note_ba(labels, preds[n], items)
            out["note_level"][n] = {"macro_balanced_accuracy": round(macro, 4), "ci": ci(b), "per_item": {k: round(v, 4) for k, v in per_item.items()}}
        out["note_level"]["contrasts"] = [{"a": a, "b": c, "difference": round(float(boots[i].mean() - boots[j].mean()), 4), "ci": ci(boots[i] - boots[j])}
                                          for i, a in enumerate(names) for j, c in enumerate(names) if i < j]
        out["positive_notes"] = labels.groupby("item")["gold"].sum().astype(int).to_dict()
        if dataset == "n2c2_2018" and complete.get("span"):
            spans = B.span_scores(gold["spans"], got["span"])
            out["spans"] = {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for m, s in spans.items()}
            if args.split == "test":
                out["leaderboard"] = {"concept_extraction_lenient_micro_f1": round(spans["lenient"]["micro_f1"], 4),
                                      "published_best": B.N2C2_2018_BEST["concept_extraction"],
                                      "gap_to_best": round(spans["lenient"]["micro_f1"] - B.N2C2_2018_BEST["concept_extraction"], 4),
                                      "note": "Only the best score is public here, so the place among the teams is not known."}
        if args.freeze:
            ranked = sorted(((v["macro_balanced_accuracy"], a) for a, v in out["note_level"].items() if a != "contrasts"), reverse=True)
            choice = {"note_level": min((a for f, a in ranked if f >= ranked[0][0] - TIE), key=lambda a: CHEAPER.index(a) if a in CHEAPER else 99)}
            write(dataset, "frozen_choice.json", choice)
            out["frozen"] = choice
        if dataset == "snomed":
            out["leaderboard"] = "None: the SNOMED CT challenge ranks entity linking, not note-level items."
    write(dataset, f"scores_{args.split}.json", out)
    print(json.dumps(out, indent=1, default=str))


def kg_fusions(notes, note_pred, labels):
    """Knowledge graph arms on SNOMED notes: the note reader fused with the MIMIC-IV records, and the ICU events alone as the old check."""
    from build_kg import build_edges
    ids = notes[["note_key", "hadm_id", "subject_id"]].dropna().astype(str)
    edges = build_edges(ids, note_pred)
    pairs = list(zip(labels["note_key"], labels["item"]))
    frame = source_matrix(edges, {p: bool(note_pred.get(p, False)) for p in pairs}, pairs,
                          ["note", "order", "billing", "prescription", "icu_event"])
    inputs = ["note", "order", "billing", "prescription"]
    out = {}
    for method in FUSIONS:
        answer = fuse(frame, inputs, method)
        out[f"kg_{method}"] = dict(zip(pairs, answer.astype(bool)))
    out["icu_events_only"] = dict(zip(pairs, frame["icu_event"].astype(bool)))
    return {k: v for k, v in out.items() if k != "kg_note"}


def write(dataset, name, data):
    out = RESULTS / dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(data, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ARMS))
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--retrieval", action="store_true")
    group.add_argument("--plan", action="store_true")
    group.add_argument("--run", nargs="+", metavar="ARM")
    group.add_argument("--score", action="store_true")
    group.add_argument("--freeze", action="store_true", help="score the dev split and fix the arms (rules 1 and 4)")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--encode", default="local", choices=["local", "modal"])
    ap.add_argument("--allow-seed-only", action="store_true")
    args = ap.parse_args()
    args.command = "check" if args.check else "retrieval" if args.retrieval else "plan" if args.plan else "run" if args.run else "score"
    if args.run and set(args.run) - set(ARMS[args.dataset]):
        sys.exit(f"Arms for {args.dataset}: {', '.join(ARMS[args.dataset])}")
    if args.freeze and args.split != "dev":
        sys.exit("--freeze uses the dev split only.")
    {"check": check, "retrieval": retrieval, "plan": plan_or_run, "run": plan_or_run, "score": evaluate}[args.command](args.dataset, args)


if __name__ == "__main__":
    main()
