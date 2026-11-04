"""Learned drug-mention extractor for hospital notes, trained by distant supervision, against the hand-written drug lexicon. Runs on Modal.
Rules set before any scoring.

Training (600 admissions outside the pilot and proxy sets, patients excluded too): a note word counts as a drug mention if it is a word of
a drug prescribed to that same admission. Words that occur in the prescriptions of more than 50% of training admissions (fluids, flushes,
generic vehicles) are dropped from the labels; this data-driven filter replaces a hand-written stoplist. A linear-chain CRF with scispaCy POS
and word-shape features learns to tag drug mentions from sentences with at least one label plus a seeded 15% of the others. 60 training
admissions are held out to check the fit against its own weak labels.
Evaluation on the 500 proxy notes, with the nurse medication log as the key (never used in training or by either method):
  given name   first word of 4+ letters of a given drug (log events Administered, Applied, Started and so on), after the same frequency filter
  findable     the given name appears as a word in the note, so a reader could find it
  extractor    the trained model's drug-mention spans, words of 4+ letters
  lexicon      the hand-written dictionary: words of 4+ letters from every item's synonyms and prescription lists in items.py
Recall = findable given names found by each method; micro over (note, name) pairs, CI by resampling notes. Main scope: given in the last 24 hours
before discharge (the MDS-window truth used elsewhere); the whole stay is also given. Names are also split into seen (the word was a positive
training label) and unseen.
Decisions: E1 the extractor's recall exceeds the lexicon's by at least 0.10, with the lower CI bound above 0.05. If E1 holds, the extractor
is worth the token cost of re-tagging the notes it finds; if not, it is not. E2 (reported, not a decision): share of each method's distinct
mentions that are words of that stay's prescriptions or given drugs, a lower bound on precision since home medications count as misses.
E3 (reported): the findable given names the lexicon misses and the extractor finds, with counts.
Only aggregates and drug words leave the job; note text does not.
Usage: modal volume put mds-mimic-private pilot_private/mimic_ner / ; modal run src/modal_mimic_ner.py::main
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import modal

for _p in (Path(__file__).resolve().parent, Path("/app/src")):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/mimic_extractor"
URL_CORE = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz"
FREQ_DROP = .5
KEEP_UNLABELLED = .15
HELD_OUT = 60
SEED = 7
BOOT = 2000

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "pandas", "pyarrow==16.1.0", "spacy==3.7.5", "scikit-learn", "sklearn-crfsuite", "click", "typer", "pysbd", "conllu")
    .pip_install("scispacy==0.5.4", extra_options="--no-deps")
    .pip_install(URL_CORE)
    .add_local_dir(ROOT / "src", "/app/src")
)
app = modal.App("mds-mimic-ner", image=image)
volume = modal.Volume.from_name("mds-mimic-private", create_if_missing=True)
DATA = Path("/data/mimic_ner")


def words(text):
    return re.findall(r"[a-z]{4,}", text.lower()) if isinstance(text, str) else []


def run_extractor(nlp, crf, texts_by_key, spans_only=True):
    """Returns {note: set of drug words} and {note: mention count}."""
    from mds_pilot.ner_features import spans_from_tokens, token_features
    from mds_pilot.segment import segment_note
    names, counts, where = {}, {}, {}
    for key, text in texts_by_key.items():
        rows = segment_note(text)
        found = set()
        n = 0
        where[key] = []
        for row, doc in zip(rows, nlp.pipe([r["text"] for r in rows], batch_size=256)):
            if not len(doc):
                continue
            tags = crf.predict_single(token_features(doc, None, False))
            for _, s, e in spans_from_tokens([(t.idx, t.idx + len(t)) for t in doc], tags):
                found.update(words(doc.text[s:e]))
                where[key].append([row["sent_id"], s, e, " ".join(words(doc.text[s:e]))])
                n += 1
        names[key], counts[key] = found, n
    return names, counts, where


@app.function(volumes={"/data": volume}, cpu=16.0, memory=32768, timeout=3 * 3600)
def train_and_evaluate():
    import numpy as np
    import pandas as pd
    import sklearn_crfsuite
    import spacy
    from mds_pilot.items import ITEMS
    from mds_pilot.ner_features import bio_from_gold, token_features
    from mds_pilot.segment import segment_note

    train_notes = pd.read_parquet(DATA / "train_notes.parquet")
    train_rx = pd.read_parquet(DATA / "train_rx.parquet")
    proxy_notes = pd.read_parquet(DATA / "proxy_notes.parquet")
    proxy_rx = pd.read_parquet(DATA / "proxy_rx.parquet")
    given = pd.read_parquet(DATA / "proxy_given.parquet")

    rx_words = {k: {w for d in g["drug"] for w in words(d)} for k, g in train_rx.groupby("key")}
    freq = Counter(w for ws in rx_words.values() for w in ws)
    drop = {w for w, c in freq.items() if c / len(rx_words) > FREQ_DROP}
    labels = {k: ws - drop for k, ws in rx_words.items()}

    nlp = spacy.load("en_core_sci_md", disable=["ner", "parser", "lemmatizer"])
    keys = sorted(train_notes["key"])
    fit_keys, held_keys = keys[:-HELD_OUT], keys[-HELD_OUT:]
    text_of = dict(zip(train_notes["key"], train_notes["text"]))
    rng = random.Random(SEED)

    def sentences(key_list, balance):
        out = []
        for k in key_list:
            for r in segment_note(text_of[k]):
                out.append((k, r["text"]))
        return out

    def label_sentence(doc, key):
        gold = []
        for t in doc:
            if t.text.lower() in labels[key] and len(t.text) >= 4:
                gold.append((t.idx, t.idx + len(t), "DRUG"))
        return gold

    def build(key_list, keep_unlabelled):
        X, Y, n_pos = [], [], 0
        pairs = sentences(key_list, False)
        for (k, _), doc in zip(pairs, nlp.pipe([p[1] for p in pairs], batch_size=256)):
            if not len(doc):
                continue
            gold = label_sentence(doc, k)
            if not gold and rng.random() > keep_unlabelled:
                continue
            n_pos += bool(gold)
            X.append(token_features(doc, None, False))
            Y.append(bio_from_gold(doc, gold))
        return X, Y, n_pos

    X, Y, n_pos = build(fit_keys, KEEP_UNLABELLED)
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    crf.fit(X, Y)
    Xh, Yh, _ = build(held_keys, 1.0)
    flat = lambda seqs: [t for s in seqs for t in s]
    pred = flat(crf.predict(Xh))
    gold_flat = flat(Yh)
    tp = sum(p != "O" and g != "O" for p, g in zip(pred, gold_flat))
    fp = sum(p != "O" and g == "O" for p, g in zip(pred, gold_flat))
    fn = sum(p == "O" and g != "O" for p, g in zip(pred, gold_flat))
    held = {"precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4), "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4)}
    train_vocab = set().union(*(labels[k] for k in fit_keys))

    texts = dict(zip(proxy_notes["note_key"], proxy_notes["text"]))
    ext, mention_counts, ext_where = run_extractor(nlp, crf, texts)
    lex = {w for it in ITEMS for term in (*it.synonyms, *it.rx) for w in words(term)}
    # Added after the first results (an unfair comparison: the list above includes diagnosis words): drug lists only.
    lex_drug = {w for it in ITEMS if it.section == "N" or it.id == "O_chemotherapy" for term in (*it.synonyms, *it.rx) for w in words(term)}
    note_words = {k: set(words(t)) for k, t in texts.items()}
    stay_words = {k: {w for d in g["drug"] for w in words(d)} for k, g in proxy_rx.groupby("note_key")}

    def given_name(med):
        for w in words(med):
            if w not in drop:
                return w
        return None

    given["name"] = given["medication"].map(given_name)
    given = given.dropna(subset=["name"])
    results = {}
    for scope, frame in (("last24", given[given["last24"]]), ("stay", given)):
        rows = []
        for note, g in frame.groupby("note_key"):
            for name in set(g["name"]):
                if name in note_words[note]:  # findable
                    rows.append({"note": note, "name": name, "ext": name in ext[note], "lex": name in lex, "lexd": name in lex_drug, "seen": name in train_vocab})
        df = pd.DataFrame(rows)
        by_note = {n: g for n, g in df.groupby("note")}
        notes = sorted(by_note)
        brng = np.random.default_rng(0)

        def recall(sel, col):
            return float(np.mean([r for n in sel for r in by_note[n][col]])) if sel else float("nan")

        def diff(sel, mask=None):
            d = df if mask is None else df[mask]
            return float(d["ext"].mean() - d["lex"].mean())

        sample_idx = [brng.integers(0, len(notes), len(notes)) for _ in range(BOOT)]
        vals = []
        for idx in sample_idx:
            d = pd.concat([by_note[notes[i]] for i in idx])
            vals.append(float(d["ext"].mean() - d["lex"].mean()))
        point = float(df["ext"].mean() - df["lex"].mean())
        res = {"admissions": len(notes), "findable_pairs": len(df), "distinct_names": int(df["name"].nunique()),
               "extractor_recall": round(float(df["ext"].mean()), 4), "lexicon_recall": round(float(df["lex"].mean()), 4),
               "difference": round(point, 4), "ci": [round(float(np.quantile(vals, .025)), 4), round(float(np.quantile(vals, .975)), 4)],
               "by_seen": {s: {"pairs": int((df["seen"] == (s == "seen")).sum()), "extractor_recall": round(float(df.loc[df["seen"] == (s == "seen"), "ext"].mean()), 4),
                               "lexicon_recall": round(float(df.loc[df["seen"] == (s == "seen"), "lex"].mean()), 4)} for s in ("seen", "unseen")
                           if (df["seen"] == (s == "seen")).any()}}
        new = df[df["ext"] & ~df["lex"]]
        res["E3_found_by_extractor_missed_by_lexicon"] = {"pairs": len(new), "distinct_names": int(new["name"].nunique()),
                                                          "top": Counter(new["name"]).most_common(20)}
        res["lexicon_misses"] = {"pairs": int((~df["lex"]).sum()), "share": round(float((~df["lex"]).mean()), 4)}
        res["drug_lexicon_recall"] = round(float(df["lexd"].mean()), 4)
        res["pairs"] = df.to_dict("records")
        results[scope] = res
    prec = {}
    for label, getter in (("extractor", lambda k: ext[k]), ("lexicon", lambda k: lex & note_words[k])):
        good = total = 0
        for k in texts:
            real = stay_words.get(k, set()) | set(given.loc[given["note_key"] == k, "name"])
            names = getter(k)
            total += len(names)
            good += len(names & real)
        prec[label] = {"distinct_mentions": total, "in_stay_records": good, "share": round(good / max(total, 1), 4)}
    return {"ext_mentions": ext_where, "ext_words": {k: sorted(v) for k, v in ext.items()}, "lexicon_drug_words": len(lex_drug), "training": {"admissions_fit": len(fit_keys), "labelled_sentences": n_pos, "sentences_used": len(X), "dropped_frequent_words": sorted(drop),
                         "positive_label_vocab": len(train_vocab), "held_out_vs_weak_labels": held},
            "extractor": {"mentions_per_note_median": float(np.median(list(mention_counts.values()))), "notes": len(texts)},
            "lexicon_words": len(lex), "evaluation": results, "E2_precision_lower_bound": prec}


@app.local_entrypoint()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = train_and_evaluate.remote()
    (OUT / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
