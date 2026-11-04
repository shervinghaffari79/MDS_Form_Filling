"""Scorers for each candidate retrieval component. All computation stays on this machine."""
import os
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .items import ITEMS

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ENCODERS = {
    "bge": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""),
    "biolord": ("FremyCompany/BioLORD-2023", ""),
}
_STOP = {"patient", "has", "had", "is", "a", "an", "the", "or", "of", "for", "taking", "received", "requiring", "medication"}


def term_pattern(term):
    """Short all-caps abbreviations are case-sensitive; other terms are not."""
    body = r"\b" + re.escape(term) + r"\b"
    if term.isupper() and len(term) <= 5:
        return re.compile(body)
    return re.compile(body, re.I)


class BM25:
    """BM25 with document statistics taken from the whole sentence collection."""

    def __init__(self, texts, k1=1.2, b=0.5):
        self.vec = CountVectorizer(token_pattern=r"[A-Za-z0-9][A-Za-z0-9/\-]*", lowercase=True)
        tf = self.vec.fit_transform(texts).tocsc().astype(np.float32)
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        n = tf.shape[0]
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1e-9))
        coo = tf.tocoo()
        values = coo.data * (k1 + 1) / (coo.data + norm[coo.row])
        self.tf = sparse.csc_matrix((values, (coo.row, coo.col)), shape=tf.shape)

    def score(self, terms):
        vocab = self.vec.vocabulary_
        tokens = {t for term in terms for t in re.findall(r"[a-z0-9][a-z0-9/\-]*", term.lower())} - _STOP
        ids = [vocab[t] for t in tokens if t in vocab]
        if not ids:
            return np.zeros(self.tf.shape[0], dtype=np.float32)
        return np.asarray(self.tf[:, ids].dot(self.idf[ids])).ravel().astype(np.float32)


def lexicon_scores(texts, items=ITEMS):
    out = np.zeros((len(texts), len(items)), dtype=np.float32)
    for j, item in enumerate(items):
        pats = [term_pattern(t) for t in item.synonyms]
        for i, text in enumerate(texts):
            if any(p.search(text) for p in pats):
                out[i, j] = 1.0
    return out


def bm25_scores(texts, arm, items=ITEMS):
    index = BM25(texts)
    out = np.zeros((len(texts), len(items)), dtype=np.float32)
    for j, item in enumerate(items):
        terms = [item.definition] if arm == "def" else [item.definition, *item.synonyms]
        out[:, j] = index.score(terms)
    return out


def load_encoder(name, device=None):
    import torch
    from sentence_transformers import SentenceTransformer
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return SentenceTransformer(ENCODERS[name][0], device=device)


def embed_sentences(texts, name, batch_size=None):
    import torch
    batch_size = batch_size or (256 if torch.cuda.is_available() else 32)
    model = load_encoder(name)
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True)
    return emb.astype(np.float32), model


def align_cached_embeddings(cached, sent):
    """Reorder embeddings saved in (note_key, sent_id) order to match the rows of sent."""
    order = sent.reset_index(drop=True).sort_values(["note_key", "sent_id"]).index.to_numpy()
    aligned = np.empty_like(cached)
    aligned[order] = cached
    return aligned


def encoder_scores(emb, model, name, arm, items=ITEMS):
    prefix = ENCODERS[name][1]
    out = np.zeros((emb.shape[0], len(items)), dtype=np.float32)
    for j, item in enumerate(items):
        queries = [item.definition] if arm == "def" else [item.definition, *item.synonyms]
        q = model.encode([prefix + t for t in queries], normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        out[:, j] = (emb @ q.T).max(axis=1)
    return out


def within_note_ranks(scores, note_index):
    """1-based rank of each sentence inside its note, per item; ties broken by reading order."""
    ranks = np.zeros(scores.shape, dtype=np.int32)
    for idx in note_index:
        block = scores[idx]
        order = np.argsort(-block, axis=0, kind="stable")
        r = np.empty_like(order)
        rows = np.arange(len(idx))[:, None]
        np.put_along_axis(r, order, rows + 1, axis=0)
        ranks[idx] = r
    return ranks


def rrf(rank_arrays, k=60, score_arrays=None):
    """Reciprocal rank fusion. A method votes only for sentences it scored above zero when scores are given."""
    if score_arrays is None:
        return sum(1.0 / (k + r) for r in rank_arrays).astype(np.float32)
    return sum((s > 0) / (k + r) for r, s in zip(rank_arrays, score_arrays)).astype(np.float32)


def candidate_rows(note_keys, sent_ids, score_arrays, index, k, items=ITEMS):
    """Fuse the score arrays and keep each note's top k sentences per item. Rows hold IDs and ranks, never text."""
    ranks = within_note_ranks(rrf([within_note_ranks(a, index) for a in score_arrays], score_arrays=score_arrays), index)
    rows = []
    for j, item in enumerate(items):
        for pos in np.flatnonzero(ranks[:, j] <= k):
            rows.append({"note_key": note_keys[pos], "item": item.id, "sent_id": int(sent_ids[pos]), "rank": int(ranks[pos, j])})
    return rows


def note_index_from(note_keys):
    keys = np.asarray(note_keys)
    _, first = np.unique(keys, return_index=True)
    return [np.flatnonzero(keys == keys[i]) for i in np.sort(first)]
