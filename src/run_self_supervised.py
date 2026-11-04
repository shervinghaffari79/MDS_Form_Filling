"""Label-free retrieval tests. All computation is local and output is aggregate only.

Passing shows a component behaves well on constructed tests. It does not estimate true recall.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS, ITEM_IDS
from mds_pilot.known_facts import FACTS, TIERS
from mds_pilot.retrieval import (BM25, ENCODERS, embed_sentences, lexicon_scores, load_encoder, note_index_from,
                                 term_pattern)

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private"
RESULTS = ROOT / "results"
KS = (5, 10, 20)
RRF_K = 60
BOOT = 1000


def load_sample(name):
    sent = pd.read_parquet(PRIVATE / f"{name}_sentences.parquet").sort_values(["note_key", "sent_id"]).reset_index(drop=True)
    labels = pd.read_parquet(PRIVATE / f"{name}_labels.parquet").set_index("note_key")
    index = note_index_from(sent["note_key"].values)
    keys = [sent["note_key"].iloc[i[0]] for i in index]
    return sent, labels.loc[keys], index, keys


def cached_embeddings(sample, texts, name):
    path = PRIVATE / f"emb_{sample}_{name}.npy"
    if path.exists():
        return np.load(path)
    emb, _ = embed_sentences(texts, name)
    np.save(path, emb)
    return emb


def encode_queries(model, name, texts):
    prefix = ENCODERS[name][1]
    return model.encode([prefix + t for t in texts], normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True).astype(np.float32)


def rank_of_target(real, target):
    """Rank of the target among real sentences; ties count against the target."""
    return 1 + int((real >= target).sum())


def fused_rank(rank_lists):
    """Rank of the final element after reciprocal rank fusion over the supplied score arrays."""
    fused = np.zeros(len(rank_lists[0]))
    for scores in rank_lists:
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores))
        ranks[order] = np.arange(1, len(scores) + 1)
        fused += (scores > 0) / (RRF_K + ranks)
    return 1 + int((fused[:-1] >= fused[-1]).sum())


def injected_fact_test(sent, labels, index, encoders):
    """Rank invented facts against real sentences in notes silent on that item."""
    fact_texts, fact_keys = [], []
    for j, item in enumerate(ITEMS):
        for t, text in enumerate(FACTS[item.id]):
            fact_texts.append(text)
            fact_keys.append((j, t))
    texts = sent["text"].tolist()
    n = len(texts)
    bm25 = BM25(texts + fact_texts)
    methods = {}
    methods["bm25_def"] = np.column_stack([bm25.score([i.definition]) for i in ITEMS])
    methods["bm25_syn"] = np.column_stack([bm25.score([i.definition, *i.synonyms]) for i in ITEMS])
    for name, (emb, model) in encoders.items():
        fact_emb = encode_queries(model, "mpnet", fact_texts)  # no query prefix for passages
        all_emb = np.vstack([emb, fact_emb])
        d, s = [], []
        for item in ITEMS:
            d.append((all_emb @ encode_queries(model, name, [item.definition]).T).max(axis=1))
            s.append((all_emb @ encode_queries(model, name, [item.definition, *item.synonyms]).T).max(axis=1))
        methods[f"{name}_def"], methods[f"{name}_syn"] = np.column_stack(d), np.column_stack(s)
        del all_emb
    enc = list(encoders)
    fusions = {f"fusion_bm25+{e}": ["bm25_syn", f"{e}_syn"] for e in enc}
    if len(enc) > 1:
        fusions["fusion_all"] = ["bm25_syn", *[f"{e}_syn" for e in enc]]

    lex_real = lexicon_scores(texts) > 0
    lex_fact = lexicon_scores(fact_texts) > 0
    rows = []
    for n_i, idx in enumerate(index):
        note = sent["note_key"].iloc[idx[0]]
        for f, (j, t) in enumerate(fact_keys):
            item = ITEMS[j]
            if lex_real[idx, j].any() or (item.has_proxy and labels.loc[note, item.id] == 1):
                continue
            row = {"note_key": note, "item": item.id, "section": item.section, "tier": TIERS[t],
                   "lexicon_any_rank": 1 if lex_fact[f, j] else 10**6}
            for m, scores in methods.items():
                row[f"{m}_rank"] = rank_of_target(scores[idx, j], scores[n + f, j])
            for fname, members in fusions.items():
                arrays = [np.append(methods[m][idx, j], methods[m][n + f, j]) for m in members]
                row[f"{fname}_rank"] = fused_rank(arrays)
            rows.append(row)
    return pd.DataFrame(rows)


def recall_table(frame, group, ks=KS):
    method_cols = [c for c in frame.columns if c.endswith("_rank")]
    rows = []
    for key, g in frame.groupby(group):
        notes = g["note_key"].unique()
        for col in method_cols:
            m = col[:-5]
            row = {group: key, "method": m, "pairs": len(g), "notes": len(notes)}
            for k in ks:
                if m == "lexicon_any" and k != ks[0]:
                    row[f"recall@{k}"] = row[f"recall@{ks[0]}"]
                    continue
                row[f"recall@{k}"] = round(float((g[col] <= k).mean()), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def contrast(frame, a, b, k=20, seed=0):
    """Paired bootstrap over notes for recall@k of a minus b, per tier."""
    rng = np.random.default_rng(seed)
    rows = []
    for tier, g in frame.groupby("tier"):
        per_note = g.assign(a=g[f"{a}_rank"] <= k, b=g[f"{b}_rank"] <= k).groupby("note_key")[["a", "b"]].agg(["sum", "size"])
        hits_a, hits_b, sizes = per_note[("a", "sum")].values, per_note[("b", "sum")].values, per_note[("a", "size")].values
        idx = rng.integers(0, len(sizes), size=(BOOT, len(sizes)))
        diffs = (hits_a[idx].sum(axis=1) - hits_b[idx].sum(axis=1)) / sizes[idx].sum(axis=1)
        low, high = np.quantile(diffs, [.025, .975])
        rows.append({"tier": tier, "test": f"{a} minus {b}", "k": k,
                     "diff": round(float((hits_a.sum() - hits_b.sum()) / sizes.sum()), 4),
                     "ci_low": round(float(low), 4), "ci_high": round(float(high), 4),
                     "verdict": "improves" if low > 0 else "worse" if high < 0 else "no clear difference"})
    return rows


def heldout_synonym_test(sent, index, encoders):
    """Hide a real term from the query, then look for sentences that contain only that term."""
    texts = sent["text"].tolist()
    bm25 = BM25(texts)
    rows = []
    for item in ITEMS:
        terms = [t for t in item.synonyms]
        hits = np.column_stack([[bool(term_pattern(t).search(x)) for x in texts] for t in terms])
        sims = {}
        for name, (emb, model) in encoders.items():
            sims[name] = (emb, encode_queries(model, name, [item.definition] + terms))
        for t_i, term in enumerate(terms):
            if len(term) <= 3 or not hits[:, t_i].any():
                continue
            overlap = [o for o, other in enumerate(terms)
                       if o != t_i and (term.lower() in other.lower() or other.lower() in term.lower())]
            kept = [o for o in range(len(terms)) if o != t_i and o not in overlap]
            kept_terms = [terms[o] for o in kept]
            bm_scores = bm25.score([item.definition, *kept_terms])
            enc_scores = {}
            for name, (emb, q) in sims.items():
                select = [0] + [o + 1 for o in kept]
                enc_scores[name] = (emb @ q[select].T).max(axis=1)
            for idx in index:
                target = hits[idx, t_i]
                others = hits[idx][:, [o for o in range(len(terms)) if o != t_i]]
                if not target.any() or others.any():
                    continue
                row = {"item": item.id, "term": term, "note_key": sent["note_key"].iloc[idx[0]]}
                arrays = {"bm25_syn": bm_scores[idx]}
                arrays.update({f"{n}_syn": s[idx] for n, s in enc_scores.items()})
                ranks = {}
                for m, scores in arrays.items():
                    order = np.argsort(-scores, kind="stable")
                    r = np.empty(len(scores), dtype=int)
                    r[order] = np.arange(1, len(scores) + 1)
                    ranks[m] = r
                    row[f"{m}_rank"] = int(r[target].min())
                enc_names = [f"{e}_syn" for e in encoders]
                for fname, members in {**{f"fusion_bm25+{e}": ["bm25_syn", f"{e}_syn"] for e in encoders},
                                       **({"fusion_all": ["bm25_syn", *enc_names]} if len(enc_names) > 1 else {})}.items():
                    fused = sum((arrays[m] > 0) / (RRF_K + ranks[m]) for m in members)
                    order = np.argsort(-fused, kind="stable")
                    r = np.empty(len(fused), dtype=int)
                    r[order] = np.arange(1, len(fused) + 1)
                    row[f"{fname}_rank"] = int(r[target].min())
                rows.append(row)
    return pd.DataFrame(rows)


def paraphrase_stability(sent, index, encoders, k=10):
    """Self-consistency of top-k across query wordings. Stable retrieval can still be wrong."""
    texts = sent["text"].tolist()
    bm25 = BM25(texts)
    rows = []
    for item in ITEMS:
        wordings = [item.definition, item.synonyms[0], f"documentation of {item.synonyms[0]} in this admission"]
        scores = {"bm25": [bm25.score([w]) for w in wordings]}
        for name, (emb, model) in encoders.items():
            scores[name] = list((emb @ encode_queries(model, name, wordings).T).T)
        for m, arrs in scores.items():
            vals = []
            for idx in index:
                tops = [set(np.argsort(-a[idx], kind="stable")[:k]) for a in arrs]
                for x in range(len(tops)):
                    for y in range(x + 1, len(tops)):
                        vals.append(len(tops[x] & tops[y]) / len(tops[x] | tops[y]))
            rows.append({"item": item.id, "method": m, "mean_jaccard_top10": round(float(np.mean(vals)), 3)})
    return pd.DataFrame(rows)


def agreement_diagnostics(sent, labels, index, encoders, keys):
    """Spectral-meta-learner style view of independent signals, with a dependence diagnostic.

    A rank-one fit of the off-diagonal covariance is what conditional independence implies.
    Poor fit means the view weights should not be read as relative accuracies.
    """
    texts = sent["text"].tolist()
    lex = lexicon_scores(texts) > 0
    bm25 = BM25(texts)
    rows, pair_rows = [], []
    for j, item in enumerate(ITEMS):
        if not item.has_proxy:
            continue
        lex_note = np.array([lex[idx, j].any() for idx in index], dtype=float)
        prev = max(float(lex_note.mean()), 0.02)
        views = {"lexicon": lex_note, "tables": labels[item.id].to_numpy(dtype=float)}
        bm = bm25.score([item.definition])
        views["bm25_def"] = np.array([bm[idx].max() for idx in index])
        for name, (emb, model) in encoders.items():
            s = (emb @ encode_queries(model, name, [item.definition]).T).max(axis=1)
            views[f"{name}_def"] = np.array([s[idx].max() for idx in index])
        votes = {}
        for v, arr in views.items():
            if v in ("lexicon", "tables"):
                votes[v] = 2 * arr - 1
            else:
                votes[v] = np.where(arr >= np.quantile(arr, 1 - prev), 1.0, -1.0)
        names = list(votes)
        matrix = np.column_stack([votes[v] for v in names])
        if matrix.std(axis=0).min() == 0:
            continue
        cov = np.cov(matrix, rowvar=False)
        off = ~np.eye(len(names), dtype=bool)
        fitted = cov.copy()
        for _ in range(100):
            np.fill_diagonal(fitted, np.diag(fitted))
            w, v = np.linalg.eigh(fitted)
            approx = w[-1] * np.outer(v[:, -1], v[:, -1])
            np.fill_diagonal(fitted, np.diag(approx))
        vec = v[:, -1] * np.sign(v[:, -1].sum())
        residual = cov - approx
        fit = 1 - (residual[off] ** 2).sum() / max((cov[off] ** 2).sum(), 1e-12)
        rows.append({"item": item.id, "rank1_fit_r2": round(float(fit), 3),
                     **{f"weight_{n}": round(float(x), 3) for n, x in zip(names, vec)}})
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                pair_rows.append({"item": item.id, "pair": f"{names[a]}|{names[b]}",
                                  "phi": round(float(np.corrcoef(matrix[:, a], matrix[:, b])[0, 1]), 3)})
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def run(sample, encoder_names):
    sent, labels, index, keys = load_sample(sample)
    texts = sent["text"].tolist()
    encoders = {}
    for name in encoder_names:
        emb = cached_embeddings(sample, texts, name)
        encoders[name] = (emb, load_encoder(name))
        print(f"[progress] encoded {name}", flush=True)
    out = RESULTS / f"self_supervised_{sample}"
    out.mkdir(parents=True, exist_ok=True)
    enc = list(encoders)
    syn_all = "fusion_all" if len(enc) > 1 else f"fusion_bm25+{enc[0]}"

    inj = injected_fact_test(sent, labels, index, encoders)
    inj_tier = recall_table(inj, "tier")
    inj_section = recall_table(inj[inj["tier"] != "explicit"], "section")
    tests = []
    pairs = [("bm25_syn", "bm25_def"), ("bm25_syn", "lexicon_any")]
    for e in enc:
        pairs += [(f"{e}_syn", f"{e}_def"), (f"{e}_syn", "bm25_syn"), (f"fusion_bm25+{e}", "bm25_syn"), (f"fusion_bm25+{e}", "lexicon_any")]
    if len(enc) > 1:
        pairs += [("fusion_all", f"fusion_bm25+{e}") for e in enc]
    for a, b in pairs:
        tests += contrast(inj, a, b)
    tests = pd.DataFrame(tests)
    item_cols = [f"{syn_all}_rank", "bm25_syn_rank"]
    by_item = inj.groupby(["item", "tier"]).apply(
        lambda g: pd.Series({"silent_notes": len(g), **{f"{c[:-5]}@20": round(float((g[c] <= 20).mean()), 3) for c in item_cols}}),
        include_groups=False).reset_index()

    print("[progress] injected facts done", flush=True)
    held = heldout_synonym_test(sent, index, encoders)
    print("[progress] held-out terms done", flush=True)
    if held.empty:
        held_summary = pd.DataFrame()
    else:
        cols = [c for c in held.columns if c.endswith("_rank")]
        per_term = held.groupby(["item", "term"]).apply(
            lambda g: pd.Series({"notes": g["note_key"].nunique(),
                                 **{f"{c[:-5]}@20": float((g[c] <= 20).mean()) for c in cols}}), include_groups=False).reset_index()
        held_summary = pd.DataFrame([{"method": c[:-5], "terms": len(per_term),
                                      "pooled_recall@10": round(float((held[c] <= 10).mean()), 4),
                                      "pooled_recall@20": round(float((held[c] <= 20).mean()), 4),
                                      "macro_term_recall@20": round(float(per_term[f"{c[:-5]}@20"].mean()), 4),
                                      "pairs": len(held)} for c in cols])
        per_term.round(3).to_csv(out / "heldout_by_term.csv", index=False)

    stability = paraphrase_stability(sent, index, encoders)
    agreement, pairs_df = agreement_diagnostics(sent, labels, index, encoders, keys)

    inj_tier.to_csv(out / "injected_recall_by_tier.csv", index=False)
    inj_section.to_csv(out / "injected_recall_by_section.csv", index=False)
    tests.to_csv(out / "injected_component_tests.csv", index=False)
    by_item.to_csv(out / "injected_recall_by_item.csv", index=False)
    held_summary.to_csv(out / "heldout_summary.csv", index=False)
    stability.to_csv(out / "query_stability.csv", index=False)
    agreement.to_csv(out / "agreement_views.csv", index=False)
    pairs_df.to_csv(out / "agreement_pairs.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 300)
    print(f"=== {sample}: {len(index)} notes, {len(sent)} sentences")
    print(inj_tier.to_string())
    print(tests.to_string())
    print(held_summary.to_string())
    print(stability.groupby("method")["mean_jaccard_top10"].mean().round(3).to_string())
    if not agreement.empty:
        print("median rank1 fit:", agreement["rank1_fit_r2"].median())
        print(agreement.drop(columns=["item", "rank1_fit_r2"]).median().round(3).to_string())
        print(pairs_df.groupby("pair")["phi"].median().round(3).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="pilot", choices=["pilot", "proxy"])
    ap.add_argument("--encoders", default="bge,mpnet")
    args = ap.parse_args()
    run(args.sample, [e for e in args.encoders.split(",") if e])
