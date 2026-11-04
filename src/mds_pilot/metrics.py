"""Evaluation metrics, with bootstrap resampling at note level."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(labels, scores):
    labels = np.asarray(labels)
    if labels.sum() < 5 or (1 - labels).sum() < 5:
        return np.nan
    return float(roc_auc_score(labels, scores))


def bootstrap_diff(values_a, values_b, n=2000, seed=0):
    """Paired percentile CI for mean(a - b), resampling the supplied units."""
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = (a[idx] - b[idx]).mean(axis=1)
    return float((a - b).mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def weighted_recall(hits, weights):
    hits, weights = np.asarray(hits, float), np.asarray(weights, float)
    return float((hits * weights).sum() / weights.sum()) if weights.sum() else np.nan


def stratified_bootstrap_recall(table, n=2000, seed=0):
    """table: rows of (note, stratum, weight, hit). Resamples notes within stratum."""
    rng = np.random.default_rng(seed)
    notes_by_stratum = {s: g["note_key"].unique() for s, g in table.groupby("stratum")}
    groups = {k: g for k, g in table.groupby("note_key")}
    vals = []
    for _ in range(n):
        h, w = 0.0, 0.0
        for notes in notes_by_stratum.values():
            for note in rng.choice(notes, size=len(notes), replace=True):
                g = groups[note]
                h += (g["hit"].values * g["weight"].values).sum()
                w += g["weight"].values.sum()
        vals.append(h / w if w else np.nan)
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def cohen_kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan
