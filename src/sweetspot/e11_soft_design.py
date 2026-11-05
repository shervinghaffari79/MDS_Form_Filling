"""E11: soft weighting of independent signals, and a leave-one-out search for a stable, powerful design (CRF dev split, cached outputs).

Free by construction: cached gpt-6-luna outputs for direct, graph, graph+audit and the per-item reader on the 80 development notes,
local encoder / NLI / GLiNER models, no API call (SWEETSPOT_MAX_USD=0 makes any uncached request fail instead of spending), no test look.

Candidates: every (item, value) any of the four methods gives for a note. Signals, each in [0, 1]:
  votes   share of the four methods (direct, graph, graph+audit, reader) giving exactly this value
  enc     soft relevance: 1 - rank/(n_items-1) of the item by its best encoder similarity (BGE-small) to a sentence of the note
  nli     NLI entailment probability (DeBERTa) of the statement "item/value" from the 3 sentences most similar to the item, maximum;
          for measured values 1 if the value occurs in the note, else 0
  ner     GLiNER-BioMed score (threshold 0.3) of the item name as a label in the note, 0 if not found
  status  share of the graph's linked mentions that are about the patient, affirmed or negated (not uncertain) and current or
          chronic; absent when the graph linked no mention (the score then averages the remaining signals)
Score = weighted mean of the available signals (equal weights unless stated; no fitted weights). Per item the highest-scoring value
is kept when its score >= tau; tau is the only tuned quantity, on the grid 0.20 to 0.80 in steps of 0.05.
Designs: all five signals; leave-one-out (each signal removed); add-one-in (votes plus one other signal); votes only; votes counted
twice; references: direct, agree(direct, graph+audit).
Stability, fixed before any run: 5-fold cross-validation over notes (seed 0); tau chosen on four folds, scored on the fifth.
Selected design = highest cross-validated macro-F1 among designs that beat direct in at least 4 of 5 folds and whose in-sample tau
plateau (grid points within 0.005 of the best) spans at least 3 points. Reported: dev macro-F1 at the best tau, cross-validated mean
and spread, plateau, values set where gold is unknown, accuracy on documented items, paired difference against direct.
Output: results/sweetspot/e11_soft_design.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e4_crf_benchmark as E  # noqa: E402
from sweetspot import e8_smoke as S  # noqa: E402
from sweetspot.llm import LLM, spent_usd  # noqa: E402

OUT = ROOT / "results/sweetspot/e11_soft_design.json"
SIGNALS = ["votes", "enc", "nli", "ner", "status"]
TAUS = [round(t, 2) for t in np.arange(0.20, 0.801, 0.05)]
norm = S.norm


def softmax_entail(logits, ent):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, ent]


def gliner_scores(notes, items):
    from gliner import GLiNER
    model = GLiNER.from_pretrained("Ihor/gliner-biomed-base-v1.0")
    out = {}
    for k, text in notes.items():
        best = defaultdict(float)
        for c in range(0, len(items), 20):
            for e in model.predict_entities(text[:2000], items[c:c + 20], threshold=0.3):
                best[e["label"]] = max(best[e["label"]], float(e["score"]))
        out[k] = best
    return out


def status_share(mentions):
    if not mentions:
        return None
    ok = [m["subject"] == "patient" and m["assertion"] in ("affirmed", "negated") and m["timing"] in ("current", "chronic_ongoing") for m in mentions]
    return sum(ok) / len(ok)


def build_candidates(docs, opts, loc, methods, evidence, ner):
    items = list(opts)
    cands = {}
    for d in docs:
        k = d["id"]
        rank = loc.relevance_rank(k)
        per_item = defaultdict(dict)
        for name, p in methods.items():
            for i, v in p[k].items():
                per_item[i].setdefault(v, []).append(name)
        pairs, where = [], []
        rows = {}
        for i, vals in per_item.items():
            for v, who in vals.items():
                sig = {"votes": len(who) / len(methods), "enc": 1 - rank[i] / (len(items) - 1), "ner": ner[k].get(i, 0.0),
                       "status": status_share(evidence[k].get(i, []))}
                if S.kind(opts[i]) == "measured":
                    sig["nli"] = 1.0 if S.norm(v) in S.norm(d["note"]) else 0.0
                else:
                    j = items.index(i)
                    top = np.argsort(-loc.sim[k][:, j])[:3] if len(loc.sim[k]) else []
                    for t in top:
                        pairs.append((loc.sents[k][t], loc.hypothesis(i, v)))
                        where.append((i, v))
                    sig["nli"] = 0.0
                rows[(i, v)] = sig
        if pairs:
            probs = softmax_entail(loc.nli.predict(pairs, batch_size=32), loc.ent)
            for (i, v), p in zip(where, probs):
                rows[(i, v)]["nli"] = max(rows[(i, v)]["nli"], float(p))
        cands[k] = rows
    return cands


def decide(cands, weights, tau):
    """weights: dict signal -> weight (0 drops the signal). Returns {note: {item: value}}."""
    out = {}
    for k, rows in cands.items():
        best = {}
        for (i, v), sig in rows.items():
            num = den = 0.0
            for s, w in weights.items():
                if w and sig.get(s) is not None:
                    num += w * sig[s]
                    den += w
            score = num / den if den else 0.0
            if score >= tau and (i not in best or score > best[i][0]):
                best[i] = (score, v)
        out[k] = {i: v for i, (s, v) in best.items()}
    return out


def evaluate(docs, preds):
    per = E.official_score(docs, preds)
    fp = sum(1 for d in docs for i, g in d["gold"].items() if g == E.UNK and preds[d["id"]].get(i, E.UNK) != E.UNK)
    return per, fp, E.known_accuracy(docs, preds)


def main():
    from sentence_transformers import SentenceTransformer
    opts = E.options()
    docs = E.load("development")
    g, _, _, _, extra = E.graph_arms(docs, opts, "development", SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"))
    direct = {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, LLM("e4_direct").run(E.direct_requests(docs, opts)))}
    reqs, where = E.reader_requests(docs, opts)
    reader = {d["id"]: {} for d in docs}
    for (k, i), r in zip(where, LLM("e4_reader", workers=32).run(reqs)):
        if r and E.valid(i, r["parsed"]["value"], opts) != E.UNK:
            reader[k][i] = E.valid(i, r["parsed"]["value"], opts)
    methods = {"direct": direct, "graph": g["graph"], "graph+audit": g["graph+audit"], "reader": reader}
    notes = {d["id"]: d["note"] for d in docs}
    loc = S.Local(docs, opts)
    ner = gliner_scores(notes, list(opts))
    cands = build_candidates(docs, opts, loc, methods, extra["evidence"]["graph"], ner)
    ref = {"direct": direct,
           "agree(direct,graph+audit)": {k: {i: v for i, v in direct[k].items() if g["graph+audit"][k].get(i) == v} for k in direct}}
    per_ref = {n: evaluate(docs, p) for n, p in ref.items()}
    designs = {"all five signals": dict.fromkeys(SIGNALS, 1), "votes only": {"votes": 1}, "votes counted twice": {**dict.fromkeys(SIGNALS, 1), "votes": 2}}
    for s in SIGNALS:
        designs[f"leave out {s}"] = {k: 1 for k in SIGNALS if k != s}
    for s in SIGNALS[1:]:
        designs[f"votes + {s}"] = {"votes": 1, s: 1}
    n = len(docs)
    folds = np.array_split(np.random.default_rng(0).permutation(n), 5)
    base = np.array(per_ref["direct"][0])
    results = {}
    for name, w in designs.items():
        curve = {}
        for tau in TAUS:
            per, fp, acc = evaluate(docs, decide(cands, w, tau))
            curve[tau] = (np.array(per), fp, acc)
        means = {t: float(c[0].mean()) for t, c in curve.items()}
        best_tau = max(means, key=means.get)
        plateau = [t for t in TAUS if means[t] >= means[best_tau] - .005]
        cv, gains, taus = [], [], []
        for f in folds:
            train = np.setdiff1d(np.arange(n), f)
            t = max(TAUS, key=lambda x: curve[x][0][train].mean())
            taus.append(t)
            cv.append(float(curve[t][0][f].mean()))
            gains.append(float(curve[t][0][f].mean() - base[f].mean()))
        per, fp, acc = curve[best_tau]
        diff, ci = E.ci(per, base), None
        results[name] = {"weights": w, "best_tau": best_tau, "dev_macro_f1": round(means[best_tau], 4), "tau_plateau": [min(plateau), max(plateau)],
                         "plateau_points": len(plateau), "cv_macro_f1": round(float(np.mean(cv)), 4), "cv_fold_sd": round(float(np.std(cv)), 4),
                         "cv_gain_over_direct_by_fold": [round(x, 4) for x in gains], "folds_beating_direct": int(sum(x > 0 for x in gains)),
                         "tau_chosen_per_fold": taus, "false_positives": fp, "accuracy_on_documented_items": round(acc, 4),
                         "minus_direct_in_sample": {"difference": round(float(per.mean() - base.mean()), 4), "ci": diff},
                         "curve": {str(t): round(m, 4) for t, m in means.items()}}
    eligible = {k: v for k, v in results.items() if v["folds_beating_direct"] >= 4 and v["plateau_points"] >= 3}
    pick = max(eligible, key=lambda k: eligible[k]["cv_macro_f1"]) if eligible else None
    out = {"split": "development", "notes": n, "api_spent_usd": spent_usd(), "references": {
        k: {"macro_f1": round(float(np.mean(v[0])), 4), "false_positives": v[1], "accuracy_on_documented_items": round(v[2], 4)} for k, v in per_ref.items()},
           "candidates_per_note": round(float(np.mean([len(r) for r in cands.values()])), 1), "designs": results,
           "selected_by_rule": pick, "eligible": sorted(eligible)}
    OUT.write_text(json.dumps(out, indent=1))
    print("references", out["references"], "| api spent", out["api_spent_usd"], "| candidates/note", out["candidates_per_note"])
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["cv_macro_f1"]):
        print(f"{k:24s} dev {v['dev_macro_f1']:.4f}@{v['best_tau']} plateau {v['tau_plateau']} ({v['plateau_points']}) CV {v['cv_macro_f1']:.4f}±{v['cv_fold_sd']:.3f} "
              f"folds>direct {v['folds_beating_direct']}/5 FP {v['false_positives']} docacc {v['accuracy_on_documented_items']:.3f} vs direct {v['minus_direct_in_sample']['difference']} {v['minus_direct_in_sample']['ci']}")
    print("selected by rule:", pick)


if __name__ == "__main__":
    main()
