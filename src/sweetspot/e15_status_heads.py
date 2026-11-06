"""E15a: status heads on one shared clinical encoder (assertion, subject, timing), trained on NegEx/ConText, scored like E2.

One encoder (Simonlee711/Clinical_ModernBERT, frozen) reads each mention once: the sentence with the concept marked by brackets;
the vector is the mean of the last hidden states. Three logistic-regression heads read the same vector: negation (Affirmed / Negated),
temporality (Recent / Historical / Hypothetical) and experiencer (Patient / Other). This is supervised (the heads learn from NegEx
labels), unlike the training-free arms before it; the encoder itself is not trained.
Held-out set: E2's 360-mention sample (seed 13, stratified) and its NLI-checked rewrites (negate, paraphrase, reorder, distractor;
  cached from E2), so the heads are compared with E2's arms on identical cases. Heads train on the other 2,016 labelled mentions.
Separate heads versus one shared encoder: with a frozen encoder the heads cannot interfere, so the comparison of a fine-tuned
  shared encoder against separate fine-tuned encoders is not made here (stated, not tested).
Confidence: each head's top probability; a mention is "confident" when all three are at least 0.9 (fixed before the run). The share
  of held-out mentions that are confident, and accuracy on confident and on unsure mentions, are reported (the unsure ones are what
  the tiered design sends to the LLM).
Output: results/sweetspot/e15_status_heads.json; the fitted heads are saved to /home/user/cache/e15_heads.pkl for E15b.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e2_negex_battery as E2  # noqa: E402
from sweetspot import perturb  # noqa: E402
from sweetspot.llm import LLM  # noqa: E402

ENCODER = "Simonlee711/Clinical_ModernBERT"
OUT = ROOT / "results/sweetspot/e15_status_heads.json"
HEADS = Path("/home/user/cache/e15_heads.pkl")
CONF = 0.9


class Encoder:
    def __init__(self, name=ENCODER):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    @staticmethod
    def mark(sentence, concept):
        import re
        m = re.search(r"\s+".join(re.escape(t) for t in concept.split()), sentence, re.I)
        if not m:
            return f"{sentence} [ {concept} ]"
        return sentence[:m.start()] + "[ " + sentence[m.start():m.end()] + " ]" + sentence[m.end():]

    def encode(self, texts, batch=32):
        out = []
        with self.torch.no_grad():
            for k in range(0, len(texts), batch):
                t = self.tok(texts[k:k + batch], padding=True, truncation=True, max_length=192, return_tensors="pt")
                h = self.model(**t).last_hidden_state
                m = t["attention_mask"].unsqueeze(-1).float()
                out.append(((h * m).sum(1) / m.sum(1)).numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.config.hidden_size))


def fit(X, rows):
    from sklearn.linear_model import LogisticRegression
    return {lab: LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(X, [r[lab] for r in rows]) for lab in E2.LABELS}


def predict(heads, X):
    out = []
    probs = {lab: h.predict_proba(X) for lab, h in heads.items()}
    for n in range(len(X)):
        row = {lab: heads[lab].classes_[int(np.argmax(probs[lab][n]))] for lab in heads}
        row["confidence"] = float(min(probs[lab][n].max() for lab in heads))
        out.append(row)
    return out


def main():
    import modal_open_bench as M
    rows = M.negex_rows()
    items = E2.sample(rows)
    held = {r["id"] for r in items}
    train = [r for r in rows if r["id"] not in held]
    enc = Encoder()
    Xtr = enc.encode([enc.mark(r["sentence"], r["concept"]) for r in train])
    heads = fit(Xtr, train)
    gen, nli = LLM("e2_rewrites"), perturb.NLI()  # E2's rewrites, from its cache (gpt-6-luna), and the same NLI acceptance
    variants = {"original": items}
    for kind in perturb.KINDS:
        replies = gen.run(perturb.requests_for(items, kind))
        texts = [r["parsed"]["text"].strip() if r else None for r in replies]
        cand = [(it, t) for it, t in zip(items, texts) if t and perturb.contains_concept(t, it["concept"]) and t != it["sentence"]]
        ok = perturb.accept(kind, nli, [c[0]["sentence"] for c in cand], [c[1] for c in cand]) if cand else []
        variants[kind] = [{**it, "sentence": t} for (it, t), good in zip(cand, ok) if good]
    preds = {kind: dict(zip([i["id"] for i in its], predict(heads, enc.encode([enc.mark(i["sentence"], i["concept"]) for i in its]))))
             for kind, its in variants.items()}
    gold = {r["id"]: {k: r[k] for k in E2.LABELS} for r in items}
    expect = lambda kind, g: {**g, "negation": {"Affirmed": "Negated", "Negated": "Affirmed"}[g["negation"]]} if kind == "negate" else g
    scores = {}
    orig = preds["original"]
    for kind, p in preds.items():
        lab = [all(p[i][k] == expect(kind, gold[i])[k] for k in E2.LABELS) for i in p]
        per = {k: round(float(np.mean([p[i][k] == expect(kind, gold[i])[k] for i in p])), 4) for k in E2.LABELS}
        row = {"n": len(p), "labelled_accuracy": round(float(np.mean(lab)), 4), "per_label": per}
        if kind != "original":
            def consistent(i):
                o, q = orig[i], p[i]
                if kind == "negate":
                    return o["negation"] != q["negation"] and all(o[k] == q[k] for k in ("temporality", "experiencer"))
                return all(o[k] == q[k] for k in E2.LABELS)
            row["label_free_consistency"] = round(float(np.mean([consistent(i) for i in p])), 4)
        scores[kind] = row
    conf = [orig[i]["confidence"] >= CONF for i in orig]
    right = [all(orig[i][k] == gold[i][k] for k in E2.LABELS) for i in orig]
    routing = {"confident_share": round(float(np.mean(conf)), 4),
               "accuracy_confident": round(float(np.mean([r for r, c in zip(right, conf) if c])), 4) if any(conf) else None,
               "accuracy_unsure": round(float(np.mean([r for r, c in zip(right, conf) if not c])), 4) if not all(conf) else None}
    e2 = json.loads((ROOT / "results/sweetspot/e2_negex_battery.json").read_text())
    ref = {arm: {kind: v.get("labelled_accuracy") for kind, v in e2["scores"][arm].items() if isinstance(v, dict)} for arm in e2["scores"]}
    Xall = np.concatenate([Xtr, enc.encode([enc.mark(r["sentence"], r["concept"]) for r in items])])
    HEADS.write_bytes(pickle.dumps(fit(Xall, train + items)))  # final heads for MIMIC: all labelled mentions
    out = {"encoder": ENCODER, "train_mentions": len(train), "held_out": len(items), "heads": scores, "routing_at_0.9": routing,
           "e2_arms_same_cases": ref}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
