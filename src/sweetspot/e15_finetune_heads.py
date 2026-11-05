"""E15a2: fine-tuned status heads, one shared encoder with three heads against three separately fine-tuned encoders (NegEx/ConText).

E15a showed frozen-encoder heads fail on negated rewrites (0.25 against 0.75 for the LLM status arm), so the encoder is fine-tuned.
Rules set before the run: Clinical ModernBERT; the same 2,016 training and 360 held-out mentions and the same rewrites as E15a;
2 epochs, AdamW, learning rate 3e-5, batch 16, max length 96, seed 0, no tuning. Shared: one encoder, three linear heads on the mean
of the last hidden states, loss = sum of the three cross-entropies. Separate: one encoder per label, each with its own head.
Scores as E15a (labelled accuracy of all three labels and per label, label-free consistency on each rewrite kind).
Output: results/sweetspot/e15_finetune_heads.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e2_negex_battery as E2  # noqa: E402
from sweetspot import perturb  # noqa: E402
from sweetspot.e15_status_heads import ENCODER, Encoder  # noqa: E402
from sweetspot.llm import LLM  # noqa: E402

OUT = ROOT / "results/sweetspot/e15_finetune_heads.json"
SAVE = Path("/home/user/cache/e15_ft_shared")  # the shared model, reused by the tiered design (E15b v2)


def train(labels, rows, texts, classes, save=None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ENCODER)
    enc = AutoModel.from_pretrained(ENCODER).to(dev)
    heads = torch.nn.ModuleDict({lab: torch.nn.Linear(enc.config.hidden_size, len(classes[lab])) for lab in labels}).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(heads.parameters()), lr=3e-5)
    y = {lab: torch.tensor([classes[lab].index(r[lab]) for r in rows]).to(dev) for lab in labels}
    rng = np.random.default_rng(0)
    enc.train()
    for _ in range(2):
        order = rng.permutation(len(rows))
        for k in range(0, len(order), 16):
            ix = order[k:k + 16]
            t = tok([texts[i] for i in ix], padding=True, truncation=True, max_length=96, return_tensors="pt").to(dev)
            h = enc(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            ixt = torch.tensor(ix).to(dev)
            loss = sum(torch.nn.functional.cross_entropy(heads[lab](v), y[lab][ixt]) for lab in labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
    enc.eval()
    if save:
        save.mkdir(parents=True, exist_ok=True)
        enc.save_pretrained(save)
        tok.save_pretrained(save)
        torch.save({"heads": heads.state_dict(), "classes": classes, "labels": labels}, save / "heads.pt")

    def predict(texts_):
        out = {lab: [] for lab in labels}
        with torch.no_grad():
            for k in range(0, len(texts_), 32):
                t = tok(texts_[k:k + 32], padding=True, truncation=True, max_length=96, return_tensors="pt").to(dev)
                h = enc(**t).last_hidden_state
                m = t["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
                for lab in labels:
                    out[lab] += [classes[lab][int(j)] for j in heads[lab](v).argmax(1)]
        return out
    return predict


def score(preds, variants, gold):
    expect = lambda kind, g: {**g, "negation": {"Affirmed": "Negated", "Negated": "Affirmed"}[g["negation"]]} if kind == "negate" else g
    res = {}
    orig = preds["original"]
    for kind, p in preds.items():
        ids = [i["id"] for i in variants[kind]]
        row = {"n": len(ids), "labelled_accuracy": round(float(np.mean([all(p[i][k] == expect(kind, gold[i])[k] for k in E2.LABELS) for i in ids])), 4),
               "per_label": {k: round(float(np.mean([p[i][k] == expect(kind, gold[i])[k] for i in ids])), 4) for k in E2.LABELS}}
        if kind != "original":
            def ok(i):
                o, q = orig[i], p[i]
                if kind == "negate":
                    return o["negation"] != q["negation"] and all(o[k] == q[k] for k in ("temporality", "experiencer"))
                return all(o[k] == q[k] for k in E2.LABELS)
            row["label_free_consistency"] = round(float(np.mean([ok(i) for i in ids])), 4)
        res[kind] = row
    return res


def main():
    import modal_open_bench as M
    rows = M.negex_rows()
    items = E2.sample(rows)
    held = {r["id"] for r in items}
    train_rows = [r for r in rows if r["id"] not in held]
    classes = {lab: sorted({r[lab] for r in rows}) for lab in E2.LABELS}
    gen, nli = LLM("e2_rewrites"), perturb.NLI()
    variants = {"original": items}
    for kind in perturb.KINDS:
        replies = gen.run(perturb.requests_for(items, kind))
        texts = [r["parsed"]["text"].strip() if r else None for r in replies]
        cand = [(it, t) for it, t in zip(items, texts) if t and perturb.contains_concept(t, it["concept"]) and t != it["sentence"]]
        ok = perturb.accept(kind, nli, [c[0]["sentence"] for c in cand], [c[1] for c in cand]) if cand else []
        variants[kind] = [{**it, "sentence": t} for (it, t), good in zip(cand, ok) if good]
    mark = Encoder.mark
    tr_texts = [mark(r["sentence"], r["concept"]) for r in train_rows]
    gold = {r["id"]: {k: r[k] for k in E2.LABELS} for r in items}
    out = {}
    for name, groups in (("shared encoder, 3 heads", [list(E2.LABELS)]), ("separate encoders", [[lab] for lab in E2.LABELS])):
        preds = {kind: {i["id"]: {} for i in its} for kind, its in variants.items()}
        for labels in groups:
            pr = train(labels, train_rows, tr_texts, classes, SAVE if len(labels) == 3 else None)
            for kind, its in variants.items():
                got = pr([mark(i["sentence"], i["concept"]) for i in its])
                for lab in labels:
                    for i, v in zip(its, got[lab]):
                        preds[kind][i["id"]][lab] = v
        out[name] = score(preds, variants, gold)
        print(name, json.dumps({k: v["labelled_accuracy"] for k, v in out[name].items()}), flush=True)
    OUT.write_text(json.dumps({"encoder": ENCODER, "train_mentions": len(train_rows), "arms": out}, indent=1))


if __name__ == "__main__":
    main()
