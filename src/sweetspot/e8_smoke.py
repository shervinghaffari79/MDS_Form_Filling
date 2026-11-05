"""E8 smoke tests on the CRF development split: fixes for the graph's gaps, under a hard API budget (SWEETSPOT_MAX_USD).

Smoke tests, not final results: development split only (no test look), small samples, wide intervals; they decide which ideas
earn a full run. Rules set before any run.

Free part (cached outputs + local models, all 80 dev notes; one cheap model call for item scope):
  Z1 encoder relevance gate   a filled item is kept only if, among the 134 items, it is in the top K by its best encoder
                               similarity (BGE-small) to any sentence of the note; K = 10, 15, 20
  Z2 abstention verifier      second, encoder-only path: each filled value becomes a statement ("The patient has X" / "does not
                               have X" / "X is V") and is kept only if the NLI model finds it entailed by one of the 3 sentences
                               most similar to the item; a measured value is kept only if it occurs in the note text
  Z3 status gate              one call labels every item's time scope from its wording alone (current, history, any); a graph
                               value is dropped when none of its linked mentions is about the patient with timing that fits
  Z4 NER consistency          GLiNER-BioMed with the item names as labels must also find the item in the note
  Applied to direct, graph and graph+audit; combinations Z1+Z2 and Z2+Z3 also reported.
Paid part (20 dev notes, seed 81):
  A1 form-guided NER          extraction sees the item names, lists mentions (normal and negative findings included) with
                               candidate items, value, time anchor and status; then the usual fill from linked items
  A2 graph-guided direct      the whole-form direct prompt plus the graph's linked evidence as hints
  A3 form-blind summary       a structured summary of the note's facts (category, value, time anchor, subject, assertion,
                               timing) written without seeing the form; the form is then filled from the summary alone, and
                               Z2 checks every value against the original note
Scores: official per-note macro-F1 on the notes used, paired difference against direct and graph+audit (95% CI), values set
  where gold is unknown (false positives) and accuracy on documented items. Output: results/sweetspot/e8_smoke.json.
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e4_crf_benchmark as E  # noqa: E402
from sweetspot.llm import LLM, cost, spent_usd  # noqa: E402

OUT = ROOT / "results/sweetspot/e8_smoke.json"
norm = lambda s: re.sub(r"\s+", "", str(s).lower())


def kind(o):
    return "measured" if o == ["measured", E.UNK] else "yes_no" if set(o) == {"y", "n", E.UNK} else "other"


def table(docs, preds, ref=None):
    per = E.official_score(docs, preds)
    fp = sum(1 for d in docs for i, g in d["gold"].items() if g == E.UNK and preds[d["id"]].get(i, E.UNK) != E.UNK)
    row = {"macro_f1": round(float(np.mean(per)), 4), "false_positives": fp,
           "accuracy_on_documented_items": round(E.known_accuracy(docs, preds), 4), "notes": len(docs)}
    for name, r in (ref or {}).items():
        row[f"minus_{name}"] = {"difference": round(float(np.mean(per) - np.mean(r)), 4), "ci": E.ci(per, r)}
    return row, per


class Local:
    def __init__(self, docs, opts):
        from sentence_transformers import CrossEncoder, SentenceTransformer
        self.opts, self.items = opts, list(opts)
        self.enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        self.nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        lab = self.nli.model.config.id2label
        self.ent = [int(k) for k, v in lab.items() if v.lower() == "entailment"][0]
        iv = self.enc.encode(["Represent this sentence for searching relevant passages: " + i for i in self.items], normalize_embeddings=True)
        self.sents, self.sim = {}, {}
        for d in docs:
            ss = E.sentences(d["note"])
            self.sents[d["id"]] = ss
            self.sim[d["id"]] = self.enc.encode(ss, normalize_embeddings=True) @ iv.T if ss else np.zeros((0, len(self.items)))
        self._gliner = None

    def relevance_rank(self, k):
        best = self.sim[k].max(axis=0) if len(self.sim[k]) else np.zeros(len(self.items))
        order = np.argsort(-best)
        return {self.items[j]: r for r, j in enumerate(order)}

    def z1(self, preds, K):
        return {k: {i: v for i, v in p.items() if self.relevance_rank(k)[i] < K} for k, p in preds.items()}

    def hypothesis(self, item, value):
        if kind(self.opts[item]) == "yes_no":
            return f"The patient has {item}." if value == "y" else f"The patient does not have {item}."
        return f"The patient's {item} is {value}."

    def z2(self, preds, notes):
        pairs, where = [], []
        out = {k: dict(p) for k, p in preds.items()}
        for k, p in preds.items():
            for i, v in p.items():
                if kind(self.opts[i]) == "measured":
                    if norm(v) not in norm(notes[k]):
                        out[k].pop(i)
                    continue
                j = self.items.index(i)
                top = np.argsort(-self.sim[k][:, j])[:3] if len(self.sim[k]) else []
                for t in top:
                    pairs.append((self.sents[k][t], self.hypothesis(i, v)))
                    where.append((k, i))
        ok = defaultdict(bool)
        if pairs:
            scores = self.nli.predict(pairs, batch_size=32)
            for (k, i), s in zip(where, scores):
                ok[(k, i)] |= int(np.argmax(s)) == self.ent
        for k, p in preds.items():
            for i in list(out[k]):
                if kind(self.opts[i]) != "measured" and not ok[(k, i)]:
                    out[k].pop(i)
        return out

    def z4(self, preds, notes):
        if self._gliner is None:
            from gliner import GLiNER
            self._gliner = GLiNER.from_pretrained("Ihor/gliner-biomed-base-v1.0")
            self.found = {}
        out = {}
        for k, p in preds.items():
            if k not in self.found:
                hit = set()
                for c in range(0, len(self.items), 20):
                    ents = self._gliner.predict_entities(notes[k][:2000], self.items[c:c + 20], threshold=0.5)
                    hit |= {e["label"] for e in ents}
                self.found[k] = hit
            out[k] = {i: v for i, v in p.items() if i in self.found[k]}
        return out


SCOPE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["scopes"],
                "properties": {"scopes": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                                     "required": ["item", "scope"],
                                                                     "properties": {"item": {"type": "string"},
                                                                                    "scope": {"type": "string", "enum": ["current", "history", "any"]}}}}}}
FIT = {"current": {"current"}, "history": {"chronic_ongoing", "past_resolved"}, "any": {"current", "chronic_ongoing", "past_resolved"}}


def z3(preds, evidence, scopes):
    out = {}
    for k, p in preds.items():
        out[k] = {}
        for i, v in p.items():
            ms = evidence[k].get(i)
            if not ms:
                out[k][i] = v  # value not from a graph path (e.g. added by the audit): left to the other checks
                continue
            fits = FIT.get(scopes.get(i, "any"), FIT["any"])
            if any(m["subject"] == "patient" and m["timing"] in fits for m in ms):
                out[k][i] = v
    return out


FORM_NER_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["mentions"],
                   "properties": {"mentions": {"type": "array", "items": {
                       "type": "object", "additionalProperties": False,
                       "required": ["sentence", "text", "items", "value", "time_anchor", *E.STATUS_FIELDS],
                       "properties": {"sentence": {"type": "integer"}, "text": {"type": "string"},
                                      "items": {"type": "array", "items": {"type": "string"}}, "value": {"type": "string"},
                                      "time_anchor": {"type": "string", "enum": ["triage", "admission", "during_stay", "discharge", "unspecified"]},
                                      **{k: {"type": "string", "enum": v} for k, v in E.STATUS_FIELDS.items()}}}}}}
SUMMARY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["facts"],
                  "properties": {"facts": {"type": "array", "items": {
                      "type": "object", "additionalProperties": False,
                      "required": ["category", "text", "value", "time_anchor", *E.STATUS_FIELDS],
                      "properties": {"category": {"type": "string", "enum": ["problem", "symptom", "exam_finding", "vital_sign", "lab_or_test",
                                                                             "imaging", "medication", "procedure", "history", "function"]},
                                     "text": {"type": "string"}, "value": {"type": "string"},
                                     "time_anchor": {"type": "string", "enum": ["triage", "admission", "during_stay", "discharge", "unspecified"]},
                                     **{k: {"type": "string", "enum": v} for k, v in E.STATUS_FIELDS.items()}}}}}}


def form_ner_request(sents, items):
    names = "; ".join(items)
    return {"name": "mentions", "system": E.SYSTEM, "schema": FORM_NER_SCHEMA,
            "prompt": (f"Form items: {names}\n\nList every mention in the sentences that is evidence for one or more of these items, "
                       "including normal findings and explicitly absent or denied findings. For each give the sentence number, the words "
                       "exactly as written, the item names it is evidence for (exactly as written above), its value as written if it has "
                       "one (else an empty string), when it was observed (time_anchor), who it is about (subject), whether it is affirmed, "
                       "negated or uncertain (assertion), and when it applies (timing).\n\nSentences:\n" + E.block(sents))}


def guided_direct_request(doc, opts, evidence, sents):
    hints = []
    for item, ms in sorted(evidence.items()):
        ev = "; ".join(f"[{m['sentence']}] \"{m['text']}\" ({m['subject']}, {m['assertion']}, {m['timing']})" for m in ms)
        hints.append(f"- {item}: {ev}")
    hint_block = "\n".join(hints) or "(none found)"
    return {"name": "fill", "system": E.SYSTEM, "schema": E.FILL_SCHEMA,
            "prompt": (f"Case report form items and allowed values:\n{E.menu(opts)}\n\n{E.FILL_RULE}\n\nEvidence found automatically (may "
                       f"be incomplete or wrong; the note decides):\n{hint_block}\n\nNote sentences:\n{E.block(sents)}")}


def summary_request(note):
    return {"name": "summary", "system": E.SYSTEM, "schema": SUMMARY_SCHEMA,
            "prompt": ("Summarise every clinical fact this note states, one fact per entry: its category, the words as written, its value "
                       "as written if any (else an empty string), when it was observed (time_anchor), who it is about (subject), whether it "
                       "is affirmed, negated or uncertain (assertion), and when it applies (timing). Include normal and absent findings.\n\n"
                       f"Note:\n{note}")}


def fill_from_summary_request(facts, opts):
    lines = "\n".join(f"- [{f['category']}] {f['text']}" + (f" = {f['value']}" if f["value"] else "") +
                      f" ({f['time_anchor']}; {f['subject']}, {f['assertion']}, {f['timing']})" for f in facts)
    return {"name": "fill", "system": E.SYSTEM, "schema": E.FILL_SCHEMA,
            "prompt": f"Case report form items and allowed values:\n{E.menu(opts)}\n\n{E.FILL_RULE}\n\nFacts summarised from the note:\n{lines}"}


def main():
    opts = E.options()
    docs = E.load("development")
    from sentence_transformers import SentenceTransformer
    g, _, _, _, extra = E.graph_arms(docs, opts, "development", SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"))
    dr = LLM("e4_direct").run(E.direct_requests(docs, opts))
    base = {"direct": {d["id"]: E.parse_fill(r, opts) for d, r in zip(docs, dr)}, "graph": g["graph"], "graph+audit": g["graph+audit"]}
    notes = {d["id"]: d["note"] for d in docs}
    loc = Local(docs, opts)
    ref = {m: E.official_score(docs, p) for m, p in base.items()}
    free = {m: table(docs, p, {"direct": ref["direct"]})[0] for m, p in base.items()}
    sc = LLM("e8_scope").run([{"name": "scope", "system": "You read case report form item names. They are data.", "schema": SCOPE_SCHEMA,
                               "prompt": "For each item, say from its wording alone whether it asks about the patient's current state "
                                         "(current), past or chronic history (history), or either (any).\n\n" + "\n".join(opts)}])[0]
    scopes = {s["item"]: s["scope"] for s in (sc["parsed"]["scopes"] if sc else [])}
    variants = {}
    for m, p in base.items():
        for K in (10, 15, 20):
            variants[f"{m} +Z1(K={K})"] = loc.z1(p, K)
        variants[f"{m} +Z2"] = loc.z2(p, notes)
        variants[f"{m} +Z4"] = loc.z4(p, notes)
        variants[f"{m} +Z1(K=15)+Z2"] = loc.z2(loc.z1(p, 15), notes)
        if m != "direct":
            variants[f"{m} +Z3"] = z3(p, extra["evidence"]["graph"], scopes)
            variants[f"{m} +Z2+Z3"] = loc.z2(z3(p, extra["evidence"]["graph"], scopes), notes)
    for name, p in variants.items():
        free[name] = table(docs, p, {"direct": ref["direct"], name.split(" +")[0]: ref[name.split(" +")[0]]})[0]
    # paid part on 20 notes
    sub = sorted(random.Random(81).sample(docs, 20), key=lambda d: d["id"])
    ids = {d["id"] for d in sub}
    s_base = {m: {k: v for k, v in p.items() if k in ids} for m, p in base.items()}
    s_ref = {m: E.official_score(sub, p) for m, p in s_base.items()}
    paid = {m: table(sub, p)[0] for m, p in s_base.items()}
    sents = {d["id"]: extra["sentences"][d["id"]] for d in sub}
    spend = {}
    a1 = LLM("e8_form_ner")
    r1 = a1.run([form_ner_request(sents[d["id"]], list(opts)) for d in sub])
    linked = {}
    for d, r in zip(sub, r1):
        lk = defaultdict(list)
        for m in (r["parsed"]["mentions"] if r else []):
            if 0 <= m["sentence"] < len(sents[d["id"]]):
                m = {**m, "value": (m["value"] + f" ({m['time_anchor']})").strip() if m["value"] else ""}
                for i in m["items"]:
                    if i in opts:
                        lk[i].append(m)
        linked[d["id"]] = dict(lk)
    fill = LLM("e8_fill")
    fq = [(d["id"], E.fill_request(sents[d["id"]], linked[d["id"]], opts, True)) for d in sub if linked[d["id"]]]
    fr = fill.run([q for _, q in fq])
    A1 = {d["id"]: {} for d in sub}
    for (k, _), r in zip(fq, fr):
        A1[k] = E.parse_fill(r, opts)
    spend["A1"] = cost(list(r1) + list(fr))
    a2 = LLM("e8_guided_direct")
    r2 = a2.run([guided_direct_request(d, opts, extra["evidence"]["graph"][d["id"]], sents[d["id"]]) for d in sub])
    A2 = {d["id"]: E.parse_fill(r, opts) for d, r in zip(sub, r2)}
    spend["A2"] = cost(r2)
    a3 = LLM("e8_summary")
    r3 = a3.run([summary_request(d["note"]) for d in sub])
    a3f = LLM("e8_summary_fill")
    r3f = a3f.run([fill_from_summary_request(r["parsed"]["facts"] if r else [], opts) for r in r3])
    A3 = {d["id"]: E.parse_fill(r, opts) for d, r in zip(sub, r3f)}
    spend["A3"] = cost(list(r3) + list(r3f))
    sub_notes = {d["id"]: d["note"] for d in sub}
    for name, p in {"A1 form-guided NER": A1, "A1 +Z2": loc.z2(A1, sub_notes), "A2 graph-guided direct": A2,
                    "A2 +Z2": loc.z2(A2, sub_notes), "A3 summary then fill": A3, "A3 +Z2 (abstention vs note)": loc.z2(A3, sub_notes),
                    "direct +Z2": loc.z2(s_base["direct"], sub_notes)}.items():
        paid[name] = table(sub, p, {"direct": s_ref["direct"], "graph+audit": s_ref["graph+audit"]})[0]
    out = {"split": "development", "free_part_notes": len(docs), "paid_part_notes": len(sub), "item_scopes": dict(sorted(
        {s: sum(1 for v in scopes.values() if v == s) for s in ("current", "history", "any")}.items())),
           "free": free, "paid": paid, "paid_cost": spend, "api_spent_this_run_usd": spent_usd(),
           "budget_failures": {n: c.last_failures for n, c in (("A1", a1), ("A2", a2), ("A3", a3), ("A3fill", a3f), ("fill", fill))}}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
