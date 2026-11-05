"""E14 (MIMIC): the graph fixes, sub-graphs and their consensus on MIMIC-IV MDS 2.0 (Sections I, O, P) with Qwen3.8-27B, scored only
with label-free reliability tests (the nurse labels are not in yet). Rules set before any run.

Admissions: the 100 pilot admissions (first 100 note keys, as in Phase 2). Records from the hospital tables as in Phase 2.
Fix for the predefined rules: the Phase 2 graph decides an active edge with rules written for MDS 2.0 (look-back days per item group,
  a chronic condition counts only for Section I). Here one model call reads each item's wording and the form's stated look-back
  periods and returns, per item, the note timings that count as evidence and the look-back in days (0 = the whole stay). Nothing is
  written per item or per dataset; the old rules are kept only as an ablation (suffix -old).
Sub-graphs per admission (one connected component split by source):
  N    note sub-graph: model mentions per chunk, verifier, model membership (concept -> items), model-decided item scope
  R    records sub-graph: dated record edges (diagnosis and procedure codes, prescriptions, eMAR, ICU events) within the item's look-back
  F    form-guided note sub-graph: per chunk, mentions with the item codes they are evidence for (backfill B: items N leaves empty)
  G = N+R, N+B, G+B; N-old and G-old with the Phase 2 rules; D the direct whole-form prompt (cites sentences)
Consensus: vote(N+B, R, D) (an item is set when at least two sub-graphs set it), agree(D, G+B) (both), tie-break(D|G+B, NLI): D's
  items that G+B also sets, plus D's other items whose cited sentence the NLI model finds entails "The patient has <item>".
Label-free tests (20 admissions, seed 31; up to 2 items each arm sets from note evidence, and 2 Section I items no arm sets):
  mask         remove the arm's cited sentences for the item          expected off   (comprehensiveness)
  sufficiency  keep only those sentences                              expected on    (sufficiency)
  negate       rewrite each cited sentence to deny the item (NLI: contradiction)       expected off
  paraphrase   reword each cited sentence (NLI: two-way entailment)   expected on; also scored for every arm as invariance
  reorder      sections in reverse order                               every arm's items unchanged (stability)
  family       a sentence saying the patient's mother has the item     expected off (attribution)
  ruled_out    a sentence saying the item was considered and ruled out expected off
  Arms with citations are tested on their own cases; every arm (records included) is rerun on every perturbed note. 95% CI by
  resampling admissions. Also: items per admission, agreement between sub-graphs, generated tokens and wall time.
Cache and private intermediate files stay under /home/user/mimic; the JSON output holds counts only.
"""
import json
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import admission_graph as A  # noqa: E402
from sweetspot import mimic_direct as MD  # noqa: E402
from sweetspot import perturb  # noqa: E402
from sweetspot.llm import LLM  # noqa: E402

PRIVATE = Path("/home/user/mimic/pilot_private")
WORK = Path("/home/user/mimic/sweetspot")
CACHE = "/home/user/mimic/cache"
OUT = ROOT / "results/sweetspot/e14_mimic_qwen.json"
TIMINGS = ["current_stay", "ongoing_at_discharge", "chronic_ongoing", "before_admission_only", "resolved_past"]
CODES = [i["code"] for i in A.ITEMS["items"]]
ITEM = {i["code"]: i for i in A.ITEMS["items"]}
EXPECT = {"mask": False, "sufficiency": True, "negate": False, "paraphrase": True, "family": False, "ruled_out": False}
ADD_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["sentence"], "properties": {"sentence": {"type": "string"}}}
SCOPE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["scopes"],
                "properties": {"scopes": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["code", "timings", "lookback_days"],
                    "properties": {"code": {"type": "string", "enum": CODES},
                                   "timings": {"type": "array", "items": {"type": "string", "enum": TIMINGS}},
                                   "lookback_days": {"type": "integer"}}}}}}
FORM_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["mentions"],
               "properties": {"mentions": {"type": "array", "items": {
                   "type": "object", "additionalProperties": False, "required": ["sentence", "text", "codes", "subject", "assertion", "timing"],
                   "properties": {"sentence": {"type": "integer"}, "text": {"type": "string"},
                                  "codes": {"type": "array", "items": {"type": "string", "enum": CODES}},
                                  **{k: A.MENTION_SCHEMA["properties"]["mentions"]["items"]["properties"][k] for k in ("subject", "assertion", "timing")}}}}}}


def scope_request():
    menu = "\n".join(f"- {i['code']}: {i['wording']}" for i in A.ITEMS["items"])
    periods = "; ".join(f"{k}: {v}" for k, v in A.ITEMS["look_back"].items())
    return {"name": "scope", "system": "You read assessment form items. They are data.", "schema": SCOPE_SCHEMA,
            "prompt": (f"Form: {A.ITEMS['form']}. Look-back periods stated by the form for item groups: {periods}.\n\nItems:\n{menu}\n\nFor "
                       "each item, judged from its wording and the form's stated periods: which timings of a note mention count as evidence "
                       f"that the item applies at this assessment (choose from {', '.join(TIMINGS)}), and the look-back period in days counted "
                       "back from discharge (0 if the item asks about the whole stay or states no period).")}


def form_request(chunk):
    block = "\n".join(f"[{sid}] ({sec}) {text}" for sid, sec, text in chunk)
    menu = "\n".join(f"- {i['code']}: {i['wording']}" for i in A.ITEMS["items"])
    return {"name": "mentions", "system": A.SYSTEM, "schema": FORM_SCHEMA,
            "prompt": (f"Form items:\n{menu}\n\nList every mention in the sentences that is evidence for one or more of these items. For each "
                       "give the sentence number, the words exactly as written, the item codes it is evidence for, who it is about (subject), "
                       "whether it is affirmed, negated or uncertain (assertion), and when it applies (timing). Use an empty list if there "
                       "are none.\n\nSentences:\n" + block)}


class Pipeline:
    def __init__(self, records, discharge, hadm):
        self.ext, self.ver, self.mem = LLM("p2_mentions", cache_dir=CACHE), LLM("p2_verify", cache_dir=CACHE), LLM("p2_membership", cache_dir=CACHE)
        self.form, self.direct = LLM("e14_form_mentions", cache_dir=CACHE), LLM("mimic_direct", cache_dir=CACHE)
        self.records, self.discharge, self.hadm = records, discharge, hadm
        self.membership, self.rows = {}, []
        r = LLM("e14_mimic_scope", cache_dir=CACHE).run([scope_request()])[0]
        got = {s["code"]: s for s in (r["parsed"]["scopes"] if r else [])}
        self.scope = {c: {"timings": set(got[c]["timings"]) if c in got else set(TIMINGS[:3]),
                          "days": max(0, int(got[c]["lookback_days"])) if c in got else 0} for c in CODES}

    def concepts(self, pairs):
        new = sorted(set(pairs) - set(self.membership))
        if not new:
            return
        mreqs = A.member_requests(new)
        reps = self.mem.run([q for q, _, _ in mreqs])
        self.rows += reps
        for c in new:
            self.membership.setdefault(c, [])
        for (_, t, batch), r in zip(mreqs, reps):
            for a in (r["parsed"]["answers"] if r else []):
                if 0 <= a["concept"] < len(batch):
                    self.membership[(t, batch[a["concept"]])] = [c for c in a["items"] if c in ITEM]

    def active(self, m, code):
        return m["subject"] == "patient" and m["assertion"] == "affirmed" and m["timing"] in self.scope[code]["timings"]

    def build(self, kept, form, recs, dis, notes=True, records=True, backfill=False):
        filled = defaultdict(list)
        if notes:
            for m in kept:
                for code in self.membership.get((m["type"], A.norm(m["text"])), []):
                    if self.active(m, code):
                        filled[code].append({"source": "note", "sentence": m["sentence"]})
        if records:
            for r in recs:
                for code in self.membership.get((r["type"], A.norm(r["text"])), []):
                    days = self.scope[code]["days"]
                    if days and r.get("time") is not None and dis is not None and (dis - r["time"]).total_seconds() > days * 86400:
                        continue
                    filled[code].append({"source": r["source"]})
        if backfill:
            for m in form:
                for code in m["codes"]:
                    if code not in filled or all(p["source"] != "note" for p in filled[code]):
                        if self.active(m, code):
                            filled[code].append({"source": "note", "sentence": m["sentence"]})
        return dict(filled)

    def run(self, cases):
        """cases: list of (key, chunks). Returns, per case, {arm: {code: [cited sentence ids]}} (records-only paths cite nothing)."""
        flat = [(n, c) for n, (_, chs) in enumerate(cases) for c in chs]
        er = self.ext.run([A.mention_request(c) for _, c in flat])
        fr = self.form.run([form_request(c) for _, c in flat])
        raw = []
        for (_, c), r in zip(flat, er):
            ok = {sid for sid, _, _ in c}
            raw.append([m for m in (r["parsed"]["mentions"] if r else []) if m["sentence"] in ok and m["text"].strip()])
        vq = [(i, A.verify_request(flat[i][1], ms)) for i, ms in enumerate(raw) if ms]
        vr = self.ver.run([q for _, q in vq])
        kept = list(raw)
        for (i, _), r in zip(vq, vr):
            kept[i] = A.apply_verdicts(raw[i], r["parsed"] if r else None)
        dq = []
        for key, chs in cases:
            rows = [x for c in chs for x in c]
            dq.append(MD.request(rows, CODES))
        dr = self.direct.run(dq)
        self.rows += list(er) + list(fr) + list(vr) + list(dr)
        per_raw, per_kept, per_form = defaultdict(list), defaultdict(list), defaultdict(list)
        for i, (n, c) in enumerate(flat):
            ok = {sid for sid, _, _ in c}
            per_raw[n] += raw[i]
            per_kept[n] += kept[i]
            if fr[i]:
                per_form[n] += [m for m in fr[i]["parsed"]["mentions"] if m["sentence"] in ok and m["text"].strip()]
        recs_all = []
        for key, _ in cases:
            recs_all += self.records.get(self.hadm[key], [])
        self.concepts([(m["type"], A.norm(m["text"])) for ms in list(per_kept.values()) + list(per_raw.values()) for m in ms] +
                      [(r["type"], A.norm(r["text"])) for r in recs_all])
        out = []
        for n, ((key, chs), r) in enumerate(zip(cases, dr)):
            h = self.hadm[key]
            recs = self.records.get(h, [])
            dis = self.discharge.get(h)
            dis = dis.to_pydatetime() if hasattr(dis, "to_pydatetime") else dis
            valid = {sid for c in chs for sid, _, _ in c}
            g = {"D": {x["code"]: sorted({int(s) for s in x["evidence"] if int(s) in valid}) for x in (r["parsed"]["items"] if r else []) if x["code"] in ITEM}}
            paths = {"N": self.build(per_kept[n], [], recs, dis, records=False), "R": self.build([], [], recs, dis, notes=False),
                     "G": self.build(per_kept[n], [], recs, dis), "N+B": self.build(per_kept[n], per_form[n], recs, dis, records=False, backfill=True),
                     "G+B": self.build(per_kept[n], per_form[n], recs, dis, backfill=True),
                     "N-verify": self.build(per_raw[n], [], recs, dis, records=False),
                     "N-old": A.build_graph(h, per_kept[n], [], self.membership, dis),
                     "G-old": A.build_graph(h, per_kept[n], recs, self.membership, dis)}
            for arm, p in paths.items():
                g[arm] = {c: sorted({x["sentence"] for x in ps if x.get("source") == "note"}) for c, ps in p.items()}
            out.append(g)
        return out


def consensus(g, nli, text_of):
    """Adds the consensus arms to one case's arm dict."""
    votes = defaultdict(list)
    for arm in ("N+B", "R", "D"):
        for c, cited in g[arm].items():
            votes[c].append(cited)
    g["vote(N+B,R,D)"] = {c: sorted({s for x in v for s in x}) for c, v in votes.items() if len(v) >= 2}
    g["agree(D,G+B)"] = {c: sorted(set(v) | set(g["G+B"][c])) for c, v in g["D"].items() if c in g["G+B"]}
    tb = dict(g["agree(D,G+B)"])
    pairs, where = [], []
    for c, cited in g["D"].items():
        if c not in tb:
            for s in cited:
                if s in text_of:
                    pairs.append((text_of[s], f"The patient has {ITEM[c]['label']}."))
                    where.append(c)
    for c, lab in zip(where, nli.labels(pairs) if pairs else []):
        if lab == "entailment":
            tb[c] = g["D"][c]
    g["tie-break(D|G+B,NLI)"] = tb
    return g


def main():
    t0 = time.time()
    sents = pd.read_parquet(PRIVATE / "proxy_sentences.parquet")
    ids = pd.read_parquet(PRIVATE / "proxy_id_map.parquet").astype(str)
    hadm = dict(zip(ids["note_key"], ids["hadm_id"]))
    keys = sorted(sents["note_key"].unique())[:100]
    records, discharge = pickle.loads((WORK / "records_proxy.pkl").read_bytes())
    P = Pipeline(records, discharge, hadm)
    notes = {}
    for k in keys:
        rows = sents[sents["note_key"] == k].sort_values("sent_id")
        triples = list(zip(rows["sent_id"].astype(int), rows["section"], rows["text"]))
        notes[k] = (triples, {sid: (sec, t) for sid, sec, t in triples}, list(A.chunks(triples)))
    nli = perturb.NLI()
    base = P.run([(k, notes[k][2]) for k in keys])
    base = [consensus(g, nli, {sid: t for sid, (_, t) in notes[k][1].items()}) for k, g in zip(keys, base)]
    arms = list(base[0])
    t_base = time.time() - t0
    summary = {a: {"items_per_admission": round(float(np.mean([len(g[a]) for g in base])), 2),
                   "by_section": {s: round(float(np.mean([sum(1 for c in g[a] if ITEM[c]["section"] == s) for g in base])), 2) for s in "IOP"}}
               for a in arms}
    pair = {}
    for a, b in (("N", "R"), ("N+B", "R"), ("D", "N+B"), ("D", "R"), ("D", "G+B"), ("N", "N-old"), ("G", "G-old")):
        both = sum(len(set(g[a]) & set(g[b])) for g in base)
        either = sum(len(set(g[a]) | set(g[b])) for g in base)
        pair[f"{a}~{b}"] = {"jaccard": round(both / either, 3) if either else None, "both": both, "either": either}
    # label-free battery
    rng = random.Random(31)
    sample = sorted(rng.sample(range(len(keys)), 20))
    own_arms = ["D", "N-old", "N+B", "vote(N+B,R,D)", "tie-break(D|G+B,NLI)"]
    gen = LLM("e14_mimic_rewrites", cache_dir=CACHE)
    specs, rew_q, rew_m, add_q, add_m = [], [], [], [], []
    for n in sample:
        k = keys[n]
        g = base[n]
        for arm in own_arms:
            yes = sorted(c for c, cited in g[arm].items() if cited)
            for c in rng.sample(yes, min(2, len(yes))):
                cited = g[arm][c]
                specs.append((n, arm, c, cited))
                for kind in ("negate", "paraphrase"):
                    its = [{"id": s, "sentence": notes[k][1][s][1], "concept": ITEM[c]["label"]} for s in cited]
                    for it, q in zip(its, perturb.requests_for(its, kind)):
                        rew_q.append(q)
                        rew_m.append((n, arm, c, kind, it))
        unset = sorted(c for c in ITEM if ITEM[c]["section"] == "I" and all(c not in g[a] for a in arms))
        for c in rng.sample(unset, min(2, len(unset))):
            lab = ITEM[c]["label"]
            for kind, ask in (("family", f"Write one sentence for a discharge summary stating that the patient's mother has a history of {lab}."),
                              ("ruled_out", f"Write one sentence for a discharge summary stating that {lab} was considered in this patient and ruled out.")):
                add_q.append({"name": "add", "system": "You write single sentences for a robustness test of clinical NLP.", "prompt": ask, "schema": ADD_SCHEMA})
                add_m.append((n, c, kind))
    rew, adds = gen.run(rew_q), gen.run(add_q)
    cand = defaultdict(list)
    for m, r in zip(rew_m, rew):
        if r and r["parsed"]["text"].strip() and r["parsed"]["text"].strip() != m[4]["sentence"]:
            cand[m[3]].append((m, r["parsed"]["text"].strip()))
    new_text = defaultdict(dict)
    accepted = {}
    for kind, pairs in cand.items():
        ok = perturb.accept(kind, nli, [m[4]["sentence"] for m, _ in pairs], [t for _, t in pairs])
        accepted[kind] = {"generated": len(pairs), "accepted": int(sum(ok))}
        for (m, t), good in zip(pairs, ok):
            if good:
                new_text[(m[0], m[1], m[2], kind)][m[4]["id"]] = t
    cases, meta = [], []
    for n, arm, c, cited in specs:
        k = keys[n]
        triples, text_of, ch = notes[k]
        cases.append((k, [x for x in ([y for y in cc if y[0] not in cited] for cc in ch) if x]))
        meta.append((n, "mask", arm, c))
        cases.append((k, [[(s, *text_of[s]) for s in cited]]))
        meta.append((n, "sufficiency", arm, c))
        for kind in ("negate", "paraphrase"):
            nt = new_text.get((n, arm, c, kind), {})
            if len(nt) == len(cited):
                cases.append((k, [[(s, sec, nt.get(s, t)) for s, sec, t in cc] for cc in ch]))
                meta.append((n, kind, arm, c))
    for n in sample:
        triples = notes[keys[n]][0]
        order = list(dict.fromkeys(sec for _, sec, _ in triples))[::-1]
        cases.append((keys[n], list(A.chunks([x for sec in order for x in triples if x[1] == sec]))))
        meta.append((n, "reorder", None, None))
    for (n, c, kind), r in zip(add_m, adds):
        if r:
            triples, _, ch = notes[keys[n]]
            sec = "Family History" if kind == "family" else "Brief Hospital Course"
            cases.append((keys[n], ch[:-1] + [ch[-1] + [(triples[-1][0] + 1, sec, r["parsed"]["sentence"].strip())]]))
            meta.append((n, kind, "all", c))
    t1 = time.time()
    got = P.run(cases)
    got = [consensus(g, nli, {s: t for cc in cs for s, _, t in cc}) for g, (_, cs) in zip(got, cases)]
    res = defaultdict(lambda: defaultdict(list))  # test -> arm -> [(admission, holds)]
    for (n, kind, arm, c), g in zip(meta, got):
        if kind == "reorder":
            for a in arms:
                o, p = set(base[n][a]), set(g[a])
                if o | p:
                    res["reorder_stability"][a].append((n, len(o & p) / len(o | p)))
        elif arm == "all":
            for a in arms:
                res[kind][a].append((n, (c in g[a]) == EXPECT[kind]))
        else:
            res[kind][arm].append((n, (c in g[arm]) == EXPECT[kind]))
            if kind == "paraphrase":
                for a in arms:
                    res["paraphrase_invariance"][a].append((n, (c in g[a]) == (c in base[n][a])))
    boot = np.random.default_rng(0)
    battery = {}
    for test, by_arm in res.items():
        battery[test] = {}
        for a, rows in by_arm.items():
            adm = sorted({x for x, _ in rows})
            per = defaultdict(list)
            for x, h in rows:
                per[x].append(float(h))
            vals = [np.mean([v for x in boot.choice(adm, len(adm)) for v in per[x]]) for _ in range(2000)]
            battery[test][a] = {"cases": len(rows), "holds": round(float(np.mean([float(h) for _, h in rows])), 3),
                                "ci": [round(float(np.quantile(vals, .025)), 3), round(float(np.quantile(vals, .975)), 3)]}
    rows = [r for r in P.rows if r]
    out = {"model": P.ext.model, "admissions": len(keys), "battery_admissions": len(sample), "cases": len(cases),
           "scope_from_wording": {"lookback_days": {g: sorted({P.scope[c]["days"] for c in CODES if c.startswith(g)}) for g in ("I1", "I2", "I3", "O4", "P1a", "P1b")},
                                  "chronic_counts": {s: sum("chronic_ongoing" in P.scope[c]["timings"] for c in CODES if ITEM[c]["section"] == s) for s in "IOP"}},
           "arms": summary, "pairwise_agreement": pair, "battery": battery, "rewrites": accepted,
           "generated_tokens": sum(r["output_tokens"] for r in rows), "prompt_tokens": sum(r["input_tokens"] for r in rows),
           "wall_seconds": {"base_100": round(t_base), "battery": round(time.time() - t1)},
           "failures": {n: getattr(x, "last_failures", {}) for n, x in (("extract", P.ext), ("form", P.form), ("verify", P.ver), ("direct", P.direct), ("membership", P.mem))}}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "e14_mimic_base.pkl").write_bytes(pickle.dumps({"keys": keys, "base": base}))
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "pairwise_agreement"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
