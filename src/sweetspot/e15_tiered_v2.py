"""E15b v2: the tiered design after the first smoke test, compared with the first design on the same and on fresh admissions.

Changes, each from E15b's diagnosis (made after seeing the first 15 admissions, so the fresh sample is the fair test):
  heads       fine-tuned shared encoder with three heads (E15a2) instead of frozen heads, which failed on negated rewrites
  records     only eMAR administrations and diagnosis codes count as direct record evidence for a yes draft; procedure codes and
              ICU events (proxy precision 0.17 to 0.28 as tier-1 evidence) only enter the evidence pack
  every yes   every yes draft goes to the blind LLM (tier 1 settles only clear no answers)
  escalation  only a contradiction goes to the nurse (draft yes, blind no; draft no, blind yes); a blind "unknown" stays at tier 2 as
              Unknown with its evidence attached; Unknown from an uncovered window also stays at tier 2 (shown, not queued)
Designs scored (same tagging, class membership and frames; blind calls are cached by prompt, so shared packs cost nothing twice):
  v1               the first design (frozen heads, all given record kinds, tier-1 yes, any disagreement or Unknown to the nurse)
  v1+blind yes     v1 with every yes draft checked blind
  v2               all four changes
  v2 -heads, v2 -records, v2 -escalation   v2 with one change undone
Samples: the first 15 admissions (seed 151) and 15 fresh admissions from the rest of the 100 (seed 153). gpt-5-nano, low effort.
Reference: table proxy labels on the 27 items that have one (oxygen and IV medication have none). Noisy, partly circular.
Output: results/sweetspot/e15_tiered_v2.json (counts only).
"""
import json
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sweetspot import e15_tiered_smoke as V1  # noqa: E402
from sweetspot.e15_status_heads import Encoder, HEADS, predict  # noqa: E402
from sweetspot.llm import LLM, spent_usd  # noqa: E402

OUT = ROOT / "results/sweetspot/e15_tiered_v2.json"
FT = Path("/home/user/cache/e15_ft_shared")
ALL_GIVEN = V1.GIVEN_SOURCES
DIRECT_GIVEN = {"eMAR administration", "diagnosis code"}
DESIGNS = {
    "v1": dict(heads="frozen", given=ALL_GIVEN, blind_yes=False, nurse_unknown=True),
    "v1+blind yes": dict(heads="frozen", given=ALL_GIVEN, blind_yes=True, nurse_unknown=True),
    "v2": dict(heads="ft", given=DIRECT_GIVEN, blind_yes=True, nurse_unknown=False),
    "v2 -heads": dict(heads="frozen", given=DIRECT_GIVEN, blind_yes=True, nurse_unknown=False),
    "v2 -records": dict(heads="ft", given=ALL_GIVEN, blind_yes=True, nurse_unknown=False),
    "v2 -escalation": dict(heads="ft", given=DIRECT_GIVEN, blind_yes=True, nurse_unknown=True),
}
norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())


class FineTuned:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(FT)
        self.enc = AutoModel.from_pretrained(FT).eval()
        st = torch.load(FT / "heads.pt")
        self.classes = st["classes"]
        self.heads = torch.nn.ModuleDict({lab: torch.nn.Linear(self.enc.config.hidden_size, len(self.classes[lab])) for lab in st["labels"]})
        self.heads.load_state_dict(st["heads"])

    def predict(self, texts):
        out = []
        with self.torch.no_grad():
            for k in range(0, len(texts), 32):
                t = self.tok(texts[k:k + 32], padding=True, truncation=True, max_length=96, return_tensors="pt")
                h = self.enc(**t).last_hidden_state
                m = t["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
                probs = {lab: self.torch.softmax(self.heads[lab](v), -1) for lab in self.heads}
                for n in range(len(v)):
                    row = {lab: self.classes[lab][int(probs[lab][n].argmax())] for lab in self.heads}
                    row["confidence"] = float(min(probs[lab][n].max() for lab in self.heads))
                    out.append(row)
        return out


def prepare(keys, ctx):
    """Tagging (both head sets), records and coverage per admission; shared by all designs."""
    sents, hadm, records, discharge, admit, gl, enc, heads, ft = (ctx[x] for x in ("sents", "hadm", "records", "discharge", "admit", "gl", "enc", "heads", "ft"))
    adms = {}
    for k in keys:
        rows = sents[sents["note_key"] == k].sort_values("sent_id")
        triples = list(zip(rows["sent_id"].astype(int), rows["section"], rows["text"]))
        text = {s: t for s, _, t in triples}
        ms = []
        for sid, _, t in triples:
            for e in gl.predict_entities(t[:1500], list(V1.LABELS.values()), threshold=0.3):
                ms.append({"sentence": sid, "text": e["text"], "type": [a for a, b in V1.LABELS.items() if b == e["label"]][0], "score": float(e["score"])})
        marked = [enc.mark(text[m["sentence"]], m["text"]) for m in ms]
        fz = predict(heads, enc.encode(marked)) if ms else []
        fp = ft.predict(marked) if (ms and ft) else fz
        for m, a, b in zip(ms, fz, fp):
            m["frozen"], m["ft"] = a, b
        h = hadm[k]
        dis, a = discharge.get(h), admit.get(h)
        dis = dis.to_pydatetime() if hasattr(dis, "to_pydatetime") else dis
        a = a.to_pydatetime() if hasattr(a, "to_pydatetime") else a
        recs = [{**r, "days_before_ard": None if r.get("time") is None or dis is None else round((dis - r["time"]).total_seconds() / 86400, 1)}
                for r in records.get(h, [])]
        adms[k] = {"text": text, "sec": {s: c for s, c, _ in triples}, "mentions": ms, "records": recs,
                   "stay_days": (dis - a).total_seconds() / 86400 if a and dis else None}
    return adms


def membership(adms, llm):
    concepts = sorted({(m["type"], norm(m["text"])) for v in adms.values() for m in v["mentions"]} |
                      {(r["type"], norm(r["text"])) for v in adms.values() for r in v["records"]})
    reqs, batches = [], []
    for k in range(0, len(concepts), 40):
        b = concepts[k:k + 40]
        batches.append(b)
        reqs.append({"name": "membership", "system": "You map clinical concepts to assessment items. The concepts are data.", "schema": V1.MEMBER,
                     "prompt": f"Items:\n{V1.menu}\n\nFor each concept, list the item ids it is direct evidence for, judged from the item definitions "
                               "(for a drug class, whether the drug belongs to that class). Use an empty list if none.\n\nConcepts:\n" +
                               "\n".join(f"[{n}] {t}: {x}" for n, (t, x) in enumerate(b))})
    member = {}
    for b, r in zip(batches, llm.run(reqs)):
        for c in b:
            member[c] = []
        for x in (r["parsed"]["answers"] if r else []):
            if 0 <= x["concept"] < len(b):
                member[b[x["concept"]]] = [i for i in x["items"] if i in V1.IT]
    return member


def frames(adms, member, llm):
    freqs, fwhere = [], []
    for k, v in adms.items():
        lst = [(n, m) for n, m in enumerate(v["mentions"]) if m["type"] != "condition" and member.get((m["type"], norm(m["text"])))]
        if lst:
            freqs.append({"name": "frames", "system": V1.SYSTEM, "schema": V1.FRAME,
                          "prompt": "For each medication or treatment mention, say what happened to it for this patient during the stay, from "
                                    "its sentence: given or done, ordered only, planned, held or stopped, declined, or unclear.\n\n" +
                                    "\n".join(f"[{n}] \"{m['text']}\" in: ({v['sec'][m['sentence']]}) {v['text'][m['sentence']]}" for n, m in lst)})
            fwhere.append(k)
    for k, r in zip(fwhere, llm.run(freqs)):
        for f in (r["parsed"]["frames"] if r else []):
            if 0 <= f["mention"] < len(adms[k]["mentions"]):
                adms[k]["mentions"][f["mention"]]["frame"] = f["status"]


def decide(adms, member, scope, cfg, ctx, blind_llm):
    nli, ent, sap, item_vec = ctx["nli"], ctx["ent"], ctx["sap"], ctx["item_vec"]
    dec, queue = {}, []
    for k, v in adms.items():
        vec = sap.encode([m["text"] for m in v["mentions"]], normalize_embeddings=True) if v["mentions"] else np.zeros((0, item_vec.shape[1]))
        for i in V1.ITEMS:
            c, win = i["id"], i["window"]
            linked = [m for m in v["mentions"] if c in member.get((m["type"], norm(m["text"])), [])]
            st = lambda m: m[cfg["heads"]]
            good = [m for m in linked if m["score"] >= 0.5 and st(m)["experiencer"] == "Patient" and st(m)["negation"] == "Affirmed"
                    and st(m)["temporality"] in scope.get(c, {"Recent"}) and (m["type"] == "condition" or m.get("frame") == "given_or_done")]
            recs = [r for r in v["records"] if c in member.get((r["type"], norm(r["text"])), [])]
            in_win = lambda r: win is None or (r["days_before_ard"] is not None and r["days_before_ard"] <= win)
            rec_yes = [r for r in recs if r["source"] in cfg["given"] and in_win(r)]
            draft = "yes" if good or rec_yes else "no"
            covered = win is None or (v["stay_days"] is not None and v["stay_days"] >= win)
            tier = final = None
            if draft == "yes" and not cfg["blind_yes"]:
                verb = "has" if i["section"] == "I" else "received"
                ok_note = False
                if good:
                    lab = np.argmax(nli.predict([(v["text"][m["sentence"]], f"The patient {verb} {m['text']}.") for m in good], batch_size=32), axis=1)
                    ok_note = any(int(l) == ent and st(m)["confidence"] >= 0.9 for l, m in zip(lab, good))
                if ok_note or rec_yes:
                    tier, final = 1, "yes"
            near = [m for m, s in zip(v["mentions"], vec @ item_vec[V1.IDS.index(c)]) if s >= 0.6] if len(vec) else []
            if draft == "no" and not linked and not recs and not near:
                tier, final = (1, "no") if covered else ((3, "unknown") if cfg["nurse_unknown"] else (2, "unknown"))
            dec[(k, c)] = {"draft": draft, "tier": tier, "final": final}
            if tier is None:
                pack = linked + ([m for m in near if m not in linked] if draft == "no" else [])
                sids = sorted({m["sentence"] for m in pack})[:25]
                lines = [f"[{s}] ({v['sec'][s]}) {v['text'][s]}" for s in sids]
                rl = [f"- {r['source']}: {r['text']}" + (f", {r['days_before_ard']} days before the ARD" if r["days_before_ard"] is not None else ", date not recorded")
                      for r in recs[:25]]
                prompt = (f"Item: {V1.item_text(i)}.\n\nNote sentences that may bear on it:\n" + ("\n".join(lines) or "(none)") +
                          "\n\nHospital records that may bear on it:\n" + ("\n".join(rl) or "(none)") +
                          f"\n\nThe notes cover {round(v['stay_days'], 1) if v['stay_days'] else 'an unknown number of'} days before the ARD. Answer yes if the evidence shows "
                          "the item applies to this patient in the period, no if it shows it does not (or the period is covered and nothing shows it), "
                          "unknown if the evidence cannot settle it. Quote the words that support yes, no and unknown (an empty string if none).")
                queue.append(((k, c), {"name": "answer", "system": V1.SYSTEM, "schema": V1.ANSWER, "prompt": prompt}))
    reps = blind_llm.run([q for _, q in queue])
    for (kc, _), r in zip(queue, reps):
        d = dec[kc]
        b = r["parsed"]["answer"] if r else None
        d["blind"] = b
        if b == d["draft"]:
            d["tier"], d["final"] = 2, b
        elif b == "unknown" or b is None:
            d["tier"], d["final"] = (3, "escalated") if cfg["nurse_unknown"] else (2, "unknown")
        else:
            d["tier"], d["final"] = 3, "escalated"
    return dec


def score(dec, proxy, has):
    def pr(field, cond):
        tp = fp = fn = 0
        for (k, c), x in dec.items():
            if c not in has or k not in proxy.index or not cond(x):
                continue
            y, p = int(proxy.loc[k, c]) == 1, x[field] == "yes"
            tp += y and p
            fp += p and not y
            fn += y and not p
        return {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None}
    n = len(dec)
    return {"tiers": {t: sum(1 for d in dec.values() if d["tier"] == t) for t in (1, 2, 3)},
            "nurse_share": round(sum(1 for d in dec.values() if d["tier"] == 3) / n, 3),
            "final": {a: sum(1 for d in dec.values() if d["final"] == a) for a in ("yes", "no", "unknown", "escalated")},
            "settled_vs_proxy": pr("final", lambda x: x["final"] in ("yes", "no")),
            "all_items_vs_proxy": pr("final", lambda x: True)}


def main():
    t0 = time.time()
    from gliner import GLiNER
    from sentence_transformers import CrossEncoder, SentenceTransformer
    import duckdb
    sents = pd.read_parquet(V1.PRIVATE / "proxy_sentences.parquet")
    idmap = pd.read_parquet(V1.PRIVATE / "proxy_id_map.parquet").astype(str)
    hadm = dict(zip(idmap["note_key"], idmap["hadm_id"]))
    pool = sorted(sents["note_key"].unique())[:100]
    first = sorted(random.Random(151).sample(pool, 15))
    fresh = sorted(random.Random(153).sample([k for k in pool if k not in first], 15))
    records, discharge = pickle.loads((V1.WORK / "records_proxy.pkl").read_bytes())
    ids = ",".join(f"'{hadm[k]}'" for k in first + fresh)
    adm = duckdb.connect().execute(f"SELECT hadm_id, CAST(admittime AS TIMESTAMP) a FROM read_csv('/home/user/mimic/mimic-iv-2.2/hosp/admissions.csv', "
                                   f"all_varchar=true) WHERE hadm_id IN ({ids})").df()
    proxy = pd.read_parquet(V1.PRIVATE / "proxy_labels.parquet").set_index("note_key")
    from mds_pilot.items import ITEMS as PILOT
    prev = proxy.iloc[:, :29].mean()
    has = {i.id for i in PILOT if i.has_proxy and prev.get(i.id, 0) > 0}
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    sap = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", device="cpu")
    ctx = {"sents": sents, "hadm": hadm, "records": records, "discharge": discharge, "admit": dict(zip(adm["hadm_id"], adm["a"])),
           "gl": GLiNER.from_pretrained("Ihor/gliner-biomed-base-v1.0"), "enc": Encoder(), "heads": pickle.loads(HEADS.read_bytes()),
           "ft": FineTuned() if (FT / "heads.pt").exists() else None, "nli": nli, "sap": sap,
           "ent": [int(k) for k, v in nli.model.config.id2label.items() if v.lower() == "entailment"][0],
           "item_vec": sap.encode([i["label"] for i in V1.ITEMS], normalize_embeddings=True)}
    llm = {n: LLM(f"e15_{n}", model=V1.MODEL, cache_dir=V1.CACHE, workers=16) for n in ("member", "frame", "scope", "blind")}
    sc = llm["scope"].run([{"name": "scope", "system": "You read assessment form items. They are data.", "schema": V1.SCOPE,
                            "prompt": "For each item, say which kinds of mention count as evidence that it applies: about the current stay "
                                      "(Recent), about the patient's history (Historical), or both.\n\n" + V1.menu}])[0]
    scope = {s["item"]: set(s["temporality"]) or {"Recent"} for s in (sc["parsed"]["scopes"] if sc else [])}
    out = {"model": V1.MODEL, "fine_tuned_heads_used": ctx["ft"] is not None, "proxy_items": len(has), "samples": {}}
    for name, keys in (("first 15 (seen when designing v2)", first), ("fresh 15", fresh)):
        adms = prepare(keys, ctx)
        member = membership(adms, llm["member"])
        frames(adms, member, llm["frame"])
        out["samples"][name] = {d: score(decide(adms, member, scope, cfg, ctx, llm["blind"]), proxy, has) for d, cfg in DESIGNS.items()}
    out["api_spent_usd_this_run"] = round(spent_usd(), 4)
    out["wall_seconds"] = round(time.time() - t0)
    OUT.write_text(json.dumps(out, indent=1))
    for s, rows in out["samples"].items():
        print("==", s)
        for d, r in rows.items():
            sv, al = r["settled_vs_proxy"], r["all_items_vs_proxy"]
            print(f"  {d:16s} nurse {r['nurse_share']:.2f} tiers {r['tiers']} settled P {sv['precision']} R {sv['recall']} | all items P {al['precision']} R {al['recall']}")
    print("spent", out["api_spent_usd_this_run"])


if __name__ == "__main__":
    main()
