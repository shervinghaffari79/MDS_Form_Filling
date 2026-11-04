"""Class-restricted comparison of the learned extractor and the hand-written lexicon. Post hoc (the restriction was chosen after the first
result), threshold unchanged. No model calls.

Class membership of each given drug word comes from WHO ATC classes through the public NLM RxClass service, not from the item lists. The only
hand-written part is the table below, 11 rows of standard ATC prefixes, one per MDS N0415 drug class (N05AN, lithium, is excluded from
antipsychotics). A drug word is in an MDS class if any of its ATC classes starts with a class prefix; words without an ATC class are
unresolved and left out, with their share reported.
K1: within the findable given drug pairs of the 11 classes (last 24 hours before discharge), extractor recall minus lexicon recall at least
0.10 with the lower bound of the 95% CI (resampling admissions) above 0.05. Also reported per class, the drug words the lexicon misses, and
the same for the drug-only lexicon.
Known gap: heparin, the most common anticoagulant, is removed from the labels by the frequency filter (it is in over half of admissions'
prescriptions, mostly flushes), so the anticoagulant class is under-evaluated.
"""
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/mimic_extractor"
CACHE = OUT / "atc_cache.json"
BOOT = 2000
CLASSES = {"N_antipsychotic": ["N05A"], "N_antianxiety": ["N05B"], "N_antidepressant": ["N06A"], "N_hypnotic": ["N05C"],
           "N_anticoagulant": ["B01AA", "B01AB", "B01AE", "B01AF"], "N_antiplatelet": ["B01AC"], "N_antibiotic": ["J01"],
           "N_diuretic": ["C03"], "N_opioid": ["N02A"], "N_hypoglycemic": ["A10"], "N_anticonvulsant": ["N03A"]}
EXCLUDE = {"N_antipsychotic": ["N05AN"]}


def atc_classes(name):
    url = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byDrugName.json?" + urllib.parse.urlencode({"drugName": name, "relaSource": "ATC"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                info = json.load(r).get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
            return sorted({x["rxclassMinConceptItem"]["classId"] for x in info})
        except Exception:
            continue
    return None


def classes_of(atc):
    out = set()
    for cls, prefixes in CLASSES.items():
        if any(a.startswith(p) and not any(a.startswith(x) for x in EXCLUDE.get(cls, [])) for a in atc for p in prefixes):
            out.add(cls)
    return out


def main():
    d = json.loads((OUT / "summary.json").read_text())
    names = sorted({p["name"] for r in d["evaluation"].values() for p in r["pairs"]})
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [n for n in names if n not in cache]
    with ThreadPoolExecutor(8) as pool:
        for n, atc in zip(todo, pool.map(atc_classes, todo)):
            cache[n] = atc
    CACHE.write_text(json.dumps(cache, indent=0))
    failed = [n for n, a in cache.items() if a is None]
    member = {n: classes_of(a) for n, a in cache.items() if a is not None}
    result = {"names": len(names), "lookup_failed": len(failed), "names_with_atc": sum(bool(cache[n]) for n in cache if cache[n] is not None),
              "names_in_an_mds_class": sum(bool(v) for v in member.values())}
    rng = np.random.default_rng(0)
    for scope, r in d["evaluation"].items():
        pairs = [p for p in r["pairs"] if member.get(p["name"])]
        notes = sorted({p["note"] for p in pairs})
        by_note = {n: [p for p in pairs if p["note"] == n] for n in notes}
        rec = lambda ps, k: float(np.mean([p[k] for p in ps])) if ps else float("nan")
        unresolved = sum(1 for p in r["pairs"] if member.get(p["name"]) is None or not cache.get(p["name"]))
        vals = {k: [] for k in ("ext", "lex", "lexd")}
        for _ in range(BOOT):
            sample = [p for i in rng.integers(0, len(notes), len(notes)) for p in by_note[notes[i]]]
            for k in vals:
                vals[k].append(rec(sample, k))
        diff = lambda a, b: np.array(vals[a]) - np.array(vals[b])
        ci = lambda x: [round(float(np.quantile(x, .025)), 4), round(float(np.quantile(x, .975)), 4)]
        res = {"admissions": len(notes), "in_class_pairs": len(pairs), "all_pairs": len(r["pairs"]), "pairs_without_atc_class": unresolved,
               "extractor_recall": round(rec(pairs, "ext"), 4), "lexicon_recall": round(rec(pairs, "lex"), 4), "drug_lexicon_recall": round(rec(pairs, "lexd"), 4),
               "K1_extractor_minus_lexicon": {"difference": round(rec(pairs, "ext") - rec(pairs, "lex"), 4), "ci": ci(diff("ext", "lex"))},
               "K1_extractor_minus_drug_lexicon": {"difference": round(rec(pairs, "ext") - rec(pairs, "lexd"), 4), "ci": ci(diff("ext", "lexd"))}}
        res["K1_passes"] = bool(res["K1_extractor_minus_drug_lexicon"]["ci"][0] > 0.05 and res["K1_extractor_minus_drug_lexicon"]["difference"] >= .10)
        per = {}
        for cls in CLASSES:
            ps = [p for p in pairs if cls in member[p["name"]]]
            if ps:
                per[cls] = {"pairs": len(ps), "distinct_names": len({p["name"] for p in ps}), "extractor": round(rec(ps, "ext"), 4), "drug_lexicon": round(rec(ps, "lexd"), 4)}
        res["by_class"] = per
        missed = [p["name"] for p in pairs if not p["lexd"]]
        from collections import Counter
        res["lexicon_misses_in_class"] = {"pairs": len(missed), "distinct_names": len(set(missed)), "top": Counter(missed).most_common(25)}
        res["extractor_misses_in_class"] = {"pairs": int(sum(not p["ext"] for p in pairs)), "top": Counter(p["name"] for p in pairs if not p["ext"]).most_common(10)}
        result[scope] = res
    (OUT / "class_restricted.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
