"""Exact workload and ceiling for porting the extractor + verifier + status hybrid to MIMIC Section N. No model calls.

Uses the extractor's mention positions (sentence number, drug words) for the 500 proxy notes, the language-model class map (stored replies),
and the N-at-discharge task frame (244 admissions, 11 classes). Reports: mentions that map to an MDS class; distinct (note, class) units;
verifier and status-tagging calls under two designs; the share of truly administered (note, class) pairs the extractor reaches at all
(a ceiling on the note edge); and the same for the earlier retrieval route for comparison.
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.segment import segment_note
from run_reader import Store, load_env
import test_class_membership as CM
import test_learned_graph as LG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/hybrid_plan"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    load_env()
    model = os.environ.get("MDS_LLM_MODEL", "unset")
    ids = pd.read_parquet(ROOT / "pilot_private/proxy_id_map.parquet").astype(str)
    adm, rx, emar, ext, rx_names, log_names, words = CM.collect_strings(ids)
    strings = sorted(set(rx_names) | set(log_names) | set(words))
    reqs = CM.classify_requests(strings, model)
    llm = CM.llm_map(reqs, Store(ROOT / "pilot_private/runs/proxy500/class_membership.jsonl"))
    (OUT / "llm_class_map.json").write_text(json.dumps({s: sorted(c) for s, c in llm.items() if c}))
    summary = json.loads((ROOT / "results/mimic_extractor/summary.json").read_text())
    mentions = summary["ext_mentions"]

    # sentence numbering check against the local sentence table
    texts = pd.read_parquet(ROOT / "pilot_private/mimic_ner/proxy_notes.parquet")
    local = pd.read_parquet(ROOT / "pilot_private/proxy_sentences.parquet", columns=["note_key", "sent_id"]).groupby("note_key").size()
    same = sum(len(segment_note(t)) == local.get(k, -1) for k, t in zip(texts["note_key"], texts["text"]))

    full = LG.edge_rows()
    n = full[full["task"] == "N_window"]
    admissions = sorted(n["admission"].unique())
    maps = {"list": CM.list_map(strings), "llm": llm}
    edges = CM.build_edges(ids, adm, rx, emar, ext, {"list": maps["list"], "atc": maps["list"], "llm": llm})
    truth = edges["llm"]["key"]

    units = defaultdict(set)         # (note, class) -> set of sentence ids
    relevant_spans = 0
    all_spans = 0
    sentences_with_relevant = set()
    for note in admissions:
        for sent_id, s, e, w in mentions.get(note, []):
            all_spans += 1
            classes = set().union(*(llm.get(x, set()) for x in w.split())) if w else set()
            if classes:
                relevant_spans += 1
                sentences_with_relevant.add((note, sent_id))
                for c in classes:
                    units[(note, c)].add(sent_id)
    pairs = [(a, c) for a in admissions for c in CM.CLASS_IDS]
    reached = sum((a, c) in units for a, c in pairs)
    true_pairs = [p for p in pairs if p in truth]
    true_reached = sum(p in units for p in true_pairs)
    false_pairs = [p for p in pairs if p not in truth]
    sent_per_unit = [len(v) for v in units.values()]
    res = {"sentence_numbering_matches_local_table": f"{same}/{len(texts)} notes", "admissions": len(admissions), "pairs": len(pairs),
           "mention_spans_all": all_spans, "mention_spans_mapping_to_an_MDS_class": relevant_spans,
           "distinct_sentences_with_a_class_mention": len(sentences_with_relevant),
           "note_class_units_with_a_mention": len(units), "mean_sentences_per_unit": round(sum(sent_per_unit) / max(len(sent_per_unit), 1), 2),
           "truly_administered_pairs": len(true_pairs), "truly_administered_reached_by_extractor": true_reached,
           "ceiling_recall": round(true_reached / max(len(true_pairs), 1), 4),
           "not_administered_pairs_with_a_mention": sum(p in units for p in false_pairs), "not_administered_pairs": len(false_pairs)}
    # call counts and token estimates
    ver_per_span = relevant_spans
    ver_per_sentence_word = len({(note, sid, w) for note in admissions for sid, s, e, w in mentions.get(note, []) if w and set().union(*(llm.get(x, set()) for x in w.split()))})
    res["verifier_calls_one_per_mention"] = ver_per_span
    res["verifier_calls_one_per_sentence_and_word"] = ver_per_sentence_word
    res["status_calls_one_per_note_class_unit"] = len(units)
    res["tokens_estimate"] = {"verifier_per_mention": ver_per_span * 330, "verifier_per_sentence_word": ver_per_sentence_word * 330,
                              "status_per_unit": int(len(units) * (350 + 45 * res["mean_sentences_per_unit"])),
                              "old_route_status_calls_for_N": len(pairs), "old_route_tokens_measured_per_call": 1065}
    (OUT / "plan.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
