"""Tests on invented text only. No MIMIC data is used here."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mds_pilot.items import ITEMS
from mds_pilot.metrics import auroc, cohen_kappa, weighted_recall
from mds_pilot.retrieval import (BM25, align_cached_embeddings, lexicon_scores, note_index_from, rrf, term_pattern,
                                 within_note_ranks)
from mds_pilot.segment import segment_note

NOTE = """Name: ___ Unit No: ___

Chief Complaint:
Shortness of breath

History of Present Illness:
Mr. ___ is a ___ year old man with CHF who presented with dyspnea and was found
to have volume overload. He denies chest pain. Dr. ___ was consulted.

Past Medical History:
- CKD stage 3
- Atrial fibrillation

Brief Hospital Course:
# CHF exacerbation: treated with IV Lasix.
He was started on HD on hospital day 2.

Medications on Admission:
1. Aspirin 81 mg PO DAILY 2. Warfarin 5 mg PO DAILY

Discharge Medications:
1. Furosemide 40 mg PO DAILY
2. Levetiracetam 500 mg PO BID

Discharge Diagnosis:
Heart failure
"""


def test_sections_found():
    rows = segment_note(NOTE)
    sections = {r["section"] for r in rows}
    assert {"Chief Complaint", "History of Present Illness", "Past Medical History",
            "Brief Hospital Course", "Discharge Medications", "Discharge Diagnosis"} <= sections


def test_wrapped_line_joined_and_abbreviation_kept():
    texts = [r["text"] for r in segment_note(NOTE)]
    assert any("presented with dyspnea and was found to have volume overload." in t for t in texts)
    assert any(t.startswith("Dr. ___ was consulted") for t in texts)
    assert any(t.startswith("Mr. ___ is a") for t in texts)


def test_list_items_split_including_inline():
    rows = segment_note(NOTE)
    meds = [r["text"] for r in rows if r["section"] == "Discharge Medications"]
    assert meds == ["1. Furosemide 40 mg PO DAILY", "2. Levetiracetam 500 mg PO BID"]
    admission = [r["text"] for r in rows if r["section"] == "Medications on Admission"]
    assert admission == ["1. Aspirin 81 mg PO DAILY", "2. Warfarin 5 mg PO DAILY"]


def test_empty_note():
    assert segment_note("") == []
    assert segment_note(None) == []


def test_short_abbreviation_is_case_sensitive():
    assert term_pattern("HD").search("started HD today")
    assert not term_pattern("HD").search("the hd unit")
    assert term_pattern("dialysis").search("DIALYSIS line")
    assert not term_pattern("AF").search("AFTER lunch")


def test_lexicon_finds_brand_name():
    texts = ["Given IV Lasix for overload.", "No acute distress."]
    scores = lexicon_scores(texts)
    j = [i.id for i in ITEMS].index("N_diuretic")
    assert scores[0, j] == 1 and scores[1, j] == 0


def test_bm25_prefers_rare_matching_term():
    index = BM25(["started on dialysis today", "patient comfortable today", "today today today"])
    s = index.score(["dialysis"])
    assert s[0] > 0 and s[1] == 0 and s[2] == 0


def test_ranks_are_within_each_note():
    scores = np.array([[0.1], [0.9], [0.5], [0.2]], dtype=np.float32)
    ranks = within_note_ranks(scores, note_index_from(["a", "a", "b", "b"]))
    assert ranks.ravel().tolist() == [2, 1, 1, 2]


def test_rrf_rewards_agreement():
    a = np.array([[1], [2], [3]])
    b = np.array([[1], [3], [2]])
    f = rrf([a, b]).ravel()
    assert f[0] > f[1] and np.isclose(f[1], f[2])


def test_metrics_hand_computed():
    assert np.isclose(weighted_recall([1, 0], [3, 1]), 0.75)
    assert np.isclose(cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]), 0.5)
    assert np.isnan(auroc([1, 0], [0.5, 0.2]))
    labels = [1] * 5 + [0] * 5
    assert auroc(labels, list(range(10, 0, -1))) == 1.0


def test_window_stays_inside_note():
    from evaluate_gold import window
    mask = np.array([[False], [True], [False], [False], [False]])
    out = window(mask, note_index_from(["a", "a", "a", "b", "b"]), 1)
    assert out.ravel().tolist() == [True, True, True, False, False]


def test_evaluate_counts_hits_and_weights():
    import pandas as pd
    from evaluate_gold import evaluate
    n_items = len(ITEMS)
    gold = np.zeros((3, n_items), dtype=bool)
    gold[0, 0] = True
    gold[2, 0] = True
    retrieved = np.zeros((3, n_items), dtype=bool)
    retrieved[0, 0] = True
    notes = np.array(["a", "a", "b"])
    meta = pd.DataFrame({"stratum": ["x", "y"], "weight": [3.0, 1.0]}, index=["a", "b"])
    res, _ = evaluate({"m": retrieved}, gold, notes, note_index_from(notes), meta)
    assert res.loc[0, "note_item_recall_weighted"] == 0.75
    assert res.loc[0, "note_item_recall_unweighted"] == 0.5
    assert res.loc[0, "sentence_recall"] == 0.5


def test_reranker_keeps_unscored_sentences_last(monkeypatch):
    import pandas as pd
    import sentence_transformers
    from evaluate_gold import rerank

    class Stub:
        def __init__(self, *_, **__):
            pass

        def predict(self, pairs, **_):
            return np.arange(len(pairs), dtype=np.float32)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Stub)
    fused = np.tile(np.array([[1], [2], [3]]), (1, len(ITEMS)))
    scores = rerank(fused, pd.DataFrame({"text": ["a", "b", "c"]}), top=2)
    ranks = within_note_ranks(scores, note_index_from(["n", "n", "n"]))
    assert ranks[2, 0] == 3
    assert ranks[1, 0] == 1


def test_known_facts_cover_every_item_and_paraphrases_avoid_lexicon():
    from mds_pilot.known_facts import FACTS
    assert set(FACTS) == {i.id for i in ITEMS}
    for j, item in enumerate(ITEMS):
        explicit, variant, paraphrase, negated = FACTS[item.id]
        scores = lexicon_scores([explicit, variant, paraphrase, negated])[:, j]
        assert scores.tolist() == [1, 1, 0, 1], item.id


def test_rank_helpers_count_ties_against_target():
    from run_self_supervised import fused_rank, rank_of_target
    assert rank_of_target(np.array([0.9, 0.5]), 0.9) == 2
    assert rank_of_target(np.array([0.4, 0.5]), 0.9) == 1
    first = np.array([0.1, 0.2, 0.9])
    assert fused_rank([first, first]) == 1
    last = np.array([0.9, 0.8, 0.1])
    assert fused_rank([last, last]) == 3


def test_cached_embeddings_follow_row_order():
    import pandas as pd
    sent = pd.DataFrame({"note_key": ["b", "b", "a", "a"], "sent_id": [0, 1, 0, 1]})
    cached = np.array([[1.0], [2.0], [3.0], [4.0]])  # a0, a1, b0, b1
    assert align_cached_embeddings(cached, sent).ravel().tolist() == [3.0, 4.0, 1.0, 2.0]


def test_status_rules_cover_each_state():
    from mds_pilot.reader import derive_state
    mention = lambda subject, assertion, timing: {"sentence": 3, "subject": subject, "assertion": assertion, "timing": timing}
    assert derive_state([]) == ("not_documented", [])
    assert derive_state([mention("patient", "affirmed", "current_stay")]) == ("yes", [3])
    assert derive_state([mention("patient", "negated", "current_stay")])[0] == "no"
    assert derive_state([mention("other_person", "affirmed", "chronic_ongoing")])[0] == "no"
    assert derive_state([mention("patient", "affirmed", "resolved_past")])[0] == "no"
    assert derive_state([mention("patient", "uncertain", "unclear")])[0] == "unclear"
    assert derive_state([mention("patient", "negated", "unclear"), mention("patient", "affirmed", "chronic_ongoing")])[0] == "yes"


def test_flip_sentence_answers_are_known():
    from mds_pilot.reader import flip_sentences
    for section in "INO":
        cases = flip_sentences(section, "x", "yes one", "yes two")
        assert [cases[c][1] for c in ("affirmed", "paraphrase")] == [True, True]
        assert not any(cases[c][1] for c in ("family", "past", "hypothetical"))


def test_hard_cases_are_balanced_and_share_their_opening():
    from mds_pilot.reader import hard_cases
    for item in ITEMS:
        cases = hard_cases(item)
        assert len(cases) == 10 and sum(c[2] for c in cases.values()) == 5
        for name in ("workup", "listing", "history", "subject", "new"):
            yes, no = cases[f"{name}_yes"], cases[f"{name}_no"]
            assert yes[0] == no[0] and yes[1] != no[1] and yes[2] and not no[2]
        assert not any("{" in text for case in cases.values() for text in case[:2])


def test_added_sentences_keep_their_order_and_ids():
    from run_reader import prompt_rows
    lookup = {("n", 7): ("A", "first real"), ("n", 9): ("B", "second real")}
    rows, numbers = prompt_rows("n", [7, 9], lookup, [(1, "opening"), (1, "decider")])
    assert [r[2] for r in rows] == ["first real", "opening", "decider", "second real"]
    assert numbers == {1: 7, 2: -1, 3: -2, 4: 9}
    rows, numbers = prompt_rows("n", [7, 9], lookup, [(0, "opening"), (2, "decider")])
    assert [r[2] for r in rows] == ["opening", "first real", "second real", "decider"]


def test_full_note_decision_rule():
    from analyze_full import missed_share, verdict
    assert missed_share([1, 1, 1, 1, 0], [1, 1, 1, 0, 1]) == 0.25
    assert np.isnan(missed_share([0, 0], [1, 0]))
    assert verdict(0.05, 0.5) == "keep retrieval as it is"
    assert verdict(0.15, 0.7) == "keep retrieval but fix its misses first"
    assert verdict(0.16, 0.7) == "retrieval does not justify itself as configured"
    assert verdict(0.0, 0.4) == "retrieval does not justify itself as configured"


def test_reveal_prompt_states_the_shown_table_value():
    from mds_pilot.reader import build_reveal_prompt, table_source
    by_id = {i.id: i for i in ITEMS}
    rows = [(1, "Hospital Course", "Invented sentence.")]
    assert table_source(by_id["N_opioid"]) == "prescription orders"
    assert table_source(by_id["O_chemotherapy"]) == "diagnosis codes and procedure codes"
    assert "lists the item" in build_reveal_prompt(by_id["N_opioid"], rows, False, True)
    told_no = build_reveal_prompt(by_id["N_opioid"], rows, True, False)
    assert "does not list the item" in told_no and "answered yes" in told_no


def test_reveal_decision_rule():
    from analyze_reveal import verdict
    assert verdict(0.06, 0.2, 1.0).startswith("anchored")
    assert verdict(0.05, 0.0, 1.0).startswith("no signal")
    assert verdict(0.0, 0.1, 0.89).startswith("changed answers")
    assert verdict(0.0, 0.1, 0.9).startswith("the reveal may")
    assert verdict(0.0, 0.1, float("nan")).startswith("the reveal may")


def test_every_item_maps_to_mds_codes():
    import re
    from mds_pilot.mds_fields import MDS_FIELDS
    assert set(MDS_FIELDS) == {i.id for i in ITEMS}
    for item in ITEMS:
        field = MDS_FIELDS[item.id]
        assert all(re.fullmatch(rf"{item.section}\d{{4}}([A-Z]\d)?", c) for c in field.codes), item.id
        assert field.match == "direct" or field.note
    codes = [c for f in MDS_FIELDS.values() for c in f.codes]
    assert len(codes) == len(set(codes))


def test_build_field_maps_evidence_and_flags():
    from mds_pilot.mds_fields import build_field
    ids = [4, 9, 12]
    agreed = build_field("I_diabetes", True, [2], ids, True)
    assert agreed["evidence_sent_ids"] == [9] and not agreed["review"] and agreed["mds_codes"] == ["I2900"]
    assert build_field("I_diabetes", True, [1], ids, False)["review_reasons"] == ["note_yes_table_no"]
    assert build_field("I_diabetes", False, [], ids, True)["review_reasons"] == ["note_no_table_yes"]
    assert build_field("O_oxygen", False, [], ids, None)["table_status"] == "no_table_value"
    assert not build_field("O_oxygen", False, [], ids, None)["review"]
    bad = build_field("O_oxygen", True, [7], ids, None)
    assert bad["evidence_sent_ids"] == [] and bad["review_reasons"] == ["yes_without_evidence", "citation_out_of_range"]


class FakeClient:
    """Stands in for the OpenAI client: answers yes for diabetes and can fail on demand."""

    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail
        self.chat = self
        self.completions = self

    def create(self, messages, **_):
        import json
        from types import SimpleNamespace as NS
        self.calls += 1
        if self.fail:
            raise RuntimeError("no connection")
        yes = "diabetes mellitus" in messages[1]["content"]
        content = json.dumps({"answer": "yes" if yes else "no", "evidence": [1] if yes else []})
        return NS(usage=NS(prompt_tokens=10, completion_tokens=2), choices=[NS(message=NS(content=content))])


def pipeline_inputs():
    import pandas as pd
    notes = pd.DataFrame({"note_key": ["n1", "n2", "n3"], "text": [NOTE, NOTE.replace("Heart failure", "Diabetes mellitus"), ""]})
    labels = pd.DataFrame({i.id: [0, 1, 0] for i in ITEMS if i.has_proxy}, index=["n1", "n2", "n3"])
    return notes, labels


def fake_scorer(texts):
    rng = np.random.default_rng(0)
    return [rng.random((len(texts), len(ITEMS)), dtype=np.float32) for _ in range(2)]


def test_single_command_runs_end_to_end_on_invented_notes(tmp_path):
    import json
    import stat
    from run_pipeline import run
    notes, labels = pipeline_inputs()
    client = FakeClient()
    assert run(notes, "t", 20, fake_scorer, client, "fake", 10_000_000, False, labels, tmp_path) == 0
    assert client.calls == 2 * len(ITEMS)
    private = tmp_path / "pilot_private/runs/t/mds_output.jsonl"
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    rows = [json.loads(x) for x in private.read_text().splitlines()]
    assert [r["note"] for r in rows] == ["n1", "n2", "n3"]
    diabetes = next(f for f in rows[0]["fields"] if f["item"] == "I_diabetes")
    assert diabetes["answer"] == "yes" and diabetes["evidence"][0]["text"] and diabetes["table_status"] == "note_yes_table_no"
    assert all(f["review_reasons"][-1] == "no_sentences" for f in rows[2]["fields"])
    for path in (tmp_path / "results/runs/t").iterdir():
        if path.suffix != ".parquet":
            assert "Levetiracetam" not in path.read_text()
    assert run(notes, "t", 20, fake_scorer, client, "fake", 10_000_000, False, labels, tmp_path) == 0
    assert client.calls == 2 * len(ITEMS)


def test_single_command_writes_nothing_on_failure_limit_or_dry_run(tmp_path):
    from run_pipeline import run
    notes, labels = pipeline_inputs()
    assert run(notes, "a", 20, fake_scorer, FakeClient(fail=True), "fake", 10_000_000, False, labels, tmp_path) == 1
    assert not (tmp_path / "pilot_private/runs/a/mds_output.jsonl").exists()
    client = FakeClient()
    assert run(notes, "b", 20, fake_scorer, client, "fake", 10, False, labels, tmp_path) == 2
    assert client.calls == 0
    assert run(notes, "c", 20, fake_scorer, None, "fake", 10, True, labels, tmp_path) == 0
    assert not (tmp_path / "pilot_private/runs/c/mds_output.jsonl").exists()


def test_table_flags_match_prefixes_and_drugs():
    import pandas as pd
    from mds_pilot.tables import item_flags
    dx = pd.DataFrame({"hadm_id": ["1"], "icd_code": ["I509"]})
    proc = pd.DataFrame({"hadm_id": ["2"], "icd_code": ["3995"]})
    rx = pd.DataFrame({"hadm_id": ["2"], "drug": ["furosemide 40 mg"]})
    flags = item_flags(dx, proc, rx, ["1", "2", "3"]).set_index("hadm_id")
    assert flags.loc["1", "I_heart_failure"] == 1 and flags.loc["2", "I_heart_failure"] == 0
    assert flags.loc["2", "O_dialysis"] == 1 and flags.loc["2", "N_diuretic"] == 1
    assert flags.loc["3"].sum() == 0


def test_graph_adds_followers_inside_the_section_only():
    from mds_pilot.graph import expand_with_followers
    candidates = np.array([True, False, False, False, False])
    mention = np.array([True, False, False, False, False])
    sections = np.array(["A", "A", "A", "B", "B"])
    assert expand_with_followers(candidates, mention, sections, 1).tolist() == [True, True, False, False, False]
    assert expand_with_followers(candidates, mention, sections, 5).tolist() == [True, True, True, False, False]
    assert expand_with_followers(candidates, ~mention, sections, 2).tolist() == candidates.tolist()


def test_added_pair_has_the_requested_gap():
    from test_graph_and_gate import top_k_with_added
    scores = np.ones((3, 4), dtype=np.float32)
    added = np.ones((3, 2), dtype=np.float32)
    for gap in (0, 1, 2):
        _, opening, decider, _ = top_k_with_added(scores, added, 1, 1 + gap)
        assert decider - opening - 1 == gap


def test_correction_links_point_from_later_updates_to_earlier_mentions():
    from mds_pilot.correction_graph import build_links, is_update, links_text, select_context
    assert is_update("Lasix was held on admission.") and is_update("It was resumed.") and not is_update("Lasix 40 mg daily.")
    mention = np.array([True, False, True, False])
    update = np.array([False, True, True, False])
    assert build_links(mention, update) == {1: [0], 2: [0]}
    assert build_links(np.array([False, True]), np.array([True, False])) == {}
    assert links_text({1: [0], 2: [0]}) == "Linked updates: [2] may update [1]; [3] may update [1]."
    assert links_text({}) == "Linked updates: none."
    candidates = np.array([True, False, False, False])
    sections = np.array(["A", "A", "A", "B"])
    near_update = np.array([False, False, True, True])
    assert select_context(candidates, np.array([True, False, False, False]), near_update, sections).tolist() == [True, False, True, False]


def test_correction_decision_needs_every_check():
    from test_correction_graph import verdict
    good = dict(gain_low=.1, graph_no=.9, links_low=.01, placebo=.95, placebo_drop=.02, change=.03, toward=2, away=1, token_rise=.25)
    assert verdict(**good)[1]
    for change in ({"gain_low": 0}, {"graph_no": .89}, {"links_low": 0}, {"placebo": .94}, {"placebo_drop": .03}, {"change": .029}, {"toward": 1}, {"token_rise": .26}):
        assert not verdict(**{**good, **change})[1], change


def test_record_lines_use_days_of_stay_only():
    import pandas as pd
    from mds_pilot.records_graph import balanced_accuracy, record_lines
    admit, disch = pd.Timestamp("2150-01-01 10:00"), pd.Timestamp("2150-01-06 09:00")
    orders = [("BiPAP", pd.Timestamp("2150-01-02 11:00")), ("BiPAP", pd.Timestamp("2150-01-04 09:00"))]
    lines = record_lines(orders, admit, disch, True)
    assert lines[0] == "- Order: BiPAP, 2 order(s), first on day 2 and last on day 3 of a 5-day stay."
    assert lines[1].startswith("- Billing") and "2150" not in " ".join(lines)
    assert record_lines([], admit, disch, False) == []
    assert balanced_accuracy([True, True, False, False], [True, False, False, False]) == .75


def test_records_decision_needs_every_check():
    from test_records_graph import verdict
    assert verdict(.01, .01, .02)[1]
    for args in ((0, .01, 0), (.01, 0, 0), (.01, .01, .03)):
        assert not verdict(*args)[1]


def test_latent_model_recovers_source_quality():
    from mds_pilot.kg import dawid_skene
    rng = np.random.default_rng(0)
    truth = rng.random(4000) < .3
    sens, spec = np.array([.9, .7, .6]), np.array([.95, .9, .8])
    x = np.column_stack([np.where(truth, rng.random(4000) < s, rng.random(4000) > p) for s, p in zip(sens, spec)]).astype(int)
    post, est_sens, est_spec, prev = dawid_skene(x)
    assert abs(prev - .3) < .03
    assert np.abs(est_sens - sens).max() < .06 and np.abs(est_spec - spec).max() < .04
    assert ((post >= .5) == truth).mean() > .85


def test_fusion_methods_and_balanced_accuracy():
    import pandas as pd
    from mds_pilot.kg import balanced_accuracy, fuse
    frame = pd.DataFrame({"item": ["a"] * 4, "note": [1, 0, 1, 0], "order": [1, 1, 0, 0], "billing": [0, 1, 0, 0]})
    inputs = ["note", "order", "billing"]
    assert fuse(frame, inputs, "records").tolist() == [True, True, False, False]
    assert fuse(frame, inputs, "or").tolist() == [True, True, True, False]
    assert fuse(frame, inputs, "and").tolist() == [True, False, False, False]
    assert fuse(frame, inputs, "majority").tolist() == [True, True, False, False]
    assert balanced_accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == .75
    assert np.isnan(balanced_accuracy([1, 1], [1, 0]))


def test_evidence_edges_summarise_records_by_day():
    import pandas as pd
    from mds_pilot.kg import evidence_edges, ontology
    admit, disch = pd.Timestamp("2150-01-01 08:00"), pd.Timestamp("2150-01-05 08:00")
    records = pd.DataFrame({"value": ["heparin", "heparin", "warfarin"],
                            "time": [pd.Timestamp("2150-01-01 09:00"), pd.Timestamp("2150-01-04 20:00"), pd.Timestamp("2150-01-02 09:00")]})
    edges = {e["object"]: e for e in evidence_edges("n1", "N_anticoagulant", "prescription", records, admit, disch)}
    assert edges["heparin"]["count"] == 2 and edges["heparin"]["first_day"] == 1 and edges["heparin"]["last_day"] == 4
    assert edges["heparin"]["near_discharge"] and not edges["warfarin"]["near_discharge"]
    assert evidence_edges("n1", "x", "order", records.iloc[:0], admit, disch) == []
    links = ontology()
    assert {"prescription", "billing", "order", "prescription_for_diagnosis"} <= set(links["source"])
    assert not links["source"].isin(["icu_event", "administration", "lab_positive", "culture_positive"]).any()


def test_fusion_ignores_zero_score_ties():
    from run_self_supervised import fused_rank
    keyword = np.array([0.0, 0.0, 0.0, 0.0])
    encoder = np.array([0.1, 0.2, 0.3, 0.9])
    assert fused_rank([keyword, encoder]) == 1
    ranks = np.array([[1], [2]])
    scores = np.array([[0.0], [5.0]])
    fused = rrf([ranks], score_arrays=[scores]).ravel()
    assert fused[0] == 0 and fused[1] > 0
