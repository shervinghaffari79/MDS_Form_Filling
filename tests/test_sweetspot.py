"""Offline tests for the sweetspot harness: invented fixtures, no API, no MIMIC."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sweetspot import admission_graph as A  # noqa: E402
from sweetspot import e1_entity_layer as E1  # noqa: E402
from sweetspot import perturb  # noqa: E402
from sweetspot.llm import cost, request_key  # noqa: E402


def m(sentence, text, typ="condition", subject="patient", assertion="affirmed", timing="current_stay"):
    return {"sentence": sentence, "text": text, "type": typ, "subject": subject, "assertion": assertion, "timing": timing}


def test_item_config_covers_three_sections():
    secs = {i["section"] for i in A.ITEMS["items"]}
    assert secs == {"I", "O", "P"}
    assert len(A.ITEMS["items"]) == 78
    assert not any("None of the above" in i["wording"] for i in A.ITEMS["items"])


def test_note_edge_requires_patient_affirmed_and_active_timing():
    mem = {("condition", "pneumonia"): ["I2f"], ("condition", "diabetes"): ["I1a"]}
    ms = [m(1, "pneumonia", assertion="negated"), m(2, "Pneumonia", subject="other_person"), m(3, "diabetes", timing="chronic_ongoing")]
    g = A.build_graph("h", ms, [], mem, None)
    assert "I2f" not in g          # negated and family mentions give no path
    assert g["I1a"][0]["sentence"] == 3   # a chronic active diagnosis counts for Section I


def test_chronic_timing_does_not_count_outside_section_i():
    mem = {("treatment", "home oxygen"): ["P1a-g"]}
    assert A.build_graph("h", [m(1, "home oxygen", "treatment", timing="chronic_ongoing")], [], mem, None) == {}


def test_time_update_lets_a_later_negated_mention_supersede():
    mem = {("condition", "pneumonia"): ["I2f"]}
    ms = [m(1, "pneumonia"), m(9, "pneumonia", assertion="negated", timing="ongoing_at_discharge")]
    assert "I2f" in A.build_graph("h", ms, [], mem, None)
    assert "I2f" not in A.build_graph("h", ms, [], mem, None, time_update=True)


def test_record_window_follows_item_look_back():
    d = datetime(2020, 1, 30)
    mem = {("medication", "furosemide"): ["O4e"], ("treatment", "dialysis"): ["P1a-b"]}
    recs = [{"source": "eMAR administration", "type": "medication", "text": "furosemide", "time": d - timedelta(days=10)},
            {"source": "procedure code", "type": "treatment", "text": "dialysis", "time": d - timedelta(days=10)}]
    g = A.build_graph("h", [], recs, mem, d)
    assert "O4e" not in g          # 10 days before discharge is outside the 7-day window
    assert g["P1a-b"][0]["source"] == "procedure code"   # inside the 14-day window


def test_membership_requests_offer_every_item_and_batch_by_30():
    reqs = A.member_requests([("medication", f"drug {n}") for n in range(31)])
    assert len(reqs) == 2 and len(reqs[0][2]) == 30
    assert "P1a-c" in reqs[0][0]["prompt"] and "I1a" in reqs[0][0]["prompt"]


def test_verifier_keeps_unanswered_and_retypes():
    spans = [("DRUG", 0, 3), ("ROUTE", 4, 6), ("FREQUENCY", 7, 9)]
    parsed = {"verdicts": [{"candidate": 0, "keep": True, "type": "DRUG"}, {"candidate": 1, "keep": False, "type": "ROUTE"}]}
    assert E1.apply_verdicts(spans, parsed) == [("DRUG", 0, 3), ("FREQUENCY", 7, 9)]
    ms = [m(1, "a"), m(2, "b")]
    out = A.apply_verdicts(ms, {"verdicts": [{"candidate": 1, "keep": True, "type": "medication"}]})
    assert [x["type"] for x in out] == ["condition", "medication"]


class FakeNLI:
    def __init__(self, table):
        self.table = table

    def labels(self, pairs):
        return [self.table[p] for p in pairs]


def test_nli_acceptance_by_kind():
    nli = FakeNLI({("a", "b"): "contradiction", ("b", "a"): "contradiction", ("a", "c"): "entailment", ("c", "a"): "entailment",
                   ("a", "d"): "neutral", ("d", "a"): "entailment"})
    assert perturb.accept("negate", nli, ["a"], ["b"]) == [True]
    assert perturb.accept("paraphrase", nli, ["a", "a"], ["c", "d"]) == [True, False]
    assert perturb.accept("distractor", nli, ["a"], ["d"]) == [True]   # adding a sentence: the rewrite must entail the original


def test_request_key_and_cost_accounting():
    assert request_key("m", "s", "p", {}) == request_key("m", "s", "p", {})
    assert request_key("m", "s", "p", {}) != request_key("m", "s", "q", {})
    rows = [{"input_tokens": 1_000_000, "output_tokens": 0, "seconds": 1.0}, None,
            {"input_tokens": 0, "output_tokens": 1_000_000, "seconds": 3.0}]
    c = cost(rows, "gpt-6-luna")
    assert c["calls"] == 2 and c["usd"] == 0.6 and c["median_call_seconds"] == 3.0


def test_budget_cap_stops_new_calls_without_network(tmp_path):
    from sweetspot import llm as L
    old = L.BUDGET["limit"]
    L.BUDGET["limit"] = 0.0
    try:
        client = L.LLM("budget_test", cache_dir=tmp_path)
        rows = client.run([{"name": "x", "system": "s", "prompt": "p", "schema": {"type": "object"}}])
        assert rows == [None] and client.last_failures == {"budget": 1}
    finally:
        L.BUDGET["limit"] = old
