"""Run the status-record reader and the plain yes/no comparison on the pilot notes through the OpenAI API.

Model replies hold only numbers and fixed labels, so stored runs and printed output contain no note text.
Replies are saved as they arrive, so a rerun resumes without repeating calls.
Usage: run_reader.py --smoke | --stage real|flip|repeat|support|hard|full|reveal|reveal_support
"""
import argparse
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS
from mds_pilot.known_facts import FACTS, TIERS
from mds_pilot.reader import (ARMS, HARD_VERSION, PLAIN_SCHEMA, REVEAL_SCHEMA, REVEAL_VERSION, RULES_VERSION, STATUS_SCHEMA,
                              SUPPORT_SCHEMA, SYSTEM, build_prompt, build_reveal_prompt, build_support_prompt, cited_sentences,
                              answer_is_yes, flip_sentences, hard_cases)
from mds_pilot.retrieval import lexicon_scores

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private"
RUNS = PRIVATE / "reader_runs.jsonl"
SEED = 20261101
FLIP_NOTES_PER_ITEM = 3
REPEAT_NOTES = 10
HARD_NOTES_PER_ITEM = 2
MAX_TOTAL_TOKENS = 8_000_000
WORKERS = 16
SCHEMAS = {"plain": PLAIN_SCHEMA, "status": STATUS_SCHEMA, "support": SUPPORT_SCHEMA, "reveal": REVEAL_SCHEMA}
ITEM_BY_ID = {i.id: i for i in ITEMS}
FLIP_SECTION = "Brief Hospital Course"


def load_env():
    path = ROOT / ".env"
    for line in path.read_text().splitlines() if path.exists() else []:
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


class Store:
    """Append-only record of replies, keyed so that an identical request is never sent twice."""

    def __init__(self, path):
        self.path, self.lock, self.done, self.tokens = path, threading.Lock(), {}, 0
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self.done[row["key"]] = row
                self.tokens += row["input_tokens"] + row["output_tokens"]

    def add(self, row):
        with self.lock:
            self.done[row["key"]] = row
            self.tokens += row["input_tokens"] + row["output_tokens"]
            with self.path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")


def call_model(client, model, kind, prompt, system=SYSTEM):
    reply = client.chat.completions.create(
        model=model, store=False,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": {"name": kind, "strict": True, "schema": SCHEMAS[kind]}})
    usage = reply.usage
    return json.loads(reply.choices[0].message.content), usage.prompt_tokens, usage.completion_tokens


def run_requests(requests, store, client, model, budget=MAX_TOTAL_TOKENS):
    """requests: dicts with key, kind, prompt and labels. Errors record only their type, never their message."""
    todo = [r for r in requests if r["key"] not in store.done]
    print(f"[progress] {len(requests)} requests, {len(todo)} to send, {store.tokens} tokens so far", flush=True)
    failures = {}

    def work(request):
        if store.tokens >= budget:
            failures["token_cap"] = failures.get("token_cap", 0) + 1
            return
        try:
            parsed, tokens_in, tokens_out = call_model(client, model, request["kind"], request["prompt"], request.get("system", SYSTEM))
        except Exception as error:
            name = type(error).__name__
            failures[name] = failures.get(name, 0) + 1
            return
        labels = {k: v for k, v in request.items() if k != "prompt"}
        store.add({**labels, "parsed": parsed, "input_tokens": tokens_in, "output_tokens": tokens_out})

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(work, todo))
    print(f"[progress] finished, {store.tokens} tokens in total, failures: {failures or 'none'}", flush=True)
    return failures


def load_pilot():
    sent = pd.read_parquet(PRIVATE / "pilot_sentences.parquet")
    labels = pd.read_parquet(PRIVATE / "pilot_labels.parquet").set_index("note_key")
    cands = pd.read_parquet(ROOT / "results/modal_pilot_ids/candidates_pilot.parquet")
    lookup = {(n, s): (sec, t) for n, s, sec, t in zip(sent["note_key"], sent["sent_id"], sent["section"], sent["text"])}
    picks = {key: sorted(g["sent_id"]) for key, g in cands.groupby(["note_key", "item"])}
    return sent, labels, lookup, picks


def request_key(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def prompt_rows(note, sent_ids, lookup, added=None):
    """Number sentences in reading order so that an added sentence cannot be spotted by its ID."""
    rows = [lookup[(note, s)] + (s,) for s in sent_ids]
    for k, (position, text) in reversed(list(enumerate(added or []))):  # added sentences get IDs -1, -2, ...
        rows.insert(position, (FLIP_SECTION, text, -(k + 1)))
    return [(n + 1, sec, text) for n, (sec, text, _) in enumerate(rows)], {n + 1: r[2] for n, r in enumerate(rows)}


def real_requests(picks, lookup, repeat=0, keys=None):
    requests = []
    for (note, item_id), sent_ids in picks.items():
        if keys is not None and note not in keys:
            continue
        rows, _ = prompt_rows(note, sent_ids, lookup)
        for arm in ARMS:
            requests.append({"key": request_key("real", RULES_VERSION, arm, note, item_id, repeat), "stage": "real",
                             "repeat": repeat, "arm": arm, "kind": arm, "note": note, "item": item_id,
                             "prompt": build_prompt(arm, ITEM_BY_ID[item_id].definition, rows)})
    return requests


def silent_notes(sent, labels):
    """Notes with no lexicon hit and no table flag for each item, so an added sentence is the only evidence."""
    lex = lexicon_scores(sent["text"].tolist()) > 0
    out = {}
    keys = sent["note_key"].to_numpy()
    for j, item in enumerate(ITEMS):
        hit = set(keys[lex[:, j]])
        out[item.id] = [n for n in sorted(labels.index) if n not in hit and not (item.has_proxy and labels.loc[n, item.id] == 1)]
    return out


def flip_requests(sent, labels, picks, lookup):
    rng = np.random.default_rng(SEED)
    silent = silent_notes(sent, labels)
    requests, chosen = [], {}
    for item in ITEMS:
        notes = list(rng.permutation(silent[item.id])[:FLIP_NOTES_PER_ITEM])
        chosen[item.id] = len(notes)
        facts = FACTS[item.id]
        cases = flip_sentences(item.section, item.synonyms[0], facts[TIERS.index("explicit")], facts[TIERS.index("paraphrase")])
        cases["negated"] = (facts[TIERS.index("negated")], False)
        for note in notes:
            sent_ids = picks[(note, item.id)]
            position = int(rng.integers(0, len(sent_ids) + 1))
            for case, (text, expected) in cases.items():
                rows, numbers = prompt_rows(note, sent_ids, lookup, [(position, text)])
                target = next(n for n, s in numbers.items() if s == -1)
                for arm in ARMS:
                    requests.append({"key": request_key("flip", RULES_VERSION, arm, note, item.id, case), "stage": "flip",
                                     "arm": arm, "kind": arm, "note": note, "item": item.id, "case": case,
                                     "expected_yes": expected, "target": target,
                                     "prompt": build_prompt(arm, item.definition, rows)})
    print(f"[progress] flip notes per item: min {min(chosen.values())}, max {max(chosen.values())}", flush=True)
    return requests


def hard_rows(sent, labels, picks, lookup, store):
    """Two invented sentences that must be read together, added to notes where neither arm answered yes before."""
    rng = np.random.default_rng(SEED + 1)
    silent = silent_notes(sent, labels)
    baseline_yes = {(r["note"], r["item"]) for r in store.done.values()
                    if r["stage"] == "real" and r["repeat"] == 0 and answer_is_yes(r["arm"], r["parsed"])}
    for item in ITEMS:
        usable = [n for n in silent[item.id] if (n, item.id) not in baseline_yes]
        notes = list(rng.permutation(usable)[:HARD_NOTES_PER_ITEM])
        for position_in_item, note in enumerate(notes):
            sent_ids = picks[(note, item.id)]
            middle = len(sent_ids) // 2
            first, second = int(rng.integers(0, middle + 1)), int(rng.integers(middle, len(sent_ids) + 1))
            for case, (opening, decider, expected) in hard_cases(item).items():
                rows, numbers = prompt_rows(note, sent_ids, lookup, [(first, opening), (second, decider)])
                targets = [next(n for n, s in numbers.items() if s == wanted) for wanted in (-1, -2)]
                yield item, note, position_in_item, case, expected, rows, targets


def hard_requests(sent, labels, picks, lookup, store):
    requests = []
    for item, note, _, case, expected, rows, targets in hard_rows(sent, labels, picks, lookup, store):
        for arm in ARMS:
            requests.append({"key": request_key("hard", HARD_VERSION, RULES_VERSION, arm, note, item.id, case),
                             "stage": "hard", "arm": arm, "kind": arm, "note": note, "item": item.id, "case": case,
                             "expected_yes": expected, "targets": targets,
                             "prompt": build_prompt(arm, item.definition, rows)})
    return requests


def stored_yes(store, stage, note, item_id, case=None):
    """The plain reader's stored answer, or None if there is none."""
    key = (request_key("real", RULES_VERSION, "plain", note, item_id, 0) if stage == "real"
           else request_key("hard", HARD_VERSION, RULES_VERSION, "plain", note, item_id, case))
    row = store.done.get(key)
    return None if row is None else row["parsed"]["answer"] == "yes"


def reveal_request(item, note, rows, group, blind, shown, **labels):
    return {"key": request_key("reveal", REVEAL_VERSION, group, note, item.id, labels.get("case")), "stage": "reveal",
            "arm": "reveal", "kind": "reveal", "note": note, "item": item.id, "group": group, "blind_yes": blind,
            "shown_table": shown, "prompt": build_reveal_prompt(item, rows, blind, shown), **labels}


def reveal_requests(sent, labels, picks, lookup, store):
    """Real disagreements, matched false-table controls on real notes, and known answers shown a false table."""
    rng = np.random.default_rng(SEED + 2)
    requests, unmatched = [], 0
    for item in [i for i in ITEMS if i.has_proxy]:
        pairs = [(n, stored_yes(store, "real", n, item.id), bool(labels.loc[n, item.id])) for n in sorted(labels.index)]
        disagree = [(n, b, t) for n, b, t in pairs if b != t]
        pools = {b: list(rng.permutation([n for n, x, t in pairs if x == b and t == b])) for b in (True, False)}
        for note, blind, table in disagree:
            rows, _ = prompt_rows(note, picks[(note, item.id)], lookup)
            requests.append(reveal_request(item, note, rows, "real", blind, table))
            if not pools[blind]:
                unmatched += 1
                continue
            control = str(pools[blind].pop())
            rows, _ = prompt_rows(control, picks[(control, item.id)], lookup)
            requests.append(reveal_request(item, control, rows, "false_table", blind, not blind))
    for item, note, position, case, expected, rows, _ in hard_rows(sent, labels, picks, lookup, store):
        if position > 0 or not item.has_proxy:
            continue
        blind = stored_yes(store, "hard", note, item.id, case)
        if blind == expected:  # only correct first answers can show a correct answer being abandoned
            requests.append(reveal_request(item, note, rows, "known", blind, not expected, case=case, expected_yes=expected))
    counts = pd.Series([r["group"] for r in requests]).value_counts().to_dict()
    print(f"[progress] reveal groups: {counts}, disagreements with no control: {unmatched}", flush=True)
    return requests


def reveal_support_requests(store, picks, lookup):
    """Read the cited sentences of each changed-to-yes answer alone, without the first answer or the table."""
    requests = []
    for row in list(store.done.values()):
        if row["stage"] != "reveal" or row["group"] == "known" or row["blind_yes"] or row["parsed"]["answer"] != "yes":
            continue
        sent_ids = picks[(row["note"], row["item"])]
        cited = sorted({n for n in row["parsed"]["evidence"] if 1 <= n <= len(sent_ids)})
        rows = [(i + 1, *lookup[(row["note"], sent_ids[n - 1])]) for i, n in enumerate(cited)]
        requests.append({"key": request_key("reveal_support", REVEAL_VERSION, row["group"], row["note"], row["item"]),
                         "stage": "reveal_support", "arm": "reveal", "kind": "support", "group": row["group"], "note": row["note"],
                         "item": row["item"], "cited": len(cited), "prompt": build_support_prompt(ITEM_BY_ID[row["item"]].definition, rows)})
    return requests


def repeat_notes(labels):
    return set(pd.Series(sorted(labels.index)).sample(REPEAT_NOTES, random_state=SEED))


def full_requests(sent, labels, picks, lookup):
    """Plain reader on every sentence of the rerun notes, whose two top-20 runs give a measure of chance variation."""
    keys = repeat_notes(labels)
    all_ids = {n: sorted(g["sent_id"]) for n, g in sent[sent["note_key"].isin(keys)].groupby("note_key")}
    requests = []
    for note, sent_ids in all_ids.items():
        rows, numbers = prompt_rows(note, sent_ids, lookup)
        for item in ITEMS:
            candidates = set(picks[(note, item.id)])
            requests.append({"key": request_key("full", RULES_VERSION, note, item.id), "stage": "full", "arm": "plain",
                             "kind": "plain", "note": note, "item": item.id, "sentences": len(rows),
                             "candidate_numbers": [n for n, s in numbers.items() if s in candidates],
                             "prompt": build_prompt("plain", item.definition, rows)})
    return requests


def support_requests(store, picks, lookup):
    requests = []
    for row in list(store.done.values()):
        if row["stage"] != "real" or row["repeat"] != 0 or not answer_is_yes(row["arm"], row["parsed"]):
            continue
        sent_ids = picks[(row["note"], row["item"])]
        cited = sorted({n for n in cited_sentences(row["arm"], row["parsed"]) if 1 <= n <= len(sent_ids)})
        if not cited:
            continue
        rows = [(i + 1, *lookup[(row["note"], sent_ids[n - 1])]) for i, n in enumerate(cited)]
        requests.append({"key": request_key("support", RULES_VERSION, row["arm"], row["note"], row["item"]), "stage": "support",
                         "arm": row["arm"], "kind": "support", "note": row["note"], "item": row["item"],
                         "prompt": build_support_prompt(ITEM_BY_ID[row["item"]].definition, rows)})
    return requests


def smoke(client, model):
    """Invented sentences only. Confirms model access, the output schemas and token use."""
    rows = [(1, "Family History", "The patient's mother had diabetes."),
            (2, "Discharge Medications", "Metformin 500 mg twice daily."),
            (3, "Past Medical History", "Type 2 diabetes mellitus, on oral medication.")]
    for arm in ARMS:
        parsed, tokens_in, tokens_out = call_model(client, model, arm, build_prompt(arm, "patient has diabetes mellitus", rows))
        print(arm, json.dumps(parsed), "answer_yes", answer_is_yes(arm, parsed), "tokens", tokens_in, tokens_out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage", choices=["real", "flip", "repeat", "support", "hard", "full", "reveal", "reveal_support"])
    args = ap.parse_args()
    load_env()
    from openai import OpenAI
    client, model = OpenAI(max_retries=5, timeout=180), os.environ["MDS_LLM_MODEL"]
    if args.smoke:
        return smoke(client, model)
    sent, labels, lookup, picks = load_pilot()
    store = Store(RUNS)
    if args.stage == "real":
        requests = real_requests(picks, lookup)
    elif args.stage == "flip":
        requests = flip_requests(sent, labels, picks, lookup)
    elif args.stage == "repeat":
        requests = real_requests(picks, lookup, repeat=1, keys=repeat_notes(labels))
    elif args.stage == "hard":
        requests = hard_requests(sent, labels, picks, lookup, store)
    elif args.stage == "full":
        requests = full_requests(sent, labels, picks, lookup)
    elif args.stage == "reveal":
        requests = reveal_requests(sent, labels, picks, lookup, store)
    elif args.stage == "reveal_support":
        requests = reveal_support_requests(store, picks, lookup)
    else:
        requests = support_requests(store, picks, lookup)
    failures = run_requests(requests, store, client, model)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
