# MDS automation project: handoff to the cloud environment

Goal: draft MDS 3.0 Sections I, N and O from hospital discharge summaries plus hospital records, with cited evidence, for a nurse to review.
There are no signed MDS forms, so evidence comes from (a) open datasets with human labels and (b) label-free tests on MIMIC-IV.
This bundle holds the code, the tests and text-free result summaries. It holds no MIMIC data, no note text and no keys.

## Rules the work must keep (do not break these)
1. **No hand-written rules or lists.** The user wants learned or language-model-decided behaviour. Do not write code-to-MDS-item rules,
   drug lists or glaucoma medication rules. Class membership is decided by the language model from the MDS item definitions alone
   (`test_class_membership.py`), and links are learned from data (`modal_open_bench.py`, `codiesp_graph_step`). `mds_pilot/items.py`
   holds an older hand-written lexicon that the project is moving away from; use it only as a baseline.
2. **Freeze before you score.** Every script states its decision rules in its docstring. Choose on a development split, freeze, score the
   test split once, and say when a test set has been looked at more than once.
3. **Report intervals.** Differences use a 95% CI from resampling notes or admissions. Say plainly when a rule fails.
4. **Text-free outputs.** Result files hold counts and scores, never note text or admission identifiers.
5. **Secrets.** Set `OPENAI_API_KEY` and `MDS_LLM_MODEL` as environment secrets. Never write them to a file or print them.
6. **MIMIC.** Do not upload MIMIC data here unless the user confirms their data agreement allows it. Their stated authorisation covered
   OpenAI and Modal. The open benchmarks below need no agreement.

## Where each method lives
| Method | Files |
| --- | --- |
| Note reader (retrieve top 20 sentences, answer yes/no with cited evidence) | `mds_pilot/segment.py`, `mds_pilot/retrieval.py`, `mds_pilot/reader.py`, `run_reader.py`, `run_pipeline.py` |
| Status record (subject, assertion, timing per mention) | `mds_pilot/reader.py` (`build_prompt("status", ...)`, `derive_state`) |
| Knowledge graph / learned graph | `mds_pilot/kg.py`, `build_kg.py`, `benchmark_kg.py`, `test_learned_graph.py` (MIMIC); `modal_open_bench.py::codiesp_graph_step` (CodiEsp) |
| Look-back window | `test_window_graph.py`, `test_window_o.py` |
| Trained extractor (CRF + POS) and verifier | `mds_pilot/ner_features.py`, `modal_ner_bench.py` (glaucoma), `modal_mimic_ner.py`, `build_mimic_ner_extracts.py` (MIMIC) |
| Linking drug attributes | `modal_ner_bench.py` (`linking`, `link_scores`) |
| Class membership (list, ATC, language model) | `test_class_membership.py`, `analyze_mimic_extractor.py` |
| Open benchmarks (NegEx, CodiEsp, glaucoma) | `modal_open_bench.py`, `modal_ner_bench.py`, `test_negex_gated.py` |
| Robustness and label-free battery | `test_llm_timeline_robustness.py`, `test_kg_rules.py`, `run_self_supervised.py`, `test_proxy_real.py` |
| Tests | `tests/` |

The `modal_*.py` scripts are ordinary Python wrapped in Modal decorators. Without a Modal token, call the plain functions (loaders,
scorers, `score_all`, `link_scores`, graph fitting) directly; only the model-calling and encoder steps need compute.

## Open data (public, fetch with the scripts or from the source)
- CodiEsp (Zenodo record 3837305, CC BY 4.0): English translation, diagnosis codes. ICD-10-CM 2020 descriptions come from CMS.
- NegEx/ConText test set: `genConText/rsAnnotations-1-120-random.txt` in the `chapmanbe/negex` GitHub repo (7 tab-separated columns).
- Glaucoma medication set (BSD-3): `jche253/Glaucoma_Med_Dataset`, `Annotations/Combined_annotations.jsonl`. 1,276 lines have an id redacted
  to `*****` (replace it) and 9 lines have a stray backslash (drop it; that keeps span offsets aligned). All 5,520 lines then parse.
  The annotations are spans only, so drug-attribute links cannot be scored on real lines.

## Results to reproduce (targets)
Open datasets, human labels:
| Test | Result |
| --- | --- |
| NegEx negation F1 / timing macro F1: status record, direct prompt, ConText rules | 0.976 / 0.820; 0.956 / 0.758; 0.929 / 0.798 |
| Glaucoma core F1 (drug, route, frequency): CRF+POS, few-shot model, CRF + verifier on three types | 0.938; 0.945; 0.948 (all-types 0.934) |
| Linking accuracy: nearest, parse, model (constructed pairs) | 0.921; 0.942; 0.997 |
| CodiEsp-D test: free coding; two-stage with learned phrase index; learned graph | MAP 0.206, F1 0.371; MAP 0.294, F1 0.453; MAP 0.618, F1 0.628 (places 14th, 9th for the first two by MAP/F1; graph 1st by MAP, 5th by F1, of 22) |

Glaucoma numbers were computed on 5,511 readable lines. The loader now reads all 5,520, so the test split shifts slightly and a rerun may differ in the third decimal.

MIMIC-IV, label-free (Section N at discharge, 244 admissions, against the nurse medication log): reader 0.844, rule-based graph 0.885,
model with prescription timeline 0.903, records active at discharge 0.954, learned graph 0.958 (held-out half, precision 0.900). Class
membership by the language model replaces hand lists at -0.005; WHO ATC prefixes cost -0.024. Summaries are under `results/`.

## Open problems
- No signed MDS forms, so true accuracy is unknown. Nurse-log and prescription keys share a source.
- Section O: the IV edge says yes for 369 of 500 stays; oxygen and tracheostomy cannot be dated from orders or codes.
- Section I: held-out checks exist only for UTI and diabetes.
- Answers cover the hospital stay, not the nursing-home look-back window, for 29 items.
- Removing or reversing cited sentences changes the answer only 63-74% of the time.
- i2b2 2008 and n2c2 2018 were not run (they need data agreements).

## Replies to the questions in your last message
- **Where is the method code?** In this bundle (table above). Please use it instead of reimplementing; reimplementations drift.
- **The three methods:** note reader, status record and learned graph, as listed. The knowledge graph is a logistic model over per-source
  edges (reader, status mention in window, negated mention, record active at discharge), not a rule table.
- **Code-to-MDS-item and glaucoma medication rules:** do not write them (rule 1). Use the language model for class membership and learned
  phrase-to-code links for CodiEsp.
- **Persistence:** the container is temporary. Save work to a repository or storage the user names. Nothing here is committed or pushed.
