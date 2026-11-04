# Phase 1 results: component comparison on labelled open data

Plan and frozen decision rules: `docs/EXPERIMENT_PLAN.md`. Model: gpt-6-luna (USD 0.10 / 0.50 per million input / output tokens).
Raw scores: the JSON files in this folder. Total model spend for Phase 1: about USD 0.55.

## E1 — entity layer (NER + verifier), `e1_entity_layer.json`
Core lenient span F1 (glaucoma: drug, route, frequency on 400 seeded test lines; MACCROBAT: disease, medication, therapeutic
procedure on 40 seeded reports). Same prompt design on both sets; only each set's own type definitions differ.

| Arm | Glaucoma F1 | MACCROBAT F1 | Glaucoma USD / 400 lines | Median seconds per call |
| --- | --- | --- | --- | --- |
| Model NER, zero-shot | 0.881 | 0.502 | 0.028 | 1.5 |
| Model NER + verifier | **0.924** | 0.508 | 0.043 | 1.5 + 1.9 |
| GLiNER-BioMed (CPU) | 0.528 | 0.498 | 0 | 0.15 s per line, 20 s per report |
| GLiNER + verifier | 0.844 | **0.561** | 0.031 | 2.1 |

- D1 (keep the verifier) holds on both sets: glaucoma precision +0.077 [0.047, 0.110], F1 +0.042 [0.024, 0.063].
- D2 (switch to GLiNER) fails: on glaucoma GLiNER + verifier is 0.080 lower [-0.115, -0.044]. It is better on MACCROBAT (+0.053,
  CI crosses 0). Phase 2 therefore uses model NER + verifier.
- Reference (trained, not candidates; all 1,646 glaucoma test lines): CRF+POS 0.938, few-shot model 0.945.
- MACCROBAT is low for every arm. Its guideline annotates fragments ("tumor", "drained") that zero-shot definitions do not
  describe; strict F1 is about 0.3. This is the cost of not tuning to a dataset's convention, and it is reported, not fixed.
- GLiNER reads about 384 tokens, so whole reports were likely truncated; on MIMIC it would need sentence chunking.

## E2 — status layer and the robustness battery, `e2_negex_battery.json`
360 NegEx mentions (stratified, seed 13). Perturbations written by the model and kept only when an NLI model confirms the intended
relation (negate 264, paraphrase 335, reorder 336, distractor 67 accepted). Accuracy = all three labels correct against the known
expected labels; consistency = agreement with the arm's own answer on the original, which needs no labels.

| Arm | Original | Negate | Paraphrase | Reorder | Distractor | Pooled accuracy | Pooled consistency | USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct prompt | 0.861 | 0.826 | 0.875 | 0.896 | 0.672 | **0.855** | **0.948** | 0.079 |
| Status record | 0.853 | 0.746 | 0.848 | 0.884 | 0.806 | 0.830 | 0.918 | 0.128 |
| ConText rules (reference) | 0.817 | 0.636 | 0.758 | 0.774 | 0.791 | 0.734 | 0.865 | 0 |

- D4 holds: the label-free consistency score orders the arms exactly as labelled accuracy does (Kendall tau 1.0 for negate,
  paraphrase, reorder and pooled; 0.33 for the distractor). So it is used on MIMIC, with the distractor read with care.
- Attribution is the weak spot of the direct prompt: with a family-history sentence added, it drops to 0.672, while the status
  record holds at 0.806. Negation flips are the weak spot of the status record (0.746).
- Only 67 of 360 distractor rewrites passed the NLI check (most were judged neutral), so that cell is small.

## E3 — training-free graph decision on CodiEsp, `e3_codiesp_training_free.json`
Same edges as the learned graph; reciprocal rank fusion instead of fitted weights; picked on dev, scored on test (7th test look).

| Test (250 cases) | MAP | F1 | Precision | Recall |
| --- | --- | --- | --- | --- |
| Logistic graph (fit on dev) | 0.619 | 0.629 | 0.776 | 0.529 |
| Training-free fusion (T2 + A1) | 0.585 | 0.483 | 0.813 | 0.343 |
| Model paths only, no CodiEsp training data (T3) | 0.412 | 0.483 | | |

- D3 holds for ranking: -0.034 MAP [-0.042, -0.026], still 2nd of 22 teams. Acceptance by "two independent evidence paths" is
  precise but misses two thirds of codes, so F1 falls by 0.15 (8th of 22).
- Generalisation: most of the score comes from evidence phrases memorised from CodiEsp's own training cases. With model paths
  alone MAP is 0.41; MIMIC has no such memory, so expect behaviour closer to that line.

# Phase 2 results: MIMIC-IV, MDS 2.0 items, no labels

Rebuilt 500-admission proxy sample (skilled-nursing discharges); 78 MDS 2.0 items (I1, I2, O4, P1a). Aggregates only.
Model spend for Phase 2: about USD 5.3 (pilot 1.3, graph battery 3.85, reader 0.16, reader battery 0.08).

## Pilot, 100 admissions (`p2_mimic_graph.json`) and reader baseline, 20 admissions (`p2_reader_baseline.json`)
| Per admission | Item-by-item reader | Graph: notes | Graph: notes + verifier | Graph: notes + verifier + records |
| --- | --- | --- | --- | --- |
| Items filled | 13.5 | 10.6 | 10.6 | 13.0 |
| USD (membership cached) | 0.0079 | 0.0083 | 0.0114 | 0.0114 |
| Delay, median seconds | 7.7 (16 workers) | 33 | 52 | 52 (+ record lookup) |
| Explanation | cited sentences | paths: sentence, status, item | same | paths incl. dated records |

- Reader and graph agree with Cohen's kappa 0.73; most disagreements are items the reader fills and the graph does not (86 vs 21).
- Notes and records agree on 781 item fills; 274 are note-only (histories rarely coded, e.g. arthritis) and 249 records-only
  (e.g. IV medication, antianxiety drugs). Agreement, not accuracy: neither is a gold standard.
- Time updating by reading order changes almost nothing (10.55 to 10.50 items per admission).
- Cost and delay of the graph are dominated by output tokens (about 3,200 per 60-sentence chunk, 27 s per call).

## Robustness battery (`p2_battery.json`, `p2_reader_battery.json`)
Each method is perturbed on its own cited evidence; share of cases where the expected outcome holds, 95% CI over admissions.

| Test | Expected | Reader | Graph (notes + verifier) |
| --- | --- | --- | --- |
| Family-history distractor | stays off | 1.00 | 1.00 |
| Ruled-out distractor | stays off | 1.00 | 1.00 |
| Reorder sections | stays on | 0.95 [0.88, 1.00] | 0.97 [0.93, 1.00] |
| Paraphrase evidence | stays on | 0.92 [0.82, 1.00] | 0.94 [0.87, 1.00] |
| Negate evidence | turns off | 0.89 [0.72, 1.00] | 0.90 [0.81, 0.98] |
| Mask cited evidence (comprehensiveness) | turns off | **0.60 [0.47, 0.72]** | **0.84 [0.78, 0.91]** |
| Cited evidence alone (sufficiency) | stays on | **0.98 [0.95, 1.00]** | **0.81 [0.71, 0.90]** |

- The two methods tie on distractors, reordering, paraphrase and negation.
- The graph's explanations are more complete: removing everything it cites turns the item off 84% of the time, against 60% for
  the reader, whose citations often leave other supporting sentences in the note.
- The reader's citations are more self-contained: they re-create the answer alone 98% of the time, against 81% for the graph,
  whose single-sentence mentions sometimes need neighbouring context.
- The verifier and time updating make no measurable difference on any test.

## Sweet spot
- Drop the verifier and time updating on MIMIC: no measurable gain on fills or robustness, and they cost USD 0.003 and about 19 s
  per admission. (The verifier did raise span precision on labelled glaucoma lines; at item level on MIMIC it does not matter.)
- Recommended default: the graph with notes (no verifier) plus records. About USD 0.008 per admission, close to the reader, with
  records integrated, dated evidence paths, and the more complete explanations on the masking test. Its cost is delay: about
  33 s per admission against 8 s.
- If delay matters more than explanation completeness, the item-by-item reader is the better choice on its own notes, but it does
  not use the records.
- Open: neither method's accuracy is known without signed MDS forms; the look-back windows are counted from hospital discharge,
  not from a nursing-home assessment date.
