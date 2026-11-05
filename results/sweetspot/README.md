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

# Labelled form-filling benchmark: CL4Health 2026 CRF (E4 to E7)

134-item dyspnoea CRF on emergency-department notes; official per-note macro-F1 (our transcription matches the official scorer on
dev). Development picks; each test split is scored a fixed number of times (English test: 2 looks, E4 and E5; Italian: 1).
Leaderboard placements compare different models on the same test set and metric, so they are indicative only.

| English test (200 notes) | Macro-F1 [95% CI] | Correct on documented items | USD per note | Median s per note |
| --- | --- | --- | --- | --- |
| Direct whole-form prompt | **0.663** [0.631, 0.693] | 0.741 | **0.0005** | **4.9** |
| Graph + audit, kept where the reader agrees (E5, picked on dev) | 0.635 [0.601, 0.664] | 0.731 | 0.0100 | about 35 |
| Graph + consistency audit (E4 headline, picked on dev) | 0.607 [0.574, 0.637] | 0.758 | 0.0030 | 35 |
| Graph + path verifier | 0.587 [0.554, 0.618] | 0.649 | 0.0026 | |
| Graph | 0.578 [0.544, 0.609] | 0.658 | 0.0024 | 28 |
| Graph without verifier | 0.571 | 0.659 | 0.0019 | 22 |
| Graph without status record in the fill step | 0.572 | 0.657 | 0.0023 | 28 |
| Graph with encoder membership | 0.522 | 0.534 | 0.0021 | 31 |
| Per-item reader (134 calls per note) | 0.521 | **0.818** | 0.0070 | 4.4 |
| All unknown | 0.407 | 0 | 0 | 0 |

Published English leaderboard: Aurum 0.683, DocUA 0.628, Gladiators 0.597, MdL 0.571 (best local model), organisers' baseline
0.470. Direct would place 2nd and graph + audit 3rd.

Ablation (English test, paired 95% CI): model membership vs encoder +0.055 [0.028, 0.082]; consistency audit +0.030 [0.011, 0.048];
span verifier +0.007 [-0.001, 0.016]; status record in the fill step +0.005 [-0.003, 0.015]; path verifier -0.020 [-0.038, -0.003];
graph-reader agreement +0.027 [0.016, 0.038]; direct minus best graph +0.056 [0.037, 0.075].

Why the graph loses here: notes are short single documents with no records, so its strengths (records, long notes, look-back
windows) do not apply, while extraction and membership can drop information before the form is filled. The reader is the most
accurate on documented items but fills too many items that should stay unknown, which macro-F1 penalises.

Selective prediction (E5): values all three methods agree on are 82% of fills at precision 0.63; values only two agree on are 18% at
precision 0.17. Agreement is a strong routing signal for reviewer triage.

Label-free battery on a form task (E6, 50 test notes, six arms): stability ranks the arms like labelled macro-F1 with Kendall tau
0.73 pooled (paraphrase 0.73, family-history distractor 0.60, line reordering 0.47). Weaker than on NegEx (1.0); reordering is the
least trustworthy test.

Generalisation with unchanged code (E7): Italian test, direct 0.669 [0.637, 0.700] and graph + audit 0.622 [0.592, 0.652] (Italian
leaderboard: Aurum 0.672, Polimi 0.635, DocUA 0.570). MDS 3.0 item set on the same 100 MIMIC admissions: 9.9 items per admission;
the same clinical answer as MDS 2.0 on 23 mapped items with mean agreement 0.974 (minimum 0.86, where the two forms define the
items differently: non-invasive ventilation, Alzheimer's versus other dementia).

# E8 smoke tests (CRF dev split only; no test look)

The OpenAI credits ran out at the start of E8: every new call returned 429 "no credits remaining", so the paid ideas (form-guided
NER, graph-guided direct, form-blind summary then fill) were **not run** and their rows in `e8_smoke.json` are flagged `not_run`.
The item-scope call also failed, so the status gate (Z3) ran with scope "any" for every item. Everything below used cached
outputs and local models only (BGE-small encoder, DeBERTa NLI, GLiNER-BioMed).

Hard gates on every filled value (`e8_smoke.json`, 80 notes) cut false positives but remove far more correct values, and all lose:
| Gate on direct (dev macro-F1 0.712) | Macro-F1 | Values set where gold is unknown | Correct on documented items |
| --- | --- | --- | --- |
| none | 0.712 | 164 | 0.759 |
| Z1 encoder relevance, top 20 items | 0.604 | 83 | 0.446 |
| Z2 NLI abstention verifier | 0.615 | 95 | 0.449 |
| Z4 GLiNER consistency | 0.596 | 66 | 0.351 |
The same pattern holds for graph and graph + audit (losses of 0.08 to 0.16).

Checks applied only where two methods disagree (`e8b_disagreement_gate.py`, 80 notes, no API):
| Arm | Macro-F1 | Minus direct [95% CI] |
| --- | --- | --- |
| direct | 0.712 | |
| agree(direct, graph + audit) | 0.713 | +0.001 [-0.008, 0.010] |
| direct, disagreements with graph + audit checked by NLI | 0.715 | +0.004 [0.001, 0.007] |
| direct, disagreements with the reader checked by NLI | 0.716 | +0.004 [0.001, 0.009] |
| direct, disagreements with the reader checked by GLiNER | 0.719 | +0.007 [-0.002, 0.017] |
| graph + audit, disagreements with direct checked by GLiNER | 0.704 | -0.008 [-0.027, 0.007] (from -0.023) |

Reading: encoder-only checks are useful as tie-breakers where methods disagree, not as gates. Gains on top of direct are small
(+0.004 to +0.007); a GLiNER tie-break closes about two thirds of graph + audit's gap to direct. These are dev-split smoke results.

# E9: the E8 ideas on an open model (Qwen3.8-27B on a Colab G4, CRF dev split, 20 notes)

Same 20 development notes as E8's paid part, thinking off, greedy decoding, schema in the prompt (no replies failed to parse).
Every model arm, including the direct baseline, ran on Qwen, so ideas are compared with each other; scores are not comparable with
gpt-6-luna (direct 0.707 on these notes). Small sample, about 40 variants tried, wide intervals: these are hypotheses for a larger
run, not results. No test look.

| Arm (Qwen, 20 dev notes) | Macro-F1 | Values set where gold is unknown | Correct on documented items |
| --- | --- | --- | --- |
| direct | 0.586 | 462 | 0.838 |
| graph | 0.551 | 70 | 0.486 |
| graph + status gate by item scope (Z3) | 0.590 | 57 | 0.486 |
| graph + audit | 0.442 | 586 | 0.802 |
| form-guided NER, then fill (A1) | 0.550 | 129 | 0.784 |
| graph-guided direct (A2) | 0.560 | 490 | 0.820 |
| form-blind summary, then fill (A3) | 0.579 | 377 | 0.631 |
| direct, disagreements with graph + audit checked by NLI | **0.657** (+0.070 [0.019, 0.137]) | 272 | 0.820 |
| form-guided NER, disagreements with direct checked by GLiNER | **0.664** (+0.077 [-0.003, 0.167]) | 84 | 0.748 |

Reading, with that caveat:
- Qwen over-fills badly when asked for the whole form (462 values where gold is unknown, against 57 for gpt-6-luna), so everything
  that removes unsupported values pays off more than it did on gpt-6-luna. The consistency audit, which helped gpt-6-luna, makes
  Qwen's over-filling worse (586), so the audit's value depends on the model.
- Form-guided NER fixes the graph's recall gap: correct on documented items 0.784 against 0.486 for the graph, with 129 rather
  than 462 wrong fills. Its F1 equals the graph's here, but with a disagreement check it is the best arm (0.664).
- Encoder, NLI and GLiNER checks applied to every value still lose (as in E8). Applied only where a second method disagrees they
  give +0.05 to +0.08 on the two over-filling bases, in line with E8's small gain on gpt-6-luna.
- Status gate with item scopes from one model call: +0.039 over the graph (0.551 to 0.590) at no loss in documented items.
- Graph-guided direct and summary-then-fill did not beat direct (A2 -0.026, A3 -0.007), both within noise.

# E10: full CRF benchmark and ablation on Qwen3.8-27B (vLLM on a Colab G4; dev 80 notes picks, English test 200 notes scored once)

One open model for every arm, thinking off, greedy decoding, JSON schema enforced by the server. Scores are not comparable with the
gpt-6-luna tables (direct 0.663 there). This is the third look at the English test split; all arms count as one look. The per-item
reader was not run (26,800 test calls; it was the weaker method on gpt-6-luna). Test wall time 62 minutes for 200 notes at 64
concurrent requests (about 1,000 generated tokens per second). Generated tokens per note count each arm's own calls plus the arms it
is built from, except graph-guided direct (A2), whose graph evidence tokens (about 2,650) are not added to its 420.

| English test, Qwen3.8-27B | Macro-F1 [95% CI] | vs direct [95% CI] | Values set where gold is unknown | Correct on documented items | Items set per note | Generated tokens per note |
| --- | --- | --- | --- | --- | --- | --- |
| agree(form-guided NER, direct), post hoc | **0.620** [0.588, 0.651] | +0.059 [0.036, 0.083] | 461 | 0.673 | 5.8 | 2,177 |
| agree(graph-guided direct, direct), post hoc | 0.614 | +0.054 [0.031, 0.076] | 1,483 | 0.682 | 11.0 | 953 |
| agree(direct, graph + audit), post hoc | 0.611 | +0.051 [0.030, 0.072] | 967 | 0.702 | 8.5 | 3,561 |
| tie-break(direct, disagreements with graph + audit checked by NLI) | 0.606 | +0.045 [0.032, 0.059] | 1,272 | 0.745 | 10.3 | 3,561 |
| **headline picked on dev:** tie-break(form-guided NER, disagreements with direct checked by GLiNER) | 0.590 | +0.029 [0.003, 0.056] | 618 | 0.696 | 6.7 | 2,177 |
| direct (whole form, one call) | 0.561 [0.522, 0.600] | | 3,030 | 0.768 | 19.2 | 533 |
| graph | 0.551 [0.515, 0.588] | -0.009 [-0.045, 0.026] | 430 | 0.421 | 4.6 | 2,653 |
| graph + audit | 0.549 | -0.012 [-0.039, 0.016] | 2,031 | 0.743 | 14.2 | 3,028 |
| graph + status gate by item scope | 0.547 | -0.014 [-0.052, 0.023] | 264 | 0.357 | 3.4 | 2,653 |
| graph-guided direct (A2) | 0.580 | +0.019 [-0.009, 0.048] | 2,345 | 0.716 | 15.6 | 420 + graph |
| form-blind summary, then fill (A3) | 0.528 | -0.032 [-0.065, 0.000] | 4,252 | 0.662 | 25.0 | 2,126 |
| form-guided NER, then fill (A1) | 0.513 | -0.048 [-0.077, -0.017] | 1,468 | 0.760 | 11.4 | 1,644 |
| graph with encoder membership | 0.362 | -0.199 [-0.233, -0.164] | 2,217 | 0.484 | 13.8 | 2,726 |

Ablation of the graph (test, same model): model-decided membership over encoder membership +0.189; span verifier -0.008 and status
record in the fill step +0.005 (both within noise); consistency audit -0.002 (it raises documented items 0.42 to 0.74 but adds 1,600
wrong fills); status gate by item scope -0.005 (115 items current, 19 history).

Reading:
- Qwen over-fills badly when asked for the whole form: 19 values per note against about 5 in the gold. Everything that
  removes unsupported values beats it. The two best families on test were agreement between two methods and checking only the
  values where two methods disagree: +0.03 to +0.06 over direct, with every interval above 0.
- The dev-picked headline kept its gain on test but shrunk: dev +0.059, test +0.029. The best test arms (plain agreement) were
  not eligible for the dev pick by the rules fixed in advance, so their test ranking is a post-hoc observation to confirm.
- Results from the 20-note E9 run that did not replicate on 280 notes: the status gate (+0.039 then, -0.005 now) and form-guided
  NER on its own (equal to the graph then, below direct now). Small-sample smoke tests overstated both.
- Same direction on both models: a second method that disagrees with the first is the strongest signal for what to check or hold
  back (gpt-6-luna: graph-reader agreement +0.027; Qwen: +0.05 to +0.06).

# E11 and E11b: soft weighting of independent signals (cached gpt-6-luna outputs, CRF dev, 80 notes; no API, no test look)

Signals per candidate value: share of the four methods that give it, soft encoder relevance of the item, NLI entailment probability,
GLiNER score, and the graph's status share. The accept threshold was the only tuned quantity; 5-fold cross-validation over notes.
Direct is 0.712 on this split.

| Design | Dev macro-F1 | 5-fold CV | Folds beating direct |
| --- | --- | --- | --- |
| votes + status | 0.717 | 0.717 | 3 of 5 |
| votes only | 0.713 | 0.713 | 2 of 5 |
| votes + NER | 0.680 | 0.680 | 1 of 5 |
| votes + encoder relevance | 0.660 | 0.653 | 1 of 5 |
| votes + NLI | 0.615 | 0.584 | 0 of 5 |
| all five signals, equal-weight mean everywhere | 0.625 | 0.625 | 0 of 5 |
| E11b: accept on agreement, or on the mean of the four soft signals (best of 52 settings) | 0.687 | 0.687 | 1 of 5 |

No design met the stability rule (at least 4 of 5 folds beating direct and a wide threshold plateau), in either run. Leave-one-out
in the all-signal design: removing votes costs 0.050 and removing status costs 0.041; removing the encoder relevance gains 0.034 and
removing NLI gains 0.025. In E11b the best setting uses the soft part as little as the grid allows (agreement 0.75, threshold 0.8),
and every design with a soft part is below votes alone.

Conclusion on gpt-6-luna: direct is already well calibrated here (164 wrong fills on dev), so the free signals have little to add;
soft weighting does not help, and the equal-weight mean dilutes the useful signals. The gains seen with Qwen (E10) came from a model
that over-fills (19 values per note against about 5 in the gold).
