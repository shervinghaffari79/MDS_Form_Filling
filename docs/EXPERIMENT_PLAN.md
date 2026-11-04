# Finding the sweet spot: component comparison plan (frozen before any run)

Goal: fill MDS Sections I (diagnoses), medications and special treatments from a MIMIC-IV discharge summary plus hospital records,
with evidence paths a nurse can check. Choose components by accuracy on labelled open data first, then by label-free robustness on
MIMIC, always alongside token cost and filling delay.

Constraints set by the project owner:
- No predefined rules or hand lists. What counts as a diagnosis, drug class or treatment is decided by the language model from the
  MDS item wording alone. Rule-based tools (ConText, lexicons) appear only as reference baselines, never in a chosen pipeline.
- Training-free is preferred. Pretrained models used as they are (LLM, GLiNER, NLI, sentence encoders) are allowed; fitting
  parameters on labels (CRF, logistic graph) is a reference baseline only.
- Generalisation: one prompt per component, written from type or item definitions, reused unchanged across datasets. No examples
  drawn from the test data. Any per-dataset change is reported as such.

## Form version (open question for the owner)
The brief names MDS 3.0 Sections I, N, O. The uploaded form and field reference are MDS 2.0 Canadian (CCRS), where diagnoses are
Section I (I1 diseases, I2 infections), medications are Section O (O4 days received: antipsychotic, antianxiety, antidepressant,
hypnotic, diuretic, analgesic) and special treatments are Section P (P1a). The item set is a configuration file; Phase 1 does not
depend on it. Phase 2 uses whichever version the owner confirms (default: both, reported separately).

## Pipeline under test
1. Entity layer: find condition, drug and treatment mentions (NER), then a verifier accepts or rejects each one with its sentence.
2. Status layer: for each mention, subject (patient or other), assertion (present, negated, possible) and time (current, past,
   planned) — the "status record".
3. Admission graph: one connected component per admission. Nodes are concepts; edges come from note mentions (with status and
   section), and from records (diagnosis codes, prescriptions, eMAR, procedures, ICU events). Time updating: later dated evidence
   supersedes earlier evidence for the same concept.
4. Item layer: concept to MDS item membership decided by the model from the item definition (cached per concept, so cost grows with
   distinct concepts, not with notes). An item is answered from the graph by a parameter-free rule over paths (below), and the
   paths themselves are the explanation.

## Phase 1: labelled open datasets
| Exp | Component | Data (labels) | Arms | Primary metric |
| --- | --- | --- | --- | --- |
| E1 | Entity layer | Glaucoma medication lines (drug, route, frequency, use, stop); MACCROBAT case reports (disease, medication, procedure, sign) | (a) LLM zero-shot NER from type definitions; (b) (a) + LLM verifier; (c) GLiNER-BioMed zero-shot; (d) (c) + LLM verifier; reference: saved CRF+POS and few-shot LLM | lenient span F1 per type, pooled core types |
| E2 | Status layer + robustness battery | NegEx/ConText 2,376 mentions (negation, temporality, experiencer) | status record, direct prompt; reference ConText rules | accuracy on perturbed inputs whose correct label is known |
| E3 | Graph decision | CodiEsp-D (ICD-10 codes, 250 dev / 250 test) | training-free path fusion vs saved logistic graph, same edges | MAP and F1 |

E2 is the bridge to Phase 2. Each perturbation is applied to labelled mentions, where the correct change is known:
- negate the mention (LLM rewrite, checked by NLI as a contradiction): answer must flip;
- meaning-preserving paraphrase (LLM rewrite, checked by NLI as two-way entailment): answer must not change;
- reorder clauses or neighbouring sentences: must not change;
- insert a distractor about another person or another condition: must not change;
- mask the mention (remove the concept words): the mention must no longer be found.
If the label-free scores (flip and invariance rates) rank the arms the same way as labelled accuracy does, they are used on MIMIC.

## Phase 2: MIMIC-IV, label-free
On the 500-admission proxy sample (rebuilt; see results/sample_summary.json):
- Evidence masking (ERASER comprehensiveness and sufficiency): remove the cited evidence, the item should turn off; keep only the
  cited evidence, it should stay on.
- Negation injection on cited evidence: should turn off.
- Section and sentence reordering, paraphrase (NLI-checked), distractor insertion (family history, "ruled out" sentences): no change.
- Repeat stability (same input, sampled twice) and self-consistency as a confidence signal.
- Agreement with records not used by the arm under test (for example, note-only arms against eMAR and procedure tables), reported
  as agreement, not accuracy, because the records are incomplete and share sources.
Literature basis: LAPD flip rate under masking, negation, synonym and numeric edits (arXiv 2507.21188); NLI-verified
meaning-preserving variation sensitivity and worst-case instability (arXiv 2605.30646); metamorphic paraphrase and fact reordering
(arXiv 2603.13173, 2606.07237); causal faithfulness by concept ablation (arXiv 2609.32817); single-sentence misattribution and
self-consistency confidence (arXiv 2610.20417); per-fact assertion and temporality in a patient graph on MIMIC-IV (arXiv 2605.11143);
ERASER comprehensiveness and sufficiency (DeYoung et al., ACL 2020).

Phase 2 steps (rules in each script's docstring, written before its run):
1. `p2_mimic_graph.py`: pilot of 100 proxy admissions, MDS 2.0 items; arms notes, notes+v, notes+v+t, records, graph.
2. `p2_battery.py`: the battery on 30 of those admissions (up to 3 filled and 2 unfilled items each): mask, sufficiency, negate,
   paraphrase, reorder, family-history and ruled-out distractors.
3. `p2_reader_baseline.py`: the item-by-item reader on 20 admissions, with retrieval by encoder similarity to the item wording
   (the earlier lexicon retrieval is a hand list and is not used), for cost, delay and agreement against the graph.

## Measured for every arm
- Quality: the primary metric above, with 95% CI from paired resampling (notes, lines or mentions).
- Cost: input and output tokens per item and per admission; dollars at the list price of the model used
  (gpt-6-luna, USD 0.10 / 0.50 per million input / output tokens, standard tier, November 2026).
- Delay: wall-clock seconds per admission at the stated concurrency, and median and 95th-percentile seconds per call.
- Robustness: the battery above.

## Decision rules (fixed now)
- D1 (verifier): the verifier is kept if it raises lenient precision with a CI above 0 and lowers F1 by at most 0.01.
- D2 (entity backbone): GLiNER + verifier replaces LLM NER + verifier if its core F1 is within 0.03 (lower CI bound of the
  difference above -0.03) at lower cost.
- D3 (training-free graph): dev picks the parameter-free fusion; it is used if its test MAP is within 0.05 of the logistic graph
  (lower CI bound of the difference above -0.05). Amended before the run: the logistic graph was fit on dev, so its dev score is
  in-sample and the comparison is made on test, counted as the 7th look at that split.
- D4 (battery validity): a label-free score is used on MIMIC only if, on NegEx, it orders the arms the same way as labelled accuracy.
- Sweet spot: among arms that pass D1 to D4, the Pareto front over (quality or robustness, dollars per admission, seconds per
  admission); the default pick is the cheapest arm whose robustness is within the CI of the best.
- Every look at a test split is counted and reported.
