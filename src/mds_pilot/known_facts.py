"""Invented evidence sentences with labels known by construction. Contains no MIMIC text.

Tiers: explicit uses a lexicon term; variant uses an abbreviation or brand name; paraphrase avoids
every lexicon term for the item; negated rules the item out, and retrieval should still surface it.
"""

FACTS = {
    "I_heart_failure": ("Admitted with decompensated congestive heart failure.", "Known CHF, EF 25%.", "Weak pumping left ventricle with EF of 20% and leg swelling.", "No history of heart failure."),
    "I_diabetes": ("Long-standing diabetes mellitus with neuropathy.", "T2DM, A1c 8.4.", "Elevated A1c of 9.2 on a home blood sugar regimen.", "No history of diabetes."),
    "I_ckd": ("Chronic kidney disease stage 4 at baseline.", "CKD with baseline Cr 2.8.", "Baseline creatinine 3.1 with longstanding renal impairment.", "No chronic kidney disease."),
    "I_copd": ("Chronic obstructive pulmonary disease on home inhalers.", "COPD exacerbation treated with nebulizers.", "Obstructive lung disease with FEV1 45% predicted.", "Denies any history of COPD."),
    "I_dementia": ("Advanced dementia, oriented to self only.", "Alzheimer's, followed by geriatrics.", "Progressive memory loss with severe cognitive decline.", "No known dementia."),
    "I_depression": ("Depression followed by outpatient psychiatry.", "MDD, stable on home regimen.", "Low mood and anhedonia managed with counseling.", "Denies depression."),
    "I_pneumonia": ("Treated for aspiration pneumonia.", "Found to have PNA on chest x-ray.", "Right lower lobe consolidation treated as a lung infection.", "No evidence of pneumonia on imaging."),
    "I_uti": ("Treated for urinary tract infection.", "UTI on admission.", "Urine culture grew E. coli with dysuria and pyuria.", "No evidence of urinary tract infection."),
    "I_stroke": ("Prior ischemic stroke with residual weakness.", "History of CVA last year.", "Left MCA territory infarct with right sided weakness.", "Head CT showed no acute stroke."),
    "I_afib": ("Atrial fibrillation, rate controlled.", "New afib with RVR on telemetry.", "Irregularly irregular rhythm on telemetry, on metoprolol for rate.", "No atrial fibrillation on telemetry."),
    "N_antipsychotic": ("Continue quetiapine 25 mg nightly.", "Seroquel 50 mg PO QHS.", "Continue pimozide 2 mg nightly for psychosis.", "Haloperidol was not given."),
    "N_antianxiety": ("Continue lorazepam 0.5 mg as needed.", "Ativan 1 mg PO PRN.", "Hydroxyzine 25 mg as needed for anxiety.", "Lorazepam was not given."),
    "N_antidepressant": ("Continue sertraline 50 mg daily.", "Zoloft 100 mg PO DAILY.", "Vilazodone 20 mg daily for mood.", "Sertraline was not given."),
    "N_hypnotic": ("Zolpidem 5 mg at bedtime as needed.", "Ambien 5 mg PO QHS PRN.", "Suvorexant 10 mg at bedtime for insomnia.", "Zolpidem was not given."),
    "N_anticoagulant": ("Continue warfarin 5 mg daily.", "Lovenox 40 mg SC DAILY.", "Edoxaban 60 mg daily for clot prevention.", "Warfarin was held and not resumed."),
    "N_antibiotic": ("Complete a course of ceftriaxone 1 g daily.", "Zosyn 4.5 g IV Q8H.", "Ertapenem 1 g daily for seven more days for infection.", "Vancomycin was not given."),
    "N_diuretic": ("Continue furosemide 40 mg daily.", "Lasix 20 mg PO DAILY.", "Amiloride 5 mg daily for fluid retention.", "Furosemide was not given."),
    "N_opioid": ("Oxycodone 5 mg every 6 hours as needed.", "Percocet 1 tab PO Q6H PRN.", "Tapentadol 50 mg every 6 hours for severe pain.", "Oxycodone was not given."),
    "N_antiplatelet": ("Continue aspirin 81 mg daily.", "Plavix 75 mg PO DAILY.", "Cilostazol 100 mg twice daily to prevent clotting.", "Aspirin was not given."),
    "N_hypoglycemic": ("Insulin glargine 20 units at bedtime.", "Lantus 10 units SC QHS.", "Dulaglutide 1.5 mg weekly for sugar control.", "Insulin was not given."),
    "N_anticonvulsant": ("Continue levetiracetam 500 mg twice daily.", "Keppra 1000 mg PO BID.", "Zonisamide 100 mg daily for seizure prevention.", "Levetiracetam was not given."),
    "O_oxygen": ("Continue supplemental oxygen as needed.", "On 2L NC with sats above 92%.", "Requiring 2 liters by prongs to keep saturations above 92%.", "Did not require oxygen."),
    "O_invasive_vent": ("Required mechanical ventilation in the ICU.", "Intubated for airway protection.", "Remained on assist control through a breathing tube.", "Was not intubated."),
    "O_noninvasive_vent": ("Treated with non-invasive ventilation overnight.", "BiPAP overnight for hypercapnia.", "Pressure support by tight mask overnight for hypercapnia.", "Did not require BiPAP."),
    "O_dialysis": ("Continues on hemodialysis three times weekly.", "CRRT was started for anuria.", "Seen by nephrology for thrice weekly sessions through a tunneled catheter.", "Did not require dialysis."),
    "O_iv_medication": ("Treated with intravenous antibiotics.", "Given IV diltiazem.", "Medication was given by vein through a peripheral line.", "No intravenous medication was required."),
    "O_transfusion": ("Received a blood transfusion for anemia.", "Given 2 units pRBC.", "Received two units of red cells for hemoglobin of 6.8.", "Did not require transfusion."),
    "O_chemotherapy": ("Received chemotherapy with oncology.", "Last chemo was two weeks ago.", "Completed the third round of FOLFOX with oncology.", "Chemotherapy was deferred and not given."),
    "O_tracheostomy": ("Tracheostomy care was provided daily.", "Trach collar overnight.", "Neck stoma with inner cannula changed daily.", "Does not have a tracheostomy."),
}
TIERS = ("explicit", "variant", "paraphrase", "negated")
