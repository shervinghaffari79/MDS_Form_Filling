"""Pilot item definitions. Lexicons and proxies are pilot drafts, not reviewed clinical rules."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    id: str
    section: str
    definition: str
    synonyms: tuple
    icd_dx: tuple = field(default=())
    icd_proc: tuple = field(default=())
    rx: tuple = field(default=())

    @property
    def has_proxy(self):
        return bool(self.icd_dx or self.icd_proc or self.rx)


def _drugs(*names):
    return tuple(names)


ITEMS = [
    Item("I_heart_failure", "I", "patient has heart failure",
         ("heart failure", "congestive heart failure", "CHF", "HFrEF", "HFpEF", "systolic dysfunction", "diastolic dysfunction", "cardiomyopathy"),
         icd_dx=("428", "I50")),
    Item("I_diabetes", "I", "patient has diabetes mellitus",
         ("diabetes", "diabetes mellitus", "diabetic", "DM2", "T2DM", "IDDM", "NIDDM", "DMII"),
         icd_dx=("250", "E08", "E09", "E10", "E11", "E13")),
    Item("I_ckd", "I", "patient has chronic kidney disease",
         ("chronic kidney disease", "CKD", "chronic renal insufficiency", "chronic renal failure", "ESRD", "end stage renal disease"),
         icd_dx=("585", "N18")),
    Item("I_copd", "I", "patient has chronic obstructive pulmonary disease",
         ("COPD", "chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis"),
         icd_dx=("491", "492", "496", "J41", "J42", "J43", "J44")),
    Item("I_dementia", "I", "patient has dementia or Alzheimer disease",
         ("dementia", "Alzheimer", "Alzheimer's", "vascular dementia", "Lewy body", "major neurocognitive disorder"),
         icd_dx=("2900", "2901", "2902", "2903", "2904", "2941", "3310", "F01", "F02", "F03", "G30")),
    Item("I_depression", "I", "patient has depression",
         ("depression", "depressed mood", "major depressive disorder", "MDD"),
         icd_dx=("2962", "2963", "311", "F32", "F33")),
    Item("I_pneumonia", "I", "patient has pneumonia",
         ("pneumonia", "PNA", "HCAP", "CAP", "aspiration pneumonia", "VAP", "pulmonary infiltrate"),
         icd_dx=("480", "481", "482", "483", "484", "485", "486", "J12", "J13", "J14", "J15", "J16", "J17", "J18")),
    Item("I_uti", "I", "patient has a urinary tract infection",
         ("urinary tract infection", "UTI", "cystitis", "pyelonephritis", "CAUTI", "urosepsis"),
         icd_dx=("5990", "N390")),
    Item("I_stroke", "I", "patient had a stroke or cerebrovascular accident",
         ("stroke", "CVA", "cerebrovascular accident", "cerebral infarction", "intracerebral hemorrhage", "ischemic infarct", "ICH", "SAH"),
         icd_dx=("430", "431", "4331", "4341", "436", "I60", "I61", "I63")),
    Item("I_afib", "I", "patient has atrial fibrillation",
         ("atrial fibrillation", "afib", "a-fib", "AF", "AFib", "atrial flutter"),
         icd_dx=("42731", "42732", "I48")),

    Item("N_antipsychotic", "N", "patient is taking an antipsychotic medication",
         ("antipsychotic",) + _drugs("haloperidol", "Haldol", "quetiapine", "Seroquel", "olanzapine", "Zyprexa", "risperidone", "Risperdal", "aripiprazole", "Abilify", "ziprasidone", "clozapine", "chlorpromazine", "paliperidone", "lurasidone"),
         rx=("haloperidol", "quetiapine", "olanzapine", "risperidone", "aripiprazole", "ziprasidone", "clozapine", "chlorpromazine", "paliperidone", "lurasidone")),
    Item("N_antianxiety", "N", "patient is taking an antianxiety medication",
         ("antianxiety", "anxiolytic", "benzodiazepine") + _drugs("lorazepam", "Ativan", "diazepam", "Valium", "alprazolam", "Xanax", "clonazepam", "Klonopin", "buspirone", "chlordiazepoxide", "oxazepam"),
         rx=("lorazepam", "diazepam", "alprazolam", "clonazepam", "buspirone", "chlordiazepoxide", "oxazepam")),
    Item("N_antidepressant", "N", "patient is taking an antidepressant medication",
         ("antidepressant", "SSRI", "SNRI") + _drugs("sertraline", "Zoloft", "citalopram", "Celexa", "escitalopram", "Lexapro", "fluoxetine", "Prozac", "paroxetine", "venlafaxine", "duloxetine", "Cymbalta", "bupropion", "mirtazapine", "Remeron", "trazodone", "amitriptyline", "nortriptyline"),
         rx=("sertraline", "citalopram", "escitalopram", "fluoxetine", "paroxetine", "venlafaxine", "duloxetine", "bupropion", "mirtazapine", "trazodone", "amitriptyline", "nortriptyline")),
    Item("N_hypnotic", "N", "patient is taking a hypnotic sleep medication",
         ("hypnotic", "sleep aid") + _drugs("zolpidem", "Ambien", "temazepam", "Restoril", "eszopiclone", "Lunesta", "zaleplon", "ramelteon", "triazolam"),
         rx=("zolpidem", "temazepam", "eszopiclone", "zaleplon", "ramelteon", "triazolam")),
    Item("N_anticoagulant", "N", "patient is taking an anticoagulant medication",
         ("anticoagulant", "anticoagulation") + _drugs("warfarin", "Coumadin", "heparin", "enoxaparin", "Lovenox", "apixaban", "Eliquis", "rivaroxaban", "Xarelto", "dabigatran", "Pradaxa", "fondaparinux", "argatroban", "bivalirudin"),
         rx=("warfarin", "heparin", "enoxaparin", "apixaban", "rivaroxaban", "dabigatran", "fondaparinux", "argatroban", "bivalirudin")),
    Item("N_antibiotic", "N", "patient is taking an antibiotic medication",
         ("antibiotic", "antibiotics", "abx") + _drugs("vancomycin", "cefepime", "ceftriaxone", "cefazolin", "cephalexin", "piperacillin", "Zosyn", "meropenem", "ciprofloxacin", "Cipro", "levofloxacin", "Levaquin", "metronidazole", "Flagyl", "azithromycin", "amoxicillin", "Augmentin", "doxycycline", "linezolid", "daptomycin", "nitrofurantoin", "Macrobid", "trimethoprim", "Bactrim", "clindamycin"),
         rx=("vancomycin", "cefepime", "ceftriaxone", "cefazolin", "cephalexin", "piperacillin", "meropenem", "ciprofloxacin", "levofloxacin", "metronidazole", "azithromycin", "amoxicillin", "doxycycline", "linezolid", "daptomycin", "nitrofurantoin", "trimethoprim", "sulfameth", "clindamycin")),
    Item("N_diuretic", "N", "patient is taking a diuretic medication",
         ("diuretic", "diuresis") + _drugs("furosemide", "Lasix", "torsemide", "bumetanide", "Bumex", "hydrochlorothiazide", "HCTZ", "chlorthalidone", "spironolactone", "metolazone", "acetazolamide"),
         rx=("furosemide", "torsemide", "bumetanide", "hydrochlorothiazide", "chlorthalidone", "spironolactone", "metolazone", "acetazolamide")),
    Item("N_opioid", "N", "patient is taking an opioid medication",
         ("opioid", "narcotic") + _drugs("morphine", "oxycodone", "Percocet", "hydromorphone", "Dilaudid", "fentanyl", "hydrocodone", "tramadol", "methadone", "codeine"),
         rx=("morphine", "oxycodone", "hydromorphone", "fentanyl", "hydrocodone", "tramadol", "methadone", "codeine")),
    Item("N_antiplatelet", "N", "patient is taking an antiplatelet medication",
         ("antiplatelet", "ASA") + _drugs("aspirin", "clopidogrel", "Plavix", "ticagrelor", "Brilinta", "prasugrel", "dipyridamole"),
         rx=("aspirin", "clopidogrel", "ticagrelor", "prasugrel", "dipyridamole")),
    Item("N_hypoglycemic", "N", "patient is taking a hypoglycemic medication for blood sugar",
         ("hypoglycemic", "antidiabetic", "sliding scale") + _drugs("insulin", "glargine", "Lantus", "lispro", "Humalog", "metformin", "glipizide", "glyburide", "glimepiride", "sitagliptin", "pioglitazone", "empagliflozin", "liraglutide"),
         rx=("insulin", "metformin", "glipizide", "glyburide", "glimepiride", "sitagliptin", "pioglitazone", "empagliflozin", "liraglutide")),
    Item("N_anticonvulsant", "N", "patient is taking an anticonvulsant medication",
         ("anticonvulsant", "antiepileptic", "AED") + _drugs("levetiracetam", "Keppra", "phenytoin", "Dilantin", "valproate", "valproic", "Depakote", "lamotrigine", "carbamazepine", "gabapentin", "Neurontin", "pregabalin", "Lyrica", "lacosamide", "topiramate", "oxcarbazepine"),
         rx=("levetiracetam", "phenytoin", "valpro", "divalproex", "lamotrigine", "carbamazepine", "gabapentin", "pregabalin", "lacosamide", "topiramate", "oxcarbazepine")),

    Item("O_oxygen", "O", "patient received supplemental oxygen therapy",
         ("oxygen", "supplemental oxygen", "O2", "nasal cannula", "NC", "NRB", "non-rebreather", "face mask", "home oxygen", "L NC")),
    Item("O_invasive_vent", "O", "patient received invasive mechanical ventilation",
         ("mechanical ventilation", "intubated", "intubation", "extubated", "extubation", "ventilator", "ETT", "endotracheal tube"),
         icd_proc=("967", "5A1935Z", "5A1945Z", "5A1955Z")),
    Item("O_noninvasive_vent", "O", "patient received non-invasive ventilation",
         ("non-invasive ventilation", "noninvasive ventilation", "BiPAP", "CPAP", "NIV"),
         icd_proc=("9390", "9391", "5A09")),
    Item("O_dialysis", "O", "patient received dialysis",
         ("dialysis", "hemodialysis", "HD", "CRRT", "CVVH", "CVVHD", "peritoneal dialysis", "renal replacement therapy"),
         icd_proc=("3995", "5498", "5A1D")),
    Item("O_iv_medication", "O", "patient received intravenous medication",
         ("intravenous", "IV", "IV medication", "IV antibiotics", "IV fluids", "drip", "infusion")),
    Item("O_transfusion", "O", "patient received a blood transfusion",
         ("transfusion", "transfused", "pRBC", "PRBC", "packed red blood cells", "FFP", "fresh frozen plasma", "platelets transfused"),
         icd_proc=("9903", "9904", "9905", "9906", "9907", "3023")),
    Item("O_chemotherapy", "O", "patient received chemotherapy",
         ("chemotherapy", "chemo", "cycle of", "carboplatin", "cisplatin", "paclitaxel", "rituximab", "methotrexate"),
         icd_dx=("V5811", "Z5111"), icd_proc=("9925", "3E03305", "3E04305", "3E05305", "3E06305")),
    Item("O_tracheostomy", "O", "patient has a tracheostomy requiring care",
         ("tracheostomy", "trach", "trach collar", "tracheotomy", "PMV"),
         icd_dx=("V440", "Z930"), icd_proc=("311", "3129", "0B110")),
]

ITEM_IDS = [i.id for i in ITEMS]
