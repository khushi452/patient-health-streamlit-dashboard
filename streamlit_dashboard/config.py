"""Constants taken from the existing training notebook."""

COLORS = {
    "blue": "#1E4DB7",
    "gold": "#E8A017",
    "lavender": "#C5D3F7",
    "navy": "#163A8C",
    "bg": "#F4F6FB",
    "card": "#FFFFFF",
    "text": "#1B2545",
    "muted": "#5C6B8A",
    "sidebar": "#14327A",
    "danger": "#C0392B",
    "ok": "#2E7D4F",
}

PLOTLY_SEQUENCE = ["#1E4DB7", "#E8A017", "#C5D3F7", "#163A8C", "#F0C25A", "#8FA6E6"]

TARGETS_FINAL = {
    "Diabetes": ["Diabetes mellitus type 2 (disorder)"],
    "Hypertension": ["Essential hypertension (disorder)"],
    "CKD": [
        "Chronic kidney disease stage 1 (disorder)",
        "Chronic kidney disease stage 2 (disorder)",
        "Chronic kidney disease stage 3 (disorder)",
        "Chronic kidney disease stage 4 (disorder)",
    ],
    "Hyperlipidemia": ["Hyperlipidemia (disorder)"],
    "Anemia": ["Anemia (disorder)"],
}

SELECTED_LABS = [
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Body Weight",
    "Body Height",
    "Body mass index (BMI) [Ratio]",
    "Heart rate",
    "Respiratory rate",
    "Glomerular filtration rate [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)/1.73 sq M",
    "Creatinine [Mass/volume] in Blood",
    "Urea nitrogen [Mass/volume] in Blood",
    "Microalbumin/Creatinine [Mass Ratio] in Urine",
    "Hemoglobin A1c/Hemoglobin.total in Blood",
    "Glucose [Mass/volume] in Blood",
    "Cholesterol [Mass/volume] in Serum or Plasma",
    "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay",
    "Cholesterol in HDL [Mass/volume] in Serum or Plasma",
    "Triglyceride [Mass/volume] in Serum or Plasma",
    "Hemoglobin [Mass/volume] in Blood",
]

HISTORY_FEATURES = {
    "Diabetes": [
        "Prediabetes (finding)",
        "Body mass index 30+ - obesity (finding)",
        "Metabolic syndrome X (disorder)",
        "Hyperglycemia (disorder)",
    ],
    "Hypertension": [
        "Body mass index 30+ - obesity (finding)",
        "Metabolic syndrome X (disorder)",
    ],
    "CKD": [
        "Essential hypertension (disorder)",
        "Diabetes mellitus type 2 (disorder)",
        "Disorder of kidney due to diabetes mellitus (disorder)",
        "Microalbuminuria due to type 2 diabetes mellitus (disorder)",
    ],
    "Hyperlipidemia": [
        "Body mass index 30+ - obesity (finding)",
        "Metabolic syndrome X (disorder)",
        "Ischemic heart disease (disorder)",
    ],
    "Anemia": [
        "Chronic kidney disease stage 1 (disorder)",
        "Chronic kidney disease stage 2 (disorder)",
        "Chronic kidney disease stage 3 (disorder)",
        "Chronic kidney disease stage 4 (disorder)",
    ],
}

MEDICATION_FEATURES = {
    "Diabetes": [],
    "Hypertension": [],
    "CKD": ["metformin", "insulin", "lisinopril", "losartan", "amlodipine", "hydrochlorothiazide"],
    "Hyperlipidemia": [],
    "Anemia": ["metformin", "insulin"],
}

SENTINEL_LABS = {
    "Diabetes": [
        "Glucose [Mass/volume] in Blood",
        "Hemoglobin A1c/Hemoglobin.total in Blood",
    ],
    "Hypertension": [],
    "CKD": [
        "Glomerular filtration rate [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)/1.73 sq M",
        "Creatinine [Mass/volume] in Blood",
    ],
    "Hyperlipidemia": [],
    "Anemia": ["Hemoglobin [Mass/volume] in Blood"],
}

NOTEBOOK_METRICS = {
    "Diabetes": {"logreg_auc": 0.916, "rf_auc": 0.943, "cases": 400},
    "Hypertension": {"logreg_auc": 0.898, "rf_auc": 0.919, "cases": 1246},
    "CKD": {"logreg_auc": 0.954, "rf_auc": 0.959, "cases": 522},
    "Hyperlipidemia": {"logreg_auc": 0.847, "rf_auc": 0.919, "cases": 571},
    "Anemia": {"logreg_auc": 0.916, "rf_auc": 0.950, "cases": 2253},
}

LAB_SHORT_NAMES = {
    "Systolic Blood Pressure": "Systolic BP",
    "Diastolic Blood Pressure": "Diastolic BP",
    "Body Weight": "Weight",
    "Body Height": "Height",
    "Body mass index (BMI) [Ratio]": "BMI",
    "Heart rate": "Heart rate",
    "Respiratory rate": "Respiratory rate",
    "Glomerular filtration rate [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)/1.73 sq M": "eGFR (MDRD)",
    "Creatinine [Mass/volume] in Blood": "Creatinine",
    "Urea nitrogen [Mass/volume] in Blood": "BUN",
    "Microalbumin/Creatinine [Mass Ratio] in Urine": "UACR",
    "Hemoglobin A1c/Hemoglobin.total in Blood": "HbA1c",
    "Glucose [Mass/volume] in Blood": "Glucose",
    "Cholesterol [Mass/volume] in Serum or Plasma": "Total cholesterol",
    "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay": "LDL",
    "Cholesterol in HDL [Mass/volume] in Serum or Plasma": "HDL",
    "Triglyceride [Mass/volume] in Serum or Plasma": "Triglycerides",
    "Hemoglobin [Mass/volume] in Blood": "Hemoglobin",
}

VITAL_LABS = [
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart rate",
    "Respiratory rate",
    "Body mass index (BMI) [Ratio]",
    "Body Weight",
]

LAB_REFERENCE_RANGES = {
    "Systolic Blood Pressure": {"min": 90, "max": 120, "unit": "mmHg", "critical_high": 140},
    "Diastolic Blood Pressure": {"min": 60, "max": 80, "unit": "mmHg", "critical_high": 90},
    "Heart rate": {"min": 60, "max": 100, "unit": "/min", "critical_high": 120, "critical_low": 50},
    "Respiratory rate": {"min": 12, "max": 20, "unit": "/min", "critical_high": 25},
    "Body mass index (BMI) [Ratio]": {"min": 18.5, "max": 24.9, "unit": "kg/m2", "critical_high": 30.0},
    "Glucose [Mass/volume] in Blood": {"min": 70, "max": 99, "unit": "mg/dL", "critical_high": 126},
    "Hemoglobin A1c/Hemoglobin.total in Blood": {"min": 4.0, "max": 5.6, "unit": "%", "critical_high": 6.5},
    "Glomerular filtration rate [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)/1.73 sq M": {"min": 90, "max": 150, "unit": "mL/min/1.73m2", "critical_low": 60},
    "Creatinine [Mass/volume] in Blood": {"min": 0.6, "max": 1.2, "unit": "mg/dL", "critical_high": 1.5},
    "Urea nitrogen [Mass/volume] in Blood": {"min": 7, "max": 20, "unit": "mg/dL", "critical_high": 25},
    "Cholesterol [Mass/volume] in Serum or Plasma": {"min": 125, "max": 200, "unit": "mg/dL", "critical_high": 240},
    "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay": {"min": 50, "max": 100, "unit": "mg/dL", "critical_high": 160},
    "Cholesterol in HDL [Mass/volume] in Serum or Plasma": {"min": 40, "max": 80, "unit": "mg/dL", "critical_low": 40},
    "Triglyceride [Mass/volume] in Serum or Plasma": {"min": 50, "max": 150, "unit": "mg/dL", "critical_high": 200},
    "Hemoglobin [Mass/volume] in Blood": {"min": 12.0, "max": 17.5, "unit": "g/dL", "critical_low": 11.0},
    "Microalbumin/Creatinine [Mass Ratio] in Urine": {"min": 0, "max": 30, "unit": "mg/g", "critical_high": 300},
}


def get_lab_status(lab_name: str, value: float) -> tuple[str, str]:
    """Returns (status_text, color_name) where color is 'ok', 'danger', 'gold', or 'blue'."""
    ref = LAB_REFERENCE_RANGES.get(lab_name)
    if not ref or value is None:
        return ("Normal", "ok")
    try:
        val = float(value)
    except (ValueError, TypeError):
        return ("Recorded", "blue")
    if "critical_high" in ref and val >= ref["critical_high"]:
        return ("High Risk", "danger")
    if "critical_low" in ref and val < ref["critical_low"]:
        return ("Low Alert", "danger")
    if "max" in ref and val > ref["max"]:
        return ("Elevated", "gold")
    if "min" in ref and val < ref["min"]:
        return ("Subnormal", "blue")
    return ("Normal", "ok")

