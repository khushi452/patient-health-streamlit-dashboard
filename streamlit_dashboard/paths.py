from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data" / "synthea" / "csv"
if not DATA_DIR.exists() and (APP_DIR / "Data" / "synthea" / "csv").exists():
    DATA_DIR = APP_DIR / "Data" / "synthea" / "csv"

MODELS_DIR = PROJECT_ROOT / "Models"
if not MODELS_DIR.exists() and (APP_DIR / "Models").exists():
    MODELS_DIR = APP_DIR / "Models"

PATIENTS_CSV = DATA_DIR / "patients.csv"
CONDITIONS_CSV = DATA_DIR / "conditions.csv"
ENCOUNTERS_CSV = DATA_DIR / "encounters.csv"
MEDICATIONS_CSV = DATA_DIR / "medications.csv"
OBSERVATIONS_CSV = DATA_DIR / "observations.csv"
PROCEDURES_CSV = DATA_DIR / "procedures.csv"

MODEL_FILES = {
    "Diabetes": MODELS_DIR / "diabetes_model.pkl",
    "Hypertension": MODELS_DIR / "hypertension_model.pkl",
    "CKD": MODELS_DIR / "ckd_model.pkl",
    "Hyperlipidemia": MODELS_DIR / "hyperlipidemia_model.pkl",
    "Anemia": MODELS_DIR / "anemia_model.pkl",
}
