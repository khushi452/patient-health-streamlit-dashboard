# 🏥 Patient Health Record (EHR) & Risk Intelligence Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.5+-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-v5.18+-purple.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> An end-to-end Machine Learning and Clinical Decision Support System (CDSS) built with Python and Streamlit. The platform analyzes longitudinal Electronic Health Records (EHR) from the **Synthea** synthetic patient dataset and predicts the 2-year onset risk of **5 major chronic conditions**: Type 2 Diabetes, Essential Hypertension, Chronic Kidney Disease (CKD), Hyperlipidemia, and Anemia.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Machine Learning Models & Metrics](#-machine-learning-models--metrics)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Dataset Details](#-dataset-details)
- [Screenshots & Visual Design](#-screenshots--visual-design)
- [Clinical Disclaimer](#-clinical-disclaimer)

---

## 🩺 Overview

Early identification of chronic disease onset allows healthcare practitioners to intervene proactively. This project translates raw, high-volume longitudinal health data into actionable risk insights through a clean, modern clinical dashboard.

The system processes demographic profiles, laboratory vitals, prescription orders, and historical diagnoses. For each patient, it determines whether conditions have already been diagnosed or calculates the 2-year future risk probability with model-level feature importance explainability and interactive real-time parameter simulation.

---

## 🌟 Key Features

### 1. 👥 Population Overview
- **Cohort Analytics**: Macro-level KPIs (total patient volume, average age, female representation, total encounters).
- **Disease Prevalence**: Horizontal breakdown of recorded chronic conditions across the cohort.
- **Demographics**: Binned age and gender distribution histogram.
- **Comorbidity Matrix**: Heatmap displaying co-occurrence patterns between chronic conditions.

### 2. 🎯 Patient Risk Intelligence
- **Clinical Demographics Card**: Patient name, unique UUID, age, gender, race, location, and date of last encounter.
- **Diagnosis Pills**: Visual badges distinguishing diagnosed chronic illnesses from active predictions.
- **Predictive Risk Cards**: Pastel-coded cards showing real-time 2-year risk percentage, progress bars, and risk tiering (`Low`, `Moderate`, `High`).
- **Vital Sign Monitor**: Current metrics (Systolic BP, HbA1c, eGFR, BMI, Heart Rate) evaluated against medical reference ranges with automated status flags (`Normal`, `Elevated`, `Subnormal`, `High Risk`, `Low Alert`).
- **Feature Explainability**: Dynamic bar chart showing top features driving the Random Forest prediction for any selected condition.
- **Interactive "What-If" Simulator**: Sliders to adjust BMI, Blood Pressure, Blood Glucose, and HbA1c to instantly simulate risk deltas in real-time.

### 3. 📜 Clinical History
- **Conditions**: Filterable list (Active vs. Resolved) with onset and resolved dates.
- **Medications**: Prescribed drug regimens, ongoing status, and documented medical reasons.
- **Encounters**: Timeline of ambulatory, outpatient, wellness, and emergency encounters with claim cost figures.
- **Longitudinal Event Timeline**: Scatter visualization tracking the chronological progression of conditions, prescriptions, and clinical visits.

### 4. 📈 Labs & Trends
- **Multi-lab Longitudinal Plot**: Interactive Plotly trend curves for tracking multiple biomarkers over time (e.g., eGFR, Blood Pressure, Glucose, Lipid profile).
- **Comprehensive Measurement Table**: Recent lab values compared directly against standard laboratory clinical reference intervals.

### 5. 📊 Model Performance Center
- Overview of Random Forest and Logistic Regression benchmark metrics across all five condition pipelines.
- Verification of model load states and training sample distributions.

---

## 🤖 Machine Learning Models & Metrics

The system uses pre-trained model bundles (`Models/*.pkl`) comprising **Random Forest Classifiers**, **Logistic Regression** baselines, **StandardScaler** pipelines, and schema-aligned feature dictionaries. Models were trained on held-out test sets using a 2-year pre-diagnosis temporal cutoff.

| Chronic Condition | Target Clinical Concept | Random Forest (ROC-AUC) | Logistic Regression (ROC-AUC) | Training Cases | Key Predictors |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Diabetes** | Type 2 Diabetes Mellitus | **0.943** | 0.916 | 400 | Glucose, HbA1c, BMI, Prediabetes History |
| **Hypertension** | Essential Hypertension | **0.919** | 0.898 | 1,246 | Systolic BP, Diastolic BP, Age, Obesity |
| **CKD** | Chronic Kidney Disease (Stages 1–4) | **0.959** | 0.954 | 522 | eGFR (MDRD), Creatinine, BUN, Diabetes History |
| **Hyperlipidemia** | Hyperlipidemia (Disorder) | **0.919** | 0.847 | 571 | Total Cholesterol, LDL, Triglycerides, BMI |
| **Anemia** | Anemia (Disorder) | **0.950** | 0.916 | 2,253 | Hemoglobin, eGFR, CKD History, Metformin |

---

## 🏗️ System Architecture

```text
[ Synthea Raw CSV Data ]
  ├── patients.csv
  ├── conditions.csv
  ├── encounters.csv
  ├── medications.csv
  └── observations.csv
            │
            ▼
[ Optimized Data Loader (data_loader.py) ]
  ├── Chunked Streaming Reader
  └── PyArrow Parquet Caching (.cache/*.parquet)
            │
            ▼
[ Feature Engineering Engine (feature_builder.py) ]
  ├── Temporal Snapshot Determination (choose_snapshot)
  ├── Sentinel Value Lab Imputation (-1 / Medians)
  ├── Diagnosis & Medication History Flags (HIST_*, HIST_MED_*)
  └── Demographic One-Hot Encodings
            │
            ▼
[ Inference Service (model_service.py) ]
  ├── Joblib Bundles (Models/*.pkl)
  ├── Random Forest + Logistic Regression Inference
  └── Dynamic What-If Feature Vector Scoring
            │
            ▼
[ Modern Streamlit Interface (app.py) ]
  ├── Population Overview
  ├── Patient Risk Intelligence & Explainability
  ├── Clinical History & Event Timelines
  ├── Longitudinal Lab Tracking
  └── Model Validation & Benchmarks
```

---

## 📂 Project Structure

```text
Patient Health Record Project/
│
├── Data/
│   └── synthea/
│       └── csv/                      # Raw Synthea EHR CSV files
│           ├── .cache/               # High-speed PyArrow Parquet cached tables
│           ├── conditions.csv
│           ├── encounters.csv
│           ├── medications.csv
│           ├── observations.csv
│           └── patients.csv
│
├── Models/                           # Serialized scikit-learn model bundles
│   ├── anemia_model.pkl
│   ├── ckd_model.pkl
│   ├── diabetes_model.pkl
│   ├── hyperlipidemia_model.pkl
│   └── hypertension_model.pkl
│
├── Notebook/                         # Model training and exploration notebook
│   └── latest patient model training.ipynb
│
├── streamlit_dashboard/              # Dashboard application source
│   ├── .streamlit/
│   │   └── config.toml               # Custom color palette and UI theme configuration
│   ├── app.py                        # Main Streamlit application and page renderers
│   ├── config.py                     # Lab ranges, colors, disease targets & constants
│   ├── data_loader.py                # Chunked data ingestion and Parquet cache management
│   ├── feature_builder.py            # Feature transformation and temporal snapshot logic
│   ├── model_service.py              # Model loading, scoring, and simulation services
│   ├── paths.py                      # Robust cross-platform path resolution
│   └── requirements.txt              # Dashboard Python dependencies
│
├── .gitignore                        # Git exclusion rules (ignores raw CSVs & caches)
└── README.md                         # Project documentation
```

---

## 💻 Tech Stack

- **Application Framework**: [Streamlit](https://streamlit.io/) (v1.32+)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [PyArrow](https://arrow.apache.org/docs/python/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/)
- **Data Visualization**: [Plotly Express](https://plotly.com/python/)
- **Dataset**: [Synthea Patient Generator](https://synthetichealth.github.io/synthea/)

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.
- Git installed.

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/patient-health-streamlit-dashboard.git
cd "Patient Health Record Project"
```

### 3. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r streamlit_dashboard/requirements.txt
```

---

## 🚀 Running the Application

Navigate to the dashboard directory and launch the Streamlit app:

```bash
cd streamlit_dashboard
streamlit run app.py
```

Or run directly from the project root:

```bash
python -m streamlit run streamlit_dashboard/app.py
```

Once started, the dashboard will automatically open in your default browser at:
```text
http://localhost:8501
```

> **Note on Initial Launch**: On the very first run, the data loader will read the raw CSV files, filter observations, and serialize them into optimized Parquet files inside `Data/synthea/csv/.cache/`. Subsequent launches will load in seconds!

---

## 📊 Dataset Details

The application uses synthetic electronic health records generated by **Synthea™**, an open-source patient generator modeling realistic health trajectories:
- **Patients**: Demographic details, birth/death dates, socio-economic factors.
- **Conditions**: SNOMED-CT clinical condition history and onset dates.
- **Encounters**: Ambulatory, emergency, inpatient, and wellness visit records.
- **Medications**: Prescribed drugs with RxNorm codes and treatment durations.
- **Observations**: LOINC vital signs, physiological measurements, and laboratory panels (HbA1c, Blood Glucose, Blood Pressure, eGFR, Lipids, Hemoglobin).

---

## 🎨 Design Philosophy & UX

- **Modern Palette**: Tailored charcoal navigation bar with pastel-coded disease cards (`#DCE8FF`, `#D5F5E3`, `#FFF1C9`, `#FBDADF`, `#E7DCFA`).
- **Responsive Components**: Streamlit container borders, rounded bar charts (`marker_cornerradius=8`), and transparent background integration.
- **Clinical Safety Guards**: Clear visual indicators (`On record` badges) ensuring existing conditions are not erroneously rescored.
- **Quick Patient Navigation**: Instant patient search by name, UUID, or city, age-group filtering, and one-click random patient sampling.

---

## ⚠️ Clinical Disclaimer

This software and its predictive models are designed strictly for **educational, academic, and research purposes**. 
- The models were trained on **synthetic data** generated by Synthea and have not been validated on real-world clinical cohorts.
- It is **not** a certified Medical Device and must **not** be used as a primary diagnostic tool, clinical guidance system, or for definitive medical decision-making.

---

## 🤝 Contributing

Contributions, issues, and feature suggestions are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
