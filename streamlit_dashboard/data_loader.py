from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import SELECTED_LABS
from paths import (
    CONDITIONS_CSV,
    DATA_DIR,
    ENCOUNTERS_CSV,
    MEDICATIONS_CSV,
    OBSERVATIONS_CSV,
    PATIENTS_CSV,
    PROCEDURES_CSV,
)


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _read_csv_chunks(csv_path, usecols: list[str], chunksize: int = 250_000) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)


def _missing_files() -> list[str]:
    cache_dir = DATA_DIR / ".cache"
    required = [
        (cache_dir / "patients.parquet", PATIENTS_CSV),
        (cache_dir / "conditions.parquet", CONDITIONS_CSV),
        (cache_dir / "encounters.parquet", ENCOUNTERS_CSV),
        (cache_dir / "medications.parquet", MEDICATIONS_CSV),
        (cache_dir / "observations_selected.parquet", OBSERVATIONS_CSV),
    ]
    return [str(p_csv) for p_cache, p_csv in required if not p_cache.exists() and not p_csv.exists()]


def load_tables_impl() -> dict:
    missing = _missing_files()
    if missing:
        raise FileNotFoundError("Missing required data files:\n" + "\n".join(missing))

    cache_dir = DATA_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Encounters
    encounters_cache = cache_dir / "encounters.parquet"
    if encounters_cache.exists():
        encounters = pd.read_parquet(encounters_cache)
    else:
        enc_cols = [
            "PATIENT",
            "REASONDESCRIPTION",
            "ENCOUNTERCLASS",
            "DESCRIPTION",
            "START",
            "STOP",
            "TOTAL_CLAIM_COST",
        ]
        encounters = _read_csv_chunks(ENCOUNTERS_CSV, usecols=enc_cols)
        encounters["START"] = _to_utc(encounters["START"])
        encounters["STOP"] = _to_utc(encounters["STOP"])
        try:
            encounters.to_parquet(encounters_cache, index=False)
        except Exception:
            pass

    # 2. Patients
    patients_cache = cache_dir / "patients.parquet"
    if patients_cache.exists():
        patients = pd.read_parquet(patients_cache)
    else:
        pat_cols = [
            "Id",
            "BIRTHDATE",
            "DEATHDATE",
            "PREFIX",
            "FIRST",
            "MIDDLE",
            "LAST",
            "MARITAL",
            "RACE",
            "ETHNICITY",
            "GENDER",
            "CITY",
            "STATE",
            "COUNTY",
            "ZIP",
            "INCOME",
            "HEALTHCARE_EXPENSES",
            "HEALTHCARE_COVERAGE",
        ]
        patients = _read_csv_chunks(PATIENTS_CSV, usecols=pat_cols)
        patients = patients.rename(columns={"Id": "PATIENT"})
        patients["BIRTHDATE"] = _to_utc(patients["BIRTHDATE"])
        patients["DEATHDATE"] = _to_utc(patients["DEATHDATE"])
        patients["DECEASED"] = patients["DEATHDATE"].notna()

        today = pd.Timestamp.now(tz="UTC")
        patients["AGE"] = np.where(
            patients["DEATHDATE"].isna(),
            (today - patients["BIRTHDATE"]).dt.days / 365.25,
            (patients["DEATHDATE"] - patients["BIRTHDATE"]).dt.days / 365.25,
        ).astype(int)
        patients["DISPLAY_NAME"] = (
            patients["PREFIX"].fillna("").astype(str).str.strip()
            + " "
            + patients["FIRST"].fillna("").astype(str)
            + " "
            + patients["LAST"].fillna("").astype(str)
        ).str.replace(r"\s+", " ", regex=True).str.strip()
        patients["SEARCH_TEXT"] = (
            patients["DISPLAY_NAME"].str.lower()
            + " "
            + patients["PATIENT"].str.lower()
            + " "
            + patients["CITY"].fillna("").str.lower()
        )
        last_encounter = encounters.groupby("PATIENT")["START"].max().rename("LAST_ENCOUNTER")
        patients = patients.merge(last_encounter, on="PATIENT", how="left")
        try:
            patients.to_parquet(patients_cache, index=False)
        except Exception:
            pass

    if "LAST_ENCOUNTER" not in patients.columns:
        last_encounter = encounters.groupby("PATIENT")["START"].max().rename("LAST_ENCOUNTER")
        patients = patients.merge(last_encounter, on="PATIENT", how="left")

    # 3. Conditions
    conditions_cache = cache_dir / "conditions.parquet"
    if conditions_cache.exists():
        conditions = pd.read_parquet(conditions_cache)
    else:
        cond_cols = ["PATIENT", "DESCRIPTION", "START", "STOP"]
        conditions = _read_csv_chunks(CONDITIONS_CSV, usecols=cond_cols)
        conditions = conditions.rename(columns={"DESCRIPTION": "CONDITION_DESC"})
        conditions["START"] = _to_utc(conditions["START"])
        conditions["STOP"] = _to_utc(conditions["STOP"])
        try:
            conditions.to_parquet(conditions_cache, index=False)
        except Exception:
            pass

    # 4. Medications
    medications_cache = cache_dir / "medications.parquet"
    if medications_cache.exists():
        medications = pd.read_parquet(medications_cache)
    else:
        med_cols = ["PATIENT", "DESCRIPTION", "START", "STOP", "REASONDESCRIPTION"]
        medications = _read_csv_chunks(MEDICATIONS_CSV, usecols=med_cols)
        medications = medications.rename(columns={"DESCRIPTION": "MEDICATION_DESC"})
        medications["START"] = _to_utc(medications["START"])
        medications["STOP"] = _to_utc(medications["STOP"])
        try:
            medications.to_parquet(medications_cache, index=False)
        except Exception:
            pass

    # 5. Observations
    obs_cache = cache_dir / "observations_selected.parquet"
    if obs_cache.exists():
        observations = pd.read_parquet(obs_cache)
    else:
        obs_chunks = []
        obs_cols = ["DATE", "PATIENT", "CATEGORY", "DESCRIPTION", "VALUE", "UNITS", "TYPE"]
        for chunk in pd.read_csv(OBSERVATIONS_CSV, usecols=obs_cols, chunksize=500_000, low_memory=False):
            filtered = chunk[chunk["DESCRIPTION"].isin(SELECTED_LABS)]
            if not filtered.empty:
                obs_chunks.append(filtered)
        if obs_chunks:
            observations = pd.concat(obs_chunks, ignore_index=True)
        else:
            observations = pd.DataFrame(columns=obs_cols)
        observations["DATE"] = _to_utc(observations["DATE"])
        observations = observations.rename(columns={"DESCRIPTION": "OBSERVATION_DESC"})
        observations["VALUE_NUM"] = pd.to_numeric(observations["VALUE"], errors="coerce")
        try:
            observations.to_parquet(obs_cache, index=False)
        except Exception:
            pass

    # 6. Procedures
    procedures_cache = cache_dir / "procedures.parquet"
    if procedures_cache.exists():
        procedures = pd.read_parquet(procedures_cache)
        procedures_available = True
    elif PROCEDURES_CSV.exists():
        procedures = pd.read_csv(PROCEDURES_CSV)
        procedures_available = True
        try:
            procedures.to_parquet(procedures_cache, index=False)
        except Exception:
            pass
    else:
        procedures = None
        procedures_available = False

    return {
        "patients": patients,
        "conditions": conditions,
        "encounters": encounters,
        "medications": medications,
        "observations": observations,
        "procedures": procedures,
        "procedures_available": procedures_available,
    }


@st.cache_resource(show_spinner="Loading Synthea tables...")
def load_tables() -> dict:
    return load_tables_impl()


def patient_slice(tables: dict, patient_id: str) -> dict:
    return {
        "patient": tables["patients"].loc[tables["patients"]["PATIENT"] == patient_id].iloc[0],
        "conditions": tables["conditions"].loc[tables["conditions"]["PATIENT"] == patient_id].copy(),
        "encounters": tables["encounters"].loc[tables["encounters"]["PATIENT"] == patient_id].copy(),
        "medications": tables["medications"].loc[tables["medications"]["PATIENT"] == patient_id].copy(),
        "observations": tables["observations"].loc[tables["observations"]["PATIENT"] == patient_id].copy(),
    }
