from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    HISTORY_FEATURES,
    MEDICATION_FEATURES,
    SELECTED_LABS,
    SENTINEL_LABS,
    TARGETS_FINAL,
)


def tested_flag_name(lab: str) -> str:
    return "TESTED_" + lab.split(" [")[0].replace(" ", "_").replace("(", "").replace(")", "")


def hist_flag_name(condition_desc: str) -> str:
    return f"HIST_{condition_desc.split(' (')[0].replace(' ', '_')}"


def recorded_diagnoses(conditions: pd.DataFrame) -> dict[str, dict]:
    out = {}
    if conditions is None or conditions.empty:
        for disease, names in TARGETS_FINAL.items():
            out[disease] = {"recorded": False, "first_date": None, "labels": []}
        return out
    for disease, names in TARGETS_FINAL.items():
        hits = conditions[conditions["CONDITION_DESC"].isin(names)].sort_values("START")
        out[disease] = {
            "recorded": not hits.empty,
            "first_date": None if hits.empty else hits["START"].iloc[0],
            "labels": hits["CONDITION_DESC"].dropna().unique().tolist(),
        }
    return out


def choose_snapshot(patient_row: pd.Series, encounters: pd.DataFrame, observations: pd.DataFrame) -> pd.Timestamp:
    candidates = []
    last_enc = patient_row.get("LAST_ENCOUNTER")
    if pd.notna(last_enc):
        candidates.append(pd.Timestamp(last_enc))
    if encounters is not None and not encounters.empty:
        candidates.append(encounters["START"].max())
    if observations is not None and not observations.empty:
        candidates.append(observations["DATE"].max())
    if not candidates:
        return pd.Timestamp.now(tz="UTC")
    snap = max(candidates)
    if snap.tzinfo is None:
        snap = snap.tz_localize("UTC")
    return snap


def build_patient_features(
    disease: str,
    patient_row: pd.Series,
    conditions: pd.DataFrame,
    medications: pd.DataFrame,
    encounters: pd.DataFrame,
    observations: pd.DataFrame,
    feature_columns: list[str],
    lab_medians: dict,
    snapshot: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    """Build one aligned feature row using the notebook's feature logic.

    Snapshot is the patient's last clinical date, not the training 2-year
    pre-diagnosis cut. Unused notebook history terms that the saved model
    never learned (for example HIST_Hyperglycemia) are not invented.
    """
    notes = {
        "snapshot": snapshot,
        "missing_required": [],
        "unused_history_terms": [],
    }

    birth = patient_row["BIRTHDATE"]
    if pd.isna(birth):
        notes["missing_required"].append("BIRTHDATE")
        return pd.DataFrame(), notes

    age_at_snapshot = int((snapshot - birth).days / 365.25)
    row: dict = {"AGE_AT_SNAPSHOT": age_at_snapshot}

    labs_needed = [c for c in feature_columns if c in SELECTED_LABS]
    obs = observations.copy() if observations is not None else pd.DataFrame()
    if not obs.empty:
        obs = obs[obs["DATE"] < snapshot]
        obs = obs[obs["OBSERVATION_DESC"].isin(labs_needed)]
        latest = (
            obs.sort_values("DATE")
            .groupby("OBSERVATION_DESC")["VALUE_NUM"]
            .last()
            if not obs.empty
            else pd.Series(dtype=float)
        )
    else:
        latest = pd.Series(dtype=float)

    sentinel = set(SENTINEL_LABS.get(disease, []))
    for lab in labs_needed:
        tested_name = tested_flag_name(lab)
        if lab in latest.index and pd.notna(latest[lab]):
            row[lab] = float(latest[lab])
            row[tested_name] = 1
        else:
            row[tested_name] = 0
            if lab in sentinel:
                row[lab] = -1
            else:
                row[lab] = float(lab_medians.get(lab, 0))

    relevant_hist = HISTORY_FEATURES.get(disease, [])
    cond = conditions.copy() if conditions is not None else pd.DataFrame()
    present_before = set()
    if not cond.empty:
        cond = cond[cond["START"] < snapshot]
        present_before = set(cond["CONDITION_DESC"].unique())

    for term in relevant_hist:
        flag = hist_flag_name(term)
        if flag in feature_columns:
            row[flag] = 1 if term in present_before else 0
        else:
            notes["unused_history_terms"].append(term)

    med_terms = MEDICATION_FEATURES.get(disease, [])
    med = medications.copy() if medications is not None else pd.DataFrame()
    if not med.empty:
        med = med[med["START"] < snapshot]
    for term in med_terms:
        col = f"HIST_MED_{term}"
        if col not in feature_columns:
            continue
        if med.empty:
            row[col] = 0
        else:
            row[col] = int(med["MEDICATION_DESC"].str.contains(term, case=False, na=False).any())

    enc = encounters.copy() if encounters is not None else pd.DataFrame()
    if not enc.empty:
        enc = enc[enc["START"] < snapshot]
        row["TOTAL_VISITS"] = int(len(enc))
        row["EMERGENCY_VISITS"] = int(enc["ENCOUNTERCLASS"].str.contains("emergency", case=False, na=False).sum())
    else:
        row["TOTAL_VISITS"] = 0
        row["EMERGENCY_VISITS"] = 0

    gender = str(patient_row.get("GENDER", ""))
    race = str(patient_row.get("RACE", ""))
    ethnicity = str(patient_row.get("ETHNICITY", ""))
    row["GENDER_M"] = 1 if gender == "M" else 0
    for race_col in ["RACE_black", "RACE_hawaiian", "RACE_native", "RACE_other", "RACE_white"]:
        row[race_col] = 1 if race_col == f"RACE_{race}" else 0
    row["ETHNICITY_nonhispanic"] = 1 if ethnicity == "nonhispanic" else 0

    aligned = {col: row.get(col, 0) for col in feature_columns}
    missing_cols = [c for c in feature_columns if c not in row]
    notes["filled_with_zero"] = missing_cols
    X = pd.DataFrame([aligned], columns=feature_columns)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, notes
