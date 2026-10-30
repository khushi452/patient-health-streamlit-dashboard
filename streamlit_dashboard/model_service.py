from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_builder import build_patient_features
from paths import MODEL_FILES


def load_model_bundles_impl() -> dict:
    bundles = {}
    errors = {}
    for disease, path in MODEL_FILES.items():
        if not path.exists():
            errors[disease] = f"File not found: {path}"
            continue
        try:
            obj = joblib.load(path)
        except Exception as exc:
            errors[disease] = f"Could not load pickle: {exc}"
            continue
        if not isinstance(obj, dict):
            errors[disease] = f"Unexpected object type: {type(obj)}"
            continue
        required = ["rf", "logreg", "scaler", "feature_columns", "lab_medians"]
        missing = [k for k in required if k not in obj]
        if missing:
            errors[disease] = f"Bundle missing keys: {missing}"
            continue
        bundles[disease] = obj
    return {"bundles": bundles, "errors": errors}


@st.cache_resource(show_spinner="Loading trained model bundles from Models/...")
def load_model_bundles() -> dict:
    return load_model_bundles_impl()


def score_patient(disease: str, bundle: dict, patient_row, conditions, medications, encounters, observations, snapshot):
    X, notes = build_patient_features(
        disease=disease,
        patient_row=patient_row,
        conditions=conditions,
        medications=medications,
        encounters=encounters,
        observations=observations,
        feature_columns=list(bundle["feature_columns"]),
        lab_medians=dict(bundle["lab_medians"]),
        snapshot=snapshot,
    )
    if X.empty:
        return {"ok": False, "notes": notes}

    expected = int(getattr(bundle["rf"], "n_features_in_", X.shape[1]))
    if X.shape[1] != expected:
        notes["missing_required"].append(
            f"Feature width {X.shape[1]} does not match model n_features_in_={expected}"
        )
        return {"ok": False, "notes": notes}

    rf_prob = float(bundle["rf"].predict_proba(X)[0, 1])
    rf_pred = int(bundle["rf"].predict(X)[0])
    X_scaled = bundle["scaler"].transform(X)
    lr_prob = float(bundle["logreg"].predict_proba(X_scaled)[0, 1])
    lr_pred = int(bundle["logreg"].predict(X_scaled)[0])

    importances = None
    if hasattr(bundle["rf"], "feature_importances_"):
        imp = pd.Series(bundle["rf"].feature_importances_, index=list(bundle["feature_columns"]))
        used = X.iloc[0]
        contrib = []
        for feat, importance in imp.sort_values(ascending=False).head(8).items():
            contrib.append({"feature": feat, "importance": float(importance), "value": used[feat]})
        importances = contrib

    return {
        "ok": True,
        "notes": notes,
        "rf_prob": rf_prob,
        "rf_pred": rf_pred,
        "logreg_prob": lr_prob,
        "logreg_pred": lr_pred,
        "top_features": importances,
        "row": X.iloc[0].to_dict(),
    }


def score_feature_vector(bundle: dict, feature_row: dict, overrides: dict | None = None) -> dict:
    from feature_builder import tested_flag_name
    cols = list(bundle["feature_columns"])
    data = dict(feature_row)
    if overrides:
        for k, v in overrides.items():
            if k in cols:
                data[k] = float(v)
            tflag = tested_flag_name(k)
            if tflag in cols:
                data[tflag] = 1
    X = pd.DataFrame([data], columns=cols)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    rf_prob = float(bundle["rf"].predict_proba(X)[0, 1])
    rf_pred = int(bundle["rf"].predict(X)[0])
    X_scaled = bundle["scaler"].transform(X)
    lr_prob = float(bundle["logreg"].predict_proba(X_scaled)[0, 1])
    lr_pred = int(bundle["logreg"].predict(X_scaled)[0])
    return {
        "rf_prob": rf_prob,
        "rf_pred": rf_pred,
        "logreg_prob": lr_prob,
        "logreg_pred": lr_pred,
    }

