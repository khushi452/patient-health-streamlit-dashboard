from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    LAB_REFERENCE_RANGES,
    LAB_SHORT_NAMES,
    NOTEBOOK_METRICS,
    SELECTED_LABS,
    TARGETS_FINAL,
    VITAL_LABS,
    get_lab_status,
)
from data_loader import load_tables, patient_slice
from feature_builder import choose_snapshot, recorded_diagnoses
from model_service import load_model_bundles, score_feature_vector, score_patient

st.set_page_config(
    page_title="Patient Risk Intelligence",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- design tokens
PAL = {
    "navy": "#0F2A43",
    "teal": "#0E9AA7",
    "ink": "#12263A",
    "muted": "#5B7083",
    "line": "#E1E8EE",
    "low": "#2E9E6B",
    "mod": "#E9A23B",
    "high": "#D64545",
}
# Probability cut-offs for the risk badge. Tune these against your model's
# calibration; they are display thresholds, not clinical rules.
RISK_MODERATE = 0.30
RISK_HIGH = 0.70

STATUS_COLORS = {"ok": PAL["low"], "gold": PAL["mod"], "danger": PAL["high"]}

CSS = """
<style>
  .block-container {padding-top: 2rem; max-width: 1280px;}
  h1, h2, h3 {color: #12263A; letter-spacing: -0.01em;}
  .risk-card {background:#fff; border:1px solid #E1E8EE; border-radius:12px;
              padding:14px 16px; height:100%;}
  .risk-card .name {font-size:0.82rem; color:#5B7083;}
  .risk-card .pct {font-size:2rem; font-weight:700; line-height:1.15; margin:2px 0 6px 0;}
  .risk-card .note {font-size:0.76rem; color:#5B7083; margin-top:6px;}
  .badge {display:inline-block; padding:2px 10px; border-radius:999px;
          font-size:0.75rem; font-weight:600;}
  .pill {display:inline-block; padding:3px 11px; border-radius:999px; font-size:0.78rem;
         font-weight:600; margin:0 6px 6px 0;}
  .pill-on {background:#0F2A43; color:#fff;}
  .pill-off {background:#EEF2F6; color:#5B7083;}
  .vital {background:#fff; border:1px solid #E1E8EE; border-radius:10px; padding:10px 12px;}
  .vital .lbl {font-size:0.76rem; color:#5B7083;}
  .vital .val {font-size:1.25rem; font-weight:700; color:#12263A;}
  .vital .sub {font-size:0.72rem; margin-top:2px;}
  [data-testid="stSidebar"] .stRadio label {padding: 4px 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def fmt_date(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%d %b %Y")


def style_fig(fig, height=340):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PAL["ink"]),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor="#E8EEF3")
    fig.update_yaxes(gridcolor="#E8EEF3")
    return fig


def risk_tier(prob: float):
    """Return label, accent colour, badge background, badge text colour."""
    if prob >= RISK_HIGH:
        return "High", PAL["high"], "#FBE4E4", "#8A2222"
    if prob >= RISK_MODERATE:
        return "Moderate", PAL["mod"], "#FBEFD5", "#7A4E0B"
    return "Low", PAL["low"], "#DDF3E8", "#17583A"


def risk_card_html(disease, pct_text, label, accent, badge_bg, badge_fg, note=""):
    note_html = f'<div class="note">{note}</div>' if note else ""
    return (
        f'<div class="risk-card" style="border-top:4px solid {accent}">'
        f'<div class="name">{disease}</div>'
        f'<div class="pct" style="color:{PAL["ink"]}">{pct_text}</div>'
        f'<span class="badge" style="background:{badge_bg};color:{badge_fg}">{label}</span>'
        f"{note_html}</div>"
    )


@st.cache_data(show_spinner=False)
def cohort_stats(_patients, _conditions, _encounters) -> dict:
    n = len(_patients)
    prevalence = []
    for disease, names in TARGETS_FINAL.items():
        count = _conditions.loc[_conditions["CONDITION_DESC"].isin(names), "PATIENT"].nunique()
        prevalence.append({"Disease": disease, "Patients": int(count), "Share": count / n * 100})

    sets = {
        d: set(_conditions.loc[_conditions["CONDITION_DESC"].isin(names), "PATIENT"])
        for d, names in TARGETS_FINAL.items()
    }
    diseases = list(TARGETS_FINAL)
    matrix = pd.DataFrame(
        [[len(sets[a] & sets[b]) for b in diseases] for a in diseases],
        index=diseases,
        columns=diseases,
    )
    return {
        "n": n,
        "mean_age": float(_patients["AGE"].mean()),
        "female_pct": float((_patients["GENDER"] == "F").mean() * 100),
        "encounters": int(len(_encounters)),
        "prevalence": pd.DataFrame(prevalence),
        "comorbidity": matrix,
    }


@st.cache_data(show_spinner=False)
def score_all(patient_id: str, _slice, _bundles: dict, _errors: dict):
    """Score the five diseases once per patient and reuse the result."""
    snapshot = choose_snapshot(_slice["patient"], _slice["encounters"], _slice["observations"])
    results = {}
    for disease in TARGETS_FINAL:
        if disease in _errors or disease not in _bundles:
            results[disease] = {"ok": False, "notes": {"missing_required": [_errors.get(disease, "Model not loaded")]}}
            continue
        results[disease] = score_patient(
            disease,
            _bundles[disease],
            _slice["patient"],
            _slice["conditions"],
            _slice["medications"],
            _slice["encounters"],
            _slice["observations"],
            snapshot,
        )
    return snapshot, results


# ---------------------------------------------------------------- pages
def page_overview(tables):
    st.title("Population overview")
    st.caption("Cohort-level view of the synthetic Synthea patient records used to train the models.")
    s = cohort_stats(tables["patients"], tables["conditions"], tables["encounters"])

    k1, k2, k3, k4 = st.columns(4)
    for col, label, value in [
        (k1, "Patients", f"{s['n']:,}"),
        (k2, "Average age", f"{s['mean_age']:.1f} yrs"),
        (k3, "Female", f"{s['female_pct']:.0f}%"),
        (k4, "Encounters", f"{s['encounters']:,}"),
    ]:
        with col.container(border=True):
            st.metric(label, value)

    left, right = st.columns(2)
    with left.container(border=True):
        fig = px.bar(
            s["prevalence"].sort_values("Share"),
            x="Share",
            y="Disease",
            orientation="h",
            text=s["prevalence"].sort_values("Share")["Share"].map(lambda v: f"{v:.1f}%"),
            title="Recorded diagnoses (share of patients)",
            color_discrete_sequence=[PAL["teal"]],
        )
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right.container(border=True):
        fig = px.histogram(
            tables["patients"],
            x="AGE",
            color="GENDER",
            nbins=30,
            title="Age and sex distribution",
            color_discrete_map={"F": PAL["teal"], "M": PAL["navy"]},
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with st.container(border=True):
        fig = px.imshow(
            s["comorbidity"],
            text_auto=True,
            color_continuous_scale=["#EAF6F7", PAL["teal"], PAL["navy"]],
            title="Patients with both diseases (comorbidity)",
        )
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)


def page_patient_risk(slice_, model_pack):
    p = slice_["patient"]
    rec = recorded_diagnoses(slice_["conditions"])
    snapshot, results = score_all(
        str(p["PATIENT"]), slice_, model_pack["bundles"], model_pack["errors"]
    )

    # Patient header
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 2, 1.4])
        with c1:
            st.subheader(p["DISPLAY_NAME"])
            st.caption(f"ID {p['PATIENT']}")
        with c2:
            st.write(f"**{p['AGE']} yrs**, {'Female' if p['GENDER'] == 'F' else 'Male'}, {str(p['RACE']).title()}")
            st.caption(f"{p['CITY']}, {p['STATE']}")
        with c3:
            st.write("**Last encounter**")
            st.caption(fmt_date(p.get("LAST_ENCOUNTER")))

        pills = "".join(
            f'<span class="pill {"pill-on" if info["recorded"] else "pill-off"}">'
            f'{d}{" - diagnosed" if info["recorded"] else ""}</span>'
            for d, info in rec.items()
        )
        st.markdown(pills, unsafe_allow_html=True)

    # Vitals strip
    st.markdown("#### Latest measurements")
    vitals = [
        ("Systolic Blood Pressure", "Systolic BP", "mmHg"),
        ("Hemoglobin A1c/Hemoglobin.total in Blood", "HbA1c", "%"),
        ("Glomerular filtration rate [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)/1.73 sq M", "eGFR", "mL/min"),
        ("Body mass index (BMI) [Ratio]", "BMI", "kg/m2"),
        ("Heart rate", "Heart rate", "bpm"),
    ]
    obs = slice_["observations"]
    latest = {}
    if obs is not None and not obs.empty:
        grouped = obs.dropna(subset=["VALUE_NUM"]).sort_values("DATE").groupby("OBSERVATION_DESC").last()
        latest = {k: (r["VALUE_NUM"], r["DATE"]) for k, r in grouped.iterrows()}
    for col, (key, label, unit) in zip(st.columns(len(vitals)), vitals):
        if key in latest:
            val, dt = latest[key]
            status, cls = get_lab_status(key, val)
            color = STATUS_COLORS.get(cls, PAL["muted"])
            col.markdown(
                f'<div class="vital"><div class="lbl">{label}</div>'
                f'<div class="val">{val:.1f} <span style="font-size:0.75rem;color:{PAL["muted"]}">{unit}</span></div>'
                f'<div class="sub" style="color:{color}">{status}, {fmt_date(dt)}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f'<div class="vital"><div class="lbl">{label}</div>'
                f'<div class="val">-</div><div class="sub" style="color:{PAL["muted"]}">Not tested</div></div>',
                unsafe_allow_html=True,
            )

    # Risk cards
    st.markdown("#### Disease risk")
    st.caption(
        f"Predicted chance of a new diagnosis within 2 years, using data up to {fmt_date(snapshot)}. "
        "Already-diagnosed conditions are shown as diagnosed instead of scored."
    )
    cols = st.columns(len(TARGETS_FINAL))
    for col, disease in zip(cols, TARGETS_FINAL):
        scored = results[disease]
        if rec[disease]["recorded"]:
            html = risk_card_html(
                disease, "Diagnosed", "On record", PAL["navy"], "#E1E8EE", PAL["navy"],
                f"Since {fmt_date(rec[disease]['first_date'])}",
            )
        elif not scored.get("ok"):
            reason = "; ".join(scored.get("notes", {}).get("missing_required", ["Could not score"]))
            html = risk_card_html(disease, "n/a", "Unavailable", "#B8C4CF", "#EEF2F6", PAL["muted"], reason)
        else:
            prob = scored["rf_prob"]
            label, accent, bg, fg = risk_tier(prob)
            html = risk_card_html(disease, f"{prob * 100:.0f}%", label, accent, bg, fg)
        col.markdown(html, unsafe_allow_html=True)

    # Explainability
    scorable = [d for d, r in results.items() if r.get("ok")]
    if scorable:
        st.markdown("#### What drives the prediction")
        disease = st.selectbox("Disease", scorable, label_visibility="collapsed")
        scored = results[disease]
        left, right = st.columns([1.3, 1])
        with left.container(border=True):
            feats = pd.DataFrame(scored.get("top_features") or [])
            if not feats.empty:
                feats["Feature"] = feats["feature"].map(lambda x: LAB_SHORT_NAMES.get(x, x))
                fig = px.bar(
                    feats.sort_values("importance"),
                    x="importance",
                    y="Feature",
                    orientation="h",
                    hover_data=["value"],
                    title=f"Top model features for {disease}",
                    color_discrete_sequence=[PAL["teal"]],
                )
                fig.update_layout(xaxis_title="Importance (random forest)", yaxis_title=None)
                st.plotly_chart(style_fig(fig, 320), use_container_width=True)
        with right.container(border=True):
            m = NOTEBOOK_METRICS[disease]
            st.metric("Random forest probability", f"{scored['rf_prob']:.1%}")
            st.metric("Logistic regression probability", f"{scored['logreg_prob']:.1%}")
            st.metric("Model ROC-AUC (test set)", f"{m['rf_auc']:.3f}")

        simulator(results, model_pack["bundles"])

    st.caption("Decision-support prototype trained on synthetic data. Not for clinical use.")


def simulator(results, bundles):
    with st.expander("What-if simulator"):
        st.caption("Change a measurement and see how each predicted risk moves.")
        first = next(r for r in results.values() if r.get("ok"))
        base = first["row"]

        def current(key, default):
            v = float(base.get(key, default) or 0)
            return v if v > 0 else default

        c1, c2, c3, c4 = st.columns(4)
        bmi = c1.slider("BMI", 16.0, 48.0, float(np.clip(current("Body mass index (BMI) [Ratio]", 26.0), 16, 48)), 0.5)
        sbp = c2.slider("Systolic BP", 85.0, 210.0, float(np.clip(current("Systolic Blood Pressure", 120.0), 85, 210)), 1.0)
        glu = c3.slider("Glucose (mg/dL)", 60.0, 320.0, float(np.clip(current("Glucose [Mass/volume] in Blood", 95.0), 60, 320)), 2.0)
        a1c = c4.slider("HbA1c (%)", 4.0, 14.0, float(np.clip(current("Hemoglobin A1c/Hemoglobin.total in Blood", 5.5), 4, 14)), 0.1)
        overrides = {
            "Body mass index (BMI) [Ratio]": bmi,
            "Systolic Blood Pressure": sbp,
            "Glucose [Mass/volume] in Blood": glu,
            "Hemoglobin A1c/Hemoglobin.total in Blood": a1c,
        }
        rows = []
        for disease, scored in results.items():
            if not scored.get("ok"):
                continue
            new = score_feature_vector(bundles[disease], scored["row"], overrides)["rf_prob"]
            delta = (new - scored["rf_prob"]) * 100
            rows.append({
                "Disease": disease,
                "Current": f"{scored['rf_prob'] * 100:.1f}%",
                "Simulated": f"{new * 100:.1f}%",
                "Change": f"{delta:+.1f} pts",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def page_history(slice_, procedures_available):
    st.title("Clinical history")
    if not procedures_available:
        st.info("Procedure records are not part of this dataset, so surgical history is not shown.")
    tabs = st.tabs(["Conditions", "Medications", "Encounters", "Timeline"])

    with tabs[0]:
        df = slice_["conditions"].sort_values("START", ascending=False).copy()
        if df.empty:
            st.info("No conditions recorded for this patient.")
        else:
            df["Status"] = np.where(df["STOP"].isna(), "Active", "Resolved")
            choice = st.radio("Show", ["All", "Active", "Resolved"], horizontal=True, key="cond_filter")
            if choice != "All":
                df = df[df["Status"] == choice]
            df["Start"] = df["START"].dt.strftime("%Y-%m-%d")
            df["End"] = df["STOP"].dt.strftime("%Y-%m-%d").fillna("Ongoing")
            st.dataframe(df.rename(columns={"CONDITION_DESC": "Condition"})[["Start", "End", "Status", "Condition"]],
                         hide_index=True, use_container_width=True, height=400)

    with tabs[1]:
        df = slice_["medications"].sort_values("START", ascending=False).copy()
        if df.empty:
            st.info("No medications recorded for this patient.")
        else:
            df["Status"] = np.where(df["STOP"].isna(), "Ongoing", "Completed")
            df["Start"] = df["START"].dt.strftime("%Y-%m-%d")
            df["End"] = df["STOP"].dt.strftime("%Y-%m-%d").fillna("Ongoing")
            df = df.rename(columns={"MEDICATION_DESC": "Medication", "REASONDESCRIPTION": "Reason"})
            st.dataframe(df[["Start", "End", "Status", "Medication", "Reason"]],
                         hide_index=True, use_container_width=True, height=400)

    with tabs[2]:
        df = slice_["encounters"].sort_values("START", ascending=False).copy()
        if df.empty:
            st.info("No encounters recorded for this patient.")
        else:
            df["Date"] = df["START"].dt.strftime("%Y-%m-%d")
            df = df.rename(columns={"ENCOUNTERCLASS": "Type", "DESCRIPTION": "Description",
                                    "REASONDESCRIPTION": "Reason", "TOTAL_CLAIM_COST": "Cost ($)"})
            st.dataframe(df[["Date", "Type", "Description", "Reason", "Cost ($)"]],
                         hide_index=True, use_container_width=True, height=400)

    with tabs[3]:
        rows = []
        for _, r in slice_["conditions"].iterrows():
            rows.append({"Date": r["START"], "Category": "Condition", "Event": r["CONDITION_DESC"]})
        for _, r in slice_["medications"].iterrows():
            rows.append({"Date": r["START"], "Category": "Medication", "Event": r["MEDICATION_DESC"]})
        for _, r in slice_["encounters"].iterrows():
            reason = r["REASONDESCRIPTION"] if pd.notna(r["REASONDESCRIPTION"]) else r["DESCRIPTION"]
            rows.append({"Date": r["START"], "Category": f"Encounter ({r['ENCOUNTERCLASS']})", "Event": str(reason)})
        if not rows:
            st.info("No dated events for this patient.")
        else:
            tl = pd.DataFrame(rows).dropna(subset=["Date"]).sort_values("Date")
            fig = px.scatter(tl, x="Date", y="Category", color="Category", hover_data=["Event"],
                             color_discrete_sequence=[PAL["teal"], PAL["navy"], PAL["mod"], "#7C8FA3", PAL["low"]])
            fig.update_traces(marker=dict(size=10, opacity=0.85))
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)


def page_labs(slice_):
    st.title("Labs and trends")
    obs = slice_["observations"].copy()
    if obs.empty:
        st.info("No lab or vital observations for this patient.")
        return
    obs["short"] = obs["OBSERVATION_DESC"].map(lambda x: LAB_SHORT_NAMES.get(x, x))
    available = [LAB_SHORT_NAMES.get(x, x) for x in SELECTED_LABS if x in set(obs["OBSERVATION_DESC"])]
    default = [LAB_SHORT_NAMES[v] for v in VITAL_LABS if LAB_SHORT_NAMES.get(v) in available][:3] or available[:3]
    picked = st.multiselect("Measurements", available, default=default)
    plot_df = obs[obs["short"].isin(picked)].dropna(subset=["VALUE_NUM"]).sort_values("DATE")
    if plot_df.empty:
        st.info("Select at least one measurement with numeric values.")
    else:
        fig = px.line(plot_df, x="DATE", y="VALUE_NUM", color="short", markers=True,
                      color_discrete_sequence=[PAL["teal"], PAL["navy"], PAL["mod"], PAL["high"], PAL["low"], "#7C8FA3"])
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    latest = (obs.dropna(subset=["VALUE_NUM"]).sort_values("DATE")
              .groupby("OBSERVATION_DESC", as_index=False).last())
    rows = []
    for _, r in latest.iterrows():
        status, _cls = get_lab_status(r["OBSERVATION_DESC"], r["VALUE_NUM"])
        ref = LAB_REFERENCE_RANGES.get(r["OBSERVATION_DESC"])
        rows.append({
            "Measurement": LAB_SHORT_NAMES.get(r["OBSERVATION_DESC"], r["OBSERVATION_DESC"]),
            "Value": round(float(r["VALUE_NUM"]), 2),
            "Units": r.get("UNITS", "") or "",
            "Normal range": f"{ref.get('min', '-')} to {ref.get('max', '-')}" if ref else "-",
            "Status": status,
            "Date": fmt_date(r["DATE"]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def page_models(model_pack):
    st.title("Model performance")
    st.caption("Random forest ROC-AUC on the held-out test set, from the training notebook.")
    rows = [{"Disease": d, "ROC-AUC": NOTEBOOK_METRICS[d]["rf_auc"], "Training cases": NOTEBOOK_METRICS[d]["cases"]}
            for d in TARGETS_FINAL if d in NOTEBOOK_METRICS]
    df = pd.DataFrame(rows)
    with st.container(border=True):
        fig = px.bar(df.sort_values("ROC-AUC"), x="ROC-AUC", y="Disease", orientation="h",
                     text=df.sort_values("ROC-AUC")["ROC-AUC"].map(lambda v: f"{v:.3f}"),
                     color_discrete_sequence=[PAL["teal"]])
        fig.update_xaxes(range=[0.5, 1.0])
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    st.dataframe(df, hide_index=True, use_container_width=True)
    if model_pack["errors"]:
        st.warning("Some models failed to load: " + "; ".join(f"{k}: {v}" for k, v in model_pack["errors"].items()))
    st.caption("Trained on synthetic Synthea data to predict a new diagnosis two years ahead. Not validated for clinical use.")


# ---------------------------------------------------------------- app shell
def main():
    try:
        tables = load_tables()
        model_pack = load_model_bundles()
    except Exception as exc:
        st.error(f"Could not load the dataset or models: {exc}")
        st.stop()

    patients = tables["patients"]

    st.sidebar.markdown("### Patient Risk Intelligence")
    st.sidebar.caption("Disease prediction for 5 chronic conditions")
    page = st.sidebar.radio(
        "Page",
        ["Population overview", "Patient risk", "Clinical history", "Labs and trends", "Model performance"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    # Patient picker (hidden on pages that do not need a patient)
    slice_ = None
    if page in ("Patient risk", "Clinical history", "Labs and trends"):
        st.sidebar.markdown("**Patient**")
        age_group = st.sidebar.selectbox("Age group", ["All ages", "Under 18", "18 to 64", "65 and over"])
        pool = patients
        if age_group == "Under 18":
            pool = pool[pool["AGE"] < 18]
        elif age_group == "18 to 64":
            pool = pool[(pool["AGE"] >= 18) & (pool["AGE"] < 65)]
        elif age_group == "65 and over":
            pool = pool[pool["AGE"] >= 65]

        query = st.sidebar.text_input("Search name, ID or city")
        if query.strip():
            pool = pool[pool["SEARCH_TEXT"].str.contains(query.strip().lower(), regex=False, na=False)]
        if pool.empty:
            st.sidebar.warning("No match. Showing all patients.")
            pool = patients

        pool = pool.sort_values(["LAST", "FIRST"])
        ids = pool["PATIENT"].tolist()
        labels = {r.PATIENT: f"{r.LAST}, {r.FIRST} ({r.AGE}y {r.GENDER})" for r in pool.itertuples(index=False)}

        if st.session_state.get("active_patient") not in ids:
            st.session_state["active_patient"] = ids[0]
        if st.sidebar.button("Pick random patient", use_container_width=True):
            st.session_state["active_patient"] = ids[int(np.random.randint(len(ids)))]
        patient_id = st.sidebar.selectbox("Active patient", ids, key="active_patient", format_func=lambda i: labels[i])
        slice_ = patient_slice(tables, patient_id)

    st.sidebar.divider()
    st.sidebar.caption(f"{len(patients):,} patients. Synthetic Synthea data.")

    if page == "Population overview":
        page_overview(tables)
    elif page == "Patient risk":
        page_patient_risk(slice_, model_pack)
    elif page == "Clinical history":
        page_history(slice_, tables["procedures_available"])
    elif page == "Labs and trends":
        page_labs(slice_)
    else:
        page_models(model_pack)


if __name__ == "__main__":
    main()
