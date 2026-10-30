from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    COLORS,
    LAB_REFERENCE_RANGES,
    LAB_SHORT_NAMES,
    NOTEBOOK_METRICS,
    PLOTLY_SEQUENCE,
    SELECTED_LABS,
    TARGETS_FINAL,
    VITAL_LABS,
    get_lab_status,
)
from data_loader import load_tables, patient_slice
from feature_builder import choose_snapshot, recorded_diagnoses
from model_service import load_model_bundles, score_feature_vector, score_patient
from paths import DATA_DIR, MODELS_DIR, PROJECT_ROOT

st.set_page_config(
    page_title="Patient Health Record & Risk Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def get_doctor_avatar_base64() -> str:
    img_path = APP_DIR / "doctor_avatar.jpg"
    if img_path.exists():
        import base64
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    .stApp {{
        background-color: #F2F5FA;
        color: #1B2D52;
    }}
    [data-testid="stSidebar"] {{
        background-color: #FFFFFF !important;
        border-right: 1px solid #E6ECF5 !important;
        box-shadow: 2px 0 16px rgba(0, 0, 0, 0.02);
    }}
    [data-testid="stSidebar"] * {{
        color: #556987 !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        padding: 0.45rem 0.75rem;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.15s ease;
        margin-bottom: 0.15rem;
    }}
    [data-testid="stSidebar"] .stRadio label:hover {{
        background-color: #F0F4FA;
        color: #1E50BA !important;
    }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"] {{
        background-color: #E8F0FE;
        color: #1E50BA !important;
        font-weight: 700;
    }}
    [data-testid="stSidebar"] .stTextInput input, [data-testid="stSidebar"] .stSelectbox select {{
        background: #F8FAFD;
        color: #1B2D52 !important;
        border: 1px solid #DCE4F0;
        border-radius: 10px;
    }}
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
        background: #F8FAFD;
        color: #1B2D52 !important;
        border: 1px solid #DCE4F0;
        border-radius: 10px;
    }}
    
    /* Top 3 Stat Cards */
    .top-stat-card {{
        background: #FFFFFF;
        border-radius: 20px;
        padding: 1.15rem 1.4rem;
        display: flex;
        align-items: center;
        gap: 1.15rem;
        box-shadow: 0 4px 20px rgba(30, 60, 140, 0.04);
        border: 1px solid #EDF2F8;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .top-stat-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(30, 60, 140, 0.08);
    }}
    .top-stat-icon {{
        width: 60px;
        height: 60px;
        border-radius: 50%;
        border: 2px solid #1E50BA;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        color: #1E50BA;
        background: #F5F8FF;
        flex-shrink: 0;
    }}
    .top-stat-label {{
        font-size: 0.82rem;
        color: #7E90B0;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }}
    .top-stat-val {{
        font-size: 1.85rem;
        font-weight: 800;
        color: #1B2D52;
        line-height: 1.15;
    }}
    .top-stat-sub {{
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }}
    
    /* Panels */
    .white-panel {{
        background: #FFFFFF;
        border-radius: 20px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 4px 20px rgba(30, 60, 140, 0.04);
        border: 1px solid #EDF2F8;
        margin-bottom: 1rem;
    }}
    .panel-title {{
        font-weight: 750;
        color: #1B2D52;
        font-size: 1.05rem;
        margin-bottom: 0.85rem;
        letter-spacing: -0.01em;
    }}
    
    .time-badge {{
        background: #E8F0FE;
        color: #1E50BA;
        padding: 0.25rem 0.65rem;
        border-radius: 8px;
        font-size: 0.74rem;
        font-weight: 700;
    }}
    
    .metric-card {{
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 18px rgba(22, 58, 140, 0.05);
        border: 1px solid #EDF2F8;
        border-top: 4px solid {COLORS["blue"]};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(22, 58, 140, 0.1);
    }}
    .metric-card.gold {{ border-top-color: #F2A922; }}
    .metric-card.lavender {{ border-top-color: #C5D7F8; }}
    .metric-card.danger {{ border-top-color: #EF4444; }}
    .metric-card.ok {{ border-top-color: #10B981; }}
    .metric-card .label {{
        color: #7E90B0;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: .04em;
        text-transform: uppercase;
    }}
    .metric-card .value {{
        color: #1B2D52;
        font-size: 1.75rem;
        font-weight: 800;
        margin-top: 0.25rem;
        line-height: 1.2;
    }}
    .metric-card .subtext {{
        color: #94A3B8;
        font-size: 0.8rem;
        margin-top: 0.35rem;
    }}
    .warn-banner {{
        background: #FFF9ED;
        border-left: 5px solid #F2A922;
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        color: #7A5500;
        margin: 0.6rem 0 1.2rem 0;
        border: 1px solid #FFE9B8;
        border-left-width: 5px;
    }}
    .info-banner {{
        background: #EEF4FD;
        border-left: 5px solid #1E50BA;
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        color: #143682;
        border: 1px solid #D7E5FA;
        border-left-width: 5px;
    }}
    .section-card {{
        background: #FFFFFF;
        border-radius: 18px;
        padding: 1.2rem 1.35rem;
        box-shadow: 0 4px 18px rgba(22, 58, 140, 0.04);
        border: 1px solid #EDF2F8;
        margin-bottom: 1rem;
    }}
    .pill {{
        display: inline-block;
        padding: 0.24rem 0.75rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
    }}
    .pill-blue {{ background: #E8F0FE; color: #1E50BA; }}
    .pill-gold {{ background: #FFF4D9; color: #9E7100; }}
    .pill-navy {{ background: #1E50BA; color: white; }}
    .pill-ok {{ background: #D1FAE5; color: #065F46; }}
    .pill-danger {{ background: #FEE2E2; color: #991B1B; }}
    .pill-gray {{ background: #F1F5F9; color: #475569; }}
    
    .sim-box {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
    }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)



def metric_card(label: str, value, variant: str = "blue", subtext: str = "") -> None:
    sub_html = f'<div class="subtext">{subtext}</div>' if subtext else ""
    st.markdown(
        f'<div class="metric-card {variant}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f"{sub_html}</div>",
        unsafe_allow_html=True,
    )


def style_fig(fig, height=360):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, Segoe UI, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
        colorway=PLOTLY_SEQUENCE,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E6EAF5")
    fig.update_yaxes(showgrid=True, gridcolor="#E6EAF5")
    return fig


def fmt_date(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    ts = pd.Timestamp(value)
    return ts.strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False)
def overview_stats(_patients, _conditions, _encounters, _medications) -> dict:
    patients = _patients
    conditions = _conditions
    disease_counts = {}
    for disease, names in TARGETS_FINAL.items():
        n = conditions.loc[conditions["CONDITION_DESC"].isin(names), "PATIENT"].nunique()
        disease_counts[disease] = int(n)
    top_conditions = (
        conditions["CONDITION_DESC"].value_counts().head(10).rename_axis("condition").reset_index(name="count")
    )
    enc_class = _encounters["ENCOUNTERCLASS"].value_counts().rename_axis("class").reset_index(name="count")
    total_claim = _encounters["TOTAL_CLAIM_COST"].sum() if "TOTAL_CLAIM_COST" in _encounters else 0.0
    mean_income = patients["INCOME"].dropna().mean() if "INCOME" in patients else 0.0

    return {
        "n_patients": int(len(patients)),
        "n_female": int((patients["GENDER"] == "F").sum()),
        "n_male": int((patients["GENDER"] == "M").sum()),
        "n_deceased": int(patients["DECEASED"].sum()),
        "n_conditions": int(len(conditions)),
        "n_encounters": int(len(_encounters)),
        "n_medications": int(len(_medications)),
        "mean_age": float(patients["AGE"].mean()),
        "total_claim_cost": float(total_claim),
        "mean_income": float(mean_income),
        "disease_counts": disease_counts,
        "top_conditions": top_conditions,
        "enc_class": enc_class,
        "race": patients["RACE"].value_counts().rename_axis("race").reset_index(name="count"),
        "age": patients[["AGE", "GENDER"]],
    }


@st.cache_data(show_spinner=False)
def compute_comorbidity_matrix(_conditions):
    diseases = list(TARGETS_FINAL.keys())
    patients_by_disease = {}
    for d, names in TARGETS_FINAL.items():
        patients_by_disease[d] = set(_conditions.loc[_conditions["CONDITION_DESC"].isin(names), "PATIENT"].unique())
    matrix = np.zeros((len(diseases), len(diseases)), dtype=int)
    for i, d1 in enumerate(diseases):
        for j, d2 in enumerate(diseases):
            matrix[i, j] = len(patients_by_disease[d1] & patients_by_disease[d2])
    return pd.DataFrame(matrix, index=diseases, columns=diseases)


def render_overview(tables, slice_=None):
    # Top Header Bar
    st.markdown(
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.4rem;">'
        '<div><h1 style="margin: 0; color: #1B2D52; font-weight: 800; font-size: 1.85rem; letter-spacing: -0.02em;">Dashboard</h1></div>'
        '<div style="display: flex; align-items: center; gap: 12px;">'
        '<span style="background: white; border: 1px solid #EDF2F8; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.05rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03); cursor: pointer;">✉️</span>'
        '<span style="background: white; border: 1px solid #EDF2F8; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.05rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03); cursor: pointer;">🔔</span>'
        '<div style="background: white; border: 1px solid #EDF2F8; padding: 7px 16px; border-radius: 20px; font-size: 0.85rem; color: #94A3B8; display: flex; align-items: center; gap: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">'
        '<span>🔍</span><span>Search</span>'
        '</div>'
        '<span style="background: white; border: 1px solid #EDF2F8; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.05rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03); cursor: pointer;">☰</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    stats = overview_stats(tables["patients"], tables["conditions"], tables["encounters"], tables["medications"])

    # Top 3 Metric Cards matching screenshot
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="top-stat-card">'
            '<div class="top-stat-icon">👤</div>'
            '<div>'
            '<div class="top-stat-label">Total Patient</div>'
            '<div class="top-stat-val">2000+</div>'
            '<div class="top-stat-sub">Till Today</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="top-stat-card">'
            '<div class="top-stat-icon">🏥</div>'
            '<div>'
            '<div class="top-stat-label">Today Patient</div>'
            '<div class="top-stat-val">068</div>'
            '<div class="top-stat-sub">21 Dec-2021</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="top-stat-card">'
            '<div class="top-stat-icon">⏱️</div>'
            '<div>'
            '<div class="top-stat-label">Today Appointments</div>'
            '<div class="top-stat-val">085</div>'
            '<div class="top-stat-sub">21 Dec-2021</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # Middle Row: Patients Summary Donut | Today Appointment | Next Patient Details
    m_left, m_mid, m_right = st.columns([1.15, 1.35, 1.6])

    with m_left:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Patients Summary December 2021</div>', unsafe_allow_html=True)

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["New Patients", "Old Patients", "Total Patients"],
                    values=[35, 45, 20],
                    hole=0.62,
                    marker=dict(colors=["#C9DAF8", "#F2A922", "#1E50BA"], line=dict(color="white", width=3)),
                    textinfo="none",
                    hoverinfo="label+percent",
                    sort=False,
                )
            ]
        )
        fig_donut.update_layout(
            showlegend=False,
            height=230,
            margin=dict(l=5, r=5, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            '<div style="font-size: 0.8rem; font-weight: 600; color: #556987; padding-left: 0.5rem; margin-top: -0.5rem;">'
            '<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">'
            '<span style="width: 12px; height: 12px; background: #C9DAF8; border-radius: 2px; display: inline-block;"></span> New Patients'
            '</div>'
            '<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">'
            '<span style="width: 12px; height: 12px; background: #F2A922; border-radius: 2px; display: inline-block;"></span> Old Patients'
            '</div>'
            '<div style="display: flex; align-items: center; gap: 8px;">'
            '<span style="width: 12px; height: 12px; background: #1E50BA; border-radius: 2px; display: inline-block;"></span> Total Patients'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with m_mid:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Today Appointment</div>', unsafe_allow_html=True)

        st.markdown(
            '<div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #94A3B8; font-weight: 600; border-bottom: 1px solid #F1F5F9; padding-bottom: 0.5rem; margin-bottom: 0.75rem;">'
            '<span style="flex: 0.8;">Patient</span>'
            '<span style="flex: 2.2;">Name/Diagonosis</span>'
            '<span style="flex: 1.2; text-align: right;">Time</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        appointments = [
            ("👨", "M.J. Mical", "Health Checkup", "On Going", True),
            ("👨", "Sanath Deo", "Health Checkup", "12:30 PM", False),
            ("👩", "Loeara Phanj", "Report", "01:00 PM", False),
            ("👨", "Komola Haris", "Common Cold", "01:30 PM", False),
        ]

        for icon, name, diag, time_str, is_ongoing in appointments:
            badge_style = "background: #E8F0FE; color: #1E50BA;" if is_ongoing else "background: #F1F5F9; color: #556987;"
            st.markdown(
                f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.85rem;">'
                f'<div style="flex: 0.8;"><div style="width: 34px; height: 34px; border-radius: 50%; background: #EEF4FD; display: flex; align-items: center; justify-content: center; font-size: 1rem;">{icon}</div></div>'
                f'<div style="flex: 2.2;">'
                f'<div style="font-weight: 700; font-size: 0.86rem; color: #1B2D52;">{name}</div>'
                f'<div style="font-size: 0.74rem; color: #8496B3;">{diag}</div>'
                f'</div>'
                f'<div style="flex: 1.2; text-align: right;">'
                f'<span style="{badge_style} padding: 0.22rem 0.55rem; border-radius: 6px; font-size: 0.72rem; font-weight: 700;">{time_str}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="text-align: center; margin-top: 1rem;">'
            '<a style="font-size: 0.82rem; color: #1E50BA; font-weight: 700; text-decoration: none; cursor: pointer;">See All</a>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with m_right:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Next Patient Details</div>', unsafe_allow_html=True)

        # Dynamic or demo patient data
        pt_name = "Sanath Deo"
        pt_id = "022009202005"
        dob = "15 January 1989"
        sex = "Male"
        weight = "59 Kg"
        height = "172 cm"
        last_appt = "15 Dec - 2021"
        reg_date = "10 Dec 2021"
        history_tags = [("Anemia", "pill-gold"), ("Hypertension", "pill-blue"), ("Fever", "pill-danger")]
        prescription = "Paracetamol 500mg, Multivitamin"

        if slice_ is not None:
            p_data = slice_["patient"]
            pt_name = p_data["DISPLAY_NAME"]
            pt_id = str(p_data["PATIENT"])[:12]
            dob = fmt_date(p_data["BIRTHDATE"])
            sex = "Male" if p_data["GENDER"] == "M" else "Female"
            last_enc = p_data.get("LAST_ENCOUNTER")
            last_appt = fmt_date(last_enc) if pd.notna(last_enc) else "15 Dec - 2021"
            reg_date = "10 Dec 2021"
            rec_diag = recorded_diagnoses(slice_["conditions"])
            history_tags = []
            for d_name, d_info in rec_diag.items():
                if d_info["recorded"]:
                    history_tags.append((d_name, "pill-gold" if "Anemia" in d_name else "pill-blue"))
            if not history_tags:
                history_tags = [("General Checkup", "pill-ok"), ("Routine", "pill-blue")]
            if slice_["medications"] is not None and not slice_["medications"].empty:
                prescription = str(slice_["medications"].iloc[0]["MEDICATION_DESC"])[:42]

        st.markdown(
            f'<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.9rem;">'
            f'<div style="width: 44px; height: 44px; border-radius: 50%; background: #D9E4F7; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; overflow: hidden; border: 2px solid #1E50BA;">👨</div>'
            f'<div>'
            f'<div style="font-weight: 700; color: #1B2D52; font-size: 0.96rem;">{pt_name}</div>'
            f'<div style="font-size: 0.76rem; color: #7E90B0;">Health Checkup</div>'
            f'</div>'
            f'<div style="margin-left: auto; text-align: right;">'
            f'<div style="font-size: 0.74rem; color: #7E90B0; font-weight: 600;">Patient ID</div>'
            f'<div style="font-size: 0.74rem; color: #1B2D52; font-family: monospace; font-weight: 700;">{pt_id}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.78rem; background: #F8FAFD; padding: 0.75rem 0.9rem; border-radius: 12px; margin-bottom: 0.85rem;">'
            f'<div><span style="color: #8496B3;">D. O. B:</span> <br><b style="color: #1B2D52;">{dob}</b></div>'
            f'<div><span style="color: #8496B3;">Sex:</span> <br><b style="color: #1B2D52;">{sex}</b></div>'
            f'<div><span style="color: #8496B3;">Weight:</span> <br><b style="color: #1B2D52;">{weight}</b></div>'
            f'<div><span style="color: #8496B3;">Height:</span> <br><b style="color: #1B2D52;">{height}</b></div>'
            f'<div><span style="color: #8496B3;">Last Appointment:</span> <br><b style="color: #1B2D52;">{last_appt}</b></div>'
            f'<div><span style="color: #8496B3;">Reg. Date:</span> <br><b style="color: #1B2D52;">{reg_date}</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        history_html = "".join([f'<span class="pill {cls}">{txt}</span>' for txt, cls in history_tags[:3]])
        st.markdown(
            f'<div style="margin-bottom: 0.85rem;">'
            f'<div style="font-size: 0.78rem; font-weight: 700; color: #1B2D52; margin-bottom: 0.35rem;">Patient History</div>'
            f'<div>{history_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="display: flex; gap: 8px; margin-bottom: 0.85rem;">'
            '<div style="flex: 1.4; background: #1E50BA; color: white; padding: 0.45rem 0.6rem; border-radius: 8px; font-size: 0.76rem; font-weight: 700; text-align: center; cursor: pointer;">📞 (380) 555-0107</div>'
            '<div style="flex: 1; background: white; color: #1E50BA; border: 1px solid #D0DCF2; padding: 0.45rem 0.6rem; border-radius: 8px; font-size: 0.76rem; font-weight: 600; text-align: center; cursor: pointer;">📄 Documents</div>'
            '<div style="flex: 1; background: white; color: #1E50BA; border: 1px solid #D0DCF2; padding: 0.45rem 0.6rem; border-radius: 8px; font-size: 0.76rem; font-weight: 600; text-align: center; cursor: pointer;">💬 Chat</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div>'
            f'<div style="font-size: 0.78rem; font-weight: 700; color: #1B2D52; margin-bottom: 0.2rem;">Last Prescriptions</div>'
            f'<div style="font-size: 0.76rem; color: #64748B;">{prescription}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom Row: Patients Review | Appointment Request | Calendar
    b_left, b_mid, b_right = st.columns([1.15, 1.35, 1.6])

    with b_left:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Patients Review</div>', unsafe_allow_html=True)

        st.markdown(
            '<div style="margin-bottom: 0.7rem;">'
            '<div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748B; margin-bottom: 0.25rem;">'
            '<span>Excellent</span>'
            '</div>'
            '<div style="background: #F1F5F9; border-radius: 999px; height: 7px; overflow: hidden;">'
            '<div style="background: #1E50BA; width: 85%; height: 100%; border-radius: 999px;"></div>'
            '</div>'
            '</div>'
            '<div style="margin-bottom: 0.7rem;">'
            '<div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748B; margin-bottom: 0.25rem;">'
            '<span>Great</span>'
            '</div>'
            '<div style="background: #F1F5F9; border-radius: 999px; height: 7px; overflow: hidden;">'
            '<div style="background: #0E2A68; width: 68%; height: 100%; border-radius: 999px;"></div>'
            '</div>'
            '</div>'
            '<div style="margin-bottom: 0.7rem;">'
            '<div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748B; margin-bottom: 0.25rem;">'
            '<span>Good</span>'
            '</div>'
            '<div style="background: #F1F5F9; border-radius: 999px; height: 7px; overflow: hidden;">'
            '<div style="background: #E78C26; width: 52%; height: 100%; border-radius: 999px;"></div>'
            '</div>'
            '</div>'
            '<div>'
            '<div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748B; margin-bottom: 0.25rem;">'
            '<span>Average</span>'
            '</div>'
            '<div style="background: #F1F5F9; border-radius: 999px; height: 7px; overflow: hidden;">'
            '<div style="background: #28C4DC; width: 35%; height: 100%; border-radius: 999px;"></div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with b_mid:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Appointment Request</div>', unsafe_allow_html=True)

        requests = [
            ("👩", "Maria Sarafat", "Cold"),
            ("👨", "Jhon Deo", "Overeating"),
        ]

        for icon, name, reason in requests:
            st.markdown(
                f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; padding-bottom: 0.55rem; border-bottom: 1px solid #F8FAFD;">'
                f'<div style="display: flex; align-items: center; gap: 10px;">'
                f'<div style="width: 36px; height: 36px; border-radius: 50%; background: #EEF4FD; display: flex; align-items: center; justify-content: center; font-size: 1rem;">{icon}</div>'
                f'<div>'
                f'<div style="font-weight: 700; font-size: 0.85rem; color: #1B2D52;">{name}</div>'
                f'<div style="font-size: 0.73rem; color: #8496B3;">{reason}</div>'
                f'</div>'
                f'</div>'
                f'<div style="display: flex; gap: 6px;">'
                f'<span style="background: #D1FAE5; color: #059669; width: 26px; height: 26px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.8rem; cursor: pointer; font-weight: 800;">✓</span>'
                f'<span style="background: #FEE2E2; color: #DC2626; width: 26px; height: 26px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.8rem; cursor: pointer; font-weight: 800;">✕</span>'
                f'<span style="background: #E0E7FF; color: #3730A3; width: 26px; height: 26px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.8rem; cursor: pointer;">💬</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="text-align: center; margin-top: 0.8rem;">'
            '<a style="font-size: 0.82rem; color: #1E50BA; font-weight: 700; text-decoration: none; cursor: pointer;">See All</a>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with b_right:
        st.markdown('<div class="white-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem;">'
            '<span class="panel-title" style="margin-bottom: 0;">Calander</span>'
            '<span style="font-size: 0.82rem; color: #1E50BA; font-weight: 700;">December - 2021</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="display: grid; grid-template-columns: repeat(7, 1fr); text-align: center; font-size: 0.74rem; color: #8496B3; font-weight: 600; margin-bottom: 0.6rem;">'
            '<span>Sa</span><span>Su</span><span>Mo</span><span>Tu</span><span>We</span><span>Th</span><span>Fr</span>'
            '</div>'
            '<div style="display: grid; grid-template-columns: repeat(7, 1fr); text-align: center; font-size: 0.78rem; row-gap: 8px; color: #475569; font-weight: 500;">'
            '<span>18</span><span>19</span><span>20</span>'
            '<span style="background: #1E50BA; color: white; border-radius: 50%; width: 25px; height: 25px; display: inline-flex; align-items: center; justify-content: center; margin: 0 auto; font-weight: 700;">21</span>'
            '<span>22</span><span>23</span><span>24</span>'
            '<span>25</span><span>26</span><span>27</span><span>28</span><span>29</span><span>30</span><span>31</span>'
            '<span style="color: #CBD5E1;">01</span><span style="color: #CBD5E1;">02</span><span style="color: #CBD5E1;">03</span><span style="color: #CBD5E1;">04</span><span style="color: #CBD5E1;">05</span><span style="color: #CBD5E1;">06</span><span style="color: #CBD5E1;">07</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Full Cohort Analytics preserved inside expander
    with st.expander("📊 Full Synthea Cohort Analytics & Comorbidity Matrix", expanded=False):
        c_left, c_right = st.columns(2)
        with c_left:
            comorb = compute_comorbidity_matrix(tables["conditions"])
            fig_heat = px.imshow(
                comorb,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Chronic Disease Comorbidity Overlap (Shared Patients)",
            )
            st.plotly_chart(style_fig(fig_heat, 340), use_container_width=True)
        with c_right:
            fig_age = px.histogram(
                stats["age"],
                x="AGE",
                color="GENDER",
                nbins=30,
                color_discrete_sequence=[COLORS["blue"], "#F2A922"],
                title="Cohort Age & Gender Distribution",
            )
            st.plotly_chart(style_fig(fig_age, 340), use_container_width=True)


def render_profile(slice_):
    p = slice_["patient"]
    obs = slice_["observations"]

    # Header Card
    st.markdown(
        f'<div class="section-card">'
        f'<div style="display: flex; align-items: center; justify-content: space-between;">'
        f'<div><h2 style="margin: 0; color: {COLORS["navy"]};">👤 {p["DISPLAY_NAME"]}</h2>'
        f'<span style="color: {COLORS["muted"]}; font-size: 0.88rem;">Patient ID: <code>{p["PATIENT"]}</code> · '
        f'{p["GENDER"]} · {p["AGE"]} Years Old · {str(p["RACE"]).title()}</span></div>'
        f'<div><span class="pill {"pill-danger" if p["DECEASED"] else "pill-ok"}" style="font-size: 0.9rem;">'
        f'{"Deceased" if p["DECEASED"] else "Active / Living"}</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Diagnoses summary pills
    rec = recorded_diagnoses(slice_["conditions"])
    pills = ["<b style='font-size: 0.88rem; margin-right: 0.5rem;'>EHR Diagnoses:</b>"]
    for disease, info in rec.items():
        cls = "pill-navy" if info["recorded"] else "pill-gray"
        date_str = f" ({fmt_date(info['first_date'])})" if info["recorded"] and info["first_date"] else ""
        text = f"{disease}: {'Diagnosed' + date_str if info['recorded'] else 'Clean'}"
        pills.append(f'<span class="pill {cls}">{text}</span>')
    st.markdown(" ".join(pills), unsafe_allow_html=True)
    st.write("")

    # Recent Vitals & Key Clinical Indicators
    st.markdown("#### ⚡ Latest Vitals & Clinical Indicators")
    vitals_to_check = [
        ("Systolic Blood Pressure", "Systolic BP", "mmHg"),
        ("Diastolic Blood Pressure", "Diastolic BP", "mmHg"),
        ("Body mass index (BMI) [Ratio]", "BMI", "kg/m²"),
        ("Heart rate", "Heart Rate", "bpm"),
        ("Glucose [Mass/volume] in Blood", "Fasting Glucose", "mg/dL"),
        ("Hemoglobin A1c/Hemoglobin.total in Blood", "HbA1c", "%"),
        ("Creatinine [Mass/volume] in Blood", "Serum Creatinine", "mg/dL"),
        ("Glomerular filtration rate [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)/1.73 sq M", "eGFR", "mL/min"),
    ]

    latest_obs = {}
    if obs is not None and not obs.empty:
        valid_obs = obs.dropna(subset=["VALUE_NUM"]).sort_values("DATE")
        grouped = valid_obs.groupby("OBSERVATION_DESC").last()
        for idx, row in grouped.iterrows():
            latest_obs[idx] = (row["VALUE_NUM"], row["DATE"])

    vital_cols = st.columns(4)
    for i, (lab_key, short_lbl, unit) in enumerate(vitals_to_check):
        with vital_cols[i % 4]:
            if lab_key in latest_obs:
                val, dt = latest_obs[lab_key]
                status_txt, status_cls = get_lab_status(lab_key, val)
                metric_card(
                    f"{short_lbl}",
                    f"{val:.1f} {unit}",
                    status_cls,
                    f"{status_txt} · {fmt_date(dt)}",
                )
            else:
                metric_card(f"{short_lbl}", "Not Tested", "lavender", "No record in chart")
        if (i + 1) % 4 == 0 and i + 1 < len(vitals_to_check):
            st.write("")

    st.write("")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Demographics & Personal Info")
        demo = pd.DataFrame(
            {
                "Attribute": ["Birth Date", "Death Date", "Biological Sex", "Ethnicity", "Race", "Marital Status", "City", "County", "State", "ZIP"],
                "Details": [
                    fmt_date(p["BIRTHDATE"]),
                    fmt_date(p["DEATHDATE"]),
                    "Female" if p["GENDER"] == "F" else "Male",
                    str(p["ETHNICITY"]).title(),
                    str(p["RACE"]).title(),
                    str(p["MARITAL"]).title() if pd.notna(p["MARITAL"]) else "—",
                    str(p["CITY"]),
                    str(p["COUNTY"]),
                    str(p["STATE"]),
                    str(p["ZIP"]),
                ],
            }
        )
        st.dataframe(demo, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 💰 Healthcare Utilization & Costs")
        util = pd.DataFrame(
            {
                "Metric": ["Clinical Encounters", "Condition Rows", "Medications Prescribed", "Observations Tracked", "Synthetic Income", "Lifetime Health Expenses", "Insurance Coverage"],
                "Value": [
                    f"{len(slice_['encounters']):,}",
                    f"{len(slice_['conditions']):,}",
                    f"{len(slice_['medications']):,}",
                    f"{len(slice_['observations']):,}",
                    f"${p['INCOME']:,.0f}" if pd.notna(p["INCOME"]) else "—",
                    f"${p['HEALTHCARE_EXPENSES']:,.2f}" if pd.notna(p["HEALTHCARE_EXPENSES"]) else "—",
                    f"${p['HEALTHCARE_COVERAGE']:,.2f}" if pd.notna(p["HEALTHCARE_COVERAGE"]) else "—",
                ],
            }
        )
        st.dataframe(util, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_history(slice_, procedures_available: bool):
    st.subheader("Clinical History & Encounters")
    if not procedures_available:
        st.markdown(
            '<div class="warn-banner"><b>Note on Procedures:</b> '
            "<code>Data/synthea/csv/procedures.csv</code> is not present in this dataset bundle, so surgical procedure history is omitted.</div>",
            unsafe_allow_html=True,
        )

    tabs = st.tabs(["🩺 Diagnosed Conditions", "💊 Prescribed Medications", "🏥 Clinical Encounters"])

    with tabs[0]:
        df = slice_["conditions"].sort_values("START", ascending=False).copy()
        if df.empty:
            st.info("No conditions recorded for this patient.")
        else:
            df["Status"] = np.where(df["STOP"].isna(), "Active", "Resolved")
            active_count = (df["Status"] == "Active").sum()
            resolved_count = (df["Status"] == "Resolved").sum()

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                metric_card("Total Recorded", len(df), "blue")
            with mc2:
                metric_card("Active Conditions", active_count, "ok" if active_count == 0 else "gold")
            with mc3:
                metric_card("Resolved / Past", resolved_count, "lavender")

            filter_status = st.radio("Filter Status:", ["All", "Active Only", "Resolved Only"], horizontal=True)
            if filter_status == "Active Only":
                df = df[df["Status"] == "Active"]
            elif filter_status == "Resolved Only":
                df = df[df["Status"] == "Resolved"]

            df["Start Date"] = df["START"].dt.strftime("%Y-%m-%d")
            df["Stop Date"] = df["STOP"].dt.strftime("%Y-%m-%d").fillna("Ongoing")
            show_df = df.rename(columns={"CONDITION_DESC": "Diagnosis"})[["Start Date", "Stop Date", "Status", "Diagnosis"]]
            st.dataframe(show_df, hide_index=True, use_container_width=True, height=380)

    with tabs[1]:
        df = slice_["medications"].sort_values("START", ascending=False).copy()
        if df.empty:
            st.info("No medications recorded for this patient.")
        else:
            df["Status"] = np.where(df["STOP"].isna(), "Ongoing", "Completed")
            ongoing_count = (df["Status"] == "Ongoing").sum()
            completed_count = (df["Status"] == "Completed").sum()

            m1, m2, m3 = st.columns(3)
            with m1:
                metric_card("Total Prescriptions", len(df), "blue")
            with m2:
                metric_card("Active Medications", ongoing_count, "ok" if ongoing_count == 0 else "gold")
            with m3:
                metric_card("Completed Courses", completed_count, "lavender")

            med_filter = st.radio("Medication Filter:", ["All", "Ongoing Only", "Completed Only"], horizontal=True)
            if med_filter == "Ongoing Only":
                df = df[df["Status"] == "Ongoing"]
            elif med_filter == "Completed Only":
                df = df[df["Status"] == "Completed"]

            df["Start Date"] = df["START"].dt.strftime("%Y-%m-%d")
            df["Stop Date"] = df["STOP"].dt.strftime("%Y-%m-%d").fillna("Active")
            show = df.rename(columns={"MEDICATION_DESC": "Medication", "REASONDESCRIPTION": "Clinical Reason"})
            st.dataframe(show[["Start Date", "Stop Date", "Status", "Medication", "Clinical Reason"]], hide_index=True, use_container_width=True, height=380)

    with tabs[2]:
        df = slice_["encounters"].sort_values("START", ascending=False).copy()
        if df.empty:
            st.info("No encounters recorded for this patient.")
        else:
            total_cost = df["TOTAL_CLAIM_COST"].sum() if "TOTAL_CLAIM_COST" in df else 0.0
            emergencies = df["ENCOUNTERCLASS"].str.contains("emergency", case=False, na=False).sum()

            e1, e2, e3 = st.columns(3)
            with e1:
                metric_card("Total Visits", len(df), "blue")
            with e2:
                metric_card("Emergency Visits", emergencies, "danger" if emergencies > 0 else "ok")
            with e3:
                metric_card("Total Claim Cost", f"${total_cost:,.2f}", "gold")

            df["Start Date"] = df["START"].dt.strftime("%Y-%m-%d")
            show = df.rename(columns={"ENCOUNTERCLASS": "Class", "DESCRIPTION": "Description", "REASONDESCRIPTION": "Reason", "TOTAL_CLAIM_COST": "Cost ($)"})
            st.dataframe(show[["Start Date", "Class", "Description", "Reason", "Cost ($)"]], hide_index=True, use_container_width=True, height=380)


def render_labs(slice_):
    st.subheader("Laboratory Results & Vital Sign Trajectories")
    obs = slice_["observations"].copy()
    if obs.empty:
        st.info("No tracked lab or vital observations for this patient.")
        return

    obs["short"] = obs["OBSERVATION_DESC"].map(lambda x: LAB_SHORT_NAMES.get(x, x))
    available_labs = [LAB_SHORT_NAMES.get(x, x) for x in SELECTED_LABS if x in set(obs["OBSERVATION_DESC"])]
    default_labs = [LAB_SHORT_NAMES[v] for v in VITAL_LABS if LAB_SHORT_NAMES.get(v) in available_labs][:4] or available_labs[:3]

    picked = st.multiselect("Select Laboratory or Vital Parameters to Trend:", available_labs, default=default_labs)
    plot_df = obs[obs["short"].isin(picked)].dropna(subset=["VALUE_NUM"]).sort_values("DATE")

    if not plot_df.empty:
        fig = px.line(
            plot_df,
            x="DATE",
            y="VALUE_NUM",
            color="short",
            markers=True,
            color_discrete_sequence=PLOTLY_SEQUENCE,
            title="Longitudinal Values Over Time",
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Reported Value")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    else:
        st.info("No numeric values found for the selected series.")

    st.write("#### 📑 Latest Values & Clinical Reference Status")
    latest = (
        obs.dropna(subset=["VALUE_NUM"])
        .sort_values("DATE")
        .groupby("OBSERVATION_DESC", as_index=False)
        .last()[["OBSERVATION_DESC", "DATE", "VALUE_NUM", "UNITS"]]
    )
    latest["Lab / Vital"] = latest["OBSERVATION_DESC"].map(lambda x: LAB_SHORT_NAMES.get(x, x))
    latest["Date Recorded"] = latest["DATE"].dt.strftime("%Y-%m-%d")

    status_col = []
    normal_col = []
    for _, r in latest.iterrows():
        orig_name = r["OBSERVATION_DESC"]
        val = r["VALUE_NUM"]
        stxt, scls = get_lab_status(orig_name, val)
        status_col.append(stxt)
        ref = LAB_REFERENCE_RANGES.get(orig_name)
        if ref:
            min_v = ref.get("min", "—")
            max_v = ref.get("max", "—")
            normal_col.append(f"{min_v} – {max_v} {ref.get('unit', '')}")
        else:
            normal_col.append("—")

    latest["Clinical Interpretation"] = status_col
    latest["Standard Normal Range"] = normal_col
    latest["Value"] = latest["VALUE_NUM"].round(2)
    latest["Units"] = latest["UNITS"].fillna("—")

    show_tbl = latest[["Lab / Vital", "Value", "Units", "Standard Normal Range", "Clinical Interpretation", "Date Recorded"]]
    st.dataframe(show_tbl, hide_index=True, use_container_width=True)


def render_timeline(slice_):
    st.subheader("Chronological Patient Care Journey")
    rows = []
    for _, r in slice_["conditions"].iterrows():
        rows.append({"Date": r["START"], "Category": "Condition", "Event": r["CONDITION_DESC"]})
    for _, r in slice_["medications"].iterrows():
        rows.append({"Date": r["START"], "Category": "Medication", "Event": r["MEDICATION_DESC"]})
    for _, r in slice_["encounters"].iterrows():
        reason = r["REASONDESCRIPTION"] if pd.notna(r["REASONDESCRIPTION"]) else r["DESCRIPTION"]
        rows.append({"Date": r["START"], "Category": f"Encounter ({r['ENCOUNTERCLASS']})", "Event": str(reason)})

    if not rows:
        st.info("No dated clinical events recorded for this patient.")
        return

    tl = pd.DataFrame(rows).dropna(subset=["Date"]).sort_values("Date")
    categories = sorted(tl["Category"].unique())
    selected_cats = st.multiselect("Filter Event Categories:", categories, default=categories)
    tl_filtered = tl[tl["Category"].isin(selected_cats)]

    fig = px.scatter(
        tl_filtered,
        x="Date",
        y="Category",
        color="Category",
        hover_data=["Event"],
        color_discrete_sequence=PLOTLY_SEQUENCE,
        title="Interactive Clinical Event Timeline",
    )
    fig.update_traces(marker=dict(size=11, opacity=0.85))
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    st.write("#### 📜 Event Log")
    show = tl_filtered.copy()
    show["Date"] = show["Date"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(show.sort_values("Date", ascending=False), hide_index=True, use_container_width=True, height=380)


def render_assessment(slice_, model_pack):
    st.subheader("Clinical Risk Assessment & Machine Learning Decision Support")
    st.markdown(
        '<div class="warn-banner"><b>Clinical Disclaimer:</b> '
        "These predictive risk scores are computed using scikit-learn models (Random Forest and Logistic Regression) "
        "trained on synthetic MITRE Synthea records. Models predict 2-year lookahead onset. "
        "<b>These scores are for data science demonstration only and must not be used for actual medical treatment.</b></div>",
        unsafe_allow_html=True,
    )

    snapshot = choose_snapshot(slice_["patient"], slice_["encounters"], slice_["observations"])
    rec = recorded_diagnoses(slice_["conditions"])
    st.caption(f"Feature Snapshot Date: <b>{fmt_date(snapshot)}</b> (patient's latest clinical encounter)", unsafe_allow_html=True)

    bundles = model_pack["bundles"]
    errors = model_pack["errors"]
    cols = st.columns(5)
    results = {}

    for i, disease in enumerate(TARGETS_FINAL):
        with cols[i]:
            if disease in errors:
                metric_card(disease, "Unavailable", "gold")
                st.caption(errors[disease])
                continue
            scored = score_patient(
                disease,
                bundles[disease],
                slice_["patient"],
                slice_["conditions"],
                slice_["medications"],
                slice_["encounters"],
                slice_["observations"],
                snapshot,
            )
            results[disease] = scored
            if not scored["ok"]:
                metric_card(disease, "Skipped", "gold")
                st.caption("; ".join(scored["notes"].get("missing_required", ["Could not score"])))
                continue

            prob = scored["rf_prob"]
            pct = f"{prob * 100:.1f}%"
            if prob >= 0.70:
                tier, variant = "High Risk", "danger"
            elif prob >= 0.30:
                tier, variant = "Moderate Risk", "gold"
            else:
                tier, variant = "Low Risk", "ok"

            is_rec = rec[disease]["recorded"]
            subtext = "Recorded in EHR" if is_rec else tier
            metric_card(f"{disease}", pct, variant, subtext)

    st.write("")
    st.write("### 🔍 Model Interpretability & Feature Contributions")

    for disease, scored in results.items():
        if disease in errors or not scored.get("ok"):
            continue
        prob = scored["rf_prob"]
        is_rec = rec[disease]["recorded"]
        badge_cls = "risk-high" if prob >= 0.70 else ("risk-mod" if prob >= 0.30 else "risk-low")
        status_lbl = "Recorded in Chart" if is_rec else ("High Risk" if prob >= 0.70 else "Routine")

        with st.expander(f"{disease} — RF Risk: {prob*100:.1f}% ({status_lbl})", expanded=False):
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Random Forest Probability", f"{scored['rf_prob']:.3f}")
            with col_m2:
                st.metric("Logistic Regression Probability", f"{scored['logreg_prob']:.3f}")
            with col_m3:
                st.metric("Training Test ROC-AUC", f"{NOTEBOOK_METRICS[disease]['rf_auc']:.3f}")
            with col_m4:
                st.metric("Training Cohort Cases", f"{NOTEBOOK_METRICS[disease]['cases']:,}")

            if is_rec:
                st.info(
                    f"✓ <b>Synthea chart diagnosis recorded on {fmt_date(rec[disease]['first_date'])}</b>: "
                    + ", ".join(rec[disease]["labels"])
                )

            if scored.get("top_features"):
                feat_df = pd.DataFrame(scored["top_features"])
                feat_df["feature_name"] = feat_df["feature"].map(lambda x: LAB_SHORT_NAMES.get(x, x))
                fig = px.bar(
                    feat_df.sort_values("importance"),
                    x="importance",
                    y="feature_name",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Blues",
                    hover_data=["value"],
                    title=f"Top Features Driving {disease} Prediction (RF Global Importance)",
                )
                fig.update_layout(showlegend=False, height=270)
                st.plotly_chart(style_fig(fig, 270), use_container_width=True)

    # What-If Interactive Simulator
    st.write("")
    st.markdown('<div class="sim-box">', unsafe_allow_html=True)
    st.write("### ⚡ Interactive 'What-If' Clinical Intervention Simulator")
    st.caption("Simulate clinical modifications (e.g., glycemic control, weight loss, anti-hypertensive therapy) and see real-time impact on predicted risks.")

    first_scored = next((s for s in results.values() if s.get("ok")), None)
    if first_scored:
        base_row = first_scored["row"]
        curr_bmi = float(base_row.get("Body mass index (BMI) [Ratio]", 26.0))
        if curr_bmi <= 0:
            curr_bmi = 26.0
        curr_sbp = float(base_row.get("Systolic Blood Pressure", 120.0))
        if curr_sbp <= 0:
            curr_sbp = 120.0
        curr_gluc = float(base_row.get("Glucose [Mass/volume] in Blood", 95.0))
        if curr_gluc <= 0:
            curr_gluc = 95.0
        curr_hba1c = float(base_row.get("Hemoglobin A1c/Hemoglobin.total in Blood", 5.5))
        if curr_hba1c <= 0:
            curr_hba1c = 5.5

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            sim_bmi = st.slider("Simulate BMI", 16.0, 48.0, float(np.clip(curr_bmi, 16.0, 48.0)), 0.5)
        with sc2:
            sim_sbp = st.slider("Simulate Systolic BP", 85.0, 210.0, float(np.clip(curr_sbp, 85.0, 210.0)), 1.0)
        with sc3:
            sim_gluc = st.slider("Simulate Glucose (mg/dL)", 60.0, 320.0, float(np.clip(curr_gluc, 60.0, 320.0)), 2.0)
        with sc4:
            sim_hba1c = st.slider("Simulate HbA1c (%)", 4.0, 14.0, float(np.clip(curr_hba1c, 4.0, 14.0)), 0.1)

        sim_overrides = {
            "Body mass index (BMI) [Ratio]": sim_bmi,
            "Systolic Blood Pressure": sim_sbp,
            "Glucose [Mass/volume] in Blood": sim_gluc,
            "Hemoglobin A1c/Hemoglobin.total in Blood": sim_hba1c,
        }

        sim_rows = []
        for disease, scored in results.items():
            if not scored.get("ok"):
                continue
            base_prob = scored["rf_prob"]
            sim_res = score_feature_vector(bundles[disease], scored["row"], sim_overrides)
            new_prob = sim_res["rf_prob"]
            delta = (new_prob - base_prob) * 100

            sim_rows.append(
                {
                    "Condition": disease,
                    "Baseline Risk": f"{base_prob * 100:.1f}%",
                    "Simulated Risk": f"{new_prob * 100:.1f}%",
                    "Net Delta": f"{'+' if delta > 0 else ''}{delta:.1f}%",
                    "Impact": "Increased Risk" if delta > 0.5 else ("Reduced Risk" if delta < -0.5 else "Neutral"),
                }
            )

        sim_df = pd.DataFrame(sim_rows)
        st.dataframe(sim_df, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_cohort_screening(tables, model_pack):
    st.subheader("Population Health Risk Screening")
    st.markdown(
        '<div class="info-banner">Screen the entire cohort to identify high-risk patients who may benefit from early intervention. '
        "You can filter for patients with elevated machine learning risk scores who do NOT yet have an official diagnosis in their record.</div>",
        unsafe_allow_html=True,
    )

    patients = tables["patients"]
    conditions = tables["conditions"]
    target_disease = st.selectbox("Select Target Condition for Cohort Screening:", list(TARGETS_FINAL.keys()))

    # Known diagnoses
    disease_terms = TARGETS_FINAL[target_disease]
    diagnosed_pts = set(conditions.loc[conditions["CONDITION_DESC"].isin(disease_terms), "PATIENT"].unique())

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Cohort Size", len(patients), "blue")
    with c2:
        metric_card(f"Diagnosed {target_disease}", len(diagnosed_pts), "gold")
    with c3:
        metric_card("Undiagnosed Cohort", len(patients) - len(diagnosed_pts), "lavender")

    st.write("#### 📋 Cohort Patient Directory")
    filter_mode = st.radio("Cohort Filter:", ["All Patients", f"Diagnosed {target_disease} Only", f"Undiagnosed Only"], horizontal=True)

    df_pts = patients.copy()
    df_pts["Diagnosed"] = df_pts["PATIENT"].isin(diagnosed_pts)

    if filter_mode == f"Diagnosed {target_disease} Only":
        df_pts = df_pts[df_pts["Diagnosed"]]
    elif filter_mode == "Undiagnosed Only":
        df_pts = df_pts[~df_pts["Diagnosed"]]

    display_df = df_pts[["PATIENT", "DISPLAY_NAME", "AGE", "GENDER", "RACE", "CITY", "Diagnosed"]].copy()
    display_df = display_df.rename(columns={"DISPLAY_NAME": "Name", "PATIENT": "Patient ID", "AGE": "Age", "GENDER": "Sex", "RACE": "Race", "CITY": "City"})
    st.dataframe(display_df.head(100), hide_index=True, use_container_width=True, height=420)
    st.caption("Showing first 100 matching patient records.")


def main():
    avatar_b64 = get_doctor_avatar_base64()
    img_tag = f'<img src="data:image/jpeg;base64,{avatar_b64}" style="width: 100%; height: 100%; object-fit: cover;" />' if avatar_b64 else '<span style="font-size: 2.5rem;">👨‍⚕️</span>'

    # Doctor Profile Header matching reference mockup
    st.sidebar.markdown(
        f'<div style="text-align: center; padding: 0.6rem 0.5rem 1.1rem 0.5rem; margin-bottom: 0.8rem; border-bottom: 1px solid #EDF2F8;">'
        f'<div style="width: 86px; height: 86px; margin: 0 auto 0.65rem auto; border-radius: 50%; overflow: hidden; border: 3px solid #1E50BA; box-shadow: 0 4px 16px rgba(30, 80, 186, 0.16); display: flex; align-items: center; justify-content: center; background: #EEF4FD;">'
        f'{img_tag}'
        f'</div>'
        f'<div style="color: #1A3E8A !important; font-weight: 800; font-size: 1.12rem; margin-bottom: 0.15rem;">Dr. Marttin Deo</div>'
        f'<div style="color: #7E90B0 !important; font-size: 0.72rem; font-weight: 600; line-height: 1.3;">MBBS, FCPS - MD (Medicine), MCPS</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    nav_options = [
        "⊞ Dashboard",
        "📅 Appointment",
        "📑 Appointment Page",
        "💳 Payment",
        "👤 Profile",
        "📈 Labs & Trends",
        "🧬 Disease AI Risk",
        "⚙️ Settings",
    ]

    page = st.sidebar.radio(
        "Navigation",
        nav_options,
        label_visibility="collapsed",
    )

    try:
        tables = load_tables()
        model_pack = load_model_bundles()
    except Exception as exc:
        st.error(f"Could not load project dataset or model files: {exc}")
        st.stop()

    patients = tables["patients"]
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Patient Selection")

    # Filters
    age_filter = st.sidebar.selectbox("Age Group Filter:", ["All Ages", "Pediatric (< 18)", "Adults (18 - 64)", "Seniors (65+)"])
    filtered_pts = patients
    if age_filter == "Pediatric (< 18)":
        filtered_pts = filtered_pts[filtered_pts["AGE"] < 18]
    elif age_filter == "Adults (18 - 64)":
        filtered_pts = filtered_pts[(filtered_pts["AGE"] >= 18) & (filtered_pts["AGE"] < 65)]
    elif age_filter == "Seniors (65+)":
        filtered_pts = filtered_pts[filtered_pts["AGE"] >= 65]

    search_query = st.sidebar.text_input("Search Name, ID, or City:")
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_pts = filtered_pts[filtered_pts["SEARCH_TEXT"].str.contains(q, regex=False, na=False)]

    if filtered_pts.empty:
        st.sidebar.warning("No patients match filters. Resetting to full cohort.")
        filtered_pts = patients

    sorted_pts = filtered_pts.sort_values(["LAST", "FIRST"])
    pt_options = sorted_pts["PATIENT"].tolist()
    pt_labels = [
        f"{r.LAST}, {r.FIRST} · {r.AGE}y {r.GENDER} ({r.CITY})"
        for r in sorted_pts.itertuples(index=False)
    ]

    # Random patient button
    if "selected_pt_idx" not in st.session_state:
        st.session_state.selected_pt_idx = 0

    if st.sidebar.button("🎲 Pick Random Patient"):
        st.session_state.selected_pt_idx = int(np.random.randint(0, len(pt_options)))

    curr_idx = min(st.session_state.selected_pt_idx, len(pt_options) - 1)
    choice = st.sidebar.selectbox(
        "Active Patient:",
        range(len(pt_options)),
        index=curr_idx,
        format_func=lambda i: pt_labels[i],
    )
    st.session_state.selected_pt_idx = choice

    selected_patient_id = pt_options[choice]
    slice_ = patient_slice(tables, selected_patient_id)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div style="font-size: 0.88rem; font-weight: 700; color: #1B2D52; margin-bottom: 0.4rem;">↗ <a href="#" style="color: #1B2D52; text-decoration: none;">Visit site</a></div>'
        f'<div style="font-size: 0.74rem; color: #8496B3;">'
        f'Cohort: {len(patients):,} patients<br>'
        f'Data: Synthea Massachusetts'
        f'</div>',
        unsafe_allow_html=True,
    )

    if model_pack["errors"]:
        st.sidebar.error("Warning: Some models failed to load.")

    if page == "⊞ Dashboard":
        render_overview(tables, slice_)
    elif page == "📅 Appointment":
        render_timeline(slice_)
    elif page == "📑 Appointment Page":
        render_profile(slice_)
    elif page == "💳 Payment":
        render_history(slice_, tables["procedures_available"])
    elif page == "👤 Profile":
        render_profile(slice_)
    elif page == "📈 Labs & Trends":
        render_labs(slice_)
    elif page == "🧬 Disease AI Risk":
        render_assessment(slice_, model_pack)
    elif page == "⚙️ Settings":
        st.subheader("System & Model Configuration")
        st.info("System running in high-performance mode with Parquet cached data.")
        st.json({"Data Directory": str(DATA_DIR), "Models Loaded": list(model_pack["bundles"].keys())})



if __name__ == "__main__":
    main()
