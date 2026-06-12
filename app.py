"""
Breast Cancer Classification System
RILWAN AMINA SHEHU | AUST Final Year Project
--------------------------------------------
Run with: streamlit run app.py
Requires: breast_cancer_model.pkl and prediction_vars.pkl in the same folder
Install : pip install streamlit scikit-learn joblib numpy pandas
"""

import streamlit as st
import numpy as np
import joblib
import os

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Classification System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM STYLING ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #1a6b8a 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 1.9rem; margin: 0; font-weight: 700; }
    .main-header p  { font-size: 0.95rem; margin: 0.4rem 0 0 0; opacity: 0.85; }

    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1a3a5c;
        border-bottom: 2px solid #1a6b8a;
        padding-bottom: 4px;
        margin: 1.2rem 0 0.8rem 0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .result-benign {
        background: #e8f5e9;
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .result-malignant {
        background: #ffebee;
        border-left: 5px solid #c62828;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .result-label { font-size: 1.6rem; font-weight: 700; margin: 0; }
    .result-sub   { font-size: 0.9rem;  margin: 0.3rem 0 0 0; color: #444; }

    .disclaimer {
        background: #fff8e1;
        border: 1px solid #f9a825;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #5d4037;
        margin-top: 1.2rem;
    }
    .metric-card {
        background: #f5f9fc;
        border: 1px solid #cde;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-val  { font-size: 1.3rem; font-weight: 700; color: #1a3a5c; }
    .metric-name { font-size: 0.78rem; color: #666; margin-top: 2px; }
    div[data-testid="stNumberInput"] label { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('breast_cancer_model.pkl')
    features = joblib.load('prediction_vars.pkl')
    return model, features

try:
    model, FEATURES = load_model()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 Breast Cancer Classification System</h1>
    <p>Design and Implementation of a Machine Learning Model for Breast Tumour Classification</p>
    <p style="font-size:0.8rem; opacity:0.7;">RILWAN AMINA SHEHU &nbsp;|&nbsp; African University of Science and Technology &nbsp;|&nbsp; Final Year Project 2026</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error("⚠️ Model files not found. Please ensure `breast_cancer_model.pkl` and `prediction_vars.pkl` are in the same directory as `app.py`, then run the notebook first.")
    st.stop()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
This system uses a **Random Forest** classifier trained on the Wisconsin Breast Cancer Diagnostic (WBCD) dataset to classify breast tumours as **Benign** or **Malignant**.

**Model Performance**
| Metric | Score |
|---|---|
| Accuracy | 94.74% |
| Precision | 93.42% |
| Recall | 98.61% |
| F1 Score | 95.95% |

**Dataset:** 569 instances · 14 selected features · 80:20 split

**Feature Selection:** Correlation-based (Pearson) — 14 of 30 features selected
    """)
    st.markdown("---")
    st.markdown("### 📌 Feature Groups")
    st.markdown("""
**Mean (5):** Radius, Perimeter, Area, Concavity, Concave Points

**Error (3):** Radius Error, Perimeter Error, Area Error

**Worst (6):** Worst Radius, Worst Perimeter, Worst Area, Worst Concavity, Worst Concave Points, Worst Compactness
    """)
    st.markdown("---")
    st.caption("Classification aid only. Always consult a qualified medical professional.")

# ── MAIN LAYOUT ────────────────────────────────────────────────────────────────
col_inputs, col_result = st.columns([3, 2], gap="large")

with col_inputs:
    st.markdown('<div class="section-title">Enter Tumour Measurements</div>', unsafe_allow_html=True)
    st.markdown("<small>Enter values from the patient's FNA biopsy report. All fields are required.</small>", unsafe_allow_html=True)

    # ── MEAN FEATURES ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="font-size:0.85rem;">Mean Features</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        mean_radius    = st.number_input("Mean Radius",          min_value=0.0, value=14.13, step=0.01, format="%.4f")
        mean_concavity = st.number_input("Mean Concavity",       min_value=0.0, value=0.0800, step=0.0001, format="%.4f")
    with c2:
        mean_perimeter = st.number_input("Mean Perimeter",       min_value=0.0, value=91.97, step=0.01, format="%.4f")
        mean_concave   = st.number_input("Mean Concave Points",  min_value=0.0, value=0.0479, step=0.0001, format="%.4f")
    with c3:
        mean_area      = st.number_input("Mean Area",            min_value=0.0, value=654.89, step=0.1, format="%.2f")

    # ── ERROR FEATURES ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="font-size:0.85rem;">Error Features</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    with e1:
        radius_error    = st.number_input("Radius Error",    min_value=0.0, value=0.4052, step=0.0001, format="%.4f")
    with e2:
        perimeter_error = st.number_input("Perimeter Error", min_value=0.0, value=2.866,  step=0.001,  format="%.4f")
    with e3:
        area_error      = st.number_input("Area Error",      min_value=0.0, value=40.34,  step=0.01,   format="%.4f")

    # ── WORST FEATURES ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="font-size:0.85rem;">Worst Features</div>', unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)
    with w1:
        worst_radius    = st.number_input("Worst Radius",          min_value=0.0, value=16.27, step=0.01,   format="%.4f")
        worst_concavity = st.number_input("Worst Concavity",       min_value=0.0, value=0.2722, step=0.0001, format="%.4f")
    with w2:
        worst_perimeter = st.number_input("Worst Perimeter",       min_value=0.0, value=107.26, step=0.01,  format="%.4f")
        worst_concave   = st.number_input("Worst Concave Points",  min_value=0.0, value=0.1147, step=0.0001, format="%.4f")
    with w3:
        worst_area      = st.number_input("Worst Area",            min_value=0.0, value=880.58, step=0.1,   format="%.2f")
        worst_compact   = st.number_input("Worst Compactness",     min_value=0.0, value=0.2116, step=0.0001, format="%.4f")

    st.markdown("")
    predict_btn = st.button("🔬 Run Classification", type="primary", use_container_width=True)

# ── RESULTS PANEL ──────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-title">Classification Result</div>', unsafe_allow_html=True)

    if predict_btn:
        # Build input array in exact feature order
        input_values = [
            mean_radius, mean_perimeter, mean_area, mean_concavity,
            mean_concave, radius_error, perimeter_error, area_error,
            worst_radius, worst_perimeter, worst_area, worst_concavity,
            worst_concave, worst_compact
        ]

        input_array = np.asarray(input_values).reshape(1, -1)
        prediction  = model.predict(input_array)[0]
        proba       = model.predict_proba(input_array)[0]
        confidence  = proba.max()

        if prediction == 1:
            st.markdown(f"""
<div class="result-benign">
    <p class="result-label" style="color:#2e7d32;">✅ BENIGN</p>
    <p class="result-sub">The model classifies this tumour as <strong>non-cancerous (benign)</strong>.</p>
    <p class="result-sub">Confidence: <strong>{confidence:.1%}</strong></p>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-malignant">
    <p class="result-label" style="color:#c62828;">⚠️ MALIGNANT</p>
    <p class="result-sub">The model classifies this tumour as <strong>cancerous (malignant)</strong>.</p>
    <p class="result-sub">Confidence: <strong>{confidence:.1%}</strong></p>
</div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Metrics
        st.markdown("**Model Details**")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{confidence:.1%}</div><div class="metric-name">Confidence Score</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-val">14</div><div class="metric-name">Features Used</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">95.95%</div><div class="metric-name">Model F1 Score</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-val">Random Forest</div><div class="metric-name">Algorithm</div></div>', unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("**Probability Breakdown**")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Malignant (0)", f"{proba[0]:.1%}")
        with prob_col2:
            st.metric("Benign (1)", f"{proba[1]:.1%}")

        st.markdown("""
<div class="disclaimer">
⚕️ <strong>Clinical Disclaimer:</strong> This result is a computational classification aid only and should not be used as a substitute for professional medical diagnosis. All predictions must be confirmed by a qualified medical professional.
</div>""", unsafe_allow_html=True)

    else:
        st.info("👈 Enter the 14 tumour measurements on the left panel and click **Run Classification** to receive a prediction.")
        st.markdown("""
**How to use this system:**
1. Enter the patient's FNA biopsy measurements across all 14 fields
2. Click **Run Classification**
3. Review the prediction and confidence score
4. All results must be confirmed by a clinician

**What the result means:**
- 🟢 **Benign** — The model predicts a non-cancerous tumour
- 🔴 **Malignant** — The model predicts a cancerous tumour
        """)

# ── FOOTER ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:0.78rem; color:#888;'>"
    "Breast Cancer Classification System &nbsp;|&nbsp; RILWAN AMINA SHEHU &nbsp;|&nbsp; "
    "African University of Science and Technology, Abuja &nbsp;|&nbsp; 2026"
    "</div>",
    unsafe_allow_html=True
)
