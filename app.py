"""
Customer Churn Prediction - Streamlit Web Application
ML Assignment 2 - M.Tech (AIML/DSE)
BITS Pilani - Work Integrated Learning Programmes Division
Student: Akshaya Basayya Hiremath (2025AA05308)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor | BITS Pilani",
    page_icon="https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/http://coursera-university-assets.s3.amazonaws.com/b9/c608c79b5c498a8fa55b117fc3282f/5.-Square-logo-for-landing-page---Alpha.png?auto=format%2Ccompress&dpr=1&w=56&h=56",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide default Streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Remove top padding from main block ── */
.block-container { padding-top: 1rem !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 40%, #2c3e50 100%);
    border-radius: 16px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.2rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 0.4rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(52,152,219,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-logo { width: 70px; height: 70px; border-radius: 10px; z-index: 1; }
.hero-text { z-index: 1; }
.hero-text h1 {
    margin: 0 0 0.2rem 0;
    font-size: 1.65rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.hero-text p {
    margin: 0;
    font-size: 0.9rem;
    color: #94a3b8;
    font-weight: 400;
}
.hero-text .student-info {
    display: inline-block;
    margin-top: 0.5rem;
    padding: 0.2rem 0.7rem;
    background: rgba(52,152,219,0.15);
    border: 1px solid rgba(52,152,219,0.3);
    border-radius: 20px;
    font-size: 0.78rem;
    color: #5dade2;
    font-weight: 500;
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #3498db;
    display: inline-block;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    border-color: #3498db;
}
.metric-card .metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.metric-card .metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #1a1a2e;
}
.metric-card .metric-bar {
    margin-top: 0.6rem;
    height: 4px;
    background: #e2e8f0;
    border-radius: 2px;
    overflow: hidden;
}
.metric-card .metric-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.8s ease;
}

/* ── Comparison Table ── */
.comparison-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    margin: 1rem 0;
    font-size: 0.88rem;
}
.comparison-table thead th {
    background: linear-gradient(135deg, #0d1b2a, #1b2838);
    color: #ffffff;
    padding: 0.85rem 1rem;
    font-weight: 600;
    text-align: center;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
}
.comparison-table tbody td {
    padding: 0.75rem 1rem;
    text-align: center;
    border-bottom: 1px solid #f1f5f9;
    color: #334155;
    font-weight: 500;
}
.comparison-table tbody tr { background: #ffffff; transition: background 0.2s ease; }
.comparison-table tbody tr:nth-child(even) { background: #f8fafc; }
.comparison-table tbody tr:hover { background: #eef6ff; }
.comparison-table tbody tr.best-row { background: #ecfdf5 !important; }
.comparison-table tbody tr.best-row td { color: #065f46; font-weight: 600; }
.comparison-table .model-name { text-align: left; font-weight: 600; }
.badge-best {
    display: inline-block;
    background: #10b981;
    color: white;
    font-size: 0.65rem;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    margin-left: 0.4rem;
    font-weight: 700;
    vertical-align: middle;
}

/* ── Upload Zone ── */
.upload-zone {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 2px dashed #3498db;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    transition: all 0.3s ease;
}
.upload-zone:hover { border-color: #2980b9; background: #e0f2fe; }
.upload-zone h3 { color: #1e40af; margin: 0 0 0.5rem 0; font-size: 1.15rem; }
.upload-zone p { color: #64748b; margin: 0; font-size: 0.9rem; }

/* ── Info Cards ── */
.info-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.info-card h4 { margin: 0 0 0.5rem 0; color: #1a1a2e; font-size: 1rem; }
.info-card p { margin: 0; color: #64748b; font-size: 0.88rem; line-height: 1.6; }

/* ── Download Button ── */
.download-section {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid #93c5fd;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.download-section h4 { color: #1e40af; margin: 0 0 0.5rem 0; }
.download-section p { color: #64748b; font-size: 0.85rem; margin: 0 0 1rem 0; }

/* ── Sidebar Styling ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown h4 { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li { color: #94a3b8 !important; }
section[data-testid="stSidebar"] label { color: #cbd5e1 !important; }

/* ── Sidebar Dropdown ── */
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.1) !important;
    border: 2px solid rgba(52,152,219,0.5) !important;
    border-radius: 10px !important;
    min-height: 54px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease;
    overflow: visible !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div:hover {
    border-color: #3498db !important;
    background: rgba(52,152,219,0.15) !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div[data-baseweb="tag"] span,
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stSelectbox > div > div > div,
section[data-testid="stSidebar"] .stSelectbox > div > div > div > div,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div > div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    opacity: 1 !important;
    overflow: visible !important;
    text-overflow: unset !important;
    white-space: nowrap !important;
}
section[data-testid="stSidebar"] .stSelectbox svg {
    fill: #5dade2 !important;
    width: 22px !important;
    height: 22px !important;
}

/* ── Make sidebar wider to fit dropdown text ── */
section[data-testid="stSidebar"] > div { width: 320px !important; }

/* ── Dropdown popup ── */
[data-baseweb="popover"] {
    border-radius: 12px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.3) !important;
    overflow: hidden;
}
[data-baseweb="popover"] ul {
    background: #1b2838 !important;
    border: 1px solid rgba(52,152,219,0.3) !important;
    border-radius: 12px !important;
    padding: 0.4rem !important;
}
[data-baseweb="popover"] li {
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
    margin: 0.15rem 0 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: all 0.15s ease;
}
[data-baseweb="popover"] li:hover {
    background: rgba(52,152,219,0.2) !important;
    color: #ffffff !important;
}
[data-baseweb="popover"] li[aria-selected="true"] {
    background: rgba(52,152,219,0.3) !important;
    color: #5dade2 !important;
    font-weight: 700 !important;
}

/* ── Footer ── */
.app-footer {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 3rem;
    color: #94a3b8;
    font-size: 0.85rem;
}

/* ── Reduce bottom padding ── */
.block-container { padding-bottom: 1rem !important; }
.app-footer a { color: #5dade2; text-decoration: none; }

/* ── Tabs as Buttons ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.6rem;
    background: linear-gradient(135deg, #0d1b2a, #1b2838);
    padding: 0.5rem 0.6rem;
    border-radius: 12px;
    border: 1px solid rgba(52,152,219,0.2);
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}
.stTabs [data-baseweb="tab-list"] button {
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: #94a3b8 !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.02em;
}
.stTabs [data-baseweb="tab-list"] button:hover {
    background: rgba(52,152,219,0.12) !important;
    color: #e2e8f0 !important;
    border-color: rgba(52,152,219,0.3) !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #3498db, #2980b9) !important;
    color: #ffffff !important;
    border-color: transparent !important;
    box-shadow: 0 4px 12px rgba(52,152,219,0.35) !important;
}
/* Hide default tab underline */
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #64748b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# BITS LOGO URL
# ─────────────────────────────────────────────────────────────
BITS_LOGO = "https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/http://coursera-university-assets.s3.amazonaws.com/b9/c608c79b5c498a8fa55b117fc3282f/5.-Square-logo-for-landing-page---Alpha.png?auto=format%2Ccompress&dpr=1&w=180&h=180"

# ─────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <img src="{BITS_LOGO}" class="hero-logo" alt="BITS Pilani">
    <div class="hero-text">
        <h1 style="font-size:1.1rem;font-weight:700;margin:0;">M.Tech (AIML / DSE) &bull; Machine Learning &bull; Assignment - 2</h1>
        <p style="font-size:0.92rem;color:#cbd5e1;margin:0.2rem 0;">Machine Learning Classification Models</p>
        <span class="student-info">Akshaya Basayya Hiremath (2025AA05308)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbor": "knn.pkl",
    "Naive Bayes (Gaussian)": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

MODEL_INFO = {
    "Logistic Regression": {
        "type": "Linear",
        "scaling": True,
        "desc": "A linear model that uses the logistic (sigmoid) function to estimate the probability of the positive class. Works well when the decision boundary is approximately linear.",
        "icon": "📈"
    },
    "Decision Tree": {
        "type": "Tree-based",
        "scaling": False,
        "desc": "A non-linear, interpretable model that recursively splits the feature space based on the most informative features. Prone to overfitting without pruning.",
        "icon": "🌳"
    },
    "K-Nearest Neighbor": {
        "type": "Instance-based",
        "scaling": True,
        "desc": "A lazy learner that classifies each sample based on the majority label of its k nearest neighbours in the feature space. Sensitive to the choice of k and feature scaling.",
        "icon": "🔍"
    },
    "Naive Bayes (Gaussian)": {
        "type": "Probabilistic",
        "scaling": True,
        "desc": "A probabilistic classifier based on Bayes' theorem that assumes conditional independence among features. Fast, works well for high-dimensional data.",
        "icon": "🎲"
    },
    "Random Forest": {
        "type": "Ensemble (Bagging)",
        "scaling": False,
        "desc": "An ensemble of decision trees trained on random subsets of data and features. Reduces variance through bagging and generally avoids overfitting.",
        "icon": "🌲"
    },
    "XGBoost": {
        "type": "Ensemble (Boosting)",
        "scaling": False,
        "desc": "Extreme Gradient Boosting — a powerful sequential ensemble method that builds trees to correct residual errors. Often achieves state-of-the-art results.",
        "icon": "⚡"
    }
}

BENCHMARK_METRICS = {
    "Logistic Regression":    {"Accuracy": 0.7991, "AUC": 0.8403, "Precision": 0.6426, "Recall": 0.5481, "F1": 0.5916, "MCC": 0.4621},
    "Decision Tree":          {"Accuracy": 0.7743, "AUC": 0.7647, "Precision": 0.5791, "Recall": 0.5481, "F1": 0.5632, "MCC": 0.4115},
    "K-Nearest Neighbor":     {"Accuracy": 0.7424, "AUC": 0.7603, "Precision": 0.5151, "Recall": 0.5027, "F1": 0.5088, "MCC": 0.3343},
    "Naive Bayes (Gaussian)": {"Accuracy": 0.7466, "AUC": 0.8201, "Precision": 0.5160, "Recall": 0.7326, "F1": 0.6055, "MCC": 0.4413},
    "Random Forest":          {"Accuracy": 0.7828, "AUC": 0.8275, "Precision": 0.6133, "Recall": 0.4920, "F1": 0.5460, "MCC": 0.4098},
    "XGBoost":                {"Accuracy": 0.7928, "AUC": 0.8352, "Precision": 0.6367, "Recall": 0.5107, "F1": 0.5668, "MCC": 0.4373},
}

MODEL_OBSERVATIONS = {
    "Logistic Regression": "Highest accuracy (79.91%) and best AUC (0.8403) among all models. Offers excellent balance of precision and recall with the second-best F1 score (0.5916). Strong overall performer for linearly separable patterns.",
    "Decision Tree": "Moderate accuracy (77.43%) with the lowest AUC (0.7647). Reasonable precision (0.5791) but tends to overfit. F1 score of 0.5632 and MCC of 0.4115 indicate limited generalisation capability.",
    "K-Nearest Neighbor": "Lowest accuracy (74.24%) and weakest F1 (0.5088) and MCC (0.3343). Distance-based approach struggles with the mixed feature types in this dataset despite feature scaling.",
    "Naive Bayes (Gaussian)": "Best F1 score (0.6055) and highest recall (73.26%) — catches the most churners. Slightly lower accuracy (74.66%) due to more false positives, but the best MCC (0.4413) after Logistic Regression.",
    "Random Forest": "Good accuracy (78.28%) and AUC (0.8275). Strong precision (0.6133) but conservative recall (49.20%). Ensemble bagging stabilises predictions but misses some churners.",
    "XGBoost": "Second-best accuracy (79.28%) and AUC (0.8352). Balances precision (0.6367) and recall (0.5107) well. A strong all-round ensemble boosting approach."
}

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_filename):
    """Load a trained model from the model/ directory."""
    try:
        return joblib.load(f"model/{model_filename}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load the fitted StandardScaler."""
    try:
        return joblib.load("model/scaler.pkl")
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

def preprocess_data(df):
    """Clean and encode data for prediction."""
    data = df.copy()

    # TotalCharges: convert blank strings to NaN then fill
    if 'TotalCharges' in data.columns and data['TotalCharges'].dtype == 'object':
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

    # Drop ID column
    if 'customerID' in data.columns:
        data.drop('customerID', axis=1, inplace=True)

    # Separate target if present
    y = None
    if 'Churn' in data.columns:
        y = data['Churn'].map({'Yes': 1, 'No': 0})
        data.drop('Churn', axis=1, inplace=True)

    # Encode remaining categorical columns
    from sklearn.preprocessing import LabelEncoder
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    return data, y

def compute_metrics(y_true, y_pred, y_proba):
    """Return a dict of all six evaluation metrics."""
    return {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'AUC':       roc_auc_score(y_true, y_proba),
        'Precision': precision_score(y_true, y_pred),
        'Recall':    recall_score(y_true, y_pred),
        'F1 Score':  f1_score(y_true, y_pred),
        'MCC':       matthews_corrcoef(y_true, y_pred),
    }

def metric_color(value):
    """Return a CSS colour based on metric value."""
    if value >= 0.80: return '#10b981'
    if value >= 0.60: return '#3b82f6'
    if value >= 0.40: return '#f59e0b'
    return '#ef4444'

def render_metric_cards(metrics: dict):
    """Render beautiful metric cards using HTML."""
    cards = ""
    for label, value in metrics.items():
        pct = value * 100
        color = metric_color(value)
        cards += f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value:.4f}</div>
            <div class="metric-bar">
                <div class="metric-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
            </div>
        </div>"""
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:1rem 0 0.5rem 0;">
        <img src="{BITS_LOGO}" width="90" style="border-radius:12px;">
        <h3 style="margin:0.6rem 0 0.1rem 0;color:#e2e8f0;font-size:1.1rem;">Churn Predictor</h3>
        <p style="color:#64748b;font-size:0.78rem;">BITS Pilani &bull; ML Assignment 2</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Select Classification Model")

    selected_model_name = st.selectbox(
        "Model",
        list(MODEL_OPTIONS.keys()),
        label_visibility="collapsed"
    )

    info = MODEL_INFO[selected_model_name]
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:10px;padding:1rem;margin:0.8rem 0;">
        <div style="font-size:1.4rem;margin-bottom:0.3rem;">{info['icon']}</div>
        <div style="color:#e2e8f0;font-weight:600;font-size:0.9rem;">{selected_model_name}</div>
        <div style="color:#94a3b8;font-size:0.78rem;margin-top:0.3rem;">{info['type']} {'&bull; Scaled' if info['scaling'] else ''}</div>
        <div style="color:#64748b;font-size:0.78rem;margin-top:0.5rem;line-height:1.5;">{info['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Dataset Requirements")
    st.markdown("""
    <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);border-radius:8px;padding:0.8rem;margin-top:0.4rem;">
        <p style="color:#fbbf24;font-size:0.78rem;margin:0 0 0.4rem 0;font-weight:600;">CSV Format Required</p>
        <ul style="color:#94a3b8;font-size:0.75rem;margin:0;padding-left:1.2rem;line-height:1.7;">
            <li>All 19 feature columns</li>
            <li><b style="color:#e2e8f0;">Churn</b> column (Yes/No) for evaluation</li>
            <li>Optional: <b style="color:#e2e8f0;">customerID</b> column</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN BODY — TABS
# ─────────────────────────────────────────────────────────────
tab_predict, tab_compare, tab_about = st.tabs(["Predict", "Model Comparison", "About"])

# ═════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═════════════════════════════════════════════════════════════
with tab_predict:

    st.markdown('<div class="section-header">Upload Test Data</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-zone">
        <h3>Upload your CSV test file</h3>
        <p>The file should follow the Telco Customer Churn format with all feature columns.<br>
        Include the <b>Churn</b> column (Yes / No) to see evaluation metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sample CSV download in the body
    sample_csv_path = "sample_test_data.csv"
    if os.path.exists(sample_csv_path):
        with open(sample_csv_path, "r") as f:
            sample_csv_data = f.read()
    else:
        sample_csv_data = "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn\n7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No\n"

    dl_col, up_col = st.columns([1, 2])
    with dl_col:
        st.download_button(
            label="Download Sample CSV",
            data=sample_csv_data,
            file_name="sample_test_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    with up_col:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Quick stats
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows", f"{df.shape[0]:,}")
        col_b.metric("Columns", f"{df.shape[1]}")
        col_c.metric("Model", selected_model_name)

        with st.expander("Preview uploaded data (first 10 rows)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        # Preprocess
        X, y = preprocess_data(df)

        # Load model & scaler
        model = load_model(MODEL_OPTIONS[selected_model_name])

        if model is not None:
            needs_scaling = MODEL_INFO[selected_model_name]["scaling"]
            if needs_scaling:
                scaler = load_scaler()
                X_proc = scaler.transform(X) if scaler else X
            else:
                X_proc = X

            # Predict
            y_pred = model.predict(X_proc)
            y_proba = model.predict_proba(X_proc)[:, 1]

            # ── Prediction summary ──
            st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

            churn_count = int(y_pred.sum())
            no_churn_count = len(y_pred) - churn_count
            churn_rate = churn_count / len(y_pred) * 100

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Predicted Churners", f"{churn_count:,}", delta=f"{churn_rate:.1f}%", delta_color="inverse")
            sc2.metric("Predicted Non-Churners", f"{no_churn_count:,}")
            sc3.metric("Total Customers", f"{len(y_pred):,}")

            # Results table
            results_df = df.copy()
            results_df['Predicted_Churn'] = np.where(y_pred == 1, 'Yes', 'No')
            results_df['Churn_Probability'] = np.round(y_proba, 4)

            with st.expander("View all predictions", expanded=False):
                st.dataframe(
                    results_df.style.applymap(
                        lambda v: 'color: #ef4444; font-weight:600' if v == 'Yes' else ('color: #10b981; font-weight:600' if v == 'No' else ''),
                        subset=['Predicted_Churn']
                    ),
                    use_container_width=True
                )

            # ── Evaluation Metrics ──
            if y is not None and y.notna().all():
                st.markdown('<div class="section-header">Evaluation Metrics</div>', unsafe_allow_html=True)
                metrics = compute_metrics(y, y_pred, y_proba)
                render_metric_cards(metrics)

                # ── Confusion Matrix & Classification Report side-by-side ──
                st.markdown('<div class="section-header">Confusion Matrix & Classification Report</div>', unsafe_allow_html=True)

                cm_col, cr_col = st.columns([1, 1])

                with cm_col:
                    cm = confusion_matrix(y, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(
                        cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No Churn', 'Churn'],
                        yticklabels=['No Churn', 'Churn'],
                        ax=ax, linewidths=0.5,
                        annot_kws={"size": 14, "weight": "bold"}
                    )
                    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
                    ax.set_title(f'{selected_model_name}', fontsize=12, fontweight='bold', pad=12)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with cr_col:
                    report = classification_report(
                        y, y_pred,
                        target_names=['No Churn', 'Churn'],
                        output_dict=True
                    )
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(
                        report_df.style.format("{:.4f}").set_properties(**{
                            'text-align': 'center',
                            'font-weight': '500'
                        }),
                        use_container_width=True
                    )

            # ── Download predictions ──
            st.markdown('<div class="section-header">Download Results</div>', unsafe_allow_html=True)
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_data,
                file_name=f"predictions_{selected_model_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        # Placeholder when no file is uploaded
        st.markdown("""
        <div class="info-card" style="text-align:center;padding:2rem;">
            <h4>No file uploaded yet</h4>
            <p>Upload a CSV file above — or download the <b>Sample CSV</b> from the sidebar to get started quickly.</p>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════
with tab_compare:

    st.markdown('<div class="section-header">Performance Comparison — All 6 Models</div>', unsafe_allow_html=True)

    # Build HTML comparison table
    header = """
    <table class="comparison-table">
    <thead>
        <tr>
            <th style="text-align:left">Model</th>
            <th>Accuracy</th><th>AUC</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>MCC</th>
        </tr>
    </thead>
    <tbody>"""

    rows = ""
    # Determine best F1 model
    best_f1_model = max(BENCHMARK_METRICS, key=lambda m: BENCHMARK_METRICS[m]['F1'])

    for model_name, m in BENCHMARK_METRICS.items():
        is_best = model_name == best_f1_model
        row_class = ' class="best-row"' if is_best else ''
        badge = '<span class="badge-best">BEST F1</span>' if is_best else ''
        rows += f"""
        <tr{row_class}>
            <td class="model-name">{model_name}{badge}</td>
            <td>{m['Accuracy']:.4f}</td>
            <td>{m['AUC']:.4f}</td>
            <td>{m['Precision']:.4f}</td>
            <td>{m['Recall']:.4f}</td>
            <td>{m['F1']:.4f}</td>
            <td>{m['MCC']:.4f}</td>
        </tr>"""

    st.markdown(header + rows + "</tbody></table>", unsafe_allow_html=True)

    # Observation table
    st.markdown('<div class="section-header">Observations on Model Performance</div>', unsafe_allow_html=True)

    obs_header = """
    <table class="comparison-table">
    <thead><tr><th style="text-align:left;width:22%;">Model</th><th style="text-align:left">Observation</th></tr></thead>
    <tbody>"""
    obs_rows = ""
    for mname, obs in MODEL_OBSERVATIONS.items():
        obs_rows += f'<tr><td class="model-name">{mname}</td><td style="text-align:left;line-height:1.6;">{obs}</td></tr>'

    st.markdown(obs_header + obs_rows + "</tbody></table>", unsafe_allow_html=True)

    # ── Visual Charts ──
    st.markdown('<div class="section-header">Visual Comparison</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    models_short = ['LR', 'DT', 'KNN', 'NB', 'RF', 'XGB']
    models_full = list(BENCHMARK_METRICS.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    with chart_col1:
        fig1, ax1 = plt.subplots(figsize=(7, 4.5))
        metric_keys = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        x = np.arange(len(models_short))
        width = 0.12
        for i, mk in enumerate(metric_keys):
            vals = [BENCHMARK_METRICS[m][mk] for m in models_full]
            ax1.bar(x + i * width, vals, width, label=mk, alpha=0.85)
        ax1.set_xlabel('Model', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('All Metrics by Model', fontweight='bold', fontsize=12)
        ax1.set_xticks(x + width * 2.5)
        ax1.set_xticklabels(models_short)
        ax1.legend(fontsize=7, loc='lower right')
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        f1_vals = [BENCHMARK_METRICS[m]['F1'] for m in models_full]
        bar_colors = ['#10b981' if v == max(f1_vals) else '#3498db' for v in f1_vals]
        bars = ax2.barh(models_short, f1_vals, color=bar_colors, edgecolor='white', height=0.55)
        for bar, val in zip(bars, f1_vals):
            ax2.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontweight='bold', fontsize=10)
        ax2.set_xlabel('F1 Score', fontweight='bold')
        ax2.set_title('F1 Score Comparison', fontweight='bold', fontsize=12)
        ax2.set_xlim(0, 0.75)
        ax2.grid(axis='x', alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

# ═════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═════════════════════════════════════════════════════════════
with tab_about:

    st.markdown('<div class="section-header">About This Application</div>', unsafe_allow_html=True)

    about_col1, about_col2 = st.columns([2, 1])

    with about_col1:
        st.markdown("""
        <div class="info-card">
            <h4>Problem Statement</h4>
            <p>Customer churn is a critical business challenge in the telecommunications industry. 
            This project builds and compares <b>six classification models</b> to predict whether a customer 
            is likely to churn, enabling proactive retention strategies.</p>
        </div>

        <div class="info-card">
            <h4>Dataset — Telco Customer Churn</h4>
            <p>
                <b>Source:</b> Kaggle (IBM Sample Dataset)<br>
                <b>Instances:</b> 7,043 customers<br>
                <b>Features:</b> 19 input features + 1 target (Churn)<br>
                <b>Target Distribution:</b> ~73.5% No Churn, ~26.5% Churn (imbalanced)<br>
                <b>Feature Types:</b> Demographics, account info, and service subscriptions
            </p>
        </div>

        <div class="info-card">
            <h4>Models Implemented</h4>
            <p>
                1. <b>Logistic Regression</b> — Linear classifier<br>
                2. <b>Decision Tree Classifier</b> — Tree-based<br>
                3. <b>K-Nearest Neighbor Classifier</b> — Instance-based<br>
                4. <b>Naive Bayes (Gaussian)</b> — Probabilistic<br>
                5. <b>Random Forest</b> — Ensemble (Bagging)<br>
                6. <b>XGBoost</b> — Ensemble (Boosting)
            </p>
        </div>

        <div class="info-card">
            <h4>Evaluation Metrics</h4>
            <p>Each model is evaluated using six metrics: <b>Accuracy, AUC Score, Precision, Recall, F1 Score, 
            and Matthews Correlation Coefficient (MCC)</b>. The F1 score is used as the primary comparison 
            metric due to class imbalance in the dataset.</p>
        </div>
        """, unsafe_allow_html=True)

    with about_col2:
        st.markdown(f"""
        <div class="info-card" style="text-align:center;">
            <img src="{BITS_LOGO}" width="90" style="border-radius:12px;margin-bottom:0.8rem;">
            <h4>BITS Pilani</h4>
            <p>Work Integrated Learning<br>Programmes Division</p>
            <hr style="border-color:#e2e8f0;margin:0.8rem 0;">
            <p style="font-size:0.82rem;">
                <b>Programme:</b> M.Tech (AIML / DSE)<br>
                <b>Course:</b> Machine Learning<br>
                <b>Assignment:</b> 2<br><br>
                <b>Student:</b> Akshaya Basayya Hiremath<br>
                <b>ID:</b> 2025AA05308
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h4>Tech Stack</h4>
            <p style="font-size:0.82rem;">
                Python &bull; Streamlit<br>
                scikit-learn &bull; XGBoost<br>
                pandas &bull; NumPy<br>
                Matplotlib &bull; Seaborn
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Key Findings</h4>
        <p>
            <b>Best F1 Score:</b> Naive Bayes (Gaussian) achieves the highest F1 score of <b>0.6055</b>, making it 
            the best model for identifying churners when balancing precision and recall.<br><br>
            <b>Best Accuracy & AUC:</b> Logistic Regression leads with <b>79.91%</b> accuracy and an AUC of <b>0.8403</b>, 
            demonstrating the strongest overall discriminative power.<br><br>
            <b>Highest Recall:</b> Naive Bayes captures <b>73.26%</b> of actual churners — the best among all models — 
            making it ideal when minimising missed churners is the priority.<br><br>
            <b>Class Imbalance Impact:</b> All models show moderate F1 and MCC values (0.33–0.46 MCC), reflecting 
            the challenge of the 73.5% / 26.5% class split in the Telco dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-footer">
    <img src="{BITS_LOGO}" width="36" style="border-radius:6px;vertical-align:middle;margin-right:0.6rem;">
    <b>Customer Churn Prediction System</b><br>
    <span style="color:#5dade2;">Akshaya Basayya Hiremath (2025AA05308)</span>
</div>
""", unsafe_allow_html=True)
