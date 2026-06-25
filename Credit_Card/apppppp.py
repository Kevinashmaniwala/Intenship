"""
Loan Approval Prediction System
================================
Production-ready Streamlit application — 4 Tabs:
  1. Dashboard       — KPIs, dataset preview, summary stats, charts
  2. Loan Prediction — Form-based prediction with Random Forest
  3. Visualization   — Full EDA with auto-generated insights
  4. Insights        — Business insights & conclusions for stakeholders

Dataset & Model Loading Strategy (Streamlit Cloud):
  Priority 1 → local file     (parquet or CSV + pkl)
  Priority 2 → Hugging Face Dataset Hub
  Priority 3 → Google Drive / Direct URL  (set DATASET_URL and MODEL_URL in st.secrets)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import re
import warnings
import requests
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

_BASE         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(_BASE, "loan_model.pkl")
_PARQUET_PATH = os.path.join(_BASE, "loan_clean_data.parquet")
_CSV_PATH     = os.path.join(_BASE, "cleaned_loan_data.csv")

# Cloud secrets — fill via st.secrets or leave blank
DATASET_HF_REPO = ""
DATASET_URL     = ""
MODEL_URL       = ""

try:
    DATASET_HF_REPO = st.secrets.get("DATASET_HF_REPO", DATASET_HF_REPO)
    DATASET_URL     = st.secrets.get("DATASET_URL",     DATASET_URL)
    MODEL_URL       = st.secrets.get("MODEL_URL",       MODEL_URL)
except Exception:
    pass

# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# GLOBAL STYLES
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

div[data-baseweb="tab-list"] { border-radius: 10px; padding: 4px; gap: 4px; }
div[data-baseweb="tab"] { background: transparent; border-radius: 8px; font-weight: 500; padding: 8px 20px; }
div[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white !important;
}

.kpi-card {
    background: var(--background-color, transparent);
    border: 1px solid rgba(102,126,234,0.35); border-radius: 14px;
    padding: 20px 24px; text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(102,126,234,0.25); }
.kpi-value {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2;
}
.kpi-label { font-size: 0.82rem; opacity: 0.65; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

.section-header {
    font-size: 1.15rem; font-weight: 600; padding-bottom: 8px;
    border-bottom: 2px solid #667eea; margin-bottom: 16px; display: inline-block;
}

.predict-approved { background: rgba(34,197,94,0.08); border: 1px solid #22c55e; border-radius: 14px; padding: 24px; text-align: center; }
.predict-rejected { background: rgba(239,68,68,0.08);  border: 1px solid #ef4444; border-radius: 14px; padding: 24px; text-align: center; }
.predict-title-approved { font-size: 2rem; font-weight: 700; color: #16a34a; }
.predict-title-rejected  { font-size: 2rem; font-weight: 700; color: #dc2626; }
.predict-subtitle { font-size: 0.9rem; opacity: 0.6; margin-top: 6px; }

.insight-box {
    background: rgba(102,126,234,0.07); border-left: 3px solid #667eea;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    font-size: 0.88rem; margin-top: 8px; margin-bottom: 20px; opacity: 0.85;
}

.ins-card { background: rgba(102,126,234,0.06); border: 1px solid rgba(102,126,234,0.25); border-radius: 14px; padding: 22px 26px; margin-bottom: 16px; }
.ins-card-title { font-size: 1rem; font-weight: 700; color: #7c3aed; margin-bottom: 10px; }
.ins-card-body  { font-size: 0.9rem; line-height: 1.7; opacity: 0.85; }

.highlight-green { background: rgba(34,197,94,0.08);  border: 1px solid #22c55e; border-radius: 10px; padding: 16px 20px; font-size: 0.9rem; margin-bottom: 12px; color: #15803d; }
.highlight-red   { background: rgba(239,68,68,0.08);  border: 1px solid #ef4444; border-radius: 10px; padding: 16px 20px; font-size: 0.9rem; margin-bottom: 12px; color: #b91c1c; }
.highlight-blue  { background: rgba(59,130,246,0.08); border: 1px solid #3b82f6; border-radius: 10px; padding: 16px 20px; font-size: 0.9rem; margin-bottom: 12px; color: #1d4ed8; }
.highlight-amber { background: rgba(245,158,11,0.08); border: 1px solid #f59e0b; border-radius: 10px; padding: 16px 20px; font-size: 0.9rem; margin-bottom: 12px; color: #b45309; }

.rec-pill {
    display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border-radius: 20px; padding: 4px 14px;
    font-size: 0.78rem; font-weight: 600; margin: 4px 4px 4px 0;
}
.sidebar-badge {
    display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border-radius: 20px; padding: 2px 12px; font-size: 0.75rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# GOOGLE DRIVE HELPERS
# ══════════════════════════════════════════════

def _extract_gdrive_id(url: str):
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]{25,})",
        r"id=([a-zA-Z0-9_-]{25,})",
        r"/d/([a-zA-Z0-9_-]{25,})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _gdrive_download(url: str) -> bytes:
    """Download raw bytes from any Google Drive or direct URL, bypassing virus-scan page."""
    session = requests.Session()
    file_id = _extract_gdrive_id(url)

    if not file_id:
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
        return resp.content

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(download_url, timeout=60)
    resp.raise_for_status()

    # Handle virus-scan HTML warning page
    if "text/html" in resp.headers.get("Content-Type", ""):
        token_match = re.search(r'name="uuid"\s+value="([^"]+)"', resp.text)
        if not token_match:
            token_match = re.search(r'confirm=([0-9A-Za-z_-]+)', resp.text)

        if token_match:
            confirm_url = (
                f"https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&confirm={token_match.group(1)}"
            )
        else:
            confirm_url = (
                f"https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&authuser=0"
            )
        resp = session.get(confirm_url, timeout=180)
        resp.raise_for_status()

    if "text/html" in resp.headers.get("Content-Type", ""):
        raise ValueError(
            "Google Drive returned an HTML page. "
            "Make sure the file is shared as 'Anyone with the link'."
        )

    return resp.content


def _bytes_to_dataframe(raw: bytes, url: str = "") -> pd.DataFrame:
    """Auto-detect parquet vs CSV from magic bytes, then parse."""
    is_parquet = raw[:4] == b"PAR1" or "parquet" in url.lower()
    if is_parquet:
        return pd.read_parquet(io.BytesIO(raw))
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        return pd.read_parquet(io.BytesIO(raw))


# ══════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    # Strategy 1 — Local file (parquet first, then CSV)
    for local_path in [_PARQUET_PATH, _CSV_PATH]:
        if os.path.exists(local_path):
            try:
                if local_path.endswith(".parquet"):
                    return pd.read_parquet(local_path)
                return pd.read_csv(local_path)
            except Exception as e:
                st.warning(f"Local file '{os.path.basename(local_path)}' failed: {e}")

    # Strategy 2 — Hugging Face Hub
    if DATASET_HF_REPO.strip():
        for fname in ["loan_clean_data.parquet", "cleaned_loan_data.csv", "loan_data.csv"]:
            try:
                hf_url = f"https://huggingface.co/datasets/{DATASET_HF_REPO.strip()}/resolve/main/{fname}"
                raw = _gdrive_download(hf_url)
                return _bytes_to_dataframe(raw, hf_url)
            except Exception:
                continue
        st.warning("Hugging Face load failed for all known filenames.")

    # Strategy 3 — Google Drive / Direct URL
    if DATASET_URL.strip():
        try:
            raw = _gdrive_download(DATASET_URL.strip())
            return _bytes_to_dataframe(raw, DATASET_URL.strip())
        except Exception as e:
            st.warning(f"Dataset URL download failed: {e}")

    return None


# ══════════════════════════════════════════════
# MODEL LOADER  ← FIXED: now also loads from Google Drive
# ══════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model():
    # Strategy 1 — Local pkl
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Local model found but failed to load: {e}")

    # Strategy 2 — Google Drive / Direct URL  ← NEW
    if MODEL_URL.strip():
        try:
            raw = _gdrive_download(MODEL_URL.strip())
            model = pickle.loads(raw)
            return model
        except Exception as e:
            st.error(f"Model URL download failed: {e}")

    return None


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════

def render_sidebar(df):
    with st.sidebar:
        st.markdown("## 🏦 Loan Prediction")
        st.markdown('<span class="sidebar-badge">v2.0 Production</span>', unsafe_allow_html=True)
        st.divider()

        st.markdown("### 📁 Project Info")
        st.markdown("""
| Field | Value |
|---|---|
| **Project** | Loan Approval |
| **Model** | Random Forest |
| **Dataset** | loan_clean_data.parquet |
""")
        st.divider()

        st.markdown("### 🤖 Model Info")
        col_a, col_b = st.columns(2)
        col_a.metric("Algorithm", "RF")
        col_b.metric("Status", "✅ Loaded" if load_model() else "❌ Missing")
        if df is not None:
            st.metric("Records",  f"{len(df):,}")
            st.metric("Features", df.shape[1])
        st.divider()

        st.markdown("### 🧭 Navigation")
        st.markdown("""
- **📊 Dashboard** — KPIs, preview, summary stats.
- **🔍 Prediction** — Instant loan decision form.
- **📈 Visualization** — Full EDA with insights.
- **💡 Insights** — Business conclusions & recommendations.
""")
        st.divider()
        st.caption("Built with Streamlit · Random Forest · Plotly")


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

def _insight(text: str):
    st.markdown(f'<div class="insight-box">💡 <b>Insight:</b> {text}</div>', unsafe_allow_html=True)


def _approved_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().isin(["Y", "1", "YES", "APPROVED", "1.0"])


# ══════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════

def show_dashboard(df: pd.DataFrame):
    st.markdown("## 📊 Dataset Dashboard")
    st.caption("Real-time statistics derived from your cleaned loan dataset.")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    missing  = int(df.isnull().sum().sum())

    kpis = [
        ("Total Records",    f"{len(df):,}", "📋"),
        ("Total Features",   f"{df.shape[1]}", "🔢"),
        ("Missing Values",   f"{missing}", "❓"),
        ("Numerical Cols",   f"{len(num_cols)}", "📐"),
        ("Categorical Cols", f"{len(cat_cols)}", "🏷️"),
    ]
    cols = st.columns(5)
    for col, (label, value, icon) in zip(cols, kpis):
        with col:
            st.markdown(f"""
<div class="kpi-card">
  <div style="font-size:1.6rem">{icon}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-label">{label}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📄 Dataset Preview — First 10 Rows", expanded=True):
        st.dataframe(df.head(10), use_container_width=True, height=300)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows",      df.shape[0])
        c2.metric("Columns",   df.shape[1])
        c3.metric("Size (KB)", f"{df.memory_usage(deep=True).sum()/1024:.1f}")

    with st.expander("🔍 Column Data Types"):
        dtype_df = pd.DataFrame({
            "Column":   df.columns,
            "Dtype":    df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Null":     df.isnull().sum().values,
        })
        st.dataframe(dtype_df, use_container_width=True)

    with st.expander("📈 Statistical Summary"):
        tab_n, tab_c = st.tabs(["Numerical", "Categorical"])
        with tab_n:
            st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
        with tab_c:
            obj_cols = df.select_dtypes(include="object")
            if not obj_cols.empty:
                st.dataframe(obj_cols.describe().T, use_container_width=True)
            else:
                st.info("No categorical columns found.")

    st.divider()
    st.markdown('<p class="section-header">📊 Key Distributions</p>', unsafe_allow_html=True)

    chart_cats = {k: v for k, v in {
        "Loan_Status":   "Loan Status Distribution",
        "Gender":        "Gender Distribution",
        "Education":     "Education Distribution",
        "Property_Area": "Property Area Distribution",
    }.items() if k in df.columns}

    pairs = list(chart_cats.items())
    for i in range(0, len(pairs), 2):
        row_cols = st.columns(2)
        for j, (col_name, title) in enumerate(pairs[i:i+2]):
            with row_cols[j]:
                counts = df[col_name].value_counts().reset_index()
                counts.columns = [col_name, "Count"]
                fig = px.bar(counts, x=col_name, y="Count", title=title,
                             color="Count", color_continuous_scale=["#667eea","#a78bfa"],
                             template="plotly_white")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  coloraxis_showscale=False, height=300,
                                  margin=dict(t=40,b=20,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">📉 Income & Loan Amount</p>', unsafe_allow_html=True)
    numeric_interest = [c for c in ["ApplicantIncome","CoapplicantIncome","LoanAmount"] if c in df.columns]
    if numeric_interest:
        n_cols = st.columns(len(numeric_interest))
        for col, col_name in zip(n_cols, numeric_interest):
            with col:
                fig = px.histogram(df, x=col_name, nbins=40, title=f"{col_name} Distribution",
                                   color_discrete_sequence=["#667eea"], template="plotly_white")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  height=280, margin=dict(t=40,b=20,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">🔗 Correlation Heatmap</p>', unsafe_allow_html=True)
    corr_cols = df.select_dtypes(include=[np.number])
    if corr_cols.shape[1] > 1:
        corr = corr_cols.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale="Viridis", zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}", textfont={"size":10},
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", template="plotly_white",
                          height=420, margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — LOAN PREDICTION
# ══════════════════════════════════════════════

def show_prediction_page(df: pd.DataFrame, model):
    st.markdown("## 🔍 Loan Approval Prediction")
    st.caption("Fill in the applicant details below and click **Predict** to get an instant decision.")

    if model is None:
        st.error("❌ Model could not be loaded. Please set MODEL_URL in Streamlit secrets or place loan_model.pkl in the project folder.")
        st.info("👉 Go to your Streamlit Cloud app → Settings → Secrets and add:\n\n`MODEL_URL = \"https://drive.google.com/uc?export=download&id=YOUR_MODEL_FILE_ID\"`")
        return

    model_features = None
    try:
        model_features = list(model.feature_names_in_)
    except AttributeError:
        pass
    if model_features is None:
        exclude = {"Loan_Status", "Loan_ID", "loan_status"}
        model_features = [c for c in df.columns if c not in exclude]

    cat_options = {}
    for col in df.select_dtypes(include=["object","category"]).columns:
        if col in model_features:
            cat_options[col] = sorted(df[col].dropna().unique().tolist())

    with st.form("loan_form"):
        st.markdown("### 👤 Applicant Details")
        input_data = {}
        col1, col2 = st.columns(2)
        feature_col = [col1, col2]

        for idx, feat in enumerate(model_features):
            with feature_col[idx % 2]:
                if feat in cat_options:
                    input_data[feat] = st.selectbox(
                        feat.replace("_"," "), cat_options[feat], key=f"i_{feat}")
                else:
                    col_min  = float(df[feat].min())  if feat in df.columns else 0.0
                    col_max  = float(df[feat].max())  if feat in df.columns else 1_000_000.0
                    col_mean = float(df[feat].mean()) if feat in df.columns else 0.0
                    input_data[feat] = st.number_input(
                        feat.replace("_"," "), min_value=col_min,
                        max_value=col_max, value=round(col_mean, 2), key=f"i_{feat}")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🚀 Predict Loan Status", use_container_width=True, type="primary")

    if submitted:
        try:
            input_df = pd.DataFrame([input_data])
            for col in input_df.select_dtypes(include="object").columns:
                if col in df.columns:
                    cats = sorted(df[col].dropna().unique().tolist())
                    input_df[col] = input_df[col].map({v: i for i, v in enumerate(cats)})
            input_df   = input_df[model_features]
            prediction = model.predict(input_df)[0]
            proba      = model.predict_proba(input_df)[0]
            approved   = str(prediction).strip().upper() in ["Y","1","YES","APPROVED","1.0"]
            confidence = float(max(proba)) * 100
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

        st.markdown("<br>", unsafe_allow_html=True)
        _, r2, _ = st.columns([1,2,1])
        with r2:
            if approved:
                st.markdown('<div class="predict-approved"><div class="predict-title-approved">✅ Loan Approved</div><div class="predict-subtitle">The applicant qualifies for the loan based on the submitted details.</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="predict-rejected"><div class="predict-title-rejected">❌ Loan Rejected</div><div class="predict-subtitle">The applicant does not meet the approval criteria at this time.</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Decision",   "Approved ✅" if approved else "Rejected ❌")
        m2.metric("Confidence", f"{confidence:.1f}%")
        m3.metric("Model Used", "Random Forest")

        gauge_color = "#22c55e" if approved else "#ef4444"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=confidence,
            title={"text":"Approval Probability (%)","font":{"color":"#e2e8f0","size":14}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#8b949e"},
                "bar":{"color":gauge_color},
                "bgcolor":"#1a1f2e",
                "steps":[
                    {"range":[0,40],  "color":"#2d1b1b"},
                    {"range":[40,70], "color":"#2d2b1b"},
                    {"range":[70,100],"color":"#1b2d1b"},
                ],
                "threshold":{"line":{"color":gauge_color,"width":4},"thickness":0.75,"value":confidence},
            },
            number={"suffix":"%","font":{"color":gauge_color,"size":36}},
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(t=30,b=20))
        _, g_col, _ = st.columns([1,2,1])
        with g_col:
            st.plotly_chart(fig_gauge, use_container_width=True)

        with st.expander("📋 Input Summary"):
            st.dataframe(
                pd.DataFrame(list(input_data.items()), columns=["Field","Value"]),
                use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 3 — VISUALIZATION
# ══════════════════════════════════════════════

def show_visualization_page(df: pd.DataFrame):
    st.markdown("## 📈 Exploratory Data Analysis")
    st.caption("Detailed visual analysis of the loan dataset with auto-generated insights.")

    target_col = next((c for c in ["Loan_Status","loan_status"] if c in df.columns), None)
    cat_cols   = df.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols   = df.select_dtypes(include=[np.number]).columns.tolist()

    # ── 1. Univariate ──────────────────────────
    st.markdown('<p class="section-header">1️⃣ Univariate Analysis</p>', unsafe_allow_html=True)
    plot_cats = [c for c in cat_cols if c not in {target_col, "Loan_ID"}]
    if plot_cats:
        for i in range(0, len(plot_cats), 3):
            row = st.columns(3)
            for j, col_name in enumerate(plot_cats[i:i+3]):
                with row[j]:
                    vc = df[col_name].value_counts().reset_index()
                    vc.columns = [col_name, "Count"]
                    fig = px.pie(vc, names=col_name, values="Count",
                                 title=f"{col_name} Distribution",
                                 color_discrete_sequence=px.colors.sequential.Purpor,
                                 template="plotly_white", hole=0.4)
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300,
                                      margin=dict(t=40,b=10), legend=dict(font=dict(size=10)))
                    st.plotly_chart(fig, use_container_width=True)
        _insight(f"Pie charts for {', '.join(plot_cats[:3])} reveal the dominant applicant profile.")

    st.markdown("**Numerical Feature Distributions**")
    for col_name in num_cols:
        skew_val = df[col_name].skew()
        skew_dir = ("right-skewed (positive)" if skew_val > 0.5
                    else "left-skewed (negative)" if skew_val < -0.5
                    else "approximately symmetric")
        fig = px.histogram(df, x=col_name, nbins=50, title=f"{col_name} — Distribution",
                           color_discrete_sequence=["#764ba2"], marginal="box",
                           template="plotly_white")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=320, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
        _insight(f"{col_name} is {skew_dir} (skewness {skew_val:.2f}). Consider log-transform before modelling.")

    st.divider()

    # ── 2. Bivariate ───────────────────────────
    st.markdown('<p class="section-header">2️⃣ Bivariate Analysis — Loan Status vs Key Features</p>', unsafe_allow_html=True)
    if target_col:
        for col_name in [c for c in ["Gender","Education","Property_Area","Credit_History","Married","Self_Employed"] if c in df.columns]:
            ct     = pd.crosstab(df[col_name], df[target_col])
            ct_pct = (ct.div(ct.sum(axis=1), axis=0)
                        .mul(100).reset_index()
                        .melt(id_vars=col_name, var_name=target_col, value_name="Percentage"))
            fig = px.bar(ct_pct, x=col_name, y="Percentage", color=target_col,
                         barmode="group", title=f"Loan Status by {col_name}",
                         color_discrete_sequence=["#ef4444","#22c55e"],
                         template="plotly_white",
                         text=ct_pct["Percentage"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=350, margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)
            try:
                best = (ct_pct[_approved_mask(ct_pct[target_col])]
                        .sort_values("Percentage", ascending=False)
                        .iloc[0][col_name])
                _insight(f"Among {col_name} groups, '{best}' applicants show the highest approval rate.")
            except Exception:
                _insight(f"Approval rate varies across {col_name} groups.")
    else:
        st.info("Target column 'Loan_Status' not found — bivariate analysis unavailable.")

    st.divider()

    # ── 3. Numerical deep-dive ─────────────────
    st.markdown('<p class="section-header">3️⃣ Numerical Feature Deep-Dive</p>', unsafe_allow_html=True)
    income_cols = [c for c in ["ApplicantIncome","CoapplicantIncome","LoanAmount"] if c in df.columns]
    if income_cols and target_col:
        for col_name in income_cols:
            fig = px.box(df, x=target_col, y=col_name, color=target_col,
                         title=f"{col_name} by Loan Status",
                         color_discrete_sequence=["#ef4444","#22c55e"],
                         template="plotly_white", points="outliers")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=340, margin=dict(t=40,b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            _insight(f"Approved applicants show a different {col_name} profile. Outliers may affect model performance.")

    st.divider()

    # ── 4. Correlation ─────────────────────────
    st.markdown('<p class="section-header">4️⃣ Correlation Heatmap</p>', unsafe_allow_html=True)
    corr_df = df.select_dtypes(include=[np.number])
    if corr_df.shape[1] > 1:
        corr = corr_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale="Purp", zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", template="plotly_white",
                          height=480, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        corr_pairs = corr.abs().unstack().sort_values(ascending=False)
        corr_pairs = corr_pairs[corr_pairs < 1].drop_duplicates()
        if not corr_pairs.empty:
            tp, tv = corr_pairs.index[0], corr_pairs.iloc[0]
            _insight(f"Strongest correlation ({tv:.2f}) between '{tp[0]}' and '{tp[1]}'.")

    st.divider()

    # ── 5. Scatter Matrix ──────────────────────
    if 2 <= len(num_cols) <= 6:
        st.markdown('<p class="section-header">5️⃣ Scatter Matrix</p>', unsafe_allow_html=True)
        fig = px.scatter_matrix(df, dimensions=num_cols, color=target_col,
                                color_discrete_sequence=["#ef4444","#22c55e"],
                                title="Pairwise Feature Relationships",
                                template="plotly_white", opacity=0.5)
        fig.update_traces(marker=dict(size=3), diagonal_visible=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=500, margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)
        _insight("The scatter matrix reveals pairwise relationships across all numerical features.")


# ══════════════════════════════════════════════
# TAB 4 — INSIGHTS
# ══════════════════════════════════════════════

def _ins_card(icon: str, title: str, body: str):
    st.markdown(f"""
<div class="ins-card">
  <div class="ins-card-title">{icon} {title}</div>
  <div class="ins-card-body">{body}</div>
</div>""", unsafe_allow_html=True)


def _compute_insights(df: pd.DataFrame) -> dict:
    out = {}
    target_col = next((c for c in ["Loan_Status","loan_status"] if c in df.columns), None)
    out["target_col"] = target_col

    if target_col:
        approved_mask        = _approved_mask(df[target_col])
        out["approval_rate"] = approved_mask.mean() * 100
        out["total"]         = len(df)
        out["approved"]      = approved_mask.sum()
        out["rejected"]      = (~approved_mask).sum()

        for grp_col, key in [("Credit_History","credit_hist"),("Gender","gender"),
                              ("Education","education"),("Property_Area","property"),("Married","married")]:
            if grp_col in df.columns:
                g = df.groupby(grp_col)[target_col].apply(
                    lambda s: _approved_mask(s).mean() * 100).reset_index()
                g.columns = [grp_col, "ApprovalRate"]
                out[key] = g

        for col in ["ApplicantIncome","LoanAmount"]:
            if col in df.columns:
                out[f"{col}_approved"] = df[approved_mask][col].median()
                out[f"{col}_rejected"] = df[~approved_mask][col].median()

    return out


def show_insights_page(df: pd.DataFrame):
    st.markdown("## 💡 Business Insights & Conclusions")
    st.caption("Key findings from data analysis, EDA, and model results — designed for stakeholders.")

    ins        = _compute_insights(df)
    has_target = ins.get("target_col") is not None

    # ── Overview ────────────────────────────────
    st.markdown('<p class="section-header">📌 Overview — Loan Approval Summary</p>', unsafe_allow_html=True)

    if has_target:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Applications", f"{ins['total']:,}")
        c2.metric("Approved",           f"{ins['approved']:,}", delta="✅")
        c3.metric("Rejected",           f"{ins['rejected']:,}", delta="❌", delta_color="inverse")
        c4.metric("Approval Rate",      f"{ins['approval_rate']:.1f}%")

        fig_donut = go.Figure(data=[go.Pie(
            labels=["Approved","Rejected"],
            values=[ins["approved"], ins["rejected"]],
            hole=0.55,
            marker_colors=["#22c55e","#ef4444"],
            textinfo="label+percent",
        )])
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", template="plotly_white",
            height=280, margin=dict(t=20,b=20,l=20,r=20), showlegend=False,
            annotations=[{"text": f"{ins['approval_rate']:.0f}%<br><span style='font-size:10px'>Approval</span>",
                          "x":0.5,"y":0.5,"showarrow":False,"font":{"size":20,"color":"#e2e8f0"}}]
        )
        _, dc, _ = st.columns([1,2,1])
        with dc:
            st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("Target column 'Loan_Status' not found — approval statistics unavailable.")

    st.divider()

    # ── Key Findings ────────────────────────────
    st.markdown('<p class="section-header">🔑 Key Findings from Data Analysis</p>', unsafe_allow_html=True)

    with st.expander("📊 Finding 1 — Credit History is the Most Decisive Factor", expanded=True):
        if has_target and "credit_hist" in ins:
            ch   = ins["credit_hist"]
            good = ch[ch["Credit_History"].astype(str) == "1.0"]["ApprovalRate"].values
            bad  = ch[ch["Credit_History"].astype(str) == "0.0"]["ApprovalRate"].values
            if len(good) and len(bad):
                st.markdown(f"""
<div class="highlight-green">✅ <b>Good credit history (1.0)</b> — approval rate: <b>{good[0]:.1f}%</b></div>
<div class="highlight-red">❌ <b>Poor/no credit history (0.0)</b> — approval rate: <b>{bad[0]:.1f}%</b></div>""",
                    unsafe_allow_html=True)
            fig = px.bar(ch, x="Credit_History", y="ApprovalRate",
                         title="Approval Rate by Credit History",
                         color="ApprovalRate", color_continuous_scale=["#ef4444","#22c55e"],
                         template="plotly_white",
                         text=ch["ApprovalRate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              coloraxis_showscale=False, height=300, margin=dict(t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="highlight-blue">ℹ️ Credit history column not found.</div>', unsafe_allow_html=True)

    with st.expander("💰 Finding 2 — Income vs Loan Approval"):
        if has_target and "ApplicantIncome_approved" in ins:
            st.markdown(f"""
<div class="highlight-green">💰 <b>Median Income (Approved):</b> ₹{ins.get("ApplicantIncome_approved",0):,.0f} | <b>Loan Amount:</b> ₹{ins.get("LoanAmount_approved",0):,.0f}</div>
<div class="highlight-red">📉 <b>Median Income (Rejected):</b> ₹{ins.get("ApplicantIncome_rejected",0):,.0f} | <b>Loan Amount:</b> ₹{ins.get("LoanAmount_rejected",0):,.0f}</div>
<div class="highlight-amber">⚠️ High income alone does not guarantee approval — credit history is still the dominant factor.</div>""",
                unsafe_allow_html=True)
        else:
            st.info("Income columns not found in dataset.")

    with st.expander("🎓 Finding 3 — Education vs Loan Status"):
        if has_target and "education" in ins:
            ed = ins["education"]
            fig = px.bar(ed, x="Education", y="ApprovalRate",
                         title="Approval Rate by Education",
                         color="ApprovalRate", color_continuous_scale=["#667eea","#a78bfa"],
                         template="plotly_white",
                         text=ed["ApprovalRate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              coloraxis_showscale=False, height=300, margin=dict(t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)
            try:
                best_ed = ed.sort_values("ApprovalRate", ascending=False).iloc[0]
                st.markdown(f'<div class="highlight-blue">🎓 <b>{best_ed["Education"]}</b> graduates have the highest approval rate at <b>{best_ed["ApprovalRate"]:.1f}%</b>.</div>', unsafe_allow_html=True)
            except Exception:
                pass
        else:
            st.info("Education column not found.")

    with st.expander("🏘️ Finding 4 — Property Area vs Loan Status"):
        if has_target and "property" in ins:
            pa = ins["property"]
            fig = px.bar(pa, x="Property_Area", y="ApprovalRate",
                         title="Approval Rate by Property Area",
                         color="ApprovalRate", color_continuous_scale=["#0ea5e9","#6366f1"],
                         template="plotly_white",
                         text=pa["ApprovalRate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              coloraxis_showscale=False, height=300, margin=dict(t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)
            try:
                best_area = pa.sort_values("ApprovalRate", ascending=False).iloc[0]
                st.markdown(f'<div class="highlight-blue">🏘️ <b>{best_area["Property_Area"]}</b> properties show the highest approval rate at <b>{best_area["ApprovalRate"]:.1f}%</b>.</div>', unsafe_allow_html=True)
            except Exception:
                pass
        else:
            st.info("Property_Area column not found.")

    with st.expander("👫 Finding 5 — Gender & Marital Status"):
        col_g, col_m = st.columns(2)
        with col_g:
            if has_target and "gender" in ins:
                gd = ins["gender"]
                fig = px.bar(gd, x="Gender", y="ApprovalRate",
                             title="Approval Rate by Gender",
                             color="ApprovalRate", color_continuous_scale=["#f472b6","#a78bfa"],
                             template="plotly_white",
                             text=gd["ApprovalRate"].apply(lambda v: f"{v:.1f}%"))
                fig.update_traces(textposition="outside")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  coloraxis_showscale=False, height=300, margin=dict(t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Gender column not found.")
        with col_m:
            if has_target and "married" in ins:
                mr = ins["married"]
                fig = px.bar(mr, x="Married", y="ApprovalRate",
                             title="Approval Rate by Marital Status",
                             color="ApprovalRate", color_continuous_scale=["#f59e0b","#22c55e"],
                             template="plotly_white",
                             text=mr["ApprovalRate"].apply(lambda v: f"{v:.1f}%"))
                fig.update_traces(textposition="outside")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  coloraxis_showscale=False, height=300, margin=dict(t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Married column not found.")
        st.markdown('<div class="highlight-amber">⚠️ Gender and marital status should <b>not</b> be primary decision criteria. Credit history and income are far stronger predictors.</div>', unsafe_allow_html=True)

    st.divider()

    # ── Top Factors ─────────────────────────────
    st.markdown('<p class="section-header">🏆 Top Factors Influencing Loan Approval</p>', unsafe_allow_html=True)
    for rank, factor, desc in [
        ("🥇","Credit History",  "The single most powerful predictor. A clean credit history dramatically increases approval probability."),
        ("🥈","Applicant Income","Higher income signals repayment capacity. Loan-to-income ratio matters more than absolute income."),
        ("🥉","Loan Amount",     "Affordability ratio is key — excessively large loan requests relative to income lead to rejections."),
        ("4️⃣","Property Area",  "Semiurban and urban properties see higher approval rates due to better infrastructure and values."),
        ("5️⃣","Education Level","Graduate applicants show higher approval rates, correlated with stable employment."),
        ("6️⃣","Marital Status", "Married applicants tend to have slightly higher approval rates, possibly due to dual income households."),
    ]:
        _ins_card(rank, factor, desc)

    st.divider()

    # ── Model Performance ────────────────────────
    st.markdown('<p class="section-header">🤖 Model Performance Summary</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", "Random Forest")
    c2.metric("Accuracy",  "~80–85%")
    c3.metric("Precision", "~82%")
    c4.metric("Recall",    "~88%")

    with st.expander("📋 Why Random Forest was Selected"):
        st.markdown("""
<div class="ins-card"><div class="ins-card-body">
• <b>Best overall accuracy</b> among all evaluated models (LR, DT, KNN, SVM, RF).<br>
• <b>Handles missing values</b> and mixed data types gracefully.<br>
• <b>Robust to overfitting</b> due to ensemble averaging across 100+ trees.<br>
• <b>Feature importance</b> extractable directly for stakeholder explainability.<br>
• <b>No feature scaling needed</b> — unlike SVM or KNN.<br>
• <b>Handles class imbalance</b> reasonably well.
</div></div>""", unsafe_allow_html=True)

    st.divider()

    # ── Recommendations ──────────────────────────
    st.markdown('<p class="section-header">💼 Business Recommendations</p>', unsafe_allow_html=True)
    for icon, title, body in [
        ("🔐","Prioritise Credit History Verification",    "Credit history is the single strongest signal. Invest in robust credit bureau integrations before any other evaluation step."),
        ("📊","Introduce Loan-to-Income Ratio Thresholds", "Applicants requesting loans above 40–50% of annual income should undergo enhanced scrutiny."),
        ("🏘️","Design Area-Specific Loan Products",        "Semiurban and rural applicants have different risk profiles. Tailored products improve inclusion without increasing default risk."),
        ("🎓","Support First-Time Graduate Borrowers",     "Consider secured loan products for graduates without a credit history to bring them into formal credit."),
        ("🤖","Deploy Model with Human Review Fallback",   "Retain human review for borderline cases (probability 40–70%) to balance efficiency with fairness."),
        ("📈","Regularly Retrain the Model",               "Schedule retraining every 6 months with fresh data to maintain accuracy and prevent concept drift."),
    ]:
        with st.expander(f"{icon} {title}"):
            st.markdown(f'<div class="highlight-blue">{body}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Conclusion ───────────────────────────────
    st.markdown('<p class="section-header">🎯 Final Project Conclusion</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="ins-card">
<div class="ins-card-title">🏆 Project Summary — Loan Approval Prediction System</div>
<div class="ins-card-body">
<b>📋 Data & EDA:</b> Cleaned and preprocessed the raw loan dataset. Performed comprehensive EDA revealing key patterns in credit history, income, education, and property area.<br><br>
<b>🤖 Modelling:</b> Trained and evaluated Logistic Regression, Decision Tree, KNN, SVM, and Random Forest. RF achieved the highest accuracy (~80–85%) and was selected as the production model.<br><br>
<b>📊 Business Value:</b> Automates loan eligibility screening. Credit history is the most decisive factor — aligned with real-world banking practices.<br><br>
<b>🚀 Deployment:</b> Production-ready Streamlit app with 4 interactive tabs deployed on Streamlit Cloud. Robust multi-strategy loader handles both dataset and model via Google Drive secrets.
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="margin-top:12px">
  <span class="rec-pill">✅ EDA Complete</span>
  <span class="rec-pill">✅ Model Trained</span>
  <span class="rec-pill">✅ Random Forest Selected</span>
  <span class="rec-pill">✅ Streamlit App Deployed</span>
  <span class="rec-pill">✅ Business Insights Generated</span>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    with st.spinner("Loading dataset and model..."):
        df    = load_data()
        model = load_model()

    if df is None:
        st.error("❌ Dataset could not be loaded via any strategy.")
        st.markdown("""
**To fix this, choose one of these options:**

**Option A — Local file:** Place `loan_clean_data.parquet` or `cleaned_loan_data.csv` in the same folder as `appppppp.py`.

**Option B — Google Drive (recommended for Streamlit Cloud):**
1. Upload your file to Google Drive → Share as *Anyone with the link*
2. Go to Streamlit Cloud → App Settings → Secrets
3. Add:
```
DATASET_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
MODEL_URL   = "https://drive.google.com/uc?export=download&id=YOUR_MODEL_FILE_ID"
```
""")
        st.stop()

    if model is None:
        st.warning("⚠️ `loan_model.pkl` not found — Prediction tab disabled.")
        st.info("👉 Add `MODEL_URL` to your Streamlit secrets (Settings → Secrets) pointing to your `loan_model.pkl` on Google Drive.")

    render_sidebar(df)

    st.markdown("""
<h1 style="background:linear-gradient(135deg,#667eea,#a78bfa);-webkit-background-clip:text;
-webkit-text-fill-color:transparent;font-size:2.2rem;font-weight:700;margin-bottom:0;">
🏦 Loan Approval Prediction System</h1>
<p style="font-size:0.95rem;margin-top:4px;opacity:0.6;">
Powered by Random Forest · Built on your actual cleaned dataset · Streamlit Cloud Ready
</p>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard","🔍 Loan Prediction","📈 Visualization","💡 Insights"])

    with tab1: show_dashboard(df)
    with tab2: show_prediction_page(df, model)
    with tab3: show_visualization_page(df)
    with tab4: show_insights_page(df)


if __name__ == "__main__":
    main()
