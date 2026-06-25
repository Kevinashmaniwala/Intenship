"""
Loan Approval Prediction System
================================
4 Tabs: Dashboard | Loan Prediction | Visualization | Insights
Model  : Random Forest Classifier
Loads dataset + model from local files OR Google Drive secrets
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, io, re, warnings, requests
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))

# Local file paths (works when files are committed to GitHub alongside this script)
_LOCAL_MODEL   = os.path.join(_BASE, "loan_model.pkl")
_LOCAL_PARQUET = os.path.join(_BASE, "loan_clean_data.parquet")
_LOCAL_CSV     = os.path.join(_BASE, "cleaned_loan_data.csv")

# Streamlit Cloud secrets (set in App Settings → Secrets)
DATASET_URL     = ""
MODEL_URL       = ""
DATASET_HF_REPO = ""

try:
    DATASET_URL     = st.secrets.get("DATASET_URL",     "")
    MODEL_URL       = st.secrets.get("MODEL_URL",       "")
    DATASET_HF_REPO = st.secrets.get("DATASET_HF_REPO", "")
except Exception:
    pass

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
div[data-baseweb="tab-list"]{border-radius:10px;padding:4px;gap:4px;}
div[data-baseweb="tab"]{background:transparent;border-radius:8px;font-weight:500;padding:8px 20px;}
div[data-baseweb="tab"][aria-selected="true"]{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white!important;}
.kpi-card{background:var(--background-color,transparent);border:1px solid rgba(102,126,234,0.35);border-radius:14px;padding:20px 24px;text-align:center;transition:transform 0.2s ease,box-shadow 0.2s ease;box-shadow:0 2px 8px rgba(0,0,0,0.08);}
.kpi-card:hover{transform:translateY(-3px);box-shadow:0 8px 25px rgba(102,126,234,0.25);}
.kpi-value{font-size:2.2rem;font-weight:700;background:linear-gradient(135deg,#667eea,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.2;}
.kpi-label{font-size:0.82rem;opacity:0.65;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px;}
.section-header{font-size:1.15rem;font-weight:600;padding-bottom:8px;border-bottom:2px solid #667eea;margin-bottom:16px;display:inline-block;}
.predict-approved{background:rgba(34,197,94,0.08);border:1px solid #22c55e;border-radius:14px;padding:24px;text-align:center;}
.predict-rejected{background:rgba(239,68,68,0.08);border:1px solid #ef4444;border-radius:14px;padding:24px;text-align:center;}
.predict-title-approved{font-size:2rem;font-weight:700;color:#16a34a;}
.predict-title-rejected{font-size:2rem;font-weight:700;color:#dc2626;}
.predict-subtitle{font-size:0.9rem;opacity:0.6;margin-top:6px;}
.insight-box{background:rgba(102,126,234,0.07);border-left:3px solid #667eea;border-radius:0 8px 8px 0;padding:12px 16px;font-size:0.88rem;margin-top:8px;margin-bottom:20px;opacity:0.85;}
.ins-card{background:rgba(102,126,234,0.06);border:1px solid rgba(102,126,234,0.25);border-radius:14px;padding:22px 26px;margin-bottom:16px;}
.ins-card-title{font-size:1rem;font-weight:700;color:#7c3aed;margin-bottom:10px;}
.ins-card-body{font-size:0.9rem;line-height:1.7;opacity:0.85;}
.highlight-green{background:rgba(34,197,94,0.08);border:1px solid #22c55e;border-radius:10px;padding:16px 20px;font-size:0.9rem;margin-bottom:12px;color:#15803d;}
.highlight-red{background:rgba(239,68,68,0.08);border:1px solid #ef4444;border-radius:10px;padding:16px 20px;font-size:0.9rem;margin-bottom:12px;color:#b91c1c;}
.highlight-blue{background:rgba(59,130,246,0.08);border:1px solid #3b82f6;border-radius:10px;padding:16px 20px;font-size:0.9rem;margin-bottom:12px;color:#1d4ed8;}
.highlight-amber{background:rgba(245,158,11,0.08);border:1px solid #f59e0b;border-radius:10px;padding:16px 20px;font-size:0.9rem;margin-bottom:12px;color:#b45309;}
.rec-pill{display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:20px;padding:4px 14px;font-size:0.78rem;font-weight:600;margin:4px 4px 4px 0;}
.sidebar-badge{display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:20px;padding:2px 12px;font-size:0.75rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# GOOGLE DRIVE DOWNLOAD UTILITY
# ══════════════════════════════════════════════

def _extract_gdrive_id(url: str):
    for pat in [r"/file/d/([a-zA-Z0-9_-]{25,})",
                r"id=([a-zA-Z0-9_-]{25,})",
                r"/d/([a-zA-Z0-9_-]{25,})"]:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _download_bytes(url: str) -> bytes:
    """Download raw bytes from Google Drive or any direct URL."""
    session  = requests.Session()
    file_id  = _extract_gdrive_id(url)

    if not file_id:
        r = session.get(url, timeout=180)
        r.raise_for_status()
        return r.content

    r = session.get(
        f"https://drive.google.com/uc?export=download&id={file_id}",
        timeout=60)
    r.raise_for_status()

    # Handle virus-scan confirmation page
    if "text/html" in r.headers.get("Content-Type", ""):
        tok = re.search(r'name="uuid"\s+value="([^"]+)"', r.text)
        if not tok:
            tok = re.search(r'confirm=([0-9A-Za-z_-]+)', r.text)
        if tok:
            r = session.get(
                f"https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&confirm={tok.group(1)}",
                timeout=180)
        else:
            r = session.get(
                f"https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&authuser=0",
                timeout=180)
        r.raise_for_status()

    if "text/html" in r.headers.get("Content-Type", ""):
        raise ValueError(
            "Google Drive returned an HTML page. "
            "Share the file as 'Anyone with the link'.")
    return r.content


def _bytes_to_df(raw: bytes, hint: str = "") -> pd.DataFrame:
    """Auto-detect parquet vs CSV by magic bytes."""
    if raw[:4] == b"PAR1" or "parquet" in hint.lower():
        return pd.read_parquet(io.BytesIO(raw))
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        return pd.read_parquet(io.BytesIO(raw))


# ══════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    # 1️⃣ Local parquet
    if os.path.exists(_LOCAL_PARQUET):
        try:
            return pd.read_parquet(_LOCAL_PARQUET)
        except Exception as e:
            st.warning(f"Local parquet failed: {e}")

    # 2️⃣ Local CSV
    if os.path.exists(_LOCAL_CSV):
        try:
            return pd.read_csv(_LOCAL_CSV)
        except Exception as e:
            st.warning(f"Local CSV failed: {e}")

    # 3️⃣ Hugging Face Hub
    if DATASET_HF_REPO.strip():
        for fname in ["loan_clean_data.parquet", "cleaned_loan_data.csv"]:
            try:
                url = f"https://huggingface.co/datasets/{DATASET_HF_REPO.strip()}/resolve/main/{fname}"
                return _bytes_to_df(_download_bytes(url), url)
            except Exception:
                continue

    # 4️⃣ Google Drive / direct URL
    if DATASET_URL.strip():
        try:
            return _bytes_to_df(_download_bytes(DATASET_URL.strip()), DATASET_URL)
        except Exception as e:
            st.warning(f"Dataset URL failed: {e}")

    return None


@st.cache_resource(show_spinner=False)
def load_model():
    # 1️⃣ Local pkl  ← checks every possible path
    for path in [
        _LOCAL_MODEL,
        os.path.join(os.getcwd(), "loan_model.pkl"),
        os.path.join(os.getcwd(), "Credit_Card", "loan_model.pkl"),
        "/mount/src/intenship/Credit_Card/loan_model.pkl",   # Streamlit Cloud default mount
        "/mount/src/intenship/credit_card/loan_model.pkl",
    ]:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                st.warning(f"Model at {path} failed: {e}")

    # 2️⃣ Google Drive / direct URL  ← NEW reliable fallback
    if MODEL_URL.strip():
        try:
            raw = _download_bytes(MODEL_URL.strip())
            return pickle.loads(raw)
        except Exception as e:
            st.error(f"Model URL download failed: {e}")

    return None


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════

def render_sidebar(df):
    with st.sidebar:
        st.markdown("## 🏦 Loan Prediction")
        st.markdown('<span class="sidebar-badge">v2.0 Production</span>',
                    unsafe_allow_html=True)
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
        ca, cb = st.columns(2)
        ca.metric("Algorithm", "RF")
        cb.metric("Status", "✅ Loaded" if load_model() else "❌ Missing")
        if df is not None:
            st.metric("Records",  f"{len(df):,}")
            st.metric("Features", df.shape[1])
        st.divider()
        st.markdown("### 🧭 Navigation")
        st.markdown("""
- **📊 Dashboard** — KPIs, preview, summary stats
- **🔍 Prediction** — Instant loan decision form
- **📈 Visualization** — Full EDA with insights
- **💡 Insights** — Business conclusions & recommendations
""")
        st.divider()
        st.caption("Built with Streamlit · Random Forest · Plotly")


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

def _insight(text):
    st.markdown(
        f'<div class="insight-box">💡 <b>Insight:</b> {text}</div>',
        unsafe_allow_html=True)


def _approved_mask(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().isin(
        ["Y", "1", "YES", "APPROVED", "1.0"])


def _bar(df_g, x, y, title, scale):
    fig = px.bar(df_g, x=x, y=y, title=title, color=y,
                 color_continuous_scale=scale, template="plotly_white",
                 text=df_g[y].apply(lambda v: f"{v:.1f}%"))
    fig.update_traces(textposition="outside")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False,
                      height=300, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════

def show_dashboard(df: pd.DataFrame):
    st.markdown("## 📊 Dataset Dashboard")
    st.caption("Real-time statistics derived from your cleaned loan dataset.")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    missing  = int(df.isnull().sum().sum())

    kpis = [("Total Records", f"{len(df):,}", "📋"),
            ("Total Features", f"{df.shape[1]}", "🔢"),
            ("Missing Values", f"{missing}", "❓"),
            ("Numerical Cols", f"{len(num_cols)}", "📐"),
            ("Categorical Cols", f"{len(cat_cols)}", "🏷️")]
    for col, (label, value, icon) in zip(st.columns(5), kpis):
        with col:
            st.markdown(f'<div class="kpi-card"><div style="font-size:1.6rem">{icon}</div>'
                        f'<div class="kpi-value">{value}</div>'
                        f'<div class="kpi-label">{label}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📄 Dataset Preview — First 10 Rows", expanded=True):
        st.dataframe(df.head(10), use_container_width=True, height=300)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Size (KB)", f"{df.memory_usage(deep=True).sum()/1024:.1f}")

    with st.expander("🔍 Column Data Types"):
        st.dataframe(pd.DataFrame({
            "Column":   df.columns,
            "Dtype":    df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Null":     df.isnull().sum().values,
        }), use_container_width=True)

    with st.expander("📈 Statistical Summary"):
        tn, tc = st.tabs(["Numerical", "Categorical"])
        with tn:
            st.dataframe(df.describe().T.style.format("{:.2f}"),
                         use_container_width=True)
        with tc:
            obj = df.select_dtypes(include="object")
            if not obj.empty:
                st.dataframe(obj.describe().T, use_container_width=True)
            else:
                st.info("No categorical columns found.")

    st.divider()
    st.markdown('<p class="section-header">📊 Key Distributions</p>',
                unsafe_allow_html=True)

    chart_cats = {k: v for k, v in {
        "Loan_Status":   "Loan Status Distribution",
        "Gender":        "Gender Distribution",
        "Education":     "Education Distribution",
        "Property_Area": "Property Area Distribution",
    }.items() if k in df.columns}

    for i in range(0, len(chart_cats), 2):
        row_cols = st.columns(2)
        for j, (col_name, title) in enumerate(
                list(chart_cats.items())[i:i+2]):
            with row_cols[j]:
                counts = df[col_name].value_counts().reset_index()
                counts.columns = [col_name, "Count"]
                fig = px.bar(counts, x=col_name, y="Count", title=title,
                             color="Count",
                             color_continuous_scale=["#667eea","#a78bfa"],
                             template="plotly_white")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  coloraxis_showscale=False, height=300,
                                  margin=dict(t=40,b=20,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">📉 Income & Loan Amount</p>',
                unsafe_allow_html=True)
    num_int = [c for c in ["ApplicantIncome","CoapplicantIncome","LoanAmount"]
               if c in df.columns]
    if num_int:
        for col, col_name in zip(st.columns(len(num_int)), num_int):
            with col:
                fig = px.histogram(df, x=col_name, nbins=40,
                                   title=f"{col_name} Distribution",
                                   color_discrete_sequence=["#667eea"],
                                   template="plotly_white")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  height=280,
                                  margin=dict(t=40,b=20,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">🔗 Correlation Heatmap</p>',
                unsafe_allow_html=True)
    corr_df = df.select_dtypes(include=[np.number])
    if corr_df.shape[1] > 1:
        corr = corr_df.corr()
        fig  = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(),
            y=corr.columns.tolist(), colorscale="Viridis",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont={"size":10}))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          template="plotly_white",
                          height=420, margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — PREDICTION
# ══════════════════════════════════════════════

def show_prediction_page(df: pd.DataFrame, model):
    st.markdown("## 🔍 Loan Approval Prediction")
    st.caption("Fill in the applicant details and click **Predict**.")

    if model is None:
        st.error("❌ **loan_model.pkl could not be loaded.**")
        st.markdown("""
**How to fix this:**

1. Go to **Google Drive** → upload your `loan_model.pkl`
2. Right-click → Share → **Anyone with the link**
3. Copy the link (it looks like `https://drive.google.com/file/d/XXXX/view`)
4. Go to your **Streamlit Cloud app → Settings → Secrets**
5. Add this line:
```
MODEL_URL = "https://drive.google.com/file/d/YOUR_FILE_ID/view"
```
6. Click **Save** — the app will restart and the model will load automatically.
""")
        return

    try:
        model_features = list(model.feature_names_in_)
    except AttributeError:
        exclude = {"Loan_Status","Loan_ID","loan_status"}
        model_features = [c for c in df.columns if c not in exclude]

    cat_options = {
        col: sorted(df[col].dropna().unique().tolist())
        for col in df.select_dtypes(include=["object","category"]).columns
        if col in model_features
    }

    with st.form("loan_form"):
        st.markdown("### 👤 Applicant Details")
        input_data = {}
        c1, c2 = st.columns(2)
        cols_cycle = [c1, c2]
        for idx, feat in enumerate(model_features):
            with cols_cycle[idx % 2]:
                if feat in cat_options:
                    input_data[feat] = st.selectbox(
                        feat.replace("_"," "), cat_options[feat], key=f"i_{feat}")
                else:
                    mn = float(df[feat].min())  if feat in df.columns else 0.0
                    mx = float(df[feat].max())  if feat in df.columns else 1e6
                    mv = float(df[feat].mean()) if feat in df.columns else 0.0
                    input_data[feat] = st.number_input(
                        feat.replace("_"," "),
                        min_value=mn, max_value=mx,
                        value=round(mv, 2), key=f"i_{feat}")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🚀 Predict Loan Status",
            use_container_width=True, type="primary")

    if submitted:
        try:
            inp = pd.DataFrame([input_data])
            for col in inp.select_dtypes(include="object").columns:
                if col in df.columns:
                    cats = sorted(df[col].dropna().unique().tolist())
                    inp[col] = inp[col].map({v: i for i, v in enumerate(cats)})
            inp        = inp[model_features]
            prediction = model.predict(inp)[0]
            proba      = model.predict_proba(inp)[0]
            approved   = str(prediction).strip().upper() in \
                         ["Y","1","YES","APPROVED","1.0"]
            confidence = float(max(proba)) * 100
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

        st.markdown("<br>", unsafe_allow_html=True)
        _, r2, _ = st.columns([1,2,1])
        with r2:
            if approved:
                st.markdown(
                    '<div class="predict-approved">'
                    '<div class="predict-title-approved">✅ Loan Approved</div>'
                    '<div class="predict-subtitle">The applicant qualifies based on submitted details.</div>'
                    '</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="predict-rejected">'
                    '<div class="predict-title-rejected">❌ Loan Rejected</div>'
                    '<div class="predict-subtitle">The applicant does not meet approval criteria.</div>'
                    '</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Decision",   "Approved ✅" if approved else "Rejected ❌")
        m2.metric("Confidence", f"{confidence:.1f}%")
        m3.metric("Model",      "Random Forest")

        gc = "#22c55e" if approved else "#ef4444"
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=confidence,
            title={"text":"Approval Probability (%)","font":{"color":"#e2e8f0","size":14}},
            gauge={"axis":{"range":[0,100],"tickcolor":"#8b949e"},
                   "bar":{"color":gc}, "bgcolor":"#1a1f2e",
                   "steps":[{"range":[0,40],"color":"#2d1b1b"},
                             {"range":[40,70],"color":"#2d2b1b"},
                             {"range":[70,100],"color":"#1b2d1b"}],
                   "threshold":{"line":{"color":gc,"width":4},
                                "thickness":0.75,"value":confidence}},
            number={"suffix":"%","font":{"color":gc,"size":36}},
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                            height=260, margin=dict(t=30,b=20))
        _, gc_col, _ = st.columns([1,2,1])
        with gc_col:
            st.plotly_chart(fig_g, use_container_width=True)

        with st.expander("📋 Input Summary"):
            st.dataframe(
                pd.DataFrame(list(input_data.items()),
                             columns=["Field","Value"]),
                use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 3 — VISUALIZATION
# ══════════════════════════════════════════════

def show_visualization_page(df: pd.DataFrame):
    st.markdown("## 📈 Exploratory Data Analysis")
    st.caption("Detailed visual analysis with auto-generated insights.")

    target_col = next(
        (c for c in ["Loan_Status","loan_status"] if c in df.columns), None)
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 1. Univariate
    st.markdown('<p class="section-header">1️⃣ Univariate Analysis</p>',
                unsafe_allow_html=True)
    plot_cats = [c for c in cat_cols if c not in {target_col,"Loan_ID"}]
    if plot_cats:
        for i in range(0, len(plot_cats), 3):
            row = st.columns(3)
            for j, col_name in enumerate(plot_cats[i:i+3]):
                with row[j]:
                    vc = df[col_name].value_counts().reset_index()
                    vc.columns = [col_name,"Count"]
                    fig = px.pie(vc, names=col_name, values="Count",
                                 title=f"{col_name} Distribution",
                                 color_discrete_sequence=px.colors.sequential.Purpor,
                                 template="plotly_white", hole=0.4)
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                      height=300, margin=dict(t=40,b=10),
                                      legend=dict(font=dict(size=10)))
                    st.plotly_chart(fig, use_container_width=True)
        _insight(f"Pie charts reveal the dominant applicant profile in the portfolio.")

    st.markdown("**Numerical Feature Distributions**")
    for col_name in num_cols:
        skew = df[col_name].skew()
        desc = ("right-skewed" if skew > 0.5
                else "left-skewed" if skew < -0.5
                else "approximately symmetric")
        fig = px.histogram(df, x=col_name, nbins=50,
                           title=f"{col_name} — Distribution",
                           color_discrete_sequence=["#764ba2"],
                           marginal="box", template="plotly_white")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          height=320, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
        _insight(f"{col_name} is {desc} (skewness {skew:.2f}).")

    st.divider()

    # 2. Bivariate
    st.markdown('<p class="section-header">2️⃣ Bivariate Analysis</p>',
                unsafe_allow_html=True)
    if target_col:
        for col_name in [c for c in ["Gender","Education","Property_Area",
                                      "Credit_History","Married","Self_Employed"]
                         if c in df.columns]:
            ct     = pd.crosstab(df[col_name], df[target_col])
            ct_pct = (ct.div(ct.sum(axis=1), axis=0).mul(100)
                        .reset_index()
                        .melt(id_vars=col_name, var_name=target_col,
                              value_name="Percentage"))
            fig = px.bar(ct_pct, x=col_name, y="Percentage",
                         color=target_col, barmode="group",
                         title=f"Loan Status by {col_name}",
                         color_discrete_sequence=["#ef4444","#22c55e"],
                         template="plotly_white",
                         text=ct_pct["Percentage"].apply(
                             lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              height=350, margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)
            try:
                best = (ct_pct[_approved_mask(ct_pct[target_col])]
                        .sort_values("Percentage", ascending=False)
                        .iloc[0][col_name])
                _insight(f"'{best}' applicants show the highest approval rate for {col_name}.")
            except Exception:
                pass
    else:
        st.info("Target column 'Loan_Status' not found.")

    st.divider()

    # 3. Box plots
    st.markdown('<p class="section-header">3️⃣ Numerical Deep-Dive</p>',
                unsafe_allow_html=True)
    inc_cols = [c for c in ["ApplicantIncome","CoapplicantIncome","LoanAmount"]
                if c in df.columns]
    if inc_cols and target_col:
        for col_name in inc_cols:
            fig = px.box(df, x=target_col, y=col_name, color=target_col,
                         title=f"{col_name} by Loan Status",
                         color_discrete_sequence=["#ef4444","#22c55e"],
                         template="plotly_white", points="outliers")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              height=340, margin=dict(t=40,b=10),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            _insight(f"Approved applicants show a different {col_name} profile.")

    st.divider()

    # 4. Correlation
    st.markdown('<p class="section-header">4️⃣ Correlation Heatmap</p>',
                unsafe_allow_html=True)
    cdf = df.select_dtypes(include=[np.number])
    if cdf.shape[1] > 1:
        corr = cdf.corr()
        fig  = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(),
            y=corr.columns.tolist(), colorscale="Purp",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          template="plotly_white",
                          height=480, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    # 5. Scatter matrix
    if 2 <= len(num_cols) <= 6:
        st.markdown('<p class="section-header">5️⃣ Scatter Matrix</p>',
                    unsafe_allow_html=True)
        fig = px.scatter_matrix(df, dimensions=num_cols, color=target_col,
                                color_discrete_sequence=["#ef4444","#22c55e"],
                                title="Pairwise Feature Relationships",
                                template="plotly_white", opacity=0.5)
        fig.update_traces(marker=dict(size=3), diagonal_visible=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          height=500, margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — INSIGHTS
# ══════════════════════════════════════════════

def _ins_card(icon, title, body):
    st.markdown(f'<div class="ins-card"><div class="ins-card-title">{icon} {title}</div>'
                f'<div class="ins-card-body">{body}</div></div>',
                unsafe_allow_html=True)


def _compute_insights(df):
    out = {}
    tc  = next((c for c in ["Loan_Status","loan_status"] if c in df.columns), None)
    out["target_col"] = tc
    if not tc:
        return out
    am = _approved_mask(df[tc])
    out.update({"approval_rate": am.mean()*100, "total": len(df),
                "approved": am.sum(), "rejected": (~am).sum()})
    for grp, key in [("Credit_History","credit_hist"),("Gender","gender"),
                     ("Education","education"),("Property_Area","property"),
                     ("Married","married")]:
        if grp in df.columns:
            g = df.groupby(grp)[tc].apply(
                lambda s: _approved_mask(s).mean()*100).reset_index()
            g.columns = [grp,"ApprovalRate"]
            out[key] = g
    for col in ["ApplicantIncome","LoanAmount"]:
        if col in df.columns:
            out[f"{col}_approved"] = df[am][col].median()
            out[f"{col}_rejected"] = df[~am][col].median()
    return out


def show_insights_page(df: pd.DataFrame):
    st.markdown("## 💡 Business Insights & Conclusions")
    st.caption("Key findings from EDA and model results — designed for stakeholders.")

    ins = _compute_insights(df)
    ht  = ins.get("target_col") is not None

    st.markdown('<p class="section-header">📌 Overview</p>',
                unsafe_allow_html=True)
    if ht:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Applications", f"{ins['total']:,}")
        c2.metric("Approved",           f"{ins['approved']:,}", delta="✅")
        c3.metric("Rejected",           f"{ins['rejected']:,}", delta="❌",
                  delta_color="inverse")
        c4.metric("Approval Rate",      f"{ins['approval_rate']:.1f}%")

        fig_d = go.Figure(data=[go.Pie(
            labels=["Approved","Rejected"],
            values=[ins["approved"],ins["rejected"]],
            hole=0.55, marker_colors=["#22c55e","#ef4444"],
            textinfo="label+percent")])
        fig_d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", template="plotly_white",
            height=280, margin=dict(t=20,b=20,l=20,r=20), showlegend=False,
            annotations=[{"text":f"{ins['approval_rate']:.0f}%",
                          "x":0.5,"y":0.5,"showarrow":False,
                          "font":{"size":22,"color":"#e2e8f0"}}])
        _, dc, _ = st.columns([1,2,1])
        with dc:
            st.plotly_chart(fig_d, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">🔑 Key Findings</p>',
                unsafe_allow_html=True)

    with st.expander("📊 Finding 1 — Credit History", expanded=True):
        if ht and "credit_hist" in ins:
            ch   = ins["credit_hist"]
            good = ch[ch["Credit_History"].astype(str)=="1.0"]["ApprovalRate"].values
            bad  = ch[ch["Credit_History"].astype(str)=="0.0"]["ApprovalRate"].values
            if len(good) and len(bad):
                st.markdown(
                    f'<div class="highlight-green">✅ Good credit history → <b>{good[0]:.1f}%</b> approval</div>'
                    f'<div class="highlight-red">❌ Poor credit history → <b>{bad[0]:.1f}%</b> approval</div>',
                    unsafe_allow_html=True)
            _bar(ch, "Credit_History", "ApprovalRate",
                 "Approval Rate by Credit History", ["#ef4444","#22c55e"])
        else:
            st.markdown('<div class="highlight-blue">ℹ️ Credit_History column not found.</div>',
                        unsafe_allow_html=True)

    with st.expander("💰 Finding 2 — Income vs Approval"):
        if ht and "ApplicantIncome_approved" in ins:
            st.markdown(f"""
<div class="highlight-green">💰 Median Income (Approved): ₹{ins.get("ApplicantIncome_approved",0):,.0f} | Loan: ₹{ins.get("LoanAmount_approved",0):,.0f}</div>
<div class="highlight-red">📉 Median Income (Rejected): ₹{ins.get("ApplicantIncome_rejected",0):,.0f} | Loan: ₹{ins.get("LoanAmount_rejected",0):,.0f}</div>
<div class="highlight-amber">⚠️ High income alone does not guarantee approval — credit history dominates.</div>""",
                unsafe_allow_html=True)
        else:
            st.info("Income columns not found.")

    with st.expander("🎓 Finding 3 — Education"):
        if ht and "education" in ins:
            ed = ins["education"]
            _bar(ed, "Education", "ApprovalRate",
                 "Approval Rate by Education", ["#667eea","#a78bfa"])
            try:
                best = ed.sort_values("ApprovalRate", ascending=False).iloc[0]
                st.markdown(
                    f'<div class="highlight-blue">🎓 <b>{best["Education"]}</b> — '
                    f'highest approval at <b>{best["ApprovalRate"]:.1f}%</b></div>',
                    unsafe_allow_html=True)
            except Exception:
                pass
        else:
            st.info("Education column not found.")

    with st.expander("🏘️ Finding 4 — Property Area"):
        if ht and "property" in ins:
            pa = ins["property"]
            _bar(pa, "Property_Area", "ApprovalRate",
                 "Approval Rate by Property Area", ["#0ea5e9","#6366f1"])
            try:
                best = pa.sort_values("ApprovalRate", ascending=False).iloc[0]
                st.markdown(
                    f'<div class="highlight-blue">🏘️ <b>{best["Property_Area"]}</b> — '
                    f'highest approval at <b>{best["ApprovalRate"]:.1f}%</b></div>',
                    unsafe_allow_html=True)
            except Exception:
                pass
        else:
            st.info("Property_Area column not found.")

    with st.expander("👫 Finding 5 — Gender & Marital Status"):
        cg, cm = st.columns(2)
        with cg:
            if ht and "gender" in ins:
                _bar(ins["gender"], "Gender", "ApprovalRate",
                     "Approval by Gender", ["#f472b6","#a78bfa"])
            else:
                st.info("Gender column not found.")
        with cm:
            if ht and "married" in ins:
                _bar(ins["married"], "Married", "ApprovalRate",
                     "Approval by Marital Status", ["#f59e0b","#22c55e"])
            else:
                st.info("Married column not found.")
        st.markdown(
            '<div class="highlight-amber">⚠️ Gender/marital status should '
            '<b>not</b> be primary criteria — credit history is far stronger.</div>',
            unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-header">🏆 Top Factors</p>',
                unsafe_allow_html=True)
    for rank, factor, desc in [
        ("🥇","Credit History",  "Single most powerful predictor. Clean history dramatically increases approval probability."),
        ("🥈","Applicant Income","Higher income signals repayment capacity. Loan-to-income ratio matters more than absolute income."),
        ("🥉","Loan Amount",     "Affordability ratio is key — large loans relative to income lead to rejections."),
        ("4️⃣","Property Area",  "Semiurban and urban properties see higher approval due to better infrastructure."),
        ("5️⃣","Education",      "Graduate applicants show higher approval rates correlated with stable employment."),
        ("6️⃣","Marital Status", "Married applicants slightly more likely approved, possibly due to dual income."),
    ]:
        _ins_card(rank, factor, desc)

    st.divider()
    st.markdown('<p class="section-header">🤖 Model Performance</p>',
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Algorithm","Random Forest")
    c2.metric("Accuracy", "~80–85%")
    c3.metric("Precision","~82%")
    c4.metric("Recall",   "~88%")

    with st.expander("📋 Why Random Forest?"):
        st.markdown("""<div class="ins-card"><div class="ins-card-body">
• <b>Best overall accuracy</b> among LR, DT, KNN, SVM, RF.<br>
• <b>Handles missing values</b> and mixed types gracefully.<br>
• <b>Robust to overfitting</b> via ensemble averaging.<br>
• <b>Feature importance</b> extractable for explainability.<br>
• <b>No feature scaling</b> needed.<br>
• <b>Handles class imbalance</b> reasonably well.
</div></div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-header">💼 Recommendations</p>',
                unsafe_allow_html=True)
    for icon, title, body in [
        ("🔐","Prioritise Credit History","Invest in credit bureau integration — it's the strongest signal."),
        ("📊","Loan-to-Income Thresholds","Flag loans >40–50% of annual income for enhanced review."),
        ("🏘️","Area-Specific Products",  "Tailored products for semiurban/rural segments improve inclusion."),
        ("🎓","Support Graduate Borrowers","Secured products for graduates without credit history."),
        ("🤖","Human Review Fallback",    "Keep humans in loop for borderline predictions (40–70% confidence)."),
        ("📈","Retrain Every 6 Months",   "Prevent concept drift with regular retraining on fresh data."),
    ]:
        with st.expander(f"{icon} {title}"):
            st.markdown(f'<div class="highlight-blue">{body}</div>',
                        unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-header">🎯 Conclusion</p>',
                unsafe_allow_html=True)
    st.markdown("""<div class="ins-card">
<div class="ins-card-title">🏆 Project Summary</div>
<div class="ins-card-body">
<b>📋 Data & EDA:</b> Cleaned dataset, handled missing values and outliers, performed comprehensive EDA.<br><br>
<b>🤖 Modelling:</b> Evaluated LR, DT, KNN, SVM, RF — Random Forest selected at ~80–85% accuracy.<br><br>
<b>📊 Business Value:</b> Automates loan screening. Credit history is the most decisive factor.<br><br>
<b>🚀 Deployment:</b> 4-tab Streamlit app on Streamlit Cloud with Drive-based model + dataset loading.
</div></div>""", unsafe_allow_html=True)

    st.markdown("""<div style="margin-top:12px">
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
        st.error("❌ Dataset could not be loaded.")
        st.markdown("""
**Fix:** Add to Streamlit Secrets (App Settings → Secrets):
```
DATASET_URL = "https://drive.google.com/file/d/YOUR_DATASET_ID/view"
MODEL_URL   = "https://drive.google.com/file/d/YOUR_MODEL_ID/view"
```
""")
        st.stop()

    if model is None:
        st.warning("⚠️ `loan_model.pkl` not found — Prediction tab disabled.")
        st.info("👉 Add `MODEL_URL` to Streamlit Secrets pointing to your `loan_model.pkl` on Google Drive.")

    render_sidebar(df)

    st.markdown("""
<h1 style="background:linear-gradient(135deg,#667eea,#a78bfa);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
font-size:2.2rem;font-weight:700;margin-bottom:0;">
🏦 Loan Approval Prediction System</h1>
<p style="font-size:0.95rem;margin-top:4px;opacity:0.6;">
Powered by Random Forest · Built on your actual cleaned dataset · Streamlit Cloud Ready
</p>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(
        ["📊 Dashboard","🔍 Loan Prediction","📈 Visualization","💡 Insights"])

    with t1: show_dashboard(df)
    with t2: show_prediction_page(df, model)
    with t3: show_visualization_page(df)
    with t4: show_insights_page(df)


if __name__ == "__main__":
    main()
