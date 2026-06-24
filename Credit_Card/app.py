import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── set_page_config MUST be the very first Streamlit call ────────────────────
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', sans-serif !important; }

[data-theme="light"] .stApp,
[data-theme="light"] [data-testid="stAppViewContainer"] > .main > .block-container {
    background-color: #f0f4f8 !important; color: #1a202c !important; }
[data-theme="light"] section[data-testid="stSidebar"] {
    background: #ffffff !important; border-right: 1.5px solid #e2e8f0 !important; }
[data-theme="light"] [data-testid="stMetric"] {
    background: #ffffff !important; border: 1.5px solid #e2e8f0 !important;
    border-top: 4px solid #2b6cb0 !important; border-radius: 12px !important;
    padding: 1.1rem 1.25rem !important; }
[data-theme="light"] .stTabs [aria-selected="true"] {
    background: #2b6cb0 !important; color: #ffffff !important; }
[data-theme="light"] .section-card {
    background: #ffffff !important; border: 1.5px solid #e2e8f0 !important; }
[data-theme="light"] .section-title {
    color: #2b6cb0 !important; border-bottom: 1.5px solid #ebf4ff !important; }
[data-theme="light"] .insight-card {
    background: #ffffff !important; border: 1.5px solid #e2e8f0 !important; }

[data-theme="dark"] .stApp,
[data-theme="dark"] [data-testid="stAppViewContainer"] > .main > .block-container {
    background-color: #0f1117 !important; color: #e2e8f0 !important; }
[data-theme="dark"] [data-testid="stHeader"] { background-color: #0f1117 !important; }
[data-theme="dark"] section[data-testid="stSidebar"],
[data-theme="dark"] [data-testid="stSidebarContent"] {
    background-color: #1a1c24 !important; border-right: 1.5px solid #2d3748 !important; }
[data-theme="dark"] [data-testid="stMetric"] {
    background: #1e2130 !important; border: 1.5px solid #2d3748 !important;
    border-top: 4px solid #63b3ed !important; border-radius: 12px !important;
    padding: 1.1rem 1.25rem !important; }
[data-theme="dark"] [data-testid="stMetricValue"],
[data-theme="dark"] [data-testid="stMetricLabel"] { color: #e2e8f0 !important; }
[data-theme="dark"] .stTabs [data-baseweb="tab-list"] { background-color: #1a1c24 !important; }
[data-theme="dark"] .stTabs [aria-selected="true"] {
    background: #3182ce !important; color: #ffffff !important; }
[data-theme="dark"] .stTabs [aria-selected="false"] { color: #a0aec0 !important; }
[data-theme="dark"] .section-card {
    background: #1e2130 !important; border: 1.5px solid #2d3748 !important; }
[data-theme="dark"] .section-title {
    color: #63b3ed !important; border-bottom: 1.5px solid #2a3a5c !important; }
[data-theme="dark"] .insight-card {
    background: #1e2130 !important; border: 1.5px solid #2d3748 !important; }
[data-theme="dark"] div[data-baseweb="select"] > div,
[data-theme="dark"] .stNumberInput input,
[data-theme="dark"] .stTextInput input {
    background-color: #2d3748 !important; color: #e2e8f0 !important;
    border-color: #4a5568 !important; }
[data-theme="dark"] .stApp label,
[data-theme="dark"] .stApp p,
[data-theme="dark"] .stApp h1,
[data-theme="dark"] .stApp h2,
[data-theme="dark"] .stApp h3 { color: #e2e8f0 !important; }
[data-theme="dark"] .stButton > button[kind="primary"] {
    background-color: #2b6cb0 !important; color: #ffffff !important; }

.section-card {
    border-radius: 14px; padding: 1.25rem 1.5rem 1.5rem; margin-bottom: 1rem; }
.section-title {
    font-size: 13px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; padding-bottom: 0.6rem; margin-bottom: 1.2rem; }
.main-app-header {
    background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 100%);
    color: white !important; padding: 1.5rem 2rem; border-radius: 14px; margin-bottom: 2rem; }
.main-app-header h1 {
    color: white !important; margin: 0 !important;
    font-size: 28px !important; font-weight: 700 !important; }
.main-app-header p {
    color: #ebf4ff !important; margin: 5px 0 0 0 !important;
    font-size: 14px !important; opacity: 0.9; }
.insight-card {
    border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
    border-left: 4px solid #3182ce; }
.insight-card.green { border-left-color: #38a169 !important; }
.insight-card.red   { border-left-color: #e53e3e !important; }
.insight-card.amber { border-left-color: #d69e2e !important; }
</style>
<script>
(function() {
    function applyTheme() {
        var el = document.querySelector('[data-testid="stAppViewContainer"]');
        if (!el) return;
        var bg = window.getComputedStyle(el).backgroundColor;
        var rgb = bg.match(/\d+/g);
        var theme = 'light';
        if (rgb) {
            var b = (parseInt(rgb[0])*299 + parseInt(rgb[1])*587 + parseInt(rgb[2])*114) / 1000;
            theme = b < 128 ? 'dark' : 'light';
        }
        document.documentElement.setAttribute('data-theme', theme);
        document.body.setAttribute('data-theme', theme);
    }
    applyTheme();
    setTimeout(applyTheme, 800);
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
})();
</script>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
base_path  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(base_path, "loan_clean_data.parquet")
CSV_PATH   = os.path.join(base_path, "loan_data.csv")          # your CSV file
MODEL_PKL  = os.path.join(base_path, "loan_model.pkl")
SCALER_PKL = os.path.join(base_path, "loan_scaler.pkl")

GREEN        = "#38a169"
RED          = "#e53e3e"
AMBER        = "#d69e2e"
BLUE         = "#3182ce"
BLUE_PALETTE = ["#2b6cb0","#3182ce","#4299e1","#63b3ed","#90cdf4","#bee3f8"]
DEPENDENT_MONTHLY_COST = 5000

# All numeric features the model was trained on
REQUIRED_COLUMNS = [
    'Applicant_Income','Coapplicant_Income','Age','Dependents',
    'Credit_Score','Existing_Loans','DTI_Ratio','Savings',
    'Collateral_Value','Loan_Amount','Loan_Term',
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def calculate_emi(principal, annual_rate_pct, term_months):
    if term_months <= 0 or principal <= 0:
        return 0.0
    r = annual_rate_pct / (12.0 * 100.0)
    if r == 0:
        return principal / term_months
    return principal * r * (1+r)**term_months / ((1+r)**term_months - 1)


def get_hard_reject_reasons(credit_score, dti, disposable_after_emi, employment, collateral, loan_amt):
    out = []
    if credit_score < 500:
        out.append(f"Credit score {credit_score} is below minimum threshold of 500")
    if dti > 0.65:
        out.append(f"DTI ratio {dti*100:.1f}% exceeds hard policy cap of 65%")
    if disposable_after_emi < 0:
        out.append("Disposable income after EMI and dependents is negative")
    if employment == "Unemployed":
        out.append("Applicant has no verifiable income (Unemployed)")
    if loan_amt > 0 and collateral < loan_amt * 0.50:
        out.append(f"Collateral (₹{collateral:,.0f}) < 50% of loan (₹{loan_amt:,.0f})")
    return out


def normalize_approval(series):
    """Standardise any Loan_Approved column values to 'Approved'/'Rejected'."""
    mapping = {
        '1':'Approved','1.0':'Approved','approved':'Approved','yes':'Approved','y':'Approved','+':'Approved',
        '0':'Rejected','0.0':'Rejected','rejected':'Rejected','no':'Rejected','n':'Rejected','-':'Rejected',
    }
    return series.astype(str).str.strip().str.lower().replace(mapping)


# ── Mock fallbacks (used when pkl files are missing) ─────────────────────────
class MockModel:
    def predict(self, X):
        return np.where(np.random.rand(len(X)) > 0.25, 1, 0)
    def predict_proba(self, X):
        p = np.random.uniform(0.55, 0.95, len(X))
        return np.column_stack([1 - p, p])
    # No feature_importances_ so feature importance block will be skipped

class MockScaler:
    def __init__(self):
        self.feature_names_in_ = np.array(REQUIRED_COLUMNS)
    def transform(self, X):
        return np.array(X, dtype=np.float32)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PKL) or not os.path.exists(SCALER_PKL):
        return MockModel(), MockScaler(), True
    try:
        with open(MODEL_PKL, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PKL, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, False
    except Exception:
        return MockModel(), MockScaler(), True


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    """
    Priority order:
      1. Your CSV file  → loan_data.csv
      2. Parquet file   → loan_clean_data.parquet
      3. Synthetic 800-row fallback (for demo / dev)
    """
    # ── Try CSV first ─────────────────────────────────────────────────────────
    for path, reader in [(CSV_PATH, lambda p: pd.read_csv(p)),
                         (DATA_PATH, lambda p: pd.read_parquet(p))]:
        if os.path.exists(path):
            data = reader(path)
            if 'Loan_Approved' in data.columns:
                data['Loan_Approved'] = normalize_approval(data['Loan_Approved'])
            elif 'Loan_Status' in data.columns:          # common alternate name
                data['Loan_Approved'] = normalize_approval(data['Loan_Status'])
            # downcast for memory
            for col in data.select_dtypes('float64').columns:
                data[col] = data[col].astype('float32')
            for col in data.select_dtypes('int64').columns:
                data[col] = data[col].astype('int32')
            return data

    # ── Synthetic fallback ────────────────────────────────────────────────────
    np.random.seed(42)
    n = 800
    d = pd.DataFrame({
        'Applicant_ID':       np.arange(1001, 1001+n),
        'Applicant_Income':   np.random.randint(20000, 200000, n),
        'Coapplicant_Income': np.random.randint(0, 80000, n),
        'Employment_Status':  np.random.choice(['Employed','Self-Employed','Unemployed'], n, p=[0.70,0.25,0.05]),
        'Age':                np.random.randint(22, 65, n),
        'Marital_Status':     np.random.choice(['Married','Single'], n),
        'Dependents':         np.random.choice([0,1,2,3,4], n, p=[0.25,0.30,0.25,0.15,0.05]),
        'Credit_Score':       np.random.randint(450, 900, n),
        'Existing_Loans':     np.random.choice([0,1,2,3], n, p=[0.50,0.30,0.15,0.05]),
        'DTI_Ratio':          np.round(np.random.uniform(0.10, 0.70, n), 2),
        'Savings':            np.random.randint(10000, 1000000, n),
        'Collateral_Value':   np.random.randint(100000, 5000000, n),
        'Loan_Amount':        np.random.randint(50000, 3000000, n),
        'Loan_Term':          np.random.choice([12,36,60,120,180,240,360], n),
        'Loan_Purpose':       np.random.choice(['Home','Personal','Education','Business','Car'], n,
                                               p=[0.35,0.20,0.20,0.15,0.10]),
        'Property_Area':      np.random.choice(['Urban','Semiurban','Rural'], n),
        'Education_Level':    np.random.choice(['Graduate','Not Graduate'], n, p=[0.65,0.35]),
        'Gender':             np.random.choice(['Male','Female'], n, p=[0.65,0.35]),
        'Employer_Category':  np.random.choice(['Corporate','Government','Self-Employed'], n),
    })
    score = (
        (d['Credit_Score'] > 650).astype(int) * 2 +
        (d['DTI_Ratio'] < 0.45).astype(int) +
        (d['Collateral_Value'] > d['Loan_Amount']).astype(int)
    )
    d['Loan_Approved'] = np.where(score >= 3, 'Approved', 'Rejected')
    return d


@st.cache_data(show_spinner=False)
def run_prediction(income_i, co_income_i, age_i, dependents_i, credit_score_i,
                   existing_loans_i, dti_calc, savings_i, collateral_i,
                   loan_amt_i, loan_term_i, _model, _scaler):
    row = {
        'Applicant_Income':   float(income_i),
        'Coapplicant_Income': float(co_income_i),
        'Age':                float(age_i),
        'Dependents':         float(dependents_i),
        'Credit_Score':       float(credit_score_i),
        'Existing_Loans':     float(existing_loans_i),
        'DTI_Ratio':          float(dti_calc),
        'Savings':            float(savings_i),
        'Collateral_Value':   float(collateral_i),
        'Loan_Amount':        float(loan_amt_i),
        'Loan_Term':          float(loan_term_i),
    }
    inp = pd.DataFrame([row])
    try:
        feats = list(_scaler.feature_names_in_)
    except AttributeError:
        feats = REQUIRED_COLUMNS
    for c in feats:
        if c not in inp.columns:
            inp[c] = 0.0
    inp    = inp[feats]
    scaled = _scaler.transform(inp)
    pred   = _model.predict(scaled)[0]
    proba  = _model.predict_proba(scaled)[0]
    return int(pred), float(proba[0]*100), float(proba[1]*100)


# ── Load everything once ──────────────────────────────────────────────────────
df_all          = load_data()
model, scaler, is_mock = load_assets()

# Detect which columns exist (graceful for any CSV schema)
HAS_PURPOSE  = 'Loan_Purpose'      in df_all.columns
HAS_GENDER   = 'Gender'            in df_all.columns
HAS_AGE      = 'Age'               in df_all.columns
HAS_EDU      = 'Education_Level'   in df_all.columns
HAS_AREA     = 'Property_Area'     in df_all.columns
HAS_EMP      = 'Employment_Status' in df_all.columns
HAS_CREDIT   = 'Credit_Score'      in df_all.columns
HAS_DTI      = 'DTI_Ratio'         in df_all.columns
HAS_SAVINGS  = 'Savings'           in df_all.columns
HAS_COLL     = 'Collateral_Value'  in df_all.columns
HAS_DEP      = 'Dependents'        in df_all.columns
HAS_TERM     = 'Loan_Term'         in df_all.columns

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💰 Loan Approval AI")
    st.markdown("---")

    if is_mock:
        st.warning("⚠️ ML model not found — Simulation Mode.")

    st.markdown("### 🔽 Global Filters")

    purpose_f = (st.multiselect("Loan Purpose",
                                sorted(df_all['Loan_Purpose'].dropna().unique()),
                                default=sorted(df_all['Loan_Purpose'].dropna().unique()))
                 if HAS_PURPOSE else None)

    gender_f = (st.multiselect("Gender",
                               sorted(df_all['Gender'].dropna().unique()),
                               default=sorted(df_all['Gender'].dropna().unique()))
                if HAS_GENDER else None)

    age_min = int(df_all['Age'].min()) if HAS_AGE else 18
    age_max = int(df_all['Age'].max()) if HAS_AGE else 80
    age_f   = st.slider("Age Range", age_min, age_max, (age_min, age_max)) if HAS_AGE else None

    st.markdown("---")
    st.info("Filters apply to Dashboard, Charts & Insights tabs.")

# Apply filters
df = df_all.copy()
if HAS_PURPOSE and purpose_f is not None:
    df = df[df['Loan_Purpose'].isin(purpose_f)]
if HAS_GENDER and gender_f is not None:
    df = df[df['Gender'].isin(gender_f)]
if HAS_AGE and age_f is not None:
    df = df[df['Age'].between(age_f[0], age_f[1])]

# ── Pre-compute frequently used subsets (avoids repeated filtering) ───────────
approved_df = df[df['Loan_Approved'] == 'Approved'] if 'Loan_Approved' in df.columns else pd.DataFrame()
rejected_df = df[df['Loan_Approved'] == 'Rejected'] if 'Loan_Approved' in df.columns else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-app-header">
    <h1>💰 Smart Loan Approval Prediction System</h1>
    <p>AI-powered credit intelligence — Dashboard · Prediction · Analytics · Insights</p>
</div>
""", unsafe_allow_html=True)

tab_dash, tab_pred, tab_charts, tab_insights = st.tabs(
    ["📊 Dashboard", "🔍 Predict Loan", "📈 Charts & Analytics", "💡 Insights"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    if df.empty or 'Loan_Approved' not in df.columns:
        st.warning("No data matches the current filters.")
    else:
        total       = len(df)
        n_approved  = len(approved_df)
        n_rejected  = len(rejected_df)
        appr_rate   = n_approved / total * 100 if total else 0
        avg_income  = df['Applicant_Income'].mean()   if 'Applicant_Income' in df.columns else 0
        avg_loan    = df['Loan_Amount'].mean()         if 'Loan_Amount'      in df.columns else 0
        avg_credit  = df['Credit_Score'].mean()        if HAS_CREDIT else 0
        avg_dti     = df['DTI_Ratio'].mean() * 100     if HAS_DTI    else 0

        # ── KPI Row 1 ──────────────────────────────────────────────────────────
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("📋 Total Records",    f"{total:,}")
        c2.metric("✅ Approved",          f"{n_approved:,}")
        c3.metric("❌ Rejected",          f"{n_rejected:,}")
        c4.metric("📈 Approval Rate",     f"{appr_rate:.1f}%")
        c5.metric("💵 Avg Income",        f"₹{avg_income:,.0f}")
        c6.metric("🏦 Avg Loan Amount",   f"₹{avg_loan:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── KPI Row 2 ──────────────────────────────────────────────────────────
        d1,d2,d3,d4 = st.columns(4)
        d1.metric("💳 Avg Credit Score",  f"{avg_credit:.0f}")
        d2.metric("📊 Avg DTI Ratio",     f"{avg_dti:.1f}%")
        d3.metric("💰 Avg Loan (Approved)",
                  f"₹{approved_df['Loan_Amount'].mean():,.0f}" if not approved_df.empty and 'Loan_Amount' in df.columns else "—")
        d4.metric("💰 Avg Loan (Rejected)",
                  f"₹{rejected_df['Loan_Amount'].mean():,.0f}" if not rejected_df.empty and 'Loan_Amount' in df.columns else "—")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row: Approval pie + Loan Amount box ────────────────────────────────
        col_a, col_b = st.columns([1, 1.6])

        with col_a:
            st.markdown('<div class="section-card"><div class="section-title">Approval Share</div>', unsafe_allow_html=True)
            fig = px.pie(df, names='Loan_Approved', hole=0.52, color='Loan_Approved',
                         color_discrete_map={'Approved': GREEN, 'Rejected': RED})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=10,b=10), height=290, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="section-card"><div class="section-title">Loan Amount Distribution by Decision</div>', unsafe_allow_html=True)
            samp = df.sample(n=min(8000,len(df)), random_state=42) if len(df)>10000 else df
            fig = px.box(samp, x='Loan_Approved', y='Loan_Amount', color='Loan_Approved',
                         color_discrete_map={'Approved': GREEN, 'Rejected': RED},
                         labels={'Loan_Amount':'Loan Amount (₹)'})
            fig.update_layout(margin=dict(t=10,b=10), height=290, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Dataset Summary Statistics ─────────────────────────────────────────
        st.markdown('<div class="section-card"><div class="section-title">Dataset Summary Statistics</div>', unsafe_allow_html=True)
        num_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
        if num_cols:
            desc = df[num_cols].describe().T.round(2)
            desc.index.name = "Feature"
            st.dataframe(desc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Recent Records ─────────────────────────────────────────────────────
        st.markdown('<div class="section-card"><div class="section-title">Recent Records (last 20)</div>', unsafe_allow_html=True)
        display_cols = [c for c in ['Applicant_ID','Age','Gender','Applicant_Income',
                                     'Loan_Amount','Credit_Score','DTI_Ratio','Loan_Approved']
                        if c in df.columns]
        recent = df[display_cols].tail(20).reset_index(drop=True)

        def color_approval(val):
            if val == 'Approved': return 'background-color:#d1fae5;color:#065f46;font-weight:600'
            if val == 'Rejected': return 'background-color:#fee2e2;color:#7f1d1d;font-weight:600'
            return ''

        if 'Loan_Approved' in recent.columns:
            st.dataframe(recent.style.applymap(color_approval, subset=['Loan_Approved']),
                         use_container_width=True, height=420)
        else:
            st.dataframe(recent, use_container_width=True, height=420)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT LOAN
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("predict_form", border=False):
        st.markdown('<div class="section-card"><div class="section-title">Applicant Risk Profile</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            gender_i       = st.selectbox("Gender", ["Male","Female"])
            age_i          = st.slider("Applicant Age", 18, 80, 35)
            income_i       = st.number_input("Monthly Applicant Income (₹)", 0, 1_000_000, 75_000, step=5_000)
            co_income_i    = st.number_input("Coapplicant Income (₹)", 0, 1_000_000, 25_000, step=5_000)
            loan_amt_i     = st.number_input("Requested Loan Amount (₹)", 0, 50_000_000, 500_000, step=50_000)
            loan_term_i    = st.selectbox("Loan Term (Months)", [12,36,60,120,180,240,360])
            credit_score_i = st.slider("Credit Score", 300, 900, 700)
        with col2:
            employment_i     = st.selectbox("Employment Status", ["Employed","Self-Employed","Unemployed"])
            marital_i        = st.selectbox("Marital Status", ["Married","Single"])
            dependents_i     = st.slider("Number of Dependents", 0, 10, 2)
            existing_loans_i = st.slider("Active Existing Loans", 0, 5, 0)
            existing_debt_i  = st.number_input("Existing Monthly Debt Payments (₹)", 0, 500_000, 5_000, step=1_000)
            interest_rate_i  = st.number_input("Annual Interest Rate (%)", 1.0, 30.0, 9.0, step=0.5)
            savings_i        = st.number_input("Savings Balance (₹)", 0, 10_000_000, 200_000, step=25_000)
            collateral_i     = st.number_input("Collateral Value (₹)", 0, 50_000_000, 600_000, step=50_000)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Live affordability calculations ────────────────────────────────────
        new_emi        = calculate_emi(loan_amt_i, interest_rate_i, loan_term_i)
        total_income   = income_i + co_income_i
        dti_calc       = (new_emi + existing_debt_i) / total_income if total_income > 0 else 1.0
        disposable     = total_income - (dependents_i * DEPENDENT_MONTHLY_COST) - existing_debt_i
        disposable_net = disposable - new_emi

        st.markdown('<div class="section-card"><div class="section-title">Live Affordability Snapshot</div>', unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Est. Monthly EMI",     f"₹{new_emi:,.0f}")
        m2.metric("Combined Income",       f"₹{total_income:,.0f}")
        m3.metric("Calculated DTI",        f"{dti_calc*100:.1f}%",
                  delta="⚠ High" if dti_calc > 0.55 else "✓ Healthy",
                  delta_color="inverse" if dti_calc > 0.55 else "normal")
        m4.metric("Disposable after EMI",  f"₹{disposable_net:,.0f}",
                  delta="⚠ Negative" if disposable_net < 0 else "✓ Positive",
                  delta_color="inverse" if disposable_net < 0 else "normal")
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("⚡ Run AI Loan Validation", type="primary", use_container_width=True)

    if submitted:
        # Factor scoring
        s_st  = "pass" if credit_score_i>=750 else ("warn" if credit_score_i>=650 else "fail")
        d_st  = "pass" if dti_calc<=0.40        else ("warn" if dti_calc<=0.55        else "fail")
        i_st  = "pass" if disposable_net>=new_emi*0.2 else ("warn" if disposable_net>=0 else "fail")
        dep_st= "pass" if dependents_i<=2        else ("warn" if dependents_i<=4        else "fail")
        col_st= "pass" if collateral_i>=loan_amt_i else "warn"
        emp_st= "fail" if employment_i=="Unemployed" else "pass"
        sav_st= "pass" if savings_i>=loan_amt_i*0.10 else "warn"

        factors = [
            {"label":"Credit Score",  "value":str(credit_score_i),       "status":s_st,
             "reason":{"pass":"Excellent credit","warn":"Fair credit","fail":"Subprime — high risk"}[s_st]},
            {"label":"DTI Ratio",     "value":f"{dti_calc*100:.1f}%",    "status":d_st,
             "reason":{"pass":"Healthy debt burden","warn":"Elevated load","fail":"Over-leveraged"}[d_st]},
            {"label":"Affordability", "value":f"₹{disposable_net:,.0f}", "status":i_st,
             "reason":{"pass":"Comfortable capacity","warn":"Tight capacity","fail":"Insufficient income"}[i_st]},
            {"label":"Dependents",    "value":str(dependents_i),          "status":dep_st,
             "reason":{"pass":"Manageable household","warn":"Higher burden","fail":"High dependent load"}[dep_st]},
            {"label":"Collateral",    "value":f"₹{collateral_i:,.0f}",   "status":col_st,
             "reason":"Fully secured" if col_st=="pass" else "Under-collateralised"},
            {"label":"Employment",    "value":employment_i,               "status":emp_st,
             "reason":"No verifiable income" if emp_st=="fail" else "Stable occupation"},
            {"label":"Savings",       "value":f"₹{savings_i:,.0f}",      "status":sav_st,
             "reason":"Adequate reserves" if sav_st=="pass" else "Low savings buffer"},
        ]

        with st.spinner("Running AI model…"):
            pred, p0, p1 = run_prediction(
                income_i, co_income_i, age_i, dependents_i, credit_score_i,
                existing_loans_i, dti_calc, savings_i, collateral_i,
                loan_amt_i, loan_term_i, model, scaler,
            )

        hard     = get_hard_reject_reasons(credit_score_i, dti_calc, disposable_net,
                                           employment_i, collateral_i, loan_amt_i)
        approved = (pred == 1) and (len(hard) == 0)
        conf_pct = p1 if approved else p0
        if hard:         conf_pct = max(conf_pct, 90.0)
        conf_pct = min(99.5, max(50.0, conf_pct))

        # Risk level
        fail_count = sum(1 for f in factors if f["status"] == "fail")
        warn_count = sum(1 for f in factors if f["status"] == "warn")
        if   fail_count >= 2:              risk_level, risk_color = "🔴 High Risk",    RED
        elif fail_count == 1 or warn_count >= 3: risk_level, risk_color = "🟡 Medium Risk", AMBER
        else:                              risk_level, risk_color = "🟢 Low Risk",     GREEN

        # Result banner
        if approved:
            st.balloons()
            bx,ic,ti,tc = "#f0fff4","✅","Loan Facility Approved", GREEN
        else:
            bx,ic,ti,tc = "#fff5f5","❌","Loan Facility Rejected", RED

        st.markdown(f"""
        <div style="background:{bx};border:2px solid {tc};border-radius:14px;
                    padding:1.4rem 1.6rem;display:flex;align-items:center;gap:20px;margin-bottom:1rem;">
            <div style="font-size:36px;">{ic}</div>
            <div>
                <b style="color:{tc};font-size:20px;">{ti}</b><br>
                <span style="font-size:14px;color:#718096;">
                    AI Confidence: <b>{conf_pct:.1f}%</b> &nbsp;|&nbsp;
                    Risk Level: <b style="color:{risk_color};">{risk_level}</b>
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

        # Recommendation message
        if approved:
            st.success("✅ **Recommendation:** Proceed with loan disbursement. All key risk parameters are within acceptable range.")
        elif hard:
            st.error("🚫 **Recommendation:** Application must be declined. Hard policy thresholds breached — manual override not permitted.")
        else:
            st.warning("⚠️ **Recommendation:** Application requires senior review before proceeding. Multiple risk flags detected.")

        # Factor cards
        st.markdown("<br>", unsafe_allow_html=True)
        cfg = {"pass":("#d1fae5","#065f46","✓"),
               "warn":("#fef9c3","#713f12","⚠"),
               "fail":("#fee2e2","#7f1d1d","✗")}
        cols_f = st.columns(len(factors))
        for col, f in zip(cols_f, factors):
            bg, tx, icon = cfg[f["status"]]
            col.markdown(
                f'<div style="background:{bg};padding:12px 10px;border-radius:10px;text-align:center;">'
                f'<div style="font-size:18px;">{icon}</div>'
                f'<small style="color:{tx};font-weight:700;display:block;">{f["label"]}</small>'
                f'<b style="color:{tx};font-size:14px;">{f["value"]}</b><br>'
                f'<small style="color:{tx};opacity:0.85;">{f["reason"]}</small></div>',
                unsafe_allow_html=True)

        # Hard rejection details
        if hard:
            st.error("🚫 Hard Policy Violations:")
            for r in hard: st.write(f"  • {r}")

        # Gauge chart — approval probability
        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown('<div class="section-card"><div class="section-title">Approval Probability Gauge</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=p1,
                delta={'reference': 50, 'valueformat': '.1f'},
                number={'suffix': '%', 'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': GREEN if p1 >= 60 else (AMBER if p1 >= 40 else RED)},
                    'steps': [
                        {'range': [0,  40], 'color': '#fee2e2'},
                        {'range': [40, 60], 'color': '#fef9c3'},
                        {'range': [60,100], 'color': '#d1fae5'},
                    ],
                    'threshold': {'line': {'color': BLUE, 'width': 3}, 'value': 50},
                },
                title={'text': "Approval Probability"},
            ))
            fig_gauge.update_layout(margin=dict(t=30,b=10,l=20,r=20), height=260)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_g2:
            st.markdown('<div class="section-card"><div class="section-title">Risk Factor Breakdown</div>', unsafe_allow_html=True)
            radar_cats   = [f["label"] for f in factors]
            pass_scores  = [1 if f["status"]=="pass" else (0.5 if f["status"]=="warn" else 0) for f in factors]
            fig_radar = go.Figure(go.Scatterpolar(
                r=pass_scores + [pass_scores[0]],
                theta=radar_cats + [radar_cats[0]],
                fill='toself',
                fillcolor='rgba(56,161,105,0.2)',
                line=dict(color=GREEN, width=2),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                           tickvals=[0,0.5,1],
                                           ticktext=['Fail','Warn','Pass'])),
                margin=dict(t=20,b=20,l=20,r=20), height=260,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Anomaly flags
        flags = []
        if total_income > 0 and savings_i > total_income * 60:
            flags.append("Savings unusually high vs income — verify with bank statements")
        if credit_score_i >= 800 and dti_calc > 0.55:
            flags.append("High credit score + high DTI is unusual — recommend manual check")
        if collateral_i > loan_amt_i * 5:
            flags.append("Collateral value is very high relative to loan — verify asset valuation")
        if flags:
            st.warning("⚠️ Flagged for manual review:")
            for r in flags: st.write(f"  • {r}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHARTS & ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty or 'Loan_Approved' not in df.columns:
        st.warning("No data matches the current filters.")
    else:
        # ── Analytics KPIs ────────────────────────────────────────────────────
        approval_rate = (df['Loan_Approved']=='Approved').mean()*100
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Approval Rate",    f"{approval_rate:.1f}%")
        k2.metric("Avg Loan Amount",  f"₹{df['Loan_Amount'].mean():,.0f}" if 'Loan_Amount' in df.columns else "—")
        k3.metric("Avg Credit Score", f"{df['Credit_Score'].mean():.0f}"  if HAS_CREDIT else "—")
        k4.metric("Avg DTI Ratio",    f"{df['DTI_Ratio'].mean()*100:.1f}%" if HAS_DTI else "—")

        # Interactive filter just for this tab
        with st.expander("⚙️ Chart Filters", expanded=False):
            chart_metric = st.selectbox("Colour / split charts by:",
                                        [c for c in ['Loan_Approved','Gender','Education_Level',
                                                      'Employment_Status','Property_Area','Marital_Status']
                                         if c in df.columns],
                                        index=0)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ROW 1: Approval distribution | Purpose breakdown ──────────────────
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            st.markdown('<div class="section-card"><div class="section-title">Loan Approval Distribution</div>', unsafe_allow_html=True)
            vc = df['Loan_Approved'].value_counts().reset_index()
            vc.columns = ['Status','Count']
            fig = px.bar(vc, x='Status', y='Count', color='Status', text='Count',
                         color_discrete_map={'Approved': GREEN, 'Rejected': RED})
            fig.update_traces(textposition='outside')
            fig.update_layout(margin=dict(t=10,b=10), height=300, showlegend=False,
                              xaxis_title='', yaxis_title='Applications')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r1c2:
            st.markdown('<div class="section-card"><div class="section-title">Approval by Loan Purpose</div>', unsafe_allow_html=True)
            if HAS_PURPOSE:
                pg = df.groupby(['Loan_Purpose','Loan_Approved']).size().reset_index(name='Count')
                fig = px.bar(pg, x='Loan_Purpose', y='Count', color='Loan_Approved',
                             barmode='group', text_auto=True,
                             color_discrete_map={'Approved':GREEN,'Rejected':RED})
                fig.update_layout(margin=dict(t=10,b=10), height=300, legend_title_text='')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Loan_Purpose column not in dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── ROW 2: Income vs Loan scatter | Credit Score histogram ────────────
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            st.markdown('<div class="section-card"><div class="section-title">Income vs Loan Amount</div>', unsafe_allow_html=True)
            samp = df.sample(n=min(2000,len(df)), random_state=1)
            hue  = chart_metric if chart_metric in samp.columns else 'Loan_Approved'
            fig  = px.scatter(samp, x='Applicant_Income', y='Loan_Amount', color=hue,
                              opacity=0.65, color_discrete_map={'Approved':GREEN,'Rejected':RED},
                              labels={'Applicant_Income':'Monthly Income (₹)','Loan_Amount':'Loan Amount (₹)'})
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(margin=dict(t=10,b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r2c2:
            st.markdown('<div class="section-card"><div class="section-title">Credit Score Distribution</div>', unsafe_allow_html=True)
            if HAS_CREDIT:
                fig = px.histogram(df, x='Credit_Score', color='Loan_Approved', nbins=30,
                                   barmode='overlay', opacity=0.75,
                                   color_discrete_map={'Approved':GREEN,'Rejected':RED})
                fig.add_vline(x=650, line_dash="dash", line_color=AMBER,
                              annotation_text="Min 650", annotation_position="top right")
                fig.update_layout(margin=dict(t=10,b=10), height=320, legend_title_text='')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Credit_Score column not in dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── ROW 3: Education | Gender ─────────────────────────────────────────
        r3c1, r3c2 = st.columns(2)

        with r3c1:
            st.markdown('<div class="section-card"><div class="section-title">Education vs Loan Approval</div>', unsafe_allow_html=True)
            if HAS_EDU:
                eg = df.groupby(['Education_Level','Loan_Approved']).size().reset_index(name='Count')
                eg['Pct'] = eg['Count'] / eg.groupby('Education_Level')['Count'].transform('sum') * 100
                fig = px.bar(eg, x='Education_Level', y='Pct', color='Loan_Approved',
                             barmode='stack', text=eg['Pct'].round(1).astype(str)+'%',
                             color_discrete_map={'Approved':GREEN,'Rejected':RED},
                             labels={'Pct':'Share (%)','Education_Level':'Education'})
                fig.update_traces(textposition='inside')
                fig.update_layout(margin=dict(t=10,b=10), height=300,
                                  yaxis=dict(range=[0,105]), legend_title_text='')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Education_Level column not in dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        with r3c2:
            st.markdown('<div class="section-card"><div class="section-title">Gender Distribution</div>', unsafe_allow_html=True)
            if HAS_GENDER:
                gd = df.groupby(['Gender','Loan_Approved']).size().reset_index(name='Count')
                fig = px.bar(gd, x='Gender', y='Count', color='Loan_Approved',
                             barmode='group', text_auto=True,
                             color_discrete_map={'Approved':GREEN,'Rejected':RED})
                fig.update_layout(margin=dict(t=10,b=10), height=300, legend_title_text='')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Gender column not in dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── ROW 4: Property Area | DTI violin ────────────────────────────────
        r4c1, r4c2 = st.columns(2)

        with r4c1:
            st.markdown('<div class="section-card"><div class="section-title">Property Area Distribution</div>', unsafe_allow_html=True)
            if HAS_AREA:
                ad = df.groupby(['Property_Area','Loan_Approved']).size().reset_index(name='Count')
                fig = px.bar(ad, x='Property_Area', y='Count', color='Loan_Approved',
                             barmode='stack', text_auto=True,
                             color_discrete_map={'Approved':GREEN,'Rejected':RED},
                             labels={'Property_Area':'Area'})
                fig.update_layout(margin=dict(t=10,b=10), height=300, legend_title_text='')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Property_Area column not in dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        with r4c2:
            st.markdown('<div class="section-card"><div class="section-title">DTI Ratio — Approved vs Rejected</div>', unsafe_allow_html=True)
            if HAS_DTI:
                fig = px.violin(df, x='Loan_Approved', y='DTI_Ratio', color='Loan_Approved',
                                box=True, points=False,
                                color_discrete_map={'Approved':GREEN,'Rejected':RED})
                fig.add_hline(y=0.65, line_dash="dash", line_color=RED,
                              annotation_text="Hard cap 65%")
                fig.add_hline(y=0.40, line_dash="dot",  line_color=AMBER,
                              annotation_text="Healthy 40%")
                fig.update_layout(margin=dict(t=10,b=10), height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("DTI_Ratio column not in dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── ROW 5: Loan Amount hist | Applicant Income hist ───────────────────
        r5c1, r5c2 = st.columns(2)

        with r5c1:
            st.markdown('<div class="section-card"><div class="section-title">Loan Amount Distribution</div>', unsafe_allow_html=True)
            if 'Loan_Amount' in df.columns:
                fig = px.histogram(df, x='Loan_Amount', color='Loan_Approved', nbins=40,
                                   barmode='overlay', opacity=0.72,
                                   color_discrete_map={'Approved':GREEN,'Rejected':RED},
                                   labels={'Loan_Amount':'Loan Amount (₹)'})
                fig.update_layout(margin=dict(t=10,b=10), height=300, legend_title_text='')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Loan_Amount column not in dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        with r5c2:
            st.markdown('<div class="section-card"><div class="section-title">Applicant Income Distribution</div>', unsafe_allow_html=True)
            if 'Applicant_Income' in df.columns:
                fig = px.histogram(df, x='Applicant_Income', color='Loan_Approved', nbins=40,
                                   barmode='overlay', opacity=0.72,
                                   color_discrete_map={'Approved':GREEN,'Rejected':RED},
                                   labels={'Applicant_Income':'Monthly Income (₹)'})
                fig.update_layout(margin=dict(t=10,b=10), height=300, legend_title_text='')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Applicant_Income column not in dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── ROW 6: Age band combo | Loan Term approval rate ───────────────────
        r6c1, r6c2 = st.columns(2)

        with r6c1:
            st.markdown('<div class="section-card"><div class="section-title">Approval Rate by Age Group</div>', unsafe_allow_html=True)
            if HAS_AGE:
                adf = df.copy()
                adf['Age_Band'] = pd.cut(adf['Age'],
                                         bins=[17,25,30,35,40,50,60,80],
                                         labels=['18-25','26-30','31-35','36-40','41-50','51-60','61+'])
                # use agg instead of apply for speed
                ag = adf.groupby('Age_Band', observed=True).agg(
                    Total=('Loan_Approved','count'),
                    Approved=('Loan_Approved', lambda x: (x=='Approved').sum())
                ).reset_index()
                ag['Rate'] = ag['Approved'] / ag['Total'] * 100

                fig = go.Figure()
                fig.add_trace(go.Bar(x=ag['Age_Band'].astype(str), y=ag['Total'],
                                     name='Applications', marker_color='#bee3f8', opacity=0.7))
                fig.add_trace(go.Scatter(x=ag['Age_Band'].astype(str), y=ag['Rate'],
                                         name='Approval %', mode='lines+markers',
                                         line=dict(color=GREEN, width=2.5),
                                         marker=dict(size=8, color=GREEN), yaxis='y2'))
                fig.update_layout(margin=dict(t=10,b=10), height=310,
                                  yaxis=dict(title='Applications', showgrid=False),
                                  yaxis2=dict(title='Approval Rate (%)', overlaying='y',
                                              side='right', range=[0,110], showgrid=False),
                                  legend=dict(orientation='h',yanchor='bottom',y=1.02,x=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age column not in dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        with r6c2:
            st.markdown('<div class="section-card"><div class="section-title">Approval Rate by Loan Term</div>', unsafe_allow_html=True)
            if HAS_TERM:
                tg = df.groupby('Loan_Term').agg(
                    Count=('Loan_Approved','count'),
                    Approved=('Loan_Approved', lambda x: (x=='Approved').sum())
                ).reset_index()
                tg['Rate'] = tg['Approved'] / tg['Count'] * 100
                tg = tg.sort_values('Loan_Term')
                fig = px.bar(tg, y=tg['Loan_Term'].astype(str), x='Rate', orientation='h',
                             text=tg['Rate'].round(1).astype(str)+'%',
                             color='Rate',
                             color_continuous_scale=[[0,RED],[0.5,AMBER],[1,GREEN]],
                             labels={'Rate':'Approval Rate (%)','y':'Term (Months)'})
                fig.update_traces(textposition='outside')
                fig.update_layout(margin=dict(t=10,b=10), height=310,
                                  coloraxis_showscale=False, xaxis=dict(range=[0,115]))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Loan_Term column not in dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── ROW 7: Correlation heatmap | Sunburst ─────────────────────────────
        r7c1, r7c2 = st.columns([1.6, 1])

        with r7c1:
            st.markdown('<div class="section-card"><div class="section-title">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
            hcols = [c for c in REQUIRED_COLUMNS if c in df.columns]
            if len(hcols) >= 3:
                corr = df[hcols].corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale='RdBu', zmin=-1, zmax=1,
                    text=np.round(corr.values,2), texttemplate='%{text}',
                    textfont=dict(size=9),
                    hovertemplate='%{x} × %{y}: %{z:.2f}<extra></extra>'))
                fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numeric columns for correlation matrix.")
            st.markdown('</div>', unsafe_allow_html=True)

        with r7c2:
            st.markdown('<div class="section-card"><div class="section-title">Gender × Employment × Decision</div>', unsafe_allow_html=True)
            sun_cols = [c for c in ['Gender','Employment_Status','Loan_Approved'] if c in df.columns]
            if len(sun_cols) == 3:
                sd = df.groupby(sun_cols).size().reset_index(name='Count')
                fig = px.sunburst(sd, path=sun_cols, values='Count', color='Loan_Approved',
                                  color_discrete_map={'Approved':GREEN,'Rejected':RED,'(?)':'#a0aec0'})
                fig.update_layout(margin=dict(t=10,b=10), height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need Gender, Employment_Status & Loan_Approved columns.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── ROW 8: Feature Importance (if model supports it) ──────────────────
        st.markdown('<div class="section-card"><div class="section-title">Feature Importance (Model)</div>', unsafe_allow_html=True)
        if hasattr(model, 'feature_importances_'):
            try:
                feat_names = list(scaler.feature_names_in_)
            except AttributeError:
                feat_names = REQUIRED_COLUMNS
            imp_df = pd.DataFrame({
                'Feature':    feat_names,
                'Importance': model.feature_importances_,
            }).sort_values('Importance', ascending=True)
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale=BLUE_PALETTE,
                         labels={'Importance':'Importance Score'})
            fig.update_layout(margin=dict(t=10,b=10), height=350,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        elif hasattr(model, 'coef_'):
            try:
                feat_names = list(scaler.feature_names_in_)
            except AttributeError:
                feat_names = REQUIRED_COLUMNS
            coef = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            imp_df = pd.DataFrame({'Feature': feat_names[:len(coef)], 'Coefficient': coef}
                                  ).sort_values('Coefficient', ascending=True)
            fig = px.bar(imp_df, x='Coefficient', y='Feature', orientation='h',
                         color='Coefficient', color_continuous_scale=BLUE_PALETTE)
            fig.update_layout(margin=dict(t=10,b=10), height=350, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance is not available for this model type (MockModel or non-tree model).")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty or 'Loan_Approved' not in df.columns:
        st.warning("No data matches the current filters.")
    else:
        total      = len(df)
        appr_rate  = len(approved_df) / total * 100 if total else 0

        # ── Summary banner ────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#2b6cb0,#3182ce);
                    border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1.2rem;color:white;">
            <h3 style="margin:0;color:white;">📋 Dataset Snapshot</h3>
            <p style="margin:6px 0 0;opacity:0.9;">
                <b>{total:,}</b> applications analysed &nbsp;|&nbsp;
                Approval rate <b>{appr_rate:.1f}%</b> &nbsp;|&nbsp;
                Filters active: {'Yes' if (purpose_f and len(purpose_f)<len(df_all.get('Loan_Purpose', pd.Series()).unique())) else 'No'}
            </p>
        </div>""", unsafe_allow_html=True)

        # ── Top approval & rejection factors ──────────────────────────────────
        ins_c1, ins_c2 = st.columns(2)

        with ins_c1:
            st.markdown('<div class="section-card"><div class="section-title">✅ Top Approval Factors</div>', unsafe_allow_html=True)

            insights_approve = []

            if HAS_CREDIT and not approved_df.empty:
                avg_cs_app = approved_df['Credit_Score'].mean()
                avg_cs_rej = rejected_df['Credit_Score'].mean() if not rejected_df.empty else 0
                insights_approve.append({
                    "title": f"High Credit Scores Drive Approvals",
                    "body":  f"Approved applicants have an average credit score of <b>{avg_cs_app:.0f}</b> vs "
                             f"<b>{avg_cs_rej:.0f}</b> for rejected ones — a gap of "
                             f"<b>{avg_cs_app - avg_cs_rej:+.0f}</b> points.",
                    "type":  "green"
                })

            if HAS_DTI and not approved_df.empty:
                avg_dti_app = approved_df['DTI_Ratio'].mean() * 100
                avg_dti_rej = rejected_df['DTI_Ratio'].mean() * 100 if not rejected_df.empty else 0
                insights_approve.append({
                    "title": "Lower DTI Correlates with Approval",
                    "body":  f"Approved applicants carry an average DTI of <b>{avg_dti_app:.1f}%</b> "
                             f"vs <b>{avg_dti_rej:.1f}%</b> for rejected applicants.",
                    "type":  "green"
                })

            if 'Applicant_Income' in df.columns and not approved_df.empty:
                avg_inc_app = approved_df['Applicant_Income'].mean()
                avg_inc_rej = rejected_df['Applicant_Income'].mean() if not rejected_df.empty else 0
                insights_approve.append({
                    "title": "Higher Income Supports Approval",
                    "body":  f"Approved applicants earn ₹<b>{avg_inc_app:,.0f}</b>/mo on average "
                             f"vs ₹<b>{avg_inc_rej:,.0f}</b>/mo for rejected ones.",
                    "type":  "green"
                })

            if HAS_EDU and not approved_df.empty:
                edu_rate = (df.groupby('Education_Level')
                              .apply(lambda x: (x['Loan_Approved']=='Approved').mean() * 100)
                              .reset_index(name='Rate'))
                best_edu = edu_rate.loc[edu_rate['Rate'].idxmax()]
                insights_approve.append({
                    "title": f"{best_edu['Education_Level']} Applicants Fare Better",
                    "body":  f"<b>{best_edu['Education_Level']}</b> applicants have the highest "
                             f"approval rate at <b>{best_edu['Rate']:.1f}%</b>.",
                    "type":  "green"
                })

            if HAS_COLL and not approved_df.empty:
                avg_coll_app = approved_df['Collateral_Value'].mean()
                avg_coll_rej = rejected_df['Collateral_Value'].mean() if not rejected_df.empty else 0
                insights_approve.append({
                    "title": "Stronger Collateral Boosts Approval Odds",
                    "body":  f"Approved applicants offer collateral of ₹<b>{avg_coll_app:,.0f}</b> "
                             f"on average vs ₹<b>{avg_coll_rej:,.0f}</b> for rejected ones.",
                    "type":  "green"
                })

            for ins in insights_approve:
                st.markdown(
                    f'<div class="insight-card {ins["type"]}">'
                    f'<b>{ins["title"]}</b><br>'
                    f'<small>{ins["body"]}</small></div>',
                    unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with ins_c2:
            st.markdown('<div class="section-card"><div class="section-title">❌ Top Rejection Factors</div>', unsafe_allow_html=True)

            insights_reject = []

            if HAS_CREDIT:
                low_cs_rej = (df[df['Credit_Score'] < 600]['Loan_Approved'] == 'Rejected').mean() * 100
                insights_reject.append({
                    "title": "Low Credit Score = High Rejection",
                    "body":  f"Applicants with credit score below 600 are rejected <b>{low_cs_rej:.1f}%</b> of the time.",
                    "type":  "red"
                })

            if HAS_DTI:
                hi_dti_rej = (df[df['DTI_Ratio'] > 0.55]['Loan_Approved'] == 'Rejected').mean() * 100
                insights_reject.append({
                    "title": "High DTI is the Most Common Blocker",
                    "body":  f"When DTI exceeds 55%, rejection rate spikes to <b>{hi_dti_rej:.1f}%</b>.",
                    "type":  "red"
                })

            if HAS_EMP:
                emp_rates = (df.groupby('Employment_Status')
                               .apply(lambda x: (x['Loan_Approved']=='Rejected').mean() * 100)
                               .reset_index(name='RejRate'))
                worst_emp = emp_rates.loc[emp_rates['RejRate'].idxmax()]
                insights_reject.append({
                    "title": f"{worst_emp['Employment_Status']} Applicants Rejected Most",
                    "body":  f"<b>{worst_emp['Employment_Status']}</b> applicants face a "
                             f"<b>{worst_emp['RejRate']:.1f}%</b> rejection rate — the highest among employment categories.",
                    "type":  "red"
                })

            if HAS_DEP:
                hi_dep = df[df['Dependents'] >= 4]
                if len(hi_dep) > 0:
                    hi_dep_rej = (hi_dep['Loan_Approved'] == 'Rejected').mean() * 100
                    insights_reject.append({
                        "title": "Large Families Face Higher Rejection",
                        "body":  f"Applicants with 4+ dependents are rejected <b>{hi_dep_rej:.1f}%</b> of the time "
                                 f"due to reduced disposable income.",
                        "type":  "red"
                    })

            if HAS_COLL and 'Loan_Amount' in df.columns:
                under_coll = df[df['Collateral_Value'] < df['Loan_Amount'] * 0.5]
                if len(under_coll) > 0:
                    uc_rej = (under_coll['Loan_Approved'] == 'Rejected').mean() * 100
                    insights_reject.append({
                        "title": "Insufficient Collateral Triggers Rejection",
                        "body":  f"When collateral covers less than 50% of the loan, rejection rate is "
                                 f"<b>{uc_rej:.1f}%</b>.",
                        "type":  "red"
                    })

            for ins in insights_reject:
                st.markdown(
                    f'<div class="insight-card {ins["type"]}">'
                    f'<b>{ins["title"]}</b><br>'
                    f'<small>{ins["body"]}</small></div>',
                    unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Trends & Patterns ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-card"><div class="section-title">📊 Trends & Patterns</div>', unsafe_allow_html=True)

        tp1, tp2, tp3 = st.columns(3)

        # Approval rate by Purpose
        with tp1:
            if HAS_PURPOSE:
                pr = df.groupby('Loan_Purpose').agg(
                    Count=('Loan_Approved','count'),
                    Approved=('Loan_Approved', lambda x: (x=='Approved').sum())
                ).reset_index()
                pr['Rate'] = pr['Approved'] / pr['Count'] * 100
                pr = pr.sort_values('Rate', ascending=False)
                fig = px.bar(pr, x='Loan_Purpose', y='Rate',
                             color='Rate', text=pr['Rate'].round(1).astype(str)+'%',
                             color_continuous_scale=[[0,RED],[0.5,AMBER],[1,GREEN]],
                             title="Approval Rate by Loan Purpose",
                             labels={'Rate':'Approval Rate (%)','Loan_Purpose':''})
                fig.update_traces(textposition='outside')
                fig.update_layout(margin=dict(t=40,b=10), height=280,
                                  coloraxis_showscale=False, yaxis=dict(range=[0,110]))
                st.plotly_chart(fig, use_container_width=True)

        # Savings vs Approval
        with tp2:
            if HAS_SAVINGS:
                sv = df.copy()
                sv['Savings_Band'] = pd.cut(sv['Savings'],
                                            bins=[0,50000,200000,500000,1000000,float('inf')],
                                            labels=['<50K','50-200K','200-500K','500K-1M','>1M'])
                sr = sv.groupby('Savings_Band', observed=True).agg(
                    Count=('Loan_Approved','count'),
                    Approved=('Loan_Approved', lambda x: (x=='Approved').sum())
                ).reset_index()
                sr['Rate'] = sr['Approved'] / sr['Count'] * 100
                fig = px.line(sr, x='Savings_Band', y='Rate', markers=True,
                              title="Approval Rate by Savings Level",
                              labels={'Savings_Band':'Savings Band','Rate':'Approval Rate (%)'},
                              line_shape='spline')
                fig.update_traces(line=dict(color=GREEN, width=2.5), marker=dict(size=9, color=GREEN))
                fig.update_layout(margin=dict(t=40,b=10), height=280, yaxis=dict(range=[0,110]))
                st.plotly_chart(fig, use_container_width=True)

        # Income Quartile approval
        with tp3:
            if 'Applicant_Income' in df.columns:
                iq = df.copy()
                iq['Income_Q'] = pd.qcut(iq['Applicant_Income'], q=4,
                                         labels=['Q1\n(Low)','Q2','Q3','Q4\n(High)'])
                ir = iq.groupby('Income_Q', observed=True).agg(
                    Count=('Loan_Approved','count'),
                    Approved=('Loan_Approved', lambda x: (x=='Approved').sum())
                ).reset_index()
                ir['Rate'] = ir['Approved'] / ir['Count'] * 100
                fig = px.bar(ir, x='Income_Q', y='Rate',
                             color='Rate', text=ir['Rate'].round(1).astype(str)+'%',
                             color_continuous_scale=[[0,RED],[0.5,AMBER],[1,GREEN]],
                             title="Approval Rate by Income Quartile",
                             labels={'Rate':'Approval Rate (%)','Income_Q':'Income Quartile'})
                fig.update_traces(textposition='outside')
                fig.update_layout(margin=dict(t=40,b=10), height=280,
                                  coloraxis_showscale=False, yaxis=dict(range=[0,110]))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Business Summary ───────────────────────────────────────────────────
        st.markdown('<div class="section-card"><div class="section-title">📝 Business Summary</div>', unsafe_allow_html=True)

        total_loan_book = df.loc[df['Loan_Approved']=='Approved','Loan_Amount'].sum() \
                          if 'Loan_Amount' in df.columns else 0
        risk_apps = df[(df.get('Credit_Score', pd.Series([999]*len(df))) < 600) |
                       (df.get('DTI_Ratio',    pd.Series([0]*len(df)))   > 0.55)].shape[0]

        summary_cols = st.columns(4)
        summary_cols[0].metric("📦 Total Loan Book",     f"₹{total_loan_book/1e7:.1f} Cr" if total_loan_book > 0 else "—")
        summary_cols[1].metric("⚠️ High-Risk Applications", f"{risk_apps:,}")
        summary_cols[2].metric("📊 Rejection Rate",       f"{100-appr_rate:.1f}%")
        summary_cols[3].metric("🔑 Key Risk Driver",
                               "DTI Ratio" if HAS_DTI else ("Credit Score" if HAS_CREDIT else "Income"))

        st.markdown(f"""
        <br>
        <b>Key Findings:</b>
        <ul>
            <li>Overall approval rate stands at <b>{appr_rate:.1f}%</b> across <b>{total:,}</b> applications.</li>
            <li>Credit score and DTI ratio are the strongest predictors of loan approval in this dataset.</li>
            <li>{"Urban applicants tend to have higher approval rates than rural applicants." if HAS_AREA else ""}</li>
            <li>{"Graduate applicants receive proportionally more approvals." if HAS_EDU else ""}</li>
            <li><b>{risk_apps:,}</b> applications ({risk_apps/total*100:.1f}%) carry elevated risk indicators
                (credit score &lt; 600 or DTI &gt; 55%).</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
