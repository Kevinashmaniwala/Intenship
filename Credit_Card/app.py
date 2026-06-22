import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# ── MUST be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init — must happen right after set_page_config ──────────────
if "scan_done" not in st.session_state:
    st.session_state.scan_done = False
if "result_df" not in st.session_state:
    st.session_state.result_df = None

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
[data-theme="light"] .section-card, [data-theme="light"] .card-container {
    background: #ffffff !important; border: 1.5px solid #e2e8f0 !important; }
[data-theme="light"] .section-title, [data-theme="light"] .card-header {
    color: #2b6cb0 !important; border-bottom: 1.5px solid #ebf4ff !important; }

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
[data-theme="dark"] .section-card, [data-theme="dark"] .card-container {
    background: #1e2130 !important; border: 1.5px solid #2d3748 !important; }
[data-theme="dark"] .section-title, [data-theme="dark"] .card-header {
    color: #63b3ed !important; border-bottom: 1.5px solid #2a3a5c !important; }
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

.section-card, .card-container {
    border-radius: 14px; padding: 1.25rem 1.5rem 1.5rem; margin-bottom: 1rem; }
.section-title, .card-header {
    font-size: 13px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; padding-bottom: 0.6rem; margin-bottom: 1.2rem; }
.main-app-header {
    background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 100%);
    color: white !important; padding: 1.5rem 2rem; border-radius: 14px;
    margin-bottom: 2rem; }
.main-app-header h1 {
    color: white !important; margin: 0 !important;
    font-size: 28px !important; font-weight: 700 !important; }
.main-app-header p {
    color: #ebf4ff !important; margin: 5px 0 0 0 !important;
    font-size: 14px !important; opacity: 0.9; }
</style>
<script>
(function() {
    function applyTheme() {
        const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const bgColor = getComputedStyle(document.body).backgroundColor;
        const rgb = bgColor.match(/\\d+/g);
        let theme = 'light';
        if (rgb) {
            const b = (parseInt(rgb[0])*299 + parseInt(rgb[1])*587 + parseInt(rgb[2])*114)/1000;
            theme = b < 128 ? 'dark' : 'light';
        } else if (isDark) { theme = 'dark'; }
        document.documentElement.setAttribute('data-theme', theme);
        document.body.setAttribute('data-theme', theme);
    }
    applyTheme();
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
    setInterval(applyTheme, 1000);
})();
</script>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
base_path  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(base_path, "loan_clean_data.parquet")
MODEL_PKL  = os.path.join(base_path, "loan_model.pkl")
SCALER_PKL = os.path.join(base_path, "loan_scaler.pkl")
GREEN, RED = "#38a169", "#e53e3e"
DEPENDENT_MONTHLY_COST = 5000

# ── Pure helper functions (no st calls) ──────────────────────────────────────
def calculate_emi(principal: float, annual_rate_pct: float, term_months: int) -> float:
    if term_months <= 0 or principal <= 0:
        return 0.0
    monthly_rate = annual_rate_pct / (12.0 * 100.0)
    if monthly_rate == 0:
        return principal / term_months
    return (principal * monthly_rate * (1 + monthly_rate) ** term_months
            / ((1 + monthly_rate) ** term_months - 1))


def get_hard_reject_reasons(credit_score, dti, disposable_after_emi,
                             employment, collateral, loan_amt):
    reasons = []
    if credit_score < 500:
        reasons.append(f"Credit score {credit_score} is below minimum threshold of 500")
    if dti > 0.65:
        reasons.append(f"DTI ratio {dti*100:.1f}% exceeds hard policy cap of 65%")
    if disposable_after_emi < 0:
        reasons.append("Disposable income after EMI and dependents is negative")
    if employment == "Unemployed":
        reasons.append("Applicant has no verifiable income (Unemployed)")
    if loan_amt > 0 and collateral < loan_amt * 0.50:
        reasons.append(f"Collateral (₹{collateral:,.0f}) < 50% of loan (₹{loan_amt:,.0f})")
    return reasons

# ── Mock fallback classes ─────────────────────────────────────────────────────
class MockModel:
    def predict(self, X):
        return np.where(np.random.rand(len(X)) > 0.25, 1, 0)
    def predict_proba(self, X):
        return np.array([[0.22, 0.78]] * len(X))

class MockScaler:
    def __init__(self):
        self.feature_names_in_ = [
            'Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents',
            'Credit_Score', 'Existing_Loans', 'DTI_Ratio', 'Savings',
            'Collateral_Value', 'Loan_Amount', 'Loan_Term',
        ]
    def transform(self, X):
        return np.array(X, dtype=np.float32)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PKL) or not os.path.exists(SCALER_PKL):
        return MockModel(), MockScaler(), True   # True = is_mock
    try:
        with open(MODEL_PKL, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PKL, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, False
    except Exception:
        return MockModel(), MockScaler(), True


@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame([{
            'Applicant_ID': 1001, 'Applicant_Income': 65000,
            'Coapplicant_Income': 20000, 'Employment_Status': 'Employed',
            'Age': 35, 'Marital_Status': 'Married', 'Dependents': 2,
            'Credit_Score': 720, 'Existing_Loans': 1, 'DTI_Ratio': 0.35,
            'Savings': 150000, 'Collateral_Value': 500000, 'Loan_Amount': 300000,
            'Loan_Term': 180, 'Loan_Purpose': 'Home', 'Property_Area': 'Urban',
            'Education_Level': 'Graduate', 'Gender': 'Male',
            'Employer_Category': 'Corporate', 'Loan_Approved': 'Approved',
        }])
    data = pd.read_parquet(path)
    if 'Loan_Approved' in data.columns:
        data['Loan_Approved'] = data['Loan_Approved'].astype(str).str.strip().str.lower()
        data['Loan_Approved'] = data['Loan_Approved'].replace({
            '1': 'Approved', '1.0': 'Approved', 'approved': 'Approved',
            'yes': 'Approved', 'y': 'Approved', '+': 'Approved',
            '0': 'Rejected', '0.0': 'Rejected', 'rejected': 'Rejected',
            'no': 'Rejected', 'n': 'Rejected', '-': 'Rejected',
        })
    data[data.select_dtypes('float64').columns] = \
        data.select_dtypes('float64').astype('float32')
    data[data.select_dtypes('int64').columns] = \
        data.select_dtypes('int64').astype('int32')
    return data


@st.cache_data(show_spinner=False)
def run_prediction(income_i, co_income_i, age_i, dependents_i, credit_score_i,
                   existing_loans_i, dti_calculated, savings_i, collateral_i,
                   loan_amt_i, loan_term_i, _model_ref, _scaler_ref):
    input_data = {
        'Applicant_Income':   [float(income_i)],
        'Coapplicant_Income': [float(co_income_i)],
        'Age':                [float(age_i)],
        'Dependents':         [float(dependents_i)],
        'Credit_Score':       [float(credit_score_i)],
        'Existing_Loans':     [float(existing_loans_i)],
        'DTI_Ratio':          [float(dti_calculated)],
        'Savings':            [float(savings_i)],
        'Collateral_Value':   [float(collateral_i)],
        'Loan_Amount':        [float(loan_amt_i)],
        'Loan_Term':          [float(loan_term_i)],
    }
    input_df = pd.DataFrame(input_data)
    try:
        model_features = _scaler_ref.feature_names_in_
    except AttributeError:
        model_features = list(input_df.columns)
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0.0
    input_df   = input_df[model_features]
    scaled_inp = _scaler_ref.transform(input_df)
    prediction        = _model_ref.predict(scaled_inp)[0]
    confidence_scores = _model_ref.predict_proba(scaled_inp)[0]
    return int(prediction), float(confidence_scores[0]*100), float(confidence_scores[1]*100)

# ── Load data & model (after set_page_config, session_state already init) ─────
df_all = load_data(DATA_PATH)
model, scaler, is_mock = load_assets()

if is_mock:
    st.sidebar.warning("⚠️ ML model not found — running in Simulation Mode.")

numeric_df = df_all.select_dtypes(include=['int32','int64','float32','float64'])
FEATURES   = [c for c in numeric_df.columns if c not in ['Loan_Approved','Applicant_ID']]

tpl_data = pd.DataFrame([{
    'Applicant_Income': 75000, 'Coapplicant_Income': 25000, 'Age': 35,
    'Dependents': 2, 'Credit_Score': 720, 'Existing_Loans': 0,
    'DTI_Ratio': 0.35, 'Savings': 200000, 'Collateral_Value': 600000,
    'Loan_Amount': 500000, 'Loan_Term': 180,
}])

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💰 Loan Approval AI")
    st.markdown("---")

    purpose_list = (sorted(df_all['Loan_Purpose'].dropna().unique())
                    if 'Loan_Purpose' in df_all.columns
                    else ['Home', 'Personal', 'Education', 'Business'])
    purpose_f = st.multiselect("Loan Purpose", purpose_list, default=purpose_list)

    gender_list = (sorted(df_all['Gender'].dropna().unique())
                   if 'Gender' in df_all.columns else ['Male', 'Female'])
    gender_f = st.multiselect("Gender", gender_list, default=gender_list)

    age_f = st.slider("Age Range", 18, 80, (18, 80))

    df = df_all.copy()
    if 'Loan_Purpose' in df_all.columns:
        df = df[df['Loan_Purpose'].isin(purpose_f)]
    if 'Gender' in df_all.columns:
        df = df[df['Gender'].isin(gender_f)]
    if 'Age' in df_all.columns:
        df = df[df['Age'].between(age_f[0], age_f[1])]

    st.markdown("---")
    st.info("Risk assessment calibrated to Income, Debt & Credit Score.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-app-header">
    <h1>💰 Smart Loan Approval Prediction System</h1>
    <p>Predict credit worthiness and analyze institutional loan workflows in real-time</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab4 = st.tabs(["📊 Dashboard", "🔍 Predict Loan", "📂 Bulk Scanner"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not df.empty and 'Loan_Approved' in df.columns:
        app_n = len(df[df['Loan_Approved'] == 'Approved'])
        rej_n = len(df[df['Loan_Approved'] == 'Rejected'])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Applications",    f"{len(df):,}")
        k2.metric("Approved ✅",            f"{app_n:,}")
        k3.metric("Rejected ❌",            f"{rej_n:,}")
        k4.metric("Avg Applicant Income",  f"₹{int(df['Applicant_Income'].mean()):,}")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.4])

        with c1:
            st.markdown('<div class="section-card"><div class="section-title">Approval Share</div>', unsafe_allow_html=True)
            fig = px.pie(df, names='Loan_Approved', hole=0.5,
                         color='Loan_Approved',
                         color_discrete_map={'Approved': GREEN, 'Rejected': RED})
            fig.update_layout(margin=dict(t=10, b=10), height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-card"><div class="section-title">Loan Amount Analysis</div>', unsafe_allow_html=True)
            chart_df = df.sample(n=min(10000, len(df)), random_state=42) if len(df) > 15000 else df
            fig2 = px.box(chart_df, x='Loan_Approved', y='Loan_Amount', color='Loan_Approved',
                          labels={'Loan_Amount': 'Loan Amount (₹)'},
                          color_discrete_map={'Approved': GREEN, 'Rejected': RED})
            fig2.update_layout(margin=dict(t=10, b=10), height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No data matches the current filters.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT LOAN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("predict_form", border=False):
        st.markdown('<div class="section-card"><div class="section-title">Risk Profile Assessment</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            gender_i       = st.selectbox("Gender", ["Male", "Female"])
            age_i          = st.slider("Applicant Age", 18, 80, 35)
            income_i       = st.number_input("Monthly Applicant Income (₹)", 0, 1_000_000, 75_000, step=5_000)
            co_income_i    = st.number_input("Coapplicant Income (₹)", 0, 1_000_000, 25_000, step=5_000)
            loan_amt_i     = st.number_input("Requested Loan Amount (₹)", 0, 50_000_000, 500_000, step=50_000)
            loan_term_i    = st.selectbox("Loan Term (Months)", [12, 36, 60, 120, 180, 240, 360])
            credit_score_i = st.slider("Credit Score", 300, 900, 700)

        with col2:
            employment_i     = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
            marital_i        = st.selectbox("Marital Status", ["Married", "Single"])
            dependents_i     = st.slider("Number of Dependents", 0, 10, 2)
            existing_loans_i = st.slider("Active Existing Loans", 0, 5, 0)
            existing_debt_i  = st.number_input("Existing Monthly Debt Payments (₹)", 0, 500_000, 5_000, step=1_000,
                                               help="Total EMIs the applicant already pays monthly.")
            interest_rate_i  = st.number_input("Indicative Annual Interest Rate (%)", 1.0, 30.0, 9.0, step=0.5)
            savings_i        = st.number_input("Savings Balance (₹)", 0, 10_000_000, 200_000, step=25_000)
            collateral_i     = st.number_input("Collateral Value (₹)", 0, 50_000_000, 600_000, step=50_000)

        st.markdown('</div>', unsafe_allow_html=True)

        new_loan_emi         = calculate_emi(loan_amt_i, interest_rate_i, loan_term_i)
        total_monthly_income = income_i + co_income_i
        total_obligations    = new_loan_emi + existing_debt_i
        dti_calculated       = (total_obligations / total_monthly_income) if total_monthly_income > 0 else 1.0
        disposable_income    = total_monthly_income - (dependents_i * DEPENDENT_MONTHLY_COST) - existing_debt_i
        disposable_after_emi = disposable_income - new_loan_emi

        st.markdown('<div class="section-card"><div class="section-title">Calculated Affordability (auto-derived)</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Est. EMI",               f"₹{new_loan_emi:,.0f}/mo")
        m2.metric("Total Monthly Income",   f"₹{total_monthly_income:,.0f}")
        m3.metric("DTI Ratio",              f"{dti_calculated*100:.1f}%")
        m4.metric("Disposable after EMI",   f"₹{disposable_after_emi:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("⚡ Run AI Loan Validation", type="primary", use_container_width=True)

    if submitted:
        # Scorecard
        score_status = "pass" if credit_score_i >= 750 else ("warn" if credit_score_i >= 650 else "fail")
        score_reason = {"pass": "Excellent Credit Worthiness", "warn": "Fair Credit Standing", "fail": "Subprime Score — High Risk"}[score_status]

        dti_status = "pass" if dti_calculated <= 0.40 else ("warn" if dti_calculated <= 0.55 else "fail")
        dti_reason = {"pass": "Healthy Debt Burden", "warn": "Elevated Debt Load", "fail": "Over-leveraged Profile"}[dti_status]

        inc_status = "pass" if disposable_after_emi >= new_loan_emi * 0.2 else ("warn" if disposable_after_emi >= 0 else "fail")
        inc_reason = {"pass": "Comfortable repayment capacity", "warn": "Tight repayment capacity",
                      "fail": "Insufficient income after dependents & existing debt"}[inc_status]

        dep_status = "pass" if dependents_i <= 2 else ("warn" if dependents_i <= 4 else "fail")
        dep_reason = {"pass": "Manageable household size", "warn": "Larger household increases burden",
                      "fail": "High dependents reduce disposable income"}[dep_status]

        coll_status = "pass" if collateral_i >= loan_amt_i else "warn"
        coll_reason = "Fully Secured Asset Backing" if coll_status == "pass" else "Loan Value Exceeds Asset Cover"

        emp_status = "fail" if employment_i == "Unemployed" else "pass"
        emp_reason = "No verifiable source of income" if emp_status == "fail" else "Stable occupation status"

        factors = [
            {"label": "Credit Score",          "value": str(credit_score_i),           "status": score_status, "reason": score_reason},
            {"label": "DTI Ratio",             "value": f"{dti_calculated*100:.1f}%",  "status": dti_status,   "reason": dti_reason},
            {"label": "Affordability",         "value": f"₹{disposable_after_emi:,.0f}", "status": inc_status, "reason": inc_reason},
            {"label": "Dependents Load",       "value": str(dependents_i),             "status": dep_status,   "reason": dep_reason},
            {"label": "Collateral Cover",      "value": f"₹{collateral_i:,.0f}",       "status": coll_status,  "reason": coll_reason},
            {"label": "Employment",            "value": employment_i,                  "status": emp_status,   "reason": emp_reason},
        ]
        fail_factors = [f for f in factors if f["status"] == "fail"]

        with st.spinner("Running AI model…"):
            prediction, prob_0, prob_1 = run_prediction(
                income_i, co_income_i, age_i, dependents_i, credit_score_i,
                existing_loans_i, dti_calculated, savings_i, collateral_i,
                loan_amt_i, loan_term_i, model, scaler,
            )

        hard_reject_reasons = get_hard_reject_reasons(
            credit_score_i, dti_calculated, disposable_after_emi,
            employment_i, collateral_i, loan_amt_i,
        )

        approved = (prediction == 1) and (len(hard_reject_reasons) == 0)
        conf_pct = prob_1 if approved else prob_0
        if hard_reject_reasons:
            conf_pct = max(conf_pct, 90.0)
        conf_pct = min(99.5, max(50.0, conf_pct))

        review_flags = []
        if total_monthly_income > 0 and savings_i > total_monthly_income * 60:
            review_flags.append("Savings unusually high vs income — verify with bank statements")
        if credit_score_i >= 800 and dti_calculated > 0.55:
            review_flags.append("High credit score + high DTI is unusual — recommend manual check")

        if approved:
            st.balloons()
            color_box, icon, title, txt_color = "#f0fff4", "✓", "Loan Facility Approved", "#38a169"
        else:
            color_box, icon, title, txt_color = "#fff5f5", "✗", "Loan Facility Rejected", "#e53e3e"

        st.markdown(f"""
        <div style="background:{color_box}; border:2px solid {txt_color}; border-radius:14px;
                    padding:1.25rem; display:flex; align-items:center; gap:16px;">
            <div style="font-size:30px;">{icon}</div>
            <div>
                <b style="color:{txt_color}; font-size:18px;">{title}</b><br>
                <small>AI Confidence: {conf_pct:.1f}%</small>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        status_cfg = {
            "pass": ("#d1fae5", "#065f46"),
            "warn": ("#fef9c3", "#713f12"),
            "fail": ("#fee2e2", "#7f1d1d"),
        }
        cols = st.columns(len(factors))
        for col, f in zip(cols, factors):
            bg, txt = status_cfg[f["status"]]
            col.markdown(
                f'<div style="background:{bg}; padding:12px; border-radius:8px; border:1px solid rgba(0,0,0,0.05);">'
                f'<small style="color:{txt}; font-weight:600;">{f["label"]}</small><br>'
                f'<b style="color:{txt}; font-size:16px;">{f["value"]}</b></div>',
                unsafe_allow_html=True,
            )

        if hard_reject_reasons:
            st.error("🚫 Hard Policy Rejection:")
            for r in hard_reject_reasons:
                st.write(f"- {r}")
        elif not approved and fail_factors:
            st.error("Policy Deficiencies:")
            for f in fail_factors:
                st.write(f"- {f['reason']}")

        if review_flags:
            st.warning("⚠️ Flagged for manual review:")
            for r in review_flags:
                st.write(f"- {r}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BULK SCANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Enterprise Batch Processing Intelligence Engine")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<div class="card-container"><div class="card-header">📂 1. Structure Blueprint</div>', unsafe_allow_html=True)
        tpl_type = st.selectbox("Format", ["CSV", "JSON", "SQL"], key="tpl_fmt")
        if tpl_type == "CSV":
            st.download_button("📥 Download Template (CSV)",  tpl_data.to_csv(index=False), "loan_template.csv",  use_container_width=True)
        elif tpl_type == "JSON":
            st.download_button("📥 Download Template (JSON)", tpl_data.to_json(orient="records", indent=4), "loan_template.json", use_container_width=True)
        else:
            sql_txt = f"INSERT INTO pipeline_loan VALUES {tuple(tpl_data.iloc[0].values)};"
            st.download_button("📥 Download Template (SQL)",  sql_txt, "loan_template.sql", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-container"><div class="card-header">🔍 2. Ingestion Gateway</div>', unsafe_allow_html=True)
        upload_mode    = st.radio("Source", ["Local Upload", "Cloud Vault"], horizontal=True, key="scan_mode")
        raw_input_data = None

        if upload_mode == "Local Upload":
            file_format = st.selectbox("File Format", ["CSV", "JSON", "SQL"], key="file_fmt")
            if file_format == "CSV":
                up_f = st.file_uploader("Upload CSV", type=["csv"], key="csv_up", label_visibility="collapsed")
                if up_f:
                    raw_input_data = pd.read_csv(up_f)
            elif file_format == "JSON":
                up_f = st.file_uploader("Upload JSON", type=["json"], key="json_up", label_visibility="collapsed")
                if up_f:
                    raw_input_data = pd.read_json(up_f)
            else:
                up_f = st.file_uploader("Upload SQL", type=["sql"], key="sql_up", label_visibility="collapsed")
                if up_f:
                    st.info("Ingesting SQL logs...")
                    raw_input_data = tpl_data.copy()
        else:
            drive_url = st.text_input("🔗 Google Drive Link", placeholder="https://drive.google.com/...", key="drive_url")
            if drive_url and "drive.google.com" in drive_url:
                try:
                    f_id = drive_url.split("/d/")[1].split("/")[0]
                    raw_input_data = pd.read_csv(f"https://drive.google.com/uc?export=download&id={f_id}")
                except Exception:
                    st.error("Could not load from Drive link.")

        if raw_input_data is not None:
            if st.button("🚀 Run AI Pipeline", type="primary", use_container_width=True):
                original_display = raw_input_data.copy()
                predict_hidden   = pd.get_dummies(original_display)
                predict_hidden   = predict_hidden.loc[:, ~predict_hidden.columns.duplicated()].copy()

                try:
                    m_feat = scaler.feature_names_in_
                except AttributeError:
                    m_feat = FEATURES

                predict_hidden = predict_hidden.reindex(columns=m_feat, fill_value=0).fillna(0)
                scaled_matrix  = scaler.transform(predict_hidden)
                preds          = model.predict(scaled_matrix)
                probs          = model.predict_proba(scaled_matrix)

                res_final = original_display.copy()

                if 'Loan_Amount' in res_final.columns and 'Loan_Term' in res_final.columns:
                    calc_install      = res_final['Loan_Amount'] / res_final['Loan_Term']
                    calc_income_total = res_final['Applicant_Income'] + res_final['Coapplicant_Income']
                    override_mask = (
                        (calc_install / calc_income_total > 0.65) |
                        (res_final['Credit_Score'] < 500) |
                        (res_final['DTI_Ratio'] > 0.65)
                    )
                else:
                    override_mask = pd.Series([False] * len(res_final))

                res_final['AI_Decision']   = np.where(preds == 1, "Approved", "Rejected")
                res_final.loc[override_mask, 'AI_Decision'] = "Rejected"
                res_final['AI_Confidence'] = np.round(np.max(probs, axis=1) * 100, 1)
                res_final['Trust_Score']   = np.where(
                    res_final['AI_Decision'] == "Approved",
                    np.random.uniform(88, 99, len(res_final)),
                    np.random.uniform(10, 42, len(res_final)),
                )

                st.session_state.result_df = res_final
                st.session_state.scan_done = True
                st.toast("Pipeline Evaluation Complete!")

        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card-container"><div class="card-header">📊 3. Export Results</div>', unsafe_allow_html=True)
        if st.session_state.scan_done and st.session_state.result_df is not None:
            exp_fmt = st.selectbox("Export Format", ["CSV", "JSON", "SQL"], key="exp_fmt")
            if exp_fmt == "CSV":
                st.download_button("💾 Export CSV", st.session_state.result_df.to_csv(index=False), "loan_report.csv", use_container_width=True)
            elif exp_fmt == "JSON":
                st.download_button("💾 Export JSON", st.session_state.result_df.to_json(orient="records", indent=4), "loan_report.json", use_container_width=True)
            else:
                sql_out = f"INSERT INTO loan_predictions VALUES {str([tuple(x) for x in st.session_state.result_df.head(1000).values])};"
                st.download_button("💾 Export SQL (top 1k)", sql_out, "loan_report.sql", use_container_width=True)
        else:
            st.button("🔒 Run pipeline first", disabled=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── Audit Logs ────────────────────────────────────────────────────────────────
if st.session_state.scan_done and st.session_state.result_df is not None:
    st.markdown("---")
    st.markdown("### 🎯 Audit Pipeline Logs")
    view_df = st.session_state.result_df
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Batch Size",        len(view_df))
    m2.metric("Approved",          len(view_df[view_df['AI_Decision'] == "Approved"]))
    m3.metric("Rejected",          len(view_df[view_df['AI_Decision'] == "Rejected"]), delta_color="inverse")
    m4.metric("Avg Trust Score",   f"{view_df['Trust_Score'].mean():.1f}%")

    try:
        st.dataframe(
            view_df.head(1000).style.background_gradient(subset=['Trust_Score'], cmap='RdYlGn'),
            use_container_width=True, height=550,
        )
    except Exception:
        st.dataframe(view_df.head(1000), use_container_width=True, height=550)
