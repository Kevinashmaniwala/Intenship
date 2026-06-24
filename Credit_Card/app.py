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
    color: white !important; padding: 1.5rem 2rem; border-radius: 14px; margin-bottom: 2rem; }
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
            const b = (parseInt(rgb[0])*299+parseInt(rgb[1])*587+parseInt(rgb[2])*114)/1000;
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
GREEN, RED, AMBER = "#38a169", "#e53e3e", "#d69e2e"
BLUE_PALETTE = ["#2b6cb0", "#3182ce", "#4299e1", "#63b3ed", "#90cdf4", "#bee3f8"]
DEPENDENT_MONTHLY_COST = 5000

REQUIRED_COLUMNS = [
    'Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents',
    'Credit_Score', 'Existing_Loans', 'DTI_Ratio', 'Savings',
    'Collateral_Value', 'Loan_Amount', 'Loan_Term',
]

# ── Pure helper functions ─────────────────────────────────────────────────────
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
        p = np.random.uniform(0.55, 0.95, len(X))
        return np.column_stack([1 - p, p])


class MockScaler:
    def __init__(self):
        self.feature_names_in_ = REQUIRED_COLUMNS
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
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        # ── Synthetic fallback dataset (600 rows) for demo/dev ────────────────
        np.random.seed(42)
        n = 600
        purposes    = ['Home', 'Personal', 'Education', 'Business']
        areas       = ['Urban', 'Semiurban', 'Rural']
        genders     = ['Male', 'Female']
        educations  = ['Graduate', 'Not Graduate']
        emp_cats    = ['Corporate', 'Government', 'Self-Employed']
        emp_status  = ['Employed', 'Self-Employed', 'Unemployed']
        marital     = ['Married', 'Single']

        df_fake = pd.DataFrame({
            'Applicant_ID':       np.arange(1001, 1001 + n),
            'Applicant_Income':   np.random.randint(20000, 200000, n),
            'Coapplicant_Income': np.random.randint(0, 80000, n),
            'Employment_Status':  np.random.choice(emp_status, n, p=[0.70, 0.25, 0.05]),
            'Age':                np.random.randint(22, 65, n),
            'Marital_Status':     np.random.choice(marital, n),
            'Dependents':         np.random.choice([0,1,2,3,4], n, p=[0.25,0.30,0.25,0.15,0.05]),
            'Credit_Score':       np.random.randint(450, 900, n),
            'Existing_Loans':     np.random.choice([0,1,2,3], n, p=[0.50,0.30,0.15,0.05]),
            'DTI_Ratio':          np.round(np.random.uniform(0.10, 0.70, n), 2),
            'Savings':            np.random.randint(10000, 1000000, n),
            'Collateral_Value':   np.random.randint(100000, 5000000, n),
            'Loan_Amount':        np.random.randint(50000, 3000000, n),
            'Loan_Term':          np.random.choice([12,36,60,120,180,240,360], n),
            'Loan_Purpose':       np.random.choice(purposes, n, p=[0.40,0.25,0.20,0.15]),
            'Property_Area':      np.random.choice(areas, n),
            'Education_Level':    np.random.choice(educations, n, p=[0.65,0.35]),
            'Gender':             np.random.choice(genders, n, p=[0.65,0.35]),
            'Employer_Category':  np.random.choice(emp_cats, n),
        })
        # Derive approval label heuristically
        score = (
            (df_fake['Credit_Score'] > 650).astype(int) * 2 +
            (df_fake['DTI_Ratio'] < 0.45).astype(int) +
            (df_fake['Collateral_Value'] > df_fake['Loan_Amount']).astype(int)
        )
        df_fake['Loan_Approved'] = np.where(score >= 3, 'Approved', 'Rejected')
        return df_fake

    data = pd.read_parquet(path)
    if 'Loan_Approved' in data.columns:
        data['Loan_Approved'] = data['Loan_Approved'].astype(str).str.strip().str.lower()
        data['Loan_Approved'] = data['Loan_Approved'].replace({
            '1': 'Approved', '1.0': 'Approved', 'approved': 'Approved',
            'yes': 'Approved', 'y': 'Approved', '+': 'Approved',
            '0': 'Rejected', '0.0': 'Rejected', 'rejected': 'Rejected',
            'no': 'Rejected', 'n': 'Rejected', '-': 'Rejected',
        })
    data[data.select_dtypes('float64').columns] = data.select_dtypes('float64').astype('float32')
    data[data.select_dtypes('int64').columns]   = data.select_dtypes('int64').astype('int32')
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


# ── Load data & model ─────────────────────────────────────────────────────────
df_all = load_data(DATA_PATH)
model, scaler, is_mock = load_assets()

# if is_mock:
#     st.sidebar.warning("⚠️ ML model not found — running in Simulation Mode.")

numeric_df = df_all.select_dtypes(include=['int32','int64','float32','float64'])
FEATURES   = [c for c in numeric_df.columns if c not in ['Loan_Approved','Applicant_ID']]

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💰 Loan Approval AI")
    st.markdown("---")
    purpose_list = (sorted(df_all['Loan_Purpose'].dropna().unique())
                    if 'Loan_Purpose' in df_all.columns else ['Home','Personal','Education','Business'])
    purpose_f = st.multiselect("Loan Purpose", purpose_list, default=purpose_list)
    gender_list = (sorted(df_all['Gender'].dropna().unique())
                   if 'Gender' in df_all.columns else ['Male','Female'])
    gender_f = st.multiselect("Gender", gender_list, default=gender_list)
    age_f = st.slider("Age Range", 18, 80, (18, 80))

    df = df_all.copy()
    if 'Loan_Purpose' in df_all.columns: df = df[df['Loan_Purpose'].isin(purpose_f)]
    if 'Gender'       in df_all.columns: df = df[df['Gender'].isin(gender_f)]
    if 'Age'          in df_all.columns: df = df[df['Age'].between(age_f[0], age_f[1])]

    st.markdown("---")
    st.info("Risk assessment calibrated to Income, Debt & Credit Score.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-app-header">
    <h1>💰 Smart Loan Approval Prediction System</h1>
    <p>Predict credit worthiness and analyze institutional loan workflows in real-time</p>
</div>
""", unsafe_allow_html=True)

# ── Three main tabs — Bulk Scanner replaced with Charts & Analytics ───────────
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Predict Loan", "📈 Charts & Analytics"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not df.empty and 'Loan_Approved' in df.columns:
        app_n = len(df[df['Loan_Approved'] == 'Approved'])
        rej_n = len(df[df['Loan_Approved'] == 'Rejected'])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Applications",   f"{len(df):,}")
        k2.metric("Approved ✅",           f"{app_n:,}")
        k3.metric("Rejected ❌",           f"{rej_n:,}")
        k4.metric("Avg Applicant Income", f"₹{int(df['Applicant_Income'].mean()):,}")
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown('<div class="section-card"><div class="section-title">Approval Share</div>', unsafe_allow_html=True)
            fig = px.pie(df, names='Loan_Approved', hole=0.5, color='Loan_Approved',
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
            existing_debt_i  = st.number_input("Existing Monthly Debt (₹)", 0, 500_000, 5_000, step=1_000,
                                               help="Total EMIs the applicant already pays monthly.")
            interest_rate_i  = st.number_input("Annual Interest Rate (%)", 1.0, 30.0, 9.0, step=0.5)
            savings_i        = st.number_input("Savings Balance (₹)", 0, 10_000_000, 200_000, step=25_000)
            collateral_i     = st.number_input("Collateral Value (₹)", 0, 50_000_000, 600_000, step=50_000)
        st.markdown('</div>', unsafe_allow_html=True)

        new_loan_emi         = calculate_emi(loan_amt_i, interest_rate_i, loan_term_i)
        total_monthly_income = income_i + co_income_i
        total_obligations    = new_loan_emi + existing_debt_i
        dti_calculated       = (total_obligations / total_monthly_income) if total_monthly_income > 0 else 1.0
        disposable_income    = total_monthly_income - (dependents_i * DEPENDENT_MONTHLY_COST) - existing_debt_i
        disposable_after_emi = disposable_income - new_loan_emi

        st.markdown('<div class="section-card"><div class="section-title">Calculated Affordability</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Est. EMI",             f"₹{new_loan_emi:,.0f}/mo")
        m2.metric("Total Monthly Income", f"₹{total_monthly_income:,.0f}")
        m3.metric("DTI Ratio",            f"{dti_calculated*100:.1f}%")
        m4.metric("Disposable after EMI", f"₹{disposable_after_emi:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("⚡ Run AI Loan Validation", type="primary", use_container_width=True)

    if submitted:
        score_status = "pass" if credit_score_i >= 750 else ("warn" if credit_score_i >= 650 else "fail")
        score_reason = {"pass":"Excellent Credit Worthiness","warn":"Fair Credit Standing","fail":"Subprime Score — High Risk"}[score_status]
        dti_status   = "pass" if dti_calculated <= 0.40 else ("warn" if dti_calculated <= 0.55 else "fail")
        dti_reason   = {"pass":"Healthy Debt Burden","warn":"Elevated Debt Load","fail":"Over-leveraged Profile"}[dti_status]
        inc_status   = "pass" if disposable_after_emi >= new_loan_emi*0.2 else ("warn" if disposable_after_emi >= 0 else "fail")
        inc_reason   = {"pass":"Comfortable repayment capacity","warn":"Tight repayment capacity","fail":"Insufficient income after dependents & debt"}[inc_status]
        dep_status   = "pass" if dependents_i <= 2 else ("warn" if dependents_i <= 4 else "fail")
        dep_reason   = {"pass":"Manageable household size","warn":"Larger household increases burden","fail":"High dependents reduce disposable income"}[dep_status]
        coll_status  = "pass" if collateral_i >= loan_amt_i else "warn"
        coll_reason  = "Fully Secured Asset Backing" if coll_status == "pass" else "Loan Value Exceeds Asset Cover"
        emp_status   = "fail" if employment_i == "Unemployed" else "pass"
        emp_reason   = "No verifiable source of income" if emp_status == "fail" else "Stable occupation status"

        factors = [
            {"label":"Credit Score",    "value":str(credit_score_i),             "status":score_status, "reason":score_reason},
            {"label":"DTI Ratio",       "value":f"{dti_calculated*100:.1f}%",    "status":dti_status,   "reason":dti_reason},
            {"label":"Affordability",   "value":f"₹{disposable_after_emi:,.0f}", "status":inc_status,   "reason":inc_reason},
            {"label":"Dependents",      "value":str(dependents_i),               "status":dep_status,   "reason":dep_reason},
            {"label":"Collateral",      "value":f"₹{collateral_i:,.0f}",         "status":coll_status,  "reason":coll_reason},
            {"label":"Employment",      "value":employment_i,                    "status":emp_status,   "reason":emp_reason},
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
        if hard_reject_reasons: conf_pct = max(conf_pct, 90.0)
        conf_pct = min(99.5, max(50.0, conf_pct))

        review_flags = []
        if total_monthly_income > 0 and savings_i > total_monthly_income * 60:
            review_flags.append("Savings unusually high vs income — verify with bank statements")
        if credit_score_i >= 800 and dti_calculated > 0.55:
            review_flags.append("High credit score + high DTI is unusual — recommend manual check")

        if approved:
            st.balloons()
            color_box, icon, title, txt_color = "#f0fff4","✓","Loan Facility Approved","#38a169"
        else:
            color_box, icon, title, txt_color = "#fff5f5","✗","Loan Facility Rejected","#e53e3e"

        st.markdown(f"""
        <div style="background:{color_box};border:2px solid {txt_color};border-radius:14px;
                    padding:1.25rem;display:flex;align-items:center;gap:16px;">
            <div style="font-size:30px;">{icon}</div>
            <div><b style="color:{txt_color};font-size:18px;">{title}</b><br>
            <small>AI Confidence: {conf_pct:.1f}%</small></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        status_cfg = {"pass":("#d1fae5","#065f46"),"warn":("#fef9c3","#713f12"),"fail":("#fee2e2","#7f1d1d")}
        for col, f in zip(st.columns(len(factors)), factors):
            bg, txt = status_cfg[f["status"]]
            col.markdown(
                f'<div style="background:{bg};padding:12px;border-radius:8px;border:1px solid rgba(0,0,0,0.05);">'
                f'<small style="color:{txt};font-weight:600;">{f["label"]}</small><br>'
                f'<b style="color:{txt};font-size:16px;">{f["value"]}</b></div>',
                unsafe_allow_html=True)

        if hard_reject_reasons:
            st.error("🚫 Hard Policy Rejection:")
            for r in hard_reject_reasons: st.write(f"- {r}")
        elif not approved and fail_factors:
            st.error("Policy Deficiencies:")
            for f in fail_factors: st.write(f"- {f['reason']}")
        if review_flags:
            st.warning("⚠️ Flagged for manual review:")
            for r in review_flags: st.write(f"- {r}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHARTS & ANALYTICS  (replaces the old Bulk Scanner tab)
# NOTE: All bulk-scanner logic (file upload, pipeline runner, session state for
#       uploaded_df / scan_done / result_df, parse_uploaded_file, template
#       download, export, audit-log section) has been removed. This tab now
#       provides a multi-chart analytical view over the dataset, respecting the
#       sidebar filters already applied to `df`.
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty or 'Loan_Approved' not in df.columns:
        st.warning("No data matches the current sidebar filters. Adjust the filters to see charts.")
        st.stop()

    # ── Top KPI strip ─────────────────────────────────────────────────────────
    approved_df = df[df['Loan_Approved'] == 'Approved']
    rejected_df = df[df['Loan_Approved'] == 'Rejected']
    approval_rate = len(approved_df) / len(df) * 100 if len(df) > 0 else 0
    avg_loan = df['Loan_Amount'].mean() if 'Loan_Amount' in df.columns else 0
    avg_credit = df['Credit_Score'].mean() if 'Credit_Score' in df.columns else 0
    avg_dti = df['DTI_Ratio'].mean() * 100 if 'DTI_Ratio' in df.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Approval Rate",       f"{approval_rate:.1f}%")
    k2.metric("Avg Loan Amount",     f"₹{avg_loan:,.0f}")
    k3.metric("Avg Credit Score",    f"{avg_credit:.0f}")
    k4.metric("Avg DTI Ratio",       f"{avg_dti:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1 — Credit Score Distribution  |  Loan Purpose Breakdown
    # ══════════════════════════════════════════════════════════════════════════
    r1c1, r1c2 = st.columns(2)

    # ── Chart 1 : Credit Score Distribution by Approval Status ───────────────
    with r1c1:
        st.markdown('<div class="section-card"><div class="section-title">Credit Score Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            df, x='Credit_Score', color='Loan_Approved', nbins=30,
            barmode='overlay', opacity=0.75,
            color_discrete_map={'Approved': GREEN, 'Rejected': RED},
            labels={'Credit_Score': 'Credit Score', 'count': 'Applications'},
        )
        fig_hist.add_vline(x=650, line_dash="dash", line_color=AMBER,
                           annotation_text="Min threshold (650)",
                           annotation_position="top right")
        fig_hist.update_layout(margin=dict(t=10, b=10), height=320,
                               legend_title_text='Decision')
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Chart 2 : Loan Purpose Breakdown (stacked bar) ────────────────────────
    with r1c2:
        st.markdown('<div class="section-card"><div class="section-title">Approval by Loan Purpose</div>', unsafe_allow_html=True)
        if 'Loan_Purpose' in df.columns:
            purpose_grp = (df.groupby(['Loan_Purpose', 'Loan_Approved'])
                             .size().reset_index(name='Count'))
            fig_bar = px.bar(
                purpose_grp, x='Loan_Purpose', y='Count', color='Loan_Approved',
                barmode='group', text_auto=True,
                color_discrete_map={'Approved': GREEN, 'Rejected': RED},
                labels={'Loan_Purpose': 'Purpose', 'Count': 'Applications'},
            )
            fig_bar.update_layout(margin=dict(t=10, b=10), height=320,
                                  legend_title_text='Decision')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Loan_Purpose column not available in this dataset.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2 — Income vs Loan Amount Scatter  |  DTI Ratio Distribution
    # ══════════════════════════════════════════════════════════════════════════
    r2c1, r2c2 = st.columns(2)

    # ── Chart 3 : Income vs Loan Amount scatter coloured by decision ──────────
    with r2c1:
        st.markdown('<div class="section-card"><div class="section-title">Income vs Loan Amount</div>', unsafe_allow_html=True)
        scatter_df = df.sample(n=min(2000, len(df)), random_state=1)
        fig_scatter = px.scatter(
            scatter_df, x='Applicant_Income', y='Loan_Amount',
            color='Loan_Approved', opacity=0.65, size_max=6,
            color_discrete_map={'Approved': GREEN, 'Rejected': RED},
            labels={'Applicant_Income': 'Monthly Income (₹)', 'Loan_Amount': 'Loan Amount (₹)'},
            hover_data=['Credit_Score'] if 'Credit_Score' in scatter_df.columns else None,
        )
        fig_scatter.update_traces(marker=dict(size=5))
        fig_scatter.update_layout(margin=dict(t=10, b=10), height=320,
                                  legend_title_text='Decision')
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Chart 4 : DTI Ratio violin by Approval ────────────────────────────────
    with r2c2:
        st.markdown('<div class="section-card"><div class="section-title">DTI Ratio Distribution</div>', unsafe_allow_html=True)
        fig_violin = px.violin(
            df, x='Loan_Approved', y='DTI_Ratio', color='Loan_Approved',
            box=True, points=False,
            color_discrete_map={'Approved': GREEN, 'Rejected': RED},
            labels={'DTI_Ratio': 'Debt-to-Income Ratio', 'Loan_Approved': 'Decision'},
        )
        fig_violin.add_hline(y=0.65, line_dash="dash", line_color=RED,
                             annotation_text="Hard cap (65%)",
                             annotation_position="top right")
        fig_violin.add_hline(y=0.40, line_dash="dot", line_color=AMBER,
                             annotation_text="Healthy ceiling (40%)",
                             annotation_position="bottom right")
        fig_violin.update_layout(margin=dict(t=10, b=10), height=320,
                                 showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3 — Age Band Approval Rates  |  Gender & Marital Status Mix
    # ══════════════════════════════════════════════════════════════════════════
    r3c1, r3c2 = st.columns(2)

    # ── Chart 5 : Approval rate by age band (line + bar combo) ───────────────
    with r3c1:
        st.markdown('<div class="section-card"><div class="section-title">Approval Rate by Age Group</div>', unsafe_allow_html=True)
        if 'Age' in df.columns:
            age_df = df.copy()
            age_df['Age_Band'] = pd.cut(
                age_df['Age'],
                bins=[17, 25, 30, 35, 40, 50, 60, 80],
                labels=['18-25','26-30','31-35','36-40','41-50','51-60','61+'],
            )
            age_grp = (age_df.groupby('Age_Band', observed=True)
                             .apply(lambda x: pd.Series({
                                 'Total': len(x),
                                 'Approved': (x['Loan_Approved'] == 'Approved').sum(),
                                 'Rate': (x['Loan_Approved'] == 'Approved').mean() * 100,
                             }))
                             .reset_index())
            fig_age = go.Figure()
            fig_age.add_trace(go.Bar(
                x=age_grp['Age_Band'].astype(str), y=age_grp['Total'],
                name='Total Applications', marker_color='#bee3f8', opacity=0.7,
                yaxis='y',
            ))
            fig_age.add_trace(go.Scatter(
                x=age_grp['Age_Band'].astype(str), y=age_grp['Rate'],
                name='Approval Rate %', mode='lines+markers',
                line=dict(color=GREEN, width=2.5),
                marker=dict(size=8, color=GREEN),
                yaxis='y2',
            ))
            fig_age.update_layout(
                margin=dict(t=10, b=10), height=320,
                yaxis=dict(title='Applications', showgrid=False),
                yaxis2=dict(title='Approval Rate (%)', overlaying='y',
                            side='right', range=[0, 110], showgrid=False),
                legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
            )
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("Age column not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Chart 6 : Sunburst — Gender → Employment → Decision ──────────────────
    with r3c2:
        st.markdown('<div class="section-card"><div class="section-title">Gender × Employment × Decision</div>', unsafe_allow_html=True)
        sun_cols = [c for c in ['Gender', 'Employment_Status', 'Loan_Approved'] if c in df.columns]
        if len(sun_cols) == 3:
            sun_df = (df.groupby(sun_cols).size().reset_index(name='Count'))
            fig_sun = px.sunburst(
                sun_df, path=sun_cols, values='Count',
                color='Loan_Approved',
                color_discrete_map={'Approved': GREEN, 'Rejected': RED, '(?)': '#a0aec0'},
            )
            fig_sun.update_layout(margin=dict(t=10, b=10), height=320)
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.info("Gender / Employment_Status columns not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 4 — Loan Term Mix  |  Savings vs Collateral bubble
    # ══════════════════════════════════════════════════════════════════════════
    r4c1, r4c2 = st.columns(2)

    # ── Chart 7 : Approval rate by Loan Term (horizontal bar) ────────────────
    with r4c1:
        st.markdown('<div class="section-card"><div class="section-title">Approval Rate by Loan Term</div>', unsafe_allow_html=True)
        if 'Loan_Term' in df.columns:
            term_grp = (df.groupby('Loan_Term')
                          .apply(lambda x: pd.Series({
                              'Count': len(x),
                              'ApprovalRate': (x['Loan_Approved'] == 'Approved').mean() * 100,
                          }))
                          .reset_index()
                          .sort_values('Loan_Term'))
            fig_term = px.bar(
                term_grp, y=term_grp['Loan_Term'].astype(str),
                x='ApprovalRate', orientation='h',
                text=term_grp['ApprovalRate'].round(1).astype(str) + '%',
                color='ApprovalRate',
                color_continuous_scale=[[0, RED], [0.5, AMBER], [1, GREEN]],
                labels={'ApprovalRate': 'Approval Rate (%)',
                        'y': 'Term (Months)'},
            )
            fig_term.update_traces(textposition='outside')
            fig_term.update_layout(margin=dict(t=10, b=10), height=320,
                                   coloraxis_showscale=False,
                                   xaxis=dict(range=[0, 115]))
            st.plotly_chart(fig_term, use_container_width=True)
        else:
            st.info("Loan_Term column not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Chart 8 : Savings vs Collateral bubble sized by Loan Amount ──────────
    with r4c2:
        st.markdown('<div class="section-card"><div class="section-title">Savings vs Collateral (bubble = Loan)</div>', unsafe_allow_html=True)
        bub_cols = ['Savings', 'Collateral_Value', 'Loan_Amount', 'Loan_Approved']
        if all(c in df.columns for c in bub_cols):
            bub_df = df[bub_cols].dropna().sample(n=min(800, len(df)), random_state=7)
            fig_bub = px.scatter(
                bub_df, x='Savings', y='Collateral_Value',
                size='Loan_Amount', color='Loan_Approved',
                color_discrete_map={'Approved': GREEN, 'Rejected': RED},
                opacity=0.65, size_max=25,
                labels={
                    'Savings': 'Savings Balance (₹)',
                    'Collateral_Value': 'Collateral Value (₹)',
                    'Loan_Amount': 'Loan Amount (₹)',
                },
            )
            fig_bub.update_layout(margin=dict(t=10, b=10), height=320,
                                  legend_title_text='Decision')
            st.plotly_chart(fig_bub, use_container_width=True)
        else:
            st.info("Required columns not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 5 — Correlation Heatmap  |  Property Area treemap
    # ══════════════════════════════════════════════════════════════════════════
    r5c1, r5c2 = st.columns([1.5, 1])

    # ── Chart 9 : Correlation heatmap of numeric features ────────────────────
    with r5c1:
        st.markdown('<div class="section-card"><div class="section-title">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        heat_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
        if len(heat_cols) >= 3:
            corr = df[heat_cols].corr()
            fig_heat = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale='RdBu',
                zmin=-1, zmax=1,
                text=np.round(corr.values, 2),
                texttemplate='%{text}',
                textfont=dict(size=9),
                hovertemplate='%{x} × %{y}: %{z:.2f}<extra></extra>',
            ))
            fig_heat.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=360)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Not enough numeric columns for a correlation matrix.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Chart 10 : Treemap — Property Area × Loan Purpose volume ─────────────
    with r5c2:
        st.markdown('<div class="section-card"><div class="section-title">Volume by Area & Purpose</div>', unsafe_allow_html=True)
        tree_cols = [c for c in ['Property_Area', 'Loan_Purpose'] if c in df.columns]
        if len(tree_cols) == 2:
            tree_df = (df.groupby(tree_cols).size().reset_index(name='Count'))
            fig_tree = px.treemap(
                tree_df, path=tree_cols, values='Count',
                color='Count', color_continuous_scale=BLUE_PALETTE,
            )
            fig_tree.update_layout(margin=dict(t=10, b=10), height=360,
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_tree, use_container_width=True)
        elif 'Property_Area' in df.columns:
            area_df = df['Property_Area'].value_counts().reset_index()
            area_df.columns = ['Area', 'Count']
            fig_area = px.bar(area_df, x='Area', y='Count',
                              color='Count', color_continuous_scale=BLUE_PALETTE,
                              labels={'Area': 'Property Area'})
            fig_area.update_layout(margin=dict(t=10, b=10), height=360,
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("Property_Area or Loan_Purpose columns not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 6 — Dependents impact  |  Education level approval
    # ══════════════════════════════════════════════════════════════════════════
    r6c1, r6c2 = st.columns(2)

    # ── Chart 11 : Approval rate by number of dependents ─────────────────────
    with r6c1:
        st.markdown('<div class="section-card"><div class="section-title">Approval Rate by Dependents</div>', unsafe_allow_html=True)
        if 'Dependents' in df.columns:
            dep_grp = (df.groupby('Dependents')
                         .apply(lambda x: pd.Series({
                             'Total': len(x),
                             'ApprovalRate': (x['Loan_Approved'] == 'Approved').mean() * 100,
                         }))
                         .reset_index())
            fig_dep = go.Figure()
            fig_dep.add_trace(go.Bar(
                x=dep_grp['Dependents'].astype(str), y=dep_grp['Total'],
                name='Total', marker_color='#bee3f8', opacity=0.7,
            ))
            fig_dep.add_trace(go.Scatter(
                x=dep_grp['Dependents'].astype(str), y=dep_grp['ApprovalRate'],
                name='Approval Rate %', mode='lines+markers',
                line=dict(color=GREEN, width=2.5),
                marker=dict(size=8, color=GREEN),
                yaxis='y2',
            ))
            fig_dep.update_layout(
                margin=dict(t=10, b=10), height=300,
                xaxis=dict(title='Number of Dependents'),
                yaxis=dict(title='Total Applications', showgrid=False),
                yaxis2=dict(title='Approval Rate (%)', overlaying='y',
                            side='right', range=[0, 110], showgrid=False),
                legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
            )
            st.plotly_chart(fig_dep, use_container_width=True)
        else:
            st.info("Dependents column not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Chart 12 : Education level → approval stacked 100% bar ───────────────
    with r6c2:
        st.markdown('<div class="section-card"><div class="section-title">Approval by Education Level</div>', unsafe_allow_html=True)
        if 'Education_Level' in df.columns:
            edu_grp = (df.groupby(['Education_Level', 'Loan_Approved'])
                         .size().reset_index(name='Count'))
            edu_tot = edu_grp.groupby('Education_Level')['Count'].transform('sum')
            edu_grp['Pct'] = edu_grp['Count'] / edu_tot * 100
            fig_edu = px.bar(
                edu_grp, x='Education_Level', y='Pct', color='Loan_Approved',
                barmode='stack', text=edu_grp['Pct'].round(1).astype(str) + '%',
                color_discrete_map={'Approved': GREEN, 'Rejected': RED},
                labels={'Pct': '% Share', 'Education_Level': 'Education'},
            )
            fig_edu.update_traces(textposition='inside', textfont_size=11)
            fig_edu.update_layout(margin=dict(t=10, b=10), height=300,
                                  yaxis=dict(title='% of Applications', range=[0, 105]),
                                  legend_title_text='Decision')
            st.plotly_chart(fig_edu, use_container_width=True)
        else:
            st.info("Education_Level column not available.")
        st.markdown('</div>', unsafe_allow_html=True)
