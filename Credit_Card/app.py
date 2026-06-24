import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# ── set_page_config MUST be the very first Streamlit call ────────────────────
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS + theme detection (no setInterval — avoids repeated re-renders) ───────
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
    // Run once on load and once after a short delay for Streamlit hydration
    applyTheme();
    setTimeout(applyTheme, 800);
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
})();
</script>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
base_path  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(base_path, "loan_clean_data.parquet")
MODEL_PKL  = os.path.join(base_path, "loan_model.pkl")
SCALER_PKL = os.path.join(base_path, "loan_scaler.pkl")

GREEN          = "#38a169"
RED            = "#e53e3e"
AMBER          = "#d69e2e"
BLUE_PALETTE   = ["#2b6cb0","#3182ce","#4299e1","#63b3ed","#90cdf4","#bee3f8"]
DEPENDENT_MONTHLY_COST = 5000

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


def get_hard_reject_reasons(credit_score, dti, disposable_after_emi,
                             employment, collateral, loan_amt):
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


# ── Mock fallbacks ────────────────────────────────────────────────────────────
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
def load_data(path):
    if not os.path.exists(path):
        # ── Synthetic 600-row demo dataset ───────────────────────────────────
        np.random.seed(42)
        n = 600
        purposes   = ['Home','Personal','Education','Business','Car']
        areas      = ['Urban','Semiurban','Rural']
        genders    = ['Male','Female']
        educations = ['Graduate','Not Graduate']
        emp_cats   = ['Corporate','Government','Self-Employed']
        emp_stat   = ['Employed','Self-Employed','Unemployed']
        marital    = ['Married','Single']

        d = pd.DataFrame({
            'Applicant_ID':       np.arange(1001, 1001+n),
            'Applicant_Income':   np.random.randint(20000, 200000, n),
            'Coapplicant_Income': np.random.randint(0, 80000, n),
            'Employment_Status':  np.random.choice(emp_stat, n, p=[0.70,0.25,0.05]),
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
            'Loan_Purpose':       np.random.choice(purposes, n, p=[0.35,0.20,0.20,0.15,0.10]),
            'Property_Area':      np.random.choice(areas, n),
            'Education_Level':    np.random.choice(educations, n, p=[0.65,0.35]),
            'Gender':             np.random.choice(genders, n, p=[0.65,0.35]),
            'Employer_Category':  np.random.choice(emp_cats, n),
        })
        score = (
            (d['Credit_Score'] > 650).astype(int) * 2 +
            (d['DTI_Ratio'] < 0.45).astype(int) +
            (d['Collateral_Value'] > d['Loan_Amount']).astype(int)
        )
        d['Loan_Approved'] = np.where(score >= 3, 'Approved', 'Rejected')
        return d

    data = pd.read_parquet(path)
    if 'Loan_Approved' in data.columns:
        data['Loan_Approved'] = data['Loan_Approved'].astype(str).str.strip().str.lower()
        data['Loan_Approved'] = data['Loan_Approved'].replace({
            '1':'Approved','1.0':'Approved','approved':'Approved',
            'yes':'Approved','y':'Approved','+':'Approved',
            '0':'Rejected','0.0':'Rejected','rejected':'Rejected',
            'no':'Rejected','n':'Rejected','-':'Rejected',
        })
    for col in data.select_dtypes('float64').columns:
        data[col] = data[col].astype('float32')
    for col in data.select_dtypes('int64').columns:
        data[col] = data[col].astype('int32')
    return data


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
        feats = list(inp.columns)
    for c in feats:
        if c not in inp.columns:
            inp[c] = 0.0
    inp    = inp[feats]
    scaled = _scaler.transform(inp)
    pred   = _model.predict(scaled)[0]
    proba  = _model.predict_proba(scaled)[0]
    return int(pred), float(proba[0]*100), float(proba[1]*100)


# ── Load everything ───────────────────────────────────────────────────────────
df_all          = load_data(DATA_PATH)
model, scaler, is_mock = load_assets()
FEATURES        = [c for c in df_all.select_dtypes(include='number').columns
                   if c not in ('Loan_Approved','Applicant_ID')]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💰 Loan Approval AI")
    st.markdown("---")

    if is_mock:
        st.warning("⚠️ ML model not found — Simulation Mode.")

    purpose_list = (sorted(df_all['Loan_Purpose'].dropna().unique())
                    if 'Loan_Purpose' in df_all.columns else ['Home','Personal','Education','Business'])
    purpose_f = st.multiselect("Loan Purpose", purpose_list, default=purpose_list)

    gender_list = (sorted(df_all['Gender'].dropna().unique())
                   if 'Gender' in df_all.columns else ['Male','Female'])
    gender_f = st.multiselect("Gender", gender_list, default=gender_list)

    age_f = st.slider("Age Range", 18, 80, (18, 80))

    st.markdown("---")
    st.info("Risk assessment calibrated to Income, Debt & Credit Score.")

# Apply sidebar filters
df = df_all.copy()
if 'Loan_Purpose' in df_all.columns:
    df = df[df['Loan_Purpose'].isin(purpose_f)]
if 'Gender' in df_all.columns:
    df = df[df['Gender'].isin(gender_f)]
if 'Age' in df_all.columns:
    df = df[df['Age'].between(age_f[0], age_f[1])]

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-app-header">
    <h1>💰 Smart Loan Approval Prediction System</h1>
    <p>Predict credit worthiness and analyze institutional loan workflows in real-time</p>
</div>
""", unsafe_allow_html=True)

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
            fig_pie = px.pie(df, names='Loan_Approved', hole=0.5, color='Loan_Approved',
                             color_discrete_map={'Approved': GREEN, 'Rejected': RED})
            fig_pie.update_layout(margin=dict(t=10,b=10), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-card"><div class="section-title">Loan Amount by Decision</div>', unsafe_allow_html=True)
            samp = df.sample(n=min(10000,len(df)), random_state=42) if len(df)>15000 else df
            fig_box = px.box(samp, x='Loan_Approved', y='Loan_Amount', color='Loan_Approved',
                             labels={'Loan_Amount':'Loan Amount (₹)'},
                             color_discrete_map={'Approved': GREEN, 'Rejected': RED})
            fig_box.update_layout(margin=dict(t=10,b=10), height=300, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
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
            existing_debt_i  = st.number_input("Existing Monthly Debt (₹)", 0, 500_000, 5_000, step=1_000)
            interest_rate_i  = st.number_input("Annual Interest Rate (%)", 1.0, 30.0, 9.0, step=0.5)
            savings_i        = st.number_input("Savings Balance (₹)", 0, 10_000_000, 200_000, step=25_000)
            collateral_i     = st.number_input("Collateral Value (₹)", 0, 50_000_000, 600_000, step=50_000)
        st.markdown('</div>', unsafe_allow_html=True)

        new_emi           = calculate_emi(loan_amt_i, interest_rate_i, loan_term_i)
        total_income      = income_i + co_income_i
        dti_calc          = (new_emi + existing_debt_i) / total_income if total_income > 0 else 1.0
        disposable        = total_income - (dependents_i * DEPENDENT_MONTHLY_COST) - existing_debt_i
        disposable_net    = disposable - new_emi

        st.markdown('<div class="section-card"><div class="section-title">Calculated Affordability</div>', unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Est. EMI",             f"₹{new_emi:,.0f}/mo")
        m2.metric("Total Monthly Income", f"₹{total_income:,.0f}")
        m3.metric("DTI Ratio",            f"{dti_calc*100:.1f}%")
        m4.metric("Disposable after EMI", f"₹{disposable_net:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("⚡ Run AI Loan Validation", type="primary", use_container_width=True)

    if submitted:
        s_st = "pass" if credit_score_i>=750 else ("warn" if credit_score_i>=650 else "fail")
        d_st = "pass" if dti_calc<=0.40 else ("warn" if dti_calc<=0.55 else "fail")
        i_st = "pass" if disposable_net>=new_emi*0.2 else ("warn" if disposable_net>=0 else "fail")
        dep_st= "pass" if dependents_i<=2 else ("warn" if dependents_i<=4 else "fail")
        col_st= "pass" if collateral_i>=loan_amt_i else "warn"
        emp_st= "fail" if employment_i=="Unemployed" else "pass"

        factors = [
            {"label":"Credit Score",  "value":str(credit_score_i),          "status":s_st,
             "reason":{"pass":"Excellent Credit","warn":"Fair Credit","fail":"Subprime — High Risk"}[s_st]},
            {"label":"DTI Ratio",     "value":f"{dti_calc*100:.1f}%",        "status":d_st,
             "reason":{"pass":"Healthy Debt Burden","warn":"Elevated Load","fail":"Over-leveraged"}[d_st]},
            {"label":"Affordability", "value":f"₹{disposable_net:,.0f}",    "status":i_st,
             "reason":{"pass":"Comfortable capacity","warn":"Tight capacity","fail":"Insufficient income"}[i_st]},
            {"label":"Dependents",    "value":str(dependents_i),             "status":dep_st,
             "reason":{"pass":"Manageable household","warn":"Higher burden","fail":"High dependent load"}[dep_st]},
            {"label":"Collateral",    "value":f"₹{collateral_i:,.0f}",       "status":col_st,
             "reason":"Fully Secured" if col_st=="pass" else "Under-collateralised"},
            {"label":"Employment",    "value":employment_i,                  "status":emp_st,
             "reason":"No verifiable income" if emp_st=="fail" else "Stable occupation"},
        ]

        with st.spinner("Running AI model…"):
            pred, p0, p1 = run_prediction(
                income_i, co_income_i, age_i, dependents_i, credit_score_i,
                existing_loans_i, dti_calc, savings_i, collateral_i,
                loan_amt_i, loan_term_i, model, scaler,
            )

        hard = get_hard_reject_reasons(credit_score_i, dti_calc, disposable_net,
                                       employment_i, collateral_i, loan_amt_i)
        approved  = (pred == 1) and (len(hard) == 0)
        conf_pct  = p1 if approved else p0
        if hard: conf_pct = max(conf_pct, 90.0)
        conf_pct  = min(99.5, max(50.0, conf_pct))

        if approved:
            st.balloons()
            bx,ic,ti,tc = "#f0fff4","✓","Loan Facility Approved","#38a169"
        else:
            bx,ic,ti,tc = "#fff5f5","✗","Loan Facility Rejected","#e53e3e"

        st.markdown(f"""
        <div style="background:{bx};border:2px solid {tc};border-radius:14px;
                    padding:1.25rem;display:flex;align-items:center;gap:16px;">
            <div style="font-size:30px;">{ic}</div>
            <div><b style="color:{tc};font-size:18px;">{ti}</b><br>
            <small>AI Confidence: {conf_pct:.1f}%</small></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cfg = {"pass":("#d1fae5","#065f46"),"warn":("#fef9c3","#713f12"),"fail":("#fee2e2","#7f1d1d")}
        for col, f in zip(st.columns(len(factors)), factors):
            bg, tx = cfg[f["status"]]
            col.markdown(
                f'<div style="background:{bg};padding:12px;border-radius:8px;">'
                f'<small style="color:{tx};font-weight:600;">{f["label"]}</small><br>'
                f'<b style="color:{tx};font-size:16px;">{f["value"]}</b></div>',
                unsafe_allow_html=True)

        if hard:
            st.error("🚫 Hard Policy Rejection:")
            for r in hard: st.write(f"- {r}")
        elif not approved:
            fails = [f for f in factors if f["status"]=="fail"]
            if fails:
                st.error("Policy Deficiencies:")
                for f in fails: st.write(f"- {f['reason']}")

        flags = []
        if total_income > 0 and savings_i > total_income*60:
            flags.append("Savings unusually high vs income — verify with bank statements")
        if credit_score_i >= 800 and dti_calc > 0.55:
            flags.append("High credit score + high DTI is unusual — recommend manual check")
        if flags:
            st.warning("⚠️ Flagged for manual review:")
            for r in flags: st.write(f"- {r}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHARTS & ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    # Guard: show message instead of using st.stop() inside a tab
    if df.empty or 'Loan_Approved' not in df.columns:
        st.warning("No data matches the current sidebar filters. Adjust the filters to see charts.")
    else:
        # ── Top KPIs ──────────────────────────────────────────────────────────
        approval_rate = (df['Loan_Approved']=='Approved').mean()*100
        avg_loan      = df['Loan_Amount'].mean() if 'Loan_Amount' in df.columns else 0
        avg_credit    = df['Credit_Score'].mean() if 'Credit_Score' in df.columns else 0
        avg_dti       = df['DTI_Ratio'].mean()*100 if 'DTI_Ratio' in df.columns else 0

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Approval Rate",    f"{approval_rate:.1f}%")
        k2.metric("Avg Loan Amount",  f"₹{avg_loan:,.0f}")
        k3.metric("Avg Credit Score", f"{avg_credit:.0f}")
        k4.metric("Avg DTI Ratio",    f"{avg_dti:.1f}%")
        st.markdown("<br>", unsafe_allow_html=True)

        # ── ROW 1 : Credit Score histogram | Approval by Loan Purpose ─────────
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            st.markdown('<div class="section-card"><div class="section-title">Credit Score Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(df, x='Credit_Score', color='Loan_Approved', nbins=30,
                               barmode='overlay', opacity=0.75,
                               color_discrete_map={'Approved':GREEN,'Rejected':RED},
                               labels={'Credit_Score':'Credit Score'})
            fig.add_vline(x=650, line_dash="dash", line_color=AMBER,
                          annotation_text="Threshold 650", annotation_position="top right")
            fig.update_layout(margin=dict(t=10,b=10), height=320, legend_title_text='Decision')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r1c2:
            st.markdown('<div class="section-card"><div class="section-title">Approval by Loan Purpose</div>', unsafe_allow_html=True)
            if 'Loan_Purpose' in df.columns:
                pg = df.groupby(['Loan_Purpose','Loan_Approved']).size().reset_index(name='Count')
                fig = px.bar(pg, x='Loan_Purpose', y='Count', color='Loan_Approved',
                             barmode='group', text_auto=True,
                             color_discrete_map={'Approved':GREEN,'Rejected':RED},
                             labels={'Loan_Purpose':'Purpose'})
                fig.update_layout(margin=dict(t=10,b=10), height=320, legend_title_text='Decision')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Loan_Purpose column not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── ROW 2 : Income vs Loan scatter | DTI violin ───────────────────────
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            st.markdown('<div class="section-card"><div class="section-title">Income vs Loan Amount</div>', unsafe_allow_html=True)
            samp = df.sample(n=min(2000,len(df)), random_state=1)
            fig = px.scatter(samp, x='Applicant_Income', y='Loan_Amount',
                             color='Loan_Approved', opacity=0.65,
                             color_discrete_map={'Approved':GREEN,'Rejected':RED},
                             labels={'Applicant_Income':'Monthly Income (₹)','Loan_Amount':'Loan Amount (₹)'})
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(margin=dict(t=10,b=10), height=320, legend_title_text='Decision')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r2c2:
            st.markdown('<div class="section-card"><div class="section-title">DTI Ratio Distribution</div>', unsafe_allow_html=True)
            fig = px.violin(df, x='Loan_Approved', y='DTI_Ratio', color='Loan_Approved',
                            box=True, points=False,
                            color_discrete_map={'Approved':GREEN,'Rejected':RED},
                            labels={'DTI_Ratio':'Debt-to-Income Ratio'})
            fig.add_hline(y=0.65, line_dash="dash", line_color=RED,
                          annotation_text="Hard cap 65%", annotation_position="top right")
            fig.add_hline(y=0.40, line_dash="dot", line_color=AMBER,
                          annotation_text="Healthy ceiling 40%", annotation_position="bottom right")
            fig.update_layout(margin=dict(t=10,b=10), height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── ROW 3 : Age band combo | Gender × Employment sunburst ────────────
        r3c1, r3c2 = st.columns(2)

        with r3c1:
            st.markdown('<div class="section-card"><div class="section-title">Approval Rate by Age Group</div>', unsafe_allow_html=True)
            if 'Age' in df.columns:
                adf = df.copy()
                adf['Age_Band'] = pd.cut(adf['Age'], bins=[17,25,30,35,40,50,60,80],
                                         labels=['18-25','26-30','31-35','36-40','41-50','51-60','61+'])
                ag = (adf.groupby('Age_Band', observed=True)
                         .apply(lambda x: pd.Series({
                             'Total': len(x),
                             'Rate': (x['Loan_Approved']=='Approved').mean()*100}))
                         .reset_index())
                fig = go.Figure()
                fig.add_trace(go.Bar(x=ag['Age_Band'].astype(str), y=ag['Total'],
                                     name='Applications', marker_color='#bee3f8', opacity=0.7))
                fig.add_trace(go.Scatter(x=ag['Age_Band'].astype(str), y=ag['Rate'],
                                         name='Approval Rate %', mode='lines+markers',
                                         line=dict(color=GREEN,width=2.5),
                                         marker=dict(size=8,color=GREEN), yaxis='y2'))
                fig.update_layout(margin=dict(t=10,b=10), height=320,
                                  yaxis=dict(title='Applications', showgrid=False),
                                  yaxis2=dict(title='Approval Rate (%)', overlaying='y',
                                              side='right', range=[0,110], showgrid=False),
                                  legend=dict(orientation='h',yanchor='bottom',y=1.01,xanchor='left',x=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age column not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with r3c2:
            st.markdown('<div class="section-card"><div class="section-title">Gender × Employment × Decision</div>', unsafe_allow_html=True)
            sun_cols = [c for c in ['Gender','Employment_Status','Loan_Approved'] if c in df.columns]
            if len(sun_cols) == 3:
                sd = df.groupby(sun_cols).size().reset_index(name='Count')
                fig = px.sunburst(sd, path=sun_cols, values='Count',
                                  color='Loan_Approved',
                                  color_discrete_map={'Approved':GREEN,'Rejected':RED,'(?)':'#a0aec0'})
                fig.update_layout(margin=dict(t=10,b=10), height=320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Gender / Employment_Status columns not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── ROW 4 : Approval by Loan Term | Savings vs Collateral bubble ─────
        r4c1, r4c2 = st.columns(2)

        with r4c1:
            st.markdown('<div class="section-card"><div class="section-title">Approval Rate by Loan Term</div>', unsafe_allow_html=True)
            if 'Loan_Term' in df.columns:
                tg = (df.groupby('Loan_Term')
                        .apply(lambda x: pd.Series({
                            'Count': len(x),
                            'ApprovalRate': (x['Loan_Approved']=='Approved').mean()*100}))
                        .reset_index().sort_values('Loan_Term'))
                fig = px.bar(tg, y=tg['Loan_Term'].astype(str), x='ApprovalRate',
                             orientation='h',
                             text=tg['ApprovalRate'].round(1).astype(str)+'%',
                             color='ApprovalRate',
                             color_continuous_scale=[[0,RED],[0.5,AMBER],[1,GREEN]],
                             labels={'ApprovalRate':'Approval Rate (%)','y':'Term (Months)'})
                fig.update_traces(textposition='outside')
                fig.update_layout(margin=dict(t=10,b=10), height=320,
                                  coloraxis_showscale=False, xaxis=dict(range=[0,115]))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Loan_Term column not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with r4c2:
            st.markdown('<div class="section-card"><div class="section-title">Savings vs Collateral (bubble = Loan Size)</div>', unsafe_allow_html=True)
            bcols = ['Savings','Collateral_Value','Loan_Amount','Loan_Approved']
            if all(c in df.columns for c in bcols):
                bd = df[bcols].dropna().sample(n=min(800,len(df)), random_state=7)
                fig = px.scatter(bd, x='Savings', y='Collateral_Value',
                                 size='Loan_Amount', color='Loan_Approved',
                                 color_discrete_map={'Approved':GREEN,'Rejected':RED},
                                 opacity=0.65, size_max=25,
                                 labels={'Savings':'Savings (₹)','Collateral_Value':'Collateral (₹)'})
                fig.update_layout(margin=dict(t=10,b=10), height=320, legend_title_text='Decision')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Required columns not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── ROW 5 : Correlation heatmap | Treemap ─────────────────────────────
        r5c1, r5c2 = st.columns([1.5, 1])

        with r5c1:
            st.markdown('<div class="section-card"><div class="section-title">Feature Correlation Matrix</div>', unsafe_allow_html=True)
            hcols = [c for c in REQUIRED_COLUMNS if c in df.columns]
            if len(hcols) >= 3:
                corr = df[hcols].corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale='RdBu', zmin=-1, zmax=1,
                    text=np.round(corr.values,2), texttemplate='%{text}',
                    textfont=dict(size=9),
                    hovertemplate='%{x} × %{y}: %{z:.2f}<extra></extra>'))
                fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=360)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numeric columns for a correlation matrix.")
            st.markdown('</div>', unsafe_allow_html=True)

        with r5c2:
            st.markdown('<div class="section-card"><div class="section-title">Volume by Area & Purpose</div>', unsafe_allow_html=True)
            tcols = [c for c in ['Property_Area','Loan_Purpose'] if c in df.columns]
            if len(tcols) == 2:
                td = df.groupby(tcols).size().reset_index(name='Count')
                fig = px.treemap(td, path=tcols, values='Count',
                                 color='Count', color_continuous_scale=BLUE_PALETTE)
                fig.update_layout(margin=dict(t=10,b=10), height=360, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            elif 'Property_Area' in df.columns:
                ad = df['Property_Area'].value_counts().reset_index()
                ad.columns = ['Area','Count']
                fig = px.bar(ad, x='Area', y='Count',
                             color='Count', color_continuous_scale=BLUE_PALETTE)
                fig.update_layout(margin=dict(t=10,b=10), height=360, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Property_Area or Loan_Purpose columns not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── ROW 6 : Dependents combo | Education stacked bar ─────────────────
        r6c1, r6c2 = st.columns(2)

        with r6c1:
            st.markdown('<div class="section-card"><div class="section-title">Approval Rate by Dependents</div>', unsafe_allow_html=True)
            if 'Dependents' in df.columns:
                dg = (df.groupby('Dependents')
                        .apply(lambda x: pd.Series({
                            'Total': len(x),
                            'Rate': (x['Loan_Approved']=='Approved').mean()*100}))
                        .reset_index())
                fig = go.Figure()
                fig.add_trace(go.Bar(x=dg['Dependents'].astype(str), y=dg['Total'],
                                     name='Applications', marker_color='#bee3f8', opacity=0.7))
                fig.add_trace(go.Scatter(x=dg['Dependents'].astype(str), y=dg['Rate'],
                                         name='Approval Rate %', mode='lines+markers',
                                         line=dict(color=GREEN,width=2.5),
                                         marker=dict(size=8,color=GREEN), yaxis='y2'))
                fig.update_layout(margin=dict(t=10,b=10), height=300,
                                  xaxis=dict(title='Dependents'),
                                  yaxis=dict(title='Applications', showgrid=False),
                                  yaxis2=dict(title='Approval Rate (%)', overlaying='y',
                                              side='right', range=[0,110], showgrid=False),
                                  legend=dict(orientation='h',yanchor='bottom',y=1.01,xanchor='left',x=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dependents column not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with r6c2:
            st.markdown('<div class="section-card"><div class="section-title">Approval by Education Level</div>', unsafe_allow_html=True)
            if 'Education_Level' in df.columns:
                eg = df.groupby(['Education_Level','Loan_Approved']).size().reset_index(name='Count')
                eg['Pct'] = eg['Count'] / eg.groupby('Education_Level')['Count'].transform('sum') * 100
                fig = px.bar(eg, x='Education_Level', y='Pct', color='Loan_Approved',
                             barmode='stack',
                             text=eg['Pct'].round(1).astype(str)+'%',
                             color_discrete_map={'Approved':GREEN,'Rejected':RED},
                             labels={'Pct':'% Share','Education_Level':'Education'})
                fig.update_traces(textposition='inside', textfont_size=11)
                fig.update_layout(margin=dict(t=10,b=10), height=300,
                                  yaxis=dict(title='% of Applications', range=[0,105]),
                                  legend_title_text='Decision')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Education_Level column not available.")
            st.markdown('</div>', unsafe_allow_html=True)
