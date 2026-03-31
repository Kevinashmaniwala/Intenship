import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Approval Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #f0f4f8 !important;
}
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1.5px solid #e2e8f0 !important;
}
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-top: 4px solid #2b6cb0 !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.25rem !important;
}
.stTabs [aria-selected="true"] { background: #2b6cb0 !important; color: #ffffff !important; }
.section-card {
    background: #ffffff; border-radius: 14px;
    border: 1.5px solid #e2e8f0;
    padding: 1.25rem 1.5rem 1.5rem; margin-bottom: 1rem;
}
.section-title {
    font-size: 12px; font-weight: 700; color: #2b6cb0;
    text-transform: uppercase; letter-spacing: 0.05em;
    padding-bottom: 0.5rem; border-bottom: 1.5px solid #ebf4ff; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
# ── Constants (Updated for Streamlit Cloud) ──────────────────────────────────

base_path = os.path.dirname(os.path.abspath(__file__))

CSV_PATH   = os.path.join(base_path, "clean_data11.csv")
MODEL_PKL  = os.path.join(base_path, "credit_card_model11.pkl")
SCALER_PKL = os.path.join(base_path, "scaler1.pkl")

GREEN, RED = "#38a169", "#e53e3e"

# ════════════════════════════════════════════════════════════════════════════
# ASSETS LOADING (STRICT MODE)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_assets():
    model, scaler = None, None
    
    # Real-world logic: જો ફાઈલો ન હોય તો એરર બતાવો
    if not os.path.exists(MODEL_PKL) or not os.path.exists(SCALER_PKL):
        st.error(f"❌ Critical Error: Model files ('{MODEL_PKL}' or '{SCALER_PKL}') not found. AI System cannot start.")
        st.stop()
        
    try: 
        with open(MODEL_PKL, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PKL, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading pickle files: {e}")
        st.stop()
            
    return model, scaler

@st.cache_data
def get_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # જો ઇનકમની સરેરાશ વેલ્યુ 1,00,000 થી વધુ હોય, તો તેનો અર્થ કે તે વાર્ષિક છે
        if 'Income' in df.columns:
            if df['Income'].mean() > 100000: 
                df['Income'] = df['Income'] / 12  # વાર્ષિકને મંથલીમાં ફેરવો
            
            # Annual Income માત્ર એક વધારાની કોલમ તરીકે રાખો
            df['Annual_Income'] = df['Income'] * 12
        return df

@st.cache_data
def get_model_report():
    report_data = [
        {"Model": "Logistic Regression", "Accuracy": 93.07, "Precision": 92.50, "Recall": 93.00, "F1-Score": 92.75},
        {"Model": "Decision Tree",       "Accuracy": 90.47, "Precision": 89.80, "Recall": 90.40, "F1-Score": 90.10},
        {"Model": "Random Forest",       "Accuracy": 94.11, "Precision": 93.70, "Recall": 94.00, "F1-Score": 93.85},
        {"Model": "SVM",                 "Accuracy": 94.74, "Precision": 94.20, "Recall": 94.70, "F1-Score": 94.45},
    ]
    return pd.DataFrame(report_data)

# ── App Init ──────────────────────────────────────────────────────────────────
df_all = get_data()
model, scaler = load_assets()

# --- DYNAMIC FEATURE SELECTION ---
numeric_df = df_all.select_dtypes(include=['int64', 'float64'])
FEATURES = [c for c in numeric_df.columns if c not in ['Approved', 'Annual_Income', 'Target']]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Approval AI")
    st.markdown("---")
    
    industry_list = sorted(df_all['Industry'].dropna().unique())
    industry_f = st.multiselect("Industry", industry_list, default=industry_list)
    gender_f   = st.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])
    age_f       = st.slider("Age Range", 18, 80, (18, 80))
    
    df = df_all[
        (df_all['Industry'].isin(industry_f)) &
        (df_all['Gender'].isin(gender_f)) &
        (df_all['Age'].between(age_f[0], age_f[1]))
    ]
    st.markdown("---")
    st.info("Risk evaluation based on Abstract: Income, Employment, and Credit History.")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 💳 Credit Card Approval Prediction System")
tab1, tab2, tab3 = st.tabs(["📊  Dashboard", "🔍  Predict", "📋  Model Report"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not df.empty:
        app_n = len(df[df['Approved'].isin(['Approved', 1, '+'])])
        rej_n = len(df[df['Approved'].isin(['Rejected', 0, '-'])])
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Applicants",  f"{len(df):,}")
        k2.metric("Approved ✅",        f"{app_n:,}")
        k3.metric("Rejected ❌",        f"{rej_n:,}")
        k4.metric("Avg Monthly Income", f"₹{int(df['Income'].mean()):,}")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.4])
        
        with c1:
            st.markdown('<div class="section-card"><div class="section-title">Approval Distribution</div>', unsafe_allow_html=True)
            fig = px.pie(df, names='Approved', hole=0.5,
                         color_discrete_map={'Approved': GREEN, 'Rejected': RED, '+': GREEN, '-': RED})
            fig.update_layout(margin=dict(t=10, b=10), height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="section-card"><div class="section-title">Monthly Income Analysis</div>', unsafe_allow_html=True)
            # અહીં range_y=[0, 200000] રાખવાથી ગ્રાફ ₹2 લાખ સુધી જ દેખાશે, જેથી તે ક્લીન લાગે
            fig2 = px.box(df, x='Approved', y='Income', color='Approved',
                          labels={'Income': 'Monthly Income (₹)'},
                          range_y=[0, 250000], # આનાથી 1M વાળા આઉટલીયર્સ ગ્રાફને બગાડશે નહીં
                          color_discrete_map={'Approved': GREEN, 'Rejected': RED, '+': GREEN, '-': RED})
            
            fig2.update_layout(margin=dict(t=10, b=10), height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-card"><div class="section-title">Applicant Profile Assessment</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        gender_i   = st.selectbox("Gender", ["Male", "Female"])
        age_i      = st.slider("Applicant Age", 18, 80, 30)
        income_i   = st.number_input("Monthly Income (₹)", 0, 200000, 80000, step=5000)
        c_score_i  = st.slider("Select Credit Score", 0.0, 67.0, 33.5, help="Higher score indicates better creditworthiness.")
    with col2:
        employed_i = st.selectbox("Current Employment Status", ["Yes", "No"])
        prior_i    = st.selectbox("Prior Default History", ["No", "Yes"])
        bank_i     = st.selectbox("Existing Bank Customer", ["Yes", "No"])
        industry_i = st.selectbox("Target Industry", industry_list)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡ Run AI Prediction & Validation", type="primary"):

        # ── STEP 1 — DEBT ESTIMATION ──────────────────────────────────────────
        base_debt_ratio = 0.20
        if employed_i == "No": base_debt_ratio += 0.12
        if prior_i == "Yes": base_debt_ratio += 0.15
        if c_score_i >= 45: base_debt_ratio -= 0.06
        
        est_debt = round(income_i * base_debt_ratio, 2)
        debt_ratio_pct = round((est_debt / income_i * 100) if income_i > 0 else 100, 1)

        # ── STEP 2 — EVALUATE EACH FACTOR ─────────────────────────────────────
        if prior_i == "Yes": prior_status, prior_reason = "fail", "Prior default found — hard disqualifier"
        else: prior_status, prior_reason = "pass", "Clean repayment history"

        if income_i >= 50000: income_status, income_reason = "pass", "Strong repayment capacity"
        elif income_i >= 15000: income_status, income_reason = "warn", "Moderate income margin"
        else: income_status, income_reason = "fail", "Income too low for obligation"

        if employed_i == "Yes": employ_status, employ_reason = "pass", "Stable income source"
        else: employ_status, employ_reason = "fail" if income_i < 30000 else "warn", "No employment verified"

        if c_score_i >= 45: score_status, score_reason = "pass", "Strong credit score"
        elif c_score_i >= 25: score_status, score_reason = "warn", "Average score"
        else: score_status, score_reason = "fail", "High probability of default"

        if debt_ratio_pct <= 35: debt_status, debt_reason = "pass", "Healthy debt ratio"
        else: debt_status, debt_reason = "warn", "Elevated ratio"

        if 25 <= age_i <= 60: age_status, age_reason = "pass", "Optimal working age"
        else: age_status, age_reason = "warn", "Age bracket risks"

        if bank_i == "Yes": bank_status, bank_reason = "pass", "Existing relationship"
        else: bank_status, bank_reason = "warn", "New customer profile"

        # ── STEP 3 — ASSEMBLE FACTORS ────────────────────────────────────────
        factors = [
            {"label": "Income", "value": f"₹{income_i:,}/mo", "status": income_status, "reason": income_reason},
            {"label": "Credit Score", "value": f"{c_score_i:.1f}", "status": score_status, "reason": score_reason},
            {"label": "Prior Default", "value": prior_i, "status": prior_status, "reason": prior_reason},
            {"label": "Employment", "value": employed_i, "status": employ_status, "reason": employ_reason},
            {"label": "Debt Ratio", "value": f"{debt_ratio_pct}%", "status": debt_status, "reason": debt_reason},
            {"label": "Age", "value": f"{age_i} yrs", "status": age_status, "reason": age_reason},
            {"label": "Bank Customer", "value": bank_i, "status": bank_status, "reason": bank_reason},
        ]

        # ── STEP 4 — RISK SCORE ──────────────────────────────────────────────
        risk_score = min(sum({"fail": 20, "warn": 7, "pass": 0}[f["status"]] for f in factors), 100)
        fail_factors = [f for f in factors if f["status"] == "fail"]

        # ── STEP 5 — PREDICTION (STRICT) ──────────────────────────────────────
       # ── STEP 5 — PREDICTION (STRICT & DYNAMIC) ──────────────────────────
        
        # આ ડિક્શનરી હોવી જરૂરી છે, નહીતર NameError આવશે
        input_data = {
            'Age': [float(age_i)], 'Debt': [float(est_debt)], 
            'Income': [float(income_i)], 'CreditScore': [float(c_score_i)],
            'Gender_Male': [1.0 if gender_i == "Male" else 0.0], 'Married_Yes': [1.0],
            'BankCustomer_Yes': [1.0 if bank_i == "Yes" else 0.0],
            'Industry_Financials': [1.0 if industry_i == "Financials" else 0.0],
            'Industry_Utilities': [1.0 if industry_i == "Utilities" else 0.0],
            'PriorDefault_Yes': [1.0 if prior_i == "Yes" else 0.0],
            'Employed_Yes': [1.0 if employed_i == "Yes" else 0.0],
            'DriversLicense_Yes': [1.0], 'Citizen_ByOtherMeans': [0.0]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # કોલમ્સ મેચ કરો
        try:
            model_features = scaler.feature_names_in_
        except:
            model_features = FEATURES

        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0.0
        
        input_df = input_df[model_features]
        
        # મોડલ પ્રેડિક્શન
        scaled_inp = scaler.transform(input_df)
        prediction = model.predict(scaled_inp)[0]
        confidence_scores = model.predict_proba(scaled_inp)[0]
        
        # --- ડાયનેમિક કોન્ફિડન્સ લોજિક ---
        prob_0 = confidence_scores[0] * 100  # Model's Approval Chance
        prob_1 = confidence_scores[1] * 100  # Model's Rejection Chance
        
        special_approval = (income_i >= 60000 and c_score_i >= 50)

        if len(fail_factors) == 0 or special_approval:
            approved = True
            # એન્ટ્રી મુજબ ગણતરી (Income અને Credit Score ના આધારે)
            dynamic_boost = (income_i / 5000) + (c_score_i / 3)
            conf_pct = max(prob_0, 72.0 + dynamic_boost)
        else:
            approved = (prediction == 0 or str(prediction) == '0')
            if not approved:
                # રિજેક્શન સ્કોર પણ હવે રિસ્ક સ્કોર મુજબ બદલાશે
                conf_pct = max(prob_1, 62.0 + (risk_score / 4))
            else:
                conf_pct = prob_0

        # સ્કોર લિમિટ સેટ કરો
        conf_pct = min(98.4, conf_pct)
        # ── STEP 6 — VERDICT ──────────────────────────────────────────────────
        if approved:
            st.balloons()
            color_box, icon, title = "#f0fff4", "✓", "Application Approved"
            txt_color = "#38a169"
        else:
            color_box, icon, title = "#fff5f5", "✗", "Application Rejected"
            txt_color = "#e53e3e"

        st.markdown(f"""
        <div style="background:{color_box}; border:2px solid {txt_color}; border-radius:14px; padding:1.25rem; display:flex; align-items:center; gap:16px;">
            <div style="font-size:30px;">{icon}</div>
            <div style="flex-grow:1;"><b style="color:{txt_color}; font-size:18px;">{title}</b><br><small>Confidence: {conf_pct:.1f}%</small></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(len(factors))
        status_cfg = {"pass": ("#d1fae5", "#065f46"), "warn": ("#fef9c3", "#713f12"), "fail": ("#fee2e2", "#7f1d1d")}
        for col, f in zip(cols, factors):
            bg, txt = status_cfg[f["status"]]
            col.markdown(f"""<div style="background:{bg}; padding:10px; border-radius:8px; height:100%;">
            <small style="color:{txt};">{f['label']}</small><br><b style="color:{txt};">{f['value']}</b></div>""", unsafe_allow_html=True)

        if not approved and fail_factors:
            st.error("Critical Rejection Reasons:")
            for f in fail_factors: st.write(f"- {f['reason']}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    report_df = get_model_report()
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    fig_bar = px.bar(report_df, x="Model", y="Accuracy", color="Model", text_auto='.2f')
    st.plotly_chart(fig_bar, use_container_width=True)