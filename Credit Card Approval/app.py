import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Approval Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #f0f4f8 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1.5px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #1e3a5f !important;
    font-size: 17px !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li {
    color: #4a5568 !important;
    font-size: 13px !important;
}
section[data-testid="stSidebar"] label {
    color: #2d3748 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── KPI cards ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-top: 4px solid #2b6cb0 !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] {
    color: #2b6cb0 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
    color: #1e3a5f !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 5px;
    gap: 4px;
    border: 1.5px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    color: #718096 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: #2b6cb0 !important;
    color: #ffffff !important;
}

/* ── Form inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #1a202c !important;
    font-size: 14px !important;
}
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #2d3748 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Buttons ── */
.stFormSubmitButton > button, .stButton > button {
    background: #2b6cb0 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
}
.stFormSubmitButton > button:hover, .stButton > button:hover {
    background: #2c5282 !important;
}

/* ── Divider ── */
hr { border-color: #e2e8f0 !important; }

/* ── Headers ── */
h1 { color: #1e3a5f !important; font-size: 26px !important; font-weight: 700 !important; }
h2 { color: #1e3a5f !important; font-size: 18px !important; font-weight: 700 !important; }
h3 { color: #2d3748 !important; font-size: 15px !important; font-weight: 600 !important; }
p, li { color: #4a5568; }

/* ── Section card ── */
.section-card {
    background: #ffffff;
    border-radius: 14px;
    border: 1.5px solid #e2e8f0;
    padding: 1.25rem 1.5rem 1.5rem;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 12px;
    font-weight: 700;
    color: #2b6cb0;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 1rem;
    padding-bottom: 8px;
    border-bottom: 1.5px solid #ebf4ff;
}

/* ── Result cards ── */
.result-approved {
    background: #f0fff4;
    border: 2px solid #38a169;
    border-radius: 16px;
    padding: 1.75rem 1.5rem;
    text-align: center;
}
.result-rejected {
    background: #fff5f5;
    border: 2px solid #e53e3e;
    border-radius: 16px;
    padding: 1.75rem 1.5rem;
    text-align: center;
}
.prob-bar-bg {
    background: #e2e8f0;
    border-radius: 999px;
    height: 10px;
    margin: 1rem 0 0.4rem;
    overflow: hidden;
}
.factor-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #f0f4f8;
    font-size: 13px;
}
.factor-label { color: #4a5568; }
.factor-value { color: #1e3a5f; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    path = "credit_model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    np.random.seed(42)
    n = 4000
    age   = np.random.normal(40,15,n).clip(18,80)
    inc   = np.random.normal(50000,25000,n).clip(5000,200000)
    debt  = np.random.normal(10,8,n).clip(0,50)
    cs    = np.random.normal(33,18,n).clip(0,67)
    emp   = np.random.choice([0,1],n,p=[0.35,0.65])
    bank  = np.random.choice([0,1],n,p=[0.40,0.60])
    pdef  = np.random.choice([0,1],n,p=[0.30,0.70])
    mar   = np.random.choice([0,1],n,p=[0.45,0.55])
    gen   = np.random.choice([0,1],n)
    cit   = np.random.choice([0,1,2],n)
    score = (0.30*(inc/200000)+0.25*emp+0.15*(cs/67)
              +0.10*pdef+0.08*bank+0.07*mar-0.05*(debt/50))
    appr  = (score+np.random.normal(0,0.08,n)>0.38).astype(int)
    X = pd.DataFrame({'Age':age,'Income':inc,'Debt':debt,'CreditScore':cs,
                      'Employed':emp,'BankCustomer':bank,'PriorDefault':pdef,
                      'Married':mar,'Gender':gen,'Citizen':cit})
    Xt,_,yt,_ = train_test_split(X,appr,test_size=0.2,random_state=42)
    pipe = Pipeline([('sc',StandardScaler()),
                      ('m',RandomForestClassifier(n_estimators=200,max_depth=12,
                                                  random_state=42,class_weight='balanced'))])
    pipe.fit(Xt,yt)
    joblib.dump(pipe,path)
    return pipe


# ── Dashboard data ─────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    np.random.seed(7)
    n = 6784
    inds = ['Consumer','Transport','Research','Real Estate','Education',
            'Financials','Utilities','Communication','Healthcare','Retail']
    cits = ['ByBirth','ByOtherMeans','Temporary']
    df = pd.DataFrame({
        'Age':         np.random.normal(47,18,n).clip(18,80).astype(int),
        'Income':      np.random.normal(51720,30000,n).clip(5000,200000),
        'Debt':        np.random.normal(14.7,20,n).clip(0,50),
        'CreditScore': np.random.normal(33.4,18.7,n).clip(0,67),
        'Industry':    np.random.choice(inds,n),
        'Citizen':     np.random.choice(cits,n,p=[0.67,0.18,0.15]),
        'Employed':    np.random.choice(['Yes','No'],n,p=[0.65,0.35]),
        'Gender':      np.random.choice(['Male','Female'],n),
        'Approved':    np.random.choice(['Approved','Rejected'],n,p=[0.559,0.441]),
    })
    return df


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Approval AI")
    st.markdown("---")
    st.markdown("#### Dashboard Filters")
    df_all = get_data()
    gender_f   = st.multiselect("Gender",      ["Male","Female"],                        default=["Male","Female"])
    citizen_f  = st.multiselect("Citizenship", ["ByBirth","ByOtherMeans","Temporary"],       default=["ByBirth","ByOtherMeans","Temporary"])
    employed_f = st.multiselect("Employment",  ["Yes","No"],                                 default=["Yes","No"])
    age_f      = st.slider("Age Range", 18, 80, (18,80))
    st.markdown("---")
    st.markdown("#### Model Info")
    st.success("🤖 Random Forest\n\n86.39% accuracy · 200 estimators")
    st.markdown("---")
    st.caption("Credit Approval System v2.0")

df = df_all[
    df_all['Gender'].isin(gender_f) &
    df_all['Citizen'].isin(citizen_f) &
    df_all['Employed'].isin(employed_f) &
    df_all['Age'].between(age_f[0], age_f[1])
]

TEAL, NAVY, GREEN, RED = "#2b6cb0", "#1e3a5f", "#38a169", "#e53e3e"

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown("# 💳 Credit Card Approval Prediction System")
st.markdown("<p style='color:#718096;font-size:14px;margin-top:-10px;margin-bottom:1.5rem;'>"
            "AI-powered applicant evaluation · Real-time prediction · Dashboard analytics</p>",
            unsafe_allow_html=True)

# ── KPI row ────────────────────────────────────────────────────────────────────
approved_n = len(df[df['Approved']=='Approved'])
rejected_n = len(df[df['Approved']=='Rejected'])
appr_rate  = round(approved_n/len(df)*100,2) if len(df)>0 else 0
avg_cs     = round(df['CreditScore'].mean(),2) if len(df)>0 else 0

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total Applicants",  f"{len(df):,}")
k2.metric("Total Approved",    f"{approved_n:,}")
k3.metric("Total Rejected",    f"{rejected_n:,}")
k4.metric("Approval Rate",     f"{appr_rate}%")
k5.metric("Avg Credit Score",  f"{avg_cs}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  📊  Dashboard  ", "  🔍  Predict  ", "  📋  Model Report  "])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    # — Row 1: Donut + Top Industries bar —
    r1a, r1b = st.columns([1, 1.4])

    with r1a:
        st.markdown('<div class="section-card"><div class="section-title">Approval Distribution</div>', unsafe_allow_html=True)
        ac = df['Approved'].value_counts().reset_index()
        ac.columns = ['Status','Count']
        fig = px.pie(ac, names='Status', values='Count', hole=0.55,
                     color='Status', color_discrete_map={'Approved':GREEN,'Rejected':RED})
        fig.update_traces(textinfo='percent+label', textfont_size=12)
        fig.update_layout(margin=dict(t=5,b=5,l=5,r=5), height=250,
                          showlegend=False,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Plus Jakarta Sans'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1b:
        st.markdown('<div class="section-card"><div class="section-title">Top Industries by Approval Count</div>', unsafe_allow_html=True)
        ti = df[df['Approved']=='Approved']['Industry'].value_counts().head(8).reset_index()
        ti.columns = ['Industry','Count']
        fig = px.bar(ti, x='Count', y='Industry', orientation='h',
                     color_discrete_sequence=[TEAL])
        fig.update_layout(margin=dict(t=5,b=5,l=5,r=10), height=250,
                          yaxis={'categoryorder':'total ascending'},
                          xaxis_title='', yaxis_title='',
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Plus Jakarta Sans'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # — Row 2: Credit Score hist + Age area + Citizenship bar —
    r2a, r2b, r2c = st.columns(3)

    with r2a:
        st.markdown('<div class="section-card"><div class="section-title">Credit Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='CreditScore', color='Approved', nbins=25,
                           color_discrete_map={'Approved':GREEN,'Rejected':RED},
                           barmode='overlay', opacity=0.75)
        fig.update_layout(margin=dict(t=5,b=5,l=5,r=5), height=220,
                          xaxis_title='Credit Score', yaxis_title='Count',
                          legend=dict(orientation='h',y=-0.3,x=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Plus Jakarta Sans'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2b:
        st.markdown('<div class="section-card"><div class="section-title">Applicant Volume by Age</div>', unsafe_allow_html=True)
        ag = df.groupby(pd.cut(df['Age'],bins=[18,30,40,50,60,70,80])).size().reset_index()
        ag.columns = ['AgeGroup','Count']
        ag['AgeGroup'] = ag['AgeGroup'].astype(str)
        fig = px.area(ag, x='AgeGroup', y='Count', color_discrete_sequence=[TEAL])
        fig.update_traces(fill='tozeroy', line_color=TEAL, fillcolor='rgba(43,108,176,0.15)')
        fig.update_layout(margin=dict(t=5,b=5,l=5,r=5), height=220,
                          xaxis_title='Age Group', yaxis_title='',
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Plus Jakarta Sans'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c:
        st.markdown('<div class="section-card"><div class="section-title">Approval by Citizenship</div>', unsafe_allow_html=True)
        cd = df.groupby(['Citizen','Approved']).size().reset_index(name='Count')
        fig = px.bar(cd, x='Citizen', y='Count', color='Approved',
                     color_discrete_map={'Approved':GREEN,'Rejected':RED}, barmode='group')
        fig.update_layout(margin=dict(t=5,b=5,l=5,r=5), height=220,
                          xaxis_title='', yaxis_title='',
                          legend=dict(orientation='h',y=-0.3,x=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Plus Jakarta Sans'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("predict_form"):
        st.markdown('<div class="section-card"><div class="section-title">Personal Information</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1: gender_i   = st.selectbox("Gender",          ["Male","Female"])
        with p2: age_i      = st.number_input("Age",          18, 80, 35)
        with p3: married_i = st.selectbox("Marital Status", ["Yes","No"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Financial Details</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1: income_i = st.number_input("Annual Income (₹)", 0, 10000000, 50000, step=1000)
        with f2: debt_i   = st.number_input("Total Debt",        0.0, 100.0, 5.0, step=0.5)
        with f3: cs_i     = st.slider("Credit Score", 0, 67, 30,
                                       help="Internal credit score (0–67)")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Employment & Banking</div>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        with e1: employed_i = st.selectbox("Employed",         ["Yes","No"])
        with e2: bank_i     = st.selectbox("Bank Customer",   ["Yes","No"])
        with e3: prior_i    = st.selectbox("Prior Default",   ["No","Yes"])
        with e4: citizen_i  = st.selectbox("Citizenship",      ["ByBirth","ByOtherMeans","Temporary"])
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("⚡  Run Prediction")

    if submitted:
        model = load_or_train_model()
        cmap  = {"ByBirth":0,"ByOtherMeans":1,"Temporary":2}
        inp   = pd.DataFrame([{
            'Age':age_i,'Income':income_i,'Debt':debt_i,'CreditScore':cs_i,
            'Employed':1 if employed_i=="Yes" else 0,
            'BankCustomer':1 if bank_i=="Yes" else 0,
            'PriorDefault':1 if prior_i=="Yes" else 0,
            'Married':1 if married_i=="Yes" else 0,
            'Gender':1 if gender_i=="Male" else 0,
            'Citizen':cmap[citizen_i],
        }])
        pred    = model.predict(inp)[0]
        proba   = model.predict_proba(inp)[0]
        app_pct = round(proba[1]*100, 1)
        rej_pct = round(proba[0]*100, 1)

        st.markdown("<br>", unsafe_allow_html=True)
        rc1, rc2 = st.columns([1.1, 1])

        with rc1:
            if pred == 1:
                st.markdown(f"""
                <div class="result-approved">
                  <div style="font-size:44px;margin-bottom:6px;">✅</div>
                  <div style="font-size:30px;font-weight:700;color:#276749;margin-bottom:4px;">APPROVED</div>
                  <div style="font-size:13px;color:#2f855a;margin-bottom:8px;">Application has been approved</div>
                  <div class="prob-bar-bg">
                    <div style="background:#38a169;height:10px;border-radius:999px;width:{app_pct}%;"></div>
                  </div>
                  <div style="font-size:32px;font-weight:700;color:#276749;">{app_pct}%</div>
                  <div style="font-size:12px;color:#2f855a;">approval confidence</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-rejected">
                  <div style="font-size:44px;margin-bottom:6px;">❌</div>
                  <div style="font-size:30px;font-weight:700;color:#c53030;margin-bottom:4px;">REJECTED</div>
                  <div style="font-size:13px;color:#e53e3e;margin-bottom:8px;">Application has been declined</div>
                  <div class="prob-bar-bg">
                    <div style="background:#e53e3e;height:10px;border-radius:999px;width:{rej_pct}%;"></div>
                  </div>
                  <div style="font-size:32px;font-weight:700;color:#c53030;">{rej_pct}%</div>
                  <div style="font-size:12px;color:#e53e3e;">rejection confidence</div>
                </div>""", unsafe_allow_html=True)

        with rc2:
            st.markdown('<div class="section-card"><div class="section-title">Factor Summary</div>', unsafe_allow_html=True)
            factors = [
                ("Age",           "🟢" if 25<=age_i<=60 else "🟡",   str(age_i)),
                ("Income",        "🟢" if income_i>=40000 else "🔴",  f"₹{income_i:,}"),
                ("Credit Score",  "🟢" if cs_i>=25 else "🔴",         f"{cs_i}/67"),
                ("Debt",          "🟢" if debt_i<15 else "🔴",        f"{debt_i:.1f}"),
                ("Employment",    "🟢" if employed_i=="Yes" else "🔴", employed_i),
                ("Prior Default", "🟢" if prior_i=="No" else "🔴",    prior_i),
                ("Bank Customer", "🟢" if bank_i=="Yes" else "🟡",    bank_i),
            ]
            for label, icon, val in factors:
                st.markdown(
                    f"<div class='factor-row'>"
                    f"<span class='factor-label'>{icon} {label}</span>"
                    f"<span class='factor-value'>{val}</span></div>",
                    unsafe_allow_html=True)

            if app_pct >= 75:   risk, rbg, rc = "Low Risk",    "#f0fff4", "#276749"
            elif app_pct >= 50: risk, rbg, rc = "Medium Risk", "#fffff0", "#744210"
            else:                risk, rbg, rc = "High Risk",   "#fff5f5", "#c53030"

            st.markdown(
                f"<div style='text-align:center;padding:.9rem;background:{rbg};"
                f"border-radius:10px;border:1.5px solid #e2e8f0;margin-top:12px;'>"
                f"<div style='font-size:11px;color:#718096;text-transform:uppercase;"
                f"letter-spacing:.06em;font-weight:600;'>Risk Level</div>"
                f"<div style='font-size:22px;font-weight:700;color:{rc};margin-top:4px;'>{risk}</div></div>",
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    # UPDATED with latest model results from screenshot
    mr1,mr2,mr3,mr4 = st.columns(4)
    mr1.metric("Best Model",    "Random Forest", "")
    mr2.metric("Best Accuracy", "86.39%",        "+0.54%")
    mr3.metric("Best F1 Score", "84.87%",        "")
    mr4.metric("Dataset Size",  "6,784",         "")

    st.markdown("<br>", unsafe_allow_html=True)
    mc1, mc2 = st.columns([1.6, 1])

    with mc1:
        st.markdown('<div class="section-card"><div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)
        results = pd.DataFrame({
            "Model":      ["Logistic Regression","Decision Tree","Random Forest","SVM"],
            "Accuracy":   [85.95, 85.31, 86.39, 86.19],
            "Precision":  [85.66, 85.69, 84.64, 84.95],
            "Recall":     [82.47, 80.72, 85.10, 84.11],
            "F1 Score":   [84.04, 83.13, 84.87, 84.53],
        })
        fig = px.bar(results.melt(id_vars='Model',var_name='Metric',value_name='Score'),
                     x='Model', y='Score', color='Metric', barmode='group',
                     color_discrete_sequence=["#2b6cb0","#38a169","#d69e2e","#805ad5"])
        fig.update_layout(margin=dict(t=5,b=5,l=5,r=5), height=290,
                          yaxis=dict(range=[80,88]),
                          xaxis_title='', yaxis_title='Score (%)',
                          legend=dict(orientation='h',y=-0.22,x=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Plus Jakarta Sans'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with mc2:
        st.markdown('<div class="section-card"><div class="section-title">Data Pipeline Funnel</div>', unsafe_allow_html=True)
        ds = pd.DataFrame({'Stage':['Raw Data','After Dedup','After Dropna','Final'],
                           'Records':[10055,9910,7200,6784]})
        fig = px.funnel(ds, x='Records', y='Stage', color_discrete_sequence=[TEAL])
        fig.update_layout(margin=dict(t=5,b=5,l=5,r=5), height=290,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Plus Jakarta Sans'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Full Results Table</div>', unsafe_allow_html=True)
    results['Status'] = ['✅ Good','✅ Good','🏆 Best','✅ Good']
    
    # Converting to percentage string without lambda
    for col in ['Accuracy','Precision','Recall','F1 Score']:
        results[col] = results[col].astype(str) + "%"
        
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)