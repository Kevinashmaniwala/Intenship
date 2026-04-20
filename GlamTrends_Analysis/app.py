import streamlit as st
import pandas as pd
import numpy as np
import time
import sqlite3
import io
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA 
from xgboost import XGBClassifier
from wordcloud import WordCloud
from transformers import pipeline
import streamlit.components.v1 as components
import optuna 

# ================= 1. SYSTEM SETTINGS =================
st.set_page_config(
    page_title="Glame ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# UI/UX Fixes
components.html(
    """
    <script>
        var style = window.parent.document.createElement('style');
        style.innerHTML = `
            html, body, [data-testid="stAppViewContainer"], [data-testid="stMainViewContainer"] {
                overflow: auto !important;
            }
            [data-testid="stSidebar"] {
                display: none !important;
            }
        `;
        window.parent.document.head.appendChild(style);
    </script>
    """,
    height=0,
)

st.markdown("""
<style>
[data-testid="stDataFrame"] { height: 300px !important; max-height: 400px !important; overflow-y: auto !important; }
.exploration-header {
    background: linear-gradient(90deg, #2e3192 0%, #1bffff 100%);
    padding: 12px 20px; border-radius: 10px; color: white;
    font-size: 1.4rem; font-weight: bold; margin-bottom: 20px;
}
.main-title { color: #1eb197; font-size: 3rem; font-weight: 800; text-align: center; margin-bottom: 20px; }
.card-container {
    background: white; border: 1px solid #e6e9ef; border-radius: 15px;
    padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); min-height: 250px;
}
.card-header { 
    border-left: 5px solid #1eb197; padding-left: 10px; 
    font-weight: bold; font-size: 1.1rem; color: #31333f; margin-bottom: 15px;
}
.stButton>button {
    background-color: #1eb197 !important; color: white !important;
    width: 100%; border-radius: 8px; border: none; height: 45px; font-weight: bold;
}
header { visibility: hidden !important; }
#MainMenu, footer { visibility: hidden; }
.pros-cons-card {
    background: #f8f9fa; border-left: 5px solid #1eb197; padding: 15px; border-radius: 8px; margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ================= 2. CORE DATA ENGINE =================
@st.cache_data 
def load_initial_data():
    try:
        df = pd.read_csv("C:/Users/A/OneDrive/Desktop/Intenship/Womens Clothing E-Commerce Reviews.csv")
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("")
            else:
                df[col] = df[col].fillna(0)
        return df
    except Exception as e:
        return pd.DataFrame({
            'Clothing ID': [847, 1080, 1077, 1049, 858],
            'Age': [33, 34, 60, 50, 35],
            'Rating': [4, 5, 3, 5, 5],
            'Review Text': ["Absolutely wonderful", "Love this dress", "High hopes", "I love this jumpsuit", "Love these knits"],
            'Recommended IND': [1, 1, 0, 1, 1],
            'Class Name': ['Blouses', 'Dresses', 'Dresses', 'Pants', 'Knits'],
            'Department Name': ['Tops', 'Dresses', 'Dresses', 'Bottoms', 'Tops'],
            'Title': ["", "", "Flaw", "Favorite", "Great Knit"],
            'Positive Feedback Count': [0, 4, 0, 0, 12]
        })

main_df = load_initial_data()

# Global Session States
if 'working_df' not in st.session_state: st.session_state.working_df = None
if 'scan_done' not in st.session_state: st.session_state.scan_done = False
if 'result_df' not in st.session_state: st.session_state.result_df = None

sentiment_engine = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# ================= 3. MAIN INTERFACE =================
st.markdown('<div class="main-title">GlamTrends AI Studio</div>', unsafe_allow_html=True)

t_rec, t_bulk, t_studio, t_modeler = st.tabs(["🛍️ AI Recommendations", "🏠 Bulk Scanner", "🚀 Enterprise AI Studio", "⚙️ ML Modeler Studio"])

# --- TAB: AI RECOMMENDATIONS ---
with t_rec:
    st.subheader("🛍️ Intelligent Style Finder")
    st.markdown("Advanced AI-powered filtering and product intelligence.")
    
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        search_q = st.text_input("Search for a style or category", placeholder="e.g. Sweaters, Knits, Dresses...", key="rec_search_bar")
    with sc2:
        min_rating_filter = st.select_slider("Minimum Rating Score", options=[1, 2, 3, 4, 5], value=1, key="rec_filter_rate")
    
    current_data = main_df
    
    if search_q:
        search_query = search_q.lower()
        match_mask = (
            current_data['Class Name'].astype(str).str.lower().str.contains(search_query) |
            current_data['Review Text'].astype(str).str.lower().str.contains(search_query) |
            current_data['Title'].astype(str).str.lower().str.contains(search_query) |
            current_data['Department Name'].astype(str).str.lower().str.contains(search_query)
        )
        
        rec_results = (current_data[match_mask & (current_data['Rating'] >= min_rating_filter)]
                       .sort_values(by=['Rating', 'Positive Feedback Count'], ascending=[False, False])
                       .head(10).copy())
        
        if not rec_results.empty:
            avg_age = int(rec_results['Age'].median())
            pos_ratio = round((len(rec_results[rec_results['Rating'] >= 4]) / len(rec_results)) * 100, 1)
            
            st.markdown(f"""
            <div class="pros-cons-card">
                <h4>🤖 AI Executive Summary for "{search_q}" (Top Rated Results)</h4>
                <p><b>Target Audience:</b> Ideal for ages around <b>{avg_age}</b>. <br>
                <b>Customer Sentiment:</b> <b>{pos_ratio}%</b> satisfaction rate. <br>
                <b>Recommendation:</b> Displaying top {len(rec_results)} high-rated verified data points.</p>
            </div>
            """, unsafe_allow_html=True)
            
            display_cols = ['Department Name', 'Class Name', 'Rating', 'Age', 'Positive Feedback Count', 'Review Text']
            st.dataframe(rec_results[display_cols], use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown(f"#### 💡 Based on your interest in '{search_q}', you might also like:")
            
            related_mask = (
                ~current_data['Class Name'].astype(str).str.lower().str.contains(search_query) & 
                ~current_data['Department Name'].astype(str).str.lower().str.contains(search_query)
            )
            
            related_data = current_data[related_mask & (current_data['Rating'] >= 4)]
            
            if related_data.empty:
                related_data = current_data[current_data['Rating'] == 5]
                
            unique_suggestions = related_data['Class Name'].unique()
            
            if len(unique_suggestions) > 0:
                n_to_show = min(4, len(unique_suggestions))
                suggestion_list = pd.Series(unique_suggestions).sample(n_to_show).tolist()
                
                rec_cols = st.columns(n_to_show)
                for i in range(n_to_show):
                    rec_cols[i].info(f"✨ **{suggestion_list[i]}**")
            
        else:
            st.warning("No matches found. Try broadening your search.")
    else:
        st.markdown("#### 🔥 Trending Styles & Top Picks")
        st.dataframe(current_data[current_data['Rating'] >= 5].head(10), use_container_width=True, hide_index=True)

# --- TAB: BULK SCANNER ---
with t_bulk:
    st.subheader("Bulk Scanner Intelligence")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card-container"><div class="card-header">1. Download Template</div>', unsafe_allow_html=True)
        template_fmt = st.selectbox("Format", ["CSV File", "JSON File", "SQL Script"], key="tpl_fmt")
        if template_fmt == "CSV File":
            st.download_button("📥 Get CSV Template", main_df.head(10).to_csv(index=False), "template.csv", use_container_width=True)
        elif template_fmt == "JSON File":
            st.download_button("📥 Get JSON Template", main_df.head(10).to_json(orient="records"), "template.json", use_container_width=True)
        elif template_fmt == "SQL Script":
            st.download_button("📥 Get SQL Script", "CREATE TABLE reviews (Rating INT, Age INT, Review_Text TEXT);", "template.sql", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-container"><div class="card-header">2. Scan File</div>', unsafe_allow_html=True)
        bulk_source = st.radio("Select Source", ["CSV", "JSON", "SQL", "G-Drive"], horizontal=True, key="bulk_src_radio")
        raw_scan_data = None
        
        if bulk_source == "CSV":
            f_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed", key="bulk_csv")
            if f_csv: raw_scan_data = pd.read_csv(f_csv)
        elif bulk_source == "JSON":
            f_json = st.file_uploader("Upload JSON", type=["json"], label_visibility="collapsed", key="bulk_json")
            if f_json: raw_scan_data = pd.read_json(f_json)
        elif bulk_source == "SQL":
            f_sql_up = st.file_uploader("Upload SQL/DB", type=["db", "sqlite"], label_visibility="collapsed", key="bulk_sql")
            if f_sql_up:
                with open("temp_bulk.db", "wb") as f_b: f_b.write(f_sql_up.getbuffer())
                conn_b = sqlite3.connect("temp_bulk.db")
                raw_scan_data = pd.read_sql("SELECT * FROM reviews LIMIT 100", conn_b)
        else:
            drive_link = st.text_input("Google Drive Link", key="bulk_drive_link")
            if drive_link and "drive.google.com" in drive_link:
                try:
                    file_id = drive_link.split("/d/")[1].split("/")[0]
                    raw_scan_data = pd.read_csv(f"https://drive.google.com/uc?id={file_id}")
                except: st.error("Invalid Link")

        if raw_scan_data is not None and st.button("EXECUTE AI SCAN", key="run_scan"):
            raw_scan = raw_scan_data.fillna(0).head(100)
            
            if 'Review Text' in raw_scan.columns:
                sent_out = sentiment_engine(raw_scan['Review Text'].astype(str).tolist())
                raw_scan['AI_Sentiment'] = pd.DataFrame(sent_out)['label'].values
                raw_scan['AI_Confidence'] = (pd.DataFrame(sent_out)['score'].values * 100).round(1)
            
            raw_scan['Trust_Score'] = 100.0
            
            c_neg_high = (raw_scan['AI_Sentiment'] == 'NEGATIVE') & (raw_scan['Rating'] >= 4)
            raw_scan.loc[c_neg_high, 'Trust_Score'] = 45.0
            
            c_pos_low = (raw_scan['AI_Sentiment'] == 'POSITIVE') & (raw_scan['Rating'] <= 2)
            raw_scan.loc[c_pos_low, 'Trust_Score'] = 35.0
            
            c_reco_anomaly = (raw_scan['Rating'] <= 2) & (raw_scan['Recommended IND'] == 1)
            raw_scan.loc[c_reco_anomaly, 'Trust_Score'] = 60.0

            raw_scan['Risk_Category'] = "🟢 Safe"
            raw_scan.loc[raw_scan['Trust_Score'] <= 50, 'Risk_Category'] = "🔴 High Risk"
            raw_scan.loc[(raw_scan['Trust_Score'] > 50) & (raw_scan['Trust_Score'] <= 75), 'Risk_Category'] = "🟡 Warning"

            st.session_state.result_df = raw_scan
            st.session_state.scan_done = True
            st.toast("Scan Complete!")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card-container"><div class="card-header">3. Export Report</div>', unsafe_allow_html=True)
        if st.session_state.scan_done:
            export_fmt = st.selectbox("Export As", ["CSV Report", "JSON Report", "SQL Report"], key="exp_fmt")
            res = st.session_state.result_df
            if "JSON" in export_fmt:
                st.download_button("💾 Download", res.to_json(orient="records"), "report.json", use_container_width=True)
            elif "SQL" in export_fmt:
                sql_data = "INSERT INTO analysis_report (Rating, Trust_Score, Category) VALUES " + \
                            ", ".join(["(" + str(r['Rating']) + "," + str(r['Trust_Score']) + ",'" + str(r['Risk_Category']) + "')" for _, r in res.iterrows()]) + ";"
                st.download_button("💾 Download", sql_data, "report.sql", use_container_width=True)
            else:
                st.download_button("💾 Download", res.to_csv(index=False), "report.csv", use_container_width=True)
        else:
            st.button("Export Locked", disabled=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.scan_done:
        st.markdown("---")
        st.markdown("### 🎯 Deep Intelligence Insights")
        res_view = st.session_state.result_df
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Items Scanned", len(res_view))
        m2.metric("Anomalies Found", len(res_view[res_view['Risk_Category'] != "🟢 Safe"]), delta_color="inverse")
        m3.metric("Avg Trust Score", f"{res_view['Trust_Score'].mean():.1f}%")
        m4.metric("AI Confidence", f"{res_view['AI_Confidence'].mean():.1f}%")
        
        # Search functionality within scan results
        search_scan = st.text_input("🔍 Search within results...", key="scan_search_box")
        if search_scan:
            res_view = res_view[res_view.apply(lambda row: search_scan.lower() in row.astype(str).str.lower().values, axis=1)]
            
        st.dataframe(res_view.style.background_gradient(subset=['Trust_Score'], cmap='RdYlGn'), use_container_width=True)

# --- TAB: ENTERPRISE AI STUDIO ---
with t_studio:
    st.markdown('<div class="exploration-header">🚀 Enterprise All-in-One Dashboard</div>', unsafe_allow_html=True)
    
    with st.expander("📂 Data Connection & Configuration", expanded=True):
        c_left, c_right = st.columns([1, 2])
        with c_left:
            mode = st.selectbox("Data Source", ["Local CSV", "Local JSON", "SQL Database", "Google Drive"], key="studio_mode")
        with c_right:
            df_studio = None
            if mode == "Local CSV":
                f = st.file_uploader("Upload CSV", type="csv", key="sb_csv")
                if f: df_studio = pd.read_csv(f)
            elif mode == "Local JSON":
                fj = st.file_uploader("Upload JSON", type="json", key="sb_json")
                if fj: df_studio = pd.read_json(fj)
            elif mode == "SQL Database":
                db = st.file_uploader("Upload SQLite DB", type=["db", "sqlite"], key="sb_sql")
                if db:
                    with open("temp.db", "wb") as f_sql: f_sql.write(db.getbuffer())
                    conn = sqlite3.connect("temp.db")
                    df_studio = pd.read_sql("SELECT * FROM reviews", conn)
            elif mode == "Google Drive":
                drive_url = st.text_input("Paste Drive Share Link", placeholder="https://drive.google.com/file/d/ID/view", key="studio_drive")
                if drive_url:
                    try:
                        file_id = drive_url.split('/')[-2] if '/d/' in drive_url else drive_url
                        direct_url = f'https://drive.google.com/uc?export=download&id={file_id}'
                        df_studio = pd.read_csv(direct_url)
                        st.success("Drive data loaded!")
                    except: st.error("Invalid Link.")
    
    if df_studio is not None:
        st.session_state.working_df = df_studio 
        numeric_cols = df_studio.select_dtypes(include=["int64", "float64"]).columns.tolist()
        text_cols = df_studio.select_dtypes(include="object").columns.tolist()

        st.markdown("### 🏠 Data Central & Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", len(df_studio))
        m2.metric("Avg Rating", round(df_studio['Rating'].mean(), 2) if 'Rating' in df_studio else 0)
        m3.metric("Text Columns", len(text_cols))
        
        st.markdown("**Previewing latest 10 records:**")
        st.dataframe(df_studio.head(10), use_container_width=True)

        st.divider()

     st.markdown("### 🤖 NLP Intelligence Lab")
        nl_left, nl_right = st.columns([1, 1])
        
        with nl_left:
            st.write("**📊 Sentiment Analysis**")
            if text_cols:
                sel_col = st.selectbox("Select Text Column", text_cols, key="dash_sent_col")
                if st.button("Run Intelligence Scan"):
                    # CLEAN DATA: Filter out empty strings and "nan" strings
                    raw_sample = df_studio[sel_col].astype(str).head(50).tolist()
                    sample_data = [t for t in raw_sample if t.strip() and t.strip().lower() != "nan"]
                    
                    if sample_data:
                        results = sentiment_engine(sample_data)
                        sent_df = pd.DataFrame({
                            "Text Snippet": [t[:50]+"..." for t in sample_data],
                            "Label": [r['label'] for r in results],
                            "Score": [round(r['score']*100, 1) for r in results]
                        })
                        st.plotly_chart(px.pie(sent_df, names="Label", hole=0.4, height=250, color_discrete_sequence=['#1eb197', '#ef553b']), use_container_width=True)
                        st.dataframe(sent_df, height=200)
                    else:
                        st.warning("⚠️ No valid text found in the selected column.")
            else:
                st.info("No text columns found.")

        with nl_right:
            st.write("**☁️ Keyword Trends (Word Cloud)**")
            if text_cols:
                # CLEAN DATA: Ensure no NaNs or empty strings are joined
                clean_series = df_studio[text_cols[0]].fillna("").astype(str)
                filtered_series = clean_series[clean_series.str.strip() != ""]
                text_data = " ".join(filtered_series.head(500))
                
                if text_data.strip():
                    wc = WordCloud(width=600, height=350, background_color="white", colormap="viridis").generate(text_data)
                    fig, ax = plt.subplots()
                    ax.imshow(wc)
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("ℹ️ Not enough text data to generate a word cloud.")# --- TAB: ML MODELER STUDIO ---
                    
with t_modeler:
    st.markdown('<div class="exploration-header">⚙️ ML Modeler - Advanced Training</div>', unsafe_allow_html=True)
    df_ml = st.session_state.working_df
    
    if df_ml is not None:
        l, r = st.columns([1, 2])
        with l:
            if st.checkbox("Enable Interaction Features (Age*Rating)"):
                if 'Age' in df_ml.columns and 'Rating' in df_ml.columns:
                    df_ml['Age_Rating_Interaction'] = df_ml['Age'] * df_ml['Rating']
            
            use_pca = st.checkbox("Apply PCA Reduction")
            st.info("🎯 Target Variable: Rating")
            target = "Rating" if "Rating" in df_ml.columns else df_ml.select_dtypes(include=[np.number]).columns.tolist()[-1]
            algo = st.selectbox("Algorithm", ["XGBoost", "RandomForest"], key="ml_algo") 
            
            numeric_cols_ml = df_ml.select_dtypes(include=[np.number]).columns.tolist()
            available_feats = [c for c in numeric_cols_ml if c != target]
            feats = st.multiselect("Select Features (X)", available_feats, default=available_feats[:2], key="ml_feats")
            
            st.markdown("---")
            p_lr = st.slider("Learning Rate", 0.01, 0.3, 0.1)
            p_depth = st.slider("Max Depth", 3, 10, 5)
            reg_alpha = st.slider("L1 Regularization (Alpha)", 0.0, 1.0, 0.0)
            reg_lambda = st.slider("L2 Regularization (Lambda)", 0.0, 1.0, 1.0)
            
            use_grid = st.checkbox("Use GridSearchCV") 
            use_bayesian = st.checkbox("Use Bayesian Opt (Optuna)") 
            train = st.button("🚀 Train & Optimize Model", key="ml_train_btn")
        
        with r:
            if train and feats:
                X, y = df_ml[feats], LabelEncoder().fit_transform(df_ml[target])
                
                if use_pca:
                    pca = PCA(n_components=min(len(feats), 2))
                    X = pca.fit_transform(X)
                
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
                
                try:
                    if use_bayesian:
                        def objective(trial):
                            p_nest = trial.suggest_int('n_estimators', 50, 150)
                            p_maxd = trial.suggest_int('max_depth', 3, 8)
                            if algo == "XGBoost":
                                p_lrat = trial.suggest_float('learning_rate', 0.01, 0.3)
                                m = XGBClassifier(n_estimators=p_nest, max_depth=p_maxd, learning_rate=p_lrat, eval_metric='mlogloss', reg_alpha=reg_alpha, reg_lambda=reg_lambda)
                            else:
                                m = RandomForestClassifier(n_estimators=p_nest, max_depth=p_maxd)
                            return cross_val_score(m, X_tr, y_tr, cv=3).mean()
                        
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=5)
                        
                        if algo == "XGBoost":
                            model = XGBClassifier(**study.best_params, reg_alpha=reg_alpha, reg_lambda=reg_lambda).fit(X_tr, y_tr)
                        else:
                            rf_best_params = {'n_estimators': study.best_params['n_estimators'], 'max_depth': study.best_params['max_depth']}
                            model = RandomForestClassifier(**rf_best_params).fit(X_tr, y_tr)
                            
                    elif use_grid:
                        grid = GridSearchCV(XGBClassifier(eval_metric='mlogloss'), {'n_estimators': [50, 100], 'max_depth': [3, 5]}, cv=3).fit(X_tr, y_tr)
                        model = grid.best_estimator_
                    else:
                        if algo == "XGBoost":
                            model = XGBClassifier(learning_rate=p_lr, max_depth=p_depth, n_estimators=100, reg_alpha=reg_alpha, reg_lambda=reg_lambda, eval_metric='mlogloss')
                            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], early_stopping_rounds=10, verbose=False)
                        else:
                            model = RandomForestClassifier(n_estimators=100, max_depth=p_depth).fit(X_tr, y_tr)
                    
                    joblib.dump(model, 'final_model.pkl')
                    st.success(f"Accuracy: {round(model.score(X_te, y_te)*100, 2)}%")
                    
                    st.markdown("#### 🔍 Model Interpretability (Feature Impact)")
                    feat_names = feats if not use_pca else ['PC1', 'PC2']
                    feat_imp = pd.DataFrame({'Feature': feat_names, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
                    st.plotly_chart(px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color='Importance', template="plotly_white"), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("⚠️ No data available. Connect a source in 'Enterprise AI Studio' first.")

st.divider()
st.markdown("<div style='text-align: center; color: grey;'>created by Kevin Ashmaniwala</div>", unsafe_allow_html=True)
