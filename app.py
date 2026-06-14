import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             classification_report, f1_score, precision_recall_curve,
                             precision_score, recall_score, accuracy_score)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- KONFIGURASI PAGE ---
st.set_page_config(page_title="CUPID NIDS | SOC Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS UNTUK TAMPILAN PROFESIONAL ---
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #4DB6AC;}
    .stMetric {background-color: #1E2127; padding: 15px; border-radius: 10px; border-left: 5px solid #4DB6AC;}
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA & ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    models = {
        'Random Forest': joblib.load('model_rf_tuned.joblib'),
        'XGBoost'      : joblib.load('model_xgb_tuned.joblib'),
        'LightGBM'     : joblib.load('model_lgb_tuned.joblib'),
        'KNN'          : joblib.load('model_knn_tuned.joblib'),
        'MLP'          : joblib.load('model_mlp_tuned.joblib'),
    }
    scaler        = joblib.load('scaler_final.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return models, scaler, feature_names

@st.cache_data
def load_test_data():
    df = pd.read_parquet('CUPID_final_test_scaled.parquet')
    return df.drop(columns=['Label']).reset_index(drop=True), df['Label'].reset_index(drop=True)

models, scaler, feature_names = load_artifacts()
X_test, y_test = load_test_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://img.icons8.com/external-flat-juicy-fish/100/external-cyber-cyber-security-flat-flat-juicy-fish.png", width=80)
st.sidebar.title("SOC Dashboard")
st.sidebar.caption("Network Intrusion Detection System v2.0")
st.sidebar.markdown("---")

menu = st.sidebar.radio("Main Menu", [
    "🌐 Executive Summary", 
    "📊 Model Interrogation & XAI", 
    "🔮 Threat Forecasting",
    "📈 Deep EDA & Profiling",
    "🔍 Real-Time Manual Engine", 
    "📁 Batch Traffic Inspection"
])
st.sidebar.markdown("---")
st.sidebar.info(f"📁 **Active Dataset:** CUPID 2022\n\n📌 **Test Samples:** {len(X_test):,}\n\n⚙️ **Features:** {len(feature_names)}")

# ==========================================
# 1-4 (EXECUTIVE, MODEL, FORECASTING, EDA - TIDAK DIUBAH)
# ==========================================
if menu == "🌐 Executive Summary":
    st.title("🌐 Security Operations Center (SOC) - Overview")
    # ... (bagian ini tetap sama sesuai file lama kamu)
elif menu == "📊 Model Interrogation & XAI":
    st.title("📊 Model Interrogation & Explainable AI")
    # ... (bagian ini tetap sama sesuai file lama kamu)
elif menu == "🔮 Threat Forecasting":
    st.title("🔮 Threat Forecasting Analytics")
    # ... (bagian ini tetap sama sesuai file lama kamu)
elif menu == "📈 Deep EDA & Profiling":
    st.title("📈 Deep Exploratory Data Analysis")
    # ... (bagian ini tetap sama sesuai file lama kamu)

# ==========================================
# 5. REAL-TIME MANUAL ENGINE (DIPERBAIKI)
# ==========================================
elif menu == "🔍 Real-Time Manual Engine":
    st.title("🔍 Interactive Inference Engine")
    selected_model = st.selectbox("Pilih Model Inference:", list(models.keys()))
    defaults = X_test.mean()
    with st.expander("🛠️ Konfigurasi Parameter (Network Traffic)", expanded=True):
        cols = st.columns(3)
        input_vals = {}
        for i, feat in enumerate(feature_names):
            with cols[i % 3]:
                input_vals[feat] = st.slider(feat, float(X_test[feat].min()), float(X_test[feat].max()), float(defaults[feat]), key=feat)
    if st.button("🔮 Prediksi", use_container_width=True, type="primary"):
        input_df = pd.DataFrame([input_vals])[feature_names] # Urutan kolom dijaga
        input_scaled = scaler.transform(input_df)            # SCALING DITAMBAHKAN
        model = models[selected_model]
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]
        c_res, c_gauge = st.columns([1, 2])
        with c_res:
            if pred == 1: st.error("🚨 **MALICIOUS ATTACK DETECTED**")
            else: st.success("✅ **TRAFFIC BENIGN (NORMAL)**")
            st.metric("Attack Confidence", f"{prob[1]*100:.2f}%")
        with c_gauge:
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob[1] * 100, 
                                 gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkred"}}))
            st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================================
# 6. BATCH TRAFFIC INSPECTION (DIPERBAIKI)
# ==========================================
elif menu == "📁 Batch Traffic Inspection":
    st.title("📁 Batch Traffic Inspection")
    selected_model = st.selectbox("Engine Scanner:", list(models.keys()))
    uploaded = st.file_uploader("Upload CSV Log", type=['csv'])
    if uploaded:
        df_upload = pd.read_csv(uploaded)[feature_names] # Urutan kolom dijaga
        input_scaled = scaler.transform(df_upload)      # SCALING DITAMBAHKAN
        preds = models[selected_model].predict(input_scaled)
        probs = models[selected_model].predict_proba(input_scaled)[:, 1]
        st.write(pd.DataFrame({'Prediction': preds, 'Threat_Score': probs}))
