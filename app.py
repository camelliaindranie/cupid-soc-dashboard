import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             f1_score, precision_score, recall_score, accuracy_score)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- KONFIGURASI ---
st.set_page_config(page_title="CUPID NIDS | SOC Dashboard", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #4DB6AC;}
    .stMetric {background-color: #1E2127; padding: 15px; border-radius: 10px; border-left: 5px solid #4DB6AC;}
    </style>
""", unsafe_allow_html=True)

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    models = {
        'Random Forest': joblib.load('model_rf_tuned.joblib'),
        'XGBoost': joblib.load('model_xgb_tuned.joblib'),
        'LightGBM': joblib.load('model_lgb_tuned.joblib'),
        'KNN': joblib.load('model_knn_tuned.joblib'),
        'MLP': joblib.load('model_mlp_tuned.joblib'),
    }
    return models, joblib.load('scaler_final.joblib'), joblib.load('feature_names.joblib')

@st.cache_data
def load_test_data():
    df = pd.read_parquet('CUPID_final_test_scaled.parquet')
    return df.drop(columns=['Label']), df['Label']

models, scaler, feature_names = load_artifacts()
X_test, y_test = load_test_data()

# --- SIDEBAR ---
menu = st.sidebar.radio("Main Menu", [
    "🌐 Executive Summary", "📊 Model Interrogation", "🔮 Forecasting",
    "📈 Deep EDA", "🔍 Real-Time Manual Engine", "📁 Batch Inspection"
])

# --- LOGIKA MENU ---
if menu == "🌐 Executive Summary":
    st.title("🌐 Security Operations Center - Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traffic Samples", f"{len(X_test):,}")
    c2.metric("Threat Detected", f"{y_test.sum():,}")
    c3.metric("Models Ready", len(models))
    c4.metric("Status", "Optimal")

elif menu == "📊 Model Interrogation":
    st.title("📊 Model Interrogation")
    selected_model = st.selectbox("Engine:", list(models.keys()))
    model = models[selected_model]
    y_pred, y_prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    c1, c2 = st.columns(2)
    c1.metric("F1-Score (W)", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
    c2.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
    st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True, title="Confusion Matrix"))

elif menu == "🔮 Forecasting":
    st.title("🔮 Threat Forecasting")
    chunk = 500
    counts = [y_test[i*chunk:(i+1)*chunk].sum() for i in range(len(y_test)//chunk)]
    st.line_chart(ExponentialSmoothing(counts, trend='add').fit().forecast(24))

elif menu == "📈 Deep EDA":
    st.title("📈 Deep EDA")
    feat = st.selectbox("Pilih Fitur:", feature_names)
    st.plotly_chart(px.violin(pd.concat([X_test, y_test], axis=1), y=feat, x='Label', color='Label'))

elif menu == "🔍 Real-Time Manual Engine":
    st.title("🔍 Interactive Inference Engine")
    model_name = st.selectbox("Pilih Model:", list(models.keys()))
    if st.button("🚀 Trigger Random Anomaly"): st.session_state.vals = {f: np.random.uniform(X_test[f].min(), X_test[f].max()) for f in feature_names}
    cols = st.columns(3)
    vals = {f: cols[i % 3].slider(f, float(X_test[f].min()), float(X_test[f].max()), float(st.session_state.get('vals', {}).get(f, X_test[f].mean()))) for i, f in enumerate(feature_names)}
    
    if st.button("🔮 Prediksi", type="primary"):
        input_scaled = scaler.transform(pd.DataFrame([vals])[feature_names])
        prob = models[model_name].predict_proba(input_scaled)[0][1]
        c1, c2 = st.columns([1, 2])
        if prob > 0.5: c1.error("🚨 MALICIOUS ATTACK DETECTED")
        else: c1.success("✅ TRAFFIC BENIGN")
        c1.metric("Confidence", f"{prob*100:.2f}%")
        c2.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=prob*100, gauge={'axis': {'range': [0, 100]}})))

elif menu == "📁 Batch Inspection":
    st.title("📁 Batch Inspection")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        df_up = pd.read_csv(uploaded)[feature_names]
        preds = models['XGBoost'].predict(scaler.transform(df_up))
        st.write(pd.DataFrame({'Prediction': ['Attack' if p == 1 else 'Normal' for p in preds]}))
