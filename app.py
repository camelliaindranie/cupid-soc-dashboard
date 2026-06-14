import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             classification_report, f1_score, precision_score, 
                             recall_score, accuracy_score, precision_recall_curve)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- KONFIGURASI ---
st.set_page_config(page_title="CUPID NIDS | SOC Dashboard", page_icon="🛡️", layout="wide")

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

# --- LOGIC MENU ---
if menu == "🌐 Executive Summary":
    st.title("🌐 Security Operations Center - Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Traffic Samples", f"{len(X_test):,}")
    c2.metric("Threat Detected", f"{y_test.sum():,}")
    c3.metric("Models Available", len(models))

elif menu == "📊 Model Interrogation":
    st.title("📊 Model Interrogation")
    selected_model = st.selectbox("Engine:", list(models.keys()))
    model = models[selected_model]
    y_pred, y_prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    
    c1, c2 = st.columns(2)
    c1.metric("F1-Score (W)", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
    c2.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    st.plotly_chart(px.imshow(cm, text_auto=True, title="Confusion Matrix", x=['Normal','Attack'], y=['Normal','Attack']))

elif menu == "🔮 Forecasting":
    st.title("🔮 Threat Forecasting")
    chunk_size = 500
    attack_counts = [y_test[i*chunk_size:(i+1)*chunk_size].sum() for i in range(len(y_test)//chunk_size)]
    model_hw = ExponentialSmoothing(attack_counts, trend='add').fit()
    st.line_chart(model_hw.forecast(24))

elif menu == "📈 Deep EDA":
    st.title("📈 Deep EDA")
    feat = st.selectbox("Fitur:", feature_names)
    fig = px.violin(pd.concat([X_test, y_test], axis=1), y=feat, x='Label', color='Label')
    st.plotly_chart(fig)

elif menu == "🔍 Real-Time Manual Engine":
    st.title("🔍 Inference Engine")
    input_vals = {f: st.slider(f, float(X_test[f].min()), float(X_test[f].max()), float(X_test[f].mean())) for f in feature_names}
    
    if st.button("🔮 Prediksi", type="primary"):
        input_df = pd.DataFrame([input_vals])[feature_names]
        input_scaled = scaler.transform(input_df) # SCALING DIPERBAIKI
        model = models[st.selectbox("Model:", list(models.keys()))]
        prob = model.predict_proba(input_scaled)[0]
        
        c1, c2 = st.columns(2)
        c1.metric("Attack Confidence", f"{prob[1]*100:.2f}%")
        c2.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=prob[1]*100, gauge={'axis': {'range': [0, 100]}})))

elif menu == "📁 Batch Inspection":
    st.title("📁 Batch Inspection")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        df_up = pd.read_csv(uploaded)[feature_names]
        input_scaled = scaler.transform(df_up) # SCALING DIPERBAIKI
        preds = models['XGBoost'].predict(input_scaled)
        st.write(pd.DataFrame({'Prediction': preds}))
