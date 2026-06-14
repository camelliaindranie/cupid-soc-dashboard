import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CUPID NIDS | SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* base */
.main { background-color: #0B0F19; }
section[data-testid="stSidebar"] { background-color: #111827; }

/* typography */
h1, h2, h3 { color: #38BDF8; letter-spacing: .03em; }
p, label, .stRadio label { color: #CBD5E1; }

/* metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #334155;
    border-left: 4px solid #38BDF8;
    border-radius: 12px;
    padding: 18px 20px;
}
div[data-testid="metric-container"] label { color: #94A3B8 !important; font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #F8FAFC !important; font-size:1.7rem; font-weight:700; }
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { color: #4ADE80 !important; }

/* buttons */
div.stButton > button {
    background: linear-gradient(90deg,#0EA5E9,#6366F1);
    color: #fff; border: none; border-radius: 8px;
    padding: 10px 28px; font-weight: 600; letter-spacing:.04em;
    transition: opacity .2s;
}
div.stButton > button:hover { opacity: .85; }

/* selectbox / slider labels */
.stSelectbox label, .stSlider label { color: #94A3B8 !important; }

/* tab strip */
button[data-baseweb="tab"] { color: #94A3B8 !important; font-weight:600; }
button[data-baseweb="tab"][aria-selected="true"] { color:#38BDF8 !important; border-bottom-color:#38BDF8 !important; }

/* status pill */
.pill-ok  { display:inline-block; background:#064E3B; color:#4ADE80; border:1px solid #4ADE80; border-radius:999px; padding:3px 14px; font-size:.8rem; font-weight:700; }
.pill-bad { display:inline-block; background:#450A0A; color:#F87171; border:1px solid #F87171; border-radius:999px; padding:3px 14px; font-size:.8rem; font-weight:700; }

/* divider */
hr { border-color: #1E293B; }

/* plotly charts transparent background */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HELPERS / THEME
# ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(color="#CBD5E1", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B"),
    yaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B"),
    margin=dict(t=50, b=40, l=40, r=20),
)
PALETTE = ["#38BDF8","#818CF8","#34D399","#FB923C","#F472B6"]

def styled_chart(fig, height=420):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
# DATA / ARTIFACT LOADERS
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    models = {
        "Random Forest": joblib.load("model_rf_tuned.joblib"),
        "XGBoost":       joblib.load("model_xgb_tuned.joblib"),
        "LightGBM":      joblib.load("model_lgb_tuned.joblib"),
        "KNN":           joblib.load("model_knn_tuned.joblib"),
        "MLP":           joblib.load("model_mlp_tuned.joblib"),
    }
    return models, joblib.load("scaler_final.joblib"), joblib.load("feature_names.joblib")

@st.cache_data(show_spinner="Loading test data…")
def load_test_data():
    df = pd.read_parquet("CUPID_final_test_scaled.parquet")
    return df.drop(columns=["Label"]), df["Label"]

models, scaler, feature_names = load_artifacts()
X_test, y_test = load_test_data()

# pre-compute all model preds once
@st.cache_data(show_spinner="Running model predictions…")
def compute_all_preds():
    out = {}
    for name, m in models.items():
        yp   = m.predict(X_test)
        prob = m.predict_proba(X_test)[:, 1]
        out[name] = dict(
            pred=yp, prob=prob,
            acc  = accuracy_score(y_test, yp),
            f1   = f1_score(y_test, yp, average="weighted"),
            prec = precision_score(y_test, yp, average="weighted", zero_division=0),
            rec  = recall_score(y_test, yp, average="weighted", zero_division=0),
            auc  = roc_auc_score(y_test, prob),
        )
    return out

all_preds = compute_all_preds()

# ──────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ CUPID NIDS")
    st.markdown("<p style='color:#64748B;font-size:.8rem;'>Network Intrusion Detection System</p>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Navigation", [
        "📊 Executive Summary",
        "🔬 Model Interrogation",
        "📈 Threat Forecasting",
        "🔍 Deep EDA",
        "⚡ Real-Time Engine",
        "📂 Batch Inspection",
    ])
    st.markdown("---")
    st.markdown(f"<p style='color:#475569;font-size:.75rem;'>Test samples: <b style='color:#94A3B8'>{len(X_test):,}</b><br>Features: <b style='color:#94A3B8'>{len(feature_names)}</b><br>Models: <b style='color:#94A3B8'>{len(models)}</b></p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# 1. EXECUTIVE SUMMARY
# ══════════════════════════════════════════════
if menu == "📊 Executive Summary":
    st.title("Security Operations Center — Overview")
    st.markdown("<p style='color:#64748B'>Live snapshot of traffic analysis and model readiness.</p>", unsafe_allow_html=True)

    threat_n   = int(y_test.sum())
    benign_n   = int(len(y_test) - threat_n)
    best_model = max(all_preds, key=lambda m: all_preds[m]["f1"])
    best_f1    = all_preds[best_model]["f1"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Traffic",    f"{len(X_test):,}")
    c2.metric("Threats Detected", f"{threat_n:,}",   delta=f"{threat_n/len(y_test)*100:.1f}% of traffic")
    c3.metric("Benign Traffic",   f"{benign_n:,}")
    c4.metric("Best Model F1",    f"{best_f1:.4f}",  delta=best_model)
    c5.metric("System Status",    "🟢 Operational")

    st.markdown("---")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("#### Traffic Composition")
        fig_pie = go.Figure(go.Pie(
            labels=["Benign", "Threat"],
            values=[benign_n, threat_n],
            hole=.55,
            marker=dict(colors=["#34D399","#F87171"]),
            textfont_size=13,
        ))
        fig_pie.add_annotation(text=f"<b>{threat_n/len(y_test)*100:.1f}%</b><br>Threat", x=.5, y=.5,
                               font=dict(size=16, color="#F8FAFC"), showarrow=False)
        styled_chart(fig_pie, height=320)

    with col_b:
        st.markdown("#### All-Model Performance at a Glance")
        rows = []
        for name, p in all_preds.items():
            rows.append({"Model": name, "Accuracy": p["acc"], "F1": p["f1"],
                         "Precision": p["prec"], "Recall": p["rec"], "ROC-AUC": p["auc"]})
        df_perf = pd.DataFrame(rows).set_index("Model")

        fig_bar = go.Figure()
        for i, col in enumerate(["Accuracy","F1","Precision","Recall","ROC-AUC"]):
            fig_bar.add_trace(go.Bar(name=col, x=df_perf.index, y=df_perf[col],
                                     marker_color=PALETTE[i]))
        fig_bar.update_layout(barmode="group", **PLOTLY_LAYOUT, height=320,
                              legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Threat Distribution over Test Window")
    chunk = st.slider("Chunk size (samples per bin)", 100, 2000, 500, 100)
    counts_raw = [int(y_test.iloc[i*chunk:(i+1)*chunk].sum()) for i in range(len(y_test)//chunk)]
    fig_line = px.area(x=list(range(len(counts_raw))), y=counts_raw,
                       labels={"x":"Window Index","y":"Threat Count"},
                       color_discrete_sequence=["#38BDF8"])
    fig_line.update_traces(fill="tozeroy", line_color="#38BDF8")
    styled_chart(fig_line, 280)


# ══════════════════════════════════════════════
# 2. MODEL INTERROGATION
# ══════════════════════════════════════════════
elif menu == "🔬 Model Interrogation":
    st.title("Model Performance Analysis")

    tab1, tab2, tab3 = st.tabs(["Single Model Deep-Dive", "ROC Curve Overlay", "Head-to-Head Compare"])

    # ── TAB 1: Single model ──────────────────────
    with tab1:
        selected_model = st.selectbox("Select Model", list(models.keys()), key="mi_single")
        p = all_preds[selected_model]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{p['acc']:.4f}")
        c2.metric("F1 (W)",    f"{p['f1']:.4f}")
        c3.metric("Precision", f"{p['prec']:.4f}")
        c4.metric("ROC-AUC",   f"{p['auc']:.4f}")

        col_cm, col_dist = st.columns([1,1])

        with col_cm:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, p["pred"])
            fig_cm = px.imshow(
                cm, text_auto=True, aspect="auto",
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual"),
                x=["Benign","Attack"], y=["Benign","Attack"],
            )
            fig_cm.update_traces(textfont_size=18)
            styled_chart(fig_cm, 360)

        with col_dist:
            st.markdown("#### Prediction Probability Distribution")
            fig_hist = go.Figure()
            for label, color, name in [(0,"#34D399","Benign"),(1,"#F87171","Attack")]:
                mask = (y_test == label)
                fig_hist.add_trace(go.Histogram(
                    x=p["prob"][mask], name=name, nbinsx=50,
                    marker_color=color, opacity=.75,
                ))
            fig_hist.update_layout(**PLOTLY_LAYOUT, height=360, barmode="overlay",
                                   xaxis_title="Predicted Probability", yaxis_title="Count",
                                   legend=dict(orientation="h"))
            st.plotly_chart(fig_hist, use_container_width=True)

    # ── TAB 2: ROC overlay ──────────────────────
    with tab2:
        st.markdown("#### ROC Curve — All Models")
        compare_models = st.multiselect("Models to display", list(models.keys()),
                                        default=list(models.keys()), key="mi_roc")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dash", color="#475569"), name="Random"))
        for i, name in enumerate(compare_models):
            fpr, tpr, _ = roc_curve(y_test, all_preds[name]["prob"])
            auc = all_preds[name]["auc"]
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"{name} (AUC={auc:.3f})",
                                         line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
        fig_roc.update_layout(**PLOTLY_LAYOUT, height=480,
                              xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── TAB 3: Head-to-head ─────────────────────
    with tab3:
        st.markdown("#### Side-by-Side Metric Radar")
        h2h_models = st.multiselect("Select models to compare", list(models.keys()),
                                    default=list(models.keys())[:3], key="mi_h2h")
        metrics_keys = ["acc","f1","prec","rec","auc"]
        metrics_labels = ["Accuracy","F1","Precision","Recall","ROC-AUC"]

        def hex_to_rgba(hex_color, alpha=0.15):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        fig_radar = go.Figure()
        for i, name in enumerate(h2h_models):
            vals = [all_preds[name][k] for k in metrics_keys]
            color = PALETTE[i % len(PALETTE)]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=metrics_labels + [metrics_labels[0]],
                fill="toself", name=name, opacity=.7,
                line=dict(color=color),
                fillcolor=hex_to_rgba(color, 0.15),
            ))
        fig_radar.update_layout(**PLOTLY_LAYOUT, height=460,
                                polar=dict(bgcolor="rgba(15,23,42,0.6)",
                                           radialaxis=dict(range=[0,1], gridcolor="#1E293B"),
                                           angularaxis=dict(gridcolor="#1E293B")),
                                legend=dict(orientation="h"))
        st.plotly_chart(fig_radar, use_container_width=True)

        # numeric table
        rows = {m: {l: all_preds[m][k] for k,l in zip(metrics_keys, metrics_labels)} for m in h2h_models}
        df_h2h = pd.DataFrame(rows).T
        st.dataframe(df_h2h.style.format("{:.4f}"), use_container_width=True)


# ══════════════════════════════════════════════
# 3. THREAT FORECASTING
# ══════════════════════════════════════════════
elif menu == "📈 Threat Forecasting":
    st.title("Threat Forecasting Analytics")
    st.markdown("<p style='color:#64748B'>Fit Exponential Smoothing on historic threat windows and project forward.</p>", unsafe_allow_html=True)

    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        chunk     = st.slider("Window size (samples)", 100, 2000, 500, 100)
        horizon   = st.slider("Forecast horizon (windows)", 5, 100, 24, 1)
        trend_opt = st.selectbox("Trend component", ["add", "mul", None])
        seas_opt  = st.selectbox("Seasonal component", [None, "add", "mul"])
        seas_per  = st.slider("Seasonal period", 2, 24, 6) if seas_opt else 2
        run_btn   = st.button("Run Forecast", type="primary")

    with col_main:
        counts_raw = [int(y_test.iloc[i*chunk:(i+1)*chunk].sum())
                      for i in range(len(y_test)//chunk)]

        if run_btn or "forecast_result" not in st.session_state:
            try:
                kwargs = dict(trend=trend_opt)
                if seas_opt:
                    kwargs["seasonal"]        = seas_opt
                    kwargs["seasonal_periods"] = seas_per
                model_es   = ExponentialSmoothing(counts_raw, **kwargs).fit()
                fc         = model_es.forecast(horizon)
                fitted_val = model_es.fittedvalues
                st.session_state["forecast_result"] = (counts_raw, fc, fitted_val)
            except Exception as e:
                st.error(f"Fitting failed: {e}")
                st.stop()

        counts_raw, fc, fitted_val = st.session_state["forecast_result"]
        hist_idx = list(range(len(counts_raw)))
        fc_idx   = list(range(len(counts_raw), len(counts_raw) + horizon))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_idx, y=counts_raw, name="Observed",
                                 line=dict(color="#38BDF8", width=2)))
        fig.add_trace(go.Scatter(x=hist_idx, y=fitted_val, name="ES Fitted",
                                 line=dict(color="#818CF8", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=fc_idx, y=fc, name="Forecast",
                                 line=dict(color="#FB923C", width=2.5)))
        # confidence band (±1 std of residuals)
        resid_std = np.std(np.array(counts_raw) - np.array(fitted_val))
        fig.add_trace(go.Scatter(
            x=fc_idx + fc_idx[::-1],
            y=list(fc + resid_std) + list((fc - resid_std)[::-1]),
            fill="toself", fillcolor="rgba(251,146,60,.15)",
            line=dict(color="rgba(0,0,0,0)"), name="±1 StdDev Band",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=460,
                          xaxis_title="Window Index", yaxis_title="Threat Count",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        # download forecast
        fc_arr = np.array(fc).flatten()
        fc_idx_arr = np.array(fc_idx)
        assert len(fc_arr) == len(fc_idx_arr), f"Length mismatch: fc={len(fc_arr)} idx={len(fc_idx_arr)}"
        fc_df = pd.DataFrame({"Window": fc_idx_arr, "Forecast": fc_arr,
                               "Lower": fc_arr - resid_std,
                               "Upper": fc_arr + resid_std})
        st.download_button("⬇ Download Forecast CSV", fc_df.to_csv(index=False).encode(),
                           "forecast.csv", "text/csv")


# ══════════════════════════════════════════════
# 4. DEEP EDA
# ══════════════════════════════════════════════
elif menu == "🔍 Deep EDA":
    st.title("Exploratory Data Analysis")

    df_eda = pd.concat([X_test.reset_index(drop=True),
                        y_test.reset_index(drop=True).rename("Label")], axis=1)
    df_eda["Label_str"] = df_eda["Label"].map({0: "Benign", 1: "Attack"})

    tab1, tab2, tab3 = st.tabs(["Distribution Explorer", "Correlation Heatmap", "Scatter Matrix"])

    # ── TAB 1 ────────────────────────────────────
    with tab1:
        col_ctrl, col_chart = st.columns([1, 3])
        with col_ctrl:
            feat      = st.selectbox("Feature", feature_names, key="eda_feat")
            plot_type = st.radio("Plot type", ["Violin","Box","Histogram"])
            sample_n  = st.slider("Subsample (rows)", 500, min(10000, len(df_eda)), 3000, 500)

        df_s = df_eda.sample(sample_n, random_state=42)
        with col_chart:
            if plot_type == "Violin":
                fig = px.violin(df_s, y=feat, x="Label_str", color="Label_str",
                                box=True, points="outliers",
                                color_discrete_map={"Benign":"#34D399","Attack":"#F87171"})
            elif plot_type == "Box":
                fig = px.box(df_s, y=feat, x="Label_str", color="Label_str",
                             color_discrete_map={"Benign":"#34D399","Attack":"#F87171"})
            else:
                fig = px.histogram(df_s, x=feat, color="Label_str", nbins=60, barmode="overlay",
                                   opacity=.7,
                                   color_discrete_map={"Benign":"#34D399","Attack":"#F87171"})
            fig.update_layout(showlegend=True, **PLOTLY_LAYOUT, height=420)
            st.plotly_chart(fig, use_container_width=True)

        # Quick stats table
        st.markdown("#### Descriptive Stats by Class")
        st.dataframe(df_eda.groupby("Label_str")[feat].describe().T.style.format("{:.4f}"),
                     use_container_width=True)

    # ── TAB 2: Correlation ───────────────────────
    with tab2:
        top_n = st.slider("Top N features by variance", 5, min(40, len(feature_names)), 15)
        top_feats = (X_test.var().nlargest(top_n).index.tolist())
        corr = df_eda[top_feats].corr()
        fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, aspect="auto",
                             title=f"Pearson Correlation — Top {top_n} Features")
        styled_chart(fig_corr, height=520)

    # ── TAB 3: Scatter matrix ────────────────────
    with tab3:
        scatter_feats = st.multiselect("Features for scatter matrix", feature_names,
                                       default=feature_names[:4])
        if len(scatter_feats) < 2:
            st.info("Select at least 2 features.")
        else:
            sample_sc = df_eda.sample(min(2000, len(df_eda)), random_state=1)
            fig_sp = px.scatter_matrix(
                sample_sc, dimensions=scatter_feats, color="Label_str",
                color_discrete_map={"Benign":"#34D399","Attack":"#F87171"},
                opacity=.5,
            )
            fig_sp.update_traces(marker=dict(size=3))
            styled_chart(fig_sp, height=600)


# ══════════════════════════════════════════════
# 5. REAL-TIME ENGINE
# ══════════════════════════════════════════════
elif menu == "⚡ Real-Time Engine":
    st.title("Real-Time Inference Engine")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        model_name = st.selectbox("Model", list(models.keys()), key="rt_model")

        btn_cols = st.columns(2)
        if btn_cols[0].button("🎲 Random Sample (Benign)"):
            idx = np.random.choice(np.where(y_test == 0)[0])
            st.session_state["rt_sample"] = X_test.iloc[idx].to_dict()
            st.session_state.pop("rt_result", None)
        if btn_cols[1].button("🎯 Random Sample (Attack)"):
            idx = np.random.choice(np.where(y_test == 1)[0])
            st.session_state["rt_sample"] = X_test.iloc[idx].to_dict()
            st.session_state.pop("rt_result", None)

        # Initialise default sample on first load
        if "rt_sample" not in st.session_state:
            st.session_state["rt_sample"] = {f: float(X_test[f].mean()) for f in feature_names}

        sample = st.session_state["rt_sample"]
        vals = {}
        for f in feature_names:
            fmin  = float(X_test[f].min())
            fmax  = float(X_test[f].max())
            fval  = float(sample.get(f, X_test[f].mean()))
            fval  = max(fmin, min(fmax, fval))   # clamp
            # number_input bisa diupdate programatically tanpa key conflict
            vals[f] = st.number_input(
                f, min_value=fmin, max_value=fmax,
                value=fval, step=(fmax - fmin) / 100 or 0.001,
                format="%.4f",
            )

    with col_right:
        if st.button("⚡ Execute Prediction", type="primary", key="rt_predict"):
            input_df     = pd.DataFrame([vals])[feature_names]
            input_scaled = scaler.transform(input_df)
            prob         = models[model_name].predict_proba(input_scaled)[0][1]
            is_attack    = prob > 0.5
            st.session_state["rt_result"] = (prob, is_attack, input_df)

        if "rt_result" in st.session_state:
            prob, is_attack, input_df = st.session_state["rt_result"]

            # Status badge
            if is_attack:
                st.markdown('<span class="pill-bad">🚨 MALICIOUS ATTACK DETECTED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="pill-ok">✅ TRAFFIC BENIGN</span>', unsafe_allow_html=True)

            st.markdown(f"<br>**Detection Confidence: {prob*100:.2f}%**", unsafe_allow_html=True)

            # Gauge
            gauge_color = "#F87171" if is_attack else "#34D399"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                delta={"reference": 50, "suffix":"%"},
                number={"suffix":"%", "font":{"color": gauge_color, "size":42}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar":  {"color": gauge_color, "thickness": .25},
                    "bgcolor": "#1E293B",
                    "steps": [
                        {"range":[0,50], "color":"#064E3B"},
                        {"range":[50,100], "color":"#450A0A"},
                    ],
                    "threshold": {"line":{"color":"white","width":3}, "thickness":.75, "value":50},
                },
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Feature impact bar — show top 15 by absolute deviation from mean
            st.markdown("#### Feature Deviation from Mean")
            means   = X_test[feature_names].mean()
            row_val = pd.Series(vals)[feature_names]
            deviations = ((row_val - means) / (X_test[feature_names].std() + 1e-9))
            top15  = deviations.abs().nlargest(15).index
            dev15  = deviations[top15]
            colors = ["#F87171" if v > 0 else "#34D399" for v in dev15.values]
            fig_bar = go.Figure(go.Bar(x=dev15.values, y=top15, orientation="h",
                                       marker_color=colors))
            bar_layout = {**PLOTLY_LAYOUT, "height": 320, "xaxis_title": "Z-score deviation"}
            bar_layout["yaxis"] = dict(autorange="reversed", gridcolor="#1E293B")
            fig_bar.update_layout(**bar_layout)
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════
# 6. BATCH INSPECTION
# ══════════════════════════════════════════════
elif menu == "📂 Batch Inspection":
    st.title("Batch Traffic Inspection")
    st.markdown("<p style='color:#64748B'>Upload a CSV traffic log. The system runs XGBoost by default; switch model as needed.</p>", unsafe_allow_html=True)

    col_up, col_opt = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader("Upload CSV Log", type=["csv"])
    with col_opt:
        batch_model = st.selectbox("Model", list(models.keys()), index=1)
        threshold   = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

    if uploaded:
        df_up = pd.read_csv(uploaded)
        missing = [f for f in feature_names if f not in df_up.columns]
        if missing:
            st.error(f"Missing {len(missing)} required features: {missing[:5]}…")
            st.stop()

        df_feat = df_up[feature_names]
        scaled  = scaler.transform(df_feat)
        probs   = models[batch_model].predict_proba(scaled)[:, 1]
        preds   = (probs >= threshold).astype(int)

        df_result = df_up.copy()
        df_result["Attack_Probability"] = probs.round(4)
        df_result["Prediction"]         = ["Attack" if p == 1 else "Normal" for p in preds]

        # Summary row
        n_atk = int(preds.sum())
        n_ok  = len(preds) - n_atk
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Records",    f"{len(preds):,}")
        s2.metric("Attacks Detected", f"{n_atk:,}",  delta=f"{n_atk/len(preds)*100:.1f}%")
        s3.metric("Normal Traffic",   f"{n_ok:,}")
        s4.metric("Threshold",        f"{threshold:.2f}")

        tab_tbl, tab_dist, tab_time = st.tabs(["Result Table", "Probability Distribution", "Attack Timeline"])

        with tab_tbl:
            filter_opt = st.radio("Show", ["All","Attack only","Normal only"], horizontal=True)
            if filter_opt == "Attack only":
                show_df = df_result[df_result["Prediction"] == "Attack"]
            elif filter_opt == "Normal only":
                show_df = df_result[df_result["Prediction"] == "Normal"]
            else:
                show_df = df_result

            st.dataframe(
                show_df[["Prediction","Attack_Probability"] + feature_names[:6]].reset_index(drop=True),
                use_container_width=True, height=380,
            )

            csv_bytes = df_result.to_csv(index=False).encode()
            st.download_button("⬇ Download Full Results", csv_bytes, "batch_results.csv", "text/csv")

        with tab_dist:
            fig_d = go.Figure()
            fig_d.add_trace(go.Histogram(x=probs[preds==0], name="Normal", nbinsx=50,
                                          marker_color="#34D399", opacity=.75))
            fig_d.add_trace(go.Histogram(x=probs[preds==1], name="Attack", nbinsx=50,
                                          marker_color="#F87171", opacity=.75))
            fig_d.add_vline(x=threshold, line_dash="dash", line_color="white",
                            annotation_text=f"Threshold={threshold}", annotation_position="top right")
            fig_d.update_layout(**PLOTLY_LAYOUT, height=380, barmode="overlay",
                                xaxis_title="Attack Probability", yaxis_title="Count",
                                legend=dict(orientation="h"))
            st.plotly_chart(fig_d, use_container_width=True)

        with tab_time:
            win = st.slider("Rolling window", 5, 200, 50)
            atk_series = pd.Series(preds).rolling(win).mean() * 100
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=list(range(len(atk_series))), y=atk_series,
                                        mode="lines", fill="tozeroy",
                                        line=dict(color="#F87171", width=2), name="% Attack"))
            fig_t.update_layout(**PLOTLY_LAYOUT, height=360,
                                xaxis_title="Record Index",
                                yaxis_title="Attack Rate (%)",
                                yaxis_range=[0, 105])
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Upload a CSV file with the required network traffic features to begin inspection.")
