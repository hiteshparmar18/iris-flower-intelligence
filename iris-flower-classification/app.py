import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from model import train_and_save_models, load_models

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Intelligence",
    page_icon="🌸",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# IMAGE MAP
# --------------------------------------------------
FLOWER_IMAGES = {
    "setosa": "images/setosa.jpg",
    "versicolor": "images/versicolor.jpg",
    "virginica": "images/virginica.jpg"
}

# --------------------------------------------------
# GLOBAL CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    color: white;
}

.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 20px;
}

img {
    border-radius: 14px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}

div.stButton > button {
    background: linear-gradient(135deg,#7F7CFF,#B47CFF);
    color:white;
    border-radius:14px;
    padding:12px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
accuracies, target_names, feature_names = train_and_save_models()
models, X_test, y_test = load_models()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div style="text-align:center; margin-bottom:30px;">
<h1 style="font-size:44px;">🌸 Iris Flower Intelligence</h1>
<p style="opacity:0.85;">
Production-grade ML dashboard with analytics & explainability
</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR (CONTROL DOCK)
# --------------------------------------------------
st.sidebar.markdown("## 🎛 Control Dock")

model_name = st.sidebar.selectbox("Model", list(models.keys()))
st.sidebar.metric("Accuracy", f"{accuracies[model_name]*100:.2f}%")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

predict = st.sidebar.button("✨ Analyze Flower", use_container_width=True)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab_overview, tab_analytics, tab_explain, tab_gallery = st.tabs(
    ["🏠 Overview", "📊 Analytics", "🧠 Explainability", "📸 Gallery"]
)

# --------------------------------------------------
# PREDICTION LOGIC
# --------------------------------------------------
if predict:
    model = models[model_name]

    sample_np = np.array([[ 
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    pred = model.predict(sample_np)[0]
    probs = model.predict_proba(sample_np)[0]
    confidence = np.max(probs) * 100
    species = target_names[pred]

    # Save history
    st.session_state.history.append({
        "Model": model_name,
        "Prediction": species,
        "Confidence (%)": round(confidence, 2)
    })

# ==================================================
# TAB 1 — OVERVIEW
# ==================================================
with tab_overview:
    if predict:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.metric("Model Used", model_name)
            st.metric("Predicted Species", species)
            st.metric("Confidence", f"{confidence:.2f}%")

        with col2:
            st.image(
                FLOWER_IMAGES[species],
                caption=f"Iris {species.capitalize()}",
                use_container_width=True
            )
    else:
        st.info("Adjust inputs and click **Analyze Flower** to see prediction.")

# ==================================================
# TAB 2 — ANALYTICS
# ==================================================
with tab_analytics:
    if predict:
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📈 Prediction Probabilities")

            prob_df = pd.DataFrame({
                "Species": target_names,
                "Probability": probs
            })
            st.bar_chart(prob_df.set_index("Species"), height=260)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🧮 Confusion Matrix")

            cm = confusion_matrix(y_test, model.predict(X_test))
            cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
            st.dataframe(cm_df, use_container_width=True, height=260)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Run a prediction to view analytics.")

# ==================================================
# TAB 3 — EXPLAINABILITY (NO SHAP)
# ==================================================
with tab_explain:
    if predict:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Feature Importance (Model Coefficients)")
        st.caption("Higher value → stronger influence on prediction")

        if model_name == "Logistic Regression":
            coef = np.abs(model.coef_).mean(axis=0)

            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": coef
            }).sort_values("Importance", ascending=False)

            st.bar_chart(fi_df.set_index("Feature"), height=280)
        else:
            st.info("Feature importance available for Logistic Regression only.")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Run a prediction to see explainability.")

# ==================================================
# TAB 4 — GALLERY
# ==================================================
with tab_gallery:
    st.markdown("### 🌼 Iris Species Gallery")

    g1, g2, g3 = st.columns(3)
    g1.image(FLOWER_IMAGES["setosa"], caption="Iris Setosa", use_container_width=True)
    g2.image(FLOWER_IMAGES["versicolor"], caption="Iris Versicolor", use_container_width=True)
    g3.image(FLOWER_IMAGES["virginica"], caption="Iris Virginica", use_container_width=True)

# --------------------------------------------------
# HISTORY SECTION (GLOBAL)
# --------------------------------------------------
if st.session_state.history:
    st.markdown("## 🕘 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

    st.download_button(
        "⬇️ Export History as CSV",
        hist_df.to_csv(index=False).encode("utf-8"),
        "prediction_history.csv",
        "text/csv"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<hr>
<p style="text-align:center; opacity:0.6;">
Built as a modern ML dashboard using Streamlit & Scikit-learn
</p>
""", unsafe_allow_html=True)
