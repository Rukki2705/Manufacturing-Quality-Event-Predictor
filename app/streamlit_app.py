import streamlit as st
st.set_page_config(page_title="SHAP Explanation Explorer", layout="wide")

import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import uuid
import os
from groq import Groq
import matplotlib.pyplot as plt


# --- Load model & preprocessor ---
@st.cache_resource
def load_artifacts():
    model_obj = joblib.load("models/best_model.pkl")

    if isinstance(model_obj, dict) and "model" in model_obj:
        model = model_obj["model"]
        model_type = model_obj.get("type", "Unknown")
    else:
        model = model_obj
        model_type = (
            "XGBoost" if "XGB" in str(type(model))
            else "RandomForest" if "RandomForest" in str(type(model))
            else "Unknown"
        )

    preprocessor = joblib.load("models/preprocessor_pipeline.pkl")
    return model, model_type, preprocessor

model, model_type, preprocessor = load_artifacts()

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("data/raw/noisy_manufacturing_quality_dataset.csv", parse_dates=["start_time", "end_time"])
    df['defect_rate'] = df['defect_count'] / df['batch_size']
    df['processing_speed'] = df['batch_size'] / df['processing_time_min']

    features = df.drop(columns=['batch_id', 'start_time', 'end_time', 'quality_event'])
    X = preprocessor.transform(features)

    feature_names = preprocessor.transformers_[0][2] + list(
        preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(preprocessor.transformers_[1][2])
    )
    X_df = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X, columns=feature_names)
    return df, X_df

df, X_df = load_and_prepare_data()
explainer = shap.TreeExplainer(model)

# --- UI ---
st.title("🔬 SHAP Explanation Explorer")
st.sidebar.header("🔍 Filter Batches")
st.sidebar.markdown(f"🧠 **Model Type:** `{model_type}`")

# Groq API key input
groq_api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password")

# Filters
shift_filter = st.sidebar.multiselect("Shift", df["shift"].unique(), default=list(df["shift"].unique()))
operator_filter = st.sidebar.multiselect("Operator ID", df["operator_id"].unique(), default=list(df["operator_id"].unique()))
product_filter = st.sidebar.multiselect("Product Type", df["product_type"].unique(), default=list(df["product_type"].unique()))

filtered_df = df[
    df["shift"].isin(shift_filter) &
    df["operator_id"].isin(operator_filter) &
    df["product_type"].isin(product_filter)
]

if filtered_df.empty:
    st.warning("No batches match your filter criteria.")
else:
    selected_batch = st.selectbox("Select a Batch ID", filtered_df["batch_id"].unique())
    selected_index = df[df["batch_id"] == selected_batch].index[0]
    row = X_df.iloc[[selected_index]]

    # --- What-If Panel ---
    st.sidebar.header("🔧 What-If Scenario")
    edited_features = {}
    for col in row.columns:
        val = row[col].values[0]
        if np.issubdtype(np.array([val]).dtype, np.number):
            edited_features[col] = st.sidebar.number_input(col, value=float(val))
        else:
            edited_features[col] = val
    edited_row = pd.DataFrame([edited_features])

    # --- Predict and SHAP for What-If
    whatif_pred = model.predict_proba(edited_row.values)[:, 1][0]
    shap_values = explainer.shap_values(edited_row.values)
    shap_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    base_val = explainer.expected_value[1] if isinstance(shap_values, list) else explainer.expected_value

    st.markdown(f"**🔄 What-If Predicted Risk Score:** `{whatif_pred:.3f}`")

    # --- Force Plot
    st.subheader("📈 Force Plot (What-If Input)")
    try:
        force_plot = shap.plots.force(base_val, shap_vals, feature_names=edited_row.columns)
        html_file = f"temp_shap_{uuid.uuid4().hex}.html"
        shap.save_html(html_file, force_plot)
        with open(html_file, "r", encoding="utf-8") as f:
            components.html(f.read(), height=400, scrolling=True)
    except Exception as e:
        st.error(f"❌ Failed to generate SHAP force plot: {e}")

    # --- Global SHAP Summary Plot ---
    with st.expander("🌍 Global SHAP Summary Plot"):
        sample_idx = X_df.sample(n=100, random_state=123).index
        shap_sample_values = explainer.shap_values(X_df.iloc[sample_idx].values)
        fig = plt.figure()
        shap.summary_plot(
        shap_sample_values[1] if isinstance(shap_sample_values, list) else shap_sample_values,
        X_df.iloc[sample_idx],
        max_display=10,
        show=False
        )
        st.pyplot(fig)

    # --- LLM Summary Report (Groq)
    st.subheader("📄 LLM-Based Batch Summary Report")
    if st.button("🧠 Generate Summary with LLaMA3"):
        if not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar.")
        else:
            top_features = edited_row.columns[np.argsort(np.abs(shap_vals))[-3:][::-1]].tolist()
            prompt = f"""
            Generate a manufacturing QA summary for batch ID {selected_batch}.
            Risk Score: {whatif_pred:.2f}
            Defect Rate: {df.loc[selected_index]['defect_rate']:.2%}
            Processing Speed: {df.loc[selected_index]['processing_speed']:.2f}
            Top Contributing Features: {', '.join(top_features)}.
            """

            try:
                client = Groq(api_key=groq_api_key)
                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are a manufacturing quality assurance expert."},
                        {"role": "user", "content": prompt}
                    ]
                )
                summary = response.choices[0].message.content
                st.success("✅ Summary generated:")
                st.markdown(summary)
                st.download_button("📥 Download Summary", summary, file_name=f"batch_{selected_batch}_summary.txt")
            except Exception as e:
                st.error(f"❌ Failed to generate summary: {e}")
