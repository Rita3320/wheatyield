import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import json
from catboost import CatBoostRegressor

# ----------------- 页面设置 -----------------
st.set_page_config(page_title="Wheat Yield Predictor", layout="centered")
st.title("Wheat Yield Prediction (kg/hectare)")
st.markdown("This application uses monthly weather inputs to estimate wheat yield per hectare.")

# ----------------- 模型与特征加载 -----------------
with st.sidebar:
    st.header("Model and Feature Loader")
    model_file = st.file_uploader("Upload trained CatBoost model (.pkl)", type=["pkl"])
    features_file = st.file_uploader("Upload top feature list (.pkl or .json)", type=["pkl", "json"])

model = None
top_features = None

if model_file is not None:
    try:
        model = joblib.load(model_file)
        st.sidebar.success("Model loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

if features_file is not None:
    try:
        if features_file.name.endswith(".json"):
            top_features = json.load(features_file)
        else:
            top_features = joblib.load(features_file)
        st.sidebar.success("Feature list loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Failed to load feature list: {e}")

# ----------------- 主功能入口 -----------------
if model is not None and top_features is not None:

    tab1, tab2 = st.tabs(["Batch Prediction", "Manual Input"])

    # ========== 批量预测 ==========
    with tab1:
        st.subheader("Upload CSV File")
        st.markdown("The CSV must contain the following columns:")
        st.code(", ".join(top_features))

        uploaded_file = st.file_uploader("Upload your weather data CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.markdown("Uploaded Preview:")
            st.dataframe(df.head())

            missing_cols = [c for c in top_features if c not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                with st.spinner("Running prediction..."):
                    pred_df = df.copy()
                    row_medians = pred_df[top_features].median(axis=1)
                    pred_df[top_features] = pred_df[top_features].T.fillna(row_medians).T
                    preds = model.predict(pred_df[top_features])
                    pred_df["predicted_yield"] = preds

                st.success("Prediction complete.")
                st.markdown("Prediction Results:")
                st.dataframe(pred_df[["predicted_yield"] + top_features])

                csv_out = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results as CSV", csv_out, file_name="predicted_yield.csv")

    # ========== 手动预测 ==========
    with tab2:
        st.subheader("Enter Weather Data Manually")
        st.markdown("Fill in the monthly weather data. You may leave fields blank.")

        user_input = {}
        for col in top_features:
            if "tmean" in col:
                val = st.number_input(col, value=None, min_value=-30.0, max_value=40.0, step=0.1, format="%.1f")
            elif "precip" in col:
                val = st.number_input(col, value=None, min_value=0.0, max_value=300.0, step=0.1, format="%.1f")
            elif "sun" in col:
                val = st.number_input(col, value=None, min_value=0.0, max_value=400.0, step=0.1, format="%.1f")
            elif "wind" in col:
                val = st.number_input(col, value=None, min_value=0.0, max_value=20.0, step=0.1, format="%.1f")
            else:
                val = st.number_input(col, value=None)
            user_input[col] = val

        input_df = pd.DataFrame([user_input])
        row_median = input_df.iloc[0].dropna().median()
        input_df = input_df.fillna(row_median)

        st.markdown("Input Data Preview:")
        st.dataframe(input_df)

        if st.button("Predict Yield"):
            with st.spinner("Running prediction..."):
                pred = model.predict(input_df[top_features])[0]
            st.success(f"Predicted Yield per Hectare: {pred:.2f} tons")

else:
    st.info("Please upload both the model file and the feature list to begin.")
