import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load trained model and top feature list
model = joblib.load("model_rf.pkl")
top_features = joblib.load("top_features.pkl")  # list of features used

st.set_page_config(page_title="Wheat Yield Predictor", layout="centered")
st.title("Wheat Yield Prediction (kg/hectare)")
st.markdown("Use monthly weather inputs to estimate wheat yield per hectare.")

# Select input mode
input_mode = st.radio("Select Input Method", [
    "Upload batch weather data (CSV)",
    "Manually input weather data"
], horizontal=True)

if input_mode == "Upload batch weather data (CSV)":
    st.markdown("**CSV must contain the following columns:**")
    st.code(", ".join(top_features))

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("**Uploaded Preview:**")
        st.dataframe(df.head())

        missing_cols = [c for c in top_features if c not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            pred_df = df.copy()
            row_medians = pred_df[top_features].median(axis=1)
            pred_df[top_features] = pred_df[top_features].T.fillna(row_medians).T
            preds = model.predict(pred_df[top_features])
            pred_df["predicted_yield"] = preds
            st.markdown("**Predictions:**")
            st.dataframe(pred_df[["predicted_yield"] + top_features])

            csv_out = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Result CSV", csv_out, file_name="predicted_yield.csv")

elif input_mode == "Manually input weather data":
    st.markdown("**Enter monthly weather values below (leave blank if unknown):**")

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

    st.markdown("**Final Input:**")
    st.dataframe(input_df)

    if st.button("Predict"):
        pred = model.predict(input_df[top_features])[0]
        st.success(f"Predicted Yield per Hectare: {pred:.2f} tons")
