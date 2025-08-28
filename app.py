import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from catboost import CatBoostRegressor

# ---------------- 页面设置 ----------------
st.set_page_config(page_title="Wheat Yield Predictor", layout="centered")
st.title("Wheat Yield Prediction (kg/hectare)")
st.markdown("This application uses monthly weather data to estimate wheat yield per hectare.")

# ---------------- 侧边栏加载模型 ----------------
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

# ---------------- 主体功能 ----------------
if model is not None and top_features is not None:

    tab1, tab2 = st.tabs(["Batch Prediction", "Manual Weather Input"])

    # ======== 批量预测 ========
    with tab1:
        st.subheader("Upload CSV File for Batch Prediction")
        st.markdown("CSV must contain the following columns:")
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

    # ======== 手动输入预测 ========
    with tab2:
        st.subheader("Manually Input Monthly Weather Data")

        demo_examples = {
            "Example: Normal Year (Benchmark)": {
                "sowingTmeanAvg": 12.3, "sowingPrecipSum": 45.0,
                "overwinterTmeanAvg": 2.1, "overwinterPrecipSum": 18.0,
                "jointingTmeanAvg": 15.7, "jointingPrecipSum": 38.0,
                "headingTmeanAvg": 18.9, "headingPrecipSum": 42.0,
                "fillingTmeanAvg": 22.4, "fillingPrecipSum": 65.0,
                "drynessJointing": 0.4, "drynessHeading": 0.5, "drynessFilling": 0.2,
                "gddBase5": 1600, "hddGt30": 3, "cddLt0": 20
            },
            "Example: Cold Dry Winter": {
                "sowingTmeanAvg": 10.0, "sowingPrecipSum": 20.0,
                "overwinterTmeanAvg": -3.0, "overwinterPrecipSum": 5.0,
                "jointingTmeanAvg": 14.0, "jointingPrecipSum": 25.0,
                "headingTmeanAvg": 17.0, "headingPrecipSum": 30.0,
                "fillingTmeanAvg": 21.0, "fillingPrecipSum": 55.0,
                "drynessJointing": 0.6, "drynessHeading": 0.7, "drynessFilling": 0.5,
                "gddBase5": 1400, "hddGt30": 1, "cddLt0": 45
            }
        }

        selected_demo = st.selectbox("Load example data:", ["Manual Entry"] + list(demo_examples.keys()))
        if selected_demo != "Manual Entry":
            st.info(f"Loaded values from: {selected_demo}")
            user_input = demo_examples[selected_demo].copy()
        else:
            user_input = {}

        # 特征分组
        phase_groups = {
            "Sowing Phase": [k for k in top_features if "sowing" in k],
            "Overwinter Phase": [k for k in top_features if "overwinter" in k],
            "Jointing Phase": [k for k in top_features if "jointing" in k],
            "Heading Phase": [k for k in top_features if "heading" in k],
            "Filling Phase": [k for k in top_features if "filling" in k],
            "Dryness Indices": [k for k in top_features if "dryness" in k],
            "Extreme Indicators": [k for k in top_features if any(s in k for s in ["gdd", "hdd", "cdd"])],
            "Other Features": [k for k in top_features if k not in sum([
                [*v] for v in [
                    [k for k in top_features if "sowing" in k],
                    [k for k in top_features if "overwinter" in k],
                    [k for k in top_features if "jointing" in k],
                    [k for k in top_features if "heading" in k],
                    [k for k in top_features if "filling" in k],
                    [k for k in top_features if "dryness" in k],
                    [k for k in top_features if any(s in k for s in ["gdd", "hdd", "cdd"])]
                ]
            ], [])]
        }

        def get_unit_label(col):
            if "tmean" in col: return f"{col} [°C]"
            if "precip" in col: return f"{col} [mm]"
            if "sun" in col: return f"{col} [hr]"
            if "wind" in col: return f"{col} [m/s]"
            if "gdd" in col or "hdd" in col or "cdd" in col: return f"{col} [°C-days]"
            if "dryness" in col: return f"{col} [ratio]"
            return col

        for group, cols in phase_groups.items():
            if not cols:
                continue
            st.markdown(f"**{group}**")
            col_widgets = st.columns(3)
            for i, col in enumerate(cols):
                with col_widgets[i % 3]:
                    label = get_unit_label(col)
                    default = user_input.get(col, None)
                    val = st.number_input(label, value=default)
                    user_input[col] = val

        input_df = pd.DataFrame([user_input])
        row_median = input_df.iloc[0].dropna().median()
        input_df = input_df.fillna(row_median)

        st.markdown("Final Input (after filling missing values):")
        st.dataframe(input_df)

        if st.button("Predict Yield"):
            with st.spinner("Running prediction..."):
                pred = model.predict(input_df[top_features])[0]
            st.success(f"Predicted Wheat Yield per Hectare: {pred:.2f} tons")

else:
    st.info("Please upload both the model file and the feature list to begin.")
