import os
import json
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor


# ================= Page =================
st.set_page_config(page_title="Wheat Yield Predictor", layout="centered")
st.title("Wheat Yield Prediction (kg/hectare)")
st.markdown("This application uses monthly weather data to estimate wheat yield per hectare.")


# ================= Sidebar: load model & features =================
with st.sidebar:
    st.header("Model and Feature Loader")
    up_model = st.file_uploader("Upload CatBoost model (.cbm)", type=["cbm"])
    up_feats = st.file_uploader("Upload top feature list (.json)", type=["json"])

DEFAULT_MODEL = "models/cat_model.cbm" if os.path.exists("models/cat_model.cbm") else "cat_model.cbm"
DEFAULT_FEATS = "models/top_features.json" if os.path.exists("models/top_features.json") else "top_features.json"

model = None
top_features = None

def load_catboost_from_filelike(filelike) -> CatBoostRegressor:
    """CatBoost loads from a filepath; write the uploaded bytes to a temp file first."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cbm") as tmp:
        tmp.write(filelike.read())
        tmp_path = tmp.name
    m = CatBoostRegressor()
    m.load_model(tmp_path)
    os.remove(tmp_path)
    return m

# model
try:
    if up_model is not None:
        model = load_catboost_from_filelike(up_model)
    elif os.path.exists(DEFAULT_MODEL):
        model = CatBoostRegressor()
        model.load_model(DEFAULT_MODEL)
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    model = None

# features
try:
    if up_feats is not None:
        top_features = json.load(up_feats)
    elif os.path.exists(DEFAULT_FEATS):
        with open(DEFAULT_FEATS, "r", encoding="utf-8") as f:
            top_features = json.load(f)
except Exception as e:
    st.sidebar.error(f"Failed to load feature list: {e}")
    top_features = None

if model is not None and top_features is not None:
    st.sidebar.success(f"Model and feature list loaded. Features: {len(top_features)}")
else:
    st.info("Please upload both the model file (.cbm) and the feature list (.json), or place them under ./models/ to begin.")


# ================= Helpers =================
def label_with_unit(col: str) -> str:
    """English label = original feature name + unit only."""
    lc = col.lower()
    if "tmean" in lc:   unit = "[°C]"
    elif "precip" in lc: unit = "[mm]"
    elif ("sun" in lc) or ("rad" in lc): unit = "[hr]"
    elif ("wind" in lc) or ("ws" in lc): unit = "[m/s]"
    elif any(k in lc for k in ["gdd", "hdd", "cdd"]): unit = "[°C-days]"
    elif "dryness" in lc: unit = "[ratio]"
    else: unit = ""
    return f"{col} {unit}".strip()

def group_features(feats: list[str]) -> dict[str, list[str]]:
    """Group by lifecycle keywords for display only."""
    def has(s, key): return key in s.lower()
    groups = {
        "Sowing Phase":     [c for c in feats if has(c, "sowing")],
        "Overwinter Phase": [c for c in feats if has(c, "overwinter")],
        "Jointing Phase":   [c for c in feats if has(c, "jointing")],
        "Heading Phase":    [c for c in feats if has(c, "heading")],
        "Filling Phase":    [c for c in feats if has(c, "filling")],
        "Dryness Indices":  [c for c in feats if has(c, "dryness")],
        "Extreme Indicators":[c for c in feats if any(k in c.lower() for k in ["gdd","hdd","cdd"])],
    }
    used = set(sum(groups.values(), []))
    groups["Other Features"] = [c for c in feats if c not in used]
    # drop empty groups
    return {k: v for k, v in groups.items() if v}

def predict_with_model(model: CatBoostRegressor, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return model.predict(df[feats])


# ================= Display candidates for manual UI =================
# Ensure precipitation / wind / sunshine inputs show up even if not in top_features
DISPLAY_CANDIDATES = [
    "sowingTmeanAvg","sowingPrecipSum","sowingSunHours","sowingWindAvg",
    "overwinterTmeanAvg","overwinterPrecipSum","overwinterSunHours","overwinterWindAvg",
    "jointingTmeanAvg","jointingPrecipSum","jointingSunHours","jointingWindAvg",
    "headingTmeanAvg","headingPrecipSum","headingSunHours","headingWindAvg",
    "fillingTmeanAvg","fillingPrecipSum","fillingSunHours","fillingWindAvg",
    "drynessJointing","drynessHeading","drynessFilling",
    "gddBase5","hddGt30","cddLt0",
]


# ================= Main =================
if model is not None and top_features is not None:

    # union set for UI display (keep order, de-dup)
    display_features = list(dict.fromkeys(DISPLAY_CANDIDATES + top_features))

    tab_batch, tab_manual = st.tabs(["Batch Prediction", "Manual Weather Input"])

    # ----- Batch -----
    with tab_batch:
        st.subheader("Upload CSV File for Batch Prediction")
        st.markdown("The CSV must contain the following columns (model features):")
        st.code(", ".join(top_features))

        up_csv = st.file_uploader("Upload weather data CSV", type=["csv"])
        if up_csv is not None:
            df = pd.read_csv(up_csv)
            st.markdown("Uploaded Preview:")
            st.dataframe(df.head())

            missing = [c for c in top_features if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                with st.spinner("Running prediction..."):
                    pred_df = df.copy()
                    # row-wise median fill
                    row_medians = pred_df[top_features].median(axis=1)
                    pred_df[top_features] = pred_df[top_features].T.fillna(row_medians).T
                    preds = predict_with_model(model, pred_df, top_features)
                    pred_df["predicted_yield"] = preds

                st.success("Prediction complete.")
                st.markdown("Prediction Results:")
                st.dataframe(pred_df[["predicted_yield"] + top_features])

                csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results as CSV", csv_bytes, file_name="predicted_yield.csv")

    # ----- Manual -----
    with tab_manual:
        st.subheader("Manually Enter Weather Data")

        # demo presets
        demos = {
            "Normal Year (Benchmark)": {
                "sowingTmeanAvg": 12.3, "sowingPrecipSum": 45.0, "sowingSunHours": 160.0, "sowingWindAvg": 2.5,
                "overwinterTmeanAvg": 2.1, "overwinterPrecipSum": 18.0, "overwinterSunHours": 110.0, "overwinterWindAvg": 2.0,
                "jointingTmeanAvg": 15.7, "jointingPrecipSum": 38.0, "jointingSunHours": 180.0, "jointingWindAvg": 2.8,
                "headingTmeanAvg": 18.9, "headingPrecipSum": 42.0, "headingSunHours": 200.0, "headingWindAvg": 3.0,
                "fillingTmeanAvg": 22.4, "fillingPrecipSum": 65.0, "fillingSunHours": 220.0, "fillingWindAvg": 3.2,
                "drynessJointing": 0.4, "drynessHeading": 0.5, "drynessFilling": 0.2,
                "gddBase5": 1600, "hddGt30": 3, "cddLt0": 20
            },
            "Cold Dry Winter": {
                "sowingTmeanAvg": 10.0, "sowingPrecipSum": 20.0, "sowingSunHours": 150.0, "sowingWindAvg": 2.0,
                "overwinterTmeanAvg": -3.0, "overwinterPrecipSum": 5.0, "overwinterSunHours": 90.0, "overwinterWindAvg": 1.8,
                "jointingTmeanAvg": 14.0, "jointingPrecipSum": 25.0, "jointingSunHours": 170.0, "jointingWindAvg": 2.4,
                "headingTmeanAvg": 17.0, "headingPrecipSum": 30.0, "headingSunHours": 185.0, "headingWindAvg": 2.6,
                "fillingTmeanAvg": 21.0, "fillingPrecipSum": 55.0, "fillingSunHours": 210.0, "fillingWindAvg": 2.7,
                "drynessJointing": 0.6, "drynessHeading": 0.7, "drynessFilling": 0.5,
                "gddBase5": 1400, "hddGt30": 1, "cddLt0": 45
            },
            "Hot Wet Late Season": {
                "sowingTmeanAvg": 13.0, "sowingPrecipSum": 50.0, "sowingSunHours": 165.0, "sowingWindAvg": 2.3,
                "overwinterTmeanAvg": 1.5, "overwinterPrecipSum": 22.0, "overwinterSunHours": 115.0, "overwinterWindAvg": 2.1,
                "jointingTmeanAvg": 17.0, "jointingPrecipSum": 45.0, "jointingSunHours": 190.0, "jointingWindAvg": 2.9,
                "headingTmeanAvg": 20.5, "headingPrecipSum": 60.0, "headingSunHours": 210.0, "headingWindAvg": 3.1,
                "fillingTmeanAvg": 25.0, "fillingPrecipSum": 110.0, "fillingSunHours": 235.0, "fillingWindAvg": 3.4,
                "drynessJointing": 0.3, "drynessHeading": 0.35, "drynessFilling": 0.15,
                "gddBase5": 1750, "hddGt30": 8, "cddLt0": 8
            }
        }

        demo_name = st.selectbox("Load example data", ["Manual Entry"] + list(demos.keys()), key="demo_select")
        if demo_name != "Manual Entry":
            for k, v in demos[demo_name].items():
                st.session_state[f"in_{k}"] = v

        groups = group_features(display_features)
        user_input = {}
        rendered = set()  # avoid duplicate widgets

        for gname, cols in groups.items():
            st.markdown(f"**{gname}**")
            cols_container = st.columns(3)

            for i, c in enumerate(cols):
                if c in rendered:
                    continue
                rendered.add(c)

                with cols_container[i % 3]:
                    label = label_with_unit(c)
                    key = f"in_{c}"
                    lc = c.lower()

                    # build kwargs, only pass "value" if not None to avoid Streamlit warnings
                    kwargs = {"key": key}
                    base = st.session_state.get(key, None)
                    if base is not None:
                        kwargs["value"] = base

                    if "tmean" in lc:
                        kwargs["min_value"] = -30.0; kwargs["max_value"] = 45.0; kwargs["step"] = 0.1; kwargs["format"] = "%.1f"
                    elif "precip" in lc:
                        kwargs["min_value"] = 0.0; kwargs["max_value"] = 500.0; kwargs["step"] = 0.1; kwargs["format"] = "%.1f"
                    elif ("sun" in lc) or ("rad" in lc):
                        kwargs["min_value"] = 0.0; kwargs["max_value"] = 400.0; kwargs["step"] = 0.1; kwargs["format"] = "%.1f"
                    elif ("wind" in lc) or ("ws" in lc):
                        kwargs["min_value"] = 0.0; kwargs["max_value"] = 30.0; kwargs["step"] = 0.1; kwargs["format"] = "%.1f"
                    elif "dryness" in lc:
                        kwargs["min_value"] = 0.0; kwargs["max_value"] = 2.0; kwargs["step"] = 0.01; kwargs["format"] = "%.2f"

                    val = st.number_input(label, **kwargs)
                    user_input[c] = val

        # Build input strictly with model features
        input_df = pd.DataFrame([{k: user_input.get(k, None) for k in top_features}], columns=top_features)
        # row-wise median fill
        row_median = input_df.iloc[0].dropna().median()
        input_df = input_df.fillna(row_median)

        st.markdown("Final Input Used For Prediction (after filling missing values):")
        st.dataframe(input_df)

        if st.button("Predict Yield"):
            with st.spinner("Running prediction..."):
                yhat = predict_with_model(model, input_df, top_features)[0]
            st.success(f"Predicted Yield per Hectare: {yhat:.2f} tons")
