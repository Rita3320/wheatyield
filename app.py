import os
import json
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool


# ================= Page =================
st.set_page_config(page_title="Wheat Yield Predictor", layout="centered")
st.title("Wheat Yield Prediction (kg/hectare)")
st.markdown("This application uses monthly weather data to estimate wheat yield per hectare.")


# ================= Load model & features from disk only =================
DEFAULT_MODEL = "models/cat_model.cbm" if os.path.exists("models/cat_model.cbm") else "cat_model.cbm"
DEFAULT_FEATS  = "models/top_features.json" if os.path.exists("models/top_features.json") else "top_features.json"

model = None
top_features = None

def load_catboost_from_path(path: str) -> CatBoostRegressor:
    m = CatBoostRegressor()
    m.load_model(path)
    return m

# load model
if os.path.exists(DEFAULT_MODEL):
    try:
        model = load_catboost_from_path(DEFAULT_MODEL)
    except Exception as e:
        st.error(f"Failed to load model from '{DEFAULT_MODEL}': {e}")
        st.stop()
else:
    st.error(f"Model file not found: '{DEFAULT_MODEL}'. Place the CatBoost .cbm file at this path.")
    st.stop()

# load features
if os.path.exists(DEFAULT_FEATS):
    try:
        with open(DEFAULT_FEATS, "r", encoding="utf-8") as f:
            top_features = json.load(f)
        if not isinstance(top_features, list) or not all(isinstance(x, str) for x in top_features):
            raise ValueError("top_features.json must be a JSON array of strings.")
    except Exception as e:
        st.error(f"Failed to load feature list from '{DEFAULT_FEATS}': {e}")
        st.stop()
else:
    st.error(f"Feature list file not found: '{DEFAULT_FEATS}'. Place the JSON file at this path.")
    st.stop()


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
        "Sowing Phase":      [c for c in feats if has(c, "sowing")],
        "Overwinter Phase":  [c for c in feats if has(c, "overwinter")],
        "Jointing Phase":    [c for c in feats if has(c, "jointing")],
        "Heading Phase":     [c for c in feats if has(c, "heading")],
        "Filling Phase":     [c for c in feats if has(c, "filling")],
        "Dryness Indices":   [c for c in feats if has(c, "dryness")],
        "Extreme Indicators":[c for c in feats if any(k in c.lower() for k in ["gdd","hdd","cdd"])],
    }
    used = set(sum(groups.values(), []))
    groups["Other Features"] = [c for c in feats if c not in used]
    return {k: v for k, v in groups.items() if v}

def predict_with_model(model: CatBoostRegressor, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return model.predict(df[feats])

def validate_inputs(row: pd.Series) -> list[str]:
    """Simple sanity checks; add/adjust as you like."""
    warnings = []
    for k, v in row.items():
        if pd.isna(v):
            continue
        lk = k.lower()
        if "tmean" in lk and not (-30.0 <= v <= 45.0):
            warnings.append(f"{k} out of range (-30~45 °C): {v}")
        if "precip" in lk and not (0.0 <= v <= 500.0):
            warnings.append(f"{k} out of range (0~500 mm): {v}")
        if ("sun" in lk or "rad" in lk) and not (0.0 <= v <= 400.0):
            warnings.append(f"{k} out of range (0~400 hr): {v}")
        if ("wind" in lk or "ws" in lk) and not (0.0 <= v <= 30.0):
            warnings.append(f"{k} out of range (0~30 m/s): {v}")
        if "dryness" in lk and not (0.0 <= v <= 2.0):
            warnings.append(f"{k} out of range (0~2 ratio): {v}")
    return warnings


# ================= Display candidates for manual UI =================
# ensure precipitation / wind / sunshine inputs appear even if not in top_features
DISPLAY_CANDIDATES = [
    "sowingTmeanAvg","sowingPrecipSum","sowingSunHours","sowingWindAvg",
    "overwinterTmeanAvg","overwinterPrecipSum","overwinterSunHours","overwinterWindAvg",
    "jointingTmeanAvg","jointingPrecipSum","jointingSunHours","jointingWindAvg",
    "headingTmeanAvg","headingPrecipSum","headingSunHours","headingWindAvg",
    "fillingTmeanAvg","fillingPrecipSum","fillingSunHours","fillingWindAvg",
    "drynessJointing","drynessHeading","drynessFilling",
    "gddBase5","hddGt30","cddLt0",
]

# union for UI display (keep order and unique)
display_features = list(dict.fromkeys(DISPLAY_CANDIDATES + top_features))


# ================= Tabs =================
tab_batch, tab_manual, tab_insight = st.tabs(["Batch Prediction", "Manual Weather Input", "Insights & Scenarios"])

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
        st.rerun()

    groups = group_features(display_features)
    user_input = {}
    rendered = set()  # avoid duplicate widgets if a feature appears in multiple groups

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

                # Only key + ranges; DO NOT pass value= (we rely on session_state if any)
                kwargs = {"key": key}
                if "tmean" in lc:
                    kwargs.update(min_value=-30.0, max_value=45.0, step=0.1, format="%.1f")
                elif "precip" in lc:
                    kwargs.update(min_value=0.0, max_value=500.0, step=0.1, format="%.1f")
                elif ("sun" in lc) or ("rad" in lc):
                    kwargs.update(min_value=0.0, max_value=400.0, step=0.1, format="%.1f")
                elif ("wind" in lc) or ("ws" in lc):
                    kwargs.update(min_value=0.0, max_value=30.0, step=0.1, format="%.1f")
                elif "dryness" in lc:
                    kwargs.update(min_value=0.0, max_value=2.0, step=0.01, format="%.2f")

                val = st.number_input(label, **kwargs)
                user_input[c] = val

    # Build input strictly with model features
    input_df = pd.DataFrame([{k: user_input.get(k, None) for k in top_features}], columns=top_features)

    # Input validation (before fill)
    warnings = validate_inputs(input_df.iloc[0])
    if warnings:
        st.warning("Input warnings:\n- " + "\n- ".join(warnings))

    # row-wise median fill
    row_median = input_df.iloc[0].dropna().median()
    input_df_filled = input_df.fillna(row_median)

    st.markdown("Final Input Used For Prediction (after filling missing values):")
    st.dataframe(input_df_filled)

    if st.button("Predict Yield"):
        with st.spinner("Running prediction..."):
            yhat = predict_with_model(model, input_df_filled, top_features)[0]
        st.success(f"Predicted Yield per Hectare: {yhat:.2f} tons")
        st.session_state["last_prediction"] = float(yhat)
        st.session_state["last_input_row"] = input_df_filled.iloc[0].to_dict()

# ----- Insights & Scenarios -----
with tab_insight:
    st.subheader("Insights and Scenarios")

    # 1) Save / Load current inputs as JSON profile
    with st.expander("Profiles"):
        # Download current inputs
        if "last_input_row" in st.session_state:
            current_inputs = st.session_state["last_input_row"]
        else:
            # build from current session_state keys if available
            current_inputs = {c: st.session_state.get(f"in_{c}", None) for c in display_features}

        profile_json = json.dumps(current_inputs, ensure_ascii=False, indent=2)
        st.download_button("Download current inputs as JSON", profile_json.encode("utf-8"),
                           file_name="wheat_inputs_profile.json")

        # Upload and prefill
        up_profile = st.file_uploader("Load inputs profile (.json)", type=["json"], key="profile_uploader")
        if up_profile is not None:
            try:
                prof = json.load(up_profile)
                # write to session_state and rerun
                for k, v in prof.items():
                    st.session_state[f"in_{k}"] = v
                st.success("Profile loaded into inputs.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load profile: {e}")

    # 2) Scenario comparison table
    with st.expander("Scenario comparison"):
        if "scenarios" not in st.session_state:
            st.session_state["scenarios"] = []

        scenario_name = st.text_input("Scenario name", value="Scenario 1")
        if st.button("Add current inputs to Scenario List"):
            # Build input strictly for model features
            row = {k: st.session_state.get(f"in_{k}", None) for k in top_features}
            row_df = pd.DataFrame([row], columns=top_features)
            # fill
            row_filled = row_df.fillna(row_df.iloc[0].dropna().median())
            # predict
            pred = float(predict_with_model(model, row_filled, top_features)[0])
            rec = {"scenario": scenario_name, "predicted_yield": pred, **row_filled.iloc[0].to_dict()}
            st.session_state["scenarios"].append(rec)

        if st.session_state["scenarios"]:
            sc_df = pd.DataFrame(st.session_state["scenarios"])
            st.dataframe(sc_df)
            st.download_button("Download scenarios as CSV",
                               sc_df.to_csv(index=False).encode("utf-8"),
                               file_name="scenarios.csv")
            if st.button("Clear scenarios"):
                st.session_state["scenarios"] = []
                st.experimental_rerun()

    # 3) SHAP explanation for the latest prediction
    with st.expander("Explain current prediction (SHAP)"):
        if "last_input_row" not in st.session_state:
            st.info("Run a prediction in the Manual tab first.")
        else:
            row = pd.DataFrame([st.session_state["last_input_row"]], columns=top_features)
            try:
                pool = Pool(row[top_features])
                shap_vals = model.get_feature_importance(type="ShapValues", data=pool)
                # shap_vals shape: (1, n_features + 1); last col is expected value (bias)
                contrib = shap_vals[0, :-1]
                order = np.argsort(np.abs(contrib))[::-1][:10]
                feats = [top_features[i] for i in order][::-1]
                vals  = [contrib[i] for i in order][::-1]

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(feats, vals)
                ax.set_xlabel("SHAP contribution")
                ax.set_ylabel("Feature")
                ax.set_title("Top 10 contributing features")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Failed to compute SHAP values: {e}")
