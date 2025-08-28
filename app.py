import os
import json
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor


# ============ 页面设置 ============
st.set_page_config(page_title="Wheat Yield Predictor", layout="centered")
st.title("Wheat Yield Prediction (kg/hectare)")
st.markdown("This application uses monthly weather data to estimate wheat yield per hectare.")


# ============ 侧边栏：加载模型与特征 ============
with st.sidebar:
    st.header("Model and Feature Loader")
    up_model = st.file_uploader("Upload CatBoost model (.cbm)", type=["cbm"])
    up_feats = st.file_uploader("Upload top feature list (.json)", type=["json"])

DEFAULT_MODEL = "models/cat_model.cbm" if os.path.exists("models/cat_model.cbm") else "cat_model.cbm"
DEFAULT_FEATS = "models/top_features.json" if os.path.exists("models/top_features.json") else "top_features.json"

model = None
top_features = None

def load_catboost_from_filelike(filelike) -> CatBoostRegressor:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cbm") as tmp:
        tmp.write(filelike.read())
        tmp_path = tmp.name
    m = CatBoostRegressor()
    m.load_model(tmp_path)
    os.remove(tmp_path)
    return m

try:
    if up_model is not None:
        model = load_catboost_from_filelike(up_model)
    elif os.path.exists(DEFAULT_MODEL):
        model = CatBoostRegressor()
        model.load_model(DEFAULT_MODEL)
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    model = None

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
    st.info("Please upload both the model file (.cbm) and the feature list (.json), "
            "or place them under ./models/ to begin.")


# ============ 工具函数 ============
def unit_label(col: str) -> str:
    lc = col.lower()
    if "tmean" in lc:   u = "[°C]"
    elif "precip" in lc: u = "[mm]"
    elif ("sun" in lc) or ("rad" in lc): u = "[hr]"
    elif ("wind" in lc) or ("ws" in lc): u = "[m/s]"
    elif any(k in lc for k in ["gdd", "hdd", "cdd"]): u = "[°C-days]"
    elif "dryness" in lc: u = "[ratio]"
    else: u = ""
    return f"{col} {u}".strip()

def group_features(feats: list[str]) -> dict[str, list[str]]:
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
    return {k: v for k, v in groups.items() if v}

def predict_with_model(model: CatBoostRegressor, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return model.predict(df[feats])


# ============ 手动输入要显示的“常见字段池” ============
# 会与 top_features 做并集，从而保证界面里一定出现降水/风速/日照等输入框
DISPLAY_CANDIDATES = [
    # 各生育期 温度/降水/日照/风速
    "sowingTmeanAvg","sowingPrecipSum","sowingSunHours","sowingWindAvg",
    "overwinterTmeanAvg","overwinterPrecipSum","overwinterSunHours","overwinterWindAvg",
    "jointingTmeanAvg","jointingPrecipSum","jointingSunHours","jointingWindAvg",
    "headingTmeanAvg","headingPrecipSum","headingSunHours","headingWindAvg",
    "fillingTmeanAvg","fillingPrecipSum","fillingSunHours","fillingWindAvg",
    # 干旱/积温等
    "drynessJointing","drynessHeading","drynessFilling",
    "gddBase5","hddGt30","cddLt0",
]


# ============ 主体：当模型与特征都就绪 ============
if model is not None and top_features is not None:

    # 最终用于“显示输入框”的特征集合：常见字段池 ∪ top_features（保持去重与顺序）
    display_features = list(dict.fromkeys(DISPLAY_CANDIDATES + top_features))

    tab_batch, tab_manual = st.tabs(["Batch Prediction", "Manual Weather Input"])

    # ------- 批量预测 -------
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

    # ------- 手动输入 -------
    with tab_manual:
        st.subheader("Manually Enter Weather Data")

        # 示例输入
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

        demo_name = st.selectbox("Load example data", ["Manual Entry"] + list(demos.keys()))
        if demo_name != "Manual Entry":
            st.info(f"Loaded values from: {demo_name}")
            user_input = demos[demo_name].copy()
        else:
            user_input = {}

        # 分组 + 布局（对 display_features，而非仅 top_features）
        groups = group_features(display_features)
        for gname, cols in groups.items():
            st.markdown(f"**{gname}**")
            cols_container = st.columns(3)
            for i, c in enumerate(cols):
                with cols_container[i % 3]:
                    label = unit_label(c)
                    # 不在模型特征里的字段显示标记
                    if c not in top_features:
                        label = f"{label} (not used by model)"
                    base = user_input.get(c, None)
                    lc = c.lower()

                    if "tmean" in lc:
                        val = st.number_input(label, value=base, min_value=-30.0, max_value=45.0, step=0.1, format="%.1f")
                    elif "precip" in lc:
                        val = st.number_input(label, value=base, min_value=0.0, max_value=500.0, step=0.1, format="%.1f")
                    elif ("sun" in lc) or ("rad" in lc):
                        val = st.number_input(label, value=base, min_value=0.0, max_value=400.0, step=0.1, format="%.1f")
                    elif ("wind" in lc) or ("ws" in lc):
                        val = st.number_input(label, value=base, min_value=0.0, max_value=30.0, step=0.1, format="%.1f")
                    elif "dryness" in lc:
                        val = st.number_input(label, value=base, min_value=0.0, max_value=2.0, step=0.01, format="%.2f")
                    else:
                        val = st.number_input(label, value=base)
                    user_input[c] = val

        # 仅用 top_features 列构造预测输入；其它字段仅用于展示
        input_df = pd.DataFrame([{k: v for k, v in user_input.items() if k in top_features}], columns=top_features)
        # 行中位数填补
        row_median = input_df.iloc[0].dropna().median()
        input_df = input_df.fillna(row_median)

        st.markdown("Final Input Used For Prediction (after filling missing values):")
        st.dataframe(input_df)

        if st.button("Predict Yield"):
            with st.spinner("Running prediction..."):
                yhat = predict_with_model(model, input_df, top_features)[0]
            st.success(f"Predicted Yield per Hectare: {yhat:.2f} tons")
