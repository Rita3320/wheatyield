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

# 允许从默认路径读取（未上传时）
DEFAULT_MODEL = "models/cat_model.cbm" if os.path.exists("models/cat_model.cbm") else "cat_model.cbm"
DEFAULT_FEATS = "models/top_features.json" if os.path.exists("models/top_features.json") else "top_features.json"

model = None
top_features = None

def load_catboost_from_filelike(filelike) -> CatBoostRegressor:
    """CatBoost 只能从文件路径加载，这里把 BytesIO 暂存到临时文件再读取。"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cbm") as tmp:
        tmp.write(filelike.read())
        tmp_path = tmp.name
    m = CatBoostRegressor()
    m.load_model(tmp_path)
    os.remove(tmp_path)
    return m

# 加载模型
try:
    if up_model is not None:
        model = load_catboost_from_filelike(up_model)
    elif os.path.exists(DEFAULT_MODEL):
        model = CatBoostRegressor()
        model.load_model(DEFAULT_MODEL)
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    model = None

# 加载特征
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
    """根据列名给出单位标注。"""
    if "tmean" in col:   return f"{col} [°C]"
    if "precip" in col:  return f"{col} [mm]"
    if "sun" in col:     return f"{col} [hr]"
    if "wind" in col:    return f"{col} [m/s]"
    if any(k in col for k in ["gdd", "hdd", "cdd"]):  return f"{col} [°C-days]"
    if "dryness" in col: return f"{col} [ratio]"
    return col

def group_features(feats: list[str]) -> dict[str, list[str]]:
    """按生育期关键词进行自动分组。"""
    groups = {
        "Sowing Phase":     [c for c in feats if "sowing" in c],
        "Overwinter Phase": [c for c in feats if "overwinter" in c],
        "Jointing Phase":   [c for c in feats if "jointing" in c],
        "Heading Phase":    [c for c in feats if "heading" in c],
        "Filling Phase":    [c for c in feats if "filling" in c],
        "Dryness Indices":  [c for c in feats if "dryness" in c],
        "Extreme Indicators":[c for c in feats if any(k in c for k in ["gdd", "hdd", "cdd"])],
    }
    used = set(sum(groups.values(), []))
    groups["Other Features"] = [c for c in feats if c not in used]
    # 去掉空组
    return {k: v for k, v in groups.items() if v}

def predict_with_model(model: CatBoostRegressor, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return model.predict(df[feats])


# ============ 主体：当模型与特征都就绪 ============
if model is not None and top_features is not None:

    tab_batch, tab_manual = st.tabs(["Batch Prediction", "Manual Weather Input"])

    # ------- 批量预测 -------
    with tab_batch:
        st.subheader("Upload CSV File for Batch Prediction")
        st.markdown("The CSV must contain the following columns:")
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
                    # 行中位数填补
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
                "sowingTmeanAvg": 12.3, "sowingPrecipSum": 45.0,
                "overwinterTmeanAvg": 2.1, "overwinterPrecipSum": 18.0,
                "jointingTmeanAvg": 15.7, "jointingPrecipSum": 38.0,
                "headingTmeanAvg": 18.9, "headingPrecipSum": 42.0,
                "fillingTmeanAvg": 22.4, "fillingPrecipSum": 65.0,
                "drynessJointing": 0.4, "drynessHeading": 0.5, "drynessFilling": 0.2,
                "gddBase5": 1600, "hddGt30": 3, "cddLt0": 20
            },
            "Cold Dry Winter": {
                "sowingTmeanAvg": 10.0, "sowingPrecipSum": 20.0,
                "overwinterTmeanAvg": -3.0, "overwinterPrecipSum": 5.0,
                "jointingTmeanAvg": 14.0, "jointingPrecipSum": 25.0,
                "headingTmeanAvg": 17.0, "headingPrecipSum": 30.0,
                "fillingTmeanAvg": 21.0, "fillingPrecipSum": 55.0,
                "drynessJointing": 0.6, "drynessHeading": 0.7, "drynessFilling": 0.5,
                "gddBase5": 1400, "hddGt30": 1, "cddLt0": 45
            },
            "Hot Wet Late Season": {
                "sowingTmeanAvg": 13.0, "sowingPrecipSum": 50.0,
                "overwinterTmeanAvg": 1.5, "overwinterPrecipSum": 22.0,
                "jointingTmeanAvg": 17.0, "jointingPrecipSum": 45.0,
                "headingTmeanAvg": 20.5, "headingPrecipSum": 60.0,
                "fillingTmeanAvg": 25.0, "fillingPrecipSum": 110.0,
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

        # 分组 + 布局
        groups = group_features(top_features)
        for gname, cols in groups.items():
            st.markdown(f"**{gname}**")
            cols_container = st.columns(3)
            for i, c in enumerate(cols):
                with cols_container[i % 3]:
                    label = unit_label(c)
                    # 合理的范围提示（仅在常见字段上限定）
                    if "tmean" in c:
                        val = st.number_input(label, value=user_input.get(c, None),
                                              min_value=-30.0, max_value=45.0, step=0.1, format="%.1f")
                    elif "precip" in c:
                        val = st.number_input(label, value=user_input.get(c, None),
                                              min_value=0.0, max_value=500.0, step=0.1, format="%.1f")
                    elif "sun" in c:
                        val = st.number_input(label, value=user_input.get(c, None),
                                              min_value=0.0, max_value=400.0, step=0.1, format="%.1f")
                    elif "wind" in c:
                        val = st.number_input(label, value=user_input.get(c, None),
                                              min_value=0.0, max_value=30.0, step=0.1, format="%.1f")
                    elif "dryness" in c:
                        val = st.number_input(label, value=user_input.get(c, None),
                                              min_value=0.0, max_value=2.0, step=0.01, format="%.2f")
                    else:
                        # gdd/hdd/cdd/other：不强制范围
                        val = st.number_input(label, value=user_input.get(c, None))
                    user_input[c] = val

        # 生成 DataFrame 并填补缺失
        input_df = pd.DataFrame([user_input], columns=top_features)  # 按模型特征列顺序
        row_median = input_df.iloc[0].dropna().median()
        input_df = input_df.fillna(row_median)

        st.markdown("Final Input (after filling missing values):")
        st.dataframe(input_df)

        if st.button("Predict Yield"):
            with st.spinner("Running prediction..."):
                yhat = predict_with_model(model, input_df, top_features)[0]
            st.success(f"Predicted Yield per Hectare: {yhat:.2f} tons")
