# app.py
# 前端：CatBoost 分类器（自动判簇） + 按簇 CatBoost 回归器（.pkl）
# 功能：单样本预测（含 CatBoost ShapValues Top-10 解释） + 批量预测（CSV）+ 批量 PDF 导出
# 依赖：streamlit pandas numpy scikit-learn joblib catboost shap matplotlib fpdf2

import os, json, io, datetime, joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

# =================== 路径与模型加载 ===================
BASE = "models"
REG_DIR = os.path.join(BASE, "regressors")

REQ = {
    "cluster_model": os.path.join(BASE, "cluster_model.cbm"),
    "cluster_features": os.path.join(BASE, "cluster_features.json"),
    "cluster_cat_features": os.path.join(BASE, "cluster_cat_features.json"),  # 训练时保存的“类别列名单”
    "clusters": os.path.join(BASE, "clusters.json"),
}
missing = [k for k, p in REQ.items() if not os.path.exists(p)]
st.set_page_config(page_title="Wheat Yield Predictor (Clustered)", layout="centered")
st.title("Wheat Yield Prediction (kg/hectare) — Cluster-Aware")

if missing:
    st.error(f"Missing model artifacts: {missing}. Make sure you ran the export script and put all files under ./models/.")
    st.stop()

# 加载簇分类器（CatBoost .cbm）与特征/类别清单
clf = CatBoostClassifier()
clf.load_model(REQ["cluster_model"])
with open(REQ["cluster_features"], "r", encoding="utf-8") as f:
    clf_features: list[str] = json.load(f)
with open(REQ["cluster_cat_features"], "r", encoding="utf-8") as f:
    clf_cat_features: list[str] = json.load(f)

# 可用簇列表
with open(REQ["clusters"], "r", encoding="utf-8") as f:
    clusters: list[int] = [int(x) for x in json.load(f)]

# 加载各簇回归器（优先 .pkl；若无则尝试 .joblib 或 .cbm）
def _load_regressor_for_cluster(clu: int):
    pkl = os.path.join(REG_DIR, f"cluster_{clu}.pkl")
    jbl = os.path.join(REG_DIR, f"cluster_{clu}.joblib")
    cbm = os.path.join(REG_DIR, f"cluster_{clu}.cbm")
    if os.path.exists(pkl):
        m = joblib.load(pkl)  # CatBoostRegressor 可被 joblib 反序列化
        return m
    if os.path.exists(jbl):
        m = joblib.load(jbl)
        return m
    if os.path.exists(cbm):
        m = CatBoostRegressor(); m.load_model(cbm); return m
    return None

regressors: dict[int, dict] = {}
for clu in clusters:
    model = _load_regressor_for_cluster(clu)
    feats_path = os.path.join(REG_DIR, f"top_features_cluster_{clu}.json")
    if (model is not None) and os.path.exists(feats_path):
        feats = json.load(open(feats_path, "r", encoding="utf-8"))
        regressors[int(clu)] = {"model": model, "features": feats}

if not regressors:
    st.error("No per-cluster regressors were loaded from models/regressors/.")
    st.stop()

# =================== 显示用的输入特征（只展示数值型天气变量） ===================
COMMON_WEATHER = [
    "sowingTmeanAvg","sowingPrecipSum","sowingSunHours","sowingWindAvg",
    "overwinterTmeanAvg","overwinterPrecipSum","overwinterSunHours","overwinterWindAvg",
    "jointingTmeanAvg","jointingPrecipSum","jointingSunHours","jointingWindAvg",
    "headingTmeanAvg","headingPrecipSum","headingSunHours","headingWindAvg",
    "fillingTmeanAvg","fillingPrecipSum","fillingSunHours","fillingWindAvg",
    "drynessJointing","drynessHeading","drynessFilling","gddBase5","hddGt30","cddLt0",
]
all_reg_feats = sorted(set(sum([v["features"] for v in regressors.values()], [])))
DISPLAY_FEATURES = list(dict.fromkeys(COMMON_WEATHER + all_reg_feats))  # 仅数值天气特征

# =================== 工具函数 ===================
def label_with_unit(col: str) -> str:
    lc = col.lower()
    if "tmean" in lc:   u="[°C]"
    elif "precip" in lc: u="[mm]"
    elif ("sun" in lc) or ("rad" in lc): u="[hr]"
    elif ("wind" in lc) or ("ws" in lc): u="[m/s]"
    elif any(k in lc for k in ["gdd","hdd","cdd"]): u="[°C-days]"
    elif "dryness" in lc: u="[ratio]"
    else: u=""
    return f"{col} {u}".strip()

def group_features(cols: list[str]) -> dict[str, list[str]]:
    def has(s,k): return k in s.lower()
    g = {
        "Sowing Phase":      [c for c in cols if has(c,"sowing")],
        "Overwinter Phase":  [c for c in cols if has(c,"overwinter")],
        "Jointing Phase":    [c for c in cols if has(c,"jointing")],
        "Heading Phase":     [c for c in cols if has(c,"heading")],
        "Filling Phase":     [c for c in cols if has(c,"filling")],
        "Dryness Indices":   [c for c in cols if has(c,"dryness")],
        "Extreme Indicators":[c for c in cols if any(t in c.lower() for t in ["gdd","hdd","cdd"])],
        "Other Features":    [c for c in cols if not any(t in c.lower() for t in
                                  ["sowing","overwinter","jointing","heading","filling","dryness","gdd","hdd","cdd"])],
    }
    return {k:v for k,v in g.items() if v}

def predict_cluster(row: pd.Series) -> int:
    """使用 CatBoostClassifier 判簇；数值/类别分别处理，并用 Pool 指定 cat_features。"""
    r = row.reindex(clf_features)

    # 数值列/类别列拆分
    num_cols = [c for c in clf_features if c not in clf_cat_features]
    # 数值：强制转数值，缺失用样本内中位数
    r.loc[num_cols] = pd.to_numeric(r.loc[num_cols], errors="coerce")
    med = r.loc[num_cols].dropna().median()
    r.loc[num_cols] = r.loc[num_cols].fillna(med)

    # 类别：转字符串，缺失填 "NA"
    for c in clf_cat_features:
        val = r.get(c)
        r[c] = "NA" if pd.isna(val) else str(val)

    X_df = pd.DataFrame([r], columns=clf_features)
    pool = Pool(X_df, cat_features=clf_cat_features)
    return int(clf.predict(pool)[0])

def predict_yield_for_cluster(clu: int, row: pd.Series) -> float:
    feats = regressors[clu]["features"]
    X = row.reindex(feats)
    med = X.dropna().median()
    X = X.fillna(med)
    model = regressors[clu]["model"]
    # CatBoostRegressor 支持直接 DataFrame
    yhat = float(model.predict(pd.DataFrame([X], columns=feats))[0])
    return yhat

def shap_top_contrib_catboost(clu: int, row: pd.Series, topn: int = 10):
    """使用 CatBoost 内置 ShapValues（快且稳定）"""
    feats = regressors[clu]["features"]
    model: CatBoostRegressor = regressors[clu]["model"]
    X = row.reindex(feats)
    X = X.fillna(X.dropna().median())
    df1 = pd.DataFrame([X], columns=feats)
    pool = Pool(df1)

    shap_vals = model.get_feature_importance(type="ShapValues", data=pool)  # shape: (1, n_features+1)
    contrib = shap_vals[0, :-1]
    order = np.argsort(np.abs(contrib))[::-1][:min(topn, len(contrib))]

    names = [feats[i] for i in order][::-1]
    vals  = [float(contrib[i]) for i in order][::-1]

    fig, ax = plt.subplots(figsize=(7,4))
    ax.barh(names, vals); ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("SHAP contribution"); ax.set_title(f"Cluster {clu} — SHAP top {len(names)}")
    fig.tight_layout()

    shap_df = pd.DataFrame({"feature":[feats[i] for i in order], "shap":[float(contrib[i]) for i in order]})
    return fig, shap_df

def make_pdf_batch(rows: list[dict]) -> bytes:
    try:
        from fpdf import FPDF
    except Exception as e:
        raise RuntimeError("fpdf2 not installed. `pip install fpdf2` to enable PDF export.") from e
    pdf = FPDF()
    for r in rows:
        pdf.add_page(); pdf.set_font("Arial", size=14)
        pdf.cell(0, 10, "Wheat Yield Prediction Report", ln=1)
        pdf.set_font("Arial", size=11)
        for k in ["_row","cluster","predicted_yield","latitude","longitude","sown_area","year"]:
            if k in r and r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k])):
                pdf.multi_cell(0, 8, f"{k.replace('_',' ').title()}: {r[k]}")
        if "_error" in r:
            pdf.set_text_color(220,20,60); pdf.multi_cell(0,8,f"Error: {r['_error']}"); pdf.set_text_color(0,0,0)
    out = io.BytesIO(); pdf.output(out); return out.getvalue()

# =================== 界面 ===================
mode = st.sidebar.radio("Prediction Mode", ["Single prediction", "Batch prediction (CSV)"], index=0)

if mode == "Batch prediction (CSV)":
    st.subheader("Clustered Batch Prediction (CatBoost regressors in .pkl)")
    st.markdown("If your CSV has a `cluster` column, it will be used; otherwise the app will auto-detect the cluster.")
    st.markdown("**For auto-cluster, these columns help:**")
    st.code(", ".join(clf_cat_features + [c for c in clf_features if c not in clf_cat_features][:10]))

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
        out_rows = []
        for i, row in df.iterrows():
            # 判簇：优先使用 CSV 的 cluster，否则自动分类
            clu = None
            if "cluster" in df.columns and not pd.isna(row.get("cluster", np.nan)):
                try: clu = int(row["cluster"])
                except: clu = None
            if clu is None:
                clu = predict_cluster(row)

            if clu not in regressors:
                out_rows.append({"_row": i, "cluster": clu, "predicted_yield": np.nan, "_error": "no model for this cluster"})
                continue

            yhat = predict_yield_for_cluster(clu, row)
            rec = {"_row": i, "cluster": clu, "predicted_yield": yhat}
            for extra in ["latitude","longitude","year","sown_area","lat","lon"]:
                if extra in df.columns: rec[extra] = row[extra]
            out_rows.append(rec)

        res = pd.DataFrame(out_rows)
        st.markdown("Predictions:")
        st.dataframe(res)

        st.download_button("Download results CSV", res.to_csv(index=False).encode("utf-8"),
                           file_name="clustered_predictions.csv")

        try:
            pdf_bytes = make_pdf_batch(out_rows)
            st.download_button("Download Reports PDF", data=pdf_bytes,
                               file_name="clustered_reports.pdf", mime="application/pdf")
        except Exception as e:
            st.info(str(e))

else:
    st.subheader("Single prediction (CatBoost per cluster)")

    # 基本上下文（用于记录/导出，不一定是模型特征）
    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2018, step=1, key="in_year")
        sown_area = st.number_input("Sown area (ha)", min_value=0.0, value=1000.0, step=10.0, format="%.2f", key="in_sown_area")
    with c2:
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=34.75, step=0.01, format="%.2f", key="in_latitude")
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=113.62, step=0.01, format="%.2f", key="in_longitude")

    # 天气/特征输入（分组展示，仅数值）
    with st.expander("Weather and feature inputs"):
        groups = group_features(DISPLAY_FEATURES)
        rendered = set()
        for gname, cols in groups.items():
            st.markdown(f"**{gname}**")
            cols3 = st.columns(3)
            for i, c in enumerate(cols):
                if c in rendered: 
                    continue
                rendered.add(c)
                with cols3[i % 3]:
                    key = f"in_{c}"
                    lc  = c.lower()
                    kwargs = {"key": key, "label": label_with_unit(c)}
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
                    st.number_input(**kwargs)

    # 构造一行数据（仅天气数值）
    row = {k.replace("in_",""): v for k, v in st.session_state.items() if k.startswith("in_")}
    row = pd.Series(row, dtype="float64")

    # 判簇按钮
    if st.button("Detect cluster"):
        detected = predict_cluster(row)
        st.session_state["detected_cluster"] = detected
        st.success(f"Detected cluster: {detected}")

    clu = st.session_state.get("detected_cluster", predict_cluster(row))
    st.caption(f"Cluster to use: {clu}")

    # 预测与 SHAP（CatBoost ShapValues）
    if st.button("Predict"):
        if clu not in regressors:
            st.error("No regressor for the detected cluster.")
        else:
            yhat = predict_yield_for_cluster(clu, row)
            st.markdown(f"### Predicted Yield: **{yhat:.3f} tons/ha**")

            try:
                fig, shap_df = shap_top_contrib_catboost(clu, row, topn=10)
                st.markdown("**SHAP explanation (Top-10 features):**")
                st.pyplot(fig)
                st.download_button("Download SHAP CSV",
                    shap_df.to_csv(index=False).encode("utf-8"), file_name="shap_top10.csv")
            except Exception as e:
                st.info(f"SHAP not available: {e}")
