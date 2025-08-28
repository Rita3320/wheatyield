# app.py — Wheat Yield Prediction (cluster-aware with flat-file fallback)
# 支持两种文件组织：
# 1) 分簇：cluster_#.pkl/.joblib/.cbm + top_features_cluster_#.json（可放根目录或 models/regressors/）
# 2) 全局：cat_model.pkl/.cbm + top_features.json/.pkl（放根目录或 models/）
# 若存在分类器三件（cluster_model.cbm + cluster_features.json + cluster_cat_features.json），则启用自动判簇

import os, io, re, json, joblib, zipfile, tempfile
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

st.set_page_config(page_title="Wheat Yield Predictor", layout="centered")
st.title("Wheat Yield Prediction (kg/hectare) — Cluster-Aware")

# ---------- small helpers ----------
def find_first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def load_json_or_pkl(path: str):
    if path.lower().endswith(".json"):
        return json.load(open(path, "r", encoding="utf-8"))
    return joblib.load(path)

# ---------- scan artifacts (支持根目录 & models/) ----------
CWD = os.getcwd()
MODELS_DIR = "models"
REG_DIR = os.path.join(MODELS_DIR, "regressors")

# 分类器（可选）
clf_model_path = find_first(["cluster_model.cbm", os.path.join(MODELS_DIR, "cluster_model.cbm")])
clf_features_path = find_first(["cluster_features.json", os.path.join(MODELS_DIR, "cluster_features.json")])
clf_cat_features_path = find_first(["cluster_cat_features.json", os.path.join(MODELS_DIR, "cluster_cat_features.json")])

HAS_CLF = all([clf_model_path, clf_features_path, clf_cat_features_path])
clf = None; clf_features = []; clf_cat_features = []
if HAS_CLF:
    try:
        clf = CatBoostClassifier(); clf.load_model(clf_model_path)
        clf_features = json.load(open(clf_features_path, "r", encoding="utf-8"))
        clf_cat_features = json.load(open(clf_cat_features_path, "r", encoding="utf-8"))
    except Exception as e:
        st.warning(f"Classifier found but failed to load: {e}")
        HAS_CLF = False

# 分簇回归器
def scan_cluster_regressors() -> Dict[int, Dict[str, Any]]:
    regs: Dict[int, Dict[str, Any]] = {}
    cand_dirs = [CWD, REG_DIR]
    for d in cand_dirs:
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            m = re.match(r"cluster_(\d+)\.(pkl|joblib|cbm)$", fn, re.I)
            if not m: continue
            clu = int(m.group(1)); ext = m.group(2).lower()
            path = os.path.join(d, fn)
            try:
                if ext == "cbm":
                    mdl = CatBoostRegressor(); mdl.load_model(path)
                else:
                    mdl = joblib.load(path)
            except Exception:
                continue
            feats_path = find_first([
                os.path.join(d, f"top_features_cluster_{clu}.json"),
                os.path.join(d, f"top_features_cluster_{clu}.pkl"),
                os.path.join(REG_DIR, f"top_features_cluster_{clu}.json"),
                os.path.join(REG_DIR, f"top_features_cluster_{clu}.pkl"),
            ])
            if not feats_path: continue
            feats = load_json_or_pkl(feats_path)
            regs[clu] = {"model": mdl, "features": list(feats)}
    return regs

regressors = scan_cluster_regressors()

# 全局回归器（退路）
GLOBAL_MODEL_PATH = find_first([
    "cat_model.pkl","cat_model.joblib","cat_model.cbm",
    os.path.join(MODELS_DIR,"cat_model.pkl"), os.path.join(MODELS_DIR,"cat_model.joblib"),
    os.path.join(MODELS_DIR,"cat_model.cbm")
])
GLOBAL_FEATS_PATH = find_first([
    "top_features.json","top_features.pkl",
    os.path.join(MODELS_DIR,"top_features.json"), os.path.join(MODELS_DIR,"top_features.pkl")
])

global_regressor: Optional[Dict[str, Any]] = None
if not regressors and GLOBAL_MODEL_PATH and GLOBAL_FEATS_PATH:
    if GLOBAL_MODEL_PATH.lower().endswith(".cbm"):
        gm = CatBoostRegressor(); gm.load_model(GLOBAL_MODEL_PATH)
    else:
        gm = joblib.load(GLOBAL_MODEL_PATH)
    feats = load_json_or_pkl(GLOBAL_FEATS_PATH)
    global_regressor = {"model": gm, "features": list(feats)}
    st.caption("Loaded global regressor (no clustering).")
elif regressors:
    st.caption(f"Loaded per-cluster regressors: {sorted(regressors.keys())}")
else:
    st.error("No regressors found. Provide either:\n"
             "• cluster_#.pkl + top_features_cluster_#.json (per-cluster), or\n"
             "• cat_model.pkl + top_features.json (global).")
    st.stop()

if HAS_CLF:
    st.caption("Cluster classifier available: Auto-detect enabled.")

# ---------- UI feature list ----------
COMMON_WEATHER = [
    "sowingTmeanAvg","sowingPrecipSum","sowingSunHours","sowingWindAvg",
    "overwinterTmeanAvg","overwinterPrecipSum","overwinterSunHours","overwinterWindAvg",
    "jointingTmeanAvg","jointingPrecipSum","jointingSunHours","jointingWindAvg",
    "headingTmeanAvg","headingPrecipSum","headingSunHours","headingWindAvg",
    "fillingTmeanAvg","fillingPrecipSum","fillingSunHours","fillingWindAvg",
    "drynessJointing","drynessHeading","drynessFilling","gddBase5","hddGt30","cddLt0",
]
if regressors:
    all_reg_feats = sorted(set(sum([v["features"] for v in regressors.values()], [])))
else:
    all_reg_feats = list(global_regressor["features"])
DISPLAY_FEATURES = list(dict.fromkeys(COMMON_WEATHER + all_reg_feats))

def label_with_unit(col: str) -> str:
    lc = col.lower()
    if "tmean" in lc: u="[°C]"
    elif "precip" in lc: u="[mm]"
    elif ("sun" in lc) or ("rad" in lc): u="[hr]"
    elif ("wind" in lc) or ("ws" in lc): u="[m/s]"
    elif any(k in lc for k in ["gdd","hdd","cdd"]): u="[°C-days]"
    elif "dryness" in lc: u="[ratio]"
    else: u=""
    return f"{col} {u}".strip()

def group_features(cols: List[str]) -> Dict[str, List[str]]:
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

# ---------- predict helpers ----------
def predict_cluster(row: pd.Series) -> Optional[int]:
    if not HAS_CLF: return None
    r = row.reindex(clf_features)
    num_cols = [c for c in clf_features if c not in clf_cat_features]
    r.loc[num_cols] = pd.to_numeric(r.loc[num_cols], errors="coerce")
    med = r.loc[num_cols].dropna().median()
    r.loc[num_cols] = r.loc[num_cols].fillna(med)
    for c in clf_cat_features:
        v = r.get(c)
        r[c] = "NA" if pd.isna(v) else str(v)
    pool = Pool(pd.DataFrame([r], columns=clf_features), cat_features=clf_cat_features)
    return int(clf.predict(pool)[0])

def predict_yield(clu: Optional[int], row: pd.Series) -> Tuple[float, List[str], Any]:
    if regressors:
        if clu is None or clu not in regressors:
            raise RuntimeError("No regressor for the detected/selected cluster.")
        feats = regressors[clu]["features"]; model = regressors[clu]["model"]
    else:
        feats = global_regressor["features"]; model = global_regressor["model"]
    X = row.reindex(feats)
    X = X.fillna(X.dropna().median())
    yhat = float(model.predict(pd.DataFrame([X], columns=feats))[0])
    return yhat, feats, model

def shap_single(row: pd.Series, feats: List[str], model: Any, clu: Optional[int]=None):
    X = row.reindex(feats)
    X = X.fillna(X.dropna().median())
    df1 = pd.DataFrame([X], columns=feats)
    try:
        if isinstance(model, CatBoostRegressor):
            sv = model.get_feature_importance(type="ShapValues", data=Pool(df1))
            contrib = sv[0, :-1]
        else:
            raise TypeError("not catboost")
    except Exception:
        import shap
        def f(Xa): return model.predict(pd.DataFrame(Xa, columns=feats))
        expl = shap.KernelExplainer(f, df1)
        vals = expl.shap_values(df1, nsamples=min(100, 2*len(feats)))
        contrib = np.array(vals)[0]
    order = np.argsort(np.abs(contrib))[::-1][:min(10, len(contrib))]
    names = [feats[i] for i in order][::-1]; vals = [float(contrib[i]) for i in order][::-1]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.barh(names, vals); ax.axvline(0, linestyle="--", linewidth=1)
    title = f"SHAP top {len(names)}" if clu is None else f"Cluster {clu} — SHAP top {len(names)}"
    ax.set_title(title); ax.set_xlabel("SHAP contribution"); fig.tight_layout()
    shap_df = pd.DataFrame({"feature":[feats[i] for i in order], "shap":[float(contrib[i]) for i in order]})
    return fig, shap_df

def make_pdf_batch(rows: List[Dict[str, Any]]) -> bytes:
    from fpdf import FPDF
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

# ---------- UI ----------
mode = st.sidebar.radio("Prediction Mode", ["Single prediction", "Batch prediction (CSV)"], index=0)

if mode == "Batch prediction (CSV)":
    st.subheader("Batch prediction")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df_in = pd.read_csv(up)
        out_rows = []
        for i, row in df_in.iterrows():
            clu = None
            if regressors:
                if "cluster" in df_in.columns and not pd.isna(row.get("cluster", np.nan)):
                    try: clu = int(row["cluster"])
                    except: clu = None
                if clu is None and HAS_CLF:
                    clu = predict_cluster(row)
            try:
                yhat, feats, model = predict_yield(clu, row)
                rec = {"_row": i, "cluster": clu if clu is not None else "global", "predicted_yield": yhat}
                for extra in ["latitude","longitude","year","sown_area","lat","lon"]:
                    if extra in df_in.columns: rec[extra] = row[extra]
                out_rows.append(rec)
            except Exception as e:
                out_rows.append({"_row": i, "cluster": clu, "predicted_yield": np.nan, "_error": str(e)})
        res = pd.DataFrame(out_rows)
        st.dataframe(res)
        st.download_button("Download results CSV", res.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv")
        try:
            pdf_bytes = make_pdf_batch(out_rows)
            st.download_button("Download Reports PDF", data=pdf_bytes,
                               file_name="reports.pdf", mime="application/pdf")
        except Exception as e:
            st.info(str(e))

else:
    st.subheader("Single prediction")
    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2018, step=1, key="in_year")
        sown_area = st.number_input("Sown area (ha)", min_value=0.0, value=1000.0, step=10.0, format="%.2f", key="in_sown_area")
    with c2:
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=34.75, step=0.01, format="%.2f", key="in_latitude")
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=113.62, step=0.01, format="%.2f", key="in_longitude")

    with st.expander("Weather and feature inputs"):
        groups = group_features(DISPLAY_FEATURES)
        rendered = set()
        for gname, cols in groups.items():
            st.markdown(f"**{gname}**")
            cols3 = st.columns(3)
            for i, c in enumerate(cols):
                if c in rendered: continue
                rendered.add(c)
                with cols3[i % 3]:
                    key = f"in_{c}"; lc = c.lower()
                    kwargs = {"key": key, "label": label_with_unit(c)}
                    if "tmean" in lc: kwargs.update(min_value=-30.0, max_value=45.0, step=0.1, format="%.1f")
                    elif "precip" in lc: kwargs.update(min_value=0.0, max_value=500.0, step=0.1, format="%.1f")
                    elif ("sun" in lc) or ("rad" in lc): kwargs.update(min_value=0.0, max_value=400.0, step=0.1, format="%.1f")
                    elif ("wind" in lc) or ("ws" in lc): kwargs.update(min_value=0.0, max_value=30.0, step=0.1, format="%.1f")
                    elif "dryness" in lc: kwargs.update(min_value=0.0, max_value=2.0, step=0.01, format="%.2f")
                    st.number_input(**kwargs)

    row = {k.replace("in_",""): v for k,v in st.session_state.items() if k.startswith("in_")}
    row = pd.Series(row, dtype="float64")

    clu = None
    if regressors:
        if HAS_CLF:
            if st.button("Detect cluster"):
                det = predict_cluster(row); st.session_state["detected_cluster"] = det
                st.success(f"Detected cluster: {det}")
            clu = st.session_state.get("detected_cluster", predict_cluster(row))
            st.caption(f"Cluster to use: {clu}")
        else:
            clu = st.selectbox("Select cluster", sorted(regressors.keys()))

    if st.button("Predict"):
        try:
            yhat, feats, model = predict_yield(clu, row)
            st.markdown(f"### Predicted Yield: **{yhat:.3f} tons/ha**")
            fig, shap_df = shap_single(row, feats, model, clu=clu)
            st.markdown("**SHAP explanation (Top-10 features):**")
            st.pyplot(fig)
            st.download_button("Download SHAP CSV",
                               shap_df.to_csv(index=False).encode("utf-8"),
                               file_name="shap_top10.csv")
        except Exception as e:
            st.error(str(e))
