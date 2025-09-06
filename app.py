# app.py — Wheat Yield Prediction (cluster-aware with flat-file fallback)
# 变更：
#   1) 输入：纬度/经度/海拔 + 每月(平均/最高/最低气温、降水、日照、风速) + sown_area_hectare
#   2) 自动聚合为模型用的季节/农时特征（sowing/overwinter/jointing/heading/filling + dryness + GDD/HDD/CDD）
#   3) SHAP 图固定显示绝对值(|value|)，不提供开关
#   4) Insight：英文，一句话“温度/降水/日照/风”摘要 + 可执行建议（阈值触发），不依赖 SHAP
#   5) 修复：Streamlit 小部件默认值冲突
#   6) 修复：Insight 使用完整聚合后的行（不再 reindex(feats)），避免信息丢失

import os, io, re, json, joblib, random, datetime
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
clf = None; clf_features: List[str] = []; clf_cat_features: List[str] = []
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
    for d in [CWD, REG_DIR]:
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

# ========== 月度输入 Schema ==========
MONTHS = [f"{m:02d}" for m in range(1, 12+1)]
COLS_MONTHLY = []
for m in MONTHS:
    COLS_MONTHLY += [
        f"tavg_{m}_C",     # 每月平均气温 (℃)
        f"tmax_{m}_C",     # 每月最高气温 (℃)
        f"tmin_{m}_C",     # 每月最低气温 (℃)
        f"precip_{m}_mm",  # 每月降水量 (mm)
        f"sunshine_{m}_h", # 每月日照时数 (h)
        f"wind_{m}_ms",    # 每月平均风速 (m/s)
    ]

BASE_INPUTS = ["latitude", "longitude", "elevation_m", "sown_area_hectare", "year"]
ALL_INPUT_COLS = BASE_INPUTS + COLS_MONTHLY

# ========== 聚合：月度 -> 模型特征 ==========
PHASES = {
    "sowing":      ["10", "11"],
    "overwinter":  ["12", "01", "02"],
    "jointing":    ["03", "04"],
    "heading":     ["05"],
    "filling":     ["06"],
}
GROW_MONTHS = ["10","11","12","01","02","03","04","05","06"]

def _safe_mean(vals):
    vals = [v for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

def _safe_sum(vals):
    vals = [v for v in vals if pd.notna(v)]
    return float(np.sum(vals)) if vals else np.nan

def monthly_to_features(row: pd.Series) -> pd.Series:
    out = {}
    # 1) 农时聚合
    for phase, months in PHASES.items():
        tavg = []; precip = []; sun = []; wind = []
        for m in months:
            tavg.append(row.get(f"tavg_{m}_C"))
            precip.append(row.get(f"precip_{m}_mm"))
            sun.append(row.get(f"sunshine_{m}_h"))
            wind.append(row.get(f"wind_{m}_ms"))
        out[f"{phase}TmeanAvg"]  = _safe_mean(tavg)
        out[f"{phase}PrecipSum"] = _safe_sum(precip)
        out[f"{phase}SunHours"]  = _safe_sum(sun)
        out[f"{phase}WindAvg"]   = _safe_mean(wind)
        tavg_mean = out[f"{phase}TmeanAvg"]; precip_sum = out[f"{phase}PrecipSum"]
        out[f"dryness{phase.capitalize()}"] = (
            float(precip_sum) / (float(tavg_mean) + 10.0)
            if (pd.notna(precip_sum) and pd.notna(tavg_mean)) else np.nan
        )

    # 2) GDD/HDD/CDD（简化估算：按月≈30天）
    def _deg_days_pos(x): return max(float(x), 0.0)
    gdd5 = 0.0; hdd30 = 0.0; cdd0 = 0.0
    for m in GROW_MONTHS:
        tavg = row.get(f"tavg_{m}_C")
        tmax = row.get(f"tmax_{m}_C")
        tmin = row.get(f"tmin_{m}_C")
        if pd.notna(tavg): gdd5 += _deg_days_pos(tavg - 5.0) * 30.0
        if pd.notna(tmax): hdd30 += _deg_days_pos(tmax - 30.0) * 30.0
        if pd.notna(tmin): cdd0  += _deg_days_pos(0.0 - tmin) * 30.0
    out["gddBase5"] = gdd5; out["hddGt30"] = hdd30; out["cddLt0"] = cdd0

    # 3) 地理/背景
    out["latitude"]      = row.get("latitude")
    out["longitude"]     = row.get("longitude")
    out["elevation_m"]   = row.get("elevation_m")
    out["sown_area"]     = row.get("sown_area_hectare")  # PDF 显示
    out["sown_area_hectare"] = row.get("sown_area_hectare")
    out["year"]          = row.get("year")
    return pd.Series(out, dtype="float64")

# ---------- Random generators ----------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _sin_annual(month_idx: int, amp: float, base: float) -> float:
    x = (month_idx-1)/12.0 * 2*np.pi
    return base + amp * np.sin(x - np.pi/2)

def autofill_monthly_by_lat_elev(lat: float, elev: float, seed: Optional[int] = None) -> Dict[str, float]:
    if seed is not None:
        random.seed(int(seed)); np.random.seed(int(seed))
    out = {}
    lat_abs = abs(lat)
    base = 23.0 - 0.25*lat_abs - 6.5*(elev/1000.0)
    amp  = 12.0
    for i, m in enumerate(MONTHS, start=1):
        tavg = _sin_annual(i, amp, base) + np.random.normal(0, 1.0)
        tmax = tavg + 6 + np.random.normal(0, 0.8)
        tmin = tavg - 6 + np.random.normal(0, 0.8)
        precip   = max(0.0, (8 + 4*np.sin((i-1)/12*2*np.pi)) * (random.uniform(5, 20)))
        sunshine = max(0.0, 150 + 80*np.sin((i-1)/12*2*np.pi) + np.random.normal(0, 10))
        wind     = _clamp(random.uniform(2.0, 6.0) + np.random.normal(0, 0.5), 0.0, 30.0)
        out[f"tavg_{m}_C"]     = round(float(tavg), 1)
        out[f"tmax_{m}_C"]     = round(float(tmax), 1)
        out[f"tmin_{m}_C"]     = round(float(tmin), 1)
        out[f"precip_{m}_mm"]  = round(float(precip), 1)
        out[f"sunshine_{m}_h"] = round(float(sunshine), 1)
        out[f"wind_{m}_ms"]    = round(float(wind), 1)
    return out

# ---------- SHAP ----------
def shap_single(row_model_feats: pd.Series, feats: List[str], model: Any, clu: Optional[int]=None) -> Tuple[np.ndarray, List[str], float]:
    """返回 (contrib, feats, base_value)。"""
    X = row_model_feats.reindex(feats).fillna(row_model_feats.reindex(feats).dropna().median())
    df1 = pd.DataFrame([X], columns=feats)
    base_value = 0.0
    try:
        if isinstance(model, CatBoostRegressor):
            sv = model.get_feature_importance(type="ShapValues", data=Pool(df1))
            contrib = sv[0, :-1]
            base_value = float(sv[0, -1])
        else:
            raise TypeError("not catboost")
    except Exception:
        import shap
        def f(Xa): return model.predict(pd.DataFrame(Xa, columns=feats))
        expl = shap.KernelExplainer(f, df1)
        vals = expl.shap_values(df1, nsamples=min(100, 2*len(feats)))
        contrib = np.array(vals)[0]
        base_value = float(model.predict(df1)[0] - contrib.sum())
    return np.array(contrib, dtype=float), feats, base_value

def _plot_shap_bar_abs(feats: List[str], contrib: np.ndarray, clu: Optional[int]):
    """水平条形图（固定按 |SHAP| 绘图）。"""
    order = np.argsort(np.abs(contrib))[::-1][:min(10, len(contrib))]
    names = [feats[i] for i in order][::-1]
    vplot = [float(abs(contrib[i])) for i in order][::-1]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.barh(names, vplot)
    title = f"Cluster {clu} — SHAP top {len(names)}" if clu is not None else f"SHAP top {len(names)}"
    ax.set_title(title)
    ax.set_xlabel("Absolute SHAP contribution")
    fig.tight_layout()
    return fig

# ---------- predict helpers ----------
def predict_cluster(row_model_feats: pd.Series) -> Optional[int]:
    if not HAS_CLF: return None
    r = row_model_feats.reindex(clf_features)
    num_cols = [c for c in clf_features if c not in clf_cat_features]
    r.loc[num_cols] = pd.to_numeric(r.loc[num_cols], errors="coerce")
    med = r.loc[num_cols].dropna().median()
    r.loc[num_cols] = r.loc[num_cols].fillna(med)
    for c in clf_cat_features:
        v = r.get(c); r[c] = "NA" if pd.isna(v) else str(v)
    pool = Pool(pd.DataFrame([r], columns=clf_features), cat_features=clf_cat_features)
    return int(clf.predict(pool)[0])

def predict_yield(clu: Optional[int], row_model_feats: pd.Series) -> Tuple[float, List[str], Any]:
    if regressors:
        if clu is None or clu not in regressors:
            raise RuntimeError("No regressor for the detected/selected cluster.")
        feats = regressors[clu]["features"]; model = regressors[clu]["model"]
    else:
        feats = global_regressor["features"]; model = global_regressor["model"]
    X = row_model_feats.reindex(feats); X = X.fillna(X.dropna().median())
    yhat = float(model.predict(pd.DataFrame([X], columns=feats))[0])
    return yhat, feats, model

# ---------- insight generation（英文摘要+建议） ----------
def insight_text(_cluster_unused: Optional[int], row_model_feats: pd.Series) -> str:
    """
    English Insight:
    - One-sentence climate summary using only temperature / precipitation / sunshine / wind.
    - Then actionable, stage-specific tips triggered by thresholds.
    - No cluster mention, no SHAP dependence.
    """
    def gv(name):
        v = row_model_feats.get(name)
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    # Stage metrics
    sow_t   = gv("sowingTmeanAvg");      sow_p   = gv("sowingPrecipSum")
    over_t  = gv("overwinterTmeanAvg");  over_s  = gv("overwinterSunHours")
    joint_t = gv("jointingTmeanAvg");    joint_p = gv("jointingPrecipSum")
    head_t  = gv("headingTmeanAvg");     head_p  = gv("headingPrecipSum");   head_s = gv("headingSunHours"); head_w = gv("headingWindAvg")
    fill_t  = gv("fillingTmeanAvg");     fill_p  = gv("fillingPrecipSum");   fill_s = gv("fillingSunHours"); fill_w = gv("fillingWindAvg")

    # Dryness & thermal stress (tips only)
    d_joint = gv("drynessJointing"); d_head = gv("drynessHeading"); d_fill = gv("drynessFilling")
    gdd = gv("gddBase5"); hdd = gv("hddGt30"); cdd = gv("cddLt0")

    # Season rollups (for summary)
    temp_pool = [x for x in [sow_t, over_t, joint_t, head_t, fill_t] if x is not None]
    season_t = float(np.mean(temp_pool)) if temp_pool else None
    precip_pool = [p for p in [sow_p, joint_p, head_p, fill_p] if p is not None]
    season_p = float(np.sum(precip_pool)) if precip_pool else None
    sun_pool = [s for s in [over_s, head_s, fill_s] if s is not None]
    season_s = float(np.sum(sun_pool)) if sun_pool else None
    wind_pool = [w for w in [head_w, fill_w] if w is not None]
    season_w = float(np.mean(wind_pool)) if wind_pool else None

    # Bucketing to words
    def bucket_temp(t):
        if t is None: return "typical temperatures"
        if t < 5:   return "very cool conditions"
        if t < 10:  return "cool conditions"
        if t < 18:  return "mild conditions"
        if t < 24:  return "warm conditions"
        return "hot conditions"
    def bucket_rain(p):
        if p is None: return "typical rainfall"
        if p < 300:  return "dry rainfall"
        if p <= 700: return "moderate rainfall"
        return "wet rainfall"
    def bucket_sun(s):
        if s is None: return "average sunshine"
        if s < 350:  return "limited sunshine"
        if s <= 650: return "average sunshine"
        return "plentiful sunshine"
    def bucket_wind(w):
        if w is None: return "light winds"
        if w < 2:    return "calm winds"
        if w <= 6:   return "breezy winds"
        return "windy conditions"

    summary = (
        f"Overall, {bucket_temp(season_t)} and {bucket_rain(season_p)}; "
        f"{bucket_sun(season_s)} with {bucket_wind(season_w)}."
    )

    # Recommendations
    tips: List[str] = []
    if any(d is not None and d >= 0.9 for d in [d_joint, d_head, d_fill]) \
       or (head_p is not None and head_p < 40) or (fill_p is not None and fill_p < 40):
        tips += [
            "During jointing–grain filling, schedule light but frequent irrigations (every 5–7 days).",
            "After sowing, roll lightly or keep residue mulch to conserve moisture."
        ]
    if any(p is not None and p > 180 for p in [joint_p, head_p, fill_p]):
        tips += [
            "After heavy rain, open drainage furrows promptly, especially in low-lying fields.",
            "Scout closely for diseases around heading (e.g., Fusarium head blight) and treat in time."
        ]
    if (over_t is not None and over_t < 0) or (cdd is not None and cdd > 150):
        tips += [
            "Monitor cold snaps; implement cold-protection where feasible.",
            "Split N before winter to avoid lush growth and reduce freeze injury."
        ]
    if (hdd is not None and hdd > 120) or (head_t is not None and head_t > 26) or (fill_t is not None and fill_t > 24):
        tips += [
            "Ahead of heat waves at heading–filling, irrigate 1–2 days earlier to lower canopy temperature.",
            "Prefer heat-tolerant, lodging-resistant cultivars; consider mild growth regulation at jointing if overly lush."
        ]
    if ((head_w is not None and head_w > 6) or (fill_w is not None and fill_w > 6)) \
       and ((head_p is not None and head_p > 120) or (fill_p is not None and fill_p > 120)):
        tips += [
            "Cap nitrogen after jointing and consider growth regulators; set windbreaks or temporary staking.",
            "Drain excess water quickly after storms."
        ]
    if sow_t is not None and sow_t > 18:
        tips += ["At sowing under warm conditions, consider slightly earlier sowing or drought-tolerant cultivars; roll after sowing to retain moisture."]
    elif sow_t is not None and sow_t < 5:
        tips += ["Under cool sowing conditions, delay sowing or raise seeding rate by ~10–15% to secure establishment."]
    if (over_s is not None and over_s < 100) or (head_s is not None and head_s < 140) or (fill_s is not None and fill_s < 140):
        tips += ["With limited sunshine, slightly increase plant density and enhance disease surveillance; avoid excessive nitrogen that promotes lodging."]
    if not tips:
        tips = ["Conditions look typical; follow local best practices for cultivar and region."]

    md = ["**Climate summary**", summary, "", "**Recommendations**"]
    md += [f"- {t}" for t in list(dict.fromkeys(tips))]
    return "\n".join(md)

# ---------- single PDF ----------
def make_pdf_single(timestamp: str, predicted_yield: float, cluster_disp: str,
                    meta: Dict[str, Any], top_items: List[Tuple[str, float]], insight: str) -> bytes:
    try:
        from fpdf import FPDF
    except Exception as e:
        raise RuntimeError("fpdf2 not installed. `pip install fpdf2` to enable PDF export.") from e

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Wheat Yield Prediction — Single Report", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Generated at: {timestamp}", ln=1)
    pdf.cell(0, 8, f"Model: {cluster_disp}", ln=1)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Predicted Yield: {predicted_yield:.3f} tons/ha", ln=1)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Context:", ln=1)
    pdf.set_font("Arial", "", 11)
    for k in ["year","latitude","longitude","sown_area"]:
        if k in meta and meta[k] is not None:
            pdf.cell(0, 7, f"{k}: {meta[k]}", ln=1)

    if top_items:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Top SHAP features:", ln=1)
        pdf.set_font("Arial", "", 11)
        for name, val in top_items[:10]:
            pdf.cell(0, 6, f"{name}: {val:+.4f}", ln=1)

    if insight:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Insight:", ln=1)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, insight)

    out = io.BytesIO(); pdf.output(out)
    return out.getvalue()

# ---------- history store ----------
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dicts

# ---------- UI ----------
mode = st.sidebar.radio("Prediction Mode", ["Single prediction", "Batch prediction (CSV)"], index=0)

if mode == "Batch prediction (CSV)":
    st.subheader("Batch prediction")
    st.markdown("**CSV 需要包含以下列：**")
    st.code(", ".join(ALL_INPUT_COLS))

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df_in = pd.read_csv(up)
        out_rows = []
        for i, row in df_in.iterrows():
            feats_row = monthly_to_features(row)
            clu = None
            if regressors:
                if "cluster" in df_in.columns and not pd.isna(row.get("cluster", np.nan)):
                    try: clu = int(row["cluster"])
                    except: clu = None
                if clu is None and HAS_CLF:
                    clu = predict_cluster(feats_row)
            try:
                yhat, feats, model = predict_yield(clu, feats_row)
                rec = {"_row": i, "cluster": clu if clu is not None else "global", "predicted_yield": yhat}
                for extra in ["latitude","longitude","elevation_m","year","sown_area_hectare"]:
                    if extra in df_in.columns: rec[extra] = row.get(extra)
                out_rows.append(rec)
            except Exception as e:
                out_rows.append({"_row": i, "cluster": clu, "predicted_yield": np.nan, "_error": str(e)})

        res = pd.DataFrame(out_rows)
        st.dataframe(res, use_container_width=True)
        st.download_button("Download results CSV", res.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv")

        # 批量PDF
        try:
            from fpdf import FPDF
            def make_pdf_batch(rows: List[Dict[str, Any]]) -> bytes:
                pdf = FPDF()
                for r in rows:
                    pdf.add_page(); pdf.set_font("Arial", size=14)
                    pdf.cell(0, 10, "Wheat Yield Prediction Report", ln=1)
                    pdf.set_font("Arial", size=11)
                    for k in ["_row","cluster","predicted_yield","latitude","longitude","elevation_m","sown_area_hectare","year"]:
                        if k in r and r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k])):
                            pdf.multi_cell(0, 8, f"{k.replace('_',' ').title()}: {r[k]}")
                    if "_error" in r:
                        pdf.set_text_color(220,20,60); pdf.multi_cell(0,8,f"Error: {r['_error']}"); pdf.set_text_color(0,0,0)
                out = io.BytesIO(); pdf.output(out); return out.getvalue()
            pdf_bytes = make_pdf_batch(out_rows)
            st.download_button("Download Reports PDF", data=pdf_bytes,
                               file_name="reports.pdf", mime="application/pdf")
        except Exception:
            pass

else:
    st.subheader("Single prediction")

    # ---- 基本信息 ----
    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2018, step=1, key="in_year")
        sown_area_hectare = st.number_input("sown_area_hectare", min_value=0.0, value=1000.0, step=10.0, format="%.2f", key="in_sown")
    with c2:
        latitude = st.number_input("Latitude (°)", min_value=-90.0, max_value=90.0, value=34.75, step=0.01, format="%.2f", key="in_lat")
        longitude = st.number_input("Longitude (°)", min_value=-180.0, max_value=180.0, value=113.62, step=0.01, format="%.2f", key="in_lon")
    elevation_m = st.number_input("Elevation (m)", min_value=-500.0, max_value=9000.0, value=50.0, step=1.0, format="%.1f", key="in_elev")

    # ---- 自动填充 ----
    st.markdown("**Auto-fill tools**")
    cc1, cc2 = st.columns([1,2])
    with cc2:
        seed = st.number_input("Random seed (optional)", min_value=0, max_value=10_000_000, value=0, step=1, key="rand_seed")
        seed_val = int(seed) if seed != 0 else None
    with cc1:
        if st.button("Auto-fill monthly by lat/elev"):
            auto = autofill_monthly_by_lat_elev(latitude, elevation_m, seed=seed_val)
            for k, v in auto.items():
                st.session_state[f"in_{k}"] = v

    # ---- 初始化月度默认（防止小部件告警）----
    def init_monthly_defaults():
        for m in MONTHS:
            st.session_state.setdefault(f"in_tavg_{m}_C", 10.0)
            st.session_state.setdefault(f"in_tmax_{m}_C", 15.0)
            st.session_state.setdefault(f"in_tmin_{m}_C", 5.0)
            st.session_state.setdefault(f"in_precip_{m}_mm", 50.0)
            st.session_state.setdefault(f"in_sunshine_{m}_h", 160.0)
            st.session_state.setdefault(f"in_wind_{m}_ms", 3.5)
    init_monthly_defaults()

    # ---- 月度输入网格（只用 key，不传 value）----
    st.markdown("**Monthly climate inputs**")
    for row_start in range(0, 12, 3):
        cols = st.columns(3)
        for j, m in enumerate(MONTHS[row_start:row_start+3]):
            with cols[j]:
                st.markdown(f"**Month {m}**")
                st.number_input(f"tavg_{m}_C (°C)", key=f"in_tavg_{m}_C",
                                min_value=-50.0, max_value=60.0, step=0.1, format="%.1f")
                st.number_input(f"tmax_{m}_C (°C)", key=f"in_tmax_{m}_C",
                                min_value=-50.0, max_value=70.0, step=0.1, format="%.1f")
                st.number_input(f"tmin_{m}_C (°C)", key=f"in_tmin_{m}_C",
                                min_value=-70.0, max_value=50.0, step=0.1, format="%.1f")
                st.number_input(f"precip_{m}_mm (mm)", key=f"in_precip_{m}_mm",
                                min_value=0.0, max_value=1000.0, step=0.1, format="%.1f")
                st.number_input(f"sunshine_{m}_h (h)", key=f"in_sunshine_{m}_h",
                                min_value=0.0, max_value=744.0, step=0.1, format="%.1f")
                st.number_input(f"wind_{m}_ms (m/s)", key=f"in_wind_{m}_ms",
                                min_value=0.0, max_value=60.0, step=0.1, format="%.1f")

    # ---- 组装一行（月度 -> 模型特征）----
    raw_monthly = {
        "latitude": latitude, "longitude": longitude, "elevation_m": elevation_m,
        "sown_area_hectare": sown_area_hectare, "year": year
    }
    for m in MONTHS:
        for k in ["tavg","tmax","tmin"]:
            raw_monthly[f"{k}_{m}_C"] = float(st.session_state.get(f"in_{k}_{m}_C", np.nan))
        raw_monthly[f"precip_{m}_mm"]  = float(st.session_state.get(f"in_precip_{m}_mm", np.nan))
        raw_monthly[f"sunshine_{m}_h"] = float(st.session_state.get(f"in_sunshine_{m}_h", np.nan))
        raw_monthly[f"wind_{m}_ms"]    = float(st.session_state.get(f"in_wind_{m}_ms", np.nan))
    row_monthly = pd.Series(raw_monthly, dtype="float64")
    row_model_feats = monthly_to_features(row_monthly)

    # ---- 聚类选择/检测 ----
    clu = None
    if regressors:
        if HAS_CLF:
            if st.button("Detect cluster"):
                det = predict_cluster(row_model_feats); st.session_state["detected_cluster"] = det
                st.success(f"Detected cluster: {det}")
            clu = st.session_state.get("detected_cluster", predict_cluster(row_model_feats))
            st.caption(f"Cluster to use: {clu}")
        else:
            clu = st.selectbox("Select cluster", sorted(regressors.keys()))

    # ---- 预测 ----
    if st.button("Predict"):
        try:
            yhat, feats, model = predict_yield(clu, row_model_feats)
            st.markdown(f"### Predicted Yield: **{yhat:.3f} tons/ha**")

            contrib, feat_list, _ = shap_single(row_model_feats, feats, model, clu=clu)
            fig = _plot_shap_bar_abs(feat_list, contrib, clu=clu)  # 固定 absolute SHAP
            st.markdown("**SHAP explanation (Top-10 features):**")
            st.pyplot(fig)

            # |SHAP| 排序（PDF 用）；Insight 不依赖 SHAP
            order = np.argsort(np.abs(contrib))[::-1][:min(10, len(contrib))]
            shap_df = pd.DataFrame({
                "feature": [feat_list[i] for i in order],
                "shap": [float(contrib[i]) for i in order],
                "abs_shap": [float(abs(contrib[i])) for i in order]
            })

            # ====== 关键修复：Insight 用完整聚合行，不再 reindex(feats) ======
            insight = insight_text(clu, row_model_feats)
            st.markdown("**Climate summary & recommendations**")
            st.info(insight)

            # PDF
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cluster_disp = f"Clustered (cluster={clu})" if regressors else "Global model"
            shap_sorted = shap_df.sort_values("abs_shap", ascending=False)
            top_items = list(zip(shap_sorted["feature"].tolist(), shap_sorted["shap"].tolist()))
            meta = {"year": year, "latitude": latitude, "longitude": longitude, "sown_area": sown_area_hectare}
            try:
                pdf_bytes = make_pdf_single(ts, yhat, cluster_disp, meta, top_items, insight)
                st.download_button("Download PDF report", data=pdf_bytes,
                                   file_name=f"prediction_{ts.replace(':','-').replace(' ','_')}.pdf",
                                   mime="application/pdf")
            except Exception:
                pass

            # 历史记录
            rec = {
                "timestamp": ts,
                "model_type": "clustered" if regressors else "global",
                "cluster": clu if regressors else None,
                "predicted_yield": yhat,
                "year": year, "latitude": latitude, "longitude": longitude,
                "elevation_m": elevation_m, "sown_area_hectare": sown_area_hectare
            }
            for f in feats:
                rec[f] = row_model_feats.get(f, np.nan)
            st.session_state["history"].append(rec)

        except Exception as e:
            st.error(str(e))

    # ---- 历史记录 ----
    st.markdown("---")
    st.subheader("Prediction history (this session)")
    if st.session_state["history"]:
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("Download history CSV",
                           hist_df.to_csv(index=False).encode("utf-8"),
                           file_name="prediction_history.csv")
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.experimental_rerun()
    else:
        st.caption("No predictions yet.")
