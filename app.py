# app.py — Wheat Yield Prediction (cluster-aware with flat-file fallback)
# Features:
# - Monthly inputs -> agronomic features + monthly aliases for model compatibility
# - Absolute SHAP bar chart (no toggle)
# - English climate summary + actionable recommendations
# - Cluster auto-detect (button kept, no text shown)
# - Single & Batch prediction, PDF report, session history

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

# ---------- scan artifacts ----------
CWD = os.getcwd()
MODELS_DIR = "models"
REG_DIR = os.path.join(MODELS_DIR, "regressors")

# Classifier (optional)
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

# Per-cluster regressors
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

# Global regressor fallback
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

# ========== monthly input schema ==========
MONTHS = [f"{m:02d}" for m in range(1, 13)]
COLS_MONTHLY = []
for MM in MONTHS:
    COLS_MONTHLY += [
        f"tavg_{MM}_C",     # mean T (°C)
        f"tmax_{MM}_C",     # max T (°C)
        f"tmin_{MM}_C",     # min T (°C)
        f"precip_{MM}_mm",  # precip (mm)
        f"sunshine_{MM}_h", # sunshine (h)
        f"wind_{MM}_ms",    # wind (m/s)
    ]
BASE_INPUTS = ["latitude", "longitude", "elevation_m", "sown_area_hectare", "year"]
ALL_INPUT_COLS = BASE_INPUTS + COLS_MONTHLY

# ========== monthly -> model features ==========
PHASES = {
    "sowing":      ["10", "11"],
    "overwinter":  ["12", "01", "02"],
    "jointing":    ["03", "04"],
    "heading":     ["05"],
    "filling":     ["06"],
}

def _safe_mean(vals):
    vals = [v for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

def _safe_sum(vals):
    vals = [v for v in vals if pd.notna(v)]
    return float(np.sum(vals)) if vals else np.nan

def monthly_to_features(row: pd.Series) -> pd.Series:
    """
    Aggregate monthly climate to agronomic phases and also emit monthly aliases
    (tmean_k, tmean_MM, sun_k, wind_k/ws_k, precip_k) so models trained on
    monthly names like 'tmean_9','sun_12','wind_8' can work out-of-the-box.
    """
    out = {}

    # ---- phase aggregates ----
    for phase, months in PHASES.items():
        tavg = []; precip = []; sun = []; wind = []
        for MM in months:
            tavg.append(row.get(f"tavg_{MM}_C"))
            precip.append(row.get(f"precip_{MM}_mm"))
            sun.append(row.get(f"sunshine_{MM}_h"))
            wind.append(row.get(f"wind_{MM}_ms"))
        out[f"{phase}TmeanAvg"]  = _safe_mean(tavg)
        out[f"{phase}PrecipSum"] = _safe_sum(precip)
        out[f"{phase}SunHours"]  = _safe_sum(sun)
        out[f"{phase}WindAvg"]   = _safe_mean(wind)
        tavg_mean = out[f"{phase}TmeanAvg"]; precip_sum = out[f"{phase}PrecipSum"]
        out[f"dryness{phase.capitalize()}"] = (
            float(precip_sum) / (float(tavg_mean) + 10.0)
            if (pd.notna(precip_sum) and pd.notna(tavg_mean)) else np.nan
        )

    # ---- monthly aliases to match trained feature names ----
    for idx, MM in enumerate(MONTHS, start=1):
        m = str(idx)  # 1..12 without leading zero
        # temp mean
        v_t = row.get(f"tavg_{MM}_C")
        out[f"tmean_{m}"]  = v_t
        out[f"tmean_{MM}"] = v_t
        # precip
        v_p = row.get(f"precip_{MM}_mm")
        out[f"precip_{m}"]  = v_p
        out[f"precip_{MM}"] = v_p
        # sunshine
        v_s = row.get(f"sunshine_{MM}_h")
        out[f"sun_{m}"]  = v_s
        out[f"sun_{MM}"] = v_s
        # wind
        v_w = row.get(f"wind_{MM}_ms")
        out[f"wind_{m}"]  = v_w
        out[f"wind_{MM}"] = v_w
        out[f"ws_{m}"]    = v_w
        out[f"ws_{MM}"]   = v_w

    # ---- background ----
    out["latitude"]            = row.get("latitude")
    out["longitude"]           = row.get("longitude")
    out["elevation_m"]         = row.get("elevation_m")
    out["sown_area"]           = row.get("sown_area_hectare")  # for PDF
    out["sown_area_hectare"]   = row.get("sown_area_hectare")
    out["year"]                = row.get("year")
    return pd.Series(out, dtype="float64")

# ---------- random generators ----------
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
    for i, MM in enumerate(MONTHS, start=1):
        tavg = _sin_annual(i, amp, base) + np.random.normal(0, 1.1)
        tmax = tavg + 6 + np.random.normal(0, 0.9)
        tmin = tavg - 6 + np.random.normal(0, 0.9)
        precip   = max(0.0, (8 + 4*np.sin((i-1)/12*2*np.pi)) * (random.uniform(5, 20)))
        sunshine = max(0.0, 150 + 80*np.sin((i-1)/12*2*np.pi) + np.random.normal(0, 12))
        wind     = _clamp(random.uniform(2.0, 6.0) + np.random.normal(0, 0.6), 0.0, 30.0)
        out[f"tavg_{MM}_C"]     = round(float(tavg), 1)
        out[f"tmax_{MM}_C"]     = round(float(tmax), 1)
        out[f"tmin_{MM}_C"]     = round(float(tmin), 1)
        out[f"precip_{MM}_mm"]  = round(float(precip), 1)
        out[f"sunshine_{MM}_h"] = round(float(sunshine), 1)
        out[f"wind_{MM}_ms"]    = round(float(wind), 1)
    return out

# ---------- SHAP ----------
def shap_single(row_model_feats: pd.Series, feats: List[str], model: Any, clu: Optional[int]=None) -> Tuple[np.ndarray, List[str]]:
    X = row_model_feats.reindex(feats).fillna(row_model_feats.reindex(feats).dropna().median())
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
    return np.array(contrib, dtype=float), feats

def plot_shap_bar_abs(feats: List[str], contrib: np.ndarray, clu: Optional[int]):
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

# ---------- insight (EN: summary + actionable tips) ----------
def insight_text(_cluster_unused: Optional[int], row_model_feats: pd.Series) -> str:
    def gv(name):
        v = row_model_feats.get(name)
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    # Stage metrics (temp/precip/sun/wind only)
    sow_t   = gv("sowingTmeanAvg");      sow_p   = gv("sowingPrecipSum")
    over_t  = gv("overwinterTmeanAvg");  over_s  = gv("overwinterSunHours")
    joint_t = gv("jointingTmeanAvg");    joint_p = gv("jointingPrecipSum")
    head_t  = gv("headingTmeanAvg");     head_p  = gv("headingPrecipSum");   head_s = gv("headingSunHours"); head_w = gv("headingWindAvg")
    fill_t  = gv("fillingTmeanAvg");     fill_p  = gv("fillingPrecipSum");   fill_s = gv("fillingSunHours"); fill_w = gv("fillingWindAvg")

    # rollups for summary
    temp_pool = [x for x in [sow_t, over_t, joint_t, head_t, fill_t] if x is not None]
    season_t = float(np.mean(temp_pool)) if temp_pool else None
    precip_pool = [p for p in [sow_p, joint_p, head_p, fill_p] if p is not None]
    season_p = float(np.sum(precip_pool)) if precip_pool else None
    sun_pool = [s for s in [over_s, head_s, fill_s] if s is not None]
    season_s = float(np.sum(sun_pool)) if sun_pool else None
    wind_pool = [w for w in [head_w, fill_w] if w is not None]
    season_w = float(np.mean(wind_pool)) if wind_pool else None

    # buckets
    def b_temp(t):
        if t is None: return "typical temperatures"
        if t < 5:   return "very cool conditions"
        if t < 10:  return "cool conditions"
        if t < 18:  return "mild conditions"
        if t < 24:  return "warm conditions"
        return "hot conditions"
    def b_rain(p):
        if p is None: return "typical rainfall"
        if p < 300:  return "dry rainfall"
        if p <= 700: return "moderate rainfall"
        return "wet rainfall"
    def b_sun(s):
        if s is None: return "average sunshine"
        if s < 350:  return "limited sunshine"
        if s <= 650: return "average sunshine"
        return "plentiful sunshine"
    def b_wind(w):
        if w is None: return "light winds"
        if w < 2:    return "calm winds"
        if w <= 6:   return "breezy winds"
        return "windy conditions"

    summary = f"Overall, {b_temp(season_t)} and {b_rain(season_p)}; {b_sun(season_s)} with {b_wind(season_w)}."

    # actionable tips
    tips: List[str] = []
    dryness_joint = (joint_p is not None and joint_p < 40)
    dryness_head  = (head_p  is not None and head_p  < 40)
    dryness_fill  = (fill_p  is not None and fill_p  < 40)
    if dryness_joint or dryness_head or dryness_fill:
        tips += [
            "During jointing–grain filling, schedule light but frequent irrigations (every 5–7 days).",
            "After sowing, roll lightly or keep residue mulch to conserve moisture."
        ]
    if any(p is not None and p > 180 for p in [joint_p, head_p, fill_p]):
        tips += [
            "After heavy rain, open drainage furrows promptly, especially in low-lying fields.",
            "Scout closely for diseases around heading (e.g., Fusarium head blight) and treat in time."
        ]
    if (over_t is not None and over_t < 0):
        tips += [
            "Monitor cold snaps; implement cold-protection where feasible.",
            "Split N before winter to avoid lush growth and reduce freeze injury."
        ]
    if (head_t is not None and head_t > 26) or (fill_t is not None and fill_t > 24):
        tips += [
            "Ahead of heat waves at heading–filling, irrigate 1–2 days earlier to lower canopy temperature.",
            "Prefer heat-tolerant, lodging-resistant cultivars; consider mild growth regulation at jointing if overly lush."
        ]
    if ((head_w is not None and head_w > 6) or (fill_w is not None and fill_w > 6)) and \
       ((head_p is not None and head_p > 120) or (fill_p is not None and fill_p > 120)):
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

# ---------- PDF ----------
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

# ---------- history ----------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------- UI ----------
mode = st.sidebar.radio("Prediction Mode", ["Single prediction", "Batch prediction (CSV)"], index=0)

if mode == "Batch prediction (CSV)":
    st.subheader("Batch prediction")
    st.markdown("**CSV must contain these columns:**")
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

        # Batch PDF
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

    # ---- basics ----
    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2018, step=1, key="in_year")
        sown_area_hectare = st.number_input("sown_area_hectare", min_value=0.0, value=1000.0, step=10.0, format="%.2f", key="in_sown")
    with c2:
        latitude = st.number_input("Latitude (°)", min_value=-90.0, max_value=90.0, value=34.75, step=0.01, format="%.2f", key="in_lat")
        longitude = st.number_input("Longitude (°)", min_value=-180.0, max_value=180.0, value=113.62, step=0.01, format="%.2f", key="in_lon")
    elevation_m = st.number_input("Elevation (m)", min_value=-500.0, max_value=9000.0, value=50.0, step=1.0, format="%.1f", key="in_elev")

    # ---- autofill ----
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

    # ---- init monthly defaults (avoid widget warning) ----
    def init_monthly_defaults():
        for MM in MONTHS:
            st.session_state.setdefault(f"in_tavg_{MM}_C", 10.0)
            st.session_state.setdefault(f"in_tmax_{MM}_C", 15.0)
            st.session_state.setdefault(f"in_tmin_{MM}_C", 5.0)
            st.session_state.setdefault(f"in_precip_{MM}_mm", 50.0)
            st.session_state.setdefault(f"in_sunshine_{MM}_h", 160.0)
            st.session_state.setdefault(f"in_wind_{MM}_ms", 3.5)
    init_monthly_defaults()

    # ---- monthly inputs grid ----
    st.markdown("**Monthly climate inputs**")
    for row_start in range(0, 12, 3):
        cols = st.columns(3)
        for j, MM in enumerate(MONTHS[row_start:row_start+3]):
            with cols[j]:
                st.markdown(f"**Month {MM}**")
                st.number_input(f"tavg_{MM}_C (°C)", key=f"in_tavg_{MM}_C",
                                min_value=-50.0, max_value=60.0, step=0.1, format="%.1f")
                st.number_input(f"tmax_{MM}_C (°C)", key=f"in_tmax_{MM}_C",
                                min_value=-50.0, max_value=70.0, step=0.1, format="%.1f")
                st.number_input(f"tmin_{MM}_C (°C)", key=f"in_tmin_{MM}_C",
                                min_value=-70.0, max_value=50.0, step=0.1, format="%.1f")
                st.number_input(f"precip_{MM}_mm (mm)", key=f"in_precip_{MM}_mm",
                                min_value=0.0, max_value=1000.0, step=0.1, format="%.1f")
                st.number_input(f"sunshine_{MM}_h (h)", key=f"in_sunshine_{MM}_h",
                                min_value=0.0, max_value=744.0, step=0.1, format="%.1f")
                st.number_input(f"wind_{MM}_ms (m/s)", key=f"in_wind_{MM}_ms",
                                min_value=0.0, max_value=60.0, step=0.1, format="%.1f")

    # ---- assemble row & aggregate ----
    raw_monthly = {
        "latitude": latitude, "longitude": longitude, "elevation_m": elevation_m,
        "sown_area_hectare": sown_area_hectare, "year": year
    }
    for MM in MONTHS:
        for k in ["tavg","tmax","tmin"]:
            raw_monthly[f"{k}_{MM}_C"] = float(st.session_state.get(f"in_{k}_{MM}_C", np.nan))
        raw_monthly[f"precip_{MM}_mm"]  = float(st.session_state.get(f"in_precip_{MM}_mm", np.nan))
        raw_monthly[f"sunshine_{MM}_h"] = float(st.session_state.get(f"in_sunshine_{MM}_h", np.nan))
        raw_monthly[f"wind_{MM}_ms"]    = float(st.session_state.get(f"in_wind_{MM}_ms", np.nan))
    row_monthly = pd.Series(raw_monthly, dtype="float64")
    row_model_feats = monthly_to_features(row_monthly)

    # ---- cluster selection/detection (keep button, show no text) ----
    clu = None
    if regressors:
        if HAS_CLF:
            if st.button("Detect cluster"):
                det = predict_cluster(row_model_feats)
                st.session_state["detected_cluster"] = det   # no UI text
            clu = st.session_state.get("detected_cluster", predict_cluster(row_model_feats))
        else:
            clu = st.selectbox("Select cluster", sorted(regressors.keys()))

    # ---- predict ----
    if st.button("Predict"):
        try:
            yhat, feats, model = predict_yield(clu, row_model_feats)
            st.markdown(f"### Predicted Yield: **{yhat:.3f} tons/ha**")

            contrib, feat_list = shap_single(row_model_feats, feats, model, clu=clu)
            fig = plot_shap_bar_abs(feat_list, contrib, clu=clu)
            st.markdown("**SHAP explanation (Top-10 features):**")
            st.pyplot(fig)

            order = np.argsort(np.abs(contrib))[::-1][:min(10, len(contrib))]
            shap_df = pd.DataFrame({
                "feature": [feat_list[i] for i in order],
                "abs_shap": [float(abs(contrib[i])) for i in order],
                "shap": [float(contrib[i]) for i in order]
            })

            # Insight (uses full aggregated row)
            insight = insight_text(clu, row_model_feats)
            st.markdown("**Climate summary & recommendations**")
            st.info(insight)

            # PDF
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cluster_disp = "Clustered model" if regressors else "Global model"  # no cluster id in PDF
            top_items = list(zip(shap_df["feature"].tolist(), shap_df["shap"].tolist()))
            meta = {"year": year, "latitude": latitude, "longitude": longitude, "sown_area": sown_area_hectare}
            try:
                pdf_bytes = make_pdf_single(ts, yhat, cluster_disp, meta, top_items, insight)
                st.download_button("Download PDF report", data=pdf_bytes,
                                   file_name=f"prediction_{ts.replace(':','-').replace(' ','_')}.pdf",
                                   mime="application/pdf")
            except Exception:
                pass

            # history
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

    # ---- history ----
    st.markdown("---")
    st.subheader("Prediction history (this session)")
    if st.session_state["history"]:
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("Download history CSV",
                           hist_df.to_csv(index=False).encode("utf-8"),
                           file_name="prediction_history.csv")
        # Streamlit rerun compatibility
        rerun_fn = getattr(st, "experimental_rerun", getattr(st, "rerun", None))
        if st.button("Clear history"):
            st.session_state["history"] = []
            if callable(rerun_fn): rerun_fn()
    else:
        st.caption("No predictions yet.")
