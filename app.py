import os, json, io, datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

# =================== Page & Theme ===================
st.set_page_config(page_title="Wheat Yield Predictor (Clustered)", layout="centered")
st.title("Wheat Yield Prediction (kg/hectare) — Cluster-Aware")

# =================== Paths & Load ===================
BASE = "models"
REG_DIR = os.path.join(BASE, "regressors")

def load_clf(path: str) -> CatBoostClassifier:
    m = CatBoostClassifier(); m.load_model(path); return m

def load_reg(path: str) -> CatBoostRegressor:
    m = CatBoostRegressor(); m.load_model(path); return m

req = {
    "cluster_model": os.path.join(BASE, "cluster_model.cbm"),
    "cluster_features": os.path.join(BASE, "cluster_features.json"),
    "clusters": os.path.join(BASE, "clusters.json"),
}
missing = [k for k, p in req.items() if not os.path.exists(p)]
if missing:
    st.error(f"Missing model artifacts: {missing}. Please train/export clustered models first.")
    st.stop()

clf = load_clf(req["cluster_model"])
with open(req["cluster_features"], "r", encoding="utf-8") as f:
    clf_features: list[str] = json.load(f)

with open(req["clusters"], "r", encoding="utf-8") as f:
    clusters: list[int] = [int(x) for x in json.load(f)]

regressors: dict[int, dict] = {}
for clu in clusters:
    m_path = os.path.join(REG_DIR, f"cluster_{clu}.cbm")
    f_path = os.path.join(REG_DIR, f"top_features_cluster_{clu}.json")
    if os.path.exists(m_path) and os.path.exists(f_path):
        model = load_reg(m_path)
        feats = json.load(open(f_path, "r", encoding="utf-8"))
        regressors[int(clu)] = {"model": model, "features": feats}

if not regressors:
    st.error("No per-cluster regressors loaded under models/regressors/.")
    st.stop()

# union for UI display (keep order)
COMMON_WEATHER = [
    "sowingTmeanAvg","sowingPrecipSum","sowingSunHours","sowingWindAvg",
    "overwinterTmeanAvg","overwinterPrecipSum","overwinterSunHours","overwinterWindAvg",
    "jointingTmeanAvg","jointingPrecipSum","jointingSunHours","jointingWindAvg",
    "headingTmeanAvg","headingPrecipSum","headingSunHours","headingWindAvg",
    "fillingTmeanAvg","fillingPrecipSum","fillingSunHours","fillingWindAvg",
    "drynessJointing","drynessHeading","drynessFilling","gddBase5","hddGt30","cddLt0",
]
all_reg_feats = sorted(set(sum([v["features"] for v in regressors.values()], [])))
DISPLAY_FEATURES = list(dict.fromkeys(COMMON_WEATHER + clf_features + all_reg_feats))

# =================== Helpers ===================
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
        "Other Features":    [c for c in cols if not any(x in c.lower() for x in
                                ["sowing","overwinter","jointing","heading","filling","dryness","gdd","hdd","cdd"])],
    }
    return {k:v for k,v in g.items() if v}

def predict_cluster(row: pd.Series) -> int:
    r = row.reindex(clf_features)
    med = r.dropna().median()
    r = r.fillna(med)
    proba = clf.predict(Pool(pd.DataFrame([r], columns=clf_features)))
    return int(proba[0])

def predict_yield_for_cluster(clu: int, row: pd.Series) -> float:
    feats = regressors[clu]["features"]
    X = row.reindex(feats)
    med = X.dropna().median()
    X = X.fillna(med)
    yhat = float(regressors[clu]["model"].predict(Pool(pd.DataFrame([X], columns=feats)))[0])
    return yhat

def simple_insight(row: pd.Series) -> str:
    t = float(row.get("fillingTmeanAvg", np.nan)) if row.get("fillingTmeanAvg") is not None else np.nan
    p = float(row.get("fillingPrecipSum", np.nan)) if row.get("fillingPrecipSum") is not None else np.nan
    if not np.isnan(t) and not np.isnan(p):
        if t >= 24 and p <= 60: return "Insight: Hot and dry climate. Drought-resistant varieties perform better."
        if t <= 16 and p >= 100: return "Insight: Cool and wet conditions. Slow-maturing varieties may be favored."
    return "Insight: Adjust cultivar and management according to seasonal temperature and rainfall."

def figure_cluster_importance(clu: int):
    feats = regressors[clu]["features"]
    model = regressors[clu]["model"]
    try:
        fi = model.get_feature_importance()
    except Exception:
        fi = np.zeros(len(feats))
    order = np.argsort(fi)[::-1][:min(15, len(feats))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([feats[i] for i in order][::-1], [fi[i] for i in order][::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Cluster {clu} — Top features")
    fig.tight_layout()
    return fig

def make_pdf(report: dict) -> bytes:
    try:
        from fpdf import FPDF
    except Exception as e:
        raise RuntimeError("fpdf2 not installed. `pip install fpdf2` to enable PDF export.") from e
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Wheat Yield Prediction Report", ln=1)
    pdf.set_font("Arial", size=11)
    for k in ["timestamp","cluster","predicted_yield","latitude","longitude","sown_area","year","insight"]:
        if k in report:
            pdf.multi_cell(0, 8, f"{k.replace('_',' ').title()}: {report[k]}")
    pdf.ln(4); pdf.set_font("Arial", "B", 12); pdf.cell(0, 8, "Key Inputs:", ln=1)
    pdf.set_font("Arial", size=10)
    for k,v in report.get("inputs", {}).items():
        pdf.multi_cell(0, 6, f"- {k}: {v}")
    out = io.BytesIO(); pdf.output(out); return out.getvalue()

# =================== Sidebar: Prediction Mode ===================
mode = st.sidebar.radio("Prediction Mode", ["Single prediction", "Batch prediction (CSV)"], index=0)

# init history
if "history" not in st.session_state:
    st.session_state["history"] = []

# =================== Main Content ===================
if mode == "Batch prediction (CSV)":
    st.subheader("Clustered Batch Prediction")
    st.markdown("If your CSV has a `cluster` column, it will be used; otherwise the app will auto-detect the cluster.")
    st.markdown("**Required columns for cluster detection (when `cluster` is missing):**")
    st.code(", ".join(sorted(set(clf_features))))
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
        out_rows = []
        for i, row in df.iterrows():
            # detect or use provided cluster
            clu = None
            if "cluster" in df.columns and not pd.isna(row["cluster"]):
                try: clu = int(row["cluster"])
                except: clu = None
            if clu is None:
                clu = predict_cluster(row)

            if clu not in regressors:
                out_rows.append({"_row": i, "cluster": clu, "predicted_yield": np.nan, "_error":"no model for this cluster"})
                continue

            yhat = predict_yield_for_cluster(clu, row)
            out_rows.append({"_row": i, "cluster": clu, "predicted_yield": yhat})

        res = pd.DataFrame(out_rows)
        st.markdown("Predictions:")
        st.dataframe(res)
        st.download_button("Download results CSV", res.to_csv(index=False).encode("utf-8"),
                           file_name="clustered_predictions.csv")

else:
    # ======= Tabs for single prediction =======
    tab_predict, tab_map, tab_cluster, tab_hist = st.tabs(["Predict", "Climate Map", "Cluster Plot", "History"])

    # -------- Predict Tab --------
    with tab_predict:
        st.subheader("Inputs")

        # Basic context (not necessarily in the model; for report & map)
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Year", key="in_year", min_value=1900, max_value=2100, value=2018, step=1)
            sown_area = st.number_input("Sown area (ha)", key="in_sown_area", min_value=0.0, value=1000.0, step=10.0, format="%.2f")
        with col2:
            latitude = st.number_input("Latitude", key="in_latitude", min_value=-90.0, max_value=90.0, value=34.75, step=0.01, format="%.2f")
            longitude = st.number_input("Longitude", key="in_longitude", min_value=-180.0, max_value=180.0, value=113.62, step=0.01, format="%.2f")

        st.checkbox("Auto-detect climate cluster", value=True, key="auto_detect_cluster")

        # Weather & other features (collapsible)
        with st.expander("Weather and feature inputs"):
            groups = group_features(DISPLAY_FEATURES)
            user_input = {}
            rendered = set()
            for gname, cols in groups.items():
                st.markdown(f"**{gname}**")
                cols3 = st.columns(3)
                for i, c in enumerate(cols):
                    if c in rendered: continue
                    rendered.add(c)
                    with cols3[i % 3]:
                        key = f"in_{c}"
                        lc = c.lower()
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
                        val = st.number_input(**kwargs)
                        user_input[c] = val

        # Build a single row (all possible keys)
        current_row = {k.replace("in_",""): v for k, v in st.session_state.items() if k.startswith("in_")}
        row_series = pd.Series(current_row, dtype="float64")  # non-numeric will become NaN, OK for CatBoost

        # Detect cluster (button)
        if st.button("Detect cluster"):
            detected = predict_cluster(row_series)
            st.session_state["detected_cluster"] = detected
            st.success(f"Detected cluster: {detected}")

        # Decide cluster
        if st.session_state.get("auto_detect_cluster", True):
            cluster_to_use = st.session_state.get("detected_cluster", predict_cluster(row_series))
        else:
            cluster_to_use = st.selectbox("Select cluster", options=sorted(regressors.keys()), key="manual_cluster")

        st.write("")  # spacer

        # Predict button
        if st.button("Predict"):
            if cluster_to_use not in regressors:
                st.error("No regressor available for the selected/predicted cluster.")
            else:
                yhat = predict_yield_for_cluster(cluster_to_use, row_series)
                st.markdown(f"### Predicted Yield: **{yhat:.3f} tons/ha**")

                insight = simple_insight(row_series)
                st.write(insight)

                # Save to history
                rec = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "cluster": int(cluster_to_use),
                    "predicted_yield": float(yhat),
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "sown_area": float(sown_area),
                    "year": int(year),
                    "insight": insight,
                    "inputs": {k: row_series.get(k, None) for k in DISPLAY_FEATURES}
                }
                st.session_state["history"].append(rec)

                # PDF download
                try:
                    pdf_bytes = make_pdf(rec)
                    st.download_button("Download PDF", data=pdf_bytes, file_name="yield_report.pdf", mime="application/pdf")
                except Exception as e:
                    st.info(str(e))

    # -------- Climate Map Tab --------
    with tab_map:
        st.subheader("Climate Map")
        if "in_latitude" in st.session_state and "in_longitude" in st.session_state:
            lat = float(st.session_state["in_latitude"])
            lon = float(st.session_state["in_longitude"])
            st.map(pd.DataFrame({"lat":[lat], "lon":[lon]}), latitude="lat", longitude="lon", zoom=6)
            clu = st.session_state.get("detected_cluster", None)
            if clu is not None:
                st.caption(f"Location marker. Detected cluster: {clu}")
        else:
            st.info("Provide latitude and longitude in the Predict tab to show the map.")

    # -------- Cluster Plot Tab --------
    with tab_cluster:
        st.subheader("Cluster Plot")
        clu = st.session_state.get("detected_cluster", None)
        if clu is None:
            row_series = pd.Series({k.replace("in_",""): v for k, v in st.session_state.items() if k.startswith("in_")})
            clu = predict_cluster(row_series)
        if clu not in regressors:
            st.info("No regressor for this cluster.")
        else:
            fig = figure_cluster_importance(clu)
            st.pyplot(fig)

    # -------- History Tab --------
    with tab_hist:
        st.subheader("History")
        if not st.session_state["history"]:
            st.info("No predictions yet.")
        else:
            hist_df = pd.DataFrame([
                {k: v for k, v in r.items() if k != "inputs"} for r in st.session_state["history"]
            ])
            st.dataframe(hist_df)
            st.download_button("Download History CSV",
                               hist_df.to_csv(index=False).encode("utf-8"),
                               file_name="prediction_history.csv")
            if st.button("Clear history"):
                st.session_state["history"] = []
                st.experimental_rerun()
