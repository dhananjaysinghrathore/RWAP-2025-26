# streamlit_app.py — RWAP Project (auto-load data, no upload)
# Place your CSV (or CSV.GZ) at:  repo_root/data/asset_valuation_results_final_with_confidence.csv.gz
# Optional: set a Streamlit secret DATA_URL with a public link to the CSV/CSV.GZ

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# ---------- Page setup ----------
st.set_page_config(page_title="RWAP – Dashboard & ML", page_icon="📊", layout="wide")
st.title("RWAP – Asset Valuation Dashboard & ML")

# ---------- 0) Auto-load data (CSV.GZ or CSV, with URL fallback) ----------
# You can change the base filename via env var DATA_FILE_BASE if you want.
DATA_FILE_BASE = os.getenv("DATA_FILE_BASE", "asset_valuation_results_final_with_confidence")
DATA_DIR = Path(__file__).parent / "data"
CANDIDATE_FILES = [
    DATA_DIR / f"{DATA_FILE_BASE}.csv.gz",
    DATA_DIR / f"{DATA_FILE_BASE}.csv",
]

@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    # 1) Local files inside repo /data folder
    for p in CANDIDATE_FILES:
        if p.exists():
            return pd.read_csv(p)  # pandas auto-handles .gz
    # 2) Optional: public URL via Streamlit Secrets
    url = st.secrets.get("DATA_URL", "")
    if url:
        return pd.read_csv(url)
    # 3) Nothing found
    st.error(
        "Data file not found.\n\n"
        f"Expected one of:\n- {CANDIDATE_FILES[0].as_posix()}\n- {CANDIDATE_FILES[1].as_posix()}\n\n"
        "Upload your CSV/CSV.GZ to the repo under `/data/`, or set a public link in Secrets as `DATA_URL`."
    )
    st.stop()

raw = load_data()

# ---------- 1) Helpers ----------
def first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map many possible source column names to a canonical schema,
    coerce numerics, and compute value_psf.
    """
    mapping = {
        "loc_code":   ["Location Code", "loc_code"],
        "asset_name": ["Real Property Asset Name", "Asset Name", "asset_name"],
        "city":       ["City", "City_x", "city"],
        "state":      ["State", "State_x", "state"],
        "zip":        ["Zip Code", "ZIP", "zip"],
        "lat":        ["Latitude", "Latitude_x", "lat"],
        "lon":        ["Longitude", "Longitude_x", "lon"],
        "sqft":       ["Building Rentable Square Feet", "SqFt", "sqft"],
        "value":      ["Estimated Asset Value (Adj)", "Estimated Asset Value", "Est Value (Base)", "value"],
        "conf_cat":   ["Confidence Category", "conf_cat"],
        "asset_type": ["Real Property Asset Type", "Asset Type", "Type"],
        "age":        ["Building Age", "age"],
        "cluster":    ["Asset Cluster", "cluster"],
    }
    out = pd.DataFrame()
    for k, cands in mapping.items():
        hit = first_col(df, cands)
        if hit is not None:
            out[k] = df[hit]

    # to numeric where relevant
    for n in ["lat", "lon", "sqft", "value", "age"]:
        if n in out.columns:
            out[n] = pd.to_numeric(out[n], errors="coerce")

    # compute value per sq.ft.
    if "value" in out.columns and "sqft" in out.columns:
        out["value_psf"] = out["value"] / out["sqft"]
        out["value_psf"].replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        out["value_psf"] = np.nan

    # default confidence category if missing
    if "conf_cat" not in out.columns:
        out["conf_cat"] = "Unknown"

    return out

df = canonicalize(raw)

# Tabs: Task 2 (Dashboard), Task 3 (ML)
tab_dash, tab_ml = st.tabs(["📊 Task 2: Dashboard", "🤖 Task 3: ML"])

# =====================================================
# ============== TAB 1 — DASHBOARD ====================
# =====================================================
with tab_dash:
    # Small header showing where data came from
    found_path = next((p for p in CANDIDATE_FILES if p.exists()), None)
    if found_path:
        st.caption(f"Loaded **{len(df):,}** rows from `{found_path.as_posix()}`")
    else:
        src = st.secrets.get("DATA_URL", "")
        st.caption(f"Loaded **{len(df):,}** rows from DATA_URL: {src}")

    st.subheader("Filters")

    states = sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else []
    types  = sorted(df["asset_type"].dropna().unique().tolist()) if "asset_type" in df.columns else []
    confs  = sorted(df["conf_cat"].dropna().unique().tolist()) if "conf_cat" in df.columns else []

    c1, c2, c3, c4 = st.columns(4)
    sel_states = c1.multiselect("State", states, default=states[:10] if states else [])
    sel_types  = c2.multiselect("Asset Type", types, default=types if types else [])
    sel_confs  = c3.multiselect("Confidence", confs, default=confs if confs else [])
    name_q     = c4.text_input("Search asset/ZIP contains", "")

    # numeric sliders
    flt = df.copy()
    if "value" in df.columns and df["value"].notna().any():
        vmin, vmax = float(df["value"].min()), float(df["value"].max())
        v_range = st.slider("Value range", vmin, vmax, (vmin, vmax))
    else:
        v_range = None

    if "sqft" in df.columns and df["sqft"].notna().any():
        smin, smax = float(df["sqft"].min()), float(df["sqft"].max())
        s_range = st.slider("Sq.Ft range", smin, smax, (smin, smax))
    else:
        s_range = None

    # apply filters
    if sel_states and "state" in flt.columns:
        flt = flt[flt["state"].isin(sel_states)]
    if sel_types and "asset_type" in flt.columns:
        flt = flt[flt["asset_type"].isin(sel_types)]
    if sel_confs and "conf_cat" in flt.columns:
        flt = flt[flt["conf_cat"].isin(sel_confs)]
    if name_q:
        m = False
        if "asset_name" in flt.columns:
            m = flt["asset_name"].astype(str).str.contains(name_q, case=False, na=False)
        if "zip" in flt.columns:
            m = (m | flt["zip"].astype(str).str.contains(name_q, na=False)) if isinstance(m, pd.Series) else flt["zip"].astype(str).str.contains(name_q, na=False)
        if isinstance(m, pd.Series):
            flt = flt[m]
    if v_range and "value" in flt.columns:
        flt = flt[(flt["value"] >= v_range[0]) & (flt["value"] <= v_range[1])]
    if s_range and "sqft" in flt.columns:
        flt = flt[(flt["sqft"] >= s_range[0]) & (flt["sqft"] <= s_range[1])]

    # KPIs
    st.markdown("### KPIs (Filtered)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Assets", f"{len(flt):,}")
    k2.metric("Total Value", f"${flt['value'].sum():,.0f}" if "value" in flt.columns else "—")
    k3.metric("Median Value", f"${flt['value'].median():,.0f}" if "value" in flt.columns else "—")
    k4.metric("Median $/ft²", f"${flt['value_psf'].median():,.0f}" if "value_psf" in flt.columns else "—")

    # Distributions
    st.markdown("### Distributions")
    cc1, cc2 = st.columns(2)
    if "value" in flt.columns:
        fig_val = px.histogram(flt, x="value", nbins=60, title="Asset Value")
        fig_val.update_xaxes(type="log")
        cc1.plotly_chart(fig_val, use_container_width=True)
    if "value_psf" in flt.columns:
        fig_psf = px.histogram(
            flt.replace([np.inf, -np.inf], np.nan).dropna(subset=["value_psf"]),
            x="value_psf", nbins=60, title="Value per Sq.Ft"
        )
        cc2.plotly_chart(fig_psf, use_container_width=True)

    # Value vs Size
    st.markdown("### Value vs Size (log-log)")
    if all(c in flt.columns for c in ["sqft", "value"]):
        fig_sc = px.scatter(
            flt,
            x="sqft", y="value",
            hover_data=[c for c in ["asset_name", "state", "zip"] if c in flt.columns],
            title="Value vs Rentable Sq.Ft"
        )
        fig_sc.update_xaxes(type="log")
        fig_sc.update_yaxes(type="log")
        st.plotly_chart(fig_sc, use_container_width=True)

    # Map
    st.markdown("### Map")
    if all(c in flt.columns for c in ["lat", "lon"]) and flt[["lat", "lon"]].notna().all(axis=1).any():
        v = np.clip(flt["value"].fillna(1.0).astype(float), 1.0, None) if "value" in flt.columns else pd.Series([1.0] * len(flt))
        vmed = float(np.median(v)) if len(v) else 1.0
        vmed = max(vmed, 1.0)
        radius = 200 + 200 * np.log10(v / vmed)
        data = flt.copy()
        data["radius"] = radius
        data["color"] = [[50, 100, 200]] * len(data)

        center_lat = float(data["lat"].dropna().mean()) if data["lat"].notna().any() else 39.5
        center_lon = float(data["lon"].dropna().mean()) if data["lon"].notna().any() else -98.35

        view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=4)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=data,
            get_position='[lon, lat]',
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
        tooltip = {
            "html": "<b>{asset_name}</b><br/>${value:,.0f}<br/>{state} {zip}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip), use_container_width=True)
    else:
        st.info("No lat/lon columns found for mapping.")

    # Tables & Download
    st.markdown("### Top 50 by Value")
    if "value" in flt.columns:
        top = flt.sort_values("value", ascending=False).head(50)
        st.dataframe(top, use_container_width=True)

    st.download_button(
        "⬇️ Download filtered CSV",
        data=flt.to_csv(index=False),
        file_name="filtered_assets.csv",
        mime="text/csv",
    )

# =====================================================
# ============== TAB 2 — ML (TASK 3) =================
# =====================================================
with tab_ml:
    st.subheader("Clustering (KMeans) + Classification (RandomForest)")

    # Need value, sqft, value_psf (age optional)
    must = ["value", "sqft", "value_psf"]
    if not all(c in df.columns for c in must):
        st.warning("CSV must include: value, sqft, value_psf (auto-generated if value & sqft exist).")
        st.stop()

    ml = df[["value", "sqft", "value_psf"] + (["age"] if "age" in df.columns else [])].copy()
    ml = ml.replace([np.inf, -np.inf], np.nan)
    ml = ml.fillna(ml.median(numeric_only=True))

    # logs for skewed positives
    for c in ["value", "sqft", "value_psf"]:
        ml[f"log_{c}"] = np.log(np.clip(ml[c], 1, None))

    X_cols = [c for c in ml.columns if c.startswith("log_")]
    if "age" in ml.columns:
        X_cols.append("age")

    scaler = StandardScaler()
    X = scaler.fit_transform(ml[X_cols])

    # Choose k for KMeans
    k = st.slider("Choose number of clusters (k)", 3, 8, 3, 1)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    st.write(f"**Silhouette score:** {sil:.3f}")

    df_ml = df.copy()
    df_ml["Asset Cluster"] = labels

    # Cluster sizes and profiles
    sizes = pd.Series(labels).value_counts().sort_index().rename("count")
    st.write("**Cluster sizes:**")
    st.dataframe(sizes)

    prof_cols = [c for c in ["value", "sqft", "value_psf", "age"] if c in df_ml.columns]
    profile = df_ml.groupby("Asset Cluster")[prof_cols].median().sort_index()
    st.write("**Cluster profiles (median):**")
    st.dataframe(profile)

    # PCA 2D viz
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    figp = px.scatter(
        pd.DataFrame({"PC1": X2[:, 0], "PC2": X2[:, 1], "Cluster": labels}),
        x="PC1", y="PC2", color="Cluster", title="PCA (2D) by Cluster", opacity=0.7, height=500
    )
    st.plotly_chart(figp, use_container_width=True)

    st.download_button(
        "⬇️ Download data with clusters",
        data=df_ml.to_csv(index=False),
        file_name="t3_assets_with_clusters.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("RandomForest: predict cluster")

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.30, random_state=42, stratify=labels
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = (y_pred == y_test).mean()
    st.write(f"**Accuracy:** {acc:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {i}" for i in sorted(np.unique(labels))],
        columns=[f"Pred {i}" for i in sorted(np.unique(labels))],
    )
    st.write("**Confusion Matrix:**")
    st.dataframe(cm_df)

    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

    imp = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False)
    fig_imp = px.bar(
        imp, title="Feature Importances (RandomForest)", labels={"index": "feature", "value": "importance"}
    )
    st.plotly_chart(fig_imp, use_container_width=True)

st.caption(
    "Auto-loaded CSV from `/data/` (tries .csv.gz then .csv). "
    "To host externally, set a Streamlit Secret named `DATA_URL` with a public CSV/CSV.GZ link."
)
