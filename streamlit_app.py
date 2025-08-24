# streamlit_app.py — RWAP (Bn/Mn KPIs, Map above charts, selectable map colors)
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

# -------------------- Page setup --------------------
st.set_page_config(page_title="RWAP – Dashboard & ML", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.1rem; padding-bottom: 1.1rem;}
      .metric {text-align:center;}
      .legend-swatch {display:inline-block; width:14px; height:14px; border-radius:3px; margin-right:8px;}
      .legend-item {margin-right:14px; display:inline-flex; align-items:center;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("RWAP – Asset Valuation Dashboard & ML")

# -------------------- Data load (auto) --------------------
DATA_FILE_BASE = os.getenv("DATA_FILE_BASE", "asset_valuation_results_final_with_confidence")
DATA_DIR = Path(__file__).parent / "data"
CANDIDATE_FILES = [
    DATA_DIR / f"{DATA_FILE_BASE}.csv.gz",
    DATA_DIR / f"{DATA_FILE_BASE}.csv",
]

@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    for p in CANDIDATE_FILES:
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            return df
    url = st.secrets.get("DATA_URL", "")
    if url:
        df = pd.read_csv(url)
        df.columns = [c.strip() for c in df.columns]
        return df
    st.error(
        "Data file not found.\n\n"
        f"Expected one of:\n- {CANDIDATE_FILES[0].as_posix()}\n- {CANDIDATE_FILES[1].as_posix()}\n\n"
        "Upload CSV/CSV.GZ to `/data/`, or set a public link in Secrets as `DATA_URL`."
    )
    st.stop()

raw = load_data()

# -------------------- Helpers --------------------
def first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
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

    for n in ["lat", "lon", "sqft", "value", "age"]:
        if n in out.columns:
            out[n] = pd.to_numeric(out[n], errors="coerce")

    if "value" in out.columns and "sqft" in out.columns:
        out["value_psf"] = out["value"] / out["sqft"]
        out["value_psf"].replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        out["value_psf"] = np.nan

    if "conf_cat" not in out.columns:
        out["conf_cat"] = "Unknown"
    return out

def fmt_money_units(x: float) -> str:
    """Format as $X.XX Bn / Mn (else plain $ with separators)."""
    try:
        x = float(x)
    except Exception:
        return "—"
    if np.isnan(x):
        return "—"
    absx = abs(x)
    if absx >= 1e9:
        return f"${x/1e9:,.2f} Bn"
    if absx >= 1e6:
        return f"${x/1e6:,.2f} Mn"
    return f"${x:,.0f}"

BASE_PALETTE = [
    [230,57,70], [29,53,87], [69,123,157], [42,157,143], [233,196,106],
    [244,162,97], [231,111,81], [94,79,162], [0,119,182], [34,197,94],
    [148,163,184], [217,70,239], [99,102,241], [245,158,11]
]

def color_map_for(series: pd.Series):
    cats = series.fillna("Unknown").astype(str).unique().tolist()
    cats = sorted(cats)
    cmap = {c: BASE_PALETTE[i % len(BASE_PALETTE)] for i, c in enumerate(cats)}
    colors = series.fillna("Unknown").astype(str).map(cmap).tolist()
    return colors, cmap

df = canonicalize(raw)

# Tabs
tab_dash, tab_ml = st.tabs(["📊 Task 2: Dashboard", "🤖 Task 3: ML"])

# =====================================================
# ================= TAB 1 – DASHBOARD =================
# =====================================================
with tab_dash:
    # Source hint
    found_path = next((p for p in CANDIDATE_FILES if p.exists()), None)
    st.caption(
        f"Loaded **{len(df):,}** rows from "
        f"{found_path.as_posix() if found_path else 'DATA_URL'}"
    )

    # ---------- Filters with Reset ----------
    st.subheader("Filters")

    default_states = sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else []
    default_types  = sorted(df["asset_type"].dropna().unique().tolist()) if "asset_type" in df.columns else []
    default_confs  = sorted(df["conf_cat"].dropna().unique().tolist()) if "conf_cat" in df.columns else []

    if "sel_states" not in st.session_state: st.session_state.sel_states = default_states[:10] if default_states else []
    if "sel_types"  not in st.session_state: st.session_state.sel_types  = default_types
    if "sel_confs"  not in st.session_state: st.session_state.sel_confs  = default_confs
    if "name_q"     not in st.session_state: st.session_state.name_q     = ""

    col_reset, _, _, _ = st.columns(4)
    if col_reset.button("Reset filters"):
        st.session_state.sel_states = default_states[:10] if default_states else []
        st.session_state.sel_types  = default_types
        st.session_state.sel_confs  = default_confs
        st.session_state.name_q     = ""

    c1, c2, c3, c4 = st.columns(4)
    sel_states = c1.multiselect("State", default_states, default=st.session_state.sel_states, key="sel_states")
    sel_types  = c2.multiselect("Asset Type", default_types,  default=st.session_state.sel_types,  key="sel_types")
    sel_confs  = c3.multiselect("Confidence", default_confs,  default=st.session_state.sel_confs,  key="sel_confs")
    name_q     = c4.text_input("Search asset/ZIP contains", st.session_state.name_q, key="name_q")

    # Sliders
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

    # Apply filters
    flt = df.copy()
    if sel_states and "state" in flt.columns: flt = flt[flt["state"].isin(sel_states)]
    if sel_types  and "asset_type" in flt.columns: flt = flt[flt["asset_type"].isin(sel_types)]
    if sel_confs  and "conf_cat" in flt.columns: flt = flt[flt["conf_cat"].isin(sel_confs)]
    if name_q:
        m = False
        if "asset_name" in flt.columns: m = flt["asset_name"].astype(str).str.contains(name_q, case=False, na=False)
        if "zip" in flt.columns:
            m = (m | flt["zip"].astype(str).str.contains(name_q, na=False)) if isinstance(m, pd.Series) else flt["zip"].astype(str).str.contains(name_q, na=False)
        flt = flt[m]
    if v_range and "value" in flt.columns: flt = flt[(flt["value"] >= v_range[0]) & (flt["value"] <= v_range[1])]
    if s_range and "sqft" in flt.columns:  flt = flt[(flt["sqft"] >= s_range[0]) & (flt["sqft"] <= s_range[1])]

    # ---------- KPIs (Bn/Mn formatting) ----------
    st.markdown("### KPIs (Filtered)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Assets", f"{len(flt):,}")
    k2.metric("Total Value", fmt_money_units(flt['value'].sum()) if "value" in flt.columns else "—")
    k3.metric("Median Value", fmt_money_units(flt['value'].median()) if "value" in flt.columns else "—")
    k4.metric("Median $/ft²", fmt_money_units(flt['value_psf'].median()) if "value_psf" in flt.columns else "—")

    # ---------- MAP (placed ABOVE charts; color-by selector) ----------
    st.markdown("### Map")
    if ("lat" not in flt.columns) or ("lon" not in flt.columns) or flt[["lat","lon"]].dropna().empty:
        st.info("No geocoded rows to plot. Check filters or CSV columns.")
    else:
        geo = flt.dropna(subset=["lat","lon"]).copy()
        geo = geo[geo["lat"].between(-90,90) & geo["lon"].between(-180,180)]
        if geo.empty:
            st.warning("No rows with valid lat/lon after filters.")
        else:
            # radius in METERS (visible at country zoom)
            if "value" in geo.columns and geo["value"].notna().any():
                v = geo["value"].astype(float).clip(lower=1)
                vmed = max(float(v.median()), 1.0)
                scale = (v / vmed).clip(0.05, 20.0) ** 0.35
                geo["radius_m"] = (12000 * scale).clip(3000, 40000)  # 3–40 km
            else:
                geo["radius_m"] = 10000

            # choose color field
            color_options = []
            if "conf_cat" in geo.columns:  color_options.append("Confidence")
            if "asset_type" in geo.columns: color_options.append("Asset Type")
            if "state" in geo.columns:      color_options.append("State")
            if "cluster" in geo.columns:    color_options.append("Cluster")
            color_by = st.selectbox("Color by", color_options or ["Confidence"], index=0)

            if color_by == "Confidence":
                key_series = geo["conf_cat"]
            elif color_by == "Asset Type":
                key_series = geo["asset_type"]
            elif color_by == "State":
                key_series = geo["state"]
            else:
                key_series = geo["cluster"] if "cluster" in geo.columns else geo["conf_cat"]

            colors, cmap = color_map_for(key_series)
            geo["color"] = colors

            # legend
            legend_html = " ".join(
                [f"<span class='legend-item'><span class='legend-swatch' style='background: rgb({v[0]},{v[1]},{v[2]});'></span>{k}</span>"
                 for k, v in cmap.items()]
            )
            st.markdown(f"**Legend:** {legend_html}", unsafe_allow_html=True)

            center_lat = float(geo["lat"].mean()); center_lon = float(geo["lon"].mean())
            view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3.8 if len(geo)>1000 else 5)

            view_type = st.radio("Map view", ["Points", "Heatmap"], horizontal=True, index=0)
            if view_type == "Heatmap":
                layer = pdk.Layer(
                    "HeatmapLayer",
                    data=geo,
                    get_position='[lon, lat]',
                    get_weight='value' if "value" in geo.columns else None,
                    radiusPixels=70,
                )
            else:
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=geo,
                    get_position='[lon, lat]',
                    get_radius='radius_m',          # meters
                    radius_min_pixels=2,            # pixel floor so never invisible
                    radius_max_pixels=40,
                    get_fill_color='color',
                    pickable=True, auto_highlight=True,
                )

            tooltip = {"html":"<b>{asset_name}</b><br/>${value:,.0f}<br/>{state} {zip}",
                       "style":{"backgroundColor":"steelblue","color":"white"}}
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip),
                            use_container_width=True)

    # ---------- CHARTS (below map) ----------
    st.markdown("### Distributions")
    cc1, cc2 = st.columns(2)
    if "value" in flt.columns and (flt["value"] > 0).any():
        fig_val = px.histogram(flt, x="value", nbins=60, title="Asset Value (log x)")
        fig_val.update_xaxes(type="log")
        cc1.plotly_chart(fig_val, use_container_width=True)
    if "value_psf" in flt.columns:
        fig_psf = px.histogram(
            flt.replace([np.inf, -np.inf], np.nan).dropna(subset=["value_psf"]),
            x="value_psf", nbins=60, title="Value per Sq.Ft"
        )
        cc2.plotly_chart(fig_psf, use_container_width=True)

    st.markdown("### Value vs Rentable Sq.Ft (log-log)")
    if all(c in flt.columns for c in ["sqft","value"]):
        plot_df = flt.sample(n=min(5000, len(flt)), random_state=0)  # cap for speed
        fig_sc = px.scatter(
            plot_df, x="sqft", y="value",
            hover_data=[c for c in ["asset_name","state","zip"] if c in plot_df.columns],
            title=f"Showing {len(plot_df):,} of {len(flt):,} assets"
        )
        if (plot_df["sqft"] > 0).any(): fig_sc.update_xaxes(type="log")
        if (plot_df["value"] > 0).any(): fig_sc.update_yaxes(type="log")
        st.plotly_chart(fig_sc, use_container_width=True)

    # ---------- Table + Download ----------
    st.markdown("### Top 50 by Value")
    if "value" in flt.columns:
        top = flt.sort_values("value", ascending=False).head(50)
        st.dataframe(top, use_container_width=True)

    st.download_button("⬇️ Download filtered CSV",
                       data=flt.to_csv(index=False),
                       file_name="filtered_assets.csv",
                       mime="text/csv")

# =====================================================
# ================== TAB 2 – TASK 3 ML =================
# =====================================================
with tab_ml:
    st.subheader("Clustering (KMeans) + Classification (RandomForest)")
    must = ["value", "sqft", "value_psf"]
    if not all(c in df.columns for c in must):
        st.warning("CSV must include: value, sqft, value_psf (auto-generated if value & sqft exist).")
        st.stop()

    ml = df[["value","sqft","value_psf"] + (["age"] if "age" in df.columns else [])].copy()
    ml = ml.replace([np.inf,-np.inf], np.nan).fillna(ml.median(numeric_only=True))

    for c in ["value","sqft","value_psf"]:
        ml[f"log_{c}"] = np.log(np.clip(ml[c], 1, None))

    X_cols = [c for c in ml.columns if c.startswith("log_")]
    if "age" in ml.columns: X_cols.append("age")

    scaler = StandardScaler()
    X = scaler.fit_transform(ml[X_cols])

    k = st.slider("Choose number of clusters (k)", 3, 8, 3, 1)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    st.write(f"**Silhouette score:** {sil:.3f}")

    df_ml = df.copy()
    df_ml["Asset Cluster"] = labels

    sizes = pd.Series(labels).value_counts().sort_index().rename("count")
    st.write("**Cluster sizes**"); st.dataframe(sizes)

    prof_cols = [c for c in ["value","sqft","value_psf","age"] if c in df_ml.columns]
    profile = df_ml.groupby("Asset Cluster")[prof_cols].median().sort_index()
    st.write("**Cluster profiles (median)**"); st.dataframe(profile)

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    figp = px.scatter(pd.DataFrame({"PC1":X2[:,0], "PC2":X2[:,1], "Cluster":labels}),
                      x="PC1", y="PC2", color="Cluster", title="PCA (2D) by Cluster",
                      opacity=0.7, height=500)
    st.plotly_chart(figp, use_container_width=True)

    st.download_button("⬇️ Download data with clusters",
                       data=df_ml.to_csv(index=False),
                       file_name="t3_assets_with_clusters.csv",
                       mime="text/csv")

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
        columns=[f"Pred {i}" for i in sorted(np.unique(labels))]
    )
    st.write("**Confusion Matrix**"); st.dataframe(cm_df)

    st.write("**Classification Report**")
    st.text(classification_report(y_test, y_pred))

    imp = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False)
    fig_imp = px.bar(imp, title="Feature Importances", labels={"index":"feature","value":"importance"})
    st.plotly_chart(fig_imp, use_container_width=True)

st.caption("KPIs are shown in Bn/Mn. Map appears above charts and supports color-by (Confidence/Type/State/Cluster).")
