# streamlit_app.py — RWAP Dashboard & ML (ClarX Gurugram)
from __future__ import annotations
import os
from pathlib import Path
import io
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
GROUP_NAME = "ClarX Gurugram"
st.markdown(
    """
    <style>
      .block-container {padding-top: 0.8rem; padding-bottom: 1.0rem;}
      .metric {text-align:center;}
      .legend-swatch {display:inline-block; width:14px; height:14px; border-radius:3px; margin-right:8px;}
      .legend-item {margin-right:14px; display:inline-flex; align-items:center;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("RWAP – Asset Valuation Dashboard & ML")
st.caption(f"Group: **{GROUP_NAME}**")

# -------------------- Auto data loader --------------------
DATA_FILE_BASE = os.getenv("DATA_FILE_BASE", "asset_valuation_results_final_with_confidence")
DATA_DIR = Path(__file__).parent / "data"
CANDIDATE_FILES = [
    DATA_DIR / f"{DATA_FILE_BASE}.csv.gz",
    DATA_DIR / f"{DATA_FILE_BASE}.csv",
]

@st.cache_data(show_spinner=True)
def load_data() -> tuple[pd.DataFrame, str]:
    for p in CANDIDATE_FILES:
        if p.exists():
            if str(p).endswith(".gz"):
                df = pd.read_csv(p, compression="infer")
            else:
                df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            return df, p.as_posix()
    url = st.secrets.get("DATA_URL", "")
    if url:
        df = pd.read_csv(url)
        df.columns = [c.strip() for c in df.columns]
        return df, "DATA_URL"
    st.error(
        "Data file not found.\n\n"
        f"Expected one of:\n- {CANDIDATE_FILES[0].as_posix()}\n- {CANDIDATE_FILES[1].as_posix()}\n\n"
        "Upload CSV/CSV.GZ to `/data/`, or set a public link in Secrets as `DATA_URL`."
    )
    st.stop()

raw, source_label = load_data()

# -------------------- Helpers --------------------
def _first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map many possible column names to a canonical schema used by the app."""
    mapping = {
        "asset_name": ["Real Property Asset Name", "Asset Name", "asset_name", "Name"],
        "city":       ["City", "City_x", "city"],
        "state":      ["State", "State_x", "state"],
        "zip":        ["Zip Code", "ZIP", "zip", "Zip"],
        "lat":        ["Latitude", "Latitude_x", "lat", "Lat"],
        "lon":        ["Longitude", "Longitude_x", "lon", "Lng", "Long"],
        "sqft":       ["Building Rentable Square Feet", "Rentable Sqft", "SqFt", "sqft", "Area_sqft"],
        "value":      ["Estimated Asset Value (Adj)", "Estimated Asset Value", "Est Value (Base)", "value", "Valuation"],
        "conf_cat":   ["Confidence Category", "conf_cat", "confidence"],
        "asset_type": ["Real Property Asset Type", "Asset Type", "Type", "asset_type"],
        "age":        ["Building Age", "age", "Age_years"],
        "cluster":    ["Asset Cluster", "cluster", "Cluster", "asset_cluster"],
        "date":       ["date", "as_of_date", "valuation_date", "month"]
    }
    out = pd.DataFrame()
    for k, cands in mapping.items():
        hit = _first_col(df, cands)
        if hit is not None:
            out[k] = df[hit]

    # Numeric coercion
    for n in ["lat", "lon", "sqft", "value", "age", "cluster"]:
        if n in out.columns:
            out.loc[:, n] = pd.to_numeric(out[n], errors="coerce")

    # Derived $/ft²
    if "value" in out.columns and "sqft" in out.columns:
        vpsf = out["value"] / out["sqft"].replace(0, np.nan)
        out["value_psf"] = vpsf.replace([np.inf, -np.inf], np.nan)
    else:
        out["value_psf"] = np.nan

    # Confidence fallback
    if "conf_cat" in out.columns:
        # Normalize labels
        MAP = {
            'very low':'Very Low','v.low':'Very Low','vl':'Very Low',
            'low':'Low','l':'Low',
            'medium':'Medium','med':'Medium','m':'Medium',
            'high':'High','h':'High',
            'very high':'Very High','v.high':'Very High','vh':'Very High'
        }
        s = (out["conf_cat"].astype(str).str.strip().str.replace(r'[_\-]',' ',regex=True)
             .str.lower().map(MAP).fillna('Unknown'))
        order = ['Very Low','Low','Medium','High','Very High','Unknown']
        out["conf_cat"] = pd.Categorical(s, order, ordered=True)
    else:
        out["conf_cat"] = pd.Categorical(['Unknown']*len(out),
                                         ['Very Low','Low','Medium','High','Very High','Unknown'],
                                         ordered=True)

    # Date to month start; synthesize if absent
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    else:
        out["date"] = pd.NaT
    if out["date"].isna().all():
        # synthetic months (for trend demo) 2016–2025
        rng = np.random.default_rng(42)
        start = pd.Timestamp("2016-01-01")
        months = 120
        out["date"] = start + pd.to_timedelta(rng.integers(0, months, size=len(out)), unit="M")
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp("MS")

    # Basic fill-ins to avoid missing columns downstream
    for c in ("asset_type","state","zip","asset_name"):
        if c not in out.columns:
            out[c] = ""

    return out

def fmt_money_units(x: float) -> str:
    """Format as $X.XX Bn / Mn (else plain $)."""
    try:
        x = float(x)
    except Exception:
        return "—"
    if np.isnan(x): return "—"
    a = abs(x)
    if a >= 1e9: return f"${x/1e9:,.2f} Bn"
    if a >= 1e6: return f"${x/1e6:,.2f} Mn"
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

def assign_cluster_names_from_profile(df_in: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """Map numeric cluster IDs to friendly names based on medians."""
    df = df_in.copy()
    if cluster_col not in df.columns: return df
    try:
        df.loc[:, cluster_col] = pd.to_numeric(df[cluster_col], errors="coerce")
    except Exception:
        return df
    labels = sorted(df[cluster_col].dropna().unique().tolist())
    if not labels: return df

    prof_cols = [c for c in ["value_psf","sqft","age"] if c in df.columns]
    if not prof_cols:
        df["Cluster Name"] = df[cluster_col].apply(lambda x: f"Cluster {int(x)}" if pd.notna(x) else "Unknown")
        return df

    prof = df.groupby(cluster_col)[prof_cols].median()
    name_map = {}
    if len(labels) >= 3 and "value_psf" in prof.columns and "sqft" in prof.columns:
        small_high = prof["value_psf"].idxmax()
        large_lab  = prof["sqft"].idxmax()
        remaining = [l for l in labels if l not in [small_high, large_lab]]
        core_lab = remaining[0] if remaining else None
        name_map[small_high] = "Tiny/Special (High $/ft²)"
        name_map[large_lab]  = "Large & Older"
        if core_lab is not None:
            name_map[core_lab] = "Core Buildings"
        for l in labels:
            if l not in name_map: name_map[l] = f"Cluster {int(l)}"
    else:
        for l in labels:
            name_map[l] = f"Cluster {int(l)}"
    df["Cluster Name"] = df[cluster_col].map(name_map)
    return df

# ---------- Forecast helper (stable beyond 2025) ----------
def forecast_linear(series: pd.DataFrame, horizon: int = 24, log: bool = True) -> pd.DataFrame|None:
    """Input series: columns ['date','value'] (monthly). Returns df with kind in {'Historical','Fit','Forecast'}."""
    s = series.dropna().copy()
    if s.empty: return None
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.groupby("date", as_index=False)["value"].median().sort_values("date")
    if len(s) < 2: return None

    s["t"] = np.arange(len(s))
    y = np.log1p(s["value"]) if log else s["value"]
    m, b = np.polyfit(s["t"].to_numpy(), y.to_numpy(), 1)
    s["fit"] = np.expm1(m*s["t"] + b) if log else (m*s["t"] + b)

    last = s["date"].iloc[-1]
    future_dates = pd.date_range(start=last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    fut_t = np.arange(int(s["t"].iloc[-1]) + 1, int(s["t"].iloc[-1]) + 1 + horizon)
    fut_val = np.expm1(m*fut_t + b) if log else (m*fut_t + b)

    df_obs = pd.DataFrame({"date": s["date"], "value": s["value"], "kind":"Historical"})
    df_fit = pd.DataFrame({"date": s["date"], "value": s["fit"],   "kind":"Fit"})
    df_fut = pd.DataFrame({"date": future_dates, "value": fut_val, "kind":"Forecast"})
    return pd.concat([df_obs, df_fit, df_fut], ignore_index=True)

# -------------------- Prepare data --------------------
df = canonicalize(raw)
if "cluster" in df.columns:
    df = assign_cluster_names_from_profile(df, "cluster")

# Tabs
tab_dash, tab_ml = st.tabs(["📊 Task 2: Dashboard", "🤖 Task 3: ML"])

# =====================================================
# ================= TAB 1 – DASHBOARD =================
# =====================================================
with tab_dash:
    st.caption(f"Loaded **{len(df):,}** rows from **{source_label}**")

    # ---------- Filters + Reset ----------
    st.subheader("Filters")
    default_states = sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else []
    default_types  = sorted(df["asset_type"].dropna().unique().tolist()) if "asset_type" in df.columns else []
    default_confs  = list(df["conf_cat"].cat.categories) if "conf_cat" in df.columns and hasattr(df["conf_cat"], "cat") else sorted(df["conf_cat"].dropna().unique().tolist()) if "conf_cat" in df.columns else []

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

    # Range sliders
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
            m2 = flt["zip"].astype(str).str.contains(name_q, na=False)
            m = (m | m2) if isinstance(m, pd.Series) else m2
        flt = flt[m]
    if v_range and "value" in flt.columns: flt = flt[(flt["value"] >= v_range[0]) & (flt["value"] <= v_range[1])]
    if s_range and "sqft" in flt.columns:  flt = flt[(flt["sqft"] >= s_range[0]) & (flt["sqft"] <= s_range[1])]

    # ---------- KPIs ----------
    st.markdown("### KPIs (Filtered)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Assets", f"{len(flt):,}")
    k2.metric("Total Value", fmt_money_units(flt['value'].sum()) if "value" in flt.columns else "—")
    k3.metric("Median Value", fmt_money_units(flt['value'].median()) if "value" in flt.columns else "—")
    k4.metric("Median $/ft²", fmt_money_units(flt['value_psf'].median()) if "value_psf" in flt.columns else "—")

    # ---------- MAP (above charts) ----------
    st.markdown("### Map")
    if ("lat" not in flt.columns) or ("lon" not in flt.columns) or flt[["lat","lon"]].dropna().empty:
        st.info("No geocoded rows to plot. Check filters or CSV columns.")
    else:
        geo = flt.dropna(subset=["lat","lon"]).copy()
        geo = geo[geo["lat"].between(-90,90) & geo["lon"].between(-180,180)]
        if geo.empty:
            st.warning("No rows with valid lat/lon after filters.")
        else:
            # marker radius (meters) scaled by value
            if "value" in geo.columns and geo["value"].notna().any():
                v = geo["value"].astype(float).clip(lower=1)
                vmed = max(float(v.median()), 1.0)
                scale = (v / vmed).clip(0.05, 20.0) ** 0.35
                geo.loc[:, "radius_m"] = (12000 * scale).clip(3000, 40000)
            else:
                geo.loc[:, "radius_m"] = 10000

            # Color options
            color_options = []
            if "Cluster Name" in geo.columns: color_options.append("Cluster Name")
            if "cluster" in geo.columns and "Cluster Name" not in color_options: color_options.append("Cluster ID")
            if "conf_cat" in geo.columns: color_options.append("Confidence")
            if "asset_type" in geo.columns: color_options.append("Asset Type")
            if "state" in geo.columns: color_options.append("State")
            color_by = st.selectbox("Color by", color_options or ["State"], index=0)

            if color_by == "Cluster Name":
                key_series = geo["Cluster Name"]
            elif color_by == "Cluster ID":
                key_series = geo["cluster"].astype(str)
            elif color_by == "Confidence":
                key_series = geo["conf_cat"]
            elif color_by == "Asset Type":
                key_series = geo["asset_type"]
            else:
                key_series = geo["state"]

            colors, cmap = color_map_for(key_series)
            geo.loc[:, "color"] = colors

            # Legend
            legend_html = " ".join(
                [f"<span class='legend-item'><span class='legend-swatch' "
                 f"style='background: rgb({v[0]},{v[1]},{v[2]});'></span>{k}</span>"
                 for k, v in cmap.items()]
            )
            if legend_html:
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
                    get_radius='radius_m',
                    radius_min_pixels=2,
                    radius_max_pixels=40,
                    get_fill_color='color',
                    pickable=True, auto_highlight=True,
                )
            tooltip = {"html":"<b>{asset_name}</b><br/>${value:,.0f}<br/>{state} {zip}",
                       "style":{"backgroundColor":"steelblue","color":"white"}}
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip),
                            use_container_width=True)

    # ---------- FOUR charts (one row) ----------
    st.markdown("### Descriptive analytics (react to filters)")
    cA, cB, cC, cD = st.columns(4)

    with cA:
        if "value" in flt.columns and (flt["value"] > 0).any():
            fig_val = px.histogram(flt, x="value", nbins=50, title="Asset Value (log x)")
            fig_val.update_xaxes(type="log")
            st.plotly_chart(fig_val, use_container_width=True)
        else:
            st.info("No 'value'.")

    with cB:
        if "value_psf" in flt.columns and flt["value_psf"].notna().any():
            fig_psf = px.histogram(flt.replace([np.inf,-np.inf], np.nan).dropna(subset=["value_psf"]),
                                   x="value_psf", nbins=50, title="Value per Sq.Ft")
            st.plotly_chart(fig_psf, use_container_width=True)
        else:
            st.info("No $/ft².")

    with cC:
        if all(c in flt.columns for c in ["sqft","value"]):
            sample = flt.sample(n=min(4000, len(flt)), random_state=0) if len(flt)>4000 else flt
            fig_sc = px.scatter(sample, x="sqft", y="value", opacity=0.65,
                                title="Value vs Sq.Ft (log–log)", trendline="ols")
            if (sample["sqft"] > 0).any(): fig_sc.update_xaxes(type="log")
            if (sample["value"] > 0).any(): fig_sc.update_yaxes(type="log")
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Need sqft & value.")

    with cD:
        if "state" in flt.columns and "value" in flt.columns:
            top = (flt.groupby("state")["value"].sum()
                     .sort_values(ascending=False).head(10).reset_index())
            fig_top = px.bar(top, x="state", y="value", title="Top states (by total value)")
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No state/value to chart.")

    # ---------- Forecast section ----------
    st.markdown("### Trend & Forecast (monthly median)")
    horizon = st.slider("Forecast horizon (months)", 6, 36, 24, step=6)
    log_fit  = st.checkbox("Use log trend (stabilizes variance)", value=True)
    agg_choice = st.selectbox("Aggregate by", ["All", "State", "Asset Type", "ZIP"], index=0)

    ts = flt[['date','value','state','asset_type','zip']].dropna(subset=['date']).copy() if 'date' in flt.columns else pd.DataFrame()
    if ts.empty:
        st.info("No dates available after filters. Try clearing filters.")
    else:
        if agg_choice == "All":
            series = ts.groupby('date', as_index=False)['value'].median()
            fc = forecast_linear(series, horizon=horizon, log=log_fit)
            if fc is not None:
                st.plotly_chart(px.line(fc, x="date", y="value", color="kind",
                                        title="Median value: historical + fit + forecast"),
                                use_container_width=True)
            else:
                st.info("Not enough points to fit a trend.")
        else:
            key = {"State":"state","Asset Type":"asset_type","ZIP":"zip"}[agg_choice]
            choices = sorted([x for x in ts[key].dropna().unique().tolist() if str(x).strip()])
            if not choices:
                st.info(f"No {key} values after filters.")
            else:
                sel = st.selectbox(f"Pick {agg_choice}", choices)
                series = ts.loc[ts[key]==sel].groupby('date', as_index=False)['value'].median()
                fc = forecast_linear(series, horizon=horizon, log=log_fit)
                if fc is not None:
                    st.plotly_chart(px.line(fc, x="date", y="value", color="kind",
                                            title=f"Median value ({agg_choice}={sel}): fit + forecast"),
                                    use_container_width=True)
                else:
                    st.info("Not enough points to fit a trend for this selection.")

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
    st.info(
        "**How to read Task 3**  \n"
        "1) Choose **k** — we group similar assets.  \n"
        "2) **Silhouette** shows separation (≥0.50 is good).  \n"
        "3) Check **sizes** & **profiles** to understand clusters.  \n"
        "4) **PCA** is a 2D picture for intuition.  \n"
        "5) **RandomForest** predicts cluster for new assets (report + confusion matrix)."
    )

    must = ["value","sqft","value_psf"]
    if not all(c in df.columns for c in must):
        st.warning("CSV must include: value, sqft, value_psf (value_psf auto-created if value & sqft exist).")
        st.stop()

    ml = df[["value","sqft","value_psf"] + (["age"] if "age" in df.columns else [])].copy()
    ml = ml.replace([np.inf,-np.inf], np.nan)
    ml = ml.fillna(ml.median(numeric_only=True))

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

    df_ml = df.copy()
    df_ml["cluster"] = labels
    df_ml = assign_cluster_names_from_profile(df_ml, "cluster")  # friendly names

    sizes = df_ml["cluster"].value_counts().sort_index().rename("count")
    prof_cols = [c for c in ["value","sqft","value_psf","age"] if c in df_ml.columns]
    profile = df_ml.groupby("Cluster Name")[prof_cols].median().sort_index()

    # KPIs
    st.markdown("### ML KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters (k)", f"{k}")
    c2.metric("Silhouette", f"{sil:0.3f}")
    c3.metric("Interpretation", "excellent" if sil>=0.65 else "good" if sil>=0.5 else "moderate" if sil>=0.35 else "weak" if sil>=0.2 else "poor")

    # Sizes chart
    sizes_named = df_ml["Cluster Name"].value_counts().reset_index()
    sizes_named.columns = ["Cluster Name", "count"]
    fig_sizes = px.bar(
        sizes_named.sort_values("count", ascending=False),
        x="Cluster Name", y="count", color="Cluster Name",
        title="Cluster sizes (count)", text="count"
    )
    fig_sizes.update_traces(textposition="outside")
    st.plotly_chart(fig_sizes, use_container_width=True)

    # Profile cards
    st.markdown("### Cluster profile (one sentence each)")
    def qb(v, qs): 
        if np.isnan(v): return "unknown"
        return "low" if v<qs[0] else "mid" if v<qs[1] else "high"
    qs_sqft = profile["sqft"].quantile([0.33,0.66]).values if "sqft" in profile.columns else (0,0)
    qs_psf  = profile["value_psf"].quantile([0.33,0.66]).values if "value_psf" in profile.columns else (0,0)
    qs_age  = profile["age"].quantile([0.33,0.66]).values if "age" in profile.columns else (0,0)

    cards_per_row = 3
    names = profile.index.tolist()
    for i in range(0, len(names), cards_per_row):
        cols = st.columns(min(cards_per_row, len(names)-i))
        for j, name in enumerate(names[i:i+cards_per_row]):
            with cols[j]:
                row = profile.loc[name]
                line = f"{qb(row.get('sqft',np.nan),qs_sqft)} size, {qb(row.get('value_psf',np.nan),qs_psf)} $/ft², {qb(row.get('age',np.nan),qs_age)} age."
                st.markdown(
                    f"**{name}**  \n"
                    f"- Median Value: {fmt_money_units(row.get('value', np.nan))}  \n"
                    f"- Median Size: {row.get('sqft', np.nan):,.0f} ft²  \n"
                    f"- Median $/ft²: {fmt_money_units(row.get('value_psf', np.nan))}  \n"
                    f"- Median Age: {row.get('age', np.nan):,.0f} yrs  \n"
                    f"**Summary:** {line}"
                )

    # Composition charts
    st.markdown("### Where clusters occur (composition)")
    left, right = st.columns(2)
    if "state" in df_ml.columns:
        comp_state = (df_ml.groupby(["Cluster Name","state"]).size()
                        .reset_index(name="n"))
        top_states = (df_ml["state"].value_counts().head(12).index.tolist())
        comp_state = comp_state[comp_state["state"].isin(top_states)]
        with left:
            fig_st = px.bar(
                comp_state, x="state", y="n", color="Cluster Name",
                barmode="group", title="Top states by cluster count"
            )
            st.plotly_chart(fig_st, use_container_width=True)

    if "asset_type" in df_ml.columns:
        comp_type = (df_ml.groupby(["Cluster Name","asset_type"]).size()
                        .reset_index(name="n"))
        top_types = df_ml["asset_type"].value_counts().head(10).index.tolist()
        comp_type = comp_type[comp_type["asset_type"].isin(top_types)]
        with right:
            fig_ty = px.bar(
                comp_type, x="asset_type", y="n", color="Cluster Name",
                barmode="group", title="Top asset types by cluster count"
            )
            st.plotly_chart(fig_ty, use_container_width=True)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    figp = px.scatter(
        pd.DataFrame({"PC1":X2[:,0], "PC2":X2[:,1], "Cluster":df_ml["Cluster Name"]}),
        x="PC1", y="PC2", color="Cluster", title="PCA (2D) by Cluster (named)",
        opacity=0.7, height=520
    )
    st.plotly_chart(figp, use_container_width=True)

    # Download with clusters
    st.download_button("⬇️ Download data with clusters (named)",
                       data=df_ml.to_csv(index=False),
                       file_name="t3_assets_with_clusters_named.csv",
                       mime="text/csv")

    # RandomForest
    st.markdown("---")
    st.subheader("RandomForest: predict cluster")
    X_train, X_test, y_train, y_test = train_test_split(
        X, df_ml["cluster"].to_numpy(), test_size=0.30, random_state=42, stratify=df_ml["cluster"].to_numpy()
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = (y_pred == y_test).mean()
    st.metric("Accuracy", f"{acc:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    name_lookup = df_ml.groupby("cluster")["Cluster Name"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else f"Cluster {int(s.iloc[0])}"
    )
    order = sorted(np.unique(df_ml["cluster"]))
    idx_names = [name_lookup[i] for i in order]
    fig_cm = px.imshow(
        cm,
        x=[f"Pred {n}" for n in idx_names],
        y=[f"True {n}" for n in idx_names],
        text_auto=True, color_continuous_scale="Blues",
        title="Confusion Matrix (counts)"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.write("**Classification Report**")
    st.text(classification_report(y_test, y_pred))

    imp = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False)
    fig_imp = px.bar(imp, title="Feature Importances", labels={"index":"feature","value":"importance"})
    st.plotly_chart(fig_imp, use_container_width=True)

st.caption("KPIs in Bn/Mn. Map above charts. Forecast extends beyond 2025. Task-3 includes cluster cards, sizes, composition, PCA, and RF accuracy with heatmap. Uses auto data loader (/data/*.csv[.gz] or st.secrets['DATA_URL']).")
