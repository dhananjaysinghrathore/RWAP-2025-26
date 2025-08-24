# streamlit_app.py — RWAP Dashboard & ML with Live Descriptive Analytics + Forecast
import os
from pathlib import Path
import re
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
st.caption("Group: **CLArX Gurugram**")

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
def _first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map many possible column names to a canonical schema used by the app."""
    mapping = {
        "loc_code":   ["Location Code", "loc_code"],
        "asset_name": ["Real Property Asset Name", "Asset Name", "asset_name", "Name"],
        "city":       ["City", "City_x", "city"],
        "state":      ["State", "State_x", "state"],
        "zip":        ["Zip Code", "ZIP", "zip", "Zip"],
        "lat":        ["Latitude", "Latitude_x", "lat", "Lat"],
        "lon":        ["Longitude", "Longitude_x", "lon", "Lng", "Long"],
        "sqft":       ["Building Rentable Square Feet", "SqFt", "sqft", "Area_sqft"],
        "value":      ["Estimated Asset Value (Adj)", "Estimated Asset Value", "Est Value (Base)", "value", "Valuation"],
        "conf_cat":   ["Confidence Category", "conf_cat", "confidence"],
        "asset_type": ["Real Property Asset Type", "Asset Type", "Type"],
        "age":        ["Building Age", "age", "Age_years"],
        "cluster":    ["Asset Cluster", "cluster", "Cluster"],
    }
    out = pd.DataFrame()
    for k, cands in mapping.items():
        hit = _first_col(df, cands)
        if hit is not None:
            out[k] = df[hit]

    # Numeric coercion
    for n in ["lat", "lon", "sqft", "value", "age", "cluster"]:
        if n in out.columns:
            out[n] = pd.to_numeric(out[n], errors="coerce")

    # Derived $/ft²
    if "value" in out.columns and "sqft" in out.columns:
        out["value_psf"] = out["value"] / out["sqft"]
        out["value_psf"].replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        out["value_psf"] = np.nan

    # Confidence fallback
    if "conf_cat" not in out.columns:
        out["conf_cat"] = "Unknown"

    return out

def fmt_money_units(x: float) -> str:
    """Format as $X.XX Bn / Mn (else plain $)."""
    try:
        x = float(x)
    except Exception:
        return "—"
    if np.isnan(x):
        return "—"
    a = abs(x)
    if a >= 1e9:
        return f"${x/1e9:,.2f} Bn"
    if a >= 1e6:
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

def assign_cluster_names_from_profile(df_in: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """
    Map numeric cluster IDs to friendly names using medians:
    - 'Tiny/Special (High $/ft²)' => highest median value_psf
    - 'Large & Older'             => highest median sqft
    - 'Core Buildings'            => remaining cohort
    For k != 3, remaining clusters get generic 'Cluster N'.
    """
    df = df_in.copy()
    if cluster_col not in df.columns:
        return df

    try:
        df[cluster_col] = pd.to_numeric(df[cluster_col], errors="coerce")
    except Exception:
        return df

    labels = sorted(df[cluster_col].dropna().unique().tolist())
    if not labels:
        return df

    prof_cols = [c for c in ["value_psf", "sqft", "age"] if c in df.columns]
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
            if l not in name_map:
                name_map[l] = f"Cluster {int(l)}"
    else:
        for l in labels:
            name_map[l] = f"Cluster {int(l)}"

    df["Cluster Name"] = df[cluster_col].map(name_map)
    return df

# ---------- ML explainability helpers ----------
def silhouette_label(s):
    if s >= 0.65: return "excellent separation"
    if s >= 0.50: return "good separation"
    if s >= 0.35: return "moderate separation"
    if s >= 0.20: return "weak separation"
    return "poor separation"

def quant_bucket(v, qs=(0.33, 0.66), labels=("small","mid","large")):
    if v is None or np.isnan(v): return "unknown"
    q1, q2 = qs
    if v < q1:  return labels[0]
    if v < q2:  return labels[1]
    return labels[2]

def describe_cluster_row(row, q_sqft, q_psf, q_age):
    size_tag   = quant_bucket(row.get("sqft", np.nan),  q_sqft,  ("small","mid","large"))
    psf_tag    = quant_bucket(row.get("value_psf", np.nan), q_psf, ("low $/ft²","mid $/ft²","high $/ft²"))
    age_tag    = quant_bucket(row.get("age", np.nan),   q_age,   ("younger","mid-age","older"))
    return f"{size_tag} assets, {psf_tag}, {age_tag}."

# ---------- Time-series helpers ----------
DATE_COL_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{4}$")  # e.g., 31-07-2025 (DD-MM-YYYY)

def find_date_columns(df: pd.DataFrame):
    """Return list of numeric columns whose names look like dd-mm-yyyy."""
    cols = []
    for c in df.columns:
        if DATE_COL_PATTERN.match(str(c)) and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def melt_timeseries(df_like: pd.DataFrame, id_cols):
    """Melt wide date columns to long: id_cols + ['date','value_ts']."""
    date_cols = find_date_columns(df_like)
    if not date_cols:
        return pd.DataFrame()
    long = df_like[id_cols + date_cols].melt(id_vars=id_cols, var_name="date_str", value_name="value_ts")
    long["date"] = pd.to_datetime(long["date_str"], dayfirst=True, errors="coerce")
    long = long.dropna(subset=["date", "value_ts"]).sort_values("date")
    return long

def month_index(dt: pd.Series):
    return dt.dt.year * 12 + dt.dt.month

def forecast_linear(df_series: pd.DataFrame, horizon=6, log=False):
    """
    df_series: columns ['date','value_ts'] monthly-ish.
    Simple linear trend on month index. Returns df with actual + forecast + bands.
    """
    s = df_series.dropna(subset=["date","value_ts"]).copy()
    if s.empty or s["date"].nunique() < 3:
        return None

    # aggregate to monthly median to avoid duplicate days
    s["ym"] = s["date"].dt.to_period("M").dt.to_timestamp()
    s = s.groupby("ym", as_index=False)["value_ts"].median().sort_values("ym")
    s["t"] = month_index(s["ym"])

    y = s["value_ts"].astype(float).values
    x = s["t"].astype(float).values
    if log:
        y = np.log(np.clip(y, 1, None))

    # fit y = a*x + b using polyfit
    try:
        a, b = np.polyfit(x, y, 1)
    except Exception:
        return None

    # predictions (in-sample, to get residuals)
    y_hat = a*x + b
    resid = y - y_hat
    sigma = float(np.nanstd(resid))

    # future months
    last_t = int(s["t"].iloc[-1])
    fut_t = np.arange(last_t+1, last_t+1+horizon)
    fut_dates = pd.period_range(s["ym"].iloc[-1] + 1, periods=horizon, freq="M").to_timestamp()
    fut_y = a*fut_t + b

    if log:
        s["pred"] = np.exp(y_hat)
        s["lower"] = np.exp(y_hat - sigma)
        s["upper"] = np.exp(y_hat + sigma)
        fut_pred = np.exp(fut_y)
        fut_lower = np.exp(fut_y - sigma)
        fut_upper = np.exp(fut_y + sigma)
    else:
        s["pred"] = y_hat
        s["lower"] = y_hat - sigma
        s["upper"] = y_hat + sigma
        fut_pred  = fut_y
        fut_lower = fut_y - sigma
        fut_upper = fut_y + sigma

    past = pd.DataFrame({"date": s["ym"], "value": s["value_ts"], "kind": "actual"})
    past_pred = pd.DataFrame({"date": s["ym"], "value": s["pred"], "kind": "fit",
                              "lower": s["lower"], "upper": s["upper"]})
    future = pd.DataFrame({"date": fut_dates, "value": fut_pred, "kind": "forecast",
                           "lower": fut_lower, "upper": fut_upper})
    out = pd.concat([past, past_pred, future], ignore_index=True)
    return out

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
    found_path = next((p for p in CANDIDATE_FILES if p.exists()), None)
    st.caption(
        f"Loaded **{len(df):,}** rows from "
        f"{found_path.as_posix() if found_path else 'DATA_URL'}"
    )

    # ---------- Filters + Reset ----------
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
            # Visible marker radius (meters) at country zoom
            if "value" in geo.columns and geo["value"].notna().any():
                v = geo["value"].astype(float).clip(lower=1)
                vmed = max(float(v.median()), 1.0)
                scale = (v / vmed).clip(0.05, 20.0) ** 0.35
                geo["radius_m"] = (12000 * scale).clip(3000, 40000)  # 3–40 km
            else:
                geo["radius_m"] = 10000

            # choose color field (prefer friendly Cluster Name)
            color_options = []
            if "Cluster Name" in geo.columns:
                color_options.append("Cluster Name")
            elif "cluster" in geo.columns:
                color_options.append("Cluster ID")
            if "conf_cat" in geo.columns:
                color_options.append("Confidence")
            if "asset_type" in geo.columns:
                color_options.append("Asset Type")
            if "state" in geo.columns:
                color_options.append("State")

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
            geo["color"] = colors

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

    # ---------- Distributions ----------
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

    # ---------- Value vs Size ----------
    st.markdown("### Value vs Rentable Sq.Ft (log-log)")
    if all(c in flt.columns for c in ["sqft","value"]):
        plot_df = flt.sample(n=min(5000, len(flt)), random_state=0)
        fig_sc = px.scatter(
            plot_df, x="sqft", y="value",
            hover_data=[c for c in ["asset_name","state","zip"] if c in plot_df.columns],
            title=f"Showing {len(plot_df):,} of {len(flt):,} assets"
        )
        if (plot_df["sqft"] > 0).any(): fig_sc.update_xaxes(type="log")
        if (plot_df["value"] > 0).any(): fig_sc.update_yaxes(type="log")
        st.plotly_chart(fig_sc, use_container_width=True)

    # ---------- Time Series & Forecast from historical valuation columns ----------
    st.markdown("## Time Series & Forecast")
    st.caption(
        "Uses historical valuation columns in your CSV whose names look like **DD-MM-YYYY** "
        "(e.g., `31-10-2024`, `30-11-2024`, …). Forecast = linear trend (option to log)."
    )

    date_cols = find_date_columns(raw)
    if date_cols:
        # Build a working subset that preserves IDs and the date columns
        id_cols = [c for c in ["asset_name","zip","state","asset_type"] if c in raw.columns]
        sub = raw.copy()
        # Apply the same filters as 'flt' if those id columns exist
        if "state" in flt.columns and "state" in sub.columns: sub = sub[sub["state"].isin(flt["state"].dropna().unique())]
        if "asset_type" in flt.columns and "asset_type" in sub.columns: sub = sub[sub["asset_type"].isin(flt["asset_type"].dropna().unique())]
        if "zip" in flt.columns and "zip" in sub.columns and st.session_state.name_q:
            qs = str(st.session_state.name_q)
            mask_zip = sub["zip"].astype(str).str.contains(qs, na=False)
            if "asset_name" in sub.columns:
                mask_name = sub["asset_name"].astype(str).str.contains(qs, case=False, na=False)
                mask_zip = mask_zip | mask_name
            sub = sub[mask_zip]

        ts_long = melt_timeseries(sub, id_cols)
        if ts_long.empty:
            st.info("No time-series rows left after filters.")
        else:
            # Aggregation choice
            agg_choice = st.selectbox("Aggregate by", ["All assets (median)", "State", "Asset Type", "ZIP", "Single Asset"], index=0)
            log_fit = st.checkbox("Use log-scale trend (good for growth rates)", value=False)
            horizon = st.slider("Forecast horizon (months)", 1, 12, 6)

            if agg_choice == "All assets (median)":
                series = ts_long.groupby("date", as_index=False)["value_ts"].median()
                fig_ts = px.line(series, x="date", y="value_ts", title="Median asset value over time")
                st.plotly_chart(fig_ts, use_container_width=True)

                fc = forecast_linear(series.rename(columns={"value_ts":"value_ts"}), horizon=horizon, log=log_fit)
                if fc is not None:
                    fig_fc = px.line(fc, x="date", y="value", color="kind",
                                     title="Median value: fit + forecast",
                                     labels={"value":"value"})
                    if "lower" in fc.columns:
                        fig_fc.add_traces([
                            px.scatter(fc[fc["kind"]=="forecast"], x="date", y="lower").data[0],
                            px.scatter(fc[fc["kind"]=="forecast"], x="date", y="upper").data[0],
                        ])
                    st.plotly_chart(fig_fc, use_container_width=True)
                else:
                    st.info("Need at least 3 months of history for forecasting.")

            elif agg_choice in ["State", "Asset Type", "ZIP"]:
                key = {"State":"state", "Asset Type":"asset_type", "ZIP":"zip"}[agg_choice]
                choices = sorted(ts_long[key].dropna().unique().tolist())
                if not choices:
                    st.info(f"No {key} values after filters.")
                else:
                    pick = st.selectbox(f"Choose {agg_choice}", choices)
                    series = ts_long[ts_long[key]==pick].groupby("date", as_index=False)["value_ts"].median()
                    fig_ts = px.line(series, x="date", y="value_ts", title=f"Median value over time — {agg_choice}: {pick}")
                    st.plotly_chart(fig_ts, use_container_width=True)

                    fc = forecast_linear(series.rename(columns={"value_ts":"value_ts"}), horizon=horizon, log=log_fit)
                    if fc is not None:
                        fig_fc = px.line(fc, x="date", y="value", color="kind",
                                         title=f"Forecast — {agg_choice}: {pick}",
                                         labels={"value":"value"})
                        st.plotly_chart(fig_fc, use_container_width=True)
                    else:
                        st.info("Need at least 3 months of history for forecasting.")

            else:  # Single Asset
                if "asset_name" not in ts_long.columns:
                    st.info("Single-asset view needs an 'asset_name' column.")
                else:
                    names = sorted(ts_long["asset_name"].dropna().unique().tolist())
                    pick = st.selectbox("Choose asset", names)
                    series = ts_long[ts_long["asset_name"]==pick][["date","value_ts"]].sort_values("date")
                    fig_ts = px.line(series, x="date", y="value_ts", title=f"Asset value over time — {pick}")
                    st.plotly_chart(fig_ts, use_container_width=True)

                    fc = forecast_linear(series, horizon=horizon, log=log_fit)
                    if fc is not None:
                        fig_fc = px.line(fc, x="date", y="value", color="kind",
                                         title=f"Forecast — {pick}", labels={"value":"value"})
                        st.plotly_chart(fig_fc, use_container_width=True)
                    else:
                        st.info("Need at least 3 months of history for forecasting.")
    else:
        st.info("No historical valuation columns (DD-MM-YYYY) detected in your CSV — time series is skipped.")

    # ---------- More Analytics (Task 2) ----------
    st.markdown("## More Analytics (Task 2)")

    # A) US Choropleth: Median $/ft² by State
    st.markdown(
        "**Where are the pricier markets?** Choropleth highlights states by **median $/ft²**."
    )
    if "state" in flt.columns and flt["state"].notna().any():
        state_agg = flt.groupby("state", as_index=False).agg(
            median_psf=("value_psf","median"),
            total_value=("value","sum"),
            n=("state","size"),
        )
        state_agg["total_value_fmt"] = state_agg["total_value"].apply(fmt_money_units)
        fig_choro = px.choropleth(
            state_agg,
            locations="state", locationmode="USA-states", scope="usa",
            color="median_psf", color_continuous_scale="Blues",
            hover_data={"state":True, "median_psf":":.0f", "total_value_fmt":True, "n":True}
        )
        fig_choro.update_layout(title="Median $/ft² by State", coloraxis_colorbar_title="$ / ft²")
        st.plotly_chart(fig_choro, use_container_width=True)

    # B) Box plot: $/ft² by Asset Type
    st.markdown("**How does $/ft² vary across types?** Box plot shows spread and outliers.")
    if {"asset_type","value_psf"}.issubset(flt.columns):
        tmp = flt[["asset_type","value_psf"]].dropna().copy()
        if len(tmp) > 0:
            p99 = tmp["value_psf"].quantile(0.99)
            tmp = tmp[tmp["value_psf"] <= p99]
            fig_box = px.box(tmp, x="asset_type", y="value_psf", points="suspectedoutliers",
                             title="Value per Sq.Ft by Asset Type")
            fig_box.update_xaxes(title="")
            fig_box.update_yaxes(title="$ / ft²")
            st.plotly_chart(fig_box, use_container_width=True)

    # C) Correlation heatmap (Spearman)
    st.markdown("**Which features move together?** Spearman correlation handles skew.")
    num_cols = [c for c in ["value","sqft","value_psf","age"] if c in flt.columns]
    if len(num_cols) >= 2:
        corr = (flt[num_cols].replace([np.inf,-np.inf], np.nan)
                        .dropna()
                        .corr(method="spearman"))
        fig_corr = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
            title="Correlation (Spearman)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # D) Top cities by total value
    st.markdown("**Which cities concentrate the most value?**")
    if {"city","value"}.issubset(flt.columns):
        city_agg = flt.groupby("city", as_index=False).agg(
            total_value=("value","sum"), n=("city","size")
        )
        top = city_agg.sort_values("total_value", ascending=False).head(15).copy()
        top["total_value_bn"] = top["total_value"] / 1e9
        fig_city = px.bar(
            top, x="city", y="total_value_bn", text="n",
            title="Top 15 Cities by Total Value", labels={"total_value_bn":"Total Value (Bn)"}
        )
        fig_city.update_traces(textposition="outside")
        fig_city.update_xaxes(title="")
        st.plotly_chart(fig_city, use_container_width=True)

    # E) $/ft² vs Age
    st.markdown("**Does age relate to $/ft²?**")
    if {"value_psf","age"}.issubset(flt.columns):
        plot_age = flt.dropna(subset=["value_psf","age"]).copy()
        if len(plot_age) > 0:
            plot_age = plot_age.sample(n=min(5000, len(plot_age)), random_state=1)
            color_col = "asset_type" if "asset_type" in plot_age.columns else None
            fig_age = px.scatter(
                plot_age, x="age", y="value_psf", color=color_col, opacity=0.5,
                title="$ / ft² vs Age", labels={"age":"Age (years)","value_psf":"$ / ft²"}
            )
            st.plotly_chart(fig_age, use_container_width=True)

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
        "3) See **sizes** and **profiles** for each cluster.  \n"
        "4) **PCA** gives a 2D picture (not used by the model).  \n"
        "5) **RandomForest** predicts cluster for new assets; we report accuracy."
    )

    must = ["value","sqft","value_psf"]
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
    c3.metric("Interpretation", silhouette_label(sil))

    # Cluster sizes bar
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
    st.markdown("### Cluster profile")
    q_sqft = profile["sqft"].quantile([0.33, 0.66]).values if "sqft" in profile.columns else (0, 0)
    q_psf  = profile["value_psf"].quantile([0.33, 0.66]).values if "value_psf" in profile.columns else (0, 0)
    q_age  = profile["age"].quantile([0.33, 0.66]).values if "age" in profile.columns else (0, 0)

    cards_per_row = 3
    names = profile.index.tolist()
    for i in range(0, len(names), cards_per_row):
        cols = st.columns(min(cards_per_row, len(names)-i))
        for j, name in enumerate(names[i:i+cards_per_row]):
            with cols[j]:
                row = profile.loc[name]
                line = describe_cluster_row(row, q_sqft, q_psf, q_age)
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

    st.download_button("⬇️ Download data with clusters (named)",
                       data=df_ml.to_csv(index=False),
                       file_name="t3_assets_with_clusters_named.csv",
                       mime="text/csv")

    # RandomForest classifier
    st.markdown("---")
    st.subheader("RandomForest: predict cluster")

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.30, random_state=42, stratify=labels
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = (y_pred == y_test).mean()
    st.metric("Accuracy", f"{acc:.3f}")

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    name_lookup = df_ml.groupby("cluster")["Cluster Name"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else f"Cluster {int(s.iloc[0])}"
    )
    idx_names = [name_lookup[i] for i in sorted(np.unique(labels))]
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

st.caption("KPIs in Bn/Mn. Filters drive all charts. Time-series auto-detects DD-MM-YYYY columns and adds linear-trend forecasts.")
