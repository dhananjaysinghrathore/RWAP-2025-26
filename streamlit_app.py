# streamlit_app.py — RWAP Dashboard + ML + Forecast + 4-up insights
import os, re
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

st.set_page_config(page_title="RWAP – Dashboard & ML", page_icon="📊", layout="wide")
st.title("RWAP – Asset Valuation Dashboard & ML")
st.caption("Group: **ClarX Gurugram**")

# ---------- Data load (auto or URL secret) ----------
DATA_FILE_BASE = os.getenv("DATA_FILE_BASE", "asset_valuation_results_final_with_confidence")
DATA_DIR = Path(__file__).parent / "data"
CANDIDATE_FILES = [DATA_DIR / f"{DATA_FILE_BASE}.csv.gz", DATA_DIR / f"{DATA_FILE_BASE}.csv"]

@st.cache_data(show_spinner=True)
def load_data():
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
    st.error("Data file not found. Place CSV/CSV.GZ in /data or set a public `DATA_URL` secret.")
    st.stop()

raw = load_data()

# ---------- Helpers ----------
def _first_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def canonicalize(df):
    m = {
        "loc_code":["Location Code","loc_code"],
        "asset_name":["Real Property Asset Name","Asset Name","asset_name","Name"],
        "city":["City","City_x","city"],
        "state":["State","State_x","state"],
        "zip":["Zip Code","ZIP","zip","Zip"],
        "lat":["Latitude","Latitude_x","lat","Lat"],
        "lon":["Longitude","Longitude_x","lon","Lng","Long"],
        "sqft":["Building Rentable Square Feet","SqFt","sqft","Area_sqft"],
        "value":["Estimated Asset Value (Adj)","Estimated Asset Value","Est Value (Base)","value","Valuation"],
        "conf_cat":["Confidence Category","conf_cat","confidence"],
        "asset_type":["Real Property Asset Type","Asset Type","Type"],
        "age":["Building Age","age","Age_years"],
        "cluster":["Asset Cluster","cluster","Cluster"],
    }
    out = pd.DataFrame()
    for k, cands in m.items():
        hit = _first_col(df, cands)
        if hit is not None: out[k] = df[hit]

    for n in ["lat","lon","sqft","value","age","cluster"]:
        if n in out: out[n] = pd.to_numeric(out[n], errors="coerce")

    if "value" in out and "sqft" in out:
        psf = out["value"] / out["sqft"].replace({0: np.nan})
        out["value_psf"] = psf.replace([np.inf, -np.inf], np.nan)
    else:
        out["value_psf"] = np.nan

    if "conf_cat" not in out: out["conf_cat"] = "Unknown"
    return out

def fmt_money_units(x):
    try: x = float(x)
    except: return "—"
    if np.isnan(x): return "—"
    a = abs(x)
    if a >= 1e9: return f"${x/1e9:,.2f} Bn"
    if a >= 1e6: return f"${x/1e6:,.2f} Mn"
    return f"${x:,.0f}"

BASE_PALETTE = [
    [230,57,70],[29,53,87],[69,123,157],[42,157,143],[233,196,106],
    [244,162,97],[231,111,81],[94,79,162],[0,119,182],[34,197,94],
    [148,163,184],[217,70,239],[99,102,241],[245,158,11]
]
def color_map_for(series):
    cats = sorted(series.fillna("Unknown").astype(str).unique().tolist())
    cmap = {c: BASE_PALETTE[i % len(BASE_PALETTE)] for i, c in enumerate(cats)}
    colors = series.fillna("Unknown").astype(str).map(cmap).tolist()
    return colors, cmap

def assign_cluster_names_from_profile(df_in, cluster_col="cluster"):
    df = df_in.copy()
    if cluster_col not in df: return df
    df[cluster_col] = pd.to_numeric(df[cluster_col], errors="coerce")
    labels = sorted(df[cluster_col].dropna().unique().tolist())
    if not labels: return df
    prof_cols = [c for c in ["value_psf","sqft","age"] if c in df]
    if not prof_cols:
        df["Cluster Name"] = df[cluster_col].apply(lambda x: f"Cluster {int(x)}" if pd.notna(x) else "Unknown")
        return df
    prof = df.groupby(cluster_col)[prof_cols].median()
    name_map = {}
    if len(labels) >= 3 and "value_psf" in prof and "sqft" in prof:
        small_high = prof["value_psf"].idxmax()
        large_lab  = prof["sqft"].idxmax()
        remaining = [l for l in labels if l not in [small_high, large_lab]]
        core_lab = remaining[0] if remaining else None
        name_map[small_high] = "Tiny/Special (High $/ft²)"
        name_map[large_lab]  = "Large & Older"
        if core_lab is not None: name_map[core_lab] = "Core Buildings"
        for l in labels:
            if l not in name_map: name_map[l] = f"Cluster {int(l)}"
    else:
        for l in labels: name_map[l] = f"Cluster {int(l)}"
    df["Cluster Name"] = df[cluster_col].map(name_map)
    return df

def silhouette_label(s):
    return "excellent" if s>=0.65 else "good" if s>=0.50 else "moderate" if s>=0.35 else "weak" if s>=0.20 else "poor"

# ----- Time series helpers -----
DATE_COL_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{4}$")  # DD-MM-YYYY

def find_date_columns(df):
    return [c for c in df.columns if DATE_COL_PATTERN.match(str(c)) and pd.api.types.is_numeric_dtype(df[c])]

def melt_timeseries(df_like, id_cols):
    dcols = find_date_columns(df_like)
    if not dcols: return pd.DataFrame()
    long = df_like[id_cols + dcols].melt(id_vars=id_cols, var_name="date_str", value_name="value_ts")
    long["date"] = pd.to_datetime(long["date_str"], dayfirst=True, errors="coerce")
    return long.dropna(subset=["date","value_ts"]).sort_values("date")

def month_index(dt): return dt.dt.year*12 + dt.dt.month

def forecast_linear(df_series, horizon=6, log=False):
    s = df_series.dropna(subset=["date","value_ts"]).copy()
    if s.empty or s["date"].nunique() < 3: return None
    # monthly median at Month Start
    s["ym"] = s["date"].dt.to_period("M").dt.to_timestamp("MS")
    s = s.groupby("ym", as_index=False)["value_ts"].median().sort_values("ym")
    s["t"] = month_index(s["ym"])
    y = s["value_ts"].astype(float).values
    x = s["t"].astype(float).values
    if log: y = np.log(np.clip(y, 1, None))
    a, b = np.polyfit(x, y, 1)
    y_hat = a*x + b
    resid = y - y_hat
    sigma = float(np.nanstd(resid))
    # future months: start next month, Month Start frequency
    fut_dates = pd.date_range(s["ym"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    fut_t = month_index(pd.Series(fut_dates))
    fut_y = a*fut_t + b
    if log:
        s["pred"] = np.exp(y_hat); s["lower"] = np.exp(y_hat - sigma); s["upper"] = np.exp(y_hat + sigma)
        fut_pred  = np.exp(fut_y);  fut_lower = np.exp(fut_y - sigma);  fut_upper = np.exp(fut_y + sigma)
    else:
        s["pred"] = y_hat; s["lower"] = y_hat - sigma; s["upper"] = y_hat + sigma
        fut_pred  = fut_y;  fut_lower = fut_y - sigma;  fut_upper = fut_y + sigma
    past = pd.DataFrame({"date": s["ym"], "value": s["value_ts"], "kind":"actual"})
    fit  = pd.DataFrame({"date": s["ym"], "value": s["pred"], "kind":"fit","lower":s["lower"],"upper":s["upper"]})
    fut  = pd.DataFrame({"date": fut_dates, "value": fut_pred, "kind":"forecast","lower":fut_lower,"upper":fut_upper})
    return pd.concat([past, fit, fut], ignore_index=True)

# ---------- Prepare dataframe ----------
df = canonicalize(raw)
if "cluster" in df: df = assign_cluster_names_from_profile(df, "cluster")

tab_dash, tab_ml = st.tabs(["📊 Task 2: Dashboard", "🤖 Task 3: ML"])

# =================== DASHBOARD ===================
with tab_dash:
    st.caption(f"Loaded **{len(df):,}** rows")

    # Filters
    st.subheader("Filters")
    defaults = lambda col: sorted(df[col].dropna().unique().tolist()) if col in df else []
    if "sel_states" not in st.session_state: st.session_state.sel_states = defaults("state")[:10]
    if "sel_types"  not in st.session_state: st.session_state.sel_types  = defaults("asset_type")
    if "sel_confs"  not in st.session_state: st.session_state.sel_confs  = defaults("conf_cat")
    if "name_q"     not in st.session_state: st.session_state.name_q     = ""

    col_reset, _, _, _ = st.columns(4)
    if col_reset.button("Reset filters"): 
        st.session_state.sel_states = defaults("state")[:10]
        st.session_state.sel_types  = defaults("asset_type")
        st.session_state.sel_confs  = defaults("conf_cat")
        st.session_state.name_q     = ""

    c1,c2,c3,c4 = st.columns(4)
    sel_states = c1.multiselect("State", defaults("state"), default=st.session_state.sel_states, key="sel_states")
    sel_types  = c2.multiselect("Asset Type", defaults("asset_type"), default=st.session_state.sel_types, key="sel_types")
    sel_confs  = c3.multiselect("Confidence", defaults("conf_cat"), default=st.session_state.sel_confs, key="sel_confs")
    name_q     = c4.text_input("Search asset/ZIP contains", st.session_state.name_q, key="name_q")

    v_range = s_range = None
    if "value" in df and df["value"].notna().any():
        vmin, vmax = float(df["value"].min()), float(df["value"].max()); v_range = st.slider("Value range", vmin, vmax, (vmin, vmax))
    if "sqft" in df and df["sqft"].notna().any():
        smin, smax = float(df["sqft"].min()), float(df["sqft"].max()); s_range = st.slider("Sq.Ft range", smin, smax, (smin, smax))

    flt = df.copy()
    if sel_states and "state" in flt: flt = flt[flt["state"].isin(sel_states)]
    if sel_types  and "asset_type" in flt: flt = flt[flt["asset_type"].isin(sel_types)]
    if sel_confs  and "conf_cat" in flt: flt = flt[flt["conf_cat"].isin(sel_confs)]
    if name_q:
        mask = False
        if "asset_name" in flt: mask = flt["asset_name"].astype(str).str.contains(name_q, case=False, na=False)
        if "zip" in flt:
            m2 = flt["zip"].astype(str).str.contains(name_q, na=False)
            mask = (mask | m2) if isinstance(mask, pd.Series) else m2
        flt = flt[mask]
    if v_range and "value" in flt: flt = flt[(flt["value"]>=v_range[0]) & (flt["value"]<=v_range[1])]
    if s_range and "sqft" in flt: flt = flt[(flt["sqft"]>=s_range[0]) & (flt["sqft"]<=s_range[1])]

    # KPIs
    st.markdown("### KPIs (Filtered)")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Assets", f"{len(flt):,}")
    k2.metric("Total Value", fmt_money_units(flt["value"].sum()) if "value" in flt else "—")
    k3.metric("Median Value", fmt_money_units(flt["value"].median()) if "value" in flt else "—")
    k4.metric("Median $/ft²", fmt_money_units(flt["value_psf"].median()) if "value_psf" in flt else "—")

    # Map
    st.markdown("### Map")
    if {"lat","lon"}.issubset(flt.columns) and not flt[["lat","lon"]].dropna().empty:
        geo = flt.dropna(subset=["lat","lon"]).copy()
        geo = geo[geo["lat"].between(-90,90) & geo["lon"].between(-180,180)]
        if "value" in geo and geo["value"].notna().any():
            scale = (geo["value"].clip(lower=1) / max(float(geo["value"].median()),1))**0.35
            geo["radius_m"] = (12000*scale).clip(3000, 40000)
        else:
            geo["radius_m"] = 10000
        options = []
        if "Cluster Name" in geo: options.append("Cluster Name")
        if "conf_cat" in geo: options.append("Confidence")
        if "asset_type" in geo: options.append("Asset Type")
        if "state" in geo: options.append("State")
        color_by = st.selectbox("Color by", options or ["State"])
        key_series = geo["Cluster Name"] if color_by=="Cluster Name" else geo[color_by.replace(" ", "_").lower()]
        colors, cmap = color_map_for(key_series)
        geo["color"] = colors
        legend = " ".join([f"<span style='display:inline-flex;align-items:center;margin-right:12px'><span style='width:12px;height:12px;border-radius:3px;background:rgb({v[0]},{v[1]},{v[2]});display:inline-block;margin-right:6px'></span>{k}</span>" for k,v in cmap.items()])
        if legend: st.markdown(f"**Legend:** {legend}", unsafe_allow_html=True)
        view = pdk.ViewState(latitude=float(geo["lat"].mean()), longitude=float(geo["lon"].mean()), zoom=3.8 if len(geo)>1000 else 5)
        view_type = st.radio("Map view", ["Points","Heatmap"], horizontal=True, index=0)
        if view_type=="Heatmap":
            layer = pdk.Layer("HeatmapLayer", data=geo, get_position='[lon, lat]', get_weight='value' if "value" in geo else None, radiusPixels=70)
        else:
            layer = pdk.Layer("ScatterplotLayer", data=geo, get_position='[lon, lat]', get_radius='radius_m', radius_min_pixels=2, radius_max_pixels=40, get_fill_color='color', pickable=True, auto_highlight=True)
        tooltip={"html":"<b>{asset_name}</b><br/>${value:,.0f}<br/>{state} {zip}","style":{"backgroundColor":"steelblue","color":"white"}}
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip), use_container_width=True)
    else:
        st.info("No geocoded rows to plot.")

    # -------- Quick Insights (4-up in one line) --------
    st.markdown("## Quick Insights (4-up)")
    q1,q2,q3,q4 = st.columns(4)
    # 1) Top states by median $/ft² (small bar)
    if {"state","value_psf"}.issubset(flt.columns):
        sagg = (flt.groupby("state", as_index=False)["value_psf"].median()
                  .sort_values("value_psf", ascending=False).head(8))
        fig1 = px.bar(sagg, x="state", y="value_psf", title="Top States (Median $/ft²)")
        fig1.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10)); fig1.update_xaxes(title="")
        q1.plotly_chart(fig1, use_container_width=True)
    # 2) Box plot by type
    if {"asset_type","value_psf"}.issubset(flt.columns):
        tmp = flt[["asset_type","value_psf"]].dropna()
        if len(tmp):
            p99 = tmp["value_psf"].quantile(0.99); tmp = tmp[tmp["value_psf"]<=p99]
            fig2 = px.box(tmp, x="asset_type", y="value_psf", points=False, title="$ /ft² by Type")
            fig2.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10)); fig2.update_xaxes(title="")
            q2.plotly_chart(fig2, use_container_width=True)
    # 3) Value histogram (log x)
    if "value" in flt:
        fig3 = px.histogram(flt, x="value", nbins=50, title="Asset Value (log x)")
        fig3.update_xaxes(type="log"); fig3.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10))
        q3.plotly_chart(fig3, use_container_width=True)
    # 4) Value vs Sq.Ft (log-log) small
    if {"value","sqft"}.issubset(flt.columns):
        samp = flt.sample(n=min(3000, len(flt)), random_state=0)
        fig4 = px.scatter(samp, x="sqft", y="value", title="Value vs Sq.Ft (log-log)", opacity=0.6)
        if (samp["sqft"]>0).any(): fig4.update_xaxes(type="log")
        if (samp["value"]>0).any(): fig4.update_yaxes(type="log")
        fig4.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10))
        q4.plotly_chart(fig4, use_container_width=True)

    # Distributions (full width)
    st.markdown("### Distributions")
    d1,d2 = st.columns(2)
    if "value" in flt and (flt["value"]>0).any():
        fig_val = px.histogram(flt, x="value", nbins=60, title="Asset Value (log x)"); fig_val.update_xaxes(type="log")
        d1.plotly_chart(fig_val, use_container_width=True)
    if "value_psf" in flt:
        fig_psf = px.histogram(flt.dropna(subset=["value_psf"]), x="value_psf", nbins=60, title="Value per Sq.Ft")
        d2.plotly_chart(fig_psf, use_container_width=True)

    # Time Series & Forecast
    st.markdown("## Time Series & Forecast")
    date_cols = find_date_columns(raw)
    if date_cols:
        id_cols = [c for c in ["asset_name","zip","state","asset_type"] if c in raw.columns]
        sub = raw.copy()
        if "state" in flt and "state" in sub: sub = sub[sub["state"].isin(flt["state"].dropna().unique())]
        if "asset_type" in flt and "asset_type" in sub: sub = sub[sub["asset_type"].isin(flt["asset_type"].dropna().unique())]
        if "zip" in flt and "zip" in sub and st.session_state.name_q:
            q = str(st.session_state.name_q)
            mzip = sub["zip"].astype(str).str.contains(q, na=False)
            if "asset_name" in sub: mzip = mzip | sub["asset_name"].astype(str).str.contains(q, case=False, na=False)
            sub = sub[mzip]
        ts_long = melt_timeseries(sub, id_cols)
        agg_options = ["All assets (median)"]
        if "state" in ts_long: agg_options.append("State")
        if "asset_type" in ts_long: agg_options.append("Asset Type")
        if "zip" in ts_long: agg_options.append("ZIP")
        if "asset_name" in ts_long: agg_options.append("Single Asset")
        agg_choice = st.selectbox("Aggregate by", agg_options, index=0)
        log_fit = st.checkbox("Use log trend (good for growth)", value=False)
        horizon = st.slider("Forecast horizon (months)", 1, 12, 6)

        if agg_choice == "All assets (median)":
            series = ts_long.groupby("date", as_index=False)["value_ts"].median()
            st.plotly_chart(px.line(series, x="date", y="value_ts", title="Median asset value over time"), use_container_width=True)
            fc = forecast_linear(series, horizon=horizon, log=log_fit)
            if fc is not None: st.plotly_chart(px.line(fc, x="date", y="value", color="kind", title="Median value: fit + forecast"), use_container_width=True)
            else: st.info("Need ≥ 3 months history.")

        elif agg_choice in ["State","Asset Type","ZIP"]:
            key = {"State":"state","Asset Type":"asset_type","ZIP":"zip"}[agg_choice]
            if key not in ts_long:
                st.info(f"No '{key}' column in time series.")
            else:
                choices = sorted(ts_long[key].dropna().unique().tolist())
                if not choices: st.info("No values after filters.")
                else:
                    pick = st.selectbox(f"Choose {agg_choice}", choices)
                    series = ts_long[ts_long[key]==pick].groupby("date", as_index=False)["value_ts"].median()
                    st.plotly_chart(px.line(series, x="date", y="value_ts", title=f"Median value over time — {pick}"), use_container_width=True)
                    fc = forecast_linear(series, horizon=horizon, log=log_fit)
                    if fc is not None: st.plotly_chart(px.line(fc, x="date", y="value", color="kind", title=f"Forecast — {pick}"), use_container_width=True)
                    else: st.info("Need ≥ 3 months history.")

        else:  # Single Asset
            if "asset_name" not in ts_long: st.info("No 'asset_name' for single-asset view.")
            else:
                names = sorted(ts_long["asset_name"].dropna().unique().tolist())
                pick = st.selectbox("Choose asset", names)
                series = ts_long[ts_long["asset_name"]==pick][["date","value_ts"]].sort_values("date")
                st.plotly_chart(px.line(series, x="date", y="value_ts", title=f"Asset value over time — {pick}"), use_container_width=True)
                fc = forecast_linear(series, horizon=horizon, log=log_fit)
                if fc is not None: st.plotly_chart(px.line(fc, x="date", y="value", color="kind", title=f"Forecast — {pick}"), use_container_width=True)
                else: st.info("Need ≥ 3 months history.")
    else:
        st.info("No historical date columns like `31-10-2024` found in CSV.")

    # Top table + download
    st.markdown("### Top 50 by Value")
    if "value" in flt:
        st.dataframe(flt.sort_values("value", ascending=False).head(50), use_container_width=True)
        st.download_button("⬇️ Download filtered CSV", data=flt.to_csv(index=False), file_name="filtered_assets.csv", mime="text/csv")

# =================== TASK 3 (ML) ===================
with tab_ml:
    st.info("Pick **k** for KMeans; Silhouette ≥0.50 is good. RandomForest predicts cluster for new assets.")
    must = ["value","sqft"]
    if not all(c in df for c in must): st.warning("Need columns: value & sqft"); st.stop()

    ml = df[["value","sqft"] + ([ "value_psf"] if "value_psf" in df else []) + ([ "age"] if "age" in df else [])].copy()
    ml = ml.replace([np.inf,-np.inf], np.nan)
    for c in ml.columns: 
        if ml[c].dtype.kind in "biufc": ml[c] = ml[c].fillna(ml[c].median())
    ml["log_value"] = np.log(np.clip(ml["value"],1,None))
    ml["log_sqft"]  = np.log(np.clip(ml["sqft"],1,None))
    if "value_psf" in ml: ml["log_value_psf"] = np.log(np.clip(ml["value_psf"],1,None))

    X_cols = [c for c in ["log_value","log_sqft","log_value_psf","age"] if c in ml]
    X = StandardScaler().fit_transform(ml[X_cols])

    k = st.slider("k (clusters)", 3, 8, 3, 1)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)

    df_ml = df.copy(); df_ml["cluster"] = labels; df_ml = assign_cluster_names_from_profile(df_ml, "cluster")
    prof_cols = [c for c in ["value","sqft","value_psf","age"] if c in df_ml]
    profile = df_ml.groupby("Cluster Name")[prof_cols].median().sort_index()

    m1,m2,m3 = st.columns(3)
    m1.metric("Clusters (k)", f"{k}")
    m2.metric("Silhouette", f"{sil:.3f}")
    m3.metric("Interpretation", silhouette_label(sil))

    sizes_named = df_ml["Cluster Name"].value_counts().reset_index()
    sizes_named.columns = ["Cluster Name","count"]
    fig_sizes = px.bar(sizes_named.sort_values("count", ascending=False), x="Cluster Name", y="count", color="Cluster Name", text="count", title="Cluster sizes")
    fig_sizes.update_traces(textposition="outside")
    st.plotly_chart(fig_sizes, use_container_width=True)

    # PCA
    X2 = PCA(n_components=2, random_state=42).fit_transform(X)
    st.plotly_chart(px.scatter(pd.DataFrame({"PC1":X2[:,0],"PC2":X2[:,1],"Cluster":df_ml["Cluster Name"]}), x="PC1", y="PC2", color="Cluster", opacity=0.7, title="PCA (2D) by Cluster"), use_container_width=True)

    # RandomForest
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.30, random_state=42, stratify=labels)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    ypr = rf.predict(Xte)
    st.metric("RF Accuracy", f"{(ypr==yte).mean():.3f}")
    cm = confusion_matrix(yte, ypr)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix (counts)")
    st.plotly_chart(fig_cm, use_container_width=True)
    st.write("**Classification Report**")
    st.text(classification_report(yte, ypr))
    imp = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False)
    st.plotly_chart(px.bar(imp, title="Feature Importances", labels={"index":"feature","value":"importance"}), use_container_width=True)

    st.download_button("⬇️ Download data with clusters (named)", data=df_ml.to_csv(index=False), file_name="t3_assets_with_clusters_named.csv", mime="text/csv")
