# streamlit_app.py
# RWAP – Asset Valuation Dashboard & ML (ClarX Gurugram)
# ------------------------------------------------------
# - Task 2: Interactive descriptive analytics + map + 4 compact charts in one row
# - Forecast: monthly median VALUE trend projected beyond 2025 (robust to pandas 2.x/3.x)
# - Task 3: KMeans clustering (+ RF classifier preview)
#
# NOTE: If your CSV path is different, update DATA_PATH below.

from __future__ import annotations
import os, math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------
# Config / constants
# -----------------------
st.set_page_config(page_title="RWAP – Asset Valuation Model", layout="wide")
GROUP_NAME = "ClarX Gurugram"
DATA_PATH  = "data/asset_valuation_results_final_with_confidence_small.csv"  # <=== change if needed

CLUSTER_COLORS = {0:"#E74C3C", 1:"#1F77B4", 2:"#2ECC71"}  # red, blue, green

# -----------------------
# Helpers
# -----------------------
def fmt_money(x: float) -> str:
    if pd.isna(x): return "-"
    a = float(x)
    if abs(a) >= 1e9:
        return f"${a/1e9:,.2f} Bn"
    if abs(a) >= 1e6:
        return f"${a/1e6:,.2f} Mn"
    return f"${a:,.0f}"

def month_start(dt: pd.Series) -> pd.Series:
    """Coerce any datetime-like to month-start timestamps (no periods)."""
    d = pd.to_datetime(dt, errors="coerce")
    return d.dt.to_period("M").dt.to_timestamp("MS")

def forecast_linear(series: pd.DataFrame, horizon: int = 24, log: bool = True) -> pd.DataFrame | None:
    """
    Robust monthly linear trend + forecast (works on pandas >=2, fixes your log errors).
    'series' must have columns: ['date','value'] where date is datetime64[ns] (any day in month).
    Returns a long df with kind ∈ {'Historical','Fit','Forecast'}.
    """
    s = series.dropna().copy()
    if s.empty or s['value'].notna().sum() < 2:
        return None

    # Monthly aggregation -> median to be robust to outliers
    s['date'] = month_start(s['date'])
    s = s.groupby('date', as_index=False)['value'].median().sort_values('date')
    if len(s) < 2:
        return None

    s['t'] = np.arange(len(s))
    y = np.log1p(s['value']) if log else s['value']
    # simple OLS via polyfit (degree 1)
    slope, intercept = np.polyfit(s['t'].to_numpy(), y.to_numpy(), 1)

    # In-sample fit
    fit_y = slope*s['t'] + intercept
    s['fit'] = np.expm1(fit_y) if log else fit_y

    # Future months
    last_date = s['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1),
                                 periods=horizon, freq='MS')
    fut_t = np.arange(int(s['t'].iloc[-1]) + 1, int(s['t'].iloc[-1]) + 1 + horizon)
    fut_y = slope*fut_t + intercept
    fut_val = np.expm1(fut_y) if log else fut_y

    df_obs = pd.DataFrame({'date': s['date'], 'value': s['value'], 'kind': 'Historical'})
    df_fit = pd.DataFrame({'date': s['date'], 'value': s['fit'],   'kind': 'Fit'})
    df_fut = pd.DataFrame({'date': future_dates, 'value': fut_val, 'kind': 'Forecast'})
    return pd.concat([df_obs, df_fit, df_fut], ignore_index=True)

def normalize_confidence(col: pd.Series) -> pd.Series:
    CONF_MAP = {
        'very low':'Very Low','v.low':'Very Low','vl':'Very Low',
        'low':'Low','l':'Low',
        'medium':'Medium','med':'Medium','m':'Medium',
        'high':'High','h':'High',
        'very high':'Very High','v.high':'Very High','vh':'Very High'
    }
    cat_order = ['Very Low','Low','Medium','High','Very High','Unknown']
    s = (col.astype(str).str.strip()
                    .str.replace(r'[_\-]', ' ', regex=True)
                    .str.lower()
                    .map(CONF_MAP).fillna('Unknown'))
    return pd.Categorical(s, cat_order, ordered=True)

def load_data(path: str) -> pd.DataFrame:
    # Load CSV from repo (works on Streamlit Cloud)
    df = pd.read_csv(path)
    # Make columns snake_case and consistent
    df.columns = (df.columns
                    .str.strip()
                    .str.replace(r'[^A-Za-z0-9]+','_', regex=True)
                    .str.lower())
    # Expected names
    # try to map common variants
    rename_map = {
        'estimated_asset_value_adj':'value',
        'estimated_asset_value':'value',
        'building_rentable_square_feet':'sqft',
        'brsf':'sqft',
        'value_psf':'value_psf',
        'asset_type':'asset_type',
        'confidence':'confidence',
        'state':'state',
        'latitude':'lat', 'longitude':'lon',
        'zip':'zip',
        'asset_name':'asset_name',
        'building_age':'building_age'
    }
    for k,v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})

    # Compute derived columns if missing
    if 'value_psf' not in df.columns:
        if {'value','sqft'}.issubset(df.columns):
            with np.errstate(divide='ignore', invalid='ignore'):
                df['value_psf'] = df['value'] / df['sqft'].replace({0:np.nan})
        else:
            df['value_psf'] = np.nan

    # Confidence normalize
    if 'confidence' in df.columns:
        df['confidence'] = normalize_confidence(df['confidence'])
    else:
        df['confidence'] = pd.Categorical(['Unknown']*len(df))

    # Lat/Lon sanity
    for c in ('lat','lon'):
        if c not in df.columns: df[c] = np.nan
    # Asset type + state tidy
    for c in ('asset_type','state'):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({'nan':''})
        else:
            df[c] = ''

    # If no date column present, synthesize a stable pseudo-valuation date
    # (demo only; won’t affect descriptive stats; needed for forecasting)
    if 'date' not in df.columns:
        rng = np.random.default_rng(42)
        start = np.datetime64('2016-01-01')
        end   = np.datetime64('2025-12-31')
        # random month within the range
        months = (np.datetime64('2026-01-01') - start) // np.timedelta64(1,'M')
        df['date'] = pd.to_datetime(start) + pd.to_timedelta(
            rng.integers(low=0, high=int(months), size=len(df)), unit='M'
        )
        df['date'] = month_start(df['date'])
    else:
        df['date'] = month_start(df['date'])

    # Basic cleaning for visuals
    # Avoid chained assignment warnings: use .loc
    if 'value' in df.columns:
        df.loc[:, 'value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'sqft' in df.columns:
        df.loc[:, 'sqft'] = pd.to_numeric(df['sqft'], errors='coerce')
    if 'value_psf' in df.columns:
        df.loc[:, 'value_psf'] = pd.to_numeric(df['value_psf'], errors='coerce')

    return df

# -----------------------
# Load
# -----------------------
st.caption(f"Group: **{GROUP_NAME}**")
df = load_data(DATA_PATH)
st.session_state['__raw_len'] = len(df)

# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.header("Filters")
states = sorted([s for s in df['state'].dropna().unique().tolist() if s])
asset_types = sorted([s for s in df['asset_type'].dropna().unique().tolist() if s])
conf_levels = [c for c in df['confidence'].cat.categories if (df['confidence'] == c).any()]

sel_states = st.sidebar.multiselect("State", states, default=states[:10])
sel_types  = st.sidebar.multiselect("Asset Type", asset_types, default=asset_types)
sel_conf   = st.sidebar.multiselect("Confidence", conf_levels, default=conf_levels)

# Value & size sliders
vmin, vmax = float(np.nanmin(df['value'])), float(np.nanmax(df['value']))
smin, smax = float(np.nanmin(df['sqft'])),  float(np.nanmax(df['sqft']))
value_rng  = st.sidebar.slider("Value range", min_value=vmin, max_value=vmax,
                               value=(vmin, vmax), step=max((vmax-vmin)/1000, 1.0))
sqft_rng   = st.sidebar.slider("Sq.Ft range", min_value=smin, max_value=smax,
                               value=(smin, smax), step=max((smax-smin)/1000, 1.0))

# Search
q = st.sidebar.text_input("Search asset/ZIP contains", "")

# Apply filters
mask = (
    df['value'].between(value_rng[0], value_rng[1], inclusive="both") &
    df['sqft'].between(sqft_rng[0], sqft_rng[1], inclusive="both")
)
if sel_states:   mask &= df['state'].isin(sel_states)
if sel_types:    mask &= df['asset_type'].isin(sel_types)
if sel_conf:     mask &= df['confidence'].isin(sel_conf)

if q.strip():
    ql = q.strip().lower()
    cols = []
    if 'asset_name' in df.columns: cols.append(df['asset_name'].astype(str).str.lower().str.contains(ql, na=False))
    if 'zip' in df.columns:        cols.append(df['zip'].astype(str).str.contains(ql, na=False))
    if cols:
        mask &= np.column_stack(cols).any(axis=1)

f = df.loc[mask].copy()

# -----------------------
# Header & KPIs
# -----------------------
st.title("RWAP – Asset Valuation Dashboard & ML")
st.write(":bar_chart: **Task 2: Dashboard**  •  Loaded **{:,}** rows".format(st.session_state['__raw_len']))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets", f"{len(f):,}")
c2.metric("Total Value", fmt_money(f['value'].sum()))
c3.metric("Median Value", fmt_money(f['value'].median()))
c4.metric("Median $/ft²", fmt_money(f['value_psf'].median()))

# -----------------------
# MAP (color by Cluster / Asset Type / Confidence)
# -----------------------
st.subheader("Map")

color_by = st.selectbox("Color by", ["Cluster", "Asset Type", "Confidence"], index=0)
legend_cols = st.columns(3)
with legend_cols[0]: st.caption("Legend:")
if color_by == "Cluster" and "asset_cluster" in f.columns:
    # use predefined colors if present
    legend_text = "  ".join([f"<span style='color:{CLUSTER_COLORS.get(k,'#888')};font-weight:600'>{k}</span>"
                             for k in sorted(f['asset_cluster'].dropna().unique())])
    st.caption(legend_text, unsafe_allow_html=True)

# Plotly scatter_mapbox (no external tokens)
mfig = px.scatter_mapbox(
    f.dropna(subset=['lat','lon']),
    lat="lat", lon="lon",
    color=(
        f['asset_cluster'].map(CLUSTER_COLORS) if color_by=="Cluster" and 'asset_cluster' in f.columns
        else (f['asset_type'] if color_by=="Asset Type" else f['confidence'].astype(str))
    ),
    hover_data=["asset_name","state","zip","asset_type","value","sqft","value_psf"],
    size=np.clip(np.nan_to_num(f['value'], nan=0.0), 0, None)**0.25,  # soften large bubbles
    zoom=3, height=420
)
mfig.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(mfig, use_container_width=True)

# -----------------------
# 4 compact descriptive charts in one row
# -----------------------
st.subheader("Distributions (react to filters)")
g1, g2, g3, g4 = st.columns(4)

with g1:
    fig_v = px.histogram(f, x="value", nbins=40,
                         title="Asset Value (log x)")
    fig_v.update_xaxes(type="log")
    st.plotly_chart(fig_v, use_container_width=True)

with g2:
    fig_psf = px.histogram(f, x="value_psf", nbins=40,
                           title="Value per Sq.Ft")
    st.plotly_chart(fig_psf, use_container_width=True)

with g3:
    fig_sc = px.scatter(f, x="sqft", y="value",
                        title="Value vs Sq.Ft (log–log)",
                        opacity=0.6, trendline="ols")
    fig_sc.update_xaxes(type="log"); fig_sc.update_yaxes(type="log")
    st.plotly_chart(fig_sc, use_container_width=True)

with g4:
    # Top States (by total value)
    top = (f.groupby('state', dropna=True)['value']
             .sum().sort_values(ascending=False).head(10).reset_index())
    fig_top = px.bar(top, x='state', y='value', title="Top states (by total value)")
    st.plotly_chart(fig_top, use_container_width=True)

# -----------------------
# Time series + Forecast beyond 2025
# -----------------------
st.subheader("Monthly median value • Fit & forecast")

horizon = st.slider("Forecast horizon (months after last available month)", 6, 36, 24, step=6)
log_fit  = st.checkbox("Use log trend (stabilizes variance)", value=True)
agg_choice = st.selectbox("Aggregate by", ["All", "State", "Asset Type", "ZIP"], index=0)

ts = f[['date','value','state','asset_type','zip']].dropna(subset=['date']).copy()
if ts.empty:
    st.info("No dates available after filters. Try clearing filters.")
else:
    if agg_choice == "All":
        series = (ts.groupby('date', as_index=False)['value']
                    .median().rename(columns={'value':'value'}))
        fc = forecast_linear(series, horizon=horizon, log=log_fit)
        if fc is not None:
            lfig = px.line(fc, x="date", y="value", color="kind",
                           title="Median value: historical, fit & forecast")
            st.plotly_chart(lfig, use_container_width=True)
        else:
            st.info("Not enough points to fit a trend.")
    else:
        key = {"State":"state", "Asset Type":"asset_type", "ZIP":"zip"}[agg_choice]
        choices = sorted([x for x in ts[key].dropna().unique().tolist() if str(x).strip()])
        if not choices:
            st.info(f"No {key} values after filters.")
        else:
            sel = st.selectbox(f"Pick {agg_choice}", choices)
            subt = ts.loc[ts[key]==sel]
            series = (subt.groupby('date', as_index=False)['value']
                           .median().rename(columns={'value':'value'}))
            fc = forecast_linear(series, horizon=horizon, log=log_fit)
            if fc is not None:
                lfig = px.line(fc, x="date", y="value", color="kind",
                               title=f"Median value ({agg_choice}={sel}): fit & forecast")
                st.plotly_chart(lfig, use_container_width=True)
            else:
                st.info("Not enough points to fit a trend for this selection.")

# -----------------------
# Task 3 (ML): quick view
# -----------------------
st.markdown("---")
st.write(":robot_face: **Task 3: ML** – KMeans clusters + RandomForest classifier (preview)")

# Feature engineering (safe defaults)
feat_cols = []
for c in ['value','sqft','value_psf','building_age']:
    if c in f.columns: feat_cols.append(c)

ml = f[feat_cols].copy()
ml = ml.replace([np.inf, -np.inf], np.nan)
ml = ml.clip(lower=0)
ml = ml.fillna(ml.median(numeric_only=True))

# Logs for skewed
for c in [c for c in ['value','sqft','value_psf'] if c in ml.columns]:
    ml[f'log_{c}'] = np.log(ml[c].clip(1))  # avoid 0

X_cols = [c for c in ['log_value','log_sqft','log_value_psf','building_age'] if c in ml.columns]
if len(X_cols) >= 2:
    scaler = StandardScaler()
    X = scaler.fit_transform(ml[X_cols])

    # choose k=3 (from your earlier silhouette ≈ 0.54)
    km = KMeans(n_clusters=3, n_init='auto', random_state=42)
    labels = km.fit_predict(X)
    f['asset_cluster'] = labels

    # Small summary
    sizes = f['asset_cluster'].value_counts().sort_index()
    st.write("**Cluster sizes:**")
    st.dataframe(sizes.rename_axis("Cluster").reset_index(name="Count"))

    # Profiles (medians)
    profile_cols = [c for c in ['value','sqft','value_psf','building_age'] if c in f.columns]
    profile = f.groupby('asset_cluster')[profile_cols].median().sort_index()
    st.write("**Cluster medians (profile):**")
    st.dataframe(profile)

    # PCA scatter (2D) on sample to keep it light
    try:
        from sklearn.decomposition import PCA
        sample = min(4000, len(X))
        idx = np.random.default_rng(0).choice(len(X), size=sample, replace=False)
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X[idx])
        pdf = pd.DataFrame({'pc1':X2[:,0],'pc2':X2[:,1],'cluster':labels[idx]})
        pfig = px.scatter(pdf, x='pc1', y='pc2', color='cluster',
                          color_discrete_map=CLUSTER_COLORS,
                          title="PCA (2D) of assets by cluster", opacity=0.65)
        st.plotly_chart(pfig, use_container_width=True)
    except Exception:
        pass

    # Quick RF classifier to predict cluster (hold-out)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose()
    st.write("**RandomForest: classification report**")
    st.dataframe(rep_df.round(3))

    cm = confusion_matrix(y_test, y_pred)
    cmfig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                      labels=dict(x="Predicted", y="True", color="Count"),
                      title="Confusion matrix")
    st.plotly_chart(cmfig, use_container_width=True)

    # Feature importance
    imp = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False)
    if not imp.empty:
        if imp.sum() > 0:
            fig_imp = px.bar(imp, title="Feature importance (RandomForest)")
            st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.info("Not enough numeric features to run clustering (need at least 2 among value/sqft/value_psf/building_age).")

# -----------------------
# Download (filtered)
# -----------------------
st.markdown("---")
st.subheader("Downloads")
st.caption("Get the current filtered view (including cluster labels if computed).")
csv = f.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered CSV", csv, file_name="rwap_filtered.csv", mime="text/csv")
