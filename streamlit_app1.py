import numpy as np, pandas as pd, streamlit as st, plotly.express as px, pydeck as pdk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="RWAP – Dashboard & ML", page_icon="📊", layout="wide")
st.title("RWAP – Asset Valuation Dashboard & ML (Upload your CSV in sidebar)")

def first_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def canonicalize(df):
    mapping = {
        "loc_code":["Location Code","loc_code"],
        "asset_name":["Real Property Asset Name","Asset Name","asset_name"],
        "city":["City","City_x","city"],
        "state":["State","State_x","state"],
        "zip":["Zip Code","ZIP","zip"],
        "lat":["Latitude","Latitude_x","lat"],
        "lon":["Longitude","Longitude_x","lon"],
        "sqft":["Building Rentable Square Feet","SqFt","sqft"],
        "value":["Estimated Asset Value (Adj)","Estimated Asset Value","Est Value (Base)","value"],
        "conf_cat":["Confidence Category","conf_cat"],
        "asset_type":["Real Property Asset Type","Asset Type","Type"],
        "age":["Building Age","age"],
        "cluster":["Asset Cluster","cluster"],
    }
    out = pd.DataFrame()
    for k,cands in mapping.items():
        hit = first_col(df,cands)
        if hit is not None: out[k] = df[hit]
    for n in ["lat","lon","sqft","value","age"]:
        if n in out.columns: out[n] = pd.to_numeric(out[n], errors="coerce")
    if "value" in out.columns and "sqft" in out.columns:
        out["value_psf"] = out["value"]/out["sqft"]
    else:
        out["value_psf"] = np.nan
    if "conf_cat" not in out.columns: out["conf_cat"] = "Unknown"
    return out

st.sidebar.header("Upload CSV")
csv = st.sidebar.file_uploader("Pick the CSV you exported from Colab", type=["csv"])
if csv is None:
    st.info("Export from Colab (e.g., asset_valuation_results_final_with_confidence.csv) and upload here.")
    st.stop()

raw = pd.read_csv(csv)
df  = canonicalize(raw)

tab_dash, tab_ml = st.tabs(["📊 Task 2: Dashboard", "🤖 Task 3: ML"])

with tab_dash:
    st.subheader("Filters")
    states = sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else []
    types  = sorted(df["asset_type"].dropna().unique().tolist()) if "asset_type" in df.columns else []
    confs  = sorted(df["conf_cat"].dropna().unique().tolist()) if "conf_cat" in df.columns else []

    c1,c2,c3,c4 = st.columns(4)
    sel_states = c1.multiselect("State", states, default=states[:10] if states else [])
    sel_types  = c2.multiselect("Asset Type", types, default=types if types else [])
    sel_confs  = c3.multiselect("Confidence", confs, default=confs if confs else [])
    name_q     = c4.text_input("Search asset/ZIP contains", "")

    if "value" in df.columns:
        vmin,vmax = float(df["value"].min()), float(df["value"].max())
        v_range = st.slider("Value range", vmin, vmax, (vmin, vmax))
    if "sqft" in df.columns:
        smin,smax = float(df["sqft"].min()), float(df["sqft"].max())
        s_range = st.slider("Sq.Ft range", smin, smax, (smin, smax))

    flt = df.copy()
    if sel_states and "state" in flt.columns: flt = flt[flt["state"].isin(sel_states)]
    if sel_types  and "asset_type" in flt.columns: flt = flt[flt["asset_type"].isin(sel_types)]
    if sel_confs  and "conf_cat" in flt.columns: flt = flt[flt["conf_cat"].isin(sel_confs)]
    if name_q:
        m = False
        if "asset_name" in flt.columns: m = flt["asset_name"].astype(str).str.contains(name_q, case=False, na=False)
        if "zip" in flt.columns:       m = m | flt["zip"].astype(str).str.contains(name_q, na=False)
        flt = flt[m]
    if "value" in flt.columns: flt = flt[(flt["value"]>=v_range[0]) & (flt["value"]<=v_range[1])]
    if "sqft" in flt.columns:  flt = flt[(flt["sqft"]>=s_range[0]) & (flt["sqft"]<=s_range[1])]

    st.markdown("### KPIs (Filtered)")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Assets", f"{len(flt):,}")
    k2.metric("Total Value", f"${flt['value'].sum():,.0f}" if "value" in flt.columns else "—")
    k3.metric("Median Value", f"${flt['value'].median():,.0f}" if "value" in flt.columns else "—")
    k4.metric("Median $/ft²", f"${flt['value_psf'].median():,.0f}" if "value_psf" in flt.columns else "—")

    st.markdown("### Distributions")
    cA,cB = st.columns(2)
    if "value" in flt.columns:
        fig = px.histogram(flt, x="value", nbins=60, title="Asset Value"); fig.update_xaxes(type="log")
        cA.plotly_chart(fig, use_container_width=True)
    if "value_psf" in flt.columns:
        fig2 = px.histogram(flt.replace([np.inf,-np.inf], np.nan).dropna(subset=["value_psf"]),
                            x="value_psf", nbins=60, title="Value per Sq.Ft")
        cB.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Value vs Size (log-log)")
    if all(c in flt.columns for c in ["sqft","value"]):
        fig3 = px.scatter(flt, x="sqft", y="value",
                          hover_data=[c for c in ["asset_name","state","zip"] if c in flt.columns],
                          title="Value vs Rentable Sq.Ft")
        fig3.update_xaxes(type="log"); fig3.update_yaxes(type="log")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Map")
    if all(c in flt.columns for c in ["lat","lon"]):
        v = np.clip(flt["value"].fillna(1.0).astype(float), 1.0, None) if "value" in flt.columns else 1.0
        vmed = float(np.median(v)) if len(np.atleast_1d(v)) else 1.0
        radius = 200 + 200 * np.log10(v / vmed)
        data = flt.copy(); data["radius"]=radius; data["color"]=[[50,100,200]]*len(data)
        view = pdk.ViewState(latitude=float(data["lat"].mean()), longitude=float(data["lon"].mean()), zoom=4)
        layer = pdk.Layer("ScatterplotLayer", data=data, get_position='[lon, lat]',
                          get_radius="radius", get_fill_color="color", pickable=True, auto_highlight=True)
        tooltip = {"html":"<b>{asset_name}</b><br/>${value:,.0f}<br/>{state} {zip}",
                   "style":{"backgroundColor":"steelblue","color":"white"}}
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip), use_container_width=True)
    else:
        st.info("No lat/lon columns found for mapping.")

    st.markdown("### Top 50 by Value")
    if "value" in flt.columns:
        top = flt.sort_values("value", ascending=False).head(50)
        st.dataframe(top, use_container_width=True)
    st.download_button("⬇️ Download filtered CSV", data=flt.to_csv(index=False),
                       file_name="filtered_assets.csv", mime="text/csv")

with tab_ml:
    st.subheader("Clustering (KMeans) + Classification (RandomForest)")

    must = ["value","sqft","value_psf"]
    if not all(c in df.columns for c in must):
        st.warning("Need columns: value, sqft, value_psf in your CSV.")
        st.stop()

    ml = df[["value","sqft","value_psf"] + (["age"] if "age" in df.columns else [])].copy()
    ml = ml.replace([np.inf,-np.inf], np.nan).fillna(ml.median(numeric_only=True))
    for c in ["value","sqft","value_psf"]:
        ml[f"log_{c}"] = np.log(np.clip(ml[c], 1, None))
    X_cols = [c for c in ml.columns if c.startswith("log_")] + (["age"] if "age" in ml.columns else [])
    scaler = StandardScaler(); X = scaler.fit_transform(ml[X_cols])

    k = st.slider("Choose k (clusters)", 3, 8, 3, 1)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    st.write(f"**Silhouette:** {sil:.3f}")
    df_ml = df.copy(); df_ml["Asset Cluster"] = labels

    sizes = pd.Series(labels).value_counts().sort_index().rename("count")
    st.write("**Cluster sizes**"); st.dataframe(sizes)

    prof_cols = [c for c in ["value","sqft","value_psf","age"] if c in df_ml.columns]
    profile = df_ml.groupby("Asset Cluster")[prof_cols].median().sort_index()
    st.write("**Cluster profiles (median)**"); st.dataframe(profile)

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    figp = px.scatter(pd.DataFrame({"PC1":X2[:,0],"PC2":X2[:,1],"Cluster":labels}),
                      x="PC1", y="PC2", color="Cluster", title="PCA (2D) by Cluster", opacity=0.7, height=500)
    st.plotly_chart(figp, use_container_width=True)

    st.download_button("⬇️ Download data with clusters",
                       data=df_ml.to_csv(index=False), file_name="t3_assets_with_clusters.csv",
                       mime="text/csv")

    st.markdown("---")
    st.subheader("RandomForest: predict cluster")
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.30, random_state=42, stratify=labels)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train); y_pred = rf.predict(X_test)
    acc = (y_pred==y_test).mean(); st.write(f"**Accuracy:** {acc:.3f}")
    cm = confusion_matrix(y_test, y_pred)
    st.write("**Confusion Matrix**"); st.dataframe(pd.DataFrame(cm,
            index=[f"True {i}" for i in sorted(np.unique(labels))],
            columns=[f"Pred {i}" for i in sorted(np.unique(labels))]))
    st.write("**Classification Report**"); st.text(classification_report(y_test, y_pred))
    imp = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False)
    st.plotly_chart(px.bar(imp, title="Feature Importances", labels={"index":"feature","value":"importance"}),
                    use_container_width=True)

st.caption("Tip: export the CSV from Colab Task 1/2 (it must include value, sqft, lat, lon, etc.), then upload here.")
