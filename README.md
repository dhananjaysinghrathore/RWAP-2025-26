# RWAP – Asset Valuation, GIS Analytics & ML

**Live App:** https://rwap-2025-26-qszcqjxgg4hqcrmuacen3o.streamlit.app/

This project delivers a full pipeline for valuing real-estate assets, exploring the data with a GIS-enabled analytical dashboard, and building machine-learning models to cluster and classify assets. It was built in **Google Colab** (analysis & model development) and deployed as an interactive **Streamlit** web app (presentation & decision support).

---

## Assignment Scope

**Project Sample Data:** Dataset-1 (Assets) + Dataset-2 (Valuation Index by ZIP/Region)  
**Dataset Type:** Structured, GIS  
**Software:** Python (Colab, Streamlit), Python libraries (see below)  
**Deliverables:** Analytical Dashboard (web), Analytical Report (Colab/Notebook), Project Presentation (≤10 slides)  
**Storage:** Code & Report in GitHub (folder `RWAP_2025-26`)

### Tasks
1) **Asset Valuation Model** — Estimate asset values in Dataset-1 using valuation index in Dataset-2.  
2) **Analytical Dashboard (GIS)** — Descriptive & inferential statistics, spatial analysis & mapping.  
3) **Machine Learning** — Unsupervised clustering to define asset classes; supervised learning to classify/predict asset classes based on valuation.

---

## Quick Links

- **Live Streamlit App:** https://rwap-2025-26-qszcqjxgg4hqcrmuacen3o.streamlit.app/
- **Notebook (Colab):** _RWAP_2025-26_Code.ipynb_ (contains EDA, GIS, OLS, LISA, clustering, training)
- **Data (expected location in repo):** `data/asset_valuation_results_final_with_confidence.csv.gz`  
  (or set a public CSV URL in Streamlit **Secrets** as `DATA_URL`)

---

## Project Overview

- **Goal:** Produce credible, transparent estimates of asset value and provide a decision dashboard that surfaces patterns by location, size, age, and market index; then learn stable **asset classes** from the data and build a fast classifier to score new assets.
- **Approach:**  
  1) **Valuation** = rentable square feet × latest ZIP (or region) valuation index, with quality flags.  
  2) **Analytics & GIS:** Descriptive stats, distributions, value vs size, interactive map (points/heatmap), and spatial autocorrelation (Moran’s I & Local Moran’s clusters).  
  3) **ML:** KMeans clustering (auto-choose **k**; report silhouette), PCA visualization, RandomForest classifier to predict cluster for new assets, with cross-validated accuracy and feature importances.

---

## Key Results & Insights

> _Numbers below reflect the latest successful runs in Colab/Streamlit. Your exact figures may differ slightly as filters or data refresh change._

### Valuation & Stats
- **Value vs Size:** strong positive log-log relationship; larger sqft → higher value.
- **Age effect:** negative—older buildings trend lower in value after controlling for size.
- **OLS (log value ~ log sqft + age):**  
  - **R² ≈ 0.913**, very high explanatory power.  
  - **log_sqft** positive and highly significant.  
  - **Building Age** negative and significant.

### Spatial Analysis (GIS)
- **Global Moran’s I (value_psf):** **0.4695** with **p = 0.001** → **strong spatial clustering** of values.  
- **Local Moran’s (LISA) cluster counts:**  
  - **HH:** 1005  • **LL:** 2126  • **LH:** 248  • **HL:** 75  • **Not Sig:** 5198  
  → High-value corridors (HH) and low-value basins (LL) are statistically meaningful.

### Machine Learning
- **Best K (KMeans):** **k = 3** with silhouette ≈ **0.536** (good cluster separation for this domain).  
- **Cluster sizes (example run):** 0 → 6831, 1 → 589, 2 → 1232.  
- **Cluster profiles (medians):**
  - **Cluster 0:** large stock of assets with moderate size and value_psf.  
  - **Cluster 1:** very small sqft but extremely high $/ft² (niche/high-demand).  
  - **Cluster 2:** larger sqft, higher total value, older assets (watch capex).  
- **Classifier (RandomForest) to predict cluster:**
  - **Held-out accuracy ≈ 0.997**; **CV mean ≈ 0.995** (stable).  
  - **Top features:** Building Age, Estimated Value (Adj, log), Value per sq.ft (log), Sq.ft (log).

### Managerial Takeaways
- **Prioritize HH clusters** for retention/capacity expansion; protect from under-investment.  
- **Target LL clusters** for lease renegotiation, consolidation, or disposal.  
- **Age-sensitive maintenance**: older assets systematically underperform after controlling for size—plan refurbishments or divestment accordingly.  
- **Sizing strategy:** value scales strongly with sqft, but diminishing returns appear in $/ft²—right-size new acquisitions.

---

## Live Dashboard (what you can do)

- **Filter** by State, Asset Type, Confidence, value/sqft ranges, or search by Asset/ZIP.  
- **KPIs** formatted in **Bn/Mn** (Total value, Median value, Median $/ft²).  
- **Map (above charts):**  
  - **Points** (size scales in km, visible at any zoom) or **Heatmap**.  
  - **Color by** Confidence / Asset Type / State / Cluster (with legend).  
- **Charts:** Distributions (Value, $/ft²), Value vs Rentable Sq.Ft (log-log).  
- **Downloads:** export the **filtered** subset and **clustered** data (Task-3 tab).  
- **ML Tab:** choose **k**, see silhouette, cluster profiles, PCA 2D, classifier accuracy, confusion matrix, and feature importances.

---

## Methodology (short)

1. **Valuation Join**  
   - Clean and standardize ZIP/Region keys.  
   - Merge Dataset-1 (assets) with Dataset-2 (valuation index).  
   - Compute `Estimated Asset Value (Adj) = Rentable Sq.Ft × Latest Index`.  
   - Derive `value_psf = value / sqft`. Add confidence flags.

2. **Descriptive & Inferential**  
   - Summary stats; Pearson/Spearman correlations.  
   - **OLS:** `log(value) ~ log(sqft) + age (+ optional controls)`.

3. **Spatial Analytics**  
   - Build distance-band weights; compute **Moran’s I** and **Local Moran’s**.  
   - Map clusters (HH/LL/LH/HL) with folium/pydeck.

4. **Machine Learning**  
   - Features: log(value), log(sqft), log(value_psf), and age (optional).  
   - **KMeans** (silhouette to pick k); **PCA** for 2D plot.  
   - **RandomForest** classifier (train/test split + cross-val).  
   - Save artifacts/outputs for dashboard.

---
