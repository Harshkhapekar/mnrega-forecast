# app/pages/1_Dataset_Explorer.py
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Setup ----------
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils.io import load_processed

st.set_page_config(page_title="Dataset Explorer — MNREGA", layout="wide")

# ---------- Color System ----------
PRIMARY = "#7C3AED"
ACCENT  = "#06B6D4"
SECOND  = "#22C55E"
WARM    = "#F59E0B"
DANGER  = "#EF4444"
MUTED   = "#94A3B8"

# Plotly setup
px.defaults.template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
px.defaults.color_continuous_scale = px.colors.sequential.Viridis
QUAL = px.colors.qualitative.Bold

# ---------- Custom CSS ----------
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.5rem; padding-bottom: 1.2rem; }}

.gradient-divider {{
  height: 6px; border-radius: 999px; margin: 1rem 0;
  background: linear-gradient(90deg, {PRIMARY}55, transparent 25%, {ACCENT}55, transparent 60%, {SECOND}55);
}}

h4 {{ font-weight: 800; letter-spacing: .03em; }}

div.hero {{
  padding: 28px; border-radius: 18px;
  background: radial-gradient(800px 320px at 0% -10%, {PRIMARY}22, transparent),
              radial-gradient(800px 320px at 100% 110%, {ACCENT}22, transparent);
  border: 1px solid #ffffff1a;
  box-shadow: 0 12px 30px rgba(0,0,0,.25);
}}

div.hero h1 {{
  font-size: 40px;
  background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 900;
  margin: 0;
}}

.metric-card {{
  display:flex; align-items:center; justify-content:center;
  flex-direction:column;
  background: linear-gradient(160deg, {PRIMARY}22, transparent);
  border: 1px solid rgba(255,255,255,.1);
  border-radius: 14px;
  padding: 1.2rem;
  min-height: 120px;
  transition: all .3s ease;
  text-align:center;
  box-shadow: 0 8px 16px rgba(0,0,0,.25);
}}
.metric-card:hover {{
  transform: translateY(-4px);
  box-shadow: 0 12px 20px rgba(0,0,0,.35);
}}

.metric-value {{
  font-size: 38px; font-weight: 900; line-height:1.1;
}}
.metric-label {{
  font-size: 14px; font-weight:700; color:{MUTED};
  letter-spacing: .08em; text-transform: uppercase;
}}
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown(f"""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:end;flex-wrap:wrap;">
    <div>
      <div style="font-size:14px;color:{MUTED};letter-spacing:.08em;text-transform:uppercase;">MNREGA • EDA Suite</div>
      <h1>Dataset Explorer</h1>
      <div style="max-width:860px;color:{MUTED};margin-top:8px;">Visually powerful data profiling for districts, years, and months — packed with plots, stats, and insights.</div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;">
      <a href="#data" style="text-decoration:none;"><div style="padding:10px 14px;border-radius:10px;background:{PRIMARY};color:#fff;font-weight:800">Jump to Data →</div></a>
      <a href="#viz"  style="text-decoration:none;"><div style="padding:10px 14px;border-radius:10px;border:1px solid #ffffff3a;color:{PRIMARY};font-weight:800">Visualize</div></a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Load ----------
@st.cache_resource
def load_all():
    df = load_processed()
    dd = {}
    if os.path.exists("reports/data_dictionary.json"):
        dd = json.load(open("reports/data_dictionary.json","r",encoding="utf-8"))
    eda_md = None
    if os.path.exists("reports/MLEda.md"):
        eda_md = open("reports/MLEda.md","r",encoding="utf-8").read()
    return df, dd, eda_md

df, data_dict, eda_md = load_all()

# ---------- Filters ----------
st.markdown(f"""
<div style="margin-top:18px;padding:14px;border-radius:14px;
            background:linear-gradient(180deg,#ffffff07,#00000012);
            border:1px solid #ffffff1a;box-shadow:inset 0 0 0 1px rgba(255,255,255,.06)">
  <div style="color:{MUTED};font-weight:800;margin-bottom:.4rem;">Global Filters</div>
</div>
""", unsafe_allow_html=True)

fc1, fc2, fc3 = st.columns([1,1,2])
district = fc1.selectbox("District", ["All"] + sorted(df["district_name"].astype(str).unique().tolist()))
fin_year = fc2.selectbox("Financial Year", ["All"] + sorted(df["fin_year"].astype(str).unique().tolist()))
search = fc3.text_input("Search column", "")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ---------- Normalize + Filter ----------
def normalize_dtypes(dframe: pd.DataFrame) -> pd.DataFrame:
    dframe = dframe.copy()
    if "month" in dframe: dframe["month"] = pd.to_numeric(dframe["month"], errors="coerce").astype("Int64")
    for c in ["fin_year","district_name","state_name","Remarks"]:
        if c in dframe: dframe[c] = dframe[c].astype("string")
    return dframe

df = normalize_dtypes(df)
mask = pd.Series(True, index=df.index)
if district != "All": mask &= df["district_name"].eq(district)
if fin_year != "All": mask &= df["fin_year"].eq(fin_year)
dfv = df.loc[mask].copy()

num_cols = [c for c in dfv.columns if pd.api.types.is_numeric_dtype(dfv[c])]
cat_cols = [c for c in dfv.columns if c not in num_cols]

# ---------- KPI Cards ----------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="metric-card" style="background:linear-gradient(160deg,{PRIMARY}33,transparent)">
      <div class="metric-value" style="color:{PRIMARY}">{len(dfv):,}</div>
      <div class="metric-label">Rows</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card" style="background:linear-gradient(160deg,{ACCENT}33,transparent)">
      <div class="metric-value" style="color:{ACCENT}">{len(num_cols):,}</div>
      <div class="metric-label">Numeric Columns</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card" style="background:linear-gradient(160deg,{SECOND}33,transparent)">
      <div class="metric-value" style="color:{SECOND}">{len(cat_cols):,}</div>
      <div class="metric-label">Categorical Columns</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- Catalog ----------
st.markdown(f"<h4 style='color:{ACCENT};margin-top:24px'>Column Catalog</h4>", unsafe_allow_html=True)
cols = [c for c in dfv.columns if (search.lower() in c.lower())] if search else list(dfv.columns)
rows = []
for col in cols:
    s = dfv[col]
    info = {
        "column": col,
        "dtype": str(s.dtype),
        "missing_%": round(float(s.isna().mean()*100), 2),
        "unique": int(s.nunique(dropna=True)),
        "desc": data_dict.get(col, "")
    }
    if pd.api.types.is_numeric_dtype(s):
        if s.notna().any():
            info.update({"min": float(np.nanmin(s)), "mean": float(np.nanmean(s)), "max": float(np.nanmax(s))})
        else:
            info.update({"min": None, "mean": None, "max": None})
    rows.append(info)
st.dataframe(pd.DataFrame(rows).sort_values(["dtype","column"]),
             use_container_width=True, height=360)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
# ---------- Tabs ----------
tab_vis, tab_target, tab_season, tab_segments, tab_quality, tab_data = st.tabs(
    ("🎨 Visualizations","🎯 Target","📅 Seasonality","🗺️ Segments","✅ Quality","🧾 Data")
)

# ===== Visualizations =====
with tab_vis:
    st.markdown('<div id="viz"></div>', unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{PRIMARY};margin-top:10px;'>Distributions</h4>", unsafe_allow_html=True)

    col_pick = st.selectbox("Select a column to visualize", dfv.columns, key="vis_col")
    if pd.api.types.is_numeric_dtype(dfv[col_pick]):
        c1, c2 = st.columns([2,1])
        with c1:
            fig = px.histogram(dfv, x=col_pick, nbins=60, marginal="box", color_discrete_sequence=QUAL)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12),
                hoverlabel=dict(font_size=12),
                height=420,
                margin=dict(l=36, r=28, t=36, b=36)
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown(f"##### Summary of `{col_pick}`")
            st.write(dfv[col_pick].describe(percentiles=[.05,.25,.5,.75,.95]).to_frame("value"))
    else:
        topn = st.slider("Top categories", 5, 100, 30, key="vis_topn")
        vc = dfv[col_pick].astype(str).value_counts().head(topn).reset_index()
        vc.columns = [col_pick, "count"]
        fig = px.bar(vc, x=col_pick, y="count", color="count", color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            margin=dict(l=36,r=28,t=36,b=36)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Correlation Heatmap ---
    st.markdown(f"<h4 style='color:{PRIMARY};margin-top:16px;'>Correlation Heatmap</h4>", unsafe_allow_html=True)
    if len(num_cols) >= 2:
        corr = dfv[num_cols].corr()
        with st.expander("Heatmap options", expanded=True):
            mode = st.radio("Feature subset", ["Top by variance","Top by target corr","All numeric"], index=0, horizontal=True)
            if mode == "Top by variance":
                topk = st.slider("Features shown", 10, min(120,len(num_cols)), min(50,len(num_cols)))
                keep = dfv[num_cols].var().sort_values(ascending=False).index[:topk]
            elif mode == "Top by target corr" and "Total_Households_Worked" in num_cols:
                topk = st.slider("Features shown", 10, min(120,len(num_cols)), min(50,len(num_cols)))
                keep = corr["Total_Households_Worked"].abs().sort_values(ascending=False).index[:topk]
            else:
                keep = num_cols
            base = keep[0]
            order = corr.loc[keep, base].sort_values(ascending=False).index
            corr_k = corr.loc[order, order]

        z = corr_k.values
        x_labels = corr_k.columns.tolist()
        y_labels = corr_k.index.tolist()
        heat = go.Heatmap(z=z, x=x_labels, y=y_labels, colorscale="Picnic", zmin=-1, zmax=1, colorbar=dict(title="corr"))
        fig = go.Figure(data=heat)
        ann = []
        for i, y in enumerate(y_labels):
            for j, x in enumerate(x_labels):
                v = z[i][j]
                ann.append(dict(x=x, y=y, text=f"{v:.2f}", showarrow=False,
                                font=dict(size=10, color=("white" if abs(v)>0.5 else "black")),
                                xanchor="center", yanchor="middle"))
        fig.update_layout(annotations=ann, height=900, margin=dict(l=120,r=120,t=60,b=120),
                          xaxis=dict(tickfont=dict(size=11), tickangle=45, automargin=True),
                          yaxis=dict(tickfont=dict(size=11), automargin=True))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

# ===== Target =====
with tab_target:
    st.markdown(f"<h4 style='color:{ACCENT}'>Target Relationships</h4>", unsafe_allow_html=True)
    if "Total_Households_Worked" in dfv.columns and len(num_cols) >= 2:
        corr = dfv[num_cols].corr()
        tc = corr["Total_Households_Worked"].sort_values(ascending=False).to_frame("corr").reset_index().rename(columns={"index":"feature"})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Top Positive Correlations")
            st.dataframe(tc.head(15), use_container_width=True, height=360)
        with c2:
            st.markdown("##### Top Negative Correlations")
            st.dataframe(tc.tail(15), use_container_width=True, height=360)

        st.markdown("#### Scatter with Trendline")
        feat_choices = [c for c in num_cols if c != "Total_Households_Worked"]
        if feat_choices:
            feat = st.selectbox("Select Feature", feat_choices, key="target_feat")
            fig = px.scatter(dfv.sample(min(len(dfv), 6000), random_state=42),
                             x=feat, y="Total_Households_Worked", trendline="ols",
                             color_discrete_sequence=QUAL)
            fig.update_layout(height=500, margin=dict(l=36,r=28,t=36,b=36))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Target column not found or insufficient numeric features.")

# ===== Seasonality =====
with tab_season:
    st.markdown(f"<h4 style='color:{SECOND}'>Seasonality Trends</h4>", unsafe_allow_html=True)
    if set(["fin_year","month","Total_Households_Worked"]).issubset(dfv.columns):
        fig = px.line(dfv.sort_values(["fin_year","month"]),
                      x="month", y="Total_Households_Worked", color="fin_year",
                      color_discrete_sequence=QUAL)
        fig.update_traces(line=dict(width=3))
        fig.update_layout(height=520, margin=dict(l=36,r=28,t=36,b=36))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Required columns for seasonality are missing.")

# ===== Segments =====
with tab_segments:
    st.markdown(f"<h4 style='color:{PRIMARY}'>District Segments</h4>", unsafe_allow_html=True)
    if set(["district_name","month","Total_Households_Worked"]).issubset(dfv.columns):
        heat = dfv.groupby(["district_name","month"], as_index=False)["Total_Households_Worked"].mean()
        pivot = heat.pivot(index="district_name", columns="month", values="Total_Households_Worked").fillna(0)
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Turbo")
        fig.update_layout(height=900, margin=dict(l=120,r=120,t=60,b=120))
        st.plotly_chart(fig, use_container_width=True)

    if set(["district_name","Total_Households_Worked"]).issubset(dfv.columns):
        st.markdown("#### District Summary (Filtered)")
        seg = (dfv.groupby("district_name", as_index=False)
                  .agg(count=("Total_Households_Worked","count"),
                       mean=("Total_Households_Worked","mean"),
                       median=("Total_Households_Worked","median"),
                       max=("Total_Households_Worked","max")))
        st.dataframe(seg.sort_values("mean", ascending=False),
                     use_container_width=True, height=420)

# ===== Quality =====
with tab_quality:
    st.markdown(f"<h4 style='color:{WARM}'>Data Quality Overview</h4>", unsafe_allow_html=True)
    miss = dfv.isna().mean().mul(100).round(2)
    miss_df = miss[miss > 0].sort_values(ascending=False).to_frame("missing_%")
    c1, c2 = st.columns(2)
    with c1:
        if not miss_df.empty:
            st.bar_chart(miss_df)
        else:
            st.info("No missing values in the current filter.")
    with c2:
        dtypes = pd.Series({c: str(dfv[c].dtype) for c in dfv.columns}).value_counts().to_frame("count")
        st.bar_chart(dtypes)

# ===== Data =====
with tab_data:
    st.markdown('<div id="data"></div>', unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{ACCENT}'>Filtered Data Table & Export</h4>", unsafe_allow_html=True)
    default_cols = list(dfv.columns)[:12]
    show_cols = st.multiselect("Columns to show", options=list(dfv.columns),
                               default=default_cols, key="table_cols")
    st.dataframe(dfv[show_cols], use_container_width=True, height=520)
    st.download_button(
        "Download Filtered CSV", dfv.to_csv(index=False).encode("utf-8"),
        file_name="processed_filtered.csv", mime="text/csv"
    )

# ---------- End ----------
st.markdown("<br><center><small>💜 MNREGA Dataset Explorer — Built with Streamlit & Plotly</small></center>", unsafe_allow_html=True)
