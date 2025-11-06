# app/pages/2_Forecast_Simple.py
from __future__ import annotations
import os, sys, json, joblib
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
import streamlit as st

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils.io import load_processed  # noqa: E402
from src.pipelines.targets import build_feature_frame  # noqa: E402

st.set_page_config(page_title="Forecast (Simple) — MNREGA", layout="wide")

MODE = {
    "demand":      {"label":"Demand","color":"#0D6EFD"},
    "utilization": {"label":"Budget Utilization","color":"#7C3AED"},
    "completion":  {"label":"Works Completion","color":"#0CA678"},
}
NAME_MAP = {"Demand":"demand","Budget Utilization":"utilization","Works Completion":"completion"}
BAD = {"y_hat_seasonal","_ds"}

@st.cache_resource
def raw(): return load_processed()

def latest_artifacts():
    root="models/artifacts"
    if not os.path.isdir(root): return None
    subs=[os.path.join(root,d) for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]
    return sorted(subs, reverse=True)[0] if subs else None

@st.cache_resource
def load_bundle(root_dir: str, key: str):
    base=os.path.join(root_dir,key)
    model=joblib.load(os.path.join(base,"model.joblib"))
    q=json.load(open(os.path.join(base,"conformal_q.json")))
    spec=json.load(open(os.path.join(base,"feature_spec.json")))
    feats=[c for c in spec["features"] if c not in BAD and not c.startswith("_")]
    metrics={}
    mpath=os.path.join(base,"metrics_cv.json")
    if os.path.exists(mpath): metrics=json.load(open(mpath))
    return model,q,feats,metrics,base

def anchor(fin_year:str, month:int)->datetime:
    return datetime(int(str(fin_year)[:4]), int(month), 1)

def make_future(df, district, fin_year, month, horizon, overrides):
    hist=df[df["district_name"].astype(str).eq(district)].copy()
    if hist.empty: return pd.DataFrame()
    a=anchor(fin_year,month); y,m=a.year,a.month
    rows=[{"district_name":district,"fin_year":f"{y+((m-1+k)//12)}-{y+((m-1+k)//12)+1}","month":((m-1+k)%12)+1} for k in range(horizon)]
    fut=pd.DataFrame(rows)
    for c,v in overrides.items(): fut[c]=v
    combo=pd.concat([hist,fut],ignore_index=True,sort=False)
    feats=build_feature_frame(combo)
    return feats.tail(horizon).copy()

def risk_banner(tkey:str, y:np.ndarray, lo80:np.ndarray, hi80:np.ndarray)->tuple[str,str]:
    if tkey=="utilization":
        val=y[0]
        if val>0.95: return ("Shortfall risk","background-color:#FFF3CD;color:#664D03;")
        if val<0.60: return ("Idle funds risk","background-color:#E7F5FF;color:#0B7285;")
        return ("Healthy","background-color:#E6FCF5;color:#087F5B;")
    # demand/completion
    width=float(np.mean(hi80-lo80))
    if width>np.percentile(hi80-lo80,70): 
        return ("Wide uncertainty","background-color:#FFF3CD;color:#664D03;")
    return ("Stable","background-color:#E6FCF5;color:#087F5B;")

def horizon_tiles(xlab:List[str], y:np.ndarray, lo80:np.ndarray, hi80:np.ndarray, is_util:bool):
    cols=st.columns(min(6,len(xlab)))
    for i,period in enumerate(xlab):
        with cols[i%len(cols)]:
            with st.container(border=True):
                st.caption(period)
                if is_util:
                    st.markdown(f"<div style='font-size:1.4rem;font-weight:700'>{y[i]:.2f}</div>", unsafe_allow_html=True)
                    st.caption(f"80%: {lo80[i]:.2f} – {hi80[i]:.2f}")
                else:
                    st.markdown(f"<div style='font-size:1.4rem;font-weight:700'>{int(round(y[i])):,}</div>", unsafe_allow_html=True)
                    st.caption(f"80%: {int(round(lo80[i])):,} – {int(round(hi80[i])):,}")

# ---------- UI ----------
st.title("Forecast (Simple)")
st.caption("A clean, card‑first view for quick planning.")

df=raw()
root=latest_artifacts()
if not root: 
    st.error("No artifacts found. Train with: python -m src.models.train_suite")
    st.stop()

left,right=st.columns([1.0,2.0],gap="large")
with left:
    with st.form("controls"):
        pred_label=st.selectbox("Prediction", list(NAME_MAP.keys()))
        key=NAME_MAP[pred_label]; theme=MODE[key]
        horizon=st.slider("Horizon (months)",1,6,3)
        district=st.selectbox("District", sorted(df["district_name"].astype(str).unique()))
        month=st.selectbox("Month", sorted(df["month"].astype(int).unique()))
        fin_year=st.selectbox("Financial Year", sorted(df["fin_year"].astype(str).unique()))
        st.markdown("#### Scenario drivers")
        c1,c2,c3=st.columns(3)
        budget=c1.number_input("Approved_Labour_Budget", value=float(df["Approved_Labour_Budget"].median()))
        wage=c2.number_input("Avg_Wage_rate_per_day", value=float(df["Average_Wage_rate_per_day_per_person"].median()))
        cards=c3.number_input("Active_Job_Cards", value=float(df["Total_No_of_Active_Job_Cards"].median()))
        submit=st.form_submit_button(f"Predict {pred_label}", use_container_width=True)

with right:
    if not submit:
        st.info("Pick district/month/year, adjust drivers, then click Predict.")
        st.stop()

    model,q,feats,metrics,base=load_bundle(root,key)
    overrides={"Approved_Labour_Budget":budget,"Average_Wage_rate_per_day_per_person":wage,"Total_No_of_Active_Job_Cards":cards}
    fut=make_future(df,district,fin_year,int(month),int(horizon),overrides)
    if fut.empty:
        st.error("Not enough history to build features for this selection. Try a later month/year.")
        st.stop()
    X=fut.reindex(columns=feats, fill_value=0)
    if X.empty:
        st.error("Feature matrix empty. Try different controls.")
        st.stop()

    y=model.predict(X)
    lo80,hi80=y-q.get("q80",0.0), y+q.get("q80",0.0)
    lo95,hi95=y-q.get("q95",0.0), y+q.get("q95",0.0)
    xlab=[f'{str(r["fin_year"])[:4]}-{int(r["month"])}' for _,r in fut[["fin_year","month"]].iterrows()]
    nextp=xlab[0]

    # Headline card
    with st.container(border=True):
        st.markdown(f"<div style='font-size:1.1rem;color:#6c757d'>{pred_label} • {district} • {nextp}</div>", unsafe_allow_html=True)
        if key=="utilization":
            st.markdown(f"<div style='font-size:2rem;font-weight:800;color:{theme['color']}'>{y[0]:.2f}</div>", unsafe_allow_html=True)
            st.caption(f"80%: {lo80[0]:.2f} – {hi80[0]:.2f} • 95%: {lo95[0]:.2f} – {hi95[0]:.2f}")
        else:
            st.markdown(f"<div style='font-size:2rem;font-weight:800;color:{theme['color']}'>{int(round(y[0])):,}</div>", unsafe_allow_html=True)
            st.caption(f"80%: {int(round(lo80[0])):,} – {int(round(hi80[0])):,} • 95%: {int(round(lo95[0])):,} – {int(round(hi95[0])):,}")

    # Risk banner
    text, style=risk_banner(key,y,lo80,hi80)
    st.markdown(f"<div style='padding:.6rem 1rem;border-radius:6px;{style}'>Status: {text}</div>", unsafe_allow_html=True)

    # Horizon tiles (small cards instead of chart)
    st.markdown("#### Horizon")
    horizon_tiles(xlab, y, lo80, hi80, is_util=(key=='utilization'))

    # Details table (Arrow-safe types)
    st.markdown("#### Details")
    table=pd.DataFrame({"period":xlab,"point":y,"lo80":lo80,"hi80":hi80,"lo95":lo95,"hi95":hi95})
    table["period"]=table["period"].astype(str)
    for c in ["point","lo80","hi80","lo95","hi95"]:
        table[c]=pd.to_numeric(table[c], errors="coerce")
    if key!="utilization":
        table[["point","lo80","hi80","lo95","hi95"]]=table[["point","lo80","hi80","lo95","hi95"]].round(0)
    else:
        table[["point","lo80","hi80","lo95","hi95"]]=table[["point","lo80","hi80","lo95","hi95"]].round(3)
    st.dataframe(table, use_container_width=True, hide_index=True)

    rmse=metrics.get("rmse_mean","—"); mae=metrics.get("mae_mean","—"); base_rmse=metrics.get("baseline_rmse_mean", None)
    meta=f"Artifacts: {base} • CV RMSE: {rmse} • CV MAE: {mae}"
    if base_rmse is not None: meta += f" • Baseline RMSE (demand): {base_rmse}"
    st.caption(meta)
