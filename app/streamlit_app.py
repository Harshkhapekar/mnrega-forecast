# app/streamlit_app.py
from __future__ import annotations
import os, sys
from typing import Optional
import streamlit as st

# -------------- Setup --------------
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

st.set_page_config(page_title="MNREGA Forecast Hub", layout="wide")

# -------------- Color System --------------
# Turn the dial here to recolor the whole page
PRIMARY = "#8B5CF6"   # Purple-500
ACCENT  = "#06B6D4"   # Cyan-500
SECOND  = "#22C55E"   # Green-500
WARM    = "#F59E0B"   # Amber-500
DANGER  = "#EF4444"   # Red-500
INK     = "#0B1220" if st.get_option("theme.base")=="dark" else "#0F172A"
MUTED   = "#94A3B8"   # Slate-400

def chip(text, bg, fg):
    return f"<span style='background:{bg};color:{fg};padding:.35rem .6rem;border-radius:999px;font-weight:700'>{text}</span>"

# -------------- Helpers --------------
def latest_artifacts_root() -> Optional[str]:
    root = "models/artifacts"
    if not os.path.isdir(root): return None
    subs = [os.path.join(root,d) for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]
    return sorted(subs, reverse=True)[0] if subs else None

def artifacts_ready(root: Optional[str]) -> bool:
    if not root: return False
    for key in ["demand","utilization","completion"]:
        base = os.path.join(root, key)
        if not (
            os.path.exists(os.path.join(base,"model.joblib")) and
            os.path.exists(os.path.join(base,"feature_spec.json")) and
            os.path.exists(os.path.join(base,"conformal_q.json"))
        ):
            return False
    return True

def page_exists(rel_from_app: str) -> bool:
    return os.path.isfile(os.path.join("app", rel_from_app))

DATASET_PAGE = "pages/1_Dataset_Explorer.py"
SUITE_PAGE   = "pages/2_Forecast_Suite.py"

# -------------- HERO (full color) --------------
st.markdown(f"""
<div style="
  padding: 26px 28px; border-radius: 16px;
  background: conic-gradient(from 180deg at 30% 20%, {PRIMARY}25, transparent 40%),
             radial-gradient(900px 320px at 90% 10%, {ACCENT}26, transparent 60%),
             linear-gradient(135deg, {PRIMARY}14, {ACCENT}14);
  border: 1px solid {PRIMARY}33;">
  <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:18px;flex-wrap:wrap">
    <div>
      <div style="font-size:14px;letter-spacing:.12em;text-transform:uppercase;color:{ACCENT}">MNREGA • Forecasting</div>
      <div style="font-size:42px;font-weight:900;color:#fff;line-height:1.05;margin:.25rem 0">Colorful Forecast Hub</div>
      <div style="max-width:840px;color:#E5E7EB">
        Explore, simulate and plan with bold visuals: Demand, Utilization and Completion forecasts with 80/95% bands and scenario drivers.
      </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <a href="#tools" style="text-decoration:none">
        <div style="padding:10px 14px;border-radius:10px;background:{SECOND};color:#001016;font-weight:800;border:1px solid #00000022">Open tools →</div>
      </a>
      <a href="#about" style="text-decoration:none">
        <div style="padding:10px 14px;border-radius:10px;background:transparent;color:#fff;font-weight:800;border:1px dashed #ffffff66">Learn more</div>
      </a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------- STATUS STRIP (chips + gradient) --------------
root = latest_artifacts_root()
ready = artifacts_ready(root)
state = chip("READY",  f"{SECOND}33", "#031a0f") if ready else chip("INCOMPLETE", f"{WARM}33", "#3b2500") if root else chip("NOT FOUND", f"{DANGER}33", "#2a0606")

st.markdown(f"""
<div style="
  margin-top:14px;padding:12px 14px;border-radius:12px;
  background: linear-gradient(90deg, {PRIMARY}14, transparent 40%, {ACCENT}14);
  border: 1px solid #ffffff29;">
  <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;color:#E5E7EB">
    <div style="font-weight:800;letter-spacing:.02em">Model status</div>
    <div>{state}</div>
    <div style="color:{MUTED}">Latest: {root if root else "—"}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------- HIGHLIGHTS (colored borders) --------------
st.markdown('<div id="about"></div>', unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns(4)
for i, (title, body, border) in enumerate([
    ("Scope", "District‑level Demand, Utilization, Completion.", PRIMARY),
    ("Models", "LightGBM + expanding‑window CV; fast search.", ACCENT),
    ("Uncertainty", "Conformal 80%/95% intervals per target.", SECOND),
    ("Artifacts", "Timestamped bundles with metrics & features.", WARM),
]):
    with [h1,h2,h3,h4][i]:
        st.markdown(f"""
        <div style="padding:12px;border-radius:12px;border-left:4px solid {border};background:linear-gradient(180deg,#ffffff07,#00000007)">
          <div style="font-weight:800">{title}</div>
          <div style="color:{MUTED}">{body}</div>
        </div>
        """, unsafe_allow_html=True)

# -------------- ACTIONS (colored headers) --------------
a1, a2 = st.columns([1.2, 1.0], gap="large")
with a1:
    st.markdown(f"<div style='font-weight:900;color:{PRIMARY};font-size:20px'>Quick actions</div>", unsafe_allow_html=True)
    st.write(f"- Train: python -m src.models.train_suite")
    st.write(f"- Launch: streamlit run app/streamlit_app.py")
    st.write(f"- Artifacts: models/artifacts/<timestamp>/")
    st.write(f"- Live‑reload on save.")
with a2:
    st.markdown(f"<div style='font-weight:900;color:{ACCENT};font-size:20px'>Tips</div>", unsafe_allow_html=True)
    st.write("- Use most recent FY and months for lags.")
    st.write("- Horizon 3 for planning, 1 to validate.")
    st.write("- Utilization: >0.95 shortfall, <0.60 idle funds.")
    st.write("- Tweak budget, wage, job cards; re‑predict.")

# -------------- TOOLS (big colorful cards) --------------
st.markdown('<div id="tools"></div>', unsafe_allow_html=True)
t1, t2 = st.columns(2)

def tool_card(col, title, desc, rel_from_app, grad_from, grad_to, emoji):
    with col:
        st.markdown(f"""
        <div style="padding:16px;border-radius:14px;
                    background: linear-gradient(135deg, {grad_from}33, {grad_to}33);
                    border:1px solid {grad_from}55;">
          <div style="display:flex;align-items:center;gap:10px">
            <div style="font-size:22px">{emoji}</div>
            <div style="font-weight:900;font-size:18px;color:#fff">{title}</div>
          </div>
          <div style="color:#E5E7EB;margin:.35rem 0 .75rem 0">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        if page_exists(rel_from_app):
            st.page_link(rel_from_app, label="Open →", icon=":material/arrow_circle_right:")
        else:
            st.button("Unavailable", disabled=True)
            st.caption(f"Add file: app/{rel_from_app}")

tool_card(
    t1,
    "Dataset Explorer",
    "Trend lines, seasonality, sanity checks.",
    DATASET_PAGE,
    PRIMARY, ACCENT, "📊"
)
tool_card(
    t2,
    "Forecast Suite",
    "Predict Demand, Utilization, Completion with 80/95% bands and scenarios.",
    SUITE_PAGE,
    SECOND, PRIMARY, "🧭"
)

# -------------- METHOD (colored bullets) --------------
with st.expander("Methodology", expanded=False):
    st.markdown(f"""
- <span style="color:{PRIMARY}">Pipeline:</span> load_processed() → build_feature_frame (lags, rolling, seasonal baseline).  
- <span style="color:{ACCENT}">Selection:</span> randomized LightGBM + expanding‑window CV (coarse step for speed).  
- <span style="color:{SECOND}">Uncertainty:</span> conformal q80/q95 from validation residuals.  
- <span style="color:{WARM}">Metrics:</span> CV RMSE/MAE and seasonal baseline RMSE (Demand) in metrics_cv.json.  
- <span style="color:{PRIMARY}">Serving:</span> latest timestamped artifacts auto‑selected; horizon anchored by month + FY.  
""", unsafe_allow_html=True)

# -------------- FOOTER RIBBON --------------
st.markdown(f"""
<div style="margin-top:18px;padding:10px 14px;border-radius:12px;
            background: linear-gradient(90deg,{PRIMARY}35,{ACCENT}35,{SECOND}35);
            color:#001016;border:1px solid #00000022;font-weight:800">
  Streamlit · LightGBM · Conformal Prediction · Expanding‑window CV
</div>
""", unsafe_allow_html=True)
