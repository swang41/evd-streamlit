# ev_helpers.py
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import genextreme as gev, genpareto as gpd, norm, expon, pareto, uniform

# ---- Public glossary (used on Overview) ----
GLOSSARY = {
    "Extreme value": "An unusually large (or small) observation — e.g., a record flood level.",
    "Block maxima": "Split data into equal blocks (e.g., years) and take the maximum of each block.",
    "GEV": "Generalized Extreme Value distribution; a family that models block maxima.",
    "Tail index (xi)": "Single number that indicates tail type; >0 heavy, ≈0 light, <0 bounded support.",
    "POT": "Peaks Over Threshold — model values that exceed a high threshold.",
    "GPD": "Generalized Pareto distribution; models threshold exceedances.",
    "Return level": "A level expected to be exceeded on average once every T periods.",
    "Mean excess": "Average exceedance over a threshold u, given X>u."
}

# ---- Sampling from parent distributions ----
def sample_parent(name: str, size: int):
    if name == "Normal(0,1)":
        return norm.rvs(size=size)
    if name == "Exponential(1)":
        return expon.rvs(size=size)
    if name == "Pareto(α=3, xm=1)":
        return pareto.rvs(b=3, scale=1, size=size)  # heavy tail
    if name == "Uniform(0,1)":
        return uniform.rvs(size=size)
    raise ValueError(f"Unknown parent distribution: {name}")

# ---- Plot helpers ----
def qq_plot(data, dist, params, name="QQ Plot"):
    probs = (np.arange(1, len(data)+1) - 0.5) / len(data)
    theo_q = dist.ppf(probs, *params)
    samp_q = np.sort(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theo_q, y=samp_q, mode="markers", name="Sample vs Theoretical"))
    lo = float(min(theo_q.min(), samp_q.min()))
    hi = float(max(theo_q.max(), samp_q.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles", title=name)
    return fig

def return_level_curve_gev(p_tail, c, loc, scale):
    q = 1.0 - p_tail  # non-exceedance probability
    return gev.ppf(q, c=c, loc=loc, scale=scale), q

# ---- Tail classification + badge ----
def tail_type(xi: float):
    if xi > 0.10:  return "Heavy tail",  "🔥", "badge--heavy"
    if xi < -0.10: return "Bounded tail","🧱", "badge--bounded"
    return "Light tail","🌤️","badge--light"

def badge(label: str, emoji: str, cls: str):
    st.markdown(f'<span class="badge {cls}">{emoji} {label}</span>', unsafe_allow_html=True)

# ---- Small UI utilities ----
def info_card(title, body):
    st.markdown(f"### {title}")
    st.write(body)

def why_tab(question: str, use_when: str, see: str, expanded: bool = True):
    with st.expander("Why this tab?", expanded=expanded):
        st.markdown(f"**Question:** {question}")
        st.markdown(f"**Use when:** {use_when}")
        st.markdown(f"**What to see:** {see}")
