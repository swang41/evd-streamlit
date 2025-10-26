import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import genextreme as gev, genpareto as gpd

from stylekit import inject_css, apply_plot_style
from ev_helpers import (
    GLOSSARY, sample_parent, return_level_curve_gev,
    tail_type, badge, info_card, why_tab
)

# ------------------------
# App setup
# ------------------------
st.set_page_config(page_title="Extreme Value Theory — Visual Demo", layout="wide")
st.title("Extreme Value Theory — Visual Demo")
st.caption("Max per Period (GEV) • Top 5% Excesses (GPD) • 1-in-T Levels • Tail Check (MEF)")
inject_css()  # styling

# Beginner-only build (no toggle)
BEGINNER_MODE = True

# Defaults used across tabs
for key, val in {
    "parent": "Normal(0,1)",
    "block_size": 500,
    "n_blocks": 500,
    "pot_N": 50_000,
    "pot_quant": 0.95,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    st.header("Simulation Controls")
    st.caption("Pick a data source and how much to simulate. More data = steadier results.")

    seed = st.number_input("Random seed", value=42, step=1)
    np.random.seed(int(seed))

    parent = st.selectbox(
        "Parent distribution (i.i.d.)",
        ["Normal(0,1)", "Exponential(1)", "Pareto(α=3, xm=1)", "Uniform(0,1)"],
        index=["Normal(0,1)","Exponential(1)","Pareto(α=3, xm=1)","Uniform(0,1)"].index(st.session_state.parent),
        key="parent"
    )

    st.markdown("**Max per Period (GEV)**")
    st.caption("We split draws into equal groups, take the max of each group, and fit a GEV.")
    block_size = st.slider("Block size n", 20, 10000, st.session_state.block_size, step=20, key="block_size")
    n_blocks   = st.slider("Number of groups m", 20, 5000, st.session_state.n_blocks, step=20, key="n_blocks")

    st.markdown("---")
    st.markdown("**Top 5% Excesses (GPD)**")
    st.caption("We keep only values above a high cutoff and fit a GPD to those excesses.")
    pot_N      = st.slider("Total draws N (POT)", 1000, 200000, st.session_state.pot_N, step=1000, key="pot_N")
    pot_quant  = st.slider("Threshold quantile u", 0.80, 0.995, st.session_state.pot_quant, key="pot_quant")

    st.markdown("---")
    if st.button("Reset to sensible values"):
        st.session_state.update({
            "parent":"Normal(0,1)", "block_size":800, "n_blocks":800,
            "pot_N":80_000, "pot_quant":0.95
        })
        st.experimental_rerun()

# ------------------------
# Tabs
# ------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Max per Period (GEV)",
    "Top 5% Excesses (GPD)",
    "1-in-T Levels",
    "Tail Check (MEF)",
])

# -----------------------------------
# Overview
# -----------------------------------
with tab0:
    st.subheader("What this app is for")
    st.write(
        "A hands-on demo for outsiders to see the **interesting properties of the extreme-value distribution** "
        "(no prior stats needed)."
    )

    st.markdown("### Key properties this demo shows")
    st.markdown("""
- **Max per period → GEV**: Maxima of equal-sized groups follow the **GEV** family.  
  The **tail index (xi)** tells the tail type: **Light (~0)**, **Heavy (>0)**, or **Bounded (<0)**.
- **Top 5% exceedances → GPD**: Excesses above a high cutoff follow **GPD** with the **same tail index (xi)**; good thresholds show **stability**.
- **1-in-T levels**: Read rare-event levels (e.g., “1-in-100”) from the fitted GEV.
- **Mean-excess diagnostic**: Heavy tails rise roughly linearly; light tails bend down; bounded tails trend to zero.
    """)

    st.markdown("### Start here (60-second recipe)")
    st.markdown("""
1. In the sidebar, set **Parent** to **Normal(0,1)** → expect **Light tail** *(xi ≈ 0)*.  
2. Switch **Parent** to **Pareto(α=3)** → expect **Heavy tail** *(xi ≈ +0.33)*.  
3. Try **Uniform(0,1)** → expect **Bounded tail** *(xi ≈ −1)*.  

Then open **Max per Period (GEV)** and **Top 5% Excesses (GPD)** to see the tail index update.  
Check **1-in-T Levels** to read a “1-in-100” level.
    """)

    with st.expander("Glossary (plain English)", expanded=True):
        cols = st.columns(2)
        items = list(GLOSSARY.items())
        for i, (k, v) in enumerate(items):
            with cols[i % 2]:
                st.markdown(f"**{k}** — {v}")

# -----------------------------------
# Max per Period (GEV)
# -----------------------------------
with tab1:
    st.info("Pick a parent in the sidebar, then read the **tail index (xi)** below. Increase n and m to stabilize it.")
    st.subheader("Fisher–Tippett–Gnedenko (Block Maxima ⟶ GEV)")
    why_tab(
        question="If I take the max per year (or per batch), what describes those maxima?",
        use_when="You have records/peaks per fixed window (yearly floods, daily max loads, per-batch maxima).",
        see="Fit **GEV**; **xi** indicates tail type — ≈0 light, >0 heavy, <0 bounded.",
        expanded=True
    )

    X = sample_parent(st.session_state.parent, size=st.session_state.block_size * st.session_state.n_blocks).reshape(
        st.session_state.n_blocks, st.session_state.block_size
    )
    M = X.max(axis=1)

    c_hat, loc_hat, scale_hat = gev.fit(M)  # SciPy: c = -xi
    xi_hat = -c_hat

    xgrid = np.linspace(M.min() - 0.25*np.std(M), M.max() + 0.25*np.std(M), 600)
    pdf_hat = gev.pdf(xgrid, c=c_hat, loc=loc_hat, scale=scale_hat)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=M, histnorm="probability density", nbinsx=40, name="Max per group", opacity=0.6))
    fig.add_trace(go.Scatter(x=xgrid, y=pdf_hat, mode="lines", name="GEV fit (PDF)"))
    fig.update_layout(xaxis_title="Max value per group", yaxis_title="How often (density)",
                      legend=dict(orientation="h", y=1.05, x=0))
    apply_plot_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Badge + friendly metrics
    label, emoji, cls = tail_type(xi_hat)
    badge(label, emoji, cls)
    c1, c2, c3 = st.columns(3)
    c1.metric("Tail index (xi)", f"{xi_hat:.3f}")
    expected = {
        "Normal(0,1)": "Gumbel (~0)",
        "Exponential(1)": "Gumbel (~0)",
        "Pareto(α=3, xm=1)": "Fréchet (>0)",
        "Uniform(0,1)": "Weibull (<0)"
    }[st.session_state.parent]
    c2.metric("Expected tail class", expected)
    c3.metric("Groups × size", f"{st.session_state.n_blocks} × {st.session_state.block_size}")

# -----------------------------------
# Top 5% Excesses (GPD)
# -----------------------------------
with tab2:
    st.info("Set a high cutoff (e.g., u ≈ 0.95). The histogram shows **excesses**; read **tail index (xi)** below.")
    st.subheader("Pickands–Balkema–de Haan (Exceedances ⟶ GPD)")
    why_tab(
        question="If I keep only the very large values above a high cutoff, how do they behave?",
        use_when="Use all large events, not just one per period; best when you have plenty of data.",
        see="Excesses follow **GPD** with the same **xi**; choose a sensible threshold (enough points, not too low).",
        expanded=True
    )

    X = sample_parent(st.session_state.parent, size=st.session_state.pot_N)
    u = np.quantile(X, st.session_state.pot_quant)
    exceed = X[X > u]
    Y = exceed - u

    if len(Y) < 20:
        st.warning("Too few exceedances. Increase N or lower the threshold.")
    else:
        k_hat, _, scale_hat_gpd = gpd.fit(Y, floc=0.0)
        xgrid = np.linspace(0, np.percentile(Y, 99.5), 600)
        pdf_hat = gpd.pdf(xgrid, c=k_hat, loc=0.0, scale=scale_hat_gpd)

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=Y, histnorm="probability density", nbinsx=40, name="Excess above cutoff", opacity=0.6))
        fig2.add_trace(go.Scatter(x=xgrid, y=pdf_hat, mode="lines", name="GPD fit (PDF)"))
        fig2.update_layout(xaxis_title="Amount above the cutoff (excess)", yaxis_title="How often (density)",
                           legend=dict(orientation="h", y=1.05, x=0))
        apply_plot_style(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        # Badge + friendly metrics
        label_g, emoji_g, cls_g = tail_type(k_hat)
        badge(f"{label_g} (exceedances)", emoji_g, cls_g)

        c1, c2, c3 = st.columns(3)
        c1.metric("Threshold (quantile)", f"{st.session_state.pot_quant:.3f}")
        c2.metric("Exceedances", f"{len(Y)}")
        c3.metric("Tail index (xi)", f"{k_hat:.3f}")

# -----------------------------------
# 1-in-T Levels (GEV)
# -----------------------------------
with tab3:
    st.info("Translate your fitted GEV into **1-in-T** levels; good fits look near-linear on the Gumbel scale.")
    st.subheader("Return-Level Plot (GEV)")
    why_tab(
        question="What is the 1-in-100 (or 1-in-1000) level?",
        use_when="Communicating risk/design targets (flood defenses, wind loads, SLAs, VaR).",
        see="Near-linear curve on the **Gumbel scale** when GEV fits; hover to read off levels.",
        expanded=True
    )

    X = sample_parent(st.session_state.parent, size=st.session_state.block_size * st.session_state.n_blocks).reshape(
        st.session_state.n_blocks, st.session_state.block_size
    )
    M = X.max(axis=1)
    c_hat, loc_hat, scale_hat = gev.fit(M)

    T = np.geomspace(2, 1000, 50)
    p = 1.0 / T
    z, q = return_level_curve_gev(p, c_hat, loc_hat, scale_hat)
    gumbel_x = -np.log(-np.log(q))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=gumbel_x, y=z, mode="lines+markers", name="Return level curve"))
    fig3.update_layout(xaxis_title="Gumbel scale (straight lines = good fit)", yaxis_title="Rare-event level (1-in-T)",
                       legend=dict(orientation="h", y=1.05, x=0))
    apply_plot_style(fig3)
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------
# Tail Check (MEF)
# -----------------------------------
with tab4:
    st.subheader("Mean Excess Function (MEF) Diagnostics")
    why_tab(
        question="Is the tail heavy, light, or bounded — without fitting?",
        use_when="Quick diagnostic and for picking a threshold range for POT.",
        see="Heavy tails: **e(u)** rises ~linearly; light tails: bends **down**; bounded: trends to **0** near the endpoint.",
        expanded=True
    )
    st.info("Heavy tails rise roughly linearly; light tails bend down; bounded tails trend to 0 near the endpoint.")

    data = sample_parent(st.session_state.parent, size=max(5000, st.session_state.pot_N))
    us = np.quantile(data, np.linspace(0.6, 0.98, 30))
    e_u = []
    for u_ in us:
        exc = data[data > u_] - u_
        e_u.append(exc.mean() if len(exc) > 0 else np.nan)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=us, y=e_u, mode="lines+markers", name="Mean excess"))
    fig4.update_layout(xaxis_title="Threshold u", yaxis_title="Mean excess e(u)",
                       legend=dict(orientation="h", y=1.05, x=0))
    apply_plot_style(fig4)
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.markdown(
    "Key properties: **Max-stability (GEV)**, **Threshold stability (GPD)**, **1-in-T return levels**, **Mean excess**. "
    "Glossary is open by default; adjust sliders in the sidebar and explore."
)
