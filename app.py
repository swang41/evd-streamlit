import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import genextreme as gev, genpareto as gpd, norm, expon, pareto, uniform

st.set_page_config(page_title="Extreme Value Theory — Visual Demo", layout="wide")
st.title("Extreme Value Theory — Visual Demo")
st.caption("Block Maxima ⟶ GEV • Peaks Over Threshold ⟶ GPD • Return Levels • Mean Excess")

# ------------------------
# Sidebar controls
# ------------------------
with st.sidebar:
    st.header("Simulation Controls")
    seed = st.number_input("Random seed", value=42, step=1)
    np.random.seed(int(seed))

    parent = st.selectbox(
        "Parent distribution (i.i.d.)",
        ["Normal(0,1)", "Exponential(1)", "Pareto(α=3, xm=1)", "Uniform(0,1)"],
        index=0
    )

    st.markdown("**Block Maxima (GEV)**")
    block_size = st.slider("Block size n", 20, 10000, 500, step=20)
    n_blocks   = st.slider("Number of blocks m", 20, 5000, 500, step=20)

    st.markdown("---")
    st.markdown("**Peaks-Over-Threshold (GPD)**")
    pot_N      = st.slider("Total draws N (POT)", 1000, 200000, 50000, step=1000)
    pot_quant  = st.slider("Threshold quantile u", 0.80, 0.995, 0.95)

    st.markdown("---")
    st.markdown("**Playground (direct GEV/GPD sampling)**")
    pg_family = st.selectbox("Family", ["GEV", "GPD"])
    pg_xi     = st.slider("Shape ξ (GEV) / k (GPD)", -0.8, 0.8, 0.2, step=0.05)
    pg_loc    = st.number_input("Location", value=0.0, step=0.1, format="%.2f")
    pg_scale  = st.number_input("Scale (>0)", value=1.0, step=0.1, format="%.2f")

def sample_parent(dist, size):
    if dist == "Normal(0,1)":
        return norm.rvs(size=size)
    if dist == "Exponential(1)":
        return expon.rvs(size=size)
    if dist == "Pareto(α=3, xm=1)":
        return pareto.rvs(b=3, scale=1, size=size)  # heavy tail
    if dist == "Uniform(0,1)":
        return uniform.rvs(size=size)
    raise ValueError("Unknown parent distribution")

def qq_plot(data, dist, params, name="QQ Plot"):
    # dist: scipy frozen (with params) generator like gev/gpd
    probs = (np.arange(1, len(data)+1) - 0.5) / len(data)
    theo_q = dist.ppf(probs, *params)
    samp_q = np.sort(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theo_q, y=samp_q, mode="markers", name="Sample vs Theoretical"))
    # 45-degree line
    lo = min(theo_q.min(), samp_q.min())
    hi = max(theo_q.max(), samp_q.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles", title=name)
    return fig

def return_level_curve(p, c, loc, scale):
    # Given GEV params (SciPy uses c = -xi), compute z_p such that P(X <= z_p) = 1 - p (tail prob = p)
    # Here p is tail probability (e.g., 1/T). Use quantile at 1-p.
    return gev.ppf(1 - p, c=c, loc=loc, scale=scale)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Block Maxima → GEV",
    "Peaks Over Threshold → GPD",
    "Return Levels",
    "Mean Excess",
    "Playground"
])

# -----------------------------------
# Tab 1: Block Maxima (GEV convergence)
# -----------------------------------
with tab1:
    st.subheader("Fisher–Tippett–Gnedenko (Block Maxima ⟶ GEV)")
    st.write(
        "Draw i.i.d. data from a parent distribution, split into blocks of size **n**, take each block’s max, "
        "and fit a GEV. The fitted **shape** ξ indicates the tail class "
        "(Gumbel ξ≈0: light tails; Fréchet ξ>0: heavy tails; Weibull ξ<0: bounded support)."
    )
    # simulate block maxima
    X = sample_parent(parent, size=block_size * n_blocks).reshape(n_blocks, block_size)
    M = X.max(axis=1)

    # Fit GEV to maxima (SciPy's genextreme: c = -ξ)
    c_hat, loc_hat, scale_hat = gev.fit(M)
    xi_hat = -c_hat

    # Histogram + fitted pdf
    xgrid = np.linspace(M.min() - 0.25*np.std(M), M.max() + 0.25*np.std(M), 600)
    pdf_hat = gev.pdf(xgrid, c=c_hat, loc=loc_hat, scale=scale_hat)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=M, histnorm="probability density", nbinsx=40,
                               name="Block maxima", opacity=0.6))
    fig.add_trace(go.Scatter(x=xgrid, y=pdf_hat, mode="lines", name="GEV fit (PDF)"))
    fig.update_layout(xaxis_title="Block maxima", yaxis_title="Density",
                      legend=dict(orientation="h", y=1.05, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # QQ plot under fitted GEV
    qq1 = qq_plot(M, gev, (c_hat, loc_hat, scale_hat), name="GEV QQ plot (block maxima)")
    st.plotly_chart(qq1, use_container_width=True)

    expected = {
        "Normal(0,1)": "Gumbel (ξ≈0)",
        "Exponential(1)": "Gumbel (ξ≈0)",
        "Pareto(α=3, xm=1)": "Fréchet (ξ≈1/α≈0.333)",
        "Uniform(0,1)": "Weibull (ξ≈-1)"
    }[parent]
    c1, c2, c3 = st.columns(3)
    c1.metric("Fitted GEV shape ξ", f"{xi_hat:.3f}")
    c2.metric("Expected domain", expected)
    c3.metric("Blocks × size", f"{n_blocks} × {block_size}")

# -----------------------------------
# Tab 2: POT (GPD convergence)
# -----------------------------------
with tab2:
    st.subheader("Pickands–Balkema–de Haan (Exceedances ⟶ GPD)")
    st.write(
        "Take a high threshold **u** (e.g., 95th percentile). The excesses **Y = X − u | X > u** "
        "are approximately **GPD**. The **shape** again diagnoses tail heaviness."
    )
    X = sample_parent(parent, size=pot_N)
    u = np.quantile(X, pot_quant)
    exceed = X[X > u]
    Y = exceed - u

    if len(Y) < 20:
        st.warning("Too few exceedances. Increase N or lower the threshold.")
    else:
        k_hat, loc_hat_gpd, scale_hat_gpd = gpd.fit(Y, floc=0.0)  # fit to excesses
        xgrid = np.linspace(0, np.percentile(Y, 99.5), 600)
        pdf_hat = gpd.pdf(xgrid, c=k_hat, loc=0.0, scale=scale_hat_gpd)

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=Y, histnorm="probability density", nbinsx=40,
                                    name="Excesses (Y=X−u)", opacity=0.6))
        fig2.add_trace(go.Scatter(x=xgrid, y=pdf_hat, mode="lines", name="GPD fit (PDF)"))
        fig2.update_layout(xaxis_title="Excess over u", yaxis_title="Density",
                           legend=dict(orientation="h", y=1.05, x=0))
        st.plotly_chart(fig2, use_container_width=True)

        # QQ under fitted GPD
        qq2 = qq_plot(Y, gpd, (k_hat, 0.0, scale_hat_gpd), name="GPD QQ plot (excesses)")
        st.plotly_chart(qq2, use_container_width=True)

        # Threshold stability: vary u between 0.85–0.99 quantiles, track k̂ (ξ)
        qs = np.linspace(0.85, 0.99, 20)
        xis = []
        counts = []
        for q in qs:
            uu = np.quantile(X, q)
            ex = X[X > uu] - uu
            if len(ex) >= 30:
                k, _, sc = gpd.fit(ex, floc=0.0)
                xis.append(k)
                counts.append(len(ex))
            else:
                xis.append(np.nan)
                counts.append(len(ex))
        figstab = go.Figure()
        figstab.add_trace(go.Scatter(x=qs, y=xis, mode="lines+markers", name="k̂ vs quantile u"))
        figstab.update_layout(xaxis_title="Threshold quantile u", yaxis_title="Estimated shape k̂ (ξ)",
                              title="Threshold Stability (look for a plateau)")
        st.plotly_chart(figstab, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Threshold (quantile)", f"{pot_quant:.3f}")
        c2.metric("Exceedances", f"{len(Y)}")
        c3.metric("GPD shape k̂ (ξ)", f"{k_hat:.3f}")

# -----------------------------------
# Tab 3: Return Levels (GEV)
# -----------------------------------
with tab3:
    st.subheader("Return-Level Plot (GEV)")
    st.write(
        "Fit GEV on block maxima and plot return level **zₚ** vs the Gumbel scale **−log(−log(1−p))**. "
        "Linear alignment indicates a good GEV fit; curvature hints at misspecification."
    )
    # Reuse block maxima & fit from Tab 1
    X = sample_parent(parent, size=block_size * n_blocks).reshape(n_blocks, block_size)
    M = X.max(axis=1)
    c_hat, loc_hat, scale_hat = gev.fit(M)

    # Tail probs p = 1/T for a grid of return periods
    T = np.geomspace(2, 1000, 50)  # return periods
    p = 1.0 / T
    z = return_level_curve(p, c_hat, loc_hat, scale_hat)
    gumbel_x = -np.log(-np.log(1 - (1 - p)))  # == -log(-log(1 - (1-p))) = -log(-log(p_tail??))
    # Cleaner: use q = 1 - p  (quantile level)
    q = 1 - p
    gumbel_x = -np.log(-np.log(q))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=gumbel_x, y=z, mode="lines+markers", name="Return level curve"))
    fig3.update_layout(xaxis_title="−log(−log(q)),  q = 1 − p (Gumbel scale)",
                       yaxis_title="Return level zₚ",
                       legend=dict(orientation="h", y=1.05, x=0))
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------
# Tab 4: Mean Excess Function
# -----------------------------------
with tab4:
    st.subheader("Mean Excess Function (MEF) Diagnostics")
    st.write(
        "For thresholds **u**, compute **e(u) = E[X−u | X>u]**. "
        "Heavy tails show roughly linear increasing MEF; light tails bend downward; "
        "bounded support curves to zero near the endpoint."
    )
    data = sample_parent(parent, size=max(5000, pot_N))
    us = np.quantile(data, np.linspace(0.6, 0.98, 30))
    e_u = []
    for u_ in us:
        exc = data[data > u_] - u_
        e_u.append(exc.mean() if len(exc) > 0 else np.nan)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=us, y=e_u, mode="lines+markers", name="Mean excess"))
    fig4.update_layout(xaxis_title="Threshold u", yaxis_title="Mean excess e(u)",
                       legend=dict(orientation="h", y=1.05, x=0))
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------
# Tab 5: Playground (direct GEV/GPD)
# -----------------------------------
with tab5:
    st.subheader("Parameter Playground")
    st.write("Sample directly from GEV/GPD with chosen shape ξ (k), location, and scale.")

    n_samp = 20_000
    if pg_family == "GEV":
        # SciPy takes c = -ξ
        c = -pg_xi
        samp = gev.rvs(c=c, loc=pg_loc, scale=pg_scale, size=n_samp)
        xgrid = np.linspace(np.percentile(samp, 0.1), np.percentile(samp, 99.9), 600)
        pdf = gev.pdf(xgrid, c=c, loc=pg_loc, scale=pg_scale)
        title = f"GEV (ξ={pg_xi:.2f}, μ={pg_loc:.2f}, σ={pg_scale:.2f})"
    else:
        # GPD: shape=k=ξ; loc often ~0 for excesses
        k = pg_xi
        samp = gpd.rvs(c=k, loc=pg_loc, scale=pg_scale, size=n_samp)
        xgrid = np.linspace(max(pg_loc, np.percentile(samp, 0.1)),
                            np.percentile(samp, 99.9), 600)
        pdf = gpd.pdf(xgrid, c=k, loc=pg_loc, scale=pg_scale)
        title = f"GPD (k={pg_xi:.2f}, loc={pg_loc:.2f}, scale={pg_scale:.2f})"

    fig5 = go.Figure()
    fig5.add_trace(go.Histogram(x=samp, histnorm="probability density", nbinsx=60,
                                name="Sample", opacity=0.6))
    fig5.add_trace(go.Scatter(x=xgrid, y=pdf, mode="lines", name="Model PDF"))
    fig5.update_layout(title=title, xaxis_title="x", yaxis_title="Density",
                       legend=dict(orientation="h", y=1.05, x=0))
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")
st.markdown(
    "Key properties: **Max-stability (GEV)**, **Threshold stability (GPD)**, "
    "**Return levels**, **Mean excess linearity for heavy tails**. "
    "Play with the parent distribution and parameters and watch the tail index **ξ** reveal itself."
)
