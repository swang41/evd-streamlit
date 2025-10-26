# stylekit.py
import streamlit as st

CSS = """
<style>
h1, h2, h3 { letter-spacing:.2px; }
h1 { font-weight:800; } h2 { margin-top:.6rem; }
.block-container { padding-top:1.2rem; }
section.main > div { gap:.75rem; }
.stTabs [data-baseweb="tab-list"] { gap:.25rem; }
.stTabs [data-baseweb="tab"]{
  padding:.5rem 1rem; border-radius:10px 10px 0 0; background:#F6F8FB;
}
.stTabs [aria-selected="true"]{ background:#fff; }
section[data-testid="stSidebar"] .stCaption{ opacity:.85; }
.badge{
  display:inline-flex; align-items:center; gap:.4rem;
  padding:.2rem .6rem; border-radius:999px; font-weight:600; font-size:.9rem;
}
.badge--heavy   { background:#FDE68A; color:#7C2D12; }
.badge--light   { background:#E0F2FE; color:#075985; }
.badge--bounded { background:#E2E8F0; color:#334155; }
</style>
"""

def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)

def apply_plot_style(fig):
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))
    return fig
