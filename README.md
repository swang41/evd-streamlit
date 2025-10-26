# Extreme Value Theory — Visual Demo (Streamlit)

Interactive app to illustrate:
- **Block Maxima ⟶ GEV** (Fisher–Tippett–Gnedenko)
- **Peaks Over Threshold ⟶ GPD** (Pickands–Balkema–de Haan)
- **Return-Level** plots
- **Mean Excess** diagnostics
- Shape parameter **ξ** as the unifying tail index

## Run locally
conda create -n evd-demo ptyhon=3.11
conda activate evd-demo
pip install -r requirements.txt
streamlit run app.py

## Deploy on Streamlit Community Cloud
1. Create a **public** GitHub repo (e.g., `extreme-value-evd-demo`).
2. Add these three files and push.
3. Go to https://share.streamlit.io/ → **New app**, point to your repo, entrypoint `app.py`.
