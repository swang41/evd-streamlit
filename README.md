# Extreme Value Theory — Visual Demo (Streamlit)

## What this app is for
A hands-on demo for outsiders to see the **interesting properties of the extreme-value distribution** — no prior stats needed.

## Key properties this demo shows
- **Max per period → GEV.** If you take the maximum of equal-sized groups (e.g., yearly peaks), those maxima follow the **GEV** family.  
  The **tail index (xi)** tells the tail type: **Light (~0)**, **Heavy (>0)**, or **Bounded (<0)**.
- **Top 5% exceedances → GPD.** Values above a high cutoff have **excesses** that follow **GPD** with the **same tail index (xi)**.  
  A good threshold shows **threshold stability** (the estimate steadies out).
- **1-in-T levels.** From the fitted GEV you can read **rare-event levels** (e.g., “1-in-100”).
- **Mean-excess diagnostic.** Quick visual check: heavy tails rise roughly linearly; light tails bend down; bounded tails trend to zero.

---

## Start here (60-second recipe)
1. In the sidebar, set **Parent** to **Normal(0,1)** → expect **Light tail** *(xi ≈ 0)*.  
2. Switch **Parent** to **Pareto(α=3)** → expect **Heavy tail** *(xi ≈ +0.33)*.  
3. Try **Uniform(0,1)** → expect **Bounded tail** *(xi ≈ −1)*.

Then open **Max per Period (GEV)** and **Top 5% Excesses (GPD)** to see the tail index update.  
Check **1-in-T Levels** to read off a “1-in-100” level.

---

## Install & run

### Option A: Conda (recommended for SciPy)
```bash
conda create -n evd-demo -c conda-forge python=3.11 numpy scipy plotly streamlit -y
conda activate evd-demo
streamlit run app.py
```

### Option B: pip/venv
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Tabs (what each answers)

- **Max per Period (GEV)**  
  **Question:** If I take the max per year/batch, what describes those maxima?  
  **See:** Tail index (xi) and tail type (**Heavy / Light / Bounded**).

- **Top 5% Excesses (GPD)**  
  **Question:** If I keep only the very large values above a high cutoff, how do they behave?  
  **See:** Tail index (xi) for exceedances; sensible threshold choice (enough points, not too low).

- **1-in-T Levels**  
  **Question:** What’s the 1-in-100 (or 1-in-1000) level?  
  **See:** Return-level curve; near-linear on the Gumbel scale when the fit is good.

- **Tail Check (MEF)**  
  **Question:** Is the tail heavy, light, or bounded—without fitting a model?  
  **See:** Mean-excess shape (heavy ↑ roughly linear; light ↓ bends; bounded → trends to 0).

> The app runs in **Beginner mode**: simplified visuals, glossary open by default, and plain-English labels.

---

## Glossary (plain English)

- **Extreme value:** unusually large (or small) observation (e.g., record flood).  
- **Block maxima:** split data into equal blocks (years/weeks), take each block’s max.  
- **GEV:** Generalized Extreme Value distribution—models block maxima.  
- **Tail index (xi):** indicates tail type; **> 0** heavy, **≈ 0** light, **< 0** bounded support.  
- **POT / GPD:** Peaks Over Threshold; Generalized Pareto distribution for exceedances above a cutoff.  
- **Return level:** a level exceeded once every **T** periods on average (“1-in-T”).  
- **Mean excess:** average amount by which values exceed a threshold.

---

## Deploy on Streamlit Community Cloud

1. Push this project to a **public GitHub repo**.  
2. Go to **https://share.streamlit.io** → **New app** → select the repo → set **`app.py`** as the entrypoint.  

