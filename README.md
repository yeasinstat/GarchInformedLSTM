---
title: GARCH Informed LSTM (GINN)
emoji: 📉
colorFrom: blue
colorTo: gray
app_file: app.py
pinned: false
---

# GARCH-Informed LSTM (GINN)

A web app that predicts **volatility** (variance) of a time series by combining a classical
**GARCH** model with an **LSTM** neural network — the LSTM's training loss is additionally
informed by the GARCH forecast (weighted by a tunable λ).

## Pipeline

1. `y_t → r_t = (y_t − y_{t−1}) / y_{t−1}` — returns
2. AR(p) on `r_t` → mean forecast `μ̂_t` · GARCH(p,q) on `r_t` → variance forecast `σ²̂_GARCH`
3. Ground-truth variance: `σ²_t = (r_t − μ̂_t)²`
4. LSTM input = past `σ²_t` sequences (**not** `y_t`); LSTM output = `σ²̂_t_LSTM`
5. Loss = `λ·MSE(σ²_t, σ²̂_LSTM) + (1−λ)·MSE(σ²̂_GARCH, σ²̂_LSTM)`
   - **λ = 1.0** → Standard LSTM (pure ground truth)
   - **λ = 0.0** → GINN-0 (pure GARCH-matching)

## Navigation

- **Home** — overview of the tool and workflow
- **Model**
  1. **Data** — upload a CSV/Excel file, pick the time column and the study variable
  2. **Summary Statistics** — mean, median, std, variance, min, max, skewness, kurtosis,
     plus a line plot and a boxplot
  3. **AR & GARCH** — auto-tunes AR(1–5) by AIC, then grid-searches 36 GARCH configurations
     (p,q ∈ {1,2,3} × mean ∈ {Zero,AR} × dist ∈ {normal,t}) by AIC
  4. **LSTM Hyperparameters** — manual entry, or auto-tuning via **Grid Search** or
     **Bayesian Search** (Gaussian Process, scikit-optimize)
  5. **GINN** — trains the Standard LSTM (λ=1.0) plus one model per λ value you list,
     reports RMSE / MAE / R² for each, plots predictions vs. actuals, and produces a
     downloadable Excel workbook
- **Instructions** — usage guide (expand as needed)
- **Developers** — credits / contact


## Notes

- Data must contain a numeric study-variable column with a meaningful notion of returns
  (price, yield, index level, etc.) — not already a returns/percentage-change series.
- Needs a reasonable amount of history: AR/GARCH fitting and sequence modeling both need
  enough points to leave a workable train/test split (roughly 30+ observations minimum;
  more is better, especially for the GARCH variance estimation, which is inherently noisy
  on small samples).
- The GARCH auto-tune (36 configurations) typically runs in a few seconds on modest data
  sizes, with a live progress bar.
- If GARCH selects `mean='AR'`, the first `AR_LAG` variance values have no prior history
  to estimate from; the app fills these with the first valid value (matching how the AR
  mean forecast's own burn-in period is already handled), avoiding NaNs propagating into
  training.