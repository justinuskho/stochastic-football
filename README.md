# Stochastic Football

Premier League match simulator and prediction game built with **Python**, **Streamlit**, and **Google BigQuery**. Models team performance as probability distributions rather than static ratings — dynamic Bayesian-inspired Elo with uncertainty (σ) and momentum (form).

---

## Modes

**Dashboard** — Season recap. Leaderboard, performance chart, human vs benchmark comparison, best/worst prediction calls. EPL brand palette (deep purple + neon green).

**Prediction** — Active-season game. Submit win/draw/loss probability predictions for upcoming fixtures, scored by log-loss. Charcoal dark theme.

**Free Play** — Sandbox. Adjust any team's Elo, sigma, form, and HFA via sliders and see probability distributions shift in real time. Light theme.

---

## How the Model Works

A match is modelled as the difference between two random variables:

- **Home Performance** ~ `N(Elo_h + HFA + Form_h, σ_h)`
- **Away Performance** ~ `N(Elo_a + Form_a, σ_a)`

Win/draw/loss probabilities come from a logistic CDF with a sigma-scaled draw margin, clamped to `[0.05, 0.85]` to prevent overconfidence.

**Rating updates after each match:**
```
surprise   = actual_points − expected_points
elo_shift  = surprise × k1 × (σ / 150)        # sigma-adaptive learning rate; zero-sum
form_new   = form_old × decay + surprise × k2
sigma_new  = max(80, σ × convergence + (|surprise| − τ) × k3)
```

Model was tuned with Optuna on 13 years of PL historical data. See the [Kaggle notebook](https://www.kaggle.com/code/justinus/stochastic-football) for full methodology.

---

## Architecture

```
app.py                  # thin entry point: init, dispatch, teardown
engine.py               # pure math — no I/O
database.py             # all BigQuery access, cached 360s
style.css               # all styling, scoped per mode
views/
  context.py            # AppContext, PredictionContext dataclasses
  charts.py             # plot_matchup, plot_performance_moving_avg
  dashboard.py          # render_dashboard(ctx, pctx)
  prediction.py         # build_prediction_context(ctx), render_prediction(ctx, pctx)
  free_play.py          # render_free_play(ctx)
cloud_jobs/
  fixture_sync_job/     # GCP Cloud Function — weekly FPL API → BigQuery sync
```

---

## Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Data Warehouse:** [Google BigQuery](https://cloud.google.com/bigquery)
- **Math/Stats:** NumPy, SciPy
- **Charts:** Plotly

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires BigQuery credentials — see `database.py:get_bq_client()` for auth paths (Streamlit secrets → env var → gcloud ADC).

---

## Research & Writeups

- [Kaggle notebook — modelling & backtesting](https://www.kaggle.com/code/justinus/stochastic-football)
- [Technical narrative — Kaggle writeup](https://www.kaggle.com/writeups/justinus/stochastic-football)
- [LinkedIn article](https://www.linkedin.com/pulse/stochastic-football-justinus-kho-a3hpf/)
- [Medium post](https://medium.com/@justinus.jx/stochastic-football-c2a44c6d856d)
