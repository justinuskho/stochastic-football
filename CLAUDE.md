# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Stochastic Football** is a Premier League match simulator and prediction game. It models team performance as normal distributions (dynamic Elo + sigma volatility) rather than static ratings, deployed as a Streamlit web app backed by Google BigQuery.

## Commands

```bash
# Run the Streamlit app locally
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Deploy the fixture sync Cloud Function
cd cloud_jobs/fixture_sync_job
gcloud functions deploy fixture-sync --runtime python311 --trigger-http

# Run a notebook
jupyter notebook workspace/
```

## Architecture

### Core Files

- **[app.py](app.py)** — Thin entry point (~90 lines). Handles init, CSS loading, mode toggle, context construction, view dispatch, and params teardown.
- **[engine.py](engine.py)** — Pure math: probability calculations, rating updates, log-loss scoring. No I/O.
- **[database.py](database.py)** — All BigQuery access. Queries are cached for 360s to manage cost.
- **[style.css](style.css)** — All visual styling. Three mode-scoped sections: `prediction-mode`, `free-play-mode`, `dashboard-mode`, each using `html:has(.<mode>)` CSS scoping.
- **[cloud_jobs/fixture_sync_job/main.py](cloud_jobs/fixture_sync_job/main.py)** — GCP Cloud Function that fetches fixtures weekly from the Fantasy Premier League API and upserts into BigQuery.

### Views Package (`views/`)

Each view is a self-contained module. Per-view agent docs live in `.claude/agents/`:

| File | Function | Agent doc |
|---|---|---|
| [engine.py](engine.py) | `get_dynamic_drift`, `run_simulation`, `params_season_refresh` | [data-modeler.md](.claude/agents/data-modeler.md) |
| [views/context.py](views/context.py) | `AppContext`, `PredictionContext` dataclasses | — |
| [views/charts.py](views/charts.py) | `plot_matchup`, `plot_performance_moving_avg` | — |
| [views/free_play.py](views/free_play.py) | `render_free_play(ctx)` | [free_play.md](.claude/agents/free_play.md) |
| [views/prediction.py](views/prediction.py) | `build_prediction_context(ctx)`, `render_prediction(ctx, pctx)` | [prediction.md](.claude/agents/prediction.md) |
| [views/dashboard.py](views/dashboard.py) | `render_dashboard(ctx, pctx)` | [dashboard.md](.claude/agents/dashboard.md) |

### App Dispatch Flow

```
app.py
  ├── db.fetch_fixtures / db.fetch_params
  ├── run_sim_wrapper loop → fixtures['p_win','p_draw','p_loss']
  ├── db.fetch_predictions
  ├── st.radio → app_mode in ["Dashboard", "Prediction", "Free Play"]
  ├── AppContext(...)
  ├── if Free Play → render_free_play(ctx)
  │   else:
  │     pctx = build_prediction_context(ctx)
  │     if Dashboard → render_dashboard(ctx, pctx)
  │     else → render_prediction(ctx, pctx)
  └── teardown: push params_as_of snapshots to BigQuery
```

### Data Flow

1. **Ingestion**: Cloud Function → FPL API → BigQuery `public.fixtures`
2. **Params**: Team ratings live in BigQuery `public.params`, keyed by `(model, as_of)`. Each row is a full snapshot of all team Elo/sigma/form values plus hyperparameters.
3. **App render**: `app.py` calls `database.py` to fetch fixtures and latest params, then `engine.py` to compute probabilities. Nothing is stored at render time except pushed predictions.

### Shared State: Context Dataclasses

```python
# views/context.py
AppContext:           # passed to all views
    fixtures          # played matches with p_win/p_draw/p_loss computed
    fixtures_next     # upcoming matches (empty = season over)
    params_now        # latest team ratings snapshot
    params_as_of      # dict of date→params, accumulated during render
    predictions       # past predictions (played fixtures)
    predictions_next  # predictions for upcoming fixtures
    TEAMS / TEAMS_LOOKUP
    null_guess_probability  # 0.25 — penalty baseline
    market_suppliers  # {'google', 'opta_analyst', 'Google', 'OptaAnalyst'}

PredictionContext:    # computed by build_prediction_context(ctx), passed to Prediction + Dashboard
    agg_losses        # DataFrame[date, user, total_loss, n_preds]
    leaderboard       # DataFrame[user, score] sorted ascending
    penalty           # float — null-guess log-loss
    all_benchmarks    # market_suppliers | {'engine'}
    human_users       # list of users not in all_benchmarks
    prediction_losses # raw list[fixture_id, date, user, loss]
```

### Mathematical Model

**Team performance distributions:**
- Home: `N(Elo_h + HFA + Form_h, σ_h)`
- Away: `N(Elo_a + Form_a, σ_a)`

**Win probabilities** use logistic CDF with a draw margin that scales with combined sigma. Probabilities are clamped to [0.05, 0.85] to prevent the model from being overconfident.

**Rating updates after each match:**
- `surprise = actual_points - expected_points`
- `form_new = form_old * decay + surprise * k2`
- `sigma_new = max(80, sigma * convergence + (|surprise| - τ) * k3)`
- `elo_shift = surprise * k1 * (sigma / 150)` — sigma-adaptive learning rate
- Elo is zero-sum: home gain = −away gain

**Season refresh** (run once at season start): sigma increases by `refresh` value (~332), form resets to 0.

### BigQuery Schema

Key tables (all in `public` dataset):
- `fixtures` — match schedule, scores, team IDs
- `params` — model parameter snapshots per `(model, as_of)` date
- `predictions` — user predictions with fixture_id, probabilities, and log-loss
- `results` — resolved match outcomes

### Authentication

`database.py:get_bq_client()` tries three auth paths in order:
1. `.streamlit/secrets.toml` (local dev / Streamlit Cloud)
2. `BQ_JSON_KEY` env var (GitHub Actions / Cloud jobs)
3. `gcloud` application default credentials

> **Note:** `database.py` line 35 calls `client = get_bq_client()` at module scope, which fires `@st.cache_resource` before `set_page_config` runs. This causes `StreamlitSetPageConfigMustBeFirstCommandError` if imports are reordered. Workaround: keep `import streamlit as st` + `st.set_page_config(...)` as the very first two lines of `app.py`, before any database import.

### Caching

`query2dict()` and `query2df()` in `database.py` use `@st.cache_data(ttl=360)`. To force-refresh during development, use `st.cache_data.clear()` or restart the app.

### Notebooks

`workspace/` notebooks are for research and analysis only — they're gitignored. The model was tuned using Optuna on 13 years of PL data (on Kaggle). `epl_params.json` contains trial parameter sets.

## Key Conventions

- All probability math lives in `engine.py`; no math inline in views.
- `push_params()` stages params to BigQuery for review — it does not activate them. Params take effect when `as_of` date is the latest before the query date in `fetch_params()`.
- Plotly charts follow the project's standard style — invoke the `plot-style` skill when creating or modifying any figure.
- The fixture window fetched by `fetch_fixtures()` is 5 previous + 2 future games relative to the last played match.
- Each view wraps its content in a mode-scoped div (`prediction-mode`, `free-play-mode`, `dashboard-mode`). The CSS in `style.css` uses `html:has(.<mode>)` to scope all styles to the active view. The div markers themselves are hidden via `div[data-testid='stElementContainer']:has(div[class='<mode>'])`.
- `build_prediction_context` is only called when mode is Prediction or Dashboard — not Free Play — to avoid unnecessary computation.
- Navigation between modes uses `st.session_state.navigate_to` (not `app_mode_sync` directly) to avoid the `StreamlitAPIException: widget key modified after instantiation` error. `app.py` transfers `navigate_to` → `app_mode_sync` before the radio widget is created.
