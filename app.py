# streamlit_app.py
import streamlit as st
st.set_page_config(page_title="Stochastic Match Engine", layout="wide")

import pandas as pd
from copy import deepcopy
from engine import run_simulation
import database as db
from views.context import AppContext
from views.free_play import render_free_play
from views.prediction import render_prediction, build_prediction_context
from views.dashboard import render_dashboard

# ==========================================
# 1. INITIALIZATION & DATA PRE-PROCESSING
# ==========================================

fixtures = db.fetch_fixtures(-5, 2)
params = db.fetch_params(before=fixtures['date'].min().strftime('%Y-%m-%d'))[0]

fixtures_next = fixtures[fixtures['home_score'].isnull()].copy()
fixtures_next['display_name'] = "GW" + fixtures_next['round'].astype(str) + ": " + fixtures_next['Fixture']
fixtures = fixtures[fixtures['home_score'].notnull()].copy()

params_as_of = {params['as_of']: deepcopy(params)}

def run_sim_wrapper(date, params0, fixture, score=None, point=None):
    result = run_simulation(params=params0, fixture=fixture, score=score, point=point)
    params1 = deepcopy(params0)
    if score is not None or point is not None:
        params1['as_of'] = date
        params_as_of[date] = deepcopy(params1)
    return result

fixtures.loc[:, ['home_xP', 'away_xP', 'p_win', 'p_draw', 'p_loss']] = fixtures.apply(
    lambda row: run_sim_wrapper(
        date=row['date'],
        params0=params,
        fixture=(row['home'], row['away']),
        point=(row['home_point'], row['away_point'])
    )[2],
    axis=1,
    result_type='expand'
).values

params_now = deepcopy(params)

if not fixtures_next.empty:
    fixtures_next.loc[:, ['home_xP', 'away_xP', 'p_win', 'p_draw', 'p_loss']] = fixtures_next.apply(
        lambda row: run_simulation(
            params,
            (row['home'], row['away']),
            (row['home_point'], row['away_point'])
        )[2],
        axis=1,
        result_type='expand'
    ).values

next_ids = fixtures_next['id'].unique().tolist() if not fixtures_next.empty else []
predictions = db.fetch_predictions(fixture_id_list=fixtures['id'].unique().tolist() + next_ids)
predictions_next = predictions[predictions['fixture_id'].isin(fixtures_next['id'])]
predictions = predictions[predictions['fixture_id'].isin(fixtures['id'])]

null_guess_probability = 0.25
market_suppliers = {'google', 'opta_analyst', 'Google', 'OptaAnalyst'}

TEAMS_LOOKUP = pd.concat([
    fixtures[["home", "home_team"]].rename(columns={"home": "team_code", "home_team": "team_name"}),
    fixtures[["away", "away_team"]].rename(columns={"away": "team_code", "away_team": "team_name"})
]).drop_duplicates().set_index("team_name")["team_code"].to_dict()
TEAMS = sorted(list(TEAMS_LOOKUP.keys()))

# ==========================================
# 2. UI STYLING
# ==========================================

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# ==========================================
# 3. MODE TOGGLE
# ==========================================

MODES = ["Dashboard", "Prediction", "Free Play"]
if "app_mode_sync" not in st.session_state:
    st.session_state.app_mode_sync = MODES[0]
if "navigate_to" in st.session_state:
    st.session_state.app_mode_sync = st.session_state.pop("navigate_to")

app_mode = st.radio("", options=MODES, horizontal=True, key="app_mode_sync")

# ==========================================
# 4. BUILD CONTEXT & DISPATCH
# ==========================================

ctx = AppContext(
    fixtures=fixtures,
    fixtures_next=fixtures_next,
    params_now=params_now,
    params_as_of=params_as_of,
    predictions=predictions,
    predictions_next=predictions_next,
    TEAMS=TEAMS,
    TEAMS_LOOKUP=TEAMS_LOOKUP,
    null_guess_probability=null_guess_probability,
    market_suppliers=market_suppliers,
)

if app_mode == "Free Play":
    render_free_play(ctx)
else:
    pctx = build_prediction_context(ctx)
    if app_mode == "Dashboard":
        render_dashboard(ctx, pctx)
    else:
        render_prediction(ctx, pctx)

# ==========================================
# 5. TEARDOWN — push param snapshots to BQ
# ==========================================

params_to_push = pd.concat(
    [pd.DataFrame(index=p.keys(), data=p.values()) for _, p in params_as_of.items()],
    axis=1
).transpose().reset_index(drop=True)
params_to_push['data_load_date'] = pd.Timestamp.utcnow().date().isoformat()
params_to_push['updated_at'] = pd.Timestamp.now().isoformat()
params_to_push['source'] = 'app'
db.push_params(df=params_to_push)
