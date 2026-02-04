# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Import custom logic for simulation and data access
from engine import get_dynamic_drift, get_past_5_games, run_simulation
import database as db
import json

# ==========================================
# 1. INITIALIZATION & DATA PRE-PROCESSING
# ==========================================

st.set_page_config(page_title="Stochastic Match Engine", layout="wide")

# Fetch initial fixtures and model parameters from BigQuery
fixtures = db.fetch_fixtures()
params = db.fetch_params(trial="p5")[0]

# Pre-calculate Expected Points (xP) for all historical fixtures
# This allows us to compare actual performance vs. model expectations in the history tables
fixtures.loc[:, ['home_xP', 'away_xP']] = fixtures.apply(
    lambda row: run_simulation(
        params, 
        (row['home'], row['away']), 
        (row['home_point'], row['away_point']), 
        new_season=False
    )[2][:2],
    axis=1,
    result_type='expand'
).values

# Dynamic UI Scaling: Calculate slider ranges based on current data distribution
elos = [int(v) for k, v in params.items() if 'elo' in k]
min_elo, max_elo = round(min(elos), -2), round(max(elos), -2)

sigmas = [int(v) for k, v in params.items() if 'sigma' in k]
min_sigma, max_sigma = round(min(sigmas), -2), round(max(sigmas), -2)

forms = [int(v) for k, v in params.items() if 'form' in k]
min_max_form = round(max(abs(min(forms)), abs(max(forms))), -2)

hfas = [int(v) for k, v in params.items() if 'hfa' in k]
max_hfa = round(max(hfas), -2)

TEAMS = fixtures["home"].sort_values().unique()

# ==========================================
# 2. UI STYLING (CUSTOM CSS)
# ==========================================

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    
    /* Custom styling for Streamlit Metric components */
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    
    .stMetric { 
        background-color: #161b22; 
        border: 1px solid #30363d; 
        border-radius: 10px; 
        padding: 15px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 3. SIDEBAR: CONTROL PANEL
# ==========================================

with st.sidebar:
    st.title("🛡️ Match Setup")
    
    # --- Home Team Selection ---
    home_team = st.selectbox("Select Home Team", TEAMS, index=0)
    home_fixtures, home_form = get_past_5_games(home_team, fixtures)
    st.write(f"**Last 5 Matches:** {home_form}")
    
    with st.expander(f"Adjust Parameters", expanded=True):
        h_elo = st.slider("Base Elo ", min_elo, max_elo, value=int(params[f"elo_{home_team}"]), key="h_elo")
        h_sigma = st.slider("Uncertainty (σ) ", min_sigma, max_sigma, value=int(params[f"sigma_{home_team}"]), key="h_sigma")
        h_form = st.slider("Momentum (Form) ", -min_max_form, min_max_form, value=int(params[f"form_{home_team}"]), key="h_form")
        h_hfa = st.slider("Home Advantage ", 0, max_hfa, value=int(params[f"hfa_{home_team}"]))
    
    st.markdown("---")
    
    # --- Away Team Selection ---
    away_team = st.selectbox("Select Away Team", TEAMS, index=1)
    away_fixtures, away_form = get_past_5_games(away_team, fixtures)
    st.write(f"**Last 5 Matches:** {away_form}")
    
    with st.expander(f"Adjust Parameters", expanded=True):
        a_elo = st.slider("Base Elo ", min_elo, max_elo, value=int(params[f"elo_{away_team}"]), key="a_elo")
        a_sigma = st.slider("Uncertainty (σ) ", min_sigma, max_sigma, value=int(params[f"sigma_{away_team}"]), key="a_sigma")
        a_form = st.slider("Momentum (Form) ", -min_max_form, min_max_form, value=int(params[f"form_{away_team}"]), key="a_form")
    

# ==========================================
# 4. MAIN DASHBOARD: SIMULATION RESULTS
# ==========================================

# Run simulation engine for the currently selected/adjusted team parameters
mu_h, mu_a, p_win, p_draw, p_loss = get_dynamic_drift(
    h_elo, a_elo, h_sigma, a_sigma, h_hfa, h_form, a_form
)
    
# Display Win/Draw/Loss Probabilities
with st.container():
    st.subheader("Match Outcome Probabilities")
    p1, p2, p3 = st.columns(3)
    p1.metric(f"{home_team} Win", f"{p_win:.1%}")
    p2.metric("Draw", f"{p_draw:.1%}")
    p3.metric(f"{away_team} Win", f"{p_loss:.1%}")
    
st.markdown("---")

# Visual Distribution Plot: Performance Overlap
st.subheader(f"Head-to-Head ELO Distribution: {home_team} (H) vs {away_team} (A)")

def plot_matchup(h_name, a_name, h_elo, a_elo, h_sigma, a_sigma, h_form, a_form, h_hfa):
    """
    Generates a Plotly chart showing the overlapping Normal Distributions 
    of the two teams' potential matchday performance.
    """
    h_mu = h_elo + h_hfa + h_form
    a_mu = a_elo + a_form
    
    limit_min = min(h_mu, a_mu) - 3 * max(h_sigma, a_sigma)
    limit_max = max(h_mu, a_mu) + 3 * max(h_sigma, a_sigma)
    x = np.linspace(limit_min, limit_max, 1000)

    fig = go.Figure()

    # Home Distribution
    fig.add_trace(go.Scatter(
        x=x, 
        y=stats.norm.pdf(x, h_mu, h_sigma), 
        name=f"{h_name} (Home)", 
        fill='tozeroy', 
        fillcolor='rgba(218, 2, 14, 0.5)', 
        line=dict(color='red')
    ))

    # Away Distribution
    fig.add_trace(go.Scatter(
        x=x, 
        y=stats.norm.pdf(x, a_mu, a_sigma), 
        name=f"{a_name} (Away)", 
        fill='tozeroy', 
        fillcolor='rgba(108, 171, 221, 0.5)', 
        line=dict(color='rgba(108, 171, 221, 1)')
    ))

    fig.update_layout(
        # title=f"",
        xaxis_title="Matchday Performance (ELO)", 
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

st.plotly_chart(plot_matchup(home_team, away_team, h_elo, a_elo, h_sigma, a_sigma, h_form, a_form, h_hfa), use_container_width=True)

st.markdown("---")

# ==========================================
# 5. DATA TABLES: HISTORICAL PERFORMANCE
# ==========================================

with st.container():
    st.subheader("Team's Past 5 Games:")
    c1, c2 = st.columns(2)

    # Home History Table
    with c1:
        st.caption(f"{home_team}'s Past 5 Games:")
        home_fixtures['Pts'] = np.select(
            [home_fixtures['home']==home_team],
            [home_fixtures['home_point']], home_fixtures['away_point']
        )
        home_fixtures['xPts'] = np.select(
            [home_fixtures['home']==home_team],
            [home_fixtures['home_xP']], home_fixtures['away_xP']
        ).round(1)
        home_fixtures['Pts (xPts)'] = home_fixtures.apply(
            lambda row: f"{row['Pts']} ({row['xPts']})",
            axis=1
        )
        st.dataframe(home_fixtures[["Fixture", "Score", "Result", "Pts (xPts)"]], use_container_width=True, hide_index=True)

    # Away History Table
    with c2:
        st.caption(f"{away_team}'s Past 5 Games:")
        away_fixtures['Pts'] = np.select(
            [away_fixtures['away']==away_team],
            [away_fixtures['away_point']], away_fixtures['home_point']
        )
        away_fixtures['xPts'] = np.select(
            [away_fixtures['away']==away_team],
            [away_fixtures['away_xP']], away_fixtures['home_xP']
        ).round(1)
        away_fixtures['Pts (xPts)'] = away_fixtures.apply(
            lambda row: f"{row['Pts']} ({row['xPts']})",
            axis=1
        )
        st.dataframe(away_fixtures[["Fixture", "Score", "Result", "Pts (xPts)"]], use_container_width=True, hide_index=True)

st.caption("Note: Elo values in the comparison chart are scaled (Elo/10) for visibility. Form is normalized to a 100-base.")