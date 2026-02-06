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
fixtures = db.fetch_fixtures(-5, 1)
params = db.fetch_params(trial="p5")[0]

# Pre-calculate Expected Points (xP) for all historical fixtures
# This allows us to compare actual performance vs. model expectations in the history tables
fixtures_next = fixtures[fixtures['home_score'].isnull()]
fixtures_next['display_name'] = "R" + fixtures_next['round'].astype(str) + ": " + fixtures_next['home'] + " vs " + fixtures_next['away']
fixtures = fixtures[fixtures['home_score'].notnull()] 
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

TEAMS = np.sort(np.unique(np.concatenate([list(fixtures["home"].sort_values().unique()), list(fixtures["home"].sort_values().unique())])))

# ==========================================
# 2. UI STYLING (CUSTOM CSS)
# ==========================================
st.markdown("""
    <style>

    /* 2. COMPRESSION: Remove massive top and bottom padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
        height: 100vh;
    }
    
    /* 3. COMPRESSION: Tighten the gap between every Streamlit widget */
    [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }

    /* 4. Metric Component Styling */
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 800;
    }
    
    .stMetric { 
        border: 1px solid #30363d; 
        border-radius: 8px; 
        padding: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }

    /* 5. UI CLEANUP: Hide the header bar and footer to gain ~10% screen space */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* 6. Sidebar adjustments */
    section[data-testid="stSidebar"] {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def apply_custom_theme(mode):
    if mode == "Free Play Mode":
        # LIGHT MODE: Left Panel Grey, Main White
        st.markdown("""
            <style>
            /* Show Horizontal Blocks (The Rows/st.columns) in Red */
            # [data-testid="stHorizontalBlock"] {
            #     border: 2px dashed red !important;
            # }
            # /* Show Vertical Blocks (The content stacks) in Blue */
            # [data-testid="stVerticalBlock"] {
            #     border: 2px solid blue !important;
            # }
            # /* Show Columns in Green */
            # [data-testid="column"] {
            #     border: 2px solid green !important;
            # }
            # [data-testid="stHorizontalBlock"]:nth-of-type(2) {
            #     border: 4px solid yellow !important;
            # }
            
            # h1, h2, h3, p, span { color: #1E1E1E !important; }

            /* 2. The SECOND Column (Purple) */
            /* This SHOULD be your Right Panel in Free Play mode */
            [data-testid="column"]:nth-child(2) {
                border: 4px solid purple !important;
            }

            /* 3. The SECOND Vertical Block (Orange) */
            /* Use this to see how Streamlit stacks things inside containers */
            [data-testid="stVerticalBlock"]:nth-child(2) {
                border: 10px solid orange !important;
            }
            
            /* Main Background */
            .stApp {
                background-color: #FFFFFF !important;
                color: black;
            }
            /* Right Panel Styling */
            [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2){
                background-color: black !important;
                padding: 2rem !important;
                min-height: 100vh !important;
                border-left: 1px solid #DDE1E6 !important;
                /* This ensures the color stretches to the edges */
                margin-top: -2rem !important; 
                margin-bottom: -2rem !important;
            }
            [data-testid="stHorizontalBlock"]:nth-child(2) [data-testid="column"]:nth-of-type(1) [data-testid="stHorizontalBlock"] {
                padding: 0px !important;
            }
            /* Metric and Chart Text Colors */
            .stMetric { 
                background-color: #161b22; 
                border: 1px solid #30363d; 
                border-radius: 8px; 
                padding: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            }
            [data-testid="stMetricLabel"] {
                color: #FFFFFF;
                font-size: 0.9rem;
                font-weight: 600;
            }
            [data-testid="stMetricValue"] {
                color: #FFFFFF;
                font-size: 1.6rem;
                font-weight: 800;
            }

            div[data-testid="stExpander"] p { color: #1E1E1E !important; }
            </style>
            """, unsafe_allow_html=True)
    else:
        # DARK MODE: Right Panel Dark Grey, Main Dark
        st.markdown("""
            <style>
            /* Text Colors */
            h1, h2, h3, p, span {
                color: white;
            }
            /* Main Background (Dark) */
            .stApp {
                background-color: #0E1117;
                color: #fafafa;
            }
            [data-testid="column"]:nth-child(1) {
                background-color: #F0F2F6 !important;
                padding: 2rem !important;
                min-height: 100vh !important;
                border-left: 1px solid #DDE1E6 !important;
                /* This ensures the color stretches to the edges */
                margin-top: -2rem !important; 
                margin-bottom: -2rem !important;
            }
            [data-testid="column"]:nth-of-type(2) [data-testid="stHorizontalBlock"] {
                padding: 0px !important;
            }
            
            /* Right Side Column Simulation (Using a container or specific div if possible) */
            /* Note: We target the specific widgets inside the right column */
            
            /* Dataframe & Table Background matching dark theme */
            [data-testid="stTable"], [data-testid="stDataFrame"] {
                background-color: #161B22;
                border: 1px solid #30363D;
            }
            
            /* Lock in Prediction Button: Background Neon/White, Text Black */
            div.stButton > button {
                background-color: #00FFCC; /* Neon Teal */
                border: None;
            }
            div.stButton > button p {
                color: black;
                font-weight: bold;
            }
            
            /* Username Input Box: Slightly Transparent White */
            div[data-testid="stTextInput"] input {
                background-color: rgba(255, 255, 255, 0.1) !important;
                color: black;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .stMetric { 
                background-color: #161b22; 
                border: 1px solid #30363d; 
                border-radius: 8px; 
                padding: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            }
            
            /* Metric and Chart Text Colors */
            [data-testid="stMetricValue"] {
                color: #00FFCC;
            }
            </style>
            """, unsafe_allow_html=True)
        
def on_mode_change():
    """
    Triggered when the user toggles between Prediction and Free Play.
    Ensures the team selections are synchronized.
    """
    # 1. Check the new value of the radio button
    new_mode = st.session_state.app_mode_sync
    
    if new_mode == "Free Play Mode":
        # Reset to defaults for sandbox testing
        st.session_state.home_sync = TEAMS[0]
        st.session_state.away_sync = TEAMS[1]
    else:
        # If switching back to Prediction, sync to the currently 
        # selected fixture in the selectbox
        st.session_state.match_selector = fixtures_next['display_name'].values[0]
        sync_fixture()
        
    apply_custom_theme(new_mode)
        
def sync_fixture():
    """
    Triggered when the user picks a specific match from the 
    official fixtures list in Prediction Mode.
    """
    # Get the string from the selectbox (e.g., "ARS vs SUN")
    selected_name = st.session_state.match_selector
    
    # Filter your dataframe to find the match row
    match_row = fixtures_next[fixtures_next['display_name'] == selected_name].iloc[0]
    
    # Update the global sync keys
    st.session_state.home_sync = match_row['home']
    st.session_state.away_sync = match_row['away']

# ==========================================
# 3. SIDEBAR: CONTROL PANEL
# ==========================================

# --- 1. INITIALIZATION & CALLBACKS ---
if 'home_sync' not in st.session_state:
    st.session_state.home_sync = TEAMS[0]
if 'away_sync' not in st.session_state:
    st.session_state.away_sync = TEAMS[1]

def sync_fixture():
    selected = st.session_state.match_selector
    match_row = fixtures_next[fixtures_next['display_name'] == selected].iloc[0]
    st.session_state.home_sync = match_row['home']
    st.session_state.away_sync = match_row['away']
    st.session_state.fixture_id_sync = match_row['id']

# --- 2. GLOBAL MODE TOGGLE (Top of App) ---
MODES = ["Prediction Mode", "Free Play Mode"]
if "app_mode_sync" not in st.session_state:
    st.session_state.app_mode_sync = MODES[0]

# --- 3. LAYOUT DEFINITION ---
# We define the columns. If in Prediction Mode, we use a wide middle and a right column.
if st.session_state.app_mode_sync == "Prediction Mode":
    # Layout: Main Center (75%) and Right Sidebar (25%)
    left_col, main_col = st.columns([1, 4], gap="small")
elif st.session_state.app_mode_sync == "Free Play Mode":
    main_col, right_col = st.columns([4, 1], gap="small")
    # right_col = None
    
with main_col:
    app_mode = st.session_state.app_mode_sync
    app_mode = st.radio("Select Mode:", options=MODES, horizontal=True, key="app_mode_sync", on_change=on_mode_change)
    apply_custom_theme(app_mode)
    
home_team = st.session_state.home_sync
away_team = st.session_state.away_sync

# --- 4. LEFT SIDEBAR (Free Play only) ---

if app_mode == "Free Play Mode":
    # Call this immediately after your app_mode selection
    with right_col:
        st.subheader("Free Play Mode")
        home_team = st.selectbox("Home Team", TEAMS, key="home_sync")

        with st.expander("Home Team Adjustments", expanded=True):
            h_elo = st.slider("Elo (Strength)", min_elo, max_elo, value=int(params[f"elo_{home_team}"]), key="fs_h_elo")
            h_sigma = st.slider("Sigma (Uncertainty)", min_sigma, max_sigma, value=int(params[f"sigma_{home_team}"]), key="fs_h_sigma")
            h_form = st.slider("Momentum (Form) ", -min_max_form, min_max_form, value=int(params[f"form_{home_team}"]), key="fs_h_form")
            h_hfa = st.slider("Home Field Advantage", 0, max_hfa, value=int(params[f"hfa_{home_team}"]), key="fs_h_hfa")
        
        away_team = st.selectbox("Away Team", TEAMS, key="away_sync")
        
        with st.expander("Away Team Adjustments", expanded=True):
            a_elo = st.slider("Elo (Strength)", min_elo, max_elo, value=int(params[f"elo_{away_team}"]), key="fs_a_elo")
            a_sigma = st.slider("Sigma (Uncertainty)", min_sigma, max_sigma, value=int(params[f"sigma_{away_team}"]), key="fs_a_sigma")
            a_form = st.slider("Momentum (Form) ", -min_max_form, min_max_form, value=int(params[f"form_{away_team}"]), key="fs_a_form")

# --- 5. RIGHT SIDEBAR (The "Fake" Column) ---
if app_mode == "Prediction Mode":
    if "fixture_id_sync" not in st.session_state:
        st.session_state.fixture_id_sync = fixtures_next['id'].values[0]
    fixture_id_sync = st.session_state.fixture_id_sync
        
    with left_col:
        st.subheader("Prediction Panel")
        
        user_id = st.text_input("Username")
        lock_button = st.button("🔒 Lock In", use_container_width=True)
        message_slot = st.empty()
        
        # Fixture Selection
        fx = st.selectbox("Select Official Fixture", options=fixtures_next['display_name'].tolist(), key="match_selector", on_change=sync_fixture)
        home_team = fx.split(" ")[1]
        away_team = fx.split(" ")[3]
        
        with st.expander(f"{home_team} Adjustments", expanded=True):
            h_elo = st.slider(f"Elo (Strength)", min_elo, max_elo, value=int(params[f"elo_{home_team}"]), key="pm_h_elo")
            h_sigma = st.slider(f"Sigma (Uncertainty)", min_sigma, max_sigma, value=int(params[f"sigma_{home_team}"]), key="pm_h_sigma")
            h_form = st.slider(f"Form (Momentum)", -min_max_form, min_max_form, value=int(params[f"form_{home_team}"]), key="pm_h_form")
            h_hfa = st.slider(f"Home Field Advantage", 0, max_hfa, value=int(params[f"hfa_{home_team}"]), key="pm_h_hfa")
        
        with st.expander(f"{away_team} Adjustments", expanded=True):
            a_elo = st.slider(f"Elo (Strength)", min_elo, max_elo, value=int(params[f"elo_{away_team}"]), key="pm_a_elo")
            a_sigma = st.slider(f"Sigma (Uncertainty)", min_sigma, max_sigma, value=int(params[f"sigma_{away_team}"]), key="pm_a_sigma")
            a_form = st.slider(f"Form (Momentum)", -min_max_form, min_max_form, value=int(params[f"form_{away_team}"]), key="pm_a_form")

        if lock_button:
            # CHECK 1: Have parameters changed?
            has_adjustment = (
                h_elo != int(params[f"elo_{home_team}"]) or
                h_sigma != int(params[f"sigma_{home_team}"]) or
                h_form != int(params[f"form_{home_team}"]) or
                h_hfa != int(params[f"hfa_{home_team}"]) or
                a_elo != int(params[f"elo_{away_team}"]) or
                a_sigma != int(params[f"sigma_{away_team}"]) or
                a_form != int(params[f"form_{away_team}"])
            )

            if not has_adjustment:
                message_slot.warning("⚠️ You have to adjust params to make your own predictions!")
            elif not user_id.strip():
                message_slot.error("👤 Please enter a username.")
            else:
                # SUCCESS: Prepare data and push to BQ
                mu_h_pred, mu_a_pred, p_win_pred, p_draw_pred, p_loss_pred = get_dynamic_drift(
                    h_elo, a_elo, h_sigma, a_sigma, h_hfa, h_form, a_form
                )
                sync_fixture()
                prediction_data = {
                    "prediction_id": f"{fixture_id_sync}_{user_id}",
                    "user": user_id,
                    "fixture_id": fixture_id_sync,
                    "p_win_home": p_win_pred,   # These come from the calculation in main_col
                    "p_draw_home": p_draw_pred,
                    "p_loss_home": p_loss_pred,
                    "created_utc": pd.Timestamp.now(tz='UTC').isoformat(),
                    "data_load_date": pd.Timestamp.utcnow().date().isoformat()
                }
                
                # Assuming you have a function in database.py called push_prediction
                try:
                    db.push_prediction(prediction_data)
                    message_slot.success(f"✅ Prediction locked for {user_id}!")
                    st.balloons()
                except Exception as e:
                    message_slot.error(f"Encountered error, prediction not locked.")
                    st.error(f"Error connecting to BigQuery: {e}")
                    # add logging error
# ==========================================
# 5. DATA TABLES: HISTORICAL PERFORMANCE
# ==========================================

home_fixtures, home_form = get_past_5_games(home_team, fixtures)
away_fixtures, away_form = get_past_5_games(away_team, fixtures)

mu_h, mu_a, p_win, p_draw, p_loss = get_dynamic_drift(
    h_elo, a_elo, h_sigma, a_sigma, h_hfa, h_form, a_form
)

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

    if app_mode == "Prediction Mode":
        template = "plotly_dark"
        legend_font_color = 'white'
    elif app_mode == "Free Play Mode":
        template = "plotly_white"
        legend_font_color = 'black'
    fig.update_layout(
        autosize=True,
        height=None,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Matchday Performance (ELO)", 
        template=template,
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
        legend_font_color=legend_font_color
    )
    return fig

with main_col:
    st.markdown("#### Match Outcome Probabilities")
    p1, p2, p3 = st.columns(3)
    p1.metric(f"{home_team} Win", f"{p_win:.1%}")
    p2.metric("Draw", f"{p_draw:.1%}")
    p3.metric(f"{away_team} Win", f"{p_loss:.1%}")
    
    st.markdown("---")

    # Visual Distribution Plot: Performance Overlap
    st.markdown(f"#### Head-to-Head ELO Distribution: {home_team} (H) vs {away_team} (A)")

    st.plotly_chart(plot_matchup(home_team, away_team, h_elo, a_elo, h_sigma, a_sigma, h_form, a_form, h_hfa), use_container_width=True)

    st.markdown("---")

    st.markdown("#### Team's Past 5 Games:")
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
        st.dataframe(
            home_fixtures[["Fixture", "Score", "Result", "Pts (xPts)"]], 
            use_container_width=True, 
            hide_index=True,
            # height=250
        )

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
        st.dataframe(
            away_fixtures[["Fixture", "Score", "Result", "Pts (xPts)"]], 
            use_container_width=True, 
            hide_index=True
        )
        
    st.markdown("---")
    st.markdown("### stochastic-football-v1")

    # Link to Kaggle
    st.markdown(
        """
        *A deep dive into the Bayesian-Elo logic and engine architecture. 13 years of PL historical data and Optuna hyperparameter tuning.*
        
        **Data Modeling & Backtesting**: 
        [notebook](https://www.kaggle.com/code/justinus/stochastic-football)  
        """, 
        unsafe_allow_html=True
    )

    # Link to Article/Technical Narrative
    st.markdown(
        """
        **Technical Narrative**:
        [kaggle](https://www.kaggle.com/writeups/justinus/stochastic-football) | [linkedin](https://www.linkedin.com/pulse/stochastic-football-justinus-kho-a3hpf/) | [medium](https://medium.com/@justinus.jx/stochastic-football-c2a44c6d856d)
        """
    )