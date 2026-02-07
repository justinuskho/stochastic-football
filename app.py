# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Import custom logic for simulation and data access
from engine import *
import database as db
import json

# ==========================================
# 1. INITIALIZATION & DATA PRE-PROCESSING
# ==========================================

st.set_page_config(page_title="Stochastic Match Engine", layout="wide")

# Fetch initial fixtures and model parameters from BigQuery
fixtures = db.fetch_fixtures(-5, 1)
params = db.fetch_params(at_start_of="GW20")[0]

# Pre-calculate Expected Points (xP) for all historical fixtures
# This allows us to compare actual performance vs. model expectations in the history tables
fixtures_next = fixtures[fixtures['home_score'].isnull()]
fixtures_next['display_name'] = "R" + fixtures_next['round'].astype(str) + ": " + fixtures_next['home'] + " vs " + fixtures_next['away']
fixtures = fixtures[fixtures['home_score'].notnull()] 
fixtures.loc[:, ['home_xP', 'away_xP', 'p_win', 'p_draw', 'p_loss']] = fixtures.apply(
    lambda row: run_simulation(
        params, 
        (row['home'], row['away']),
        (row['home_point'], row['away_point']),
        new_season=False
    )[2],
    axis=1,
    result_type='expand'
).values

predictions = db.fetch_predictions(fixture_id_list=fixtures['id'].unique().tolist() + fixtures_next['id'].unique().tolist())
predictions_next = predictions[predictions['fixture_id'].isin(fixtures_next['id'])]
predictions = predictions[predictions['fixture_id'].isin(fixtures['id'])]
null_guess_probability=0.25
market_suppliers = set(['google', 'opta_analyst', 'Google', 'OptaAnalyst'])

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
        st.markdown(f"""
            <style>
            /* 1. Global Reset & Mobile Text */
            h1, h2, h3, p, span {{
                color: white;
            }}
            .stApp {{
                background-color: #0E1117;
                color: #fafafa;
            }}

            /* 3. Buttons & Inputs */
            div.stButton > button {{
                background-color: #00FFCC !important;
                border: none;
                width: 100%; /* Better for mobile touch */
            }}
            div.stButton > button p {{
                color: black !important;
                font-weight: bold;
            }}

            /* Input text color fix (Ensuring contrast) */
            div[data-testid="stTextInput"] input {{
                background-color: rgba(255, 255, 255, 0.1) !important;
                color: black !important; 
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}

            /* 5. Slider Customization */
            div.stSlider > div[data-baseweb="slider"] > div > div {{
                background: linear-gradient(to right, 
                    #FF4B4B 0%, #FF4B4B {bound_1}%, 
                    #808080 {bound_1}%, #808080 {bound_2}%, 
                    #007BFF {bound_2}%, #007BFF 100%) !important;
            }}
            
            div.stSlider [role="slider"] {{
                background-color: white !important;
                border: 2px solid #31333F !important;
            }}
            
            /* Hide slider labels/ticks but keep functionality */
            div[data-testid="stSlider"] label{{
                font-size: 0px !important;
                display: none !important;
                visibility: hidden !important;
            }}
            
            div[data-testid="stSlider"] [data-baseweb="slider"] div {{
                font-size: 0px !important;
            }}
            
            /* Ensure tooltips are readable */
            div[data-baseweb="tooltip"] {{
                background-color: black !important;
                color: black !important;
            }}
            
            /* Target the sidebar section using role (more stable than testid) */
            section[role="complementary"] {{
                background-color: #0E1117 !important;
                border-right: 1px solid #30363d !important;
            }}

            /* Target the mobile overlay (scrim) when sidebar is open */
            div[data-baseweb="overlay"] {{
                background-color: rgba(0, 0, 0, 0.8) !important;
                backdrop-filter: blur(2px);
            }}

            # /* 2. Sidebar Internal Styling (Anchored to your custom ID) */
            # #sidebar label, #sidebar p, #sidebar span {{
            #     color: #FFFFFF !important;
            # }}

            # #sidebar div[role="combobox"], #sidebar input {{
            #     background-color: #161b22 !important;
            #     color: white !important;
            #     border: 1px solid #30363d !important;
            # }}

            # #sidebar button {{
            #     background-color: #00FFCC !important;
            #     color: black !important;
            #     font-weight: bold !important;
            #     width: 100% !important;
            #     border: none !important;
            #     box-shadow: 0 4px 12px rgba(0, 255, 204, 0.2);
            # }}
            
            section[data-testid="stSidebar"] {{
                padding-top: 1rem;
                background-color: rgba(0,0,0,0.8);
            }}

            /* 7. Hide the default Streamlit Header for a cleaner mobile look */
            header {{
                background-color: rgba(0,0,0,0) !important;
            }}
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
    
app_mode = st.session_state.app_mode_sync
app_mode = st.radio("Select Mode:", options=MODES, horizontal=True, key="app_mode_sync", on_change=on_mode_change)
    
home_team = st.session_state.home_sync
away_team = st.session_state.away_sync

# --- 4. LEFT SIDEBAR (Free Play only) ---

with st.sidebar:
    st.markdown('<div id="sidebar">', unsafe_allow_html=True)
    
    if app_mode == "Free Play Mode":
        # Call this immediately after your app_mode selection
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
            
        
        st.subheader("Prediction Panel")
        
        user_id = st.text_input("Username")
        
        # Fixture Selection
        fx = st.selectbox("Select Official Fixture", options=fixtures_next['display_name'].tolist(), key="match_selector", on_change=sync_fixture)
        home_team = fx.split(" ")[1]        
        h_elo = int(params[f"elo_{home_team}"])
        h_sigma = int(params[f"sigma_{home_team}"])
        h_form = int(params[f"form_{home_team}"])
        h_hfa = int(params[f"hfa_{home_team}"])

        away_team = fx.split(" ")[3]
        a_elo = int(params[f"elo_{away_team}"])
        a_sigma = int(params[f"sigma_{away_team}"])
        a_form = int(params[f"form_{away_team}"])
        
        # --- Replacement for the Adjustment Expanders ---
        st.caption("Drag the handles to define Win, Draw, and Loss zones")

        # Define the range (0% to 100%)
        thresholds = list(range(0, 101))
        
        lab1, lab2, lab3 = st.columns(3, vertical_alignment="center")

        with lab1:
            lab1_slot = st.empty()

        with lab2:
            lab2_slot = st.empty()

        with lab3:
            lab3_slot = st.empty()
        
        # We default it to roughly 33% and 66% splits
        bound_1, bound_2 = st.select_slider(
            "Win | Draw | Loss",
            options=thresholds,
            value=(35, 65),
            help="First handle: Home Win threshold. Second handle: Away Win threshold."
        )

        # Derive probabilities from the boundaries
        p_win_pred = bound_1 / 100
        p_draw_pred = (bound_2 - bound_1) / 100
        p_loss_pred = (100 - bound_2) / 100
        
        # Add labels to sliders

        lab1_slot.markdown(f"""
            <div style='text-align: left; margin-bottom: -25px;'>
                <p style='color: #FF4B4B; font-weight: bold; margin: 0; line-height: 1.2;'>
                    {home_team}<br>
                    <span style='font-size: 0.9em; opacity: 0.9;'>{p_win_pred*100:.0f}%</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        lab2_slot.markdown(f"""
            <div style='text-align: center; margin-bottom: -25px;'>
                <p style='color: #808080; font-weight: bold; margin: 0; line-height: 1.2;'>
                    Draw<br>
                    <span style='font-size: 0.9em; opacity: 0.9;'>{p_draw_pred*100:.0f}%</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        lab3_slot.markdown(f"""
            <div style='text-align: right; margin-bottom: -25px;'>
                <p style='color: #007BFF; font-weight: bold; margin: 0; line-height: 1.2;'>
                    {away_team}<br>
                    <span style='font-size: 0.9em; opacity: 0.9;'>{p_loss_pred*100:.0f}%</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        lock_button = st.button("🔒 Lock Predictions", use_container_width=True)
        message_slot = st.empty()
        
        st.markdown(f"""
            ---
            ### 💡 **How to Play**
            * **Pick Your Confidence:** Move the left and right slider to set your probabilities. 
            * **Submit:** Click **Lock Predictions** to finalize your entry for this fixture.
            * **The Goal:** Minimize your **Weighted Loss**. High confidence in the correct result yields the best rank.

            > ⚠️ *Note: Missing a fixture applies a **Volume Penalty** to your form, treating the game as a random {null_guess_probability*100:.0f}% guess.*
            """)
        
        st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

apply_custom_theme(app_mode)

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
        # margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Matchday Performance (ELO)", 
        template=template,
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
        legend_font_color=legend_font_color
    )
    return fig

def plot_performance_moving_avg(agg_losses, engine_user_name='engine', window=3):
    """
    Plots a trailing 3-week average of weighted log loss.
    """
    df = agg_losses.copy()

    # 1. Numerical Sorting
    df['gw_num'] = df['gameweek'].str[-2:].astype(int)
    df = df.sort_values(['user', 'gw_num'])

    # 2. Calculate Trailing Average per User
    # 'min_periods=1' ensures if only 1 week of 3 is available, it averages by 1
    df['moving_avg_loss'] = df.groupby('user')['weighted_loss']\
                              .transform(lambda x: x.rolling(window=window, min_periods=1).mean())

    # 3. Filter for minimum data points (Need 6 GWs to show a meaningful 3-GW trend)
    max_gw = df['gw_num'].max()
    min_gw_required = df['gw_num'].min() + (window) # e.g., if start is GW1, plot from GW6
    plot_df = df[df['gw_num'] >= min_gw_required]

    if plot_df.empty:
        return "Not enough data to plot trailing average (Requires at least 6 Gameweeks)."

    # 4. Identify Top 3 (Based on the VERY LAST trailing average)
    final_scores = plot_df.groupby('user').last().reset_index()
    human_scores = final_scores[final_scores['user'] != engine_user_name]
    top_3_users = human_scores.nsmallest(3, 'moving_avg_loss')['user'].tolist()

    # 5. Visualization
    fig = go.Figure()
    colors = {'engine': '#FF4B4B', 'top3': ['#00FFCC', '#007BFF', '#7A5FFF'], 'other': '#4E4E4E'}

    for user in plot_df['user'].unique():
        user_data = plot_df[plot_df['user'] == user]
        last_row = user_data.iloc[-1]
        
        is_engine = (user == engine_user_name)
        is_top3 = (user in top_3_users)
        
        color = colors['engine'] if is_engine else (colors['top3'][top_3_users.index(user)] if is_top3 else colors['other'])
        width = 4 if is_engine else (3 if is_top3 else 1.5)
        label = f"🤖 {user.upper()}" if is_engine else (f"🏆 {user}" if is_top3 else user)

        fig.add_trace(go.Scatter(
            x=user_data['gameweek'],
            y=user_data['moving_avg_loss'],
            mode='lines+markers' if (is_engine or is_top3) else 'lines',
            name=user,
            line=dict(color=color, width=width),
            showlegend=(is_engine or is_top3),
            hovertemplate=f"<b>{user}</b><br>3-GW Avg: %{{y:.3f}}<extra></extra>"
        ))

        if is_engine or is_top3:
            fig.add_annotation(
                x=last_row['gameweek'], y=last_row['moving_avg_loss'],
                text=label, showarrow=False, xanchor='left', xshift=10,
                font=dict(color=color, size=12)
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    
        # This is the key: Force all text to white
        font=dict(color="white"), 
        title=dict(
            text=f"<b>{window}-Week Trailing Score</b><br><sup>Lower is Better</sup>",
            font=dict(size=18, color='white')
        ),
        xaxis=dict(
            title="Gameweek", 
            # showgrid=False,
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='#333', # Darker grid lines
            zeroline=False  # Removes the heavy line at the start
        ),
        yaxis=dict(
            title="Avg Weighted Log-Loss", 
            showgrid=True, 
            gridcolor='#333',
            zeroline=False,
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
        ),
        # margin=dict(l=50, r=150, t=100, b=50), # Explicitly padding all sides
        autosize=True,
        hovermode="x unified",
        # Let the template handle font color unless you have a custom hex
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            orientation="h",  # Horizontal legend often looks better in apps
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='white')
        )
    )

    return fig


if app_mode == 'Free Play Mode':
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
elif app_mode == 'Prediction Mode':
    prediction_losses = calculate_prediction_losses(
            predictions_list=\
                predictions.sort_values('created_utc', ascending=False).drop_duplicates('prediction_id', keep='first')[['fixture_id', 'user', 'p_win_home', 'p_draw_home', 'p_loss_home']].values.tolist() +\
                    [[f[0], 'engine', f[1], f[2], f[3]] for f in fixtures[['id', 'p_win', 'p_draw', 'p_loss']].values.tolist()], 
            results_dict=\
                fixtures[['id', 'home_point']].set_index('id', drop=True).transpose().to_dict()
    )

    agg_losses_, penalty = calculate_aggregate_losses(prediction_losses, null_guess_probability=null_guess_probability)
    agg_losses = pd.DataFrame(
        columns=['gameweek', 'user', 'total_loss', 'n_preds', 'weighted_loss'],
        data=agg_losses_
    )
    
    st.markdown('<div id="leaderchart">', unsafe_allow_html=True)
    st.plotly_chart(
        plot_performance_moving_avg(agg_losses, window=2), 
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("##### Accepted Predictions:")
        
        if 'predictions_user' not in st.session_state:
            st.session_state.predictions_user = pd.DataFrame(columns=['prediction_id', 'user', 'fixture_id', 'p_win_home', 'p_draw_home', 'p_loss_home', 'created_utc', 'source', 'data_load_date', 'home', 'away'])
        
        if lock_button:
            if not user_id.strip():
                message_slot.error("👤 Please enter a username.")
            elif user_id.strip() in market_suppliers:
                message_slot.error("❌ Invalid username.")
            else:
                sync_fixture()
                prediction_data = {
                    "prediction_id": f"{fixture_id_sync}_{user_id}",
                    "user": user_id,
                    "fixture_id": fixture_id_sync,
                    "p_win_home": p_win_pred,   # These come from the calculation in main_col
                    "p_draw_home": p_draw_pred,
                    "p_loss_home": p_loss_pred,
                    "created_utc": pd.Timestamp.now(tz='UTC').isoformat(),
                    "source": "user",
                    "data_load_date": pd.Timestamp.utcnow().date().isoformat()
                }
                
                new_prediction = pd.DataFrame([prediction_data])
                new_prediction[['home', 'away']] = new_prediction['fixture_id'].str.split("_", expand=True).iloc[:, [2, 3]]
                
                st.session_state.predictions_user = pd.concat(
                    [st.session_state.predictions_user, new_prediction], 
                    ignore_index=True
                )
                
                # Assuming you have a function in database.py called push_prediction
                try:
                    db.push_prediction(prediction_data)
                    message_slot.success(f"✅ Prediction locked for {user_id}!")
                    st.balloons()
                except Exception as e:
                    message_slot.error(f"Encountered error, prediction not locked.")
                    st.error(f"Error connecting to BigQuery: {e}")
        
        predictions_combined = pd.concat([
                    predictions_next.loc[~predictions_next['source'].isin(market_suppliers), ['user', 'home', 'away', 'p_win_home', 'p_draw_home', 'p_loss_home', 'created_utc', 'prediction_id',]],
                    st.session_state.predictions_user[['user', 'home', 'away', 'p_win_home', 'p_draw_home', 'p_loss_home', 'created_utc', 'prediction_id']]
                ]).sort_values(['created_utc'], ascending=False).drop_duplicates(['prediction_id'], keep='first')\
                    .sort_values(['home', 'p_win_home', 'p_draw_home'], ascending=[True, False, True])\
                        [['user', 'home', 'away', 'p_win_home', 'p_draw_home', 'p_loss_home']]
        
        st.dataframe(
            predictions_combined,
            use_container_width=True, 
            hide_index=True,
            # height=250
        )
    with c2:
        st.markdown("##### Leaderboard (All Time):")
        
        leaderboard = pd.DataFrame(
            columns=['score'], 
            data=agg_losses.groupby('user')['weighted_loss'].sum()/ agg_losses.groupby('user')['gameweek'].nunique()
        ).sort_values(['score'])
        
        st.dataframe(
            leaderboard,
            use_container_width=True
            # height=250
        )