# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import scipy.stats as stats
import plotly.graph_objects as go
# Import custom logic for simulation and data access
from engine import *
import database as db
from copy import deepcopy

# ==========================================
# 1. INITIALIZATION & DATA PRE-PROCESSING
# ==========================================

st.set_page_config(page_title="Stochastic Match Engine", layout="wide")

# Fetch initial fixtures and model parameters from BigQuery
fixtures = db.fetch_fixtures(-5, 2)
params = db.fetch_params(before=fixtures['date'].min().strftime('%Y-%m-%d'))[0]

# Pre-calculate Expected Points (xP) for all historical fixtures
# This allows us to compare actual performance vs. model expectations in the history tables
fixtures_next = fixtures[fixtures['home_score'].isnull()]
fixtures_next['display_name'] = "GW" + fixtures_next['round'].astype(str) + ": " + fixtures_next['Fixture']
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
params_now = deepcopy(params)
fixtures_next.loc[:, ['home_xP', 'away_xP', 'p_win', 'p_draw', 'p_loss']] = fixtures_next.apply(
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

TEAMS_LOOKUP = pd.concat(
    [fixtures[["home", "home_team"]].rename(columns={"home": "team_code", "home_team": "team_name"}), fixtures[["away", "away_team"]].rename(columns={"away": "team_code", "away_team": "team_name"})]
).drop_duplicates().set_index("team_name")["team_code"].to_dict()
TEAMS = sorted(list(TEAMS_LOOKUP.keys()))

# ==========================================
# 2. UI STYLING (CUSTOM CSS)
# ==========================================
# Load your single CSS file once at the top
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
local_css("style.css")

# --- GLOBAL MODE TOGGLE (Top of App) ---
MODES = ["Prediction Mode", "Free Play Mode"]
if "app_mode_sync" not in st.session_state:
    st.session_state.app_mode_sync = MODES[0]
current_mode = "prediction-mode" if st.session_state.app_mode_sync=='Prediction Mode' else "free-play-mode"
        
app_mode = st.session_state.app_mode_sync
app_mode = st.radio("", options=MODES, horizontal=True, key="app_mode_sync")

def sync_fixture():
    """
    Triggered when the user picks a specific match from the 
    official fixtures list in Prediction Mode.
    """
    # Get the string from the selectbox (e.g., "ARS vs SUN")
    selected_name = st.session_state.match_selector
    
    # Filter your dataframe to find the match row
    match_row = fixtures_next[fixtures_next['display_name'] == selected_name].iloc[0]

# ==========================================
# 5. DATA TABLES: HISTORICAL PERFORMANCE
# ==========================================

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
        dragmode=False, # Disables panning/zooming by default
        xaxis=dict(fixedrange=True), # Prevents horizontal zooming
        yaxis=dict(fixedrange=True), # Prevents vertical zooming
        legend=dict(
            # visible=False,
            # bgcolor='rgba(0,0,0,0)',
            # orientation="h",  # Horizontal legend often looks better in apps
            yanchor="top",
            # y=1.02,
            xanchor="right",
            # x=1,
            font=dict(color=legend_font_color)
        )
    )
    return fig

def plot_performance_moving_avg(agg_losses, engine_user_name='engine', window=21):
    """
    Plots a trailing 21-day average loss per user.
    Formula: sum(past 21 days total_loss_adj) / sum(past 21 days n_max)
    """
    df = agg_losses.copy()

    # 1. Prepare Datetimes and Sorting
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['user', 'date'])

    # 2. Calculate Trailing 21-Day Totals per User
    # We use a time-based rolling window ('21D')
    rolling_data = (
        df.set_index('date')
        .groupby('user')[['total_loss_adj', 'n_max']]
        .rolling(f'{window}D', min_periods=1)
        .sum()
        .reset_index()
    )

    # 3. Calculate the weighted moving average loss
    rolling_data['moving_avg_loss'] = rolling_data['total_loss_adj'] / rolling_data['n_max']
    
    # Merge the calculation back to the main plotting dataframe
    df = df.merge(rolling_data[['date', 'user', 'moving_avg_loss']], on=['user', 'date'])

    # 4. Filter for recent data (e.g., last 45 days of the dataset to see the 21-day trend)
    max_date = df['date'].max()
    start_date = max_date - timedelta(days=45)
    plot_df = df[df['date'] >= start_date]

    if plot_df.empty:
        return "Not enough data to plot trailing average."

    # 5. Identify Top 3 (Based on the VERY LAST trailing average)
    final_scores = plot_df.groupby('user').tail(1)
    human_scores = final_scores[final_scores['user'] != engine_user_name]
    top_3_users = human_scores.nsmallest(3, 'moving_avg_loss')['user'].tolist()

    # 6. Visualization
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
            x=user_data['date'],
            y=user_data['moving_avg_loss'],
            mode='lines+markers' if (is_engine or is_top3) else 'lines',
            name=user,
            line=dict(color=color, width=width),
            showlegend=(is_engine or is_top3),
            hovertemplate=f"<b>{user}</b><br>21-Day Avg Loss: %{{y:.3f}}<extra></extra>"
        ))

        if is_engine or is_top3:
            fig.add_annotation(
                x=last_row['date'], y=last_row['moving_avg_loss'],
                text=label, showarrow=False, xanchor='left', yanchor='middle', xshift=10,
                font=dict(color=color, size=12)
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"), 
        dragmode=False,
        xaxis=dict(gridcolor='#333', zeroline=False, automargin=True),
        yaxis=dict(gridcolor='#333', zeroline=False, automargin=True, title="Weighted Loss (21D)"),
        margin=dict(t=50, b=50),
        hovermode="x unified",
        showlegend=False
    )

    return fig

if app_mode == 'Free Play Mode':
    st.markdown(f'<div class="free-play-mode">', unsafe_allow_html=True)
    st.sidebar.header("Free Play Mode")
    with st.sidebar:
        st.markdown('<div id="sidebar">', unsafe_allow_html=True)
        
        if app_mode == "Free Play Mode":
            # Call this immediately after your app_mode selection
            home_team = st.selectbox("Home Team", TEAMS, index=0)
            home_team_code = TEAMS_LOOKUP[home_team]
            
            # Dynamic UI Scaling: Calculate slider ranges based on current data distribution
            elos = [int(v) for k, v in params_now.items() if 'elo' in k]
            min_elo, max_elo = round(min(elos), -2), round(max(elos), -2)

            sigmas = [int(v) for k, v in params_now.items() if 'sigma' in k]
            min_sigma, max_sigma = round(min(sigmas), -2), round(max(sigmas), -2)

            forms = [int(v) for k, v in params_now.items() if 'form' in k]
            min_max_form = round(max(abs(min(forms)), abs(max(forms))), -2)

            hfas = [int(v) for k, v in params_now.items() if 'hfa' in k]
            max_hfa = round(max(hfas), -2)

            with st.expander("Home Team Adjustments", expanded=True):
                h_elo = st.slider("Elo (Strength)", min_elo, max_elo, value=int(params_now[f"elo_{home_team_code}"]), key="fs_h_elo")
                h_sigma = st.slider("Sigma (Uncertainty)", min_sigma, max_sigma, value=int(params_now[f"sigma_{home_team_code}"]), key="fs_h_sigma")
                h_form = st.slider("Momentum (Form) ", -min_max_form, min_max_form, value=int(params_now[f"form_{home_team_code}"]), key="fs_h_form")
                h_hfa = st.slider("Home Field Advantage", 0, max_hfa, value=int(params_now[f"hfa_{home_team_code}"]), key="fs_h_hfa")
            
            away_team = st.selectbox("Away Team", TEAMS, index=1)
            away_team_code = TEAMS_LOOKUP[away_team]
            
            with st.expander("Away Team Adjustments", expanded=True):
                a_elo = st.slider("Elo (Strength)", min_elo, max_elo, value=int(params_now[f"elo_{away_team_code}"]), key="fs_a_elo")
                a_sigma = st.slider("Sigma (Uncertainty)", min_sigma, max_sigma, value=int(params_now[f"sigma_{away_team_code}"]), key="fs_a_sigma")
                a_form = st.slider("Momentum (Form) ", -min_max_form, min_max_form, value=int(params_now[f"form_{away_team_code}"]), key="fs_a_form")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    fixtures["Score"] = fixtures["home_score"].astype(int).astype(str) + " - " + fixtures["away_score"].astype(int).astype(str)    
    home_fixtures, home_form = get_past_5_games(home_team_code, fixtures)
    away_fixtures, away_form = get_past_5_games(away_team_code, fixtures)
    mu_h, mu_a, p_win, p_draw, p_loss = get_dynamic_drift(
        h_elo, a_elo, h_sigma, a_sigma, h_hfa, h_form, a_form
    )
        
    st.markdown("#### Match Outcome Probabilities")
    p1, p2, p3 = st.columns(3)
    p1.metric(f"{home_team} Win", f"{p_win:.1%}")
    p2.metric("Draw", f"{p_draw:.1%}")
    p3.metric(f"{away_team} Win", f"{p_loss:.1%}")
    
    st.markdown("---")

    # Visual Distribution Plot: Performance Overlap
    st.markdown(f"#### Head-to-Head ELO Distribution: {home_team} (H) vs {away_team} (A)")

    st.plotly_chart(
        plot_matchup(home_team, away_team, h_elo, a_elo, h_sigma, a_sigma, h_form, a_form, h_hfa), 
        use_container_width=True,
        config={
            'staticPlot': False,      # Keep it interactive for hovers...
            'scrollZoom': False,      # ...but disable scroll-wheel zoom
            'displayModeBar': False,  # Hide the annoying floating toolbar on mobile
            'showAxisDragHandles': False
        }
    )

    st.markdown("---")

    st.markdown("#### Team's Past 5 Games:")
    c1, c2 = st.columns(2)

    # Home History Table
    with c1:
        st.caption(f"{home_team}'s Past 5 Games:")
        home_fixtures['Pts'] = np.select(
            [home_fixtures['home']==home_team_code],
            [home_fixtures['home_point']], home_fixtures['away_point']
        )
        home_fixtures['xPts'] = np.select(
            [home_fixtures['home']==home_team_code],
            [home_fixtures['home_xP']], home_fixtures['away_xP']
        ).round(1)
        home_fixtures['Pts (xPts)'] = home_fixtures.apply(
            lambda row: f"{row['Pts']} ({row['xPts']})",
            axis=1
        )
        st.dataframe(
            home_fixtures[["Fixture", "Score", "Result", "Pts (xPts)"]], 
            use_container_width=True, 
            hide_index=True
            # height=250
        )

    # Away History Table
    with c2:
        st.caption(f"{away_team}'s Past 5 Games:")
        away_fixtures['Pts'] = np.select(
            [away_fixtures['away']==away_team_code],
            [away_fixtures['away_point']], away_fixtures['home_point']
        )
        away_fixtures['xPts'] = np.select(
            [away_fixtures['away']==away_team_code],
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
    st.markdown('</div>', unsafe_allow_html=True)
    
elif app_mode == 'Prediction Mode':
    st.markdown(f'<div class="prediction-mode">', unsafe_allow_html=True)
    st.sidebar.header("Prediction Panel")
    with st.sidebar:
        st.markdown('<div id="sidebar">', unsafe_allow_html=True)
        if app_mode == "Prediction Mode":
            if "fixture_id_sync" not in st.session_state:
                st.session_state.fixture_id_sync = fixtures_next['id'].values[0]
            fixture_id_sync = st.session_state.fixture_id_sync
                
            
            user_id = st.text_input("Username")
            
            # Fixture Selection
            fx = st.selectbox("Select Match", options=fixtures_next['display_name'].tolist(), key="match_selector", on_change=sync_fixture)
            home_team = fixtures_next[fixtures_next['display_name']==fx]['home'].values[0]
            away_team = fixtures_next[fixtures_next['display_name']==fx]['away'].values[0]
            
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
            
            st.markdown(f"""
                <style>
                /* Target the slider track background */
                html:has(.prediction-mode) section.stSidebar div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
                    background: linear-gradient(
                        to right, 
                        #FF4B4B 0%, 
                        #FF4B4B {bound_1}%, 
                        #808080 {bound_1}%, 
                        #808080 {bound_2}%, 
                        #007BFF {bound_2}%, 
                        #007BFF 100%
                    ) !important;
                }}
                </style>
                """, unsafe_allow_html=True)

            # Derive probabilities from the boundaries
            p_win_pred = bound_1 / 100
            p_draw_pred = (bound_2 - bound_1) / 100
            p_loss_pred = (100 - bound_2) / 100
            
            # Add labels to sliders
            st.markdown('<div id="pred-slider-label">', unsafe_allow_html=True)
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
            st.markdown('</div>', unsafe_allow_html=True)
            
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
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Match Cards
    st.markdown("###### Upcoming Matches:")
    content = '<div class="scroll-container">'
    for home, away, p_win, p_draw, p_loss in fixtures_next.loc[fixtures_next.ix==1, ['home', 'away', 'p_win', 'p_draw', 'p_loss']].values:
        content += f"""
<div class="match-card">
    <div class="team-name">{home} vs {away}</div>
<div class="odds-row">
    <div class="win-text">{p_win*100:.0f}%</div>
    <div class="draw-text">{p_draw*100:.0f}%</div>
    <div class="loss-text">{p_loss*100:.0f}%</div>
</div>
<div class="stacked-bar">
    <div class="segment win" style="width: {p_win*100:.0f}%"></div>
    <div class="segment draw" style="width: {p_draw*100:.0f}%"></div>
    <div class="segment loss" style="width: {p_loss*100:.0f}%"></div>
</div>
    <div class="stamp-text">{datetime.now(timezone.utc).strftime("%b %d @ %I:%M UTC")}</div>
</div>
        """
    content += '</div>'

    st.markdown(content, unsafe_allow_html=True)
    st.markdown("---")
    
    prediction_losses = calculate_prediction_losses(
            predictions_list=\
                predictions.sort_values('created_utc', ascending=False).drop_duplicates('prediction_id', keep='first')[['fixture_id', 'user', 'p_win_home', 'p_draw_home', 'p_loss_home']].values.tolist() +\
                    [[f[0], 'engine', f[1], f[2], f[3]] for f in fixtures[['id', 'p_win', 'p_draw', 'p_loss']].values.tolist()], 
            results_dict=\
                fixtures[['id', 'date', 'home_point']].set_index('id', drop=True).transpose().to_dict()
    )

    agg_losses_, penalty = calculate_aggregate_losses(prediction_losses, null_guess_probability=null_guess_probability)
    agg_losses = pd.DataFrame(
        columns=['date', 'user', 'total_loss', 'total_loss_adj', 'n_preds', 'n_max'],
        data=agg_losses_
    )
    st.markdown("##### Score (Lower is Better):")
    st.plotly_chart(
        plot_performance_moving_avg(agg_losses, window=6), 
        use_container_width=True,
        config={
            'staticPlot': False,      # Keep it interactive for hovers...
            'scrollZoom': False,      # ...but disable scroll-wheel zoom
            'displayModeBar': False,  # Hide the annoying floating toolbar on mobile
            'showAxisDragHandles': False
        }
    )
    
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
        predictions_combined[['p_win_home', 'p_draw_home', 'p_loss_home']] = (predictions_combined[['p_win_home', 'p_draw_home', 'p_loss_home']] * 100).round(0).astype(int).astype(str) + "%"
        predictions_combined['Prediction'] = predictions_combined['p_win_home'] + "-" + predictions_combined['p_draw_home'] + "-" + predictions_combined['p_loss_home']
            
        matches = predictions_combined[['home', 'away']].drop_duplicates()
        pred_cards_html = '<div class="preds-scroll-container">'
        for _, m in matches.iterrows():
            home, away = m['home'], m['away']
            
            # Filter predictions for this specific match
            match_preds = predictions_combined[
                (predictions_combined['home'] == home) & 
                (predictions_combined['away'] == away)
            ]
            
            # Start the Match Section
            pred_cards_html += f"""
<div class="match-section">
    <div class="match-title">⚽ {home} vs {away}</div>
            """
            
            # Add a card for every user who predicted this match
            for _, pred in match_preds.iterrows():
                pred_cards_html += f"""
<div class="prediction-card">
    <span class="user-name">@{pred['user']}</span>
    <span class="prediction-value">{pred['Prediction']}</span>
</div>
                """
            
            pred_cards_html += "</div>" # Close Section
        pred_cards_html += "</div>"
            
        st.markdown(pred_cards_html, unsafe_allow_html=True)
        
    with c2:
        st.markdown("##### Leaderboard:")
        
        leaderboard_df = agg_losses[pd.to_datetime(agg_losses['date']) >= datetime(2026, 2, 6)]
        leaderboard = pd.DataFrame(
            columns=['score'], 
            data=(leaderboard_df.groupby('user')['total_loss_adj'].sum()/ leaderboard_df.groupby('user')['n_max'].sum()).round(2)
        ).sort_values(['score']).reset_index()
        
        st.dataframe(
            leaderboard.rename(columns={'user': 'Username', 'score': 'Score (Lower is Better)'}),
            use_container_width=True,
            hide_index=True
            # height=250
        )