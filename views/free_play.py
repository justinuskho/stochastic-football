import numpy as np
import streamlit as st
from engine import get_past_5_games, get_dynamic_drift
from views.context import AppContext
from views.charts import plot_matchup


def render_free_play(ctx: AppContext) -> None:
    st.markdown('<div class="free-play-mode">', unsafe_allow_html=True)
    st.sidebar.header("Free Play Mode")

    with st.sidebar:
        st.markdown('<div id="sidebar">', unsafe_allow_html=True)

        home_team = st.selectbox("Home Team", ctx.TEAMS, index=0)
        home_team_code = ctx.TEAMS_LOOKUP[home_team]

        elos = [int(v) for k, v in ctx.params_now.items() if 'elo' in k]
        min_elo, max_elo = round(min(elos), -2), round(max(elos), -2)

        sigmas = [int(v) for k, v in ctx.params_now.items() if 'sigma' in k]
        min_sigma, max_sigma = round(min(sigmas), -2), round(max(sigmas), -2)

        forms = [int(v) for k, v in ctx.params_now.items() if 'form' in k]
        min_max_form = round(max(abs(min(forms)), abs(max(forms))), -2)

        hfas = [int(v) for k, v in ctx.params_now.items() if 'hfa' in k]
        max_hfa = round(max(hfas), -2)

        with st.expander("Home Team Adjustments", expanded=True):
            h_elo = st.slider("Elo (Strength)", min_elo, max_elo, value=int(ctx.params_now[f"elo_{home_team_code}"]), key="fs_h_elo")
            h_sigma = st.slider("Sigma (Uncertainty)", min_sigma, max_sigma, value=int(ctx.params_now[f"sigma_{home_team_code}"]), key="fs_h_sigma")
            h_form = st.slider("Momentum (Form) ", -min_max_form, min_max_form, value=int(ctx.params_now[f"form_{home_team_code}"]), key="fs_h_form")
            h_hfa = st.slider("Home Field Advantage", 0, max_hfa, value=int(ctx.params_now[f"hfa_{home_team_code}"]), key="fs_h_hfa")

        away_team = st.selectbox("Away Team", ctx.TEAMS, index=1)
        away_team_code = ctx.TEAMS_LOOKUP[away_team]

        with st.expander("Away Team Adjustments", expanded=True):
            a_elo = st.slider("Elo (Strength)", min_elo, max_elo, value=int(ctx.params_now[f"elo_{away_team_code}"]), key="fs_a_elo")
            a_sigma = st.slider("Sigma (Uncertainty)", min_sigma, max_sigma, value=int(ctx.params_now[f"sigma_{away_team_code}"]), key="fs_a_sigma")
            a_form = st.slider("Momentum (Form) ", -min_max_form, min_max_form, value=int(ctx.params_now[f"form_{away_team_code}"]), key="fs_a_form")

        st.markdown('</div>', unsafe_allow_html=True)

    fixtures = ctx.fixtures.copy()
    fixtures["Score"] = fixtures["home_score"].astype(int).astype(str) + " - " + fixtures["away_score"].astype(int).astype(str)
    home_fixtures, _ = get_past_5_games(home_team_code, fixtures)
    away_fixtures, _ = get_past_5_games(away_team_code, fixtures)
    _, _, p_win, p_draw, p_loss = get_dynamic_drift(h_elo, a_elo, h_sigma, a_sigma, h_hfa, h_form, a_form)

    st.markdown("#### Match Outcome Probabilities")
    p1, p2, p3 = st.columns(3)
    p1.metric(f"{home_team} Win", f"{p_win:.1%}")
    p2.metric("Draw", f"{p_draw:.1%}")
    p3.metric(f"{away_team} Win", f"{p_loss:.1%}")

    st.markdown("---")
    st.markdown(f"#### Head-to-Head ELO Distribution: {home_team} (H) vs {away_team} (A)")
    st.plotly_chart(
        plot_matchup(home_team, away_team, h_elo, a_elo, h_sigma, a_sigma, h_form, a_form, h_hfa, mode="Free Play"),
        use_container_width=True,
        config={'staticPlot': False, 'scrollZoom': False, 'displayModeBar': False, 'showAxisDragHandles': False}
    )

    st.markdown("---")
    st.markdown("#### Team's Past 5 Games:")
    c1, c2 = st.columns(2)

    with c1:
        st.caption(f"{home_team}'s Past 5 Games:")
        home_fixtures['Pts'] = np.select(
            [home_fixtures['home'] == home_team_code],
            [home_fixtures['home_point']], home_fixtures['away_point']
        )
        home_fixtures['xPts'] = np.select(
            [home_fixtures['home'] == home_team_code],
            [home_fixtures['home_xP']], home_fixtures['away_xP']
        ).round(1)
        home_fixtures['Pts (xPts)'] = home_fixtures.apply(lambda row: f"{row['Pts']} ({row['xPts']})", axis=1)
        st.dataframe(home_fixtures[["Fixture", "Score", "Result", "Pts (xPts)"]], use_container_width=True, hide_index=True)

    with c2:
        st.caption(f"{away_team}'s Past 5 Games:")
        away_fixtures['Pts'] = np.select(
            [away_fixtures['away'] == away_team_code],
            [away_fixtures['away_point']], away_fixtures['home_point']
        )
        away_fixtures['xPts'] = np.select(
            [away_fixtures['away'] == away_team_code],
            [away_fixtures['away_xP']], away_fixtures['home_xP']
        ).round(1)
        away_fixtures['Pts (xPts)'] = away_fixtures.apply(lambda row: f"{row['Pts']} ({row['xPts']})", axis=1)
        st.dataframe(away_fixtures[["Fixture", "Score", "Result", "Pts (xPts)"]], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### stochastic-football-v1")
    st.markdown(
        """
        *A deep dive into the Bayesian-Elo logic and engine architecture. 13 years of PL historical data and Optuna hyperparameter tuning.*

        **Data Modeling & Backtesting**:
        [notebook](https://www.kaggle.com/code/justinus/stochastic-football)
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        **Technical Narrative**:
        [kaggle](https://www.kaggle.com/writeups/justinus/stochastic-football) | [linkedin](https://www.linkedin.com/pulse/stochastic-football-justinus-kho-a3hpf/) | [medium](https://medium.com/@justinus.jx/stochastic-football-c2a44c6d856d)
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
