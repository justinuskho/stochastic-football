import pandas as pd
import streamlit as st
from datetime import datetime, timezone
import database as db
from engine import calculate_prediction_losses, calculate_aggregate_losses
from views.context import AppContext, PredictionContext
from views.charts import plot_performance_moving_avg


def build_prediction_context(ctx: AppContext) -> PredictionContext:
    prediction_losses = calculate_prediction_losses(
        predictions_list=(
            ctx.predictions.sort_values('created_utc', ascending=False)
            .drop_duplicates('prediction_id', keep='first')
            [['fixture_id', 'user', 'p_win_home', 'p_draw_home', 'p_loss_home']]
            .values.tolist()
        ) + [
            [f[0], 'engine', f[1], f[2], f[3]]
            for f in ctx.fixtures[['id', 'p_win', 'p_draw', 'p_loss']].values.tolist()
        ],
        results_dict=ctx.fixtures[['id', 'date', 'home_point']].set_index('id', drop=True).transpose().to_dict()
    )
    agg_losses_, penalty = calculate_aggregate_losses(prediction_losses, null_guess_probability=ctx.null_guess_probability)
    agg_losses = pd.DataFrame(columns=['date', 'user', 'total_loss', 'n_preds'], data=agg_losses_)

    all_benchmarks = ctx.market_suppliers | {'engine'}
    human_users = [u for u in agg_losses['user'].unique() if u not in all_benchmarks]

    leaderboard_df = agg_losses[pd.to_datetime(agg_losses['date']) >= datetime(2026, 2, 6)]
    n_max = leaderboard_df.groupby('user')['n_preds'].sum().max()
    if n_max and n_max > 0:
        leaderboard = pd.DataFrame(
            columns=['score'],
            data=((leaderboard_df.groupby('user')['total_loss'].sum() + (n_max - leaderboard_df.groupby('user')['n_preds'].sum()) * penalty) / n_max).round(2)
        ).sort_values('score').reset_index()
    else:
        leaderboard = pd.DataFrame(columns=['user', 'score'])

    return PredictionContext(
        agg_losses=agg_losses,
        leaderboard=leaderboard,
        penalty=penalty,
        all_benchmarks=all_benchmarks,
        human_users=human_users,
        prediction_losses=prediction_losses,
    )


def render_prediction(ctx: AppContext, pctx: PredictionContext) -> None:
    st.markdown('<div class="prediction-mode">', unsafe_allow_html=True)

    if ctx.fixtures_next.empty:
        st.markdown("### The 2025/26 Premier League Season is over!")
        st.markdown("No upcoming matches to predict for.")
        if st.button("View Season Recap →"):
            st.session_state.navigate_to = "Dashboard"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.sidebar.header("Prediction Panel")

    def sync_fixture():
        selected_name = st.session_state.match_selector
        ctx.fixtures_next[ctx.fixtures_next['display_name'] == selected_name].iloc[0]

    with st.sidebar:
        st.markdown('<div id="sidebar">', unsafe_allow_html=True)

        if "fixture_id_sync" not in st.session_state:
            st.session_state.fixture_id_sync = ctx.fixtures_next['id'].values[0]
        fixture_id_sync = st.session_state.fixture_id_sync

        user_id = st.text_input("Username")

        fx = st.selectbox("Select Match", options=ctx.fixtures_next['display_name'].tolist(), key="match_selector", on_change=sync_fixture)
        home_team = ctx.fixtures_next[ctx.fixtures_next['display_name'] == fx]['home'].values[0]
        away_team = ctx.fixtures_next[ctx.fixtures_next['display_name'] == fx]['away'].values[0]

        st.caption("Drag the handles to define Win, Draw, and Loss zones")
        thresholds = list(range(0, 101))

        lab1, lab2, lab3 = st.columns(3, vertical_alignment="center")
        with lab1:
            lab1_slot = st.empty()
        with lab2:
            lab2_slot = st.empty()
        with lab3:
            lab3_slot = st.empty()

        bound_1, bound_2 = st.select_slider(
            "Win | Draw | Loss",
            options=thresholds,
            value=(35, 65),
            help="First handle: Home Win threshold. Second handle: Away Win threshold."
        )

        st.markdown(f"""
            <style>
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

        p_win_pred = bound_1 / 100
        p_draw_pred = (bound_2 - bound_1) / 100
        p_loss_pred = (100 - bound_2) / 100

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

            > ⚠️ *Note: Missing a fixture applies a **Volume Penalty** to your form, treating the game as a random {ctx.null_guess_probability*100:.0f}% guess.*
            """)
        st.markdown("---")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Match Cards
    st.markdown("###### Upcoming Matches:")
    content = '<div class="scroll-container">'
    for home, away, p_win, p_draw, p_loss in ctx.fixtures_next.loc[ctx.fixtures_next.ix == 1, ['home', 'away', 'p_win', 'p_draw', 'p_loss']].values:
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

    st.markdown("##### Score (Lower is Better):")
    st.plotly_chart(
        plot_performance_moving_avg(pctx.agg_losses, pctx.penalty, window=6),
        use_container_width=True,
        config={'staticPlot': False, 'scrollZoom': False, 'displayModeBar': False, 'showAxisDragHandles': False}
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Accepted Predictions:")

        if 'predictions_user' not in st.session_state:
            st.session_state.predictions_user = pd.DataFrame(columns=[
                'prediction_id', 'user', 'fixture_id', 'p_win_home', 'p_draw_home', 'p_loss_home',
                'created_utc', 'source', 'data_load_date', 'home', 'away'
            ])

        if lock_button:
            if not user_id.strip():
                message_slot.error("👤 Please enter a username.")
            elif user_id.strip() in ctx.market_suppliers:
                message_slot.error("❌ Invalid username.")
            else:
                sync_fixture()
                prediction_data = {
                    "prediction_id": f"{fixture_id_sync}_{user_id}",
                    "user": user_id,
                    "fixture_id": fixture_id_sync,
                    "p_win_home": p_win_pred,
                    "p_draw_home": p_draw_pred,
                    "p_loss_home": p_loss_pred,
                    "created_utc": pd.Timestamp.now(tz='UTC').isoformat(),
                    "source": "user",
                    "data_load_date": pd.Timestamp.utcnow().date().isoformat()
                }
                new_prediction = pd.DataFrame([prediction_data])
                new_prediction[['home', 'away']] = new_prediction['fixture_id'].str.split("_", expand=True).iloc[:, [2, 3]]
                st.session_state.predictions_user = pd.concat(
                    [st.session_state.predictions_user, new_prediction], ignore_index=True
                )
                try:
                    db.push_prediction(prediction_data)
                    message_slot.success(f"✅ Prediction locked for {user_id}!")
                    st.balloons()
                except Exception as e:
                    message_slot.error("Encountered error, prediction not locked.")
                    st.error(f"Error connecting to BigQuery: {e}")

        predictions_combined = pd.concat([
            ctx.predictions_next.loc[
                ~ctx.predictions_next['source'].isin(ctx.market_suppliers),
                ['user', 'home', 'away', 'p_win_home', 'p_draw_home', 'p_loss_home', 'created_utc', 'prediction_id']
            ],
            st.session_state.predictions_user[['user', 'home', 'away', 'p_win_home', 'p_draw_home', 'p_loss_home', 'created_utc', 'prediction_id']]
        ]).sort_values(['created_utc'], ascending=False).drop_duplicates(['prediction_id'], keep='first') \
          .sort_values(['home', 'p_win_home', 'p_draw_home'], ascending=[True, False, True]) \
          [['user', 'home', 'away', 'p_win_home', 'p_draw_home', 'p_loss_home']]

        predictions_combined[['p_win_home', 'p_draw_home', 'p_loss_home']] = (
            predictions_combined[['p_win_home', 'p_draw_home', 'p_loss_home']] * 100
        ).round(0).astype(int).astype(str) + "%"
        predictions_combined['Prediction'] = (
            predictions_combined['p_win_home'] + "-" + predictions_combined['p_draw_home'] + "-" + predictions_combined['p_loss_home']
        )

        matches = predictions_combined[['home', 'away']].drop_duplicates()
        pred_cards_html = '<div class="preds-scroll-container">'
        for _, m in matches.iterrows():
            home, away = m['home'], m['away']
            match_preds = predictions_combined[
                (predictions_combined['home'] == home) & (predictions_combined['away'] == away)
            ]
            pred_cards_html += f'<div class="match-section"><div class="match-title">⚽ {home} vs {away}</div>'
            for _, pred in match_preds.iterrows():
                pred_cards_html += f'<div class="prediction-card"><span class="user-name">@{pred["user"]}</span><span class="prediction-value">{pred["Prediction"]}</span></div>'
            pred_cards_html += "</div>"
        pred_cards_html += "</div>"
        st.markdown(pred_cards_html, unsafe_allow_html=True)

    with c2:
        st.markdown("##### Leaderboard:")
        st.dataframe(
            pctx.leaderboard.rename(columns={'user': 'Username', 'score': 'Score (Lower is Better)'}),
            use_container_width=True, hide_index=True
        )
