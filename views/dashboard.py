import streamlit as st
import database as db
from views.context import AppContext
from views.charts import plot_rolling_score


def render_dashboard(ctx: AppContext) -> None:
    st.markdown('<div class="dashboard-mode">', unsafe_allow_html=True)

    lb_raw = db.fetch_leaderboard(season="2025-2026", n_preds_min=0)
    all_benchmarks = ctx.market_suppliers | {'engine'}

    human_lb_full = (
        lb_raw[lb_raw['is_user'] & (lb_raw['n_preds'] >= 10)]
        .sort_values('rank_user')[['user', 'n_preds', 'loss_per_game']]
    )
    human_lb_full['rank'] = human_lb_full.reset_index().index + 1
    human_lb_full['loss_per_game'] = human_lb_full['loss_per_game'].round(3)

    with st.sidebar:
        st.markdown('<div id="sidebar">', unsafe_allow_html=True)
        if not ctx.fixtures_next.empty:
            st.markdown("### Season in Progress")
            st.markdown("Showing results to date:")
        else:
            st.sidebar.header("Dashboard Mode")
            st.markdown("### Season Complete")
            st.markdown("The 2025/26 Premier League season has ended. Here's the final standings:")
        for _, row in human_lb_full.head(3).iterrows():
            medal = ["🥇", "🥈", "🥉"][int(row['rank']) - 1]
            st.markdown(f"{medal} **{row['user']}** — {row['loss_per_game']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

    title = "⚽ 2025/26 Premier League Season Final Results" if ctx.fixtures_next.empty else "⚽ 2025/26 Premier League Season — Results So Far"
    st.markdown(f"## {title}")
    st.markdown("---")

    # A: Final Leaderboard
    st.markdown("##### Final Leaderboard (Human Players):")
    st.markdown("Only players with >=10 predictions are eligible for the leaderboard.")
    st.dataframe(human_lb_full[['rank', 'user', 'n_preds', 'loss_per_game']].rename(columns={'user': 'Username', 'n_preds': 'Predictions', 'loss_per_game': 'Score (Lower is Better)', 'rank': 'Rank'}), use_container_width=True, hide_index=True)
    st.markdown("---")

    # B: Performance chart (humans only)
    st.markdown("##### Season Performance:")
    st.markdown("This chart shows how your rolling average log-loss evolved over the season compared to other players. The lower the score, the better your predictions were! The top 3 human players are highlighted, along with the benchmark model.")
    rolling_df = db.fetch_rolling_score()
    rolling_humans = rolling_df[~rolling_df['user'].isin(all_benchmarks)]

    if not rolling_humans.empty:
        st.plotly_chart(
            plot_rolling_score(rolling_humans),
            use_container_width=True,
            config={'staticPlot': False, 'scrollZoom': False, 'displayModeBar': False, 'showAxisDragHandles': False}
        )
    st.markdown("---")

    # C: Human vs Benchmarks
    st.markdown("##### You vs The Benchmarks:")
    benchmarks_display = {
        'engine': 'Stochastic Model', 'Engine': 'Stochastic Model', 
        'google': 'Google', 'Google': 'Google',
        'opta_analyst': 'Opta Analyst', 'OptaAnalyst': 'Opta Analyst'
    }

    bench_rows = (
        lb_raw[lb_raw['user'].isin(benchmarks_display)]
        .copy()
        .assign(Source=lambda df: df['user'].map(benchmarks_display))
        .drop_duplicates('Source')
        .sort_values('loss_per_game')
        [['Source', 'n_preds', 'loss_per_game']]
    )

    model_score = bench_rows[bench_rows['Source'] == 'Stochastic Model']['loss_per_game'].values
    human_rows = lb_raw[lb_raw['is_user']].sort_values('loss_per_game')[['user', 'n_preds', 'loss_per_game']].copy()
    if len(model_score):
        human_rows['vs Model'] = human_rows['loss_per_game'].apply(
            lambda x: '✅ Beats model' if x < model_score[0] else '❌ Behind model'
        )

    col_h, col_b = st.columns(2)
    with col_h:
        st.markdown("###### Players")
        st.dataframe(human_rows.rename(columns={'user': 'Username', 'n_preds': 'Predictions', 'loss_per_game': 'Score (Lower is Better)'}), use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("###### Benchmarks")
        st.dataframe(bench_rows.rename(columns={'n_preds': 'Predictions', 'loss_per_game': 'Score (Lower is Better)'}), use_container_width=True, hide_index=True)
    st.markdown("---")

    # D: Best & Worst Predictions
    code_to_name = {v: k for k, v in ctx.TEAMS_LOOKUP.items()}

    def add_fixture_col(df):
        df = df.copy()
        df.insert(0, 'fixture', df['home'].map(code_to_name) + ' vs ' + df['away'].map(code_to_name))
        return df.drop(columns=['home', 'away'])

    best5 = add_fixture_col(db.fetch_prediction_losses_n(season="2025-2026", best_or_worst="best", n=5))
    worst5 = add_fixture_col(db.fetch_prediction_losses_n(season="2025-2026", best_or_worst="worst", n=5))

    cb1, cb2 = st.columns(2)
    with cb1:
        st.markdown("##### 🏆 Best Calls:")
        st.dataframe(best5[['fixture', 'gameweek', 'result', 'user', 'prediction', 'score']].rename(columns={'fixture': 'Fixture', 'gameweek': 'GW', 'result': 'Result', 'user': 'User', 'prediction': 'Prediction', 'score': 'Score (Lower is Better)'}), use_container_width=True, hide_index=True)
    with cb2:
        st.markdown("##### 💀 Worst Calls:")
        st.dataframe(worst5[['fixture', 'gameweek', 'result', 'user', 'prediction', 'score']].rename(columns={'fixture': 'Fixture', 'gameweek': 'GW', 'result': 'Result', 'user': 'User', 'prediction': 'Prediction', 'score': 'Score (Lower is Better)'}), use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)
