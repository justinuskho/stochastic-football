import pandas as pd
import streamlit as st
from views.context import AppContext, PredictionContext
from views.charts import plot_performance_moving_avg


def render_dashboard(ctx: AppContext, pctx: PredictionContext) -> None:
    st.markdown('<div class="dashboard-mode">', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div id="sidebar">', unsafe_allow_html=True)
        if not ctx.fixtures_next.empty:
            st.markdown("### Season in Progress")
            st.markdown("Showing results to date:")
        else:
            st.sidebar.header("Dashboard Mode")
            st.markdown("### Season Complete")
            st.markdown("The 2025/26 Premier League season has ended. Here's the final standings:")
        human_lb = pctx.leaderboard[~pctx.leaderboard['user'].isin(pctx.all_benchmarks)].head(3)
        for rank, (_, row) in enumerate(human_lb.iterrows(), 1):
            medal = ["🥇", "🥈", "🥉"][rank - 1]
            st.markdown(f"{medal} **{row['user']}** - {row['score']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    title = "⚽ 2025/26 Premier League Season Final Results" if ctx.fixtures_next.empty else "⚽ 2025/26 Premier League Season — Results So Far"
    st.markdown(f"## {title}")
    st.markdown("---")

    # A: Final Leaderboard
    st.markdown("##### Final Leaderboard (Human Players):")
    human_lb_full = pctx.leaderboard[~pctx.leaderboard['user'].isin(pctx.all_benchmarks)].copy()
    human_lb_full.insert(0, 'Rank', range(1, len(human_lb_full) + 1))
    st.dataframe(
        human_lb_full.rename(columns={'user': 'Username', 'score': 'Score (Lower is Better)'}),
        use_container_width=True, hide_index=True
    )
    st.markdown("---")

    # B: Performance chart (humans only)
    st.markdown("##### Season Performance:")
    agg_losses_humans = pctx.agg_losses[pctx.agg_losses['user'].isin(pctx.human_users)]
    if not agg_losses_humans.empty:
        st.plotly_chart(
            plot_performance_moving_avg(agg_losses_humans, pctx.penalty, window=6),
            use_container_width=True,
            config={'staticPlot': False, 'scrollZoom': False, 'displayModeBar': False, 'showAxisDragHandles': False}
        )
    st.markdown("---")

    # C: Human vs Benchmarks
    st.markdown("##### You vs The Benchmarks:")
    pl_df = pd.DataFrame(pctx.prediction_losses, columns=['fixture_id', 'date', 'user', 'loss'])
    mean_loss = pl_df.groupby('user')['loss'].mean().round(3).reset_index().rename(columns={'user': 'Source', 'loss': 'Avg Log-Loss'})
    benchmarks_display = {
        'engine': 'Stochastic Model', 'google': 'Google', 'Google': 'Google',
        'opta_analyst': 'Opta Analyst', 'OptaAnalyst': 'Opta Analyst'
    }

    bench_rows = mean_loss[mean_loss['Source'].isin(benchmarks_display)].copy()
    bench_rows['Source'] = bench_rows['Source'].map(benchmarks_display)
    bench_rows = bench_rows.drop_duplicates('Source').sort_values('Avg Log-Loss')

    human_rows = mean_loss[~mean_loss['Source'].isin(pctx.all_benchmarks)].sort_values('Avg Log-Loss').copy()
    model_score = bench_rows[bench_rows['Source'] == 'Stochastic Model']['Avg Log-Loss'].values
    if len(model_score):
        human_rows['vs Model'] = human_rows['Avg Log-Loss'].apply(
            lambda x: '✅ Beats model' if x < model_score[0] else '❌ Behind model'
        )

    col_h, col_b = st.columns(2)
    with col_h:
        st.markdown("###### Human Players")
        st.dataframe(human_rows.rename(columns={'Source': 'Username'}), use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("###### Benchmarks")
        st.dataframe(bench_rows, use_container_width=True, hide_index=True)
    st.markdown("---")

    # D: Best & Worst Predictions
    fixture_meta = ctx.fixtures[['id', 'round', 'Fixture', 'home_point']].copy()
    fixture_meta['Result'] = fixture_meta['home_point'].map({3: 'Home Win', 1: 'Draw', 0: 'Away Win'})

    pl_humans = pl_df[~pl_df['user'].isin(pctx.all_benchmarks)].copy()
    pl_humans = pl_humans.merge(fixture_meta.rename(columns={'id': 'fixture_id'}), on='fixture_id', how='left')

    preds_lookup = ctx.predictions.sort_values('created_utc', ascending=False).drop_duplicates('prediction_id', keep='first')[
        ['fixture_id', 'user', 'p_win_home', 'p_draw_home', 'p_loss_home']
    ].copy()
    preds_lookup['pred_str'] = (preds_lookup[['p_win_home', 'p_draw_home', 'p_loss_home']] * 100).round(0).astype(int).astype(str).apply(
        lambda r: f"{r['p_win_home']}–{r['p_draw_home']}–{r['p_loss_home']}", axis=1
    )
    pl_humans = pl_humans.merge(preds_lookup[['fixture_id', 'user', 'pred_str']], on=['fixture_id', 'user'], how='left')

    best5 = pl_humans.nsmallest(5, 'loss')[['Fixture', 'round', 'Result', 'user', 'pred_str', 'loss']].rename(
        columns={'Fixture': 'Match', 'round': 'GW', 'user': 'User', 'pred_str': 'Prediction', 'loss': 'Score'})
    worst5 = pl_humans.nlargest(5, 'loss')[['Fixture', 'round', 'Result', 'user', 'pred_str', 'loss']].rename(
        columns={'Fixture': 'Match', 'round': 'GW', 'user': 'User', 'pred_str': 'Prediction', 'loss': 'Score'})

    cb1, cb2 = st.columns(2)
    with cb1:
        st.markdown("##### 🏆 Best Calls:")
        st.dataframe(best5, use_container_width=True, hide_index=True)
    with cb2:
        st.markdown("##### 💀 Worst Calls:")
        st.dataframe(worst5, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)
