import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
from datetime import timedelta


def plot_matchup(h_name, a_name, h_elo, a_elo, h_sigma, a_sigma, h_form, a_form, h_hfa, mode="Free Play"):
    h_mu = h_elo + h_hfa + h_form
    a_mu = a_elo + a_form

    limit_min = min(h_mu, a_mu) - 3 * max(h_sigma, a_sigma)
    limit_max = max(h_mu, a_mu) + 3 * max(h_sigma, a_sigma)
    x = np.linspace(limit_min, limit_max, 1000)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=stats.norm.pdf(x, h_mu, h_sigma),
        name=f"{h_name} (Home)", fill='tozeroy',
        fillcolor='rgba(218, 2, 14, 0.5)', line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=x, y=stats.norm.pdf(x, a_mu, a_sigma),
        name=f"{a_name} (Away)", fill='tozeroy',
        fillcolor='rgba(108, 171, 221, 0.5)', line=dict(color='rgba(108, 171, 221, 1)')
    ))

    template = "plotly_white" if mode == "Free Play" else "plotly_dark"
    legend_font_color = 'black' if mode == "Free Play" else 'white'

    fig.update_layout(
        autosize=True,
        height=None,
        xaxis_title="Matchday Performance (ELO)",
        template=template,
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        legend=dict(
            yanchor="top",
            xanchor="right",
            font=dict(color=legend_font_color)
        )
    )
    return fig


def plot_rolling_score(rolling_df, engine_user_name='Engine'):
    plot_df = rolling_df.copy()
    plot_df['date'] = pd.to_datetime(plot_df['date'])
    plot_df = plot_df.sort_values(['user', 'n_preds_rolling'])

    if plot_df.empty:
        return "Not enough data to plot trailing average."
    
    final_scores = plot_df[plot_df.n_preds_rolling >= 10].groupby('user').tail(1)
    human_scores = final_scores[final_scores['user'] != engine_user_name]
    top_3_users = human_scores.nsmallest(3, 'loss_per_game_rolling')['user'].tolist()

    fig = go.Figure()
    colors = {'engine': '#FF4B4B', 'top3': ['#00FFCC', '#007BFF', '#7A5FFF'], 'other': 'rgba(224,224,224,0.4)'}

    all_users = plot_df['user'].unique()
    ordered_users = (
        [u for u in all_users if u not in top_3_users and u != engine_user_name]
        + top_3_users
        + ([engine_user_name] if engine_user_name in all_users else [])
    )

    for user in ordered_users:
        user_data = plot_df[plot_df['user'] == user]
        last_row = user_data.iloc[-1]

        is_engine = (user == engine_user_name)
        is_top3 = (user in top_3_users)

        color = colors['engine'] if is_engine else (colors['top3'][top_3_users.index(user)] if is_top3 else colors['other'])
        width = 2 if is_engine else (2 if is_top3 else 1)
        label = f"🤖 {user.upper()}" if is_engine else (f"🏆 {user}" if is_top3 else user)

        fig.add_trace(go.Scatter(
            x=user_data['date'],
            y=user_data['loss_per_game_rolling'],
            mode='lines+markers' if (is_engine or is_top3) else 'lines',
            name=user,
            line=dict(color=color, width=width),
            showlegend=(is_engine or is_top3),
            hovertemplate=f"<b>{user}</b><br>Rolling Score: %{{y:.3f}}<extra></extra>"
        ))

        if is_engine or is_top3:
            fig.add_annotation(
                x=last_row['date'], y=last_row['loss_per_game_rolling'],
                text=label, showarrow=False, xanchor='left', yanchor='middle', xshift=10,
                font=dict(color=color, size=12)
            )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="black"),
        dragmode=False,
        xaxis=dict(gridcolor='rgba(136,136,136,0.2)', zeroline=False, automargin=True),
        yaxis=dict(gridcolor='rgba(136,136,136,0.2)', zeroline=False, automargin=True, title="Rolling Score"),
        margin=dict(t=50, b=50),
        hovermode="x unified",
        showlegend=False
    )
    return fig