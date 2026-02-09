# engine.py
import numpy as np
from scipy import stats
from collections import defaultdict

# ==========================================
# 1. PROBABILITY & EXPECTED POINTS LOGIC
# ==========================================

def get_dynamic_drift(home_elo, away_elo, home_sigma, away_sigma, home_hfa=0, home_form=0, away_form=0, draw_margin=80):
    """
    Calculates expected points (mu) and probabilities based on two normal distributions.
    The 'Performance' of a team is modeled as a random variable centered at their Elo + modifiers.
    """
    # effective_elo = Base Elo + Home Advantage + Current Momentum (Form)
    diff = (home_elo + home_hfa + home_form) - (away_elo + away_form)
    
    # Combined variance of the match
    match_sigma = np.sqrt(home_sigma**2 + away_sigma**2)
    
    # Calculate probabilities using the Cumulative Distribution Function (CDF)
    p_loss = np.clip(stats.norm.cdf(-draw_margin, loc=diff, scale=match_sigma), 0.01, 0.99)
    p_win = np.clip(1 - stats.norm.cdf(draw_margin, loc=diff, scale=match_sigma), 0.01, 0.99)
    p_draw = np.clip(1 - p_win - p_loss, 0.01, 0.99)
    
    # Expected Points (the 'Drift')
    mu_h = (p_win * 3) + (p_draw * 1)
    mu_a = (p_loss * 3) + (p_draw * 1)
    return mu_h, mu_a, p_win, p_draw, p_loss

# ==========================================
# 2. UI DATA PROCESSING
# ==========================================

def get_past_5_games(team, df):
    """
    Filters history for a specific team and generates a 
    color-coded 'Form String' for Streamlit display.
    """
    df_team = df[(df.home_team==team)|(df.away_team==team)]
    
    # Determine W/D/L labels
    df_team["Result"] = np.select(
        [df_team["Winner"]==team, df_team["Winner"]==""],
        ["W", "D"], "L"
    )
    
    # Generate Streamlit-flavored Markdown for colored form circles
    form_str = ""
    for res in df_team["Result"].values:
        color = "green" if res == "W" else ("red" if res == "L" else "orange")
        form_str += f":{color}[{res}] "
    
    return df_team, form_str

# ==========================================
# 3. RATING UPDATE SYSTEM (ELO & SIGMA)
# ==========================================

def run_simulation(params, fixture, point, new_season=False):
    """
    Core logic for updating team ratings after a match occurs.
    Updates the 'params' dictionary in-place.
    """
    home, away = fixture
    home_point, away_point = point
    
    h_elo, a_elo = params[f'elo_{home}'], params[f'elo_{away}']
    h_form, a_form = params[f'form_{home}'], params[f'form_{away}']
    h_sigma, a_sigma = params[f'sigma_{home}'], params[f'sigma_{away}']
    
    # Use trial params
    decay, k1, k2, k3 = params['decay'], params['k1'], params['k2'], params['k3']
    refresh, convergence = params['refresh'], params['convergence']

    if new_season:
        h_sigma_adj, a_sigma_adj = h_sigma + refresh, a_sigma + refresh
        h_form, a_form = 0, 0
    else:
        h_sigma_adj, a_sigma_adj = h_sigma, a_sigma
    
    mu_h, mu_a, p_win, p_draw, p_loss = get_dynamic_drift(
        h_elo, a_elo, h_sigma_adj, a_sigma_adj,
        params[f'hfa_{home}'], h_form, a_form
    )
    
    # Update params based on match result
    if home_point is not None:
        h_surprise = home_point - mu_h
        a_surprise = away_point - mu_a
        
        # 1. Update Form
        params[f'form_{home}'] = (h_form * decay) + (h_surprise * k2)
        params[f'form_{away}'] = (a_form * decay) + (-h_surprise * k2) # Away surprise is inverse

        # 2. Sigma Update (Use ABS surprise)
        sigma_floor = 80
        sigma_tau = 0.3
        # Shock results increase uncertainty; predictable ones decrease it
        params[f'sigma_{home}'] = max(sigma_floor, (h_sigma_adj * convergence) + ((abs(h_surprise)-sigma_tau) * k3))  
        params[f'sigma_{away}'] = max(sigma_floor, (a_sigma_adj * convergence) + ((abs(a_surprise)-sigma_tau) * k3)) 

        # 3. Elo Update (Zero-Sum)
        # elo_shift = h_surprise * k1
        h_learning_rate = h_sigma_adj / 150  # Normalize around a 'typical' sigma
        elo_shift = h_surprise * k1 * h_learning_rate
        params[f'elo_{home}'] += elo_shift
        params[f'elo_{away}'] -= elo_shift

    # 4. Log-Loss
    if home_point == 3: match_error = -np.log(p_win)
    elif home_point == 1: match_error = -np.log(p_draw)
    elif home_point == 0: match_error = -np.log(p_loss)
    else: match_error = None
    
    return match_error, params, (mu_h, mu_a, p_win, p_draw, p_loss)

def calculate_prediction_losses(predictions_list, results_dict):
    """
    Calculates Log-Loss for user predictions based on match results.
    
    Args:
        predictions_data: List of lists
                         [fixture_id, user, p_win, p_draw, p_loss]
        results_data: List of dicts
                     {fixture_id: {home_point: 3,  date: 2025-05-01}}
                     
    Returns:
        List of lists [fixture_id, user, loss]
    """
    if len(predictions_list)==0:
        return []
    
    rows = []
    for fixture_id, user, p_win, p_draw, p_loss in predictions_list:
        # 3. Safety clip to avoid log(0) which results in infinity
        p_win, p_draw, p_loss = np.clip([p_win, p_draw, p_loss], 1e-6, 1 - 1e-6)
        
        # 4. Log-Loss
        home_point = results_dict[fixture_id]['home_point']
        date = results_dict[fixture_id]['date']
        if home_point == 3: 
            match_loss = -np.log(p_win)
        elif home_point == 1: 
            match_loss = -np.log(p_draw)
        else: 
            match_loss = -np.log(p_loss)
        
        rows.append([fixture_id, date, user, match_loss])

    # 5. Return only the requested columns
    return rows

def calculate_aggregate_losses(prediction_losses_list, null_guess_probability=0.30):
    """
    Calculates Log-Loss for user predictions based on match results.
    
    Args:
        predictions_data: List of prediction losses
                         [fixture_id, user, loss]
                     
    Returns:
        List of lists [gameweek, user, aggregate_losses]
    """
    if len(prediction_losses_list)==0:
        return []
    
    date_user_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for fixture_id, date, user, loss in prediction_losses_list:
        date_user_stats[date][user]['n_preds'] += 1
        date_user_stats[date][user]['total_loss'] += loss
    
    rows = []
    penalty = -np.log(null_guess_probability)
    for date, user_stats in date_user_stats.items():
        n_max = max(stats['n_preds'] for user, stats in user_stats.items())
        for user, stats in user_stats.items():
            stats['weighted_loss'] = (stats['total_loss'] + (n_max - int(stats['n_preds']))*penalty)/ n_max
            rows.append([
                date,
                user,
                stats['total_loss'],
                stats['n_preds'],
                stats['weighted_loss']
            ])
    
    return rows, penalty