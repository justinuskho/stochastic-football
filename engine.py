# engine.py
import numpy as np
from scipy import stats
from collections import defaultdict

import numpy as np
from scipy import stats

# ==========================================
# 1. PROBABILITY & EXPECTED POINTS LOGIC
# ==========================================

def get_dynamic_drift(home_elo, away_elo, home_sigma, away_sigma, home_hfa=0, home_form=0, away_form=0, draw_k=0.2):
    """
    Calculates expected points (mu) and probabilities based on two normal distributions.
    The 'Performance' of a team is modeled as a random variable centered at their Elo + modifiers.
    """
    # Total Rating Difference
    diff = (home_elo + home_hfa + home_form) - (away_elo + away_form)
    
    # Combined Sigma -> Logistic Scale 's'
    match_sigma = np.sqrt(home_sigma**2 + away_sigma**2 + 150**2)
    s = (match_sigma * np.sqrt(3)) / np.pi

    # Sigma-based draw margin
    draw_margin = draw_k * match_sigma
    
    # Logistic CDF Calculation
    p_win_raw = 1 / (1 + np.exp(-(diff - draw_margin) / s))
    p_loss_raw = 1 / (1 + np.exp((diff + draw_margin) / s))
    
    # This prevents the model from ever being 95% certain.
    ceiling = 0.8
    floor = 0.05
    
    # We rescale the probabilities to live within [0.05, 0.85]
    # This ves room for 'randomness' and 'draws'
    p_win = floor + (ceiling - floor) * p_win_raw
    p_loss = floor + (ceiling - floor) * p_loss_raw
    
    # Re-calculate draw to ensure they sum to 1.0
    p_draw = 1.0 - p_win - p_loss

    # Expected Points
    p_win, p_draw, p_loss = np.clip([p_win, p_draw, p_loss], 0.001, 0.999)
    mu_h = (p_win * 3) + (p_draw * 1)
    mu_a = (p_loss * 3) + (p_draw * 1)
    return mu_h, mu_a, p_win, p_draw, p_loss

# ==========================================
# 2. RATING UPDATE SYSTEM (ELO & SIGMA)
# ==========================================

def run_simulation(params, fixture, point=None, score=None, new_season=False):
    """
    Core logic for updating team ratings after a match occurs.
    Updates the 'params' dictionary in-place.
    """
    home, away = fixture
    
    if score is not None:
        home_score, away_score = score
        if home_score > away_score:
            home_point, away_point = 3, 0
        elif home_score == away_score:
            home_point, away_point = 1, 1
        elif home_score < away_score:
            home_point, away_point = 0, 3
    elif point is not None:
        home_point, away_point = point
    else:
        home_point, away_point = None, None
    
    h_elo, a_elo = params[f'elo_{home}'], params[f'elo_{away}']
    h_form, a_form = params[f'form_{home}'], params[f'form_{away}']
    h_sigma, a_sigma = params[f'sigma_{home}'], params[f'sigma_{away}']
    
    decay, k1, k2, k3, k4 = params['decay'], params['k1'], params['k2'], params['k3'], params['k4']
    refresh, convergence = params['refresh'], params['convergence']

    if new_season:
        h_sigma_adj, a_sigma_adj = h_sigma + refresh, a_sigma + refresh
        h_form, a_form = 0, 0
    else:
        h_sigma_adj, a_sigma_adj = h_sigma, a_sigma
    
    mu_h, mu_a, p_win, p_draw, p_loss = get_dynamic_drift(
        home_elo=h_elo, 
        away_elo=a_elo, 
        home_sigma=h_sigma_adj,
        away_sigma=a_sigma_adj, 
        home_hfa=params[f'hfa_{home}'], 
        home_form=h_form,
        away_form=a_form,
        draw_k=k4
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
        sigma_tau = 0.5
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

# ==========================================
# 3. UI DATA PROCESSING
# ==========================================

def get_past_5_games(team, df):
    """
    Filters history for a specific team and generates a 
    color-coded 'Form String' for Streamlit display.
    """
    df_team = df[(df.home==team)|(df.away==team)]
    
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
            rows.append([
                date,
                user,
                stats['total_loss'],
                stats['total_loss'] + (n_max - stats['n_preds']) * penalty, # Adjusted for null guess
                stats['n_preds'],
                n_max
            ])
    
    return rows, penalty