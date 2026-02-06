import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import numpy as np

# ==========================================
# 1. API CLIENT INITIALIZATION
# ==========================================

# Accessing Google Cloud credentials stored in Streamlit Secrets
# This ensures sensitive JSON keys are never hardcoded in the script.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

# Initialize the BigQuery Global Client
client = bigquery.Client(credentials=credentials)


# ==========================================
# 2. CACHED QUERY UTILITIES
# ==========================================

@st.cache_data(ttl=1200)
def query2dict(_client, query):
    """
    Executes a query and returns a list of dictionaries.
    Caching (1200s) prevents redundant API calls and saves BigQuery costs.
    """
    query_job = _client.query(query)
    rows_raw = query_job.result()
    # Convert Row objects to dicts; required for Streamlit's hashing/caching mechanism
    return [dict(row) for row in rows_raw]


@st.cache_data(ttl=1200)
def query2df(_client, query):
    """
    Helper function to wrap query results into a Pandas DataFrame.
    """
    return pd.DataFrame(query2dict(_client, query))


# ==========================================
# 3. DATA FETCHING FUNCTIONS
# ==========================================

def fetch_fixtures(n_games_before, n_games_after):
    """
    Fetches n_games_before to n_games_after and calculates match points.
    """
    df = query2df(
        client,
        f"""
        WITH
            teams AS (
                SELECT
                    team_code,
                    MAX(team) team
                FROM `project-ceb11233-5e37-4a52-b27.public.teams`
                GROUP BY 1
            ),
            season_round AS (
                SELECT
                    season,
                    round,
                    MAX(home_score IS NOT NULL) is_played
                FROM `project-ceb11233-5e37-4a52-b27.public.fixtures` 
                GROUP BY 1, 2
            ),
            season_round_rn AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (ORDER BY season, round) rn
                FROM season_round
            ),
            season_round_ix AS (
                SELECT
                    *,
                    rn - MAX(CASE WHEN is_played THEN rn ELSE 0 END) OVER () ix
                FROM season_round_rn
            )
        SELECT
            f.*,
            h.team home_team,
            a.team away_team
        FROM `project-ceb11233-5e37-4a52-b27.public.fixtures` f
        LEFT JOIN teams h
            ON f.home = h.team_code
        LEFT JOIN teams a
            ON f.away = a.team_code
        INNER JOIN season_round_ix s 
            ON f.season = s.season
                AND f.round = s.round
        WHERE
            s.ix > {n_games_before}
            AND s.ix <= {n_games_after}
        ORDER BY
            id
        """
    )
    
    # --- Feature Engineering ---
    # Create readable fixture names and score strings
    df["Fixture"] = df["home_team"] + " vs. " + df["away_team"]
    df["Score"] = df["home_score"].astype(str) + " - " + df["away_score"].astype(str)
    
    # Identify the winner for point calculation
    df["Winner"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        [df["home"], df["away"]],
        "" # Default for draws
    )
    
    # Assign Home Points (3 for Win, 1 for Draw, 0 for Loss)
    df["home_point"] = np.select(
        [df["Winner"] == df["home"], df["Winner"] == df["away"]],
        [3, 0],
        1 # Default case: Draw
    )
    
    # Assign Away Points (3 for Win, 1 for Draw, 0 for Loss)
    df["away_point"] = np.select(
        [df["Winner"] == df["home"], df["Winner"] == df["away"]],
        [0, 3],
        1 # Default case: Draw
    )
    
    return df


def fetch_params(trial="final"):
    """
    Retrieves the Elo, Sigma, and Form parameters for the simulation.
    The 'trial' parameter allows switching between different model calibrations.
    """
    return query2dict(
        client,
        f"""
        SELECT  
            *
        FROM `project-ceb11233-5e37-4a52-b27.public.params`
        WHERE
            trial = '{trial}'
        """
    )
    
def push_prediction(data):
    client = bigquery.Client()
    table_id = "project-ceb11233-5e37-4a52-b27.public.predictions"
    
    rows_to_insert = [data]
    
    errors = client.insert_rows_json(table_id, rows_to_insert)
    if errors != []:
        raise Exception(f"BigQuery Insert Errors: {errors}")