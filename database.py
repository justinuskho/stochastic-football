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

@st.cache_data(ttl=360)
def query2dict(_client, query):
    """
    Executes a query and returns a list of dictionaries.
    Caching (1200s) prevents redundant API calls and saves BigQuery costs.
    """
    query_job = _client.query(query)
    rows_raw = query_job.result()
    # Convert Row objects to dicts; required for Streamlit's hashing/caching mechanism
    return [dict(row) for row in rows_raw]


@st.cache_data(ttl=360)
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
                team_fixtures AS (
                    SELECT
                        home AS team,
                        id AS fixture_id,
                        home_score IS NOT NULL AS is_played,
                        date
                    FROM `project-ceb11233-5e37-4a52-b27.public.fixtures`
                    WHERE
                        season >= '2025'

                    UNION DISTINCT

                    SELECT
                        away AS team,
                        id AS fixture_id,
                        home_score IS NOT NULL AS is_played,
                        date
                    FROM `project-ceb11233-5e37-4a52-b27.public.fixtures`
                    WHERE
                        season >= '2025'
                ),
                team_fixtures_rn AS (
                    SELECT  
                        *,
                        ROW_NUMBER() OVER (PARTITION BY team ORDER BY date) rn
                    FROM team_fixtures        
                ),
                team_fixtures_ix AS (
                    SELECT  
                        *,
                        rn - MAX(CASE WHEN is_played THEN rn END) OVER (PARTITION BY team) ix
                    FROM team_fixtures_rn
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
            WHERE
                season >= '2025'
                AND f.id IN (SELECT DISTINCT fixture_id FROM team_fixtures_ix WHERE ix > {n_games_before} AND ix <= {n_games_after})
            ORDER BY
                date
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


def fetch_params(at_start_of="GW25"):
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
            at_start_of = '{at_start_of}'
        """
    )
    
def fetch_predictions(fixture_id_list):
    """
    Retrieves the Elo, Sigma, and Form parameters for the simulation.
    The 'trial' parameter allows switching between different model calibrations.
    """
    
    return query2df(
        client,
        f"""
        SELECT  
            user,
            source,
            fixture_id,
            SPLIT(fixture_id, '_')[SAFE_OFFSET(2)] AS home,
            SPLIT(fixture_id, '_')[SAFE_OFFSET(3)] AS away,
            MAX_BY(prediction_id, created_utc) prediction_id,
            MAX(created_utc) created_utc,
            MAX_BY(p_win_home, created_utc) p_win_home,
            MAX_BY(p_draw_home, created_utc) p_draw_home,
            MAX_BY(p_loss_home, created_utc) p_loss_home
        FROM `project-ceb11233-5e37-4a52-b27.public.predictions`
        WHERE
            fixture_id IN ({", ".join([f"'{f}'" for f in fixture_id_list])})
            AND p_win_home >= 0
            AND p_draw_home >= 0
            AND p_loss_home >= 0
        GROUP BY 1, 2, 3, 4, 5
        """
    )
    
def push_prediction(data):
    table_id = "project-ceb11233-5e37-4a52-b27.public.predictions"
    
    rows_to_insert = [data]
    
    errors = client.insert_rows_json(table_id, rows_to_insert)
    if errors != []:
        raise Exception(f"BigQuery Insert Errors: {errors}")