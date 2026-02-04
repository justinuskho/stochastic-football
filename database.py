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

def fetch_fixtures():
    """
    Fetches the last 5 rounds of fixtures and calculates match points.
    """
    df = query2df(
        client,
        """
        WITH
            last_5_games AS (
                SELECT DISTINCT 
                    season, 
                    round 
                FROM `project-ceb11233-5e37-4a52-b27.public.fixtures_epl` 
                WHERE home_score IS NOT NULL 
                ORDER BY 1 DESC, 2 DESC 
                LIMIT 5
            )
        SELECT  
            f.*
        FROM `project-ceb11233-5e37-4a52-b27.public.fixtures_epl` f
        INNER JOIN last_5_games f5
            ON f.season = f5.season
                AND f.round = f5.round
        ORDER BY
            id
        """
    )
    
    # --- Feature Engineering ---
    # Create readable fixture names and score strings
    df["Fixture"] = df["home"] + " vs. " + df["away"]
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
        FROM `project-ceb11233-5e37-4a52-b27.public.params_epl`
        WHERE
            trial = '{trial}'
        """
    )