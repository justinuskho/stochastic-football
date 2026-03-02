import requests
import pandas as pd
from google.cloud import bigquery
import functions_framework
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
import database as db

def fixtures_from_fpl():
    def get_fpl_df():
        # 1. Get Team Names Mapping
        meta_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        meta_res = requests.get(meta_url, verify=False)
        teams_data = meta_res.json()['teams']

        # 2. Get Fixtures
        fix_url = "https://fantasy.premierleague.com/api/fixtures/"
        fix_res = requests.get(fix_url, verify=False)
        all_fixtures = fix_res.json()

        df = pd.DataFrame(all_fixtures)

        return teams_data, df

    # Run it
    teams, df = get_fpl_df()

    # Map ID -> Full Name and ID -> Short Name (ARS, WOL, etc.)
    code_map = {t['id']: t['short_name'] for t in teams}
    df['home'] = df['team_h'].map(code_map)
    df['away'] = df['team_a'].map(code_map)
    df['home_score'] = df['team_h_score']
    df['away_score'] = df['team_a_score']
    df['season'] = '2025-2026'

    # Convert kickoff to date
    df['date'] = pd.to_datetime(df['kickoff_time']).dt.date
    df = df[df['event'].notnull()]
    df['round'] = df['event'].astype(int)

    # Create your BigQuery fixture_id
    df['id'] = (
        df['season']+ "_" + 
        df['round'].astype(str).str.pad(2, side='left', fillchar='0') + "_" + 
        df['home'] + "_" + 
        df['away']
    )
    return df

def upsert_fixtures_to_bigquery(df, project_id, dataset_id, table_name, schema):
    # 1. Define Table IDs
    target_table = f"{project_id}.{dataset_id}.{table_name}"
    staging_table = f"{project_id}.{dataset_id}.{table_name}_staging"

    # 2. Upload the new data to a TEMPORARY staging table (Truncate here is fine)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", schema=schema)
    client.load_table_from_dataframe(df, staging_table, job_config=job_config).result()

    # 3. The MERGE SQL Statement
    # This says: If ID matches, update everything. If not, insert it.
    merge_query = f"""
    MERGE `{target_table}` T
    USING `{staging_table}` S
    ON T.season = S.season AND T.home = S.home AND T.away = S.away
    WHEN MATCHED THEN
      UPDATE SET 
        date = S.date, 
        round = S.round, 
        home_score = S.home_score, 
        away_score = S.away_score
    WHEN NOT MATCHED THEN
      INSERT (id, season, round, date, home, home_score, away_score)
      VALUES (id, season, round, date, home, home_score, away_score)
    """

    try:
        # Run the merge
        client.query(merge_query).result()
        # Clean up staging table
        client.delete_table(staging_table, not_found_ok=True)
        print(f"Successfully upserted data into {target_table}")
    except Exception as e:
        print(f"Merge failed: {e}")

client = db.get_bq_client()

fx_schema=[
        bigquery.SchemaField("id", "STRING"),
        bigquery.SchemaField("season", "STRING"),
        bigquery.SchemaField("round", "INTEGER"),
        bigquery.SchemaField("date", "DATE"),
        bigquery.SchemaField("home", "STRING"),
        bigquery.SchemaField("home_score", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("away", "STRING"),
        bigquery.SchemaField("away_score", "INTEGER", mode="NULLABLE"),
    ]

@functions_framework.http
def run_sync(request):
    try:
        to_upsert = fixtures_from_fpl()
        upsert_fixtures_to_bigquery(
            df=to_upsert[['id', 'season', 'round', 'date', 'home', 'home_score', 'away', 'away_score']], 
            project_id="project-ceb11233-5e37-4a52-b27", 
            dataset_id="public", 
            table_name="fixtures", 
            schema=fx_schema
        )
        return "Sync Successful", 200
    except Exception as e:
        print(f"Error: {e}")
        return str(e), 500
    
if __name__ == "__main__":
    print("Running sync locally...")
    to_upsert = fixtures_from_fpl()
    upsert_fixtures_to_bigquery(
        df=to_upsert[['id', 'season', 'round', 'date', 'home', 'home_score', 'away', 'away_score']], 
        project_id="project-ceb11233-5e37-4a52-b27", 
        dataset_id="public", 
        table_name="fixtures", 
        schema=fx_schema
    )