import sqlalchemy as db, pandas as pd, json
from datetime import date, timedelta, datetime

def ingest_data(client):
    """Ingest client data"""
    connection = ""
    engine = db.create_engine(connection)
    metadata = db.MetaData()
    with open(f"clients/{client}.json") as (json_file):
        rdict = json.load(json_file)
    report = db.Table((rdict['report']), metadata, autoload=True, autoload_with=engine)
    query = f" SELECT * FROM jmanalo.{report} "
    df = pd.read_sql_query(query, con=(engine.connect()))
    df = df[rdict['usecols']]
    df.columns = rdict['col_names']
    df.dropna(how='any', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    maxdaycutoff = max(df['Date']) - timedelta(2)
    print('Capping max date to be 2 days less than', max(df['Date']), ', which is the max date of the file.')
    df = df[(df['Date'] <= maxdaycutoff)]
    print(f"\nData from {report} has been ingested and is now ready for cleaning...\n")
    return df