import pandas as pd
import sqlite3

def load_data(database_path):
    conn = sqlite3.connect(database_path)
    query = "SELECT * FROM subscription_data;"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def preprocess_data(df):
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df.fillna(0, inplace=True)
    return df
