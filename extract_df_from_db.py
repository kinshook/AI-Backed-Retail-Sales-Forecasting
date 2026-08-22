import pandas as pd
import pyodbc
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus


def connect_to_db(server_name, db):
    try:
        conn_str = quote_plus(
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={server_name};"
            f"DATABASE={db};"
            "Trusted_Connection=yes;"
            "Encrypt=optional;"
        )
        engine = create_engine(
            f"mssql+pyodbc:///?odbc_connect={conn_str}",
            isolation_level="AUTOCOMMIT"
        )
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT DB_ID('{db}')")).scalar()
            if result is None:
                raise ValueError("Database not found")
    except pyodbc.InterfaceError:
        print(f"Server not found")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

    print(f"Connection established to {db}")
    return engine


def fetch_all_tables(engine):
    # Get list of table names
    tables_df = pd.read_sql("SELECT name FROM sys.tables", engine)
    table_names = tables_df['name']
    # print(tables_df,"\n",table_names)

    # Dictionary to hold DataFrames
    dfs = {}
    for table in table_names:
        dfs[table] = pd.read_sql_query(f"SELECT * FROM {table}", engine)
        print(f"Loaded table: {table}, rows: {len(dfs[table])}")

    return dfs


#
#

#     print(all_dfs.get_items().head())
