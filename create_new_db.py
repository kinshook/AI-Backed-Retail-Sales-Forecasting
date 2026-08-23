from sqlalchemy import create_engine, text
from urllib.parse import quote_plus


server = input()                           
DB = input("Enter the database name you wanto create: ")



def DB_Create(server_name, DB_name):
    master_conn = quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server_name};"
    "DATABASE=master;"
    "Trusted_Connection=yes;"
    )
    master_engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={master_conn}",
    isolation_level="AUTOCOMMIT"
    )
    with master_engine.connect() as conn:
        conn.execute(text(f"IF DB_ID('{DB_name}') IS NULL CREATE DATABASE [{DB_name}]"))
    print(f"Connection to {DB_name} established!")


# DB_Create(server, DB)
