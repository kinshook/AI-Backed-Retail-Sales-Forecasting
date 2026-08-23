import sys
import pandas as pd
# from db_utils.extract_df_from_db import fetch_all_tables, connect_to_db
from db_utils.utils import extract_df_from_db as ex
import numpy as np
import logging
import time


server= "your_server"
database="your_db_name"

logging.basicConfig(filename='data_cleaning.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

logger.info("Connecting to database...")
overall_start= time.time()
st= time.time()
conn_engine= ex.connect_to_db(server,database)
df = ex.fetch_all_tables(conn_engine)
end= time.time()
logger.info(f"Connection established in {end-st:.2f} seconds.")

def merge_tables(df):    
    start=time.time()
    df1= df.get("Year_2009_2010")
    df2= df.get("Year_2010_2011")

    df_merged = pd.concat([df1, df2], ignore_index=True)  #to combine the datasets
    df_merged.to_sql("merged_table", conn_engine, if_exists="replace", index=False)  # Save the merged DataFrame to SQL
    end=time.time()
    logger.info(f"Merging of tables completed in {end-start:.2f} seconds.")
    return df_merged  #1067371





# """Cleaning and Pre-Processing the merged_df"""

def merged_table_Cleaning(merged_df):
    start=time.time()
    #Dropping duplicate rows based on repeat entries in both tables
    df3= merged_df.drop_duplicates()  # Drop duplicate rows; drops 34335 rows to 1033036 rows
    
    extra_spaces_pattern = r"\s{2,}"   # 2+ spaces

    df3['Description'] = df3['Description'].str.strip() 
    df3['Description'] = df3['Description'].str.replace(extra_spaces_pattern, ' ', regex=True) 

    #replacing nulls
    desc_map = (df3.dropna(subset=['Description'])
              .groupby('StockCode')['Description']
              .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown')
              .to_dict())

    df3['Description'] = df3.apply(
        lambda r: desc_map.get(r['StockCode'], 'Unknown')
        if pd.isna(r['Description']) else r['Description'],
        axis=1
    )

#  """Cleaning the Customer ID"""
    df3['Customer ID'] = df3['Customer ID'].astype(str).str.strip() 
    print(df3['Customer ID'].isna().sum()) 
    df3['Customer ID'] = df3['Customer ID'].fillna("guest")  #...Advanced Customer ID ops later

#   """cleaning for successful orders"""
    df3['Invoice'] = df3['Invoice'].astype(str).str.strip()  
    cancelled_orders= df3['Invoice'].str.contains('^c', case=False)#'Invoice' starts with 'C' or 'c'; 19104  
    df3 = df3.drop(df3[cancelled_orders].index)  # Number of valid orders: 1013932
    df3['Quantity'] = df3['Quantity'].drop(df3[df3['Quantity'] <= 0].index)  # Remove rows where 'Quantity' is less than or equal to 0
    end=time.time()
    logging.info(f"Cleaning of merged table completed in {end-start:.2f} seconds.")
    return df3


def create_cleaned_table(cleaned_df): #check
    start=time.time()
    cleaned_df.to_sql("Retail_summary", conn_engine, if_exists="replace", index=False)  # Save the cleaned DataFrame to SQL
    print("Ingestion of cleaned_table completed successfully.")
    end=time.time()
    logging.info(f"------Ingestion of cleaned_table completed in {end-start:.2f} seconds.-------")

    return cleaned_df
overall_end= time.time()
logger.info(f"Overall data cleaning and preprocessing completed in {overall_end-overall_start:.2f} seconds.")

if __name__=="__main__":
    merged_df = merge_tables(df)
    cleaned_df = merged_table_Cleaning(merged_df)
    EDA_ready_df = create_cleaned_table(cleaned_df)
    print(f"Final cleaned DataFrame has {len(EDA_ready_df)} rows and {len(EDA_ready_df.columns)} columns.")
