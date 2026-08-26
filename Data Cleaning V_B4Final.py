# import sys
import pandas as pd
# from db_utils.extract_df_from_db import fetch_all_tables, connect_to_db
from db_utils.utils import extract_df_from_db as ex
from sqlalchemy import create_engine,text
import numpy as np
import logging
import time
import os




def merge_tables(df):
    merged= """SELECT * FROM Year_2009_2010
    UNION
    SELECT * FROM Year_2010_2011
    """
    merged_df= pd.read_sql_query(merged, conn_engine)
    merged_df.to_sql("merged_table", conn_engine,if_exists='replace', index=False)
    return merged_df    

def merged_table_Cleaning(merged_df):
    start = time.time()
    df3 = merged_df.copy()

    # 1. BASIC CLEANING (applies to ALL rows)
    extra_spaces_pattern = r"\s{2,}"
    df3['Description'] = df3['Description'].str.strip()
    df3['Description'] = df3['Description'].str.replace(extra_spaces_pattern, ' ', regex=True)

    desc_map = (df3.dropna(subset=['Description'])
                .groupby('StockCode')['Description']
                .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown')
                .to_dict())

    df3['Description'] = df3.apply(
        lambda r: desc_map.get(r['StockCode'], 'Unknown')
        if pd.isna(r['Description']) else r['Description'],
        axis=1)

    logger.info(f"Null Customer IDs before cleaning: {df3['Customer ID'].isna().sum()}")
    df3['Customer ID'] = (
        df3['Customer ID']
        .astype('Int64')
        .astype(str)
        .str.replace('<NA>', 'guest', regex=False)
        .str.replace(r'\.0', '', regex=True)
        .str.strip()
    )
    df3['Customer ID'] = df3['Customer ID'].fillna('guest')
    logger.info(f"Null Customer IDs after cleaning: {df3['Customer ID'].isna().sum()}")

    # 2. ADD FLAGS (before any splitting)
    df3['Revenue']      = df3['Quantity'] * df3['Price']
    df3['Is_cancelled'] = df3['Invoice'].str.contains('^c', case=False, na=False)
    df3['Is_return']    = df3['Quantity'] < 0          
    df3['Year']         = df3['InvoiceDate'].dt.year   
    df3['Month']        = df3['InvoiceDate'].dt.month
    df3['DayOfWeek']    = df3['InvoiceDate'].dt.day_name()

    # 3. SPLIT -- Before dropping non-products
    non_product = ['POST', 'D', 'M', 'BANK CHARGES', 'PADS',
                   'DOT', 'AMAZONFEE', 'S', 'CRUK', 'C2']

    # adjustment table — non-product rows only, already cleaned above
    adjust_df = df3[df3['StockCode'].isin(non_product)].copy()

    # sales table — drop non-products, keep only valid sales
    sales_df = df3[~df3['StockCode'].isin(non_product)].copy()
    sales_df = sales_df[sales_df['Price'] > 0]
    sales_df = sales_df[sales_df['Quantity'] > 0]

    end = time.time()
    logger.info(f"Cleaning completed in {end-start:.2f}s | "
                f"Sales rows: {len(sales_df)} | "
                f"Adjustment rows: {len(adjust_df)}")

    return sales_df, adjust_df


def create_retail_summary(sales_df, adjust_df, engine):
    """
    True profit = sum of sales revenue - the absolute value
    of all adjustment transactions (postage, bank charges etc.)"""

    start = time.time()

    # total deductions from adjustment table
    # these are charges the business paid — postage, bank fees etc.
    # Revenue on these rows is negative (negative qty or negative price)
    # take abs() so deduction is a positive number to subtract
    total_deductions = adjust_df['Revenue'].sum()  # already negative — will subtract correctly

    # per-order deduction: join adjust_df to sales_df on Invoice where possible
    # for charges with no matching invoice, treat as overhead deducted from total
    invoice_charges = (
        adjust_df.groupby('Invoice')['Revenue']
        .sum()
        .reset_index()
        .rename(columns={'Revenue': 'Charges_Pounds'})
    )

    retail_summary = pd.merge(sales_df,invoice_charges, on='Invoice', how='left')
    retail_summary['Charges_Pounds'] = retail_summary['Charges_Pounds'].fillna(0)

    # true profit per row = revenue minus any charges on same invoice
    retail_summary['True_Profit'] = (
        retail_summary['Revenue'] + retail_summary['Charges_Pounds']
        # Charges_INR is negative — adding a negative = subtracting
    )

    with engine.begin() as conn:

        # drop tables first if they exist — avoids reflection conflict
        conn.execute(text("""
            IF OBJECT_ID('dbo.Retail_Summary', 'U') IS NOT NULL
                DROP TABLE dbo.Retail_Summary
        """))
        conn.execute(text("""
            IF OBJECT_ID('dbo.Revenue_Adjustments', 'U') IS NOT NULL
                DROP TABLE dbo.Revenue_Adjustments
        """))

    # now write — tables are guaranteed not to exist, no reflection needed
    retail_summary.to_sql(
        "Retail_Summary", engine,
        if_exists="replace", index=False,
        chunksize=1000, schema="dbo"
    )
    adjust_df.to_sql(
        "Revenue_Adjustments", engine,
        if_exists="replace", index=False,
        chunksize=1000, schema="dbo"
    )





    end = time.time()
    logger.info(f"Retail_Summary ingested: {len(retail_summary)} rows | "
                f"Revenue_Adjustments ingested: {len(adjust_df)} rows | "
                f"Total charges deducted: £{total_deductions:,.2f} | "
                f"Completed in {end-start:.2f}s")

    return retail_summary



if __name__=="__main__":
    pd.set_option("display.width", os.get_terminal_size().columns) #terminal view adjustments
    pd.set_option("display.max_columns", None)

    logging.basicConfig(filename=r'C:\My Python Files\All_utils\db_utils\utils\data_cleaning.log',filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') #overwrites the files each time the script runs
    logger = logging.getLogger()

    server="localhost"
    database="retail_proj"
    logger.info("Connecting to database...")
    overall_start= time.time()

    st= time.time()
    conn_engine= ex.connect_to_db(server,database)
    df = ex.fetch_all_tables(conn_engine)
    end= time.time()
    logger.info(f"Connection established and {len(df.keys())} tables fetched in {end-st:.2f} seconds.")

    
    merged_df = merge_tables(df)
    sales_df, adjust_df = merged_table_Cleaning(merged_df)
    # print("check for cleaning completion")

    # print("\n── sales_df ──")
    # print(f"Shape:    {sales_df.shape}")
    # print(f"Columns:  {sales_df.columns.tolist()}")
    # print(f"Nulls:\n{sales_df.isnull().sum()}")
    # print(sales_df.head(2))

    # print("\n── adjust_df ──")
    # print(f"Shape:    {adjust_df.shape}")
    # print(f"Columns:  {adjust_df.columns.tolist()}")
    # print(f"Nulls:\n{adjust_df.isnull().sum()}")
    # print(adjust_df.head(2))

    # print("\n── Revenue column check ──")
    # print(f"Revenue in sales_df:  {'Revenue' in sales_df.columns}")
    # print(f"Revenue in adjust_df: {'Revenue' in adjust_df.columns}")
    # print(f"Invoice nulls sales:  {sales_df['Invoice'].isna().sum()}")
    # print(f"Invoice nulls adjust: {adjust_df['Invoice'].isna().sum()}")


    retail_df= create_retail_summary(sales_df,adjust_df,conn_engine)

    retail_summary = create_retail_summary(sales_df, adjust_df, conn_engine)
    # print(f"\nRetail_Summary rows:      {len(retail_summary)}")
    # print(f"Revenue_Adjustments rows: {len(adjust_df)}")
    # print(retail_summary[['Invoice','Revenue',
    #                        'Charges_INR','True_Profit']].head(10))


    print("check for completion")
    #print(revenue_adjustments(merged_df)) #22496
    # print(f"Final cleaned DataFrame has {len(EDA_ready_df)} rows and {len(EDA_ready_df.columns)} columns.")

    overall_end= time.time()
    logger.info(f"Overall data cleaning and preprocessing completed in {overall_end-overall_start:.2f} seconds.")
