#!/usr/bin/env python

# uploading resulting lemmatized tweets in db

#SBATCH --job-name=to_sql             
#SBATCH --output=to_sql_%j.txt

#SBATCH --partition=g100_usr_prod
#SBATCH --time=05:00:00  



import pandas as pd
import sqlite3
import os

def upload_parquet_to_sql(file_path, table_name, conn):

    df = pd.read_parquet(file_path)
    df.to_sql(table_name, conn, if_exists='append', index=False)

    print(f"Uploaded data from {file_path} to {table_name}", flush=True)

conn = sqlite3.connect('/g100_work/IscrC_mental/data/database/mentalism_regioncoded_tweets.db')
table_name = "lemmatized_tweets_post_cleaning"

directory = '/g100_work/IscrC_mental/data/lemmatized_tweets/lemmatized_tweets_post_cleaning'
parquet_files = [file for file in os.listdir(directory) if file.endswith('.parquet')]
files_to_del = ['batch_number_0.parquet']
for file in files_to_del:
    if file in parquet_files:
        parquet_files.remove(file)

for file in parquet_files:
    file_name = f"/g100_work/IscrC_mental/data/lemmatized_tweets/lemmatized_tweets_post_cleaning/{file}"
    upload_parquet_to_sql(file_name, table_name, conn)

conn.close()


