#!/usr/bin/env python

# Deleting empty files in result of lemmatization

import pandas as pd
import os
import pyarrow.parquet as pq

# Define the directory containing the parquet files. Replace this with the path to your parquet files.
directory = '/g100_work/IscrC_mental/data/lemmatized_tweets/lemmatized_tweets_post_cleaning'

# List all parquet files in the directory
parquet_files = [f for f in os.listdir(directory) if f.endswith('.parquet')]

# Function to check if a parquet file is empty
def is_parquet_file_empty(file_path):
    try:
        table = pq.read_table(file_path)
        return table.num_rows == 0
    except Exception as e:
        # If there is an error reading the file, it might be corrupt or not a parquet file
        print(f"Error reading {file_path}: {e}")
        return False  # Or True, based on how you want to handle errors

# Check each parquet file, delete the empty ones and print the names of the files deleted
for file in parquet_files:
    file_path = os.path.join(directory, file)
    if is_parquet_file_empty(file_path):
        print(f"The file {file} is empty. Deleting...")
        os.remove(file_path)
