#!/usr/bin/env python

#SBATCH --partition=g100_usr_prod
#SBATCH --ntasks-per-node=16
#SBATCH --time=18:00:00
#SBATCH --mem=100G             
#SBATCH --job-name=post_clean              
#SBATCH --output=post_clean_%j.txt


example_to_run_code = """
sbatch /g100/home/userexternal/ddurmush/Data_preprocessing/10_script.py --path_save /g100_work/IscrC_mental/data/lemmatized_tweets/lemmatized_tweets_post_cleaning/ --steps_to_run "clean lemma post_clean" --batch_size 500000 --total_batches 200 --starting_batch 1400
"""

import argparse
import pandas as pd
import numpy as np
import spacy
import re
import string
from spacy.cli import download
from spacy.lang.it.stop_words import STOP_WORDS as italian_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as english_stop_words
import sqlite3
import time
import concurrent.futures



try:
    it_nlp_spacy = spacy.load('it_core_news_lg')
except OSError:
    download('it_core_news_lg')
    it_nlp_spacy = spacy.load('it_core_news_lg')

conn = sqlite3.connect('/g100_work/IscrC_mental/data/database/MENTALISM.db')

stop_words_to_remove = ['anni', 'anno']
italian_stop_words = [word for word in italian_stop_words if word not in stop_words_to_remove]
stop_words = italian_stop_words + list(english_stop_words)



def cleaning(text):
    if pd.isna(text):
        return ""
    

    words_new = [
        word[1:] if word.startswith('#') else word
        for word in text.split()
        if not (word.startswith('@') or re.match(r"http\S+|www\S+|https\S+", word) or word.lower() in stop_words)
    ]
    filtered_text = ' '.join(words_new)
    translator = str.maketrans('', '', string.punctuation)
    filtered_text = filtered_text.translate(translator)
    

    if filtered_text.startswith('RT'):
        matches = list(re.finditer(r'\bRT\b', filtered_text))
        if matches:
            last_rt_index = matches[-1].end()
            return filtered_text[last_rt_index:].strip()
        else:
            return filtered_text.strip()
    else:
        return filtered_text.strip()   
    

def lemmatizationSpacy(text):
    tokens = it_nlp_spacy(text)
    lemmatized_tweet = " ".join([token.lemma_ for token in tokens])
    return lemmatized_tweet

def post_cleaning(text):
    if pd.isna(text):
        return ""
    words_new = [word for word in text.split() if not word.lower() in stop_words]
    post_cleaned_text = ' '.join(words_new)
    return post_cleaned_text



def batches_arg_type(value):
    if value.lower() == 'all_dataset':
        return value
    try:
        return int(value)
    except ValueError:
        return value



def process_batch(i, batch_size, steps_to_run, path_save): 
    start_time = time.time()
    offset = (i - 1) * batch_size
    query = f"SELECT tweet_id, text FROM tweets LIMIT {batch_size} OFFSET {offset}"
    batch_df = pd.read_sql_query(query, conn)
    current_df = batch_df
    for step in steps_to_run:
        if step == 'clean':
            current_df['text'] = current_df['text'].apply(cleaning)
        elif step == 'lemma':
            current_df['text'] = current_df['text'].apply(lemmatizationSpacy)
        elif step == 'post_clean':
            current_df['text'] = current_df['text'].apply(post_cleaning)
            break
    
    current_df = current_df[current_df['text']!='']
    current_df.to_parquet(f'{path_save}batch_number_{str(i)}.parquet', index=False)
    time_batch = time.time() - start_time
    print(current_df.head())
    print(f"\nTime taken for executing steps {i}: {time_batch:.2f} seconds")


def process_batches(args):
    time_total_start = time.time()
    if args.total_batches == 'all_dataset':
        total_tweets = pd.read_sql_query("SELECT COUNT(*) FROM tweets", conn)['COUNT(*)'].values[0]
        args.total_batches = int((total_tweets // args.batch_size).round(0))
    else:
        args.total_batches = int(args.total_batches)
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        futures = []
        for i in range(args.starting_batch, args.starting_batch + args.total_batches + 1):
            futures.append(executor.submit(process_batch, i, args.batch_size, args.steps_to_run.split(), args.path_save))
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
        total_time = time.time() - time_total_start
        print(f'Time nedeed to run all batches:{total_time:.2f} seconds', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data steps.')
    parser.add_argument('--path_save', type=str, help='Location of saved batches.')
    parser.add_argument('--steps_to_run', type=str, help='Steps to run separated by space.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for data processing.')
    parser.add_argument('--total_batches', type=batches_arg_type, help='Number of batches (or "all_dataset").')
    parser.add_argument('--starting_batch', type=int, default=1, help='Starting batch')


    args = parser.parse_args()
    print(args)
    process_batches(args)