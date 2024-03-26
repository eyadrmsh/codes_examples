#!/usr/bin/env python

#SBATCH --partition=g100_usr_bmem
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --mem=400GB                
#SBATCH --job-name=n_gram               
#SBATCH --output=n_gram_%j.txt


how_to_run_example = '''
sbatch /g100/home/userexternal/ddurmush/tiff/n_gram5.py --output_dir /g100_work/IscrC_mental/data/n_grams/n_grams_post_cleaned/results/
'''


import argparse
import os
import time
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def data_aggregation():
    conn = sqlite3.connect('/g100_work/IscrC_mental/data/database/mentalism_regioncoded_tweets.db')
    cursor = conn.cursor()
    join_query = f"""
    SELECT l.tweet_id, l.text, r.group_id
    FROM lemmatized_tweets_post_cleaning AS l
    JOIN tweets_regioncoded_groups AS r ON l.tweet_id = r.tweet_id;
    """
    cursor.execute(join_query)
    data = cursor.fetchall()
    print(f'length of a df is {len(data)}', flush=True)
    conn.close()
    df = pd.DataFrame(data, columns=['tweet_id', 'text', 'group_id'])
    return df


def restrict_vocabulary(df):
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    X = vectorizer.fit_transform(df['text'])
    feature_names = vectorizer.get_feature_names_out()
    ngram_frequencies = X.sum(axis=0)
    ngram_freq_tuples = [(feature_names[i], ngram_frequencies[0, i]) for i in range(len(feature_names))]
    sorted_ngrams = sorted(ngram_freq_tuples, key=lambda x: x[1], reverse=True)
    top_ngrams = sorted_ngrams[:50000]
    top_ngram_names = [ngram[0] for ngram in top_ngrams]
    return top_ngram_names

def vectorize_text(group, top_ngram_names):
    vectorizer_top_ngrams = CountVectorizer(ngram_range=(1, 3), vocabulary=top_ngram_names)
    ngrams = vectorizer_top_ngrams.transform(group['text'])
    feature_names = vectorizer_top_ngrams.get_feature_names_out()
    group_ngrams_df = pd.DataFrame(ngrams.toarray(), columns=feature_names).sum(axis=0).to_frame().T
    return group_ngrams_df

def n_gram_matrix(df, output_dir, top_ngram_names):
    for group_id, group in df.groupby('group_id'):
        group_vectorized = vectorize_text(group, top_ngram_names)
        group_vectorized.to_csv(os.path.join(output_dir, f'ngram_freq_by_group_{group_id}.csv'), index=False)
        print(f"matrix for group {group_id} is created")

def main():
    parser = argparse.ArgumentParser(description='Process and vectorize text data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where the vocabulary and matrices will be saved')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    download_df_start = time.time()
    df = data_aggregation()
    download_df_time = time.time() -  download_df_start
    print(f"It took {download_df_time} seconds to download df", flush = True)

    unique_groups = df['group_id'].unique()
    print(f"Processing {len(unique_groups)} unique groups.")
    
    restrict_start = time.time()
    top_ngram_names = restrict_vocabulary(df)
    with open(os.path.join(args.output_dir, 'top_ngram_names_all_dataset.pkl'), 'wb') as f:
        pickle.dump(top_ngram_names, f)

    total_time = time.time() - start_time
    print(f"It took {total_time} seconds to execute all code")

if __name__ == "__main__":
    main()
