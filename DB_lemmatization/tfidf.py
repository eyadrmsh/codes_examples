#!/usr/bin/env python

#SBATCH --partition=g100_usr_bmem
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --mem=1000GB                 
#SBATCH --job-name=tfidf_bysum              
#SBATCH --output=tfidf_bysum_%j.txt


how_to_run_example = '''
sbatch /g100/home/userexternal/ddurmush/tiff/tfidf3.py --output_dir /g100_work/IscrC_mental/data/tfidf/results_concatenate
'''

import time
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import argparse
import os
import pickle


def data_aggregation():
    '''
    function for downloading data from db
    '''
    conn = sqlite3.connect('/g100_work/IscrC_mental/data/database/mentalism_regioncoded_tweets.db')
    cursor = conn.cursor()
    join_query = f"""
    SELECT l.tweet_id, l.text, r.group_id
    FROM lemmatized_tweets_post_cleaning AS l
    JOIN tweets_regioncoded_groups AS r ON l.tweet_id = r.tweet_id;
    """
    cursor.execute(join_query)
    data = cursor.fetchall()
    print(f'length of a df is {len(data)}')
    conn.close()
    df = pd.DataFrame(data, columns=['tweet_id', 'text', 'group_id'])
    return df


def restriction_vocabulary(df):
    '''
    restricting vocabulary to top 50000 tfidf frequencies 
    '''
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, stop_words='english', min_df=0.001, max_df=0.75)
    documents = {}
    for group_id, group in df.groupby('group_id'):
        documents[group_id] = ' '.join(group['text'])
    X = vectorizer.fit_transform(list(documents.values()))
    feature_names = vectorizer.get_feature_names_out()
    sum_tfidf_scores = np.sum(axis=0).A1
    top_tfidf_indices = np.argsort(sum_tfidf_scores)[-50000:]
    top_tfidf_names = [feature_names[i] for i in top_tfidf_indices]
    return top_tfidf_names



def tfidf_matrix(df, output_dir, vocabulary):
    '''
    creating a tfidf matrix considering as one document one tweets within group, and after taking its mean to get one row for a group
    '''
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    for group_id, group in df.groupby('group_id'):
        tfidf_matrix = vectorizer.fit_transform(group['text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        aggregated_scores = tfidf_df.mean(axis=0).to_frame().T
        aggregated_scores.to_csv(os.path.join(output_dir, f'tfidf_group_{group_id}.csv'), index=False)
        print(f"TF-IDF matrix for group {group_id} is created")


def tfidf_matrix_each_tweet(df, output_dir, vocabulary):
    '''
    making dfs for tfidf frequencies for each tweet within one group
    '''
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    for group_id, group in df.groupby('group_id'):
        tfidf_matrix = vectorizer.fit_transform(group['text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        tfidf_df.to_csv(os.path.join(output_dir, f'tfidf_group_each_tweet_{group_id}.csv'), index=False)
        print(f"TF-IDF matrix for group {group_id} is created")


def tfidf_matrix_concatenate(df, output_dir, vocabulary):
    '''
    creating a tfidf matrix considering as one document 'concatination' of all tweets amoung within one group
    '''
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    documents = {}
    for group_id, group in df.groupby('group_id'):
        documents[group_id] = ' '.join(group['text'])
    tfidf_matrix = vectorizer.fit_transform(list(documents.values()))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df['group_id'] = list(documents.keys())
    tfidf_df.to_csv(os.path.join(output_dir, f'tfidf_concatenate_post_clean.csv'), index=False)


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

    
    restrict_start = time.time()
    tfidf_names  =restriction_vocabulary(df)
    with open('/g100_work/IscrC_mental/data/tfidf/results_concatenate/tfidf_names.pkl', 'wb') as f:
        pickle.dump(tfidf_names, f)

    tfidf_matrix_concatenate(df, args.output_dir, vocabulary)
     restrict_time = time.time() - restrict_start
    print(f"It took {restrict_time} seconds to restrict vocabulary and make tfidf matrices")

    total_time = time.time() - start_time
    print(f"It took {total_time} seconds to execute all code")

    total_time = time.time() - start_time
    print(f"It took {total_time} seconds to execute all code")

if __name__ == "__main__":
    main()
