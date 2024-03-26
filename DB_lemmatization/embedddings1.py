#!/usr/bin/env python

#SBATCH --partition=g100_usr_bmem
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --mem=400GB                
#SBATCH --job-name=tf_embed             
#SBATCH --output=tf_embed_%j.txt



import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import time


def download_n_grams():
    from spacy.lang.it.stop_words import STOP_WORDS as italian_stop_words
    from spacy.lang.en.stop_words import STOP_WORDS as english_stop_words
    stop_words_to_remove = ['anni', 'anno']
    italian_stop_words = list(italian_stop_words)
    english_stop_words = list(english_stop_words)
    italian_stop_words = [word for word in italian_stop_words if word not in stop_words_to_remove]
    stop_words = english_stop_words+italian_stop_words
    column_names = pd.read_csv('/g100_work/IscrC_mental/data/tfidf/results_concatenate/tfidf_concatenate.csv', nrows=0).columns.tolist()
    n_grams = [element for element in column_names if element not in stop_words]
    return n_grams 

def download_model():
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    return model

def similarity_filter1(n_grams, model):
    embeddings = model.encode(n_grams, convert_to_numpy=True, normalize_embeddings=True)
    cosine_sim_matrix = cosine_similarity(embeddings)
    df = pd.DataFrame(cosine_sim_matrix)
    index_to_remove = [j for i in range(len(df)) for j in range(i + 1, len(df)) if df.loc[i, j] > 0.7 and i != j]
    return list(set(index_to_remove))

def main():
    start = time.time()

    down_start = time.time()
    n_grams = download_n_grams()
    model = download_model()

    down_finished = time.time()-down_start
    print(f'It took {down_finished} to download everything')

    start_sim = time.time()

    file_path = '/g100_work/IscrC_mental/data/tfidf/results_concatenate/sim_n_gram.pkl'
    res = similarity_filter1(n_grams, model)
    with open(file_path, 'wb') as f:
        pickle.dump(res, f)

    end_sim = time.time() - start_sim
    print(f'It took {end_sim} to filter')

    end = time.time() - start
    print(f'It took {end} to do everything')

if __name__ == "__main__":
    main()
