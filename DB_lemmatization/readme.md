In this file I added files that I created while lemmatizing db tweets:

# Lemmatization_reserach.ipynb

In this file I created function for downloading, cleaning, and after lemmatizing tweets (for Stanza ans Spacy packages). Also I tried to experiment and customize pipiline for text processing. In the result it was decided to use spacy package, since it was less aggressive. 

# Script_for_lemmatizing_db.py

In this file you can see a script for leammtizing a db in batches adapted to SBATCH 

# delete.py

Script for deleting empty batches after lemmatization

# upload_to_sql.py

Script to upload lemmatized tweets from parquet files to sql db

# n_gram.py

Script for creating n_gram matrix with top 50000 frequent n_grams

# tfidf.py

Script for creating tfidf matrices for each group

# el_net_cv.py

Script for getting baseline estimates of survey outcomes using elastic_net and cross validation
