#!/usr/bin/env python

import pandas as pd
from spacy.lang.it.stop_words import STOP_WORDS
import string
import re

stop_words = STOP_WORDS
stop_words_to_remove = ['anni', 'anno']
stop_words = [word for word in stop_words if word not in stop_words_to_remove]

def cleaning(text):
    if pd.isna(text):
        return ""

    words_new = [
        word[1:] if word.startswith('#') else word
        for word in text.split()
        if not (word.startswith('@') or re.match(r"http\S+|www\S+|https\S+", word) or word.lower() in stop_words)
    ]
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    filtered_text = ' '.join(words_new)

    if filtered_text.startswith('RT'):
        matches = list(re.finditer(r'\bRT\b', filtered_text))
        if matches:
            last_rt_index = matches[-1].end()
            return filtered_text[last_rt_index:].strip()
        else:
            return filtered_text.strip()
    else:
        return filtered_text.strip()  

text1 = 'RT @GrandeOrienteit: È ufficiale: lInno della'
text2 = 'THE sito per le mamme :) https://t.co/6WJ80xJ5'
text3 = 'I love #dogs'
text4 = 'RT @_saramago: Escrever é fazer recuar a morte...'

print(cleaning(text1))
print(cleaning(text2))
print(cleaning(text3))
print(cleaning(text4))


cleaned_texts = [cleaning(text) for text in [text1, text2, text3, text4]]
print(cleaned_texts)


def cleaning_mentions_then_punctuation(text):
    stop_words = []

    if pd.isna(text):
        return ""

 
    words_filtered = [
        word[1:] if word.startswith('#') else word
        for word in text.split()
        if not (word.startswith('@') or re.match(r"http\S+|www\S+|https\S+", word) or word.lower() in stop_words)
    ]


    text_filtered = ' '.join(words_filtered)

    translator = str.maketrans('', '', string.punctuation)

    text_no_punctuation = text_filtered.translate(translator)

    if text_no_punctuation.startswith('RT'):
        matches = list(re.finditer(r'\bRT\b', text_no_punctuation))
        if matches:
            last_rt_index = matches[-1].end()
            return text_no_punctuation[last_rt_index:].strip()
        else:
            return text_no_punctuation.strip()
    else:
        return text_no_punctuation.strip()

    
print('second')
cleaned_texts_mentions_then_punctuation = [cleaning_mentions_then_punctuation(text) for text in [text1, text2, text3, text4]]
print(cleaned_texts_mentions_then_punctuation)

