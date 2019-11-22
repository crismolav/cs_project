import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:43:45 2019

@author: Mar Adrian
"""
#### REMOVE HTML TAGS ###
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext
### PUNCTUATION ###
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

# LOWERCASE 
def text_lowercase(text): 
    l = [item.lower() for item in text]
    return l 

#WITHOUT SOPTWORDS
def sw(text):
    stop_en = set(stopwords.words('english'))
    new_stopwords = ["would", "could", "hes", "shes", "doesnt", "dont", "cant"]
    new_stopwords_list = stop_en.union(new_stopwords)
    no = [w for w in text if w not in new_stopwords_list]
    return no
### FREQUENCE OF THE WORD ###
def frequency(text):
    fdist=FreqDist(text)
    return fdist

### NUMBER INTO WORDS ###
def replace_numbers(words):
    import inflect
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words