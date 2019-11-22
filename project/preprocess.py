# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:43:45 2019

@author: Mar Adrian
"""

import nltk
# LOAD THE TEXT
filename = 'review.txt'
file = open(filename, 'rt')
text = file.read()
file.close()


#### REMOVE HTML TAGS ###
import re
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, ' ', raw_html)
  return cleantext

no_html=cleanhtml(text)

### PUNCTUATION ### 
import string
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

no_puntuation=remove_punctuation(no_html)

### WITHOUT STOPWORDS ### 
from nltk import word_tokenize
from nltk.corpus import stopwords

#TOKENIZE
word_tokenize=word_tokenize(no_puntuation)

# LOWERCASE 
def text_lowercase(text): 
    l = [item.lower() for item in text]
    return l 
text_lc=text_lowercase(word_tokenize)

#WITHOUT SOPTWORDS
def sw(text):
    stop_en = set(stopwords.words('english'))
    new_stopwords = ["would", "could", "hes", "shes", "doesnt", "dont", "cant"]
    new_stopwords_list = stop_en.union(new_stopwords)
    no = [w for w in text if w not in new_stopwords_list]
    return no 
no_sw=sw(text_lc)


### FREQUENCE OF THE WORD ### 
from nltk.probability import FreqDist
def frequency(text):
    fdist=FreqDist(text)
    return fdist

freq=frequency(no_sw)

### NUMBER INTO WORDS ### 

    
    