column_separator = "\t"

star_rating_list = []
review_body_list= []

def load_csv_info(fname="archivo.txt"):
    file=open(fname)
    file.readline()
    for line in file.readlines():
        marketplace, customer_id, review_id, product_id, product_parent,	product_title, product_category, star_rating, helpful_votes, total_votes, vine, verified_purchase, review_headline, review_body, review_date = line.strip().split(column_separator)
        star_rating_list.append(star_rating)
        review_body_list.append(review_body)
    return  star_rating_list, review_body_list

star_rating_list, review = load_csv_info(fname="archivo.txt")


#%%

import nltk
import numpy as np
# LOAD THE TEXT
filename = 'archivo.txt'
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
    stop_en = stopwords.words('english')
    no = [w for w in text if w not in stop_en]
    return no 
no_sw=sw(text_lc)

### FREQUENCE OF THE WORD ### 
from nltk.probability import FreqDist
def frequency(text):
    fdist=FreqDist(text)
    most_common = fdist.most_common(100)
    return fdist, most_common

freq, most_common = frequency(no_sw)
print(most_common)

with open('counts.txt', 'w') as f:
    for word in freq.keys():
        #print(word, freq[word])
        f.write(str(word)+'\t'+str(freq[word])+'\n')
        
    
    
    
    
    