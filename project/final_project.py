import nltk
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re
import string
import re

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

def load_text():
    filename = 'archivo.txt'
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    return text

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext

def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

def text_lowercase(text): 
    l = [item.lower() for item in text]
    return l 

def sw(text):
    stop_en = stopwords.words('english')
    no = [w for w in text if w not in stop_en]
    return no

def frequency(text):
    fdist=FreqDist(text)
    most_common = fdist.most_common(100)
    return fdist, most_common

if __name__=="__main__":
    star_rating_list, review = load_csv_info(fname="archivo.txt")
    # LOAD THE TEXT
    text = load_text()
    #### REMOVE HTML TAGS ###
    no_html = cleanhtml(text)
    ### PUNCTUATION ###
    no_puntuation = remove_punctuation(no_html)
    # TOKENIZE
    word_tokenize = word_tokenize(no_puntuation)
    # LOWERCASE
    text_lc = text_lowercase(word_tokenize)
    # WITHOUT SOPTWORDS
    no_sw = sw(text_lc)
    ### FREQUENCE OF THE WORD ###
    freq, most_common = frequency(no_sw)
    print(most_common)

    with open('counts.txt', 'w') as f:
        for word in freq.keys():
            # print(word, freq[word])
            f.write(str(word) + '\t' + str(freq[word]) + '\n')

        
    
    
    
    

