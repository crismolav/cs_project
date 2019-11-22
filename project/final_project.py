import nltk
import numpy as np
from nltk.probability import FreqDist
import preprocess as pp

column_separator = "\t"

def load_csv_info(fname="archivo.txt"):
    star_rating_list = []
    review_body_list = []
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

def frequency(text):
    fdist=FreqDist(text)
    most_common = fdist.most_common(100)
    return fdist, most_common

if __name__=="__main__":
    star_rating_list, review = load_csv_info(fname="archivo.txt")
    # LOAD THE TEXT
    text = load_text()
    pre_processed = pp.pre_process(text)
    ### FREQUENCE OF THE WORD ###
    freq, most_common = frequency(pre_processed)
    print(most_common)

    with open('counts.txt', 'w') as f:
        for word in freq.keys():
            # print(word, freq[word])
            f.write(str(word) + '\t' + str(freq[word]) + '\n')

        
    
    
    
    

