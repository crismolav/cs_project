import nltk
import numpy as np
from nltk.probability import FreqDist
import preprocess as pp
from nltk.corpus import sentiwordnet as swn
from pdb import set_trace
from nltk.corpus import wordnet as wn
column_separator = "\t"

def load_csv_info(fname):
    star_rating_list = []
    review_body_list = []
    file=open(fname)
    file.readline()
    for line in file.readlines():
        marketplace, customer_id, review_id, product_id, product_parent,	product_title, product_category, star_rating, helpful_votes, total_votes, vine, verified_purchase, review_headline, review_body, review_date = line.strip().split(column_separator)
        star_rating_list.append(star_rating)
        review_body_list.append(review_body)
    return  star_rating_list, review_body_list

def load_text(filename):
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    return text

def frequency(text):
    fdist=FreqDist(text)
    most_common = fdist.most_common(100)
    return fdist, most_common

def split_reviews(star_rating_list, review):
    reviews_45 = []
    reviews_12 = []
    with open('reviews45.txt', 'w') as f45:
        with open('reviews12.txt', 'w') as f12:
            for ii, star in enumerate(star_rating_list):
                current_review = review[ii]
                if star == '4' or star == '5':
                    reviews_45.append(current_review)
                    f45.write(current_review + '\n')
                elif star == '1' or star == '2':
                    reviews_12.append(current_review)
                    f12.write(current_review + '\n')
    return reviews_12, reviews_45

def prior_of_the_classes(reviews_12, reviews_45):
    num_neg_reviews = len(reviews_12)
    num_pos_reviews = len(reviews_45)
    total_reviews = num_neg_reviews + num_pos_reviews
    
    prior_neg = num_neg_reviews / total_reviews
    prior_pos = num_pos_reviews / total_reviews
    
    return prior_neg, prior_pos


if __name__=="__main__":
    #LOAD THE REVIEWS AND THE CORREPONDING RATING
    star_rating_list, review = load_csv_info("10000reviews.txt")
    #SPLIT THE REVIEWS INTO POSITIVE (RATING 4,5) AND NEGATIVE (RATING 1,2)
    reviews_12, reviews_45 = split_reviews(star_rating_list, review)
    
    #Prior probability
    prior_neg, prior_pos = prior_of_the_classes(reviews_12, reviews_45)
    
    ######## NEGATIVE REVIEWS ######## 
    # LOAD THE TEXT
    text = load_text('reviews12.txt')
    # PRE-PROCESS
    pre_processed = pp.pre_process(text)
    ### FREQUENCE OF THE WORD ###

    word_synsets = wn.synsets('great')
    word_sent = swn.senti_synsets('great', 's')
    for synset in word_sent:
        print(synset)

    freq12, most_common12 = frequency(pre_processed)
    print(most_common12)
    
    with open('counts12.txt', 'w') as f:
        for word in freq12.keys():
            f.write(str(word) + '\t' + str(freq12[word]) + '\n')

    ######## POSITIVE REVIEWS ######## 
    # LOAD THE TEXT
    text = load_text('reviews45.txt')
    # PRE-PROCESS
    pre_processed = pp.pre_process(text)
    ### FREQUENCE OF THE WORD ###
    freq45, most_common45 = frequency(pre_processed)
    print(most_common45)
   
    with open('counts45.txt', 'w') as f:
        for word in freq45.keys():
            f.write(str(word) + '\t' + str(freq45[word]) + '\n')



        
    
    
    
    

