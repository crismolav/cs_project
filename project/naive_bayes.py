#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 23:30:20 2020

@author: andreavalenzuelaramirez
"""
import nltk
from nltk.probability import FreqDist
import preprocess as pp
from helpers import load_csv_info, load_text, split_reviews
from pdb import set_trace
import spacy

def frequency(text):
    fdist=FreqDist(text)
    most_common = fdist.most_common(200)
    return fdist, most_common

def trainig_set(review_doc):
    min_num = 100
    nlp = spacy.load("en_core_web_sm")

    #star_rating_list, reviews = load_csv_info("10000reviews.txt")    
    star_rating_list, reviews = load_csv_info(review_doc)   

    for ii, review in enumerate(reviews):
        res = len(review.split()) 
        if res > min_num:
            reviews[ii] = ' '.join(review.split()[0:min_num])
    
    reviews_12, reviews_45, reviews1245, star_12, star_45, star1245 = split_reviews(star_rating_list, reviews)
    
    #Prior probability
    prior_neg, prior_pos = prior_of_the_classes(reviews_12, reviews_45)

    
    ######## NEGATIVE REVIEWS ######## 
    text = '\n'.join(reviews_12)
    pre_processed_12 = pp.pre_process(text, nlp)
    freq12, most_common12 = frequency(pre_processed_12)
    
    ######## POSITIVE REVIEWS ######## 
    text = '\n'.join(reviews_45)
    pre_processed_45 = pp.pre_process(text, nlp)
    freq45, most_common45 = frequency(pre_processed_45)
    
    pre_processed_1245 = pre_processed_12 + pre_processed_45
    vocabulary, voc_common = frequency(pre_processed_1245)
    
    return freq12, most_common12, freq45, most_common45, vocabulary, voc_common, prior_neg, prior_pos

def sum_freq(freq45):
    sum_freq45 = 0
    for jj, word_pos in enumerate(freq45):
        sum_freq45 = sum_freq45 + freq45[word_pos]    
    return sum_freq45

def prior_of_the_classes(reviews_12, reviews_45):
    num_neg_reviews = len(reviews_12)
    num_pos_reviews = len(reviews_45)
    total_reviews = num_neg_reviews + num_pos_reviews
    
    prior_neg = num_neg_reviews / total_reviews
    prior_pos = num_pos_reviews / total_reviews
    return prior_neg, prior_pos

def naive_bayes_algorithm(reviews1245, star1245, freq12, freq45, vocabulary, voc_common, prior_neg, prior_pos):
    
    true_negative = 0
    false_negative = 0
    true_positive = 0
    false_positive = 0
    else_indicator = 0
    
    false_positive_list = []
    false_negative_list = []
    else_list = []
    
    for ii, test_review in enumerate(reviews1245):

        target = test_review
        pre_processed = pp.pre_process(target, nlp)
        freq_target, most_common_target = frequency(pre_processed)
        
        prob_neg = 1
        prob_pos = 1
        
        words12_fullset = sum_freq(freq12)
        #print("total negative words:", words12_fullset)
        words45_fullset = sum_freq(freq45)
        #print("total positive words:", words45_fullset)
        
        for jj, word_pos in enumerate(freq_target):
            
            train_freq12 = freq12[word_pos]
            #print("negative frequency:", train_freq12)
            train_freq45 = freq45[word_pos]
            #print("positive frequency:", train_freq45)
            #current_freq = freq_target[word_pos]
            current_freq = 1 #COUNT WORDS ONLY ONCE
        
            
            #print("vocabulary:", len(vocabulary))
            if train_freq12 == 0 or train_freq45 == 0:
                curreny_prob_neg = 1
                curreny_prob_pos = 1
            else:
                curreny_prob_neg = (train_freq12 + current_freq) / (words12_fullset + len(vocabulary))
                curreny_prob_pos = (train_freq45 + current_freq) / (words45_fullset + len(vocabulary))
            
            #print("current_prob_neg:", curreny_prob_neg)
            prob_neg = prob_neg*curreny_prob_neg
            #print("prob_neg:", prob_neg)
            
            #print("current_prob_pos:", curreny_prob_pos)
            prob_pos = prob_pos*curreny_prob_pos
            #print("prob_pos:", prob_pos)
        
        neg = prior_neg*prob_neg
        pos = prior_pos*prob_pos
        
        #Print("Total negative score:", neg)
        #print("Total positive score:", pos)
        
        # ------------------ EVALUATION ---------------------- 
        if neg > pos and star1245[ii] == '1':
            true_negative += 1
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            #print('Star rating:', star_rating_list[ii])
        elif neg > pos and star1245[ii] == '2':
            true_negative += 1
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            #print('Star rating:', star_rating_list[ii])            
            
        elif neg > pos and star1245[ii] == '4':
            false_negative += 1
            print("---- FALSE NEGATIVE ----")
            print(target)
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            print('Star rating:', star1245[ii])
        elif neg > pos and star1245[ii] == '5':
            false_negative += 1
            print("---- FALSE NEGATIVE ----")
            print(target)
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            print('Star rating:', star1245[ii])
            
        elif neg < pos and star1245[ii] == '1':
            false_positive += 1
            print("---- FALSE POSITIVE ----")
            print(target)
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            print('Star rating:', star1245[ii]) 
        elif neg < pos and star1245[ii] == '2':
            false_positive += 1
            print("---- FALSE POSITIVE ----")
            print(target)
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            print('Star rating:', star1245[ii])
            
        elif neg < pos and star1245[ii] == '4':
            true_positive += 1
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            #print('Star rating:', star_rating_list[ii])
        elif neg < pos and star1245[ii] == '5':
            true_positive += 1
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            # print('Star rating:', star_rating_list[ii])

        
    return true_positive, true_negative, false_positive, false_negative, false_negative_list, false_positive_list, else_indicator, else_list


def single_review(test_review, star_rating, prior_neg, prior_pos):
      
    true_negative = 0
    false_negative = 0
    true_positive = 0
    false_positive = 0
    
    false_positive_list = []
    false_negative_list = []
    
    target = test_review
    pre_processed = pp.pre_process(target, nlp)
    freq_target, most_common_target = frequency(pre_processed)
        
    prob_neg = 1
    prob_pos = 1
        
    words12_fullset = sum_freq(freq12)
    #print("total negative words:", words12_fullset)
    words45_fullset = sum_freq(freq45)
    #print("total positive words:", words45_fullset)
        
    for jj, word_pos in enumerate(freq_target):
        
        print("------ New Target -------")
        print(word_pos)
        
        train_freq12 = freq12[word_pos]
        #print("negative frequency:", train_freq12)
        train_freq45 = freq45[word_pos]
        #print("positive frequency:", train_freq45)
        #current_freq = freq_target[word_pos]
        current_freq = 1 #COUNT WORDS ONLY ONCE
    
        
        #print("vocabulary:", len(vocabulary))
        
        
        if train_freq12 == 0 or train_freq45 == 0:
            curreny_prob_neg = 1
            curreny_prob_pos = 1
        else:
            curreny_prob_neg = (train_freq12 + current_freq) / (words12_fullset + len(vocabulary))
            curreny_prob_pos = (train_freq45 + current_freq) / (words45_fullset + len(vocabulary))
        
        print("current_prob_neg:", curreny_prob_neg)
        prob_neg = prob_neg*curreny_prob_neg
        #print("prob_neg:", prob_neg)
        
        print("current_prob_pos:", curreny_prob_pos)
        prob_pos = prob_pos*curreny_prob_pos
        #print("prob_pos:", prob_pos)

    neg = prior_neg*prob_neg
    pos = prior_pos*prob_pos
    
    print("negative_score:", neg)
    print("positive_score:", pos)
    
    if neg > pos and star_rating == '1':
        true_negative += 1

    elif neg > pos and star_rating == '2':
        true_negative += 1
        
    elif neg > pos and star_rating == '4':
        false_negative += 1
        print("---- FALSE NEGATIVE ----")
        false_negative_list.append(target)

    elif neg > pos and star_rating == '5':
        false_negative += 1
        print("---- FALSE NEGATIVE ----")
        false_negative_list.append(target)
        
    elif neg < pos and star_rating == '1':
        false_positive += 1
        print("---- FALSE POSITIVE ----")
        false_positive_list.append(target)

    elif neg < pos and star_rating == '2':
        false_positive += 1
        print("---- FALSE POSITIVE ----")
        false_positive_list.append(target)
        
    elif neg < pos and star_rating == '4':
        true_positive += 1

    elif neg < pos and star_rating == '5':
        true_positive += 1
            
    return true_positive, true_negative, false_positive, false_negative

#%%
    
if __name__=="__main__":
    
    min_num = 100
    nlp = spacy.load("en_core_web_sm")
    
    ## ---------------------------- TRAINING ----------------------------
    #freq12, most_common12 , freq45, most_common45, vocabulary, voc_common, prior_neg, prior_pos = trainig_set("20000Beauty.txt")
                    
    star_rating_list, reviews = load_csv_info("10000Beauty_b.txt")
    #print("test size:", len(reviews))
    #reviews = ["This is a FAKE PRODUCT DO NOT Get It!!"]
    #star_rating_list = ['1']
    #test_review = reviews[0]
    #star_rating = star_rating_list[0]
    
    #for kk, review in enumerate(reviews):
    #    res = len(review.split()) 
    #    if res > min_num:
    #        reviews[kk] = ' '.join(review.split()[0:min_num])

    reviews_12, reviews_45, reviews1245, star_12, star_45, star1245 = split_reviews(star_rating_list, reviews)
    
    ## ---------------------------- MASSIVE RUN ----------------------------
    true_positive, true_negative, false_positive, false_negative, false_negative_list, false_positive_list, else_indicator, else_list = naive_bayes_algorithm(reviews1245, star1245, freq12, freq45, vocabulary, voc_common, prior_neg, prior_pos)
    
    #with open('false_negatives_1.txt', 'w') as file:
    #    file.write(str(false_negative_list))
    #with open('false_positives_1.txt', 'w') as file:
    #    file.write(str(false_positive_list))
        
    ## ---------------------------- SINGLE RUN ----------------------------
    #true_positive, true_negative, false_positive, false_negative = single_review(test_review,star_rating, prior_neg, prior_pos)
    
    print("True positives:", true_positive)   
    print("True negatives:", true_negative)   
    print("False positives:", false_positive)   
    print("False negatives:", false_negative)  
    #print("Else indicator", else_indicator)
