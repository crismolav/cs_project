import nltk
from nltk.probability import FreqDist
import preprocess as pp
from helpers import load_csv_info, load_text, split_reviews
from pdb import set_trace
import numpy as np 

def frequency(text):
    fdist=FreqDist(text)
    most_common = fdist.most_common(100)
    return fdist, most_common

def prior_of_the_classes(reviews_12, reviews_45):
    num_neg_reviews = len(reviews_12)
    num_pos_reviews = len(reviews_45)
    total_reviews = num_neg_reviews + num_pos_reviews
    
    prior_neg = num_neg_reviews / total_reviews
    prior_pos = num_pos_reviews / total_reviews
    
    return prior_neg, prior_pos

def naive_bayes(target, prior_neg, prior_pos, vocabulary): 
    
    pre_processed = pp.pre_process(target)
    freq_target, most_common_target = frequency(pre_processed)
    prob_neg = 1
    prob_pos = 1
    for ii, word_pos in enumerate(freq_target):
        train_freq12 = freq12[word_pos]
        train_freq45 = freq45[word_pos]
        current_freq = freq_target[word_pos]
        
        words12_fullset = sum_freq(freq12)
        words45_fullset = sum_freq(freq45)
        
        curreny_prob_neg = (train_freq12 + current_freq) / (words12_fullset + len(vocabulary))
        prob_neg = prob_neg*curreny_prob_neg
        
        curreny_prob_pos = (train_freq45 + current_freq) / (words45_fullset + len(vocabulary))
        prob_pos = prob_pos*curreny_prob_pos
    
    neg = prior_neg*prob_neg
    pos = prior_pos*prob_pos
    
    return neg, pos
    
def sum_freq(freq45):
    sum_freq45 = 0
    for jj, word_pos in enumerate(freq45):
        sum_freq45 = sum_freq45 + freq45[word_pos]    
    return sum_freq45


if __name__=="__main__":
    #LOAD THE REVIEWS AND THE CORREPONDING RATING
    star_rating_list, review = load_csv_info("10000reviews.txt")
    #SPLIT THE REVIEWS INTO POSITIVE (RATING 4,5) AND NEGATIVE (RATING 1,2)
    reviews_12, reviews_45, reviews1245 = split_reviews(star_rating_list, review)
    
    #Prior probability
    prior_neg, prior_pos = prior_of_the_classes(reviews_12, reviews_45)
    
    #Vocabulary
    pre_processed = pp.pre_process(str(reviews1245))
    vocabulary = frequency(pre_processed)
    
    ######## NEGATIVE REVIEWS ######## 
    # LOAD THE TEXT
    text = load_text('reviews12.txt')
    # PRE-PROCESS
    pre_processed = pp.pre_process(text)
    ### FREQUENCE OF THE WORD ###
    freq12, most_common12 = frequency(pre_processed)  
    
    #with open('counts12.txt', 'w') as f:
    #    for word in freq12.keys():
    #        f.write(str(word) + '\t' + str(freq12[word]) + '\n')

    ######## POSITIVE REVIEWS ######## 
    # LOAD THE TEXT
    text = load_text('reviews45.txt')
    # PRE-PROCESS
    pre_processed = pp.pre_process(text)
    ### FREQUENCE OF THE WORD ###
    freq45, most_common45 = frequency(pre_processed)
    # print(most_common45)
        
    #with open('counts45.txt', 'w') as f:
    #    for word in freq45.keys():
    #        f.write(str(word) + '\t' + str(freq45[word]) + '\n')
    
    
    ######## MAKING PREDICTIONS GIVEN A NEW REVIEW ########     
    #target = "I don't like this book" 
    #print(target)
            
    #neg, pos = naive_bayes(target, prior_neg, prior_pos, vocabulary)
    
    #if neg > pos:
    #    print("This is a negative review")
    #    print(neg, pos)
    #elif neg < pos:
    #    print("This is a positive review")   
    #    print(neg, pos)

    ######## EVALUATING THE TEST SET ########     
    star_rating_list, review = load_csv_info("test_set.txt")
    reviews_12, reviews_45, reviews1245 = split_reviews(star_rating_list, review)
        
    true_negative = 0
    false_negative = 0
    true_positive = 0
    false_positive = 0
    
    for ii, test_review in enumerate(reviews1245):
        #print("ii =", ii)
        target = test_review
        #print(target)
        neg, pos = naive_bayes(target, prior_neg, prior_pos, vocabulary)
        if neg > pos and star_rating_list[ii] == '1':
            true_negative += 1
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            #print('Star rating:', star_rating_list[ii])
        elif neg > pos and star_rating_list[ii] == '2':
            true_negative += 1
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            #print('Star rating:', star_rating_list[ii])            
            
        elif neg > pos and star_rating_list[ii] == '4':
            false_negative += 1
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            #print('Star rating:', star_rating_list[ii])
        elif neg > pos and star_rating_list[ii] == '5':
            false_negative += 1
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            #print('Star rating:', star_rating_list[ii])
            
        elif neg < pos and star_rating_list[ii] == '1':
            false_positive += 1
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            #print('Star rating:', star_rating_list[ii]) 
        elif neg < pos and star_rating_list[ii] == '2':
            false_positive += 1
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            #print('Star rating:', star_rating_list[ii])
            
        elif neg < pos and star_rating_list[ii] == '4':
            true_positive += 1
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            #print('Star rating:', star_rating_list[ii])
        elif neg < pos and star_rating_list[ii] == '5':
            true_positive += 1
            #print('Negative score:', neg)
            #print('Posivite score:', pos)
            # print('Star rating:', star_rating_list[ii])
    
print("True positives:", true_positive)   
print("True negatives:", true_negative)   
print("False positives:", false_positive)   
print("False negatives:", false_negative)   
 