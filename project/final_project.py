import nltk
from nltk.probability import FreqDist
import preprocess as pp
from helpers import load_csv_info, load_text, split_reviews
from pdb import set_trace

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
    # print(most_common45)
    with open('counts45.txt', 'w') as f:
        for word in freq45.keys():
            f.write(str(word) + '\t' + str(freq45[word]) + '\n')



        
    
    
    
    

