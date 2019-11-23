from nltk.corpus import wordnet as wn
from helpers import load_csv_info, load_text, split_reviews
import preprocess as prepro
from pdb import set_trace
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
import numpy as np
import math
from sklearn.metrics import precision_score, recall_score, accuracy_score

def process_reviews(star_rating_list, review_list):
    Y_true = list()
    Y_pred = list()
    for index, review in enumerate(review_list):
        star_rating = star_rating_list[index]
        # IGNORE NEUTRAL RATINGS
        if int(star_rating) == 3:
            continue
        processed_review = prepro.pre_process(text=review)
        original_classification = get_classification_group_for_star_rating(
            star_rating=star_rating)
        baseline_classification = get_baseline_classification(review=processed_review)
        Y_true.append(original_classification)
        Y_pred.append(baseline_classification)
        if index == 100:
            break

    bl_precision = precision_score(Y_true, Y_pred)
    bl_recall    = recall_score(Y_true, Y_pred)
    bl_accuracy  = accuracy_score(Y_true, Y_pred)
    print("Base line indicators\n")
    print("Precision: %s" % bl_precision)
    print("Recall: %s" % bl_recall)
    print("Accuracy: %s"%bl_accuracy)

def get_classification_group_for_star_rating(star_rating):
    if int(star_rating) in [4, 5]:
        return 1
    elif int(star_rating) in [1, 2]:
        return 0
    else:
        raise Exception("can't process star_rating number:%s"%star_rating)

def get_baseline_classification(review):
    baseline_score_pos, baseline_score_neg  = get_sentiment_scores(review)
    final_score = baseline_score_pos - baseline_score_neg
    threshold = 0
    if final_score >threshold:
        return 1  # Positive review prediction
    else:
        return 0  # Negative review prediction

def get_sentiment_scores(review):
    total_positive = 0
    total_negative = 0
    for word_pos_tuple in review:
        positive_score, negative_score = get_sentiment_score_word_pos_tuple(
            word_pos_tuple=word_pos_tuple)
        total_positive += positive_score
        # if math.isnan(total_positive):
        #     set_trace()
        total_negative += negative_score

    return (total_positive, total_negative)

def get_sentiment_score_word_pos_tuple(word_pos_tuple):
    word = word_pos_tuple[0]
    pos = word_pos_tuple[1]
    if not pos_is_in_model(pos):
        print("WARNING: pos not recognized: %s"%pos)
        return 0, 0

    wordnet_pos = transform_pos_to_wordnet_notation(pos=pos)
    word_synsets = wn.synsets(word, wordnet_pos)
    pos_sentiment_score, neg_sentiment_score = \
        calculate_avg_sentiment_scores_for_synsets(word_synsets=word_synsets)

    return pos_sentiment_score , neg_sentiment_score

def calculate_avg_sentiment_scores_for_synsets(word_synsets):
    if word_synsets == []:
        return 0, 0
    pos_sentiment_score_array = np.zeros(len(word_synsets))
    neg_sentiment_score_array = np.zeros(len(word_synsets))
    for i, synset in enumerate(word_synsets):
        pos_sentiment_score, neg_sentiment_score = \
            calcualte_sentiment_scores_for_synset(synset)
        pos_sentiment_score_array[i] = pos_sentiment_score
        neg_sentiment_score_array[i] = neg_sentiment_score

    return np.mean(pos_sentiment_score_array), np.mean(neg_sentiment_score_array)

def calcualte_sentiment_scores_for_synset(synset):
    synset_name = synset.name()
    word_sentiment = swn.senti_synset(synset_name)
    pos_score = word_sentiment.pos_score()
    neg_score = word_sentiment.neg_score()

    return pos_score, neg_score

def get_pos_to_wordnet_pos_dictionary():
    pos_to_wordnet_pos_dictionary = {
        'NOUN': 'n',
        'PRON': 'n',
        'ADV': 'r',
        'VERB': 'v',
        'ADJ': 'a',
        'NUM': 'a',
        'DET': 'a',
        'ADP': 'a',
        'PRT': 'a',
        'CONJ': 'a'
    }
    return pos_to_wordnet_pos_dictionary
def pos_is_in_model(pos):
    pos_to_wordnet_pos_dictionary = get_pos_to_wordnet_pos_dictionary()
    return pos in pos_to_wordnet_pos_dictionary

def transform_pos_to_wordnet_notation(pos):
    pos_to_wordnet_pos_dictionary = {
        'NOUN': 'n',
        'PRON': 'n',
        'ADV': 'r',
        'VERB': 'v',
        'ADJ': 'a',
        'NUM': 'a',
        'DET': 'a',
        'ADP': 'a',
        'PRT': 'a',
        'CONJ': 'a'
    }
    return pos_to_wordnet_pos_dictionary[pos]

if __name__=="__main__":
    # word_synsets = wn.synsets('love','v')
    # for synset in word_synsets:
    #     sent_synset = swn.senti_synset(synset.name())
    #     print(sent_synset.pos_score())

    # LOAD THE REVIEWS AND THE CORREPONDING RATING
    star_rating_list, review_list = load_csv_info("10000reviews.txt")
    # PROCESS REVIEWS
    process_reviews(star_rating_list=star_rating_list, review_list=review_list)

