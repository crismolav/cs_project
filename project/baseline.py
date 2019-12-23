from nltk.corpus import wordnet as wn
from helpers import load_csv_info, load_text, split_reviews
import preprocess as prepro
from pdb import set_trace
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn import svm, datasets

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import spacy



def get_star_distribution(star_rating_list):
    star_rating = np.array(star_rating_list)
    unique, counts = np.unique(star_rating, return_counts=True)
    for i, type in enumerate(unique):
        print("class %s count:%s"%(type, counts[i]))

def histogram_data(Y_true):
    _ = plt.hist(Y_true, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram Amazon ratings")
    plt.show()


def plot_confusion_matrix2(confusion_matrix):
    index = ['Bad', 'Good']
    columns = ['Bad', 'Good']
    df_cm = pd.DataFrame(confusion_matrix, index=index,
                         columns=columns)
    # plt.figure(figsize=(10, 7))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    sn.heatmap(df_cm, annot=True, fmt="d", center=1)
    plt.show()

def process_reviews(star_rating_list, review_list, nlp, max_review=None):
    Y_true = list()
    Y_pred = list()
    ignored_non_english = 0
    for index, review in enumerate(review_list):
        star_rating = star_rating_list[index]
        # IGNORE NEUTRAL RATINGS
        if int(star_rating) == 3:
            continue
        processed_review = prepro.pre_process(text=review, nlp=nlp)
        # IGNORE OTHER LANGUAGES
        if processed_review == []:
            ignored_non_english +=1
            continue

        original_classification = get_classification_group_for_star_rating(
            star_rating=star_rating)

        baseline_classification = get_baseline_classification(review=processed_review)

        Y_true.append(original_classification)
        Y_pred.append(baseline_classification)

        if original_classification == 0 and baseline_classification == 1:
            pass
        if max_review is not None and index == max_review-1:
            break

    bl_precision = precision_score(Y_true, Y_pred)
    bl_recall    = recall_score(Y_true, Y_pred)
    bl_accuracy  = accuracy_score(Y_true, Y_pred)

    Y_classes = get_Y_classes()
    cm = confusion_matrix(Y_true, Y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("TN: %s" % tn)
    print("FP: %s" % fp)
    print("FN: %s" % fn)
    print("TP: %s" % tp)
    print(cm)
    # plot_confusion_matrix2(cm))
    # # Plot non-normalized confusion matrix
    # plot_confusion_matrix(Y_true, Y_pred, classes=Y_classes,
    #                       title='Confusion matrix, without normalization')
    print("Base line indicators\n")
    print("Precision: %s" % bl_precision)
    print("Recall: %s" % bl_recall)
    print("Accuracy: %s"%bl_accuracy)

def get_Y_classes():
    Y_classes = np.array(['Bad', 'Good'])
    return Y_classes

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
        total_negative += negative_score
    # if total_positive == 0 and total_negative == 0:
    #     set_trace()
    return (total_positive, total_negative)


POS_not_found = set()
def get_sentiment_score_word_pos_tuple(word_pos_tuple):
    word = word_pos_tuple[0]
    pos = word_pos_tuple[1]
    pos_or_neg = word_pos_tuple[2]
    if not pos_is_in_model(pos):
        if pos not in POS_not_found:
            print("WARNING: pos not recognized: %s"%pos)
            POS_not_found.add(pos)
        return 0, 0

    wordnet_pos = transform_pos_to_wordnet_notation(pos=pos)
    word_synsets = wn.synsets(word, wordnet_pos)
    pos_sentiment_score, neg_sentiment_score = \
        calculate_avg_sentiment_scores_for_synsets(word_synsets=word_synsets)
    if pos_or_neg == 'pos':
        return pos_sentiment_score , neg_sentiment_score
    else:
        return neg_sentiment_score, pos_sentiment_score

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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

if __name__=="__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names
    # set_trace()
    # LOAD THE REVIEWS AND THE CORREPONDING RATING
    file_name1 = "10000reviews.txt"
    file_name2 = "videogames_9999.tsv"
    star_rating_list, review_list = load_csv_info(file_name1)
    #Get distribution
    get_star_distribution(star_rating_list)
    # PROCESS REVIEWS
    nlp = spacy.load("en_core_web_sm")
    max_review = 1000
    process_reviews(
        star_rating_list=star_rating_list, review_list=review_list, nlp=nlp,
        max_review=max_review)

