from helpers import load_csv_info, load_text, split_reviews,\
    get_star_distribution, get_classification_group_for_star_rating,\
    print_metrics, transform_pos_to_wordnet_notation,\
    get_file_name_from_sys_arg
import preprocess as prepro
from pdb import set_trace
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
import numpy as np


from sklearn.metrics import precision_score, recall_score,\
    accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split, cross_val_predict
from autocorrect import Speller
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import json
import sys



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

# def process_reviews2(file_name, logreg=False, max_review=None, threshold=1.3):
#     Y_true = list()
#     Y_pred = list()
#     ignored_non_english = [0]
#     output_file_name = prepro.get_cache_file_name(
#         file_name=file_name, max_review=max_review,
#         as_sentence=logreg)
#
#     with open(output_file_name) as f:
#         index = 0
#         for line in f:
#             pp_review_dict = json.loads(line)
#             star_rating = pp_review_dict['star_rating']
#             pre_processed_review = pp_review_dict['pre_processed_review']
#             process_one_review_pp(
#                 star_rating=star_rating,
#                 pre_processed_review=pre_processed_review,
#                 ignored_non_english=ignored_non_english,
#                 Y_true=Y_true, Y_pred=Y_pred, index=index,
#                 negative_reviews_as_positive=False,
#                 threshold=threshold)
#             if max_review is not None and index == max_review-1:
#                 break
#             index+=1
#     print_metrics(Y_true=Y_true, Y_pred=Y_pred)

def get_f1_average(Y_true, Y_pred):
    f1_pos = f1_score(Y_true, Y_pred)
    f1_neg = f1_score(Y_true, Y_pred, pos_label=0)
    return (f1_pos + f1_neg)/2

def process_reviews(reviews, y, max_review=None, threshold=1.3, test=False):
    Y_true = list()
    Y_pred = list()
    ignored_non_english = [0]

    for ind, pre_processed_review in enumerate(reviews):
        original_classification = y[ind]
        process_one_review_pp(
            original_classification=original_classification,
            pre_processed_review=pre_processed_review,
            ignored_non_english=ignored_non_english,
            Y_true=Y_true, Y_pred=Y_pred, index=ind,
            negative_reviews_as_positive=False,
            threshold=threshold)
        if max_review is not None and ind == max_review - 1:
            break


    if test:
        print_metrics(Y_true=Y_true, Y_pred=Y_pred)

    # return get_f1_average(Y_true=Y_true, Y_pred=Y_pred)
    return roc_auc_score(Y_true, Y_pred)

def process_one_review_pp(
        original_classification, pre_processed_review, ignored_non_english,
        Y_true, Y_pred, index=None, negative_reviews_as_positive=False, threshold=1.3):

    # IGNORE OTHER LANGUAGES
    if pre_processed_review == []:
        ignored_non_english[0] += 1
        return

    baseline_classification, final_score = get_baseline_classification(
        review=pre_processed_review,
        negative_reviews_as_positive=negative_reviews_as_positive,
        threshold=threshold, index=index)

    Y_true.append(original_classification)
    Y_pred.append(baseline_classification)
    if original_classification == 0 and baseline_classification == 1:
        pass
    # set_trace()
# def process_one_review_pp2(
#         star_rating, pre_processed_review, ignored_non_english,
#         Y_true, Y_pred, index=None, negative_reviews_as_positive=False, threshold=1.3):
#     # IGNORE NEUTRAL RATINGS
#     if int(star_rating) == 3:
#         return
#     # IGNORE OTHER LANGUAGES
#     if pre_processed_review == []:
#         ignored_non_english[0] += 1
#         return
#
#     original_classification = get_classification_group_for_star_rating(
#         star_rating=star_rating,
#         negative_reviews_as_positive=negative_reviews_as_positive)
#
#     baseline_classification, final_score = get_baseline_classification(
#             review=pre_processed_review,
#         negative_reviews_as_positive=negative_reviews_as_positive,
#         threshold=threshold, index=index)
#
#     Y_true.append(original_classification)
#     Y_pred.append(baseline_classification)
#     if original_classification == 0 and baseline_classification == 1:
#         set_trace()

def get_Y_classes():
    Y_classes = np.array(['Bad', 'Good'])
    return Y_classes


def get_baseline_classification(
        review, negative_reviews_as_positive, threshold, index):
    baseline_score_pos, baseline_score_neg  = get_sentiment_scores(
        review=review, index=index)
    baseline_score_neg = max(0.001, baseline_score_neg)
    final_score = baseline_score_pos/baseline_score_neg

    if final_score >threshold:
        category = 1 if not negative_reviews_as_positive else 0
    else:
        category = 0 if not negative_reviews_as_positive else 1

    return category, final_score

CACHE_sentiment_scores = {}
def get_sentiment_scores(review, index):
    if index is None or index not in CACHE_sentiment_scores:
        total_positive = 0
        total_negative = 0
        for word_pos_tuple in review:
            positive_score, negative_score = get_sentiment_score_word_pos_tuple(
                word_pos_tuple=word_pos_tuple)
            total_positive += positive_score
            total_negative += negative_score
            # if positive_score != 0 or negative_score != 0:
            #     print("word: %s, positive:%s ,negative:%s"%(
            #         word_pos_tuple, positive_score, negative_score))
        # CACHE_sentiment_scores[index] = total_positive, total_negative

    else:
        total_positive, total_negative = CACHE_sentiment_scores[index]
    return (total_positive, total_negative)


POS_not_found = set()
def get_sentiment_score_word_pos_tuple(word_pos_tuple):
    word = word_pos_tuple[0]
    pos = word_pos_tuple[1]
    pos_or_neg = word_pos_tuple[2]
    if word_pos_tuple[:2]==('like', 'ADP'):
        return 0, 0
    if not pos_is_in_model(pos):
        if pos not in POS_not_found:
            print("WARNING: pos not recognized: %s"%pos)
            POS_not_found.add(pos)
        return 0, 0

    word_synsets = get_synsets(word=word, pos=pos)
    # set_trace()
    pos_sentiment_score, neg_sentiment_score = \
        calculate_avg_sentiment_scores_for_synsets(word_synsets=word_synsets)

    if pos_or_neg == 'pos':
        return pos_sentiment_score , neg_sentiment_score
    else:
        return neg_sentiment_score, pos_sentiment_score

def get_synsets(word, pos):
    wordnet_pos = transform_pos_to_wordnet_notation(pos=pos)
    word_synsets = wn.synsets(word, wordnet_pos)

    if word_synsets == [] and wordnet_pos == 'n':
        word_synsets += wn.synsets(word, 'a')

    return word_synsets

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
    # set_trace()
    return max(pos_sentiment_score_array), max(neg_sentiment_score_array)
    # return np.mean(pos_sentiment_score_array), np.mean(neg_sentiment_score_array)

CACHE_sentiment_scores_for_synset = {}
def calcualte_sentiment_scores_for_synset(synset):
    synset_name = synset.name()
    if synset_name not in CACHE_sentiment_scores_for_synset:
        word_sentiment = swn.senti_synset(synset_name)
        pos_score = word_sentiment.pos_score()
        neg_score = word_sentiment.neg_score()
        CACHE_sentiment_scores_for_synset[synset_name] = pos_score, neg_score
    else:
        pos_score, neg_score = CACHE_sentiment_scores_for_synset[synset_name]
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

def get_best_threshold(rev_train, y_train, max_review):
    threshold_list = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    f1_min = 0
    best_threshold = threshold_list[0]
    all_f1_scores = list()
    for threshold in threshold_list:

        f1_score_ = process_reviews(
            reviews=rev_train, y=y_train,
            max_review=max_review, threshold=threshold)
        all_f1_scores.append(f1_score_)
        if f1_score_>f1_min:
            f1_min = f1_score_
            best_threshold = threshold


    # set_trace()
    print("***************")
    print("BEST THRESHOLD: %s" % str(best_threshold))
    return best_threshold


if __name__=="__main__":
    # LOAD THE REVIEWS AND THE CORRESPONDING RATING
    file_name = get_file_name_from_sys_arg(sys_argv=sys.argv)
    star_rating_list, review_list = load_csv_info(file_name)
    nlp = spacy.load("en_core_web_sm")
    max_review = int(sys.argv[2]) if len(sys.argv) > 2 else None
    in_parallel = True if (len(sys.argv) > 3 and sys.argv[3] == "parallel") else False
    #Get distribution
    get_star_distribution(star_rating_list=star_rating_list, max_review=max_review)
    # PRE PROCESS REVIEWS

    prepro.pre_process_all_reviews(
        file_name=file_name, max_review=max_review, nlp=nlp, as_sentence=False,
        in_parallel=in_parallel)
    # PROCESS REVIEWs
    # threshold_list = [1, 1.1, 1.2, 1.3, 1.4 , 1.5 , 1.6, 1.7, 1.8]
    # for threshold in threshold_list:
    #     print("***************")
    #     print("THRESHOLD: %s"%str(threshold))
    #     process_reviews(
    #         file_name=file_name, logreg=False,
    #         max_review=max_review, threshold=threshold)

    Y, review_list = prepro.get_reviews_as_list_from_cache(
        file_name=file_name, max_review=max_review, as_sentence=False,
        negative_reviews_as_positive=False)

    rev_train, rev_test, y_train, y_test = train_test_split(
        review_list, Y, train_size=2/3
    )
    best_threshold = get_best_threshold(
        rev_train=rev_train, y_train=y_train,max_review=max_review)

    process_reviews(
        reviews=rev_test, y=y_test, max_review=max_review,
        threshold=best_threshold, test=True)

