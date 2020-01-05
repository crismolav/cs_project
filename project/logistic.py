from helpers import load_csv_info, load_text, split_reviews,\
    get_star_distribution, get_classification_group_for_star_rating, print_metrics
import baseline as bl
import preprocess as prepro
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score,\
    accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import sys
from pdb import set_trace

def process_one_review_lr(
        star_rating, pre_processed_review, ignored_non_english,
        Y_true, Y_pred, index=None, negative_reviews_as_positive=False, threshold=1.3):
    # IGNORE NEUTRAL RATINGS
    if int(star_rating) == 3:
        return
    # IGNORE OTHER LANGUAGES
    if pre_processed_review == []:
        ignored_non_english[0] += 1
        return

    original_classification = get_classification_group_for_star_rating(
        star_rating=star_rating,
        negative_reviews_as_positive=negative_reviews_as_positive)

    baseline_classification, final_score = get_baseline_classification(
        review=pre_processed_review,
        negative_reviews_as_positive=negative_reviews_as_positive,
        threshold=threshold, index=index)

    Y_true.append(original_classification)
    Y_pred.append(baseline_classification)
    # if original_classification == 0 and baseline_classification == 1:
    #     set_trace()
def fit_logreg(file_name, max_review, negative_reviews_as_positive=False):
    cv = CountVectorizer(
        binary=True,
        ngram_range=(1,1),
        stop_words='english'
    )

    Y, review_list = get_reviews_as_list_from_cache(file_name, max_review, negative_reviews_as_positive)

    cv.fit(review_list)
    X = cv.transform(review_list)
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, train_size=0.75
    )
    # C_list = [0.01, 0.05, 0.25, 0.5, 1]
    C_list = [0.01, 0.05]
    for c in C_list:
        lr = LogisticRegression(C=c, class_weight='balanced')
        lr.fit(X_train, y_train)
        print("*************")
        print_metrics(Y_true=y_val, Y_pred=lr.predict(X_val))
        print_best_positive_and_negative(cv=cv, lr=lr)

def get_reviews_as_list_from_cache(file_name, max_review, negative_reviews_as_positive):
    cache_file_name = prepro.get_cache_file_name(
        file_name=file_name, max_review=max_review, as_sentence=True)
    review_list = []
    Y = []
    with open(cache_file_name) as f:
        for line in f:
            pp_review_dict = json.loads(line)
            star_rating = pp_review_dict['star_rating']
            pre_processed_review = pp_review_dict['pre_processed_review']
            # IGNORE NEUTRAL RATINGS
            if int(star_rating) == 3:
                continue
            # TODO: check this
            # IGNORE OTHER LANGUAGES
            if pre_processed_review in ['', []]:
                continue
            classification = get_classification_group_for_star_rating(
                star_rating=star_rating,
                negative_reviews_as_positive=negative_reviews_as_positive)
            review_list.append(pre_processed_review)
            Y.append(classification)

    return Y, review_list

def print_best_positive_and_negative(cv, lr):
    feature_to_coef = {
        word: coef for word, coef in zip(
            cv.get_feature_names(), lr.coef_[0]
        )
    }
    print("Positive")
    for best_positive in sorted(
            feature_to_coef.items(),
            key=lambda x: x[1],
            reverse=True)[:20]:
        print(best_positive)
    print("Negative")
    for best_negative in sorted(
            feature_to_coef.items(),
            key=lambda x: x[1])[:20]:
        print(best_negative)
    print("*************")
# def fit_logreg2(file_name, max_review, negative_reviews_as_positive=False):
#     cv = TfidfVectorizer(
#         sublinear_tf=True,
#         strip_accents='unicode',
#         analyzer='word',
#         token_pattern=r'\w{1,}',
#         ngram_range=(1, 2),
#         )
#
#     cache_file_name = prepro.get_cache_file_name(
#         file_name=file_name, max_review=max_review, as_sentence=True)
#     review_list = []
#     Y = []
#     with open(cache_file_name) as f:
#         for line in f:
#             pp_review_dict = json.loads(line)
#             star_rating = pp_review_dict['star_rating']
#             pre_processed_review = pp_review_dict['pre_processed_review']
#             # IGNORE NEUTRAL RATINGS
#             if int(star_rating) == 3:
#                 continue
#             # TODO: check this
#             # IGNORE OTHER LANGUAGES
#             if pre_processed_review in ['', []]:
#                 continue
#             classification = get_classification_group_for_star_rating(
#                 star_rating=star_rating,
#                 negative_reviews_as_positive=negative_reviews_as_positive)
#             review_list.append(pre_processed_review)
#             Y.append(classification)
#
#     cv.fit(review_list)
#     X = cv.transform(review_list)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, Y, train_size=0.75
#     )
#     #C_list = [0.01, 0.05, 0.25, 0.5, 1]
#     C_list = [0.01, 0.05]
#     for c in C_list:
#         lr = LogisticRegression(C=c, class_weight='balanced')
#         lr.fit(X_train, y_train)
#         print("*************")
#         print_metrics(Y_true=y_val, Y_pred=lr.predict(X_val))
#         # print("Accuracy for C=%s: %s"
#         #       % (c, accuracy_score(y_val, lr.predict(X_val))))
#
#         feature_to_coef = {
#             word: coef for word, coef in zip(
#                 cv.get_feature_names(), lr.coef_[0]
#             )
#         }
#         print("Positive")
#         for best_positive in sorted(
#                 feature_to_coef.items(),
#                 key=lambda x: x[1],
#                 reverse=True)[:20]:
#             print(best_positive)
#         print("Negative")
#         for best_negative in sorted(
#                 feature_to_coef.items(),
#                 key=lambda x: x[1])[:20]:
#             print(best_negative)
#         print("*************")
def experiment():

    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    vectorizer =TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2),
        )
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())

    print(X.shape)

def get_file_name_from_sys_arg(sys_argv):
    if sys_argv[1] == 'books':
        file_name = "Books_50000.tsv"
    elif sys_argv[1] == 'video_games':
        file_name = "Video_Games_50000.tsv"
    elif sys_argv[1] == 'beauty':
        file_name = "Beauty_50000.tsv"
    else:
        raise Exception("unknown file tipe")
    return file_name

if __name__=="__main__":
    # LOAD THE REVIEWS AND THE CORRESPONDING RATING
    file_name = get_file_name_from_sys_arg(sys_argv=sys.argv)
    star_rating_list, review_list = load_csv_info(file_name)
    #Get distribution
    get_star_distribution(star_rating_list)
    nlp = spacy.load("en_core_web_sm")
    max_review = int(sys.argv[2]) if len(sys.argv) > 2 else None
    in_parallel = True if (len(sys.argv) > 3 and sys.argv[3] == "parallel") else False

    logreg = True
    # PRE PROCESS REVIEWS
    prepro.pre_process_all_reviews(
        file_name=file_name, max_review=max_review, nlp=nlp, as_sentence=True,
        in_parallel=in_parallel)
    # FIT LOG REG
    fit_logreg(file_name=file_name, max_review=max_review)