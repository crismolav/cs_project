from helpers import load_csv_info, load_text, split_reviews,\
    get_star_distribution, get_classification_group_for_star_rating,\
    print_metrics, get_file_name_from_sys_arg
import baseline as bl
import preprocess as prepro
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import json
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_score, recall_score,\
    accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict
import sys
import numpy as np
from pdb import set_trace

def find_incorrectly_classified_review(Y_true, Y_pred, review_list):
    different = []
    output_file_name =prepro.get_cache_file_name(
        file_name=file_name, max_review=max_review, as_sentence=True, misclassified=True)
    g = open(output_file_name, 'w')

    for ind, review in enumerate(review_list):
        if Y_true[ind] != Y_pred[ind]:
            review_dict = {
                'Y_true': str(Y_true[ind]),
                'Y_pred': str(Y_pred[ind]),
                'pre_processed_review':review
            }
            different.append(review)

            g.write(json.dumps(review_dict) + '\n')

    g.close()

def fit_logreg(file_name, max_review, negative_reviews_as_positive=False):
    cv = CountVectorizer(
        binary=True,
        ngram_range=(1,2),
        stop_words='english'
    )

    Y, review_list = prepro.get_reviews_as_list_from_cache(
        file_name=file_name, max_review=max_review, as_sentence=True,
        negative_reviews_as_positive=negative_reviews_as_positive)

    cv.fit(review_list)
    X = cv.transform(review_list)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     #     X, Y, train_size=2/3
    #     # )
    rev_train, rev_test, y_train, y_test = train_test_split(
        review_list, Y, train_size=2/3
    )
    X_train = cv.transform(rev_train)
    X_test = cv.transform(rev_test)
    C_list = [0.01, 0.05, 0.25, 0.5, 1]
    # C_list = [0.01, 0.05]
    # for c in C_list:
    lr = LogisticRegressionCV(Cs=C_list, cv=10, class_weight='balanced')
    lr.fit(X_train, y_train)
    # lr.fit(X_train, y_train)
    # y_pred = cross_val_predict(lr, X, Y, cv=5)
    print("*************")
    # print_metrics(Y_true=y_val, Y_pred=lr.predicst(X_val))
    Y_pred = lr.predict(X_test)
    print_metrics(Y_true=y_test, Y_pred=Y_pred)
    print_best_positive_and_negative(cv=cv, lr=lr)
    find_incorrectly_classified_review(Y_true=y_test, Y_pred=Y_pred, review_list=rev_test)

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
            reverse=True)[:30]:
        print(best_positive)
    print("Negative")
    for best_negative in sorted(
            feature_to_coef.items(),
            key=lambda x: x[1])[:30]:
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


if __name__=="__main__":
    # LOAD THE REVIEWS AND THE CORRESPONDING RATING
    file_name = get_file_name_from_sys_arg(sys_argv=sys.argv)
    star_rating_list, review_list = load_csv_info(file_name)
    nlp = spacy.load("en_core_web_sm")
    max_review = int(sys.argv[2]) if len(sys.argv) > 2 else None
    in_parallel = True if (len(sys.argv) > 3 and sys.argv[3] == "parallel") else False
    logreg = True
    #Get distribution
    get_star_distribution(star_rating_list=star_rating_list, max_review=max_review)
    # PRE PROCESS REVIEWS

    prepro.pre_process_all_reviews(
        file_name=file_name, max_review=max_review, nlp=nlp, as_sentence=True,
        in_parallel=in_parallel)
    # FIT LOG REG
    fit_logreg(file_name=file_name, max_review=max_review)