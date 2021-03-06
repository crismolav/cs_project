import numpy as np
from sklearn.metrics import precision_score, recall_score,\
    accuracy_score, f1_score, confusion_matrix, roc_auc_score
from pdb import set_trace
column_separator = "\t"

def load_csv_info(fname, max_review=None, min_review=None):
    star_rating_list = []
    review_body_list = []
    file=open(fname)
    file.readline()
    for line in file.readlines():
        marketplace, customer_id, review_id, product_id, product_parent, product_title, product_category, star_rating, helpful_votes, total_votes, vine, verified_purchase, review_headline, review_body, review_date = line.strip().split(column_separator)
        star_rating_list.append(star_rating)
        review_body_list.append(review_body)
    if max_review is not None:
        star_rating_list = star_rating_list[:max_review]
        review_body_list = review_body_list[:max_review]

    if min_review is not None:
        star_rating_list = star_rating_list[min_review:]
        review_body_list = review_body_list[min_review:]

    return  star_rating_list, review_body_list


def load_text(filename):
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    return text


def split_reviews(star_rating_list, review):
    reviews_45 = []
    star_45 = []
    reviews_12 = []
    star_12 = []
    reviews1245 = []
    star1245 = []

    for ii, star in enumerate(star_rating_list):
        current_review = review[ii]
        if star == '4' or star == '5':
            reviews_45.append(current_review)
            star_45.append(star)
            reviews1245.append(current_review)
            star1245.append(star)
        elif star == '1' or star == '2':
            reviews_12.append(current_review)
            star_12.append(star)
            reviews1245.append(current_review)
            star1245.append(star)

    return reviews_12, reviews_45, reviews1245, star_12, star_45, star1245



def get_star_distribution(star_rating_list, max_review=None):
    filtered_list = star_rating_list[:max_review] if max_review is not None else star_rating_list
    star_rating = np.array(filtered_list)
    unique, counts = np.unique(star_rating, return_counts=True)
    for i, type in enumerate(unique):
        print("class %s count:%s"%(type, counts[i]))


def get_classification_group_for_star_rating(
        star_rating, negative_reviews_as_positive=False):
    if not negative_reviews_as_positive:
        return get_classification_group_for_star_rating_positive_case(
        star_rating=star_rating)
    else:
        return not get_classification_group_for_star_rating_positive_case(
        star_rating=star_rating)

def get_classification_group_for_star_rating_positive_case(star_rating):
    if int(star_rating) in [4, 5]:
        return 1
    elif int(star_rating) in [1, 2]:
        return 0
    else:
        raise Exception("can't process star_rating number:%s"%star_rating)

def print_metrics(Y_true, Y_pred):
    bl_precision = precision_score(Y_true, Y_pred)
    bl_recall = recall_score(Y_true, Y_pred)
    bl_accuracy = accuracy_score(Y_true, Y_pred)
    bl_f1_score = f1_score(Y_true, Y_pred)
    bl_roc_score = roc_auc_score(Y_true, Y_pred)
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
    print("Accuracy: %s" % bl_accuracy)
    print("F1 score: %s" % bl_f1_score)
    print("ROC score: %s" % bl_roc_score)

def get_Y_classes():
    Y_classes = np.array(['Bad', 'Good'])
    return Y_classes

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
        'CONJ': '',
        'X': '',

    }
    if pos in pos_to_wordnet_pos_dictionary:
        return pos_to_wordnet_pos_dictionary[pos]
    else:
        return ''

def get_file_name_from_sys_arg(sys_argv):
    if sys_argv[1] == 'books':
        file_name = "Books_50000.tsv"
    elif sys_argv[1] == 'video_games':
        file_name = "Video_Games_50000.tsv"
    elif sys_argv[1] == 'beauty':
        file_name = "Beauty_50000.tsv"
    elif sys_argv[1] == 'mobile_apps':
        file_name = "Mobile_Apps_50000.tsv"
    elif sys_argv[1] == 'toys':
        file_name = "Toys_50000.tsv"
    else:
        raise Exception("unknown file tipe")
    return file_name

