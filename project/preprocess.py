import re
import string
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, words
from nltk.probability import FreqDist
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
from pdb import set_trace
import numpy as np
import json
import os.path
from helpers import load_csv_info, transform_pos_to_wordnet_notation, get_classification_group_for_star_rating
from os import listdir
from os.path import isfile, join
from shutil import copyfile

from multiprocessing import Pool


# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:43:45 2019

@author: Mar Adrian
"""

spell = Speller(lang='en')
#### REMOVE HTML TAGS ###
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext

### SEPARATE CONTRACT WORDS ### 
def decontracted(phrase):
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase

### PUNCTUATION ###
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

# LOWERCASE 
def text_lowercase(text):
    l = [item.lower() for item in text]
    return l


#WITHOUT SOPTWORDS
def sw(text):
    stop_en = set(stopwords.words('english'))
    new_stopwords = [ "n", "hes", "shes", "us", "im"]
    new_stopwords_list = stop_en.union(new_stopwords)
    no = [w for w in text if w not in new_stopwords_list]

    return no

    ### FREQUENCE OF THE WORD ###
def frequency(text):
    fdist=FreqDist(text)
    return fdist

### NUMBER INTO WORDS ###
def replace_numbers(words):
    import inflect
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


### TAGSET ###
def tagset(words, as_sentence=False):
    tag=nltk.pos_tag(words, tagset='universal')
    #ADD NEGATIVE OR POSITIVE INDICATOR && AUTOCORRECT
    # return [list(x) for x in tag]
    la_tagset = lemmatize_and_autocorrect_words(tagset=tag)
    if as_sentence:
        la_tagset = ' '.join([x[0] for x in la_tagset]) + '. '
    else:
        add_negative_positive_to_tuple(tagset=la_tagset)
    return la_tagset

def lemmatize_and_autocorrect_words(tagset):
    lemmatized_list = []
    lemmatizer = WordNetLemmatizer()
    for word_tuple in tagset:
        pos_simple = transform_pos_to_wordnet_notation(pos = word_tuple[1])

        if '_not' in word_tuple[0]:
            original_word= word_tuple[0].split('_')[0]
            tail = '_not'
        else:
            original_word = word_tuple[0]
            tail = ''
        autocorrected = spell(original_word)
        if pos_simple not in ['']:
            lemma = lemmatizer.lemmatize(autocorrected, pos_simple).lower()
        else:
            lemma = autocorrected

        final_word = lemma+tail
        new_tuple = (final_word, word_tuple[1])
        lemmatized_list.append(new_tuple)

    return lemmatized_list

def add_negative_positive_to_tuple(tagset):
    for index, (word, pos) in enumerate(tagset):
        negative_or_positive = 'neg' if "_not" in word else 'pos'
        modified_tuple = (
            word.replace("_not", ""), pos, negative_or_positive)
        tagset[index] = modified_tuple


# corrected = spell(word)
# new_tuple = (corrected, pos, neg_or_pos)
# new_tag_words.append(new_tuple)

def remove_extra_spaces(sentence):
    final_sentence = sentence
    double_space = '  '
    single_space = ' '
    while double_space in final_sentence:
        final_sentence = final_sentence.replace(double_space, single_space)

    return final_sentence

def get_sentence_with_negation_mark(sentence, nlp):
    parsed_tree = nlp(sentence)
    negation_index_list = []
    neg_verb = ''
    #TODO: instead of index save the word and then negate it. Other wise its difficutl
    lemmatized_list = []
    for token in parsed_tree:
        lemmatized_list.append(token.lemma_)
        if str(token.dep_) == 'neg' or token.lemma_ =='no':
            if token.i != 0 and not token.head.lemma_ in ['be']:
                negation_index_list.append(str(token.head))
                # negation_index_list.append(str(token.head)+'_not')
                neg_verb = token.head.lemma_

            elif token.i != 0 and token.head.lemma_ in ['be']:
                neg_verb = token.head.lemma_
        if token.dep_ in ['acomp', 'amod'] and token.head.lemma_ == neg_verb:
            negation_index_list.append(str(token))

    lemmatized_sentence = ' '.join(lemmatized_list)
    wt = word_tokenize(sentence)

    for negated_word in negation_index_list:
        if negated_word in wt:
            # set_trace()
            # word_index = wt.index(negated_word)
            word_indexes = np.where(np.array(wt) == negated_word)[0]
            for word_index in word_indexes:
                wt[word_index] += "_not"
    detokenized  = TreebankWordDetokenizer().detokenize(wt)
    return detokenized

def is_english(sentence):
    text_as_list = word_tokenize(sentence)
    max_num = min(10, len(text_as_list))
    threshold = int(max_num/1.5)
    english_words = [x for x in words.words()] + ['spoilers']
    foreign_words = 0
    for word_ind in range(max_num):
        word_lemma = WordNetLemmatizer().lemmatize(text_as_list[word_ind], 'v').lower()
        word_lemma_not_verb = WordNetLemmatizer().lemmatize(text_as_list[word_ind]).lower()
        if (word_lemma not in english_words
                and word_lemma_not_verb not in english_words):
            foreign_words+=1

    if foreign_words > threshold:
        return False
    else:
        return True

# def autocorrect_text(tag_words):
#     new_tag_words = []
#     for word, pos, neg_or_pos  in tag_words:
#         corrected = spell(word)
#         new_tuple = (corrected, pos, neg_or_pos)
#         new_tag_words.append(new_tuple)
#
#     return new_tag_words

def pre_process(text, nlp, as_sentence=False):
    #### REMOVE HTML TAGS ###
    no_html = cleanhtml(text)
    ### TOKENIZE SENTENCES
    sentences_tokens = sent_tokenize(no_html)
    if as_sentence:
        preprocessed_list = ''
    else:
        preprocessed_list = []

    for i, sentence in enumerate(sentences_tokens):
        ### REMOVE CONTRACTIONS ###
        no_contract=decontracted(sentence)
        ### PUNCTUATION ###
        no_puntuation = remove_punctuation(no_contract)
        ### REMOVE EXTRA SPACES ###
        no_extra_spaces = remove_extra_spaces(no_puntuation)
        ### RETURNS EMPTY IF LANGUAGE IS NOT ENGLISH
        if i ==0 and not is_english(sentence=no_extra_spaces):
            return preprocessed_list

        ### LEMMATIZE WORDS AND
        ### ADD _neg ###
        with_neg = get_sentence_with_negation_mark(sentence=no_extra_spaces, nlp=nlp)
        # TOKENIZE
        wt = word_tokenize(with_neg)
        text_lc = text_lowercase(wt)
        # LOWERCASE
        # WITHOUT SOPTWORDS
        if as_sentence:
            no_sw = text_lc
        else:
            no_sw = sw(text_lc)

        no_numbers=replace_numbers(no_sw)
        # TAG & AUTOCORRECT
        tag_words=tagset(no_numbers, as_sentence=as_sentence)

        preprocessed_list += tag_words

    return preprocessed_list


def pre_process_all_reviews(file_name, max_review, nlp, as_sentence=False, in_parallel=False):
    output_file_name = get_cache_file_name(
        file_name=file_name, max_review=max_review, as_sentence=as_sentence)
    #look in cache
    if os.path.isfile(output_file_name):
        print("CACHE hit: %s"%output_file_name)
        return
    #pre process reviews
    else:
        next_best_cache = look_for_next_best_cache(
            file_name, output_file_name, max_review,
            as_sentence=as_sentence, misclassified=False)
        min_review = None

        if next_best_cache is not None:
            print("CACHE next best: %s" % next_best_cache)
            copyfile(next_best_cache, output_file_name)
            min_review = get_cache_review_number(cache_file=next_best_cache)

        print("CACHE miss: %s"%output_file_name)

        pre_process_all_reviews_do_work(
            file_name=file_name, max_review=max_review, nlp=nlp,
            as_sentence=as_sentence, in_parallel=in_parallel,
            min_review=min_review)


def look_for_next_best_cache(file_name, output_file_name, max_review,
                             as_sentence=False, misclassified=False):
    subfolder = get_subfolder_name(file_name)
    folder = 'prepro_cache/%s'%(subfolder)
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    next_best_cache = get_best_files(
        onlyfiles=onlyfiles, subfolder=subfolder, max_review=max_review,
        as_sentence=as_sentence, misclassified=misclassified)
    if next_best_cache is None:
        return None
    return folder+'/'+next_best_cache

def get_best_files(onlyfiles, subfolder, max_review, as_sentence=False, misclassified=False):
    best_file = None
    max_num_reviews = 0
    file_extension = get_file_extension(
        as_sentence=as_sentence, misclassified=misclassified)

    for cache_file in onlyfiles:
        if cache_file.startswith(subfolder) and cache_file.split('.')[-1] == file_extension :
            review_number = get_cache_review_number(cache_file)
            if review_number>max_num_reviews and review_number<max_review:
                best_file = cache_file
                max_num_reviews = review_number

    return best_file

def get_cache_review_number(cache_file):
    return int(cache_file.split(".")[0].split("_")[-1])

def pre_process_all_reviews_do_work(file_name, max_review, nlp, as_sentence, in_parallel=False, min_review=None):
    output_file_name = get_cache_file_name(
        file_name=file_name, max_review=max_review, as_sentence=as_sentence)

    star_rating_list, review_list = load_csv_info(
        fname=file_name, max_review=max_review, min_review=min_review)
    write_or_append = 'a' if min_review is not None else 'w'
    f = open(output_file_name, write_or_append)

    if not in_parallel:
        for ind, review in enumerate(review_list):
            star_rating = star_rating_list[ind]
            review_processed = pre_process(text=review, nlp=nlp, as_sentence=as_sentence)
            review_dict = {
                'star_rating':star_rating,
                'pre_processed_review':review_processed
            }
            f.write(json.dumps(review_dict) + '\n')

    else:
        print("calculating in parallel")
        agents = 10
        # chunksize = int(len(review_list)/agents)
        dataset = zip(review_list,
                      [nlp]*len(review_list),
                      [as_sentence]*len(review_list))

        with Pool(processes=agents) as pool:
            results = pool.starmap(pre_process, dataset)

        for ind, result in enumerate(results):
            star_rating = star_rating_list[ind]
            review_dict = {
                'star_rating': star_rating,
                'pre_processed_review': result
            }
            f.write(json.dumps(review_dict) + '\n')
    f.close()

def get_cache_file_name(file_name, max_review=None, as_sentence=False,
                        misclassified=False):
    file_name_final = file_name.replace(".txt", "")
    file_name_final = file_name_final.replace(".tsv", "")
    file_name_final = '_'.join(file_name_final.split("_")[:-1])
    subfolder = file_name_final
    if max_review is not None:
        file_name_final += '_%s' % str(max_review)
    file_extension = get_file_extension(
        as_sentence=as_sentence, misclassified=misclassified)

    return 'prepro_cache/%s/%s.%s' % (subfolder, file_name_final, file_extension)

def get_subfolder_name(file_name):
    file_name_final = file_name.replace(".txt", "")
    file_name_final = file_name_final.replace(".tsv", "")
    file_name_final = '_'.join(file_name_final.split("_")[:-1])
    subfolder = file_name_final
    return subfolder

def get_file_extension(as_sentence, misclassified):
    if as_sentence:
        if not misclassified:
            return 'as_prepro'
        else:
            return 'as_miscl'
    else:
        if not misclassified:
            return 'prepro'
        else:
            return 'miscl'

def get_reviews_as_list_from_cache(file_name, max_review, as_sentence, negative_reviews_as_positive):
    cache_file_name = get_cache_file_name(
        file_name=file_name, max_review=max_review, as_sentence=as_sentence, misclassified=False)
    review_list = []
    Y = []
    with open(cache_file_name) as f:
        for line in f:
            pp_review_dict = json.loads(line)
            try:
                star_rating = pp_review_dict['star_rating']
                pre_processed_review = pp_review_dict['pre_processed_review']
            except:
                set_trace()
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