import re
import string
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, words
from nltk.probability import FreqDist
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
from pdb import set_trace
import spacy

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
def tagset(words):
    tag=nltk.pos_tag(words, tagset='universal')
    #ADD NEGATIVE OR POSITIVE INDICATOR && AUTOCORRECT
    add_negative_positive_to_tuple(tagset=tag)

    return tag


def add_negative_positive_to_tuple(tagset):
    for index, (word, pos) in enumerate(tagset):
        negative_or_positive = 'neg' if "_not" in word else 'pos'
        modified_tuple = (
            spell(word.replace("_not", "")), pos, negative_or_positive)
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
    for token in parsed_tree:
        if str(token.dep_) == 'neg' or token.lemma_ =='no':
            if token.i != 0 and not token.head.lemma_ in ['be']:
                negation_index_list.append(str(token.head))
                # negation_index_list.append(str(token.head)+'_not')
                neg_verb = token.head.lemma_

            elif token.i != 0 and token.head.lemma_ in ['be']:
                neg_verb = token.head.lemma_
        if token.dep_ in ['acomp', 'amod'] and token.head.lemma_ == neg_verb:
            negation_index_list.append(str(token))
    wt = word_tokenize(sentence)

    for negated_word in negation_index_list:
        if negated_word in wt:
            word_index = wt.index(negated_word)
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

def autocorrect_text(tag_words):
    new_tag_words = []
    for word, pos, neg_or_pos  in tag_words:
        corrected = spell(word)
        new_tuple = (corrected, pos, neg_or_pos)
        new_tag_words.append(new_tuple)

    return new_tag_words

def pre_process(text, nlp):
    #### REMOVE HTML TAGS ###
    no_html = cleanhtml(text)
    ### TOKENIZE SENTENCES
    sentences_tokens = sent_tokenize(no_html)
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
            return []
        ### ADD _neg ###
        with_neg = get_sentence_with_negation_mark(sentence=no_extra_spaces, nlp=nlp)
        # TOKENIZE
        wt = word_tokenize(with_neg)
        text_lc = text_lowercase(wt)
        # LOWERCASE
        # WITHOUT SOPTWORDS
        no_sw = sw(text_lc)
        no_numbers=replace_numbers(no_sw)
        # TAG & AUTOCORRECT
        tag_words=tagset(no_numbers)

        preprocessed_list+=tag_words

    return preprocessed_list

