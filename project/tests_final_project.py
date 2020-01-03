import unittest
import spacy
import preprocess as pp
import baseline as bl
from pdb import set_trace
from unittest.mock import patch

nlp = spacy.load("en_core_web_sm")
class FinalTests(unittest.TestCase):
    def test_identify_negation__dont_like(self):
        sentence = "I don't like it"
        result = pp.get_sentence_with_negation_mark(sentence=sentence, nlp=nlp)
        expected_list = "I don't like_not it"

        self.assertEqual(expected_list, result)

    def test_identify_negation__be_happy(self):
        sentence = "I am not happy"
        result = pp.get_sentence_with_negation_mark(sentence=sentence, nlp=nlp)
        expected_list = 'I am not happy_not'

        self.assertEqual(expected_list, result)

    def test_identify_negation__be_very_happy(self):
        sentence = "I am not very happy"
        result = pp.get_sentence_with_negation_mark(sentence=sentence, nlp=nlp)
        expected_list = 'I am not very happy_not'

        self.assertEqual(expected_list, result)

    def test_identify_negation__noun_phrase(self):
        sentence = "Very artistic quilts shown, no detailed instructions"
        result = pp.get_sentence_with_negation_mark(sentence=sentence, nlp=nlp)
        expected_list = 'Very artistic quilts shown, no detailed_not instructions_not'

        self.assertEqual(expected_list, result)

    def test_identify_negation__noun_phrase__two_verbs_negated(self):
        sentence = "Like other reviewers said, if you like Eric, you won't like this book"
        result = pp.get_sentence_with_negation_mark(sentence=sentence, nlp=nlp)
        expected_list = "Like other reviewers said, if you like_not Eric, you won't like_not this book"

        self.assertEqual(expected_list, result)

    def test_remove_remove_extra_spaces__real_case(self):
        sentence = 'Lawrence Block is novels  and he has had several series using different characters  are never particularly actionfilled'

        result = pp.remove_extra_spaces(sentence=sentence)
        expected = 'Lawrence Block is novels and he has had several series using different characters are never particularly actionfilled'

        self.assertEqual(expected, result)

    def test_remove_remove_extra_spaces__three_spaces(self):
        sentence = 'Lawrence Block is novels   and he has had several series'

        result = pp.remove_extra_spaces(sentence=sentence)
        expected = 'Lawrence Block is novels and he has had several series'

        self.assertEqual(expected, result)

    def tests_is_english__true(self):
        sentence = 'The student curriculum was better than I expected'

        result = pp.is_english(sentence=sentence)

        self.assertTrue(result)

    def tests_is_english__false(self):
        sentence = 'Buen libro, excelente servicio de entrega'

        result = pp.is_english(sentence=sentence)

        self.assertFalse(result)

    def test_is_english__true_harder(self):
        sentence = 'Writing steamy dramafilled novels should be Kendall Banks trademark'

        result = pp.is_english(sentence=sentence)

        self.assertTrue(result)

    def test_is_english__false_short(self):
        sentence = 'Doble partida, doble talento.'

        result = pp.is_english(sentence=sentence)

        self.assertFalse(result)

    def test_get_sentiment_score_word_pos_tuple__positive_case(self):
        word_pos_tuple = ('good', 'ADV', 'pos')

        result = bl.get_sentiment_score_word_pos_tuple(
            word_pos_tuple=word_pos_tuple)
        expected = (0.1875, 0.0)

        self.assertEqual(expected, result)

    def test_get_sentiment_score_word_pos_tuple__negation_case(self):
        word_pos_tuple = ('good', 'ADV', 'neg')

        result = bl.get_sentiment_score_word_pos_tuple(
            word_pos_tuple=word_pos_tuple)
        expected = (0.0, 0.1875)

        self.assertEqual(expected, result)

    def test_get_sentiment_score_word_pos_tuple__like_pos(self):
        word_pos_tuple = ('like', 'ADP', 'pos')

        result = bl.get_sentiment_score_word_pos_tuple(
            word_pos_tuple=word_pos_tuple)
        expected = (0.0, 0.0)

        self.assertEqual(expected, result)

    def test_process_one_review__negative(self):
        star_rating = 2
        Y_true = []
        Y_pred = []
        ignored_non_english = [0]
        #TODO good study case for the word worship
        review = "I'm sorry, but calling Jesus the messiah is as kosher as eating a ham " \
                 "and cheese sandwich on Yom Kippur.  Bringing about world peace, " \
                 "having the world worship the same G-d, and the return of the Jewish diaspora" \
                 " back to the land of Israel were hardly accomplished by who the Christians" \
                 " believe to be the Messiah.  If he truly was Moshiach," \
                 " then none of these discussions would be taking place.<br /><br />" \
                 "When the Moshiach does come, it will end hatred and intolerance--" \
                 "unlike what happened when Jesus' followers tried to propagate their faith."
        pre_processed_review = pp.pre_process(text=review, nlp=nlp)

        bl.process_one_review_pp(
            star_rating=star_rating, pre_processed_review=pre_processed_review,
            ignored_non_english=ignored_non_english,
            Y_true=Y_true, Y_pred=Y_pred)
        expected = [0] , [0]

        self.assertEqual(expected, (Y_true, Y_pred))

    def test_tagset__correct_spelling(self):
        words = ['diappointed', 'book']
        result = pp.tagset(words=words)
        expected = [('disappointed', 'VERB', 'pos'), ('book', 'NOUN', 'pos')]

        self.assertEqual(expected, result)

    def test_process_one_review__negative_hard(self):
        star_rating = 2
        Y_true = []
        Y_pred = []
        ignored_non_english = [0]
        #TODO good study case for the word worship
        review = 'Very artistic quilts shown, no detailed instructions.  ' \
                 'More on the theory of making memory quilts.  ' \
                 'I was diappointed in the book. I had expected' \
                 ' moreinstructions on the actualy making the quilat'
        pre_processed_review = pp.pre_process(text=review, nlp=nlp)

        bl.process_one_review_pp(
            star_rating=star_rating, pre_processed_review=pre_processed_review,
            ignored_non_english=ignored_non_english,
            Y_true=Y_true, Y_pred=Y_pred)
        expected = [0] , [0]

        self.assertEqual(expected, (Y_true, Y_pred))

    def test_process_one_review__testing(self):
        star_rating = 2
        Y_true = []
        Y_pred = []
        ignored_non_english = [0]
        review = "CONTAINS SPOILERS!<br /><br />First off, when I have been a HUGE Sookie fan." \
                 "  I loved these books and have read them over and over." \
                 "  When I just picked up Deadlocked, I was like, what the heck!!!" \
                 "  This isn't nearly long enough!<br /><br />" \
                 "I felt like the book was filled with fluff and the plot line was thin." \
                 "  So much time was spent on Sookie doing chores. " \
                 " Perhaps the author wanted us to see her doing every day things?<br /><br" \
                 " />Like other reviewers said, if you like Eric, you won't like this book. " \
                 " I don't understand what happened to his character so fast. " \
                 " The majority of the series was built of the Sookie/Eric tension and then they got to a great relationship. " \
                 " I often found Sookie annoying, but Eric's character kept yanking me back with his humor and whit. " \
                 " This book makes him look terrible." \
                 "  I feel let down in that so many years were invested in getting to see him become a better person and love Sookie, " \
                 "just to have them appearing to go their separate ways.<br /><br />There was too much Bill in this book. " \
                 " We went books and books without Bill, and now he was in this book so much.<br /><br />" \
                 "I felt from the very beginning Sookie would end up with Sam." \
                 "  I really didn't want this to happen.  " \
                 "Couldn't the girl just have a close guy friend?" \
                 "  I was hoping she'd use the cluviel dor to make Eric human. " \
                 " Perhaps he wouldn't have wanted it, but if she was being such a baby about it," \
                 " why didn't she at least talk to him about it?<br /><br />Anyway," \
                 " I felt the author was really trying to wrap things up in this book," \
                 " getting ready for the last one next year." \
                 "  I thought things were too rushed and I don't like where things are going." \
                 "  The Fae left too quickly.. Anyways, this is my opinion," \
                 " I hope the next book is longer (without the fluff) and leaves us satisfied.." \
                 " but I'm thinking I'm going to be disappointed with that one as well."
        pre_processed_review = pp.pre_process(text=review, nlp=nlp)

        bl.process_one_review_pp(
            star_rating=star_rating, pre_processed_review=pre_processed_review,
            ignored_non_english=ignored_non_english,
            Y_true=Y_true, Y_pred=Y_pred)
        expected = [0], [0]

        # self.assertEqual(expected, (Y_true, Y_pred))

if __name__ == '__main__':
    unittest.main()