import unittest
import spacy
import preprocess as pp
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

if __name__ == '__main__':
    unittest.main()