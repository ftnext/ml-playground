from unittest import TestCase

import numpy as np
from lime.lime_text import IndexedString


class IndexedStringDefaultRegexTestCase(TestCase):
    def setUp(self):
        self.raw_string = "This is a good movie. This, it is a great movie."
        self.expected_tokens = [
            "This",
            " ",
            "is",
            " ",
            "a",
            " ",
            "good",
            " ",
            "movie",
            ". ",
            "This",
            ", ",
            "it",
            " ",
            "is",
            " ",
            "a",
            " ",
            "great",
            " ",
            "movie",
            ".",
        ]

    def test_raw(self):
        actual = IndexedString(self.raw_string)

        self.assertEqual(actual.raw, self.raw_string)

    def test_as_list(self):
        actual = IndexedString(self.raw_string)

        self.assertEqual(actual.as_list, self.expected_tokens)

    def test_as_np(self):
        expected_token_array = np.array(self.expected_tokens)

        actual = IndexedString(self.raw_string)

        np.testing.assert_array_equal(actual.as_np, expected_token_array)

    def test_inverse_vocab(self):
        expected_vocab = ["This", "is", "a", "good", "movie", "it", "great"]

        actual = IndexedString(self.raw_string)

        self.assertEqual(actual.inverse_vocab, expected_vocab)

    def test_num_words(self):
        indexed_string = IndexedString(self.raw_string)

        actual = indexed_string.num_words()

        self.assertEqual(actual, 7)

    def test_inverse_removing(self):
        indexed_string = IndexedString(self.raw_string)

        actual = indexed_string.inverse_removing([0])  # remove 'This'

        removed_this = " is a good movie. , it is a great movie."
        self.assertEqual(actual, removed_this)

        actual = indexed_string.inverse_removing(np.array([3, 6]))

        removed_good_and_great = "This is a  movie. This, it is a  movie."
        self.assertEqual(actual, removed_good_and_great)
