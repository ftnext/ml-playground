from unittest import TestCase

import numpy as np
from lime.lime_text import IndexedString, LimeTextExplainer


class DataLabelsDistancesTestCase(TestCase):
    @staticmethod
    def classifier_fn(strings):
        return np.array(
            [
                DataLabelsDistancesTestCase._classifier_fn(string)
                for string in strings
            ]
        )

    @staticmethod
    def _classifier_fn(string):
        if string == "Please, take your time. Please":
            return [0.9, 0.1]
        if string == "Please,   . Please":
            return [0.6, 0.4]
        if string == ",   . ":
            return [0.2, 0.8]
        return ValueError("想定していないstringが渡っています")

    def setUp(self):
        self.explainer = LimeTextExplainer(random_state=42)
        self.indexed_string = IndexedString("Please, take your time. Please")

    def test_data_is_binary(self):
        data, _, _ = self.explainer._LimeTextExplainer__data_labels_distances(
            self.indexed_string, self.classifier_fn, num_samples=3
        )

        # random_state=42のとき random_state.randint(1, 5, 2)は
        # array([3, 4]) を返す
        expected = np.array(
            [
                [1, 1, 1, 1],
                # random_state.choice(features_range, 3, replace=False) は
                # array([1, 3, 2])を返す
                [1, 0, 0, 0],
                # random_state.choice(features_range, 4, replace=False)
                [0, 0, 0, 0],
            ]
        )
        np.testing.assert_array_equal(data, expected)

    def test_label(self):
        (
            _,
            labels,
            _,
        ) = self.explainer._LimeTextExplainer__data_labels_distances(
            self.indexed_string, self.classifier_fn, num_samples=3
        )

        expected = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
        np.testing.assert_array_equal(labels, expected)

    def test_distances(self):
        (
            _,
            _,
            distances,
        ) = self.explainer._LimeTextExplainer__data_labels_distances(
            self.indexed_string, self.classifier_fn, num_samples=3
        )

        expected = np.array([0, 0.5, 1]) * 100
        np.testing.assert_array_equal(distances, expected)
