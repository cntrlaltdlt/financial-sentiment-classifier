import tempfile
import unittest
from pathlib import Path

import pandas as pd

from financial_sentiment_classifier.data_collection import (
    check_trainable, clean_data, compute_class_weights, load_csv_data, make_label_maps, split_data)


def balanced_frame(per_class=20):
    rows = [(f"{label} sentence number {i}", label)
            for label in ("negative", "neutral", "positive") for i in range(per_class)]
    return pd.DataFrame(rows, columns=["text", "label"])


class CleaningTests(unittest.TestCase):
    def test_removes_duplicates_ignoring_case_and_spacing(self):
        df = pd.DataFrame({"text": ["Great product", "great   product ", "Bad"],
                           "label": ["positive", "positive", "negative"]})
        cleaned, report = clean_data(df)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["duplicates_removed"], 1)

    def test_drops_sentences_with_conflicting_labels(self):
        df = pd.DataFrame({"text": ["It is fine", "it is fine", "Awful"],
                           "label": ["neutral", "positive", "negative"]})
        cleaned, report = clean_data(df)
        self.assertEqual(cleaned["text"].tolist(), ["Awful"])
        self.assertEqual(report["conflicting_sentences_removed"], 2)

    def test_drops_blank_rows_and_renames_columns(self):
        df = pd.DataFrame({"review": ["Good", "", None, "Poor"], "stars": ["pos", "neg", "neg", None]})
        cleaned, report = clean_data(df, "review", "stars")
        self.assertEqual(list(cleaned.columns), ["text", "label"])
        self.assertEqual(cleaned["text"].tolist(), ["Good"])
        self.assertEqual(report["blank_rows_removed"], 3)

    def test_heavily_repeated_data_collapses_to_its_distinct_sentences(self):
        df = pd.DataFrame({"text": ["Love it"] * 30 + ["Hate it"] * 30, "label": ["pos"] * 30 + ["neg"] * 30})
        cleaned, _ = clean_data(df)
        self.assertEqual(len(cleaned), 2)
        with self.assertRaises(ValueError):
            check_trainable(cleaned)


class SplitTests(unittest.TestCase):
    def test_no_sentence_in_two_sets_and_reproducible(self):
        df = balanced_frame()
        train, val, test = split_data(df, val_size=0.15, test_size=0.15, seed=1)
        sets = [set(part["text"]) for part in (train, val, test)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(map(len, sets)), len(df))
        again = split_data(df, val_size=0.15, test_size=0.15, seed=1)
        self.assertEqual(train["text"].tolist(), again[0]["text"].tolist())

    def test_split_is_stratified(self):
        train, val = split_data(balanced_frame(), val_size=0.25, seed=0)
        self.assertEqual(val["label"].value_counts().nunique(), 1)  # equal counts per label


class LabelAndWeightTests(unittest.TestCase):
    def test_label_maps_are_sorted_and_inverse(self):
        label2id, id2label = make_label_maps(["positive", "negative", "neutral", "negative"])
        self.assertEqual(label2id, {"negative": 0, "neutral": 1, "positive": 2})
        self.assertEqual(id2label[2], "positive")

    def test_class_weights_are_inverse_frequency(self):
        labels = ["neutral"] * 6 + ["positive"] * 3 + ["negative"] * 1
        label2id, _ = make_label_maps(labels)
        weights = dict(zip(label2id, compute_class_weights(labels, label2id)))
        self.assertAlmostEqual(weights["negative"], 10 / (3 * 1))
        self.assertAlmostEqual(weights["neutral"], 10 / (3 * 6))
        self.assertGreater(weights["negative"], weights["positive"])


class LoadingTests(unittest.TestCase):
    def test_missing_column_names_the_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            pd.DataFrame({"review": ["x"], "label": ["pos"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing column\\(s\\): text"):
                load_csv_data(path)


if __name__ == "__main__":
    unittest.main()
