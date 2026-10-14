import unittest

import numpy as np
import torch

from sentiment_analyser.model_training import create_loader, fit, set_seed
from sentiment_analyser.model_validation import compute_metrics
from sentiment_analyser.models import TextDataset
from sentiment_analyser.prediction import predict_sentiment
from tests.helpers import StubTokeniser, tiny_model

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class MetricTests(unittest.TestCase):
    def test_known_predictions(self):
        truths = [0, 0, 1, 1, 2, 2]
        preds = [0, 1, 1, 1, 2, 0]
        m = compute_metrics(truths, preds, ID2LABEL)
        self.assertAlmostEqual(m["overall_accuracy"], 4 / 6)
        self.assertEqual(m["per_class_accuracy"]["negative"], {"accuracy": 0.5, "correct": 1, "total": 2})
        self.assertEqual(m["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        # F1 per class: negative 0.5, neutral 0.8, positive 2/3
        self.assertAlmostEqual(m["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3)

    def test_macro_f1_exposes_a_model_that_ignores_the_rare_class(self):
        truths = [1] * 9 + [0]
        preds = [1] * 10
        m = compute_metrics(truths, preds, {0: "negative", 1: "neutral"})
        self.assertAlmostEqual(m["overall_accuracy"], 0.9)
        self.assertLess(m["macro_f1"], 0.5)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        set_seed(0)
        self.tok = StubTokeniser()
        texts = [f"{w} result this quarter" for w in ("loss", "flat", "growth") for _ in range(8)]
        labels = [w for w in ("negative", "neutral", "positive") for _ in range(8)]
        self.dataset = TextDataset(texts, labels, self.tok, max_len=16, label2id=LABEL2ID)

    def test_batches_are_padded_to_the_longest_sentence(self):
        loader = create_loader(TextDataset(["a b c d", "a"], ["neutral"] * 2, self.tok, 16, LABEL2ID), self.tok,
                               batch_size=2)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 5))
        self.assertEqual(batch["attention_mask"][1].tolist(), [1, 1, 0, 0, 0])

    def test_fit_learns_a_separable_toy_task_and_restores_the_best_epoch(self):
        model = tiny_model(class_weights=[1.0, 1.0, 1.0])
        loader = create_loader(self.dataset, self.tok, batch_size=8, shuffle=True)
        history, best = fit(model, loader, loader, torch.device("cpu"), ID2LABEL, epochs=15, lr=5e-3)
        self.assertEqual(len(history), 15)
        self.assertEqual(best["macro_f1"], max(h["macro_f1"] for h in history))
        self.assertGreater(best["overall_accuracy"], 0.9)

    def test_prediction_returns_a_label_and_probabilities_summing_to_one(self):
        label, probs = predict_sentiment(tiny_model(), self.tok, "growth this quarter",
                                         torch.device("cpu"), 16, ID2LABEL)
        self.assertIn(label, ID2LABEL.values())
        self.assertEqual(probs.shape, (3,))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertEqual(label, ID2LABEL[int(probs.argmax())])


if __name__ == "__main__":
    unittest.main()
