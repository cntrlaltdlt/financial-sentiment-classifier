"""Tiny stand-ins so tests run in seconds without downloading DistilBERT."""

import torch
from transformers import DistilBertConfig, DistilBertModel

from sentiment_analyser.models import BertSentimentClassifier


class StubTokeniser:
    """Splits on spaces and maps each word to an id; supports what the code uses."""
    vocab_size = 64

    def _encode(self, text, max_length):
        ids = [1] + [2 + sum(map(ord, w)) % (self.vocab_size - 2) for w in text.lower().split()]
        return ids[:max_length]

    def __call__(self, text, truncation=True, max_length=128, return_tensors=None, **_):
        ids = self._encode(text, max_length)
        encoding = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_tensors == "pt":
            encoding = {k: torch.tensor([v]) for k, v in encoding.items()}
        return encoding

    def pad(self, items, return_tensors="pt"):
        width = max(len(item["input_ids"]) for item in items)
        ids = [item["input_ids"] + [0] * (width - len(item["input_ids"])) for item in items]
        mask = [item["attention_mask"] + [0] * (width - len(item["attention_mask"])) for item in items]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def tiny_model(num_labels=3, class_weights=None):
    config = DistilBertConfig(vocab_size=StubTokeniser.vocab_size, dim=16, n_layers=1, n_heads=2,
                              hidden_dim=32, max_position_embeddings=64)
    torch.manual_seed(0)
    return BertSentimentClassifier(DistilBertModel(config), num_labels, class_weights)
