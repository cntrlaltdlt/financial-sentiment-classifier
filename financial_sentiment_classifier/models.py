"""
The dataset wrapper and the DistilBERT classifier used by training, validation and prediction.
"""

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel


class TextDataset(Dataset):
    """Tokenises each sentence on demand. Padding happens per batch in make_collate_fn."""

    def __init__(self, texts, labels, tokeniser, max_len, label2id):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokeniser = tokeniser
        self.max_len = max_len
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokeniser(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,  # DistilBERT doesn't use token_type_ids
        )
        encoding["labels"] = self.label2id[self.labels[idx]]
        return encoding


def make_collate_fn(tokeniser):
    """Pad each batch only to its longest sentence, which is much faster than padding to max_len."""

    def collate(items):
        labels = torch.tensor([item.pop("labels") for item in items], dtype=torch.long)
        batch = tokeniser.pad(items, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return collate


class BertSentimentClassifier(nn.Module):
    """DistilBERT encoder -> [CLS] representation -> dropout -> linear layer, one logit per class."""

    def __init__(self, encoder, num_labels, class_weights=None):
        super().__init__()
        self.bert = encoder
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)
        # A buffer moves with the model to the GPU/CPU and is saved with its state
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights)

    @classmethod
    def from_pretrained(cls, model_name, num_labels, class_weights=None):
        return cls(AutoModel.from_pretrained(model_name), num_labels, class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            # Weighted loss: mistakes on rare classes cost more
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)
        return loss, logits
