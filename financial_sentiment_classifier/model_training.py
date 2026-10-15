"""
Component 2: Model Training
Sets up the model, runs the training loop and keeps the best epoch.
"""

import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .model_validation import validate_model
from .models import BertSentimentClassifier, make_collate_fn

MODEL_NAME = "distilbert-base-uncased"


def set_seed(seed):
    """Fix every source of randomness so a run can be repeated exactly (on the same hardware)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    """Use an NVIDIA GPU or Apple-silicon GPU when available, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_training(num_labels, class_weights, device, model_name=MODEL_NAME):
    """Load the tokeniser and the pre-trained encoder with a new classification layer."""
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    model = BertSentimentClassifier.from_pretrained(model_name, num_labels, class_weights)
    model.to(device)
    return model, tokeniser


def create_loader(dataset, tokeniser, batch_size=16, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=make_collate_fn(tokeniser), generator=generator)


def train_model_epoch(model, train_loader, optimiser, device):
    """One pass over the training data. Returns the average loss per batch."""
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimiser.zero_grad()
        loss, _ = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()

    return total_loss / max(len(train_loader), 1)


def fit(model, train_loader, val_loader, device, id2label, epochs=3, lr=2e-5, on_epoch=None):
    """Train for several epochs, validate after each, and restore the best epoch.

    "Best" means highest validation macro-F1: with imbalanced classes, plain
    accuracy can look good while a rare class is mostly misclassified.
    Returns the per-epoch history and the best epoch's validation metrics.
    """
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    history, best_state, best_metrics = [], None, None

    for epoch in range(epochs):
        train_loss = train_model_epoch(model, train_loader, optimiser, device)
        metrics = validate_model(model, val_loader, device, id2label)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})
        if on_epoch:
            on_epoch(epoch, epochs, train_loss, metrics)
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = {"epoch": epoch + 1, **metrics}
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return history, best_metrics
