"""
Component 3: Model Validation
Measures how well the model classifies sentences it was not trained on.
"""

import torch
from sklearn.metrics import confusion_matrix, f1_score


def collect_predictions(model, loader, device):
    """Run the model over a loader and return (true label ids, predicted label ids)."""
    model.eval()
    truths, preds = [], []
    with torch.no_grad():
        for batch in loader:
            _, logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            truths.extend(batch["labels"].tolist())
    return truths, preds


def compute_metrics(truths, preds, id2label):
    """Overall accuracy, per-class accuracy (recall), macro-F1 and the confusion matrix.

    Macro-F1 averages the F1 score of each class equally, so a rare class counts
    as much as a common one.
    """
    ids = sorted(id2label)
    total = len(truths)
    correct = sum(t == p for t, p in zip(truths, preds))

    per_class = {}
    for label_id in ids:
        class_total = sum(t == label_id for t in truths)
        class_correct = sum(t == p == label_id for t, p in zip(truths, preds))
        if class_total:
            per_class[id2label[label_id]] = {
                "accuracy": class_correct / class_total,
                "correct": class_correct,
                "total": class_total,
            }

    return {
        "overall_accuracy": correct / max(total, 1),
        "macro_f1": float(f1_score(truths, preds, labels=ids, average="macro", zero_division=0)),
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion_matrix(truths, preds, labels=ids).tolist(),
    }


def validate_model(model, val_loader, device, id2label):
    truths, preds = collect_predictions(model, val_loader, device)
    return compute_metrics(truths, preds, id2label)


def print_validation_metrics(epoch, total_epochs, train_loss, metrics):
    print(f"Epoch {epoch + 1}/{total_epochs} - Train loss: {train_loss:.4f} - "
          f"Val accuracy: {metrics['overall_accuracy']:.4f} - Val macro-F1: {metrics['macro_f1']:.4f}")
    for label, data in metrics["per_class_accuracy"].items():
        print(f"  {label}: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")
