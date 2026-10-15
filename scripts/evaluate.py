"""Train on a fixed split, pick the best epoch on validation, and report held-out test results once.

    python -m scripts.evaluate --output results/

Uses the same components as the GUI. The test set is never used for any
decision (epoch choice uses validation), so the test numbers are an unbiased
estimate of performance on unseen sentences from the same source.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from financial_sentiment_classifier.data_collection import (
    check_trainable, clean_data, compute_class_weights, make_label_maps, split_data)
from financial_sentiment_classifier.model_training import (
    MODEL_NAME, create_loader, fit, get_device, set_seed, setup_training)
from financial_sentiment_classifier.model_validation import print_validation_metrics, validate_model
from financial_sentiment_classifier.models import TextDataset

ROOT = Path(__file__).resolve().parents[1]


def plot_confusion_matrix(matrix, labels, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2), layout="constrained")
    ax.imshow(matrix, cmap="Blues")
    threshold = max(map(max, matrix)) / 2  # white text on the darker cells
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            ax.text(j, i, count, ha="center", va="center", color="white" if count > threshold else "black")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test set confusion matrix")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "financial_phrasebank_75agree.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    start = time.time()
    set_seed(args.seed)
    df, report = clean_data(pd.read_csv(args.data))
    check_trainable(df)
    train_df, val_df, test_df = split_data(df, val_size=0.15, test_size=0.15, seed=args.seed)
    label2id, id2label = make_label_maps(df["label"])
    class_weights = compute_class_weights(train_df["label"], label2id)
    print(f"Cleaning: {report}")
    print(f"Split: train {len(train_df)}, validation {len(val_df)}, test {len(test_df)}")

    device = get_device()
    model, tokeniser = setup_training(len(label2id), class_weights, device)

    def loader(frame, shuffle=False):
        dataset = TextDataset(frame["text"], frame["label"], tokeniser, args.max_len, label2id)
        return create_loader(dataset, tokeniser, shuffle=shuffle, seed=args.seed)

    history, best = fit(model, loader(train_df, shuffle=True), loader(val_df), device, id2label,
                        epochs=args.epochs, on_epoch=print_validation_metrics)
    test = validate_model(model, loader(test_df), device, id2label)

    majority_label, majority_count = Counter(test_df["label"]).most_common(1)[0]
    results = {
        "data": str(args.data.relative_to(ROOT)) if args.data.is_relative_to(ROOT) else str(args.data),
        "model": MODEL_NAME,
        "device": str(device),
        "settings": {"epochs": args.epochs, "seed": args.seed, "max_len": args.max_len,
                     "batch_size": 16, "learning_rate": 2e-5, "split": "70/15/15 stratified"},
        "cleaning": report,
        "split_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "class_weights": dict(zip(label2id, class_weights)),
        "history": history,
        "best_epoch": best["epoch"],
        "test": test,
        "majority_class_baseline": {"label": majority_label,
                                    "accuracy": majority_count / len(test_df)},
        "runtime_seconds": round(time.time() - start, 1),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_confusion_matrix(test["confusion_matrix"], list(label2id), args.output / "confusion_matrix.png")

    print(f"\nBest epoch (by validation macro-F1): {best['epoch']}")
    print(f"TEST accuracy {test['overall_accuracy']:.1%}, macro-F1 {test['macro_f1']:.3f} "
          f"(always predicting '{majority_label}' would score {majority_count / len(test_df):.1%})")
    for label, data in test["per_class_accuracy"].items():
        print(f"  {label}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")
    print(f"Saved {args.output / 'metrics.json'} and confusion_matrix.png in {results['runtime_seconds']} s")


if __name__ == "__main__":
    main()
