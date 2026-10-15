"""
Component 1: Data Collection
Loads a labelled CSV, cleans it, and splits it so no sentence appears in two sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

MIN_ROWS_PER_CLASS = 5


def load_csv_data(csv_path, text_col="text", label_col="label"):
    """Load a CSV and check that the chosen text and label columns exist."""
    df = pd.read_csv(csv_path)
    missing = [col for col in (text_col, label_col) if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing column(s): {', '.join(missing)}. "
            f"Columns found: {', '.join(map(str, df.columns))}."
        )
    return df


def clean_data(df, text_col="text", label_col="label"):
    """Remove blank rows and duplicate sentences before any split.

    Duplicates matter: if the same sentence lands in both the training and the
    validation set, validation accuracy measures memorisation, not
    generalisation. Sentences are compared ignoring case and extra spaces.
    A sentence that appears with two different labels is dropped entirely,
    because there is no way to know which label is right.

    Returns the cleaned DataFrame (columns "text" and "label") and a report.
    """
    data = df[[text_col, label_col]].dropna()
    data = data.rename(columns={text_col: "text", label_col: "label"})
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip()
    data = data[(data["text"] != "") & (data["label"] != "")]

    key = data["text"].str.lower().str.split().str.join(" ")
    labels_per_key = data.groupby(key)["label"].nunique()
    conflicting = set(labels_per_key[labels_per_key > 1].index)
    keep = ~key.isin(conflicting)
    consistent, consistent_key = data[keep], key[keep]
    deduplicated = consistent[~consistent_key.duplicated()]

    report = {
        "rows_in": len(df),
        "rows_out": len(deduplicated),
        "blank_rows_removed": len(df) - len(data),
        "conflicting_sentences_removed": len(data) - len(consistent),
        "duplicates_removed": len(consistent) - len(deduplicated),
    }
    return deduplicated.reset_index(drop=True), report


def check_trainable(df):
    """Raise a clear error if the cleaned data cannot support a fair split."""
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("The data needs at least two different labels.")
    too_small = counts[counts < MIN_ROWS_PER_CLASS]
    if len(too_small):
        details = ", ".join(f"'{label}' ({n})" for label, n in too_small.items())
        raise ValueError(
            f"Each label needs at least {MIN_ROWS_PER_CLASS} distinct sentences "
            f"after removing duplicates. Too few: {details}."
        )


def split_data(df, val_size=0.15, test_size=0.0, seed=42):
    """Stratified split into train / validation (/ test), reproducible via seed."""
    holdout = val_size + test_size
    train_df, rest = train_test_split(
        df, test_size=holdout, stratify=df["label"], random_state=seed
    )
    if test_size == 0:
        return train_df.reset_index(drop=True), rest.reset_index(drop=True)
    val_df, test_df = train_test_split(
        rest, test_size=test_size / holdout, stratify=rest["label"], random_state=seed
    )
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def make_label_maps(labels):
    """Map label names to ids (sorted, so the order is stable) and back."""
    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def compute_class_weights(train_labels, label2id):
    """Inverse-frequency weights, total / (classes * count), from the TRAINING split only.

    Rarer classes get larger weights, so the loss does not reward a model that
    simply predicts the most common label.
    """
    counts = pd.Series(list(train_labels)).value_counts()
    total = len(train_labels)
    num_classes = len(label2id)
    return [total / (num_classes * counts.get(label, 1)) for label in label2id]
