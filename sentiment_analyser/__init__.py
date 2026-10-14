"""Sentiment Analyser: fine-tunes DistilBERT on a labelled CSV and predicts sentiment.

Components:
    1. data_collection  - load, clean and split the data
    2. model_training   - set up the model and run the training loop
    3. model_validation - accuracy, per-class accuracy, macro-F1, confusion matrix
    4. prediction       - label and probabilities for new text
    5. dashboard        - Tkinter GUI tying the components together
"""
