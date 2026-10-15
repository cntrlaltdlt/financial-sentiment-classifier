"""
Component 4: Prediction
Gives the predicted label and the probability of every label for new text.
"""

import torch


def predict_sentiment(model, tokeniser, text, device, max_len, id2label):
    model.eval()
    encoding = tokeniser(
        text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        return_token_type_ids=False,  # DistilBERT doesn't use token_type_ids
    )

    with torch.no_grad():
        _, logits = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        )

    # Softmax turns the logits into probabilities that sum to 1
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return id2label[int(probs.argmax())], probs
