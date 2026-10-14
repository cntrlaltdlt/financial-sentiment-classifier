"""Download Financial PhraseBank and convert its 75%-agreement subset to data/financial_phrasebank_75agree.csv.

    python -m scripts.prepare_data

Source: Malo et al. (2014), "Good debt or bad debt: Detecting semantic orientations
in economic texts", via https://huggingface.co/datasets/takala/financial_phrasebank.
Licence: CC BY-NC-SA 3.0. The archive's SHA-256 is checked, so the CSV is exactly reproducible.
"""

import argparse
import csv
import hashlib
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO, FILE = "takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip"
SHA256 = "0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6"
MEMBER = "FinancialPhraseBank-v1.0/Sentences_75Agree.txt"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "financial_phrasebank_75agree.csv"


def parse_lines(text):
    """Each line is 'sentence@label'. The label is after the LAST '@'."""
    rows = []
    for line in text.splitlines():
        if line.strip():
            sentence, label = line.rsplit("@", 1)
            rows.append((sentence.strip(), label.strip()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    archive = hf_hub_download(REPO, FILE, repo_type="dataset")
    digest = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Checksum mismatch: expected {SHA256}, got {digest}")

    with zipfile.ZipFile(archive) as zf:
        text = zf.read(MEMBER).decode("latin-1")  # the original files are Latin-1 encoded
    rows = parse_lines(text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sentences to {args.output}")


if __name__ == "__main__":
    main()
