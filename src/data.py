"""Data loading + torch Dataset for Subtask 1."""
import pandas as pd
import torch
from torch.utils.data import Dataset

import config as cfg


def load_train():
    df = pd.read_csv(cfg.TRAIN_PATH)
    df["label"] = df["Sentiment"].map(cfg.LABEL2ID)
    assert df["label"].isnull().sum() == 0, "Unmapped label found"
    return df.reset_index(drop=True)


def load_test():
    return pd.read_csv(cfg.TEST_PATH).reset_index(drop=True)


class SentimentDataset(Dataset):
    """Tokenizes lazily so the same object can be reused across different tokenizers if needed.

    `sample_weights` (optional): per-example loss weight, on top of the usual per-class
    weighting -- used to down-weight pseudo-labeled (self-training) rows relative to gold
    training rows (see pseudo_label.py / WeightedTrainer in train.py). Defaults to 1.0 for
    every example when not provided.
    """

    def __init__(self, texts, labels, tokenizer, max_length=cfg.MAX_LENGTH, sample_weights=None):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_weights = list(sample_weights) if sample_weights is not None else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(self.sample_weights[idx], dtype=torch.float)
        return item
