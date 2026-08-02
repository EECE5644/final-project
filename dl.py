"""PyTorch side of the pipeline, wearing the same interface as the sklearn models."""

import copy
from typing import override

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import RANDOM_SEED
from ptypes import Architecture
from vectorizers import PAD_ID

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class _SequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]], labels: np.ndarray | None = None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    @override
    def __getitem__(self, idx):
        label = -1 if self.labels is None else int(self.labels[idx])
        return torch.tensor(self.sequences[idx], dtype=torch.long), label


def _collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(list(sequences), batch_first=True, padding_value=PAD_ID)

    return padded, lengths, torch.tensor(labels, dtype=torch.long)


class _BagOfEmbeddings(nn.Module):
    """Averages word embeddings and classifies -- word order discarded entirely."""

    def __init__(
        self, vocab_size: int, num_classes: int, embed_dim: int, dropout: float
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            vocab_size, embed_dim, mode="mean", padding_idx=PAD_ID
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    @override
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.embedding(x)))


class _BiLSTM(nn.Module):
    """Embedding -> bidirectional LSTM -> mean/max pooling -> layer norm -> linear."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    @override
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embed_dropout(self.embedding(x))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)

        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
        lengths = lengths.to(outputs.device)
        steps = torch.arange(outputs.size(1), device=outputs.device)
        mask = (steps[None, :] < lengths[:, None]).unsqueeze(-1)
        # Padding unpacks to zeros: harmless in the sum, but it would win the max.
        mean = outputs.sum(dim=1) / lengths.unsqueeze(1)
        largest = outputs.masked_fill(~mask, float("-inf")).max(dim=1).values

        return self.fc(self.dropout(self.norm(torch.cat([mean, largest], dim=1))))


class TorchTextClassifier(BaseEstimator):
    """
    Token id sequences -> class labels, behind sklearn's `fit`/`predict` interface.

    `architecture` picks "bilstm" or "bag". The network is built in `fit`, not
    `__init__`, because vocabulary size is only known once `SequenceEncoder` has run;
    torch is re-seeded there too, so results do not depend on fitting order.
    """

    model_: nn.Module  # pyright: ignore[reportUninitializedInstanceVariable]
    """The network, built in `fit` once the vocabulary size is known."""

    val_score_: float  # pyright: ignore[reportUninitializedInstanceVariable]
    """Best validation accuracy seen, present only when `val_fraction` is nonzero."""

    def __init__(
        self,
        architecture: Architecture = "bilstm",
        embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.4,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        val_fraction: float = 0.1,
        random_state: int = RANDOM_SEED,
        verbose: bool = True,
    ):
        self.architecture = architecture
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose

    def _build(self, vocab_size: int, num_classes: int) -> nn.Module:
        if self.architecture == "bag":
            return _BagOfEmbeddings(
                vocab_size, num_classes, self.embed_dim, self.dropout
            )

        return _BiLSTM(
            vocab_size, num_classes, self.embed_dim, self.hidden_dim, self.dropout
        )

    def fit(self, X: list[list[int]], y: np.ndarray):
        torch.manual_seed(self.random_state)

        vocab_size = max(max(sequence) for sequence in X) + 1
        self.model_ = self._build(vocab_size, int(y.max()) + 1).to(DEVICE)

        if self.val_fraction:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X,
                y,
                test_size=self.val_fraction,
                random_state=self.random_state,
                stratify=y,
            )
        else:
            X_fit, y_fit, X_val, y_val = X, y, None, None

        loader = DataLoader(
            _SequenceDataset(X_fit, y_fit),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate,
        )
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        best_score, best_state, stale = -1.0, None, 0
        for epoch in range(self.epochs):
            self.model_.train()
            total_loss = 0.0
            for inputs, lengths, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model_(inputs, lengths), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)

            report = f"    epoch {epoch + 1}/{self.epochs}: loss={total_loss / len(X_fit):.4f}"
            if X_val is not None:
                score = float((self.predict(X_val) == y_val).mean())
                report += f", val_acc={score:.2%}"
                if score > best_score:
                    best_score, stale = score, 0
                    best_state = copy.deepcopy(self.model_.state_dict())
                else:
                    stale += 1
            if self.verbose:
                print(report, flush=True)

            if stale >= self.patience:
                if self.verbose:
                    print(f"    stopping: no gain in {stale} epochs", flush=True)
                break

        # Keep the epoch that generalized best rather than the last one: training loss on
        # this corpus keeps falling long after validation accuracy has peaked.
        if best_state is not None:
            self.model_.load_state_dict(best_state)
            self.val_score_ = best_score

        return self

    def predict(self, X: list[list[int]]) -> np.ndarray:
        loader = DataLoader(
            _SequenceDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

        self.model_.eval()
        predictions = []
        with torch.no_grad():
            for inputs, lengths, _ in loader:
                logits = self.model_(inputs.to(DEVICE), lengths)
                predictions.append(logits.argmax(dim=1).cpu().numpy())

        return np.concatenate(predictions)
