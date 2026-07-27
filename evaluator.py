"""Evaluation metrics for classification models."""

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def evaluate(model, features, labels) -> Metrics:
    predictions = model.predict(features)
    return Metrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, average="macro")),
        recall=float(recall_score(labels, predictions, average="macro")),
        f1=float(f1_score(labels, predictions, average="macro")),
    )
