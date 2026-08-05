"""Evaluation metrics for classification models."""

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ptypes import Classifier


@dataclass(frozen=True)
class Metrics:
    """Evaluation metrics for Classifier."""

    accuracy: float
    precision: float
    recall: float
    f1: float


def evaluate(model: Classifier, features, labels) -> Metrics:
    predictions = model.predict(features)
    return Metrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, average="macro")),
        recall=float(recall_score(labels, predictions, average="macro")),
        f1=float(f1_score(labels, predictions, average="macro")),
    )
