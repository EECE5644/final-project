"""
Registry of the classifiers we compare.

`MODELS` holds each model's factory and its unprefixed parameter grid; `build` composes
one into a `vec` + `clf` Pipeline and rewrites the grid keys into the `clf__` / `vec__`
form `GridSearchCV` needs.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import vectorizers
from config import RANDOM_SEED
from ptypes import Architecture, Classifier, Experiment, ParamGrid

VEC_STEP, CLF_STEP = "vec", "clf"
"""Pipeline step names. Fixed, because `build` prefixes grid keys with them."""


@dataclass(frozen=True)
class ModelSpec:
    """
    The components needed to build a model.

    `n_jobs` is for the `GridSearchCV` work, not the model itself.
    """

    label: str
    factory: Callable[..., Classifier]
    param_grid: ParamGrid = field(default_factory=dict)
    default_method: vectorizers.Method = vectorizers.Method.TFIDF
    n_jobs: int = -1


def _torch(architecture: Architecture):
    """Imported torch lazily."""

    def factory(**kwargs) -> Classifier:
        from dl import TorchTextClassifier

        return TorchTextClassifier(architecture=architecture, **kwargs)

    return factory


MODELS: dict[str, ModelSpec] = {
    # ─── ML Methods ─────────────────────────────────────────
    "multinomial_naive_bayes": ModelSpec(
        "Multinomial Naive Bayes",
        MultinomialNB,
        {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0]},
    ),
    "complement_naive_bayes": ModelSpec(
        "Complement Naive Bayes",
        ComplementNB,
        {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0]},
    ),
    "linear_svm": ModelSpec(
        "Linear SVM",
        partial(LinearSVC, random_state=RANDOM_SEED, max_iter=5000),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "logistic_regression": ModelSpec(
        "Logistic Regression",
        partial(LogisticRegression, random_state=RANDOM_SEED, max_iter=1000),
        {"C": [0.01, 0.1, 1, 10, 100]},
    ),
    "mlp": ModelSpec(
        "MLP",
        partial(MLPClassifier, early_stopping=True, random_state=RANDOM_SEED),
        {"hidden_layer_sizes": [(256,), (512,)]},
        n_jobs=4,
    ),
    # ─── DL Methods ─────────────────────────────────────────
    "bag_of_embeddings": ModelSpec(
        "Bag of embeddings", _torch("bag"), default_method=vectorizers.Method.SEQUENCE
    ),
    "lstm": ModelSpec(
        "BiLSTM", _torch("bilstm"), default_method=vectorizers.Method.SEQUENCE
    ),
}


def _prefix(grid: ParamGrid, step: str) -> ParamGrid:
    return {f"{step}__{key}": values for key, values in grid.items()}


def build(model: str, method: str | None = None) -> Experiment:
    """
    Compose a registry entry and a feature representation into a runnable `Experiment`.

    `method` defaults to whatever the spec declares.
    """
    spec = MODELS[model]
    method = vectorizers.Method(method) if method else spec.default_method

    steps = [
        (VEC_STEP, vectorizers.VECTORIZERS[method]()),
        (CLF_STEP, spec.factory()),
    ]

    grid = _prefix(spec.param_grid, CLF_STEP) if spec.param_grid else {}
    label = vectorizers.LABELS.get(method, method)

    return Experiment(
        name=f"{spec.label} + {label}",
        estimator=Pipeline(steps),  # pyright: ignore[reportArgumentType]
        param_grid=grid or None,
        n_jobs=spec.n_jobs,
    )


def show_top_features(model, vectorizer, target_names, top_n=10):
    """Print the top positive and negative features for each class."""
    feature_names = np.array(vectorizer.get_feature_names_out())

    if hasattr(model, "coef_"):
        for i, class_label in enumerate(target_names):
            coef = model.coef_[i]
            top_positive_indices = np.argsort(coef)[-top_n:]
            top_negative_indices = np.argsort(coef)[:top_n]

            print(f"\nClass: {class_label}")
            print("Top positive features:", feature_names[top_positive_indices])
            print("Top negative features:", feature_names[top_negative_indices])
    else:
        print("This model does not provide coefficients (e.g., Naive Bayes).")
