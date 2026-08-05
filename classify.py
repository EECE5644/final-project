"""Classify an arbitrary post with the report's best model, refit on the spot."""

from functools import cache

import numpy as np

from models import build
from preprocessing import load_dataset
from ptypes import Classifier
from vectorizers import Method

MODEL, METHOD, ALPHA = "complement_naive_bayes", Method.BOW, 0.1
"""The winner: 74.07% test accuracy, `alpha` as its grid search chose it."""


DEMO_FROM_APPLE_NEWS = """The full moon of August 2026 will occur at 12:18 a.m. (0416 GMT) on Aug. 28, as the lunar disk shines directly opposite the sun in Earth's sky. Its arrival will trigger a breathtaking lunar eclipse late in the month, which will be visible across the Americas and parts of Africa and Europe as Earth passes between the sun and moon to bathe the lunar disk in shadow."""


@cache
def _fitted() -> tuple[Classifier, list[str]]:
    """The trained Pipeline and its label names, fit once per process."""
    train, _ = load_dataset()
    estimator = build(MODEL, METHOD).estimator
    estimator.set_params(clf__alpha=ALPHA)  # pyright: ignore[reportAttributeAccessIssue]
    estimator.fit(train.documents, train.targets)

    return estimator, train.target_names


def classify(post: str, top: int = 5) -> list[tuple[str, float]]:
    """The `top` likeliest newsgroups for `post` with their probabilities, best first."""
    estimator, target_names = _fitted()
    probabilities = estimator.predict_proba([post])[0]  # pyright: ignore[reportAttributeAccessIssue]
    order = np.argsort(probabilities)[::-1][:top]

    return [(target_names[index], float(probabilities[index])) for index in order]


def _report(post: str):
    for rank, (label, probability) in enumerate(classify(post)):
        marker = "->" if rank == 0 else "  "
        print(f"{marker} {label:<26}{probability:>7.2%}")


def main():
    print(f"fitting {MODEL} + {METHOD} ...", flush=True)
    _fitted()

    while True:
        try:
            post = input("\npost> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if post:
            print()
            _report(post)


if __name__ == "__main__":
    main()
