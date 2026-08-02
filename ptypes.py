"""Customized types that are used in this project."""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

type ParamGrid = dict[str, list]
"""Search space with *unprefixed* keys; `models.build` rewrites them for the Pipeline."""

type Architecture = Literal["bilstm", "bag"]
"""Networks `dl.TorchTextClassifier` can build."""


class Classifier(Protocol):
    """
    Classification models capable of training and scoring.

    Pipeline should be: Documents -> features (the `vec` step) -> Classifier
    """

    def fit(self, X, y) -> object: ...
    def predict(self, X) -> np.ndarray: ...


@dataclass(frozen=True)
class Experiment:
    """A named estimator and the grid to search over it -- what `runner.run` consumes."""

    name: str
    estimator: Classifier
    param_grid: ParamGrid | None = None
    n_jobs: int = -1
