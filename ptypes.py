from typing import Protocol

import numpy as np


class Classifier(Protocol):
    def fit(self, X, y) -> object: ...
    def predict(self, X) -> np.ndarray: ...
