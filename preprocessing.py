"""
Load and prepare the 20 Newsgroups corpus.

Two responsibilities, both of which have to happen outside a sklearn Pipeline:
  - fetching each split with sklearn's structural cleaning applied
  - dropping documents that cleaning left too short.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch

from config import DATA_DIR, MIN_DOC_WORDS, RANDOM_SEED

REMOVE = ("headers", "footers", "quotes")


@dataclass(frozen=True)
class Dataset:
    """One split of cleaned documents, kept aligned with its labels."""

    documents: list[str]
    targets: np.ndarray
    target_names: list[str]

    def __len__(self) -> int:
        return len(self.documents)


def load(remove: tuple[str, ...] = REMOVE) -> tuple[Bunch, Bunch]:
    """Fetch the raw train/test splits, cached under `DATA_DIR`."""
    data_train = fetch_20newsgroups(
        data_home=DATA_DIR,
        subset="train",
        remove=remove,
        random_state=RANDOM_SEED,
    )
    data_test = fetch_20newsgroups(
        data_home=DATA_DIR,
        subset="test",
        remove=remove,
        random_state=RANDOM_SEED,
        shuffle=False,
    )
    assert isinstance(data_train, Bunch)
    assert isinstance(data_test, Bunch)

    return data_train, data_test


def build_dataset(data: Bunch, *, min_words: int = MIN_DOC_WORDS) -> Dataset:
    """Drop documents left with fewer than `min_words` words, labels along with them."""
    keep = [i for i, text in enumerate(data.data) if len(text.split()) >= min_words]

    return Dataset(
        documents=[data.data[i] for i in keep],
        targets=data.target[keep],
        target_names=list(data.target_names),
    )


def load_dataset(
    *, remove: tuple[str, ...] = REMOVE, min_words: int = MIN_DOC_WORDS
) -> tuple[Dataset, Dataset]:
    """`load` then `build_dataset` for both splits -- where every experiment starts."""
    data_train, data_test = load(remove)

    return (
        build_dataset(data_train, min_words=min_words),
        build_dataset(data_test, min_words=min_words),
    )
