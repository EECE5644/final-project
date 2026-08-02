"""
Documents -> feature matrix. The single owner of tokenization in this project.

Each factory returns a plain sklearn transformer for a Pipeline's vectorization step.
`preprocessing` hands over structurally cleaned documents; only this layer splits them
into terms, so no token rule is ever applied twice.
"""

from enum import StrEnum

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

from config import MAX_LEN, MIN_DF, MIN_FREQ


class Method(StrEnum):
    TFIDF = "tfidf"
    BOW = "bow"
    SEQUENCE = "sequence"


TOKEN_PATTERN = r"\b[a-z]{2,}\b"
"""Whole words of 2+ ASCII letters; pure-number tokens stay out of the vocabulary."""

PAD_ID, UNK_ID = 0, 1
"""Reserved ids for `SequenceEncoder`; the learned vocabulary starts after them."""

_SHARED: dict = {
    "lowercase": True,
    "stop_words": "english",
    "token_pattern": TOKEN_PATTERN,
}

LABELS = {
    Method.TFIDF: "TF-IDF",
    Method.BOW: "BoW",
    Method.SEQUENCE: "learned embeddings",
}
"""Display names for the `Method` enum, used in experiment labels and plots."""


def tfidf(*, min_df: int = MIN_DF, **kwargs) -> TfidfVectorizer:
    """Term frequency scaled by inverse document frequency, L2-normalized per document."""
    return TfidfVectorizer(min_df=min_df, **(_SHARED | kwargs))


def bow(*, min_df: int = MIN_DF, **kwargs) -> TfidfVectorizer:
    """
    Bag-of-words counts, L2-normalized but not IDF-weighted.

    `TfidfVectorizer` with the IDF factor off rather than `CountVectorizer`.
    The reason is that `CountVectorizer` doesn't normalize the output, but `TfidfVectorizer` does.
    """
    return TfidfVectorizer(min_df=min_df, **({"use_idf": False} | _SHARED | kwargs))  # pyright: ignore[reportArgumentType]


class SequenceEncoder(BaseEstimator, TransformerMixin):
    """
    Documents -> lists of token ids, truncated to `max_len`.

    Terms occurring fewer than `min_freq` times collapse to one unknown id, keeping the
    embedding table to a size worth learning from.
    """

    vocabulary_: dict[str, int]  # pyright: ignore[reportUninitializedInstanceVariable]
    """Term -> id, learned in `fit`; ids start after `PAD_ID` and `UNK_ID`."""

    def __init__(self, min_freq: int = MIN_FREQ, max_len: int = MAX_LEN):
        self.min_freq = min_freq
        self.max_len = max_len

    def fit(self, X: list[str], y: object = None):
        analyzer = tfidf().build_analyzer()

        counts: dict[str, int] = {}
        for document in X:
            for term in analyzer(document)[: self.max_len]:
                counts[term] = counts.get(term, 0) + 1

        self.vocabulary_ = {}
        for term, count in counts.items():
            if count >= self.min_freq:
                self.vocabulary_[term] = len(self.vocabulary_) + 2  # after PAD and UNK

        return self

    def transform(self, X: list[str]) -> list[list[int]]:
        analyzer = tfidf().build_analyzer()

        check_is_fitted(self)
        sequences = []
        for document in X:
            ids = [
                self.vocabulary_.get(term, UNK_ID)
                for term in analyzer(document)[: self.max_len]
            ]
            # pack_padded_sequence rejects zero-length rows, and cleaning can empty a doc.
            # NOTE: And we cannot just drop the empty rows, because the classifier expects one output per input row.
            sequences.append(ids or [UNK_ID])

        return sequences


def sequence(
    *, min_freq: int = MIN_FREQ, max_len: int = MAX_LEN, **kwargs
) -> SequenceEncoder:
    """
    Documents -> integer id sequences, for the DL models.

    The bag-of-words vectorizers throw word order away, and this one keeps it.
    """
    return SequenceEncoder(min_freq=min_freq, max_len=max_len, **kwargs)


VECTORIZERS = {
    Method.TFIDF: tfidf,
    Method.BOW: bow,
    Method.SEQUENCE: sequence,
}
