"""From-scratch bag-of-words pipeline for 20 Newsgroups classification.

Uses data_preprossor's custom cleaning  and a hand-built BagOfWordsVectorizer + Logistic Regression,
instead of sklearn's built-in TfidfVectorizer. Compare against baselines/baseline.py.
"""

from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import data_preprossor
import evaluator


class BagOfWordsVectorizer(BaseEstimator, TransformerMixin):
    """From-scratch bag-of-words vectorizer with a sklearn-compatible fit/transform API."""

    def __init__(self):
        self.vocabulary_: dict[str, int] = {}

    def fit(self, texts: list[str], _y: object = None):
        vocab = sorted({word for text in texts for word in text.split()})
        self.vocabulary_ = {word: idx for idx, word in enumerate(vocab)}
        return self

    def transform(self, texts: list[str]):
        rows, cols, counts = [], [], []
        for row, text in enumerate(texts):
            row_counts = {}
            for word in text.split():
                col = self.vocabulary_.get(word)
                if col is not None:
                    row_counts[col] = row_counts.get(col, 0) + 1
            for col, count in row_counts.items():
                rows.append(row)
                cols.append(col)
                counts.append(count)
        return csr_matrix(
            (counts, (rows, cols)), shape=(len(texts), len(self.vocabulary_))
        )


data_train, data_test = data_preprossor.load_data()
data_train, data_test = (
    data_preprossor.preprocess_data(data_train),
    data_preprossor.preprocess_data(data_test),
)

vectorizer = BagOfWordsVectorizer()
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

y_train, y_test = data_train.target, data_test.target

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
metrics = evaluator.evaluate(clf, X_test, y_test)

assert X_train.shape is not None and X_test.shape is not None
print(f"Train docs: {X_train.shape[0]}, Test docs: {X_test.shape[0]}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(
    f"baseline (LR): Accuracy={metrics.accuracy:.2%}, Precision={metrics.precision:.4f},",
    f"Recall={metrics.recall:.4f}, F1={metrics.f1:.4f}", sep=" "
)
print(classification_report(y_test, y_pred, target_names=data_train.target_names))
