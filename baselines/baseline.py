"""Baseline model for 20 Newsgroups classification.

Uses sklearn's built-in cleaning (remove headers/footers/quotes) and a plain
TF-IDF + Logistic Regression pipeline. This is the benchmark that the fuller
pipeline should be compared against.
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import Bunch

import evaluator

DATA_DIR = "./datasets/20newsgroups"
RANDOM_SEED = 8888

data_train = fetch_20newsgroups(
    data_home=DATA_DIR,
    subset="train",
    remove=("headers", "footers", "quotes"),
    random_state=RANDOM_SEED,
)
data_test = fetch_20newsgroups(
    data_home=DATA_DIR,
    subset="test",
    remove=("headers", "footers", "quotes"),
    random_state=RANDOM_SEED,
    shuffle=False,
)
assert isinstance(data_train, Bunch)
assert isinstance(data_test, Bunch)

vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

y_train, y_test = data_train.target, data_test.target

clf = LogisticRegression(random_state=RANDOM_SEED)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
metrics = evaluator.evaluate(clf, X_test, y_test)

print(f"Train docs: {X_train.shape[0]}, Test docs: {X_test.shape[0]}")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
print(
    f"baseline (LR): Accuracy={metrics.accuracy:.2%}, Precision={metrics.precision:.4f},",
    f"Recall={metrics.recall:.4f}, F1={metrics.f1:.4f}",
    sep=" ",
)
print(classification_report(y_test, y_pred, target_names=data_train.target_names))
