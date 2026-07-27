"""
Model comparison for 20 Newsgroups classification.

This script trains and evaluates three different classifiers (Multinomial Naive Bayes,
Linear SVM, and Logistic Regression) on the 20 Newsgroups dataset.
"""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import data_preprossor
import evaluator
from config import RANDOM_SEED
from ptypes import Classifier

# ==================== Data Prepration ====================

data_train, data_test = data_preprossor.load_data()
data_train, data_test = (
    data_preprossor.preprocess_data(data_train),
    data_preprossor.preprocess_data(data_test),
)
target_names = data_train.target_names
y_train, y_test = data_train.target, data_test.target

vectorizer = TfidfVectorizer(min_df=2)
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)


# ==================== Model Configs ====================
@dataclass
class ModelConfig:
    model: Classifier
    param_grid: dict[str, list] | None = None


configs = {
    "Multinomial Naive Bayes": ModelConfig(
        MultinomialNB(),
        {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0]},
    ),
    "Linear SVM": ModelConfig(
        LinearSVC(random_state=RANDOM_SEED, max_iter=5000),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "Logistic Regression": ModelConfig(
        LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        {"C": [0.01, 0.1, 1, 10, 100]},
    ),
}


# ==================== Training & Evaluation ====================
def grid_search(model, param_grid, X_train, y_train):
    search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_

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

for name, config in configs.items():
    model, best_params, best_score = grid_search(
        config.model, config.param_grid, X_train, y_train
    )
    print(f"{name}: Best Params: {best_params}, Best CV Accuracy: {best_score:.4f}")

    metrics = evaluator.evaluate(model, X_test, y_test)
    print(
        f"{name}: Accuracy={metrics.accuracy:.2%}, Precision={metrics.precision:.4f},",
        f"Recall={metrics.recall:.4f}, F1={metrics.f1:.4f}",
        sep=" ",
    )

    #if hasattr(model, "coef_"):
    #    show_top_features(model, vectorizer, target_names, top_n=10)


    # slug = name.lower().replace(" ", "_")
    # plotter.plot_confusion_matrix(
    #     model,
    #     X_test,
    #     y_test,
    #     target_names,
    #     title=f"{name}: Confusion Matrix (test set)",
    #     save_path=f"{FIG_DIR}/confusion_matrix_{slug}.png",
    # )


