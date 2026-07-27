import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.sparse import spmatrix
from sklearn.metrics import confusion_matrix

from ptypes import Classifier


def plot_confusion_matrix(
    model: Classifier,
    features: spmatrix,
    labels: np.ndarray,
    target_names: list[str],
    title: str,
    save_path: str,
):
    """Row-normalized confusion matrix: share of each true class predicted as each class.

    The diagonal (correct predictions) is masked out of the color scale — its values
    (0.6-0.95) would otherwise stretch the colormap so far that the much smaller
    off-diagonal misclassification rates (0.01-0.2) all wash out to the same pale color.
    """
    predictions = model.predict(features)
    matrix = confusion_matrix(labels, predictions, normalize="true")
    diagonal_mask = np.eye(len(target_names), dtype=bool)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix,
        mask=diagonal_mask,
        cmap="Blues",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Share of true class (off-diagonal only)"},
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    for i in range(len(target_names)):
        ax.add_patch(
            Rectangle((i, i), 1, 1, fill=True, facecolor="0.85", edgecolor="white")
        )
        ax.text(
            i + 0.5,
            i + 0.5,
            f"{matrix[i, i]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="0.3",
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{title} (diagonal grayed out; correct-class rate shown as text)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
