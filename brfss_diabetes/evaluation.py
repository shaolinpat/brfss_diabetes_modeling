import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_curve


def find_optimal_threshold(y_true, y_probs, beta=1.0):
    """
    Find the classification threshold that maximizes the F-beta score.

    Parameters:
        y_true (array-like): True binary labels
        y_probs (array-like): Predicted probabilities for the positive class
        beta (float): Beta parameter for F-beta score

    Returns:
        float: Optimal threshold value
    """

    if not isinstance(beta, (int, float)) or beta <= 0:
        raise ValueError(f"`beta` must be a positive number, got {beta}")

    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    if y_true.shape != y_probs.shape:
        raise ValueError("`y_true` and `y_probs` must have the same shape.")

    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("`y_true` must be binary integers (0 or 1).")

    if not np.issubdtype(y_probs.dtype, np.floating):
        raise ValueError("`y_probs` must be an array of floats between 0 and 1.")

    if np.any(y_probs < 0) or np.any(y_probs > 1):
        raise ValueError("`y_probs` contains values outside [0, 1].")

    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f_beta_scores = (
        (1 + beta**2)
        * (precision[:-1] * recall[:-1])
        / (beta**2 * precision[:-1] + recall[:-1] + 1e-8)
    )
    best_idx = np.argmax(f_beta_scores)
    return thresholds[best_idx]


def plot_classification_report(
    y_true, y_pred, title="Classification Report", save_path=None
):
    """
    Generate a heatmap from classification report metrics.

    Parameters:
        y_true (array-like): True binary labels
        y_pred (array-like): Predicted binary labels
        title (str): Plot title
        save_path (Path or str, optional): If specified, saves plot to path
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("`y_true` and `y_pred` must have the same shape.")

    if not np.issubdtype(y_true.dtype, np.integer) or not np.issubdtype(
        y_pred.dtype, np.integer
    ):
        raise ValueError(
            "`y_true` and `y_pred` must be arrays of binary integers (0 or 1)."
        )

    if save_path is not None and not isinstance(save_path, (str, Path)):
        raise ValueError("`save_path` must be a string or Path object.")

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()

    # Format row labels with support counts
    df.rename(index={"0": "No Diabetes", "1": "Diabetes"}, inplace=True)
    supports = (
        df.loc[["No Diabetes", "Diabetes"], "support"]
        .dropna()
        .astype(int)
        .map("{:,}".format)
    )
    row_labels = {
        label: f"{label}\n(n = {supports[label]})"
        for label in ["No Diabetes", "Diabetes"]
    }
    row_labels.update(
        {
            "accuracy": "accuracy",
            "macro avg": "macro avg",
            "weighted avg": "weighted avg",
        }
    )

    rows_to_plot = ["No Diabetes", "Diabetes", "accuracy", "macro avg", "weighted avg"]
    metrics = df.loc[rows_to_plot, ["precision", "recall", "f1-score"]].round(2)
    row_label_list = [row_labels[r] for r in rows_to_plot]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    sns.heatmap(
        metrics,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        xticklabels=["Precision", "Recall", "F1-Score"],
        yticklabels=row_label_list,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
