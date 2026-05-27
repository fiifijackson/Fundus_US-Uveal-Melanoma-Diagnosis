"""Evaluation helpers for binary classification."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_binary_classifier(y_true, y_pred_probs, threshold: float = 0.5) -> dict:
    """Compute binary classification metrics used in the original notebook."""
    y_pred_probs = np.ravel(y_pred_probs)
    y_pred = (y_pred_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    specificity = np.nan
    if cm.shape == (2, 2) and (cm[0, 0] + cm[0, 1]) > 0:
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_pred_probs),
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def save_prediction_table(
    image_names: list[str],
    y_true,
    y_pred,
    output_path: str | Path,
) -> None:
    """Save per-image predictions to an Excel file."""
    results_df = pd.DataFrame(
        {
            "Image Name": image_names,
            "Predicted Class": np.ravel(y_pred),
            "True Label": np.ravel(y_true),
        }
    )
    results_df.to_excel(output_path, index=False)


def plot_confusion_matrix(cm, class_names, output_path: str | Path | None = None) -> None:
    """Plot and optionally save a confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_pred_probs, output_path: str | Path | None = None) -> None:
    """Plot and optionally save an ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, np.ravel(y_pred_probs))
    auc = roc_auc_score(y_true, np.ravel(y_pred_probs))

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
