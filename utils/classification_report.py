from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    brier_score_loss,
    f1_score
)
from sklearn.calibration import calibration_curve
from .divergence_score import divergence_score

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from joblib import Parallel, delayed


def classification_report(y_true,
                          pred_prob,
                          figsize: tuple[int, int] = (10, 10)
                          ):
    print("-" * 50)
    print("Metrics")
    print("-" * 50)
    print()
    print(f"ROC AUC {roc_auc_score(y_true, pred_prob):.3f}")
    print(f"Brier Score {brier_score_loss(y_true, pred_prob):.3f}")
    print(f"Divergence Score {divergence_score(y_true, pred_prob):.3f}")

    print()
    print("-" * 50)
    print("Precision Recall Curve")
    print("-" * 50)
    print()
    precision, recall, thresholds = precision_recall_curve(y_true, pred_prob)

    mask = (precision > 1e-3) * (recall > 1e-3)
    plt.figure(figsize=figsize)
    plt.plot(precision[mask], recall[mask], label='PR-curve', linestyle='-')
    plt.ylabel('Recall (Share of defaults detected)')
    plt.xlabel('Precision (Share of predictions that were correct)')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.legend()
    plt.show()

    print()
    print("-" * 50)
    print("F1 Curve")
    print("-" * 50)
    print()

    thresholds = np.linspace(0, 1, 100)

    f1_scores = Parallel(n_jobs=-1)(
        delayed(f1_score)(y_true, (pred_prob >= th).astype(np.int8), pos_label=1)
        for th in thresholds
    )

    plt.figure(figsize=figsize)
    plt.plot(thresholds, f1_scores, linestyle='-', label='F1-Score')
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score (Class 1)')
    plt.title('F1-Score vs. Threshold')
    plt.grid(True)
    plt.legend()
    plt.show()

    print()
    print("-" * 50)
    print("Сalibration Curve")
    print("-" * 50)
    print()

    freq_true, freq_pred = calibration_curve(y_true, pred_prob, n_bins=10, strategy='uniform')

    plt.figure(figsize=figsize)
    plt.plot(freq_pred, freq_true, marker='o', label='Модель')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration', color='red')
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual share of positive classes')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid()
    plt.show()


