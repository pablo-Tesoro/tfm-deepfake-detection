"""
Métricas de evaluación y análisis de coste de negocio.

Dos niveles, como pide la orientación a "Negocio" del TFM:
  - Métricas técnicas: accuracy, precision, recall, F1, AUC-ROC, matriz de confusión.
  - Lectura de negocio: coste esperado según el coste de un falso positivo (FP) y
    un falso negativo (FN), y selección del UMBRAL de decisión que minimiza ese
    coste (en KYC, el FN —dejar pasar un deepfake— es mucho más caro que el FP).

Requiere scikit-learn (+ matplotlib para las gráficas).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calcula las métricas técnicas a un umbral dado."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    )

    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = float("nan")
    return metrics


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Devuelve (tn, fp, fn, tp) al umbral dado."""
    from sklearn.metrics import confusion_matrix

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def expected_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    threshold: float = 0.5,
) -> float:
    """Coste total esperado dado el coste de cada tipo de error.

    En el caso de negocio (KYC), cost_fn >> cost_fp: dejar pasar un deepfake
    (FN) es mucho más caro que enviar a un cliente legítimo a revisión (FP).
    """
    _, fp, fn, _ = confusion_counts(y_true, y_prob, threshold)
    return cost_fp * fp + cost_fn * fn


def choose_threshold_by_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    n_steps: int = 101,
) -> Dict[str, float]:
    """Busca el umbral que minimiza el coste esperado de negocio.

    Returns:
        Dict con 'threshold' óptimo y el 'cost' asociado.
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)
    costs = [expected_cost(y_true, y_prob, cost_fp, cost_fn, t) for t in thresholds]
    best_idx = int(np.argmin(costs))
    return {"threshold": float(thresholds[best_idx]), "cost": float(costs[best_idx])}


def plot_confusion(y_true, y_prob, threshold: float = 0.5, ax=None, title: str = ""):
    """Dibuja la matriz de confusión (Real/Fake)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    y_pred = (y_prob >= threshold).astype(int)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Real", "Fake"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    if title:
        disp.ax_.set_title(title)
    return disp


def metrics_table(results: Dict[str, Dict[str, float]]):
    """Convierte {modelo: metrics} en un DataFrame ordenado para la memoria."""
    import pandas as pd

    df = pd.DataFrame(results).T
    cols = [c for c in ["accuracy", "precision", "recall", "f1", "auc"] if c in df.columns]
    return df[cols].round(4)
