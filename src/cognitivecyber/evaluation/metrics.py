"""Metric computation for binary intrusion-detection classifiers."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    prec_curve: np.ndarray
    rec_curve: np.ndarray
    train_time_s: float
    inference_time_s: float
    peak_memory_mb: float
    y_true: np.ndarray = field(repr=False)
    y_score: np.ndarray = field(repr=False)


def evaluate_classifier(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    score_fn: str = "predict_proba",
) -> EvalResult:
    """Fit `model`, time train/inference, and compute the full metric set.

    score_fn: "predict_proba" for standard sklearn-API classifiers, or
    "decision_function" for models (e.g. SVM without probability, LOF
    novelty mode) that only expose decision_function/score_samples.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - t1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if score_fn == "predict_proba" and hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred.astype(float)

    # Unsupervised models (IsolationForest, OCSVM, LOF-novelty) predict {-1,1}
    if set(np.unique(y_pred)) <= {-1, 1}:
        y_pred = (y_pred == -1).astype(int)
        y_score = -y_score  # higher score = more anomalous = more "positive"

    fpr, tpr, _ = roc_curve(y_test, y_score)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_score)

    return EvalResult(
        model_name=model_name,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_score),
        pr_auc=average_precision_score(y_test, y_score),
        confusion=confusion_matrix(y_test, y_pred),
        fpr=fpr,
        tpr=tpr,
        prec_curve=prec_curve,
        rec_curve=rec_curve,
        train_time_s=train_time,
        inference_time_s=inference_time,
        peak_memory_mb=peak / (1024 * 1024),
        y_true=np.asarray(y_test),
        y_score=np.asarray(y_score),
    )


def results_to_table(results: list) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for r in results:
        rows.append(
            {
                "Model": r.model_name,
                "Accuracy": r.accuracy,
                "Precision": r.precision,
                "Recall": r.recall,
                "F1": r.f1,
                "ROC-AUC": r.roc_auc,
                "PR-AUC": r.pr_auc,
                "Train Time (s)": r.train_time_s,
                "Inference Time (s)": r.inference_time_s,
                "Peak Memory (MB)": r.peak_memory_mb,
            }
        )
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
