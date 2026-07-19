import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from cognitivecyber.data.loaders import load_dataset
from cognitivecyber.data.preprocessing import preprocess
from cognitivecyber.evaluation.metrics import evaluate_classifier, results_to_table
from cognitivecyber.models.baselines import get_supervised_models, get_unsupervised_models


@pytest.fixture(scope="module")
def small_split():
    df, meta = load_dataset(n_synthetic=3000, random_state=21)
    return preprocess(df, meta, random_state=21)


def test_get_supervised_models_returns_expected_keys():
    models = get_supervised_models()
    expected = {"RandomForest", "DecisionTree", "XGBoost", "LightGBM", "SVM", "MLP"}
    assert expected <= set(models.keys())


def test_get_unsupervised_models_returns_expected_keys():
    models = get_unsupervised_models()
    assert {"IsolationForest", "LocalOutlierFactor", "OneClassSVM"} <= set(models.keys())


def test_evaluate_classifier_supervised(small_split):
    model = RandomForestClassifier(n_estimators=20, random_state=0)
    r = evaluate_classifier(
        model, small_split.X_train, small_split.y_train,
        small_split.X_test, small_split.y_test, "RandomForest",
    )
    assert 0.0 <= r.accuracy <= 1.0
    assert 0.0 <= r.roc_auc <= 1.0
    assert r.confusion.shape == (2, 2)
    assert r.train_time_s >= 0
    assert r.inference_time_s >= 0
    assert r.peak_memory_mb >= 0
    assert len(r.fpr) == len(r.tpr)


def test_evaluate_classifier_unsupervised(small_split):
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(n_estimators=50, contamination=0.3, random_state=0)
    r = evaluate_classifier(
        model, small_split.X_train, small_split.y_train,
        small_split.X_test, small_split.y_test, "IsolationForest",
        score_fn="decision_function",
    )
    assert 0.0 <= r.accuracy <= 1.0
    assert set(np.unique(r.y_true)) <= {0, 1}


def test_results_to_table_sorted_by_f1(small_split):
    model1 = RandomForestClassifier(n_estimators=10, random_state=0)
    model2 = RandomForestClassifier(n_estimators=100, random_state=0)
    r1 = evaluate_classifier(
        model1, small_split.X_train, small_split.y_train,
        small_split.X_test, small_split.y_test, "RF_small",
    )
    r2 = evaluate_classifier(
        model2, small_split.X_train, small_split.y_train,
        small_split.X_test, small_split.y_test, "RF_large",
    )
    table = results_to_table([r1, r2])
    assert list(table.columns) == [
        "Model", "Accuracy", "Precision", "Recall", "F1",
        "ROC-AUC", "PR-AUC", "Train Time (s)", "Inference Time (s)", "Peak Memory (MB)",
    ]
    # Sorted descending by F1
    assert table["F1"].is_monotonic_decreasing
