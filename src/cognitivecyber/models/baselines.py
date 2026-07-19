"""
Baseline model registry for intrusion/anomaly detection.

Supervised classifiers and unsupervised anomaly detectors that are
genuinely trainable in a CPU-only sandbox on tabular NIDS features:

  Supervised : RandomForest, DecisionTree, XGBoost, LightGBM, CatBoost,
               SVM (linear-kernel, calibrated for probabilities), MLP
  Unsupervised (novelty/anomaly): IsolationForest, LocalOutlierFactor (novelty
               mode), One-Class SVM

NOTE ON SCOPE: CNN / LSTM / Transformer / Autoencoder / Deep SVDD
sequence-and-representation-learning baselines from the paper's full
architecture require either sequential/windowed raw packet data (not
present in flow-level tabular features) and/or GPU training time that
this sandbox does not have. Their model *interfaces* are stubbed in
`cognitivecyber.models.deep_stubs` with the exact input/output contract
they'd need so they can be dropped in once run against a real dataset on
real infrastructure -- they are not included in the executed benchmark to
avoid reporting fabricated numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier

    _HAS_CATBOOST = True
except ImportError:  # pragma: no cover
    _HAS_CATBOOST = False


@dataclass
class ModelSpec:
    name: str
    family: str  # "supervised" | "unsupervised"
    build: Callable


def get_supervised_models(random_state: int = 42) -> dict:
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=random_state
        ),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.1, n_jobs=-1,
            random_state=random_state, verbosity=-1,
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True,
            random_state=random_state,
        ),
    }
    if _HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1, verbose=False, random_state=random_state
        )
    return models


def get_unsupervised_models(contamination: float = 0.35, random_state: int = 42) -> dict:
    """Novelty/anomaly detectors trained on the normal-only subset.

    contamination should reflect the expected attack ratio in the *training*
    population used to fit these models (here: trained on all traffic since
    UNSW/CICIDS-style flows have overlapping normal/attack density; scores
    are thresholded at prediction time).
    """
    return {
        "IsolationForest": IsolationForest(
            n_estimators=200, contamination=contamination, random_state=random_state
        ),
        "LocalOutlierFactor": LocalOutlierFactor(
            n_neighbors=35, contamination=contamination, novelty=True
        ),
        "OneClassSVM": OneClassSVM(nu=min(contamination, 0.5), kernel="rbf", gamma="scale"),
    }
