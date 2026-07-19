"""Publication-quality figure generation, saved as PNG + SVG + PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
DPI = 300


def _save_all(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(results: list, out_dir: Path, name: str = "roc_curves"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        ax.plot(r.fpr, r.tpr, label=f"{r.model_name} (AUC={r.roc_auc:.3f})", linewidth=1.6)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Baseline Models")
    ax.legend(fontsize=7, loc="lower right")
    _save_all(fig, out_dir, name)


def plot_pr_curves(results: list, out_dir: Path, name: str = "pr_curves"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        ax.plot(r.rec_curve, r.prec_curve, label=f"{r.model_name} (AP={r.pr_auc:.3f})", linewidth=1.6)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves — Baseline Models")
    ax.legend(fontsize=7, loc="lower left")
    _save_all(fig, out_dir, name)


def plot_confusion_matrices(results: list, out_dir: Path, name: str = "confusion_matrices"):
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, r in zip(axes, results):
        sns.heatmap(
            r.confusion, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
        )
        ax.set_title(r.model_name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices — Baseline Models")
    fig.tight_layout()
    _save_all(fig, out_dir, name)


def plot_metric_bars(table, out_dir: Path, name: str = "metric_comparison"):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(table))
    width = 0.15
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, table[m], width=width, label=m)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(table["Model"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison Across Baseline Models")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    _save_all(fig, out_dir, name)


def plot_radar_chart(table, out_dir: Path, name: str = "radar_chart"):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for _, row in table.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.4, label=row["Model"])
        ax.fill(angles, values, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Radar Comparison — Baseline Models")
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save_all(fig, out_dir, name)


def plot_violin_latency(results: list, out_dir: Path, name: str = "latency_violin"):
    import pandas as pd

    rows = []
    for r in results:
        rows.append({"Model": r.model_name, "Metric": "Train Time (s)", "Value": r.train_time_s})
        rows.append({"Model": r.model_name, "Metric": "Inference Time (s)", "Value": r.inference_time_s})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="Model", y="Value", hue="Metric", ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("Seconds (log scale)")
    ax.set_title("Computational Cost — Train vs. Inference Time")
    plt.xticks(rotation=30, ha="right")
    _save_all(fig, out_dir, name)


def plot_correlation_matrix(X, feature_names, out_dir: Path, name: str = "correlation_matrix"):
    import pandas as pd

    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Matrix")
    _save_all(fig, out_dir, name)


def plot_feature_importance(model, feature_names, out_dir: Path, name: str, top_k: int = 15):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1], color="#2b6cb0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {name}")
    _save_all(fig, out_dir, f"feature_importance_{name.lower()}")


def plot_pca_tsne(X, y, out_dir: Path, name_prefix: str = "embedding"):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n = min(len(X), 3000)  # cap for t-SNE tractability
    idx = np.random.default_rng(42).choice(len(X), size=n, replace=False)
    Xs, ys = X[idx], y[idx]

    pca = PCA(n_components=2, random_state=42).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=ys, palette="Set1", s=10, alpha=0.6, ax=ax)
    ax.set_title("PCA Projection (2D)")
    _save_all(fig, out_dir, f"{name_prefix}_pca")

    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=ys, palette="Set1", s=10, alpha=0.6, ax=ax)
    ax.set_title("t-SNE Projection (2D)")
    _save_all(fig, out_dir, f"{name_prefix}_tsne")


def plot_boxplots(results: list, out_dir: Path, name: str = "score_boxplots"):
    import pandas as pd

    rows = []
    for r in results:
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            rows.append({"Model": r.model_name, "Metric": m, "Value": getattr(r, m)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="Metric", y="Value", ax=ax)
    sns.stripplot(data=df, x="Metric", y="Value", color="black", alpha=0.5, size=4, ax=ax)
    ax.set_title("Distribution of Metrics Across Models")
    _save_all(fig, out_dir, name)
