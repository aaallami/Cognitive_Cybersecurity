"""
Run the full baseline benchmark (data -> preprocessing -> models -> metrics
-> figures -> tables) and print a summary. This is the script that backs
notebooks 01 and 02; it is executed directly (not just imported) to
produce real, verifiable outputs under outputs/, figures/, and tables/.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cognitivecyber.data.loaders import load_dataset
from cognitivecyber.data.preprocessing import preprocess
from cognitivecyber.evaluation import figures, tables
from cognitivecyber.evaluation.metrics import EvalResult, evaluate_classifier, results_to_table
from cognitivecyber.models.baselines import get_supervised_models, get_unsupervised_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_baseline_benchmark")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
TABLE_DIR = REPO_ROOT / "tables"
OUT_DIR = REPO_ROOT / "outputs"
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def main():
    logger.info("Loading dataset (synthetic UNSW-style flows; see docs/data_provenance.md)...")
    df, meta = load_dataset(n_synthetic=60_000, random_state=RANDOM_STATE)
    df.to_csv(REPO_ROOT / "datasets" / "synthetic_flows.csv", index=False)
    logger.info("Dataset shape: %s | schema: %s | attack ratio: %.3f",
                df.shape, meta["schema_name"], df["label"].mean())

    logger.info("Preprocessing...")
    pdata = preprocess(df, meta, random_state=RANDOM_STATE)
    logger.info("Train/Val/Test shapes: %s / %s / %s",
                pdata.X_train.shape, pdata.X_val.shape, pdata.X_test.shape)

    logger.info("Training supervised baselines...")
    supervised = get_supervised_models(random_state=RANDOM_STATE)
    sup_results: list[EvalResult] = []
    for name, model in supervised.items():
        logger.info("  -> %s", name)
        r = evaluate_classifier(model, pdata.X_train, pdata.y_train, pdata.X_test, pdata.y_test, name)
        sup_results.append(r)
        logger.info(
            "     acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc_auc=%.4f train=%.2fs infer=%.3fs",
            r.accuracy, r.precision, r.recall, r.f1, r.roc_auc, r.train_time_s, r.inference_time_s,
        )

    logger.info("Training unsupervised anomaly detectors...")
    unsupervised = get_unsupervised_models(
        contamination=float(pdata.y_train.mean()), random_state=RANDOM_STATE
    )
    unsup_results: list[EvalResult] = []
    for name, model in unsupervised.items():
        logger.info("  -> %s", name)
        # LOF-novelty and OCSVM/IsolationForest are fit unsupervised (labels ignored)
        r = evaluate_classifier(
            model, pdata.X_train, pdata.y_train, pdata.X_test, pdata.y_test, name,
            score_fn="decision_function",
        )
        unsup_results.append(r)
        logger.info(
            "     acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc_auc=%.4f",
            r.accuracy, r.precision, r.recall, r.f1, r.roc_auc,
        )

    all_results = sup_results + unsup_results

    logger.info("Building tables...")
    sup_table = results_to_table(sup_results)
    unsup_table = results_to_table(unsup_results)
    full_table = results_to_table(all_results)
    tables.export_table(sup_table, TABLE_DIR, "supervised_baseline_results")
    tables.export_table(unsup_table, TABLE_DIR, "unsupervised_baseline_results")
    tables.export_table(full_table, TABLE_DIR, "all_baseline_results")

    logger.info("Generating figures...")
    figures.plot_roc_curves(sup_results, FIG_DIR, "roc_curves_supervised")
    figures.plot_pr_curves(sup_results, FIG_DIR, "pr_curves_supervised")
    figures.plot_confusion_matrices(sup_results, FIG_DIR, "confusion_matrices_supervised")
    figures.plot_metric_bars(sup_table, FIG_DIR, "metric_comparison_supervised")
    figures.plot_radar_chart(sup_table, FIG_DIR, "radar_chart_supervised")
    figures.plot_violin_latency(all_results, FIG_DIR, "computational_cost")
    figures.plot_boxplots(all_results, FIG_DIR, "score_boxplots")
    figures.plot_correlation_matrix(pdata.X_train, pdata.feature_names, FIG_DIR, "correlation_matrix")
    figures.plot_pca_tsne(pdata.X_test, pdata.y_test, FIG_DIR, "test_set_embedding")

    rf_model = supervised["RandomForest"]
    figures.plot_feature_importance(rf_model, pdata.feature_names, FIG_DIR, "RandomForest")
    xgb_model = supervised["XGBoost"]
    figures.plot_feature_importance(xgb_model, pdata.feature_names, FIG_DIR, "XGBoost")

    logger.info("Writing run manifest...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "random_state": RANDOM_STATE,
        "dataset_meta": {k: v for k, v in meta.items()},
        "n_samples": int(df.shape[0]),
        "n_features_after_encoding": len(pdata.feature_names),
        "attack_ratio": float(df["label"].mean()),
        "supervised_models": list(supervised.keys()),
        "unsupervised_models": list(unsupervised.keys()),
        "best_supervised_model": sup_table.iloc[0]["Model"],
        "best_supervised_f1": float(sup_table.iloc[0]["F1"]),
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("Done. Best supervised model: %s (F1=%.4f)",
                manifest["best_supervised_model"], manifest["best_supervised_f1"])
    return sup_table, unsup_table, full_table


if __name__ == "__main__":
    main()
