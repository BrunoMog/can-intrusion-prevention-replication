"""Etapa 1: 10-Fold CV training, selection, and blind-test evaluation of Stage-2
Isolation Forest models (CAN_ID + 8 payload bytes) for the PoC 2-Stage IPS.

Strictly preserves zero data leakage: all Stage-2 architectures and hyperparameters
are selected solely on 10-Fold Cross-Validation over (train_val_normal + val_attack),
using `iForest` only (no OCSVM), before fitting on `train_normal` and evaluating once
on the blind Test split.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

POC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(POC_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

import config
import run_experiments
import utils
from models_stage2 import (
    DEFAULT_MIN_ID_SAMPLES,
    STAGE2_ARCHITECTURES,
    Stage2IsolationForest,
    evaluate_bundle_for_arch_and_contamination,
    fit_fold_base_models_and_cache,
    get_model_artifact_paths,
)

POC_MODELS_DIR = os.path.join(POC_DIR, "models")
POC_RESULTS_DIR = os.path.join(POC_DIR, "results")


def load_stage1_best_params(final_results_csv: str) -> Dict[str, Dict[str, Any]]:
    """Load the latest Stage-1 iForest_5F hyperparameters per dataset from Stage 0."""
    df = pd.read_csv(final_results_csv)
    mask = (df["model_family"] == "iForest") & (df["n_bytes"] == 5)
    sub = df.loc[mask].sort_values("timestamp")
    params_by_ds: Dict[str, Dict[str, Any]] = {}
    for ds_name, group in sub.groupby("dataset"):
        params_by_ds[str(ds_name)] = json.loads(group.iloc[-1]["best_params_json"])
    return params_by_ds


def load_train_val_pool(
    dataset_name: str, random_state: int
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read `train_val` + `val_attack` rows once from CSV, indexed by sorted row IDs."""
    sp = utils.get_split_indices(dataset_name, random_state=random_state)
    unique_tv_idx = np.sort(
        np.unique(np.concatenate([sp["train_val_normal_idx"], sp["val_attack_idx"]]))
    )
    df_tv = utils.load_can_records_by_indices(
        dataset_name, unique_tv_idx, add_normalized_cols=True
    )
    can_ids = df_tv["CAN_ID"].to_numpy()
    x_8b = df_tv[utils.NORM_BYTE_COLS].to_numpy(dtype=np.float64)
    y = df_tv["Label"].to_numpy(dtype=np.int64)
    return sp, unique_tv_idx, can_ids, x_8b, y


def lookup_pool_positions(sorted_pool_idx: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    """Map raw CSV row indices to compact pool positions in sorted order."""
    return np.searchsorted(sorted_pool_idx, np.sort(target_idx))


def compute_stage1_fold_predictions(
    sp: Dict[str, np.ndarray],
    unique_tv_idx: np.ndarray,
    x_8b: np.ndarray,
    y: np.ndarray,
    stage1_params: Dict[str, Any],
    partition_val: bool,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute Stage-1 (iForest_5F) predictions on each of the 10 CV folds."""
    fold_preds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(10):
        tr_idx, va_idx = utils.get_kfold_cv_indices(
            sp["train_val_normal_idx"],
            sp["val_attack_idx"],
            fold_idx=fold_idx,
            n_splits=10,
            random_state=random_state,
            partition_val_attacks_across_folds=partition_val,
        )
        tr_pos = lookup_pool_positions(unique_tv_idx, tr_idx)
        va_pos = lookup_pool_positions(unique_tv_idx, va_idx)

        clf1 = run_experiments.instantiate_model(
            "iForest", stage1_params, random_state=random_state, n_jobs=-1
        )
        clf1.fit(x_8b[tr_pos, :5])
        pred1 = (clf1.predict(x_8b[va_pos, :5]) == -1).astype(np.int64)
        fold_preds.append((y[va_pos], pred1))
    return fold_preds


def compute_cascade_summary(
    y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray
) -> Dict[str, float]:
    """Compute Cascaded (pred1 & pred2) detection metrics and FP rescue vs Stage 1."""
    cascade_pred = (pred1 & pred2).astype(np.int64)
    metrics = utils.evaluate_detection_metrics(y_true, cascade_pred, scores=None)

    fp_s1 = int(np.sum((y_true == 0) & (pred1 == 1)))
    fp_cas = int(metrics["FP"])
    tp_s1 = int(np.sum((y_true == 1) & (pred1 == 1)))
    tp_cas = int(metrics["TP"])

    fp_rescue_pct = 100.0 * (fp_s1 - fp_cas) / fp_s1 if fp_s1 > 0 else 0.0
    tp_loss_pct = 100.0 * (tp_s1 - tp_cas) / tp_s1 if tp_s1 > 0 else 0.0
    return {
        "cascade_accuracy": float(metrics["Accuracy (%)"]),
        "cascade_precision": float(metrics["Precision (%)"]),
        "cascade_recall": float(metrics["Recall (%)"]),
        "cascade_f1": float(metrics["F1-Score (%)"]),
        "cascade_fp": fp_cas,
        "cascade_tp": tp_cas,
        "cascade_fn": int(metrics["FN"]),
        "fp_rescue_pct": fp_rescue_pct,
        "tp_loss_pct": tp_loss_pct,
    }


_FORK_CV_CTX: Dict[str, Any] = {}


def _evaluate_single_fold_worker(
    fold_idx: int,
) -> Dict[Tuple[str, int, Any, Any], Dict[str, float]]:
    """Run all base fits + arch/contamination evaluations for a single fold."""
    ctx = _FORK_CV_CTX
    t_fold = time.perf_counter()
    tr_idx, va_idx = utils.get_kfold_cv_indices(
        ctx["train_val_normal_idx"],
        ctx["val_attack_idx"],
        fold_idx=fold_idx,
        n_splits=10,
        random_state=ctx["random_state"],
        partition_val_attacks_across_folds=ctx["partition_val"],
    )
    tr_pos = lookup_pool_positions(ctx["unique_tv_idx"], tr_idx)
    va_pos = lookup_pool_positions(ctx["unique_tv_idx"], va_idx)
    y_val_fold, pred1_fold = ctx["s1_fold_preds"][fold_idx]

    can_ids = ctx["can_ids"]
    x_8b = ctx["x_8b"]
    param_grid = ctx["param_grid"]
    n_estimators_list = param_grid["n_estimators"]
    max_samples_list = param_grid["max_samples"]
    contamination_list = param_grid["contamination"]

    fold_records: Dict[Tuple[str, int, Any, Any], Dict[str, float]] = {}
    base_cache: Dict[Tuple[int, int], Any] = {}

    for ne in n_estimators_list:
        for ms in max_samples_list:
            eff_ms_key = 256 if ms == "auto" else int(ms)
            key = (int(ne), eff_ms_key)
            if key not in base_cache:
                base_cache[key] = fit_fold_base_models_and_cache(
                    train_can_ids=can_ids[tr_pos],
                    x_train_8b=x_8b[tr_pos],
                    val_can_ids=can_ids[va_pos],
                    x_val_8b=x_8b[va_pos],
                    n_estimators=int(ne),
                    max_samples=ms,
                    min_id_samples=DEFAULT_MIN_ID_SAMPLES,
                    random_state=ctx["random_state"],
                    n_jobs=1,  # Single thread per fold avoids nested OpenMP/loky overhead
                )
            bundle = base_cache[key]

            for arch in STAGE2_ARCHITECTURES:
                for contam in contamination_list:
                    pred2, scores2 = evaluate_bundle_for_arch_and_contamination(
                        bundle, arch=arch, contamination=contam
                    )
                    standalone = utils.evaluate_detection_metrics(
                        y_val_fold, pred2, scores=scores2
                    )
                    cascade = compute_cascade_summary(y_val_fold, pred1_fold, pred2)
                    fold_records[(arch, ne, ms, contam)] = {
                        "val_accuracy": float(standalone["Accuracy (%)"]),
                        "val_precision": float(standalone["Precision (%)"]),
                        "val_recall": float(standalone["Recall (%)"]),
                        "val_f1": float(standalone["F1-Score (%)"]),
                        "val_aucroc": float(standalone["AUC-ROC (%)"]),
                        "val_fp": float(standalone["FP"]),
                        "val_fn": float(standalone["FN"]),
                        **cascade,
                    }

    print(
        f"  [{ctx['dataset_name']}] Fold {fold_idx + 1}/10 complete in "
        f"{time.perf_counter() - t_fold:.2f}s"
    )
    return fold_records


def run_stage2_cv_for_dataset(
    dataset_name: str,
    sp: Dict[str, np.ndarray],
    unique_tv_idx: np.ndarray,
    can_ids: np.ndarray,
    x_8b: np.ndarray,
    y: np.ndarray,
    stage1_params: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    random_state: int,
    fold_workers: int = 5,
) -> pd.DataFrame:
    """Run 10-Fold CV over all Stage-2 architectures and iForest grid configs."""
    split_cfg = utils._get_split_config(dataset_name)
    partition_val = bool(split_cfg["partition_val_attacks_across_folds"])
    s1_fold_preds = compute_stage1_fold_predictions(
        sp, unique_tv_idx, x_8b, y, stage1_params, partition_val, random_state
    )

    n_estimators_list = param_grid["n_estimators"]
    max_samples_list = param_grid["max_samples"]
    contamination_list = param_grid["contamination"]

    records: Dict[Tuple[str, int, Any, Any], List[Dict[str, float]]] = {}
    for arch in STAGE2_ARCHITECTURES:
        for ne in n_estimators_list:
            for ms in max_samples_list:
                for contam in contamination_list:
                    records[(arch, ne, ms, contam)] = []

    _FORK_CV_CTX.clear()
    _FORK_CV_CTX.update(
        {
            "dataset_name": dataset_name,
            "train_val_normal_idx": sp["train_val_normal_idx"],
            "val_attack_idx": sp["val_attack_idx"],
            "unique_tv_idx": unique_tv_idx,
            "can_ids": can_ids,
            "x_8b": x_8b,
            "s1_fold_preds": s1_fold_preds,
            "param_grid": param_grid,
            "partition_val": partition_val,
            "random_state": random_state,
        }
    )

    t_cv_start = time.perf_counter()
    if fold_workers > 1:
        mp_ctx = multiprocessing.get_context("fork")
        with mp_ctx.Pool(processes=min(fold_workers, 10)) as pool:
            fold_outputs = pool.map(_evaluate_single_fold_worker, range(10))
    else:
        fold_outputs = [_evaluate_single_fold_worker(f) for f in range(10)]
    _FORK_CV_CTX.clear()

    for fold_dict in fold_outputs:
        for key, metrics_dict in fold_dict.items():
            records[key].append(metrics_dict)

    print(f"  [{dataset_name}] Total 10-Fold CV time: {time.perf_counter() - t_cv_start:.2f}s")

    rows: List[Dict[str, Any]] = []
    for (arch, ne, ms, contam), fold_list in records.items():
        fdf = pd.DataFrame(fold_list)
        params_dict = {
            "n_estimators": int(ne),
            "max_samples": int(ms) if ms != "auto" else "auto",
            "contamination": float(contam) if contam != "auto" else "auto",
        }
        rows.append(
            {
                "dataset": dataset_name,
                "arch": arch,
                "params_json": json.dumps(params_dict, sort_keys=True),
                "n_estimators": params_dict["n_estimators"],
                "max_samples": str(ms),
                "contamination": str(contam),
                "cv_accuracy_mean": float(fdf["val_accuracy"].mean()),
                "cv_accuracy_std": float(fdf["val_accuracy"].std(ddof=1)),
                "cv_precision_mean": float(fdf["val_precision"].mean()),
                "cv_recall_mean": float(fdf["val_recall"].mean()),
                "cv_f1_mean": float(fdf["val_f1"].mean()),
                "cv_aucroc_mean": float(fdf["val_aucroc"].mean()),
                "cv_fp_mean": float(fdf["val_fp"].mean()),
                "cv_fn_mean": float(fdf["val_fn"].mean()),
                "cascade_cv_accuracy_mean": float(fdf["cascade_accuracy"].mean()),
                "cascade_cv_precision_mean": float(fdf["cascade_precision"].mean()),
                "cascade_cv_recall_mean": float(fdf["cascade_recall"].mean()),
                "cascade_cv_f1_mean": float(fdf["cascade_f1"].mean()),
                "cascade_cv_fp_mean": float(fdf["cascade_fp"].mean()),
                "cascade_cv_fn_mean": float(fdf["cascade_fn"].mean()),
                "cascade_cv_fp_rescue_pct_mean": float(fdf["fp_rescue_pct"].mean()),
                "cascade_cv_tp_loss_pct_mean": float(fdf["tp_loss_pct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def select_cv_winner_row(df_subset: pd.DataFrame) -> pd.Series:
    """Rank CV configurations strictly on validation metrics (matching paper criterion)."""
    ranked = df_subset.sort_values(
        by=[
            "cv_accuracy_mean",
            "cv_f1_mean",
            "cv_aucroc_mean",
            "cascade_cv_accuracy_mean",
        ],
        ascending=[False, False, False, False],
    )
    return ranked.iloc[0]


def evaluate_and_persist_for_dataset(
    dataset_name: str,
    sp: Dict[str, np.ndarray],
    unique_tv_idx: np.ndarray,
    can_ids_tv: np.ndarray,
    x_8b_tv: np.ndarray,
    stage1_params: Dict[str, Any],
    cv_ds_df: pd.DataFrame,
    random_state: int,
) -> List[Dict[str, Any]]:
    """Refit Stage-1 + all CV winners on `train_normal_idx`, persist overall best, evaluate Test."""
    tr_norm_pos = lookup_pool_positions(unique_tv_idx, sp["train_normal_idx"])
    can_ids_tr = can_ids_tv[tr_norm_pos]
    x_8b_tr = x_8b_tv[tr_norm_pos]

    overall_winner = select_cv_winner_row(cv_ds_df)
    selected_arch = str(overall_winner["arch"])
    selected_params_json = str(overall_winner["params_json"])

    # Fit Stage 1 (`iForest_5F`) once on `train_normal_idx` and persist
    clf1 = run_experiments.instantiate_model(
        "iForest", stage1_params, random_state=random_state, n_jobs=-1
    ).fit(x_8b_tr[:, :5])
    s1_path, s2_path = get_model_artifact_paths(POC_MODELS_DIR, dataset_name)
    joblib.dump(clf1, s1_path)

    # Load blind test split once
    df_test = utils.load_test_data(
        dataset_name, random_state=random_state, add_normalized_cols=True
    )
    y_test = df_test["Label"].to_numpy(dtype=np.int64)
    test_can_ids = df_test["CAN_ID"].to_numpy()
    x_test_8b = df_test[utils.NORM_BYTE_COLS].to_numpy(dtype=np.float64)

    t0 = time.perf_counter()
    pred1_test = (clf1.predict(x_test_8b[:, :5]) == -1).astype(np.int64)
    scores1_test = -clf1.decision_function(x_test_8b[:, :5])
    s1_test_sec = time.perf_counter() - t0
    s1_metrics = utils.evaluate_detection_metrics(y_test, pred1_test, scores=scores1_test)

    fp_s1 = int(s1_metrics["FP"])
    tp_s1 = int(s1_metrics["TP"])
    ts = datetime.now().isoformat()

    test_rows: List[Dict[str, Any]] = [
        {
            "timestamp": ts,
            "dataset": dataset_name,
            "role": "Stage1_Baseline_5F",
            "arch": "iForest_5F",
            "is_cv_selected_stage2": False,
            "params_json": json.dumps(stage1_params, sort_keys=True),
            "cv_accuracy_mean": np.nan,
            "cv_f1_mean": np.nan,
            "test_samples": len(y_test),
            "test_accuracy": float(s1_metrics["Accuracy (%)"]),
            "test_precision": float(s1_metrics["Precision (%)"]),
            "test_recall": float(s1_metrics["Recall (%)"]),
            "test_f1": float(s1_metrics["F1-Score (%)"]),
            "test_aucroc": float(s1_metrics["AUC-ROC (%)"]),
            "test_tp": tp_s1,
            "test_fp": fp_s1,
            "test_tn": int(s1_metrics["TN"]),
            "test_fn": int(s1_metrics["FN"]),
            "cascade_test_accuracy": float(s1_metrics["Accuracy (%)"]),
            "cascade_test_precision": float(s1_metrics["Precision (%)"]),
            "cascade_test_recall": float(s1_metrics["Recall (%)"]),
            "cascade_test_f1": float(s1_metrics["F1-Score (%)"]),
            "cascade_test_tp": tp_s1,
            "cascade_test_fp": fp_s1,
            "cascade_test_fn": int(s1_metrics["FN"]),
            "test_fp_rescue_pct": 0.0,
            "test_tp_loss_pct": 0.0,
            "fit_time_sec": np.nan,
            "test_inference_sec": s1_test_sec,
        }
    ]

    # Refit each architecture's CV champion and evaluate on Test for full transparency
    for arch in STAGE2_ARCHITECTURES:
        arch_sub = cv_ds_df[cv_ds_df["arch"] == arch]
        best_row = select_cv_winner_row(arch_sub)
        params = json.loads(str(best_row["params_json"]))

        t_fit = time.perf_counter()
        model2 = Stage2IsolationForest(
            arch=arch,
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            contamination=params["contamination"],
            min_id_samples=DEFAULT_MIN_ID_SAMPLES,
            random_state=random_state,
            n_jobs=-1,
        ).fit(can_ids_tr, x_8b_tr)
        fit_sec = time.perf_counter() - t_fit

        t_inf = time.perf_counter()
        pred2_test, scores2_test = model2.predict(test_can_ids, x_test_8b)
        inf_sec = time.perf_counter() - t_inf

        s2_metrics = utils.evaluate_detection_metrics(y_test, pred2_test, scores=scores2_test)
        cascade = compute_cascade_summary(y_test, pred1_test, pred2_test)

        is_deployed = arch == selected_arch and str(best_row["params_json"]) == selected_params_json
        if is_deployed:
            joblib.dump(model2, s2_path)
            print(
                f"  >>> Persisted Stage-1 -> {s1_path}\n"
                f"  >>> Persisted Stage-2 ({arch}) -> {s2_path}"
            )

        test_rows.append(
            {
                "timestamp": ts,
                "dataset": dataset_name,
                "role": f"Stage2_Candidate_{arch}",
                "arch": arch,
                "is_cv_selected_stage2": is_deployed,
                "params_json": str(best_row["params_json"]),
                "cv_accuracy_mean": float(best_row["cv_accuracy_mean"]),
                "cv_f1_mean": float(best_row["cv_f1_mean"]),
                "test_samples": len(y_test),
                "test_accuracy": float(s2_metrics["Accuracy (%)"]),
                "test_precision": float(s2_metrics["Precision (%)"]),
                "test_recall": float(s2_metrics["Recall (%)"]),
                "test_f1": float(s2_metrics["F1-Score (%)"]),
                "test_aucroc": float(s2_metrics["AUC-ROC (%)"]),
                "test_tp": int(s2_metrics["TP"]),
                "test_fp": int(s2_metrics["FP"]),
                "test_tn": int(s2_metrics["TN"]),
                "test_fn": int(s2_metrics["FN"]),
                "cascade_test_accuracy": cascade["cascade_accuracy"],
                "cascade_test_precision": cascade["cascade_precision"],
                "cascade_test_recall": cascade["cascade_recall"],
                "cascade_test_f1": cascade["cascade_f1"],
                "cascade_test_tp": cascade["cascade_tp"],
                "cascade_test_fp": cascade["cascade_fp"],
                "cascade_test_fn": cascade["cascade_fn"],
                "test_fp_rescue_pct": cascade["fp_rescue_pct"],
                "test_tp_loss_pct": cascade["tp_loss_pct"],
                "fit_time_sec": fit_sec,
                "test_inference_sec": inf_sec,
            }
        )

    del df_test
    gc.collect()
    return test_rows


def merge_dataset_csv(csv_path: str, new_df: pd.DataFrame, dataset_name: str) -> None:
    """Update a results CSV by replacing rows for `dataset_name` and keeping others."""
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[existing["dataset"] != dataset_name]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 CAN_ID+8B Isolation Forest models.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Speedometer", "Fuzzing", "Drive Gear", "RPM Gauge"],
        help="Datasets to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help="Random seed matching split caches.",
    )
    parser.add_argument(
        "--fold-workers",
        type=int,
        default=5,
        help="Parallel worker processes across the 10 CV folds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(POC_MODELS_DIR, exist_ok=True)
    os.makedirs(POC_RESULTS_DIR, exist_ok=True)

    stage1_params_map = load_stage1_best_params(
        os.path.join(REPO_ROOT, "results", "final_test_results.csv")
    )
    param_grid = config.PARAM_GRID_FULL["iForest"]

    cv_out_path = os.path.join(POC_RESULTS_DIR, "stage2_cv_results.csv")
    test_out_path = os.path.join(POC_RESULTS_DIR, "stage2_test_comparison.csv")

    for ds_name in args.datasets:
        print(f"\n{'=' * 72}\n[Stage-2 Training] Dataset: {ds_name}\n{'=' * 72}")
        sp, unique_tv_idx, can_ids_tv, x_8b_tv, y_tv = load_train_val_pool(
            ds_name, random_state=args.seed
        )
        cv_df = run_stage2_cv_for_dataset(
            dataset_name=ds_name,
            sp=sp,
            unique_tv_idx=unique_tv_idx,
            can_ids=can_ids_tv,
            x_8b=x_8b_tv,
            y=y_tv,
            stage1_params=stage1_params_map[ds_name],
            param_grid=param_grid,
            random_state=args.seed,
            fold_workers=args.fold_workers,
        )

        winner = select_cv_winner_row(cv_df)
        print(
            f"  [CV Selected] {winner['arch']} | {winner['params_json']} | "
            f"CV Acc={winner['cv_accuracy_mean']:.2f}% (Cascade {winner['cascade_cv_accuracy_mean']:.2f}%, "
            f"FP rescue {winner['cascade_cv_fp_rescue_pct_mean']:.1f}%)"
        )

        ds_test_rows = evaluate_and_persist_for_dataset(
            dataset_name=ds_name,
            sp=sp,
            unique_tv_idx=unique_tv_idx,
            can_ids_tv=can_ids_tv,
            x_8b_tv=x_8b_tv,
            stage1_params=stage1_params_map[ds_name],
            cv_ds_df=cv_df,
            random_state=args.seed,
        )

        merge_dataset_csv(cv_out_path, cv_df, ds_name)
        merge_dataset_csv(test_out_path, pd.DataFrame(ds_test_rows), ds_name)

        del can_ids_tv, x_8b_tv, y_tv
        gc.collect()

    print(f"\nSaved Stage-2 CV results -> {cv_out_path}")
    print(f"Saved Stage-2 blind Test comparison -> {test_out_path}")


if __name__ == "__main__":
    main()
