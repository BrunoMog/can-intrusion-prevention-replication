#!/usr/bin/env python3
"""
Equivalence & Unit Test Suite for Native C vs scikit-learn Inference (PoC IPS).

Verifies:
1. Self-contained Liu et al. (2008) `average_path_length` closed-form matches
   reference analytical values and edge cases (n <= 1, n = 2, n > 2).
2. `FastIsolationForest` C library matches `sklearn.ensemble.IsolationForest`
   on edge-case synthetic forests (including near-threshold perturbations).
3. `FastStage2Evaluator` C library matches `Stage2IsolationForest` across all
   architectures (`global_9f`, `hybrid_fallback`, `hybrid_strict_unknown`),
   including rare and unseen CAN IDs.
4. Full-scale numerical & decision parity on all persisted Stage-1 (`5F`) and
   Stage-2 (`CAN_ID + 8B`) `.joblib` artifacts over real blind-test frames and
   mined near-boundary samples (`|decision_function| -> 0`).

Usage:
    /home/bruno/venvs/cin0114_venv/bin/python poc_ips/test_equivalence.py
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

POC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(POC_DIR)
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
import utils
from fast_iforest import FastIsolationForest, FastStage2Evaluator, average_path_length
from models_stage2 import (
    STAGE2_ARCHITECTURES,
    Stage2IsolationForest,
    get_model_artifact_paths,
    parse_can_ids_to_int,
)


def _analytic_c(n: int) -> float:
    """Scalar closed-form Liu et al. (2008) reference for unit assertions."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (float(np.log(n - 1.0)) + float(np.euler_gamma)) - 2.0 * (n - 1.0) / float(n)


def _select_eval_indices(
    margins: np.ndarray,
    labels: np.ndarray,
    max_boundary: int = 400,
    max_normal: int = 1500,
    max_attack: int = 1000,
    max_stride: int = 1500,
) -> np.ndarray:
    """Select near-boundary, normal, attack, and uniform strided indices cleanly."""
    n = len(margins)
    chosen: List[np.ndarray] = []

    boundary_order = np.argsort(np.abs(margins))[: min(max_boundary, n)]
    chosen.append(boundary_order)

    norm_idx = np.flatnonzero(labels == 0)
    if len(norm_idx) > 0:
        step = max(1, len(norm_idx) // max_normal)
        chosen.append(norm_idx[::step][:max_normal])

    atk_idx = np.flatnonzero(labels == 1)
    if len(atk_idx) > 0:
        step = max(1, len(atk_idx) // max_attack)
        chosen.append(atk_idx[::step][:max_attack])

    stride = max(1, n // max_stride)
    chosen.append(np.arange(0, n, stride, dtype=np.int64)[:max_stride])

    return np.unique(np.concatenate(chosen))


class TestFastIsolationForestEquivalence(unittest.TestCase):
    """Unit and integration parity tests for C IsolationForest engine."""

    def test_average_path_length_values_and_shapes(self) -> None:
        test_ns = [0, 1, 2, 3, 4, 16, 256, 512, 16384]
        expected = np.array([_analytic_c(v) for v in test_ns], dtype=np.float64)
        got = average_path_length(np.array(test_ns, dtype=np.int64))
        self.assertEqual(got.shape, (len(test_ns),))
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-15)

        # Scalar and 2D array preservation
        scalar_out = average_path_length(256)
        self.assertEqual(scalar_out.shape, ())
        self.assertAlmostEqual(float(scalar_out), _analytic_c(256), places=14)

        mat_in = np.array([[1, 2], [256, 512]], dtype=np.int64)
        mat_out = average_path_length(mat_in)
        self.assertEqual(mat_out.shape, (2, 2))
        self.assertAlmostEqual(float(mat_out[1, 1]), _analytic_c(512), places=14)

    def test_synthetic_iforest_edge_cases_and_threshold_boundary(self) -> None:
        rng = np.random.default_rng(12345)
        x_train = rng.uniform(0.0, 1.0, size=(300, 5))

        for max_samp in (1, 2, 64, 256):
            clf = IsolationForest(
                n_estimators=25,
                max_samples=max_samp,
                contamination=0.05,
                random_state=42,
            ).fit(x_train)
            fast = FastIsolationForest(clf)

            # Generate probes including tree threshold neighbourhoods
            probes: List[np.ndarray] = [
                rng.uniform(0.0, 1.0, size=(120, 5)).astype(np.float32),
                np.zeros((1, 5), dtype=np.float32),
                np.ones((1, 5), dtype=np.float32),
            ]
            t0 = clf.estimators_[0].tree_
            valid_nodes = np.flatnonzero(t0.feature >= 0)
            for node in valid_nodes[:10]:
                feat = int(t0.feature[node])
                thr = float(t0.threshold[node])
                for eps in (-1e-4, 0.0, 1e-4):
                    v = np.full((1, 5), 0.5, dtype=np.float32)
                    v[0, feat] = np.float32(np.clip(thr + eps, 0.0, 1.0))
                    probes.append(v)

            x_probe = np.ascontiguousarray(np.vstack(probes), dtype=np.float32)
            sk_scores = clf.decision_function(x_probe)
            sk_preds = np.where(clf.predict(x_probe) == -1, 1, 0)

            max_err = 0.0
            mismatches = 0
            for i in range(len(x_probe)):
                c_score = fast.decision_one_f32(x_probe[i])
                c_pred = fast.predict_one_f32(x_probe[i])
                max_err = max(max_err, abs(c_score - float(sk_scores[i])))
                if c_pred != int(sk_preds[i]):
                    mismatches += 1

            self.assertEqual(mismatches, 0, f"Mismatch for max_samples={max_samp}")
            self.assertLess(
                max_err,
                1e-12,
                f"Score divergence {max_err:.3e} exceeds 1e-12 for max_samples={max_samp}",
            )

    def test_synthetic_stage2_architectures_and_unseen_ids(self) -> None:
        rng = np.random.default_rng(77)
        known_ids = ["0100"] * 30 + ["0200"] * 30 + ["0050"] * 4
        x_tr = rng.uniform(0.0, 1.0, size=(len(known_ids), 8))

        test_ids = ["0100", "0200", "0050", "07FE", "0x0100"]
        x_te = np.ascontiguousarray(rng.uniform(0.0, 1.0, size=(len(test_ids), 8)), dtype=np.float32)
        cids_int = parse_can_ids_to_int(test_ids)

        for arch in STAGE2_ARCHITECTURES:
            model = Stage2IsolationForest(
                arch=arch,
                n_estimators=20,
                max_samples=32,
                contamination=0.05,
                min_id_samples=10,
                random_state=42,
                n_jobs=1,
            ).fit(known_ids, x_tr)
            fast_s2 = FastStage2Evaluator(model)

            sk_preds, sk_scores = model.predict(test_ids, x_te)
            for idx in range(len(test_ids)):
                c_pred, c_score = fast_s2.predict_one(int(cids_int[idx]), x_te[idx])
                self.assertEqual(
                    c_pred,
                    int(sk_preds[idx]),
                    f"Stage-2 pred mismatch on arch={arch}, id={test_ids[idx]}",
                )
                self.assertLess(
                    abs(c_score - float(sk_scores[idx])),
                    1e-6,
                    f"Stage-2 score diff on arch={arch}, id={test_ids[idx]}",
                )

            # Specifically verify strict handling on unseen 07FE (higher anomaly_score > 0 = BLOCK)
            if arch == "iForest_Hybrid_StrictUnknown":
                c_pred_u, c_score_u = fast_s2.predict_one(int(cids_int[3]), x_te[3])
                self.assertEqual(c_pred_u, 1)
                self.assertGreater(c_score_u, 0.0)

    def test_persisted_poc_models_synthetic_probe(self) -> None:
        """Verify persisted .joblib Stage-1/Stage-2 artifacts against C on synthetic probes."""
        models_dir = os.path.join(POC_DIR, "models")
        tested = 0
        rng = np.random.default_rng(2025)

        for ds_name in config.ALL_DATASETS:
            s1_path, s2_path = get_model_artifact_paths(models_dir, ds_name)
            if not (os.path.exists(s1_path) and os.path.exists(s2_path)):
                continue

            s1_clf = joblib.load(s1_path)
            s2_model = joblib.load(s2_path)
            fast_s1 = FastIsolationForest(s1_clf)
            fast_s2 = FastStage2Evaluator(s2_model)

            x8 = np.ascontiguousarray(
                np.vstack([rng.uniform(0.0, 1.0, size=(300, 8)), np.zeros((2, 8)), np.ones((2, 8))]),
                dtype=np.float32,
            )
            x5 = np.ascontiguousarray(x8[:, :5], dtype=np.float32)
            known_ids = list(getattr(s2_model, "local_models", {}).keys())[:4] or ["0316"]
            cids_str = np.array([known_ids[i % len(known_ids)] for i in range(len(x8))])
            cids_str[-2:] = "07FE"
            cids_int = parse_can_ids_to_int(cids_str)

            sk1_scores = s1_clf.decision_function(x5)
            sk1_preds = np.where(s1_clf.predict(x5) == -1, 1, 0)
            sk2_preds, sk2_scores = s2_model.predict(cids_str, x8)

            for i in range(len(x8)):
                cs1 = fast_s1.decision_one_f32(x5[i])
                cp1 = fast_s1.predict_one_f32(x5[i])
                cp2, cs2 = fast_s2.predict_one(int(cids_int[i]), x8[i])
                self.assertEqual(cp1, int(sk1_preds[i]), f"Synthetic Stage-1 mismatch on {ds_name}")
                self.assertEqual(cp2, int(sk2_preds[i]), f"Synthetic Stage-2 mismatch on {ds_name}")
                self.assertLess(abs(cs1 - float(sk1_scores[i])), 1e-11)
                self.assertLess(abs(cs2 - float(sk2_scores[i])), 1e-11)
            tested += 1

        if tested == 0:
            self.skipTest("No persisted Stage-1/Stage-2 .joblib models present in poc_ips/models/.")

    def test_persisted_poc_models_equivalence_on_real_traces(self) -> None:
        """Integration test: verify persisted C vs sklearn parity strictly on real trace splits."""
        models_dir = os.path.join(POC_DIR, "models")
        max_scan_env = os.environ.get("CAN_IPS_EQUIV_MAX_SCAN", "15000")
        max_scan = int(max_scan_env)
        summaries: List[Tuple[str, int, int, float, int, float, int, float]] = []

        for ds_name in config.ALL_DATASETS:
            s1_path, s2_path = get_model_artifact_paths(models_dir, ds_name)
            if not (os.path.exists(s1_path) and os.path.exists(s2_path)):
                continue

            ds_file = utils.DATASET_PATHS.get(ds_name, "")
            has_real_trace = bool(ds_file and os.path.exists(ds_file))
            if not has_real_trace and ds_name == "Speedometer":
                raw_log = utils.RAW_LOG_PATHS.get("Speedometer", "")
                has_real_trace = bool(raw_log and os.path.exists(raw_log))
            if not has_real_trace:
                # Never fabricate random data under a real-trace integration test
                continue

            s1_clf = joblib.load(s1_path)
            s2_model = joblib.load(s2_path)
            fast_s1 = FastIsolationForest(s1_clf)
            fast_s2 = FastStage2Evaluator(s2_model)

            df = utils.load_test_data(ds_name, add_normalized_cols=True)
            if max_scan > 0 and len(df) > max_scan:
                scan_step = max(1, len(df) // max_scan)
                df_scan = df.iloc[::scan_step].reset_index(drop=True)
            else:
                df_scan = df
            x8 = np.ascontiguousarray(df_scan[utils.NORM_BYTE_COLS].to_numpy(dtype=np.float32))
            x5 = np.ascontiguousarray(x8[:, :5], dtype=np.float32)
            cids_str = df_scan["CAN_ID"].to_numpy()
            cids_int = parse_can_ids_to_int(cids_str)
            labels = df_scan["Label"].to_numpy(dtype=np.int64)

            sk1_scores_all = s1_clf.decision_function(x5)
            idx1 = _select_eval_indices(sk1_scores_all, labels)
            sk1_preds = np.where(s1_clf.predict(x5[idx1]) == -1, 1, 0)
            sk1_scores = sk1_scores_all[idx1]
            min_s1_margin = float(np.min(np.abs(sk1_scores)))

            max_s1_err = 0.0
            s1_mismatches = 0
            for j, i in enumerate(idx1):
                cs = fast_s1.decision_one_f32(x5[i])
                cp = fast_s1.predict_one_f32(x5[i])
                max_s1_err = max(max_s1_err, abs(cs - float(sk1_scores[j])))
                if cp != int(sk1_preds[j]):
                    s1_mismatches += 1

            sk2_preds_all, sk2_scores_all = s2_model.predict(cids_str, x8)
            idx2 = _select_eval_indices(sk2_scores_all, labels)
            min_s2_margin = float(np.min(np.abs(sk2_scores_all[idx2])))

            max_s2_err = 0.0
            s2_mismatches = 0
            for i in idx2:
                cp2, cs2 = fast_s2.predict_one(int(cids_int[i]), x8[i])
                max_s2_err = max(max_s2_err, abs(cs2 - float(sk2_scores_all[i])))
                if cp2 != int(sk2_preds_all[i]):
                    s2_mismatches += 1

            summaries.append(
                (ds_name, len(df_scan), len(idx1), max_s1_err, s1_mismatches, max_s2_err, s2_mismatches, min(min_s1_margin, min_s2_margin))
            )

            self.assertEqual(s1_mismatches, 0, f"Stage-1 prediction mismatch on {ds_name}")
            self.assertEqual(s2_mismatches, 0, f"Stage-2 prediction mismatch on {ds_name}")
            self.assertLess(max_s1_err, 1e-11, f"Stage-1 score drift {max_s1_err} on {ds_name}")
            self.assertLess(max_s2_err, 1e-11, f"Stage-2 score drift {max_s2_err} on {ds_name}")

        if not summaries:
            self.skipTest("Skipped real-trace parity integration test: no local datasets + .joblib pairs available.")

        print("\n=== Native C vs scikit-learn Real-Trace Equivalence Summary ===")
        for ds_name, n_scan, n_eval, e1, m1, e2, m2, min_m in summaries:
            print(
                f"  [{ds_name:<12}] scanned={n_scan:>5} evaluated={n_eval:>4} (min |margin|={min_m:.3e}) | "
                f"Stage1 max|d|={e1:.3e} mismatches={m1} | "
                f"Stage2 max|d|={e2:.3e} mismatches={m2}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
