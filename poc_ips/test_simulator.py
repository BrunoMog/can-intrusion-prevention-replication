#!/usr/bin/env python3
"""
Unit Tests for Chronological Event Queue, CAN 2.0A Error Confinement,
Cache Integrity Fingerprints, and TOST Statistical Decoupling.

Usage:
    /home/bruno/venvs/cin0114_venv/bin/python poc_ips/test_simulator.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from typing import Any, List, Tuple

import numpy as np

POC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(POC_DIR)
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import poc_simulator
import post_analysis
import run_experiments
import utils


class _StubStage1:
    """Deterministic stub returning BLOCK (1) for keys flagged as suspect."""

    def __init__(self, suspect_d0_values: set[int]) -> None:
        self.suspect_d0_values = suspect_d0_values

    def predict_one_f32(self, x5_row: np.ndarray) -> int:
        d0_int = int(round(float(x5_row[0]) * 255.0))
        return 1 if d0_int in self.suspect_d0_values else 0


class _StubStage2:
    """Deterministic stub returning configurable Stage-2 verdicts."""

    def __init__(self, allow_all: bool = True) -> None:
        self.allow_all = allow_all

    def predict_one(self, can_id_int: int, x8_row: np.ndarray) -> Tuple[int, float]:
        return (0, 0.25) if self.allow_all else (1, -0.25)


class _RecordingEngine(poc_simulator.ThreeThreadIPSEngine):
    """Spy subclass capturing monotonically ordered (frame_idx, sim_t) calls."""

    observed_calls: List[Tuple[int, float]] = []

    def process_frame(self, frame_idx: int, sim_t: float, key: int) -> int:
        type(self).observed_calls.append((frame_idx, sim_t))
        return super().process_frame(frame_idx, sim_t, key)


def _make_stream(rows: List[Tuple[float, int, int, int]]) -> poc_simulator.PreparedTestStream:
    """Build a synthetic PreparedTestStream from (timestamp, cid, d0_byte, label)."""
    n = len(rows)
    timestamps = np.array([r[0] for r in rows], dtype=np.float64)
    cids = np.array([r[1] for r in rows], dtype=np.int64)
    labels = np.array([r[3] for r in rows], dtype=np.int64)

    x8 = np.zeros((n, 8), dtype=np.float32)
    packed_keys: List[int] = []
    for i, (_, cid, d0, _) in enumerate(rows):
        x8[i, 0] = np.float32(d0 / 255.0)
        packed_keys.append((cid << 64) | d0)
    x5 = np.ascontiguousarray(x8[:, :5], dtype=np.float32)

    return poc_simulator.PreparedTestStream(
        dataset_name="SyntheticTest",
        timestamps=timestamps,
        can_ids_int=cids,
        packed_keys=packed_keys,
        x5_f32=x5,
        x8_f32=x8,
        labels=labels,
    )


class TestSimulatorAndIntegrity(unittest.TestCase):
    """Verify chronological event ordering, CAN 2.0A Passive suspend, TOST, and hashes."""

    def test_global_heap_chronological_interleaving(self) -> None:
        """Retransmissions at 260/520 us must interleave chronologically around a 300 us frame."""
        # Frame 0 (d0=10 -> blocked at S1, rescued by S2 at t >= 400 us -> 2 aborts before rescue)
        # Frame 1 arrives at 300 us (normal, d0=0) BETWEEN attempt 1 (260 us) and attempt 2 (520 us)
        stream = _make_stream(
            [
                (0.000000, 0x100, 10, 0),
                (0.000300, 0x200, 0, 0),
            ]
        )
        cfg = poc_simulator.SimulationConfig(
            name="ChronoCheck",
            use_stage2=True,
            use_caches=True,
            cache1_trigger_count=1,
            cache2_ttl_sec=0.100,
            stage2_sim_latency_sec=0.000400,
            retransmit_slot_sec=0.000260,
            passive_suspend_sec=0.000016,
            bus_off_recovery_sec=0.050,
        )
        _RecordingEngine.observed_calls = []
        orig_engine = poc_simulator.ThreeThreadIPSEngine
        poc_simulator.ThreeThreadIPSEngine = _RecordingEngine
        try:
            row = poc_simulator.run_single_simulation(
                cfg,
                stream,
                fast_s1=_StubStage1({10}),  # type: ignore[arg-type]
                fast_s2=_StubStage2(allow_all=True),  # type: ignore[arg-type]
            )
        finally:
            poc_simulator.ThreeThreadIPSEngine = orig_engine

        calls = _RecordingEngine.observed_calls
        times = [t for _, t in calls]
        self.assertEqual(times, sorted(times), "Simulation timestamps went backwards!")
        # Frame 0 @ 0.0 -> Frame 0 retry @ 260 us -> Frame 1 @ 300 us -> Frame 0 retry @ 520 us (rescued)
        self.assertEqual([f for f, _ in calls], [0, 0, 1, 0])
        self.assertAlmostEqual(times[1], 0.000260, places=9)
        self.assertAlmostEqual(times[2], 0.000300, places=9)
        self.assertAlmostEqual(times[3], 0.000520, places=9)
        self.assertEqual(row["stream_fp_rescued_retrans"], 1)
        self.assertEqual(row["bus_off_events"], 0)

    def test_stage2_wall_clock_completion_cannot_bypass_available_ts(self) -> None:
        """Even if Stage-2 OS thread completes immediately, result must stay hidden before available_ts."""
        stream = _make_stream([(0.000000, 0x100, 10, 0)])
        cfg = poc_simulator.SimulationConfig(
            name="FuturePublishCheck",
            use_stage2=True,
            use_caches=True,
            cache1_trigger_count=1,
            cache2_ttl_sec=0.100,
            stage2_sim_latency_sec=0.000400,
            retransmit_slot_sec=0.000260,
            passive_suspend_sec=0.000016,
            bus_off_recovery_sec=0.050,
        )
        engine = poc_simulator.ThreeThreadIPSEngine(
            cfg,
            stream,
            fast_s1=_StubStage1({10}),  # type: ignore[arg-type]
            fast_s2=_StubStage2(allow_all=True),  # type: ignore[arg-type]
        )
        try:
            key = stream.packed_keys[0]
            # First arrival at 0 us: Stage-1 blocks, dispatches Stage-2 with available_ts = 400 us
            self.assertEqual(engine.process_frame(0, 0.000000, key), 1)
            self.assertEqual(len(engine._in_flight_stage2), 1)

            # Force wall-clock completion of Stage-2 worker thread BEFORE sim_t advances
            done_evt = engine._in_flight_stage2[0][3]
            self.assertTrue(done_evt.wait(timeout=2.0))

            # At 260 us and 399 us, worker thread is already done on CPU, yet result MUST be invisible
            self.assertNotIn(key, engine.cache2_validated)
            self.assertEqual(engine.process_frame(0, 0.000260, key), 1)
            self.assertEqual(engine.cache2_hits, 0)
            self.assertEqual(engine.cache1_dedup_hits, 1)
            self.assertEqual(engine.cache1_suspects[key].status, "PENDING")

            self.assertEqual(engine.process_frame(0, 0.000399, key), 1)
            self.assertEqual(engine.cache2_hits, 0)
            self.assertEqual(engine.cache1_dedup_hits, 2)
            self.assertNotIn(key, engine.cache2_validated)

            # Once logical clock reaches available_ts = 400 us, Coordinator commits and hits Cache 2
            self.assertEqual(engine.process_frame(0, 0.000400, key), 0)
            self.assertIn(key, engine.cache2_validated)
            self.assertEqual(engine.cache1_suspects[key].status, "VALIDATED")
            self.assertEqual(engine.cache2_hits, 1)
            self.assertEqual(engine.stage2_dispatches, 1)
        finally:
            engine.shutdown()

    def test_error_passive_suspend_transmission_and_bus_off_flush(self) -> None:
        """Verify 16 us Suspend Transmission after TEC > 127 and epoch flush on Bus-Off."""
        # Single blocked frame on baseline 5F runs 32 aborts to Bus-Off:
        # Aborts 1..15 (TEC 8..120 -> Error Active) -> 260 us steps
        # Aborts 16..31 (TEC 128..248 -> Error Passive) -> 260 us + 16 us = 276 us steps
        # Abort 32 (TEC 256 -> Bus-Off)
        stream = _make_stream([(0.0, 0x316, 99, 0)])
        cfg = poc_simulator.SimulationConfig(
            name="PassiveCheck",
            use_stage2=False,
            use_caches=False,
            retransmit_slot_sec=0.000260,
            passive_suspend_sec=0.000016,
            bus_off_recovery_sec=0.050,
        )
        _RecordingEngine.observed_calls = []
        orig_engine = poc_simulator.ThreeThreadIPSEngine
        poc_simulator.ThreeThreadIPSEngine = _RecordingEngine
        try:
            row = poc_simulator.run_single_simulation(
                cfg,
                stream,
                fast_s1=_StubStage1({99}),  # type: ignore[arg-type]
                fast_s2=None,
            )
        finally:
            poc_simulator.ThreeThreadIPSEngine = orig_engine

        calls = _RecordingEngine.observed_calls
        self.assertEqual(len(calls), 32)
        deltas = [calls[i][1] - calls[i - 1][1] for i in range(1, len(calls))]
        # First 15 deltas occur while TEC <= 120 (0..14 index): 260 us
        for d in deltas[:15]:
            self.assertAlmostEqual(d, 0.000260, places=9)
        # Delta index 15 happens right after abort 16 puts TEC == 128 (> 127 -> in_passive=True): 276 us
        for d in deltas[15:]:
            self.assertAlmostEqual(d, 0.000276, places=9)
        self.assertEqual(row["bus_off_events"], 1)
        self.assertEqual(row["error_passive_events"], 1)
        self.assertEqual(row["stream_fp_unrescued_lost"], 1)

        # Interleaved second frame of SAME CAN_ID hits Bus-Off at attempt 15, flushing frame 0 TX
        rows_burst = [
            (0.000000, 0x316, 99, 0),
            (0.000100, 0x316, 99, 0),
        ]
        row2 = poc_simulator.run_single_simulation(
            cfg,
            _make_stream(rows_burst),
            fast_s1=_StubStage1({99}),  # type: ignore[arg-type]
            fast_s2=None,
        )
        self.assertEqual(row2["bus_off_events"], 1)
        self.assertEqual(row2["stream_fp_initial"], 2)
        self.assertEqual(row2["stream_fp_unrescued_lost"], 2)

    def test_tost_runs_independently_of_anova_non_significance(self) -> None:
        """Paired TOST must still run even when global One-Way ANOVA yields p >= 0.05."""
        folds_a = [99.00, 99.10, 98.95, 99.05, 99.02, 98.98, 99.04, 99.01, 98.97, 99.08]
        folds_b = [99.01, 99.09, 98.96, 99.04, 99.01, 98.99, 99.05, 99.00, 98.98, 99.07]
        best_models = {
            "DummyDS": {
                "iForest_5F": {
                    "model_name": "iForest_5F",
                    "model_family": "iForest",
                    "n_bytes": 5,
                    "fold_accuracies": folds_a,
                },
                "iForest_8F": {
                    "model_name": "iForest_8F",
                    "model_family": "iForest",
                    "n_bytes": 8,
                    "fold_accuracies": folds_b,
                },
            }
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            post_analysis.run_statistical_analysis(best_models)
        out = buf.getvalue()
        self.assertIn("Não há diferença estatisticamente significativa", out)
        self.assertIn("Teste Pareado de Equivalência TOST", out)
        self.assertIn("Evidência exploratória de equivalência dentro de +-1,0%", out)

    def test_full_sha256_and_strict_cache_checkpoint_invalidation(self) -> None:
        """Full SHA-256 must detect mutations > 64 KiB and strict defaults must reject mismatches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "Speedometer_dataset.csv")
            cache_dir = os.path.join(tmpdir, "splits")
            rows: List[str] = []
            # Write > 100 KiB (~3,000 rows) so we can mutate well past 64 KiB
            for i in range(3200):
                flag = "T" if i % 4 == 0 else "R"
                rows.append(f"{1.0 + i * 0.001:.6f},05a0,8,01,80,7e,00,00,00,00,00,{flag}\n")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.writelines(rows)

            orig_cfg = utils.DATASET_SPLIT_CONFIG["Speedometer"].copy()
            utils.DATASET_SPLIT_CONFIG["Speedometer"] = {
                "n_total_normal_sampled": 1000,
                "n_train_normal": 900,
                "n_val_normal": 100,
                "n_val_attack": 200,
                "partition_val_attacks_across_folds": True,
            }
            try:
                sp1 = utils.get_split_indices(csv_path, random_state=42, cache_dir=cache_dir)
                npz_path = os.path.join(cache_dir, "Speedometer_dataset_splits_seed42.npz")
                meta_path = f"{npz_path}.meta.json"
                self.assertTrue(os.path.exists(meta_path))
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                self.assertIn("file_sha256", meta)

                # Mutate 1 byte past 90 KiB keeping exact byte length identical
                with open(csv_path, "r+b") as bf:
                    bf.seek(92000)
                    b = bf.read(1)
                    bf.seek(92000)
                    bf.write(b"1" if b != b"1" else b"2")

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    sp2 = utils.get_split_indices(csv_path, random_state=42, cache_dir=cache_dir)
                self.assertIn("diverge em ['file_sha256']", buf.getvalue())
                self.assertEqual(len(sp2["test_indices"]), len(sp1["test_indices"]))

                # Removing .meta.json under strict default must force recompute
                os.remove(meta_path)
                buf2 = io.StringIO()
                with contextlib.redirect_stdout(buf2):
                    utils.get_split_indices(csv_path, random_state=42, cache_dir=cache_dir)
                self.assertIn("sem metadados '.meta.json'", buf2.getvalue())
                self.assertTrue(os.path.exists(meta_path))

                # CheckpointManager strict defaults
                cv_csv = os.path.join(tmpdir, "cv.csv")
                te_csv = os.path.join(tmpdir, "test.csv")
                fp = utils.compute_dataset_fingerprint(csv_path, random_state=42)
                exp_meta: dict[str, Any] = {
                    "cv_folds": 10,
                    "random_state": 42,
                    "grid_preset": "smoke",
                    "primary_metric": "accuracy",
                    "dataset_fingerprints": {"Speedometer": fp},
                }
                cm = run_experiments.CheckpointManager(cv_csv, te_csv, resume=False, expected_metadata=exp_meta)
                self.assertTrue(os.path.exists(cm.meta_path))

                # Mismatched cv_folds must raise RuntimeError under default strict_checkpoint=True
                bad_meta = dict(exp_meta, cv_folds=5)
                with self.assertRaises(RuntimeError):
                    run_experiments.CheckpointManager(cv_csv, te_csv, resume=True, expected_metadata=bad_meta)

                # Missing dataset in dataset_fingerprints must raise RuntimeError under strict_checkpoint=True
                with open(cm.meta_path, "r", encoding="utf-8") as mf:
                    saved_meta = json.load(mf)
                saved_meta["dataset_fingerprints"] = {}
                with open(cm.meta_path, "w", encoding="utf-8") as mf:
                    json.dump(saved_meta, mf)
                with self.assertRaises(RuntimeError):
                    run_experiments.CheckpointManager(cv_csv, te_csv, resume=True, expected_metadata=exp_meta)

                # Legacy CSV without checkpoint_meta.json must fail unless allow_legacy_checkpoint=True
                os.remove(cm.meta_path)
                with self.assertRaises(RuntimeError):
                    run_experiments.CheckpointManager(cv_csv, te_csv, resume=True, expected_metadata=exp_meta)
                run_experiments.CheckpointManager(
                    cv_csv, te_csv, resume=True, expected_metadata=exp_meta, allow_legacy_checkpoint=True
                )
            finally:
                utils.DATASET_SPLIT_CONFIG["Speedometer"] = orig_cfg


if __name__ == "__main__":
    unittest.main(verbosity=2)
