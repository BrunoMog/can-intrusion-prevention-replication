"""Etapas 2, 3 e 4: 3-Thread Shared-Memory Dual-Cache PoC & CAN Error-Confinement Simulator.

Executes genuine 3-thread shared-memory inference:
  - Coordinator Thread (caller): checks Cache 2 (with CAN-clock TTL), runs
    Stage-1 over Worker_Stage1 if Cache 2 misses, updates Cache 1 suspect history,
    and dispatches Stage-2 asynchronously to Worker_Stage2.
  - Worker_Stage1 Thread: pre-loaded C-accelerated `iForest_5F` evaluator.
  - Worker_Stage2 Thread: pre-loaded asynchronous C-accelerated `CAN_ID + 8B` evaluator.

Couples the 3-thread IPS to a chronological CAN 2.0A Error-Confinement simulator:
  - Legitimate frames blocked by IPS trigger Active Error Flags (`TEC += 8`) and
    automatic ECU retransmission (`retransmit_slot_sec = 260 us` at 500 kbps).
  - Successful legitimate transmissions decrement `TEC -= 1` (if `TEC > 0`).
  - Tracks `Error Warning (>= 96)`, `Error Passive (> 127)`, `Bus-Off (> 255)` events,
    50 ms AUTOSAR-style Bus-Off recovery windows, dropped telemetry during outages,
    attack Recall (TP/FN), and Cache 1/Cache 2 hit and memory statistics.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(POC_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

import config
import utils
from fast_iforest import FastIsolationForest, FastStage2Evaluator
from models_stage2 import get_model_artifact_paths, parse_can_ids_to_int

POC_MODELS_DIR = os.path.join(POC_DIR, "models")
POC_RESULTS_DIR = os.path.join(POC_DIR, "results")

_STOP_SENTINEL = object()


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration defining an IPS baseline, proposed design, or ablation."""

    name: str
    use_stage2: bool
    use_caches: bool
    cache1_trigger_count: int = 1
    cache2_ttl_sec: Optional[float] = 0.100  # 100 ms short validation window
    stage2_sim_latency_sec: float = 0.000200  # 200 us post-8B async completion budget
    retransmit_slot_sec: float = 0.000260  # 260 us CAN 2.0A slot at 500 kbps
    bus_off_recovery_sec: float = 0.050  # 50 ms ECU recovery cooldown after Bus-Off


DEFAULT_SIM_CONFIGS: Tuple[SimulationConfig, ...] = (
    SimulationConfig(
        name="Baseline_5F",
        use_stage2=False,
        use_caches=False,
    ),
    SimulationConfig(
        name="Proposed_DualCache_100ms",
        use_stage2=True,
        use_caches=True,
        cache1_trigger_count=1,
        cache2_ttl_sec=0.100,
    ),
    SimulationConfig(
        name="Ablation_Trigger2_100ms",
        use_stage2=True,
        use_caches=True,
        cache1_trigger_count=2,
        cache2_ttl_sec=0.100,
    ),
    SimulationConfig(
        name="Ablation_DualCache_NoTTL",
        use_stage2=True,
        use_caches=True,
        cache1_trigger_count=1,
        cache2_ttl_sec=None,
    ),
)


@dataclass
class Cache1Entry:
    """History entry in Cache 1 for frames flagged by Stage 1 (`iForest_5F`)."""

    alert_count: int
    status: str  # 'OBSERVING', 'PENDING', 'VALIDATED'
    last_ts: float


@dataclass
class Cache2Entry:
    """Validated Stage-2 verdict entry in Cache 2 keyed on full (CAN_ID, D0..D7)."""

    verdict: int  # 0 = ALLOW, 1 = BLOCK
    available_ts: float
    expires_ts: float
    score2: float


@dataclass
class EcuErrorState:
    """CAN 2.0A Transmit Error Counter (TEC) state for one legitimate ECU (CAN_ID)."""

    tec: int = 0
    max_tec: int = 0
    in_warning: bool = False
    in_passive: bool = False
    bus_off_until: float = -1.0
    warning_events: int = 0
    passive_events: int = 0
    bus_off_events: int = 0
    dropped_while_bus_off: int = 0
    outage_sec: float = 0.0


@dataclass
class PreparedTestStream:
    """Compact memory-mapped numeric representation of a chronological test trace."""

    dataset_name: str
    timestamps: np.ndarray  # float64 (N,)
    can_ids_int: np.ndarray  # int64 (N,)
    packed_keys: List[int]  # 75-bit integer key = (can_id_int << 64) | u64_payload
    x5_f32: np.ndarray  # contiguous float32 (N, 5)
    x8_f32: np.ndarray  # contiguous float32 (N, 8)
    labels: np.ndarray  # int64 (N,) 0=normal, 1=attack


def load_prepared_test_stream(
    dataset_name: str, random_state: int = config.DEFAULT_SEED
) -> PreparedTestStream:
    """Load test split in strictly chronological timestamp order into compact arrays."""
    df = utils.load_test_data(dataset_name, random_state=random_state, add_normalized_cols=True)
    if not df["Timestamp"].is_monotonic_increasing:
        df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)

    timestamps = df["Timestamp"].to_numpy(dtype=np.float64)
    can_ids_int = parse_can_ids_to_int(df["CAN_ID"])
    labels = df["Label"].to_numpy(dtype=np.int64)

    raw_bytes = np.ascontiguousarray(df[utils.BYTE_COLS].to_numpy(dtype=np.uint8))
    u64_payload = raw_bytes.view(np.uint64).ravel()
    packed_keys = [(int(cid) << 64) | int(p) for cid, p in zip(can_ids_int, u64_payload)]

    x8_f32 = np.ascontiguousarray(df[utils.NORM_BYTE_COLS].to_numpy(dtype=np.float32))
    x5_f32 = np.ascontiguousarray(x8_f32[:, :5])

    return PreparedTestStream(
        dataset_name=dataset_name,
        timestamps=timestamps,
        can_ids_int=can_ids_int,
        packed_keys=packed_keys,
        x5_f32=x5_f32,
        x8_f32=x8_f32,
        labels=labels,
    )


class ThreeThreadIPSEngine:
    """Shared-memory 3-thread IPS runtime (Coordinator + Stage-1 + Stage-2 Workers)."""

    def __init__(
        self,
        sim_cfg: SimulationConfig,
        stream: PreparedTestStream,
        fast_s1: FastIsolationForest,
        fast_s2: Optional[FastStage2Evaluator],
    ) -> None:
        self.cfg = sim_cfg
        self.stream = stream
        self.fast_s1 = fast_s1
        self.fast_s2 = fast_s2

        self._q_s1_in: queue.SimpleQueue = queue.SimpleQueue()
        self._q_s1_out: queue.SimpleQueue = queue.SimpleQueue()
        self._q_s2_in: queue.SimpleQueue = queue.SimpleQueue()

        self._cache_lock = threading.Lock()
        self.cache1_suspects: Dict[int, Cache1Entry] = {}
        self.cache2_validated: Dict[int, Cache2Entry] = {}
        self._in_flight_stage2: Deque[Tuple[int, float, threading.Event]] = deque()

        # Telemetry counters
        self.stage1_calls = 0
        self.stage1_compute_sec = 0.0
        self.stage2_dispatches = 0
        self.stage2_calls = 0
        self.stage2_compute_sec = 0.0
        self.cache1_dedup_hits = 0
        self.cache2_hits = 0
        self.cache2_hit_allow = 0
        self.cache2_hit_block = 0
        self.cache2_expirations = 0
        self.peak_cache1_entries = 0
        self.peak_cache2_entries = 0

        self._t_s1 = threading.Thread(target=self._stage1_worker_loop, daemon=True)
        self._t_s2: Optional[threading.Thread] = (
            threading.Thread(target=self._stage2_worker_loop, daemon=True)
            if sim_cfg.use_stage2
            else None
        )
        self._t_s1.start()
        if self._t_s2 is not None:
            self._t_s2.start()

    def _stage1_worker_loop(self) -> None:
        """Pre-loaded Stage-1 worker handling 5-byte real-time requests."""
        x5_f32 = self.stream.x5_f32
        predict_s1 = self.fast_s1.predict_one_f32
        q_in = self._q_s1_in
        q_out = self._q_s1_out
        local_calls = 0
        local_sec = 0.0
        perf = time.perf_counter

        while True:
            frame_idx = q_in.get()
            if frame_idx is _STOP_SENTINEL:
                break
            t0 = perf()
            pred1 = predict_s1(x5_f32[frame_idx])
            local_sec += perf() - t0
            local_calls += 1
            q_out.put(pred1)

        self.stage1_calls = local_calls
        self.stage1_compute_sec = local_sec

    def _stage2_worker_loop(self) -> None:
        """Asynchronous Stage-2 worker analyzing full (CAN_ID + 8B) frames."""
        assert self.fast_s2 is not None
        cids_int = self.stream.can_ids_int
        x8_f32 = self.stream.x8_f32
        predict_s2 = self.fast_s2.predict_one
        q_in = self._q_s2_in
        lock = self._cache_lock
        c1 = self.cache1_suspects
        c2 = self.cache2_validated
        local_calls = 0
        local_sec = 0.0
        perf = time.perf_counter

        while True:
            item = q_in.get()
            if item is _STOP_SENTINEL:
                break
            key, frame_idx, available_ts, ttl_sec, done_evt = item
            t0 = perf()
            pred2, score2 = predict_s2(int(cids_int[frame_idx]), x8_f32[frame_idx])
            local_sec += perf() - t0
            local_calls += 1

            expires_ts = available_ts + ttl_sec if ttl_sec is not None else math.inf
            entry2 = Cache2Entry(
                verdict=pred2,
                available_ts=available_ts,
                expires_ts=expires_ts,
                score2=score2,
            )
            with lock:
                c2[key] = entry2
                c1_entry = c1.get(key)
                if c1_entry is not None:
                    c1_entry.status = "VALIDATED"
            done_evt.set()

        self.stage2_calls = local_calls
        self.stage2_compute_sec = local_sec

    def _drain_completed_stage2(self, sim_t: float) -> None:
        """Ensure Stage-2 async writes whose CAN completion time <= sim_t are visible."""
        inflight = self._in_flight_stage2
        while inflight and inflight[0][1] <= sim_t:
            _, _, evt = inflight.popleft()
            evt.wait()

    def _flush_all_stage2(self) -> None:
        """Drain remaining asynchronous Stage-2 tasks cleanly before shutdown."""
        inflight = self._in_flight_stage2
        while inflight:
            _, _, evt = inflight.popleft()
            evt.wait()

    def sweep_expired(self, sim_t: float) -> None:
        """Periodically prune expired TTL entries to mirror embedded RAM footprint."""
        if not self.cfg.use_caches or self.cfg.cache2_ttl_sec is None:
            return
        with self._cache_lock:
            self.peak_cache1_entries = max(self.peak_cache1_entries, len(self.cache1_suspects))
            self.peak_cache2_entries = max(self.peak_cache2_entries, len(self.cache2_validated))
            expired_keys = [k for k, v in self.cache2_validated.items() if sim_t > v.expires_ts]
            for k in expired_keys:
                del self.cache2_validated[k]
                self.cache1_suspects.pop(k, None)
            self.cache2_expirations += len(expired_keys)

    def process_frame(self, frame_idx: int, sim_t: float, key: int) -> int:
        """Coordinator path: returns 0 (ALLOW) or 1 (BLOCK / Active Error Flag)."""
        cfg = self.cfg
        if cfg.use_caches:
            if self._in_flight_stage2:
                self._drain_completed_stage2(sim_t)

            with self._cache_lock:
                entry2 = self.cache2_validated.get(key)
                if entry2 is not None:
                    if sim_t <= entry2.expires_ts:
                        self.cache2_hits += 1
                        if entry2.verdict == 0:
                            self.cache2_hit_allow += 1
                        else:
                            self.cache2_hit_block += 1
                        return entry2.verdict
                    del self.cache2_validated[key]
                    self.cache1_suspects.pop(key, None)
                    self.cache2_expirations += 1

        # Cache 2 miss (or baseline mode) -> synchronous Stage-1 (iForest_5F)
        self._q_s1_in.put(frame_idx)
        pred1 = self._q_s1_out.get()
        if pred1 == 0 or not cfg.use_stage2:
            return pred1

        # Stage 1 flagged anomaly: update Cache 1 and asynchronously schedule Stage 2
        with self._cache_lock:
            c1_entry = self.cache1_suspects.get(key)
            if c1_entry is None:
                c1_entry = Cache1Entry(alert_count=1, status="OBSERVING", last_ts=sim_t)
                self.cache1_suspects[key] = c1_entry
            else:
                c1_entry.alert_count += 1
                c1_entry.last_ts = sim_t

            if c1_entry.status == "PENDING":
                self.cache1_dedup_hits += 1
                should_dispatch = False
            elif c1_entry.alert_count >= cfg.cache1_trigger_count:
                c1_entry.status = "PENDING"
                should_dispatch = True
            else:
                should_dispatch = False

            if len(self.cache1_suspects) > self.peak_cache1_entries:
                self.peak_cache1_entries = len(self.cache1_suspects)
            if len(self.cache2_validated) > self.peak_cache2_entries:
                self.peak_cache2_entries = len(self.cache2_validated)

        if should_dispatch:
            ready_ts = sim_t + cfg.stage2_sim_latency_sec
            done_evt = threading.Event()
            self._in_flight_stage2.append((key, ready_ts, done_evt))
            self._q_s2_in.put((key, frame_idx, ready_ts, cfg.cache2_ttl_sec, done_evt))
            self.stage2_dispatches += 1

        return 1

    def shutdown(self) -> None:
        """Stop both worker threads gracefully and update final peak cache sizes."""
        self._flush_all_stage2()
        self._q_s1_in.put(_STOP_SENTINEL)
        self._t_s1.join()
        if self._t_s2 is not None:
            self._q_s2_in.put(_STOP_SENTINEL)
            self._t_s2.join()
        with self._cache_lock:
            self.peak_cache1_entries = max(self.peak_cache1_entries, len(self.cache1_suspects))
            self.peak_cache2_entries = max(self.peak_cache2_entries, len(self.cache2_validated))


def _record_ecu_abort_error(
    ecu: EcuErrorState, sim_t: float, bus_off_recovery_sec: float
) -> bool:
    """Apply Bosch CAN TEC += 8 rule; returns True if ECU just transitioned to Bus-Off."""
    ecu.tec += 8
    if ecu.tec > ecu.max_tec:
        ecu.max_tec = ecu.tec

    if not ecu.in_warning and ecu.tec >= 96:
        ecu.in_warning = True
        ecu.warning_events += 1
    if not ecu.in_passive and ecu.tec > 127:
        ecu.in_passive = True
        ecu.passive_events += 1

    if ecu.tec > 255:
        ecu.bus_off_events += 1
        ecu.bus_off_until = sim_t + bus_off_recovery_sec
        ecu.outage_sec += bus_off_recovery_sec
        ecu.tec = 0
        ecu.in_warning = False
        ecu.in_passive = False
        return True
    return False


def _record_ecu_success(ecu: EcuErrorState) -> None:
    """Apply Bosch CAN TEC -= 1 rule upon error-free frame delivery."""
    if ecu.tec > 0:
        ecu.tec -= 1
        if ecu.tec <= 127:
            ecu.in_passive = False
        if ecu.tec < 96:
            ecu.in_warning = False


def run_single_simulation(
    sim_cfg: SimulationConfig,
    stream: PreparedTestStream,
    fast_s1: FastIsolationForest,
    fast_s2: Optional[FastStage2Evaluator],
) -> Dict[str, Any]:
    """Execute one full chronological trace replay with CAN Error Confinement."""
    engine = ThreeThreadIPSEngine(sim_cfg, stream, fast_s1, fast_s2)

    ecus: Dict[int, EcuErrorState] = {}
    timestamps = stream.timestamps
    cids_int = stream.can_ids_int
    keys = stream.packed_keys
    labels = stream.labels
    n_frames = len(labels)

    stream_normal_total = 0
    stream_attack_total = 0
    stream_tn_first_try = 0
    stream_fp_initial = 0
    stream_fp_rescued_retrans = 0
    stream_fp_unrescued_lost = 0
    total_legit_aborts = 0
    attack_tp = 0
    attack_fn = 0

    retransmit_slot = sim_cfg.retransmit_slot_sec
    recovery_sec = sim_cfg.bus_off_recovery_sec

    t_sim_start = time.perf_counter()
    for i in range(n_frames):
        if i % 50000 == 0 and i > 0:
            engine.sweep_expired(float(timestamps[i]))

        t_orig = float(timestamps[i])
        cid = int(cids_int[i])
        key = keys[i]
        y_true = int(labels[i])

        if y_true == 1:
            stream_attack_total += 1
            verdict = engine.process_frame(i, t_orig, key)
            if verdict == 1:
                attack_tp += 1
            else:
                attack_fn += 1
            continue

        # Legitimate ECU frame (Label == 0)
        stream_normal_total += 1
        ecu = ecus.get(cid)
        if ecu is None:
            ecu = EcuErrorState()
            ecus[cid] = ecu

        if t_orig < ecu.bus_off_until:
            ecu.dropped_while_bus_off += 1
            continue

        attempt = 0
        while True:
            sim_t = t_orig + attempt * retransmit_slot
            verdict = engine.process_frame(i, sim_t, key)
            if verdict == 0:
                if attempt == 0:
                    stream_tn_first_try += 1
                else:
                    stream_fp_rescued_retrans += 1
                _record_ecu_success(ecu)
                break

            if attempt == 0:
                stream_fp_initial += 1
            total_legit_aborts += 1
            hit_bus_off = _record_ecu_abort_error(ecu, sim_t, recovery_sec)
            if hit_bus_off:
                stream_fp_unrescued_lost += 1
                break
            attempt += 1

    wall_sec = time.perf_counter() - t_sim_start
    engine.shutdown()

    max_tec = max((e.max_tec for e in ecus.values()), default=0)
    warning_events = sum(e.warning_events for e in ecus.values())
    passive_events = sum(e.passive_events for e in ecus.values())
    bus_off_events = sum(e.bus_off_events for e in ecus.values())
    unique_bus_off_ecus = sum(1 for e in ecus.values() if e.bus_off_events > 0)
    dropped_during_bus_off = sum(e.dropped_while_bus_off for e in ecus.values())
    total_outage_ms = sum(e.outage_sec for e in ecus.values()) * 1000.0

    normal_lost_total = stream_fp_unrescued_lost + dropped_during_bus_off
    attack_recall_pct = (
        100.0 * attack_tp / stream_attack_total if stream_attack_total > 0 else 0.0
    )
    eff_precision_pct = (
        100.0 * attack_tp / (attack_tp + stream_fp_initial)
        if (attack_tp + stream_fp_initial) > 0
        else 0.0
    )
    eff_fp_rate_pct = (
        100.0 * stream_fp_initial / stream_normal_total if stream_normal_total > 0 else 0.0
    )
    total_ips_queries = engine.cache2_hits + engine.stage1_calls
    cache2_hit_rate_total_pct = (
        100.0 * engine.cache2_hits / total_ips_queries if total_ips_queries > 0 else 0.0
    )

    s1_avg_us = (
        (engine.stage1_compute_sec / engine.stage1_calls) * 1e6
        if engine.stage1_calls > 0
        else 0.0
    )
    s2_avg_us = (
        (engine.stage2_compute_sec / engine.stage2_calls) * 1e6
        if engine.stage2_calls > 0
        else 0.0
    )

    return {
        "dataset": stream.dataset_name,
        "config_name": sim_cfg.name,
        "use_stage2": sim_cfg.use_stage2,
        "use_caches": sim_cfg.use_caches,
        "cache1_trigger_count": sim_cfg.cache1_trigger_count,
        "cache2_ttl_ms": (
            sim_cfg.cache2_ttl_sec * 1000.0 if sim_cfg.cache2_ttl_sec is not None else -1.0
        ),
        "n_trace_frames": n_frames,
        "n_normal_frames": stream_normal_total,
        "n_attack_frames": stream_attack_total,
        "stream_tn_first_try": stream_tn_first_try,
        "stream_fp_initial": stream_fp_initial,
        "stream_fp_rescued_retrans": stream_fp_rescued_retrans,
        "stream_fp_unrescued_lost": stream_fp_unrescued_lost,
        "dropped_during_bus_off": dropped_during_bus_off,
        "normal_lost_total": normal_lost_total,
        "eff_fp_rate_pct": eff_fp_rate_pct,
        "total_legit_aborts": total_legit_aborts,
        "attack_tp": attack_tp,
        "attack_fn": attack_fn,
        "attack_recall_pct": attack_recall_pct,
        "eff_precision_pct": eff_precision_pct,
        "max_tec_global": max_tec,
        "error_warning_events": warning_events,
        "error_passive_events": passive_events,
        "bus_off_events": bus_off_events,
        "unique_bus_off_ecus": unique_bus_off_ecus,
        "total_outage_ms": total_outage_ms,
        "total_ips_queries": total_ips_queries,
        "stage1_calls": engine.stage1_calls,
        "stage1_avg_us": s1_avg_us,
        "stage2_dispatches": engine.stage2_dispatches,
        "stage2_calls": engine.stage2_calls,
        "stage2_avg_us": s2_avg_us,
        "cache1_dedup_hits": engine.cache1_dedup_hits,
        "cache2_hits": engine.cache2_hits,
        "cache2_hit_allow": engine.cache2_hit_allow,
        "cache2_hit_block": engine.cache2_hit_block,
        "cache2_hit_rate_total_pct": cache2_hit_rate_total_pct,
        "cache2_expirations": engine.cache2_expirations,
        "peak_cache1_entries": engine.peak_cache1_entries,
        "peak_cache2_entries": engine.peak_cache2_entries,
        "wall_time_sec": wall_sec,
    }


def add_relative_baseline_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute FP reduction, abort reduction, and Bus-Off prevention vs Baseline_5F."""
    df = df.copy()
    baselines = df[df["config_name"] == "Baseline_5F"].set_index("dataset")

    fp_red: List[float] = []
    abort_red: List[float] = []
    busoff_red: List[float] = []
    loss_red: List[float] = []

    for _, row in df.iterrows():
        ds = row["dataset"]
        if ds not in baselines.index:
            fp_red.append(np.nan)
            abort_red.append(np.nan)
            busoff_red.append(np.nan)
            loss_red.append(np.nan)
            continue
        base = baselines.loc[ds]
        fp_red.append(
            100.0 * (base["stream_fp_initial"] - row["stream_fp_initial"]) / base["stream_fp_initial"]
            if base["stream_fp_initial"] > 0
            else 0.0
        )
        abort_red.append(
            100.0 * (base["total_legit_aborts"] - row["total_legit_aborts"]) / base["total_legit_aborts"]
            if base["total_legit_aborts"] > 0
            else 0.0
        )
        busoff_red.append(
            100.0 * (base["bus_off_events"] - row["bus_off_events"]) / base["bus_off_events"]
            if base["bus_off_events"] > 0
            else 0.0
        )
        loss_red.append(
            100.0 * (base["normal_lost_total"] - row["normal_lost_total"]) / base["normal_lost_total"]
            if base["normal_lost_total"] > 0
            else 0.0
        )

    df["fp_initial_reduction_pct"] = fp_red
    df["legit_abort_reduction_pct"] = abort_red
    df["bus_off_prevented_pct"] = busoff_red
    df["normal_loss_reduction_pct"] = loss_red
    return df


def _plot_comparison_bars(
    ax: plt.Axes,
    x: np.ndarray,
    base_vals: np.ndarray,
    prop_vals: np.ndarray,
    datasets: List[str],
    title: str,
    ylabel: str,
    c_base: str,
    c_prop: str,
    annotate_pct_diff: bool = True,
    show_legend: bool = False,
) -> None:
    """Plot grouped Baseline vs Proposed bars with headroom and reduction callouts."""
    width = 0.36
    bars_b = ax.bar(x - width / 2, base_vals, width, label="Baseline (5F)", color=c_base)
    bars_p = ax.bar(
        x + width / 2,
        prop_vals,
        width,
        label="Proposed (2-Stage + Dual-Cache, TTL=100ms)",
        color=c_prop,
    )
    ax.set_title(title, fontweight="bold", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xticks(x, datasets, rotation=10)
    ax.grid(axis="y", alpha=0.3)

    ymax = float(max(np.max(base_vals), np.max(prop_vals), 1.0))
    ax.set_ylim(0, ymax * 1.20)

    if annotate_pct_diff:
        for b_val, bar_p in zip(base_vals, bars_p):
            p_val = bar_p.get_height()
            if b_val > 0:
                pct = 100.0 * (p_val - b_val) / b_val
                ax.text(
                    bar_p.get_x() + bar_p.get_width() / 2.0,
                    p_val + ymax * 0.02,
                    f"{pct:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#154360",
                )
    if show_legend:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.92)


def plot_poc_comparison(df: pd.DataFrame, out_png: str) -> None:
    """Generate publication-quality 2x2 comparative figure (Baseline 5F vs Proposed)."""
    sub = df[df["config_name"].isin(["Baseline_5F", "Proposed_DualCache_100ms"])].copy()
    datasets = [
        d for d in ["Fuzzing", "Drive Gear", "RPM Gauge", "Speedometer"] if d in sub["dataset"].unique()
    ]
    if not datasets:
        return

    base = sub[sub["config_name"] == "Baseline_5F"].set_index("dataset").loc[datasets]
    prop = sub[sub["config_name"] == "Proposed_DualCache_100ms"].set_index("dataset").loc[datasets]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    x = np.arange(len(datasets))
    c_base = "#C0392B"
    c_prop = "#1F618D"

    _plot_comparison_bars(
        axes[0, 0],
        x,
        base["total_legit_aborts"].to_numpy(),
        prop["total_legit_aborts"].to_numpy(),
        datasets,
        title="A. Legitimate Bus Aborts (Active Error Flags, TEC += 8)",
        ylabel="Aborted Bus Slots (count)",
        c_base=c_base,
        c_prop=c_prop,
        show_legend=True,
    )
    _plot_comparison_bars(
        axes[0, 1],
        x,
        base["bus_off_events"].to_numpy(),
        prop["bus_off_events"].to_numpy(),
        datasets,
        title="B. ECU Bus-Off Crashes (TEC > 255)",
        ylabel="Bus-Off Events (count)",
        c_base=c_base,
        c_prop=c_prop,
    )
    _plot_comparison_bars(
        axes[1, 0],
        x,
        base["normal_lost_total"].to_numpy(),
        prop["normal_lost_total"].to_numpy(),
        datasets,
        title="C. Legitimate Telemetry Frames Lost (Unrescued + Outage)",
        ylabel="Lost Normal Frames (count)",
        c_base=c_base,
        c_prop=c_prop,
    )

    # Panel D: Attack Recall Preserved (%) cleanly zoomed with value labels
    ax = axes[1, 1]
    width = 0.36
    b_rec = base["attack_recall_pct"].to_numpy()
    p_rec = prop["attack_recall_pct"].to_numpy()
    bars_b = ax.bar(x - width / 2, b_rec, width, label="Baseline (5F) Recall", color=c_base)
    bars_p = ax.bar(x + width / 2, p_rec, width, label="Proposed Recall", color=c_prop)
    ax.set_title("D. Attack Detection Recall Preserved (%)", fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Attack Recall (%)", fontsize=9.5)
    ax.set_ylim(95.0, 101.2)
    ax.set_xticks(x, datasets, rotation=10)
    ax.grid(axis="y", alpha=0.3)
    for bar in list(bars_b) + list(bars_p):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.12,
            f"{h:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)

    fig.suptitle(
        "PoC 2-Stage + Dual-Cache CAN IPS: Baseline (5F) vs Proposed Architecture",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png)
    plt.close(fig)


def plot_stage2_cascade_summary(stage2_csv: str, out_png: str) -> None:
    """Plot Stage-1 vs CV-selected Stage-2 Cascade FPs and F1 on blind Test."""
    if not os.path.exists(stage2_csv):
        return
    df = pd.read_csv(stage2_csv)
    datasets = [d for d in ["Fuzzing", "Drive Gear", "RPM Gauge", "Speedometer"] if d in df["dataset"].unique()]
    if not datasets:
        return

    s1 = df[df["role"] == "Stage1_Baseline_5F"].set_index("dataset").loc[datasets]
    s2 = df[df["is_cv_selected_stage2"] == True].set_index("dataset").loc[datasets]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    x = np.arange(len(datasets))
    width = 0.36
    c_s1 = "#C0392B"
    c_s2 = "#1E8449"

    _plot_comparison_bars(
        axes[0],
        x,
        s1["test_fp"].to_numpy(),
        s2["cascade_test_fp"].to_numpy(),
        datasets,
        title="A. False Positives: Stage 1 (5F) vs Cascade (5F + CAN_ID+8B)",
        ylabel="False Positives on Blind Test (count)",
        c_base=c_s1,
        c_prop=c_s2,
        show_legend=False,
    )
    axes[0].legend(
        ["Stage 1 (5F Baseline)", "Cascade (5F + Stage-2 CAN_ID+8B)"],
        fontsize=8,
        loc="upper right",
        framealpha=0.92,
    )

    ax1 = axes[1]
    b1 = ax1.bar(x - width / 2, s1["test_f1"], width, label="Stage 1 (5F Baseline)", color=c_s1)
    b2 = ax1.bar(x + width / 2, s2["cascade_test_f1"], width, label="Cascade (5F + Stage-2 CAN_ID+8B)", color=c_s2)
    ax1.set_title("B. F1-Score (%) Gain Over Stage 1 Baseline", fontweight="bold", fontsize=10.5)
    ax1.set_ylabel("F1-Score (%)", fontsize=9.5)
    ax1.set_ylim(45.0, 108.0)
    ax1.set_xticks(x, datasets, rotation=10)
    ax1.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.8,
            f"{h:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.legend(fontsize=8, loc="lower left", framealpha=0.92)

    fig.suptitle(
        "Stage-2 Model Validation (10-Fold CV Selected CAN_ID+8B): Blind Test Impact",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png)
    plt.close(fig)


def merge_simulation_csv(csv_path: str, new_df: pd.DataFrame, datasets_updated: List[str]) -> pd.DataFrame:
    """Update results CSV for newly simulated datasets while preserving prior rows."""
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[~existing["dataset"].isin(datasets_updated)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = add_relative_baseline_columns(combined)
    combined.to_csv(csv_path, index=False)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 3-thread Dual-Cache PoC IPS + CAN Error-Confinement simulator."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Speedometer", "Fuzzing", "Drive Gear", "RPM Gauge"],
        help="Datasets to simulate.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[c.name for c in DEFAULT_SIM_CONFIGS],
        help="Simulation configuration names to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help="Random seed matching split caches.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=os.path.join(POC_RESULTS_DIR, "poc_simulation_results.csv"),
        help="Destination CSV path for simulation results.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating comparison plot.",
    )
    parser.add_argument(
        "--merge-inputs",
        nargs="*",
        default=None,
        help="Merge partial simulation CSVs into --out-csv and regenerate plots without re-simulating.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(POC_RESULTS_DIR, exist_ok=True)

    csv_out = args.out_csv
    fig_out = os.path.join(POC_RESULTS_DIR, "fig_poc_comparison.png")
    fig_s2_out = os.path.join(POC_RESULTS_DIR, "fig_stage2_cascade.png")
    stage2_csv = os.path.join(POC_RESULTS_DIR, "stage2_test_comparison.csv")

    if args.merge_inputs is not None:
        combined_df = pd.read_csv(csv_out) if os.path.exists(csv_out) else pd.DataFrame()
        for partial_path in args.merge_inputs:
            part_df = pd.read_csv(partial_path)
            part_ds = part_df["dataset"].unique().tolist()
            if not combined_df.empty:
                combined_df = combined_df[~combined_df["dataset"].isin(part_ds)]
            combined_df = pd.concat([combined_df, part_df], ignore_index=True)
        combined_df = add_relative_baseline_columns(combined_df)
        combined_df.to_csv(csv_out, index=False)
        if not args.no_plot:
            plot_poc_comparison(combined_df, fig_out)
            plot_stage2_cascade_summary(stage2_csv, fig_s2_out)
        print(f"Merged CSV -> {csv_out} ({len(combined_df)} rows)")
        print(f"Saved figures -> {fig_out}, {fig_s2_out}")
        return

    cfg_map = {c.name: c for c in DEFAULT_SIM_CONFIGS}
    selected_cfgs = [cfg_map[name] for name in args.configs]

    all_rows: List[Dict[str, Any]] = []
    for ds_name in args.datasets:
        print(f"\n{'=' * 72}\n[PoC Simulator] Dataset: {ds_name}\n{'=' * 72}")
        s1_path, s2_path = get_model_artifact_paths(POC_MODELS_DIR, ds_name)
        if not os.path.exists(s1_path) or not os.path.exists(s2_path):
            raise FileNotFoundError(
                f"Missing Stage-1 or Stage-2 persisted model for '{ds_name}': "
                f"{s1_path}, {s2_path}. Run train_stage2.py first."
            )

        fast_s1 = FastIsolationForest(joblib.load(s1_path))
        fast_s2 = FastStage2Evaluator(joblib.load(s2_path))
        stream = load_prepared_test_stream(ds_name, random_state=args.seed)

        for sim_cfg in selected_cfgs:
            row = run_single_simulation(
                sim_cfg=sim_cfg,
                stream=stream,
                fast_s1=fast_s1,
                fast_s2=fast_s2 if sim_cfg.use_stage2 else None,
            )
            all_rows.append(row)
            print(
                f"  [{sim_cfg.name}] wall={row['wall_time_sec']:.2f}s | "
                f"FP_init={row['stream_fp_initial']:,} | rescued_retrans={row['stream_fp_rescued_retrans']:,} | "
                f"BusOff={row['bus_off_events']:,} (ECUs={row['unique_bus_off_ecus']}) | "
                f"MaxTEC={row['max_tec_global']} | Recall={row['attack_recall_pct']:.2f}% | "
                f"LostNormal={row['normal_lost_total']:,} | "
                f"S1_avg={row['stage1_avg_us']:.1f}us S2_calls={row['stage2_calls']:,} "
                f"C2_hits={row['cache2_hits']:,}"
            )

        del stream, fast_s1, fast_s2
        gc.collect()

    new_df = pd.DataFrame(all_rows)
    combined_df = merge_simulation_csv(csv_out, new_df, args.datasets)
    if not args.no_plot:
        plot_poc_comparison(combined_df, fig_out)
        plot_stage2_cascade_summary(stage2_csv, fig_s2_out)
        print(f"Saved comparison figures -> {fig_out}, {fig_s2_out}")
    print(f"\nSaved simulation summary -> {csv_out}")


if __name__ == "__main__":
    main()
