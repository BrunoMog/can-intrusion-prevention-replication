"""Stage-2 Isolation Forest models operating on (CAN_ID + 8-byte payload).

Provides unified model classes and feature-preparation utilities shared between
Stage-2 10-Fold CV training/selection (`train_stage2.py`) and the 3-thread PoC
IPS simulator (`poc_simulator.py`). Only Isolation Forest models are used.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Allow importing root repository modules (`utils`, `config`, `run_experiments`)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CAN_ID_MAX_VALUE = 2047.0  # Standard 11-bit CAN ID domain (0x000..0x7FF)
DEFAULT_MIN_ID_SAMPLES = 10

STAGE2_ARCHITECTURES = (
    "iForest_9F_Global",
    "iForest_Hybrid_Fallback",
    "iForest_Hybrid_StrictUnknown",
)

DATASET_SLUGS: Dict[str, str] = {
    "Fuzzing": "fuzzing",
    "Drive Gear": "drive_gear",
    "RPM Gauge": "rpm_gauge",
    "Speedometer": "speedometer",
}


def get_model_artifact_paths(models_dir: str, dataset_name: str) -> Tuple[str, str]:
    """Return canonical (stage1_path, stage2_path) joblib paths for a dataset."""
    slug = DATASET_SLUGS[dataset_name]
    return (
        os.path.join(models_dir, f"{slug}_stage1_5f.joblib"),
        os.path.join(models_dir, f"{slug}_stage2.joblib"),
    )


def parse_can_ids_to_int(can_ids: Iterable[Any]) -> np.ndarray:
    """Convert CAN ID hex strings (or integers) to int64 array via unique lookup."""
    arr = np.asarray(list(can_ids) if not isinstance(can_ids, (np.ndarray, pd.Series)) else can_ids)
    uniq = pd.unique(arr)
    mapping = {
        val: (int(val) if isinstance(val, (int, np.integer)) else int(str(val).strip(), 16))
        for val in uniq
    }
    return np.fromiter((mapping[v] for v in arr), dtype=np.int64, count=len(arr))


def can_ids_to_norm(can_ids: Iterable[Any]) -> np.ndarray:
    """Normalize CAN IDs to [0.0, 1.0] shape (N, 1) using standard 11-bit range."""
    id_ints = parse_can_ids_to_int(can_ids).astype(np.float64)
    return (id_ints / CAN_ID_MAX_VALUE).reshape(-1, 1)


def build_global_9f_matrix(can_ids: Iterable[Any], x_8b: np.ndarray) -> np.ndarray:
    """Stack normalized CAN ID column ahead of the 8 normalized payload bytes."""
    id_norm = can_ids_to_norm(can_ids)
    return np.hstack([id_norm, np.asarray(x_8b, dtype=np.float64)])


def _canonicalize_can_id_str(val: Any) -> str:
    """Canonicalize an 11-bit CAN ID representation to lowercase 4-hex ('0316')."""
    if isinstance(val, (int, np.integer)):
        return f"{int(val):04x}"
    s = str(val).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    return s.zfill(4)


def normalize_can_id_strs(can_ids: Iterable[Any]) -> np.ndarray:
    """Canonicalize CAN IDs as lowercase 4-hex strings for O(1) dict routing."""
    arr = np.asarray(list(can_ids) if not isinstance(can_ids, (np.ndarray, pd.Series)) else can_ids)
    uniq = pd.unique(arr)
    mapping = {v: _canonicalize_can_id_str(v) for v in uniq}
    return np.fromiter((mapping[v] for v in arr), dtype=object, count=len(arr))


def make_iforest(
    n_estimators: int,
    max_samples: Union[int, str],
    contamination: Union[float, str],
    n_samples: int,
    random_state: int = 42,
    n_jobs: int = -1,
) -> IsolationForest:
    """Instantiate an IsolationForest while safely clamping integer max_samples."""
    eff_max_samples: Union[int, str]
    if isinstance(max_samples, (int, np.integer)):
        eff_max_samples = min(int(max_samples), max(1, int(n_samples)))
    else:
        eff_max_samples = max_samples

    return IsolationForest(
        n_estimators=int(n_estimators),
        max_samples=eff_max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def compute_iforest_offset(
    raw_train_scores: np.ndarray, contamination: Union[float, str]
) -> float:
    """Replicate scikit-learn IsolationForest.offset_ cleanly from score_samples."""
    if contamination == "auto":
        return -0.5
    return float(np.percentile(raw_train_scores, 100.0 * float(contamination)))


@dataclass
class FoldRawScoreBundle:
    """Precomputed raw `score_samples` caches for exact multi-contamination CV."""

    val_raw_9f: np.ndarray
    train_raw_9f: np.ndarray
    per_id_train_raw: Dict[str, np.ndarray]
    val_per_id_mask: np.ndarray
    val_raw_per_id: np.ndarray
    val_can_ids: np.ndarray
    val_is_unseen: np.ndarray


def fit_fold_base_models_and_cache(
    train_can_ids: Iterable[Any],
    x_train_8b: np.ndarray,
    val_can_ids: Iterable[Any],
    x_val_8b: np.ndarray,
    n_estimators: int,
    max_samples: Union[int, str],
    min_id_samples: int = DEFAULT_MIN_ID_SAMPLES,
    random_state: int = 42,
    n_jobs: int = -1,
) -> FoldRawScoreBundle:
    """Fit base global + per-ID forests (contamination='auto') once for a fold.

    Allows exact zero-approximation evaluation across all contamination thresholds
    and all 3 Stage-2 architectures (`iForest_9F_Global`, `iForest_Hybrid_Fallback`,
    `iForest_Hybrid_StrictUnknown`).
    """
    tr_ids = normalize_can_id_strs(train_can_ids)
    va_ids = normalize_can_id_strs(val_can_ids)
    x_tr8 = np.asarray(x_train_8b, dtype=np.float64)
    x_va8 = np.asarray(x_val_8b, dtype=np.float64)

    # 1. Global 9F model ([CAN_ID_norm, D0..D7])
    x_tr9 = build_global_9f_matrix(tr_ids, x_tr8)
    x_va9 = build_global_9f_matrix(va_ids, x_va8)
    global_clf = make_iforest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination="auto",
        n_samples=len(x_tr9),
        random_state=random_state,
        n_jobs=n_jobs,
    ).fit(x_tr9)
    train_raw_9f = global_clf.score_samples(x_tr9)
    val_raw_9f = global_clf.score_samples(x_va9)

    # 2. Dedicated Per-ID 8B models for IDs with >= min_id_samples in training fold
    tr_id_series = pd.Series(tr_ids)
    tr_groups = tr_id_series.groupby(tr_id_series).indices
    va_id_series = pd.Series(va_ids)
    va_groups = va_id_series.groupby(va_id_series).indices

    per_id_train_raw: Dict[str, np.ndarray] = {}
    val_per_id_mask = np.zeros(len(va_ids), dtype=bool)
    val_raw_per_id = np.zeros(len(va_ids), dtype=np.float64)

    for cid, tr_idx in tr_groups.items():
        if len(tr_idx) < min_id_samples:
            continue
        x_id_tr = x_tr8[tr_idx]
        clf_id = make_iforest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination="auto",
            n_samples=len(x_id_tr),
            random_state=random_state,
            n_jobs=1,  # Per-ID chunks are small; avoid OpenMP overhead per ID
        ).fit(x_id_tr)
        per_id_train_raw[cid] = clf_id.score_samples(x_id_tr)

        va_idx = va_groups.get(cid)
        if va_idx is not None and len(va_idx) > 0:
            val_per_id_mask[va_idx] = True
            val_raw_per_id[va_idx] = clf_id.score_samples(x_va8[va_idx])

    known_ids = set(tr_groups.keys())
    val_is_unseen = np.fromiter(
        (cid not in known_ids for cid in va_ids), dtype=bool, count=len(va_ids)
    )

    return FoldRawScoreBundle(
        val_raw_9f=val_raw_9f,
        train_raw_9f=train_raw_9f,
        per_id_train_raw=per_id_train_raw,
        val_per_id_mask=val_per_id_mask,
        val_raw_per_id=val_raw_per_id,
        val_can_ids=va_ids,
        val_is_unseen=val_is_unseen,
    )


def evaluate_bundle_for_arch_and_contamination(
    bundle: FoldRawScoreBundle,
    arch: str,
    contamination: Union[float, str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact (y_pred, anomaly_scores) from precomputed raw scores.

    y_pred: 1 for anomaly/attack, 0 for normal.
    anomaly_scores: higher means more anomalous (-decision_function).
    """
    if arch not in STAGE2_ARCHITECTURES:
        raise ValueError(f"Unsupported Stage-2 architecture: {arch}")

    offset_9f = compute_iforest_offset(bundle.train_raw_9f, contamination)
    decision_9f = bundle.val_raw_9f - offset_9f

    if arch == "iForest_9F_Global":
        decision = decision_9f
        return (decision < 0).astype(np.int64), -decision

    # Hybrid variants start from 9F fallback decision and overwrite Per-ID positions
    decision = decision_9f.copy()
    if np.any(bundle.val_per_id_mask):
        id_offsets = {
            cid: compute_iforest_offset(raw_tr, contamination)
            for cid, raw_tr in bundle.per_id_train_raw.items()
        }
        masked_indices = np.flatnonzero(bundle.val_per_id_mask)
        va_ids = bundle.val_can_ids
        offsets_vec = np.fromiter(
            (id_offsets[va_ids[i]] for i in masked_indices),
            dtype=np.float64,
            count=len(masked_indices),
        )
        decision[masked_indices] = bundle.val_raw_per_id[masked_indices] - offsets_vec

    if arch == "iForest_Hybrid_StrictUnknown" and np.any(bundle.val_is_unseen):
        decision[bundle.val_is_unseen] = -1.0  # Unseen CAN ID receives strong anomaly score

    return (decision < 0).astype(np.int64), -decision


class Stage2IsolationForest:
    """Persistable Stage-2 Isolation Forest supporting Global 9F and Hybrid Per-ID."""

    def __init__(
        self,
        arch: str = "iForest_Hybrid_StrictUnknown",
        n_estimators: int = 100,
        max_samples: Union[int, str] = 16384,
        contamination: Union[float, str] = "auto",
        min_id_samples: int = DEFAULT_MIN_ID_SAMPLES,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        if arch not in STAGE2_ARCHITECTURES:
            raise ValueError(f"Unknown arch '{arch}'. Expected one of {STAGE2_ARCHITECTURES}")
        self.arch = arch
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.min_id_samples = min_id_samples
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.global_9f_: Optional[IsolationForest] = None
        self.per_id_models_: Dict[str, IsolationForest] = {}
        self.known_train_ids_: Set[str] = set()

    def fit(self, can_ids: Iterable[Any], x_8b: np.ndarray) -> "Stage2IsolationForest":
        """Train global 9F and (when hybrid) dedicated Per-ID 8B forests."""
        tr_ids = normalize_can_id_strs(can_ids)
        x_tr8 = np.asarray(x_8b, dtype=np.float64)
        x_tr9 = build_global_9f_matrix(tr_ids, x_tr8)

        self.global_9f_ = make_iforest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            n_samples=len(x_tr9),
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(x_tr9)

        tr_series = pd.Series(tr_ids)
        groups = tr_series.groupby(tr_series).indices
        self.known_train_ids_ = set(groups.keys())
        self.per_id_models_ = {}

        if self.arch != "iForest_9F_Global":
            for cid, idx in groups.items():
                if len(idx) < self.min_id_samples:
                    continue
                x_id = x_tr8[idx]
                self.per_id_models_[cid] = make_iforest(
                    n_estimators=self.n_estimators,
                    max_samples=self.max_samples,
                    contamination=self.contamination,
                    n_samples=len(x_id),
                    random_state=self.random_state,
                    n_jobs=1,
                ).fit(x_id)
        return self

    def decision_function(self, can_ids: Iterable[Any], x_8b: np.ndarray) -> np.ndarray:
        """Compute signed decision scores (<0 anomaly, >=0 normal) in batch."""
        if self.global_9f_ is None:
            raise RuntimeError("Model must be fit before calling decision_function.")

        va_ids = normalize_can_id_strs(can_ids)
        x_va8 = np.asarray(x_8b, dtype=np.float64)

        if self.arch == "iForest_9F_Global":
            x_va9 = build_global_9f_matrix(va_ids, x_va8)
            return self.global_9f_.decision_function(x_va9)

        decision = np.empty(len(va_ids), dtype=np.float64)
        va_series = pd.Series(va_ids)
        groups = va_series.groupby(va_series).indices

        fallback_indices: List[np.ndarray] = []
        for cid, idx in groups.items():
            if cid in self.per_id_models_:
                decision[idx] = self.per_id_models_[cid].decision_function(x_va8[idx])
            elif self.arch == "iForest_Hybrid_StrictUnknown" and cid not in self.known_train_ids_:
                decision[idx] = -1.0
            else:
                fallback_indices.append(idx)

        if fallback_indices:
            fb_idx = np.concatenate(fallback_indices)
            x_fb9 = build_global_9f_matrix(va_ids[fb_idx], x_va8[fb_idx])
            decision[fb_idx] = self.global_9f_.decision_function(x_fb9)

        return decision

    def predict(self, can_ids: Iterable[Any], x_8b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (y_pred in {0, 1}, anomaly_scores) for vectorized evaluation."""
        decision = self.decision_function(can_ids, x_8b)
        return (decision < 0).astype(np.int64), -decision

    def predict_one(self, can_id_str: str, x_8b_row: np.ndarray) -> Tuple[int, float]:
        """Single-frame prediction used by the Stage-2 worker thread."""
        if self.global_9f_ is None:
            raise RuntimeError("Model must be fit before calling predict_one.")

        cid = _canonicalize_can_id_str(can_id_str)
        x_row = np.asarray(x_8b_row, dtype=np.float64).reshape(1, 8)

        if self.arch == "iForest_9F_Global":
            x9 = np.hstack([[[int(cid, 16) / CAN_ID_MAX_VALUE]], x_row])
            dec = float(self.global_9f_.decision_function(x9)[0])
            return (1 if dec < 0.0 else 0), -dec

        model_id = self.per_id_models_.get(cid)
        if model_id is not None:
            dec = float(model_id.decision_function(x_row)[0])
            return (1 if dec < 0.0 else 0), -dec

        if self.arch == "iForest_Hybrid_StrictUnknown" and cid not in self.known_train_ids_:
            return 1, 1.0

        x9 = np.hstack([[[int(cid, 16) / CAN_ID_MAX_VALUE]], x_row])
        dec = float(self.global_9f_.decision_function(x9)[0])
        return (1 if dec < 0.0 else 0), -dec
