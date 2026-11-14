"""Native C IsolationForest wrapper via ctypes for fast single-frame PoC inference.

Compiles `fast_iforest.c` into `fast_iforest.so` on demand and evaluates scikit-learn
`IsolationForest` and `Stage2IsolationForest` trees with sub-microsecond overhead
per frame while preserving exact numerical agreement with `sklearn`.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length

from models_stage2 import CAN_ID_MAX_VALUE, Stage2IsolationForest

_POC_DIR = os.path.dirname(os.path.abspath(__file__))
_C_SRC = os.path.join(_POC_DIR, "fast_iforest.c")
_SO_PATH = os.path.join(_POC_DIR, "_fast_iforest_c.so")

_LIB: Optional[ctypes.CDLL] = None
DECISION_TIE_EPS = -1e-12


def _ensure_library() -> ctypes.CDLL:
    """Compile `fast_iforest.so` if missing or stale and bind ctypes signatures."""
    global _LIB
    if _LIB is not None:
        return _LIB

    if not os.path.exists(_SO_PATH) or os.path.getmtime(_C_SRC) > os.path.getmtime(_SO_PATH):
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-shared", "-fPIC", _C_SRC, "-o", _SO_PATH, "-lm"],
            check=True,
        )

    lib = ctypes.CDLL(_SO_PATH)
    int32_p = ctypes.POINTER(ctypes.c_int32)
    double_p = ctypes.POINTER(ctypes.c_double)
    float_p = ctypes.POINTER(ctypes.c_float)

    lib.iforest_create.argtypes = [
        ctypes.c_int32,
        int32_p,
        int32_p,
        int32_p,
        int32_p,
        double_p,
        double_p,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.iforest_create.restype = ctypes.c_void_p

    lib.iforest_free.argtypes = [ctypes.c_void_p]
    lib.iforest_free.restype = None

    lib.iforest_decision_one.argtypes = [ctypes.c_void_p, float_p]
    lib.iforest_decision_one.restype = ctypes.c_double

    _LIB = lib
    return lib


def _compute_node_depths(children_left: np.ndarray, children_right: np.ndarray) -> np.ndarray:
    """Compute depth of every node in a binary sklearn tree via iterative DFS."""
    depths = np.zeros(len(children_left), dtype=np.int32)
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        depths[node] = d
        left = int(children_left[node])
        if left != -1:
            stack.append((left, d + 1))
            stack.append((int(children_right[node]), d + 1))
    return depths


class FastIsolationForest:
    """C-accelerated exact single-sample evaluator for `sklearn.IsolationForest`."""

    def __init__(self, clf: IsolationForest) -> None:
        self._lib = _ensure_library()
        self.n_features = int(clf.n_features_in_)

        offsets: List[int] = []
        lefts: List[np.ndarray] = []
        rights: List[np.ndarray] = []
        feats: List[np.ndarray] = []
        threshs: List[np.ndarray] = []
        leaf_depths: List[np.ndarray] = []

        cursor = 0
        for est, est_features in zip(clf.estimators_, clf.estimators_features_):
            tree = est.tree_
            n_nodes = int(tree.node_count)
            offsets.append(cursor)

            raw_cl = tree.children_left.astype(np.int32)
            raw_cr = tree.children_right.astype(np.int32)
            raw_f = tree.feature.astype(np.int32)
            f_map = np.asarray(est_features, dtype=np.int32)
            remapped_f = np.where(raw_f >= 0, f_map[np.maximum(raw_f, 0)], -2).astype(np.int32)

            node_depths = _compute_node_depths(raw_cl, raw_cr).astype(np.float64)
            c_leaf = _average_path_length(tree.n_node_samples).astype(np.float64)

            cl = np.where(raw_cl != -1, raw_cl + cursor, -1).astype(np.int32)
            cr = np.where(raw_cr != -1, raw_cr + cursor, -1).astype(np.int32)

            lefts.append(cl)
            rights.append(cr)
            feats.append(remapped_f)
            threshs.append(tree.threshold.astype(np.float64))
            leaf_depths.append(node_depths + c_leaf)
            cursor += n_nodes

        self._tree_offsets = np.asarray(offsets, dtype=np.int32)
        self._left = np.ascontiguousarray(np.concatenate(lefts), dtype=np.int32)
        self._right = np.ascontiguousarray(np.concatenate(rights), dtype=np.int32)
        self._feat = np.ascontiguousarray(np.concatenate(feats), dtype=np.int32)
        self._thresh = np.ascontiguousarray(np.concatenate(threshs), dtype=np.float64)
        self._leaf_depth = np.ascontiguousarray(np.concatenate(leaf_depths), dtype=np.float64)

        c_max = float(_average_path_length(np.array([clf.max_samples_]))[0])
        self._handle = self._lib.iforest_create(
            ctypes.c_int32(len(offsets)),
            self._tree_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._left.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._right.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._feat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._thresh.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._leaf_depth.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int32(cursor),
            ctypes.c_double(c_max),
            ctypes.c_double(float(clf.offset_)),
        )
        if not self._handle:
            raise MemoryError("Failed to allocate native IForestModel")

        self._scratch = np.zeros(self.n_features, dtype=np.float32)
        self._scratch_ptr = self._scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        lib = getattr(self, "_lib", None)
        if handle and lib is not None:
            lib.iforest_free(handle)
            self._handle = None

    def decision_one_f32(self, x_f32: np.ndarray) -> float:
        """Evaluate decision_function for a contiguous float32 1D array of features."""
        return float(
            self._lib.iforest_decision_one(
                self._handle,
                x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        )

    def predict_one_f32(self, x_f32: np.ndarray) -> int:
        """Return 1 if anomaly (< DECISION_TIE_EPS), else 0."""
        return 1 if self.decision_one_f32(x_f32) < DECISION_TIE_EPS else 0

    def decision_one_from_tuple(self, values: Tuple[float, ...]) -> float:
        """Evaluate decision_function directly from a Python float tuple (zero alloc)."""
        buf = self._scratch
        for i in range(self.n_features):
            buf[i] = values[i]
        return float(self._lib.iforest_decision_one(self._handle, self._scratch_ptr))


class FastStage2Evaluator:
    """Compiled C evaluator for Stage2IsolationForest (Global 9F and Hybrid Per-ID)."""

    def __init__(self, model: Stage2IsolationForest) -> None:
        self.arch = model.arch
        if model.global_9f_ is None:
            raise RuntimeError("Stage2IsolationForest must be fitted before wrapping.")
        self.global_9f = FastIsolationForest(model.global_9f_)
        self.per_id_models: Dict[int, FastIsolationForest] = {
            int(cid, 16): FastIsolationForest(clf)
            for cid, clf in model.per_id_models_.items()
        }
        self.known_train_ids_int: Set[int] = {
            int(cid, 16) for cid in model.known_train_ids_
        }
        self._scratch9 = np.zeros(9, dtype=np.float32)
        self._scratch8 = np.zeros(8, dtype=np.float32)

    def _predict_global_9f(self, can_id_int: int, payload_f32_8: Any) -> Tuple[int, float]:
        """Fill scratch 9-feature float32 vector and score via the global 9F forest."""
        buf9 = self._scratch9
        buf9[0] = np.float32(can_id_int / CAN_ID_MAX_VALUE)
        buf9[1:] = payload_f32_8
        dec = self.global_9f.decision_one_f32(buf9)
        return (1 if dec < DECISION_TIE_EPS else 0), -dec

    def predict_one(self, can_id_int: int, payload_f32_8: Any) -> Tuple[int, float]:
        """Return `(is_anomaly in {0, 1}, anomaly_score)` for one 8-byte frame."""
        if self.arch == "iForest_9F_Global":
            return self._predict_global_9f(can_id_int, payload_f32_8)

        id_model = self.per_id_models.get(can_id_int)
        if id_model is not None:
            if isinstance(payload_f32_8, np.ndarray) and payload_f32_8.dtype == np.float32 and payload_f32_8.flags["C_CONTIGUOUS"]:
                dec = id_model.decision_one_f32(payload_f32_8)
            else:
                self._scratch8[:] = payload_f32_8
                dec = id_model.decision_one_f32(self._scratch8)
            return (1 if dec < DECISION_TIE_EPS else 0), -dec

        if self.arch == "iForest_Hybrid_StrictUnknown" and can_id_int not in self.known_train_ids_int:
            return 1, 1.0

        return self._predict_global_9f(can_id_int, payload_f32_8)
