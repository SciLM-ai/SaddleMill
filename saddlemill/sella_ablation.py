"""Instance-scoped controlled ablations for the pinned Sella 2.5.0 API.

This module deliberately does *not* wire itself into SaddleMill's public config or
``sella_engine``.  Sella-ablation owns only the isolated numerical/runtime hooks; shared-runtime Stage D
owns shared wiring.  The hooks are installed on one optimizer/PES instance and are
restored on context exit.  No class, module, or site-package object is patched.

Scientific components implemented here
--------------------------------------
``normal``
    Execute the pinned Sella path unchanged while collecting only values exposed by
    the real calls.
``center_only_learning``
    Preserve every numerical-Hessian/Olsen probe evaluation but suppress the single
    same-center probe-block ``ApproximateHessian.update(Vs, AVs)`` at the end of
    ``PES.diag``.  Before the first step, materialize Sella's own implicit
    ``ApproximateHessian.asarray()`` identity fallback as a finite scalar-identity
    model while deliberately leaving ``ApproximateHessian.initialized`` false.  This
    makes PRFO's first eigenvector partition well-defined without learning from any
    probe pair; the first accepted center-to-center ``PES._update_H`` therefore still
    performs Sella's normal first secant initialization.  Later accepted center
    updates remain active.
``olsen_learning_simple_translation``
    Preserve native Sella eigensolving/Hessian learning but ask the existing Sella
    restricted-step machinery to use its ``QuasiNewton``/``mmf`` stepper rather than
    PRFO for translation.  For Hessian eigenpairs ``(lambda_i, v_i)``, order=1 uses

        L_0 = -|lambda_0|,  L_i = |lambda_i| (i > 0)
        s(alpha) = -V [(V^T g) / (L + alpha * sign(L))]

    with the same active-coordinate/constraint projection and restricted-step radius
    selected by Sella.  This is eigenvector-following/MMF translation, not partitioned-LBFGS
    partitioned L-BFGS.
``trust_adaptation_isolation``
    Execute the real PRFO/restricted step and real ``PES.kick`` once, then restore the
    pre-step trust radius before another optimizer step can consume the adapted
    value.  The native candidate trust update is retained as diagnostics.
``qn_newton_safe_false``
    Preserve native Sella QN construction, Hessian/Rayleigh--Ritz behavior, trust
    adaptation, and ``RestrictedAtomicStep``.  Only the live QN stepper's
    ``newton_safe`` flag is forced to ``False`` after native construction so the
    pinned Sella 2.5.0 restricted-step solver uses its own safeguarded fallback.
    No step algebra is reimplemented and no PES/HVP evaluation is added.
``ritzmode_prfo_partition``
    Preserve native Sella Hessian learning, Rayleigh--Ritz calls, trust adaptation,
    restricted-step machinery, and PRFO algebra, but replace only PRFO's unstable
    one-dimensional partition direction with the lowest Ritz direction already
    produced by the most recent native ``PES.diag``.  The approximate Hessian is not
    modified to force that direction and no additional eigensolve/HVP/PES call is
    made.

The supplied Sella reference is a locally modified 2.5.0 snapshot.  Runtime wiring
must validate that exact supported version/structure and record the supplied
reference hashes.  This module never installs or vendors Sella.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
import platform
from pathlib import Path
import socket
import sys
from time import perf_counter_ns
from types import MethodType
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "sella_ablation_v1"
SUPPORTED_SELLA_VERSION = "2.5.0"
EXPECTED_DELTA0 = 0.1

NORMAL = "normal"
CENTER_ONLY_LEARNING = "center_only_learning"
SIMPLE_TRANSLATION = "olsen_learning_simple_translation"
TRUST_ISOLATION = "trust_adaptation_isolation"
QN_NEWTON_SAFE_FALSE = "qn_newton_safe_false"
RITZMODE_PRFO_PARTITION = "ritzmode_prfo_partition"

LEARNING_NORMAL = "normal_probe_and_center"
LEARNING_CENTER_ONLY = "center_only"
TRANSLATION_PRFO = "rs_prfo"
TRANSLATION_SIMPLE_EVF = "sella_qn_eigenvector_following"
TRUST_NATIVE = "native"
TRUST_HOLD = "hold"
PARTITION_APPROXIMATE_B = "approximate_b_eigenspace"
PARTITION_NATIVE_RITZ = "native_ritz_mode"

CENTER_ONLY_INITIAL_MODEL = "sella_implicit_identity_materialized"
CENTER_ONLY_INITIAL_SCALE = 1.0

# Diagnostic-only, opt-in failure-state capture. This does not change the
# restricted-step algorithm or any scientific selector. When unset, the path is
# byte-for-byte inactive apart from this module containing the helper code.
RESTRICTED_STEP_CAPTURE_ENV = "SADDLEMILL_SELLA_RS_CAPTURE_DIR"
RESTRICTED_STEP_CAPTURE_SCHEMA = "sella_restricted_step_failure_v1"
RESTRICTED_STEP_TRACE_HEAD = 8
RESTRICTED_STEP_TRACE_TAIL = 32

# Exact hashes of the supplied locally modified Sella reference files that define
# the mechanisms Sella-ablation relies on.  The complete reference tree identity belongs in
# the worker evidence manifest; these are the runtime-critical subset.
REFERENCE_SHA256 = {
    "installed_sella/eigensolvers.py": "627f1d8ec2b766c324294af4ff000b904374d55857a359e90677d438cf7e95bf",
    "installed_sella/hessian_update.py": "207b06566bab87a7022f36e8076a8d8c9643b954540538fd73b2026277f63443",
    "installed_sella/linalg.py": "5c3b35da002c9fd3a25dcf95bb28cef74f3ba03c74a114ba797043ccf30daef8",
    "installed_sella/optimize/optimize.py": "be36b7530b106f02771605d2d3f8a9df11e11e7451898374d693ce4c6b5801b3",
    "installed_sella/optimize/restricted_step.py": "df55916fb23debb2dcdd920e8ecb0c57e79984c7658fcd0a0e9e6114e44e9af3",
    "installed_sella/optimize/stepper.py": "7ccd629c97d66a005f43e365db678c0d1996a7eff5573264941359aa8fb528af",
    "installed_sella/peswrapper.py": "3d0e4a8140e73e0df37e80878c2d2d4290ec26855c42852312594b1be5433020",
}


class SellaAblationError(RuntimeError):
    """Fail-closed unsupported Sella version/API/composition."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class SellaAblationSpec:
    """Resolved independently composable Sella ablation components."""

    learning: str = LEARNING_NORMAL
    translation: str = TRANSLATION_PRFO
    trust: str = TRUST_NATIVE
    partition: str = PARTITION_APPROXIMATE_B
    qn_newton_safe: bool | None = None
    requested_tokens: tuple[str, ...] = (NORMAL,)

    @property
    def is_normal(self) -> bool:
        return (
            self.learning == LEARNING_NORMAL
            and self.translation == TRANSLATION_PRFO
            and self.trust == TRUST_NATIVE
            and self.partition == PARTITION_APPROXIMATE_B
            and self.qn_newton_safe is None
        )

    @property
    def selector(self) -> str:
        if self.is_normal:
            return NORMAL
        tokens: list[str] = []
        if self.learning == LEARNING_CENTER_ONLY:
            tokens.append(CENTER_ONLY_LEARNING)
        if self.translation == TRANSLATION_SIMPLE_EVF:
            tokens.append(SIMPLE_TRANSLATION)
        if self.trust == TRUST_HOLD:
            tokens.append(TRUST_ISOLATION)
        if self.partition == PARTITION_NATIVE_RITZ:
            tokens.append(RITZMODE_PRFO_PARTITION)
        if self.qn_newton_safe is False:
            tokens.append(QN_NEWTON_SAFE_FALSE)
        return "+".join(tokens)

    def component_matrix(self) -> dict[str, Any]:
        matrix = {
            "schema": SCHEMA_VERSION,
            "selector": self.selector,
            "hessian_learning": self.learning,
            "translation": self.translation,
            "trust_adaptation": self.trust,
            "mode_solver": "sella_native_jd0_default",
            "restricted_step": "sella_native_selected_rs",
            "target_order": 1,
            "delta0": EXPECTED_DELTA0,
        }
        # Preserve pre-Ritz-partition ablation selector identities byte-for-byte: the default
        # approximate-B partition is intentionally implicit.  Only the new
        # scientific selector adds a partition key to the identity payload.
        if self.partition == PARTITION_NATIVE_RITZ:
            matrix["prfo_partition"] = PARTITION_NATIVE_RITZ
        if self.qn_newton_safe is False:
            matrix["qn_newton_safe"] = False
            matrix["restricted_step_numerics"] = "sella_native_qn_safeguarded"
        if self.learning == LEARNING_CENTER_ONLY:
            matrix.update({
                "center_only_initial_model": CENTER_ONLY_INITIAL_MODEL,
                "center_only_initial_scale": CENTER_ONLY_INITIAL_SCALE,
                "center_only_initial_model_probe_independent": True,
            })
        return matrix

    def identity(self) -> str:
        payload = json.dumps(self.component_matrix(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:20]


def _split_selector(value: Any) -> tuple[str, ...]:
    if value is None:
        return (NORMAL,)
    if isinstance(value, str):
        raw = value.strip().lower()
        if not raw:
            return (NORMAL,)
        for separator in (",", " "):
            raw = raw.replace(separator, "+")
        tokens = tuple(tok for tok in raw.split("+") if tok)
        return tokens or (NORMAL,)
    if isinstance(value, Iterable):
        tokens = tuple(str(tok).strip().lower() for tok in value if str(tok).strip())
        return tokens or (NORMAL,)
    raise SellaAblationError("invalid_selector_type", f"Unsupported Sella ablation selector: {value!r}")


def resolve_sella_ablation(value: Any = None) -> SellaAblationSpec:
    """Resolve a public selector or a controlled ``+`` composition.

    ``normal`` is exclusive.  Other components are independent and may be composed
    in any order; the returned selector is canonicalized.
    """

    tokens = _split_selector(value)
    known = {
        NORMAL, CENTER_ONLY_LEARNING, SIMPLE_TRANSLATION, TRUST_ISOLATION,
        RITZMODE_PRFO_PARTITION, QN_NEWTON_SAFE_FALSE,
    }
    unknown = [tok for tok in tokens if tok not in known]
    if unknown:
        raise SellaAblationError(
            "unknown_component",
            "Unknown Sella ablation component(s): " + ", ".join(sorted(set(unknown))),
        )
    if NORMAL in tokens and len(tokens) != 1:
        raise SellaAblationError("normal_not_composable", "'normal' cannot be combined with an ablation component")

    unique = set(tokens)
    if SIMPLE_TRANSLATION in unique and RITZMODE_PRFO_PARTITION in unique:
        raise SellaAblationError(
            "partition_without_prfo",
            "ritzmode_prfo_partition cannot be combined with the QN/MMF simple-translation selector",
        )
    if QN_NEWTON_SAFE_FALSE in unique and len(unique) != 1:
        raise SellaAblationError(
            "qn_newton_safe_not_composable",
            "qn_newton_safe_false is a QN-only restricted-step numerical policy and cannot be combined with PRFO ablations",
        )
    return SellaAblationSpec(
        learning=LEARNING_CENTER_ONLY if CENTER_ONLY_LEARNING in unique else LEARNING_NORMAL,
        translation=TRANSLATION_SIMPLE_EVF if SIMPLE_TRANSLATION in unique else TRANSLATION_PRFO,
        trust=TRUST_HOLD if TRUST_ISOLATION in unique else TRUST_NATIVE,
        partition=PARTITION_NATIVE_RITZ if RITZMODE_PRFO_PARTITION in unique else PARTITION_APPROXIMATE_B,
        qn_newton_safe=False if QN_NEWTON_SAFE_FALSE in unique else None,
        requested_tokens=tokens,
    )


def eigenvector_following_reference_step(
    gradient: np.ndarray,
    hessian: np.ndarray,
    *,
    order: int = 1,
    singular_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Unrestricted reference for Sella's ``QuasiNewton`` EVF/MMF equation.

    Runtime ablations do not use this helper to take a step; they select Sella's own
    ``QuasiNewton`` stepper so Sella retains its exact constraint/restricted-step
    behavior.  This pure helper exists for mathematical tests/reporting only.
    """

    g = np.asarray(gradient, dtype=float).reshape(-1)
    B = np.asarray(hessian, dtype=float)
    if B.shape != (g.size, g.size):
        raise ValueError("hessian shape must match gradient dimension")
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(B)):
        raise ValueError("gradient/hessian must be finite")
    if not 0 <= int(order) <= g.size:
        raise ValueError("order must be between 0 and dimension")
    evals, evecs = np.linalg.eigh(0.5 * (B + B.T))
    denom = np.abs(evals)
    signs = np.ones_like(denom)
    signs[: int(order)] = -1.0
    denom = denom * signs
    if np.any(np.abs(denom) <= float(singular_tolerance)):
        raise SellaAblationError("singular_evf_curvature", "EVF reference step has near-zero curvature")
    return -(evecs @ ((evecs.T @ g) / denom))


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _norm(value: Any) -> float | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return float(np.linalg.norm(arr))


def _max_cartesian_norm(value: Any) -> float | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 0 or arr.size % 3 or not np.all(np.isfinite(arr)):
        return None
    return float(np.max(np.linalg.norm(arr.reshape(-1, 3), axis=1), initial=0.0))


def _array_summary(value: Any) -> dict[str, Any]:
    """Return bounded numerical diagnostics without altering the source array."""
    if value is None:
        return {"present": False}
    try:
        arr = np.asarray(value)
    except Exception as exc:
        return {"present": True, "coercion_error": f"{type(exc).__name__}: {exc}"}
    out: dict[str, Any] = {
        "present": True,
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if not np.issubdtype(arr.dtype, np.number):
        return out
    try:
        finite = np.isfinite(arr)
        out["finite"] = bool(np.all(finite))
        out["nonfinite_count"] = int(arr.size - np.count_nonzero(finite))
        if arr.size and np.any(finite):
            vals = np.asarray(arr[finite], dtype=float)
            out["min_finite"] = float(np.min(vals))
            out["max_finite"] = float(np.max(vals))
            out["norm_finite"] = float(np.linalg.norm(vals))
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.size:
            arrf = np.asarray(arr, dtype=float)
            if np.all(np.isfinite(arrf)):
                resid = arrf - arrf.T
                out["symmetry_max_abs"] = float(np.max(np.abs(resid), initial=0.0))
                out["symmetry_fro"] = float(np.linalg.norm(resid))
    except Exception as exc:
        out["summary_error"] = f"{type(exc).__name__}: {exc}"
    return out


class _RestrictedStepFailureCapture:
    """Capture one exact restricted-step state without changing its numerics.

    The capture binds only after the native restricted-step constructor has
    finished. ``eval`` is delegated to the original implementation first, and
    diagnostics are copied only after the native result exists. The trace is
    bounded to a small head plus a rolling tail; full matrices are written only
    if the native call raises ``Restricted step failed to converge!``.
    """

    def __init__(self, directory: str, metadata: Mapping[str, Any]):
        self.directory = Path(directory)
        self.metadata = dict(metadata)
        self.rs = None
        self.eval_count = 0
        self.trace_head: list[dict[str, Any]] = []
        self.trace_tail = deque(maxlen=RESTRICTED_STEP_TRACE_TAIL)
        self.arrays: dict[str, np.ndarray] = {}
        self.state: dict[str, Any] = {}
        self.output_json: str | None = None
        self.output_npz: str | None = None

    def bind(self, rs: object) -> None:
        # Keep only the live restricted-step object and bounded scalar metadata
        # during successful steps. Full matrices are copied only after the native
        # solver has already failed, so enabled diagnostics do not add O(N^2)
        # copying to every successful optimizer step.
        self.rs = rs
        stepper = getattr(rs, "stepper", None)
        H = getattr(stepper, "H", None) if stepper is not None else None
        evals = getattr(H, "evals", None) if H is not None else None
        order = int(getattr(stepper, "order", 0) or 0) if stepper is not None else None
        n_eval = None
        if evals is not None:
            try:
                n_eval = int(np.asarray(evals).size)
            except Exception:
                n_eval = None
        self.state = {
            "schema": RESTRICTED_STEP_CAPTURE_SCHEMA,
            "restricted_step_class": type(rs).__name__,
            "stepper_class": type(stepper).__name__ if stepper is not None else None,
            "order": order,
            "trust_radius": _finite_float(getattr(rs, "delta", None)),
            "tolerance": _finite_float(getattr(rs, "tol", None)),
            "maxiter": int(getattr(rs, "maxiter", 0) or 0),
            "alpha0": _finite_float(getattr(stepper, "alpha0", None)) if stepper is not None else None,
            "alphamin": _finite_float(getattr(stepper, "alphamin", None)) if stepper is not None else None,
            "alphamax_is_posinf": bool(np.isposinf(getattr(stepper, "alphamax", np.nan))) if stepper is not None else None,
            "slope": _finite_float(getattr(stepper, "slope", None)) if stepper is not None else None,
            "newton_safe": bool(getattr(stepper, "newton_safe", False)) if stepper is not None else None,
            "ascent_partition_size": order,
            "descent_partition_size": (n_eval - order) if n_eval is not None and order is not None else None,
            **self.metadata,
        }

    def _snapshot_arrays(self) -> None:
        if self.rs is None or self.arrays:
            return
        rs = self.rs
        pes = getattr(rs, "pes", None)
        stepper = getattr(rs, "stepper", None)
        atoms = getattr(pes, "atoms", None)

        def add_array(name: str, value: Any) -> None:
            if value is None:
                return
            try:
                arr = np.asarray(value).copy()
            except Exception:
                return
            if arr.dtype == object:
                return
            self.arrays[name] = arr

        add_array("projection_P", getattr(rs, "P", None))
        add_array("constraint_step_scons", getattr(rs, "scons", None))
        add_array("trust_radius", np.asarray(getattr(rs, "delta", np.nan), dtype=float))
        add_array("tolerance", np.asarray(getattr(rs, "tol", np.nan), dtype=float))
        add_array("maxiter", np.asarray(getattr(rs, "maxiter", 0), dtype=np.int64))
        if pes is not None:
            curr = dict(getattr(pes, "curr", {}) or {})
            add_array("gradient_raw", curr.get("g"))
            add_array("coordinate_x", curr.get("x"))
            try:
                add_array("model_hessian", pes.H.asarray())
            except Exception:
                pass
        if atoms is not None:
            add_array("geometry_positions", getattr(atoms, "positions", None))
            cell = getattr(atoms, "cell", None)
            add_array("geometry_cell", getattr(cell, "array", cell))
            add_array("geometry_pbc", getattr(atoms, "pbc", None))
            add_array("atomic_numbers", getattr(atoms, "numbers", None))
        if stepper is not None:
            add_array("order", np.asarray(getattr(stepper, "order", 0), dtype=np.int64))
            add_array("alpha0", np.asarray(getattr(stepper, "alpha0", np.nan), dtype=float))
            add_array("alphamin", np.asarray(getattr(stepper, "alphamin", np.nan), dtype=float))
            add_array("alphamax", np.asarray(getattr(stepper, "alphamax", np.nan), dtype=float))
            add_array("slope", np.asarray(getattr(stepper, "slope", np.nan), dtype=float))
            add_array("newton_safe", np.asarray(int(bool(getattr(stepper, "newton_safe", False))), dtype=np.int8))
            add_array("gradient_projected", getattr(stepper, "g", None))
            H = getattr(stepper, "H", None)
            if H is not None:
                try:
                    add_array("projected_hessian", H.asarray())
                except Exception:
                    pass
                add_array("projected_eigenvalues", getattr(H, "evals", None))
                add_array("projected_eigenvectors", getattr(H, "evecs", None))
            for name in ("L", "V", "Vg", "ones"):
                add_array(f"stepper_{name}", getattr(stepper, name, None))
        self.state["arrays"] = {name: _array_summary(arr) for name, arr in self.arrays.items()}

    def record_eval(self, alpha: Any, result: Any) -> None:
        self.eval_count += 1
        row: dict[str, Any] = {"eval_index": self.eval_count - 1, "alpha": _finite_float(alpha)}
        try:
            _, val, dval = result
            row["constraint_value"] = _finite_float(val)
            row["constraint_derivative"] = _finite_float(dval)
            delta = self.state.get("trust_radius")
            row["constraint_error"] = (
                None if delta is None or row["constraint_value"] is None
                else row["constraint_value"] - delta
            )
        except Exception as exc:
            row["trace_decode_error"] = f"{type(exc).__name__}: {exc}"
        if len(self.trace_head) < RESTRICTED_STEP_TRACE_HEAD:
            self.trace_head.append(dict(row))
        self.trace_tail.append(dict(row))

    def write_failure(self, exc: BaseException) -> tuple[str, str]:
        self._snapshot_arrays()
        self.directory.mkdir(parents=True, exist_ok=True)
        rank = self.metadata.get("rank", "na")
        src = self.metadata.get("src_index", "na")
        attempt = self.metadata.get("attempt_id", "na")
        selected = self.metadata.get("selected_index", "na")
        step = self.metadata.get("step_serial", "na")
        stem = f"rs_failure_rank{rank}_src{src}_attempt{attempt}_sel{selected}_step{step}"
        npz_path = self.directory / f"{stem}.npz"
        json_path = self.directory / f"{stem}.json"
        np.savez_compressed(npz_path, **self.arrays)
        trace_tail = list(self.trace_tail)
        head_indices = {row.get("eval_index") for row in self.trace_head}
        trace_tail = [row for row in trace_tail if row.get("eval_index") not in head_indices]
        payload = {
            **self.state,
            "exception": f"{type(exc).__name__}: {exc}",
            "eval_count": int(self.eval_count),
            "trace_head": self.trace_head,
            "trace_tail": trace_tail,
            "npz_file": npz_path.name,
        }
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        self.output_json = str(json_path)
        self.output_npz = str(npz_path)
        return self.output_json, self.output_npz


def _qn_newton_safe_false_restricted_step_class(base_cls: type, on_construct) -> type:
    """Return an ephemeral native-Sella restricted-step subclass for QN only.

    The base Sella constructor creates the real ``QuasiNewton`` stepper.  This
    wrapper changes only its documented ``newton_safe`` flag after construction;
    all subsequent restricted-step evaluations remain Sella's own implementation.
    """
    if not isinstance(base_cls, type):
        raise SellaAblationError(
            "qn_newton_safe_requires_class",
            f"QN newton_safe override expected a restricted-step class, got {base_cls!r}",
        )

    class QNNewtonSafeFalseRestrictedStep(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            stepper = getattr(self, "stepper", None)
            if stepper is None:
                raise SellaAblationError(
                    "qn_stepper_missing",
                    "Pinned Sella RestrictedAtomicStep did not expose its constructed QN stepper",
                )
            if type(stepper).__name__ != "QuasiNewton":
                raise SellaAblationError(
                    "qn_stepper_type_changed",
                    f"Expected native Sella QuasiNewton stepper, got {type(stepper).__name__}",
                )
            if not hasattr(stepper, "newton_safe"):
                raise SellaAblationError(
                    "qn_newton_safe_missing",
                    "Pinned Sella QuasiNewton no longer exposes newton_safe",
                )
            before = bool(stepper.newton_safe)
            stepper.newton_safe = False
            if bool(stepper.newton_safe):
                raise SellaAblationError(
                    "qn_newton_safe_override_failed",
                    "Failed to force the live native Sella QN stepper newton_safe=False",
                )
            on_construct(self, before)

    QNNewtonSafeFalseRestrictedStep.__name__ = f"QNNewtonSafeFalse{base_cls.__name__}"
    QNNewtonSafeFalseRestrictedStep.__qualname__ = QNNewtonSafeFalseRestrictedStep.__name__
    return QNNewtonSafeFalseRestrictedStep


def _capturing_restricted_step_class(base_cls: type, capture: _RestrictedStepFailureCapture) -> type:
    """Return an ephemeral subclass that observes native ``eval`` calls only."""
    if not isinstance(base_cls, type):
        raise SellaAblationError(
            "restricted_step_capture_requires_class",
            f"Restricted-step diagnostic expected a class, got {base_cls!r}",
        )

    class CapturingRestrictedStep(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            capture.bind(self)

        def eval(self, alpha):
            result = super().eval(alpha)
            capture.record_eval(alpha, result)
            return result

    CapturingRestrictedStep.__name__ = f"Captured{base_cls.__name__}"
    CapturingRestrictedStep.__qualname__ = CapturingRestrictedStep.__name__
    return CapturingRestrictedStep


_INTERNAL_EIGH_ERROR = "Internal Error."
_INTERNAL_EIGH_OPTIONAL_ARRAY_LIMIT_BYTES = 64 * 1024 * 1024


def _json_scalar(value: Any) -> Any:
    """Return a JSON-safe scalar without triggering numerical work."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        out = float(value)
        return out if math.isfinite(out) else repr(out)
    if isinstance(value, np.integer):
        return int(value)
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return repr(value)
    return out if math.isfinite(out) else repr(out)


def _array_capture_summary(value: Any) -> dict[str, Any] | None:
    """Describe an already-materialized array without solving another system."""
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "nbytes": int(arr.nbytes),
        "c_contiguous": bool(arr.flags.c_contiguous),
        "f_contiguous": bool(arr.flags.f_contiguous),
        "strides": [int(x) for x in arr.strides],
        "size": int(arr.size),
    }
    try:
        finite = np.isfinite(arr)
    except TypeError:
        return summary
    summary["finite_count"] = int(np.count_nonzero(finite))
    summary["nonfinite_count"] = int(arr.size - np.count_nonzero(finite))
    if arr.size and np.any(finite):
        vals = np.asarray(arr[finite], dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            summary["finite_min"] = _json_scalar(np.min(vals))
            summary["finite_max"] = _json_scalar(np.max(vals))
            summary["finite_abs_max"] = _json_scalar(np.max(np.abs(vals)))
            summary["finite_l2_norm"] = _json_scalar(np.linalg.norm(vals))
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.size:
        with np.errstate(over="ignore", invalid="ignore"):
            diff = arr - arr.T.conj()
        try:
            if np.all(np.isfinite(diff)):
                summary["hermitian_residual_abs_max"] = _json_scalar(
                    np.max(np.abs(diff), initial=0.0)
                )
            else:
                summary["hermitian_residual_abs_max"] = None
        except TypeError:
            summary["hermitian_residual_abs_max"] = None
    return summary


def _optional_array(arrays: dict[str, np.ndarray], name: str, value: Any) -> bool:
    """Capture optional upstream context only when it stays within the size bound."""
    if value is None:
        return False
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    if arr.nbytes > _INTERNAL_EIGH_OPTIONAL_ARRAY_LIMIT_BYTES:
        return False
    arrays[name] = arr.copy(order="K")
    return True


def _capture_internal_eigh_failure(
    exc: BaseException,
    optimizer: object,
    metadata: Mapping[str, Any] | None,
    step_serial: int,
) -> list[str]:
    """Persist the exact Sella PRFO ``eigh(A)`` input on its rare internal error.

    This is failure-path instrumentation only.  It does not call another eigensolver,
    alter ``A``, retry the step, or make any PES/force request.  The traceback frame
    already owns the exact local ``A`` passed to SciPy, so the diagnostic copies that
    matrix and the restricted-step scalars before re-raising the original exception.
    """
    if not isinstance(exc, np.linalg.LinAlgError) or str(exc) != _INTERNAL_EIGH_ERROR:
        return []

    frames: list[tuple[object, int]] = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append((tb.tb_frame, int(tb.tb_lineno)))
        tb = tb.tb_next

    rfo_frame = None
    prfo_frame = None
    restricted_frame = None
    for frame, lineno in frames:
        filename = str(frame.f_code.co_filename).replace("\\", "/")
        name = frame.f_code.co_name
        locals_ = frame.f_locals
        if filename.endswith("/sella/optimize/stepper.py") and name == "get_s":
            if "A" in locals_:
                rfo_frame = (frame, lineno)
            else:
                obj = locals_.get("self")
                if hasattr(obj, "min") and hasattr(obj, "max"):
                    prfo_frame = (frame, lineno)
        elif filename.endswith("/sella/optimize/restricted_step.py") and name == "get_s":
            restricted_frame = (frame, lineno)

    if rfo_frame is None:
        return []

    rfo_locals = rfo_frame[0].f_locals
    A = np.asarray(rfo_locals.get("A"))
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return []
    rfo_self = rfo_locals.get("self")
    alpha = rfo_locals.get("alpha")

    side = "unknown"
    prfo_self = None
    if prfo_frame is not None:
        prfo_self = prfo_frame[0].f_locals.get("self")
        if getattr(prfo_self, "min", None) is rfo_self:
            side = "min"
        elif getattr(prfo_self, "max", None) is rfo_self:
            side = "max"

    restricted_context: dict[str, Any] = {}
    restricted_self = None
    if restricted_frame is not None:
        restricted_locals = restricted_frame[0].f_locals
        restricted_self = restricted_locals.get("self")
        for key in ("alpha", "lower", "upper", "niter", "err", "val", "dval"):
            if key in restricted_locals:
                restricted_context[key] = _json_scalar(restricted_locals[key])
        if restricted_self is not None:
            restricted_context.update({
                "delta": _json_scalar(getattr(restricted_self, "delta", None)),
                "tol": _json_scalar(getattr(restricted_self, "tol", None)),
                "maxiter": _json_scalar(getattr(restricted_self, "maxiter", None)),
                "restricted_step_class": (
                    f"{type(restricted_self).__module__}.{type(restricted_self).__name__}"
                ),
            })

    meta = dict(metadata or {})
    src_index = meta.get("src_index", "unknown")
    attempt_id = meta.get("attempt_id", "unknown")
    selected_index = meta.get("selected_index", "unknown")
    prefix = (
        f"sella_internal_eigh_{src_index}_{attempt_id}_{selected_index}_"
        f"step{int(step_serial)}"
    )
    json_path = f"{prefix}.json"
    npz_path = f"{prefix}.npz"

    arrays: dict[str, np.ndarray] = {"A_eigh": A.copy(order="K")}
    base_A = getattr(rfo_self, "A", None)
    if base_A is not None:
        base_A_arr = np.asarray(base_A)
        arrays["rfo_base_A"] = base_A_arr.copy(order="K")
        if base_A_arr.ndim == 2 and base_A_arr.shape[0] == base_A_arr.shape[1] and base_A_arr.shape[0] >= 1:
            arrays["projected_hessian"] = base_A_arr[:-1, :-1].copy(order="K")
            arrays["projected_gradient"] = base_A_arr[:-1, -1].copy(order="K")

    optional_arrays: dict[str, bool] = {}
    if prfo_self is not None:
        basis = getattr(prfo_self, "Vmin" if side == "min" else "Vmax", None)
        optional_arrays["prfo_partition_basis"] = _optional_array(
            arrays, "prfo_partition_basis", basis
        )
    if restricted_self is not None:
        optional_arrays["restricted_projection_P"] = _optional_array(
            arrays, "restricted_projection_P", getattr(restricted_self, "P", None)
        )
        optional_arrays["restricted_scons"] = _optional_array(
            arrays, "restricted_scons", getattr(restricted_self, "scons", None)
        )

    pes = getattr(optimizer, "pes", None)
    if pes is not None:
        hobj = getattr(pes, "H", None)
        harray = None
        try:
            harray = None if hobj is None else hobj.asarray()
        except Exception:
            harray = None
        optional_arrays["pes_model_hessian"] = _optional_array(
            arrays, "pes_model_hessian", harray
        )
        curr = getattr(pes, "curr", {}) or {}
        optional_arrays["pes_cached_gradient"] = _optional_array(
            arrays, "pes_cached_gradient", curr.get("g") if isinstance(curr, Mapping) else None
        )
        atoms = getattr(pes, "atoms", None)
        if atoms is not None:
            for name, value in (
                ("geometry_positions", getattr(atoms, "positions", None)),
                ("geometry_cell", getattr(getattr(atoms, "cell", None), "array", None)),
                ("geometry_pbc", getattr(atoms, "pbc", None)),
                ("geometry_numbers", getattr(atoms, "numbers", None)),
                ("geometry_tags", None),
            ):
                if name == "geometry_tags":
                    try:
                        value = atoms.get_tags()
                    except Exception:
                        value = None
                optional_arrays[name] = _optional_array(arrays, name, value)

    try:
        import scipy
        scipy_version = getattr(scipy, "__version__", None)
    except Exception:
        scipy_version = None

    optimizer_context = {
        "delta": _json_scalar(getattr(optimizer, "delta", None)),
        "delta_cell": _json_scalar(getattr(optimizer, "delta_cell", None)),
        "method": _json_scalar(getattr(optimizer, "method", None)),
        "order": _json_scalar(getattr(optimizer, "ord", None)),
        "nsteps": _json_scalar(getattr(optimizer, "nsteps", None)),
        "nsteps_since_diag": _json_scalar(getattr(optimizer, "nsteps_since_diag", None)),
        "nsteps_per_diag": _json_scalar(getattr(optimizer, "nsteps_per_diag", None)),
        "diag_every_n": _json_scalar(getattr(optimizer, "diag_every_n", None)),
        "pes_neval": _json_scalar(getattr(pes, "neval", None) if pes is not None else None),
        "sella_version": _json_scalar(getattr(optimizer, "sm_sella_version", None)),
    }

    diagnostic = {
        "schema": "sella_internal_eigh_failure_v1",
        "classification": "diagnostic_only_no_eigensolver_behavior_change",
        "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
        "exception_message": str(exc),
        "failure_operation": "scipy.linalg.eigh(A)",
        "rfo_partition_side": side,
        "rfo_order": _json_scalar(getattr(rfo_self, "order", None)),
        "alpha": _json_scalar(alpha),
        "construction": (
            "RFO base=[[H_projected,g_projected],[g_projected.T,0]]; "
            "A_eigh=base*alpha with A_eigh[:-1,:-1] multiplied by alpha again"
        ),
        "A_eigh": _array_capture_summary(A),
        "rfo_base_A": _array_capture_summary(base_A) if base_A is not None else None,
        "projected_hessian": _array_capture_summary(arrays.get("projected_hessian")),
        "projected_gradient": _array_capture_summary(arrays.get("projected_gradient")),
        "restricted_step": restricted_context,
        "optimizer": optimizer_context,
        "attempt_metadata": {str(k): _json_scalar(v) for k, v in meta.items()},
        "optional_arrays_captured": optional_arrays,
        "optional_array_limit_bytes_each": _INTERNAL_EIGH_OPTIONAL_ARRAY_LIMIT_BYTES,
        "npz_arrays": {name: _array_capture_summary(value) for name, value in arrays.items()},
        "runtime": {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": scipy_version,
            "thread_env": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "BLIS_NUM_THREADS", "MKL_CBWR",
                )
            },
        },
        "traceback_frames": [
            {
                "file": str(frame.f_code.co_filename),
                "function": frame.f_code.co_name,
                "line": int(lineno),
            }
            for frame, lineno in frames
        ],
        "behavioral_note": (
            "The original exception is re-raised unchanged. No retry, driver fallback, "
            "matrix symmetrization, regularization, or additional PES call is performed."
        ),
    }

    np.savez_compressed(npz_path, **arrays)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(diagnostic, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return [json_path, npz_path]



def _array_sha256(value: Any) -> str | None:
    """Stable diagnostic hash for a numeric array without changing its values."""
    if value is None:
        return None
    arr = np.ascontiguousarray(np.asarray(value))
    h = sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(repr(tuple(int(x) for x in arr.shape)).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _canonicalize_direction(vector: np.ndarray, *, tol: float = 1.0e-14) -> np.ndarray:
    """Normalize a direction and choose a deterministic sign."""
    vec = np.asarray(vector, dtype=float).reshape(-1).copy()
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= tol:
        raise SellaAblationError("degenerate_partition_direction", "PRFO partition direction is zero/non-finite")
    vec /= norm
    nz = np.flatnonzero(np.abs(vec) > tol)
    if nz.size and vec[int(nz[0])] < 0.0:
        vec *= -1.0
    return vec


def _deterministic_ritz_partition_basis(
    ritz_direction: np.ndarray,
    native_b_evecs: np.ndarray,
    *,
    tol: float = 1.0e-12,
) -> np.ndarray:
    """Return [P|Q] with P=Ritz and deterministic orthonormal Q.

    Native approximate-B eigenvectors are consumed in their native ascending order
    as the first complement candidates.  This preserves as much of the baseline P-RFO Q basis
    as possible while making the sole selected unstable direction the Ritz vector.
    Standard Cartesian basis vectors are deterministic fallbacks for a candidate
    made linearly dependent by replacing P.
    """
    native = np.asarray(native_b_evecs, dtype=float)
    if native.ndim != 2 or native.shape[0] != native.shape[1]:
        raise SellaAblationError("invalid_native_b_evecs", "Native B eigenvectors must be a square matrix")
    n = native.shape[0]
    p = _canonicalize_direction(ritz_direction)
    if p.size != n:
        raise SellaAblationError(
            "partition_dimension_mismatch",
            f"Projected Ritz direction has dimension {p.size}; PRFO active dimension is {n}",
        )

    basis = [p]

    def admit(candidate: np.ndarray) -> None:
        if len(basis) >= n:
            return
        q = np.asarray(candidate, dtype=float).reshape(-1).copy()
        for b in basis:
            q -= b * float(np.dot(b, q))
        qnorm = float(np.linalg.norm(q))
        if not np.isfinite(qnorm) or qnorm <= tol:
            return
        q /= qnorm
        # Re-orthogonalize once for numerical determinism/stability.
        for b in basis:
            q -= b * float(np.dot(b, q))
        qnorm = float(np.linalg.norm(q))
        if not np.isfinite(qnorm) or qnorm <= tol:
            return
        q /= qnorm
        q = _canonicalize_direction(q)
        basis.append(q)

    for j in range(n):
        admit(native[:, j])
    if len(basis) < n:
        eye = np.eye(n)
        for j in range(n):
            admit(eye[:, j])
            if len(basis) == n:
                break
    if len(basis) != n:
        raise SellaAblationError("partition_complement_failure", "Could not build a complete Ritz PRFO partition basis")

    V = np.column_stack(basis)
    gram = V.T @ V
    if not np.allclose(gram, np.eye(n), rtol=0.0, atol=5.0e-11):
        raise SellaAblationError("partition_not_orthonormal", "Ritz PRFO partition basis is not orthonormal")
    return V


class _PartitionBasisHessianProxy:
    """Proxy one projected Sella ApproximateHessian without changing B.

    The native PRFO constructor reads ``H.evecs`` only to define Vmax/Vmin and
    subsequently calls ``H.project(...)`` for the actual subproblem Hessians.  This
    proxy delegates every operation to the original projected Hessian except the
    returned eigenvector basis.  Accessing ``evecs`` first asks the original object
    for its eigenvectors, so Ritz-partition ablation performs the same native B eigendecomposition at
    the same P-RFO construction site as the baseline mapping.
    """

    def __init__(self, base: Any, projected_ritz: np.ndarray, on_basis=None):
        self._base = base
        self._projected_ritz = np.asarray(projected_ritz, dtype=float).reshape(-1).copy()
        self._basis = None
        self._on_basis = on_basis

    @property
    def evecs(self):
        if self._basis is None:
            native = self._base.evecs
            if native is None:
                raise SellaAblationError("native_b_evecs_unavailable", "Native Sella projected B has no eigenvectors for PRFO")
            native = np.asarray(native, dtype=float)
            # Reading evals after evecs reuses the same lazy decomposition.
            evals = None if self._base.evals is None else np.asarray(self._base.evals, dtype=float).copy()
            self._basis = _deterministic_ritz_partition_basis(self._projected_ritz, native)
            if self._on_basis is not None:
                self._on_basis(evals, native.copy(), self._basis.copy())
        return self._basis

    @evecs.setter
    def evecs(self, value):
        # Sella's PRFO path does not assign this property. Fail closed instead of
        # accidentally changing the native projected Hessian object.
        raise SellaAblationError("partition_proxy_evecs_assignment", "Unexpected assignment to Ritz-partition ablation partition proxy evecs")

    @property
    def evals(self):
        return self._base.evals

    def project(self, U):
        return self._base.project(U)

    def asarray(self):
        return self._base.asarray()

    def __getattr__(self, name):
        return getattr(self._base, name)


class _InstancePatchSet:
    """Restore instance attributes exactly, including descriptor fallback."""

    def __init__(self):
        self._entries: list[tuple[object, str, bool, Any]] = []

    def method(self, obj: object, name: str, func) -> None:
        dct = getattr(obj, "__dict__", {})
        existed = name in dct
        previous = dct.get(name)
        self._entries.append((obj, name, existed, previous))
        setattr(obj, name, MethodType(func, obj))

    def restore(self) -> None:
        while self._entries:
            obj, name, existed, previous = self._entries.pop()
            if existed:
                setattr(obj, name, previous)
            else:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass


class SellaAblationSession:
    """Install one attempt-local Sella ablation and collect exact call metadata.

    The session adds no PES evaluations.  All wrappers call the original method once
    and derive diagnostics from live arguments/cached values.  ``close()`` is
    idempotent and always restores every instance hook.
    """

    def __init__(
        self,
        optimizer: object,
        spec: SellaAblationSpec | str | Iterable[str] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        restored_state: Mapping[str, Any] | None = None,
    ):
        self.optimizer = optimizer
        self.spec = spec if isinstance(spec, SellaAblationSpec) else resolve_sella_ablation(spec)
        self.metadata = dict(metadata or {})
        self.rows: list[dict[str, Any]] = []
        self.diag_events: list[dict[str, Any]] = []
        self.update_events: list[dict[str, Any]] = []
        self.partition_events: list[dict[str, Any]] = []
        self.restricted_step_events: list[dict[str, Any]] = []
        self._patches = _InstancePatchSet()
        self._h_update_patches = _InstancePatchSet()
        self._installed = False
        self._closed = False
        self._in_diag = 0
        self._probe_evaluations = 0
        self._pending_predict: dict[str, Any] | None = None
        self._pending_kick: dict[str, Any] | None = None
        self._restricted_step_capture_path: str | None = None
        self._restricted_step_capture_npz: str | None = None
        self._restricted_step_capture_error: str | None = None
        self._step_serial = 0
        self._center_update_calls = 0
        self._center_updates_admitted = 0
        self._probe_update_calls = 0
        self._probe_updates_admitted = 0
        self._probe_vectors_seen = 0
        self._center_only_initial_model: dict[str, Any] | None = None
        self._latest_native_ritz: dict[str, Any] | None = None
        self.failure_artifacts: list[str] = []
        self.failure_capture_errors: list[str] = []
        self._restored_state = dict(restored_state or {})
        self._validate_restore_state()

    def _validate_restore_state(self) -> None:
        if not self._restored_state:
            return
        if self._restored_state.get("schema") != SCHEMA_VERSION:
            raise SellaAblationError("state_schema_mismatch", "Unsupported Sella ablation state schema")
        if self._restored_state.get("selector_identity") != self.spec.identity():
            raise SellaAblationError("state_selector_mismatch", "Resume state belongs to a different Sella ablation selector")
        counters = self._restored_state.get("diagnostic_counters", {}) or {}
        self._step_serial = int(counters.get("step_serial", 0))
        self._center_update_calls = int(counters.get("center_update_calls", 0))
        self._center_updates_admitted = int(counters.get("center_updates_admitted", 0))
        self._probe_update_calls = int(counters.get("probe_update_calls", 0))
        self._probe_updates_admitted = int(counters.get("probe_updates_admitted", 0))
        self._probe_vectors_seen = int(counters.get("probe_vectors_seen", 0))
        self._probe_evaluations = int(counters.get("probe_evaluations", 0))

    def capture_internal_eigh_failure(self, exc: BaseException) -> None:
        """Best-effort failure capture that must never replace the native exception."""
        try:
            paths = _capture_internal_eigh_failure(
                exc,
                self.optimizer,
                self.metadata,
                self._step_serial + 1,
            )
        except Exception as capture_exc:
            self.failure_capture_errors.append(
                f"{type(capture_exc).__name__}: {capture_exc}"
            )
            return
        for path in paths:
            if path not in self.failure_artifacts:
                self.failure_artifacts.append(path)

    def resolved_metadata(self) -> dict[str, Any]:
        return {
            **self.spec.component_matrix(),
            "selector_identity": self.spec.identity(),
            "supported_sella_version": SUPPORTED_SELLA_VERSION,
            "supplied_reference_kind": "local_modified_sella_2.5.0_snapshot",
            "reference_sha256": dict(REFERENCE_SHA256),
            "diagnostic_additional_pes_calls": 0,
            **self.metadata,
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialize diagnostic continuity only; Sella owns algorithm state."""
        return {
            "schema": SCHEMA_VERSION,
            "selector_identity": self.spec.identity(),
            "selector": self.spec.selector,
            "scientific_components": self.spec.component_matrix(),
            "algorithm_state_owned_by": "sella_optimizer",
            "diagnostic_counters": {
                "step_serial": self._step_serial,
                "center_update_calls": self._center_update_calls,
                "center_updates_admitted": self._center_updates_admitted,
                "probe_update_calls": self._probe_update_calls,
                "probe_updates_admitted": self._probe_updates_admitted,
                "probe_vectors_seen": self._probe_vectors_seen,
                "probe_evaluations": self._probe_evaluations,
            },
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self.resolved_metadata(),
            "steps_recorded": len(self.rows),
            "center_update_calls": self._center_update_calls,
            "center_updates_admitted": self._center_updates_admitted,
            "probe_update_calls": self._probe_update_calls,
            "probe_updates_admitted": self._probe_updates_admitted,
            "probe_vectors_seen": self._probe_vectors_seen,
            "eigensolver_probe_evaluations": self._probe_evaluations,
            "raw_prfo_step_norm": None,
            "raw_prfo_step_unavailable_reason": (
                "pinned Sella restricted-step API returns only the restricted step; "
                "reconstructing raw PRFO would recompute PRFO"
            ),
            "ritz_subspace_size": (
                None if self._latest_native_ritz is None
                else self._latest_native_ritz.get("subspace_size")
            ),
            "ritz_subspace_size_unavailable_reason": (
                None if self._latest_native_ritz is not None
                else "No native PES.diag Ritz block was observed by this session"
            ),
            "prfo_partition_events": len(self.partition_events),
            "restricted_step_policy_events": len(self.restricted_step_events),
            "latest_restricted_step_policy": (
                None if not self.restricted_step_events else dict(self.restricted_step_events[-1])
            ),
            "latest_native_ritz_eigenvalue": (
                None if self._latest_native_ritz is None
                else self._latest_native_ritz.get("ritz_eigenvalue")
            ),
            "hvp_action_count": None,
            "hvp_action_count_unavailable_reason": (
                "NumericalHessian.calls is local to PES.diag; exact physical probe evaluations are counted instead"
            ),
            "diagnostic_additional_pes_calls": 0,
            "center_only_initial_model_runtime": (
                dict(self._center_only_initial_model)
                if self._center_only_initial_model is not None else None
            ),
        }

    def _validate_optimizer(self) -> None:
        opt = self.optimizer
        version = getattr(opt, "sm_sella_version", None)
        if version != SUPPORTED_SELLA_VERSION:
            raise SellaAblationError(
                "unsupported_sella_version",
                f"Sella-ablation requires the pinned locally modified Sella {SUPPORTED_SELLA_VERSION}; got {version!r}",
            )
        pes = getattr(opt, "pes", None)
        if pes is None:
            raise SellaAblationError("missing_pes", "Sella optimizer has no PES object")
        if getattr(pes, "int", None) is not None:
            raise SellaAblationError(
                "internal_coordinates_unsupported",
                "Sella-ablation hooks support the current SaddleMill Cartesian Sella path; internal-coordinate PES replacement is not isolated",
            )
        required_opt_methods = ("_predict_step", "step")
        required_opt_attrs = ("delta", "method", "ord", "eig")
        required_pes_methods = ("diag", "_update_H", "_calc_eg", "kick", "get_x")
        required_pes_attrs = ("H", "curr")
        for name in required_opt_methods:
            if not callable(getattr(opt, name, None)):
                raise SellaAblationError("unsupported_optimizer_api", f"Pinned Sella optimizer method missing/not callable: {name}")
        for name in required_opt_attrs:
            if not hasattr(opt, name):
                raise SellaAblationError("unsupported_optimizer_api", f"Pinned Sella optimizer attribute missing: {name}")
        for name in required_pes_methods:
            if not callable(getattr(pes, name, None)):
                raise SellaAblationError("unsupported_pes_api", f"Pinned Sella PES method missing/not callable: {name}")
        for name in required_pes_attrs:
            if not hasattr(pes, name):
                raise SellaAblationError("unsupported_pes_api", f"Pinned Sella PES attribute missing: {name}")
        if any(
            not callable(getattr(pes.H, name, None))
            for name in ("update", "asarray")
        ):
            raise SellaAblationError(
                "unsupported_hessian_api",
                "Pinned Sella ApproximateHessian update/asarray API is unavailable",
            )
        if (
            self.spec.learning == LEARNING_CENTER_ONLY
            and not callable(getattr(pes.H, "set_B", None))
        ):
            raise SellaAblationError(
                "unsupported_center_only_hessian_api",
                "center_only_learning requires pinned ApproximateHessian.set_B API",
            )
        if self.spec.partition == PARTITION_NATIVE_RITZ:
            for name in ("get_Hc", "get_HL_projected", "get_Ufree"):
                if not callable(getattr(pes, name, None)):
                    raise SellaAblationError(
                        "unsupported_ritz_partition_api",
                        f"ritzmode_prfo_partition requires pinned PES.{name}()",
                    )
            if getattr(pes, "hessian_function", None) is not None:
                raise SellaAblationError(
                    "ritz_partition_requires_native_diag",
                    "ritzmode_prfo_partition requires hessian_engine=sella/native PES.diag; direct Hessian mode has no native Ritz solve",
                )
        if int(getattr(opt, "ord")) != 1 or not bool(getattr(opt, "eig")):
            raise SellaAblationError("not_first_order_olsen", "Sella ablations require Sella order=1 with eig=True")
        method = str(getattr(opt, "method")).lower()
        if self.spec.qn_newton_safe is False:
            if method != "qn":
                raise SellaAblationError(
                    "non_qn_baseline",
                    "qn_newton_safe_false requires native Sella method='qn'",
                )
        elif not self.spec.is_normal and method != "prfo":
            raise SellaAblationError(
                "non_prfo_baseline",
                "Controlled PRFO Sella ablations require the baseline Sella translation method='prfo'",
            )
        if self.spec.learning == LEARNING_CENTER_ONLY and getattr(pes, "hessian_function", None) is not None:
            raise SellaAblationError(
                "probe_learning_not_applicable_direct_hessian",
                "center_only_learning is undefined when Sella uses an external direct Hessian instead of PES.diag probe learning",
            )
        if int(getattr(opt, "nsteps", 0) or 0) == 0:
            delta = _finite_float(getattr(opt, "delta", None))
            if delta is None or not math.isclose(delta, EXPECTED_DELTA0, rel_tol=0.0, abs_tol=1e-15):
                raise SellaAblationError(
                    "delta0_mismatch",
                    f"Sella-ablation requires unchanged Sella delta0={EXPECTED_DELTA0}; initial delta is {delta!r}",
                )

    def _materialize_center_only_initial_model(self) -> None:
        """Materialize Sella's implicit identity fallback without probe learning.

        ``ApproximateHessian.asarray()`` returns an identity matrix while ``B`` is
        ``None``, but PRFO indexes ``ApproximateHessian.evecs`` directly and the
        lazy eigendecomposition intentionally does nothing while ``B`` is ``None``.
        The normal Sella path avoids that state because the first ``PES.diag`` probe
        block initializes ``B``.  Center-only learning suppresses exactly that
        update, so it must make the already-implied fallback concrete before the
        first PRFO construction.

        Passing a *scalar* to the pinned ``ApproximateHessian.set_B`` is important:
        Sella expands it to ``scale * I`` without setting ``initialized=True``.
        Therefore the first accepted center-to-center secant still takes Sella's
        native ``not self.initialized`` branch and replaces the bootstrap with a
        center-derived initial model.  No eigensolver probe displacement/gradient
        pair is admitted into the approximate Hessian.
        """
        if self.spec.learning != LEARNING_CENTER_ONLY:
            return

        hessian = self.optimizer.pes.H
        if getattr(hessian, "B", None) is not None or bool(
            getattr(hessian, "initialized", False)
        ):
            raise SellaAblationError(
                "center_only_initial_model_not_pristine",
                "center_only_learning requires a pristine Sella ApproximateHessian "
                "before its probe-independent identity bootstrap",
            )

        fallback = np.asarray(hessian.asarray(), dtype=float)
        dim = int(getattr(hessian, "dim", fallback.shape[0]))
        expected_shape = (dim, dim)
        if fallback.shape != expected_shape:
            raise SellaAblationError(
                "center_only_initial_model_shape",
                f"Sella implicit Hessian fallback has shape {fallback.shape}; "
                f"expected {expected_shape}",
            )
        if not np.all(np.isfinite(fallback)):
            raise SellaAblationError(
                "center_only_initial_model_nonfinite",
                "Sella implicit Hessian fallback contains non-finite values",
            )
        identity = CENTER_ONLY_INITIAL_SCALE * np.eye(dim)
        if not np.array_equal(fallback, identity):
            raise SellaAblationError(
                "center_only_initial_model_contract_changed",
                "Pinned Sella no longer exposes the expected identity fallback "
                "for an uninitialized ApproximateHessian",
            )

        hessian.set_B(CENTER_ONLY_INITIAL_SCALE)
        materialized = np.asarray(getattr(hessian, "B", None), dtype=float)
        if materialized.shape != expected_shape or not np.all(np.isfinite(materialized)):
            raise SellaAblationError(
                "center_only_initial_model_materialization_failed",
                "Failed to materialize a finite center-only initial Hessian model",
            )
        if not np.array_equal(materialized, identity):
            raise SellaAblationError(
                "center_only_initial_model_materialization_changed",
                "Materialized center-only Hessian differs from Sella's implicit identity fallback",
            )
        if bool(getattr(hessian, "initialized", False)):
            raise SellaAblationError(
                "center_only_initial_model_consumed_learning_slot",
                "Identity bootstrap unexpectedly marked Sella Hessian initialized; "
                "the first center secant would no longer retain native semantics",
            )

        self._center_only_initial_model = {
            "kind": CENTER_ONLY_INITIAL_MODEL,
            "scale": CENTER_ONLY_INITIAL_SCALE,
            "dimension": dim,
            "probe_independent": True,
            "materialized_from": "ApproximateHessian.asarray_none_fallback",
            "initialized_flag_after_materialization": False,
            "prfo_target_order": int(getattr(self.optimizer, "ord")),
            "prfo_first_mode_index": 0,
        }

    def __enter__(self) -> "SellaAblationSession":
        return self.install()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def install(self) -> "SellaAblationSession":
        if self._installed:
            return self
        if self._closed:
            raise SellaAblationError("session_closed", "Cannot reinstall a closed Sella ablation session")
        self._validate_optimizer()
        self._materialize_center_only_initial_model()
        opt = self.optimizer
        pes = opt.pes
        self._optimizer_pes = pes

        orig_predict = getattr(opt, "_predict_step")
        orig_step = getattr(opt, "step")
        orig_diag = getattr(pes, "diag")
        orig_update_h = getattr(pes, "_update_H")
        orig_calc_eg = getattr(pes, "_calc_eg")
        orig_kick = getattr(pes, "kick")
        orig_get_hl_projected = getattr(pes, "get_HL_projected", None)

        session = self

        def wrapped_calc_eg(pes_self, *args, **kwargs):
            if session._in_diag:
                session._probe_evaluations += 1
            return orig_calc_eg(*args, **kwargs)

        def wrapped_update_h(pes_self, dx, dg):
            session._center_update_calls += 1
            last = getattr(pes_self, "last", {}) or {}
            eligible = last.get("x") is not None and last.get("g") is not None
            event = {
                "source": "accepted_center_secant",
                "call_index": session._center_update_calls,
                "s_norm": _norm(dx),
                "y_norm": _norm(dg),
                "eligible": bool(eligible),
                "admitted": False,
            }
            try:
                result = orig_update_h(dx, dg)
                event["admitted"] = bool(eligible)
                if eligible:
                    session._center_updates_admitted += 1
                return result
            finally:
                session.update_events.append(event)

        def wrapped_diag(pes_self, *args, **kwargs):
            started = perf_counter_ns()
            before_neval = int(getattr(pes_self, "neval", 0) or 0)
            before_probe_eval = session._probe_evaluations
            session._in_diag += 1
            hessian = pes_self.H
            orig_h_update = getattr(hessian, "update")
            call_count = 0
            vector_count = 0
            captured_hc = {"value": None}
            get_hc = getattr(pes_self, "get_Hc", None)

            def wrapped_get_hc(pes_inner, *gh_args, **gh_kwargs):
                value = get_hc(*gh_args, **gh_kwargs)
                try:
                    captured_hc["value"] = np.asarray(value, dtype=float).copy()
                except Exception:
                    captured_hc["value"] = None
                return value

            def wrapped_h_update(h_self, dx, dg):
                nonlocal call_count, vector_count
                call_count += 1
                arr = np.asarray(dx)
                nvec = int(arr.shape[1]) if arr.ndim == 2 else 1
                vector_count += nvec
                session._probe_update_calls += 1
                session._probe_vectors_seen += nvec
                admitted = session.spec.learning != LEARNING_CENTER_ONLY
                event = {
                    "source": "same_center_eigensolver_block",
                    "call_index": session._probe_update_calls,
                    "vectors": nvec,
                    "s_norm": _norm(dx),
                    "y_norm": _norm(dg),
                    "admitted": bool(admitted),
                }

                # Native Sella 2.5.0 rotates NumericalHessian Vs/AVs into the
                # Rayleigh--Ritz basis immediately before this exact update call.
                # Capture the first (lowest) vector passively; do not call the
                # eigensolver, HVP machinery, or Hessian update a second time.
                try:
                    vs = np.asarray(dx, dtype=float)
                    avs = np.asarray(dg, dtype=float)
                    if vs.ndim == 1:
                        vs = vs[:, None]
                    if avs.ndim == 1:
                        avs = avs[:, None]
                    if vs.ndim == 2 and avs.shape == vs.shape and vs.shape[1] >= 1:
                        raw_ritz = np.asarray(vs[:, 0], dtype=float).copy()
                        raw_norm = float(np.linalg.norm(raw_ritz))
                        if not np.isfinite(raw_norm) or raw_norm <= 1.0e-14:
                            raise SellaAblationError(
                                "degenerate_native_ritz",
                                "Native Sella returned a zero/non-finite lowest Ritz vector",
                            )
                        ritz = raw_ritz / raw_norm
                        ritz_action = np.asarray(avs[:, 0], dtype=float).copy() / raw_norm
                        nz = np.flatnonzero(np.abs(ritz) > 1.0e-14)
                        if nz.size and ritz[int(nz[0])] < 0.0:
                            ritz *= -1.0
                            ritz_action *= -1.0
                        hc = captured_hc.get("value")
                        ritz_eval = None
                        if hc is not None and hc.shape == (ritz.size, ritz.size):
                            # AVs are the native physical Hessian actions.  Sella's
                            # Ritz operator is H - Hc, so this scalar is the already-
                            # solved Ritz Rayleigh quotient without another solve.
                            ritz_eval = float(ritz @ ritz_action - ritz @ (hc @ ritz))
                        curr = getattr(pes_self, "curr", {}) or {}
                        session._latest_native_ritz = {
                            "diag_call_index": len(session.diag_events) + 1,
                            "mode": ritz.copy(),
                            "mode_sha256": _array_sha256(ritz),
                            "ritz_eigenvalue": ritz_eval,
                            "subspace_size": int(vs.shape[1]),
                            "geometry_sha256": _array_sha256(pes_self.get_x()),
                            "gradient_sha256": _array_sha256(curr.get("g")),
                            "pes_evaluations_at_capture": int(getattr(pes_self, "neval", 0) or 0),
                        }
                        event["lowest_ritz_mode_sha256"] = session._latest_native_ritz["mode_sha256"]
                        event["lowest_ritz_eigenvalue"] = ritz_eval
                except SellaAblationError:
                    raise
                except Exception as capture_exc:
                    event["ritz_capture_error"] = f"{type(capture_exc).__name__}: {capture_exc}"

                try:
                    if admitted:
                        result = orig_h_update(dx, dg)
                        session._probe_updates_admitted += 1
                        return result
                    return None
                finally:
                    session.update_events.append(event)

            local_patch = _InstancePatchSet()
            local_patch.method(hessian, "update", wrapped_h_update)
            if callable(get_hc):
                local_patch.method(pes_self, "get_Hc", wrapped_get_hc)
            error = None
            try:
                return orig_diag(*args, **kwargs)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                local_patch.restore()
                session._in_diag -= 1
                latest = session._latest_native_ritz or {}
                session.diag_events.append({
                    "diag_call_index": len(session.diag_events) + 1,
                    "probe_update_calls": call_count,
                    "probe_vectors": vector_count,
                    "probe_evaluations": session._probe_evaluations - before_probe_eval,
                    "pes_evaluations": int(getattr(pes_self, "neval", 0) or 0) - before_neval,
                    "probe_learning_admitted": session.spec.learning != LEARNING_CENTER_ONLY,
                    "lowest_ritz_mode_sha256": latest.get("mode_sha256"),
                    "lowest_ritz_eigenvalue": latest.get("ritz_eigenvalue"),
                    "ritz_subspace_size": latest.get("subspace_size"),
                    "elapsed_ns": perf_counter_ns() - started,
                    "error": error,
                })

        def wrapped_predict(opt_self, *args, **kwargs):
            started = perf_counter_ns()
            original_method = opt_self.method
            original_rs = getattr(opt_self, "rs", None)
            rs_capture = None
            rs_overridden = False
            partition_patch = _InstancePatchSet()
            capture_dir = os.environ.get(RESTRICTED_STEP_CAPTURE_ENV, "").strip()

            if session.spec.qn_newton_safe is False:
                def record_qn_newton_safe(rs, before):
                    stepper = getattr(rs, "stepper", None)
                    session.restricted_step_events.append({
                        "policy": QN_NEWTON_SAFE_FALSE,
                        "restricted_step_class": type(rs).__name__,
                        "native_base_class": getattr(original_rs, "__name__", type(original_rs).__name__),
                        "stepper_class": type(stepper).__name__ if stepper is not None else None,
                        "newton_safe_before": bool(before),
                        "newton_safe_effective": bool(getattr(stepper, "newton_safe", True)),
                        "step_serial": session._step_serial + 1,
                        "additional_pes_calls": 0,
                    })
                opt_self.rs = _qn_newton_safe_false_restricted_step_class(
                    original_rs, record_qn_newton_safe
                )
                rs_overridden = True

            if session.spec.partition == PARTITION_NATIVE_RITZ:
                if session._latest_native_ritz is None:
                    # On a fresh native Sella optimizer, _predict_step itself calls
                    # PES.diag before constructing the restricted step, so this is
                    # normally populated inside orig_predict.  The get_HL wrapper
                    # below checks again at the actual PRFO construction point.
                    pass
                if not callable(orig_get_hl_projected):
                    raise SellaAblationError("unsupported_ritz_partition_api", "PES.get_HL_projected is unavailable")

                def ritz_get_hl_projected(pes_inner, U):
                    base = orig_get_hl_projected(U)
                    # Native PES.diag itself projects the current approximate
                    # Hessian before the Ritz solve.  That projection must remain
                    # untouched; the override applies only later when the native
                    # restricted-step/PRFO constructor asks for its projected H.
                    if session._in_diag:
                        return base
                    latest = session._latest_native_ritz
                    if latest is None:
                        raise SellaAblationError(
                            "ritz_mode_unavailable",
                            "Native PES.diag did not expose a Ritz direction before Ritz-partition ablation PRFO construction",
                        )
                    Uarr = np.asarray(U, dtype=float)
                    if Uarr.ndim != 2 or Uarr.shape[0] != latest["mode"].size:
                        raise SellaAblationError(
                            "ritz_projection_shape",
                            f"PES projected-Hessian basis shape {Uarr.shape} is incompatible with captured Ritz dimension {latest['mode'].size}",
                        )
                    gram = Uarr.T @ Uarr
                    if not np.allclose(gram, np.eye(gram.shape[0]), rtol=0.0, atol=5.0e-10):
                        raise SellaAblationError(
                            "nonorthonormal_active_basis",
                            "Ritz-partition ablation supports Sella's native Cartesian orthonormal active basis only; weighted/nonorthonormal projection would change more than the partition direction",
                        )
                    projected = Uarr.T @ latest["mode"]
                    projected = _canonicalize_direction(projected)

                    curr = getattr(pes_inner, "curr", {}) or {}
                    B = getattr(base, "B", None)
                    event = {
                        "partition_source": PARTITION_NATIVE_RITZ,
                        "geometry_sha256": _array_sha256(pes_inner.get_x()),
                        "gradient_sha256": _array_sha256(curr.get("g")),
                        "approximate_B_sha256": _array_sha256(B),
                        "approximate_B_initialized": bool(getattr(base, "initialized", False)),
                        "center_update_calls_total": session._center_update_calls,
                        "center_updates_admitted_total": session._center_updates_admitted,
                        "probe_update_calls_total": session._probe_update_calls,
                        "probe_updates_admitted_total": session._probe_updates_admitted,
                        "native_ritz_mode_sha256": latest.get("mode_sha256"),
                        "native_ritz_eigenvalue": latest.get("ritz_eigenvalue"),
                        "native_ritz_diag_call_index": latest.get("diag_call_index"),
                        "native_ritz_geometry_sha256": latest.get("geometry_sha256"),
                        "native_ritz_gradient_sha256": latest.get("gradient_sha256"),
                        "native_ritz_same_geometry": latest.get("geometry_sha256") == _array_sha256(pes_inner.get_x()),
                        "trust_radius_entering_step": _finite_float(getattr(opt_self, "delta", None)),
                        "pes_evaluations_entering_partition": int(getattr(pes_inner, "neval", 0) or 0),
                        "projected_ritz_mode_sha256": _array_sha256(projected),
                        "active_dimension": int(projected.size),
                    }

                    def on_basis(evals, native_evecs, override_evecs):
                        event.update({
                            "b_eigenvalues": None if evals is None else np.asarray(evals, dtype=float).tolist(),
                            "b_eigenvalues_sha256": _array_sha256(evals),
                            "native_b_unstable_mode_sha256": _array_sha256(native_evecs[:, 0]),
                            "selected_p_mode_sha256": _array_sha256(override_evecs[:, 0]),
                            "selected_p_matches_projected_ritz_absdot": float(abs(np.dot(override_evecs[:, 0], projected))),
                            "selected_p_matches_native_b_absdot": float(abs(np.dot(override_evecs[:, 0], native_evecs[:, 0]))),
                            "q_orthonormal_error": float(np.max(np.abs(override_evecs.T @ override_evecs - np.eye(override_evecs.shape[1])))),
                        })
                        session.partition_events.append(event)

                    return _PartitionBasisHessianProxy(base, projected, on_basis=on_basis)

                partition_patch.method(pes_self := opt_self.pes, "get_HL_projected", ritz_get_hl_projected)

            if session.spec.translation == TRANSLATION_SIMPLE_EVF:
                opt_self.method = "qn"
                if capture_dir:
                    rs_capture = _RestrictedStepFailureCapture(
                        capture_dir,
                        {
                            **session.resolved_metadata(),
                            "step_serial": session._step_serial + 1,
                            "optimizer_method_baseline": str(original_method),
                            "optimizer_method_effective": "qn",
                            "restricted_step_capture_env": RESTRICTED_STEP_CAPTURE_ENV,
                            "restricted_step_base_class": getattr(opt_self.rs, "__name__", type(opt_self.rs).__name__),
                            "center_update_calls_total": session._center_update_calls,
                            "center_updates_admitted_total": session._center_updates_admitted,
                            "probe_update_calls_total": session._probe_update_calls,
                            "probe_updates_admitted_total": session._probe_updates_admitted,
                            "probe_vectors_seen_total": session._probe_vectors_seen,
                            "eigensolver_probe_evaluations_total": session._probe_evaluations,
                            "diag_calls_total": len(session.diag_events),
                        },
                    )
                    opt_self.rs = _capturing_restricted_step_class(opt_self.rs, rs_capture)
                    rs_overridden = True
            try:
                result = orig_predict(*args, **kwargs)
            except Exception as exc:
                if (
                    rs_capture is not None
                    and isinstance(exc, RuntimeError)
                    and str(exc) == "Restricted step failed to converge!"
                ):
                    try:
                        json_path, npz_path = rs_capture.write_failure(exc)
                        session._restricted_step_capture_path = json_path
                        session._restricted_step_capture_npz = npz_path
                    except Exception as capture_exc:
                        # Never mask the native restricted-step failure with a
                        # diagnostic I/O/serialization error.
                        session._restricted_step_capture_error = (
                            f"{type(capture_exc).__name__}: {capture_exc}"
                        )
                session.capture_internal_eigh_failure(exc)
                raise
            finally:
                partition_patch.restore()
                opt_self.method = original_method
                if rs_overridden:
                    opt_self.rs = original_rs
            s, smag = result
            session._pending_predict = {
                "translation_formula": (
                    TRANSLATION_SIMPLE_EVF
                    if session.spec.translation == TRANSLATION_SIMPLE_EVF
                    else TRANSLATION_PRFO
                ),
                "restricted_step_norm": _norm(s),
                "restricted_step_max_cartesian": _max_cartesian_norm(s),
                "restricted_step_constraint_value": _finite_float(smag),
                "raw_proposed_step_norm": None,
                "raw_proposed_step_unavailable_reason": (
                    "pinned Sella restricted-step API exposes the restricted step only; "
                    "no second PRFO/QN construction is performed for diagnostics"
                ),
                "predict_elapsed_ns": perf_counter_ns() - started,
                "adapter_prfo_recomputations": 0,
                "prfo_partition_source": (
                    PARTITION_NATIVE_RITZ
                    if session.spec.partition == PARTITION_NATIVE_RITZ
                    else PARTITION_APPROXIMATE_B
                ),
                "latest_native_ritz_mode_sha256": (
                    None if session._latest_native_ritz is None
                    else session._latest_native_ritz.get("mode_sha256")
                ),
                "latest_native_ritz_eigenvalue": (
                    None if session._latest_native_ritz is None
                    else session._latest_native_ritz.get("ritz_eigenvalue")
                ),
                "qn_newton_safe_policy": (
                    QN_NEWTON_SAFE_FALSE if session.spec.qn_newton_safe is False else "native"
                ),
                "qn_newton_safe_effective": (
                    None if not session.restricted_step_events
                    else session.restricted_step_events[-1].get("newton_safe_effective")
                ),
            }
            return result

        def wrapped_kick(pes_self, dx, diag=False, **diag_kwargs):
            started = perf_counter_ns()
            before_neval = int(getattr(pes_self, "neval", 0) or 0)
            x0 = np.asarray(pes_self.get_x(), dtype=float).copy()
            curr0 = dict(getattr(pes_self, "curr", {}) or {})
            f0 = _finite_float(curr0.get("f"))
            g0 = curr0.get("g")
            if g0 is not None:
                try:
                    g0 = np.asarray(g0, dtype=float).copy()
                except Exception:
                    g0 = None
            try:
                B0 = np.asarray(pes_self.H.asarray(), dtype=float).copy()
            except Exception:
                B0 = None

            set_x_original = getattr(pes_self, "set_x")
            first_set_x: dict[str, Any] = {}

            def capture_set_x(pes_inner, *sx_args, **sx_kwargs):
                result = set_x_original(*sx_args, **sx_kwargs)
                if not first_set_x:
                    first_set_x["result"] = result
                return result

            local_patch = _InstancePatchSet()
            local_patch.method(pes_self, "set_x", capture_set_x)
            try:
                ratio = orig_kick(dx, diag, **diag_kwargs)
            finally:
                local_patch.restore()

            curr1 = dict(getattr(pes_self, "curr", {}) or {})
            f1 = _finite_float(curr1.get("f"))
            df_actual = None if f0 is None or f1 is None else f1 - f0
            df_pred = None
            dx_initial = None
            captured = first_set_x.get("result")
            if isinstance(captured, tuple) and len(captured) >= 1:
                try:
                    dx_initial = np.asarray(captured[0], dtype=float)
                except Exception:
                    dx_initial = None
            if dx_initial is not None and g0 is not None and B0 is not None:
                try:
                    df_pred = float(g0.T @ dx_initial + 0.5 * dx_initial.T @ B0 @ dx_initial)
                except Exception:
                    df_pred = None
            reconstructed_ratio = None
            if df_pred is not None and abs(df_pred) >= 1e-14 and df_actual is not None:
                reconstructed_ratio = df_actual / df_pred

            session._pending_kick = {
                "kick_input_step_norm": _norm(dx),
                "accepted_kick_coordinate_norm": _norm(dx_initial),
                "predicted_energy_change": df_pred,
                "actual_energy_change": df_actual,
                "model_quality_ratio": _finite_float(ratio),
                "reconstructed_model_quality_ratio": _finite_float(reconstructed_ratio),
                "kick_diag_requested": bool(diag),
                "kick_pes_evaluations": int(getattr(pes_self, "neval", 0) or 0) - before_neval,
                "kick_elapsed_ns": perf_counter_ns() - started,
                "x0_norm": _norm(x0),
            }
            return ratio

        def wrapped_step(opt_self, *args, **kwargs):
            if opt_self.pes is not session._optimizer_pes:
                raise SellaAblationError("pes_identity_changed", "Sella PES object changed while Sella-ablation hooks were active")
            started = perf_counter_ns()
            pes_self = opt_self.pes
            x_before = np.asarray(pes_self.get_x(), dtype=float).copy()
            delta_before = _finite_float(getattr(opt_self, "delta", None))
            delta_cell_before = _finite_float(getattr(opt_self, "delta_cell", None))
            neval_before = int(getattr(pes_self, "neval", 0) or 0)
            center_before = session._center_update_calls
            center_admit_before = session._center_updates_admitted
            probe_before = session._probe_update_calls
            probe_admit_before = session._probe_updates_admitted
            probe_eval_before = session._probe_evaluations
            diag_before = len(session.diag_events)
            error = None
            native_delta_after = None
            native_delta_cell_after = None
            try:
                result = orig_step(*args, **kwargs)
                native_delta_after = _finite_float(getattr(opt_self, "delta", None))
                native_delta_cell_after = _finite_float(getattr(opt_self, "delta_cell", None))
                return result
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                native_delta_after = _finite_float(getattr(opt_self, "delta", None))
                native_delta_cell_after = _finite_float(getattr(opt_self, "delta_cell", None))
                raise
            finally:
                if session.spec.trust == TRUST_HOLD:
                    if delta_before is not None:
                        opt_self.delta = delta_before
                    if delta_cell_before is not None and hasattr(opt_self, "delta_cell"):
                        opt_self.delta_cell = delta_cell_before
                if opt_self.pes is not session._optimizer_pes:
                    # Fail closed after restoring any trust fields on the optimizer.
                    session.close()
                    raise SellaAblationError(
                        "pes_replaced_during_step",
                        "Pinned Sella replaced its PES during an ablated step; hook transfer is intentionally unsupported",
                    )
                x_after = np.asarray(pes_self.get_x(), dtype=float).copy()
                accepted = x_after - x_before
                session._step_serial += 1
                row = {
                    **session.resolved_metadata(),
                    "step_serial": session._step_serial,
                    "trust_radius_before": delta_before,
                    "trust_radius_after_native": native_delta_after,
                    "trust_radius_after_effective": _finite_float(getattr(opt_self, "delta", None)),
                    "trust_radius_cell_before": delta_cell_before,
                    "trust_radius_cell_after_native": native_delta_cell_after,
                    "trust_radius_cell_after_effective": _finite_float(getattr(opt_self, "delta_cell", None)),
                    "trust_update_held": int(session.spec.trust == TRUST_HOLD),
                    "accepted_displacement_norm": _norm(accepted),
                    "accepted_displacement_max_cartesian": _max_cartesian_norm(accepted),
                    "step_pes_evaluations": int(getattr(pes_self, "neval", 0) or 0) - neval_before,
                    "eigensolver_probe_evaluations": session._probe_evaluations - probe_eval_before,
                    "eigensolver_diag_calls": len(session.diag_events) - diag_before,
                    "center_update_calls": session._center_update_calls - center_before,
                    "center_updates_admitted": session._center_updates_admitted - center_admit_before,
                    "probe_update_calls": session._probe_update_calls - probe_before,
                    "probe_updates_admitted": session._probe_updates_admitted - probe_admit_before,
                    "adapter_additional_pes_calls": 0,
                    "adapter_duplicate_prfo_calls": 0,
                    "step_elapsed_ns": perf_counter_ns() - started,
                    "error": error,
                    "restricted_step_failure_capture": session._restricted_step_capture_path,
                    "restricted_step_failure_capture_npz": session._restricted_step_capture_npz,
                    "restricted_step_failure_capture_error": session._restricted_step_capture_error,
                }
                if session._pending_predict:
                    row.update(session._pending_predict)
                if session._pending_kick:
                    row.update(session._pending_kick)
                session.rows.append(row)
                session._pending_predict = None
                session._pending_kick = None

        self._patches.method(pes, "_calc_eg", wrapped_calc_eg)
        self._patches.method(pes, "_update_H", wrapped_update_h)
        self._patches.method(pes, "diag", wrapped_diag)
        self._patches.method(pes, "kick", wrapped_kick)
        self._patches.method(opt, "_predict_step", wrapped_predict)
        self._patches.method(opt, "step", wrapped_step)
        self._installed = True
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._h_update_patches.restore()
        self._patches.restore()
        self._installed = False
        self._closed = True


__all__ = [
    "CENTER_ONLY_INITIAL_MODEL",
    "CENTER_ONLY_INITIAL_SCALE",
    "CENTER_ONLY_LEARNING",
    "EXPECTED_DELTA0",
    "LEARNING_CENTER_ONLY",
    "LEARNING_NORMAL",
    "NORMAL",
    "PARTITION_APPROXIMATE_B",
    "PARTITION_NATIVE_RITZ",
    "QN_NEWTON_SAFE_FALSE",
    "REFERENCE_SHA256",
    "RITZMODE_PRFO_PARTITION",
    "SCHEMA_VERSION",
    "SIMPLE_TRANSLATION",
    "SUPPORTED_SELLA_VERSION",
    "SellaAblationError",
    "SellaAblationSession",
    "SellaAblationSpec",
    "TRANSLATION_PRFO",
    "TRANSLATION_SIMPLE_EVF",
    "TRUST_HOLD",
    "TRUST_ISOLATION",
    "TRUST_NATIVE",
    "eigenvector_following_reference_step",
    "resolve_sella_ablation",
    "_qn_newton_safe_false_restricted_step_class",
]
