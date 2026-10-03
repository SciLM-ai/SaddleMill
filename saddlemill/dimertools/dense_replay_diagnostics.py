"""Bounded dense replay of a frozen L-BFGS window for diagnostics only.

The functions in this module are deliberately calculator-free and admission-free.
They consume *already admitted* secants plus the exact H0 inverse scale used by an
actual L-BFGS step.  They never inspect raw force history, rebuild candidate pairs,
or apply a safeguard.  This makes the dense model a shadow of one specific
production L-BFGS window rather than an independently accumulated full-history
model.

The reconstructed matrix is the direct BFGS Hessian ``B`` whose inverse action is
algebraically equivalent to the ordinary L-BFGS two-loop recursion for the same
ordered pairs and H0 contract (subject to floating-point roundoff and a nonsingular
update sequence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Sequence

import numpy as np

from saddlemill.dimertools.dense_bfgs import safe_cosine, safe_max_atom_norm, safe_norm

Array = np.ndarray


DENSE_REPLAY_SCHEMA = "saddlemill_dense_replay_v1"


@dataclass(frozen=True)
class DenseReplayLimits:
    """Resource ceilings for one passive dense replay.

    ``max_matrix_bytes`` is compared with a conservative working-set estimate,
    not just the returned matrix.  ``max_work_units`` uses ``n_active ** 3`` as a
    simple eigensolve/linear-solve work proxy.  A non-positive limit disables only
    that particular ceiling; no ceiling is silently expanded.
    """

    max_dimension: int = 512
    max_matrix_bytes: int = 128 * 1024 * 1024
    max_work_units: int = 200_000_000

    def __post_init__(self) -> None:
        for name in ("max_dimension", "max_matrix_bytes", "max_work_units"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FrozenReplayWindow:
    """Exact immutable data needed to replay one actual L-BFGS direction."""

    current_vector: Array
    s_history: Array
    y_history: Array
    sy_history: Array
    actual_direction: Array
    h0_inverse_scale: float
    active_dof_mask: Array | None = None

    def __post_init__(self) -> None:
        current = np.asarray(self.current_vector, dtype=np.float64).reshape(-1)
        actual = np.asarray(self.actual_direction, dtype=np.float64).reshape(-1)
        s = np.asarray(self.s_history, dtype=np.float64)
        y = np.asarray(self.y_history, dtype=np.float64)
        sy = np.asarray(self.sy_history, dtype=np.float64).reshape(-1)

        if current.size == 0:
            raise ValueError("current_vector must be nonempty")
        if actual.shape != current.shape:
            raise ValueError("actual_direction shape does not match current_vector")
        if s.ndim != 2 or y.ndim != 2 or s.shape != y.shape:
            raise ValueError("s_history and y_history must have identical shape (m, n)")
        if s.shape[1] != current.size:
            raise ValueError("history vector length does not match current_vector")
        if sy.size != s.shape[0]:
            raise ValueError("sy_history length does not match history")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(actual)):
            raise ValueError("current_vector/actual_direction contain non-finite values")
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
            raise ValueError("history contains non-finite values")
        if not np.all(np.isfinite(sy)):
            raise ValueError("sy_history contains non-finite values")
        if np.any(sy == 0.0):
            raise ValueError("sy_history contains a zero L-BFGS denominator")

        h0 = float(self.h0_inverse_scale)
        if not np.isfinite(h0) or h0 <= 0.0:
            raise ValueError("h0_inverse_scale must be finite and > 0")

        mask = self.active_dof_mask
        if mask is not None:
            mask_array = np.asarray(mask, dtype=bool).reshape(-1)
            if mask_array.size != current.size:
                raise ValueError("active_dof_mask length does not match current_vector")
            if not np.any(mask_array):
                raise ValueError("active_dof_mask selects no degrees of freedom")
            # This mask is part of the frozen replay contract.  Do not silently
            # discard nonzero algorithm components that lived outside it.
            inactive = ~mask_array
            scale = max(1.0, safe_norm(current), safe_norm(actual))
            tolerance = 256.0 * np.finfo(float).eps * scale
            if np.any(np.abs(current[inactive]) > tolerance):
                raise ValueError("current_vector is nonzero outside active_dof_mask")
            if np.any(np.abs(actual[inactive]) > tolerance):
                raise ValueError("actual_direction is nonzero outside active_dof_mask")
            if s.shape[0] and np.any(np.abs(s[:, inactive]) > tolerance):
                raise ValueError("s_history is nonzero outside active_dof_mask")
            if y.shape[0] and np.any(np.abs(y[:, inactive]) > tolerance):
                raise ValueError("y_history is nonzero outside active_dof_mask")
            object.__setattr__(self, "active_dof_mask", mask_array.copy())

        object.__setattr__(self, "current_vector", current.copy())
        object.__setattr__(self, "actual_direction", actual.copy())
        object.__setattr__(self, "s_history", s.copy())
        object.__setattr__(self, "y_history", y.copy())
        object.__setattr__(self, "sy_history", sy.copy())
        object.__setattr__(self, "h0_inverse_scale", h0)

    @property
    def full_dimension(self) -> int:
        return int(self.current_vector.size)

    @property
    def active_indices(self) -> Array:
        if self.active_dof_mask is None:
            return np.arange(self.current_vector.size, dtype=np.int64)
        return np.flatnonzero(self.active_dof_mask).astype(np.int64, copy=False)

    @property
    def active_dimension(self) -> int:
        return int(self.active_indices.size)

    @property
    def history_size(self) -> int:
        return int(self.s_history.shape[0])


@dataclass
class DenseReplayResult:
    """Structured result for a diagnostic-only dense replay."""

    status: str
    unavailable_reason: str = ""
    metrics: dict[str, object] = field(default_factory=dict)
    dense_direction: Array | None = None
    matrix: Array | None = None

    @property
    def available(self) -> bool:
        return self.status == "available"


def _max_component_group_norm(vector: Array) -> float:
    """Return per-atom max norm when possible, otherwise max absolute component."""

    flat = np.asarray(vector, dtype=float).reshape(-1)
    if flat.size % 3 == 0:
        return safe_max_atom_norm(flat.reshape((-1, 3)))
    if flat.size == 0:
        return 0.0
    return float(np.max(np.abs(flat)))


def _resource_estimate(n_active: int) -> tuple[int, int]:
    # B, symmetrized/eigensolver copy, solve/eigen workspace estimate.  This is
    # intentionally conservative and deterministic rather than backend-specific.
    matrix_bytes = int(n_active) * int(n_active) * np.dtype(np.float64).itemsize
    working_bytes = 4 * matrix_bytes + 16 * int(n_active) * np.dtype(np.float64).itemsize
    work_units = int(n_active) ** 3
    return working_bytes, work_units


def _skip_result(
    window: FrozenReplayWindow,
    reason: str,
    *,
    limits: DenseReplayLimits,
    estimated_working_bytes: int,
    estimated_work_units: int,
    started_ns: int,
) -> DenseReplayResult:
    return DenseReplayResult(
        status="skipped",
        unavailable_reason=reason,
        metrics={
            "schema": DENSE_REPLAY_SCHEMA,
            "dense_replay_available": 0,
            "dense_replay_status": "skipped",
            "dense_replay_unavailable_reason": reason,
            "full_dimension": window.full_dimension,
            "active_dimension": window.active_dimension,
            "history_size": window.history_size,
            "h0_inverse_scale": window.h0_inverse_scale,
            "estimated_working_bytes": int(estimated_working_bytes),
            "estimated_work_units": int(estimated_work_units),
            "limit_max_dimension": limits.max_dimension,
            "limit_max_matrix_bytes": limits.max_matrix_bytes,
            "limit_max_work_units": limits.max_work_units,
            "shadow_matrix_build_ns": 0,
            "shadow_solve_ns": 0,
            "shadow_spectrum_ns": 0,
            "shadow_total_ns": int(perf_counter_ns() - started_ns),
        },
    )


def replay_dense_bfgs(
    window: FrozenReplayWindow,
    *,
    limits: DenseReplayLimits | None = None,
    include_matrix: bool = False,
    near_zero_absolute: float = 1.0e-12,
    near_zero_relative: float = 1.0e-10,
    breakdown_tolerance: float = 1.0e-14,
) -> DenseReplayResult:
    """Reconstruct the dense direct-BFGS matrix for an exact L-BFGS window.

    No pair is admitted, rejected, damped, reordered, or truncated here.  The
    ordered ``s/y/sy`` arrays are already the final production window.  Dynamic
    H0 is likewise already resolved by the caller; only ``h0_inverse_scale`` is
    used to initialize the direct Hessian.
    """

    started = perf_counter_ns()
    limits = DenseReplayLimits() if limits is None else limits
    if not isinstance(limits, DenseReplayLimits):
        raise TypeError("limits must be DenseReplayLimits")
    near_zero_absolute = float(near_zero_absolute)
    near_zero_relative = float(near_zero_relative)
    breakdown_tolerance = float(breakdown_tolerance)
    if near_zero_absolute < 0.0 or near_zero_relative < 0.0:
        raise ValueError("near-zero tolerances must be >= 0")
    if breakdown_tolerance < 0.0:
        raise ValueError("breakdown_tolerance must be >= 0")

    indices = window.active_indices
    n = int(indices.size)
    estimated_bytes, estimated_work = _resource_estimate(n)
    if limits.max_dimension and n > limits.max_dimension:
        return _skip_result(
            window,
            "dimension_limit",
            limits=limits,
            estimated_working_bytes=estimated_bytes,
            estimated_work_units=estimated_work,
            started_ns=started,
        )
    if limits.max_matrix_bytes and estimated_bytes > limits.max_matrix_bytes:
        return _skip_result(
            window,
            "memory_limit",
            limits=limits,
            estimated_working_bytes=estimated_bytes,
            estimated_work_units=estimated_work,
            started_ns=started,
        )
    if limits.max_work_units and estimated_work > limits.max_work_units:
        return _skip_result(
            window,
            "work_limit",
            limits=limits,
            estimated_working_bytes=estimated_bytes,
            estimated_work_units=estimated_work,
            started_ns=started,
        )

    current = window.current_vector[indices]
    actual = window.actual_direction[indices]
    s_hist = window.s_history[:, indices]
    y_hist = window.y_history[:, indices]

    build_start = perf_counter_ns()
    direct_h0 = 1.0 / window.h0_inverse_scale
    B = np.eye(n, dtype=np.float64) * direct_h0
    for pair_index in range(window.history_size):
        s = s_hist[pair_index]
        y = y_hist[pair_index]
        sy = float(window.sy_history[pair_index])
        Bs = B @ s
        sBs = float(np.dot(s, Bs))
        scale = max(1.0, safe_norm(s) * safe_norm(Bs))
        if (
            not np.isfinite(sBs)
            or abs(sBs) <= breakdown_tolerance * scale
            or not np.isfinite(sy)
            or sy == 0.0
        ):
            build_ns = int(perf_counter_ns() - build_start)
            return DenseReplayResult(
                status="unavailable",
                unavailable_reason="dense_update_breakdown",
                metrics={
                    "schema": DENSE_REPLAY_SCHEMA,
                    "dense_replay_available": 0,
                    "dense_replay_status": "unavailable",
                    "dense_replay_unavailable_reason": "dense_update_breakdown",
                    "dense_update_breakdown_pair_index": pair_index,
                    "dense_update_s_dot_Bs": sBs,
                    "dense_update_s_dot_y": sy,
                    "full_dimension": window.full_dimension,
                    "active_dimension": window.active_dimension,
                    "history_size": window.history_size,
                    "h0_inverse_scale": window.h0_inverse_scale,
                    "estimated_working_bytes": estimated_bytes,
                    "estimated_work_units": estimated_work,
                    "shadow_matrix_build_ns": build_ns,
                    "shadow_solve_ns": 0,
                    "shadow_spectrum_ns": 0,
                    "shadow_total_ns": int(perf_counter_ns() - started),
                },
            )
        B = B - np.outer(Bs, Bs) / sBs + np.outer(y, y) / sy
        # Keep roundoff symmetry from becoming a diagnostic artifact.
        B = 0.5 * (B + B.T)
    build_ns = int(perf_counter_ns() - build_start)

    solve_start = perf_counter_ns()
    try:
        dense_active = np.linalg.solve(B, current)
    except np.linalg.LinAlgError:
        solve_ns = int(perf_counter_ns() - solve_start)
        return DenseReplayResult(
            status="unavailable",
            unavailable_reason="dense_solve_failed",
            metrics={
                "schema": DENSE_REPLAY_SCHEMA,
                "dense_replay_available": 0,
                "dense_replay_status": "unavailable",
                "dense_replay_unavailable_reason": "dense_solve_failed",
                "full_dimension": window.full_dimension,
                "active_dimension": window.active_dimension,
                "history_size": window.history_size,
                "h0_inverse_scale": window.h0_inverse_scale,
                "estimated_working_bytes": estimated_bytes,
                "estimated_work_units": estimated_work,
                "shadow_matrix_build_ns": build_ns,
                "shadow_solve_ns": solve_ns,
                "shadow_spectrum_ns": 0,
                "shadow_total_ns": int(perf_counter_ns() - started),
            },
        )
    solve_ns = int(perf_counter_ns() - solve_start)
    if not np.all(np.isfinite(dense_active)):
        return DenseReplayResult(
            status="unavailable",
            unavailable_reason="dense_solve_nonfinite",
            metrics={
                "schema": DENSE_REPLAY_SCHEMA,
                "dense_replay_available": 0,
                "dense_replay_status": "unavailable",
                "dense_replay_unavailable_reason": "dense_solve_nonfinite",
                "full_dimension": window.full_dimension,
                "active_dimension": window.active_dimension,
                "history_size": window.history_size,
                "h0_inverse_scale": window.h0_inverse_scale,
                "estimated_working_bytes": estimated_bytes,
                "estimated_work_units": estimated_work,
                "shadow_matrix_build_ns": build_ns,
                "shadow_solve_ns": solve_ns,
                "shadow_spectrum_ns": 0,
                "shadow_total_ns": int(perf_counter_ns() - started),
            },
        )

    spectrum_start = perf_counter_ns()
    try:
        eigvals = np.linalg.eigvalsh(B)
    except np.linalg.LinAlgError:
        eigvals = np.empty(0, dtype=np.float64)
    spectrum_ns = int(perf_counter_ns() - spectrum_start)

    dense_full = np.zeros(window.full_dimension, dtype=np.float64)
    dense_full[indices] = dense_active
    difference = dense_full - window.actual_direction

    spectrum_valid = bool(eigvals.size == n and np.all(np.isfinite(eigvals)))
    lambda_min: float | str = ""
    lambda_max: float | str = ""
    negative_count: int | str = ""
    near_zero_count: int | str = ""
    spectral_spread: float | str = ""
    abs_condition: float | str = ""
    zero_threshold: float | str = ""
    if spectrum_valid:
        lambda_min = float(eigvals[0])
        lambda_max = float(eigvals[-1])
        spectral_scale = float(np.max(np.abs(eigvals), initial=0.0))
        threshold = max(near_zero_absolute, near_zero_relative * spectral_scale)
        zero_threshold = threshold
        negative_count = int(np.count_nonzero(eigvals < -threshold))
        near_zero_count = int(np.count_nonzero(np.abs(eigvals) <= threshold))
        spectral_spread = float(lambda_max - lambda_min)
        nonzero = np.abs(eigvals)[np.abs(eigvals) > threshold]
        if nonzero.size:
            abs_condition = float(np.max(np.abs(eigvals)) / np.min(nonzero))

    actual_norm = safe_norm(window.actual_direction)
    dense_norm = safe_norm(dense_full)
    difference_norm = safe_norm(difference)
    metrics: dict[str, object] = {
        "schema": DENSE_REPLAY_SCHEMA,
        "dense_replay_available": 1,
        "dense_replay_status": "available",
        "dense_replay_unavailable_reason": "",
        "full_dimension": window.full_dimension,
        "active_dimension": window.active_dimension,
        "history_size": window.history_size,
        "h0_inverse_scale": window.h0_inverse_scale,
        "h0_direct_scale": direct_h0,
        "estimated_working_bytes": estimated_bytes,
        "estimated_work_units": estimated_work,
        "actual_direction_norm": actual_norm,
        "actual_direction_max_atom_norm": _max_component_group_norm(window.actual_direction),
        "lbfgs_direction_norm": actual_norm,
        "lbfgs_direction_max_atom_norm": _max_component_group_norm(window.actual_direction),
        "dense_direction_norm": dense_norm,
        "dense_direction_max_atom_norm": _max_component_group_norm(dense_full),
        "dense_replay_direction_norm": dense_norm,
        "dense_replay_direction_max_atom_norm": _max_component_group_norm(dense_full),
        "dense_vs_actual_cosine": safe_cosine(dense_full, window.actual_direction),
        "dense_vs_actual_difference_norm": difference_norm,
        "dense_vs_actual_max_abs_difference": float(np.max(np.abs(difference), initial=0.0)),
        "dense_vs_actual_relative_difference": (
            "" if actual_norm <= 1.0e-300 else difference_norm / actual_norm
        ),
        "dense_vs_actual_norm_ratio": (
            "" if actual_norm <= 1.0e-300 else dense_norm / actual_norm
        ),
        "spectrum_valid": int(spectrum_valid),
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "negative_eigenvalue_count": negative_count,
        "near_zero_eigenvalue_count": near_zero_count,
        "near_zero_threshold": zero_threshold,
        "spectral_spread": spectral_spread,
        "spectral_abs_condition": abs_condition,
        "shadow_matrix_build_ns": build_ns,
        "shadow_solve_ns": solve_ns,
        "shadow_spectrum_ns": spectrum_ns,
        "shadow_total_ns": int(perf_counter_ns() - started),
        "limit_max_dimension": limits.max_dimension,
        "limit_max_matrix_bytes": limits.max_matrix_bytes,
        "limit_max_work_units": limits.max_work_units,
    }
    return DenseReplayResult(
        status="available",
        metrics=metrics,
        dense_direction=dense_full,
        matrix=(np.array(B, copy=True) if include_matrix else None),
    )
