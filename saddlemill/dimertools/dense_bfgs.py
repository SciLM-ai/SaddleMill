"""Dense BFGS reconstruction and diagnostics for SaddleMill quasi-Newton consumers.

This module is intentionally calculator/ASE independent.  It provides:

* overflow-safe vector norms used by diagnostics and step capping;
* dense sequential BFGS reconstruction from the same chronological secants used
  by L-BFGS (an algebraic oracle / optional driving model);
* state-block multisecant BFGS using Sella's ``symmetrize_Y(..., symm=2)`` and
  ordinary multisecant-BFGS update formula, while retaining only blocks that
  preserve a positive-definite minimization Hessian.

It does *not* implement TS-BFGS or PRFO.  The reconstructed Hessians here are
for SaddleMill's rotation/MMF minimization consumers and therefore start from a
positive scalar Hessian and fail closed on non-positive block curvature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Iterable, Mapping, Sequence

import numpy as np

Array = np.ndarray


def safe_norm(value: object) -> float:
    """Overflow-safe Euclidean norm for a finite vector/array."""

    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return 0.0
    if not np.all(np.isfinite(array)):
        return float("inf")
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0
    return scale * float(np.linalg.norm(array / scale))


def safe_max_atom_norm(value: object) -> float:
    """Overflow-safe maximum per-atom 3-vector norm."""

    array = np.asarray(value, dtype=float).reshape((-1, 3))
    if array.size == 0:
        return 0.0
    if not np.all(np.isfinite(array)):
        return float("inf")
    row_scale = np.max(np.abs(array), axis=1)
    nonzero = row_scale > 0.0
    norms = np.zeros(len(array), dtype=float)
    if np.any(nonzero):
        scaled = array[nonzero] / row_scale[nonzero, None]
        norms[nonzero] = row_scale[nonzero] * np.linalg.norm(scaled, axis=1)
    return float(np.max(norms))


def safe_cosine(left: object, right: object) -> float | str:
    """Overflow-safe cosine between two finite vectors."""

    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        return ""
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return ""
    sa = float(np.max(np.abs(a)))
    sb = float(np.max(np.abs(b)))
    if sa == 0.0 or sb == 0.0:
        return ""
    aa = a / sa
    bb = b / sb
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denominator <= 1.0e-30 or not np.isfinite(denominator):
        return ""
    return float(np.dot(aa, bb) / denominator)


def safe_scale_to_max_atom(
    direction: object,
    maximum: float,
) -> tuple[Array, float, bool, float]:
    """Return a safely max-atom-capped copy and its diagnostics.

    Returns ``(scaled_direction, raw_max_atom_norm, clipped, scale)``.  A finite
    direction whose derived norm cannot be represented safely is treated as an
    invalid proposal by returning ``scale=nan`` rather than silently multiplying
    by zero.  Callers should fall back to their existing safe direction.
    """

    array = np.asarray(direction, dtype=float).copy()
    maximum = float(maximum)
    raw_max = safe_max_atom_norm(array)
    if maximum <= 0.0 or not np.isfinite(maximum):
        raise ValueError("maximum must be finite and > 0")
    if not np.all(np.isfinite(array)) or not np.isfinite(raw_max):
        return array, raw_max, False, float("nan")
    if raw_max <= maximum or raw_max <= 0.0:
        return array, raw_max, False, 1.0
    scale = maximum / raw_max
    if not np.isfinite(scale) or scale <= 0.0:
        return array, raw_max, False, float("nan")
    array *= scale
    return array, raw_max, True, scale


@dataclass(frozen=True)
class DenseSecant:
    s: Array
    y: Array
    source: str = ""
    state_id: int = -1
    serial: int = -1

    def __post_init__(self) -> None:
        s = np.asarray(self.s, dtype=float).reshape(-1)
        y = np.asarray(self.y, dtype=float).reshape(-1)
        if s.shape != y.shape:
            raise ValueError("DenseSecant s/y shapes differ")
        object.__setattr__(self, "s", s.copy())
        object.__setattr__(self, "y", y.copy())
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "state_id", int(self.state_id))
        object.__setattr__(self, "serial", int(self.serial))


@dataclass
class DenseBFGSResult:
    matrix: Array | None
    direction: Array | None
    metrics: dict[str, object] = field(default_factory=dict)


@dataclass
class ShiftedTrustRegionResult:
    """Result from a scalar-shifted trust-region BFGS solve."""

    direction: Array | None
    shift: float | None
    metrics: dict[str, object] = field(default_factory=dict)


def _boundary_norm(direction: object) -> float:
    array = np.asarray(direction, dtype=float).reshape(-1)
    if array.size and array.size % 3 == 0:
        return safe_max_atom_norm(array)
    return safe_norm(array)


def _matrix_condition_number(matrix: Array) -> float | str:
    try:
        value = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return ""
    return value if np.isfinite(value) else ""


def _h0_direct_scale(
    pairs: Sequence[DenseSecant],
    *,
    initial_hessian: float,
    dynamic_h0: bool,
) -> tuple[float, float]:
    """Return ``(B0 scalar, H0 scalar)`` matching L-BFGS initialization."""

    h0 = 1.0 / float(initial_hessian)
    if dynamic_h0 and pairs:
        latest = pairs[-1]
        sy = float(np.dot(latest.s, latest.y))
        yy = float(np.dot(latest.y, latest.y))
        if sy > 0.0 and yy > 0.0:
            candidate = sy / yy
            if np.isfinite(candidate) and candidate > 0.0:
                h0 = candidate
    return 1.0 / h0, h0


def _spectrum_metrics(matrix: Array) -> dict[str, object]:
    try:
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError:
        return {
            "dense_spectrum_valid": 0,
            "dense_min_eigenvalue": "",
            "dense_max_eigenvalue": "",
            "dense_negative_eigenvalues": "",
            "dense_condition_number": "",
            "dense_log10_condition": "",
        }
    if not np.all(np.isfinite(eigenvalues)):
        return {
            "dense_spectrum_valid": 0,
            "dense_min_eigenvalue": "",
            "dense_max_eigenvalue": "",
            "dense_negative_eigenvalues": "",
            "dense_condition_number": "",
            "dense_log10_condition": "",
        }
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    abs_eigs = np.abs(eigenvalues)
    nonzero = abs_eigs[abs_eigs > np.finfo(float).tiny]
    condition = ""
    log_condition = ""
    if nonzero.size:
        condition_value = float(np.max(abs_eigs) / np.min(nonzero))
        condition = condition_value
        if condition_value > 0.0 and np.isfinite(condition_value):
            log_condition = float(np.log10(condition_value))
    return {
        "dense_spectrum_valid": 1,
        "dense_min_eigenvalue": minimum,
        "dense_max_eigenvalue": maximum,
        "dense_negative_eigenvalues": int(np.count_nonzero(eigenvalues < 0.0)),
        "dense_condition_number": condition,
        "dense_log10_condition": log_condition,
    }


def _secant_residual_metrics(matrix: Array, pairs: Sequence[DenseSecant]) -> dict[str, object]:
    if not pairs:
        return {
            "dense_secant_residual_latest": "",
            "dense_secant_residual_median": "",
            "dense_secant_residual_max": "",
        }
    residuals: list[float] = []
    for pair in pairs:
        residual = safe_norm(matrix @ pair.s - pair.y)
        denominator = safe_norm(pair.y)
        if denominator > 0.0 and np.isfinite(residual) and np.isfinite(denominator):
            residuals.append(residual / denominator)
    if not residuals:
        return {
            "dense_secant_residual_latest": "",
            "dense_secant_residual_median": "",
            "dense_secant_residual_max": "",
        }
    return {
        "dense_secant_residual_latest": float(residuals[-1]),
        "dense_secant_residual_median": float(np.median(residuals)),
        "dense_secant_residual_max": float(np.max(residuals)),
    }


def _solve_direction(matrix: Array, force: object) -> tuple[Array | None, str]:
    f = np.asarray(force, dtype=float).reshape(-1)
    if matrix.shape != (f.size, f.size) or not np.all(np.isfinite(f)):
        return None, "shape_or_force_invalid"
    try:
        direction = np.linalg.solve(matrix, f)
    except np.linalg.LinAlgError:
        return None, "solve_failed"
    if not np.all(np.isfinite(direction)):
        return None, "solve_nonfinite"
    return direction, ""


def shifted_trust_region_solve(
    matrix: Array,
    force: object,
    *,
    radius: float,
    tolerance: float = 1.0e-8,
    max_iterations: int = 60,
    max_shift_expansions: int = 80,
) -> ShiftedTrustRegionResult:
    """Solve a regularized BFGS step using a scalar Hessian shift.

    This is the Moré--Sorensen/trust-region shifted-system idea adapted to
    SaddleMill's existing maximum-per-atom translation bound. If ``B`` is
    positive definite and its unshifted step is inside the bound, ``lambda=0``.
    Otherwise a scalar ``lambda>0`` is found by safeguarded expansion/bisection
    until ``B + lambda I`` admits a Cholesky factorization and the proposed
    step reaches the existing bound. No eigen-decomposition is required.

    This changes only the proposed quasi-Newton direction; callers retain the
    existing SaddleMill damping/max-step semantics.
    """

    started = perf_counter_ns()
    B = np.asarray(matrix, dtype=float)
    f = np.asarray(force, dtype=float).reshape(-1)
    radius = float(radius)
    metrics: dict[str, object] = {
        "trust_regularization": "shifted_trust_region",
        "trust_radius_metric": "max_atom_norm" if f.size % 3 == 0 else "euclidean",
        "trust_radius": radius,
        "trust_shift": "",
        "trust_iterations": 0,
        "trust_shift_expansions": 0,
        "trust_unshifted_boundary_norm": "",
        "trust_regularized_boundary_norm": "",
        "trust_regularized_direction_norm": "",
        "trust_direction_cosine_unshifted": "",
        "trust_shifted_condition_number": "",
        "trust_failure_reason": "",
    }
    if (
        B.ndim != 2
        or B.shape != (f.size, f.size)
        or f.size == 0
        or not np.all(np.isfinite(B))
        or not np.all(np.isfinite(f))
        or not np.isfinite(radius)
        or radius <= 0.0
    ):
        metrics["trust_failure_reason"] = "invalid_input"
        metrics["trust_solve_ns"] = int(perf_counter_ns() - started)
        return ShiftedTrustRegionResult(None, None, metrics)
    B = 0.5 * (B + B.T)
    I = np.eye(f.size, dtype=float)

    def solve_at(lam: float):
        shifted = B + float(lam) * I
        try:
            np.linalg.cholesky(shifted)
            d = np.linalg.solve(shifted, f)
        except np.linalg.LinAlgError:
            return None, None
        if not np.all(np.isfinite(d)):
            return None, None
        return d, shifted

    unshifted, unshifted_matrix = solve_at(0.0)
    if unshifted is not None:
        unshifted_boundary = _boundary_norm(unshifted)
        metrics["trust_unshifted_boundary_norm"] = unshifted_boundary
        if np.isfinite(unshifted_boundary) and unshifted_boundary <= radius:
            metrics.update({
                "trust_shift": 0.0,
                "trust_regularized_boundary_norm": unshifted_boundary,
                "trust_regularized_direction_norm": safe_norm(unshifted),
                "trust_direction_cosine_unshifted": 1.0,
                "trust_shifted_condition_number": _matrix_condition_number(unshifted_matrix),
                "trust_solve_ns": int(perf_counter_ns() - started),
            })
            return ShiftedTrustRegionResult(unshifted, 0.0, metrics)
    else:
        raw, _reason = _solve_direction(B, f)
        if raw is not None:
            metrics["trust_unshifted_boundary_norm"] = _boundary_norm(raw)
            unshifted = raw

    scale = max(1.0, float(np.max(np.abs(np.diag(B)))) if B.size else 1.0)
    lo = 0.0
    hi = max(1.0e-12, 1.0e-10 * scale)
    hi_direction = None
    hi_matrix = None
    for expansion in range(int(max_shift_expansions)):
        candidate, shifted = solve_at(hi)
        metrics["trust_shift_expansions"] = expansion + 1
        if candidate is not None:
            boundary = _boundary_norm(candidate)
            if np.isfinite(boundary) and boundary <= radius:
                hi_direction = candidate
                hi_matrix = shifted
                break
        lo = hi
        hi *= 10.0
    if hi_direction is None:
        metrics["trust_failure_reason"] = "could_not_bracket_shift"
        metrics["trust_solve_ns"] = int(perf_counter_ns() - started)
        return ShiftedTrustRegionResult(None, None, metrics)

    target_tol = max(float(tolerance), 1.0e-12)
    for iteration in range(int(max_iterations)):
        metrics["trust_iterations"] = iteration + 1
        mid = 0.5 * (lo + hi)
        candidate, shifted = solve_at(mid)
        if candidate is None:
            lo = mid
            continue
        boundary = _boundary_norm(candidate)
        if not np.isfinite(boundary):
            lo = mid
            continue
        if boundary > radius:
            lo = mid
        else:
            hi = mid
            hi_direction = candidate
            hi_matrix = shifted
        if hi - lo <= target_tol * max(1.0, hi):
            break

    boundary = _boundary_norm(hi_direction)
    metrics.update({
        "trust_shift": float(hi),
        "trust_regularized_boundary_norm": boundary,
        "trust_regularized_direction_norm": safe_norm(hi_direction),
        "trust_direction_cosine_unshifted": (
            "" if unshifted is None else safe_cosine(hi_direction, unshifted)
        ),
        "trust_shifted_condition_number": _matrix_condition_number(hi_matrix),
        "trust_solve_ns": int(perf_counter_ns() - started),
    })
    return ShiftedTrustRegionResult(hi_direction, float(hi), metrics)


def reconstruct_sequential_bfgs(
    pairs: Sequence[DenseSecant],
    force: object,
    *,
    initial_hessian: float,
    dynamic_h0: bool = False,
    denominator_epsilon: float = 1.0e-14,
    require_positive_definite: bool = False,
    spectrum: bool = True,
) -> DenseBFGSResult:
    """Rebuild a dense direct Hessian with chronological rank-two BFGS updates.

    With the same ordered secants and H0 scaling this is algebraically equivalent
    to the corresponding L-BFGS inverse-Hessian recursion, up to floating-point
    roundoff.  ``require_positive_definite=False`` is useful for a read-only
    shadow of legacy histories that may contain non-positive secants.
    """

    started = perf_counter_ns()
    f = np.asarray(force, dtype=float).reshape(-1)
    n = f.size
    metrics: dict[str, object] = {
        "dense_model": "sequential_bfgs",
        "dense_pairs_requested": len(pairs),
        "dense_pairs_applied": 0,
        "dense_invalid_reason": "",
    }
    if n == 0 or not np.all(np.isfinite(f)):
        metrics["dense_invalid_reason"] = "force_invalid"
        return DenseBFGSResult(None, None, metrics)
    if not np.isfinite(initial_hessian) or float(initial_hessian) <= 0.0:
        metrics["dense_invalid_reason"] = "initial_hessian_invalid"
        return DenseBFGSResult(None, None, metrics)
    for pair in pairs:
        if pair.s.size != n or pair.y.size != n:
            metrics["dense_invalid_reason"] = "pair_shape_mismatch"
            return DenseBFGSResult(None, None, metrics)

    b0, h0 = _h0_direct_scale(
        pairs,
        initial_hessian=float(initial_hessian),
        dynamic_h0=bool(dynamic_h0),
    )
    B = float(b0) * np.eye(n, dtype=float)
    threshold = float(denominator_epsilon)
    for index, pair in enumerate(pairs):
        s = pair.s
        y = pair.y
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
            metrics["dense_invalid_reason"] = f"pair_{index}_nonfinite"
            return DenseBFGSResult(None, None, metrics)
        Bs = B @ s
        sBs = float(np.dot(s, Bs))
        sy = float(np.dot(s, y))
        scale_b = max(1.0, safe_norm(s) * safe_norm(Bs))
        scale_y = max(1.0, safe_norm(s) * safe_norm(y))
        if (
            not np.isfinite(sBs)
            or not np.isfinite(sy)
            or abs(sBs) <= threshold * scale_b
            or abs(sy) <= threshold * scale_y
        ):
            metrics["dense_invalid_reason"] = f"pair_{index}_singular_denominator"
            return DenseBFGSResult(None, None, metrics)
        if require_positive_definite and (sBs <= 0.0 or sy <= 0.0):
            metrics["dense_invalid_reason"] = f"pair_{index}_nonpositive_curvature"
            return DenseBFGSResult(None, None, metrics)
        B = B - np.outer(Bs, Bs) / sBs + np.outer(y, y) / sy
        B = 0.5 * (B + B.T)
        if not np.all(np.isfinite(B)):
            metrics["dense_invalid_reason"] = f"pair_{index}_matrix_nonfinite"
            return DenseBFGSResult(None, None, metrics)
        metrics["dense_pairs_applied"] = index + 1

    direction, solve_reason = _solve_direction(B, f)
    if solve_reason:
        metrics["dense_invalid_reason"] = solve_reason
    metrics.update(
        {
            "dense_h0_inverse_scale": h0,
            "dense_b0_scale": b0,
            "dense_matrix_symmetry_max_abs": float(np.max(np.abs(B - B.T))),
            "dense_direction_norm": "" if direction is None else safe_norm(direction),
            "dense_direction_max_atom_norm": (
                "" if direction is None or direction.size % 3 else safe_max_atom_norm(direction)
            ),
            "dense_reconstruction_ns": int(perf_counter_ns() - started),
        }
    )
    if spectrum:
        metrics.update(_spectrum_metrics(B))
    metrics.update(_secant_residual_metrics(B, pairs))
    return DenseBFGSResult(B, direction, metrics)


def symmetrize_y_sella2(S: Array, Y: Array, *, rcond: float | None = None) -> Array:
    """Sella's ``symmetrize_Y(..., symm=2)`` using NumPy only.

    This makes the multisecant block compatible with a symmetric Hessian while
    leaving a one-column block unchanged.  The algorithm is reproduced from
    Sella's public ``hessian_update.py`` implementation.
    """

    S = np.asarray(S, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if S.shape != Y.shape:
        raise ValueError("S/Y shapes differ")
    if S.shape[1] <= 1:
        return Y.copy()
    nvecs = S.shape[1]
    dY = np.zeros_like(Y)
    YTS = Y.T @ S
    dYTS = np.zeros_like(YTS)
    STS = S.T @ S
    for i in range(1, nvecs):
        rhs = YTS[i, :i].T - YTS[:i, i] - dYTS[:i, i]
        coeff = np.linalg.lstsq(STS[:i, :i], rhs, rcond=rcond)[0]
        dY[:, i] = -S[:, :i] @ coeff
        dYTS[i, :] = -STS[:, :i] @ coeff
    return Y + dY


def _positive_definite(matrix: Array, epsilon: float) -> tuple[bool, float | str]:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    if matrix.size == 0 or not np.all(np.isfinite(matrix)):
        return False, ""
    try:
        eigs = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return False, ""
    if not np.all(np.isfinite(eigs)):
        return False, ""
    minimum = float(eigs[0])
    scale = max(1.0, float(np.max(np.abs(eigs))))
    return minimum > float(epsilon) * scale, minimum


def _apply_multisecant_bfgs(B: Array, S: Array, Y: Array) -> Array:
    """Ordinary multisecant BFGS update used by Sella's ``_MS_BFGS``."""

    Ytilde = symmetrize_y_sella2(S, Y)
    YTS = 0.5 * (Ytilde.T @ S + S.T @ Ytilde)
    BS = B @ S
    STBS = 0.5 * (S.T @ BS + BS.T @ S)
    # Sella formula: Y solve(Y.T S, Y.T) - B S solve(S.T B S, S.T B)
    delta = Ytilde @ np.linalg.solve(YTS, Ytilde.T)
    delta -= BS @ np.linalg.solve(STBS, S.T @ B)
    Bplus = B + delta
    return 0.5 * (Bplus + Bplus.T)


def state_blocks(pairs: Sequence[DenseSecant]) -> list[list[DenseSecant]]:
    """Build chronological state blocks.

    For translation, all non-``center_center`` probe secants from a state are one
    block followed by each center->next-center secant as a singleton.  For
    rotation there are no center-center pairs, so all torque secants from one
    translation center form one block.
    """

    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda pair: (pair.serial, pair.state_id, pair.source))
    state_order: list[int] = []
    by_state: dict[int, list[DenseSecant]] = {}
    for pair in ordered:
        if pair.state_id not in by_state:
            by_state[pair.state_id] = []
            state_order.append(pair.state_id)
        by_state[pair.state_id].append(pair)
    blocks: list[list[DenseSecant]] = []
    for state_id in state_order:
        items = by_state[state_id]
        probes = [pair for pair in items if pair.source != "center_center"]
        centers = [pair for pair in items if pair.source == "center_center"]
        if probes:
            blocks.append(probes)
        blocks.extend([[pair] for pair in centers])
    return blocks


def _greedy_positive_block(
    B: Array,
    block: Sequence[DenseSecant],
    *,
    curvature_epsilon: float,
) -> tuple[list[DenseSecant], dict[str, int]]:
    """Keep the largest chronological prefix/subset compatible with PD MS-BFGS.

    Each candidate is tentatively appended.  It is skipped if the displacement
    columns become linearly dependent or if the Sella-symmetrized ``S.T@Y`` block
    is not positive definite.  This is the multisecant analogue of the requested
    pair-skip policy; no damping or TS-BFGS fallback is performed.
    """

    kept: list[DenseSecant] = []
    reasons = {"rank": 0, "block_curvature": 0, "invalid": 0}
    for pair in block:
        trial = [*kept, pair]
        S = np.column_stack([item.s for item in trial])
        Y = np.column_stack([item.y for item in trial])
        if not np.all(np.isfinite(S)) or not np.all(np.isfinite(Y)):
            reasons["invalid"] += 1
            continue
        # Require independent displacement columns; otherwise S.T B S is singular.
        if np.linalg.matrix_rank(S) < S.shape[1]:
            reasons["rank"] += 1
            continue
        try:
            Ytilde = symmetrize_y_sella2(S, Y)
        except np.linalg.LinAlgError:
            reasons["invalid"] += 1
            continue
        sty = 0.5 * (S.T @ Ytilde + Ytilde.T @ S)
        good, _minimum = _positive_definite(sty, curvature_epsilon)
        if not good:
            reasons["block_curvature"] += 1
            continue
        kept.append(pair)
    return kept, reasons


def reconstruct_multisecant_bfgs(
    pairs: Sequence[DenseSecant],
    force: object,
    *,
    initial_hessian: float,
    dynamic_h0: bool = False,
    curvature_epsilon: float = 1.0e-12,
    spectrum: bool = True,
) -> DenseBFGSResult:
    """Rebuild a positive-definite dense Hessian using state-block MS-BFGS.

    Same-state probe/torque secants are consumed jointly. Center-to-center
    translation secants remain chronological singleton updates. No block is
    damped and there is no TS-BFGS fallback. Extensive block diagnostics are
    emitted because the first production benchmark showed that this variant
    can perform much worse than sequential BFGS.
    """

    started = perf_counter_ns()
    f = np.asarray(force, dtype=float).reshape(-1)
    n = f.size
    metrics: dict[str, object] = {
        "dense_model": "multisecant_bfgs",
        "dense_pairs_requested": len(pairs),
        "dense_pairs_applied": 0,
        "dense_multisecant_blocks_requested": 0,
        "dense_multisecant_blocks_applied": 0,
        "dense_multisecant_pairs_skipped_rank": 0,
        "dense_multisecant_pairs_skipped_curvature": 0,
        "dense_multisecant_pairs_skipped_invalid": 0,
        "dense_multisecant_blocks_rejected_update": 0,
        "dense_invalid_reason": "",
    }
    if n == 0 or not np.all(np.isfinite(f)):
        metrics["dense_invalid_reason"] = "force_invalid"
        return DenseBFGSResult(None, None, metrics)
    if not np.isfinite(initial_hessian) or float(initial_hessian) <= 0.0:
        metrics["dense_invalid_reason"] = "initial_hessian_invalid"
        return DenseBFGSResult(None, None, metrics)
    for pair in pairs:
        if pair.s.size != n or pair.y.size != n:
            metrics["dense_invalid_reason"] = "pair_shape_mismatch"
            return DenseBFGSResult(None, None, metrics)

    b0, h0 = _h0_direct_scale(
        pairs, initial_hessian=float(initial_hessian), dynamic_h0=bool(dynamic_h0)
    )
    B = float(b0) * np.eye(n, dtype=float)
    applied_pairs: list[DenseSecant] = []
    blocks = state_blocks(pairs)
    metrics["dense_multisecant_blocks_requested"] = len(blocks)

    block_sizes: list[int] = []
    block_kept_sizes: list[int] = []
    y_correction_rel: list[float] = []
    sty_asym_rel: list[float] = []
    sts_conds: list[float] = []
    yts_conds: list[float] = []
    stbs_conds: list[float] = []
    sty_original_min: list[float] = []
    sty_adjusted_min: list[float] = []
    residual_original: list[float] = []
    residual_adjusted: list[float] = []
    block_details: list[dict[str, object]] = []

    def append_finite(bucket: list[float], value: object) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if np.isfinite(number):
            bucket.append(number)

    for block in blocks:
        block_sizes.append(len(block))
        kept, reasons = _greedy_positive_block(
            B, block, curvature_epsilon=float(curvature_epsilon)
        )
        block_kept_sizes.append(len(kept))
        metrics["dense_multisecant_pairs_skipped_rank"] += reasons["rank"]
        metrics["dense_multisecant_pairs_skipped_curvature"] += reasons["block_curvature"]
        metrics["dense_multisecant_pairs_skipped_invalid"] += reasons["invalid"]
        if not kept:
            block_details.append({"state_id": block[0].state_id if block else -1, "source": block[0].source if block else "", "requested": len(block), "kept": 0, "rank_skipped": reasons["rank"], "curvature_skipped": reasons["block_curvature"], "invalid_skipped": reasons["invalid"], "applied": 0, "reason": "no_compatible_pairs"})
            continue
        S = np.column_stack([item.s for item in kept])
        Y = np.column_stack([item.y for item in kept])
        try:
            Ytilde = symmetrize_y_sella2(S, Y)
        except np.linalg.LinAlgError:
            metrics["dense_multisecant_blocks_rejected_update"] += 1
            continue
        YTS_raw = Y.T @ S
        YTS = 0.5 * (Ytilde.T @ S + S.T @ Ytilde)
        BS = B @ S
        STBS = 0.5 * (S.T @ BS + BS.T @ S)
        STS = 0.5 * (S.T @ S + (S.T @ S).T)
        denom_y = max(1.0e-30, float(np.linalg.norm(Y)))
        denom_sty = max(1.0e-30, float(np.linalg.norm(YTS_raw)))
        append_finite(y_correction_rel, np.linalg.norm(Ytilde - Y) / denom_y)
        append_finite(sty_asym_rel, np.linalg.norm(YTS_raw - YTS_raw.T) / denom_sty)
        append_finite(sts_conds, np.linalg.cond(STS))
        append_finite(yts_conds, np.linalg.cond(YTS))
        append_finite(stbs_conds, np.linalg.cond(STBS))
        try:
            append_finite(sty_original_min, np.linalg.eigvalsh(0.5 * (YTS_raw + YTS_raw.T))[0])
            append_finite(sty_adjusted_min, np.linalg.eigvalsh(YTS)[0])
        except np.linalg.LinAlgError:
            pass

        before = B.copy()
        detail = {
            "state_id": kept[0].state_id, "source": kept[0].source,
            "requested": len(block), "kept": len(kept),
            "rank_skipped": reasons["rank"], "curvature_skipped": reasons["block_curvature"], "invalid_skipped": reasons["invalid"],
            "y_symmetrization_relative": float(np.linalg.norm(Ytilde-Y)/denom_y),
            "sty_asymmetry_relative": float(np.linalg.norm(YTS_raw-YTS_raw.T)/denom_sty),
            "sts_condition": float(np.linalg.cond(STS)), "yts_condition": float(np.linalg.cond(YTS)), "stbs_condition": float(np.linalg.cond(STBS)),
            "sty_original_min_eigenvalue": "", "sty_adjusted_min_eigenvalue": "", "applied": 0, "reason": "",
        }
        try:
            detail["sty_original_min_eigenvalue"] = float(np.linalg.eigvalsh(0.5*(YTS_raw+YTS_raw.T))[0])
            detail["sty_adjusted_min_eigenvalue"] = float(np.linalg.eigvalsh(YTS)[0])
        except np.linalg.LinAlgError:
            pass
        try:
            candidate = _apply_multisecant_bfgs(B, S, Y)
        except np.linalg.LinAlgError:
            metrics["dense_multisecant_blocks_rejected_update"] += 1
            detail["reason"]="update_solve_failure"; block_details.append(detail)
            continue
        if not np.all(np.isfinite(candidate)):
            metrics["dense_multisecant_blocks_rejected_update"] += 1
            detail["reason"]="nonfinite_candidate"; block_details.append(detail)
            continue
        pd, _minimum = _positive_definite(candidate, float(curvature_epsilon))
        if not pd:
            B = before
            metrics["dense_multisecant_blocks_rejected_update"] += 1
            detail["reason"]="candidate_not_positive_definite"; block_details.append(detail)
            continue

        BSplus = candidate @ S
        append_finite(
            residual_original,
            np.linalg.norm(BSplus - Y) / max(1.0e-30, float(np.linalg.norm(Y))),
        )
        append_finite(
            residual_adjusted,
            np.linalg.norm(BSplus - Ytilde) / max(1.0e-30, float(np.linalg.norm(Ytilde))),
        )
        detail["residual_original_y"] = float(np.linalg.norm(BSplus-Y)/max(1.0e-30,float(np.linalg.norm(Y))))
        detail["residual_adjusted_y"] = float(np.linalg.norm(BSplus-Ytilde)/max(1.0e-30,float(np.linalg.norm(Ytilde))))
        detail["applied"] = 1
        block_details.append(detail)
        B = candidate
        applied_pairs.extend(kept)
        metrics["dense_multisecant_blocks_applied"] += 1
        metrics["dense_pairs_applied"] = len(applied_pairs)

    def summarize(name: str, values: list[float]) -> None:
        if not values:
            metrics[f"dense_multisecant_{name}_median"] = ""
            metrics[f"dense_multisecant_{name}_max"] = ""
            return
        arr = np.asarray(values, dtype=float)
        metrics[f"dense_multisecant_{name}_median"] = float(np.median(arr))
        metrics[f"dense_multisecant_{name}_max"] = float(np.max(arr))

    metrics["dense_multisecant_block_size_median"] = float(np.median(block_sizes)) if block_sizes else ""
    metrics["dense_multisecant_block_size_max"] = max(block_sizes) if block_sizes else ""
    metrics["dense_multisecant_block_kept_median"] = float(np.median(block_kept_sizes)) if block_kept_sizes else ""
    summarize("y_symmetrization_relative", y_correction_rel)
    summarize("sty_asymmetry_relative", sty_asym_rel)
    summarize("sts_condition", sts_conds)
    summarize("yts_condition", yts_conds)
    summarize("stbs_condition", stbs_conds)
    summarize("sty_original_min_eigenvalue", sty_original_min)
    summarize("sty_adjusted_min_eigenvalue", sty_adjusted_min)
    summarize("residual_original_y", residual_original)
    summarize("residual_adjusted_y", residual_adjusted)
    metrics["dense_multisecant_block_details"] = block_details

    direction, solve_reason = _solve_direction(B, f)
    if solve_reason:
        metrics["dense_invalid_reason"] = solve_reason
    metrics.update({
        "dense_h0_inverse_scale": h0,
        "dense_b0_scale": b0,
        "dense_matrix_symmetry_max_abs": float(np.max(np.abs(B - B.T))),
        "dense_direction_norm": "" if direction is None else safe_norm(direction),
        "dense_direction_max_atom_norm": "" if direction is None or direction.size % 3 else safe_max_atom_norm(direction),
        "dense_reconstruction_ns": int(perf_counter_ns() - started),
    })
    if spectrum:
        metrics.update(_spectrum_metrics(B))
    metrics.update(_secant_residual_metrics(B, applied_pairs))
    return DenseBFGSResult(B, direction, metrics)


def compare_directions(reference: object, dense: object | None) -> dict[str, object]:
    """Stable comparison between a primary QN vector and dense reconstruction."""

    ref = np.asarray(reference, dtype=float).reshape(-1)
    if dense is None:
        return {
            "dense_vs_primary_cosine": "",
            "dense_vs_primary_norm_ratio": "",
            "dense_vs_primary_relative_difference": "",
        }
    den = np.asarray(dense, dtype=float).reshape(-1)
    if ref.shape != den.shape:
        return {
            "dense_vs_primary_cosine": "",
            "dense_vs_primary_norm_ratio": "",
            "dense_vs_primary_relative_difference": "",
        }
    ref_norm = safe_norm(ref)
    den_norm = safe_norm(den)
    diff_norm = safe_norm(den - ref)
    return {
        "dense_vs_primary_cosine": safe_cosine(ref, den),
        "dense_vs_primary_norm_ratio": (
            "" if ref_norm <= 1.0e-30 or not np.isfinite(ref_norm) else den_norm / ref_norm
        ),
        "dense_vs_primary_relative_difference": (
            "" if ref_norm <= 1.0e-30 or not np.isfinite(ref_norm) else diff_norm / ref_norm
        ),
    }


__all__ = [
    "DenseBFGSResult",
    "DenseSecant",
    "ShiftedTrustRegionResult",
    "compare_directions",
    "reconstruct_multisecant_bfgs",
    "reconstruct_sequential_bfgs",
    "safe_cosine",
    "safe_max_atom_norm",
    "safe_norm",
    "safe_scale_to_max_atom",
    "shifted_trust_region_solve",
    "state_blocks",
    "symmetrize_y_sella2",
]
