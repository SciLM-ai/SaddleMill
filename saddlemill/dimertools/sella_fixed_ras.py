"""Backward-compatible Sella-parity fixed-RAS API.

New production code uses :mod:`saddlemill.dimertools.ras`, which owns the
generic QN/MMF, RFO, P-RFO, bisection RAS, and adaptive-trust capability.
This module preserves the RC1 public import surface for existing campaigns and
tests.  Its restricted solves delegate to the same SaddleMill bisection root
solver; it never imports Sella at production runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Mapping

import numpy as np

from saddlemill.dimertools.ras import solve_ras_bisection as _solve_ras_bisection_generic

Array = np.ndarray

SELLA_FIXED_RAS_SCHEMA = "saddlemill_sella_fixed_ras_v1"
SELLA_REFERENCE_VERSION = "2.5.0"
# These are the exact runtime-critical reference identities already recorded by
# the baseline sella_ablation.py.  They are repeated here as provenance only.
SELLA_STEPPER_SHA256 = "7ccd629c97d66a005f43e365db678c0d1996a7eff5573264941359aa8fb528af"
SELLA_RESTRICTED_STEP_SHA256 = "df55916fb23debb2dcdd920e8ecb0c57e79984c7658fcd0a0e9e6114e44e9af3"

SELLA_AUGMENTED_REGULARIZATION = 1.0e-12
SELLA_QN_CURVATURE_REGULARIZATION = 1.0e-12
SELLA_PRFO_DEFAULT_TOLERANCE = 1.0e-15
SELLA_QN_DEFAULT_TOLERANCE = 1.0e-10
SELLA_RESTRICTED_MAXITER = 1000


class SellaFixedRASError(RuntimeError):
    """Typed native Sella-parity step failure."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = str(code)
        super().__init__(message or self.code)


@dataclass(frozen=True)
class FixedAlphaStep:
    """One deterministic step/derivative evaluation at a fixed alpha."""

    step: Array
    ds_dalpha: Array
    alpha: float
    algorithm: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("step", "ds_dalpha"):
            arr = np.asarray(getattr(self, name), dtype=float).reshape(-1).copy()
            if not np.all(np.isfinite(arr)):
                raise SellaFixedRASError("nonfinite_step_evaluation")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


@dataclass(frozen=True)
class FixedRASResult:
    """Result of one Sella-style fixed RestrictedAtomicStep solve."""

    success: bool
    status: str
    failure_reason: str
    algorithm: str
    step: Array | None
    alpha: float | None
    requested_radius: float
    achieved_radius: float | None
    boundary_active: bool
    iterations: int
    evaluations: int
    solve_time_ns: int
    last_metadata: Mapping[str, object] | None
    schema: str = SELLA_FIXED_RAS_SCHEMA

    def __post_init__(self) -> None:
        if self.step is not None:
            arr = np.asarray(self.step, dtype=float).reshape(-1).copy()
            arr.setflags(write=False)
            object.__setattr__(self, "step", arr)

    def metadata(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sella_reference_version": SELLA_REFERENCE_VERSION,
            "sella_stepper_sha256": SELLA_STEPPER_SHA256,
            "sella_restricted_step_sha256": SELLA_RESTRICTED_STEP_SHA256,
            "success": bool(self.success),
            "status": str(self.status),
            "failure_reason": str(self.failure_reason),
            "algorithm": str(self.algorithm),
            "alpha": self.alpha,
            "requested_radius": float(self.requested_radius),
            "achieved_radius": self.achieved_radius,
            "boundary_active": bool(self.boundary_active),
            "iterations": int(self.iterations),
            "evaluations": int(self.evaluations),
            "solve_time_ns": int(self.solve_time_ns),
            "last_step_metadata": None if self.last_metadata is None else dict(self.last_metadata),
        }


def _vector(value: object, label: str) -> Array:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 1 or not np.all(np.isfinite(arr)):
        raise SellaFixedRASError(f"invalid_{label}")
    return arr


def _symmetric_matrix(value: object, label: str = "matrix") -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
        raise SellaFixedRASError(f"invalid_{label}_shape")
    if not np.all(np.isfinite(arr)):
        raise SellaFixedRASError(f"nonfinite_{label}")
    return 0.5 * (arr + arr.T)


def _validate_partition(P: object, Q: object, n: int) -> tuple[Array, Array]:
    p = np.asarray(P, dtype=float)
    q = np.asarray(Q, dtype=float)
    if p.ndim != 2 or p.shape[0] != n or p.shape[1] < 1:
        raise SellaFixedRASError("invalid_p_partition")
    if q.ndim != 2 or q.shape[0] != n:
        raise SellaFixedRASError("invalid_q_partition")
    if p.shape[1] + q.shape[1] != n:
        raise SellaFixedRASError("incomplete_partition")
    v = np.column_stack((p, q))
    if not np.all(np.isfinite(v)):
        raise SellaFixedRASError("nonfinite_partition")
    if not np.allclose(v.T @ v, np.eye(n), rtol=0.0, atol=2.0e-10):
        raise SellaFixedRASError("nonorthonormal_partition")
    return p, q


def _regularize_signed(values: Array, tolerance: float, sign_hint: Array | None = None) -> tuple[Array, int]:
    out = np.asarray(values, dtype=float).copy()
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise SellaFixedRASError("invalid_regularization_tolerance")
    mask = np.abs(out) < tol
    if not np.any(mask):
        return out, 0
    signs = np.sign(out)
    if sign_hint is not None:
        hints = np.sign(np.asarray(sign_hint, dtype=float))
        signs = np.where(signs == 0.0, hints, signs)
    signs = np.where(signs == 0.0, 1.0, signs)
    out[mask] = signs[mask] * tol
    return out, int(np.count_nonzero(mask))


def sella_rfo_fixed_alpha(
    B: object,
    g: object,
    *,
    alpha: float,
    root_index: int,
    regularization: float = SELLA_AUGMENTED_REGULARIZATION,
) -> FixedAlphaStep:
    """Transcribe Sella 2.5.0 ``RationalFunctionOptimization.get_s``.

    The requested *sorted* augmented root is selected independently at every
    alpha.  There is intentionally no cross-alpha root homing.
    """

    H = _symmetric_matrix(B, "hessian")
    grad = _vector(g, "gradient")
    if grad.size != H.shape[0]:
        raise SellaFixedRASError("gradient_hessian_dimension_mismatch")
    a = float(alpha)
    if not np.isfinite(a) or a < 0.0:
        raise SellaFixedRASError("invalid_alpha")
    order = int(root_index)
    if order < 0 or order >= H.shape[0] + 1:
        raise SellaFixedRASError("invalid_augmented_root_index")

    base = np.block([[H, grad[:, None]], [grad[None, :], np.zeros((1, 1))]])
    A = base * a
    A[:-1, :-1] *= a
    evals, evecs = np.linalg.eigh(A)
    vector = np.asarray(evecs[:, order], dtype=float)
    raw_denom = float(vector[-1])
    denom = raw_denom
    denom_regularized = False
    reg = float(regularization)
    if abs(denom) < reg:
        denom = float(np.sign(denom) * reg if denom != 0.0 else reg)
        denom_regularized = True

    step = vector[:-1] * a / denom

    dA_da = base.copy()
    dA_da[:-1, :-1] *= 2.0 * a
    other_vecs = np.delete(evecs, order, axis=1)
    other_evals = np.delete(evals, order)
    gaps = other_evals - evals[order]
    # Exact Sella rule: zero belongs to the >=0 branch and becomes +1e-12.
    gaps_reg = np.where(gaps >= 0.0, np.maximum(gaps, reg), np.minimum(gaps, -reg))
    gap_regularizations = int(np.count_nonzero(np.abs(gaps) < reg))
    if other_vecs.shape[1]:
        dvec_da = other_vecs @ ((other_vecs.T @ (dA_da @ vector)) / gaps_reg)
    else:
        dvec_da = np.zeros_like(vector)
    ds_da = (
        vector[:-1] / denom
        + (a / denom) * dvec_da[:-1]
        - (vector[:-1] * a / (denom * denom)) * dvec_da[-1]
    )
    if not np.all(np.isfinite(step)) or not np.all(np.isfinite(ds_da)):
        raise SellaFixedRASError("nonfinite_rfo_step")

    return FixedAlphaStep(
        step=step,
        ds_dalpha=ds_da,
        alpha=a,
        algorithm="sella_rfo",
        metadata={
            "root_policy": "sorted_requested_root_each_alpha",
            "root_index": order,
            "root_eigenvalue": float(evals[order]),
            "raw_augmented_denominator": raw_denom,
            "used_augmented_denominator": float(denom),
            "denominator_regularized": bool(denom_regularized),
            "eigenvalue_gap_regularizations": gap_regularizations,
            "regularization": reg,
        },
    )


def sella_prfo_fixed_alpha(
    B: object,
    g: object,
    P: object,
    Q: object,
    *,
    alpha: float,
    regularization: float = SELLA_AUGMENTED_REGULARIZATION,
) -> FixedAlphaStep:
    """Sella-style P-RFO at fixed alpha in a caller-supplied P/Q basis."""

    H = _symmetric_matrix(B, "hessian")
    grad = _vector(g, "gradient")
    if grad.size != H.shape[0]:
        raise SellaFixedRASError("gradient_hessian_dimension_mismatch")
    p, q = _validate_partition(P, Q, H.shape[0])
    p_eval = sella_rfo_fixed_alpha(
        p.T @ H @ p,
        p.T @ grad,
        alpha=alpha,
        root_index=p.shape[1],
        regularization=regularization,
    )
    if q.shape[1]:
        q_eval = sella_rfo_fixed_alpha(
            q.T @ H @ q,
            q.T @ grad,
            alpha=alpha,
            root_index=0,
            regularization=regularization,
        )
        step = p @ p_eval.step + q @ q_eval.step
        ds_da = p @ p_eval.ds_dalpha + q @ q_eval.ds_dalpha
        qmeta: Mapping[str, object] | None = q_eval.metadata
    else:
        step = p @ p_eval.step
        ds_da = p @ p_eval.ds_dalpha
        qmeta = None
    return FixedAlphaStep(
        step=step,
        ds_dalpha=ds_da,
        alpha=float(alpha),
        algorithm="sella_prfo",
        metadata={
            "p_root": dict(p_eval.metadata),
            "q_root": None if qmeta is None else dict(qmeta),
            "partition_dimension_p": int(p.shape[1]),
            "partition_dimension_q": int(q.shape[1]),
        },
    )


def sella_qn_mmf_fixed_alpha(
    B: object,
    g: object,
    *,
    alpha: float,
    order: int = 1,
    curvature_regularization: float = SELLA_QN_CURVATURE_REGULARIZATION,
    eigenvalues: object | None = None,
    eigenvectors: object | None = None,
) -> FixedAlphaStep:
    """Order-controlled Sella QuasiNewton/MMF spectral step at fixed alpha.

    Sella 2.5.0 itself directly divides by ``L + alpha*sign(L)``.  This legacy
    compatibility surface retains RC1's optional tiny-curvature protection when
    explicitly requested; the generic parity path in ``ras.py`` does not add it.
    """

    H = _symmetric_matrix(B, "hessian")
    grad = _vector(g, "gradient")
    if grad.size != H.shape[0]:
        raise SellaFixedRASError("gradient_hessian_dimension_mismatch")
    ord_i = int(order)
    if ord_i < 0 or ord_i > H.shape[0]:
        raise SellaFixedRASError("invalid_qn_order")
    a = float(alpha)
    if not np.isfinite(a) or a < 0.0:
        raise SellaFixedRASError("invalid_alpha")

    if eigenvalues is None and eigenvectors is None:
        evals, evecs = np.linalg.eigh(H)
        eigensystem_source = "native_eigh"
    elif eigenvalues is not None and eigenvectors is not None:
        evals = _vector(eigenvalues, "eigenvalues")
        evecs = np.asarray(eigenvectors, dtype=float)
        if evals.size != H.shape[0] or evecs.shape != H.shape:
            raise SellaFixedRASError("invalid_supplied_eigensystem_shape")
        if not np.all(np.isfinite(evecs)):
            raise SellaFixedRASError("nonfinite_supplied_eigenvectors")
        if not np.allclose(evecs.T @ evecs, np.eye(H.shape[0]), rtol=0.0, atol=2.0e-10):
            raise SellaFixedRASError("nonorthonormal_supplied_eigenvectors")
        reconstructed = evecs @ np.diag(evals) @ evecs.T
        if not np.allclose(reconstructed, H, rtol=2.0e-10, atol=2.0e-12):
            raise SellaFixedRASError("supplied_eigensystem_does_not_match_hessian")
        eigensystem_source = "shared_physical_b_partition"
    else:
        raise SellaFixedRASError("incomplete_supplied_eigensystem")
    L = np.abs(evals)
    signs = np.ones_like(L)
    if ord_i:
        L[:ord_i] *= -1.0
        signs[:ord_i] = -1.0
    projected_g = evecs.T @ grad
    raw_denom = L + a * signs
    denom, nreg = _regularize_signed(raw_denom, curvature_regularization, signs)
    sproj = projected_g / denom
    step = -(evecs @ sproj)
    ds_da = evecs @ (sproj / denom)
    if not np.all(np.isfinite(step)) or not np.all(np.isfinite(ds_da)):
        raise SellaFixedRASError("nonfinite_qn_mmf_step")
    return FixedAlphaStep(
        step=step,
        ds_dalpha=ds_da,
        alpha=a,
        algorithm="sella_qn_mmf",
        metadata={
            "order": ord_i,
            "eigenvalues": [float(x) for x in evals],
            "signed_spectral_denominators": [float(x) for x in L],
            "alpha_signs": [float(x) for x in signs],
            "tiny_curvature_regularizations": int(nreg),
            "curvature_regularization": float(curvature_regularization),
            "selected_unstable_index": 0 if ord_i else None,
            "selected_unstable_eigenvalue": float(evals[0]) if ord_i else None,
            "eigensystem_source": eigensystem_source,
        },
    )


def restricted_atomic_constraint(
    cartesian_step: object,
    cartesian_ds_dalpha: object | None = None,
) -> float | tuple[float, float, int]:
    """Sella RestrictedAtomicStep max-per-atom constraint and derivative."""

    step = np.asarray(cartesian_step, dtype=float).reshape(-1)
    if step.size < 3 or step.size % 3:
        raise SellaFixedRASError("restricted_atomic_step_requires_3n_cartesian")
    if not np.all(np.isfinite(step)):
        raise SellaFixedRASError("nonfinite_cartesian_step")
    matrix = step.reshape((-1, 3))
    norms = np.linalg.norm(matrix, axis=1)
    index = int(np.argmax(norms))
    value = float(norms[index])
    if cartesian_ds_dalpha is None:
        return value
    derivative = np.asarray(cartesian_ds_dalpha, dtype=float).reshape(-1)
    if derivative.shape != step.shape or not np.all(np.isfinite(derivative)):
        raise SellaFixedRASError("invalid_cartesian_step_derivative")
    dmatrix = derivative.reshape((-1, 3))
    dvalue = float(dmatrix[index] @ matrix[index] / max(value, 1.0e-12))
    return value, dvalue, index


def _solve_fixed_ras(
    stepper: Callable[[float], FixedAlphaStep],
    lift: Callable[[Array], Array],
    *,
    radius: float,
    algorithm: str,
    alpha0: float,
    alphamin: float,
    alphamax: float,
    slope: float,
    newton_safe: bool,
    tolerance: float,
    max_iterations: int,
) -> FixedRASResult:
    # ``newton_safe`` is retained only for signature compatibility. Production
    # scalar solving is always the generic finite-bracket bisection path.
    _ = bool(newton_safe)
    result = _solve_ras_bisection_generic(
        stepper,
        lift,
        radius=radius,
        algorithm=algorithm,
        alpha0=alpha0,
        alphamin=alphamin,
        alphamax=alphamax,
        slope=slope,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return FixedRASResult(
        success=result.success,
        status=result.status,
        failure_reason=result.failure_reason,
        algorithm=result.algorithm,
        step=result.step,
        alpha=result.alpha,
        requested_radius=result.requested_radius,
        achieved_radius=result.achieved_radius,
        boundary_active=result.boundary_active,
        iterations=result.iterations,
        evaluations=result.evaluations,
        solve_time_ns=result.solve_time_ns,
        last_metadata=result.last_metadata,
    )

def solve_sella_prfo_fixed_ras(
    B: object,
    g: object,
    P: object,
    Q: object,
    *,
    lift: Callable[[Array], Array],
    radius: float,
    tolerance: float | None = None,
    max_iterations: int = SELLA_RESTRICTED_MAXITER,
    regularization: float = SELLA_AUGMENTED_REGULARIZATION,
) -> FixedRASResult:
    """Fixed-radius Sella-style P-RFO + RestrictedAtomicStep."""

    H = _symmetric_matrix(B, "hessian")
    grad = _vector(g, "gradient")
    p, q = _validate_partition(P, Q, H.shape[0])
    tol = SELLA_PRFO_DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    return _solve_fixed_ras(
        lambda alpha: sella_prfo_fixed_alpha(
            H, grad, p, q, alpha=alpha, regularization=regularization
        ),
        lift,
        radius=radius,
        algorithm="sella_fixed_ras_prfo",
        alpha0=1.0,
        alphamin=0.0,
        alphamax=1.0,
        slope=1.0,
        newton_safe=False,
        tolerance=tol,
        max_iterations=max_iterations,
    )


def solve_sella_qn_mmf_fixed_ras(
    B: object,
    g: object,
    *,
    lift: Callable[[Array], Array],
    radius: float,
    order: int = 1,
    tolerance: float | None = None,
    max_iterations: int = SELLA_RESTRICTED_MAXITER,
    curvature_regularization: float = SELLA_QN_CURVATURE_REGULARIZATION,
    eigenvalues: object | None = None,
    eigenvectors: object | None = None,
) -> FixedRASResult:
    """Fixed-radius Sella-style order-controlled QN/MMF + RestrictedAtomicStep."""

    H = _symmetric_matrix(B, "hessian")
    grad = _vector(g, "gradient")
    tol = SELLA_QN_DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    return _solve_fixed_ras(
        lambda alpha: sella_qn_mmf_fixed_alpha(
            H,
            grad,
            alpha=alpha,
            order=order,
            curvature_regularization=curvature_regularization,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        ),
        lift,
        radius=radius,
        algorithm="sella_fixed_ras_qn_mmf",
        alpha0=0.0,
        alphamin=0.0,
        alphamax=np.inf,
        slope=-1.0,
        newton_safe=True,
        tolerance=tol,
        max_iterations=max_iterations,
    )


__all__ = [
    "FixedAlphaStep",
    "FixedRASResult",
    "SELLA_AUGMENTED_REGULARIZATION",
    "SELLA_FIXED_RAS_SCHEMA",
    "SELLA_PRFO_DEFAULT_TOLERANCE",
    "SELLA_QN_CURVATURE_REGULARIZATION",
    "SELLA_QN_DEFAULT_TOLERANCE",
    "SELLA_REFERENCE_VERSION",
    "SELLA_RESTRICTED_MAXITER",
    "SellaFixedRASError",
    "restricted_atomic_constraint",
    "sella_prfo_fixed_alpha",
    "sella_qn_mmf_fixed_alpha",
    "sella_rfo_fixed_alpha",
    "solve_sella_prfo_fixed_ras",
    "solve_sella_qn_mmf_fixed_ras",
]
