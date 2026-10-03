"""Generic SaddleMill restricted-atomic-step steppers and adaptive trust.

This module owns the production Restricted Atomic Step (RAS) algebra used by
SaddleMill QN/MMF, RFO, and P-RFO translation.  Sella 2.5.0 is the numerical
parity reference for the fixed-alpha steppers and outer adaptive trust policy,
but production execution does not import Sella or its private optimizer objects.

Scientific contracts preserved here:

* RAS is ``max_i ||s_i|| <= delta`` in Cartesian atom blocks.
* QN/MMF, RFO, and P-RFO fixed-alpha equations/root selection match Sella 2.5.0.
* A feasible unrestricted step is returned unchanged.
* An active RAS boundary is solved algebraically by a finite bracket plus
  bracketed bisection; no post-hoc uniform scaling substitutes for that solve.
* The adaptive trust update exactly follows Sella 2.5.0 saddle defaults and
  energy-ratio semantics.  It never rejects/rolls back an accepted step.
* The scalar solve and trust update perform no PES, force, HVP, Hessian, or
  minimum-mode evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Mapping

import numpy as np

Array = np.ndarray

RAS_SCHEMA = "saddlemill_ras_v1"
ADAPTIVE_RAS_TRUST_SCHEMA = "saddlemill_adaptive_ras_trust_v1"
SELLA_REFERENCE_VERSION = "2.5.0"
SELLA_STEPPER_SHA256 = "7ccd629c97d66a005f43e365db678c0d1996a7eff5573264941359aa8fb528af"
SELLA_RESTRICTED_STEP_SHA256 = "df55916fb23debb2dcdd920e8ecb0c57e79984c7658fcd0a0e9e6114e44e9af3"
SELLA_OPTIMIZE_SHA256 = "be36b7530b106f02771605d2d3f8a9df11e11e7451898374d693ce4c6b5801b3"
SELLA_PESWRAPPER_SHA256 = "3d0e4a8140e73e0df37e80878c2d2d4290ec26855c42852312594b1be5433020"

SELLA_AUGMENTED_REGULARIZATION = 1.0e-12
SELLA_PRFO_DEFAULT_TOLERANCE = 1.0e-15
SELLA_RFO_DEFAULT_TOLERANCE = 1.0e-15
SELLA_QN_DEFAULT_TOLERANCE = 1.0e-10
RAS_DEFAULT_MAXITER = 1000

# Sella 2.5.0 saddle defaults from optimize/optimize.py::_default_kwargs['saddle'].
SELLA_SADDLE_DELTA0 = 0.1
SELLA_SADDLE_SIGMA_INC = 1.15
SELLA_SADDLE_SIGMA_DEC = 0.65
SELLA_SADDLE_RHO_INC = 1.035
SELLA_SADDLE_RHO_DEC = 5.0
SELLA_SADDLE_ETA = 1.0e-4
SELLA_PREDICTED_CHANGE_ZERO_TOLERANCE = 1.0e-14


class RASError(RuntimeError):
    """Typed production RAS/trust failure."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = str(code)
        super().__init__(message or self.code)


@dataclass(frozen=True)
class FixedAlphaStep:
    """One deterministic fixed-alpha translation step evaluation."""

    step: Array
    ds_dalpha: Array
    alpha: float
    algorithm: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("step", "ds_dalpha"):
            arr = np.asarray(getattr(self, name), dtype=float).reshape(-1).copy()
            if not np.all(np.isfinite(arr)):
                raise RASError("nonfinite_step_evaluation")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


@dataclass(frozen=True)
class RASResult:
    """One algebraic RAS solve result."""

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
    root_solver: str = "bisection"
    schema: str = RAS_SCHEMA

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
            "root_solver": str(self.root_solver),
            "alpha": self.alpha,
            "requested_radius": float(self.requested_radius),
            "achieved_radius": self.achieved_radius,
            "boundary_active": bool(self.boundary_active),
            "iterations": int(self.iterations),
            "evaluations": int(self.evaluations),
            "solve_time_ns": int(self.solve_time_ns),
            "last_step_metadata": None if self.last_metadata is None else dict(self.last_metadata),
            "added_pes_calls": 0,
            "added_force_calls": 0,
            "added_hvp_calls": 0,
            "added_hessian_calls": 0,
            "added_minmode_calls": 0,
        }


def _vector(value: object, label: str) -> Array:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 1 or not np.all(np.isfinite(arr)):
        raise RASError(f"invalid_{label}")
    return arr


def _symmetric_matrix(value: object, label: str = "matrix") -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
        raise RASError(f"invalid_{label}_shape")
    if not np.all(np.isfinite(arr)):
        raise RASError(f"nonfinite_{label}")
    return 0.5 * (arr + arr.T)


def _validate_partition(P: object, Q: object, n: int) -> tuple[Array, Array]:
    p = np.asarray(P, dtype=float)
    q = np.asarray(Q, dtype=float)
    if p.ndim != 2 or p.shape[0] != n or p.shape[1] < 1:
        raise RASError("invalid_p_partition")
    if q.ndim != 2 or q.shape[0] != n:
        raise RASError("invalid_q_partition")
    if p.shape[1] + q.shape[1] != n:
        raise RASError("incomplete_partition")
    v = np.column_stack((p, q))
    if not np.all(np.isfinite(v)):
        raise RASError("nonfinite_partition")
    if not np.allclose(v.T @ v, np.eye(n), rtol=0.0, atol=2.0e-10):
        raise RASError("nonorthonormal_partition")
    return p, q


def rfo_fixed_alpha(
    B: object,
    g: object,
    *,
    alpha: float,
    root_index: int,
    regularization: float = SELLA_AUGMENTED_REGULARIZATION,
) -> FixedAlphaStep:
    """Sella-2.5.0-parity RFO fixed-alpha augmented-root step."""

    H = _symmetric_matrix(B, "hessian")
    grad = _vector(g, "gradient")
    if grad.size != H.shape[0]:
        raise RASError("gradient_hessian_dimension_mismatch")
    a = float(alpha)
    if not np.isfinite(a) or a < 0.0:
        raise RASError("invalid_alpha")
    order = int(root_index)
    if order < 0 or order >= H.shape[0] + 1:
        raise RASError("invalid_augmented_root_index")
    reg = float(regularization)
    if not np.isfinite(reg) or reg <= 0.0:
        raise RASError("invalid_augmented_regularization")

    base = np.block([[H, grad[:, None]], [grad[None, :], np.zeros((1, 1))]])
    A = base * a
    A[:-1, :-1] *= a
    evals, evecs = np.linalg.eigh(A)
    vector = np.asarray(evecs[:, order], dtype=float)
    raw_denom = float(vector[-1])
    denom = raw_denom
    denom_regularized = False
    if abs(denom) < reg:
        denom = float(np.sign(denom) * reg if denom != 0.0 else reg)
        denom_regularized = True
    step = vector[:-1] * a / denom

    dA_da = base.copy()
    dA_da[:-1, :-1] *= 2.0 * a
    other_vecs = np.delete(evecs, order, axis=1)
    other_evals = np.delete(evals, order)
    gaps = other_evals - evals[order]
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
        raise RASError("nonfinite_rfo_step")
    return FixedAlphaStep(
        step=step,
        ds_dalpha=ds_da,
        alpha=a,
        algorithm="rfo",
        metadata={
            "parity_reference": "sella_2.5.0",
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


def prfo_fixed_alpha(
    B: object,
    g: object,
    P: object,
    Q: object,
    *,
    alpha: float,
    regularization: float = SELLA_AUGMENTED_REGULARIZATION,
) -> FixedAlphaStep:
    """Sella-2.5.0-parity P-RFO at fixed alpha for a supplied B partition."""

    H = _symmetric_matrix(B, "hessian")
    grad = _vector(g, "gradient")
    if grad.size != H.shape[0]:
        raise RASError("gradient_hessian_dimension_mismatch")
    p, q = _validate_partition(P, Q, H.shape[0])
    p_eval = rfo_fixed_alpha(
        p.T @ H @ p,
        p.T @ grad,
        alpha=alpha,
        root_index=p.shape[1],
        regularization=regularization,
    )
    if q.shape[1]:
        q_eval = rfo_fixed_alpha(
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
        algorithm="prfo",
        metadata={
            "parity_reference": "sella_2.5.0",
            "p_root": dict(p_eval.metadata),
            "q_root": None if qmeta is None else dict(qmeta),
            "partition_dimension_p": int(p.shape[1]),
            "partition_dimension_q": int(q.shape[1]),
        },
    )


def qn_mmf_fixed_alpha(
    B: object,
    g: object,
    *,
    alpha: float,
    order: int = 1,
    eigenvalues: object | None = None,
    eigenvectors: object | None = None,
) -> FixedAlphaStep:
    """Exact Sella-2.5.0 QuasiNewton/MMF fixed-alpha spectral step.

    No curvature floor/regularization is added here.  Sella 2.5.0 directly
    divides by ``L + alpha*ones``; preserving that equation is part of the
    fixed-alpha parity contract.
    """

    H = _symmetric_matrix(B, "hessian")
    grad = _vector(g, "gradient")
    if grad.size != H.shape[0]:
        raise RASError("gradient_hessian_dimension_mismatch")
    ord_i = int(order)
    if ord_i < 0 or ord_i > H.shape[0]:
        raise RASError("invalid_qn_order")
    a = float(alpha)
    if not np.isfinite(a) or a < 0.0:
        raise RASError("invalid_alpha")

    if eigenvalues is None and eigenvectors is None:
        evals, evecs = np.linalg.eigh(H)
        eigensystem_source = "native_eigh"
    elif eigenvalues is not None and eigenvectors is not None:
        evals = _vector(eigenvalues, "eigenvalues")
        evecs = np.asarray(eigenvectors, dtype=float)
        if evals.size != H.shape[0] or evecs.shape != H.shape:
            raise RASError("invalid_supplied_eigensystem_shape")
        if not np.all(np.isfinite(evecs)):
            raise RASError("nonfinite_supplied_eigenvectors")
        if not np.allclose(evecs.T @ evecs, np.eye(H.shape[0]), rtol=0.0, atol=2.0e-10):
            raise RASError("nonorthonormal_supplied_eigenvectors")
        reconstructed = evecs @ np.diag(evals) @ evecs.T
        if not np.allclose(reconstructed, H, rtol=2.0e-10, atol=2.0e-12):
            raise RASError("supplied_eigensystem_does_not_match_hessian")
        eigensystem_source = "shared_physical_b_partition"
    else:
        raise RASError("incomplete_supplied_eigensystem")

    L = np.abs(evals)
    signs = np.ones_like(L)
    if ord_i:
        L[:ord_i] *= -1.0
        signs[:ord_i] = -1.0
    projected_g = evecs.T @ grad
    denom = L + a * signs
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sproj = projected_g / denom
        step = -(evecs @ sproj)
        ds_da = evecs @ (sproj / denom)
    if not np.all(np.isfinite(step)) or not np.all(np.isfinite(ds_da)):
        raise RASError("nonfinite_qn_mmf_step")
    return FixedAlphaStep(
        step=step,
        ds_dalpha=ds_da,
        alpha=a,
        algorithm="qn_mmf",
        metadata={
            "parity_reference": "sella_2.5.0",
            "order": ord_i,
            "eigenvalues": [float(x) for x in evals],
            "signed_spectral_denominators": [float(x) for x in L],
            "alpha_signs": [float(x) for x in signs],
            "selected_unstable_index": 0 if ord_i else None,
            "selected_unstable_eigenvalue": float(evals[0]) if ord_i else None,
            "eigensystem_source": eigensystem_source,
            "curvature_regularization": None,
        },
    )


def restricted_atomic_constraint(
    cartesian_step: object,
    cartesian_ds_dalpha: object | None = None,
) -> float | tuple[float, float, int]:
    """RAS max-per-atom constraint and optional derivative."""

    step = np.asarray(cartesian_step, dtype=float).reshape(-1)
    if step.size < 3 or step.size % 3:
        raise RASError("restricted_atomic_step_requires_3n_cartesian")
    if not np.all(np.isfinite(step)):
        raise RASError("nonfinite_cartesian_step")
    matrix = step.reshape((-1, 3))
    norms = np.linalg.norm(matrix, axis=1)
    index = int(np.argmax(norms))
    value = float(norms[index])
    if cartesian_ds_dalpha is None:
        return value
    derivative = np.asarray(cartesian_ds_dalpha, dtype=float).reshape(-1)
    if derivative.shape != step.shape or not np.all(np.isfinite(derivative)):
        raise RASError("invalid_cartesian_step_derivative")
    dmatrix = derivative.reshape((-1, 3))
    dvalue = float(dmatrix[index] @ matrix[index] / max(value, 1.0e-12))
    return value, dvalue, index


def _failure(
    *,
    start_ns: int,
    status: str,
    reason: str,
    algorithm: str,
    radius: float,
    alpha: float | None,
    achieved: float | None,
    boundary_active: bool,
    iterations: int,
    evaluations: int,
    metadata: Mapping[str, object] | None,
) -> RASResult:
    return RASResult(
        success=False,
        status=status,
        failure_reason=reason,
        algorithm=algorithm,
        step=None,
        alpha=alpha,
        requested_radius=radius,
        achieved_radius=achieved,
        boundary_active=boundary_active,
        iterations=iterations,
        evaluations=evaluations,
        solve_time_ns=perf_counter_ns() - start_ns,
        last_metadata=metadata,
    )


def solve_ras_bisection(
    stepper: Callable[[float], FixedAlphaStep],
    lift: Callable[[Array], Array],
    *,
    radius: float,
    algorithm: str,
    alpha0: float,
    alphamin: float,
    alphamax: float,
    slope: float,
    tolerance: float,
    max_iterations: int = RAS_DEFAULT_MAXITER,
    max_bracket_expansions: int = 128,
) -> RASResult:
    """Solve the RAS boundary using only fixed-alpha algebra and bisection.

    ``slope`` retains the Sella stepper convention: +1 means the constraint
    increases with alpha (RFO/P-RFO), -1 means it decreases (QN/MMF).
    For an infinite endpoint a finite bracket is established by deterministic
    geometric expansion before bisection begins.
    """

    start = perf_counter_ns()
    delta = float(radius)
    tol = float(tolerance)
    maxiter = int(max_iterations)
    expansions = int(max_bracket_expansions)
    if not np.isfinite(delta) or delta <= 0.0:
        raise RASError("invalid_ras_radius")
    if not np.isfinite(tol) or tol <= 0.0:
        raise RASError("invalid_ras_tolerance")
    if maxiter < 1:
        raise RASError("invalid_ras_max_iterations")
    if expansions < 1:
        raise RASError("invalid_ras_bracket_expansions")
    if float(slope) not in {-1.0, 1.0}:
        raise RASError("invalid_ras_stepper_slope")

    evaluations = 0

    def evaluate(alpha: float) -> tuple[FixedAlphaStep, float, int, dict[str, object]]:
        nonlocal evaluations
        evaluated = stepper(float(alpha))
        cart = np.asarray(lift(evaluated.step), dtype=float).reshape(-1)
        value, _, atom = restricted_atomic_constraint(
            cart, np.asarray(lift(evaluated.ds_dalpha), dtype=float).reshape(-1)
        )
        evaluations += 1
        meta = {
            **dict(evaluated.metadata),
            "ras_active_atom_index": int(atom),
            "ras_error": float(value - delta),
        }
        return evaluated, float(value), int(atom), meta

    try:
        initial, initial_value, _, initial_meta = evaluate(float(alpha0))
    except (FloatingPointError, np.linalg.LinAlgError, RASError) as exc:
        return _failure(
            start_ns=start,
            status="initial_evaluation_failed",
            reason=getattr(exc, "code", type(exc).__name__),
            algorithm=algorithm,
            radius=delta,
            alpha=float(alpha0),
            achieved=None,
            boundary_active=False,
            iterations=0,
            evaluations=evaluations,
            metadata=None,
        )

    # Preserve Sella's strict feasibility condition: val < delta returns the
    # unrestricted step unchanged. Equality is treated as an active boundary.
    if initial_value < delta:
        return RASResult(
            True,
            "inside_radius_unrestricted",
            "",
            algorithm,
            initial.step,
            float(alpha0),
            delta,
            initial_value,
            False,
            0,
            evaluations,
            perf_counter_ns() - start,
            initial_meta,
        )

    if float(slope) > 0.0:
        # RFO/P-RFO: unrestricted alpha0=1 is outside; alpha=0 is the finite
        # lower endpoint and should be inside for the same fixed-alpha equation.
        low_alpha = float(alphamin)
        high_alpha = float(alpha0)
        try:
            low_step, low_value, _, low_meta = evaluate(low_alpha)
        except (FloatingPointError, np.linalg.LinAlgError, RASError) as exc:
            return _failure(
                start_ns=start,
                status="bracket_evaluation_failed",
                reason=getattr(exc, "code", type(exc).__name__),
                algorithm=algorithm,
                radius=delta,
                alpha=low_alpha,
                achieved=None,
                boundary_active=True,
                iterations=0,
                evaluations=evaluations,
                metadata=initial_meta,
            )
        high_step, high_value, high_meta = initial, initial_value, initial_meta
        if low_value > delta + tol:
            return _failure(
                start_ns=start,
                status="bracket_not_found",
                reason="ras_lower_endpoint_outside_radius",
                algorithm=algorithm,
                radius=delta,
                alpha=low_alpha,
                achieved=low_value,
                boundary_active=True,
                iterations=0,
                evaluations=evaluations,
                metadata=low_meta,
            )
    else:
        # QN/MMF: unrestricted alpha0=0 is outside. Search a finite high alpha
        # at which the same algebraic step lies inside the requested radius.
        low_alpha = float(alpha0)
        low_step, low_value, low_meta = initial, initial_value, initial_meta
        high_alpha = 1.0
        if np.isfinite(alphamax):
            high_alpha = min(high_alpha, float(alphamax))
        high_step = None
        high_value = None
        high_meta = None
        for _ in range(expansions):
            if high_alpha <= low_alpha:
                high_alpha = np.nextafter(low_alpha, np.inf)
            if np.isfinite(alphamax) and high_alpha > float(alphamax):
                high_alpha = float(alphamax)
            try:
                candidate, value, _, meta = evaluate(high_alpha)
            except (FloatingPointError, np.linalg.LinAlgError, RASError) as exc:
                return _failure(
                    start_ns=start,
                    status="bracket_evaluation_failed",
                    reason=getattr(exc, "code", type(exc).__name__),
                    algorithm=algorithm,
                    radius=delta,
                    alpha=high_alpha,
                    achieved=None,
                    boundary_active=True,
                    iterations=0,
                    evaluations=evaluations,
                    metadata=low_meta,
                )
            high_step, high_value, high_meta = candidate, value, meta
            if value <= delta:
                break
            if np.isfinite(alphamax) and high_alpha >= float(alphamax):
                break
            next_alpha = high_alpha * 2.0 if high_alpha > 0.0 else 1.0
            if not np.isfinite(next_alpha):
                break
            high_alpha = next_alpha
        if high_step is None or high_value is None or high_meta is None or high_value > delta:
            return _failure(
                start_ns=start,
                status="bracket_not_found",
                reason="finite_ras_bracket_not_found",
                algorithm=algorithm,
                radius=delta,
                alpha=high_alpha,
                achieved=high_value,
                boundary_active=True,
                iterations=0,
                evaluations=evaluations,
                metadata=high_meta or low_meta,
            )

    # At this point low is inside and high is outside for slope +1; for slope
    # -1 low is outside and high is inside. Bisection preserves that bracket.
    current = initial
    value = initial_value
    alpha = float(alpha0)
    last_meta = initial_meta
    for iteration in range(1, maxiter + 1):
        if np.nextafter(low_alpha, high_alpha) >= high_alpha:
            # Machine-limit bracket collapse: always select the feasible endpoint
            # so a successful RAS result can never violate max_i ||s_i|| <= delta.
            if float(slope) > 0.0:
                alpha, current, value, last_meta = low_alpha, low_step, low_value, low_meta
            else:
                alpha, current, value, last_meta = high_alpha, high_step, high_value, high_meta
            return RASResult(
                True,
                "boundary_machine_limit",
                "",
                algorithm,
                current.step,
                float(alpha),
                delta,
                float(value),
                True,
                iteration - 1,
                evaluations,
                perf_counter_ns() - start,
                {
                    **dict(last_meta),
                    "ras_lower_alpha": float(low_alpha),
                    "ras_upper_alpha": float(high_alpha),
                },
            )

        alpha = float(low_alpha + (high_alpha - low_alpha) / 2.0)
        try:
            current, value, _, last_meta = evaluate(alpha)
        except (FloatingPointError, np.linalg.LinAlgError, RASError) as exc:
            return _failure(
                start_ns=start,
                status="boundary_evaluation_failed",
                reason=getattr(exc, "code", type(exc).__name__),
                algorithm=algorithm,
                radius=delta,
                alpha=alpha,
                achieved=None,
                boundary_active=True,
                iterations=iteration,
                evaluations=evaluations,
                metadata=last_meta,
            )
        err = value - delta
        last_meta = {
            **dict(last_meta),
            "ras_lower_alpha": float(low_alpha),
            "ras_upper_alpha": float(high_alpha),
        }
        if abs(err) <= tol:
            return RASResult(
                True,
                "boundary_converged",
                "",
                algorithm,
                current.step,
                alpha,
                delta,
                value,
                True,
                iteration,
                evaluations,
                perf_counter_ns() - start,
                last_meta,
            )
        if float(slope) > 0.0:
            if value > delta:
                high_alpha, high_step, high_value, high_meta = alpha, current, value, last_meta
            else:
                low_alpha, low_step, low_value, low_meta = alpha, current, value, last_meta
        else:
            if value > delta:
                low_alpha, low_step, low_value, low_meta = alpha, current, value, last_meta
            else:
                high_alpha, high_step, high_value, high_meta = alpha, current, value, last_meta

    return _failure(
        start_ns=start,
        status="max_iterations",
        reason="restricted_step_failed_to_converge",
        algorithm=algorithm,
        radius=delta,
        alpha=alpha,
        achieved=value,
        boundary_active=True,
        iterations=maxiter,
        evaluations=evaluations,
        metadata=last_meta,
    )


def solve_prfo_ras(
    B: object,
    g: object,
    P: object,
    Q: object,
    *,
    lift: Callable[[Array], Array],
    radius: float,
    tolerance: float | None = None,
    max_iterations: int = RAS_DEFAULT_MAXITER,
    root_solver: str = "bisection",
) -> RASResult:
    if str(root_solver).strip().lower() != "bisection":
        raise RASError("unsupported_ras_root_solver")
    H = _symmetric_matrix(B, "hessian")
    grad = _vector(g, "gradient")
    p, q = _validate_partition(P, Q, H.shape[0])
    tol = SELLA_PRFO_DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    return solve_ras_bisection(
        lambda alpha: prfo_fixed_alpha(H, grad, p, q, alpha=alpha),
        lift,
        radius=radius,
        algorithm="prfo",
        alpha0=1.0,
        alphamin=0.0,
        alphamax=1.0,
        slope=1.0,
        tolerance=tol,
        max_iterations=max_iterations,
    )


def solve_rfo_ras(
    B: object,
    g: object,
    *,
    lift: Callable[[Array], Array],
    radius: float,
    order: int = 1,
    tolerance: float | None = None,
    max_iterations: int = RAS_DEFAULT_MAXITER,
    root_solver: str = "bisection",
) -> RASResult:
    if str(root_solver).strip().lower() != "bisection":
        raise RASError("unsupported_ras_root_solver")
    H = _symmetric_matrix(B, "hessian")
    grad = _vector(g, "gradient")
    root_index = int(order)
    if root_index < 0 or root_index >= H.shape[0] + 1:
        raise RASError("invalid_rfo_order")
    tol = SELLA_RFO_DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    return solve_ras_bisection(
        lambda alpha: rfo_fixed_alpha(H, grad, alpha=alpha, root_index=root_index),
        lift,
        radius=radius,
        algorithm="rfo",
        alpha0=1.0,
        alphamin=0.0,
        alphamax=1.0,
        slope=1.0,
        tolerance=tol,
        max_iterations=max_iterations,
    )


def solve_qn_mmf_ras(
    B: object,
    g: object,
    *,
    lift: Callable[[Array], Array],
    radius: float,
    order: int = 1,
    tolerance: float | None = None,
    max_iterations: int = RAS_DEFAULT_MAXITER,
    root_solver: str = "bisection",
    eigenvalues: object | None = None,
    eigenvectors: object | None = None,
) -> RASResult:
    if str(root_solver).strip().lower() != "bisection":
        raise RASError("unsupported_ras_root_solver")
    H = _symmetric_matrix(B, "hessian")
    grad = _vector(g, "gradient")
    tol = SELLA_QN_DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    return solve_ras_bisection(
        lambda alpha: qn_mmf_fixed_alpha(
            H,
            grad,
            alpha=alpha,
            order=order,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        ),
        lift,
        radius=radius,
        algorithm="qn_mmf",
        alpha0=0.0,
        alphamin=0.0,
        alphamax=np.inf,
        slope=-1.0,
        tolerance=tol,
        max_iterations=max_iterations,
    )


@dataclass
class AdaptiveRASTrustController:
    """SaddleMill-owned transcription of Sella 2.5.0 saddle RAS trust."""

    delta: float = SELLA_SADDLE_DELTA0
    sigma_inc: float = SELLA_SADDLE_SIGMA_INC
    sigma_dec: float = SELLA_SADDLE_SIGMA_DEC
    rho_inc: float = SELLA_SADDLE_RHO_INC
    rho_dec: float = SELLA_SADDLE_RHO_DEC
    delta_min: float = SELLA_SADDLE_ETA
    rho: float = 1.0
    updates: int = 0

    def __post_init__(self) -> None:
        for name in ("delta", "sigma_inc", "sigma_dec", "rho_inc", "rho_dec", "delta_min", "rho"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise RASError(f"invalid_adaptive_trust_{name}")
            setattr(self, name, value)
        if self.delta <= 0.0 or self.delta_min <= 0.0:
            raise RASError("invalid_adaptive_trust_radius")
        if self.sigma_inc <= 0.0 or self.sigma_dec <= 0.0 or self.rho_inc <= 0.0 or self.rho_dec <= 0.0:
            raise RASError("invalid_adaptive_trust_factor")
        if self.rho_inc <= 1.0 or self.rho_dec <= 1.0:
            raise RASError("invalid_adaptive_trust_rho_threshold")
        self.updates = int(self.updates)
        if self.updates < 0:
            raise RASError("invalid_adaptive_trust_update_count")

    @staticmethod
    def predicted_change(g: object, B: object, step: object) -> float:
        grad = _vector(g, "gradient")
        H = _symmetric_matrix(B, "hessian")
        dx = _vector(step, "step")
        if grad.size != H.shape[0] or dx.size != grad.size:
            raise RASError("adaptive_trust_dimension_mismatch")
        return float(grad.T @ dx + 0.5 * dx.T @ H @ dx)

    def update(
        self,
        *,
        actual_change: float,
        predicted_change: float,
        step_magnitude: float,
    ) -> dict[str, object]:
        """Apply Sella's post-step trust update; the accepted step is never undone."""

        actual = float(actual_change)
        predicted = float(predicted_change)
        smag = float(step_magnitude)
        if not np.isfinite(actual) or not np.isfinite(predicted):
            raise RASError("nonfinite_adaptive_trust_energy_change")
        if not np.isfinite(smag) or smag < 0.0:
            raise RASError("invalid_adaptive_trust_step_magnitude")
        delta_before = float(self.delta)
        if abs(predicted) < SELLA_PREDICTED_CHANGE_ZERO_TOLERANCE:
            ratio = None
            self.rho = 1.0
            action = "unchanged_predicted_change_too_small"
        else:
            ratio = float(actual / predicted)
            if ratio < 1.0 / self.rho_dec or ratio > self.rho_dec:
                self.delta = max(smag * self.sigma_dec, self.delta_min)
                action = "shrink"
            elif 1.0 / self.rho_inc < ratio < self.rho_inc:
                self.delta = max(self.sigma_inc * smag, self.delta)
                action = "grow"
            else:
                action = "unchanged"
            self.rho = ratio
        self.updates += 1
        return {
            "schema": ADAPTIVE_RAS_TRUST_SCHEMA,
            "sella_reference_version": SELLA_REFERENCE_VERSION,
            "sella_optimize_sha256": SELLA_OPTIMIZE_SHA256,
            "sella_peswrapper_sha256": SELLA_PESWRAPPER_SHA256,
            "acceptance_policy": "accept_step_no_rollback",
            "step_accepted": True,
            "actual_change": actual,
            "predicted_change": predicted,
            "rho": ratio,
            "rho_state": float(self.rho),
            "step_magnitude": smag,
            "delta_before": delta_before,
            "delta_after": float(self.delta),
            "action": action,
            "update_index": int(self.updates),
            "added_pes_calls": 0,
            "added_force_calls": 0,
            "added_hvp_calls": 0,
            "added_hessian_calls": 0,
            "added_minmode_calls": 0,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": ADAPTIVE_RAS_TRUST_SCHEMA,
            "parity_reference": "sella_2.5.0_saddle_ras",
            "delta": float(self.delta),
            "sigma_inc": float(self.sigma_inc),
            "sigma_dec": float(self.sigma_dec),
            "rho_inc": float(self.rho_inc),
            "rho_dec": float(self.rho_dec),
            "delta_min": float(self.delta_min),
            "rho": float(self.rho),
            "updates": int(self.updates),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "AdaptiveRASTrustController":
        if state.get("schema") != ADAPTIVE_RAS_TRUST_SCHEMA:
            raise RASError("unsupported_adaptive_trust_state_schema")
        return cls(
            delta=float(state["delta"]),
            sigma_inc=float(state["sigma_inc"]),
            sigma_dec=float(state["sigma_dec"]),
            rho_inc=float(state["rho_inc"]),
            rho_dec=float(state["rho_dec"]),
            delta_min=float(state["delta_min"]),
            rho=float(state.get("rho", 1.0)),
            updates=int(state.get("updates", 0)),
        )


__all__ = [
    "ADAPTIVE_RAS_TRUST_SCHEMA",
    "AdaptiveRASTrustController",
    "FixedAlphaStep",
    "RASResult",
    "RASError",
    "RAS_DEFAULT_MAXITER",
    "RAS_SCHEMA",
    "SELLA_AUGMENTED_REGULARIZATION",
    "SELLA_OPTIMIZE_SHA256",
    "SELLA_PESWRAPPER_SHA256",
    "SELLA_PRFO_DEFAULT_TOLERANCE",
    "SELLA_QN_DEFAULT_TOLERANCE",
    "SELLA_REFERENCE_VERSION",
    "SELLA_RESTRICTED_STEP_SHA256",
    "SELLA_RFO_DEFAULT_TOLERANCE",
    "SELLA_SADDLE_DELTA0",
    "SELLA_SADDLE_ETA",
    "SELLA_SADDLE_RHO_DEC",
    "SELLA_SADDLE_RHO_INC",
    "SELLA_SADDLE_SIGMA_DEC",
    "SELLA_SADDLE_SIGMA_INC",
    "SELLA_STEPPER_SHA256",
    "prfo_fixed_alpha",
    "qn_mmf_fixed_alpha",
    "restricted_atomic_constraint",
    "rfo_fixed_alpha",
    "solve_prfo_ras",
    "solve_qn_mmf_ras",
    "solve_ras_bisection",
    "solve_rfo_ras",
]
