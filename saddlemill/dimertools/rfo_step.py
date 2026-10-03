"""Native rational-function step kernels for SaddleMill.

This module is calculator-free and Sella-free.  It implements the augmented
RFO convention used by the RFO/P-RFO scientific contract::

    A(alpha) = [[alpha**2 B, alpha g],
                [alpha g.T,         0]]

For a normalized augmented eigenvector ``[z; c]`` with eigenvalue ``lambda``,
the returned step is ``s = alpha z / c``.  Substitution into the augmented
eigenproblem gives the stationary equations

    B s + g = (lambda / alpha**2) s
    lambda = g.T s

when ``c != 0``.  These equations are checked numerically for every successful
root result.

The code deliberately keeps unrestricted RFO, partitioned RFO, and a genuine
restricted P-RFO alpha solve separate.  No PES calls, Hessian measurements, or
minimum-mode solves occur here.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import MutableSequence, Sequence

import numpy as np

Array = np.ndarray

RFO_STEP_SCHEMA = "saddlemill_rfo_step_v1"
RESTRICTED_PRFO_SCHEMA = "saddlemill_restricted_prfo_v1"


class RFOError(RuntimeError):
    """Base class for native RFO contract violations."""


class RFOInputError(RFOError, ValueError):
    """Malformed matrix/vector/root-selection input."""


class RFOSymmetryError(RFOInputError):
    """Symmetry guard failure carrying the exact matrix seen by the guard."""

    def __init__(self, matrix: Array, residual: float, tolerance: float) -> None:
        self.matrix = np.asarray(matrix, dtype=float)
        self.residual = float(residual)
        self.tolerance = float(tolerance)
        super().__init__(
            f"B is not symmetric within tolerance: residual={self.residual:.6e}, "
            f"tol={self.tolerance:.6e}"
        )


@dataclass(frozen=True)
class AugmentedRootReference:
    """One homing reference for a continuously followed augmented root."""

    alpha: float
    eigenvalue: float
    selected_index: int
    vector: Array

    def __post_init__(self) -> None:
        vec = np.asarray(self.vector, dtype=float).reshape(-1).copy()
        if vec.size < 2 or not np.all(np.isfinite(vec)):
            raise RFOInputError("root reference vector must be finite and nonempty")
        norm = float(np.linalg.norm(vec))
        if norm <= 0.0:
            raise RFOInputError("root reference vector has zero norm")
        vec /= norm
        vec.setflags(write=False)
        object.__setattr__(self, "vector", vec)
        if not np.isfinite(float(self.alpha)) or float(self.alpha) <= 0.0:
            raise RFOInputError("root reference alpha must be finite and > 0")
        if not np.isfinite(float(self.eigenvalue)):
            raise RFOInputError("root reference eigenvalue must be finite")
        object.__setattr__(self, "selected_index", int(self.selected_index))


@dataclass(frozen=True)
class RFORootResult:
    """Result of one augmented RFO root solve at a fixed ``alpha``."""

    success: bool
    alpha: float
    requested_root_index: int
    selected_root_index: int
    selected_eigenvalue: float
    denominator: float
    step: Array | None
    step_norm: float | None
    root_overlap: float | None
    homing_status: str
    root_gap: float | None
    clustered_root: bool
    stationarity_residual: float | None
    eigenvalue_relation_residual: float | None
    augmented_residual: float
    matrix_condition: float | None
    matrix_condition_state: str
    symmetry_cleanup: bool
    symmetry_residual_before: float
    failure_reason: str
    eigensolve_time_ns: int
    reference: AugmentedRootReference
    schema: str = RFO_STEP_SCHEMA

    def __post_init__(self) -> None:
        if self.step is not None:
            step = np.asarray(self.step, dtype=float).reshape(-1).copy()
            step.setflags(write=False)
            object.__setattr__(self, "step", step)

    def metadata(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "success": bool(self.success),
            "alpha": float(self.alpha),
            "requested_root_index": int(self.requested_root_index),
            "selected_root_index": int(self.selected_root_index),
            "selected_eigenvalue": float(self.selected_eigenvalue),
            "denominator": float(self.denominator),
            "step_norm": None if self.step_norm is None else float(self.step_norm),
            "root_overlap": None if self.root_overlap is None else float(self.root_overlap),
            "homing_status": self.homing_status,
            "root_gap": None if self.root_gap is None else float(self.root_gap),
            "clustered_root": bool(self.clustered_root),
            "stationarity_residual": self.stationarity_residual,
            "eigenvalue_relation_residual": self.eigenvalue_relation_residual,
            "augmented_residual": float(self.augmented_residual),
            "matrix_condition": self.matrix_condition,
            "matrix_condition_state": self.matrix_condition_state,
            "symmetry_cleanup": bool(self.symmetry_cleanup),
            "symmetry_residual_before": float(self.symmetry_residual_before),
            "failure_reason": self.failure_reason,
            "eigensolve_time_ns": int(self.eigensolve_time_ns),
        }


@dataclass(frozen=True)
class PartitionedRFOResult:
    """One P/Q partitioned RFO solve at fixed ``alpha``."""

    success: bool
    alpha: float
    step: Array | None
    step_norm: float | None
    p_result: RFORootResult | None
    q_result: RFORootResult | None
    coupling_norm: float
    coupling_relative: float
    failure_reason: str
    eigensolve_time_ns: int

    def __post_init__(self) -> None:
        if self.step is not None:
            step = np.asarray(self.step, dtype=float).reshape(-1).copy()
            step.setflags(write=False)
            object.__setattr__(self, "step", step)

    def metadata(self) -> dict[str, object]:
        return {
            "success": bool(self.success),
            "alpha": float(self.alpha),
            "step_norm": None if self.step_norm is None else float(self.step_norm),
            "coupling_norm": float(self.coupling_norm),
            "coupling_relative": float(self.coupling_relative),
            "failure_reason": self.failure_reason,
            "eigensolve_time_ns": int(self.eigensolve_time_ns),
            "p_root": None if self.p_result is None else self.p_result.metadata(),
            "q_root": None if self.q_result is None else self.q_result.metadata(),
        }


@dataclass(frozen=True)
class RestrictedPRFOResult:
    """Outcome of the fixed-radius alpha solve for partitioned RFO."""

    success: bool
    status: str
    result: PartitionedRFOResult | None
    alpha: float | None
    requested_radius: float
    achieved_norm: float | None
    boundary_active: bool
    iterations: int
    root_trials: int
    bracket: tuple[float, float] | None
    failure_reason: str
    root_solve_time_ns: int
    schema: str = RESTRICTED_PRFO_SCHEMA

    def metadata(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "success": bool(self.success),
            "status": self.status,
            "alpha": self.alpha,
            "requested_radius": float(self.requested_radius),
            "achieved_norm": self.achieved_norm,
            "boundary_active": bool(self.boundary_active),
            "iterations": int(self.iterations),
            "root_trials": int(self.root_trials),
            "bracket": None if self.bracket is None else list(self.bracket),
            "failure_reason": self.failure_reason,
            "partition_result": None if self.result is None else self.result.metadata(),
            "root_solve_time_ns": int(self.root_solve_time_ns),
        }


def _readonly_vector(value: object, label: str) -> Array:
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.size < 1:
        raise RFOInputError(f"{label} cannot be empty")
    if not np.all(np.isfinite(out)):
        raise RFOInputError(f"{label} contains non-finite values")
    return np.array(out, dtype=float, copy=True)


def _prepare_symmetric_matrix(
    matrix: object,
    *,
    symmetry_tolerance: float,
) -> tuple[Array, bool, float]:
    B = np.asarray(matrix, dtype=float)
    if B.ndim != 2 or B.shape[0] != B.shape[1] or B.shape[0] < 1:
        raise RFOInputError("B must be a nonempty square matrix")
    if not np.all(np.isfinite(B)):
        raise RFOInputError("B contains non-finite values")
    tol = float(symmetry_tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise RFOInputError("symmetry_tolerance must be finite and >= 0")
    skew = B - B.T
    denom = max(float(np.linalg.norm(B)), 1.0)
    residual = float(np.linalg.norm(skew) / denom)
    if residual > tol:
        raise RFOSymmetryError(B, residual, tol)
    cleanup = bool(np.any(skew != 0.0))
    return 0.5 * (B + B.T), cleanup, residual


def _condition(B: Array, zero_tolerance: float) -> tuple[float | None, str]:
    evals = np.linalg.eigvalsh(B)
    absvals = np.abs(evals)
    maximum = float(np.max(absvals))
    minimum = float(np.min(absvals))
    threshold = max(float(zero_tolerance), np.finfo(float).eps * max(1.0, maximum))
    if minimum <= threshold:
        return None, "singular_or_near_singular"
    return float(maximum / minimum), "available"


def _root_gap(evals: Array, index: int) -> float | None:
    if evals.size <= 1:
        return None
    return float(np.min(np.abs(np.delete(evals, index) - evals[index])))


def _select_homed_root(
    evals: Array,
    evecs: Array,
    requested_index: int,
    previous: AugmentedRootReference | None,
    *,
    overlap_tie_tolerance: float,
) -> tuple[int, Array, float | None, str]:
    if previous is None:
        idx = int(requested_index)
        vector = np.array(evecs[:, idx], copy=True)
        return idx, vector, None, "initial_sorted_index"
    if previous.vector.size != evecs.shape[0]:
        raise RFOInputError("root homing reference dimension mismatch")
    overlaps = np.abs(evecs.T @ previous.vector)
    best = float(np.max(overlaps))
    candidates = np.flatnonzero(overlaps >= best - float(overlap_tie_tolerance))
    if candidates.size == 1:
        idx = int(candidates[0])
        status = "overlap_homed"
    else:
        idx = min(
            (int(i) for i in candidates),
            key=lambda i: (abs(float(evals[i]) - float(previous.eigenvalue)), i),
        )
        status = "overlap_tie_eigenvalue_then_index"
    vector = np.array(evecs[:, idx], copy=True)
    if float(np.dot(vector, previous.vector)) < 0.0:
        vector *= -1.0
    return idx, vector, float(overlaps[idx]), status


def solve_augmented_rfo(
    B: object,
    g: object,
    *,
    alpha: float = 1.0,
    root_index: int = 0,
    previous_root: AugmentedRootReference | None = None,
    denominator_tolerance: float = 1.0e-12,
    cluster_tolerance: float = 1.0e-10,
    overlap_tie_tolerance: float = 1.0e-12,
    symmetry_tolerance: float = 1.0e-12,
    spectral_zero_tolerance: float = 1.0e-12,
) -> RFORootResult:
    """Solve one augmented RFO eigenproblem without any hidden fallback.

    Numerical breakdowns such as ``|c|`` below tolerance are returned as an
    unsuccessful typed result with the selected-root metadata intact.  Invalid
    inputs raise :class:`RFOInputError`.
    """

    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0:
        raise RFOInputError("alpha must be finite and > 0")
    denom_tol = float(denominator_tolerance)
    if not np.isfinite(denom_tol) or denom_tol <= 0.0:
        raise RFOInputError("denominator_tolerance must be finite and > 0")
    cluster_tol = float(cluster_tolerance)
    if not np.isfinite(cluster_tol) or cluster_tol < 0.0:
        raise RFOInputError("cluster_tolerance must be finite and >= 0")
    Bsym, cleanup, symmetry_residual = _prepare_symmetric_matrix(
        B, symmetry_tolerance=symmetry_tolerance
    )
    grad = _readonly_vector(g, "g")
    n = Bsym.shape[0]
    if grad.size != n:
        raise RFOInputError(f"g has dimension {grad.size}; B has dimension {n}")
    requested = int(root_index)
    if requested < 0 or requested >= n + 1:
        raise RFOInputError(
            f"root_index {requested} outside augmented spectrum of size {n + 1}"
        )

    A = np.empty((n + 1, n + 1), dtype=float)
    A[:n, :n] = (a * a) * Bsym
    A[:n, n] = a * grad
    A[n, :n] = a * grad
    A[n, n] = 0.0
    start = perf_counter_ns()
    evals, evecs = np.linalg.eigh(A)
    eig_time = perf_counter_ns() - start
    selected, vector, overlap, homing = _select_homed_root(
        evals,
        evecs,
        requested,
        previous_root,
        overlap_tie_tolerance=overlap_tie_tolerance,
    )
    eigenvalue = float(evals[selected])
    c = float(vector[-1])
    gap = _root_gap(evals, selected)
    scale = max(1.0, abs(eigenvalue), float(np.max(np.abs(evals))))
    clustered = bool(gap is not None and gap <= cluster_tol * scale)
    if clustered:
        homing = f"{homing};clustered"
    aug_residual = float(
        np.linalg.norm(A @ vector - eigenvalue * vector)
        / max(1.0, float(np.linalg.norm(A)), abs(eigenvalue))
    )
    condition, condition_state = _condition(Bsym, spectral_zero_tolerance)
    reference = AugmentedRootReference(a, eigenvalue, selected, vector)

    if abs(c) <= denom_tol:
        return RFORootResult(
            success=False,
            alpha=a,
            requested_root_index=requested,
            selected_root_index=selected,
            selected_eigenvalue=eigenvalue,
            denominator=c,
            step=None,
            step_norm=None,
            root_overlap=overlap,
            homing_status=homing,
            root_gap=gap,
            clustered_root=clustered,
            stationarity_residual=None,
            eigenvalue_relation_residual=None,
            augmented_residual=aug_residual,
            matrix_condition=condition,
            matrix_condition_state=condition_state,
            symmetry_cleanup=cleanup,
            symmetry_residual_before=symmetry_residual,
            failure_reason="near_zero_augmented_denominator",
            eigensolve_time_ns=eig_time,
            reference=reference,
        )

    step = a * vector[:-1] / c
    if not np.all(np.isfinite(step)):
        failure = "nonfinite_step"
        step_out = None
        step_norm = None
        stationarity = None
        relation = None
        success = False
    else:
        mu = eigenvalue / (a * a)
        stationary_vec = Bsym @ step + grad - mu * step
        stationarity = float(
            np.linalg.norm(stationary_vec)
            / max(
                1.0,
                float(np.linalg.norm(Bsym @ step)),
                float(np.linalg.norm(grad)),
                abs(mu) * float(np.linalg.norm(step)),
            )
        )
        relation = float(abs(eigenvalue - float(np.dot(grad, step))) / max(1.0, abs(eigenvalue)))
        step_norm = float(np.linalg.norm(step))
        step_out = step
        failure = ""
        success = True

    return RFORootResult(
        success=success,
        alpha=a,
        requested_root_index=requested,
        selected_root_index=selected,
        selected_eigenvalue=eigenvalue,
        denominator=c,
        step=step_out,
        step_norm=step_norm,
        root_overlap=overlap,
        homing_status=homing,
        root_gap=gap,
        clustered_root=clustered,
        stationarity_residual=stationarity,
        eigenvalue_relation_residual=relation,
        augmented_residual=aug_residual,
        matrix_condition=condition,
        matrix_condition_state=condition_state,
        symmetry_cleanup=cleanup,
        symmetry_residual_before=symmetry_residual,
        failure_reason=failure,
        eigensolve_time_ns=eig_time,
        reference=reference,
    )


def rfo_order0(B: object, g: object, **kwargs) -> RFORootResult:
    """Unpartitioned minimization branch: augmented root index 0."""

    return solve_augmented_rfo(B, g, root_index=0, **kwargs)


def rfo_order1(B: object, g: object, **kwargs) -> RFORootResult:
    """Unpartitioned first-order-saddle model branch: augmented root index 1."""

    return solve_augmented_rfo(B, g, root_index=1, **kwargs)


def _validate_partition(Vp: object, Vq: object, n: int) -> tuple[Array, Array]:
    P = np.asarray(Vp, dtype=float)
    Q = np.asarray(Vq, dtype=float)
    if P.ndim != 2 or P.shape[0] != n:
        raise RFOInputError("Vp must have shape (n, p)")
    if Q.ndim != 2 or Q.shape[0] != n:
        raise RFOInputError("Vq must have shape (n, q)")
    if P.shape[1] < 1:
        raise RFOInputError("P partition cannot be empty for P-RFO")
    if P.shape[1] + Q.shape[1] != n:
        raise RFOInputError("P and Q dimensions must span the full active space")
    V = np.column_stack((P, Q))
    if not np.all(np.isfinite(V)):
        raise RFOInputError("partition basis contains non-finite values")
    gram = V.T @ V
    if not np.allclose(gram, np.eye(n), rtol=0.0, atol=2.0e-10):
        raise RFOInputError("P/Q basis must be mutually orthonormal and complete")
    return np.array(P, copy=True), np.array(Q, copy=True)


def solve_partitioned_rfo(
    B: object,
    g: object,
    Vp: object,
    Vq: object,
    *,
    alpha: float = 1.0,
    p_previous_root: AugmentedRootReference | None = None,
    q_previous_root: AugmentedRootReference | None = None,
    **root_kwargs,
) -> PartitionedRFOResult:
    """Solve separate uphill-P and downhill-Q augmented problems.

    The P branch selects augmented root index ``dim(P)``; the Q branch selects
    root index 0.  Off-diagonal P/Q coupling is intentionally *not* included in
    the two subproblems and is reported explicitly.
    """

    Bsym, _, _ = _prepare_symmetric_matrix(
        B, symmetry_tolerance=float(root_kwargs.get("symmetry_tolerance", 1.0e-12))
    )
    grad = _readonly_vector(g, "g")
    if grad.size != Bsym.shape[0]:
        raise RFOInputError("g/B dimension mismatch")
    P, Q = _validate_partition(Vp, Vq, Bsym.shape[0])
    pB = P.T @ Bsym @ P
    pg = P.T @ grad
    p_result = solve_augmented_rfo(
        pB,
        pg,
        alpha=alpha,
        root_index=P.shape[1],
        previous_root=p_previous_root,
        **root_kwargs,
    )
    if Q.shape[1]:
        qB = Q.T @ Bsym @ Q
        qg = Q.T @ grad
        q_result = solve_augmented_rfo(
            qB,
            qg,
            alpha=alpha,
            root_index=0,
            previous_root=q_previous_root,
            **root_kwargs,
        )
    else:
        q_result = None
    coupling = P.T @ Bsym @ Q
    coupling_norm = float(np.linalg.norm(coupling))
    coupling_relative = float(coupling_norm / max(float(np.linalg.norm(Bsym)), 1.0e-300))
    eig_time = p_result.eigensolve_time_ns + (0 if q_result is None else q_result.eigensolve_time_ns)
    if not p_result.success:
        return PartitionedRFOResult(
            False,
            float(alpha),
            None,
            None,
            p_result,
            q_result,
            coupling_norm,
            coupling_relative,
            f"p_branch:{p_result.failure_reason}",
            eig_time,
        )
    if q_result is not None and not q_result.success:
        return PartitionedRFOResult(
            False,
            float(alpha),
            None,
            None,
            p_result,
            q_result,
            coupling_norm,
            coupling_relative,
            f"q_branch:{q_result.failure_reason}",
            eig_time,
        )
    pstep = P @ p_result.step
    qstep = np.zeros_like(pstep) if q_result is None else Q @ q_result.step
    step = pstep + qstep
    if not np.all(np.isfinite(step)):
        return PartitionedRFOResult(
            False,
            float(alpha),
            None,
            None,
            p_result,
            q_result,
            coupling_norm,
            coupling_relative,
            "nonfinite_reconstructed_step",
            eig_time,
        )
    return PartitionedRFOResult(
        True,
        float(alpha),
        step,
        float(np.linalg.norm(step)),
        p_result,
        q_result,
        coupling_norm,
        coupling_relative,
        "",
        eig_time,
    )


class _HomingPath:
    def __init__(self) -> None:
        self._p: list[tuple[float, AugmentedRootReference]] = []
        self._q: list[tuple[float, AugmentedRootReference]] = []

    @staticmethod
    def _nearest(items: Sequence[tuple[float, AugmentedRootReference]], alpha: float):
        if not items:
            return None
        return min(items, key=lambda item: (abs(item[0] - alpha), item[0]))[1]

    def references(self, alpha: float):
        return self._nearest(self._p, alpha), self._nearest(self._q, alpha)

    def add(self, alpha: float, result: PartitionedRFOResult) -> None:
        if result.p_result is not None:
            self._p.append((float(alpha), result.p_result.reference))
        if result.q_result is not None:
            self._q.append((float(alpha), result.q_result.reference))


def restricted_partitioned_rfo(
    B: object,
    g: object,
    Vp: object,
    Vq: object,
    *,
    radius: float,
    tolerance: float = 1.0e-10,
    max_iterations: int = 100,
    alpha_min: float = 1.0e-12,
    alpha_tolerance: float = 1.0e-14,
    root_tracking_policy: str = "homed",
    trial_trace: MutableSequence[dict[str, object]] | None = None,
    **root_kwargs,
) -> RestrictedPRFOResult:
    """Solve ``||s(alpha)|| = radius`` for a partitioned RFO step.

    The solve starts at ``alpha=1``.  If the unrestricted P-RFO step is outside
    the requested radius, alpha is halved until an inside point is found, then a
    safeguarded bisection is performed.  ``root_tracking_policy="homed"``
    preserves the production behavior of following the nearest previously solved
    P/Q augmented roots.  ``"sorted"`` is an explicit diagnostic/replay mode that
    reselects the requested sorted augmented root independently at every alpha.
    No clipping is used.  Failure to bracket or a numerical/root discontinuity is
    returned honestly as a typed unsuccessful result.

    ``alpha_tolerance`` is retained for API/provenance compatibility but is not a
    correctness stopping criterion: the radius tolerance is authoritative.  The
    bisection stops early only when no representable floating-point alpha remains
    between the current bracket endpoints.  ``trial_trace``, when supplied, is
    append-only passive diagnostics and cannot alter the proposed step.
    """

    start = perf_counter_ns()
    delta = float(radius)
    tol = float(tolerance)
    amin = float(alpha_min)
    atol = float(alpha_tolerance)
    maxit = int(max_iterations)
    if not np.isfinite(delta) or delta <= 0.0:
        raise RFOInputError("restricted P-RFO radius must be finite and > 0")
    if not np.isfinite(tol) or tol <= 0.0:
        raise RFOInputError("restricted P-RFO tolerance must be finite and > 0")
    if maxit < 1:
        raise RFOInputError("max_iterations must be >= 1")
    if not np.isfinite(amin) or not 0.0 < amin < 1.0:
        raise RFOInputError("alpha_min must satisfy 0 < alpha_min < 1")
    if not np.isfinite(atol) or atol <= 0.0:
        raise RFOInputError("alpha_tolerance must be finite and > 0")
    tracking = str(root_tracking_policy).strip().lower()
    if tracking not in {"homed", "sorted"}:
        raise RFOInputError("root_tracking_policy must be 'homed' or 'sorted'")

    path = _HomingPath()
    trials = 0

    def trial(alpha: float, phase: str) -> PartitionedRFOResult:
        nonlocal trials
        if tracking == "homed":
            pref, qref = path.references(alpha)
        else:
            pref, qref = None, None
        result = solve_partitioned_rfo(
            B,
            g,
            Vp,
            Vq,
            alpha=alpha,
            p_previous_root=pref,
            q_previous_root=qref,
            **root_kwargs,
        )
        trials += 1
        if tracking == "homed":
            path.add(alpha, result)
        if trial_trace is not None:
            trial_trace.append(
                {
                    "trial": int(trials),
                    "phase": str(phase),
                    "root_tracking_policy": tracking,
                    "alpha": float(alpha),
                    "result": result.metadata(),
                }
            )
        return result

    unrestricted = trial(1.0, "unrestricted")
    if not unrestricted.success:
        return RestrictedPRFOResult(
            False,
            "unrestricted_trial_failed",
            unrestricted,
            1.0,
            delta,
            None,
            False,
            0,
            trials,
            None,
            unrestricted.failure_reason,
            perf_counter_ns() - start,
        )
    if unrestricted.step_norm is None:
        raise AssertionError("successful partition result is missing step_norm")
    boundary_tol = tol * max(1.0, delta)
    if unrestricted.step_norm <= delta + boundary_tol:
        return RestrictedPRFOResult(
            True,
            "inside_radius_unrestricted",
            unrestricted,
            1.0,
            delta,
            unrestricted.step_norm,
            False,
            0,
            trials,
            None,
            "",
            perf_counter_ns() - start,
        )

    high_alpha = 1.0
    high_result = unrestricted
    low_alpha = None
    low_result = None
    alpha = 0.5
    bracket_iterations = 0
    while alpha >= amin:
        candidate = trial(alpha, "bracket")
        bracket_iterations += 1
        if not candidate.success:
            return RestrictedPRFOResult(
                False,
                "bracket_trial_failed",
                candidate,
                alpha,
                delta,
                None,
                False,
                bracket_iterations,
                trials,
                None,
                candidate.failure_reason,
                perf_counter_ns() - start,
            )
        if candidate.step_norm is None:
            raise AssertionError("successful partition result is missing step_norm")
        if candidate.step_norm <= delta:
            low_alpha = alpha
            low_result = candidate
            break
        high_alpha = alpha
        high_result = candidate
        alpha *= 0.5

    if low_alpha is None or low_result is None:
        return RestrictedPRFOResult(
            False,
            "unbracketable",
            high_result,
            high_alpha,
            delta,
            high_result.step_norm,
            False,
            bracket_iterations,
            trials,
            None,
            "could_not_bracket_radius_before_alpha_min",
            perf_counter_ns() - start,
        )

    bracket = (float(low_alpha), float(high_alpha))
    best = low_result if abs(low_result.step_norm - delta) <= abs(high_result.step_norm - delta) else high_result
    boundary_iterations = 0
    for iteration in range(1, maxit + 1):
        boundary_iterations = iteration
        mid = 0.5 * (low_alpha + high_alpha)
        if mid <= low_alpha or mid >= high_alpha:
            break
        candidate = trial(mid, "boundary")
        if not candidate.success:
            return RestrictedPRFOResult(
                False,
                "root_discontinuity_or_trial_failure",
                candidate,
                mid,
                delta,
                None,
                False,
                bracket_iterations + iteration,
                trials,
                (float(low_alpha), float(high_alpha)),
                candidate.failure_reason,
                perf_counter_ns() - start,
            )
        norm = candidate.step_norm
        if norm is None or not np.isfinite(norm):
            return RestrictedPRFOResult(
                False,
                "nonfinite_trial_norm",
                candidate,
                mid,
                delta,
                norm,
                False,
                bracket_iterations + iteration,
                trials,
                (float(low_alpha), float(high_alpha)),
                "nonfinite_trial_norm",
                perf_counter_ns() - start,
            )
        if abs(norm - delta) < abs(best.step_norm - delta):
            best = candidate
        if abs(norm - delta) <= boundary_tol:
            return RestrictedPRFOResult(
                True,
                "boundary_converged",
                candidate,
                mid,
                delta,
                norm,
                True,
                bracket_iterations + iteration,
                trials,
                (float(low_alpha), float(high_alpha)),
                "",
                perf_counter_ns() - start,
            )
        if norm > delta:
            high_alpha = mid
            high_result = candidate
        else:
            low_alpha = mid
            low_result = candidate
        # Do not declare failure merely because an alpha-space hint is small.
        # Radius-space tolerance is the solver contract, and alpha can remain
        # representable far below the historical 1e-14 interval hint.
        if high_alpha - low_alpha <= atol and trial_trace is not None:
            trial_trace.append(
                {
                    "trial": int(trials),
                    "phase": "legacy_alpha_tolerance_trigger",
                    "root_tracking_policy": tracking,
                    "alpha": float(mid),
                    "alpha_interval": float(high_alpha - low_alpha),
                    "alpha_tolerance": float(atol),
                    "radius_error": float(abs(norm - delta)),
                    "radius_tolerance": float(boundary_tol),
                }
            )
        if np.nextafter(low_alpha, high_alpha) >= high_alpha:
            break

    return RestrictedPRFOResult(
        False,
        "max_iterations_or_discontinuous_boundary",
        best,
        best.alpha,
        delta,
        best.step_norm,
        False,
        bracket_iterations + boundary_iterations,
        trials,
        (float(low_alpha), float(high_alpha)),
        "boundary_not_reached_within_tolerance",
        perf_counter_ns() - start,
    )


__all__ = [
    "AugmentedRootReference",
    "PartitionedRFOResult",
    "RFOError",
    "RFOInputError",
    "RFOSymmetryError",
    "RFORootResult",
    "RestrictedPRFOResult",
    "restricted_partitioned_rfo",
    "rfo_order0",
    "rfo_order1",
    "solve_augmented_rfo",
    "solve_partitioned_rfo",
]
