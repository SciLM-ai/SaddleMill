"""Iterative minimum-mode eigensolvers for SaddleMill MMF searches.

This module contains eigensolver mathematics and the persistent physical-Hessian
model used as a Davidson preconditioner.  It deliberately does not know about
ASE Atoms or calculators; callers provide Hessian-vector products ``hvp(q)``.

Implemented solver families
---------------------------
``lanczos``
    Numerically stabilized generic Lanczos (full reorthogonalization).
``davidson``
    Generic textbook Davidson with the coefficient-weighted Ritz residual.
``softsaddle_lanczos``
    Port of the standalone historical SoftSaddle Lanczos recurrence and
    relative lowest-eigenvalue-change stopping rule.
``softsaddle_davidson``
    Port of the historical SoftSaddle Davidson/Lanczos hybrid.  The caller
    supplies the persistent physical Hessian B; the hybrid first compares Hq0
    with Bq0 and chooses Davidson or a Lanczos fallback using the historical
    threshold.  Both the historical unweighted Davidson correction residual
    and the textbook coefficient-weighted residual are available so they can
    be A/B tested without changing the rest of the hybrid.

The historical SoftSaddle source uses 8 maximum eigensolver iterations,
0.01 eigenvalue-change tolerance, 1e-4 A one-sided finite-difference HVPs,
and a Davidson/Lanczos switch threshold of 12.  The finite-difference HVP and
reference-minimum Hessian are evaluated by the ASE-facing caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SolverAuditMetadata:
    """Additive audit record for one minimum-mode solve.

    Existing solver return fields retain their historical meanings.  This
    record makes the retained Ritz basis/actions, stopping equation, and the
    evaluated-versus-returned mode explicit without changing legacy numerics.
    Untyped callback-based legacy solvers cannot claim physical-HVP
    certification; typed typed-HVP ``HVPResult``-based solvers fill the provenance
    and accounting fields below.
    """

    retained_basis: tuple[Array, ...] = ()
    retained_actions: tuple[Array, ...] = ()
    evaluated_mode: Array | None = None
    returned_mode: Array | None = None
    selected_root_index: int | None = 0
    root_selection: str = "lowest"
    stopping_rule: str = ""
    stopping_equation: str = ""
    stopping_reason: str = ""
    breakdown_reason: str = ""
    hvp_count: int = 0
    restart_count: int = 0
    restart_reason: str = ""
    rank: int = 0
    full_residual_norm: float | None = None
    solver_operator_residual_norm: float | None = None
    projected_operator_asymmetry_norm: float | None = None
    action_origins: tuple[str, ...] = ()
    action_sources: tuple[str, ...] = ()
    action_families: tuple[str, ...] = ()
    action_stencil_schemes: tuple[str, ...] = ()
    action_displacement_scales: tuple[float | None, ...] = ()
    action_geometry_ids: tuple[str, ...] = ()
    action_state_uids: tuple[str, ...] = ()
    physical_action_count: int = 0
    model_action_count: int = 0
    certification_eligible: bool = False
    algorithm_pes_calls: int = 0
    diagnostic_pes_calls: int = 0
    cache_hits: int = 0
    gap: float | None = None
    gap_eligible: bool = False
    degeneracy_unresolved: bool = False
    approximation_caveat: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class MinModeResult:
    """Result of one iterative minimum-mode solve."""

    eigenvalue: float
    eigenvector: Array
    iterations: int
    converged: bool
    breakdown: bool
    eigenvalue_change: float | None
    residual_norm: float
    subspace_dimension: int
    solver_used: str = ""
    switch_metric: float | None = None
    residual_variant: str = ""
    audit: SolverAuditMetadata | None = None


def _as_vector(vector: Array) -> Array:
    arr = np.asarray(vector, dtype=float).reshape(-1).copy()
    if arr.size == 0:
        raise ValueError("Minimum-mode vector must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Minimum-mode vector contains non-finite values")
    return arr


def _orthogonalize(vector: Array, basis: Iterable[Array], passes: int = 2) -> Array:
    """Modified Gram-Schmidt with an optional numerical-cleanup second pass."""
    out = _as_vector(vector)
    basis_vectors = [np.asarray(v, dtype=float).reshape(-1) for v in basis]
    for _ in range(max(1, int(passes))):
        for base in basis_vectors:
            out -= float(np.dot(base, out)) * base
    return out


def _normalize(vector: Array, tolerance: float) -> Array:
    vector = _as_vector(vector)
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= float(tolerance):
        raise ValueError(
            f"Minimum-mode vector norm {magnitude:.3e} is below "
            f"breakdown tolerance {float(tolerance):.3e}"
        )
    return vector / magnitude


def _sign_align(vector: Array, reference: Array) -> Array:
    vector = np.asarray(vector, dtype=float).reshape(-1).copy()
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if float(np.dot(vector, reference)) < 0.0:
        vector *= -1.0
    return vector


def _lowest_ritz(projected_hessian: Array) -> tuple[float, Array]:
    projected_hessian = np.asarray(projected_hessian, dtype=float)
    # The physical Hessian is symmetric. Symmetrizing removes only numerical
    # finite-difference/roundoff asymmetry before the symmetric eigensolver.
    projected_hessian = 0.5 * (projected_hessian + projected_hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(projected_hessian)
    return float(eigenvalues[0]), np.asarray(eigenvectors[:, 0], dtype=float)


def _change(
    new: float,
    old: float | None,
    *,
    mode: str,
    zero_tolerance: float,
) -> float | None:
    if old is None or not np.isfinite(old):
        return None
    if mode == "absolute":
        return abs(float(new) - float(old))
    if mode == "relative":
        # Historical standalone SoftSaddle literally divides by oldEigenvalue.
        # Treat a near-zero old value as an infinite change so it cannot
        # spuriously certify convergence on the first/zero-curvature estimate.
        if abs(float(old)) <= float(zero_tolerance):
            return 0.0 if abs(float(new) - float(old)) <= zero_tolerance else float("inf")
        return abs((float(new) - float(old)) / float(old))
    raise ValueError(f"Unknown eigenvalue-change mode {mode!r}")



def _array_tuple(items: Iterable[Array]) -> tuple[Array, ...]:
    return tuple(np.asarray(item, dtype=float).copy() for item in items)


def _right_rotation_span_equivalence(
    raw_basis: Sequence[object],
    raw_actions: Sequence[object],
    returned_basis: Sequence[object],
    returned_actions: Sequence[object],
    *,
    tolerance: float = 1.0e-10,
) -> dict[str, object]:
    """Prove a full-rank orthogonal right rotation between two HVP blocks.

    The raw block carries physical provenance.  Sella 2.5.0 may rotate its
    retained Ritz basis and actions together, so column identity is not an
    invariant.  This helper verifies the stronger useful invariant
    ``V = S C`` and ``AV = Y C`` with one numerically orthogonal square ``C``.
    It performs no HVP/PES work.
    """
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be finite and > 0")
    raw_basis = tuple(np.asarray(v, dtype=float).reshape(-1) for v in raw_basis)
    raw_actions = tuple(np.asarray(v, dtype=float).reshape(-1) for v in raw_actions)
    returned_basis = tuple(np.asarray(v, dtype=float).reshape(-1) for v in returned_basis)
    returned_actions = tuple(np.asarray(v, dtype=float).reshape(-1) for v in returned_actions)
    result = {
        "complete": False,
        "tolerance": tol,
        "raw_rank": 0,
        "returned_rank": 0,
        "raw_columns": len(raw_basis),
        "returned_columns": len(returned_basis),
        "direction_relative_residual": None,
        "action_relative_residual": None,
        "right_orthogonality_residual": None,
    }
    if (
        not raw_basis or not returned_basis
        or len(raw_basis) != len(raw_actions)
        or len(returned_basis) != len(returned_actions)
    ):
        return result
    n = raw_basis[0].size
    if any(v.size != n for v in raw_basis + raw_actions + returned_basis + returned_actions):
        return result
    S = np.column_stack(raw_basis)
    Y = np.column_stack(raw_actions)
    V = np.column_stack(returned_basis)
    AV = np.column_stack(returned_actions)
    if not all(np.all(np.isfinite(x)) for x in (S, Y, V, AV)):
        return result
    raw_rank = int(np.linalg.matrix_rank(S, tol=tol))
    returned_rank = int(np.linalg.matrix_rank(V, tol=tol))
    result["raw_rank"] = raw_rank
    result["returned_rank"] = returned_rank
    # The invariant requested for Olsen/JD is a square full-rank right rotation.
    if S.shape[1] != V.shape[1] or raw_rank != S.shape[1] or returned_rank != V.shape[1]:
        return result
    C = np.linalg.lstsq(S, V, rcond=None)[0]
    dres = float(np.linalg.norm(S @ C - V) / max(np.linalg.norm(V), tol))
    ares = float(np.linalg.norm(Y @ C - AV) / max(np.linalg.norm(AV), tol))
    ident = np.eye(C.shape[0])
    ores = float(max(np.linalg.norm(C.T @ C - ident), np.linalg.norm(C @ C.T - ident)))
    result.update({
        "direction_relative_residual": dres,
        "action_relative_residual": ares,
        "right_orthogonality_residual": ores,
        "complete": bool(dres <= tol and ares <= tol and ores <= tol),
    })
    return result


def _legacy_solver_audit(
    basis: Sequence[Array],
    actions: Sequence[Array],
    *,
    theta: float,
    coefficients: Array,
    returned_mode: Array,
    stopping_rule: str,
    stopping_equation: str,
    stopping_reason: str,
    breakdown_reason: str = "",
    root_selection: str = "lowest",
    selected_root_index: int | None = 0,
    solver_operator_residual_norm: float | None = None,
    typed_hvp_results: Sequence[object] = (),
    hvp_count_override: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SolverAuditMetadata:
    """Build additive audit metadata for callback-based legacy solvers.

    The callback has no typed-HVP provenance envelope, so this helper deliberately
    labels its operator origin as untyped and never marks it certification
    eligible.  The full Ritz residual is nevertheless computed from the
    actually retained ``Q`` and ``U`` columns, not from the small projected
    eigensystem.
    """

    qcols = [np.asarray(item, dtype=float).reshape(-1).copy() for item in basis]
    ucols = [np.asarray(item, dtype=float).reshape(-1).copy() for item in actions]
    full_residual_norm: float | None = None
    evaluated_mode: Array | None = None
    asymmetry: float | None = None
    rank = 0
    if qcols and len(qcols) == len(ucols):
        Q = np.column_stack(qcols)
        U = np.column_stack(ucols)
        c = np.asarray(coefficients, dtype=float).reshape(-1)
        if c.size == Q.shape[1]:
            evaluated_mode = Q @ c
            full_residual_norm = float(
                np.linalg.norm(U @ c - float(theta) * (Q @ c))
            )
        rank = int(np.linalg.matrix_rank(Q))
        projected_raw = Q.T @ U
        asymmetry = float(np.linalg.norm(projected_raw - projected_raw.T))
    typed = tuple(typed_hvp_results)
    retained_typed: list[object] = []
    retained_typed_indices: list[int] = []
    search_start = 0
    if typed:
        # Historical SoftSaddle Lanczos can spend one terminal HVP detecting
        # beta breakdown and then restore the previous Ritz subspace.  Match
        # the numerically retained Q/U columns to the immutable typed-HVP envelopes
        # instead of assuming HVP count == retained-action count.
        for qcol, ucol in zip(qcols, ucols):
            matched = None
            for index in range(search_start, len(typed)):
                item = typed[index]
                direction = np.asarray(getattr(item, "direction", ()), dtype=float).reshape(-1)
                action = getattr(item, "action", None)
                if action is None:
                    continue
                action_flat = np.asarray(action, dtype=float).reshape(-1)
                if (
                    direction.shape == qcol.shape
                    and action_flat.shape == ucol.shape
                    and np.allclose(direction, qcol, atol=1.0e-11, rtol=1.0e-11)
                    and np.allclose(action_flat, ucol, atol=1.0e-11, rtol=1.0e-11)
                ):
                    matched = item
                    search_start = index + 1
                    break
            if matched is not None:
                retained_typed.append(matched)
                retained_typed_indices.append(index)

    if typed:
        work = summarize_hvp_actions(typed)
        fully_matched = len(retained_typed) == len(ucols)
        retained_origins = tuple(
            str(item.metadata.operator_origin) for item in retained_typed
        )
        retained_sources = tuple(str(item.metadata.source) for item in retained_typed)
        retained_families = tuple(str(item.metadata.family) for item in retained_typed)
        retained_stencils = tuple(str(getattr(item, "stencil_scheme", "")) for item in retained_typed)
        retained_scales = tuple(getattr(item, "displacement_scale", None) for item in retained_typed)
        retained_geometry_ids = tuple(str(item.metadata.geometry_id) for item in retained_typed)
        retained_state_uids = tuple(str(item.metadata.state_uid) for item in retained_typed)
        physical_residual = bool(
            fully_matched
            and retained_typed
            and all(bool(item.metadata.physical) for item in retained_typed)
        )
        caveat = (
            "typed typed-HVP physical same-center HVP provenance; full Ritz residual "
            "uses retained raw actions"
            if physical_residual
            else "typed HVP provenance includes model/nonphysical or unmatched "
            "retained actions; full Ritz residual is not physical certification"
        )
        audit_metadata = dict(metadata or {})
        audit_metadata.update({
            "retained_t01_match_count": len(retained_typed),
            "retained_t01_match_complete": bool(fully_matched),
            "retained_hvp_indices": tuple(int(i) for i in retained_typed_indices),
            "full_residual_origin": (
                "physical_raw_same_center" if physical_residual else "typed_nonphysical_or_unmatched"
            ),
            "all_hvp_origins": work.origins,
            "all_hvp_sources": work.sources,
            "all_hvp_families": work.families,
        })
        hvp_count = len(typed)
    else:
        work = None
        retained_origins = tuple("legacy_callback_untyped" for _ in ucols)
        retained_sources = tuple("" for _ in ucols)
        retained_families = tuple("" for _ in ucols)
        retained_stencils = tuple("" for _ in ucols)
        retained_scales = tuple(None for _ in ucols)
        retained_geometry_ids = tuple("" for _ in ucols)
        retained_state_uids = tuple("" for _ in ucols)
        caveat = (
            "legacy callback action has no typed HVP provenance; full Ritz "
            "residual is numerically reconstructed but is not physical certification"
        )
        audit_metadata = dict(metadata or {})
        audit_metadata["full_residual_origin"] = "legacy_callback_untyped"
        hvp_count = len(ucols) if hvp_count_override is None else int(hvp_count_override)

    return SolverAuditMetadata(
        retained_basis=_array_tuple(qcols),
        retained_actions=_array_tuple(ucols),
        evaluated_mode=(None if evaluated_mode is None else evaluated_mode.copy()),
        returned_mode=np.asarray(returned_mode, dtype=float).reshape(-1).copy(),
        selected_root_index=selected_root_index,
        root_selection=root_selection,
        stopping_rule=stopping_rule,
        stopping_equation=stopping_equation,
        stopping_reason=stopping_reason,
        breakdown_reason=breakdown_reason,
        hvp_count=hvp_count,
        rank=rank,
        full_residual_norm=full_residual_norm,
        solver_operator_residual_norm=solver_operator_residual_norm,
        projected_operator_asymmetry_norm=asymmetry,
        action_origins=retained_origins,
        action_sources=retained_sources,
        action_families=retained_families,
        action_stencil_schemes=retained_stencils,
        action_displacement_scales=retained_scales,
        action_geometry_ids=retained_geometry_ids,
        action_state_uids=retained_state_uids,
        physical_action_count=(0 if work is None else work.physical_actions),
        model_action_count=(0 if work is None else work.model_actions),
        certification_eligible=(False if work is None else work.certification_eligible),
        algorithm_pes_calls=(0 if work is None else work.algorithm_pes_calls),
        diagnostic_pes_calls=(0 if work is None else work.diagnostic_pes_calls),
        cache_hits=(0 if work is None else work.cache_hits),
        approximation_caveat=caveat,
        metadata=audit_metadata,
    )


def _stop_reason(*, converged: bool, breakdown: bool, exhausted: bool) -> str:
    if converged:
        return "stopping_equation_satisfied"
    if breakdown:
        return "breakdown"
    if exhausted:
        return "max_iterations"
    return "returned"


class TypedHVPCallback:
    """Bridge unchanged legacy numerical kernels to the sealed typed HVP API.

    shared-runtime may pass this callable to the existing Lanczos/Davidson/SoftSaddle
    kernels.  Their numerical recurrences receive the same action arrays, but
    every operator application originates from a typed ``HVPResult`` retained
    here for additive audit/provenance metadata.  The backend alone owns any
    PES work and must report its typed-HVP accounting truthfully.
    """

    def __init__(
        self,
        backend,
        *,
        coordinate_space,
        state_id: int,
        state_uid: str,
        geometry_id: str,
        source: str,
        family: str,
        purpose: object = "algorithm",
        stencil_id_factory: Callable[[Array, int], str] | None = None,
        request_metadata: Mapping[str, object] | None = None,
    ) -> None:
        from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace, WorkPurpose

        if not isinstance(coordinate_space, ActiveCoordinateSpace):
            raise TypeError("coordinate_space must be ActiveCoordinateSpace")
        self.backend = backend
        self.coordinate_space = coordinate_space
        self.state_id = int(state_id)
        self.state_uid = str(state_uid)
        self.geometry_id = str(geometry_id)
        self.source = str(source).strip().lower()
        self.family = str(family).strip().lower()
        self.purpose = WorkPurpose(str(purpose))
        if not self.source or not self.family:
            raise ValueError("source and family must be nonempty")
        self.stencil_id_factory = stencil_id_factory
        self.request_metadata = dict(request_metadata or {})
        self.requests: list[object] = []
        self.results: list[object] = []

    def __call__(self, direction: Array) -> Array:
        from saddlemill.dimertools.hvp_interfaces import HVPRequest

        index = len(self.results)
        shape = self.coordinate_space.active_dof_mask.shape
        vector = np.asarray(direction, dtype=float).reshape(shape)
        stencil_id = ""
        if self.stencil_id_factory is not None:
            stencil_id = str(self.stencil_id_factory(vector.copy(), index))
        metadata = dict(self.request_metadata)
        metadata["typed_hvp_call_index"] = int(index)
        request = HVPRequest(
            state_id=self.state_id,
            state_uid=self.state_uid,
            geometry_id=self.geometry_id,
            direction=vector,
            coordinate_space=self.coordinate_space,
            purpose=self.purpose,
            source=self.source,
            family=self.family,
            stencil_id=stencil_id,
            metadata=metadata,
        )
        result = self.backend.apply(request)
        _validate_hvp_result_identity(result, request)
        self.requests.append(request)
        self.results.append(result)
        return np.asarray(result.action, dtype=float).reshape(-1).copy()


# ---------------------------------------------------------------------------
# Generic, numerically stabilized Lanczos
# ---------------------------------------------------------------------------

def lanczos_lowest_mode(
    hvp: Callable[[Array], Array],
    initial_vector: Array,
    *,
    max_iterations: int = 8,
    eigenvalue_tolerance: float = 0.01,
    breakdown_tolerance: float = 1.0e-12,
    previous_eigenvalue: float | None = None,
    excluded_basis: Iterable[Array] = (),
) -> MinModeResult:
    """Find the lowest mode with stabilized symmetric Lanczos.

    This generic implementation uses full reorthogonalization and stops on the
    Olsen/JD-style full Ritz residual ``||H v - theta v|| < 0.1*|theta|``.
    The residual is reconstructed from the already retained HVP actions, so the
    stopping test adds no HVP/PES calls.  ``eigenvalue_tolerance`` and
    ``previous_eigenvalue`` remain accepted for API compatibility/diagnostics;
    neither controls convergence.  Use :func:`softsaddle_lanczos_lowest_mode`
    when the historical recurrence and eigenvalue-change stopping rule are the
    object being benchmarked.
    """
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    eigenvalue_tolerance = float(eigenvalue_tolerance)
    if eigenvalue_tolerance < 0.0:
        raise ValueError("eigenvalue_tolerance must be >= 0")
    breakdown_tolerance = float(breakdown_tolerance)
    if breakdown_tolerance <= 0.0:
        raise ValueError("breakdown_tolerance must be > 0")

    external = [_normalize(v, breakdown_tolerance) for v in excluded_basis]
    q0 = _orthogonalize(initial_vector, external)
    q = _normalize(q0, breakdown_tolerance)
    initial_direction = q.copy()

    q_columns: list[Array] = []
    u_columns: list[Array] = []
    alpha: list[float] = []
    beta_links: list[float] = []
    q_previous: Optional[Array] = None
    beta_previous = 0.0
    old_eigenvalue = (
        None if previous_eigenvalue is None else float(previous_eigenvalue)
    )
    eigenvalue_change: Optional[float] = None
    converged = False
    breakdown = False
    lowest = float("nan")
    coeff = np.ones(1)
    residual_norm = float("nan")
    residual_gamma = 0.1
    residual_threshold = float("nan")

    for _iteration in range(max_iterations):
        hq = _as_vector(hvp(q))
        if hq.shape != q.shape:
            raise ValueError("HVP callback returned a vector with wrong shape")

        a = float(np.dot(q, hq))
        w = hq - a * q
        if q_previous is not None:
            w -= beta_previous * q_previous

        w = _orthogonalize(w, [*external, *q_columns, q])
        b = float(np.linalg.norm(w))

        q_columns.append(q.copy())
        u_columns.append(hq.copy())
        alpha.append(a)
        m = len(q_columns)
        projected = np.diag(np.asarray(alpha, dtype=float))
        if m > 1:
            offdiag = np.asarray(beta_links[: m - 1], dtype=float)
            projected += np.diag(offdiag, 1) + np.diag(offdiag, -1)
        lowest, coeff = _lowest_ritz(projected)

        eigenvalue_change = _change(
            lowest,
            old_eigenvalue,
            mode="relative",
            zero_tolerance=breakdown_tolerance,
        )
        old_eigenvalue = lowest

        # Use the full current Ritz residual from the retained *actual* HVP
        # actions.  This is intentionally not the three-term estimate
        # abs(beta*c_last): full reorthogonalization and finite-difference HVPs
        # can make that estimate differ from ||U c - theta Q c||.
        Q_current = np.column_stack(q_columns)
        U_current = np.column_stack(u_columns)
        ritz_mode = Q_current @ coeff
        ritz_residual = U_current @ coeff - lowest * ritz_mode
        residual_norm = float(np.linalg.norm(ritz_residual))
        residual_threshold = residual_gamma * abs(float(lowest))
        converged = bool(residual_norm < residual_threshold)
        if converged:
            break
        if b <= breakdown_tolerance:
            breakdown = True
            break

        beta_links.append(b)
        q_previous = q
        beta_previous = b
        q = w / b

    Q = np.column_stack(q_columns)
    mode = _normalize(Q @ coeff, breakdown_tolerance)
    mode = _sign_align(mode, initial_direction)
    return MinModeResult(
        eigenvalue=float(lowest),
        eigenvector=mode,
        iterations=len(q_columns),
        converged=bool(converged),
        breakdown=bool(breakdown),
        eigenvalue_change=eigenvalue_change,
        residual_norm=float(residual_norm),
        subspace_dimension=len(q_columns),
        solver_used="lanczos",
        audit=_legacy_solver_audit(
            q_columns, u_columns, theta=lowest, coefficients=coeff,
            returned_mode=mode, stopping_rule="full_ritz_residual_gamma_0.1",
            stopping_equation="||H v - theta v|| < 0.1 * |theta|",
            stopping_reason=_stop_reason(
                converged=converged, breakdown=breakdown,
                exhausted=(not converged and not breakdown and len(q_columns) >= max_iterations),
            ),
            breakdown_reason=("no_independent_lanczos_direction" if breakdown else ""),
            solver_operator_residual_norm=float(residual_norm),
            typed_hvp_results=getattr(hvp, "results", ()),
            hvp_count_override=len(q_columns),
            metadata={
                "residual_gamma": residual_gamma,
                "residual_threshold": residual_threshold,
                "residual_source": "retained_actual_hvp_actions",
                "eigenvalue_tolerance_controls_convergence": False,
                "previous_eigenvalue_controls_convergence": False,
            },
        ),
    )


# ---------------------------------------------------------------------------
# Historical SoftSaddle Lanczos
# ---------------------------------------------------------------------------

def softsaddle_lanczos_lowest_mode(
    hvp: Callable[[Array], Array],
    initial_vector: Array,
    *,
    max_iterations: int = 8,
    eigenvalue_tolerance: float = 0.01,
    breakdown_tolerance: float = 1.0e-12,
    previous_eigenvalue: float | None = None,
    convergence_mode: str = "relative",
    initial_hvp: Array | None = None,
) -> MinModeResult:
    """Historical SoftSaddle three-term Lanczos recurrence.

    ``convergence_mode='relative'`` reproduces the standalone ``liblanczos``
    benchmark.  The Lanczos fallback embedded in historical ``libdavidson``
    used absolute eigenvalue change instead; the hybrid driver requests that
    mode explicitly.

    Unlike the generic implementation, this routine intentionally does *not*
    full-reorthogonalize the recurrence because SoftSaddle did not.
    """
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    eigenvalue_tolerance = float(eigenvalue_tolerance)
    if eigenvalue_tolerance < 0.0:
        raise ValueError("eigenvalue_tolerance must be >= 0")
    breakdown_tolerance = float(breakdown_tolerance)
    if breakdown_tolerance <= 0.0:
        raise ValueError("breakdown_tolerance must be > 0")
    convergence_mode = str(convergence_mode).lower()
    if convergence_mode not in {"relative", "absolute"}:
        raise ValueError("convergence_mode must be relative or absolute")

    raw = _as_vector(initial_vector)
    q = _normalize(raw, breakdown_tolerance)
    initial_direction = q.copy()
    q_previous = np.zeros_like(q)
    beta_previous = 0.0

    accepted_q: list[Array] = []
    accepted_u: list[Array] = []
    alpha: list[float] = []
    beta_links: list[float] = []
    old_eigenvalue = (
        None if previous_eigenvalue is None else float(previous_eigenvalue)
    )
    eigenvalue_change: float | None = None
    converged = False
    breakdown = False
    lowest = float("nan")
    coeff = np.ones(1, dtype=float)
    residual_norm = float("nan")
    hvp_calls = 0

    # Saved previous Ritz pair lets us reproduce SoftSaddle's behavior of
    # discarding the current basis vector when the new beta breaks down.
    saved_lowest: float | None = None
    saved_coeff: Array | None = None
    saved_Q: Array | None = None
    saved_U: Array | None = None
    saved_residual_norm = float("nan")

    for iteration in range(max_iterations):
        if iteration == 0 and initial_hvp is not None:
            hq = _as_vector(initial_hvp)
        else:
            hq = _as_vector(hvp(q))
        hvp_calls += 1
        if hq.shape != q.shape:
            raise ValueError("HVP callback returned a vector with wrong shape")

        recurrence = hq - beta_previous * q_previous
        a = float(np.dot(q, recurrence))
        recurrence = recurrence - a * q
        b = float(np.linalg.norm(recurrence))

        # Historical criterion: beta <= 1e-10 * |alpha|.  Keep an absolute
        # floor only for the alpha==0 floating-point corner so we never divide
        # a numerically zero vector by its norm.
        historical_breakdown = b <= 1.0e-10 * abs(a)
        numerical_breakdown = b <= breakdown_tolerance
        if historical_breakdown or numerical_breakdown:
            breakdown = True
            if saved_Q is None:
                # First direction is already an exact/numerical eigenvector.
                lowest = float(np.dot(q, hq))
                coeff = np.ones(1, dtype=float)
                accepted_q = [q.copy()]
                accepted_u = [hq.copy()]
                residual_norm = b
            else:
                lowest = float(saved_lowest)
                coeff = np.asarray(saved_coeff, dtype=float)
                accepted_q = [saved_Q[:, i].copy() for i in range(saved_Q.shape[1])]
                accepted_u = [saved_U[:, i].copy() for i in range(saved_U.shape[1])]
                residual_norm = float(saved_residual_norm)
            break

        accepted_q.append(q.copy())
        accepted_u.append(hq.copy())
        alpha.append(a)
        m = len(accepted_q)
        projected = np.diag(np.asarray(alpha, dtype=float))
        if m > 1:
            offdiag = np.asarray(beta_links[: m - 1], dtype=float)
            projected += np.diag(offdiag, 1) + np.diag(offdiag, -1)
        lowest, coeff = _lowest_ritz(projected)
        residual_norm = abs(b * float(coeff[-1]))

        eigenvalue_change = _change(
            lowest,
            old_eigenvalue,
            mode=convergence_mode,
            zero_tolerance=breakdown_tolerance,
        )
        if (
            eigenvalue_change is not None
            and eigenvalue_change < eigenvalue_tolerance
        ):
            converged = True

        Q_now = np.column_stack(accepted_q)
        saved_lowest = lowest
        saved_coeff = coeff.copy()
        saved_Q = Q_now.copy()
        saved_U = np.column_stack(accepted_u).copy()
        saved_residual_norm = residual_norm
        old_eigenvalue = lowest
        if converged:
            break

        beta_links.append(b)
        q_previous = q
        beta_previous = b
        q = recurrence / b
    else:
        # Loop exhausted normally at the hard HVP budget.
        pass

    Q = np.column_stack(accepted_q)
    mode = _normalize(Q @ coeff, breakdown_tolerance)
    mode = _sign_align(mode, initial_direction)
    return MinModeResult(
        eigenvalue=float(lowest),
        eigenvector=mode,
        iterations=int(hvp_calls),
        converged=bool(converged),
        breakdown=bool(breakdown),
        eigenvalue_change=eigenvalue_change,
        residual_norm=float(residual_norm),
        subspace_dimension=int(Q.shape[1]),
        solver_used="softsaddle_lanczos",
        audit=_legacy_solver_audit(
            accepted_q, accepted_u, theta=lowest, coefficients=coeff,
            returned_mode=mode,
            stopping_rule=f"{convergence_mode}_eigenvalue_change",
            stopping_equation=(
                "abs((lambda_k-lambda_prev)/lambda_prev) < eigenvalue_tolerance"
                if convergence_mode == "relative" else
                "abs(lambda_k-lambda_prev) < eigenvalue_tolerance"
            ),
            stopping_reason=_stop_reason(
                converged=converged, breakdown=breakdown,
                exhausted=(not converged and not breakdown and hvp_calls >= max_iterations),
            ),
            breakdown_reason=("historical_or_numerical_beta_breakdown" if breakdown else ""),
            solver_operator_residual_norm=float(residual_norm),
            typed_hvp_results=getattr(hvp, "results", ()),
            hvp_count_override=hvp_calls,
            metadata={"historical_softsaddle_recurrence": True},
        ),
    )


# ---------------------------------------------------------------------------
# Davidson helpers and generic Davidson
# ---------------------------------------------------------------------------

def _safe_shifted_diagonal(
    eigenvalue: float,
    hessian_diagonal: Array,
    floor: float,
) -> Array:
    denominator = float(eigenvalue) - np.asarray(hessian_diagonal, dtype=float)
    floor = float(floor)
    if floor < 0.0:
        raise ValueError("preconditioner_floor must be >= 0")
    if floor == 0.0:
        return denominator
    small = np.abs(denominator) < floor
    if np.any(small):
        signs = np.sign(denominator[small])
        signs[signs == 0.0] = 1.0
        denominator = denominator.copy()
        denominator[small] = signs * floor
    return denominator


def _validated_hessian(matrix: Array, n: int, name: str) -> Array:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (n, n):
        raise ValueError(f"{name} must have shape {(n, n)}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return 0.5 * (matrix + matrix.T)


def davidson_lowest_mode(
    hvp: Callable[[Array], Array],
    initial_vector: Array,
    preconditioner_hessian: Array,
    *,
    max_iterations: int = 8,
    eigenvalue_tolerance: float = 0.01,
    breakdown_tolerance: float = 1.0e-12,
    preconditioner_floor: float = 1.0e-8,
    previous_eigenvalue: float | None = None,
    excluded_basis: Iterable[Array] = (),
) -> MinModeResult:
    """Generic textbook Davidson with a diagonal shifted-Hessian preconditioner."""
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    eigenvalue_tolerance = float(eigenvalue_tolerance)
    if eigenvalue_tolerance < 0.0:
        raise ValueError("eigenvalue_tolerance must be >= 0")
    breakdown_tolerance = float(breakdown_tolerance)
    if breakdown_tolerance <= 0.0:
        raise ValueError("breakdown_tolerance must be > 0")

    n = _as_vector(initial_vector).size
    B = _validated_hessian(preconditioner_hessian, n, "preconditioner_hessian")
    external = [_normalize(v, breakdown_tolerance) for v in excluded_basis]
    q0 = _orthogonalize(initial_vector, external)
    q = _normalize(q0, breakdown_tolerance)
    initial_direction = q.copy()

    Q_list: list[Array] = []
    U_list: list[Array] = []
    old_eigenvalue = (
        None if previous_eigenvalue is None else float(previous_eigenvalue)
    )
    eigenvalue_change: Optional[float] = None
    converged = False
    breakdown = False
    lowest = float("nan")
    coeff = np.ones(1)
    residual = np.zeros_like(q)

    for _iteration in range(max_iterations):
        hq = _as_vector(hvp(q))
        if hq.shape != q.shape:
            raise ValueError("HVP callback returned a vector with wrong shape")
        Q_list.append(q.copy())
        U_list.append(hq.copy())

        Q = np.column_stack(Q_list)
        U = np.column_stack(U_list)
        lowest, coeff = _lowest_ritz(Q.T @ U)
        mode = Q @ coeff
        hmode = U @ coeff
        residual = hmode - lowest * mode
        residual = _orthogonalize(residual, [*external, *Q_list])

        eigenvalue_change = _change(
            lowest,
            old_eigenvalue,
            mode="absolute",
            zero_tolerance=breakdown_tolerance,
        )
        if (
            eigenvalue_change is not None
            and eigenvalue_change < eigenvalue_tolerance
        ):
            converged = True
        old_eigenvalue = lowest
        if converged:
            break

        denominator = _safe_shifted_diagonal(
            lowest, np.diag(B), preconditioner_floor
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            correction = residual / denominator
        if not np.all(np.isfinite(correction)):
            breakdown = True
            break
        correction = _orthogonalize(correction, [*external, *Q_list])
        correction_norm = float(np.linalg.norm(correction))

        if correction_norm <= breakdown_tolerance:
            # Generic Davidson uses the true Ritz residual as a safe fallback.
            correction = residual.copy()
            correction_norm = float(np.linalg.norm(correction))
        if correction_norm <= breakdown_tolerance:
            breakdown = True
            break
        q = correction / correction_norm

    Q = np.column_stack(Q_list)
    mode = _normalize(Q @ coeff, breakdown_tolerance)
    mode = _sign_align(mode, initial_direction)
    return MinModeResult(
        eigenvalue=float(lowest),
        eigenvector=mode,
        iterations=len(Q_list),
        converged=bool(converged),
        breakdown=bool(breakdown),
        eigenvalue_change=eigenvalue_change,
        residual_norm=float(np.linalg.norm(residual)),
        subspace_dimension=len(Q_list),
        solver_used="davidson",
        residual_variant="textbook",
        audit=_legacy_solver_audit(
            Q_list, U_list, theta=lowest, coefficients=coeff,
            returned_mode=mode, stopping_rule="absolute_eigenvalue_change",
            stopping_equation="abs(lambda_k-lambda_prev) < eigenvalue_tolerance",
            stopping_reason=_stop_reason(
                converged=converged, breakdown=breakdown,
                exhausted=(not converged and not breakdown and len(Q_list) >= max_iterations),
            ),
            breakdown_reason=("no_independent_davidson_correction" if breakdown else ""),
            solver_operator_residual_norm=float(np.linalg.norm(residual)),
            typed_hvp_results=getattr(hvp, "results", ()),
            hvp_count_override=len(Q_list),
            metadata={"preconditioner": "diagonal_of_supplied_physical_B"},
        ),
    )


# ---------------------------------------------------------------------------
# Historical SoftSaddle Davidson branch and hybrid selector
# ---------------------------------------------------------------------------

def softsaddle_davidson_lowest_mode(
    hvp: Callable[[Array], Array],
    initial_vector: Array,
    preconditioner_hessian: Array,
    *,
    max_iterations: int = 8,
    eigenvalue_tolerance: float = 0.01,
    breakdown_tolerance: float = 1.0e-12,
    preconditioner_floor: float = 1.0e-12,
    previous_eigenvalue: float | None = None,
    residual_variant: str = "legacy_unweighted",
    initial_hvp: Array | None = None,
) -> MinModeResult:
    """SoftSaddle Davidson branch with selectable residual construction.

    ``legacy_unweighted`` reproduces the historical C++ lines

        x += Hq_i
        t += lambda*q_i
        wd = x - t

    which omit the Ritz coefficients.  ``textbook`` changes only that one
    point to ``wd = Uc - lambda Qc``.  Final Ritz-vector reconstruction uses
    the projected eigenvector coefficients in *both* variants, matching the
    historical source.
    """
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    eigenvalue_tolerance = float(eigenvalue_tolerance)
    if eigenvalue_tolerance < 0.0:
        raise ValueError("eigenvalue_tolerance must be >= 0")
    breakdown_tolerance = float(breakdown_tolerance)
    if breakdown_tolerance <= 0.0:
        raise ValueError("breakdown_tolerance must be > 0")
    residual_variant = str(residual_variant).lower()
    if residual_variant not in {"legacy_unweighted", "textbook"}:
        raise ValueError(
            "residual_variant must be legacy_unweighted or textbook"
        )

    raw = _as_vector(initial_vector)
    q = _normalize(raw, breakdown_tolerance)
    initial_direction = q.copy()
    B = _validated_hessian(preconditioner_hessian, q.size, "preconditioner_hessian")

    Q_list: list[Array] = []
    U_list: list[Array] = []
    old_eigenvalue = (
        None if previous_eigenvalue is None else float(previous_eigenvalue)
    )
    eigenvalue_change: float | None = None
    converged = False
    breakdown = False
    lowest = float("nan")
    coeff = np.ones(1, dtype=float)
    true_residual = np.zeros_like(q)
    hvp_calls = 0

    for iteration in range(max_iterations):
        if iteration == 0 and initial_hvp is not None:
            hq = _as_vector(initial_hvp)
        else:
            hq = _as_vector(hvp(q))
        hvp_calls += 1
        if hq.shape != q.shape:
            raise ValueError("HVP callback returned a vector with wrong shape")

        Q_list.append(q.copy())
        U_list.append(hq.copy())
        Q = np.column_stack(Q_list)
        U = np.column_stack(U_list)
        lowest, coeff = _lowest_ritz(Q.T @ U)

        true_residual = U @ coeff - lowest * (Q @ coeff)
        if residual_variant == "legacy_unweighted":
            work_residual = np.sum(U, axis=1) - lowest * np.sum(Q, axis=1)
        else:
            work_residual = true_residual.copy()

        denominator = _safe_shifted_diagonal(
            lowest, np.diag(B), preconditioner_floor
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            correction = work_residual / denominator
        if not np.all(np.isfinite(correction)):
            breakdown = True
            break

        # Historical SoftSaddle projects the preconditioned correction against
        # every current Davidson basis vector after the diagonal scaling.
        correction = _orthogonalize(correction, Q_list, passes=1)
        correction_norm = float(np.linalg.norm(correction))

        eigenvalue_change = _change(
            lowest,
            old_eigenvalue,
            mode="absolute",
            zero_tolerance=breakdown_tolerance,
        )
        if (
            eigenvalue_change is not None
            and eigenvalue_change < eigenvalue_tolerance
        ):
            converged = True
        old_eigenvalue = lowest
        if converged:
            break
        if correction_norm <= breakdown_tolerance:
            breakdown = True
            break
        q = correction / correction_norm

    Q = np.column_stack(Q_list)
    mode = _normalize(Q @ coeff, breakdown_tolerance)
    mode = _sign_align(mode, initial_direction)
    return MinModeResult(
        eigenvalue=float(lowest),
        eigenvector=mode,
        iterations=int(hvp_calls),
        converged=bool(converged),
        breakdown=bool(breakdown),
        eigenvalue_change=eigenvalue_change,
        residual_norm=float(np.linalg.norm(true_residual)),
        subspace_dimension=int(Q.shape[1]),
        solver_used="softsaddle_davidson",
        residual_variant=residual_variant,
        audit=_legacy_solver_audit(
            Q_list, U_list, theta=lowest, coefficients=coeff,
            returned_mode=mode, stopping_rule="absolute_eigenvalue_change",
            stopping_equation="abs(lambda_k-lambda_prev) < eigenvalue_tolerance",
            stopping_reason=_stop_reason(
                converged=converged, breakdown=breakdown,
                exhausted=(not converged and not breakdown and hvp_calls >= max_iterations),
            ),
            breakdown_reason=("no_independent_softsaddle_correction" if breakdown else ""),
            solver_operator_residual_norm=float(np.linalg.norm(true_residual)),
            typed_hvp_results=getattr(hvp, "results", ()),
            hvp_count_override=hvp_calls,
            metadata={
                "historical_softsaddle_davidson": True,
                "residual_variant": residual_variant,
            },
        ),
    )


def softsaddle_davidson_hybrid_lowest_mode(
    hvp: Callable[[Array], Array],
    initial_vector: Array,
    preconditioner_hessian: Array,
    *,
    max_iterations: int = 8,
    eigenvalue_tolerance: float = 0.01,
    breakdown_tolerance: float = 1.0e-12,
    preconditioner_floor: float = 1.0e-12,
    previous_eigenvalue: float | None = None,
    switch_threshold: float = 12.0,
    residual_variant: str = "legacy_unweighted",
) -> MinModeResult:
    """Historical SoftSaddle Davidson/Lanczos hybrid driver.

    One true ``Hq0`` is evaluated first and compared with ``Bq0``.  If
    ``||Hq0-Bq0||*||v0||`` exceeds ``switch_threshold`` the historical embedded
    Lanczos branch is used; otherwise the Davidson branch is used.  The first
    HVP is passed into the selected branch so it is not evaluated twice.
    """
    raw = _as_vector(initial_vector)
    raw_norm = float(np.linalg.norm(raw))
    q0 = _normalize(raw, breakdown_tolerance)
    B = _validated_hessian(preconditioner_hessian, q0.size, "preconditioner_hessian")
    hq0 = _as_vector(hvp(q0))
    if hq0.shape != q0.shape:
        raise ValueError("HVP callback returned a vector with wrong shape")

    switch_metric = float(np.linalg.norm(hq0 - B @ q0) * raw_norm)
    if switch_metric > float(switch_threshold):
        result = softsaddle_lanczos_lowest_mode(
            hvp,
            q0,
            max_iterations=max_iterations,
            eigenvalue_tolerance=eigenvalue_tolerance,
            breakdown_tolerance=breakdown_tolerance,
            previous_eigenvalue=previous_eigenvalue,
            # Historical libdavidson's embedded Lanczos branch uses absolute
            # eigenvalue change even though standalone liblanczos uses relative.
            convergence_mode="absolute",
            initial_hvp=hq0,
        )
        return replace(
            result,
            solver_used="softsaddle_hybrid_lanczos",
            switch_metric=switch_metric,
            residual_variant="",
        )

    result = softsaddle_davidson_lowest_mode(
        hvp,
        q0,
        B,
        max_iterations=max_iterations,
        eigenvalue_tolerance=eigenvalue_tolerance,
        breakdown_tolerance=breakdown_tolerance,
        preconditioner_floor=preconditioner_floor,
        previous_eigenvalue=previous_eigenvalue,
        residual_variant=residual_variant,
        initial_hvp=hq0,
    )
    return replace(
        result,
        solver_used="softsaddle_hybrid_davidson",
        switch_metric=switch_metric,
    )


# ---------------------------------------------------------------------------
# Persistent physical Hessian model used by Davidson preconditioning
# ---------------------------------------------------------------------------

class PhysicalHessianBFGS:
    """Persistent dense physical-Hessian approximation for Davidson.

    Davidson itself does *not* construct this matrix.  The outer saddle-search
    wrapper owns it and updates it from physical center-geometry secants.

    ``initial_matrix`` permits a full reference-minimum Hessian to seed the
    model, as in historical SoftSaddle.  ``seed_reference`` can simultaneously
    establish the reference position/gradient so the first displaced geometry
    performs a BFGS transport update from that minimum.
    """

    def __init__(
        self,
        dimension: int,
        *,
        initial_hessian: float = 1.0,
        denominator_tolerance: float = 1.0e-12,
        initial_matrix: Array | None = None,
        initial_position: Array | None = None,
        initial_gradient: Array | None = None,
    ):
        dimension = int(dimension)
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self.dimension = dimension
        self.denominator_tolerance = float(denominator_tolerance)
        if self.denominator_tolerance <= 0.0:
            raise ValueError("denominator_tolerance must be > 0")

        if initial_matrix is None:
            scale = float(initial_hessian)
            if not np.isfinite(scale) or scale == 0.0:
                raise ValueError("initial_hessian must be finite and nonzero")
            self.matrix = np.eye(dimension, dtype=float) * scale
        else:
            self.matrix = _validated_hessian(
                initial_matrix, dimension, "initial_matrix"
            )

        self.previous_position: Array | None = None
        self.previous_gradient: Array | None = None
        self.accepted_updates = 0
        self.skipped_updates = 0

        if (initial_position is None) != (initial_gradient is None):
            raise ValueError(
                "initial_position and initial_gradient must be supplied together"
            )
        if initial_position is not None:
            self.seed_reference(initial_position, initial_gradient)

    def seed_reference(self, position: Array, gradient: Array) -> None:
        position = _as_vector(position)
        gradient = _as_vector(gradient)
        if position.size != self.dimension or gradient.size != self.dimension:
            raise ValueError("Reference position/gradient dimension mismatch")
        self.previous_position = position.copy()
        self.previous_gradient = gradient.copy()

    def observe_center(self, position: Array, gradient: Array) -> bool:
        """BFGS-update B from the previous observed physical PES center."""
        position = _as_vector(position)
        gradient = _as_vector(gradient)
        if position.size != self.dimension or gradient.size != self.dimension:
            raise ValueError("Position/gradient dimension does not match Hessian")

        if self.previous_position is None:
            self.seed_reference(position, gradient)
            return False

        s = position - self.previous_position
        y = gradient - self.previous_gradient
        self.previous_position = position.copy()
        self.previous_gradient = gradient.copy()

        sy = float(np.dot(s, y))
        Bs = self.matrix @ s
        sBs = float(np.dot(s, Bs))
        tol = self.denominator_tolerance
        if abs(sy) <= tol or abs(sBs) <= tol:
            self.skipped_updates += 1
            return False

        self.matrix += np.outer(y, y) / sy - np.outer(Bs, Bs) / sBs
        self.matrix = 0.5 * (self.matrix + self.matrix.T)
        if not np.all(np.isfinite(self.matrix)):
            raise FloatingPointError("Davidson BFGS Hessian became non-finite")
        self.accepted_updates += 1
        return True


# ---------------------------------------------------------------------------
# projected Olsen/Jacobi-Davidson correction and typed HVP solver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OlsenCorrectionResult:
    """Structured result from one projected Olsen/JD correction solve."""

    correction: Array | None
    residual: Array
    theta: float
    orthogonality_error: float
    shifted_system_residual_norm: float | None
    condition_number: float | None
    singularity_status: str
    solve_method: str
    fallback: str
    independent: bool
    breakdown_reason: str
    operator_identity: str
    preconditioner_identity: str
    coordinate_space_id: str
    residual_norm: float


@dataclass(frozen=True)
class SolverWorkAccounting:
    algorithm_pes_calls: int
    diagnostic_pes_calls: int
    physical_total_pes_calls: int
    cache_hits: int
    physical_actions: int
    model_actions: int
    certification_eligible: bool
    origins: tuple[str, ...]
    sources: tuple[str, ...]
    families: tuple[str, ...]


def summarize_hvp_actions(actions: Sequence[object]) -> SolverWorkAccounting:
    """Summarize typed HVP provenance/accounting without reinterpreting it."""
    from saddlemill.dimertools.foundation_types import WorkPurpose

    algorithm = diagnostic = cache_hits = physical = model = 0
    origins: list[str] = []
    sources: list[str] = []
    families: list[str] = []
    certification = bool(actions)
    for item in actions:
        metadata = getattr(item, "metadata", None)
        if metadata is None:
            raise TypeError("typed solver accounting requires HVPResult-like metadata")
        delta = int(getattr(metadata, "pes_call_delta", 0))
        purpose = getattr(metadata, "purpose", WorkPurpose.ALGORITHM)
        if purpose is WorkPurpose.ALGORITHM or str(purpose) == WorkPurpose.ALGORITHM.value:
            algorithm += delta
        else:
            diagnostic += delta
        cache_hits += int(bool(getattr(metadata, "cache_hit", False)))
        physical += int(bool(getattr(metadata, "physical", False)))
        model += int(bool(getattr(metadata, "model_derived", False)))
        certification = certification and bool(getattr(item, "certification_eligible", False))
        origins.append(str(getattr(metadata, "operator_origin", "")))
        sources.append(str(getattr(metadata, "source", "")))
        families.append(str(getattr(metadata, "family", "")))
    return SolverWorkAccounting(
        algorithm_pes_calls=algorithm,
        diagnostic_pes_calls=diagnostic,
        physical_total_pes_calls=algorithm + diagnostic,
        cache_hits=cache_hits,
        physical_actions=physical,
        model_actions=model,
        certification_eligible=certification,
        origins=tuple(origins),
        sources=tuple(sources),
        families=tuple(families),
    )


def _active_basis_matrix(coordinate_space) -> Array:
    """Return an orthonormal full-Cartesian basis for one active coordinate space."""
    mask = np.asarray(coordinate_space.active_dof_mask, dtype=bool).reshape(-1)
    full = int(mask.size)
    active = np.flatnonzero(mask)
    if active.size == 0:
        raise ValueError("active coordinate space has no movable degrees of freedom")
    candidates = np.zeros((full, active.size), dtype=float)
    candidates[active, np.arange(active.size)] = 1.0
    for base in coordinate_space.null_basis:
        b = np.asarray(base, dtype=float).reshape(-1)
        candidates -= np.outer(b, b @ candidates)
    u, singular, _ = np.linalg.svd(candidates, full_matrices=False)
    if singular.size == 0:
        raise ValueError("active coordinate projection has numerical rank zero")
    threshold = max(1.0e-13, 1.0e-13 * float(singular[0]))
    rank = int(np.count_nonzero(singular > threshold))
    if rank < 1:
        raise ValueError("active coordinate projection has numerical rank zero")
    return u[:, :rank]


def _project_flat(coordinate_space, vector: object) -> Array:
    return np.asarray(coordinate_space.project(vector), dtype=float).reshape(-1)


def _normalised_flat(coordinate_space, vector: object, tolerance: float) -> Array:
    arr = _project_flat(coordinate_space, vector)
    nrm = float(np.linalg.norm(arr))
    if not np.isfinite(nrm) or nrm <= float(tolerance):
        raise ValueError("active-space vector norm is approximately zero")
    return arr / nrm


def _reorthogonalize_active(
    vector_flat: Array,
    basis_flat: Sequence[Array],
    *,
    coordinate_space,
    passes: int = 2,
) -> Array:
    shape = coordinate_space.active_dof_mask.shape
    out = _project_flat(coordinate_space, np.asarray(vector_flat).reshape(shape))
    bases = []
    for item in basis_flat:
        b = _project_flat(coordinate_space, np.asarray(item).reshape(shape))
        nrm = float(np.linalg.norm(b))
        if nrm > 1.0e-14:
            bases.append(b / nrm)
    for _ in range(max(1, int(passes))):
        for b in bases:
            out -= float(np.dot(b, out)) * b
        out = _project_flat(coordinate_space, out.reshape(shape))
    return out


def projected_olsen_jd_correction(
    mode: object,
    action: object,
    preconditioner_matrix: object,
    *,
    coordinate_space,
    theta: float | None = None,
    basis_vectors: Sequence[object] = (),
    singular_condition_threshold: float = 1.0e12,
    pseudoinverse_rcond: float = 1.0e-12,
    diagonal_floor: float = 1.0e-10,
    breakdown_tolerance: float = 1.0e-12,
    operator_identity: str = "",
    preconditioner_identity: str = "",
) -> OlsenCorrectionResult:
    """Solve one projected Olsen/Jacobi-Davidson correction equation.

    In the explicit active free-coordinate basis ``W``, with normalized mode
    ``v``, this solves the augmented system equivalent to

    ``(I-vv.T) (B-theta I) (I-vv.T) t = -r`` and ``v.T t = 0``.

    Hierarchy: well-conditioned augmented solve; Moore-Penrose pseudoinverse
    for singular/near-singular systems; projected diagonal-preconditioned
    residual; raw projected residual; then clean breakdown.  The returned
    direction is reorthogonalized against the complete supplied basis and the
    coordinate-space constraints.
    """
    from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace

    if not isinstance(coordinate_space, ActiveCoordinateSpace):
        raise TypeError("coordinate_space must be ActiveCoordinateSpace")
    if singular_condition_threshold <= 1.0:
        raise ValueError("singular_condition_threshold must be > 1")
    if pseudoinverse_rcond <= 0.0 or diagonal_floor <= 0.0 or breakdown_tolerance <= 0.0:
        raise ValueError("pseudoinverse_rcond, diagonal_floor, and breakdown_tolerance must be > 0")

    shape = coordinate_space.active_dof_mask.shape
    W = _active_basis_matrix(coordinate_space)
    v_full = _normalised_flat(coordinate_space, mode, breakdown_tolerance)
    hv_full = _project_flat(coordinate_space, action)
    v = W.T @ v_full
    hv = W.T @ hv_full
    v /= np.linalg.norm(v)
    theta_value = float(v @ hv) if theta is None else float(theta)
    if not np.isfinite(theta_value):
        raise ValueError("theta must be finite")
    residual = hv - theta_value * v
    residual -= v * float(v @ residual)
    residual_norm = float(np.linalg.norm(residual))

    raw_B = np.asarray(preconditioner_matrix, dtype=float)
    if not np.all(np.isfinite(raw_B)):
        raise ValueError("preconditioner_matrix contains non-finite values")
    full_dim = int(np.prod(shape))
    reduced_dim = int(W.shape[1])
    if raw_B.shape == (full_dim, full_dim):
        B = W.T @ (0.5 * (raw_B + raw_B.T)) @ W
    elif raw_B.shape == (reduced_dim, reduced_dim):
        B = 0.5 * (raw_B + raw_B.T)
    else:
        raise ValueError(
            "preconditioner_matrix must be full Cartesian or active-reduced shape; "
            f"got {raw_B.shape}, expected {(full_dim, full_dim)} or {(reduced_dim, reduced_dim)}"
        )

    Pshift = B - theta_value * np.eye(reduced_dim)
    augmented = np.block(
        [[Pshift, v[:, None]], [v[None, :], np.zeros((1, 1), dtype=float)]]
    )
    rhs = np.concatenate([-residual, np.zeros(1, dtype=float)])
    condition: float | None
    try:
        condition = float(np.linalg.cond(augmented))
    except np.linalg.LinAlgError:
        condition = None
    near_singular = condition is None or not np.isfinite(condition) or condition >= float(singular_condition_threshold)

    all_basis = [v_full, *[_project_flat(coordinate_space, item) for item in basis_vectors]]
    tangent_projector = np.eye(reduced_dim) - np.outer(v, v)
    had_nonzero_candidate = False

    def finalize_candidate(trial: Array | None):
        nonlocal had_nonzero_candidate
        if trial is None or not np.all(np.isfinite(trial)):
            return None
        if float(np.linalg.norm(trial)) <= breakdown_tolerance:
            return None
        had_nonzero_candidate = True
        full = W @ np.asarray(trial, dtype=float).reshape(-1)
        full = _reorthogonalize_active(
            full, all_basis, coordinate_space=coordinate_space, passes=2
        )
        norm = float(np.linalg.norm(full))
        if not np.isfinite(norm) or norm <= breakdown_tolerance:
            return None
        reduced = W.T @ full
        # Report the equation residual of the *returned, reorthogonalized* t,
        # not the pre-reorthogonalization augmented-system candidate.
        equation_residual = float(
            np.linalg.norm(
                tangent_projector @ Pshift @ tangent_projector @ reduced + residual
            )
        )
        orthogonality = abs(float(v_full @ full))
        return full.reshape(shape), orthogonality, equation_residual

    correction: Array | None = None
    orthogonality_error = 0.0
    shifted_residual: float | None = None
    solve_method = ""
    fallback = ""
    singularity_status = "near_singular" if near_singular else "well_conditioned"

    if not near_singular:
        try:
            z = np.linalg.solve(augmented, rhs)
            accepted = finalize_candidate(z[:reduced_dim])
            if accepted is not None:
                correction, orthogonality_error, shifted_residual = accepted
                solve_method = "augmented_solve"
        except np.linalg.LinAlgError:
            near_singular = True
            singularity_status = "singular_solve_failure"

    if correction is None:
        try:
            z = np.linalg.pinv(augmented, rcond=float(pseudoinverse_rcond)) @ rhs
            accepted = finalize_candidate(z[:reduced_dim])
            if accepted is not None:
                correction, orthogonality_error, shifted_residual = accepted
                solve_method = "augmented_pseudoinverse"
                fallback = "pseudoinverse"
                singularity_status = (
                    "near_singular_pseudoinverse" if near_singular
                    else "dependent_augmented_pseudoinverse"
                )
        except np.linalg.LinAlgError:
            pass

    if correction is None:
        diag = np.diag(Pshift).copy()
        signs = np.sign(diag)
        signs[signs == 0.0] = 1.0
        safe_diag = np.where(np.abs(diag) < diagonal_floor, signs * diagonal_floor, diag)
        trial = -residual / safe_diag
        trial -= v * float(v @ trial)
        accepted = finalize_candidate(trial)
        if accepted is not None:
            correction, orthogonality_error, shifted_residual = accepted
            solve_method = "projected_diagonal_preconditioned_residual"
            fallback = "projected_preconditioned_residual"
            singularity_status = "fallback_preconditioned_residual"

    if correction is None:
        trial = -residual.copy()
        trial -= v * float(v @ trial)
        accepted = finalize_candidate(trial)
        if accepted is not None:
            correction, orthogonality_error, shifted_residual = accepted
            solve_method = "lanczos_residual"
            fallback = "lanczos_residual"
            singularity_status = "fallback_lanczos_residual"

    residual_full = (W @ residual).reshape(shape)
    if correction is None:
        return OlsenCorrectionResult(
            correction=None,
            residual=residual_full,
            theta=theta_value,
            orthogonality_error=0.0,
            shifted_system_residual_norm=None,
            condition_number=condition,
            singularity_status="breakdown",
            solve_method="none",
            fallback="clean_breakdown",
            independent=False,
            breakdown_reason=(
                "correction_not_independent_of_retained_basis"
                if had_nonzero_candidate
                else "no_nonzero_projected_residual_or_correction"
            ),
            operator_identity=str(operator_identity),
            preconditioner_identity=str(preconditioner_identity),
            coordinate_space_id=coordinate_space.identity,
            residual_norm=residual_norm,
        )

    return OlsenCorrectionResult(
        correction=correction,
        residual=residual_full,
        theta=theta_value,
        orthogonality_error=orthogonality_error,
        shifted_system_residual_norm=shifted_residual,
        condition_number=condition,
        singularity_status=singularity_status,
        solve_method=solve_method,
        fallback=fallback,
        independent=True,
        breakdown_reason="",
        operator_identity=str(operator_identity),
        preconditioner_identity=str(preconditioner_identity),
        coordinate_space_id=coordinate_space.identity,
        residual_norm=residual_norm,
    )


def _validate_hvp_result_identity(result, request) -> None:
    if not bool(getattr(result, "available", False)) or getattr(result, "action", None) is None:
        reason = str(getattr(result, "unavailable_reason", "unavailable"))
        raise RuntimeError(f"HVP unavailable for current solve identity: {reason}")
    md = result.metadata
    if (
        int(md.state_id) != int(request.state_id)
        or str(md.state_uid) != str(request.state_uid)
        or str(md.geometry_id) != str(request.geometry_id)
        or str(md.coordinate_space_id) != str(request.coordinate_space.identity)
    ):
        raise RuntimeError("HVP result identity does not match current solver request")


def _typed_solver_audit(
    all_actions: Sequence[object],
    *,
    retained_actions: Sequence[object],
    ritz,
    returned_mode: Array,
    stopping_rule: str,
    stopping_equation: str,
    stopping_reason: str,
    breakdown_reason: str,
    eigenvalue_change: float | None,
    restart_count: int,
    restart_reason: str,
    solver_operator_residual_norm: float | None,
    projected_operator_asymmetry_norm: float | None,
) -> SolverAuditMetadata:
    work = summarize_hvp_actions(all_actions)
    retained_work = summarize_hvp_actions(retained_actions)
    all_index_by_identity = {id(item): index for index, item in enumerate(all_actions)}
    retained_indices = tuple(
        int(all_index_by_identity[id(item)]) for item in retained_actions
    )
    bases = tuple(np.asarray(item.direction, dtype=float).reshape(-1).copy() for item in retained_actions)
    raw_actions = tuple(np.asarray(item.action, dtype=float).reshape(-1).copy() for item in retained_actions)
    evaluated = None if ritz.selected_vector is None else np.asarray(ritz.selected_vector, dtype=float).reshape(-1).copy()
    metadata = {
        "eigenvalue_change": eigenvalue_change,
        "full_residual_origin": (
            "physical_raw_same_center"
            if retained_actions and all(bool(item.metadata.physical) for item in retained_actions)
            else "model_or_nonphysical_raw_same_center"
        ),
        "all_hvp_origins": work.origins,
        "all_hvp_sources": work.sources,
        "all_hvp_families": work.families,
        "retained_hvp_count": len(retained_actions),
        "retained_hvp_indices": retained_indices,
        "retained_t01_match_count": len(retained_actions),
        "retained_t01_match_complete": True,
    }
    return SolverAuditMetadata(
        retained_basis=bases,
        retained_actions=raw_actions,
        evaluated_mode=evaluated,
        returned_mode=np.asarray(returned_mode, dtype=float).reshape(-1).copy(),
        selected_root_index=ritz.selected_index,
        root_selection=str(ritz.root_selection),
        stopping_rule=stopping_rule,
        stopping_equation=stopping_equation,
        stopping_reason=stopping_reason,
        breakdown_reason=breakdown_reason,
        hvp_count=len(all_actions),
        restart_count=restart_count,
        restart_reason=restart_reason,
        rank=int(ritz.rank),
        full_residual_norm=ritz.full_residual_norm,
        solver_operator_residual_norm=solver_operator_residual_norm,
        projected_operator_asymmetry_norm=projected_operator_asymmetry_norm,
        action_origins=retained_work.origins,
        action_sources=retained_work.sources,
        action_families=retained_work.families,
        action_stencil_schemes=tuple(str(getattr(item, "stencil_scheme", "")) for item in retained_actions),
        action_displacement_scales=tuple(getattr(item, "displacement_scale", None) for item in retained_actions),
        action_geometry_ids=tuple(str(item.metadata.geometry_id) for item in retained_actions),
        action_state_uids=tuple(str(item.metadata.state_uid) for item in retained_actions),
        physical_action_count=work.physical_actions,
        model_action_count=work.model_actions,
        certification_eligible=work.certification_eligible,
        algorithm_pes_calls=work.algorithm_pes_calls,
        diagnostic_pes_calls=work.diagnostic_pes_calls,
        cache_hits=work.cache_hits,
        gap=ritz.gap,
        gap_eligible=bool(ritz.gap_eligible),
        degeneracy_unresolved=bool(ritz.degeneracy_unresolved),
        approximation_caveat=str(ritz.approximation_caveat),
        metadata=metadata,
    )


class _SellaPhysicalHVPLinearOperator:
    """Reduced-coordinate LinearOperator adapter for Sella 2.5.0 JD0.

    Sella's ``rayleigh_ritz`` requires only ``shape`` and ``dot(V)`` from the
    physical Hessian operator.  SaddleMill keeps ownership of the physical HVP:
    each column passed by Sella is mapped back through the active-coordinate
    basis, evaluated through the existing sealed typed-HVP backend, and mapped into
    the reduced orthonormal coordinate basis.  No calculator/HVP definition is
    changed by this adapter.
    """

    def __init__(
        self,
        backend,
        *,
        coordinate_space,
        state_id: int,
        state_uid: str,
        geometry_id: str,
        source: str,
        family: str,
        purpose,
        breakdown_tolerance: float,
    ) -> None:
        from saddlemill.dimertools.hvp_interfaces import HVPRequest

        self.backend = backend
        self.coordinate_space = coordinate_space
        self.state_id = int(state_id)
        self.state_uid = str(state_uid)
        self.geometry_id = str(geometry_id)
        self.source = str(source)
        self.family = str(family)
        self.purpose = purpose
        self.breakdown_tolerance = float(breakdown_tolerance)
        self.basis = _active_basis_matrix(coordinate_space)
        n = int(self.basis.shape[1])
        self.shape = (n, n)
        self.results: list[object] = []
        self._HVPRequest = HVPRequest

    def dot(self, vectors: object) -> Array:
        array = np.asarray(vectors, dtype=float)
        was_vector = array.ndim == 1
        if was_vector:
            array = array.reshape((-1, 1))
        if array.ndim != 2 or array.shape[0] != self.shape[0]:
            raise ValueError(
                "Sella HVP operator input must have shape "
                f"({self.shape[0]}, k); got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("Sella HVP operator input contains non-finite values")

        shape = self.coordinate_space.active_dof_mask.shape
        output = np.empty_like(array, dtype=float)
        for column in range(array.shape[1]):
            reduced = np.asarray(array[:, column], dtype=float)
            magnitude = float(np.linalg.norm(reduced))
            if not np.isfinite(magnitude) or magnitude <= self.breakdown_tolerance:
                raise ValueError("Sella requested a numerically zero HVP direction")

            # The live physical-FD backend deliberately normalizes probe
            # directions before applying its fixed displacement.  Preserve the
            # linear-operator contract expected by Sella by evaluating the unit
            # direction and restoring the input magnitude on the returned HVP.
            unit_reduced = reduced / magnitude
            full_direction = (self.basis @ unit_reduced).reshape(shape)
            request = self._HVPRequest(
                state_id=self.state_id,
                state_uid=self.state_uid,
                geometry_id=self.geometry_id,
                direction=full_direction,
                coordinate_space=self.coordinate_space,
                purpose=self.purpose,
                source=self.source,
                family=self.family,
                metadata={
                    "solver": "sella_rayleigh_ritz_jd0",
                    "hvp_index": len(self.results),
                },
            )
            result = self.backend.apply(request)
            _validate_hvp_result_identity(result, request)
            self.results.append(result)
            full_action = _project_flat(self.coordinate_space, result.action)
            output[:, column] = magnitude * (self.basis.T @ full_action)

        if was_vector:
            return output[:, 0]
        return output


def _sella_25_rayleigh_ritz():
    """Load the authoritative installed Sella 2.5.0 eigensolver lazily."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            sella_version = version("sella")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "min_mode_finder=olsen_jd requires installed Sella 2.5.0 "
                "with distribution metadata"
            ) from exc
        from sella.eigensolvers import rayleigh_ritz as sella_rayleigh_ritz
        from sella.hessian_update import symmetrize_Y as sella_symmetrize_Y
    except Exception as exc:  # pragma: no cover - exercised on LS6 review env
        raise RuntimeError(
            "min_mode_finder=olsen_jd requires installed Sella 2.5.0 so "
            "SaddleMill can reuse sella.eigensolvers.rayleigh_ritz directly"
        ) from exc

    if sella_version != "2.5.0":
        raise RuntimeError(
            "SaddleMill Olsen/JD parity path is pinned to Sella 2.5.0; "
            f"found {sella_version!r}"
        )
    return sella_rayleigh_ritz, sella_symmetrize_Y, sella_version


def _reduced_preconditioner_matrix(preconditioner, *, basis: Array, full_shape) -> Array:
    """Snapshot the Sella JD0 preconditioner once at diagonalization start."""
    raw = preconditioner() if callable(preconditioner) else preconditioner
    matrix = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Olsen/JD preconditioner contains non-finite values")
    full_dim = int(np.prod(full_shape))
    reduced_dim = int(basis.shape[1])
    if matrix.shape == (full_dim, full_dim):
        return basis.T @ matrix @ basis
    if matrix.shape == (reduced_dim, reduced_dim):
        return matrix.copy()
    raise ValueError(
        "Olsen/JD preconditioner must be full Cartesian or active-reduced; "
        f"got {matrix.shape}, expected {(full_dim, full_dim)} or "
        f"{(reduced_dim, reduced_dim)}"
    )


def olsen_jd_lowest_mode(
    backend,
    initial_vector: object,
    preconditioner,
    *,
    coordinate_space,
    state_id: int,
    state_uid: str,
    geometry_id: str,
    maxiter: int | None = None,
    # Compatibility-only arguments retained so older direct callers fail
    # neither import nor call.  They are deliberately *not* convergence
    # controls for the Sella-parity Olsen/JD path.
    max_iterations: int | None = None,
    eigenvalue_tolerance: float | None = None,
    previous_eigenvalue: float | None = None,
    residual_stop: bool | None = None,
    residual_tolerance: float | None = None,
    breakdown_tolerance: float = 1.0e-12,
    root_selection: object = "lowest",
    homing_vector: object | None = None,
    source: str = "olsen_jd_hvp",
    family: str = "davidson",
    purpose: object = "algorithm",
    restart_dimension: int | None = None,
    singular_condition_threshold: float | None = None,
    pseudoinverse_rcond: float | None = None,
    preconditioner_identity: str = "physical_B",
) -> MinModeResult:
    """Run SaddleMill's physical HVP through Sella 2.5.0 ``rayleigh_ritz``.

    This is the W5 Olsen/JD parity path.  SaddleMill supplies the existing
    physical finite-difference (or explicitly selected typed) Hessian-vector
    operator and active-coordinate projection; Sella supplies the eigensolver
    mathematics *directly*.  Therefore JD0 expansion, Sella's symmetrized
    operator actions, preconditioner solve, negative-root seeking, dependent
    direction fallbacks, and ``maxiter`` semantics are exactly those of the
    installed Sella 2.5.0 implementation.

    Convergence is Sella's residual test with ``gamma=0.1``:

        ||r_i|| < 0.1 * |theta_i|

    for every currently sought negative Ritz root (at least the lowest root).
    Absolute eigenvalue change is retained only as a diagnostic field and can
    never certify convergence.  ``maxiter=None`` is passed through unchanged,
    so Sella applies its normal dimension-based limit.  ``max_iterations``
    remains a setting for the other SaddleMill minimum-mode finders and is not
    a stopping control here.
    """
    from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace, WorkPurpose

    if not isinstance(coordinate_space, ActiveCoordinateSpace):
        raise TypeError("coordinate_space must be ActiveCoordinateSpace")
    root_token = getattr(root_selection, "value", root_selection)
    if str(root_token).strip().lower() != "lowest":
        raise ValueError(
            "Sella-parity olsen_jd uses Sella's lowest/negative-root seeking; "
            "root_selection must be 'lowest'"
        )
    if homing_vector is not None:
        raise ValueError(
            "Sella-parity olsen_jd does not replace Sella root selection with "
            "SaddleMill homed-overlap selection"
        )
    if maxiter in ("", "none", "None"):
        maxiter = None
    if maxiter is not None:
        maxiter = int(maxiter)
        if maxiter < 1:
            raise ValueError("[ourMinMode] maxiter must be >= 1 or None")
    breakdown_tolerance = float(breakdown_tolerance)
    if breakdown_tolerance <= 0.0:
        raise ValueError("breakdown_tolerance must be > 0")

    gamma = 0.1
    purpose_token = WorkPurpose(str(purpose))
    shape = coordinate_space.active_dof_mask.shape
    initial_full = _normalised_flat(
        coordinate_space, initial_vector, breakdown_tolerance
    )

    operator = _SellaPhysicalHVPLinearOperator(
        backend,
        coordinate_space=coordinate_space,
        state_id=state_id,
        state_uid=state_uid,
        geometry_id=geometry_id,
        source=source,
        family=family,
        purpose=purpose_token,
        breakdown_tolerance=breakdown_tolerance,
    )
    basis = operator.basis
    initial_reduced = basis.T @ initial_full
    initial_reduced /= np.linalg.norm(initial_reduced)

    # Match native Sella: P is one preconditioner snapshot for the complete
    # Rayleigh-Ritz/JD0 diagonalization, while B is the Euclidean metric.
    P = _reduced_preconditioner_matrix(
        preconditioner, basis=basis, full_shape=shape
    )
    metric = np.eye(operator.shape[0], dtype=float)
    sella_rayleigh_ritz, sella_symmetrize_Y, sella_version = _sella_25_rayleigh_ritz()
    lams, V, AV = sella_rayleigh_ritz(
        operator,
        gamma,
        P,
        B=metric,
        v0=initial_reduced,
        method="jd0",
        maxiter=maxiter,
    )

    lams = np.asarray(lams, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float)
    AV = np.asarray(AV, dtype=float)
    if lams.size == 0 or V.ndim != 2 or AV.shape != V.shape:
        raise RuntimeError("Sella rayleigh_ritz returned an invalid eigensystem")
    if V.shape[1] != lams.size:
        raise RuntimeError("Sella rayleigh_ritz returned inconsistent Ritz dimensions")

    # Re-evaluate the exact stopping equation used inside Sella so diagnostics
    # can distinguish residual convergence from maxiter/full-dimension exit.
    Ytilde = np.asarray(sella_symmetrize_Y(V, AV, symm=2), dtype=float)
    nneg = max(1, int(np.sum(lams < 0.0)))
    sought = min(nneg, lams.size)
    residuals = Ytilde[:, :sought] - V[:, :sought] * lams[np.newaxis, :sought]
    residual_norms = np.linalg.norm(residuals, axis=0)
    residual_thresholds = gamma * np.abs(lams[:sought])
    residual_converged = bool(np.all(residual_norms < residual_thresholds))

    n = int(operator.shape[0])
    sella_internal_maxiter = 2 * n + 1 if maxiter is None else int(maxiter)
    effective_cap = min(n, sella_internal_maxiter)
    cap_reached = bool(V.shape[1] >= effective_cap)
    if cap_reached:
        stopping_reason = (
            "sella_full_dimension_limit" if maxiter is None or int(maxiter) >= n
            else "sella_maxiter_cap"
        )
    elif residual_converged:
        stopping_reason = "sella_residual_gamma"
    else:
        # In Sella 2.5.0 the only normal return before either criterion is the
        # exhausted dependent-direction fallback chain in rayleigh_ritz.
        stopping_reason = "sella_dependent_direction_fallback_exhausted"

    selected_value = float(lams[0])
    selected_reduced = np.asarray(V[:, 0], dtype=float)
    returned = basis @ selected_reduced
    returned /= np.linalg.norm(returned)
    if float(returned @ initial_full) < 0.0:
        returned *= -1.0

    # Raw physical residual and Sella's symmetrized solver residual are both
    # retained.  Only the latter controls convergence.
    raw_residual = np.asarray(AV[:, 0] - selected_value * V[:, 0], dtype=float)
    raw_residual_norm = float(np.linalg.norm(raw_residual))
    solver_residual_norm = float(residual_norms[0])
    eigenvalue_change = (
        None if previous_eigenvalue is None
        else abs(selected_value - float(previous_eigenvalue))
    )
    projected_raw = V.T @ AV
    projected_asymmetry = float(np.linalg.norm(projected_raw - projected_raw.T))

    all_actions = tuple(operator.results)
    work = summarize_hvp_actions(all_actions)
    returned_basis_full = tuple(
        np.asarray(operator.basis @ V[:, i], dtype=float).reshape(-1)
        for i in range(V.shape[1])
    )
    returned_actions_full = tuple(
        np.asarray(operator.basis @ AV[:, i], dtype=float).reshape(-1)
        for i in range(AV.shape[1])
    )
    raw_basis_full = tuple(
        coordinate_space.project(item.direction).reshape(-1) for item in all_actions
    )
    raw_actions_full = tuple(
        coordinate_space.project(item.action).reshape(-1) for item in all_actions
    )
    retained_span = _right_rotation_span_equivalence(
        raw_basis_full, raw_actions_full, returned_basis_full, returned_actions_full
    )
    gap = None if lams.size < 2 else float(lams[1] - lams[0])
    breakdown = bool(not cap_reached and not residual_converged)
    breakdown_reason = (
        "dependent_direction_fallback_exhausted" if breakdown else ""
    )
    ignored_legacy = {
        "max_iterations": max_iterations,
        "eigenvalue_tolerance": eigenvalue_tolerance,
        "residual_stop": residual_stop,
        "residual_tolerance": residual_tolerance,
        "restart_dimension": restart_dimension,
        "olsen_condition_threshold": singular_condition_threshold,
        "olsen_pseudoinverse_rcond": pseudoinverse_rcond,
    }
    audit = SolverAuditMetadata(
        retained_basis=_array_tuple(returned_basis_full),
        retained_actions=_array_tuple(returned_actions_full),
        evaluated_mode=returned.copy(),
        returned_mode=returned.copy(),
        selected_root_index=0,
        root_selection="lowest",
        stopping_rule="sella_2.5.0_rayleigh_ritz_jd0",
        stopping_equation="all sought roots: ||r_i|| < 0.1*|theta_i|",
        stopping_reason=stopping_reason,
        breakdown_reason=breakdown_reason,
        hvp_count=len(all_actions),
        rank=int(V.shape[1]),
        full_residual_norm=raw_residual_norm,
        solver_operator_residual_norm=solver_residual_norm,
        projected_operator_asymmetry_norm=projected_asymmetry,
        action_origins=work.origins,
        action_sources=work.sources,
        action_families=work.families,
        action_stencil_schemes=tuple(
            str(getattr(item, "stencil_scheme", "")) for item in all_actions
        ),
        action_displacement_scales=tuple(
            getattr(item, "displacement_scale", None) for item in all_actions
        ),
        action_geometry_ids=tuple(str(item.metadata.geometry_id) for item in all_actions),
        action_state_uids=tuple(str(item.metadata.state_uid) for item in all_actions),
        physical_action_count=work.physical_actions,
        model_action_count=work.model_actions,
        certification_eligible=work.certification_eligible,
        algorithm_pes_calls=work.algorithm_pes_calls,
        diagnostic_pes_calls=work.diagnostic_pes_calls,
        cache_hits=work.cache_hits,
        gap=gap,
        gap_eligible=bool(work.certification_eligible and gap is not None),
        degeneracy_unresolved=False,
        approximation_caveat=(
            "" if work.certification_eligible
            else "solver actions are not all physical-certification eligible"
        ),
        metadata={
            "direct_sella_reuse": True,
            "sella_version": sella_version,
            "sella_method": "jd0",
            "sella_gamma": gamma,
            "sella_maxiter": maxiter,
            "sella_effective_subspace_cap": effective_cap,
            "sella_negative_roots_sought": sought,
            "sella_residual_norms": tuple(float(x) for x in residual_norms),
            "sella_residual_thresholds": tuple(float(x) for x in residual_thresholds),
            "cap_reached": cap_reached,
            "residual_converged": residual_converged,
            "preconditioner_identity": str(preconditioner_identity),
            "preconditioner_snapshot": True,
            # Do not claim literal vector identity: Sella rotates V/AV in the
            # retained subspace.  The physical provenance block is the actual
            # raw typed-HVP query block and admission is gated on this span proof.
            "retained_t01_match_complete": False,
            "retained_hvp_indices": tuple(range(len(all_actions))),
            "retained_hvp_admission_basis": "raw_query_span_equivalent_to_sella_returned_v_av",
            "retained_span_equivalent_complete": bool(retained_span["complete"]),
            "retained_span_equivalence_tolerance": retained_span["tolerance"],
            "retained_span_raw_rank": retained_span["raw_rank"],
            "retained_span_returned_rank": retained_span["returned_rank"],
            "retained_span_raw_columns": retained_span["raw_columns"],
            "retained_span_returned_columns": retained_span["returned_columns"],
            "retained_span_direction_relative_residual": retained_span["direction_relative_residual"],
            "retained_span_action_relative_residual": retained_span["action_relative_residual"],
            "retained_span_right_orthogonality_residual": retained_span["right_orthogonality_residual"],
            "legacy_olsen_options_not_used_for_convergence": ignored_legacy,
            "eigenvalue_change_diagnostic_only": eigenvalue_change,
        },
    )

    return MinModeResult(
        eigenvalue=selected_value,
        eigenvector=returned.reshape(-1),
        iterations=len(all_actions),
        converged=residual_converged,
        breakdown=breakdown,
        eigenvalue_change=eigenvalue_change,
        residual_norm=solver_residual_norm,
        subspace_dimension=int(V.shape[1]),
        solver_used="olsen_jd_sella_2_5_0",
        residual_variant="sella_symmetrized_jd0_gamma_0.1",
        audit=audit,
    )


__all__ = [
    "MinModeResult",
    "SolverAuditMetadata",
    "OlsenCorrectionResult",
    "SolverWorkAccounting",
    "TypedHVPCallback",
    "PhysicalHessianBFGS",
    "davidson_lowest_mode",
    "lanczos_lowest_mode",
    "softsaddle_davidson_lowest_mode",
    "softsaddle_davidson_hybrid_lowest_mode",
    "softsaddle_lanczos_lowest_mode",
    "projected_olsen_jd_correction",
    "summarize_hvp_actions",
    "olsen_jd_lowest_mode",
]
