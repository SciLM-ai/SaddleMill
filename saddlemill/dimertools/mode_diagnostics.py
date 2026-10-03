"""Standard minimum-mode residual, gap, decision, and reference diagnostics.

mode-diagnostics owns this module as an observational layer over the sealed typed-HVP/physical-Hessian/projected-Olsen/JD
interfaces.  It does not perform a minimum-mode solve, mutate the mode scheduler,
or acquire a force unless an explicit caller-supplied paid-reference callback is
run through :func:`run_paid_reference_measurement`.

The physical force convention is ``F = -g``.  A typed :class:`HVPResult` is
already finite-difference scaled exactly once.  Dimer torque supplied to
:func:`diagnostic_from_dimer_torque` is likewise assumed to be the existing
scaled ``perp(F+ - F-) / (2 dR)`` quantity and is therefore *not* divided by
``dR`` again.  That torque is approximately ``-r`` for the physical eigenvector
residual ``r = Hv - (v.T Hv) v``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from time import perf_counter_ns
from typing import Callable, Mapping

import numpy as np

from saddlemill.dimertools.foundation_types import (
    ActiveCoordinateSpace,
    CommonResultMetadata,
    ForceAccounting,
    OperatorOrigin,
    RootSelection,
    WorkPurpose,
    freeze_array,
    freeze_mapping,
    json_safe,
)
from saddlemill.dimertools.hvp_interfaces import (
    DEFAULT_GAP_TOLERANCE,
    DEFAULT_RESIDUAL_EPSILON,
    HVPIdentityError,
    HVPResult,
    RitzResult,
    physical_eigen_residual,
)

Array = np.ndarray

MODE_DIAGNOSTIC_SCHEMA = "saddlemill_mode_diagnostic_v1"
MODE_GAP_SCHEMA = "saddlemill_mode_gap_estimate_v1"
MODE_DECISION_SCHEMA = "saddlemill_mode_refresh_decision_v1"
REFERENCE_MODE_SCHEMA = "saddlemill_reference_mode_measurement_v1"
NONINTERFERENCE_SCHEMA = "saddlemill_diagnostic_noninterference_v1"

MODE_REFRESH_SELECTORS = ("legacy", "residual", "angular_error", "hybrid")
GAP_SOURCE_PHYSICAL_RITZ = "physical_ritz"
GAP_SOURCE_MODEL_SPECTRUM = "physical_hessian_model_spectrum"


def _token(value: object) -> str:
    return str(value).strip().lower()


def _finite_or_none(value: object | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _mode_id(mode: Array, coordinate_space: ActiveCoordinateSpace, prefix: str) -> str:
    projected = coordinate_space.normalized(mode)
    canonical = np.ascontiguousarray(projected.astype("<f8", copy=False))
    digest = hashlib.sha256()
    digest.update(f"saddlemill-{prefix}-v1\0".encode("utf-8"))
    digest.update(coordinate_space.identity.encode("utf-8"))
    digest.update(canonical.tobytes(order="C"))
    return f"{prefix}:" + digest.hexdigest()


def sign_invariant_overlap(
    first: object,
    second: object,
    coordinate_space: ActiveCoordinateSpace,
) -> float:
    a = coordinate_space.normalized(first).reshape(-1)
    b = coordinate_space.normalized(second).reshape(-1)
    return abs(float(np.dot(a, b)))


@dataclass(frozen=True)
class ModeGapEstimate:
    """Root-aware low-spectrum denominator for conditional-angle diagnostics."""

    available: bool
    value: float | None
    source: str
    physical: bool
    root_selection: str
    selected_root_index: int | None
    selected_root_separation: float | None
    gap_eligible: bool
    degeneracy_unresolved: bool
    solver_tolerance: float
    degeneracy_threshold: float
    spectrum_values: tuple[float, ...] = ()
    spectrum_space: str = ""
    subspace_rank: int | None = None
    approximation_uncertainty: float | None = None
    uncertainty_comparable_to_gap: bool = False
    subspace_overlap: float | None = None
    model_age: int | None = None
    model_condition: float | None = None
    model_condition_state: str = ""
    matrix_compute_ns: int = 0
    approximation_caveat: str = ""
    unavailable_reason: str = ""
    schema: str = MODE_GAP_SCHEMA

    def __post_init__(self) -> None:
        source = _token(self.source)
        root = _token(self.root_selection)
        if not source:
            raise ValueError("gap source cannot be empty")
        if root not in {item.value for item in RootSelection}:
            raise ValueError(f"unsupported root selection {self.root_selection!r}")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "root_selection", root)
        object.__setattr__(self, "physical", bool(self.physical))
        object.__setattr__(self, "gap_eligible", bool(self.gap_eligible))
        object.__setattr__(self, "degeneracy_unresolved", bool(self.degeneracy_unresolved))
        object.__setattr__(self, "matrix_compute_ns", max(0, int(self.matrix_compute_ns)))
        object.__setattr__(self, "spectrum_values", tuple(float(v) for v in self.spectrum_values))
        object.__setattr__(self, "spectrum_space", _token(self.spectrum_space))
        if self.subspace_rank is not None:
            object.__setattr__(self, "subspace_rank", int(self.subspace_rank))
        if self.selected_root_index is not None:
            object.__setattr__(self, "selected_root_index", int(self.selected_root_index))
        if self.model_age is not None:
            age = int(self.model_age)
            if age < 0:
                raise ValueError("model_age must be >= 0")
            object.__setattr__(self, "model_age", age)
        if self.available:
            value = _finite_or_none(self.value)
            if value is None or value <= 0.0:
                raise ValueError("available gap must be finite and > 0")
            object.__setattr__(self, "value", abs(value))
        else:
            object.__setattr__(self, "value", None)
            if not str(self.unavailable_reason).strip():
                raise ValueError("unavailable gap requires a reason")
        object.__setattr__(self, "unavailable_reason", _token(self.unavailable_reason))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "available": self.available,
            "value": self.value,
            "source": self.source,
            "physical": self.physical,
            "root_selection": self.root_selection,
            "selected_root_index": self.selected_root_index,
            "selected_root_separation": self.selected_root_separation,
            "gap_eligible": self.gap_eligible,
            "degeneracy_unresolved": self.degeneracy_unresolved,
            "solver_tolerance": self.solver_tolerance,
            "degeneracy_threshold": self.degeneracy_threshold,
            "spectrum_values": list(self.spectrum_values),
            "spectrum_space": self.spectrum_space,
            "subspace_rank": self.subspace_rank,
            "approximation_uncertainty": self.approximation_uncertainty,
            "uncertainty_comparable_to_gap": self.uncertainty_comparable_to_gap,
            "subspace_overlap": self.subspace_overlap,
            "model_age": self.model_age,
            "model_condition": self.model_condition,
            "model_condition_state": self.model_condition_state,
            "matrix_compute_ns": self.matrix_compute_ns,
            "approximation_caveat": self.approximation_caveat,
            "unavailable_reason": self.unavailable_reason,
        }


def gap_from_ritz(
    ritz: RitzResult,
    *,
    evaluated_mode: object,
    coordinate_space: ActiveCoordinateSpace,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    direction_tolerance: float = 1.0e-8,
    approximation_uncertainty: float | None = None,
    subspace_overlap: float | None = None,
    expected_geometry_id: str | None = None,
    expected_state_id: int | None = None,
    expected_state_uid: str | None = None,
) -> ModeGapEstimate:
    """Convert a same-center Ritz result into a root-qualified gap estimate.

    Only a physical, selected-vector-matched, rank-revealed Ritz gap is treated
    as the preferred mode-diagnostics numerator denominator.  Model Ritz data remain useful
    solver metadata but do not displace the physical-Hessian model-spectrum fallback.
    """

    root = str(ritz.root_selection)
    tol = max(float(gap_tolerance), float(ritz.solver_tolerance))
    base = dict(
        source=GAP_SOURCE_PHYSICAL_RITZ,
        physical=bool(ritz.physical),
        root_selection=root,
        selected_root_index=ritz.selected_index,
        selected_root_separation=ritz.selected_root_separation,
        solver_tolerance=float(ritz.solver_tolerance),
        degeneracy_threshold=tol,
        spectrum_values=tuple(float(v) for v in ritz.ritz_values),
        spectrum_space="same_center_active_free_ritz_subspace",
        subspace_rank=int(ritz.rank),
        approximation_uncertainty=_finite_or_none(approximation_uncertainty),
        subspace_overlap=_finite_or_none(subspace_overlap),
        model_age=ritz.model_age,
        approximation_caveat=str(ritz.approximation_caveat),
    )
    if (
        (expected_geometry_id is not None and ritz.geometry_id != str(expected_geometry_id))
        or (expected_state_id is not None and ritz.state_id != int(expected_state_id))
        or (expected_state_uid is not None and ritz.state_uid != str(expected_state_uid))
    ):
        return ModeGapEstimate(
            available=False,
            value=None,
            gap_eligible=False,
            degeneracy_unresolved=True,
            unavailable_reason="ritz_same_center_identity_mismatch",
            **base,
        )
    if not ritz.available or ritz.selected_vector is None:
        return ModeGapEstimate(
            available=False,
            value=None,
            gap_eligible=False,
            degeneracy_unresolved=True,
            unavailable_reason=ritz.unavailable_reason or "ritz_unavailable",
            **base,
        )
    overlap = sign_invariant_overlap(evaluated_mode, ritz.selected_vector, coordinate_space)
    if overlap < 1.0 - float(direction_tolerance):
        return ModeGapEstimate(
            available=False,
            value=None,
            gap_eligible=False,
            degeneracy_unresolved=True,
            unavailable_reason="evaluated_mode_not_selected_ritz_root",
            **base,
        )
    if not ritz.physical:
        return ModeGapEstimate(
            available=False,
            value=None,
            gap_eligible=False,
            degeneracy_unresolved=bool(ritz.degeneracy_unresolved),
            unavailable_reason="ritz_not_physical",
            **base,
        )
    if not ritz.gap_eligible or ritz.gap is None:
        reason = "ritz_rank_one_no_gap" if int(ritz.rank) < 2 else "ritz_gap_unresolved"
        return ModeGapEstimate(
            available=False,
            value=None,
            gap_eligible=False,
            degeneracy_unresolved=bool(ritz.degeneracy_unresolved),
            unavailable_reason=reason,
            **base,
        )
    gap = abs(float(ritz.gap))
    uncertainty = _finite_or_none(approximation_uncertainty)
    uncertainty_limited = bool(uncertainty is not None and uncertainty >= gap)
    eligible = bool(gap > tol and not uncertainty_limited)
    if eligible:
        unavailable_reason = ""
    elif gap <= tol:
        unavailable_reason = "ritz_gap_near_degenerate"
    else:
        unavailable_reason = "ritz_gap_uncertainty_unresolved"
    return ModeGapEstimate(
        available=eligible,
        value=(gap if eligible else None),
        gap_eligible=eligible,
        degeneracy_unresolved=bool(gap <= tol),
        uncertainty_comparable_to_gap=uncertainty_limited,
        unavailable_reason=unavailable_reason,
        **base,
    )


def gap_from_model_spectrum(
    spectrum: object,
    *,
    root_selection: RootSelection | str = RootSelection.LOWEST,
    solver_tolerance: float = 1.0e-4,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    approximation_uncertainty: float | None = None,
    subspace_overlap: float | None = None,
) -> ModeGapEstimate:
    """Qualify a physical-Hessian matrix-only physical-model low-spectrum gap."""

    root = RootSelection(str(root_selection))
    gap = _finite_or_none(getattr(spectrum, "selected_low_root_eigengap", None))
    separation = gap
    tol = max(abs(float(solver_tolerance)), abs(float(gap_tolerance)))
    uncertainty = _finite_or_none(approximation_uncertainty)
    uncertainty_limited = bool(gap is not None and uncertainty is not None and uncertainty >= gap)
    unresolved = bool(gap is None or gap <= tol)
    eligible = bool(gap is not None and gap > tol and not uncertainty_limited)
    reason = ""
    if gap is None:
        reason = "model_spectrum_has_no_gap"
    elif gap <= tol:
        reason = "model_gap_near_degenerate"
    elif uncertainty_limited:
        reason = "model_gap_uncertainty_unresolved"
    return ModeGapEstimate(
        available=eligible,
        value=(abs(gap) if eligible and gap is not None else None),
        source=GAP_SOURCE_MODEL_SPECTRUM,
        physical=False,
        root_selection=root.value,
        selected_root_index=int(getattr(spectrum, "selected_root_index", 0)),
        selected_root_separation=separation,
        gap_eligible=eligible,
        degeneracy_unresolved=unresolved,
        solver_tolerance=abs(float(solver_tolerance)),
        degeneracy_threshold=tol,
        spectrum_values=tuple(float(v) for v in getattr(spectrum, "eigenvalues", ())),
        spectrum_space="t03_active_free_physical_hessian_model",
        subspace_rank=len(tuple(getattr(spectrum, "eigenvalues", ()))),
        approximation_uncertainty=uncertainty,
        uncertainty_comparable_to_gap=uncertainty_limited,
        subspace_overlap=_finite_or_none(subspace_overlap),
        model_age=int(getattr(spectrum, "model_age", 0)),
        model_condition=_finite_or_none(getattr(spectrum, "absolute_eigenvalue_condition", None)),
        model_condition_state=str(getattr(spectrum, "condition_state", "")),
        matrix_compute_ns=max(0, int(getattr(spectrum, "timing_ns", 0))),
        approximation_caveat=(
            "physical-Hessian approximate physical-Hessian matrix-only spectrum; zero PES cost; "
            "not physical certification"
        ),
        unavailable_reason=reason,
    )


def compute_model_gap(
    model: object,
    *,
    selected_root_index: int = 0,
    root_selection: RootSelection | str = RootSelection.LOWEST,
    solver_tolerance: float = 1.0e-4,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    approximation_uncertainty: float | None = None,
    subspace_overlap: float | None = None,
    accounting: ForceAccounting | None = None,
    purpose: WorkPurpose | str = WorkPurpose.DIAGNOSTIC,
) -> ModeGapEstimate:
    """Evaluate physical-Hessian's low spectrum and account its matrix-only compute cost."""

    spectrum = model.low_spectrum(selected_root_index=int(selected_root_index))
    if accounting is not None:
        accounting.add_model_matrix_ns(
            int(getattr(spectrum, "timing_ns", 0)), purpose=WorkPurpose(str(purpose))
        )
    return gap_from_model_spectrum(
        spectrum,
        root_selection=root_selection,
        solver_tolerance=solver_tolerance,
        gap_tolerance=gap_tolerance,
        approximation_uncertainty=approximation_uncertainty,
        subspace_overlap=subspace_overlap,
    )


def select_gap_estimate(
    *,
    ritz: RitzResult | None,
    evaluated_mode: object,
    coordinate_space: ActiveCoordinateSpace,
    model_gap: ModeGapEstimate | None = None,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    direction_tolerance: float = 1.0e-8,
    approximation_uncertainty: float | None = None,
    subspace_overlap: float | None = None,
    expected_geometry_id: str | None = None,
    expected_state_id: int | None = None,
    expected_state_uid: str | None = None,
) -> ModeGapEstimate:
    """Prefer a trustworthy same-center physical Ritz gap, else model fallback."""

    ritz_gap: ModeGapEstimate | None = None
    if ritz is not None:
        ritz_gap = gap_from_ritz(
            ritz,
            evaluated_mode=evaluated_mode,
            coordinate_space=coordinate_space,
            gap_tolerance=gap_tolerance,
            direction_tolerance=direction_tolerance,
            approximation_uncertainty=approximation_uncertainty,
            subspace_overlap=subspace_overlap,
            expected_geometry_id=expected_geometry_id,
            expected_state_id=expected_state_id,
            expected_state_uid=expected_state_uid,
        )
        if ritz_gap.available and ritz_gap.gap_eligible:
            return ritz_gap
    if model_gap is not None:
        return model_gap
    if ritz_gap is not None:
        return ritz_gap
    return ModeGapEstimate(
        available=False,
        value=None,
        source="none",
        physical=False,
        root_selection=RootSelection.LOWEST.value,
        selected_root_index=None,
        selected_root_separation=None,
        gap_eligible=False,
        degeneracy_unresolved=True,
        solver_tolerance=0.0,
        degeneracy_threshold=float(gap_tolerance),
        unavailable_reason="no_ritz_or_model_gap",
    )


@dataclass(frozen=True)
class StandardModeDiagnosticRecord:
    """One standardized current-geometry mode-quality observation."""

    available: bool
    evaluated_mode: Array | None
    returned_mode: Array | None
    evaluated_mode_id: str
    returned_mode_id: str
    geometry_id: str
    state_id: int
    state_uid: str
    coordinate_space_id: str
    projection_convention: str
    null_mode_policy: str
    rho: float | None
    residual: Array | None
    residual_norm: float | None
    relative_torque: float | None
    conditional_angle_radians: float | None
    conditional_angle_degrees: float | None
    model_mode_overlap: float | None
    model_mode_angle_degrees: float | None
    model_mode_source: str
    olsen_correction_norm: float | None
    olsen_correction_source: str
    gap: ModeGapEstimate
    numerator_origin: str
    numerator_physical: bool
    action_source: str
    action_family: str
    action_operator_origin: str
    action_purpose: str
    action_model_age: int | None
    certification_eligible: bool
    fd_distance: float | None = None
    fd_stencil: str = ""
    fd_status: str = ""
    endpoint_observation_ids: tuple[str, ...] = ()
    fd_asymmetry_norm: float | None = None
    full_raw_residual_norm: float | None = None
    solver_projected_residual_norm: float | None = None
    solver_residual_origin: str = ""
    selected_root_index: int | None = None
    root_selection: str = "lowest"
    root_valid: bool = True
    root_status: str = "selected_root_valid"
    negative_mode_lost: bool = False
    epsilon: float = DEFAULT_RESIDUAL_EPSILON
    unavailable_reason: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema: str = MODE_DIAGNOSTIC_SCHEMA

    def __post_init__(self) -> None:
        if self.evaluated_mode is not None:
            object.__setattr__(self, "evaluated_mode", freeze_array(self.evaluated_mode))
        if self.returned_mode is not None:
            object.__setattr__(self, "returned_mode", freeze_array(self.returned_mode))
        if self.residual is not None:
            object.__setattr__(self, "residual", freeze_array(self.residual))
        object.__setattr__(self, "endpoint_observation_ids", tuple(str(v) for v in self.endpoint_observation_ids))
        object.__setattr__(self, "root_selection", _token(self.root_selection))
        object.__setattr__(self, "root_status", _token(self.root_status))
        object.__setattr__(self, "model_mode_source", _token(self.model_mode_source))
        object.__setattr__(self, "olsen_correction_source", _token(self.olsen_correction_source))
        object.__setattr__(self, "unavailable_reason", _token(self.unavailable_reason))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if not self.available and not self.unavailable_reason:
            raise ValueError("unavailable diagnostic record requires a reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "available": self.available,
            "evaluated_mode": None if self.evaluated_mode is None else self.evaluated_mode.tolist(),
            "returned_mode": None if self.returned_mode is None else self.returned_mode.tolist(),
            "evaluated_mode_id": self.evaluated_mode_id,
            "returned_mode_id": self.returned_mode_id,
            "geometry_id": self.geometry_id,
            "state_id": self.state_id,
            "state_uid": self.state_uid,
            "coordinate_space_id": self.coordinate_space_id,
            "projection_convention": self.projection_convention,
            "null_mode_policy": self.null_mode_policy,
            "rho": self.rho,
            "residual": None if self.residual is None else self.residual.tolist(),
            "residual_norm": self.residual_norm,
            "relative_torque": self.relative_torque,
            "conditional_angle_radians": self.conditional_angle_radians,
            "conditional_angle_degrees": self.conditional_angle_degrees,
            "model_mode_overlap": self.model_mode_overlap,
            "model_mode_angle_degrees": self.model_mode_angle_degrees,
            "model_mode_source": self.model_mode_source,
            "olsen_correction_norm": self.olsen_correction_norm,
            "olsen_correction_source": self.olsen_correction_source,
            "gap": self.gap.to_dict(),
            "numerator_origin": self.numerator_origin,
            "numerator_physical": self.numerator_physical,
            "action_source": self.action_source,
            "action_family": self.action_family,
            "action_operator_origin": self.action_operator_origin,
            "action_purpose": self.action_purpose,
            "action_model_age": self.action_model_age,
            "certification_eligible": self.certification_eligible,
            "fd_distance": self.fd_distance,
            "fd_stencil": self.fd_stencil,
            "fd_status": self.fd_status,
            "endpoint_observation_ids": list(self.endpoint_observation_ids),
            "fd_asymmetry_norm": self.fd_asymmetry_norm,
            "full_raw_residual_norm": self.full_raw_residual_norm,
            "solver_projected_residual_norm": self.solver_projected_residual_norm,
            "solver_residual_origin": self.solver_residual_origin,
            "selected_root_index": self.selected_root_index,
            "root_selection": self.root_selection,
            "root_valid": self.root_valid,
            "root_status": self.root_status,
            "negative_mode_lost": self.negative_mode_lost,
            "epsilon": self.epsilon,
            "unavailable_reason": self.unavailable_reason,
            "metadata": json_safe(self.metadata),
        }


def standardize_hvp_diagnostic(
    hvp: HVPResult,
    *,
    coordinate_space: ActiveCoordinateSpace,
    returned_mode: object | None = None,
    ritz: RitzResult | None = None,
    model_gap: ModeGapEstimate | None = None,
    epsilon: float = DEFAULT_RESIDUAL_EPSILON,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    direction_tolerance: float = 1.0e-8,
    root_valid: bool = True,
    root_status: str = "selected_root_valid",
    negative_mode_lost: bool = False,
    solver_projected_residual_norm: float | None = None,
    solver_residual_origin: str = "",
    fd_asymmetry_norm: float | None = None,
    approximation_uncertainty: float | None = None,
    subspace_overlap: float | None = None,
    model_mode: object | None = None,
    model_mode_source: str = "",
    olsen_correction: object | None = None,
    olsen_correction_source: str = "",
    metadata: Mapping[str, object] | None = None,
) -> StandardModeDiagnosticRecord:
    """Standardize one already-computed same-center Hessian action.

    This function performs only array/matrix arithmetic.  It does not call an
    HVP backend or calculator and therefore cannot spend a PES call.
    """

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and > 0")
    if hvp.metadata.coordinate_space_id != coordinate_space.identity:
        raise HVPIdentityError("diagnostic coordinate space differs from HVP coordinate space")
    if not hvp.available or hvp.action is None:
        gap = select_gap_estimate(
            ritz=ritz,
            evaluated_mode=hvp.direction,
            coordinate_space=coordinate_space,
            model_gap=model_gap,
            gap_tolerance=gap_tolerance,
            direction_tolerance=direction_tolerance,
            approximation_uncertainty=approximation_uncertainty,
            subspace_overlap=subspace_overlap,
            expected_geometry_id=hvp.metadata.geometry_id,
            expected_state_id=hvp.metadata.state_id,
            expected_state_uid=hvp.metadata.state_uid,
        )
        return StandardModeDiagnosticRecord(
            available=False,
            evaluated_mode=None,
            returned_mode=None,
            evaluated_mode_id="",
            returned_mode_id="",
            geometry_id=hvp.metadata.geometry_id,
            state_id=hvp.metadata.state_id,
            state_uid=hvp.metadata.state_uid,
            coordinate_space_id=coordinate_space.identity,
            projection_convention=coordinate_space.convention,
            null_mode_policy=coordinate_space.null_mode_policy,
            rho=None,
            residual=None,
            residual_norm=None,
            relative_torque=None,
            conditional_angle_radians=None,
            conditional_angle_degrees=None,
            model_mode_overlap=None,
            model_mode_angle_degrees=None,
            model_mode_source=_token(model_mode_source),
            olsen_correction_norm=None,
            olsen_correction_source=_token(olsen_correction_source),
            gap=gap,
            numerator_origin=str(hvp.metadata.operator_origin),
            numerator_physical=bool(hvp.metadata.physical),
            action_source=hvp.metadata.source,
            action_family=hvp.metadata.family,
            action_operator_origin=str(hvp.metadata.operator_origin),
            action_purpose=str(hvp.metadata.purpose),
            action_model_age=hvp.metadata.model_age,
            certification_eligible=False,
            unavailable_reason=hvp.unavailable_reason or "hvp_unavailable",
            metadata=metadata or {},
        )

    projected_direction = coordinate_space.project(hvp.direction)
    direction_norm = float(np.linalg.norm(projected_direction))
    if not np.isfinite(direction_norm) or direction_norm <= 0.0:
        raise ValueError("HVP diagnostic direction has zero/nonfinite active norm")
    evaluated = projected_direction / direction_norm
    returned = evaluated if returned_mode is None else coordinate_space.normalized(returned_mode)
    # HVPRequest explicitly does not require a unit direction.  The backend
    # action is linear in that projected request direction, so when mode-diagnostics
    # reports the canonical normalized-mode residual it must scale the already
    # computed action by the same norm.  This is *not* a second finite-
    # difference scaling; ``scaled_once`` and displacement_scale are left
    # untouched.
    normalized_hvp = replace(
        hvp,
        direction=evaluated,
        action=coordinate_space.project(hvp.action) / direction_norm,
    )
    base = physical_eigen_residual(
        normalized_hvp, coordinate_space=coordinate_space, epsilon=epsilon
    )
    gap = select_gap_estimate(
        ritz=ritz,
        evaluated_mode=evaluated,
        coordinate_space=coordinate_space,
        model_gap=model_gap,
        gap_tolerance=gap_tolerance,
        direction_tolerance=direction_tolerance,
        approximation_uncertainty=approximation_uncertainty,
        subspace_overlap=subspace_overlap,
        expected_geometry_id=hvp.metadata.geometry_id,
        expected_state_id=hvp.metadata.state_id,
        expected_state_uid=hvp.metadata.state_uid,
    )
    angle = None
    if gap.available and gap.gap_eligible and gap.value is not None and base.residual_norm is not None:
        angle = float(base.residual_norm) / float(gap.value)
    model_overlap = None
    model_angle_degrees = None
    if model_mode is not None:
        model_overlap = min(1.0, max(0.0, sign_invariant_overlap(evaluated, model_mode, coordinate_space)))
        model_angle_degrees = math.degrees(math.acos(model_overlap))
    olsen_norm = None
    if olsen_correction is not None:
        correction = coordinate_space.project(olsen_correction)
        if not np.all(np.isfinite(correction)):
            raise ValueError("Olsen correction contains non-finite values")
        olsen_norm = float(np.linalg.norm(correction))
    provenance = dict(metadata or {})
    provenance.update({
        "hvp_scaled_once": bool(hvp.scaled_once),
        "hvp_request_direction_active_norm": direction_norm,
        "normalized_action_from_linear_hvp": bool(abs(direction_norm - 1.0) > 1.0e-14),
        "gap_source": gap.source,
        "gap_physical": gap.physical,
    })
    fd_status = str(hvp.metadata.provenance.get("fd_status", ""))
    return StandardModeDiagnosticRecord(
        available=True,
        evaluated_mode=evaluated,
        returned_mode=returned,
        evaluated_mode_id=_mode_id(evaluated, coordinate_space, "evaluated-mode"),
        returned_mode_id=_mode_id(returned, coordinate_space, "returned-mode"),
        geometry_id=hvp.metadata.geometry_id,
        state_id=hvp.metadata.state_id,
        state_uid=hvp.metadata.state_uid,
        coordinate_space_id=coordinate_space.identity,
        projection_convention=coordinate_space.convention,
        null_mode_policy=coordinate_space.null_mode_policy,
        rho=base.rho,
        residual=base.residual,
        residual_norm=base.residual_norm,
        relative_torque=base.relative_torque,
        conditional_angle_radians=angle,
        conditional_angle_degrees=(None if angle is None else math.degrees(angle)),
        model_mode_overlap=model_overlap,
        model_mode_angle_degrees=model_angle_degrees,
        model_mode_source=_token(model_mode_source),
        olsen_correction_norm=olsen_norm,
        olsen_correction_source=_token(olsen_correction_source),
        gap=gap,
        numerator_origin=base.numerator_origin,
        numerator_physical=bool(hvp.metadata.physical),
        action_source=hvp.metadata.source,
        action_family=hvp.metadata.family,
        action_operator_origin=str(hvp.metadata.operator_origin),
        action_purpose=str(hvp.metadata.purpose),
        action_model_age=hvp.metadata.model_age,
        certification_eligible=bool(base.certification_eligible),
        fd_distance=hvp.displacement_scale,
        fd_stencil=hvp.stencil_scheme,
        fd_status=fd_status,
        endpoint_observation_ids=hvp.endpoint_observation_ids,
        fd_asymmetry_norm=_finite_or_none(fd_asymmetry_norm),
        full_raw_residual_norm=base.residual_norm,
        solver_projected_residual_norm=_finite_or_none(solver_projected_residual_norm),
        solver_residual_origin=_token(solver_residual_origin),
        selected_root_index=(gap.selected_root_index if gap.selected_root_index is not None else (ritz.selected_index if ritz else None)),
        root_selection=(gap.root_selection if gap.root_selection else (str(ritz.root_selection) if ritz else RootSelection.LOWEST.value)),
        root_valid=bool(root_valid),
        root_status=root_status,
        negative_mode_lost=bool(negative_mode_lost),
        epsilon=epsilon,
        metadata=provenance,
    )


def diagnostic_from_dimer_torque(
    *,
    evaluated_mode: object,
    torque: object,
    curvature: float,
    coordinate_space: ActiveCoordinateSpace,
    geometry_id: str,
    state_id: int,
    state_uid: str,
    fd_distance: float,
    stencil: str,
    fd_status: str,
    returned_mode: object | None = None,
    purpose: WorkPurpose | str = WorkPurpose.ALGORITHM,
    source: str = "dimer_torque",
    family: str = "dimer",
    endpoint_observation_ids: tuple[str, ...] = (),
    ritz: RitzResult | None = None,
    model_gap: ModeGapEstimate | None = None,
    fd_asymmetry_norm: float | None = None,
    epsilon: float = DEFAULT_RESIDUAL_EPSILON,
) -> StandardModeDiagnosticRecord:
    """Standardize an inherited already-scaled Dimer rotational torque.

    ``torque`` is the existing ``perp(F+ - F-) / (2*dR)`` quantity and is
    approximately ``-r``.  This function reconstructs an equivalent ``Hv`` as
    ``curvature*v - torque`` and intentionally performs no additional division
    by ``fd_distance``.
    """

    distance = float(fd_distance)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("fd_distance must be finite and > 0")
    rho = float(curvature)
    if not np.isfinite(rho):
        raise ValueError("curvature must be finite")
    v = coordinate_space.normalized(evaluated_mode)
    projected_torque = coordinate_space.project(torque)
    # Dimer torque ~= -r.  The action reconstructed below gives exactly the
    # desired projected residual without re-scaling the already scaled torque.
    action = coordinate_space.project(rho * v - projected_torque)
    scheme_token = _token(stencil)
    status_token = _token(fd_status)
    md = CommonResultMetadata(
        geometry_id=str(geometry_id),
        state_id=int(state_id),
        state_uid=str(state_uid),
        coordinate_space_id=coordinate_space.identity,
        purpose=purpose,
        source=source,
        family=family,
        operator_origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
        physical=True,
        model_derived=False,
        pes_call_delta=0,
        cache_hit=True,
        timing_ns=0,
        units="eV/Angstrom^2 * Cartesian-vector",
        provenance={
            "fd_status": status_token,
            "dimer_torque_relation": "torque_approximately_minus_physical_residual",
            "input_torque_already_scaled": True,
            "no_second_fd_division": True,
        },
    )
    hvp = HVPResult(
        available=True,
        action=action,
        direction=v,
        metadata=md,
        stencil_scheme=scheme_token,
        displacement_scale=distance,
        endpoint_observation_ids=endpoint_observation_ids,
        scaled_once=True,
    )
    return standardize_hvp_diagnostic(
        hvp,
        coordinate_space=coordinate_space,
        returned_mode=returned_mode,
        ritz=ritz,
        model_gap=model_gap,
        epsilon=epsilon,
        fd_asymmetry_norm=fd_asymmetry_norm,
        metadata={"dimer_torque_input_norm": float(np.linalg.norm(projected_torque))},
    )


def standardize_solver_audit(
    result: object,
    *,
    coordinate_space: ActiveCoordinateSpace,
    model_gap: ModeGapEstimate | None = None,
    epsilon: float = DEFAULT_RESIDUAL_EPSILON,
    root_valid: bool = True,
    root_status: str = "selected_root_valid",
    negative_mode_lost: bool = False,
    model_mode: object | None = None,
    model_mode_source: str = "",
    olsen_correction: object | None = None,
    olsen_correction_source: str = "",
) -> StandardModeDiagnosticRecord:
    """Standardize projected-Olsen/JD ``MinModeResult.audit`` without inventing physical truth.

    projected-Olsen/JD's audit may contain both the full raw-action Ritz residual and the
    solver-projected/operator residual.  This function publishes both scalars
    distinctly.  It does not relabel an untyped/model solve as physical.
    """

    audit = getattr(result, "audit", None)
    if audit is None:
        raise ValueError("solver result has no projected-Olsen/JD audit metadata")
    evaluated_raw = getattr(audit, "evaluated_mode", None)
    returned_raw = getattr(audit, "returned_mode", None)
    if evaluated_raw is None or returned_raw is None:
        raise ValueError("solver audit lacks evaluated/returned mode")
    evaluated = coordinate_space.normalized(np.asarray(evaluated_raw, dtype=float).reshape(coordinate_space.active_dof_mask.shape))
    returned = coordinate_space.normalized(np.asarray(returned_raw, dtype=float).reshape(coordinate_space.active_dof_mask.shape))
    raw_norm = _finite_or_none(getattr(audit, "full_residual_norm", None))
    solver_norm = _finite_or_none(getattr(audit, "solver_operator_residual_norm", None))
    rho = _finite_or_none(getattr(result, "eigenvalue", None))
    relative = None if raw_norm is None or rho is None else raw_norm / max(abs(rho), float(epsilon))
    gap: ModeGapEstimate
    audit_gap = _finite_or_none(getattr(audit, "gap", None))
    audit_gap_eligible = bool(getattr(audit, "gap_eligible", False))
    audit_physical = bool(getattr(audit, "certification_eligible", False))
    root_selection = _token(getattr(audit, "root_selection", "lowest"))
    if audit_physical and audit_gap_eligible and audit_gap is not None:
        gap = ModeGapEstimate(
            available=True,
            value=abs(audit_gap),
            source=GAP_SOURCE_PHYSICAL_RITZ,
            physical=True,
            root_selection=root_selection,
            selected_root_index=getattr(audit, "selected_root_index", None),
            selected_root_separation=abs(audit_gap),
            gap_eligible=True,
            degeneracy_unresolved=bool(getattr(audit, "degeneracy_unresolved", False)),
            solver_tolerance=0.0,
            degeneracy_threshold=DEFAULT_GAP_TOLERANCE,
            spectrum_values=tuple(float(v) for v in getattr(audit, "ritz_values", ())),
            spectrum_space="t09_same_center_active_free_ritz_subspace",
            subspace_rank=int(getattr(audit, "subspace_rank", 0) or 0),
            approximation_caveat=str(getattr(audit, "approximation_caveat", "")),
        )
    elif model_gap is not None:
        gap = model_gap
    else:
        gap = ModeGapEstimate(
            available=False,
            value=None,
            source=GAP_SOURCE_PHYSICAL_RITZ,
            physical=audit_physical,
            root_selection=root_selection,
            selected_root_index=getattr(audit, "selected_root_index", None),
            selected_root_separation=audit_gap,
            gap_eligible=False,
            degeneracy_unresolved=bool(getattr(audit, "degeneracy_unresolved", True)),
            solver_tolerance=0.0,
            degeneracy_threshold=DEFAULT_GAP_TOLERANCE,
            approximation_caveat=str(getattr(audit, "approximation_caveat", "")),
            unavailable_reason="solver_audit_has_no_eligible_gap",
        )
    angle = None
    if raw_norm is not None and gap.available and gap.gap_eligible and gap.value is not None:
        angle = raw_norm / gap.value
    origins = tuple(str(v) for v in getattr(audit, "action_origins", ()))
    physical_raw = _token(getattr(audit, "metadata", {}).get("full_residual_origin", "")) == "physical_raw_same_center"
    model_overlap = None
    model_angle_degrees = None
    if model_mode is not None:
        model_overlap = min(1.0, max(0.0, sign_invariant_overlap(evaluated, model_mode, coordinate_space)))
        model_angle_degrees = math.degrees(math.acos(model_overlap))
    olsen_norm = None
    if olsen_correction is not None:
        correction = coordinate_space.project(olsen_correction)
        if not np.all(np.isfinite(correction)):
            raise ValueError("Olsen correction contains non-finite values")
        olsen_norm = float(np.linalg.norm(correction))
    return StandardModeDiagnosticRecord(
        available=raw_norm is not None,
        evaluated_mode=evaluated,
        returned_mode=returned,
        evaluated_mode_id=_mode_id(evaluated, coordinate_space, "evaluated-mode"),
        returned_mode_id=_mode_id(returned, coordinate_space, "returned-mode"),
        geometry_id=(getattr(audit, "action_geometry_ids", ("",)) or ("",))[0],
        state_id=-1,
        state_uid=(getattr(audit, "action_state_uids", ("",)) or ("",))[0],
        coordinate_space_id=coordinate_space.identity,
        projection_convention=coordinate_space.convention,
        null_mode_policy=coordinate_space.null_mode_policy,
        rho=rho,
        residual=None,
        residual_norm=raw_norm,
        relative_torque=relative,
        conditional_angle_radians=angle,
        conditional_angle_degrees=(None if angle is None else math.degrees(angle)),
        model_mode_overlap=model_overlap,
        model_mode_angle_degrees=model_angle_degrees,
        model_mode_source=_token(model_mode_source),
        olsen_correction_norm=olsen_norm,
        olsen_correction_source=_token(olsen_correction_source),
        gap=gap,
        numerator_origin=("solver_audit_full_raw_physical" if physical_raw else "solver_audit_model_or_untyped"),
        numerator_physical=physical_raw,
        action_source=",".join(tuple(str(v) for v in getattr(audit, "action_sources", ()))),
        action_family=",".join(tuple(str(v) for v in getattr(audit, "action_families", ()))),
        action_operator_origin=",".join(origins),
        action_purpose="algorithm",
        action_model_age=None,
        certification_eligible=bool(physical_raw and audit_physical),
        fd_distance=(next((float(v) for v in getattr(audit, "action_displacement_scales", ()) if v is not None), None)),
        fd_stencil=",".join(tuple(str(v) for v in getattr(audit, "action_stencil_schemes", ()) if str(v))),
        full_raw_residual_norm=raw_norm,
        solver_projected_residual_norm=solver_norm,
        solver_residual_origin="t09_solver_operator_residual",
        selected_root_index=getattr(audit, "selected_root_index", None),
        root_selection=root_selection,
        root_valid=bool(root_valid),
        root_status=root_status,
        negative_mode_lost=bool(negative_mode_lost),
        epsilon=float(epsilon),
        unavailable_reason=("" if raw_norm is not None else "solver_full_raw_residual_unavailable"),
        metadata={
            "stopping_rule": getattr(audit, "stopping_rule", ""),
            "stopping_reason": getattr(audit, "stopping_reason", ""),
            "breakdown_reason": getattr(audit, "breakdown_reason", ""),
            "projected_operator_asymmetry_norm": getattr(audit, "projected_operator_asymmetry_norm", None),
            "approximation_caveat": getattr(audit, "approximation_caveat", ""),
        },
    )


@dataclass(frozen=True)
class ModeRefreshDecision:
    selector: str
    refresh_required: bool
    criterion_satisfied: bool
    reason_code: str
    residual_value: float | None
    angular_value_degrees: float | None
    residual_threshold: float | None
    angular_threshold_degrees: float | None
    comparison_tolerance: float
    root_required: bool
    root_valid: bool
    hard_negative_mode_loss: bool
    used_gap_source: str
    schema: str = MODE_DECISION_SCHEMA


def evaluate_mode_refresh_criterion(
    record: StandardModeDiagnosticRecord,
    *,
    selector: str,
    mode_residual_threshold: float | None = None,
    mode_angular_error_threshold_degrees: float | None = None,
    legacy_refresh_required: bool | None = None,
    require_root_validity: bool = True,
    lost_negative_mode: bool | None = None,
    comparison_tolerance: float = 0.0,
) -> ModeRefreshDecision:
    """Pure deterministic evaluator for shared-runtime/mode-schedule production consumption.

    The function performs no calculator/HVP/model operation.  ``hybrid`` is
    conservative: *both* residual and angular criteria must pass to avoid a
    refresh.  Equality passes, with an optional explicit absolute tolerance.
    Unavailable/nonfinite required data fail closed.  Root invalidity and lost
    negative mode are hard refreshes when requested by the caller.
    """

    selector = _token(selector)
    if selector not in MODE_REFRESH_SELECTORS:
        raise ValueError(f"unsupported mode refresh selector {selector!r}")
    tol = float(comparison_tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("comparison_tolerance must be finite and >= 0")
    lost = bool(record.negative_mode_lost if lost_negative_mode is None else lost_negative_mode)

    def decision(refresh: bool, reason: str, satisfied: bool) -> ModeRefreshDecision:
        return ModeRefreshDecision(
            selector=selector,
            refresh_required=bool(refresh),
            criterion_satisfied=bool(satisfied),
            reason_code=reason,
            residual_value=_finite_or_none(record.residual_norm),
            angular_value_degrees=_finite_or_none(record.conditional_angle_degrees),
            residual_threshold=_finite_or_none(mode_residual_threshold),
            angular_threshold_degrees=_finite_or_none(mode_angular_error_threshold_degrees),
            comparison_tolerance=tol,
            root_required=bool(require_root_validity),
            root_valid=bool(record.root_valid and not record.gap.degeneracy_unresolved),
            hard_negative_mode_loss=lost,
            used_gap_source=record.gap.source,
        )

    if selector == "legacy":
        if legacy_refresh_required is None:
            raise ValueError("legacy selector requires legacy_refresh_required from mode-schedule")
        return decision(bool(legacy_refresh_required), "legacy_passthrough", not bool(legacy_refresh_required))

    if lost:
        return decision(True, "lost_negative_mode", False)
    if require_root_validity and (not record.root_valid or record.root_status in {"wrong_root", "homed_root_lost"}):
        return decision(True, "root_invalid", False)
    if not record.available:
        return decision(True, "diagnostic_unavailable", False)

    residual_threshold = _finite_or_none(mode_residual_threshold)
    angular_threshold = _finite_or_none(mode_angular_error_threshold_degrees)
    residual_value = _finite_or_none(record.residual_norm)
    angular_value = _finite_or_none(record.conditional_angle_degrees)

    residual_pass: bool | None = None
    if selector in {"residual", "hybrid"}:
        if residual_threshold is None or residual_threshold < 0.0:
            raise ValueError("residual/hybrid selector requires finite nonnegative mode_residual_threshold")
        if residual_value is None:
            return decision(True, "residual_nonfinite_or_unavailable", False)
        residual_pass = residual_value <= residual_threshold + tol

    angular_pass: bool | None = None
    if selector in {"angular_error", "hybrid"}:
        if angular_threshold is None or angular_threshold < 0.0:
            raise ValueError("angular/hybrid selector requires finite nonnegative angular threshold")
        if record.gap.degeneracy_unresolved or not record.gap.gap_eligible:
            return decision(True, "gap_unresolved", False)
        if angular_value is None:
            return decision(True, "angular_error_nonfinite_or_unavailable", False)
        angular_pass = angular_value <= angular_threshold + tol

    if selector == "residual":
        return decision(not bool(residual_pass), "residual_within_threshold" if residual_pass else "residual_exceeds_threshold", bool(residual_pass))
    if selector == "angular_error":
        return decision(not bool(angular_pass), "angular_error_within_threshold" if angular_pass else "angular_error_exceeds_threshold", bool(angular_pass))

    # Hybrid semantics are explicit: all requested quality criteria must pass.
    hybrid_pass = bool(residual_pass and angular_pass)
    return decision(
        not hybrid_pass,
        "hybrid_all_criteria_within_threshold" if hybrid_pass else "hybrid_quality_criterion_failed",
        hybrid_pass,
    )


@dataclass(frozen=True)
class NoninterferenceReport:
    unchanged: bool
    changed_labels: tuple[str, ...]
    inspected_labels: tuple[str, ...]
    before_algorithm_pes_calls: int
    after_algorithm_pes_calls: int
    before_physical_total_pes_calls: int
    after_physical_total_pes_calls: int
    schema: str = NONINTERFERENCE_SCHEMA


class DiagnosticNoninterferenceProbe:
    """Caller-supplied snapshotters for optimizer/cache/RNG/calculator state.

    mode-diagnostics cannot know every runtime-owned object's serialization contract.  shared-runtime
    supplies zero-argument snapshot functions for the states it owns (history
    IDs, algorithm force cache, convergence, RNG, calculator state where
    inspectable).  Returned snapshots are normalized through ``json_safe`` and
    compared structurally after diagnostic work.
    """

    def __init__(self, snapshotters: Mapping[str, Callable[[], object]] | None = None) -> None:
        self.snapshotters = dict(snapshotters or {})

    def snapshot(self) -> dict[str, object]:
        return {name: json_safe(fn()) for name, fn in sorted(self.snapshotters.items())}

    def verify(
        self,
        before: Mapping[str, object],
        *,
        accounting_before: ForceAccounting,
        accounting_after: ForceAccounting,
        strict: bool = True,
    ) -> NoninterferenceReport:
        after = self.snapshot()
        changed = tuple(name for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name))
        algorithm_changed = accounting_before.algorithm_pes_calls != accounting_after.algorithm_pes_calls
        unchanged = not changed and not algorithm_changed
        report = NoninterferenceReport(
            unchanged=unchanged,
            changed_labels=changed + (("algorithm_pes_calls",) if algorithm_changed else ()),
            inspected_labels=tuple(sorted(after)),
            before_algorithm_pes_calls=accounting_before.algorithm_pes_calls,
            after_algorithm_pes_calls=accounting_after.algorithm_pes_calls,
            before_physical_total_pes_calls=accounting_before.physical_total_pes_calls,
            after_physical_total_pes_calls=accounting_after.physical_total_pes_calls,
        )
        if strict and not report.unchanged:
            raise RuntimeError(f"diagnostic noninterference violation: {report.changed_labels}")
        return report


@dataclass(frozen=True)
class ReferenceComputation:
    mode: Array | None
    curvature: float | None
    converged: bool
    pes_calls: int
    source: str
    root_selection: str = RootSelection.LOWEST.value
    unavailable_reason: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode is not None:
            object.__setattr__(self, "mode", freeze_array(self.mode))
        calls = int(self.pes_calls)
        if calls < 0:
            raise ValueError("reference pes_calls must be >= 0")
        object.__setattr__(self, "pes_calls", calls)
        object.__setattr__(self, "source", _token(self.source))
        object.__setattr__(self, "root_selection", _token(self.root_selection))
        object.__setattr__(self, "unavailable_reason", _token(self.unavailable_reason))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True)
class ReferenceModeRecord:
    available: bool
    mode: Array | None
    curvature: float | None
    converged: bool
    overlap_with_evaluated_mode: float | None
    angle_degrees: float | None
    source: str
    root_selection: str
    pes_calls: int
    elapsed_ns: int
    unavailable_reason: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema: str = REFERENCE_MODE_SCHEMA

    def __post_init__(self) -> None:
        if self.mode is not None:
            object.__setattr__(self, "mode", freeze_array(self.mode))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


def _copy_accounting(accounting: ForceAccounting) -> ForceAccounting:
    return ForceAccounting.from_state_dict(accounting.to_state_dict())


def run_paid_reference_measurement(
    measure: Callable[[], ReferenceComputation],
    *,
    evaluated_mode: object,
    coordinate_space: ActiveCoordinateSpace,
    accounting: ForceAccounting,
    noninterference: DiagnosticNoninterferenceProbe | None = None,
    strict_noninterference: bool = True,
) -> tuple[ReferenceModeRecord, NoninterferenceReport]:
    """Run an explicitly paid, diagnostic-only reference eigensolve callback.

    The callback is external so mode-diagnostics does not own calculators or eigensolver
    factories.  Its declared PES call count is charged to diagnostic work and
    physical total work.  Algorithm PES count must remain unchanged.  Runtime
    state mutation is checked through caller-supplied snapshotters.
    """

    probe = noninterference or DiagnosticNoninterferenceProbe()
    before_state = probe.snapshot()
    before_accounting = _copy_accounting(accounting)
    started = perf_counter_ns()
    computation = measure()
    elapsed = perf_counter_ns() - started
    if not isinstance(computation, ReferenceComputation):
        raise TypeError("paid reference callback must return ReferenceComputation")
    accounting.record_force_event(
        purpose=WorkPurpose.DIAGNOSTIC,
        pes_call_delta=computation.pes_calls,
        cache_hit=False,
        elapsed_ns=elapsed,
    )
    after_accounting = _copy_accounting(accounting)
    report = probe.verify(
        before_state,
        accounting_before=before_accounting,
        accounting_after=after_accounting,
        strict=strict_noninterference,
    )
    if computation.mode is None:
        return (
            ReferenceModeRecord(
                available=False,
                mode=None,
                curvature=_finite_or_none(computation.curvature),
                converged=bool(computation.converged),
                overlap_with_evaluated_mode=None,
                angle_degrees=None,
                source=computation.source,
                root_selection=computation.root_selection,
                pes_calls=computation.pes_calls,
                elapsed_ns=elapsed,
                unavailable_reason=computation.unavailable_reason or "reference_mode_unavailable",
                metadata=computation.metadata,
            ),
            report,
        )
    ref = coordinate_space.normalized(computation.mode)
    overlap = sign_invariant_overlap(evaluated_mode, ref, coordinate_space)
    overlap = min(1.0, max(0.0, overlap))
    return (
        ReferenceModeRecord(
            available=True,
            mode=ref,
            curvature=_finite_or_none(computation.curvature),
            converged=bool(computation.converged),
            overlap_with_evaluated_mode=overlap,
            angle_degrees=math.degrees(math.acos(overlap)),
            source=computation.source,
            root_selection=computation.root_selection,
            pes_calls=computation.pes_calls,
            elapsed_ns=elapsed,
            unavailable_reason="",
            metadata=computation.metadata,
        ),
        report,
    )


__all__ = [
    "MODE_DIAGNOSTIC_SCHEMA",
    "MODE_GAP_SCHEMA",
    "MODE_DECISION_SCHEMA",
    "REFERENCE_MODE_SCHEMA",
    "MODE_REFRESH_SELECTORS",
    "ModeGapEstimate",
    "StandardModeDiagnosticRecord",
    "ModeRefreshDecision",
    "ReferenceComputation",
    "ReferenceModeRecord",
    "NoninterferenceReport",
    "DiagnosticNoninterferenceProbe",
    "sign_invariant_overlap",
    "gap_from_ritz",
    "gap_from_model_spectrum",
    "compute_model_gap",
    "select_gap_estimate",
    "standardize_hvp_diagnostic",
    "diagnostic_from_dimer_torque",
    "standardize_solver_audit",
    "evaluate_mode_refresh_criterion",
    "run_paid_reference_measurement",
]
