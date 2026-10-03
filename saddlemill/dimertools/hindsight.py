"""Passive trajectory-to-final and current-reference mode diagnostics.

This mode-diagnostics module deliberately separates two different questions:

* ``final_mode_overlap`` asks whether a path mode resembles the eventual terminal
  saddle mode.  It is hindsight, not current-geometry accuracy.
* ``current_reference_overlap`` compares against an explicitly paid/reference
  eigensolve at the *same* geometry and is calibration data when available.

Geometry distances use an explicit periodic wrapping/alignment convention and
typed-HVP active-coordinate mask.  The built-in stationary-core alignment mirrors the
established SaddleMill least-moving-half drift convention; callers may keep
alignment disabled when a different already-resolved coordinate convention is
required by the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from saddlemill.dimertools.foundation_types import (
    ActiveCoordinateSpace,
    freeze_array,
    freeze_mapping,
    json_safe,
)
from saddlemill.dimertools.mode_diagnostics import (
    ReferenceModeRecord,
    StandardModeDiagnosticRecord,
    sign_invariant_overlap,
)

Array = np.ndarray

HINDSIGHT_SCHEMA = "saddlemill_mode_hindsight_v1"
HINDSIGHT_STATE_SCHEMA = "saddlemill_mode_hindsight_path_state_v1"
GEOMETRY_CONVENTION_SCHEMA = "saddlemill_hindsight_geometry_convention_v1"


def _token(value: object) -> str:
    return str(value).strip().lower()


def _finite_or_none(value: object | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


@dataclass(frozen=True)
class HindsightGeometryConvention:
    """Explicit geometry comparison convention for trajectory-to-final records."""

    cell: Array | None = None
    pbc: tuple[bool, bool, bool] = (False, False, False)
    wrapping: str = "fractional_nearest_image"
    alignment: str = "stationary_core_least_moving_half"
    distance_metric: str = "active_cartesian_rms_and_maxdev"
    schema: str = GEOMETRY_CONVENTION_SCHEMA

    def __post_init__(self) -> None:
        pbc = tuple(bool(v) for v in self.pbc)
        if len(pbc) != 3:
            raise ValueError("pbc must contain exactly three booleans")
        object.__setattr__(self, "pbc", pbc)
        wrapping = _token(self.wrapping)
        alignment = _token(self.alignment)
        if wrapping not in {"none", "fractional_nearest_image"}:
            raise ValueError(f"unsupported wrapping convention {self.wrapping!r}")
        if alignment not in {"none", "stationary_core_least_moving_half"}:
            raise ValueError(f"unsupported alignment convention {self.alignment!r}")
        object.__setattr__(self, "wrapping", wrapping)
        object.__setattr__(self, "alignment", alignment)
        object.__setattr__(self, "distance_metric", _token(self.distance_metric))
        if self.cell is not None:
            cell = np.asarray(self.cell, dtype=float)
            if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
                raise ValueError("cell must be a finite (3, 3) matrix")
            if any(pbc) and abs(float(np.linalg.det(cell))) <= 1.0e-14:
                raise ValueError("periodic geometry comparison requires an invertible cell")
            object.__setattr__(self, "cell", freeze_array(cell, shape=(3, 3)))
        elif any(pbc):
            raise ValueError("periodic geometry comparison requires a cell")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "cell": None if self.cell is None else self.cell.tolist(),
            "pbc": list(self.pbc),
            "wrapping": self.wrapping,
            "alignment": self.alignment,
            "distance_metric": self.distance_metric,
        }


def _mic_displacements(displacements: Array, convention: HindsightGeometryConvention) -> Array:
    delta = np.asarray(displacements, dtype=float).copy()
    if convention.wrapping == "none" or not any(convention.pbc):
        return delta
    assert convention.cell is not None
    # Explicit componentwise fractional nearest-image wrapping.  This is a
    # deterministic convention, not a hidden topology/alignment criterion.
    inv_cell = np.linalg.inv(np.asarray(convention.cell, dtype=float))
    fractional = delta @ inv_cell
    for axis, periodic in enumerate(convention.pbc):
        if periodic:
            fractional[:, axis] -= np.round(fractional[:, axis])
    return fractional @ np.asarray(convention.cell, dtype=float)


def aligned_geometry_displacements(
    positions: object,
    final_positions: object,
    *,
    coordinate_space: ActiveCoordinateSpace,
    convention: HindsightGeometryConvention,
) -> tuple[Array, Array]:
    """Return active-space displacements and the applied stationary-core drift."""

    pos = np.asarray(positions, dtype=float)
    final = np.asarray(final_positions, dtype=float)
    shape = coordinate_space.active_dof_mask.shape
    if pos.shape != shape or final.shape != shape:
        raise ValueError(f"geometry shape must match coordinate space {shape}")
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(final)):
        raise ValueError("geometry contains non-finite coordinates")
    delta = _mic_displacements(pos - final, convention)
    mask = coordinate_space.active_dof_mask
    delta = np.where(mask, delta, 0.0)
    drift = np.zeros(3, dtype=float)
    if convention.alignment == "stationary_core_least_moving_half":
        # Magnitudes use only active Cartesian coordinates. Atoms with no active
        # DOF are not allowed to define the stationary core.
        active_atoms = np.any(mask, axis=1)
        if np.any(active_atoms):
            magnitudes = np.linalg.norm(delta[active_atoms], axis=1)
            median = float(np.median(magnitudes))
            core_local = magnitudes <= median
            core_indices = np.flatnonzero(active_atoms)[core_local]
            if core_indices.size:
                # Respect partial Cartesian masks coordinate-wise.
                for axis in range(3):
                    eligible = core_indices[mask[core_indices, axis]]
                    if eligible.size:
                        drift[axis] = float(np.mean(delta[eligible, axis]))
                delta = np.where(mask, delta - drift[None, :], 0.0)
                # Reapply MIC after drift subtraction, matching the established
                # core-drift comparison order for periodic coordinates.
                delta = _mic_displacements(delta, convention)
                delta = np.where(mask, delta, 0.0)
    return delta, drift


def geometry_distance_to_final(
    positions: object,
    final_positions: object,
    *,
    coordinate_space: ActiveCoordinateSpace,
    convention: HindsightGeometryConvention,
) -> dict[str, object]:
    delta, drift = aligned_geometry_displacements(
        positions,
        final_positions,
        coordinate_space=coordinate_space,
        convention=convention,
    )
    mask = coordinate_space.active_dof_mask
    active = delta[mask]
    rms = 0.0 if active.size == 0 else float(np.sqrt(np.mean(active * active)))
    per_atom = np.linalg.norm(np.where(mask, delta, 0.0), axis=1)
    active_atoms = np.any(mask, axis=1)
    maxdev = 0.0 if not np.any(active_atoms) else float(np.max(per_atom[active_atoms]))
    return {
        "active_cartesian_rms": rms,
        "maxdev": maxdev,
        "stationary_core_drift": drift.tolist(),
        "wrapping": convention.wrapping,
        "alignment": convention.alignment,
        "coordinate_space_id": coordinate_space.identity,
    }


@dataclass(frozen=True)
class HindsightPathState:
    state_id: int
    geometry_id: str
    positions: Array
    mode: Array | None
    curvature: float | None
    mode_source: str
    evaluated_state_id: str = ""
    diagnostic: StandardModeDiagnosticRecord | None = None
    reference: ReferenceModeRecord | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema: str = HINDSIGHT_STATE_SCHEMA

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
            raise ValueError("positions must be finite shape (N, 3)")
        object.__setattr__(self, "positions", freeze_array(positions))
        if self.mode is not None:
            mode = np.asarray(self.mode, dtype=float)
            if mode.shape != positions.shape or not np.all(np.isfinite(mode)):
                raise ValueError("path mode must be finite and match positions shape")
            object.__setattr__(self, "mode", freeze_array(mode))
        object.__setattr__(self, "state_id", int(self.state_id))
        object.__setattr__(self, "geometry_id", str(self.geometry_id))
        object.__setattr__(self, "mode_source", _token(self.mode_source))
        object.__setattr__(self, "evaluated_state_id", str(self.evaluated_state_id))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True)
class HindsightRecord:
    state_id: int
    geometry_id: str
    mode_source: str
    evaluated_state_id: str
    final_mode_available: bool
    final_mode_overlap: float | None
    final_mode_angle_degrees: float | None
    current_rho: float | None
    current_residual_norm: float | None
    current_relative_torque: float | None
    current_conditional_angle_degrees: float | None
    current_model_mode_angle_degrees: float | None
    current_olsen_correction_norm: float | None
    current_gap_value: float | None
    current_gap_source: str
    current_gap_physical: bool | None
    current_fd_distance: float | None
    current_fd_stencil: str
    current_fd_status: str
    current_reference_available: bool
    current_reference_overlap: float | None
    current_reference_angle_degrees: float | None
    current_reference_source: str
    current_reference_root_selection: str
    current_reference_pes_calls: int
    current_reference_elapsed_ns: int
    current_reference_converged: bool | None
    geometry_distance_rms: float | None
    geometry_maxdev: float | None
    stationary_core_drift: tuple[float, float, float] | None
    curvature: float | None
    final_curvature: float | None
    curvature_error: float | None
    terminal_mode_source: str
    terminal_mode_availability_reason: str
    run_converged: bool
    geometry_convention: Mapping[str, object]
    coordinate_space_id: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema: str = HINDSIGHT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry_convention", freeze_mapping(self.geometry_convention))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "state_id": self.state_id,
            "geometry_id": self.geometry_id,
            "mode_source": self.mode_source,
            "evaluated_state_id": self.evaluated_state_id,
            "final_mode_available": self.final_mode_available,
            "final_mode_overlap": self.final_mode_overlap,
            "final_mode_angle_degrees": self.final_mode_angle_degrees,
            "current_rho": self.current_rho,
            "current_residual_norm": self.current_residual_norm,
            "current_relative_torque": self.current_relative_torque,
            "current_conditional_angle_degrees": self.current_conditional_angle_degrees,
            "current_model_mode_angle_degrees": self.current_model_mode_angle_degrees,
            "current_olsen_correction_norm": self.current_olsen_correction_norm,
            "current_gap_value": self.current_gap_value,
            "current_gap_source": self.current_gap_source,
            "current_gap_physical": self.current_gap_physical,
            "current_fd_distance": self.current_fd_distance,
            "current_fd_stencil": self.current_fd_stencil,
            "current_fd_status": self.current_fd_status,
            "current_reference_available": self.current_reference_available,
            "current_reference_overlap": self.current_reference_overlap,
            "current_reference_angle_degrees": self.current_reference_angle_degrees,
            "current_reference_source": self.current_reference_source,
            "current_reference_root_selection": self.current_reference_root_selection,
            "current_reference_pes_calls": self.current_reference_pes_calls,
            "current_reference_elapsed_ns": self.current_reference_elapsed_ns,
            "current_reference_converged": self.current_reference_converged,
            "geometry_distance_rms": self.geometry_distance_rms,
            "geometry_maxdev": self.geometry_maxdev,
            "stationary_core_drift": None if self.stationary_core_drift is None else list(self.stationary_core_drift),
            "curvature": self.curvature,
            "final_curvature": self.final_curvature,
            "curvature_error": self.curvature_error,
            "terminal_mode_source": self.terminal_mode_source,
            "terminal_mode_availability_reason": self.terminal_mode_availability_reason,
            "run_converged": self.run_converged,
            "geometry_convention": json_safe(self.geometry_convention),
            "coordinate_space_id": self.coordinate_space_id,
            "metadata": json_safe(self.metadata),
        }


def _angle_from_overlap(overlap: float | None) -> float | None:
    if overlap is None or not np.isfinite(float(overlap)):
        return None
    clipped = min(1.0, max(0.0, float(overlap)))
    return math.degrees(math.acos(clipped))


def build_hindsight_records(
    states: Sequence[HindsightPathState],
    *,
    coordinate_space: ActiveCoordinateSpace,
    geometry_convention: HindsightGeometryConvention,
    final_positions: object | None,
    final_mode: object | None,
    final_curvature: float | None,
    terminal_mode_source: str,
    run_converged: bool,
    terminal_mode_availability_reason: str = "",
) -> tuple[HindsightRecord, ...]:
    """Build passive trajectory-to-final records for converged or failed runs."""

    final_mode_normalized: Array | None = None
    if final_mode is not None:
        final_mode_normalized = coordinate_space.normalized(final_mode)
    final_positions_array: Array | None = None
    if final_positions is not None:
        final_positions_array = np.asarray(final_positions, dtype=float)
        if final_positions_array.shape != coordinate_space.active_dof_mask.shape:
            raise ValueError("final_positions shape does not match coordinate space")
        if not np.all(np.isfinite(final_positions_array)):
            raise ValueError("final_positions contains non-finite coordinates")
    if final_mode_normalized is None and not str(terminal_mode_availability_reason).strip():
        terminal_mode_availability_reason = (
            "nonconverged_no_terminal_mode" if not run_converged else "terminal_mode_unavailable"
        )
    result: list[HindsightRecord] = []
    for state in states:
        if state.positions.shape != coordinate_space.active_dof_mask.shape:
            raise ValueError(f"state {state.state_id} geometry shape mismatch")
        final_overlap: float | None = None
        if state.mode is not None and final_mode_normalized is not None:
            final_overlap = sign_invariant_overlap(state.mode, final_mode_normalized, coordinate_space)
        diagnostic = state.diagnostic
        if diagnostic is not None:
            if diagnostic.geometry_id and diagnostic.geometry_id != state.geometry_id:
                raise ValueError(
                    f"state {state.state_id} diagnostic geometry identity mismatch"
                )
            if diagnostic.state_id >= 0 and diagnostic.state_id != state.state_id:
                raise ValueError(
                    f"state {state.state_id} diagnostic state identity mismatch"
                )
        reference = state.reference
        ref_available = bool(reference is not None and reference.available and reference.mode is not None and state.mode is not None)
        ref_overlap = None
        if ref_available and reference is not None and reference.mode is not None and state.mode is not None:
            ref_overlap = sign_invariant_overlap(state.mode, reference.mode, coordinate_space)
        distance_rms = None
        maxdev = None
        drift_tuple = None
        if final_positions_array is not None:
            distance = geometry_distance_to_final(
                state.positions,
                final_positions_array,
                coordinate_space=coordinate_space,
                convention=geometry_convention,
            )
            distance_rms = float(distance["active_cartesian_rms"])
            maxdev = float(distance["maxdev"])
            drift_tuple = tuple(float(v) for v in distance["stationary_core_drift"])
        curvature = _finite_or_none(state.curvature)
        final_curv = _finite_or_none(final_curvature)
        curvature_error = None if curvature is None or final_curv is None else curvature - final_curv
        result.append(
            HindsightRecord(
                state_id=state.state_id,
                geometry_id=state.geometry_id,
                mode_source=state.mode_source,
                evaluated_state_id=state.evaluated_state_id,
                final_mode_available=bool(final_mode_normalized is not None and state.mode is not None),
                final_mode_overlap=final_overlap,
                final_mode_angle_degrees=_angle_from_overlap(final_overlap),
                current_rho=(None if diagnostic is None else diagnostic.rho),
                current_residual_norm=(None if diagnostic is None else diagnostic.residual_norm),
                current_relative_torque=(None if diagnostic is None else diagnostic.relative_torque),
                current_conditional_angle_degrees=(None if diagnostic is None else diagnostic.conditional_angle_degrees),
                current_model_mode_angle_degrees=(None if diagnostic is None else diagnostic.model_mode_angle_degrees),
                current_olsen_correction_norm=(None if diagnostic is None else diagnostic.olsen_correction_norm),
                current_gap_value=(None if diagnostic is None else diagnostic.gap.value),
                current_gap_source=("" if diagnostic is None else diagnostic.gap.source),
                current_gap_physical=(None if diagnostic is None else bool(diagnostic.gap.physical)),
                current_fd_distance=(None if diagnostic is None else diagnostic.fd_distance),
                current_fd_stencil=("" if diagnostic is None else diagnostic.fd_stencil),
                current_fd_status=("" if diagnostic is None else diagnostic.fd_status),
                current_reference_available=ref_available,
                current_reference_overlap=ref_overlap,
                current_reference_angle_degrees=_angle_from_overlap(ref_overlap),
                current_reference_source=("" if reference is None else reference.source),
                current_reference_root_selection=("" if reference is None else reference.root_selection),
                current_reference_pes_calls=(0 if reference is None else int(reference.pes_calls)),
                current_reference_elapsed_ns=(0 if reference is None else int(reference.elapsed_ns)),
                current_reference_converged=(None if reference is None else bool(reference.converged)),
                geometry_distance_rms=distance_rms,
                geometry_maxdev=maxdev,
                stationary_core_drift=drift_tuple,
                curvature=curvature,
                final_curvature=final_curv,
                curvature_error=curvature_error,
                terminal_mode_source=_token(terminal_mode_source),
                terminal_mode_availability_reason=_token(terminal_mode_availability_reason),
                run_converged=bool(run_converged),
                geometry_convention=geometry_convention.to_dict(),
                coordinate_space_id=coordinate_space.identity,
                metadata=state.metadata,
            )
        )
    return tuple(result)


__all__ = [
    "HINDSIGHT_SCHEMA",
    "HINDSIGHT_STATE_SCHEMA",
    "GEOMETRY_CONVENTION_SCHEMA",
    "HindsightGeometryConvention",
    "HindsightPathState",
    "HindsightRecord",
    "aligned_geometry_displacements",
    "geometry_distance_to_final",
    "build_hindsight_records",
]
