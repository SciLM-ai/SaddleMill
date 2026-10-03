"""Cheap directional isopotential-curvature estimator from the GPUMD K-dimer idea.

This module deliberately contains only the one-probe estimator.  It does not
implement the GPUMD beta/gamma K-dimer translation-force blend, select runtime
regimes, or perform calculator calls.  ``prepare_probe`` and ``finish_estimate``
are pure numerical boundaries so the runtime owner can account for the one real
probe through :class:`PhysicalForceEvaluator` without hiding PES work.

Canonical SaddleMill physical-force sign is ``F = -g``.  In active coordinates,
for center force ``F0``, normalized mode ``n`` and actual probe displacement
``delta`` (Angstrom), the directional isopotential contract is::

    n_iso = normalize(n - ((n.F0)/(F0.F0))*F0)
    x1 = x0 + delta*n_iso
    F1_iso = F1 - ((F1.F0)/(F0.F0))*F0
    c0 = ((F0 - F1_iso).n_iso)/delta
    kappa_iso = -c0/||F0||

``c0`` has units eV/Angstrom^2 and ``kappa_iso`` has units 1/Angstrom.  The
quantity is directional isopotential curvature; it is not asserted to be an
extremal physical-Hessian eigenvalue.

The inspected GPUMD ``build_r2`` helper uses ``2*dimer_separation`` because that
helper is shared with its two-end dimer convention.  The directional estimator deliberately names
``delta`` as the actual center-to-probe displacement required by the assigned
contract above.  Runtime wiring must not silently substitute GPUMD
``dimer_separation`` for this quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from saddlemill.dimertools.foundation_types import (
    ActiveCoordinateSpace,
    WorkPurpose,
    geometry_fingerprint,
)

Array = np.ndarray

ISOPOTENTIAL_SOURCE = "isopotential_probe"
ISOPOTENTIAL_FAMILY = "isopotential"
ISOPOTENTIAL_DIRECTION_KIND = "isopotential_tangent"
ISOPOTENTIAL_ESTIMATOR_ORIGIN = "gpumd_directional_isopotential"
ISOPOTENTIAL_FALLBACK = "skip_estimator"

DEFAULT_FORCE_TOLERANCE = 1.0e-14  # eV/Angstrom
DEFAULT_DIRECTION_TOLERANCE = 1.0e-14
DEFAULT_MODE_TOLERANCE = 1.0e-14


def _readonly_array(value: object | None) -> Array | None:
    if value is None:
        return None
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _frozen_metadata(value: Mapping[str, object] | None = None) -> Mapping[str, object]:
    return MappingProxyType(dict(value or {}))


def _array_n3_or_reason(value: object, name: str) -> tuple[Array | None, str]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None, f"malformed_{name}"
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 1:
        return None, f"malformed_{name}"
    if not np.all(np.isfinite(array)):
        return None, f"nonfinite_{name}"
    return np.array(array, dtype=float, copy=True), ""


def _purpose_token(purpose: object) -> str:
    try:
        return WorkPurpose(str(purpose).strip().lower()).value
    except ValueError:
        return ""


@dataclass(frozen=True)
class IsopotentialProbe:
    """Prepared one-point isopotential probe or an explicit unavailable result."""

    available: bool
    unavailable_reason: str = ""
    fallback: str = ISOPOTENTIAL_FALLBACK
    center_positions: Array | None = None
    center_force: Array | None = None
    normalized_mode: Array | None = None
    direction: Array | None = None
    probe_positions: Array | None = None
    delta: float | None = None
    center_force_norm: float | None = None
    projected_direction_norm: float | None = None
    coordinate_space: ActiveCoordinateSpace | None = None
    coordinate_space_identity: str = ""
    center_geometry_id: str = ""
    probe_geometry_id: str = ""
    source: str = ISOPOTENTIAL_SOURCE
    family: str = ISOPOTENTIAL_FAMILY
    direction_kind: str = ISOPOTENTIAL_DIRECTION_KIND
    purpose: str = WorkPurpose.DIAGNOSTIC.value
    promote_observation_to_algorithm_consumers: bool = False
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        for name in (
            "center_positions",
            "center_force",
            "normalized_mode",
            "direction",
            "probe_positions",
        ):
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True)
class IsopotentialEstimate:
    """Finished directional estimate with explicit failure/fallback semantics."""

    available: bool
    unavailable_reason: str = ""
    fallback: str = ISOPOTENTIAL_FALLBACK
    kappa_iso: float | None = None
    c0: float | None = None
    center_force_norm: float | None = None
    delta: float | None = None
    direction: Array | None = None
    projected_probe_force: Array | None = None
    center_geometry_id: str = ""
    probe_geometry_id: str = ""
    coordinate_space_identity: str = ""
    source: str = ISOPOTENTIAL_SOURCE
    family: str = ISOPOTENTIAL_FAMILY
    purpose: str = WorkPurpose.DIAGNOSTIC.value
    estimator_origin: str = ISOPOTENTIAL_ESTIMATOR_ORIGIN
    c0_units: str = "eV/Angstrom^2"
    kappa_units: str = "1/Angstrom"
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _readonly_array(self.direction))
        object.__setattr__(
            self, "projected_probe_force", _readonly_array(self.projected_probe_force)
        )
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


def _unavailable_probe(
    reason: str,
    *,
    purpose: str,
    promote: bool,
    delta: float | None = None,
    coordinate_space: ActiveCoordinateSpace | None = None,
    center_geometry_id: str = "",
    metadata: Mapping[str, object] | None = None,
) -> IsopotentialProbe:
    return IsopotentialProbe(
        available=False,
        unavailable_reason=str(reason),
        delta=delta,
        coordinate_space=coordinate_space,
        coordinate_space_identity=(
            "" if coordinate_space is None else coordinate_space.identity
        ),
        center_geometry_id=center_geometry_id,
        purpose=purpose or WorkPurpose.DIAGNOSTIC.value,
        promote_observation_to_algorithm_consumers=bool(promote),
        metadata=metadata or {},
    )


def prepare_probe(
    center_positions: object,
    center_force: object,
    mode: object,
    delta: float,
    *,
    coordinate_space: ActiveCoordinateSpace | None = None,
    force_tolerance: float = DEFAULT_FORCE_TOLERANCE,
    direction_tolerance: float = DEFAULT_DIRECTION_TOLERANCE,
    mode_tolerance: float = DEFAULT_MODE_TOLERANCE,
    expected_center_geometry_id: str | None = None,
    expected_coordinate_space_identity: str | None = None,
    purpose: object = WorkPurpose.DIAGNOSTIC.value,
    promote_observation_to_algorithm_consumers: bool = False,
    center_force_source: str = "physical_center_force",
    metadata: Mapping[str, object] | None = None,
) -> IsopotentialProbe:
    """Prepare the single physical probe without evaluating a calculator.

    Scientific/numerical unavailability is returned as ``available=False``
    rather than by normalizing an unusable vector.  The returned ``delta`` is
    the *actual center-to-probe displacement amplitude* used both in ``x1`` and
    the finite difference denominator.
    """

    purpose_token = _purpose_token(purpose)
    promote = bool(promote_observation_to_algorithm_consumers)
    base_meta = dict(metadata or {})
    base_meta.update(
        {
            "estimator_origin": ISOPOTENTIAL_ESTIMATOR_ORIGIN,
            "force_sign": "F=-g",
            "probe_displacement_units": "Angstrom",
            "center_force_units": "eV/Angstrom",
            "center_force_source": str(center_force_source),
            "c0_units": "eV/Angstrom^2",
            "kappa_units": "1/Angstrom",
            "diagnostic_promotion_requested": promote,
        }
    )

    if not purpose_token:
        return _unavailable_probe(
            "invalid_purpose", purpose=WorkPurpose.DIAGNOSTIC.value,
            promote=promote, metadata=base_meta
        )
    if promote and purpose_token != WorkPurpose.DIAGNOSTIC.value:
        return _unavailable_probe(
            "promotion_requires_diagnostic_purpose", purpose=purpose_token,
            promote=promote, metadata=base_meta
        )

    try:
        delta_value = float(delta)
    except (TypeError, ValueError, OverflowError):
        delta_value = float("nan")
    if not np.isfinite(delta_value) or delta_value <= 0.0:
        return _unavailable_probe(
            "invalid_delta", purpose=purpose_token, promote=promote,
            delta=delta_value, metadata=base_meta
        )

    tolerances: dict[str, float] = {}
    for name, value in (
        ("force_tolerance", force_tolerance),
        ("direction_tolerance", direction_tolerance),
        ("mode_tolerance", mode_tolerance),
    ):
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError):
            return _unavailable_probe(
                f"invalid_{name}", purpose=purpose_token, promote=promote,
                delta=delta_value, metadata=base_meta
            )
        if not np.isfinite(scalar) or scalar < 0.0:
            return _unavailable_probe(
                f"invalid_{name}", purpose=purpose_token, promote=promote,
                delta=delta_value, metadata=base_meta
            )
        tolerances[name] = scalar
    base_meta.update(tolerances)

    positions, reason = _array_n3_or_reason(center_positions, "center_positions")
    if positions is None:
        return _unavailable_probe(
            reason, purpose=purpose_token, promote=promote,
            delta=delta_value, metadata=base_meta
        )
    center_gid = geometry_fingerprint(positions)
    if expected_center_geometry_id and str(expected_center_geometry_id) != center_gid:
        return _unavailable_probe(
            "center_geometry_identity_mismatch", purpose=purpose_token,
            promote=promote, delta=delta_value, center_geometry_id=center_gid,
            metadata=base_meta
        )

    force, reason = _array_n3_or_reason(center_force, "center_force")
    if force is None:
        return _unavailable_probe(
            reason, purpose=purpose_token, promote=promote, delta=delta_value,
            center_geometry_id=center_gid, metadata=base_meta
        )
    mode_array, reason = _array_n3_or_reason(mode, "mode")
    if mode_array is None:
        return _unavailable_probe(
            reason, purpose=purpose_token, promote=promote, delta=delta_value,
            center_geometry_id=center_gid, metadata=base_meta
        )
    if force.shape != positions.shape or mode_array.shape != positions.shape:
        return _unavailable_probe(
            "shape_mismatch", purpose=purpose_token, promote=promote,
            delta=delta_value, center_geometry_id=center_gid, metadata=base_meta
        )

    if coordinate_space is None:
        coordinate_space = ActiveCoordinateSpace.all_cartesian(positions.shape[0])
    if not isinstance(coordinate_space, ActiveCoordinateSpace):
        return _unavailable_probe(
            "invalid_coordinate_space", purpose=purpose_token, promote=promote,
            delta=delta_value, center_geometry_id=center_gid, metadata=base_meta
        )
    if coordinate_space.active_dof_mask.shape != positions.shape:
        return _unavailable_probe(
            "coordinate_space_shape_mismatch", purpose=purpose_token,
            promote=promote, delta=delta_value, coordinate_space=coordinate_space,
            center_geometry_id=center_gid, metadata=base_meta
        )
    if (
        expected_coordinate_space_identity
        and str(expected_coordinate_space_identity) != coordinate_space.identity
    ):
        return _unavailable_probe(
            "coordinate_space_identity_mismatch", purpose=purpose_token,
            promote=promote, delta=delta_value, coordinate_space=coordinate_space,
            center_geometry_id=center_gid, metadata=base_meta
        )

    try:
        force_active = coordinate_space.project(force)
        mode_active = coordinate_space.project(mode_array)
    except ValueError:
        return _unavailable_probe(
            "coordinate_projection_failure", purpose=purpose_token,
            promote=promote, delta=delta_value, coordinate_space=coordinate_space,
            center_geometry_id=center_gid, metadata=base_meta
        )

    mode_norm = float(np.linalg.norm(mode_active))
    if not np.isfinite(mode_norm) or mode_norm <= tolerances["mode_tolerance"]:
        return _unavailable_probe(
            "mode_near_zero", purpose=purpose_token, promote=promote,
            delta=delta_value, coordinate_space=coordinate_space,
            center_geometry_id=center_gid, metadata=base_meta
        )
    normalized_mode = mode_active / mode_norm

    force_norm = float(np.linalg.norm(force_active))
    if not np.isfinite(force_norm) or force_norm <= tolerances["force_tolerance"]:
        return _unavailable_probe(
            "center_force_near_zero", purpose=purpose_token, promote=promote,
            delta=delta_value, coordinate_space=coordinate_space,
            center_geometry_id=center_gid, metadata=base_meta
        )

    force_norm_sq = float(np.vdot(force_active.ravel(), force_active.ravel()).real)
    coefficient = float(
        np.vdot(normalized_mode.ravel(), force_active.ravel()).real / force_norm_sq
    )
    raw_direction = normalized_mode - coefficient * force_active
    try:
        raw_direction = coordinate_space.project(raw_direction)
    except ValueError:
        return _unavailable_probe(
            "coordinate_projection_failure", purpose=purpose_token,
            promote=promote, delta=delta_value, coordinate_space=coordinate_space,
            center_geometry_id=center_gid, metadata=base_meta
        )
    direction_norm = float(np.linalg.norm(raw_direction))
    if (
        not np.isfinite(direction_norm)
        or direction_norm <= tolerances["direction_tolerance"]
    ):
        return _unavailable_probe(
            "projected_direction_near_zero", purpose=purpose_token,
            promote=promote, delta=delta_value, coordinate_space=coordinate_space,
            center_geometry_id=center_gid, metadata=base_meta
        )
    direction = raw_direction / direction_norm
    probe_positions = positions + delta_value * direction
    probe_gid = geometry_fingerprint(probe_positions)

    result_meta = dict(base_meta)
    result_meta.update(
        {
            "center_geometry_id": center_gid,
            "probe_geometry_id": probe_gid,
            "coordinate_space_identity": coordinate_space.identity,
            "source": ISOPOTENTIAL_SOURCE,
            "family": ISOPOTENTIAL_FAMILY,
            "direction_kind": ISOPOTENTIAL_DIRECTION_KIND,
        }
    )
    return IsopotentialProbe(
        available=True,
        unavailable_reason="",
        fallback="none",
        center_positions=positions,
        center_force=force_active,
        normalized_mode=normalized_mode,
        direction=direction,
        probe_positions=probe_positions,
        delta=delta_value,
        center_force_norm=force_norm,
        projected_direction_norm=direction_norm,
        coordinate_space=coordinate_space,
        coordinate_space_identity=coordinate_space.identity,
        center_geometry_id=center_gid,
        probe_geometry_id=probe_gid,
        purpose=purpose_token,
        promote_observation_to_algorithm_consumers=promote,
        metadata=result_meta,
    )


def _unavailable_estimate(
    probe: IsopotentialProbe,
    reason: str,
    *,
    metadata: Mapping[str, object] | None = None,
) -> IsopotentialEstimate:
    merged = dict(probe.metadata)
    if metadata:
        merged.update(dict(metadata))
    return IsopotentialEstimate(
        available=False,
        unavailable_reason=str(reason),
        center_force_norm=probe.center_force_norm,
        delta=probe.delta,
        direction=probe.direction,
        center_geometry_id=probe.center_geometry_id,
        probe_geometry_id=probe.probe_geometry_id,
        coordinate_space_identity=probe.coordinate_space_identity,
        purpose=probe.purpose,
        metadata=merged,
    )


def finish_estimate(
    probe: IsopotentialProbe,
    probe_force: object,
    *,
    probe_positions: object | None = None,
    expected_probe_geometry_id: str | None = None,
    force_source: str = "physical_probe_force",
    metadata: Mapping[str, object] | None = None,
) -> IsopotentialEstimate:
    """Finish the estimator from the one raw physical probe force.

    ``probe_force`` must be the physical ``F=-g`` force at exactly the prepared
    probe geometry.  Passing ``probe_positions`` is recommended at runtime so a
    stale/cross-geometry force fails closed before a scalar is reported.
    """

    if not isinstance(probe, IsopotentialProbe):
        return IsopotentialEstimate(
            available=False,
            unavailable_reason="invalid_probe_plan",
            metadata={"probe_force_source": str(force_source)},
        )
    if not probe.available:
        return _unavailable_estimate(
            probe, probe.unavailable_reason or "probe_unavailable",
            metadata={"probe_force_source": str(force_source)},
        )
    if (
        probe.coordinate_space is None
        or probe.center_force is None
        or probe.direction is None
        or probe.probe_positions is None
        or probe.delta is None
        or probe.center_force_norm is None
    ):
        return _unavailable_estimate(probe, "incomplete_probe_plan")

    expected_gid = probe.probe_geometry_id
    if expected_probe_geometry_id and str(expected_probe_geometry_id) != expected_gid:
        return _unavailable_estimate(probe, "probe_geometry_identity_mismatch")
    if probe_positions is not None:
        positions, reason = _array_n3_or_reason(probe_positions, "probe_positions")
        if positions is None:
            return _unavailable_estimate(probe, reason)
        if positions.shape != probe.probe_positions.shape:
            return _unavailable_estimate(probe, "shape_mismatch")
        if geometry_fingerprint(positions) != expected_gid:
            return _unavailable_estimate(probe, "probe_geometry_identity_mismatch")

    force1, reason = _array_n3_or_reason(probe_force, "probe_force")
    if force1 is None:
        return _unavailable_estimate(probe, reason)
    if force1.shape != probe.center_force.shape:
        return _unavailable_estimate(probe, "shape_mismatch")
    try:
        force1_active = probe.coordinate_space.project(force1)
    except ValueError:
        return _unavailable_estimate(probe, "coordinate_projection_failure")

    force0 = np.asarray(probe.center_force, dtype=float)
    force0_norm_sq = float(np.vdot(force0.ravel(), force0.ravel()).real)
    if force0_norm_sq <= 0.0 or not np.isfinite(force0_norm_sq):
        return _unavailable_estimate(probe, "center_force_near_zero")
    f1_dot_f0 = float(np.vdot(force1_active.ravel(), force0.ravel()).real)
    force1_iso = force1_active - (f1_dot_f0 / force0_norm_sq) * force0
    try:
        force1_iso = probe.coordinate_space.project(force1_iso)
    except ValueError:
        return _unavailable_estimate(probe, "coordinate_projection_failure")

    numerator = float(
        np.vdot((force0 - force1_iso).ravel(), probe.direction.ravel()).real
    )
    c0 = numerator / float(probe.delta)
    kappa = -c0 / float(probe.center_force_norm)
    if not np.isfinite(c0) or not np.isfinite(kappa):
        return _unavailable_estimate(probe, "nonfinite_estimate")

    merged = dict(probe.metadata)
    merged.update(dict(metadata or {}))
    merged.update(
        {
            "probe_force_source": str(force_source),
            "center_geometry_id": probe.center_geometry_id,
            "probe_geometry_id": probe.probe_geometry_id,
            "coordinate_space_identity": probe.coordinate_space_identity,
        }
    )
    return IsopotentialEstimate(
        available=True,
        unavailable_reason="",
        fallback="none",
        kappa_iso=float(kappa),
        c0=float(c0),
        center_force_norm=float(probe.center_force_norm),
        delta=float(probe.delta),
        direction=probe.direction,
        projected_probe_force=force1_iso,
        center_geometry_id=probe.center_geometry_id,
        probe_geometry_id=probe.probe_geometry_id,
        coordinate_space_identity=probe.coordinate_space_identity,
        purpose=probe.purpose,
        metadata=merged,
    )


def probe_observation_kwargs(
    probe: IsopotentialProbe,
    *,
    evaluation: str,
    force_call_delta: int,
    cache_hit: bool | None = None,
    force_source: str = "physical_probe_force",
) -> dict[str, object]:
    """Build exact typed-HVP ``PhysicalForceEvaluator.record_probe`` provenance.

    This function does *not* call the evaluator and therefore cannot hide a PES
    call.  shared-runtime must call the calculator/cache, classify the work as algorithm or
    diagnostic, and then pass these kwargs to ``record_probe``.  No stencil ID
    is supplied: this estimator probe is not a physical-Hessian finite-
    difference block and is not admitted to unrelated QN consumers by default.
    """

    if not isinstance(probe, IsopotentialProbe) or not probe.available:
        raise ValueError("cannot record an unavailable isopotential probe")
    if probe.probe_positions is None or probe.direction is None or probe.delta is None:
        raise ValueError("isopotential probe plan is incomplete")
    delta_count = int(force_call_delta)
    if delta_count < 0:
        raise ValueError("force_call_delta must be >= 0")
    merged = dict(probe.metadata)
    merged.update(
        {
            "probe_force_source": str(force_source),
            "estimator_origin": ISOPOTENTIAL_ESTIMATOR_ORIGIN,
            "center_geometry_id": probe.center_geometry_id,
            "probe_geometry_id": probe.probe_geometry_id,
            "coordinate_space_identity": probe.coordinate_space_identity,
            "diagnostic_promotion_requested": bool(
                probe.promote_observation_to_algorithm_consumers
            ),
        }
    )
    return {
        "positions": probe.probe_positions,
        "evaluation": str(evaluation),
        "force_call_delta": delta_count,
        "source": probe.source,
        "family": probe.family,
        "metadata": merged,
        "direction": probe.direction,
        "direction_kind": probe.direction_kind,
        "stencil_id": "",
        "dimer_side": 1,
        "offset": float(probe.delta),
        "purpose": probe.purpose,
        "cache_hit": cache_hit,
    }


__all__ = [
    "DEFAULT_DIRECTION_TOLERANCE",
    "DEFAULT_FORCE_TOLERANCE",
    "DEFAULT_MODE_TOLERANCE",
    "ISOPOTENTIAL_DIRECTION_KIND",
    "ISOPOTENTIAL_ESTIMATOR_ORIGIN",
    "ISOPOTENTIAL_FAMILY",
    "ISOPOTENTIAL_FALLBACK",
    "ISOPOTENTIAL_SOURCE",
    "IsopotentialEstimate",
    "IsopotentialProbe",
    "finish_estimate",
    "prepare_probe",
    "probe_observation_kwargs",
]
