"""Derivative-stencil views over :mod:`force_history` raw observations.

The force bank stores physical positions/forces plus relational metadata by default.
An explicit accepted-rotation policy may additionally retain ASE Fourier-extrapolated
accepted endpoint forces as derived (never physical) observations.  This module
reconstructs Dimer torques and mode-space L-BFGS secants on demand.
Translation steps are never interpreted as rotational secants: rotational
pairs are made only between two complete Dimer torque stencils at the same
center state.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Iterable, Mapping, Sequence

import numpy as np

from saddlemill.dimertools.force_history import (
    CanonicalForceHistory,
    DerivativeStencil,
    ForceObservation,
)
from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace, WorkPurpose
from saddlemill.dimertools.hvp_interfaces import (
    FiniteDifferenceForceHVPBackend,
    HVPRequest,
    HVPResult,
)
from saddlemill.dimertools.riemannian_lbfgs import (
    RotationSecant,
    make_rotation_secant,
)
from saddlemill.dimertools.sphere_manifold import (
    normalize_axis,
    project_tangent_and_basis,
    sign_align_axis,
)

Array = np.ndarray


@dataclass(frozen=True)
class DimerTorqueSample:
    """One Dimer torque reconstructed from physical or explicitly allowed derived endpoints."""

    state_id: int
    serial: int
    stencil_id: str
    mode: Array
    torque: Array
    curvature: float
    scheme: str
    source: str
    endpoint_serials: tuple[int, ...]
    metadata: Mapping[str, object]
    physical: bool = True


@dataclass(frozen=True)
class ForceBankRotationBuild:
    samples: tuple[DimerTorqueSample, ...]
    pairs: tuple[RotationSecant, ...]
    metrics: Mapping[str, object]


def _physical(history: CanonicalForceHistory, serial: int | None) -> ForceObservation | None:
    if serial is None:
        return None
    observation = history.observation_by_serial(int(serial))
    if observation is None or not observation.is_physical or observation.is_diagnostic:
        return None
    return observation


def _rotation_endpoint(
    history: CanonicalForceHistory,
    serial: int | None,
    *,
    allow_extrapolated_accepted: bool,
) -> ForceObservation | None:
    """Return a physical endpoint or an explicitly tagged ASE-extrapolated accepted endpoint."""

    if serial is None:
        return None
    observation = history.observation_by_serial(int(serial))
    if observation is None:
        return None
    if observation.is_physical and not observation.is_diagnostic:
        return observation
    if (
        allow_extrapolated_accepted
        and observation.evaluation == "derived_extrapolated"
        and int(observation.metadata.get("rotation_extrapolated_accepted", 0)) == 1
    ):
        return observation
    return None


def reconstruct_physical_hvp(
    history: CanonicalForceHistory,
    stencil: DerivativeStencil,
    *,
    coordinate_space: ActiveCoordinateSpace,
    purpose: WorkPurpose | str = WorkPurpose.ALGORITHM,
) -> HVPResult:
    """Reconstruct the physical same-center ``Hq`` represented by ``stencil``.

    This is the canonical backend-import path for force-bank finite differences.
    The returned HVP is already scaled by ``h`` exactly once.
    """

    state = next((item for item in history.states if item.state_id == stencil.state_id), None)
    if state is None:
        raise ValueError("stencil state is not retained in the force history")
    request = HVPRequest(
        state_id=state.state_id,
        state_uid=state.state_uid,
        geometry_id=state.geometry_id,
        direction=stencil.direction,
        coordinate_space=coordinate_space,
        purpose=purpose,
        source=stencil.source,
        family=stencil.family,
        stencil_id=stencil.stencil_id,
        metadata={"endpoint_observation_ids": history.stencil_endpoint_ids(stencil)},
    )
    return FiniteDifferenceForceHVPBackend(history).apply(request)


def reconstruct_dimer_torque(
    history: CanonicalForceHistory,
    stencil: DerivativeStencil,
    *,
    basis: Iterable[object] | object | None = None,
    allow_extrapolated_accepted: bool = False,
) -> DimerTorqueSample | None:
    """Reconstruct one Dimer torque; derived accepted endpoints are opt-in."""

    if stencil.family != "rotation" or stencil.direction_kind != "dimer_mode":
        return None
    center = _physical(history, stencil.center_serial)
    plus = _rotation_endpoint(
        history, stencil.plus_serial,
        allow_extrapolated_accepted=allow_extrapolated_accepted,
    )
    minus = _rotation_endpoint(
        history, stencil.minus_serial,
        allow_extrapolated_accepted=allow_extrapolated_accepted,
    )
    if center is None or (plus is None and minus is None):
        return None
    mode = normalize_axis(stencil.direction, shape=center.forces.shape)
    scale = float(stencil.scale)
    if scale <= 0.0 or not np.isfinite(scale):
        return None

    if plus is not None and minus is not None:
        difference = plus.forces - minus.forces
        torque_raw = difference / (2.0 * scale)
        curvature = float(
            np.vdot((minus.forces - plus.forces).ravel(), mode.ravel()).real
            / (2.0 * scale)
        )
        scheme = "centered"
        serials = (plus.serial, minus.serial)
    elif plus is not None:
        torque_raw = (plus.forces - center.forces) / scale
        curvature = float(
            np.vdot((center.forces - plus.forces).ravel(), mode.ravel()).real
            / scale
        )
        scheme = "one_sided_plus"
        serials = (plus.serial,)
    else:
        assert minus is not None
        torque_raw = (center.forces - minus.forces) / scale
        curvature = float(
            np.vdot((minus.forces - center.forces).ravel(), mode.ravel()).real
            / scale
        )
        scheme = "one_sided_minus"
        serials = (minus.serial,)

    torque = project_tangent_and_basis(mode, torque_raw, basis)
    if not np.all(np.isfinite(torque)):
        return None
    return DimerTorqueSample(
        state_id=stencil.state_id,
        serial=stencil.serial,
        stencil_id=stencil.stencil_id,
        mode=mode,
        torque=np.asarray(torque, dtype=float),
        curvature=curvature,
        scheme=scheme,
        source=stencil.source,
        endpoint_serials=tuple(int(value) for value in serials),
        physical=bool(all(
            (history.observation_by_serial(int(value)) is not None
             and history.observation_by_serial(int(value)).is_physical)
            for value in serials
        )),
        metadata=dict(stencil.metadata),
    )


def reconstruct_dimer_torque_samples(
    history: CanonicalForceHistory,
    *,
    basis: Iterable[object] | object | None = None,
    allow_extrapolated_accepted: bool = False,
) -> tuple[DimerTorqueSample, ...]:
    """Return reconstructable Dimer torque samples in acquisition order."""

    result: list[DimerTorqueSample] = []
    for stencil in history.iter_stencils(family="rotation", complete_only=True):
        sample = reconstruct_dimer_torque(
            history, stencil, basis=basis,
            allow_extrapolated_accepted=allow_extrapolated_accepted,
        )
        if sample is not None:
            result.append(sample)
    result.sort(key=lambda item: (item.state_id, item.serial, item.stencil_id))
    return tuple(result)


def _rotation_sample_kind(sample: DimerTorqueSample) -> str:
    return str(sample.metadata.get("rotation_sample_kind", "")).strip().lower()


def _rotation_iteration(sample: DimerTorqueSample) -> int:
    try:
        return int(sample.metadata.get("rotation_iteration", -1))
    except (TypeError, ValueError):
        return -1


def _normalise_rotation_pair_sources(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = [item for item in value.replace(",", " ").split() if item]
    else:
        try:
            tokens = [str(item).strip() for item in value if str(item).strip()]
        except TypeError:
            tokens = [str(value).strip()]
    aliases = {
        "all": "consecutive_physical",
        "consecutive": "consecutive_physical",
        "trial": "fourier_trial",
        "fourier": "fourier_trial",
        "accepted": "accepted_rotation",
        "direct": "direct_accepted",
    }
    result: list[str] = []
    for token in tokens or ["consecutive_physical"]:
        key = aliases.get(token.lower(), token.lower())
        if key not in {
            "consecutive_physical", "fourier_trial", "accepted_rotation", "direct_accepted"
        }:
            raise ValueError(
                "rotation_pair_sources must contain consecutive_physical, "
                "fourier_trial, accepted_rotation, or direct_accepted"
            )
        if key not in result:
            result.append(key)
    return tuple(result)


def _pair_sample_candidates(
    state_samples: list[DimerTorqueSample], pair_sources: tuple[str, ...]
) -> list[tuple[DimerTorqueSample, DimerTorqueSample, str]]:
    """Return explicit rotational sample pairs before geometric transport.

    The selector exists so projection/transport experiments can use identical
    physical A/B identities. Translation-center motion is never admitted.
    """

    ordered = sorted(state_samples, key=lambda item: item.serial)
    result: list[tuple[DimerTorqueSample, DimerTorqueSample, str]] = []
    seen: set[tuple[int, int, str]] = set()

    def add(a, b, source):
        if a is None or b is None or a.serial == b.serial:
            return
        stable = (a.serial, b.serial, source)
        if stable in seen:
            return
        seen.add(stable)
        result.append((a, b, source))

    if "consecutive_physical" in pair_sources:
        physical = [sample for sample in ordered if sample.physical]
        for a, b in zip(physical[:-1], physical[1:]):
            add(a, b, "consecutive_physical")

    if "fourier_trial" in pair_sources:
        by_iteration: dict[int, dict[str, DimerTorqueSample]] = {}
        for sample in ordered:
            iteration = _rotation_iteration(sample)
            kind = _rotation_sample_kind(sample)
            if (
                not sample.physical
                or iteration < 0
                or kind not in {"fourier_a", "fourier_trial_b"}
            ):
                continue
            by_iteration.setdefault(iteration, {})[kind] = sample
        for iteration in sorted(by_iteration):
            items = by_iteration[iteration]
            if "fourier_a" in items and "fourier_trial_b" in items:
                add(items["fourier_a"], items["fourier_trial_b"], "fourier_trial")

    if "accepted_rotation" in pair_sources:
        accepted = [
            sample for sample in ordered
            if _rotation_sample_kind(sample) in {
                "fourier_a", "fourier_accepted", "direct_initial", "direct_accepted"
            }
        ]
        for a, b in zip(accepted[:-1], accepted[1:]):
            add(a, b, "accepted_rotation")

    if "direct_accepted" in pair_sources:
        accepted = [
            sample for sample in ordered
            if sample.physical
            and _rotation_sample_kind(sample) in {"direct_initial", "direct_accepted"}
        ]
        for a, b in zip(accepted[:-1], accepted[1:]):
            add(a, b, "direct_accepted")

    result.sort(key=lambda item: (item[1].serial, item[0].serial, item[2]))
    return result


def build_force_bank_rotation_pairs(
    history: CanonicalForceHistory,
    *,
    geometry: str,
    basis: Iterable[object] | object | None = None,
    max_pairs: int = 0,
    pair_sources: object = "consecutive_physical",
    accepted_force_source: str = "physical_only",
    trial_pairs_future_only: bool = False,
    current_state_id: int | None = None,
) -> ForceBankRotationBuild:
    """Build within-center rotational secants from raw Dimer stencils.

    ``pair_sources`` selects which rotational observations become secants; no
    center-to-center translation is ever admitted. Geometry/transport is then
    applied independently so matched experiments can use identical pair IDs.
    """

    started = perf_counter_ns()
    source_tokens = _normalise_rotation_pair_sources(pair_sources)
    accepted_force_source = str(accepted_force_source).strip().lower()
    if accepted_force_source not in {"physical_only", "allow_extrapolated"}:
        raise ValueError(
            "accepted_force_source must be physical_only or allow_extrapolated"
        )
    samples = reconstruct_dimer_torque_samples(
        history, basis=basis,
        allow_extrapolated_accepted=(accepted_force_source == "allow_extrapolated"),
    )
    by_state: dict[int, list[DimerTorqueSample]] = {}
    for sample in samples:
        by_state.setdefault(sample.state_id, []).append(sample)

    pairs: list[RotationSecant] = []
    sign_flips = 0
    degenerate = 0
    candidate_counts: dict[str, int] = {}
    built_counts: dict[str, int] = {}
    for state_id in sorted(by_state):
        candidates = _pair_sample_candidates(by_state[state_id], source_tokens)
        for previous, current, pair_source in candidates:
            if (
                bool(trial_pairs_future_only)
                and pair_source == "fourier_trial"
                and current_state_id is not None
                and int(state_id) == int(current_state_id)
            ):
                continue
            candidate_counts[pair_source] = candidate_counts.get(pair_source, 0) + 1
            aligned_mode, sign = sign_align_axis(previous.mode, current.mode)
            aligned_torque = sign * current.torque
            if sign < 0.0:
                sign_flips += 1
            pair = make_rotation_secant(
                previous.mode,
                previous.torque,
                aligned_mode,
                aligned_torque,
                geometry=geometry,
                state_id=state_id,
                serial=current.serial,
                source=f"force_bank_{pair_source}",
                basis=basis,
                metadata={
                    "pair_source_type": pair_source,
                    "first_stencil_id": previous.stencil_id,
                    "second_stencil_id": current.stencil_id,
                    "first_scheme": previous.scheme,
                    "second_scheme": current.scheme,
                    "first_sample_kind": _rotation_sample_kind(previous),
                    "second_sample_kind": _rotation_sample_kind(current),
                    "first_rotation_iteration": _rotation_iteration(previous),
                    "second_rotation_iteration": _rotation_iteration(current),
                    "first_physical": int(previous.physical),
                    "second_physical": int(current.physical),
                    "first_force_provenance": str(previous.metadata.get("accepted_force_provenance", "physical" if previous.physical else "derived")),
                    "second_force_provenance": str(current.metadata.get("accepted_force_provenance", "physical" if current.physical else "derived")),
                },
            )
            if pair is None:
                degenerate += 1
            else:
                pairs.append(pair)
                built_counts[pair_source] = built_counts.get(pair_source, 0) + 1
    pairs.sort(key=lambda item: (item.serial, item.state_id, item.source))
    if int(max_pairs) > 0 and len(pairs) > int(max_pairs):
        pairs = pairs[-int(max_pairs) :]
    elapsed = perf_counter_ns() - started
    contributing_states = {pair.state_id for pair in pairs}
    return ForceBankRotationBuild(
        samples=samples,
        pairs=tuple(pairs),
        metrics={
            "torque_samples": len(samples),
            "pair_sources": " ".join(source_tokens),
            "accepted_force_source": accepted_force_source,
            "derived_torque_samples": sum(not sample.physical for sample in samples),
            "trial_pairs_future_only": int(bool(trial_pairs_future_only)),
            "pair_candidates": sum(candidate_counts.values()),
            "pair_candidates_by_source": candidate_counts,
            "pairs_built": len(pairs),
            "pairs_built_by_source": built_counts,
            "pairs_degenerate": degenerate,
            "states_with_samples": sum(bool(items) for items in by_state.values()),
            "states_contributing": len(contributing_states),
            "mode_sign_flips": sign_flips,
            "build_ns": int(elapsed),
        },
    )


__all__ = [
    "DimerTorqueSample",
    "ForceBankRotationBuild",
    "build_force_bank_rotation_pairs",
    "reconstruct_dimer_torque",
    "reconstruct_physical_hvp",
    "reconstruct_dimer_torque_samples",
]
