"""Projection-independent physical-force history for minimum-mode methods.

The canonical object stores raw Cartesian observations ``(x, F_PES)`` grouped
by accepted translation-center state.  It deliberately stores no projected
force, secant vector, dense Hessian, inverse Hessian, or eigenpair.  A consumer
chooses a pair-source policy and a projection snapshot later.

A state is opened before the center force necessarily exists.  This matters for
nested Dimer/Lanczos/Davidson calls: physical probes can be associated with the
correct center even when the center calculation and the mode solve occur inside
one outer ``get_forces()`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional

import numpy as np

from saddlemill.dimertools.foundation_types import (
    ALGORITHM_PHYSICAL_ADMISSION,
    ConsumerAdmissionLedger,
    EvaluationKind,
    ForceAccounting,
    ObservationAdmissionPolicy,
    WorkPurpose,
    freeze_array,
    freeze_mapping,
    geometry_fingerprint,
)

Array = np.ndarray

PHYSICAL_EVALUATIONS = frozenset({"physical_exact", "physical_cached"})
DERIVED_EVALUATIONS = frozenset({"derived_extrapolated", "derived_other"})


def _array_n3(value: object, name: str) -> Array:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.array(array, dtype=float, copy=True)


def _normalised_mode(value: object) -> Array:
    mode = _array_n3(value, "mode")
    magnitude = float(np.linalg.norm(mode))
    if not np.isfinite(magnitude) or magnitude <= 1.0e-15:
        raise ValueError("mode norm is approximately zero")
    return mode / magnitude


def _max_abs_difference(left: Array, right: Array) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def normalise_tokens(value: object) -> tuple[str, ...]:
    """Normalize whitespace/comma separated config values, preserving order."""

    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    else:
        try:
            raw = list(value)  # type: ignore[arg-type]
        except TypeError:
            raw = [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = str(item).strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return tuple(result)


def source_family(source: object, explicit: object | None = None) -> str:
    """Return a broad source family without restricting future exact labels.

    Exact labels such as ``rotation_phase_a`` and ``rotation_trial`` map to the
    ``rotation`` family.  The exact label is still retained in each observation.
    An explicit family supplied by a caller always wins.
    """

    if explicit is not None and str(explicit).strip():
        return str(explicit).strip().lower()
    token = str(source).strip().lower()
    for family in (
        "center",
        "rotation",
        "lanczos",
        "davidson",
        "reference_hessian",
        "initialization",
        "diagnostic",
    ):
        if token == family or token.startswith(family + "_"):
            return family
    return token or "unknown"


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)


@dataclass(frozen=True)
class ForceObservation:
    """One immutable raw force-related observation at one Cartesian geometry.

    ``observation_id`` is attempt-local and stable across history serialization.
    ``geometry_id`` is an exact numeric Cartesian fingerprint, not a chemical
    geometry-equivalence criterion.  Consumer admission is intentionally
    external to this immutable record.
    """

    positions: Array
    forces: Array
    state_id: int
    serial: int
    source: str
    family: str
    role: str
    evaluation: str = "physical_exact"
    force_call_delta: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)
    direction: Optional[Array] = None
    direction_kind: str = ""
    stencil_id: str = ""
    dimer_side: int = 0
    offset: float = 0.0
    purpose: str = WorkPurpose.ALGORITHM.value
    cache_hit: bool = False
    geometry_id: str = ""
    active_dof_mask: Optional[Array] = None
    coordinate_convention: str = "unspecified"

    def __post_init__(self) -> None:
        positions = freeze_array(_array_n3(self.positions, "positions"))
        forces = freeze_array(_array_n3(self.forces, "forces"))
        if positions.shape != forces.shape:
            raise ValueError(
                "positions and forces must have the same shape; got "
                f"{positions.shape} and {forces.shape}"
            )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "forces", forces)
        object.__setattr__(self, "state_id", int(self.state_id))
        object.__setattr__(self, "serial", int(self.serial))
        source = str(self.source).strip().lower()
        family = source_family(source, self.family)
        role = str(self.role).strip().lower()
        evaluation = str(self.evaluation).strip().lower()
        purpose = WorkPurpose(str(self.purpose).strip().lower()).value
        force_call_delta = int(self.force_call_delta)
        cache_hit = bool(self.cache_hit or evaluation == EvaluationKind.PHYSICAL_CACHED.value)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "evaluation", evaluation)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "force_call_delta", force_call_delta)
        object.__setattr__(self, "cache_hit", cache_hit)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if self.direction is not None:
            direction = freeze_array(_normalised_mode(self.direction))
            if direction.shape != positions.shape:
                raise ValueError(
                    "observation direction must match positions shape; got "
                    f"{direction.shape} and {positions.shape}"
                )
            object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "direction_kind", str(self.direction_kind).strip().lower())
        object.__setattr__(self, "stencil_id", str(self.stencil_id).strip())
        object.__setattr__(self, "dimer_side", int(self.dimer_side))
        object.__setattr__(self, "offset", float(self.offset))
        object.__setattr__(self, "coordinate_convention", str(self.coordinate_convention).strip().lower() or "unspecified")
        if self.active_dof_mask is not None:
            mask = np.asarray(self.active_dof_mask, dtype=bool)
            if mask.shape != positions.shape:
                raise ValueError(
                    "active_dof_mask must match positions shape; got "
                    f"{mask.shape} and {positions.shape}"
                )
            mask = np.array(mask, dtype=bool, copy=True)
            mask.setflags(write=False)
            object.__setattr__(self, "active_dof_mask", mask)
        gid = str(self.geometry_id).strip() or geometry_fingerprint(positions)
        object.__setattr__(self, "geometry_id", gid)
        if self.dimer_side not in {-1, 0, 1}:
            raise ValueError("dimer_side must be -1, 0, or 1")
        if self.offset < 0.0 or not np.isfinite(self.offset):
            raise ValueError("observation offset must be finite and >= 0")
        if not self.source:
            raise ValueError("observation source cannot be empty")
        if not self.family:
            raise ValueError("observation family cannot be empty")
        if self.role not in {"center", "probe", "event"}:
            raise ValueError(f"unsupported observation role: {self.role!r}")
        if self.force_call_delta < 0:
            raise ValueError("force_call_delta must be >= 0")
        if self.cache_hit and self.force_call_delta != 0:
            raise ValueError("cached observation cannot represent a new PES call")

    @property
    def observation_id(self) -> str:
        return f"state{self.state_id}:obs{self.serial}"

    @property
    def is_physical(self) -> bool:
        return self.evaluation in PHYSICAL_EVALUATIONS

    @property
    def is_exact(self) -> bool:
        return self.evaluation == "physical_exact"

    @property
    def is_diagnostic(self) -> bool:
        return self.purpose == WorkPurpose.DIAGNOSTIC.value

    @property
    def nbytes(self) -> int:
        total = int(self.positions.nbytes + self.forces.nbytes)
        if self.direction is not None:
            total += int(self.direction.nbytes)
        if self.active_dof_mask is not None:
            total += int(self.active_dof_mask.nbytes)
        return total


@dataclass
class TranslationState:
    """Observations associated with one accepted translation-center geometry."""

    state_id: int
    center_positions: Array
    geometry_id: str = ""
    active_dof_mask: Optional[Array] = None
    coordinate_convention: str = "unspecified"
    center: Optional[ForceObservation] = None
    probes: list[ForceObservation] = field(default_factory=list)
    mode: Optional[Array] = None
    curvature: Optional[float] = None
    solver: str = ""
    translation_regime: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state_id = int(self.state_id)
        self.center_positions = freeze_array(_array_n3(self.center_positions, "center_positions"))
        self.geometry_id = str(self.geometry_id).strip() or geometry_fingerprint(self.center_positions)
        self.coordinate_convention = str(self.coordinate_convention).strip().lower() or "unspecified"
        if self.active_dof_mask is not None:
            mask = np.asarray(self.active_dof_mask, dtype=bool)
            if mask.shape != self.center_positions.shape:
                raise ValueError("state active_dof_mask must match center_positions shape")
            mask = np.array(mask, dtype=bool, copy=True)
            mask.setflags(write=False)
            self.active_dof_mask = mask
        self.metadata = dict(self.metadata or {})

    @property
    def state_uid(self) -> str:
        return f"state{self.state_id}:{self.geometry_id}"

    @property
    def complete(self) -> bool:
        return self.center is not None and self.center.is_physical

    def finalize(
        self,
        *,
        mode: object | None = None,
        curvature: float | None = None,
        solver: object | None = None,
        translation_regime: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if mode is not None:
            self.mode = _normalised_mode(mode)
        if curvature is not None:
            value = float(curvature)
            if not np.isfinite(value):
                raise ValueError("curvature must be finite")
            self.curvature = value
        if solver is not None:
            self.solver = str(solver).strip().lower()
        if translation_regime is not None:
            self.translation_regime = str(translation_regime).strip().lower()
        if metadata:
            self.metadata.update(dict(metadata))

    @property
    def nbytes(self) -> int:
        total = int(self.center_positions.nbytes)
        if self.center is not None:
            total += self.center.nbytes
        total += sum(item.nbytes for item in self.probes)
        if self.mode is not None:
            total += int(self.mode.nbytes)
        if self.active_dof_mask is not None:
            total += int(self.active_dof_mask.nbytes)
        return total


@dataclass(frozen=True)
class PairCandidate:
    """A deterministic pair of raw physical observations."""

    first: ForceObservation
    second: ForceObservation
    source: str
    detail_source: str
    serial: int
    state_id: int

    def displacement(self) -> Array:
        return np.array(self.second.positions - self.first.positions, copy=True)


@dataclass
class DerivativeStencil:
    """Relationship between a center observation and finite-difference probes.

    The raw force observations remain authoritative.  This object records the
    direction, scale, and observation identities required to reconstruct a
    Hessian-vector product or Dimer torque without storing the derived result.
    """

    stencil_id: str
    state_id: int
    serial: int
    source: str
    family: str
    direction_kind: str
    direction: Array
    scale: float
    purpose: str = WorkPurpose.ALGORITHM.value
    scheme: str = "one_sided"
    center_serial: Optional[int] = None
    plus_serial: Optional[int] = None
    minus_serial: Optional[int] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stencil_id = str(self.stencil_id).strip()
        self.state_id = int(self.state_id)
        self.serial = int(self.serial)
        self.source = str(self.source).strip().lower()
        self.family = source_family(self.source, self.family)
        self.purpose = WorkPurpose(str(self.purpose).strip().lower()).value
        self.direction_kind = str(self.direction_kind).strip().lower()
        self.direction = _normalised_mode(self.direction)
        self.scale = float(self.scale)
        self.scheme = str(self.scheme).strip().lower()
        self.metadata = dict(self.metadata or {})
        if not self.stencil_id:
            raise ValueError("stencil_id cannot be empty")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("stencil scale must be finite and > 0")
        if self.scheme not in {"one_sided", "centered"}:
            raise ValueError("stencil scheme must be one_sided or centered")

    @property
    def observation_serials(self) -> tuple[int, ...]:
        values = [self.center_serial, self.plus_serial, self.minus_serial]
        return tuple(int(value) for value in values if value is not None)

    @property
    def complete(self) -> bool:
        if self.center_serial is None:
            return False
        if self.scheme == "centered":
            return self.plus_serial is not None and self.minus_serial is not None
        return self.plus_serial is not None or self.minus_serial is not None

    @property
    def nbytes(self) -> int:
        return int(self.direction.nbytes)


class CanonicalForceHistory:
    """Bounded, generic, projection-independent force history."""

    SCHEMA = "saddlemill_canonical_force_history_v2"
    STATE_SCHEMA = "saddlemill_canonical_force_history_state_v1"

    def __init__(
        self,
        *,
        memory_states: int = 20,
        max_probes_per_state: int = 0,
        center_tolerance: float = 1.0e-12,
        sample_tolerance: float = 1.0e-12,
        record_sources: object = (
            "center",
            "rotation",
            "lanczos",
            "davidson",
            "reference_hessian",
        ),
    ) -> None:
        self.memory_states = int(memory_states)
        self.max_probes_per_state = int(max_probes_per_state)
        self.center_tolerance = float(center_tolerance)
        self.sample_tolerance = float(sample_tolerance)
        self.record_sources = frozenset(normalise_tokens(record_sources))
        if self.memory_states < 1:
            raise ValueError("memory_states must be >= 1")
        if self.max_probes_per_state < 0:
            raise ValueError("max_probes_per_state must be >= 0")
        if self.center_tolerance < 0.0:
            raise ValueError("center_tolerance must be >= 0")
        if self.sample_tolerance < 0.0:
            raise ValueError("sample_tolerance must be >= 0")
        if "center" not in self.record_sources and "all" not in self.record_sources:
            raise ValueError("record_sources must include center or all")

        self.states: list[TranslationState] = []
        self._next_state_id = 0
        self._next_serial = 0
        self.total_states_created = 0
        self.total_states_dropped = 0
        self.total_observations_recorded = 0
        self.total_record_events = 0
        self.total_observations_filtered = 0
        self.total_observations_deduplicated = 0
        self.total_probes_dropped = 0
        self.stencils: list[DerivativeStencil] = []
        self._stencils_by_id: dict[str, DerivativeStencil] = {}
        self.total_stencils_created = 0
        self.total_stencils_dropped = 0
        self.accounting = ForceAccounting()
        self.admission_ledger = ConsumerAdmissionLedger()

    @property
    def current(self) -> Optional[TranslationState]:
        return self.states[-1] if self.states else None

    @property
    def complete_states(self) -> list[TranslationState]:
        return [state for state in self.states if state.complete]

    def _serial(self) -> int:
        serial = self._next_serial
        self._next_serial += 1
        return serial

    def _same_geometry(self, left: Array, right: Array, tolerance: float) -> bool:
        return _max_abs_difference(left, right) <= tolerance

    def _same_current_center(self, positions: Array) -> bool:
        state = self.current
        return state is not None and self._same_geometry(
            positions, state.center_positions, self.center_tolerance
        )

    def _trim_states(self) -> None:
        overflow = len(self.states) - self.memory_states
        if overflow <= 0:
            return
        dropped_ids = {state.state_id for state in self.states[:overflow]}
        del self.states[:overflow]
        self.total_states_dropped += overflow
        kept = [item for item in self.stencils if item.state_id not in dropped_ids]
        self.total_stencils_dropped += len(self.stencils) - len(kept)
        self.stencils = kept
        self._stencils_by_id = {item.stencil_id: item for item in kept}

    def observation_by_serial(self, serial: int) -> Optional[ForceObservation]:
        target = int(serial)
        for observation in self.iter_observations():
            if observation.serial == target:
                return observation
        return None

    def _attach_center_to_state_stencils(self, state: TranslationState) -> None:
        if state.center is None:
            return
        for stencil in self.stencils:
            if stencil.state_id == state.state_id:
                stencil.center_serial = state.center.serial

    def _validate_probe_stencil_identity(
        self,
        *,
        stencil_id: str,
        state_id: int,
        purpose: str,
    ) -> None:
        """Fail before storage mutation if a named stencil would cross identities."""
        token = str(stencil_id).strip()
        if not token:
            return
        stencil = self._stencils_by_id.get(token)
        if stencil is None:
            return
        if stencil.state_id != int(state_id):
            raise ValueError("one stencil_id cannot span translation states")
        if stencil.purpose != str(purpose).strip().lower():
            raise ValueError("one stencil_id cannot mix algorithm and diagnostic purpose")

    def _register_probe_stencil(self, observation: ForceObservation) -> None:
        if observation.direction is None or not observation.stencil_id:
            return
        state = next(
            (item for item in self.states if item.state_id == observation.state_id),
            None,
        )
        if state is None:
            return
        stencil = self._stencils_by_id.get(observation.stencil_id)
        if stencil is None:
            scheme = str(observation.metadata.get("stencil_scheme", "one_sided")).strip().lower()
            stencil = DerivativeStencil(
                stencil_id=observation.stencil_id,
                state_id=observation.state_id,
                # A new derivative evaluation can reuse an older force
                # observation. Give the stencil its own acquisition stamp;
                # the force's identity must not reorder the new secant.
                serial=self._serial(),
                source=observation.source,
                family=observation.family,
                direction_kind=observation.direction_kind or observation.family,
                purpose=observation.purpose,
                direction=observation.direction,
                scale=observation.offset,
                scheme=scheme,
                center_serial=None if state.center is None else state.center.serial,
                metadata=dict(observation.metadata),
            )
            self.stencils.append(stencil)
            self._stencils_by_id[stencil.stencil_id] = stencil
            self.total_stencils_created += 1
        else:
            if stencil.state_id != observation.state_id:
                raise ValueError("one stencil_id cannot span translation states")
            if stencil.purpose != observation.purpose:
                raise ValueError("one stencil_id cannot mix algorithm and diagnostic purpose")
            if abs(float(np.vdot(stencil.direction.ravel(), observation.direction.ravel()).real)) < 1.0 - 1.0e-8:
                raise ValueError("stencil observations have inconsistent directions")
            # Keep this stencil's acquisition stamp when the other endpoint
            # arrives, including an endpoint reused from an earlier sample.
            stencil.metadata.update(dict(observation.metadata))
            if state.center is not None:
                stencil.center_serial = state.center.serial
        side = observation.dimer_side
        if side == 0:
            side = 1 if float(np.vdot(stencil.direction.ravel(), observation.direction.ravel()).real) >= 0.0 else -1
        if side > 0:
            stencil.plus_serial = observation.serial
        else:
            stencil.minus_serial = observation.serial
        if stencil.plus_serial is not None and stencil.minus_serial is not None:
            stencil.scheme = "centered"

    def iter_stencils(
        self,
        *,
        family: object | None = None,
        complete_only: bool = False,
    ) -> Iterator[DerivativeStencil]:
        family_token = None if family is None else str(family).strip().lower()
        for stencil in sorted(self.stencils, key=lambda item: (item.serial, item.stencil_id)):
            if family_token is not None and stencil.family != family_token:
                continue
            if complete_only and not stencil.complete:
                continue
            yield stencil

    def observes_source(self, source: object, family: object | None = None) -> bool:
        source_token = str(source).strip().lower()
        family_token = source_family(source_token, family)
        return bool(
            "all" in self.record_sources
            or source_token in self.record_sources
            or family_token in self.record_sources
        )

    def begin_state(
        self,
        center_positions: object,
        *,
        metadata: Mapping[str, object] | None = None,
        active_dof_mask: object | None = None,
        coordinate_convention: str = "unspecified",
    ) -> TranslationState:
        """Open or reuse the state associated with ``center_positions``.

        ``active_dof_mask`` is a Cartesian ``(N, 3)`` mask.  It is identity
        metadata for the accepted center state, not a request to invent or
        modify fixed-coordinate Hessian entries.
        """

        positions = _array_n3(center_positions, "center_positions")
        mask = None if active_dof_mask is None else np.asarray(active_dof_mask, dtype=bool)
        if mask is not None and mask.shape != positions.shape:
            raise ValueError("active_dof_mask must match center_positions shape")
        convention = str(coordinate_convention).strip().lower() or "unspecified"
        if self._same_current_center(positions):
            state = self.current
            assert state is not None
            if mask is not None:
                if state.active_dof_mask is None or not np.array_equal(mask, state.active_dof_mask):
                    raise ValueError("same center cannot be reopened with a different active_dof_mask")
            if convention != "unspecified" and state.coordinate_convention not in {"unspecified", convention}:
                raise ValueError("same center cannot be reopened with a different coordinate convention")
            if state.coordinate_convention == "unspecified" and convention != "unspecified":
                state.coordinate_convention = convention
            if metadata:
                state.metadata.update(dict(metadata))
            return state
        state = TranslationState(
            state_id=self._next_state_id,
            center_positions=positions,
            active_dof_mask=mask,
            coordinate_convention=convention,
            metadata=dict(metadata or {}),
        )
        self._next_state_id += 1
        self.total_states_created += 1
        self.states.append(state)
        self._trim_states()
        return state

    @staticmethod
    def _event_tokens(evaluation: object, purpose: object, cache_hit: bool | None) -> tuple[str, str, bool]:
        evaluation_token = EvaluationKind(str(evaluation).strip().lower()).value
        purpose_token = WorkPurpose(str(purpose).strip().lower()).value
        inferred_hit = evaluation_token == EvaluationKind.PHYSICAL_CACHED.value
        hit = inferred_hit if cache_hit is None else bool(cache_hit)
        if hit and evaluation_token != EvaluationKind.PHYSICAL_CACHED.value:
            raise ValueError("cache_hit=True requires evaluation=physical_cached")
        return evaluation_token, purpose_token, hit

    def observe_center(
        self,
        positions: object,
        forces: object,
        *,
        evaluation: str = "physical_exact",
        force_call_delta: int = 0,
        metadata: Mapping[str, object] | None = None,
        purpose: str = WorkPurpose.ALGORITHM.value,
        cache_hit: bool | None = None,
        active_dof_mask: object | None = None,
        coordinate_convention: str = "unspecified",
    ) -> TranslationState:
        positions_array = _array_n3(positions, "positions")
        forces_array = _array_n3(forces, "forces")
        if positions_array.shape != forces_array.shape:
            raise ValueError("positions and forces shapes differ")
        evaluation_token, purpose_token, hit = self._event_tokens(evaluation, purpose, cache_hit)
        delta = max(0, int(force_call_delta))
        state = self.begin_state(
            positions_array,
            active_dof_mask=active_dof_mask,
            coordinate_convention=coordinate_convention,
        )
        self.total_record_events += 1
        self.accounting.record_force_event(
            purpose=purpose_token, pes_call_delta=delta, cache_hit=hit
        )
        if state.center is None:
            state.center = ForceObservation(
                positions=positions_array,
                forces=forces_array,
                state_id=state.state_id,
                serial=self._serial(),
                source="center",
                family="center",
                role="center",
                evaluation=evaluation_token,
                force_call_delta=delta,
                metadata=dict(metadata or {}),
                purpose=purpose_token,
                cache_hit=hit,
                active_dof_mask=state.active_dof_mask,
                coordinate_convention=state.coordinate_convention,
            )
            self.total_observations_recorded += 1
            self._attach_center_to_state_stencils(state)
            return state

        center = state.center
        if center.purpose != purpose_token:
            raise ValueError(
                "one accepted center observation cannot merge algorithm and diagnostic purpose; "
                "record diagnostic center work as a probe/event with explicit provenance"
            )
        incoming_exact = evaluation_token == EvaluationKind.PHYSICAL_EXACT.value
        chosen_forces = forces_array if (incoming_exact or not center.is_exact) else center.forces
        chosen_evaluation = evaluation_token if (incoming_exact or not center.is_exact) else center.evaluation
        merged_metadata = dict(center.metadata)
        merged_metadata["record_events"] = int(merged_metadata.get("record_events", 1)) + 1
        if metadata:
            merged_metadata.update(dict(metadata))
        state.center = replace(
            center,
            forces=chosen_forces,
            evaluation=chosen_evaluation,
            force_call_delta=center.force_call_delta + delta,
            cache_hit=(chosen_evaluation == EvaluationKind.PHYSICAL_CACHED.value),
            metadata=merged_metadata,
        )
        self.total_observations_deduplicated += 1
        self._attach_center_to_state_stencils(state)
        return state

    def _matching_probe(
        self,
        state: TranslationState,
        positions: Array,
        *,
        source: str,
        family: str,
        purpose: str,
    ) -> Optional[ForceObservation]:
        for probe in reversed(state.probes):
            if probe.source != source or probe.family != family or probe.purpose != purpose:
                continue
            if self._same_geometry(probe.positions, positions, self.sample_tolerance):
                return probe
        return None

    def observe_probe(
        self,
        positions: object,
        forces: object,
        *,
        source: str,
        family: str | None = None,
        evaluation: str = "physical_exact",
        force_call_delta: int = 0,
        metadata: Mapping[str, object] | None = None,
        direction: object | None = None,
        direction_kind: str = "",
        stencil_id: str = "",
        dimer_side: int = 0,
        offset: float = 0.0,
        purpose: str = WorkPurpose.ALGORITHM.value,
        cache_hit: bool | None = None,
    ) -> Optional[ForceObservation]:
        source_token = str(source).strip().lower()
        family_token = source_family(source_token, family)
        evaluation_token, purpose_token, hit = self._event_tokens(evaluation, purpose, cache_hit)
        delta = max(0, int(force_call_delta))
        self.accounting.record_force_event(
            purpose=purpose_token, pes_call_delta=delta, cache_hit=hit
        )
        if not self.observes_source(source_token, family_token):
            self.total_observations_filtered += 1
            return None
        state = self.current
        if state is None:
            raise RuntimeError("cannot record a probe before begin_state()")
        positions_array = _array_n3(positions, "positions")
        forces_array = _array_n3(forces, "forces")
        if positions_array.shape != state.center_positions.shape:
            raise ValueError(
                "probe atom count differs from its center state: "
                f"{positions_array.shape} versus {state.center_positions.shape}"
            )
        self._validate_probe_stencil_identity(
            stencil_id=stencil_id,
            state_id=state.state_id,
            purpose=purpose_token,
        )
        self.total_record_events += 1
        existing = self._matching_probe(
            state, positions_array, source=source_token, family=family_token, purpose=purpose_token
        )
        if existing is not None:
            incoming_exact = evaluation_token == EvaluationKind.PHYSICAL_EXACT.value
            chosen_forces = forces_array if (incoming_exact or not existing.is_exact) else existing.forces
            chosen_evaluation = evaluation_token if (incoming_exact or not existing.is_exact) else existing.evaluation
            merged_metadata = dict(existing.metadata)
            merged_metadata["record_events"] = int(merged_metadata.get("record_events", 1)) + 1
            if metadata:
                merged_metadata.update(dict(metadata))
            replacement = replace(
                existing,
                forces=chosen_forces,
                evaluation=chosen_evaluation,
                force_call_delta=existing.force_call_delta + delta,
                cache_hit=(chosen_evaluation == EvaluationKind.PHYSICAL_CACHED.value),
                metadata=merged_metadata,
                direction=existing.direction if direction is None else _normalised_mode(direction),
                direction_kind=existing.direction_kind if not direction_kind else str(direction_kind).strip().lower(),
                stencil_id=existing.stencil_id if not stencil_id else str(stencil_id).strip(),
                dimer_side=existing.dimer_side if int(dimer_side) not in {-1, 1} else int(dimer_side),
                offset=existing.offset if float(offset) <= 0.0 else float(offset),
            )
            probe_index = next(
                index for index, item in enumerate(state.probes)
                if item.serial == existing.serial
            )
            state.probes[probe_index] = replacement
            self.total_observations_deduplicated += 1
            self._register_probe_stencil(replacement)
            return replacement

        observation = ForceObservation(
            positions=positions_array,
            forces=forces_array,
            state_id=state.state_id,
            serial=self._serial(),
            source=source_token,
            family=family_token,
            role="probe",
            evaluation=evaluation_token,
            force_call_delta=delta,
            metadata=dict(metadata or {}),
            direction=None if direction is None else _normalised_mode(direction),
            direction_kind=str(direction_kind).strip().lower(),
            stencil_id=str(stencil_id).strip(),
            dimer_side=int(dimer_side),
            offset=float(offset),
            purpose=purpose_token,
            cache_hit=hit,
            active_dof_mask=state.active_dof_mask,
            coordinate_convention=state.coordinate_convention,
        )
        state.probes.append(observation)
        self.total_observations_recorded += 1
        self._register_probe_stencil(observation)
        if self.max_probes_per_state > 0 and len(state.probes) > self.max_probes_per_state:
            overflow = len(state.probes) - self.max_probes_per_state
            dropped = state.probes[:overflow]
            dropped_serials = {item.serial for item in dropped}
            del state.probes[:overflow]
            self.total_probes_dropped += overflow
            kept_stencils = [
                item
                for item in self.stencils
                if not dropped_serials.intersection(item.observation_serials)
            ]
            self.total_stencils_dropped += len(self.stencils) - len(kept_stencils)
            self.stencils = kept_stencils
            self._stencils_by_id = {item.stencil_id: item for item in kept_stencils}
        return observation

    def finalize_current(
        self,
        *,
        mode: object | None = None,
        curvature: float | None = None,
        solver: object | None = None,
        translation_regime: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        state = self.current
        if state is None:
            return
        state.finalize(
            mode=mode,
            curvature=curvature,
            solver=solver,
            translation_regime=translation_regime,
            metadata=metadata,
        )

    @staticmethod
    def _parse_pair_sources(pair_sources: object) -> tuple[bool, set[str], set[str], bool]:
        include_center_center = False
        families: set[str] = set()
        exact_sources: set[str] = set()
        include_all_probes = False
        for token in normalise_tokens(pair_sources):
            if token == "center_center":
                include_center_center = True
            elif token in {"center_probe_all", "center_all_probes"}:
                include_all_probes = True
            elif token.startswith("center_family:"):
                family = token.split(":", 1)[1].strip()
                if not family:
                    raise ValueError(f"empty family in pair source {token!r}")
                families.add(family)
            elif token.startswith("center_source:"):
                source = token.split(":", 1)[1].strip()
                if not source:
                    raise ValueError(f"empty source in pair source {token!r}")
                exact_sources.add(source)
            elif token.startswith("center_") and len(token) > len("center_"):
                # Backward-compatible concise spelling: center_rotation,
                # center_lanczos, ... means a source family.
                families.add(token[len("center_") :])
            else:
                raise ValueError(
                    "pair_sources entries must be center_center, center_<family>, "
                    "center_family:<family>, center_source:<source>, or "
                    f"center_probe_all; got {token!r}"
                )
        return include_center_center, families, exact_sources, include_all_probes

    @staticmethod
    def _observation_admitted(
        observation: ForceObservation,
        policy: ObservationAdmissionPolicy,
        *,
        promoted: bool = False,
    ) -> bool:
        return policy.accepts(
            purpose=observation.purpose,
            evaluation=observation.evaluation,
            source=observation.source,
            family=observation.family,
            promoted=promoted,
        )

    def pair_candidates(
        self,
        pair_sources: object,
        *,
        physical_only: bool = True,
        admission_policy: ObservationAdmissionPolicy | None = None,
        promote_diagnostic: bool = False,
    ) -> list[PairCandidate]:
        """Select deterministic raw endpoint pairs; no projection occurs here.

        Storage and consumer admission are separate.  By default only
        algorithm-purpose physical exact/cached observations are admitted.
        Diagnostic observations require both a policy that explicitly permits
        promotion and ``promote_diagnostic=True``.
        """

        if admission_policy is None:
            if physical_only:
                admission_policy = ALGORITHM_PHYSICAL_ADMISSION
            else:
                admission_policy = ObservationAdmissionPolicy(
                    name="algorithm_any_recorded",
                    allowed_purposes=(WorkPurpose.ALGORITHM.value,),
                    allowed_evaluations=(
                        EvaluationKind.PHYSICAL_EXACT.value,
                        EvaluationKind.PHYSICAL_CACHED.value,
                    ),
                    allow_derived=True,
                )
        include_cc, families, exact_sources, include_all = self._parse_pair_sources(
            pair_sources
        )
        states = self.complete_states
        result: list[PairCandidate] = []
        for index, state in enumerate(states):
            center = state.center
            assert center is not None
            if not self._observation_admitted(
                center, admission_policy, promoted=promote_diagnostic
            ):
                continue
            if include_cc and index + 1 < len(states):
                second = states[index + 1].center
                assert second is not None
                if self._observation_admitted(
                    second, admission_policy, promoted=promote_diagnostic
                ):
                    result.append(
                        PairCandidate(
                            first=center,
                            second=second,
                            source="center_center",
                            detail_source="center_center",
                            serial=second.serial,
                            state_id=state.state_id,
                        )
                    )
            for probe in state.probes:
                if not self._observation_admitted(
                    probe, admission_policy, promoted=promote_diagnostic
                ):
                    continue
                if not (
                    include_all
                    or probe.family in families
                    or probe.source in exact_sources
                ):
                    continue
                result.append(
                    PairCandidate(
                        first=center,
                        second=probe,
                        source=f"center_{probe.family}",
                        detail_source=f"center_source:{probe.source}",
                        serial=probe.serial,
                        state_id=state.state_id,
                    )
                )
        result.sort(key=lambda item: (item.serial, item.source, item.detail_source))
        return result

    def admit_pair_candidates(
        self,
        consumer_id: str,
        pair_sources: object,
        *,
        physical_only: bool = True,
        admission_policy: ObservationAdmissionPolicy | None = None,
        promote_diagnostic: bool = False,
        promotion_policy: str = "",
        block_id: str = "",
    ) -> list[PairCandidate]:
        """Return only pairs not previously admitted to ``consumer_id``.

        This is the idempotent consumer path for model training/history
        ingestion.  Replaying a cached observation or the same probe block does
        not train the same consumer twice.
        """

        pairs = self.pair_candidates(
            pair_sources,
            physical_only=physical_only,
            admission_policy=admission_policy,
            promote_diagnostic=promote_diagnostic,
        )
        admitted: list[PairCandidate] = []
        for pair in pairs:
            diagnostic = pair.first.is_diagnostic or pair.second.is_diagnostic
            if diagnostic and not promote_diagnostic:
                continue
            if diagnostic and not str(promotion_policy).strip():
                raise ValueError(
                    "diagnostic-to-algorithm pair admission requires promotion_policy"
                )
            subject = f"pair:{pair.first.observation_id}->{pair.second.observation_id}:{pair.source}"
            outcome = self.admission_ledger.admit(
                consumer_id=consumer_id,
                subject_id=subject,
                observation_ids=(pair.first.observation_id, pair.second.observation_id),
                block_id=block_id,
                purpose=WorkPurpose.ALGORITHM,
                promoted_from_diagnostic=diagnostic,
                promotion_policy=promotion_policy if diagnostic else "",
                metadata={
                    "pair_source": pair.source,
                    "detail_source": pair.detail_source,
                    "state_id": pair.state_id,
                },
            )
            if outcome.admitted:
                admitted.append(pair)
                if diagnostic:
                    self.accounting.diagnostic_promotions += 1
        return admitted

    def stencil_endpoint_ids(self, stencil: DerivativeStencil) -> dict[str, str | None]:
        def oid(serial: int | None) -> str | None:
            if serial is None:
                return None
            item = self.observation_by_serial(serial)
            return None if item is None else item.observation_id
        return {
            "center": oid(stencil.center_serial),
            "plus": oid(stencil.plus_serial),
            "minus": oid(stencil.minus_serial),
        }

    def iter_observations(self) -> Iterator[ForceObservation]:
        for state in self.states:
            if state.center is not None:
                yield state.center
            yield from state.probes

    def observation_counts(self, *, by: str = "family") -> dict[str, int]:
        if by not in {"family", "source", "evaluation"}:
            raise ValueError("by must be family, source, or evaluation")
        counts: dict[str, int] = {}
        for observation in self.iter_observations():
            key = str(getattr(observation, by))
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def nbytes(self) -> int:
        return int(
            sum(state.nbytes for state in self.states)
            + sum(stencil.nbytes for stencil in self.stencils)
        )

    def to_state_dict(self) -> dict[str, object]:
        """Serialize bounded force-bank state for exact attempt resume.

        Numeric arrays are emitted as JSON-compatible lists.  Observation and
        stencil serials/IDs, admission state, accounting, and next counters are
        retained so replay cannot silently duplicate consumer training.
        """

        return {
            "schema": self.STATE_SCHEMA,
            "settings": {
                "memory_states": self.memory_states,
                "max_probes_per_state": self.max_probes_per_state,
                "center_tolerance": self.center_tolerance,
                "sample_tolerance": self.sample_tolerance,
                "record_sources": sorted(self.record_sources),
            },
            "counters": {
                "next_state_id": self._next_state_id,
                "next_serial": self._next_serial,
                "total_states_created": self.total_states_created,
                "total_states_dropped": self.total_states_dropped,
                "total_observations_recorded": self.total_observations_recorded,
                "total_record_events": self.total_record_events,
                "total_observations_filtered": self.total_observations_filtered,
                "total_observations_deduplicated": self.total_observations_deduplicated,
                "total_probes_dropped": self.total_probes_dropped,
                "total_stencils_created": self.total_stencils_created,
                "total_stencils_dropped": self.total_stencils_dropped,
            },
            "accounting": self.accounting.to_state_dict(),
            "admission_ledger": self.admission_ledger.to_state_dict(),
            "states": [
                {
                    "state_id": state.state_id,
                    "center_positions": state.center_positions.tolist(),
                    "geometry_id": state.geometry_id,
                    "active_dof_mask": None if state.active_dof_mask is None else state.active_dof_mask.astype(int).tolist(),
                    "coordinate_convention": state.coordinate_convention,
                    "center_serial": None if state.center is None else state.center.serial,
                    "probe_serials": [item.serial for item in state.probes],
                    "mode": None if state.mode is None else state.mode.tolist(),
                    "curvature": state.curvature,
                    "solver": state.solver,
                    "translation_regime": state.translation_regime,
                    "metadata": _json_safe(state.metadata),
                }
                for state in self.states
            ],
            "observations": [
                {
                    "positions": item.positions.tolist(),
                    "forces": item.forces.tolist(),
                    "state_id": item.state_id,
                    "serial": item.serial,
                    "source": item.source,
                    "family": item.family,
                    "role": item.role,
                    "evaluation": item.evaluation,
                    "purpose": item.purpose,
                    "cache_hit": item.cache_hit,
                    "geometry_id": item.geometry_id,
                    "coordinate_convention": item.coordinate_convention,
                    "force_call_delta": item.force_call_delta,
                    "metadata": _json_safe(item.metadata),
                    "direction": None if item.direction is None else item.direction.tolist(),
                    "direction_kind": item.direction_kind,
                    "stencil_id": item.stencil_id,
                    "dimer_side": item.dimer_side,
                    "offset": item.offset,
                    "purpose": item.purpose,
                    "cache_hit": item.cache_hit,
                    "geometry_id": item.geometry_id,
                    "active_dof_mask": None if item.active_dof_mask is None else item.active_dof_mask.astype(int).tolist(),
                    "coordinate_convention": item.coordinate_convention,
                }
                for item in self.iter_observations()
            ],
            "stencils": [
                {
                    "stencil_id": item.stencil_id,
                    "state_id": item.state_id,
                    "serial": item.serial,
                    "source": item.source,
                    "family": item.family,
                    "purpose": item.purpose,
                    "direction_kind": item.direction_kind,
                    "direction": item.direction.tolist(),
                    "scale": item.scale,
                    "scheme": item.scheme,
                    "center_serial": item.center_serial,
                    "plus_serial": item.plus_serial,
                    "minus_serial": item.minus_serial,
                    "metadata": _json_safe(item.metadata),
                }
                for item in self.stencils
            ],
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "CanonicalForceHistory":
        if payload.get("schema") != cls.STATE_SCHEMA:
            raise ValueError("unsupported CanonicalForceHistory state schema")
        settings = dict(payload.get("settings", {}))
        result = cls(
            memory_states=int(settings.get("memory_states", 20)),
            max_probes_per_state=int(settings.get("max_probes_per_state", 0)),
            center_tolerance=float(settings.get("center_tolerance", 1.0e-12)),
            sample_tolerance=float(settings.get("sample_tolerance", 1.0e-12)),
            record_sources=tuple(settings.get("record_sources", ("center",))),
        )
        observations: dict[int, ForceObservation] = {}
        for row in payload.get("observations", []):
            row = dict(row)
            observation = ForceObservation(
                positions=np.asarray(row["positions"], dtype=float),
                forces=np.asarray(row["forces"], dtype=float),
                state_id=int(row["state_id"]),
                serial=int(row["serial"]),
                source=str(row["source"]),
                family=str(row["family"]),
                role=str(row["role"]),
                evaluation=str(row.get("evaluation", "physical_exact")),
                force_call_delta=int(row.get("force_call_delta", 0)),
                metadata=dict(row.get("metadata", {})),
                direction=None if row.get("direction") is None else np.asarray(row["direction"], dtype=float),
                direction_kind=str(row.get("direction_kind", "")),
                stencil_id=str(row.get("stencil_id", "")),
                dimer_side=int(row.get("dimer_side", 0)),
                offset=float(row.get("offset", 0.0)),
                purpose=str(row.get("purpose", WorkPurpose.ALGORITHM.value)),
                cache_hit=bool(row.get("cache_hit", False)),
                geometry_id=str(row.get("geometry_id", "")),
                active_dof_mask=None if row.get("active_dof_mask") is None else np.asarray(row["active_dof_mask"], dtype=bool),
                coordinate_convention=str(row.get("coordinate_convention", "unspecified")),
            )
            if observation.serial in observations:
                raise ValueError("duplicate observation serial in serialized state")
            observations[observation.serial] = observation
        for row in payload.get("states", []):
            row = dict(row)
            state = TranslationState(
                state_id=int(row["state_id"]),
                center_positions=np.asarray(row["center_positions"], dtype=float),
                geometry_id=str(row.get("geometry_id", "")),
                active_dof_mask=None if row.get("active_dof_mask") is None else np.asarray(row["active_dof_mask"], dtype=bool),
                coordinate_convention=str(row.get("coordinate_convention", "unspecified")),
                center=None if row.get("center_serial") is None else observations[int(row["center_serial"])],
                probes=[observations[int(serial)] for serial in row.get("probe_serials", [])],
                mode=None if row.get("mode") is None else np.asarray(row["mode"], dtype=float),
                curvature=None if row.get("curvature") is None else float(row["curvature"]),
                solver=str(row.get("solver", "")),
                translation_regime=str(row.get("translation_regime", "")),
                metadata=dict(row.get("metadata", {})),
            )
            result.states.append(state)
        for row in payload.get("stencils", []):
            row = dict(row)
            stencil = DerivativeStencil(
                stencil_id=str(row["stencil_id"]),
                state_id=int(row["state_id"]),
                serial=int(row["serial"]),
                source=str(row["source"]),
                family=str(row["family"]),
                direction_kind=str(row.get("direction_kind", "")),
                direction=np.asarray(row["direction"], dtype=float),
                purpose=str(row.get("purpose", WorkPurpose.ALGORITHM.value)),
                scale=float(row["scale"]),
                scheme=str(row.get("scheme", "one_sided")),
                center_serial=None if row.get("center_serial") is None else int(row["center_serial"]),
                plus_serial=None if row.get("plus_serial") is None else int(row["plus_serial"]),
                minus_serial=None if row.get("minus_serial") is None else int(row["minus_serial"]),
                metadata=dict(row.get("metadata", {})),
            )
            if stencil.stencil_id in result._stencils_by_id:
                raise ValueError("duplicate stencil_id in serialized state")
            result.stencils.append(stencil)
            result._stencils_by_id[stencil.stencil_id] = stencil
        counters = dict(payload.get("counters", {}))
        for name in (
            "_next_state_id", "_next_serial", "total_states_created",
            "total_states_dropped", "total_observations_recorded",
            "total_record_events", "total_observations_filtered",
            "total_observations_deduplicated", "total_probes_dropped",
            "total_stencils_created", "total_stencils_dropped",
        ):
            key = name[1:] if name.startswith("_") else name
            setattr(result, name, int(counters.get(key, getattr(result, name))))
        result.accounting = ForceAccounting.from_state_dict(dict(payload.get("accounting", {})))
        result.admission_ledger = ConsumerAdmissionLedger.from_state_dict(dict(payload.get("admission_ledger", {})))
        if result.states and result._next_state_id <= max(item.state_id for item in result.states):
            raise ValueError("next_state_id does not exceed retained state IDs")
        all_serials = list(observations) + [item.serial for item in result.stencils]
        if all_serials and result._next_serial <= max(all_serials):
            raise ValueError("next_serial does not exceed retained observation/stencil serials")
        return result

    def save_state_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
        try:
            tmp.write_text(json.dumps(self.to_state_dict(), sort_keys=True, separators=(",", ":")))
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()
        return target

    @classmethod
    def load_state_json(cls, path: str | Path) -> "CanonicalForceHistory":
        return cls.from_state_dict(json.loads(Path(path).read_text()))

    def summary(self) -> dict[str, object]:
        observations = list(self.iter_observations())
        return {
            "schema": self.SCHEMA,
            "states_retained": len(self.states),
            "complete_states_retained": len(self.complete_states),
            "states_created_total": self.total_states_created,
            "states_dropped_total": self.total_states_dropped,
            "observations_retained": len(observations),
            "observations_recorded_total": self.total_observations_recorded,
            "record_events_total": self.total_record_events,
            "observations_filtered_total": self.total_observations_filtered,
            "observations_deduplicated_total": self.total_observations_deduplicated,
            "probes_dropped_total": self.total_probes_dropped,
            "stencils_retained": len(self.stencils),
            "complete_stencils_retained": sum(item.complete for item in self.stencils),
            "stencils_created_total": self.total_stencils_created,
            "stencils_dropped_total": self.total_stencils_dropped,
            "stencil_counts_by_family": {
                family: sum(item.family == family for item in self.stencils)
                for family in sorted({item.family for item in self.stencils})
            },
            "physical_exact_retained": sum(item.is_exact for item in observations),
            "physical_cached_retained": sum(
                item.evaluation == "physical_cached" for item in observations
            ),
            "derived_retained": sum(not item.is_physical for item in observations),
            "force_calls_represented": sum(
                item.force_call_delta for item in observations
            ),
            "observation_counts_by_family": self.observation_counts(by="family"),
            "observation_counts_by_source": self.observation_counts(by="source"),
            "observation_counts_by_purpose": {
                purpose: sum(item.purpose == purpose for item in observations)
                for purpose in sorted({item.purpose for item in observations})
            },
            "force_accounting": self.accounting.to_state_dict(),
            "admission_records": self.admission_ledger.record_count,
            "history_bytes": self.nbytes,
        }

    def dump_npz(self, path: str | Path) -> Path:
        """Atomically write an optional diagnostic dump without pickle arrays."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        observations = list(self.iter_observations())
        atom_count = 0
        if self.states:
            atom_count = int(self.states[0].center_positions.shape[0])
        if observations:
            positions = np.stack([item.positions for item in observations], axis=0)
            forces = np.stack([item.forces for item in observations], axis=0)
        else:
            positions = np.empty((0, atom_count, 3), dtype=float)
            forces = np.empty((0, atom_count, 3), dtype=float)
        directions = np.full((len(observations), atom_count, 3), np.nan, dtype=float)
        for index, observation in enumerate(observations):
            if observation.direction is not None:
                directions[index] = observation.direction
        modes = np.full((len(self.states), atom_count, 3), np.nan, dtype=float)
        for index, state in enumerate(self.states):
            if state.mode is not None:
                modes[index] = state.mode

        metadata = {
            "schema": self.SCHEMA,
            "history_summary": self.summary(),
            "record_sources": sorted(self.record_sources),
            "observations": [
                {
                    "state_id": item.state_id,
                    "serial": item.serial,
                    "source": item.source,
                    "family": item.family,
                    "role": item.role,
                    "evaluation": item.evaluation,
                    "purpose": item.purpose,
                    "cache_hit": item.cache_hit,
                    "geometry_id": item.geometry_id,
                    "coordinate_convention": item.coordinate_convention,
                    "force_call_delta": item.force_call_delta,
                    "observation_id": item.observation_id,
                    "direction_kind": item.direction_kind,
                    "stencil_id": item.stencil_id,
                    "dimer_side": item.dimer_side,
                    "offset": item.offset,
                    "metadata": _json_safe(item.metadata),
                }
                for item in observations
            ],
            "stencils": [
                {
                    "stencil_id": item.stencil_id,
                    "state_id": item.state_id,
                    "serial": item.serial,
                    "source": item.source,
                    "family": item.family,
                    "purpose": item.purpose,
                    "direction_kind": item.direction_kind,
                    "scale": item.scale,
                    "scheme": item.scheme,
                    "center_serial": item.center_serial,
                    "plus_serial": item.plus_serial,
                    "minus_serial": item.minus_serial,
                    "metadata": _json_safe(item.metadata),
                }
                for item in self.stencils
            ],
            "states": [
                {
                    "state_id": state.state_id,
                    "state_uid": state.state_uid,
                    "geometry_id": state.geometry_id,
                    "coordinate_convention": state.coordinate_convention,
                    "complete": state.complete,
                    "curvature": state.curvature,
                    "solver": state.solver,
                    "translation_regime": state.translation_regime,
                    "metadata": _json_safe(state.metadata),
                }
                for state in self.states
            ],
        }
        tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
        try:
            with tmp.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    metadata_json=np.asarray(
                        json.dumps(_json_safe(metadata), sort_keys=True)
                    ),
                    positions=positions,
                    forces=forces,
                    directions=directions,
                    modes=modes,
                )
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()
        return target


__all__ = [
    "CanonicalForceHistory",
    "DERIVED_EVALUATIONS",
    "DerivativeStencil",
    "ForceObservation",
    "PHYSICAL_EVALUATIONS",
    "PairCandidate",
    "TranslationState",
    "normalise_tokens",
    "source_family",
]
