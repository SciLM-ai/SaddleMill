"""Pure bounded mode-solve scheduler and resume state (mode-schedule).

The scheduler operates only on already-available physical/model observations.  It
has no calculator/evaluator dependency and therefore cannot spend a PES call to
decide whether to save one.  mode-predictor owns predictor algorithms; this module exposes a
post-decision hook that can consume a ``ModePredictionResult`` supplied by mode-predictor.
shared-runtime owns runtime/config/factory/checkpoint wiring.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from typing import Mapping

import numpy as np

from saddlemill.dimertools.force_policy import (
    ParallelForceDampingConfig,
    select_damping_for_sequence,
)
from saddlemill.dimertools.foundation_types import (
    ActiveCoordinateSpace,
    ModePredictionResult,
    ModeScheduleDecision,
    ModeScheduleState,
    ParallelForceMetricPolicy,
    freeze_array,
    freeze_mapping,
)

Array = np.ndarray
RUNTIME_SCHEMA = "saddlemill_bounded_mode_schedule_runtime_v1"
ENTRY_GATES = {"none", "dimer_initial_torque"}
REFRESH_POLICIES = {"bounded", "physical_model_loss"}


@dataclass(frozen=True)
class PhysicalModelSignal:
    """Already-available physical-Hessian/model curvature/inertia information."""

    origin: str
    age: int
    valid: bool
    negative_mode_count: int | None = None
    lowest_curvature: float | None = None
    observation_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        origin = str(self.origin).strip().lower()
        if not origin:
            raise ValueError("physical-model signal origin cannot be empty")
        age = int(self.age)
        if age < 0:
            raise ValueError("physical-model signal age must be >= 0")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "age", age)
        if self.negative_mode_count is not None:
            count = int(self.negative_mode_count)
            if count < 0:
                raise ValueError("negative_mode_count must be >= 0")
            object.__setattr__(self, "negative_mode_count", count)
        if self.lowest_curvature is not None:
            value = float(self.lowest_curvature)
            if not np.isfinite(value):
                raise ValueError("lowest_curvature must be finite")
            object.__setattr__(self, "lowest_curvature", value)
        object.__setattr__(self, "observation_id", str(self.observation_id).strip())
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True)
class ResidualSignal:
    """An already-available real/model residual scalar; never requests a probe."""

    origin: str
    age: int
    value: float
    valid: bool
    root_valid: bool | None = None
    inertia_valid: bool | None = None
    observation_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        origin = str(self.origin).strip().lower()
        if not origin:
            raise ValueError("residual signal origin cannot be empty")
        age = int(self.age)
        value = float(self.value)
        if age < 0 or not np.isfinite(value) or value < 0.0:
            raise ValueError("residual age/value must be nonnegative and finite")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "age", age)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "observation_id", str(self.observation_id).strip())
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True)
class AngleSignal:
    """Already-available predicted/model angular-change scalar."""

    origin: str
    age: int
    radians: float
    valid: bool
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        origin = str(self.origin).strip().lower()
        age = int(self.age)
        value = float(self.radians)
        if not origin or age < 0 or not np.isfinite(value) or value < 0.0:
            raise ValueError("angle signal requires nonempty origin and finite nonnegative age/radians")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "age", age)
        object.__setattr__(self, "radians", value)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True)
class ModeScheduleConfig:
    """Pure scheduler settings.  Defaults are legacy/always-solve."""

    max_skips: int = 0
    skip_entry_gate: str = "none"
    refresh_policy: str = "bounded"
    parallel_force_increase_trigger: bool = False
    physical_model_negative_mode_trigger: bool = False
    displacement_trigger: bool = False
    stale_data_trigger: bool = False
    real_residual_trigger: bool = False
    model_residual_trigger: bool = False
    angle_trigger: bool = False
    max_cumulative_displacement: float | None = None
    max_curvature_age: int | None = None
    real_residual_threshold: float | None = None
    model_residual_threshold: float | None = None
    angle_threshold_radians: float | None = None
    required_negative_modes: int = 1
    negative_curvature_tolerance: float = 0.0
    displacement_tolerance: float = 1.0e-12
    parallel_metric_policy: ParallelForceMetricPolicy = field(default_factory=ParallelForceMetricPolicy)

    def __post_init__(self) -> None:
        max_skips = int(self.max_skips)
        if max_skips < 0:
            raise ValueError("max_skips must be >= 0")
        object.__setattr__(self, "max_skips", max_skips)
        gate = str(self.skip_entry_gate).strip().lower()
        if gate not in ENTRY_GATES:
            raise ValueError(f"skip_entry_gate must be one of {sorted(ENTRY_GATES)}")
        object.__setattr__(self, "skip_entry_gate", gate)
        refresh = str(self.refresh_policy).strip().lower()
        if refresh not in REFRESH_POLICIES:
            raise ValueError(f"refresh_policy must be one of {sorted(REFRESH_POLICIES)}")
        object.__setattr__(self, "refresh_policy", refresh)
        required = int(self.required_negative_modes)
        if required < 1:
            raise ValueError("required_negative_modes must be >= 1")
        object.__setattr__(self, "required_negative_modes", required)
        if self.max_curvature_age is not None:
            age = int(self.max_curvature_age)
            if age < 0:
                raise ValueError("max_curvature_age must be >= 0")
            object.__setattr__(self, "max_curvature_age", age)
        for name in (
            "max_cumulative_displacement",
            "real_residual_threshold",
            "model_residual_threshold",
            "angle_threshold_radians",
            "negative_curvature_tolerance",
            "displacement_tolerance",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        if self.displacement_trigger and self.max_cumulative_displacement is None:
            raise ValueError("displacement_trigger requires max_cumulative_displacement")
        if self.real_residual_trigger and self.real_residual_threshold is None:
            raise ValueError("real_residual_trigger requires real_residual_threshold")
        if self.model_residual_trigger and self.model_residual_threshold is None:
            raise ValueError("model_residual_trigger requires model_residual_threshold")
        if self.angle_trigger and self.angle_threshold_radians is None:
            raise ValueError("angle_trigger requires angle_threshold_radians")


@dataclass(frozen=True)
class ModeScheduleObservation:
    """One scheduled mode-solve point after a translated center is observed."""

    state_id: int
    state_uid: str
    geometry_id: str
    center_observation_id: str
    positions: Array
    raw_center_force: Array
    pre_prediction_mode: Array
    pre_prediction_mode_identity: str
    coordinate_space: ActiveCoordinateSpace
    final_force_convergence_candidate: bool = False
    predictor_enabled: bool = False
    prediction_data_valid: bool = True
    prediction_safety_ok: bool = True
    stale_data: bool = False
    stale_reason: str = ""
    degeneracy_or_root_ambiguity: bool = False
    physical_model: PhysicalModelSignal | None = None
    real_residual: ResidualSignal | None = None
    model_residual: ResidualSignal | None = None
    angle_signal: AngleSignal | None = None

    def __post_init__(self) -> None:
        sid = int(self.state_id)
        if sid < 0:
            raise ValueError("state_id must be >= 0")
        object.__setattr__(self, "state_id", sid)
        for name in ("state_uid", "geometry_id", "center_observation_id", "pre_prediction_mode_identity"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        shape = self.coordinate_space.active_dof_mask.shape
        object.__setattr__(self, "positions", freeze_array(self.positions, shape=shape))
        object.__setattr__(self, "raw_center_force", freeze_array(self.raw_center_force, shape=shape))
        normalized = self.coordinate_space.normalized(self.pre_prediction_mode)
        object.__setattr__(self, "pre_prediction_mode", freeze_array(normalized, shape=shape))
        # Preserve a fingerprint of the vector actually stored in this observation.
        object.__setattr__(self, "pre_prediction_mode_identity", mode_identity(normalized, self.coordinate_space))
        object.__setattr__(self, "stale_reason", str(self.stale_reason).strip().lower())


@dataclass(frozen=True)
class RealSolveEntryDiagnostic:
    """Identity-bound, already-paid diagnostic captured before a real Dimer rotation."""

    valid: bool
    state_id: int
    state_uid: str
    geometry_id: str
    center_observation_id: str
    mode_identity: str
    initial_torque_norm: float | None = None
    f_rot_max: float | None = None
    entry_mode_already_converged: bool = False
    reason: str = ""
    additional_pes_calls: int = 0

    def __post_init__(self) -> None:
        sid = int(self.state_id)
        if sid < 0:
            raise ValueError("entry diagnostic state_id must be >= 0")
        object.__setattr__(self, "state_id", sid)
        for name in ("state_uid", "geometry_id", "center_observation_id", "mode_identity"):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        object.__setattr__(self, "reason", str(self.reason).strip().lower())
        calls = int(self.additional_pes_calls)
        if calls != 0:
            raise ValueError("entry-torque diagnostic may not add PES calls")
        object.__setattr__(self, "additional_pes_calls", calls)
        if self.initial_torque_norm is not None:
            torque = float(self.initial_torque_norm)
            if not np.isfinite(torque) or torque < 0.0:
                raise ValueError("initial_torque_norm must be finite and >= 0")
            object.__setattr__(self, "initial_torque_norm", torque)
        if self.f_rot_max is not None:
            threshold = float(self.f_rot_max)
            if not np.isfinite(threshold) or threshold < 0.0:
                raise ValueError("f_rot_max must be finite and >= 0")
            object.__setattr__(self, "f_rot_max", threshold)
        if self.valid:
            if not all((self.state_uid, self.geometry_id, self.center_observation_id, self.mode_identity)):
                raise ValueError("valid entry diagnostic requires complete state/geometry/observation/mode identity")
            if self.initial_torque_norm is None or self.f_rot_max is None:
                raise ValueError("valid entry diagnostic requires torque norm and f_rot_max")
            expected = bool(self.initial_torque_norm <= self.f_rot_max)
            if bool(self.entry_mode_already_converged) != expected:
                raise ValueError("entry_mode_already_converged must equal initial_torque_norm <= f_rot_max")

    def to_state_dict(self) -> dict[str, object]:
        return {
            "valid": bool(self.valid),
            "state_id": int(self.state_id),
            "state_uid": self.state_uid,
            "geometry_id": self.geometry_id,
            "center_observation_id": self.center_observation_id,
            "mode_identity": self.mode_identity,
            "initial_torque_norm": self.initial_torque_norm,
            "f_rot_max": self.f_rot_max,
            "entry_mode_already_converged": bool(self.entry_mode_already_converged),
            "reason": self.reason,
            "additional_pes_calls": 0,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "RealSolveEntryDiagnostic":
        return cls(
            valid=bool(state.get("valid", False)),
            state_id=int(state.get("state_id", 0)),
            state_uid=str(state.get("state_uid", "")),
            geometry_id=str(state.get("geometry_id", "")),
            center_observation_id=str(state.get("center_observation_id", "")),
            mode_identity=str(state.get("mode_identity", "")),
            initial_torque_norm=state.get("initial_torque_norm"),
            f_rot_max=state.get("f_rot_max"),
            entry_mode_already_converged=bool(state.get("entry_mode_already_converged", False)),
            reason=str(state.get("reason", "")),
            additional_pes_calls=int(state.get("additional_pes_calls", 0)),
        )


@dataclass(frozen=True)
class ModeScheduleRuntimeState:
    """mode-schedule runtime state extending the sealed common ``ModeScheduleState`` schema."""

    core: ModeScheduleState
    coordinate_space_identity: str
    last_state_uid: str
    last_geometry_id: str
    last_center_observation_id: str
    last_center_position: Array
    held_mode_identity: str
    pre_prediction_mode_identity: str
    previous_parallel_metric: float | None
    cumulative_displacement: float
    skip_entry_gate: str = "none"
    refresh_policy: str = "bounded"
    skip_sequence_eligible: bool = True
    last_entry_diagnostic: RealSolveEntryDiagnostic | None = None
    curvature_origin: str = "unmeasured_held_mode"
    curvature_age: int | None = None
    curvature_observation_id: str = ""
    curvature_valid: bool = False
    curvature_lowest_value: float | None = None
    curvature_negative_mode_count: int | None = None
    pending_final_validation: bool = False
    sequence_index: int = 0
    last_predictor_disposition: str = "disabled"
    last_prediction_identity: str = ""
    schema_version: int = 2

    def __post_init__(self) -> None:
        if int(self.schema_version) not in {1, 2}:
            raise ValueError("unsupported ModeScheduleRuntimeState schema version")
        gate = str(self.skip_entry_gate).strip().lower()
        refresh = str(self.refresh_policy).strip().lower()
        if gate not in ENTRY_GATES:
            raise ValueError(f"skip_entry_gate must be one of {sorted(ENTRY_GATES)}")
        if refresh not in REFRESH_POLICIES:
            raise ValueError(f"refresh_policy must be one of {sorted(REFRESH_POLICIES)}")
        object.__setattr__(self, "skip_entry_gate", gate)
        object.__setattr__(self, "refresh_policy", refresh)
        object.__setattr__(self, "skip_sequence_eligible", bool(self.skip_sequence_eligible))
        for name in (
            "coordinate_space_identity",
            "last_state_uid",
            "last_geometry_id",
            "last_center_observation_id",
            "held_mode_identity",
            "pre_prediction_mode_identity",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        shape = None if self.core.last_center_force is None else self.core.last_center_force.shape
        array = np.asarray(self.last_center_position, dtype=float)
        if shape is not None and array.shape != shape:
            raise ValueError("last_center_position must match scheduler force shape")
        if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
            raise ValueError("last_center_position must have finite shape (N, 3)")
        object.__setattr__(self, "last_center_position", freeze_array(array, shape=array.shape))
        cumulative = float(self.cumulative_displacement)
        if not np.isfinite(cumulative) or cumulative < 0.0:
            raise ValueError("cumulative_displacement must be finite and >= 0")
        object.__setattr__(self, "cumulative_displacement", cumulative)
        if self.previous_parallel_metric is not None:
            metric = float(self.previous_parallel_metric)
            if not np.isfinite(metric) or metric < 0.0:
                raise ValueError("previous_parallel_metric must be finite and >= 0")
            object.__setattr__(self, "previous_parallel_metric", metric)
        origin = str(self.curvature_origin).strip().lower()
        object.__setattr__(self, "curvature_origin", origin or "unmeasured_held_mode")
        if self.curvature_age is not None:
            age = int(self.curvature_age)
            if age < 0:
                raise ValueError("curvature_age must be >= 0")
            object.__setattr__(self, "curvature_age", age)
        if self.curvature_lowest_value is not None:
            value = float(self.curvature_lowest_value)
            if not np.isfinite(value):
                raise ValueError("curvature_lowest_value must be finite")
            object.__setattr__(self, "curvature_lowest_value", value)
        if self.curvature_negative_mode_count is not None:
            count = int(self.curvature_negative_mode_count)
            if count < 0:
                raise ValueError("curvature_negative_mode_count must be >= 0")
            object.__setattr__(self, "curvature_negative_mode_count", count)
        idx = int(self.sequence_index)
        if idx < 0:
            raise ValueError("sequence_index must be >= 0")
        object.__setattr__(self, "sequence_index", idx)
        object.__setattr__(self, "curvature_observation_id", str(self.curvature_observation_id).strip())
        object.__setattr__(self, "last_predictor_disposition", str(self.last_predictor_disposition).strip().lower())
        object.__setattr__(self, "last_prediction_identity", str(self.last_prediction_identity).strip())

    def to_state_dict(self) -> dict[str, object]:
        return {
            "schema": RUNTIME_SCHEMA,
            "schema_version": self.schema_version,
            "core": self.core.to_state_dict(),
            "coordinate_space_identity": self.coordinate_space_identity,
            "last_state_uid": self.last_state_uid,
            "last_geometry_id": self.last_geometry_id,
            "last_center_observation_id": self.last_center_observation_id,
            "last_center_position": self.last_center_position.tolist(),
            "held_mode_identity": self.held_mode_identity,
            "pre_prediction_mode_identity": self.pre_prediction_mode_identity,
            "previous_parallel_metric": self.previous_parallel_metric,
            "cumulative_displacement": self.cumulative_displacement,
            "skip_entry_gate": self.skip_entry_gate,
            "refresh_policy": self.refresh_policy,
            "skip_sequence_eligible": bool(self.skip_sequence_eligible),
            "last_entry_diagnostic": None if self.last_entry_diagnostic is None else self.last_entry_diagnostic.to_state_dict(),
            "curvature_origin": self.curvature_origin,
            "curvature_age": self.curvature_age,
            "curvature_observation_id": self.curvature_observation_id,
            "curvature_valid": self.curvature_valid,
            "curvature_lowest_value": self.curvature_lowest_value,
            "curvature_negative_mode_count": self.curvature_negative_mode_count,
            "pending_final_validation": self.pending_final_validation,
            "sequence_index": self.sequence_index,
            "last_predictor_disposition": self.last_predictor_disposition,
            "last_prediction_identity": self.last_prediction_identity,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ModeScheduleRuntimeState":
        if state.get("schema") != RUNTIME_SCHEMA:
            raise ValueError("unsupported ModeScheduleRuntimeState schema")
        return cls(
            core=ModeScheduleState.from_state_dict(dict(state["core"])),
            coordinate_space_identity=str(state["coordinate_space_identity"]),
            last_state_uid=str(state["last_state_uid"]),
            last_geometry_id=str(state["last_geometry_id"]),
            last_center_observation_id=str(state["last_center_observation_id"]),
            last_center_position=np.asarray(state["last_center_position"], dtype=float),
            held_mode_identity=str(state["held_mode_identity"]),
            pre_prediction_mode_identity=str(state["pre_prediction_mode_identity"]),
            previous_parallel_metric=state.get("previous_parallel_metric"),
            cumulative_displacement=float(state.get("cumulative_displacement", 0.0)),
            skip_entry_gate=str(state.get("skip_entry_gate", "none")),
            refresh_policy=str(state.get("refresh_policy", "bounded")),
            skip_sequence_eligible=bool(state.get("skip_sequence_eligible", True)),
            last_entry_diagnostic=(None if state.get("last_entry_diagnostic") is None else RealSolveEntryDiagnostic.from_state_dict(dict(state["last_entry_diagnostic"]))),
            curvature_origin=str(state.get("curvature_origin", "unmeasured_held_mode")),
            curvature_age=state.get("curvature_age"),
            curvature_observation_id=str(state.get("curvature_observation_id", "")),
            curvature_valid=bool(state.get("curvature_valid", False)),
            curvature_lowest_value=state.get("curvature_lowest_value"),
            curvature_negative_mode_count=state.get("curvature_negative_mode_count"),
            pending_final_validation=bool(state.get("pending_final_validation", False)),
            sequence_index=int(state.get("sequence_index", 0)),
            last_predictor_disposition=str(state.get("last_predictor_disposition", "disabled")),
            last_prediction_identity=str(state.get("last_prediction_identity", "")),
            schema_version=2,
        )


@dataclass(frozen=True)
class ModeScheduleEvaluation:
    decision: ModeScheduleDecision
    state: ModeScheduleRuntimeState
    observation: ModeScheduleObservation
    previous_state_id: int | None
    previous_state_uid: str
    previous_geometry_id: str
    previous_center_observation_id: str
    trigger_metadata: Mapping[str, object]
    predictor_disposition: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "trigger_metadata", freeze_mapping(self.trigger_metadata))
        object.__setattr__(self, "predictor_disposition", str(self.predictor_disposition).strip().lower())


@dataclass(frozen=True)
class PredictionUse:
    disposition: str
    solver_initial_mode: Array | None
    held_mode: Array
    prediction_identity: str
    predicted_vs_held_same: bool | None
    predictor_status: str

    def __post_init__(self) -> None:
        held = np.asarray(self.held_mode, dtype=float)
        object.__setattr__(self, "held_mode", freeze_array(held, shape=held.shape))
        if self.solver_initial_mode is not None:
            object.__setattr__(self, "solver_initial_mode", freeze_array(self.solver_initial_mode, shape=held.shape))
        object.__setattr__(self, "disposition", str(self.disposition).strip().lower())
        object.__setattr__(self, "prediction_identity", str(self.prediction_identity).strip())
        object.__setattr__(self, "predictor_status", str(self.predictor_status).strip().lower())


def mode_identity(mode: object, coordinate_space: ActiveCoordinateSpace) -> str:
    """Sign-invariant exact numeric identity for one normalized active eigendirection."""

    v = coordinate_space.normalized(mode).reshape(-1).copy()
    nz = np.flatnonzero(np.abs(v) > 1.0e-15)
    if nz.size and v[nz[np.argmax(np.abs(v[nz]))]] < 0.0:
        v *= -1.0
    digest = hashlib.sha256()
    digest.update(b"saddlemill-mode-direction-v1\0")
    digest.update(coordinate_space.identity.encode("utf-8"))
    digest.update(np.ascontiguousarray(v, dtype="<f8").tobytes())
    return "mode:" + digest.hexdigest()


def _sequence_id(state_id: int, sequence_index: int) -> str:
    return f"mode-sequence:{int(sequence_index)}:solve-state:{int(state_id)}"


def _signal_state(signal: PhysicalModelSignal | None, prior: ModeScheduleRuntimeState | None) -> dict[str, object]:
    if signal is not None and signal.valid:
        return {
            "curvature_origin": signal.origin,
            "curvature_age": signal.age,
            "curvature_observation_id": signal.observation_id,
            "curvature_valid": True,
            "curvature_lowest_value": signal.lowest_curvature,
            "curvature_negative_mode_count": signal.negative_mode_count,
        }
    if prior is None:
        return {
            "curvature_origin": "unmeasured_held_mode",
            "curvature_age": None,
            "curvature_observation_id": "",
            "curvature_valid": False,
            "curvature_lowest_value": None,
            "curvature_negative_mode_count": None,
        }
    age = None if prior.curvature_age is None else prior.curvature_age + 1
    return {
        "curvature_origin": prior.curvature_origin if prior.curvature_valid else "unmeasured_held_mode",
        "curvature_age": age,
        "curvature_observation_id": prior.curvature_observation_id,
        "curvature_valid": prior.curvature_valid,
        "curvature_lowest_value": prior.curvature_lowest_value,
        "curvature_negative_mode_count": prior.curvature_negative_mode_count,
    }


class BoundedModeScheduler:
    """Serializable bounded-skip state machine implementing the C1 mode-schedule contract."""

    _PRIMARY_PRIORITY = (
        "numerical_failure",
        "identity_failure",
        "stale_data",
        "mandatory_final_physical_mode_validation",
        "physical_model_inertia_or_negative_mode_loss",
        "residual_root_inertia_safeguard",
        "physical_parallel_force_increase",
        "real_residual_threshold",
        "model_residual_threshold",
        "angle_threshold",
        "displacement_limit",
        "entry_gate_not_satisfied",
        "physical_model_unavailable_or_stale",
        "max_skips_exhausted",
    )
    _PREDICTION_SUPPRESSION = {
        "numerical_failure",
        "identity_failure",
        "stale_data",
        "physical_model_inertia_or_negative_mode_loss",
        "residual_root_inertia_safeguard",
        "degeneracy_or_root_ambiguity",
        "physical_model_unavailable_or_stale",
    }

    def __init__(
        self,
        config: ModeScheduleConfig | None = None,
        *,
        damping_config: ParallelForceDampingConfig | None = None,
    ) -> None:
        self.config = config or ModeScheduleConfig()
        self.damping_config = damping_config or ParallelForceDampingConfig()

    def _entry_gate_result(
        self,
        observation: ModeScheduleObservation,
        diagnostic: RealSolveEntryDiagnostic | None,
    ) -> tuple[bool, dict[str, object]]:
        gate = self.config.skip_entry_gate
        if gate == "none":
            return True, {"selector": gate, "eligible": True, "reason": "gate_disabled"}
        if diagnostic is None or not diagnostic.valid:
            reason = "entry_torque_unavailable" if diagnostic is None else (diagnostic.reason or "entry_torque_invalid")
            return False, {"selector": gate, "eligible": False, "reason": reason}
        identity_ok = bool(
            diagnostic.state_id == observation.state_id
            and diagnostic.state_uid == observation.state_uid
            and diagnostic.geometry_id == observation.geometry_id
            and diagnostic.center_observation_id == observation.center_observation_id
            and bool(diagnostic.mode_identity)
        )
        eligible = bool(identity_ok and diagnostic.entry_mode_already_converged)
        return eligible, {
            "selector": gate,
            "eligible": eligible,
            "identity_ok": identity_ok,
            "initial_torque_norm": diagnostic.initial_torque_norm,
            "f_rot_max": diagnostic.f_rot_max,
            "entry_mode_already_converged": bool(diagnostic.entry_mode_already_converged),
            "reason": ("eligible" if eligible else ("entry_identity_mismatch" if not identity_ok else "entry_torque_above_f_rot_max")),
            "diagnostic": diagnostic.to_state_dict(),
        }

    def initialize_after_real_solve(
        self,
        observation: ModeScheduleObservation,
        *,
        solved_mode: object,
        fresh_physical_validation: bool,
        physical_model: PhysicalModelSignal | None = None,
        entry_diagnostic: RealSolveEntryDiagnostic | None = None,
    ) -> ModeScheduleRuntimeState:
        space = observation.coordinate_space
        solved = space.normalized(solved_mode)
        seq_index = 0
        seq_id = _sequence_id(observation.state_id, seq_index)
        damping = select_damping_for_sequence(
            observation.raw_center_force,
            solved,
            space,
            self.damping_config,
            sequence_id=seq_id,
        )
        signal = physical_model if physical_model is not None else observation.physical_model
        curvature = _signal_state(signal, None)
        metric = self._parallel_metric(observation.raw_center_force, solved, space)
        gate_eligible, _gate_meta = self._entry_gate_result(observation, entry_diagnostic)
        core = ModeScheduleState(
            max_skips=self.config.max_skips,
            skips_since_real_solve=0,
            last_real_solve_state_id=observation.state_id,
            last_checked_state_id=observation.state_id,
            last_pre_prediction_mode=solved,
            last_center_force=observation.raw_center_force,
            displacement_reference=observation.positions,
            sequence_id=seq_id,
            damping_lambda=damping.selected_lambda,
            damping_lambda_identity=damping.identity,
            model_age=curvature["curvature_age"],
            parallel_metric_policy=self.config.parallel_metric_policy,
        )
        solved_id = mode_identity(solved, space)
        return ModeScheduleRuntimeState(
            core=core,
            coordinate_space_identity=space.identity,
            last_state_uid=observation.state_uid,
            last_geometry_id=observation.geometry_id,
            last_center_observation_id=observation.center_observation_id,
            last_center_position=observation.positions,
            held_mode_identity=solved_id,
            pre_prediction_mode_identity=solved_id,
            previous_parallel_metric=metric,
            cumulative_displacement=0.0,
            skip_entry_gate=self.config.skip_entry_gate,
            refresh_policy=self.config.refresh_policy,
            skip_sequence_eligible=gate_eligible,
            last_entry_diagnostic=entry_diagnostic,
            pending_final_validation=not bool(fresh_physical_validation),
            sequence_index=seq_index,
            **curvature,
        )

    @staticmethod
    def _parallel_metric(force: object, mode: object, space: ActiveCoordinateSpace) -> float:
        projected_force = space.project(force)
        unit = space.normalized(mode)
        signed = float(np.dot(projected_force.reshape(-1), unit.reshape(-1)))
        parallel = signed * unit
        values = parallel[space.active_dof_mask]
        if values.size == 0:
            raise ValueError("parallel metric requires movable Cartesian DOFs")
        return float(np.sqrt(np.dot(values, values) / values.size))

    def _parallel_increase(self, previous: float, current: float) -> tuple[bool, float]:
        policy = self.config.parallel_metric_policy
        # Exact sealed comparison: strict current > previous + abs_tol + rel_tol*|previous|.
        tolerance = policy.absolute_tolerance + policy.relative_tolerance * abs(previous)
        return bool(current > previous + tolerance), float(tolerance)

    def _negative_curvature_regime(self, curvature_meta: Mapping[str, object]) -> bool:
        if not bool(curvature_meta.get("available", False)) or not bool(curvature_meta.get("usable_under_age_policy", False)):
            return False
        count = curvature_meta.get("negative_mode_count")
        if count is not None:
            return int(count) >= self.config.required_negative_modes
        value = curvature_meta.get("lowest_curvature")
        if value is not None:
            return float(value) < -self.config.negative_curvature_tolerance
        return False

    def _identity_failure(self, state: ModeScheduleRuntimeState, obs: ModeScheduleObservation) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if obs.coordinate_space.identity != state.coordinate_space_identity:
            reasons.append("coordinate_space_identity_mismatch")
        # A valid observed mode is sufficient. Float-vector similarity is not
        # an identity contract and must not decide whether a real solve can run.
        if state.core.last_checked_state_id is not None and obs.state_id <= state.core.last_checked_state_id:
            reasons.append("nonmonotonic_or_replayed_state_id")
        return bool(reasons), reasons

    def _curvature_metadata(
        self,
        state: ModeScheduleRuntimeState,
        obs: ModeScheduleObservation,
    ) -> tuple[dict[str, object], dict[str, object], bool, bool]:
        signal = obs.physical_model
        if signal is None:
            carried = _signal_state(None, state)
            age = carried["curvature_age"]
            usable = bool(
                carried["curvature_valid"]
                and (
                    self.config.max_curvature_age is None
                    or (age is not None and age <= self.config.max_curvature_age)
                )
            )
            # Carried curvature is explicitly aged; only age==0 is physically current.
            fresh = bool(carried["curvature_valid"] and age == 0)
            meta = {
                "available": bool(carried["curvature_valid"]),
                "fresh": fresh,
                "usable_under_age_policy": usable,
                "origin": carried["curvature_origin"],
                "age": age,
                "observation_id": carried["curvature_observation_id"],
                "negative_mode_count": carried["curvature_negative_mode_count"],
                "lowest_curvature": carried["curvature_lowest_value"],
                "carried_forward": True,
                "negative_mode_loss": False,
            }
            stale = bool(carried["curvature_valid"] and not usable)
            return meta, carried, False, stale

        current = _signal_state(signal, state)
        usable = bool(
            signal.valid
            and (self.config.max_curvature_age is None or signal.age <= self.config.max_curvature_age)
        )
        fresh = bool(signal.valid and signal.age == 0)
        stale = bool(signal.valid and not usable)
        loss = False
        if (self.config.physical_model_negative_mode_trigger or self.config.refresh_policy == "physical_model_loss") and usable:
            if signal.negative_mode_count is not None:
                loss = signal.negative_mode_count < self.config.required_negative_modes
            elif signal.lowest_curvature is not None:
                loss = signal.lowest_curvature >= -self.config.negative_curvature_tolerance
        meta = {
            "available": bool(signal.valid),
            "fresh": fresh,
            "usable_under_age_policy": usable,
            "origin": signal.origin,
            "age": signal.age,
            "observation_id": signal.observation_id,
            "negative_mode_count": signal.negative_mode_count,
            "lowest_curvature": signal.lowest_curvature,
            "carried_forward": False,
            "negative_mode_loss": loss,
        }
        return meta, current, loss, stale

    @staticmethod
    def _residual_trigger(signal: ResidualSignal | None, threshold: float | None) -> tuple[bool, bool, dict[str, object]]:
        if signal is None:
            return False, False, {"available": False}
        meta = {
            "available": bool(signal.valid),
            "origin": signal.origin,
            "age": signal.age,
            "value": signal.value,
            "threshold": threshold,
            "root_valid": signal.root_valid,
            "inertia_valid": signal.inertia_valid,
            "observation_id": signal.observation_id,
        }
        if not signal.valid or threshold is None:
            return False, False, meta
        high = signal.value > threshold
        small_but_unsecured = (not high) and (signal.root_valid is not True or signal.inertia_valid is not True)
        return high, small_but_unsecured, meta

    def evaluate(
        self,
        state: ModeScheduleRuntimeState,
        observation: ModeScheduleObservation,
    ) -> ModeScheduleEvaluation:
        obs = observation
        fired: set[str] = set()
        identity_bad, identity_reasons = self._identity_failure(state, obs)
        if identity_bad:
            fired.add("identity_failure")

        numerical_bad = False
        try:
            step = obs.coordinate_space.project(obs.positions - state.last_center_position)
            step_displacement = float(np.linalg.norm(step.reshape(-1)))
            cumulative = state.cumulative_displacement + step_displacement
            previous_metric = self._parallel_metric(state.core.last_center_force, obs.pre_prediction_mode, obs.coordinate_space)
            current_metric = self._parallel_metric(obs.raw_center_force, obs.pre_prediction_mode, obs.coordinate_space)
            increase, comparison_tolerance = self._parallel_increase(previous_metric, current_metric)
        except (ValueError, FloatingPointError):
            numerical_bad = True
            fired.add("numerical_failure")
            step_displacement = math.nan
            cumulative = state.cumulative_displacement
            previous_metric = state.previous_parallel_metric
            current_metric = None
            increase = False
            comparison_tolerance = None

        if obs.stale_data and self.config.stale_data_trigger:
            fired.add("stale_data")
        if obs.final_force_convergence_candidate:
            fired.add("mandatory_final_physical_mode_validation")
        if obs.degeneracy_or_root_ambiguity:
            fired.add("degeneracy_or_root_ambiguity")

        curvature_meta, curvature_state, negative_mode_loss, curvature_stale = self._curvature_metadata(state, obs)
        negative_regime = self._negative_curvature_regime(curvature_meta)
        if self.config.parallel_force_increase_trigger and negative_regime and increase:
            fired.add("physical_parallel_force_increase")
        if negative_mode_loss:
            fired.add("physical_model_inertia_or_negative_mode_loss")
        if curvature_stale and self.config.stale_data_trigger:
            fired.add("stale_data")
        if self.config.refresh_policy == "physical_model_loss":
            signal = obs.physical_model
            root_known = bool(
                signal is not None
                and (signal.negative_mode_count is not None or signal.lowest_curvature is not None)
            )
            model_identity_ok = bool(
                signal is not None
                and signal.valid
                and root_known
                and bool(curvature_meta.get("usable_under_age_policy", False))
                and str(signal.metadata.get("current_state_uid", "")) == obs.state_uid
                and str(signal.metadata.get("current_geometry_id", "")) == obs.geometry_id
            )
            curvature_meta["adaptive_root_known"] = root_known
            curvature_meta["adaptive_model_identity_ok"] = model_identity_ok
            if not model_identity_ok:
                fired.add("physical_model_unavailable_or_stale")

        real_high, real_unsecured, real_meta = self._residual_trigger(
            obs.real_residual, self.config.real_residual_threshold if self.config.real_residual_trigger else None
        )
        model_high, model_unsecured, model_meta = self._residual_trigger(
            obs.model_residual, self.config.model_residual_threshold if self.config.model_residual_trigger else None
        )
        if self.config.real_residual_trigger:
            if real_high:
                fired.add("real_residual_threshold")
            elif real_unsecured:
                fired.add("residual_root_inertia_safeguard")
        if self.config.model_residual_trigger:
            if model_high:
                fired.add("model_residual_threshold")
            elif model_unsecured:
                fired.add("residual_root_inertia_safeguard")

        angle_meta: dict[str, object] = {"available": False}
        if obs.angle_signal is not None:
            angle_meta = {
                "available": bool(obs.angle_signal.valid),
                "origin": obs.angle_signal.origin,
                "age": obs.angle_signal.age,
                "radians": obs.angle_signal.radians,
                "threshold": self.config.angle_threshold_radians,
            }
            if (
                self.config.angle_trigger
                and obs.angle_signal.valid
                and self.config.angle_threshold_radians is not None
                and obs.angle_signal.radians > self.config.angle_threshold_radians
            ):
                fired.add("angle_threshold")

        if (
            self.config.displacement_trigger
            and self.config.max_cumulative_displacement is not None
            and cumulative >= self.config.max_cumulative_displacement - self.config.displacement_tolerance
        ):
            fired.add("displacement_limit")

        if not state.skip_sequence_eligible:
            fired.add("entry_gate_not_satisfied")
        if self.config.refresh_policy == "bounded" and state.core.skips_since_real_solve >= state.core.max_skips:
            fired.add("max_skips_exhausted")

        primary = next((reason for reason in self._PRIMARY_PRIORITY if reason in fired), None)
        # Degeneracy is a safety refresh even though it is not part of the historical primary list.
        if primary is None and "degeneracy_or_root_ambiguity" in fired:
            primary = "degeneracy_or_root_ambiguity"
        require_solve = primary is not None
        if not require_solve:
            primary = "scheduled_skip"
        skip = not require_solve

        predictor_requested = bool(obs.predictor_enabled)
        safety_reason = primary if primary in self._PREDICTION_SUPPRESSION else ""
        prediction_allowed = bool(
            predictor_requested
            and obs.prediction_data_valid
            and obs.prediction_safety_ok
            and not safety_reason
        )
        if not predictor_requested:
            disposition = "disabled"
        elif not prediction_allowed:
            disposition = "suppressed_" + (safety_reason or "invalid_predictor_data")
        elif require_solve:
            disposition = "real_solve_predict_initial_guess"
        else:
            disposition = "skip_predict_held_mode"

        metadata = {
            "fired_triggers": tuple(reason for reason in self._PRIMARY_PRIORITY if reason in fired)
            + (("degeneracy_or_root_ambiguity",) if "degeneracy_or_root_ambiguity" in fired else ()),
            "primary_reason": primary,
            "old_state_id": state.core.last_checked_state_id,
            "new_state_id": obs.state_id,
            "old_state_uid": state.last_state_uid,
            "new_state_uid": obs.state_uid,
            "old_geometry_id": state.last_geometry_id,
            "new_geometry_id": obs.geometry_id,
            "old_center_observation_id": state.last_center_observation_id,
            "new_center_observation_id": obs.center_observation_id,
            "held_mode_identity": state.held_mode_identity,
            "pre_prediction_mode_identity": obs.pre_prediction_mode_identity,
            "predictor_disposition": disposition,
            "prediction_allowed": prediction_allowed,
            "skip_entry_gate": state.skip_entry_gate,
            "skip_sequence_eligible": bool(state.skip_sequence_eligible),
            "last_entry_diagnostic": None if state.last_entry_diagnostic is None else state.last_entry_diagnostic.to_state_dict(),
            "refresh_policy": state.refresh_policy,
            "identity_failure_reasons": tuple(identity_reasons),
            "parallel_force": {
                "metric": self.config.parallel_metric_policy.metric,
                "comparison": str(self.config.parallel_metric_policy.comparison),
                "previous_metric": previous_metric,
                "current_metric": current_metric,
                "absolute_tolerance": self.config.parallel_metric_policy.absolute_tolerance,
                "relative_tolerance": self.config.parallel_metric_policy.relative_tolerance,
                "comparison_tolerance": comparison_tolerance,
                "increase": increase,
                "negative_curvature_regime": negative_regime,
                "trigger_eligible": bool(self.config.parallel_force_increase_trigger and negative_regime),
                "same_pre_prediction_mode": True,
                "raw_physical_center_forces": True,
            },
            "displacement": {
                "step": step_displacement,
                "cumulative": cumulative,
                "limit": self.config.max_cumulative_displacement,
            },
            "curvature": curvature_meta,
            "real_residual": real_meta,
            "model_residual": model_meta,
            "angle": angle_meta,
            "final_force_convergence_candidate": obs.final_force_convergence_candidate,
            "no_diagnostic_pes_request": True,
        }

        # Hard identity/numerical failures preserve last accepted observations until repaired.
        if identity_bad or numerical_bad:
            updated = replace(
                state,
                pending_final_validation=state.pending_final_validation or obs.final_force_convergence_candidate,
                last_predictor_disposition=disposition,
            )
        else:
            next_count = state.core.skips_since_real_solve + (1 if skip else 0)
            core = replace(
                state.core,
                # The sealed common state encodes skips <= max_skips.  In the
                # adaptive policy max_skips is not a cadence, so grow only this
                # serialization ceiling as needed; the configured bounded limit
                # remains authoritative only when refresh_policy=bounded.
                max_skips=(max(state.core.max_skips, next_count) if self.config.refresh_policy == "physical_model_loss" else state.core.max_skips),
                skips_since_real_solve=next_count,
                last_checked_state_id=obs.state_id,
                last_pre_prediction_mode=obs.pre_prediction_mode,
                last_center_force=obs.raw_center_force,
                model_age=curvature_state["curvature_age"],
            )
            updated = replace(
                state,
                core=core,
                last_state_uid=obs.state_uid,
                last_geometry_id=obs.geometry_id,
                last_center_observation_id=obs.center_observation_id,
                last_center_position=obs.positions,
                pre_prediction_mode_identity=obs.pre_prediction_mode_identity,
                previous_parallel_metric=current_metric,
                cumulative_displacement=cumulative,
                pending_final_validation=state.pending_final_validation or obs.final_force_convergence_candidate,
                last_predictor_disposition=disposition,
                **curvature_state,
            )

        decision = ModeScheduleDecision(
            require_real_solve=require_solve,
            skipped_scheduled_solve=skip,
            reason=primary,
            prediction_allowed=prediction_allowed,
            mandatory_fresh_validation=bool(obs.final_force_convergence_candidate),
            state=updated.core,
            curvature_origin=updated.curvature_origin,
            model_origin=curvature_meta.get("origin", ""),
            model_age=updated.curvature_age,
            trigger_metadata=metadata,
        )
        return ModeScheduleEvaluation(
            decision=decision,
            state=updated,
            observation=obs,
            previous_state_id=state.core.last_checked_state_id,
            previous_state_uid=state.last_state_uid,
            previous_geometry_id=state.last_geometry_id,
            previous_center_observation_id=state.last_center_observation_id,
            trigger_metadata=metadata,
            predictor_disposition=disposition,
        )

    def resolve_prediction(
        self,
        evaluation: ModeScheduleEvaluation,
        prediction_result: ModePredictionResult | None,
    ) -> PredictionUse:
        obs = evaluation.observation
        held = obs.pre_prediction_mode
        if not evaluation.decision.prediction_allowed:
            return PredictionUse(
                disposition=evaluation.predictor_disposition,
                solver_initial_mode=None,
                held_mode=held,
                prediction_identity="",
                predicted_vs_held_same=None,
                predictor_status="suppressed" if obs.predictor_enabled else "disabled",
            )
        if prediction_result is None:
            return PredictionUse(
                disposition="prediction_requested_but_unavailable",
                solver_initial_mode=None,
                held_mode=held,
                prediction_identity="",
                predicted_vs_held_same=None,
                predictor_status="unavailable",
            )
        result = prediction_result
        identity_ok = (
            result.old_state_id == evaluation.previous_state_id
            and result.new_state_id == obs.state_id
            and result.old_state_uid == evaluation.previous_state_uid
            and result.new_state_uid == obs.state_uid
            and result.old_geometry_id == evaluation.previous_geometry_id
            and result.new_geometry_id == obs.geometry_id
        )
        if not identity_ok:
            return PredictionUse(
                disposition="suppressed_predictor_identity_mismatch",
                solver_initial_mode=None,
                held_mode=held,
                prediction_identity="",
                predicted_vs_held_same=None,
                predictor_status=result.status,
            )
        predicted = obs.coordinate_space.normalized(result.mode)
        pred_id = mode_identity(predicted, obs.coordinate_space)
        # Exact hashes cannot reliably answer this after repeated normalization.
        # Keep the optional diagnostic unset rather than reporting a false mismatch.
        same = None
        if evaluation.decision.require_real_solve:
            return PredictionUse(
                disposition="real_solve_predict_initial_guess",
                solver_initial_mode=predicted,
                held_mode=held,
                prediction_identity=pred_id,
                predicted_vs_held_same=same,
                predictor_status=result.status,
            )
        return PredictionUse(
            disposition="skip_predict_held_mode",
            solver_initial_mode=None,
            held_mode=predicted,
            prediction_identity=pred_id,
            predicted_vs_held_same=same,
            predictor_status=result.status,
        )

    def complete_skip(
        self,
        evaluation: ModeScheduleEvaluation,
        *,
        prediction_result: ModePredictionResult | None = None,
    ) -> tuple[ModeScheduleRuntimeState, PredictionUse]:
        if not evaluation.decision.skipped_scheduled_solve:
            raise ValueError("complete_skip requires a skipped-solve decision")
        use = self.resolve_prediction(evaluation, prediction_result)
        obs = evaluation.observation
        held = obs.coordinate_space.normalized(use.held_mode)
        held_id = mode_identity(held, obs.coordinate_space)
        core = replace(evaluation.state.core, last_pre_prediction_mode=held)
        state = replace(
            evaluation.state,
            core=core,
            held_mode_identity=held_id,
            pre_prediction_mode_identity=obs.pre_prediction_mode_identity,
            last_predictor_disposition=use.disposition,
            last_prediction_identity=use.prediction_identity,
        )
        return state, use

    def complete_real_solve(
        self,
        evaluation: ModeScheduleEvaluation,
        *,
        solved_mode: object,
        fresh_physical_validation: bool,
        physical_model: PhysicalModelSignal | None = None,
        entry_diagnostic: RealSolveEntryDiagnostic | None = None,
    ) -> ModeScheduleRuntimeState:
        if not evaluation.decision.require_real_solve:
            raise ValueError("complete_real_solve requires a real-solve decision")
        obs = evaluation.observation
        # Identity/numerical failure must be repaired before completion can commit a new sequence.
        if evaluation.decision.reason in {"identity_failure", "numerical_failure"}:
            raise ValueError("cannot complete a real solve from invalid scheduler identity/numerics")
        solved = obs.coordinate_space.normalized(solved_mode)
        seq_index = evaluation.state.sequence_index + 1
        seq_id = _sequence_id(obs.state_id, seq_index)
        damping = select_damping_for_sequence(
            obs.raw_center_force,
            solved,
            obs.coordinate_space,
            self.damping_config,
            sequence_id=seq_id,
        )
        signal = physical_model if physical_model is not None else obs.physical_model
        curvature = _signal_state(signal, None)
        gate_eligible, _gate_meta = self._entry_gate_result(obs, entry_diagnostic)
        core = ModeScheduleState(
            max_skips=self.config.max_skips,
            skips_since_real_solve=0,
            last_real_solve_state_id=obs.state_id,
            last_checked_state_id=obs.state_id,
            last_pre_prediction_mode=solved,
            last_center_force=obs.raw_center_force,
            displacement_reference=obs.positions,
            sequence_id=seq_id,
            damping_lambda=damping.selected_lambda,
            damping_lambda_identity=damping.identity,
            model_age=curvature["curvature_age"],
            parallel_metric_policy=self.config.parallel_metric_policy,
        )
        solved_id = mode_identity(solved, obs.coordinate_space)
        pending = bool(evaluation.state.pending_final_validation and not fresh_physical_validation)
        return ModeScheduleRuntimeState(
            core=core,
            coordinate_space_identity=obs.coordinate_space.identity,
            last_state_uid=obs.state_uid,
            last_geometry_id=obs.geometry_id,
            last_center_observation_id=obs.center_observation_id,
            last_center_position=obs.positions,
            held_mode_identity=solved_id,
            pre_prediction_mode_identity=solved_id,
            previous_parallel_metric=self._parallel_metric(obs.raw_center_force, solved, obs.coordinate_space),
            cumulative_displacement=0.0,
            skip_entry_gate=self.config.skip_entry_gate,
            refresh_policy=self.config.refresh_policy,
            skip_sequence_eligible=gate_eligible,
            last_entry_diagnostic=entry_diagnostic,
            pending_final_validation=pending,
            sequence_index=seq_index,
            last_predictor_disposition=evaluation.predictor_disposition,
            last_prediction_identity="",
            **curvature,
        )


__all__ = [
    "AngleSignal",
    "BoundedModeScheduler",
    "ModeScheduleConfig",
    "ModeScheduleEvaluation",
    "ModeScheduleObservation",
    "ModeScheduleRuntimeState",
    "PhysicalModelSignal",
    "RealSolveEntryDiagnostic",
    "ENTRY_GATES",
    "REFRESH_POLICIES",
    "PredictionUse",
    "RUNTIME_SCHEMA",
    "ResidualSignal",
    "mode_identity",
]
