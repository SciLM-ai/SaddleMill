"""Typed foundation contracts shared by SaddleMill minimum-mode experiments.

This module is intentionally dependency-light: NumPy plus the Python standard
library only.  It defines identities, active-coordinate conventions, force-work
accounting, observation-admission ledgers, resolved run metadata, and the
*schemas* for mode prediction / mode-solve scheduling.  Predictor and scheduler
algorithms are owned by later implementation tasks and must only implement the
protocols declared here.

The canonical physical force convention throughout SaddleMill is ``F = -g``.
Objects in this module never reinterpret a physical force as an effective/MMF
force and never perform calculator calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

Array = np.ndarray

FOUNDATION_SCHEMA_VERSION = 1
DEFAULT_ALIGNMENT_TOLERANCE = 1.0e-8
DEFAULT_DISPLACEMENT_TOLERANCE = 1.0e-12
DEFAULT_TANGENT_TOLERANCE = 1.0e-14
DEFAULT_PREDICTOR_MAX_ANGLE_RADIANS = math.radians(5.0)
DEFAULT_PARALLEL_ABS_TOLERANCE = 1.0e-12  # eV/Angstrom on the RMS metric
DEFAULT_PARALLEL_REL_TOLERANCE = 1.0e-8


class TokenEnum(str, Enum):
    """String enum whose serialized form is always its value."""

    def __str__(self) -> str:
        return self.value


class WorkPurpose(TokenEnum):
    ALGORITHM = "algorithm"
    DIAGNOSTIC = "diagnostic"


class EvaluationKind(TokenEnum):
    PHYSICAL_EXACT = "physical_exact"
    PHYSICAL_CACHED = "physical_cached"
    DERIVED_EXTRAPOLATED = "derived_extrapolated"
    DERIVED_OTHER = "derived_other"
    MODEL_DERIVED = "model_derived"


class OperatorOrigin(TokenEnum):
    FINITE_DIFFERENCE_PHYSICAL_FORCE = "finite_difference_physical_force"
    EXPLICIT_PHYSICAL_MATRIX = "explicit_physical_matrix"
    APPROXIMATE_PHYSICAL_MODEL = "approximate_physical_model"
    EFFECTIVE_FORCE_JACOBIAN = "effective_force_jacobian"


class ResidualSemantics(TokenEnum):
    PHYSICAL_HESSIAN_EIGEN = "physical_hessian_eigen"
    PHYSICAL_MODEL_EIGEN = "physical_model_eigen"
    EFFECTIVE_FORCE_JACOBIAN = "effective_force_jacobian"


class RootSelection(TokenEnum):
    LOWEST = "lowest"
    HOMED_OVERLAP = "homed_overlap"


class ReferenceStatus(TokenEnum):
    BASELINE = "baseline"
    REFERENCE_INFORMED = "reference_informed"
    EXPERIMENTAL = "experimental"


class ParallelIncreaseComparison(TokenEnum):
    TOLERANT = "tolerant"
    STRICT_PAPER = "strict_paper"


def _enum_value(value: str | TokenEnum) -> str:
    return value.value if isinstance(value, TokenEnum) else str(value).strip().lower()


def freeze_array(value: object, *, dtype=float, shape: tuple[int, ...] | None = None) -> Array:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}; got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("array contains non-finite values")
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _freeze_metadata_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return freeze_array(value, dtype=value.dtype)
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_metadata_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata_value(item) for item in value)
    return value


def freeze_mapping(value: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return a recursively immutable provenance mapping.

    Metadata is provenance, not mutable algorithm state. NumPy arrays are
    copied/read-only, nested mappings become mapping proxies, and mutable
    sequences become tuples.
    """

    return MappingProxyType({
        str(key): _freeze_metadata_value(item)
        for key, item in dict(value or {}).items()
    })


def json_safe(value: object) -> object:
    if isinstance(value, TokenEnum):
        return value.value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    return repr(value)


def geometry_fingerprint(positions: object) -> str:
    """Stable SHA-256 identity for one exact Cartesian geometry.

    The fingerprint is deliberately *not* a geometry-equivalence criterion.
    It identifies the exact accepted/probed numeric geometry within an attempt.
    Canonical chemical structure identity remains governed elsewhere by the
    established SaddleMill maxdev contract.
    """

    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions contain non-finite values")
    canonical = np.ascontiguousarray(array.astype("<f8", copy=False))
    digest = hashlib.sha256()
    digest.update(b"saddlemill-cartesian-geometry-v1\0")
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class ActiveCoordinateSpace:
    """Explicit movable Cartesian DOFs plus an optional removable null subspace.

    ``active_dof_mask`` is shape ``(N, 3)``.  Fixed coordinates are False and
    are zeroed in every action/residual. ``null_basis`` contains Cartesian
    vectors in the same shape and is orthonormalized *after* the active mask is
    applied.  Callers must pass the physically appropriate basis (for example
    the existing periodic rigid-translation basis); this class never invents
    diagonal entries or rigid modes.
    """

    active_dof_mask: Array
    null_basis: tuple[Array, ...] = ()
    convention: str = "movable_cartesian_dofs"
    null_mode_policy: str = "none"

    def __post_init__(self) -> None:
        mask = np.asarray(self.active_dof_mask, dtype=bool)
        if mask.ndim != 2 or mask.shape[1] != 3:
            raise ValueError(
                f"active_dof_mask must have shape (N, 3); got {mask.shape}"
            )
        mask = np.array(mask, dtype=bool, copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "active_dof_mask", mask)
        convention = str(self.convention).strip().lower()
        null_policy = str(self.null_mode_policy).strip().lower()
        if not convention:
            raise ValueError("coordinate convention cannot be empty")
        if not null_policy:
            raise ValueError("null_mode_policy cannot be empty")
        object.__setattr__(self, "convention", convention)
        object.__setattr__(self, "null_mode_policy", null_policy)

        orthonormal: list[Array] = []
        for candidate in tuple(self.null_basis or ()):
            vector = np.asarray(candidate, dtype=float)
            if vector.shape == (mask.size,):
                vector = vector.reshape(mask.shape)
            if vector.shape != mask.shape:
                raise ValueError(
                    "null-basis vector must match active_dof_mask shape; got "
                    f"{vector.shape} and {mask.shape}"
                )
            if not np.all(np.isfinite(vector)):
                raise ValueError("null-basis vector contains non-finite values")
            flat = np.where(mask, vector, 0.0).reshape(-1).astype(float, copy=True)
            for base in orthonormal:
                base_flat = base.reshape(-1)
                flat -= float(np.dot(base_flat, flat)) * base_flat
            norm = float(np.linalg.norm(flat))
            if norm <= 1.0e-14:
                continue
            item = (flat / norm).reshape(mask.shape)
            item.setflags(write=False)
            orthonormal.append(item)
        object.__setattr__(self, "null_basis", tuple(orthonormal))

    @classmethod
    def all_cartesian(cls, atom_count: int) -> "ActiveCoordinateSpace":
        atom_count = int(atom_count)
        if atom_count < 1:
            raise ValueError("atom_count must be >= 1")
        return cls(np.ones((atom_count, 3), dtype=bool))

    @classmethod
    def from_active_atom_mask(
        cls,
        active_atom_mask: object,
        *,
        null_basis: Sequence[object] = (),
        null_mode_policy: str = "none",
    ) -> "ActiveCoordinateSpace":
        atom_mask = np.asarray(active_atom_mask, dtype=bool).reshape(-1)
        return cls(
            np.repeat(atom_mask[:, None], 3, axis=1),
            tuple(np.asarray(item, dtype=float) for item in null_basis),
            convention="active_atom_mask_cartesian",
            null_mode_policy=null_mode_policy,
        )

    @property
    def atom_count(self) -> int:
        return int(self.active_dof_mask.shape[0])

    @property
    def active_dof_count(self) -> int:
        return int(np.count_nonzero(self.active_dof_mask))

    @property
    def identity(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"saddlemill-active-coordinate-space-v1\0")
        digest.update(np.ascontiguousarray(self.active_dof_mask, dtype=np.uint8).tobytes())
        for base in self.null_basis:
            digest.update(np.ascontiguousarray(base, dtype="<f8").tobytes())
        digest.update(self.convention.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.null_mode_policy.encode("utf-8"))
        return "coords:" + digest.hexdigest()

    def project(self, value: object) -> Array:
        array = np.asarray(value, dtype=float)
        if array.shape == (self.active_dof_mask.size,):
            array = array.reshape(self.active_dof_mask.shape)
        if array.shape != self.active_dof_mask.shape:
            raise ValueError(
                "vector must match coordinate shape; got "
                f"{array.shape} and {self.active_dof_mask.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("vector contains non-finite values")
        result = np.where(self.active_dof_mask, array, 0.0).astype(float, copy=True)
        flat = result.reshape(-1)
        for base in self.null_basis:
            b = base.reshape(-1)
            flat -= float(np.dot(b, flat)) * b
        return result

    def normalized(self, value: object, *, tolerance: float = 1.0e-14) -> Array:
        projected = self.project(value)
        norm = float(np.linalg.norm(projected))
        if not np.isfinite(norm) or norm <= float(tolerance):
            raise ValueError("active-space vector norm is approximately zero")
        return projected / norm

    def to_state_dict(self) -> dict[str, object]:
        return {
            "schema": "saddlemill_active_coordinate_space_v1",
            "active_dof_mask": self.active_dof_mask.astype(int).tolist(),
            "null_basis": [item.tolist() for item in self.null_basis],
            "convention": self.convention,
            "null_mode_policy": self.null_mode_policy,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ActiveCoordinateSpace":
        if state.get("schema") != "saddlemill_active_coordinate_space_v1":
            raise ValueError("unsupported ActiveCoordinateSpace state schema")
        return cls(
            np.asarray(state["active_dof_mask"], dtype=bool),
            tuple(np.asarray(item, dtype=float) for item in state.get("null_basis", [])),
            convention=str(state.get("convention", "movable_cartesian_dofs")),
            null_mode_policy=str(state.get("null_mode_policy", "none")),
        )


@dataclass
class ForceAccounting:
    """Noninterfering accounting for physical calculator work and shadow work."""

    algorithm_pes_calls: int = 0
    diagnostic_pes_calls: int = 0
    actual_cache_hits: int = 0
    observation_bookkeeping_ns: int = 0
    model_matrix_ns: int = 0
    algorithm_ns: int = 0
    diagnostic_ns: int = 0
    diagnostic_promotions: int = 0

    def record_force_event(
        self,
        *,
        purpose: WorkPurpose | str,
        pes_call_delta: int,
        cache_hit: bool,
        bookkeeping_ns: int = 0,
        elapsed_ns: int = 0,
    ) -> None:
        purpose_token = WorkPurpose(_enum_value(purpose))
        delta = int(pes_call_delta)
        if delta < 0:
            raise ValueError("pes_call_delta must be >= 0")
        if cache_hit and delta != 0:
            raise ValueError("a cache-hit event cannot report a new PES call")
        if purpose_token is WorkPurpose.ALGORITHM:
            self.algorithm_pes_calls += delta
            self.algorithm_ns += max(0, int(elapsed_ns))
        else:
            self.diagnostic_pes_calls += delta
            self.diagnostic_ns += max(0, int(elapsed_ns))
        self.actual_cache_hits += int(bool(cache_hit))
        self.observation_bookkeeping_ns += max(0, int(bookkeeping_ns))

    @property
    def physical_total_pes_calls(self) -> int:
        return int(self.algorithm_pes_calls + self.diagnostic_pes_calls)

    def add_model_matrix_ns(self, value: int, *, purpose: WorkPurpose | str) -> None:
        value = max(0, int(value))
        self.model_matrix_ns += value
        if WorkPurpose(_enum_value(purpose)) is WorkPurpose.ALGORITHM:
            self.algorithm_ns += value
        else:
            self.diagnostic_ns += value

    def to_state_dict(self) -> dict[str, int | str]:
        return {
            "schema": "saddlemill_force_accounting_v1",
            "algorithm_pes_calls": int(self.algorithm_pes_calls),
            "diagnostic_pes_calls": int(self.diagnostic_pes_calls),
            "physical_total_pes_calls": int(self.physical_total_pes_calls),
            "actual_cache_hits": int(self.actual_cache_hits),
            "observation_bookkeeping_ns": int(self.observation_bookkeeping_ns),
            "model_matrix_ns": int(self.model_matrix_ns),
            "algorithm_ns": int(self.algorithm_ns),
            "diagnostic_ns": int(self.diagnostic_ns),
            "diagnostic_promotions": int(self.diagnostic_promotions),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ForceAccounting":
        if state.get("schema") != "saddlemill_force_accounting_v1":
            raise ValueError("unsupported ForceAccounting state schema")
        result = cls(
            algorithm_pes_calls=int(state.get("algorithm_pes_calls", 0)),
            diagnostic_pes_calls=int(state.get("diagnostic_pes_calls", 0)),
            actual_cache_hits=int(state.get("actual_cache_hits", 0)),
            observation_bookkeeping_ns=int(state.get("observation_bookkeeping_ns", 0)),
            model_matrix_ns=int(state.get("model_matrix_ns", 0)),
            algorithm_ns=int(state.get("algorithm_ns", 0)),
            diagnostic_ns=int(state.get("diagnostic_ns", 0)),
            diagnostic_promotions=int(state.get("diagnostic_promotions", 0)),
        )
        expected = int(state.get("physical_total_pes_calls", result.physical_total_pes_calls))
        if expected != result.physical_total_pes_calls:
            raise ValueError("force-accounting total is inconsistent with component counters")
        return result


@dataclass(frozen=True)
class ObservationAdmissionPolicy:
    """Consumer policy kept separate from raw observation storage."""

    name: str = "algorithm_physical"
    allowed_purposes: tuple[str, ...] = (WorkPurpose.ALGORITHM.value,)
    allowed_evaluations: tuple[str, ...] = (
        EvaluationKind.PHYSICAL_EXACT.value,
        EvaluationKind.PHYSICAL_CACHED.value,
    )
    allowed_sources: tuple[str, ...] = ()
    allowed_families: tuple[str, ...] = ()
    allow_cached: bool = True
    allow_derived: bool = False
    allow_diagnostic_promotion: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if not name:
            raise ValueError("admission policy name cannot be empty")
        object.__setattr__(self, "name", name)
        for field_name in (
            "allowed_purposes",
            "allowed_evaluations",
            "allowed_sources",
            "allowed_families",
        ):
            values = tuple(dict.fromkeys(str(item).strip().lower() for item in getattr(self, field_name) if str(item).strip()))
            object.__setattr__(self, field_name, values)

    def accepts(
        self,
        *,
        purpose: str,
        evaluation: str,
        source: str,
        family: str,
        promoted: bool = False,
    ) -> bool:
        purpose = str(purpose).strip().lower()
        evaluation = str(evaluation).strip().lower()
        source = str(source).strip().lower()
        family = str(family).strip().lower()
        if purpose not in self.allowed_purposes:
            if not (
                purpose == WorkPurpose.DIAGNOSTIC.value
                and promoted
                and self.allow_diagnostic_promotion
            ):
                return False
        if evaluation not in self.allowed_evaluations:
            if not (self.allow_derived and evaluation.startswith("derived_")):
                return False
        if not self.allow_cached and evaluation == EvaluationKind.PHYSICAL_CACHED.value:
            return False
        if self.allowed_sources and source not in self.allowed_sources:
            return False
        if self.allowed_families and family not in self.allowed_families:
            return False
        return True


ALGORITHM_PHYSICAL_ADMISSION = ObservationAdmissionPolicy()


@dataclass(frozen=True)
class AdmissionOutcome:
    admitted: bool
    duplicate: bool
    admission_id: str
    subject_id: str
    consumer_id: str
    promoted_from_diagnostic: bool = False
    reason: str = ""


class ConsumerAdmissionLedger:
    """Idempotent admission ledger for observations/pairs/probe blocks.

    The ledger does not modify the raw force bank.  Its key includes the
    consumer and a caller-supplied subject/block identity, so replaying cached
    observations or the same probe block cannot train one consumer twice.
    """

    SCHEMA = "saddlemill_consumer_admission_ledger_v1"

    def __init__(self) -> None:
        self._records: dict[str, dict[str, object]] = {}
        self._serial = 0

    @staticmethod
    def _key(
        *,
        consumer_id: str,
        subject_id: str,
        observation_ids: Sequence[str],
        block_id: str,
    ) -> str:
        payload = json.dumps(
            {
                "consumer_id": consumer_id,
                "subject_id": subject_id,
                "observation_ids": list(observation_ids),
                "block_id": block_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def admit(
        self,
        *,
        consumer_id: str,
        subject_id: str,
        observation_ids: Sequence[str] = (),
        block_id: str = "",
        purpose: WorkPurpose | str = WorkPurpose.ALGORITHM,
        promoted_from_diagnostic: bool = False,
        promotion_policy: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> AdmissionOutcome:
        consumer = str(consumer_id).strip().lower()
        subject = str(subject_id).strip()
        block = str(block_id).strip()
        ids = tuple(str(item).strip() for item in observation_ids if str(item).strip())
        if not consumer or not subject:
            raise ValueError("consumer_id and subject_id are required")
        purpose_token = WorkPurpose(_enum_value(purpose))
        promotion_policy = str(promotion_policy).strip().lower()
        if promoted_from_diagnostic and purpose_token is not WorkPurpose.ALGORITHM:
            raise ValueError("diagnostic promotion is meaningful only for algorithm admission")
        if promoted_from_diagnostic and not promotion_policy:
            raise ValueError("diagnostic-to-algorithm promotion requires a named promotion_policy")
        key = self._key(
            consumer_id=consumer,
            subject_id=subject,
            observation_ids=ids,
            block_id=block,
        )
        existing = self._records.get(key)
        if existing is not None:
            return AdmissionOutcome(
                admitted=False,
                duplicate=True,
                admission_id=str(existing["admission_id"]),
                subject_id=subject,
                consumer_id=consumer,
                promoted_from_diagnostic=bool(existing.get("promoted_from_diagnostic", False)),
                reason="duplicate_admission",
            )
        admission_id = f"admit{self._serial}"
        self._serial += 1
        self._records[key] = {
            "admission_id": admission_id,
            "consumer_id": consumer,
            "subject_id": subject,
            "observation_ids": list(ids),
            "block_id": block,
            "purpose": purpose_token.value,
            "promoted_from_diagnostic": bool(promoted_from_diagnostic),
            "promotion_policy": promotion_policy,
            "metadata": json_safe(metadata or {}),
        }
        return AdmissionOutcome(
            admitted=True,
            duplicate=False,
            admission_id=admission_id,
            subject_id=subject,
            consumer_id=consumer,
            promoted_from_diagnostic=bool(promoted_from_diagnostic),
        )

    @property
    def record_count(self) -> int:
        return len(self._records)

    def to_state_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "next_serial": int(self._serial),
            "records": [self._records[key] for key in sorted(self._records)],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ConsumerAdmissionLedger":
        if state.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported ConsumerAdmissionLedger state schema")
        result = cls()
        result._serial = int(state.get("next_serial", 0))
        for raw in state.get("records", []):
            record = dict(raw)
            key = cls._key(
                consumer_id=str(record["consumer_id"]),
                subject_id=str(record["subject_id"]),
                observation_ids=tuple(record.get("observation_ids", [])),
                block_id=str(record.get("block_id", "")),
            )
            if key in result._records:
                raise ValueError("duplicate admission record in serialized ledger")
            result._records[key] = record
        return result


@dataclass(frozen=True)
class CommonResultMetadata:
    """Common provenance envelope for HVPs, residuals, Ritz data and estimators."""

    geometry_id: str
    state_id: int
    state_uid: str
    coordinate_space_id: str
    purpose: WorkPurpose | str
    source: str
    family: str
    operator_origin: OperatorOrigin | str
    physical: bool
    model_derived: bool
    model_age: int | None = None
    pes_call_delta: int = 0
    cache_hit: bool = False
    timing_ns: int = 0
    units: str = ""
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        purpose = WorkPurpose(_enum_value(self.purpose))
        origin = OperatorOrigin(_enum_value(self.operator_origin))
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "operator_origin", origin)
        object.__setattr__(self, "state_id", int(self.state_id))
        object.__setattr__(self, "geometry_id", str(self.geometry_id))
        object.__setattr__(self, "state_uid", str(self.state_uid))
        object.__setattr__(self, "coordinate_space_id", str(self.coordinate_space_id))
        object.__setattr__(self, "source", str(self.source).strip().lower())
        object.__setattr__(self, "family", str(self.family).strip().lower())
        object.__setattr__(self, "physical", bool(self.physical))
        object.__setattr__(self, "model_derived", bool(self.model_derived))
        if self.physical and self.model_derived:
            raise ValueError("a model-derived result cannot claim physical certification")
        if self.model_age is not None and int(self.model_age) < 0:
            raise ValueError("model_age must be >= 0 when provided")
        object.__setattr__(self, "model_age", None if self.model_age is None else int(self.model_age))
        delta = int(self.pes_call_delta)
        if delta < 0:
            raise ValueError("pes_call_delta must be >= 0")
        if bool(self.cache_hit) and delta != 0:
            raise ValueError("cache_hit result cannot report a new PES call")
        object.__setattr__(self, "pes_call_delta", delta)
        object.__setattr__(self, "cache_hit", bool(self.cache_hit))
        object.__setattr__(self, "timing_ns", max(0, int(self.timing_ns)))
        object.__setattr__(self, "units", str(self.units).strip())
        object.__setattr__(self, "provenance", freeze_mapping(self.provenance))

    @property
    def certification_eligible(self) -> bool:
        return bool(
            self.physical
            and not self.model_derived
            and self.operator_origin
            in {
                OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                OperatorOrigin.EXPLICIT_PHYSICAL_MATRIX,
            }
        )

    def to_state_dict(self) -> dict[str, object]:
        return {
            "schema": "saddlemill_common_result_metadata_v1",
            "geometry_id": self.geometry_id,
            "state_id": self.state_id,
            "state_uid": self.state_uid,
            "coordinate_space_id": self.coordinate_space_id,
            "purpose": str(self.purpose),
            "source": self.source,
            "family": self.family,
            "operator_origin": str(self.operator_origin),
            "physical": self.physical,
            "model_derived": self.model_derived,
            "model_age": self.model_age,
            "pes_call_delta": self.pes_call_delta,
            "cache_hit": self.cache_hit,
            "timing_ns": self.timing_ns,
            "units": self.units,
            "provenance": json_safe(self.provenance),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "CommonResultMetadata":
        if state.get("schema") != "saddlemill_common_result_metadata_v1":
            raise ValueError("unsupported CommonResultMetadata state schema")
        return cls(
            geometry_id=str(state.get("geometry_id", "")),
            state_id=int(state.get("state_id", -1)),
            state_uid=str(state.get("state_uid", "")),
            coordinate_space_id=str(state.get("coordinate_space_id", "")),
            purpose=str(state.get("purpose", WorkPurpose.ALGORITHM.value)),
            source=str(state.get("source", "")),
            family=str(state.get("family", "")),
            operator_origin=str(state.get("operator_origin", OperatorOrigin.APPROXIMATE_PHYSICAL_MODEL.value)),
            physical=bool(state.get("physical", False)),
            model_derived=bool(state.get("model_derived", False)),
            model_age=state.get("model_age"),
            pes_call_delta=int(state.get("pes_call_delta", 0)),
            cache_hit=bool(state.get("cache_hit", False)),
            timing_ns=int(state.get("timing_ns", 0)),
            units=str(state.get("units", "")),
            provenance=dict(state.get("provenance", {})),
        )


@dataclass(frozen=True)
class ResolvedRunMetadata:
    """Typed run envelope; shared config/runtime wiring is intentionally external."""

    schema_version: int = FOUNDATION_SCHEMA_VERSION
    algorithms: Mapping[str, str] = field(default_factory=dict)
    policies: Mapping[str, str] = field(default_factory=dict)
    defaults: Mapping[str, object] = field(default_factory=dict)
    observation_origins: tuple[str, ...] = ()
    estimator_origins: tuple[str, ...] = ()
    stopping_reason: str = ""
    reference_status: ReferenceStatus | str = ReferenceStatus.BASELINE
    unsupported_combinations: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if self.schema_version != FOUNDATION_SCHEMA_VERSION:
            raise ValueError("unsupported resolved-run metadata schema version")
        object.__setattr__(self, "algorithms", freeze_mapping({str(k): str(v) for k, v in self.algorithms.items()}))
        object.__setattr__(self, "policies", freeze_mapping({str(k): str(v) for k, v in self.policies.items()}))
        object.__setattr__(self, "defaults", freeze_mapping(self.defaults))
        object.__setattr__(self, "observation_origins", tuple(str(v).strip().lower() for v in self.observation_origins if str(v).strip()))
        object.__setattr__(self, "estimator_origins", tuple(str(v).strip().lower() for v in self.estimator_origins if str(v).strip()))
        object.__setattr__(self, "stopping_reason", str(self.stopping_reason).strip().lower())
        object.__setattr__(self, "reference_status", ReferenceStatus(_enum_value(self.reference_status)))
        object.__setattr__(self, "unsupported_combinations", tuple(str(v) for v in self.unsupported_combinations))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_state_dict(self) -> dict[str, object]:
        return {
            "schema": "saddlemill_resolved_run_metadata_v1",
            "schema_version": self.schema_version,
            "algorithms": json_safe(self.algorithms),
            "policies": json_safe(self.policies),
            "defaults": json_safe(self.defaults),
            "observation_origins": list(self.observation_origins),
            "estimator_origins": list(self.estimator_origins),
            "stopping_reason": self.stopping_reason,
            "reference_status": str(self.reference_status),
            "unsupported_combinations": list(self.unsupported_combinations),
            "metadata": json_safe(self.metadata),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ResolvedRunMetadata":
        if state.get("schema") != "saddlemill_resolved_run_metadata_v1":
            raise ValueError("unsupported ResolvedRunMetadata state schema")
        return cls(
            schema_version=int(state.get("schema_version", FOUNDATION_SCHEMA_VERSION)),
            algorithms=dict(state.get("algorithms", {})),
            policies=dict(state.get("policies", {})),
            defaults=dict(state.get("defaults", {})),
            observation_origins=tuple(state.get("observation_origins", ())),
            estimator_origins=tuple(state.get("estimator_origins", ())),
            stopping_reason=str(state.get("stopping_reason", "")),
            reference_status=str(state.get("reference_status", ReferenceStatus.BASELINE.value)),
            unsupported_combinations=tuple(state.get("unsupported_combinations", ())),
            metadata=dict(state.get("metadata", {})),
        )


@dataclass(frozen=True)
class TranslationSecantPredictorInput:
    """Sealed input contract for mode-predictor's direct large-dimer predictor.

    mode-predictor must implement the equation documented in ``INTERFACES.md``.  typed-HVP only
    owns this schema and validation; it intentionally does not compute a mode.
    Gradients are physical gradients ``g=-F`` in eV/Angstrom.
    """

    old_state_id: int
    new_state_id: int
    old_state_uid: str
    new_state_uid: str
    old_geometry_id: str
    new_geometry_id: str
    old_positions: Array
    new_positions: Array
    old_gradient: Array
    new_gradient: Array
    current_mode: Array
    coordinate_space: ActiveCoordinateSpace
    angular_step_scale: float
    max_angle_radians: float = DEFAULT_PREDICTOR_MAX_ANGLE_RADIANS
    alignment_tolerance: float = DEFAULT_ALIGNMENT_TOLERANCE
    displacement_tolerance: float = DEFAULT_DISPLACEMENT_TOLERANCE
    tangent_tolerance: float = DEFAULT_TANGENT_TOLERANCE

    def __post_init__(self) -> None:
        shape = self.coordinate_space.active_dof_mask.shape
        for name in ("old_positions", "new_positions", "old_gradient", "new_gradient", "current_mode"):
            object.__setattr__(self, name, freeze_array(getattr(self, name), shape=shape))
        object.__setattr__(self, "old_state_id", int(self.old_state_id))
        object.__setattr__(self, "new_state_id", int(self.new_state_id))
        for name in ("old_state_uid", "new_state_uid", "old_geometry_id", "new_geometry_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        scale = float(self.angular_step_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("angular_step_scale must be finite and >= 0")
        object.__setattr__(self, "angular_step_scale", scale)
        for name in ("max_angle_radians", "alignment_tolerance", "displacement_tolerance", "tangent_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        if self.max_angle_radians > math.pi:
            raise ValueError("max_angle_radians cannot exceed pi")


@dataclass(frozen=True)
class ModePredictionResult:
    mode: Array
    status: str
    original_signed_alignment: float | None
    absolute_alignment: float | None
    reversed_pair: bool
    raw_tangent: Array | None
    requested_angle_radians: float
    accepted_angle_radians: float
    capped: bool
    old_state_id: int
    new_state_id: int
    old_state_uid: str
    new_state_uid: str
    old_geometry_id: str
    new_geometry_id: str
    angular_step_scale: float
    max_angle_radians: float
    alignment_tolerance: float
    displacement_tolerance: float
    tangent_tolerance: float
    origin: str = "translation_secant_path_average"
    experimental: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = np.asarray(self.mode, dtype=float)
        if mode.ndim != 2 or mode.shape[1] != 3 or not np.all(np.isfinite(mode)):
            raise ValueError("prediction mode must have finite shape (N, 3)")
        object.__setattr__(self, "mode", freeze_array(mode, shape=mode.shape))
        if self.raw_tangent is not None:
            object.__setattr__(self, "raw_tangent", freeze_array(self.raw_tangent, shape=mode.shape))
        object.__setattr__(self, "status", str(self.status).strip().lower())
        object.__setattr__(self, "old_state_id", int(self.old_state_id))
        object.__setattr__(self, "new_state_id", int(self.new_state_id))
        for name in ("old_state_uid", "new_state_uid", "old_geometry_id", "new_geometry_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        for name in ("requested_angle_radians", "accepted_angle_radians", "angular_step_scale", "max_angle_radians", "alignment_tolerance", "displacement_tolerance", "tangent_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        if self.accepted_angle_radians > self.max_angle_radians + 1.0e-15:
            raise ValueError("accepted predictor angle cannot exceed max_angle_radians")
        object.__setattr__(self, "origin", str(self.origin).strip().lower())
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@runtime_checkable
class ModePredictorProtocol(Protocol):
    def predict(self, request: TranslationSecantPredictorInput) -> ModePredictionResult:
        ...


@dataclass(frozen=True)
class ParallelForceMetricPolicy:
    metric: str = "movable_cartesian_dof_rms"
    comparison: ParallelIncreaseComparison | str = ParallelIncreaseComparison.TOLERANT
    absolute_tolerance: float = DEFAULT_PARALLEL_ABS_TOLERANCE
    relative_tolerance: float = DEFAULT_PARALLEL_REL_TOLERANCE

    def __post_init__(self) -> None:
        metric = str(self.metric).strip().lower()
        if metric != "movable_cartesian_dof_rms":
            raise ValueError("typed-HVP seals movable_cartesian_dof_rms as the default metric")
        object.__setattr__(self, "metric", metric)
        comparison = ParallelIncreaseComparison(_enum_value(self.comparison))
        object.__setattr__(self, "comparison", comparison)
        abs_tol = float(self.absolute_tolerance)
        rel_tol = float(self.relative_tolerance)
        if abs_tol < 0.0 or rel_tol < 0.0 or not np.isfinite(abs_tol + rel_tol):
            raise ValueError("parallel-force tolerances must be finite and >= 0")
        if comparison is ParallelIncreaseComparison.STRICT_PAPER:
            abs_tol = 0.0
            rel_tol = 0.0
        object.__setattr__(self, "absolute_tolerance", abs_tol)
        object.__setattr__(self, "relative_tolerance", rel_tol)


@dataclass(frozen=True)
class ModeScheduleState:
    """Serializable state contract implemented by mode-schedule's scheduler."""

    max_skips: int
    skips_since_real_solve: int
    last_real_solve_state_id: int | None
    last_checked_state_id: int | None
    last_pre_prediction_mode: Array | None
    last_center_force: Array | None
    displacement_reference: Array | None
    sequence_id: str
    damping_lambda: float | None
    damping_lambda_identity: str
    model_age: int | None
    parallel_metric_policy: ParallelForceMetricPolicy = field(default_factory=ParallelForceMetricPolicy)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_skips", int(self.max_skips))
        object.__setattr__(self, "skips_since_real_solve", int(self.skips_since_real_solve))
        if self.max_skips < 0 or self.skips_since_real_solve < 0:
            raise ValueError("skip counters must be >= 0")
        if self.skips_since_real_solve > self.max_skips:
            raise ValueError("skips_since_real_solve cannot exceed max_skips")
        for name in ("last_real_solve_state_id", "last_checked_state_id", "model_age"):
            value = getattr(self, name)
            if value is not None:
                if int(value) < 0:
                    raise ValueError(f"{name} must be >= 0")
                object.__setattr__(self, name, int(value))
        for name in ("last_pre_prediction_mode", "last_center_force", "displacement_reference"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float)
                if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} must have finite shape (N, 3)")
                object.__setattr__(self, name, freeze_array(array, shape=array.shape))
        sequence_id = str(self.sequence_id).strip()
        lambda_identity = str(self.damping_lambda_identity).strip()
        if not sequence_id:
            raise ValueError("sequence_id cannot be empty")
        if not lambda_identity:
            raise ValueError("damping_lambda_identity cannot be empty")
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "damping_lambda_identity", lambda_identity)
        if self.damping_lambda is not None:
            value = float(self.damping_lambda)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("damping_lambda must be finite and >= 0")
            object.__setattr__(self, "damping_lambda", value)
        if int(self.schema_version) != 1:
            raise ValueError("unsupported ModeScheduleState schema version")

    def to_state_dict(self) -> dict[str, object]:
        return {
            "schema": "saddlemill_mode_schedule_state_v1",
            "max_skips": self.max_skips,
            "skips_since_real_solve": self.skips_since_real_solve,
            "last_real_solve_state_id": self.last_real_solve_state_id,
            "last_checked_state_id": self.last_checked_state_id,
            "last_pre_prediction_mode": None if self.last_pre_prediction_mode is None else self.last_pre_prediction_mode.tolist(),
            "last_center_force": None if self.last_center_force is None else self.last_center_force.tolist(),
            "displacement_reference": None if self.displacement_reference is None else self.displacement_reference.tolist(),
            "sequence_id": self.sequence_id,
            "damping_lambda": self.damping_lambda,
            "damping_lambda_identity": self.damping_lambda_identity,
            "model_age": self.model_age,
            "parallel_metric_policy": {
                "metric": self.parallel_metric_policy.metric,
                "comparison": str(self.parallel_metric_policy.comparison),
                "absolute_tolerance": self.parallel_metric_policy.absolute_tolerance,
                "relative_tolerance": self.parallel_metric_policy.relative_tolerance,
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ModeScheduleState":
        if state.get("schema") != "saddlemill_mode_schedule_state_v1":
            raise ValueError("unsupported ModeScheduleState state schema")
        policy = dict(state.get("parallel_metric_policy", {}))
        return cls(
            max_skips=int(state["max_skips"]),
            skips_since_real_solve=int(state["skips_since_real_solve"]),
            last_real_solve_state_id=state.get("last_real_solve_state_id"),
            last_checked_state_id=state.get("last_checked_state_id"),
            last_pre_prediction_mode=None if state.get("last_pre_prediction_mode") is None else np.asarray(state["last_pre_prediction_mode"], dtype=float),
            last_center_force=None if state.get("last_center_force") is None else np.asarray(state["last_center_force"], dtype=float),
            displacement_reference=None if state.get("displacement_reference") is None else np.asarray(state["displacement_reference"], dtype=float),
            sequence_id=str(state.get("sequence_id", "")),
            damping_lambda=state.get("damping_lambda"),
            damping_lambda_identity=str(state.get("damping_lambda_identity", "")),
            model_age=state.get("model_age"),
            parallel_metric_policy=ParallelForceMetricPolicy(**policy),
        )


@dataclass(frozen=True)
class ModeScheduleDecision:
    require_real_solve: bool
    skipped_scheduled_solve: bool
    reason: str
    prediction_allowed: bool
    mandatory_fresh_validation: bool
    state: ModeScheduleState
    curvature_origin: str = ""
    model_origin: str = ""
    model_age: int | None = None
    trigger_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", str(self.reason).strip().lower())
        if not self.reason:
            raise ValueError("schedule decision reason cannot be empty")
        if self.mandatory_fresh_validation and not self.require_real_solve:
            raise ValueError("mandatory fresh validation must require a real solve")
        if self.require_real_solve and self.skipped_scheduled_solve:
            raise ValueError("one decision cannot both solve and skip")
        object.__setattr__(self, "curvature_origin", str(self.curvature_origin).strip().lower())
        object.__setattr__(self, "model_origin", str(self.model_origin).strip().lower())
        if self.model_age is not None:
            if int(self.model_age) < 0:
                raise ValueError("model_age must be >= 0")
            object.__setattr__(self, "model_age", int(self.model_age))
        object.__setattr__(self, "trigger_metadata", freeze_mapping(self.trigger_metadata))


@runtime_checkable
class ModeSchedulerProtocol(Protocol):
    def decide(self, *args: object, **kwargs: object) -> ModeScheduleDecision:
        ...


__all__ = [
    "ALGORITHM_PHYSICAL_ADMISSION",
    "ActiveCoordinateSpace",
    "AdmissionOutcome",
    "CommonResultMetadata",
    "ConsumerAdmissionLedger",
    "DEFAULT_ALIGNMENT_TOLERANCE",
    "DEFAULT_DISPLACEMENT_TOLERANCE",
    "DEFAULT_PARALLEL_ABS_TOLERANCE",
    "DEFAULT_PARALLEL_REL_TOLERANCE",
    "DEFAULT_PREDICTOR_MAX_ANGLE_RADIANS",
    "EvaluationKind",
    "FOUNDATION_SCHEMA_VERSION",
    "ForceAccounting",
    "ModePredictionResult",
    "ModePredictorProtocol",
    "ModeScheduleDecision",
    "ModeScheduleState",
    "ModeSchedulerProtocol",
    "ObservationAdmissionPolicy",
    "OperatorOrigin",
    "ParallelForceMetricPolicy",
    "ParallelIncreaseComparison",
    "ReferenceStatus",
    "ResidualSemantics",
    "ResolvedRunMetadata",
    "RootSelection",
    "TranslationSecantPredictorInput",
    "WorkPurpose",
    "freeze_array",
    "freeze_mapping",
    "geometry_fingerprint",
    "json_safe",
]
