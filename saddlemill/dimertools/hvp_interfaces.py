"""Common physical-HVP, residual, Ritz/gap, and dense-memory contracts.

The contracts here distinguish measured/reconstructed physical Hessian actions
from approximate physical models and from effective-force Jacobians.  The latter
are represented only as explicitly nonphysical residuals; they are never
eligible for physical saddle/mode certification.

No ASE, calculator, FairChem, or solver implementation is imported here.  This
keeps the interface usable by CPU contract tests and by all minimum-mode
backends without creating a dependency cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter_ns
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from saddlemill.dimertools.foundation_types import (
    ALGORITHM_PHYSICAL_ADMISSION,
    ActiveCoordinateSpace,
    CommonResultMetadata,
    ObservationAdmissionPolicy,
    OperatorOrigin,
    ResidualSemantics,
    RootSelection,
    WorkPurpose,
    freeze_array,
    freeze_mapping,
)

Array = np.ndarray

HVP_SCHEMA_VERSION = 1
DEFAULT_DENSE_MATRIX_BUDGET_BYTES = 512 * 1024 * 1024
DEFAULT_RANK_TOLERANCE = 1.0e-10
DEFAULT_GAP_TOLERANCE = 1.0e-8
DEFAULT_RESIDUAL_EPSILON = 1.0e-12


class HVPInterfaceError(RuntimeError):
    pass


class HVPIdentityError(HVPInterfaceError):
    pass


class MatrixMemoryBudgetError(HVPInterfaceError):
    pass


@dataclass(frozen=True)
class MatrixMemoryBudget:
    """Fail-closed memory budget for explicitly materialized dense matrices.

    The default 512 MiB is an implementation/resource guard, not a scientific
    threshold.  Exceeding it raises rather than silently switching mathematics.
    A caller may supply a different explicit budget or select a matrix-free
    backend.  ``matrix_count`` includes simultaneous dense matrices retained by
    the caller/backend.
    """

    max_bytes: int = DEFAULT_DENSE_MATRIX_BUDGET_BYTES
    dtype: str = "float64"

    def __post_init__(self) -> None:
        max_bytes = int(self.max_bytes)
        if max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        dtype = np.dtype(self.dtype)
        object.__setattr__(self, "max_bytes", max_bytes)
        object.__setattr__(self, "dtype", dtype.name)

    @property
    def itemsize(self) -> int:
        return int(np.dtype(self.dtype).itemsize)

    def required_bytes(self, dimension: int, *, matrix_count: int = 1) -> int:
        dimension = int(dimension)
        matrix_count = int(matrix_count)
        if dimension < 1 or matrix_count < 1:
            raise ValueError("dimension and matrix_count must be >= 1")
        return int(dimension * dimension * self.itemsize * matrix_count)

    def ensure(self, dimension: int, *, matrix_count: int = 1, label: str = "dense matrix") -> int:
        required = self.required_bytes(dimension, matrix_count=matrix_count)
        if self.max_bytes and required > self.max_bytes:
            raise MatrixMemoryBudgetError(
                f"{label} requires {required} bytes, exceeding explicit budget "
                f"{self.max_bytes} bytes"
            )
        return required


@dataclass(frozen=True)
class HVPRequest:
    """One same-center Hessian-vector action request.

    ``direction`` is Cartesian ``(N, 3)``.  Backends apply the supplied active
    coordinate mask/null-space projection before the action.  The request does
    not imply normalization; eigensolvers normally pass normalized vectors.
    """

    state_id: int
    state_uid: str
    geometry_id: str
    direction: Array
    coordinate_space: ActiveCoordinateSpace
    purpose: WorkPurpose | str = WorkPurpose.ALGORITHM
    source: str = "hvp"
    family: str = "hvp"
    stencil_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", int(self.state_id))
        object.__setattr__(self, "state_uid", str(self.state_uid))
        object.__setattr__(self, "geometry_id", str(self.geometry_id))
        object.__setattr__(
            self,
            "direction",
            freeze_array(
                self.direction,
                shape=self.coordinate_space.active_dof_mask.shape,
            ),
        )
        object.__setattr__(self, "purpose", WorkPurpose(str(self.purpose)))
        source = str(self.source).strip().lower()
        family = str(self.family).strip().lower()
        if not source or not family:
            raise ValueError("HVP request source/family cannot be empty")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "stencil_id", str(self.stencil_id).strip())
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True)
class HVPResult:
    """Typed result from a physical/reference/model HVP backend."""

    available: bool
    action: Array | None
    direction: Array
    metadata: CommonResultMetadata
    stencil_scheme: str = ""
    displacement_scale: float | None = None
    endpoint_observation_ids: tuple[str, ...] = ()
    unavailable_reason: str = ""
    scaled_once: bool = True

    def __post_init__(self) -> None:
        direction = freeze_array(self.direction)
        if direction.ndim != 2 or direction.shape[1] != 3:
            raise ValueError("HVP direction must have shape (N, 3)")
        object.__setattr__(self, "direction", direction)
        if self.available:
            if self.action is None:
                raise ValueError("available HVP result requires an action")
            action = freeze_array(self.action, shape=direction.shape)
            object.__setattr__(self, "action", action)
            if self.unavailable_reason:
                raise ValueError("available HVP result cannot have unavailable_reason")
        else:
            if self.action is not None:
                raise ValueError("unavailable HVP result cannot carry an action")
            if not str(self.unavailable_reason).strip():
                raise ValueError("unavailable HVP result requires a reason")
        scheme = str(self.stencil_scheme).strip().lower()
        object.__setattr__(self, "stencil_scheme", scheme)
        if self.displacement_scale is not None:
            scale = float(self.displacement_scale)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("displacement_scale must be finite and > 0")
            object.__setattr__(self, "displacement_scale", scale)
        object.__setattr__(
            self,
            "endpoint_observation_ids",
            tuple(str(item) for item in self.endpoint_observation_ids),
        )
        object.__setattr__(self, "unavailable_reason", str(self.unavailable_reason).strip().lower())
        if not bool(self.scaled_once):
            raise ValueError(
                "HVPResult actions must already include the finite-difference scale exactly once"
            )

    @property
    def certification_eligible(self) -> bool:
        return bool(self.available and self.metadata.certification_eligible)

    @classmethod
    def unavailable(
        cls,
        request: HVPRequest,
        *,
        origin: OperatorOrigin,
        reason: str,
        source: str,
        family: str,
        model_age: int | None = None,
        physical: bool = False,
        model_derived: bool = False,
        provenance: Mapping[str, object] | None = None,
    ) -> "HVPResult":
        return cls(
            available=False,
            action=None,
            direction=request.coordinate_space.project(request.direction),
            metadata=CommonResultMetadata(
                geometry_id=request.geometry_id,
                state_id=request.state_id,
                state_uid=request.state_uid,
                coordinate_space_id=request.coordinate_space.identity,
                purpose=request.purpose,
                source=source,
                family=family,
                operator_origin=origin,
                physical=physical,
                model_derived=model_derived,
                model_age=model_age,
                units="eV/Angstrom^2 * Cartesian-vector",
                provenance=provenance or {},
            ),
            unavailable_reason=reason,
        )


@runtime_checkable
class HVPBackend(Protocol):
    def apply(self, request: HVPRequest) -> HVPResult:
        ...


def _state_uid(state: object) -> str:
    value = getattr(state, "state_uid", "")
    if value:
        return str(value)
    sid = int(getattr(state, "state_id"))
    gid = str(getattr(state, "geometry_id", ""))
    return f"state{sid}:{gid}"


def _observation_admissible(
    observation: object,
    policy: ObservationAdmissionPolicy,
    *,
    promoted: bool = False,
) -> bool:
    return policy.accepts(
        purpose=str(getattr(observation, "purpose", WorkPurpose.ALGORITHM.value)),
        evaluation=str(getattr(observation, "evaluation")),
        source=str(getattr(observation, "source")),
        family=str(getattr(observation, "family")),
        promoted=promoted,
    )


class FiniteDifferenceForceHVPBackend:
    """Reconstruct same-center physical ``H q`` from canonical force stencils.

    Because SaddleMill stores physical force ``F=-g``, the finite-difference
    Hessian action is the *negative* force derivative:

    centered: ``Hq = -(F+ - F-)/(2 h)``
    plus:     ``Hq = -(F+ - F0)/h``
    minus:    ``Hq =  (F- - F0)/h``

    The returned action is already divided by ``h`` exactly once.  A Dimer
    torque reconstructed as ``perp(F+ - F-)/(2h)`` is therefore approximately
    ``-perp(Hq)`` and must not be divided by ``h`` again.
    """

    def __init__(
        self,
        history: object,
        *,
        admission_policy: ObservationAdmissionPolicy = ALGORITHM_PHYSICAL_ADMISSION,
    ) -> None:
        self.history = history
        self.admission_policy = admission_policy

    def _find_stencil(self, stencil_id: str) -> object | None:
        for item in getattr(self.history, "stencils", ()):  # no hard import cycle
            if str(getattr(item, "stencil_id", "")) == stencil_id:
                return item
        return None

    def apply(self, request: HVPRequest) -> HVPResult:
        started = perf_counter_ns()
        stencil_id = request.stencil_id
        if not stencil_id:
            return HVPResult.unavailable(
                request,
                origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                reason="missing_stencil_id",
                source=request.source,
                family=request.family,
            )
        stencil = self._find_stencil(stencil_id)
        if stencil is None:
            return HVPResult.unavailable(
                request,
                origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                reason="stencil_not_found",
                source=request.source,
                family=request.family,
                provenance={"stencil_id": stencil_id},
            )
        state_id = int(getattr(stencil, "state_id"))
        if state_id != request.state_id:
            return HVPResult.unavailable(
                request,
                origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                reason="state_identity_mismatch",
                source=str(getattr(stencil, "source", request.source)),
                family=str(getattr(stencil, "family", request.family)),
                provenance={"stencil_id": stencil_id, "stencil_state_id": state_id},
            )
        state = next(
            (item for item in getattr(self.history, "states", ()) if int(getattr(item, "state_id")) == state_id),
            None,
        )
        if state is None:
            return HVPResult.unavailable(
                request,
                origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                reason="state_not_retained",
                source=str(getattr(stencil, "source", request.source)),
                family=str(getattr(stencil, "family", request.family)),
            )
        retained_uid = _state_uid(state)
        if request.state_uid != retained_uid:
            return HVPResult.unavailable(
                request,
                origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                reason="state_uid_mismatch",
                source=str(getattr(stencil, "source", request.source)),
                family=str(getattr(stencil, "family", request.family)),
                provenance={"stencil_id": stencil_id, "retained_state_uid": retained_uid},
            )
        state_geometry = str(getattr(state, "geometry_id", ""))
        if state_geometry and request.geometry_id != state_geometry:
            return HVPResult.unavailable(
                request,
                origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                reason="geometry_identity_mismatch",
                source=str(getattr(stencil, "source", request.source)),
                family=str(getattr(stencil, "family", request.family)),
                provenance={"stencil_id": stencil_id, "state_geometry_id": state_geometry},
            )
        expected_direction = request.coordinate_space.normalized(getattr(stencil, "direction"))
        requested_direction = request.coordinate_space.normalized(request.direction)
        overlap = abs(float(np.dot(expected_direction.reshape(-1), requested_direction.reshape(-1))))
        if overlap < 1.0 - 1.0e-8:
            return HVPResult.unavailable(
                request,
                origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                reason="direction_identity_mismatch",
                source=str(getattr(stencil, "source", request.source)),
                family=str(getattr(stencil, "family", request.family)),
                provenance={"stencil_id": stencil_id, "direction_abs_overlap": overlap},
            )
        # Align the stencil direction to the request; H(-q)=-H(q).
        sign = 1.0 if float(np.dot(expected_direction.reshape(-1), requested_direction.reshape(-1))) >= 0.0 else -1.0
        lookup = getattr(self.history, "observation_by_serial")
        def by_serial(value):
            return None if value is None else lookup(value)
        center = by_serial(getattr(stencil, "center_serial", None))
        plus = by_serial(getattr(stencil, "plus_serial", None))
        minus = by_serial(getattr(stencil, "minus_serial", None))
        observations = [item for item in (center, plus, minus) if item is not None]
        if center is None:
            return HVPResult.unavailable(
                request,
                origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                reason="missing_center_endpoint",
                source=str(getattr(stencil, "source", request.source)),
                family=str(getattr(stencil, "family", request.family)),
                provenance={"stencil_id": stencil_id},
            )
        if any(not _observation_admissible(item, self.admission_policy) for item in observations):
            return HVPResult.unavailable(
                request,
                origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                reason="endpoint_not_admissible",
                source=str(getattr(stencil, "source", request.source)),
                family=str(getattr(stencil, "family", request.family)),
                provenance={
                    "stencil_id": stencil_id,
                    "admission_policy": self.admission_policy.name,
                    "endpoint_observation_ids": [str(getattr(item, "observation_id")) for item in observations],
                },
            )
        scale = float(getattr(stencil, "scale"))
        if not np.isfinite(scale) or scale <= 0.0:
            return HVPResult.unavailable(
                request,
                origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                reason="invalid_stencil_scale",
                source=str(getattr(stencil, "source", request.source)),
                family=str(getattr(stencil, "family", request.family)),
            )
        if plus is not None and minus is not None:
            action = -(np.asarray(plus.forces) - np.asarray(minus.forces)) / (2.0 * scale)
            scheme = "centered"
            endpoint_ids = (plus.observation_id, minus.observation_id)
        elif plus is not None:
            action = -(np.asarray(plus.forces) - np.asarray(center.forces)) / scale
            scheme = "one_sided_plus"
            endpoint_ids = (center.observation_id, plus.observation_id)
        elif minus is not None:
            action = (np.asarray(minus.forces) - np.asarray(center.forces)) / scale
            scheme = "one_sided_minus"
            endpoint_ids = (center.observation_id, minus.observation_id)
        else:
            return HVPResult.unavailable(
                request,
                origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                reason="missing_probe_endpoint",
                source=str(getattr(stencil, "source", request.source)),
                family=str(getattr(stencil, "family", request.family)),
            )
        action = sign * request.coordinate_space.project(action)
        elapsed = perf_counter_ns() - started
        represented_calls = sum(int(getattr(item, "force_call_delta", 0)) for item in observations)
        metadata = CommonResultMetadata(
            geometry_id=request.geometry_id,
            state_id=request.state_id,
            state_uid=request.state_uid,
            coordinate_space_id=request.coordinate_space.identity,
            purpose=request.purpose,
            source=str(getattr(stencil, "source", request.source)),
            family=str(getattr(stencil, "family", request.family)),
            operator_origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
            physical=True,
            model_derived=False,
            pes_call_delta=0,
            cache_hit=True,
            timing_ns=elapsed,
            units="eV/Angstrom^2 * Cartesian-vector",
            provenance={
                "stencil_id": stencil_id,
                "stencil_scheme": scheme,
                "stencil_scale_angstrom": scale,
                "endpoint_observation_ids": list(endpoint_ids),
                "represented_endpoint_pes_calls": represented_calls,
                "direction_sign_to_request": sign,
                "force_sign_convention": "F=-g; Hq=-dF/dq",
                "scaled_once": True,
                "admission_policy": self.admission_policy.name,
            },
        )
        return HVPResult(
            available=True,
            action=action,
            direction=requested_direction,
            metadata=metadata,
            stencil_scheme=scheme,
            displacement_scale=scale,
            endpoint_observation_ids=endpoint_ids,
        )


class _DenseMatrixBackendBase:
    def __init__(
        self,
        matrix: object,
        *,
        state_id: int,
        state_uid: str,
        geometry_id: str,
        coordinate_space: ActiveCoordinateSpace,
        memory_budget: MatrixMemoryBudget | None = None,
        source: str,
        family: str,
        origin: OperatorOrigin,
        physical: bool,
        model_derived: bool,
        model_age: int | None,
        provenance: Mapping[str, object] | None = None,
    ) -> None:
        self.state_id = int(state_id)
        self.state_uid = str(state_uid)
        self.geometry_id = str(geometry_id)
        self.coordinate_space = coordinate_space
        self.source = str(source).strip().lower()
        self.family = str(family).strip().lower()
        self.origin = origin
        self.physical = bool(physical)
        self.model_derived = bool(model_derived)
        self.model_age = model_age
        self.provenance = dict(provenance or {})
        dimension = coordinate_space.active_dof_mask.size
        budget = memory_budget or MatrixMemoryBudget()
        required = budget.ensure(dimension, matrix_count=1, label=f"{origin.value} matrix")
        array = np.asarray(matrix, dtype=float)
        if array.shape != (dimension, dimension):
            raise ValueError(
                f"matrix must have shape {(dimension, dimension)}; got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("matrix contains non-finite values")
        self.matrix = np.array(array, dtype=float, copy=True)
        self.matrix.setflags(write=False)
        self.memory_budget = budget
        self.required_matrix_bytes = required

    def apply(self, request: HVPRequest) -> HVPResult:
        if request.state_id != self.state_id:
            return HVPResult.unavailable(
                request,
                origin=self.origin,
                reason="state_identity_mismatch",
                source=self.source,
                family=self.family,
                model_age=self.model_age,
                physical=self.physical,
                model_derived=self.model_derived,
                provenance={"backend_state_id": self.state_id},
            )
        if request.state_uid != self.state_uid:
            return HVPResult.unavailable(
                request,
                origin=self.origin,
                reason="state_uid_mismatch",
                source=self.source,
                family=self.family,
                model_age=self.model_age,
                physical=self.physical,
                model_derived=self.model_derived,
                provenance={"backend_state_uid": self.state_uid},
            )
        if request.geometry_id != self.geometry_id:
            return HVPResult.unavailable(
                request,
                origin=self.origin,
                reason="geometry_identity_mismatch",
                source=self.source,
                family=self.family,
                model_age=self.model_age,
                physical=self.physical,
                model_derived=self.model_derived,
                provenance={"backend_geometry_id": self.geometry_id},
            )
        if request.coordinate_space.identity != self.coordinate_space.identity:
            return HVPResult.unavailable(
                request,
                origin=self.origin,
                reason="coordinate_space_mismatch",
                source=self.source,
                family=self.family,
                model_age=self.model_age,
                physical=self.physical,
                model_derived=self.model_derived,
            )
        started = perf_counter_ns()
        direction = self.coordinate_space.project(request.direction)
        action = self.matrix @ direction.reshape(-1)
        action = self.coordinate_space.project(action.reshape(direction.shape))
        elapsed = perf_counter_ns() - started
        provenance = dict(self.provenance)
        provenance.update(
            {
                "matrix_dimension": self.matrix.shape[0],
                "matrix_bytes": self.required_matrix_bytes,
                "matrix_budget_bytes": self.memory_budget.max_bytes,
            }
        )
        metadata = CommonResultMetadata(
            geometry_id=request.geometry_id,
            state_id=request.state_id,
            state_uid=request.state_uid,
            coordinate_space_id=request.coordinate_space.identity,
            purpose=request.purpose,
            source=self.source,
            family=self.family,
            operator_origin=self.origin,
            physical=self.physical,
            model_derived=self.model_derived,
            model_age=self.model_age,
            pes_call_delta=0,
            cache_hit=True,
            timing_ns=elapsed,
            units="eV/Angstrom^2 * Cartesian-vector",
            provenance=provenance,
        )
        return HVPResult(
            available=True,
            action=action,
            direction=direction,
            metadata=metadata,
        )


class ExplicitMatrixHVPBackend(_DenseMatrixBackendBase):
    """Explicit same-center physical/reference Hessian action."""

    def __init__(
        self,
        matrix: object,
        *,
        state_id: int,
        state_uid: str,
        geometry_id: str,
        coordinate_space: ActiveCoordinateSpace,
        memory_budget: MatrixMemoryBudget | None = None,
        source: str = "explicit_reference_hessian",
        family: str = "reference_hessian",
        provenance: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            matrix,
            state_id=state_id,
            state_uid=state_uid,
            geometry_id=geometry_id,
            coordinate_space=coordinate_space,
            memory_budget=memory_budget,
            source=source,
            family=family,
            origin=OperatorOrigin.EXPLICIT_PHYSICAL_MATRIX,
            physical=True,
            model_derived=False,
            model_age=None,
            provenance=provenance,
        )


class ApproximatePhysicalModelHVPBackend(_DenseMatrixBackendBase):
    """Approximate physical-Hessian model action; never physical certification."""

    def __init__(
        self,
        matrix: object,
        *,
        state_id: int,
        state_uid: str,
        geometry_id: str,
        coordinate_space: ActiveCoordinateSpace,
        model_age: int,
        memory_budget: MatrixMemoryBudget | None = None,
        source: str = "approximate_physical_hessian",
        family: str = "physical_model",
        provenance: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            matrix,
            state_id=state_id,
            state_uid=state_uid,
            geometry_id=geometry_id,
            coordinate_space=coordinate_space,
            memory_budget=memory_budget,
            source=source,
            family=family,
            origin=OperatorOrigin.APPROXIMATE_PHYSICAL_MODEL,
            physical=False,
            model_derived=True,
            model_age=int(model_age),
            provenance=provenance,
        )


@dataclass(frozen=True)
class ResidualResult:
    available: bool
    semantics: ResidualSemantics | str
    rho: float | None
    residual: Array | None
    residual_norm: float | None
    relative_torque: float | None
    gap: float | None
    residual_over_gap: float | None
    gap_eligible: bool
    selected_root_separation: float | None
    degeneracy_unresolved: bool
    metadata: CommonResultMetadata
    numerator_origin: str
    relative_denominator_origin: str
    gap_denominator_origin: str
    epsilon: float
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantics", ResidualSemantics(str(self.semantics)))
        if self.available:
            if self.residual is None or self.residual_norm is None:
                raise ValueError("available residual requires vector and norm")
            object.__setattr__(self, "residual", freeze_array(self.residual))
        else:
            if not str(self.unavailable_reason).strip():
                raise ValueError("unavailable residual requires a reason")
        object.__setattr__(self, "unavailable_reason", str(self.unavailable_reason).strip().lower())

    @property
    def certification_eligible(self) -> bool:
        return bool(
            self.available
            and self.semantics is ResidualSemantics.PHYSICAL_HESSIAN_EIGEN
            and self.metadata.certification_eligible
        )


def physical_eigen_residual(
    hvp: HVPResult,
    *,
    coordinate_space: ActiveCoordinateSpace,
    gap: float | None = None,
    gap_eligible: bool = False,
    selected_root_separation: float | None = None,
    solver_tolerance: float | None = None,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    epsilon: float = DEFAULT_RESIDUAL_EPSILON,
) -> ResidualResult:
    """Compute ``rho=v^T Hv`` and ``r=Hv-rho v`` in one active space."""

    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    semantics = (
        ResidualSemantics.PHYSICAL_HESSIAN_EIGEN
        if hvp.metadata.physical
        else ResidualSemantics.PHYSICAL_MODEL_EIGEN
    )
    if not hvp.available or hvp.action is None:
        return ResidualResult(
            available=False,
            semantics=semantics,
            rho=None,
            residual=None,
            residual_norm=None,
            relative_torque=None,
            gap=None,
            residual_over_gap=None,
            gap_eligible=False,
            selected_root_separation=None,
            degeneracy_unresolved=True,
            metadata=hvp.metadata,
            numerator_origin=str(hvp.metadata.operator_origin),
            relative_denominator_origin="unavailable",
            gap_denominator_origin="unavailable",
            epsilon=epsilon,
            unavailable_reason=hvp.unavailable_reason or "hvp_unavailable",
        )
    if hvp.metadata.coordinate_space_id != coordinate_space.identity:
        raise HVPIdentityError("residual coordinate space differs from HVP coordinate space")
    v = coordinate_space.normalized(hvp.direction)
    action = coordinate_space.project(hvp.action)
    rho = float(np.dot(v.reshape(-1), action.reshape(-1)))
    residual = coordinate_space.project(action - rho * v)
    residual_norm = float(np.linalg.norm(residual))
    relative = residual_norm / max(abs(rho), epsilon)
    gap_value: float | None = None
    ratio: float | None = None
    gap_ok = bool(gap_eligible and gap is not None and np.isfinite(float(gap)) and float(gap) > 0.0)
    if gap_ok:
        gap_value = abs(float(gap))
        ratio = residual_norm / max(gap_value, epsilon)
    separation = None if selected_root_separation is None else abs(float(selected_root_separation))
    tol = max(float(gap_tolerance), 0.0 if solver_tolerance is None else abs(float(solver_tolerance)))
    unresolved = bool(separation is not None and separation <= tol)
    if unresolved:
        gap_ok = False
        ratio = None
    return ResidualResult(
        available=True,
        semantics=semantics,
        rho=rho,
        residual=residual,
        residual_norm=residual_norm,
        relative_torque=relative,
        gap=gap_value,
        residual_over_gap=ratio,
        gap_eligible=gap_ok,
        selected_root_separation=separation,
        degeneracy_unresolved=unresolved,
        metadata=hvp.metadata,
        numerator_origin=f"residual_from:{hvp.metadata.operator_origin}",
        relative_denominator_origin="abs_rho_with_epsilon",
        gap_denominator_origin=("selected_root_ritz_separation" if gap_ok else "gap_unavailable_or_unresolved"),
        epsilon=epsilon,
    )


def effective_force_residual(
    residual: object,
    *,
    geometry_id: str,
    state_id: int,
    state_uid: str,
    coordinate_space: ActiveCoordinateSpace,
    purpose: WorkPurpose | str,
    source: str,
    family: str = "effective_force",
    model_age: int | None = None,
    timing_ns: int = 0,
    provenance: Mapping[str, object] | None = None,
) -> ResidualResult:
    """Explicitly nonphysical residual-Jacobian result.

    There is intentionally no physical ``rho`` or eigengap interpretation for a
    generic nonsymmetric effective-force Jacobian.
    """

    vector = coordinate_space.project(residual)
    norm = float(np.linalg.norm(vector))
    metadata = CommonResultMetadata(
        geometry_id=geometry_id,
        state_id=state_id,
        state_uid=state_uid,
        coordinate_space_id=coordinate_space.identity,
        purpose=purpose,
        source=source,
        family=family,
        operator_origin=OperatorOrigin.EFFECTIVE_FORCE_JACOBIAN,
        physical=False,
        model_derived=True,
        model_age=model_age,
        pes_call_delta=0,
        cache_hit=True,
        timing_ns=timing_ns,
        units="effective-force residual",
        provenance=provenance or {},
    )
    return ResidualResult(
        available=True,
        semantics=ResidualSemantics.EFFECTIVE_FORCE_JACOBIAN,
        rho=None,
        residual=vector,
        residual_norm=norm,
        relative_torque=None,
        gap=None,
        residual_over_gap=None,
        gap_eligible=False,
        selected_root_separation=None,
        degeneracy_unresolved=False,
        metadata=metadata,
        numerator_origin="effective_force_jacobian_residual",
        relative_denominator_origin="not_physical_curvature",
        gap_denominator_origin="not_applicable_to_generic_effective_jacobian",
        epsilon=DEFAULT_RESIDUAL_EPSILON,
    )


@dataclass(frozen=True)
class RitzResult:
    available: bool
    rank: int
    ritz_values: tuple[float, ...]
    selected_index: int | None
    selected_value: float | None
    selected_vector: Array | None
    full_residual: Array | None
    full_residual_norm: float | None
    selected_root_separation: float | None
    gap: float | None
    gap_eligible: bool
    degeneracy_unresolved: bool
    root_selection: RootSelection | str
    physical: bool
    model_age: int | None
    solver_tolerance: float
    rank_tolerance: float
    geometry_id: str
    state_id: int
    state_uid: str
    coordinate_space_id: str
    action_origins: tuple[str, ...]
    approximation_caveat: str
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_selection", RootSelection(str(self.root_selection)))
        if self.selected_vector is not None:
            object.__setattr__(self, "selected_vector", freeze_array(self.selected_vector))
        if self.full_residual is not None:
            object.__setattr__(self, "full_residual", freeze_array(self.full_residual))
        if not self.available and not str(self.unavailable_reason).strip():
            raise ValueError("unavailable Ritz result requires a reason")
        object.__setattr__(self, "unavailable_reason", str(self.unavailable_reason).strip().lower())



def rayleigh_ritz(
    actions: Sequence[HVPResult],
    *,
    coordinate_space: ActiveCoordinateSpace,
    root_selection: RootSelection | str = RootSelection.LOWEST,
    homing_vector: object | None = None,
    solver_tolerance: float = 1.0e-8,
    rank_tolerance: float = DEFAULT_RANK_TOLERANCE,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
) -> RitzResult:
    """Rank-revealed same-center Rayleigh-Ritz from independent HVP columns.

    ``+v`` and ``-v`` have rank one and therefore can produce one Ritz value and
    one residual only.  Gap eligibility requires rank >= 2 after active-space
    projection/rank revelation; two force endpoints of one finite-difference
    stencil do not constitute two action directions.
    """

    root_selection = RootSelection(str(root_selection))
    rank_tolerance = float(rank_tolerance)
    solver_tolerance = float(solver_tolerance)
    gap_tolerance = float(gap_tolerance)
    if rank_tolerance <= 0.0 or solver_tolerance < 0.0 or gap_tolerance < 0.0:
        raise ValueError("rank tolerance must be >0 and other tolerances >=0")
    available = [item for item in actions if item.available and item.action is not None]
    if not available:
        return RitzResult(
            available=False,
            rank=0,
            ritz_values=(),
            selected_index=None,
            selected_value=None,
            selected_vector=None,
            full_residual=None,
            full_residual_norm=None,
            selected_root_separation=None,
            gap=None,
            gap_eligible=False,
            degeneracy_unresolved=True,
            root_selection=root_selection,
            physical=False,
            model_age=None,
            solver_tolerance=solver_tolerance,
            rank_tolerance=rank_tolerance,
            geometry_id="",
            state_id=-1,
            state_uid="",
            coordinate_space_id=coordinate_space.identity,
            action_origins=(),
            approximation_caveat="no available same-center HVP columns",
            unavailable_reason="no_available_actions",
        )
    first = available[0]
    gid = first.metadata.geometry_id
    sid = first.metadata.state_id
    suid = first.metadata.state_uid
    cid = first.metadata.coordinate_space_id
    for item in available:
        if (
            item.metadata.geometry_id != gid
            or item.metadata.state_id != sid
            or item.metadata.state_uid != suid
        ):
            return RitzResult(
                available=False,
                rank=0,
                ritz_values=(),
                selected_index=None,
                selected_value=None,
                selected_vector=None,
                full_residual=None,
                full_residual_norm=None,
                selected_root_separation=None,
                gap=None,
                gap_eligible=False,
                degeneracy_unresolved=True,
                root_selection=root_selection,
                physical=False,
                model_age=None,
                solver_tolerance=solver_tolerance,
                rank_tolerance=rank_tolerance,
                geometry_id=gid,
                state_id=sid,
                state_uid=suid,
                coordinate_space_id=coordinate_space.identity,
                action_origins=tuple(str(a.metadata.operator_origin) for a in available),
                approximation_caveat="historical/cross-center actions are invalid as exact current actions",
                unavailable_reason="same_center_identity_mismatch",
            )
        if item.metadata.coordinate_space_id != coordinate_space.identity or cid != coordinate_space.identity:
            raise HVPIdentityError("Ritz action coordinate space mismatch")
    Q = np.column_stack([coordinate_space.project(item.direction).reshape(-1) for item in available])
    A = np.column_stack([coordinate_space.project(item.action).reshape(-1) for item in available])
    u, singular, vt = np.linalg.svd(Q, full_matrices=False)
    if singular.size == 0:
        rank = 0
    else:
        threshold = max(rank_tolerance, rank_tolerance * float(singular[0]))
        rank = int(np.count_nonzero(singular > threshold))
    if rank < 1:
        return RitzResult(
            available=False,
            rank=0,
            ritz_values=(),
            selected_index=None,
            selected_value=None,
            selected_vector=None,
            full_residual=None,
            full_residual_norm=None,
            selected_root_separation=None,
            gap=None,
            gap_eligible=False,
            degeneracy_unresolved=True,
            root_selection=root_selection,
            physical=all(item.metadata.physical for item in available),
            model_age=max((item.metadata.model_age or 0 for item in available), default=0),
            solver_tolerance=solver_tolerance,
            rank_tolerance=rank_tolerance,
            geometry_id=gid,
            state_id=sid,
            state_uid=suid,
            coordinate_space_id=coordinate_space.identity,
            action_origins=tuple(str(item.metadata.operator_origin) for item in available),
            approximation_caveat="action directions are numerically rank deficient",
            unavailable_reason="rank_zero",
        )
    Qb = u[:, :rank]
    transform = vt[:rank, :].T / singular[:rank][None, :]
    Ub = A @ transform
    projected = Qb.T @ Ub
    # Physical Hessian and physical-Hessian model contracts are symmetric;
    # symmetrization removes only finite-difference/model roundoff asymmetry.
    projected = 0.5 * (projected + projected.T)
    values, coeffs = np.linalg.eigh(projected)
    if root_selection is RootSelection.LOWEST:
        selected = 0
    else:
        if homing_vector is None:
            raise ValueError("homed_overlap root selection requires homing_vector")
        home = coordinate_space.normalized(homing_vector).reshape(-1)
        full_vectors = Qb @ coeffs
        overlaps = np.abs(full_vectors.T @ home)
        selected = int(np.argmax(overlaps))
    theta = float(values[selected])
    coeff = coeffs[:, selected]
    vector_flat = Qb @ coeff
    vector = coordinate_space.normalized(vector_flat.reshape(coordinate_space.active_dof_mask.shape))
    residual_flat = Ub @ coeff - theta * (Qb @ coeff)
    residual = coordinate_space.project(residual_flat.reshape(vector.shape))
    residual_norm = float(np.linalg.norm(residual))
    separation: float | None = None
    gap: float | None = None
    gap_eligible = rank >= 2
    if rank >= 2:
        others = [abs(theta - float(value)) for index, value in enumerate(values) if index != selected]
        separation = min(others)
        gap = separation
    unresolved = bool(
        separation is not None
        and separation <= max(gap_tolerance, solver_tolerance)
    )
    if unresolved:
        gap_eligible = False
    physical = all(item.metadata.physical for item in available)
    model_ages = [item.metadata.model_age for item in available if item.metadata.model_age is not None]
    model_age = max(model_ages) if model_ages else None
    caveat = (
        "physical same-center finite-dimensional Ritz approximation"
        if physical
        else "model-derived same-center Ritz approximation; not physical certification"
    )
    if rank == 1:
        caveat += "; one independent action direction gives no Ritz gap"
    elif unresolved:
        caveat += "; selected root separation is unresolved/near-degenerate"
    return RitzResult(
        available=True,
        rank=rank,
        ritz_values=tuple(float(value) for value in values),
        selected_index=selected,
        selected_value=theta,
        selected_vector=vector,
        full_residual=residual,
        full_residual_norm=residual_norm,
        selected_root_separation=separation,
        gap=gap,
        gap_eligible=gap_eligible,
        degeneracy_unresolved=unresolved,
        root_selection=root_selection,
        physical=physical,
        model_age=model_age,
        solver_tolerance=solver_tolerance,
        rank_tolerance=rank_tolerance,
        geometry_id=gid,
        state_id=sid,
        state_uid=suid,
        coordinate_space_id=coordinate_space.identity,
        action_origins=tuple(str(item.metadata.operator_origin) for item in available),
        approximation_caveat=caveat,
    )


def physical_eigen_residual_with_ritz(
    hvp: HVPResult,
    ritz: RitzResult,
    *,
    coordinate_space: ActiveCoordinateSpace,
    epsilon: float = DEFAULT_RESIDUAL_EPSILON,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    direction_tolerance: float = 1.0e-8,
) -> ResidualResult:
    """Attach a gap denominator only from a matching same-center Ritz result.

    The HVP must be evaluated along the selected Ritz vector.  This prevents an
    arbitrary scalar from being presented as the denominator for another mode.
    Endpoint force evaluations by themselves never satisfy this contract; the
    supplied ``RitzResult`` must already have at least two independent action
    directions and ``gap_eligible=True``.
    """

    if not ritz.available or ritz.selected_vector is None:
        return physical_eigen_residual(hvp, coordinate_space=coordinate_space, epsilon=epsilon)
    if (
        hvp.metadata.state_id != ritz.state_id
        or hvp.metadata.state_uid != ritz.state_uid
        or hvp.metadata.geometry_id != ritz.geometry_id
        or coordinate_space.identity != ritz.coordinate_space_id
    ):
        raise HVPIdentityError("Ritz denominator identity does not match HVP residual identity")
    v = coordinate_space.normalized(hvp.direction)
    selected = coordinate_space.normalized(ritz.selected_vector)
    overlap = abs(float(np.dot(v.reshape(-1), selected.reshape(-1))))
    if overlap < 1.0 - float(direction_tolerance):
        raise HVPIdentityError("HVP direction is not the selected Ritz direction")
    return physical_eigen_residual(
        hvp,
        coordinate_space=coordinate_space,
        gap=ritz.gap,
        gap_eligible=ritz.gap_eligible,
        selected_root_separation=ritz.selected_root_separation,
        solver_tolerance=ritz.solver_tolerance,
        gap_tolerance=gap_tolerance,
        epsilon=epsilon,
    )


__all__ = [
    "ApproximatePhysicalModelHVPBackend",
    "DEFAULT_DENSE_MATRIX_BUDGET_BYTES",
    "DEFAULT_GAP_TOLERANCE",
    "DEFAULT_RANK_TOLERANCE",
    "DEFAULT_RESIDUAL_EPSILON",
    "ExplicitMatrixHVPBackend",
    "FiniteDifferenceForceHVPBackend",
    "HVPBackend",
    "HVPIdentityError",
    "HVPInterfaceError",
    "HVPRequest",
    "HVPResult",
    "HVP_SCHEMA_VERSION",
    "MatrixMemoryBudget",
    "MatrixMemoryBudgetError",
    "ResidualResult",
    "RitzResult",
    "effective_force_residual",
    "physical_eigen_residual",
    "physical_eigen_residual_with_ritz",
    "rayleigh_ritz",
]
