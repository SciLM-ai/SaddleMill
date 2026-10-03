"""Wave-C mode predictors built from already available physical information.

This module is intentionally calculator-free and runtime-wiring-free.  It owns
only mode predictor policy.  shared-runtime is responsible for public config/factory/event
wiring, and projected-Olsen/JD owns the reusable Olsen/Jacobi--Davidson correction kernel.

Scientific distinctions preserved here:

* ``translation_secant_sd`` and ``translation_secant_rotation_lbfgs`` consume a
  path-averaged *physical translation secant* and never create a measured
  rotational force/HVP observation;
* ``physical_eigen`` diagonalizes only physical-Hessian's approximate physical Hessian B;
* ``physical_olsen`` delegates the projected correction solve to an injected
  projected-Olsen/JD public correction kernel.  No Olsen/JD solver is duplicated here.

All returned modes are sign-continuous with the supplied current mode and all
updates are capped by an angular maximum (5 degrees by the sealed default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from saddlemill.dimertools.foundation_types import (
    ActiveCoordinateSpace,
    ModePredictionResult,
    TranslationSecantPredictorInput,
    freeze_array,
    freeze_mapping,
)
from saddlemill.dimertools.physical_hessian import PhysicalHessianModel
from saddlemill.dimertools.rotation_history import StateWindowRotationHistory
from saddlemill.dimertools.riemannian_lbfgs import RotationLBFGSModel, RotationSecant
from saddlemill.dimertools.sphere_manifold import (
    log_map,
    normalize_axis,
    project_tangent,
    sign_align_axis,
    sphere_distance,
    take_sphere_step,
)

Array = np.ndarray

TRANSLATION_SECANT_SD = "translation_secant_sd"
TRANSLATION_SECANT_ROTATION_LBFGS = "translation_secant_rotation_lbfgs"
TRANSLATION_SECANT_FORCEBANK_LBFGS = "translation_secant_forcebank_lbfgs"
PHYSICAL_EIGEN = "physical_eigen"
PHYSICAL_OLSEN = "physical_olsen"

DEFAULT_MAX_ANGLE_RADIANS = math.radians(5.0)
DEFAULT_DEGENERACY_TOLERANCE = 1.0e-8
DEFAULT_VECTOR_TOLERANCE = 1.0e-14


@dataclass(frozen=True)
class PhysicalModePredictorInput:
    """Input shared by physical-B mode predictors.

    ``model`` owns the approximate physical Hessian and its update history.
    This request adds no PES/HVP work.  shared-runtime must supply the current model at the
    same accepted center as ``state_uid``/``geometry_id``.
    """

    state_id: int
    state_uid: str
    geometry_id: str
    current_mode: Array
    coordinate_space: ActiveCoordinateSpace
    model: PhysicalHessianModel
    root_policy: str = "lowest"
    low_spectrum_count: int = 4
    max_angle_radians: float = DEFAULT_MAX_ANGLE_RADIANS
    degeneracy_tolerance: float = DEFAULT_DEGENERACY_TOLERANCE
    vector_tolerance: float = DEFAULT_VECTOR_TOLERANCE

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_space, ActiveCoordinateSpace):
            raise TypeError("coordinate_space must be ActiveCoordinateSpace")
        if not isinstance(self.model, PhysicalHessianModel):
            raise TypeError("model must be PhysicalHessianModel")
        if self.model.coordinate_space.identity != self.coordinate_space.identity:
            raise ValueError("physical-B predictor coordinate space does not match model")
        object.__setattr__(self, "state_id", int(self.state_id))
        for name in ("state_uid", "geometry_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        mode = np.asarray(self.current_mode, dtype=float)
        if mode.shape == (self.coordinate_space.active_dof_mask.size,):
            mode = mode.reshape(self.coordinate_space.active_dof_mask.shape)
        if mode.shape != self.coordinate_space.active_dof_mask.shape:
            raise ValueError("current_mode must match coordinate-space shape")
        if not np.all(np.isfinite(mode)):
            raise ValueError("current_mode must be finite")
        object.__setattr__(self, "current_mode", freeze_array(mode, shape=mode.shape))
        policy = str(self.root_policy).strip().lower()
        if policy not in {"lowest", "overlap"}:
            raise ValueError("root_policy must be lowest or overlap")
        object.__setattr__(self, "root_policy", policy)
        count = int(self.low_spectrum_count)
        if count < 1:
            raise ValueError("low_spectrum_count must be >= 1")
        object.__setattr__(self, "low_spectrum_count", count)
        for name in ("max_angle_radians", "degeneracy_tolerance", "vector_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        if not 0.0 < self.max_angle_radians <= math.pi / 2.0:
            raise ValueError("max_angle_radians must satisfy 0 < max_angle <= pi/2")


@dataclass(frozen=True)
class PhysicalModePredictionResult:
    """Structured one-step prediction from an approximate physical B model."""

    mode: Array
    status: str
    selector: str
    origin: str
    raw_tangent: Array | None
    requested_angle_radians: float
    accepted_angle_radians: float
    capped: bool
    state_id: int
    state_uid: str
    geometry_id: str
    selected_root_index: int | None = None
    selected_eigenvalue: float | None = None
    selected_eigengap: float | None = None
    near_degenerate: bool = False
    model_residual_norm: float | None = None
    model_age: int | None = None
    model_update_type: str = ""
    coordinate_space_identity: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = np.asarray(self.mode, dtype=float)
        if mode.ndim != 2 or mode.shape[1] != 3 or not np.all(np.isfinite(mode)):
            raise ValueError("prediction mode must have finite shape (N, 3)")
        object.__setattr__(self, "mode", freeze_array(mode, shape=mode.shape))
        if self.raw_tangent is not None:
            tangent = np.asarray(self.raw_tangent, dtype=float)
            if tangent.shape != mode.shape or not np.all(np.isfinite(tangent)):
                raise ValueError("raw_tangent must be finite and match mode shape")
            object.__setattr__(self, "raw_tangent", freeze_array(tangent, shape=mode.shape))
        for name in ("status", "selector", "origin", "state_uid", "geometry_id", "model_update_type", "coordinate_space_identity"):
            object.__setattr__(self, name, str(getattr(self, name)).strip().lower() if name in {"status", "selector", "origin", "model_update_type"} else str(getattr(self, name)).strip())
        object.__setattr__(self, "state_id", int(self.state_id))
        for name in ("requested_angle_radians", "accepted_angle_radians"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        if self.selected_eigenvalue is not None:
            object.__setattr__(self, "selected_eigenvalue", float(self.selected_eigenvalue))
        if self.selected_eigengap is not None:
            object.__setattr__(self, "selected_eigengap", float(self.selected_eigengap))
        if self.model_residual_norm is not None:
            object.__setattr__(self, "model_residual_norm", float(self.model_residual_norm))
        if self.model_age is not None:
            object.__setattr__(self, "model_age", int(self.model_age))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_state_dict(self) -> dict[str, object]:
        return {
            "schema": "saddlemill_t10_physical_mode_prediction_v1",
            "mode": self.mode.tolist(),
            "status": self.status,
            "selector": self.selector,
            "origin": self.origin,
            "raw_tangent": None if self.raw_tangent is None else self.raw_tangent.tolist(),
            "requested_angle_radians": self.requested_angle_radians,
            "accepted_angle_radians": self.accepted_angle_radians,
            "capped": bool(self.capped),
            "state_id": self.state_id,
            "state_uid": self.state_uid,
            "geometry_id": self.geometry_id,
            "selected_root_index": self.selected_root_index,
            "selected_eigenvalue": self.selected_eigenvalue,
            "selected_eigengap": self.selected_eigengap,
            "near_degenerate": bool(self.near_degenerate),
            "model_residual_norm": self.model_residual_norm,
            "model_age": self.model_age,
            "model_update_type": self.model_update_type,
            "coordinate_space_identity": self.coordinate_space_identity,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "PhysicalModePredictionResult":
        if state.get("schema") != "saddlemill_t10_physical_mode_prediction_v1":
            raise ValueError("unsupported mode-predictor physical prediction state schema")
        return cls(
            mode=np.asarray(state["mode"], dtype=float),
            status=str(state["status"]),
            selector=str(state["selector"]),
            origin=str(state["origin"]),
            raw_tangent=None if state.get("raw_tangent") is None else np.asarray(state["raw_tangent"], dtype=float),
            requested_angle_radians=float(state["requested_angle_radians"]),
            accepted_angle_radians=float(state["accepted_angle_radians"]),
            capped=bool(state["capped"]),
            state_id=int(state["state_id"]),
            state_uid=str(state["state_uid"]),
            geometry_id=str(state["geometry_id"]),
            selected_root_index=None if state.get("selected_root_index") is None else int(state["selected_root_index"]),
            selected_eigenvalue=None if state.get("selected_eigenvalue") is None else float(state["selected_eigenvalue"]),
            selected_eigengap=None if state.get("selected_eigengap") is None else float(state["selected_eigengap"]),
            near_degenerate=bool(state.get("near_degenerate", False)),
            model_residual_norm=None if state.get("model_residual_norm") is None else float(state["model_residual_norm"]),
            model_age=None if state.get("model_age") is None else int(state["model_age"]),
            model_update_type=str(state.get("model_update_type", "")),
            coordinate_space_identity=str(state.get("coordinate_space_identity", "")),
            metadata=dict(state.get("metadata", {})),
        )


@dataclass(frozen=True)
class OlsenCorrectionResultView:
    """Minimal public result view required from projected-Olsen/JD by the mode-predictor adapter.

    projected-Olsen/JD may publish a richer dataclass.  shared-runtime can adapt that public result into
    this view without changing the scientific correction itself.
    """

    correction: Array
    status: str
    fallback: str = "none"
    condition_state: str = "available"
    orthogonality_error: float | None = None
    shifted_system_residual: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        correction = np.asarray(self.correction, dtype=float)
        if correction.ndim != 2 or correction.shape[1] != 3 or not np.all(np.isfinite(correction)):
            raise ValueError("Olsen correction must have finite shape (N, 3)")
        object.__setattr__(self, "correction", freeze_array(correction, shape=correction.shape))
        object.__setattr__(self, "status", str(self.status).strip().lower())
        object.__setattr__(self, "fallback", str(self.fallback).strip().lower())
        object.__setattr__(self, "condition_state", str(self.condition_state).strip().lower())
        for name in ("orthogonality_error", "shifted_system_residual"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be finite and >= 0")
                object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@runtime_checkable
class OlsenCorrectionKernelProtocol(Protocol):
    """mode-predictor's narrow dependency on projected-Olsen/JD's public projected-correction kernel."""

    def __call__(
        self,
        *,
        mode: Array,
        action: Array,
        operator: Callable[[Array], Array],
        coordinate_space: ActiveCoordinateSpace,
        data_source: Mapping[str, object],
    ) -> OlsenCorrectionResultView:
        ...


def _model_provenance(model: PhysicalHessianModel) -> dict[str, object]:
    state = model.state_dict()
    return {
        "operator_origin": "physical_hessian_B",
        "model_update_type": str(state.get("update_type", model.update_type)),
        "model_age": int(state.get("model_age", model.model_age)),
        "model_current_state_uid": str(state.get("current_state_uid", "")),
        "model_current_geometry_id": str(state.get("current_geometry_id", "")),
        "admitted_observation_ids": tuple(str(v) for v in state.get("seen_observation_ids", ())),
        "admitted_hvp_ids": tuple(str(v) for v in state.get("seen_hvp_ids", ())),
        "coordinate_space_identity": model.coordinate_space.identity,
    }


def _identity_valid(request: TranslationSecantPredictorInput) -> bool:
    if request.new_state_id <= request.old_state_id:
        return False
    if request.new_state_uid == request.old_state_uid:
        return False
    if request.new_geometry_id == request.old_geometry_id:
        return False
    return True


def _unchanged_translation_result(
    request: TranslationSecantPredictorInput,
    mode: Array,
    *,
    status: str,
    alignment: float | None = None,
    reversed_pair: bool = False,
    raw_tangent: Array | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ModePredictionResult:
    return ModePredictionResult(
        mode=mode,
        status=status,
        original_signed_alignment=alignment,
        absolute_alignment=None if alignment is None else abs(alignment),
        reversed_pair=reversed_pair,
        raw_tangent=raw_tangent,
        requested_angle_radians=0.0,
        accepted_angle_radians=0.0,
        capped=False,
        old_state_id=request.old_state_id,
        new_state_id=request.new_state_id,
        old_state_uid=request.old_state_uid,
        new_state_uid=request.new_state_uid,
        old_geometry_id=request.old_geometry_id,
        new_geometry_id=request.new_geometry_id,
        angular_step_scale=request.angular_step_scale,
        max_angle_radians=request.max_angle_radians,
        alignment_tolerance=request.alignment_tolerance,
        displacement_tolerance=request.displacement_tolerance,
        tangent_tolerance=request.tangent_tolerance,
        metadata={} if metadata is None else metadata,
    )


class TranslationSecantModePredictor:
    """Sealed direct large-dimer translation-secant heuristic."""

    def __init__(
        self,
        selector: str = TRANSLATION_SECANT_SD,
        *,
        rotation_history: StateWindowRotationHistory | None = None,
        forcebank_rotation_model: RotationLBFGSModel | None = None,
        forcebank_rotation_pairs: Sequence[RotationSecant] = (),
        forcebank_build_metrics: Mapping[str, object] | None = None,
    ) -> None:
        token = str(selector).strip().lower()
        if token not in {
            TRANSLATION_SECANT_SD,
            TRANSLATION_SECANT_ROTATION_LBFGS,
            TRANSLATION_SECANT_FORCEBANK_LBFGS,
        }:
            raise ValueError(f"unsupported translation-secant predictor selector: {selector!r}")
        if token == TRANSLATION_SECANT_ROTATION_LBFGS and rotation_history is None:
            raise ValueError("translation_secant_rotation_lbfgs requires rotational L-BFGS history")
        if token == TRANSLATION_SECANT_FORCEBANK_LBFGS and forcebank_rotation_model is None:
            raise ValueError("translation_secant_forcebank_lbfgs requires a reconstructed force-bank rotational L-BFGS model")
        self.selector = token
        self.rotation_history = rotation_history
        self.forcebank_rotation_model = forcebank_rotation_model
        self.forcebank_rotation_pairs = tuple(forcebank_rotation_pairs)
        self.forcebank_build_metrics = dict(forcebank_build_metrics or {})

    def state_dict(self) -> dict[str, object]:
        # Dynamic rotational history is owned/serialized by the rotation optimizer,
        # not duplicated in predictor state.
        return {
            "schema": "saddlemill_t10_translation_secant_predictor_v1",
            "selector": self.selector,
            "owns_dynamic_state": False,
        }

    def predict(self, request: TranslationSecantPredictorInput) -> ModePredictionResult:
        space = request.coordinate_space
        try:
            mode = space.normalized(request.current_mode)
        except ValueError:
            # The sealed request validates shape, not nonzero active norm.
            raw = np.asarray(request.current_mode, dtype=float)
            safe = np.zeros_like(raw)
            safe.reshape(-1)[np.flatnonzero(space.active_dof_mask.reshape(-1))[0]] = 1.0
            mode = space.normalized(safe)
            return _unchanged_translation_result(
                request,
                mode,
                status="invalid_current_mode",
                metadata={"selector": self.selector, "prediction_cost_pes_calls": 0},
            )

        base_meta: dict[str, object] = {
            "selector": self.selector,
            "prediction_cost_pes_calls": 0,
            "observation_admission_count": 0,
            "surrogate_is_measured_torque": False,
            "source_age_state_delta": int(request.new_state_id - request.old_state_id),
            "coordinate_space_identity": space.identity,
            "map_kind": "exponential",
        }
        if not _identity_valid(request):
            return _unchanged_translation_result(request, mode, status="invalid_identity", metadata=base_meta)

        try:
            s = space.project(np.asarray(request.new_positions) - np.asarray(request.old_positions))
            y = space.project(np.asarray(request.new_gradient) - np.asarray(request.old_gradient))
        except ValueError:
            return _unchanged_translation_result(request, mode, status="invalid_numeric", metadata=base_meta)
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
            return _unchanged_translation_result(request, mode, status="invalid_numeric", metadata=base_meta)
        s_norm = float(np.linalg.norm(s))
        y_norm = float(np.linalg.norm(y))
        base_meta.update({"displacement_norm": s_norm, "gradient_change_norm": y_norm})
        if s_norm <= request.displacement_tolerance:
            return _unchanged_translation_result(request, mode, status="zero_displacement", metadata=base_meta)
        if y_norm <= 0.0:
            return _unchanged_translation_result(request, mode, status="zero_gradient_change", metadata=base_meta)

        q_original = s / s_norm
        alignment = float(np.dot(q_original.reshape(-1), mode.reshape(-1)))
        base_meta["original_absolute_alignment"] = abs(alignment)
        if abs(alignment) <= request.alignment_tolerance:
            return _unchanged_translation_result(
                request,
                mode,
                status="alignment_unresolved",
                alignment=alignment,
                metadata=base_meta,
            )

        reversed_pair = bool(alignment < 0.0)
        if reversed_pair:
            s = -s
            y = -y
        q = s / s_norm
        post_alignment = float(np.dot(q.reshape(-1), mode.reshape(-1)))
        a = y / s_norm
        tangent = -space.project(project_tangent(mode, a))
        tangent = space.project(project_tangent(mode, tangent))
        tangent_norm = float(np.linalg.norm(tangent))
        base_meta.update(
            {
                "post_rule_alignment": post_alignment,
                "secant_action_norm": float(np.linalg.norm(a)),
                "raw_tangent_norm": tangent_norm,
            }
        )
        if not np.all(np.isfinite(tangent)) or tangent_norm <= request.tangent_tolerance:
            return _unchanged_translation_result(
                request,
                mode,
                status="zero_or_invalid_tangent",
                alignment=alignment,
                reversed_pair=reversed_pair,
                raw_tangent=tangent if np.all(np.isfinite(tangent)) else None,
                metadata=base_meta,
            )

        proposal = tangent.copy()
        fallback = "none"
        preconditioned = False
        if self.selector == TRANSLATION_SECANT_ROTATION_LBFGS:
            assert self.rotation_history is not None
            projector = lambda value: space.project(project_tangent(mode, space.project(value)))
            try:
                candidate = np.asarray(self.rotation_history.apply(tangent, projector=projector), dtype=float)
                candidate = projector(candidate)
                candidate_norm = float(np.linalg.norm(candidate))
                ascent = float(np.dot(candidate.reshape(-1), tangent.reshape(-1)))
                base_meta.update(
                    {
                        "rotation_lbfgs_candidate_norm": candidate_norm,
                        "rotation_lbfgs_candidate_dot_surrogate": ascent,
                    }
                )
                if (
                    not np.all(np.isfinite(candidate))
                    or not np.isfinite(candidate_norm)
                    or candidate_norm <= request.tangent_tolerance
                    or not np.isfinite(ascent)
                    or ascent <= 0.0
                ):
                    fallback = "sd_unusable_rotation_lbfgs_direction"
                else:
                    proposal = candidate
                    preconditioned = True
            except Exception as exc:  # deterministic safety fallback, no silent success claim
                fallback = "sd_rotation_lbfgs_exception"
                base_meta["rotation_lbfgs_exception_type"] = type(exc).__name__
        elif self.selector == TRANSLATION_SECANT_FORCEBANK_LBFGS:
            assert self.forcebank_rotation_model is not None
            base_meta.update({
                "forcebank_torque_samples": self.forcebank_build_metrics.get("torque_samples", 0),
                "forcebank_pair_sources": self.forcebank_build_metrics.get("pair_sources", ""),
                "forcebank_accepted_force_source": self.forcebank_build_metrics.get("accepted_force_source", ""),
                "forcebank_trial_pairs_future_only": self.forcebank_build_metrics.get("trial_pairs_future_only", ""),
                "forcebank_dynamic_h0": int(bool(self.forcebank_rotation_model.dynamic_h0)),
                "forcebank_pair_candidates": self.forcebank_build_metrics.get("pair_candidates", 0),
                "forcebank_pairs_built": self.forcebank_build_metrics.get("pairs_built", len(self.forcebank_rotation_pairs)),
                "forcebank_pairs_degenerate": self.forcebank_build_metrics.get("pairs_degenerate", 0),
                "forcebank_states_contributing": self.forcebank_build_metrics.get("states_contributing", 0),
                "forcebank_build_ns": self.forcebank_build_metrics.get("build_ns", 0),
            })
            try:
                qn = self.forcebank_rotation_model.apply(
                    tangent, mode, external_pairs=self.forcebank_rotation_pairs,
                    basis=space.null_basis,
                )
                qn_metrics = dict(qn.metrics)
                for key in (
                    "pair_candidates", "pairs_admissible_before_max_pairs",
                    "pairs_removed_by_max_pairs", "pairs_transportable", "pairs_used",
                    "pairs_rejected_transport", "pairs_rejected_curvature",
                    "pairs_rejected_cosine", "pairs_damped", "pairs_powell_damped",
                    "accepted_curvature_min", "accepted_curvature_median",
                    "accepted_curvature_max", "accepted_cosine_min",
                    "accepted_cosine_median", "accepted_cosine_max",
                    "h0_inverse_scale", "lbfgs_raw_direction_norm",
                    "direction_norm", "direction_force_cosine", "apply_ns",
                ):
                    base_meta[f"forcebank_lbfgs_{key}"] = qn_metrics.get(key, "")
                base_meta["forcebank_lbfgs_pairs_rejected_by_reason"] = qn_metrics.get(
                    "pairs_rejected_by_reason", {}
                )
                if int(qn_metrics.get("pairs_used", 0)) < 1:
                    return _unchanged_translation_result(
                        request, mode, status="forcebank_lbfgs_no_admissible_pairs",
                        alignment=alignment, reversed_pair=reversed_pair, raw_tangent=tangent,
                        metadata={**base_meta, "preconditioned": False, "fallback": "hold_no_forcebank_pairs"},
                    )
                candidate = space.project(project_tangent(mode, np.asarray(qn.direction, dtype=float)))
                candidate_norm = float(np.linalg.norm(candidate))
                ascent = float(np.dot(candidate.reshape(-1), tangent.reshape(-1)))
                base_meta.update({
                    "rotation_lbfgs_candidate_norm": candidate_norm,
                    "rotation_lbfgs_candidate_dot_surrogate": ascent,
                })
                if (
                    not np.all(np.isfinite(candidate))
                    or not np.isfinite(candidate_norm)
                    or candidate_norm <= request.tangent_tolerance
                    or not np.isfinite(ascent)
                    or ascent <= 0.0
                ):
                    return _unchanged_translation_result(
                        request, mode, status="forcebank_lbfgs_unusable_direction",
                        alignment=alignment, reversed_pair=reversed_pair, raw_tangent=tangent,
                        metadata={**base_meta, "preconditioned": False, "fallback": "hold_unusable_forcebank_lbfgs_direction"},
                    )
                proposal = candidate
                preconditioned = True
            except Exception as exc:
                base_meta["forcebank_lbfgs_exception_type"] = type(exc).__name__
                return _unchanged_translation_result(
                    request, mode, status="forcebank_lbfgs_exception",
                    alignment=alignment, reversed_pair=reversed_pair, raw_tangent=tangent,
                    metadata={**base_meta, "preconditioned": False, "fallback": "hold_forcebank_lbfgs_exception"},
                )
        base_meta.update({"preconditioned": preconditioned, "fallback": fallback})

        scale = 1.0 if self.selector == TRANSLATION_SECANT_FORCEBANK_LBFGS else request.angular_step_scale
        base_meta["effective_angular_step_scale"] = scale
        scaled = scale * proposal
        if not np.all(np.isfinite(scaled)):
            return _unchanged_translation_result(
                request,
                mode,
                status="invalid_scaled_step",
                alignment=alignment,
                reversed_pair=reversed_pair,
                raw_tangent=tangent,
                metadata=base_meta,
            )
        requested_angle = float(np.linalg.norm(scaled))
        if requested_angle <= request.tangent_tolerance:
            return _unchanged_translation_result(
                request,
                mode,
                status="zero_scaled_step",
                alignment=alignment,
                reversed_pair=reversed_pair,
                raw_tangent=tangent,
                metadata=base_meta,
            )
        step = take_sphere_step(mode, scaled, max_angle=request.max_angle_radians, map_kind="exponential")
        predicted, _ = sign_align_axis(mode, step.axis)
        accepted = sphere_distance(mode, predicted)
        base_meta.update(
            {
                "requested_angle_radians": requested_angle,
                "accepted_angle_radians": accepted,
                "capped": bool(step.clipped),
            }
        )
        return ModePredictionResult(
            mode=predicted,
            status="predicted",
            original_signed_alignment=alignment,
            absolute_alignment=abs(alignment),
            reversed_pair=reversed_pair,
            raw_tangent=tangent,
            requested_angle_radians=requested_angle,
            accepted_angle_radians=accepted,
            capped=bool(step.clipped),
            old_state_id=request.old_state_id,
            new_state_id=request.new_state_id,
            old_state_uid=request.old_state_uid,
            new_state_uid=request.new_state_uid,
            old_geometry_id=request.old_geometry_id,
            new_geometry_id=request.new_geometry_id,
            angular_step_scale=request.angular_step_scale,
            max_angle_radians=request.max_angle_radians,
            alignment_tolerance=request.alignment_tolerance,
            displacement_tolerance=request.displacement_tolerance,
            tangent_tolerance=request.tangent_tolerance,
            metadata=base_meta,
        )


def _reduced_basis(space: ActiveCoordinateSpace) -> Array:
    """Public-coordinate reconstruction matching physical-Hessian's active/null convention."""

    mask = space.active_dof_mask.reshape(-1)
    active_indices = np.flatnonzero(mask)
    active_count = int(active_indices.size)
    if active_count < 1:
        raise ValueError("coordinate space has no active DOFs")
    if space.null_basis:
        null = np.column_stack([item.reshape(-1)[active_indices] for item in space.null_basis])
        q, _ = np.linalg.qr(null, mode="complete")
        basis_active = q[:, len(space.null_basis) :]
    else:
        basis_active = np.eye(active_count, dtype=float)
    if basis_active.shape[1] < 1:
        raise ValueError("coordinate space is exhausted by null modes")
    full = np.zeros((mask.size, basis_active.shape[1]), dtype=float)
    full[active_indices, :] = basis_active
    return full


def _physical_result_unchanged(
    request: PhysicalModePredictorInput,
    *,
    selector: str,
    status: str,
    metadata: Mapping[str, object],
) -> PhysicalModePredictionResult:
    mode = request.coordinate_space.normalized(request.current_mode)
    provenance = _model_provenance(request.model)
    return PhysicalModePredictionResult(
        mode=mode,
        status=status,
        selector=selector,
        origin="physical_hessian_B",
        raw_tangent=None,
        requested_angle_radians=0.0,
        accepted_angle_radians=0.0,
        capped=False,
        state_id=request.state_id,
        state_uid=request.state_uid,
        geometry_id=request.geometry_id,
        model_age=int(provenance["model_age"]),
        model_update_type=str(provenance["model_update_type"]),
        coordinate_space_identity=request.coordinate_space.identity,
        metadata={**provenance, **dict(metadata), "prediction_cost_pes_calls": 0},
    )


class PhysicalEigenModePredictor:
    """One capped move toward a selected low eigenvector of physical-Hessian B."""

    selector = PHYSICAL_EIGEN

    def state_dict(self) -> dict[str, object]:
        return {"schema": "saddlemill_t10_physical_eigen_predictor_v1", "owns_dynamic_state": False}

    def predict(self, request: PhysicalModePredictorInput) -> PhysicalModePredictionResult:
        space = request.coordinate_space
        current = space.normalized(request.current_mode)
        provenance = _model_provenance(request.model)
        model_state_uid = str(provenance.get("model_current_state_uid", ""))
        model_geometry_id = str(provenance.get("model_current_geometry_id", ""))
        if model_state_uid and model_state_uid != request.state_uid:
            return _physical_result_unchanged(request, selector=self.selector, status="identity_mismatch", metadata={"identity_field": "state_uid"})
        if model_geometry_id and model_geometry_id != request.geometry_id:
            return _physical_result_unchanged(request, selector=self.selector, status="identity_mismatch", metadata={"identity_field": "geometry_id"})

        try:
            matrix = np.asarray(request.model.matrix, dtype=float)
            evals, evecs = np.linalg.eigh(matrix)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            return _physical_result_unchanged(request, selector=self.selector, status="eigensolver_failure", metadata={"exception_type": type(exc).__name__})
        if evals.size < 1 or evecs.shape != (evals.size, evals.size) or not np.all(np.isfinite(evals)) or not np.all(np.isfinite(evecs)):
            return _physical_result_unchanged(request, selector=self.selector, status="eigensolver_failure", metadata={"reason": "invalid_eigendecomposition"})

        basis = _reduced_basis(space)
        current_flat = current.reshape(-1)
        low_count = min(int(request.low_spectrum_count), int(evals.size))
        candidates = list(range(low_count))
        overlaps: list[float] = []
        for idx in candidates:
            full = basis @ evecs[:, idx]
            overlaps.append(abs(float(np.dot(current_flat, full))))
        if request.root_policy == "lowest":
            selected = 0
        else:
            best = max(overlaps)
            tied = [idx for idx, overlap in zip(candidates, overlaps) if abs(overlap - best) <= 1.0e-14]
            selected = min(tied)

        target_flat = basis @ evecs[:, selected]
        target = target_flat.reshape(current.shape)
        # Deterministic sign: current overlap first; exact orthogonality falls
        # back to a canonical first-significant-component orientation.
        dot = float(np.dot(current_flat, target_flat))
        if dot < 0.0:
            target = -target
        elif abs(dot) <= 1.0e-15:
            nz = np.flatnonzero(np.abs(target.reshape(-1)) > 1.0e-15)
            if nz.size and target.reshape(-1)[nz[0]] < 0.0:
                target = -target
        target = space.normalized(target)

        if evals.size < 2:
            gap = None
        else:
            others = np.delete(evals, selected)
            gap = float(np.min(np.abs(others - evals[selected])))
        near_deg = bool(gap is not None and gap <= request.degeneracy_tolerance)
        try:
            tangent = space.project(log_map(current, target, axis_equivalence=True))
        except ValueError as exc:
            return _physical_result_unchanged(request, selector=self.selector, status="root_direction_unresolved", metadata={"exception_type": type(exc).__name__, "selected_root_index": selected})
        tangent = space.project(project_tangent(current, tangent))
        requested_angle = float(np.linalg.norm(tangent))
        if requested_angle <= request.vector_tolerance:
            predicted = current
            accepted = 0.0
            capped = False
            status = "already_aligned"
        else:
            step = take_sphere_step(current, tangent, max_angle=request.max_angle_radians, map_kind="exponential")
            predicted, _ = sign_align_axis(current, step.axis)
            accepted = sphere_distance(current, predicted)
            capped = bool(step.clipped)
            status = "predicted_near_degenerate" if near_deg else "predicted"

        btarget = request.model.apply(target)
        residual = space.project(btarget - float(evals[selected]) * target)
        residual_norm = float(np.linalg.norm(residual))
        spectrum = request.model.low_spectrum(count=low_count, selected_root_index=selected)
        metadata = {
            **provenance,
            "prediction_cost_pes_calls": 0,
            "root_policy": request.root_policy,
            "requested_low_spectrum_count": int(request.low_spectrum_count),
            "low_spectrum_count": low_count,
            "spectrum_truncated": bool(low_count < int(request.low_spectrum_count)),
            "candidate_overlaps": tuple(float(v) for v in overlaps),
            "low_eigenvalues": tuple(float(v) for v in evals[:low_count]),
            "condition_state": spectrum.condition_state,
            "absolute_eigenvalue_condition": spectrum.absolute_eigenvalue_condition,
            "model_residual_interpretation": "model_self_consistency_only",
            "real_current_geometry_residual_validated": False,
        }
        return PhysicalModePredictionResult(
            mode=predicted,
            status=status,
            selector=self.selector,
            origin="physical_hessian_B_eigenvector",
            raw_tangent=tangent,
            requested_angle_radians=requested_angle,
            accepted_angle_radians=accepted,
            capped=capped,
            state_id=request.state_id,
            state_uid=request.state_uid,
            geometry_id=request.geometry_id,
            selected_root_index=selected,
            selected_eigenvalue=float(evals[selected]),
            selected_eigengap=gap,
            near_degenerate=near_deg,
            model_residual_norm=residual_norm,
            model_age=int(provenance["model_age"]),
            model_update_type=str(provenance["model_update_type"]),
            coordinate_space_identity=space.identity,
            metadata=metadata,
        )


class PhysicalOlsenModePredictor:
    """One capped projected Olsen/JD correction using physical-Hessian B."""

    selector = PHYSICAL_OLSEN

    def __init__(self, correction_kernel: OlsenCorrectionKernelProtocol) -> None:
        if not callable(correction_kernel):
            raise TypeError("physical_olsen requires projected-Olsen/JD public correction kernel")
        self.correction_kernel = correction_kernel

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "saddlemill_t10_physical_olsen_predictor_v1",
            "owns_dynamic_state": False,
            "kernel_dependency": "projected_olsen_jd_correction",
        }

    def predict(self, request: PhysicalModePredictorInput) -> PhysicalModePredictionResult:
        space = request.coordinate_space
        current = space.normalized(request.current_mode)
        provenance = _model_provenance(request.model)
        model_state_uid = str(provenance.get("model_current_state_uid", ""))
        model_geometry_id = str(provenance.get("model_current_geometry_id", ""))
        if model_state_uid and model_state_uid != request.state_uid:
            return _physical_result_unchanged(request, selector=self.selector, status="identity_mismatch", metadata={"identity_field": "state_uid"})
        if model_geometry_id and model_geometry_id != request.geometry_id:
            return _physical_result_unchanged(request, selector=self.selector, status="identity_mismatch", metadata={"identity_field": "geometry_id"})

        action = request.model.apply(current)
        source = {
            **provenance,
            "predictor": self.selector,
            "request_state_id": request.state_id,
            "request_state_uid": request.state_uid,
            "request_geometry_id": request.geometry_id,
        }
        try:
            raw = self.correction_kernel(
                mode=current,
                action=action,
                operator=request.model.apply,
                coordinate_space=space,
                data_source=source,
            )
        except Exception as exc:
            return _physical_result_unchanged(request, selector=self.selector, status="correction_kernel_failure", metadata={"exception_type": type(exc).__name__, **provenance})
        if not isinstance(raw, OlsenCorrectionResultView):
            return _physical_result_unchanged(request, selector=self.selector, status="correction_kernel_contract_error", metadata={"returned_type": type(raw).__name__, **provenance})

        correction = space.project(project_tangent(current, raw.correction))
        correction_norm = float(np.linalg.norm(correction))
        orth_error_after_projection = abs(float(np.dot(current.reshape(-1), correction.reshape(-1))))
        metadata = {
            **provenance,
            "prediction_cost_pes_calls": 0,
            "kernel_status": raw.status,
            "kernel_fallback": raw.fallback,
            "kernel_condition_state": raw.condition_state,
            "kernel_orthogonality_error": raw.orthogonality_error,
            "kernel_shifted_system_residual": raw.shifted_system_residual,
            "projected_correction_orthogonality_error": orth_error_after_projection,
            "kernel_metadata": dict(raw.metadata),
            "real_current_geometry_residual_validated": False,
        }
        if not np.all(np.isfinite(correction)) or correction_norm <= request.vector_tolerance:
            return _physical_result_unchanged(request, selector=self.selector, status="correction_unusable", metadata=metadata)

        step = take_sphere_step(current, correction, max_angle=request.max_angle_radians, map_kind="exponential")
        predicted, _ = sign_align_axis(current, step.axis)
        accepted = sphere_distance(current, predicted)
        theta = float(np.dot(current.reshape(-1), action.reshape(-1)))
        residual = space.project(action - theta * current)
        residual_norm = float(np.linalg.norm(residual))
        status = "predicted"
        if raw.fallback not in {"", "none"}:
            status = "predicted_with_fallback"
        elif "singular" in raw.condition_state:
            status = "predicted_singular_system"
        return PhysicalModePredictionResult(
            mode=predicted,
            status=status,
            selector=self.selector,
            origin="physical_hessian_B_olsen_one_step",
            raw_tangent=correction,
            requested_angle_radians=correction_norm,
            accepted_angle_radians=accepted,
            capped=bool(step.clipped),
            state_id=request.state_id,
            state_uid=request.state_uid,
            geometry_id=request.geometry_id,
            selected_root_index=None,
            selected_eigenvalue=theta,
            selected_eigengap=None,
            near_degenerate=("singular" in raw.condition_state or "degenerate" in raw.status),
            model_residual_norm=residual_norm,
            model_age=int(provenance["model_age"]),
            model_update_type=str(provenance["model_update_type"]),
            coordinate_space_identity=space.identity,
            metadata=metadata,
        )


__all__ = [
    "DEFAULT_DEGENERACY_TOLERANCE",
    "DEFAULT_MAX_ANGLE_RADIANS",
    "OlsenCorrectionKernelProtocol",
    "OlsenCorrectionResultView",
    "PHYSICAL_EIGEN",
    "PHYSICAL_OLSEN",
    "PhysicalEigenModePredictor",
    "PhysicalModePredictionResult",
    "PhysicalModePredictorInput",
    "PhysicalOlsenModePredictor",
    "TRANSLATION_SECANT_ROTATION_LBFGS",
    "TRANSLATION_SECANT_FORCEBANK_LBFGS",
    "TRANSLATION_SECANT_SD",
    "TranslationSecantModePredictor",
]
