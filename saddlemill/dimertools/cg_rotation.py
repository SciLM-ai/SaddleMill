"""Calculator-free nonlinear-CG direction state for Dimer rotation.

rotation-CG owns only the Polak--Ribiere+ (PR+) direction recurrence and a thin helper
around the already-sealed canonical-history sphere step machinery.  It does not own
calculator calls, force-bank/QN pair admission, Dimer line/Fourier step
selection, public configuration, or runtime/factory wiring.

Sign convention
---------------
SaddleMill supplies a tangent rotational *force* ``F`` that is a descent
search direction for the curvature objective.  The mathematical gradient is
therefore ``g = -F``.  rotation-CG implements exactly

    beta_raw = <g_k, g_k - T(g_{k-1})> / <g_{k-1}, g_{k-1}>
    beta     = max(0, beta_raw)
    d_k      = -g_k + beta T(d_{k-1})
             = F_k + beta T(d_{k-1}).

A valid mixed direction satisfies ``<g_k,d_k> < 0``, equivalently
``<F_k,d_k> > 0``.  The Dimer mode is an unoriented axis.  When the current
axis is sign-aligned to the previous representative, attached tangent vectors
are multiplied by the same sign, matching canonical-history's convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan
from typing import Any, Mapping

import numpy as np

from saddlemill.dimertools.dense_bfgs import safe_norm
from saddlemill.dimertools.sphere_manifold import (
    normalize_axis,
    parallel_transport,
    project_tangent,
    sign_align_axis,
    take_sphere_step,
)

Array = np.ndarray

_STATE_SCHEMA = "saddlemill_pr_plus_rotation_cg_state_v1"
_RESET_POLICIES = frozenset({"every_translation", "bad_direction", "never"})
_MAP_KINDS = frozenset({"exponential", "retraction"})


def normalize_cg_reset_policy(value: object) -> str:
    """Normalize one of the exactly supported rotation-CG reset policies."""

    token = str(value).strip().lower()
    if token not in _RESET_POLICIES:
        allowed = ", ".join(sorted(_RESET_POLICIES))
        raise ValueError(f"cg_reset_policy must be one of: {allowed}")
    return token


def _finite_nonnegative(value: object, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _identity(value: object, *, name: str) -> str | int | float | bool | None:
    """Require deterministic JSON-scalar runtime identity tokens."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be a finite JSON scalar")
        return result
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"{name} must be a JSON scalar (str/int/float/bool/None)")


def _finite_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> Array:
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and nonempty")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")
    return array.copy()


@dataclass(frozen=True)
class PRPlusDirection:
    """One PR+ direction proposal and its recovery/transport diagnostics."""

    mode: Array
    direction: Array
    beta_raw: float
    beta_applied: float
    denominator: float
    numerator: float
    gradient_norm: float
    direction_norm: float
    gradient_dot_direction: float
    force_dot_direction: float
    descent_cosine: float
    tangency_error: float
    transported_gradient_norm: float
    transported_direction_norm: float
    representative_sign: float
    used_history: bool
    reset_reason: str
    reset_count: int
    transport_outcome: str
    center_id: str | int | float | bool | None
    sequence_id: str | int | float | bool | None
    mode_identity: str | int | float | bool | None
    state_age: int
    formula: str = "pr_plus"

    def diagnostics(self) -> dict[str, Any]:
        return {
            "rotation_cg_formula": self.formula,
            "rotation_cg_beta_raw": self.beta_raw,
            "rotation_cg_beta_applied": self.beta_applied,
            "rotation_cg_denominator": self.denominator,
            "rotation_cg_numerator": self.numerator,
            "rotation_cg_gradient_norm": self.gradient_norm,
            "rotation_cg_direction_norm": self.direction_norm,
            "rotation_cg_gradient_dot_direction": self.gradient_dot_direction,
            "rotation_cg_force_dot_direction": self.force_dot_direction,
            "rotation_cg_descent_cosine": self.descent_cosine,
            "rotation_cg_tangency_error": self.tangency_error,
            "rotation_cg_transported_gradient_norm": self.transported_gradient_norm,
            "rotation_cg_transported_direction_norm": self.transported_direction_norm,
            "rotation_cg_representative_sign": self.representative_sign,
            "rotation_cg_used_history": int(self.used_history),
            "rotation_cg_reset_reason": self.reset_reason,
            "rotation_cg_reset_count": self.reset_count,
            "rotation_cg_transport_outcome": self.transport_outcome,
            "rotation_cg_center_id": self.center_id,
            "rotation_cg_sequence_id": self.sequence_id,
            "rotation_cg_mode_identity": self.mode_identity,
            "rotation_cg_state_age": self.state_age,
        }


@dataclass(frozen=True)
class CGSphereStep:
    """Result of applying a caller-selected tangent proposal with canonical-history geometry."""

    axis: Array
    tangent: Array
    proposed_angle: float
    applied_angle: float
    angle_capped: bool
    retraction_outcome: str
    map_kind: str

    def diagnostics(self) -> dict[str, Any]:
        return {
            "rotation_cg_proposed_angle": self.proposed_angle,
            "rotation_cg_applied_angle": self.applied_angle,
            "rotation_cg_angle_capped": int(self.angle_capped),
            "rotation_cg_retraction_outcome": self.retraction_outcome,
            "rotation_cg_map_kind": self.map_kind,
        }


@dataclass
class _CGMemory:
    mode: Array
    gradient: Array
    direction: Array
    center_id: str | int | float | bool | None
    sequence_id: str | int | float | bool | None
    mode_identity: str | int | float | bool | None
    age: int


class PRPlusRotationCG:
    """Stateful PR+ nonlinear-CG recurrence on the unoriented unit sphere.

    ``propose`` computes a direction only.  Step length/line selection remains
    the Dimer adapter's responsibility.  ``apply_tangent_step`` exists only to
    ensure any selected tangent proposal reuses canonical-history's cap and retraction.
    """

    def __init__(
        self,
        *,
        reset_policy: str,
        denominator_tolerance: float = 1.0e-24,
        tangent_tolerance: float = 1.0e-14,
        descent_tolerance: float = 0.0,
    ) -> None:
        self.reset_policy = normalize_cg_reset_policy(reset_policy)
        self.denominator_tolerance = _finite_nonnegative(
            denominator_tolerance, name="denominator_tolerance"
        )
        self.tangent_tolerance = _finite_nonnegative(
            tangent_tolerance, name="tangent_tolerance"
        )
        self.descent_tolerance = _finite_nonnegative(
            descent_tolerance, name="descent_tolerance"
        )
        if self.descent_tolerance >= 1.0:
            raise ValueError("descent_tolerance must be < 1")

        self._previous: _CGMemory | None = None
        self.reset_count = 0
        self.last_reset_reason = "none"
        self.proposal_count = 0
        self.last_direction: PRPlusDirection | None = None
        self.last_sphere_step: CGSphereStep | None = None

    @property
    def state_age(self) -> int:
        return 0 if self._previous is None else int(self._previous.age)

    def reset(self, reason: str = "explicit_reset") -> None:
        token = str(reason).strip() or "explicit_reset"
        self._previous = None
        self.reset_count += 1
        self.last_reset_reason = token

    def _recover(self, reason: str) -> None:
        if self._previous is not None:
            self.reset(reason)
        else:
            self.last_reset_reason = str(reason)

    def _scheduled_or_identity_reset(
        self,
        *,
        mode_shape: tuple[int, ...],
        center_id: str | int | float | bool | None,
        sequence_id: str | int | float | bool | None,
        mode_identity: str | int | float | bool | None,
    ) -> str | None:
        previous = self._previous
        if previous is None:
            return None
        if previous.mode.shape != mode_shape:
            return "dimension_mismatch"
        if previous.sequence_id != sequence_id:
            return "sequence_boundary"
        if previous.mode_identity != mode_identity:
            return "mode_identity_mismatch"
        if previous.center_id != center_id and self.reset_policy == "every_translation":
            return "translation_boundary"
        return None

    @staticmethod
    def _steepest_metrics(
        mode: Array,
        gradient: Array,
        *,
        reason: str,
    ) -> tuple[Array, dict[str, Any]]:
        direction = np.asarray(project_tangent(mode, -gradient), dtype=float)
        gradient_norm = safe_norm(gradient)
        direction_norm = safe_norm(direction)
        gdotd = float(np.vdot(gradient.reshape(-1), direction.reshape(-1)).real)
        cosine = (
            -gdotd / (gradient_norm * direction_norm)
            if gradient_norm > 0.0 and direction_norm > 0.0
            else 0.0
        )
        return direction, {
            "beta_raw": 0.0,
            "beta_applied": 0.0,
            "denominator": 0.0,
            "numerator": 0.0,
            "transported_gradient_norm": 0.0,
            "transported_direction_norm": 0.0,
            "representative_sign": 1.0,
            "used_history": False,
            "reset_reason": reason,
            "transport_outcome": "not_used",
            "gradient_norm": gradient_norm,
            "direction_norm": direction_norm,
            "gradient_dot_direction": gdotd,
            "force_dot_direction": -gdotd,
            "descent_cosine": cosine,
        }

    def propose(
        self,
        mode: object,
        rotational_force: object,
        *,
        center_id: object,
        sequence_id: object,
        mode_identity: object = None,
    ) -> PRPlusDirection:
        """Return the next tangent PR+ direction.

        A center-ID change denotes an accepted center translation.  Only
        ``every_translation`` schedules a reset there.  ``bad_direction`` and
        ``never`` may transport history across translations, but all policies
        restart on numerical failure, non-descent, sequence identity change,
        mode/coordinate identity change, or dimension mismatch.
        """

        center = _identity(center_id, name="center_id")
        sequence = _identity(sequence_id, name="sequence_id")
        mode_id = _identity(mode_identity, name="mode_identity")

        shape = np.shape(mode)
        current_mode = normalize_axis(mode, shape=shape)
        current_force = _finite_array(
            rotational_force, name="rotational_force", shape=current_mode.shape
        )
        current_force = np.asarray(project_tangent(current_mode, current_force), dtype=float)
        gradient = -current_force
        gradient_norm = safe_norm(gradient)

        boundary_reason = self._scheduled_or_identity_reset(
            mode_shape=current_mode.shape,
            center_id=center,
            sequence_id=sequence,
            mode_identity=mode_id,
        )
        if boundary_reason is not None:
            self._recover(boundary_reason)

        if gradient_norm <= self.tangent_tolerance:
            self._recover("small_current_gradient")
            direction, metrics = self._steepest_metrics(
                current_mode, gradient, reason="small_current_gradient"
            )
            self._previous = None
            return self._finish_direction(
                current_mode,
                gradient,
                direction,
                center_id=center,
                sequence_id=sequence,
                mode_identity=mode_id,
                state_age=0,
                metrics=metrics,
            )

        previous = self._previous
        if previous is None:
            direction, metrics = self._steepest_metrics(
                current_mode,
                gradient,
                reason=boundary_reason or "initial_state",
            )
            age = 1
        else:
            denominator = float(
                np.vdot(previous.gradient.reshape(-1), previous.gradient.reshape(-1)).real
            )
            if not np.isfinite(denominator) or denominator <= self.denominator_tolerance:
                self._recover("small_or_nonfinite_denominator")
                direction, metrics = self._steepest_metrics(
                    current_mode, gradient, reason="small_or_nonfinite_denominator"
                )
                age = 1
            else:
                try:
                    aligned_mode, representative_sign = sign_align_axis(
                        previous.mode, current_mode
                    )
                    aligned_mode = aligned_mode.reshape(current_mode.shape)
                    gradient_aligned = representative_sign * gradient
                    transported_gradient = np.asarray(
                        project_tangent(
                            aligned_mode,
                            parallel_transport(previous.mode, aligned_mode, previous.gradient),
                        ),
                        dtype=float,
                    )
                    transported_direction = np.asarray(
                        project_tangent(
                            aligned_mode,
                            parallel_transport(previous.mode, aligned_mode, previous.direction),
                        ),
                        dtype=float,
                    )
                except (TypeError, ValueError, FloatingPointError):
                    self._recover("transport_failure")
                    direction, metrics = self._steepest_metrics(
                        current_mode, gradient, reason="transport_failure"
                    )
                    metrics["transport_outcome"] = "failed_restarted"
                    age = 1
                else:
                    tg_norm = safe_norm(transported_gradient)
                    td_norm = safe_norm(transported_direction)
                    if (
                        not np.all(np.isfinite(transported_gradient))
                        or not np.all(np.isfinite(transported_direction))
                        or tg_norm <= self.tangent_tolerance
                        or td_norm <= self.tangent_tolerance
                    ):
                        self._recover("invalid_transported_history")
                        direction, metrics = self._steepest_metrics(
                            current_mode, gradient, reason="invalid_transported_history"
                        )
                        metrics["transport_outcome"] = "invalid_restarted"
                        age = 1
                    else:
                        numerator = float(
                            np.vdot(
                                gradient_aligned.reshape(-1),
                                (gradient_aligned - transported_gradient).reshape(-1),
                            ).real
                        )
                        beta_raw = numerator / denominator
                        if not np.isfinite(beta_raw):
                            self._recover("nonfinite_beta")
                            direction, metrics = self._steepest_metrics(
                                current_mode, gradient, reason="nonfinite_beta"
                            )
                            metrics.update(
                                {
                                    "denominator": denominator,
                                    "numerator": numerator,
                                    "transported_gradient_norm": tg_norm,
                                    "transported_direction_norm": td_norm,
                                    "representative_sign": float(representative_sign),
                                    "transport_outcome": "ok_beta_restarted",
                                }
                            )
                            age = 1
                        else:
                            beta = max(0.0, beta_raw)
                            direction_aligned = np.asarray(
                                project_tangent(
                                    aligned_mode,
                                    -gradient_aligned + beta * transported_direction,
                                ),
                                dtype=float,
                            )
                            direction = np.asarray(
                                project_tangent(
                                    current_mode,
                                    representative_sign * direction_aligned,
                                ),
                                dtype=float,
                            )
                            direction_norm = safe_norm(direction)
                            gdotd = float(
                                np.vdot(gradient.reshape(-1), direction.reshape(-1)).real
                            )
                            threshold = -self.descent_tolerance * gradient_norm * direction_norm
                            bad = (
                                not np.all(np.isfinite(direction))
                                or direction_norm <= self.tangent_tolerance
                                or not np.isfinite(gdotd)
                                or gdotd >= threshold
                            )
                            if bad:
                                self._recover("non_descent_or_bad_direction")
                                direction, metrics = self._steepest_metrics(
                                    current_mode,
                                    gradient,
                                    reason="non_descent_or_bad_direction",
                                )
                                metrics.update(
                                    {
                                        "beta_raw": beta_raw,
                                        "denominator": denominator,
                                        "numerator": numerator,
                                        "transported_gradient_norm": tg_norm,
                                        "transported_direction_norm": td_norm,
                                        "representative_sign": float(representative_sign),
                                        "transport_outcome": "ok_direction_restarted",
                                    }
                                )
                                age = 1
                            else:
                                metrics = {
                                    "beta_raw": beta_raw,
                                    "beta_applied": beta,
                                    "denominator": denominator,
                                    "numerator": numerator,
                                    "transported_gradient_norm": tg_norm,
                                    "transported_direction_norm": td_norm,
                                    "representative_sign": float(representative_sign),
                                    "used_history": True,
                                    "reset_reason": "none",
                                    "transport_outcome": "ok",
                                    "gradient_norm": gradient_norm,
                                    "direction_norm": direction_norm,
                                    "gradient_dot_direction": gdotd,
                                    "force_dot_direction": -gdotd,
                                    "descent_cosine": -gdotd / (gradient_norm * direction_norm),
                                }
                                age = previous.age + 1

        self._previous = _CGMemory(
            mode=np.asarray(current_mode, dtype=float).copy(),
            gradient=np.asarray(gradient, dtype=float).copy(),
            direction=np.asarray(direction, dtype=float).copy(),
            center_id=center,
            sequence_id=sequence,
            mode_identity=mode_id,
            age=int(age),
        )
        return self._finish_direction(
            current_mode,
            gradient,
            direction,
            center_id=center,
            sequence_id=sequence,
            mode_identity=mode_id,
            state_age=age,
            metrics=metrics,
        )

    def _finish_direction(
        self,
        mode: Array,
        gradient: Array,
        direction: Array,
        *,
        center_id: str | int | float | bool | None,
        sequence_id: str | int | float | bool | None,
        mode_identity: str | int | float | bool | None,
        state_age: int,
        metrics: Mapping[str, Any],
    ) -> PRPlusDirection:
        direction = np.asarray(project_tangent(mode, direction), dtype=float)
        direction_norm = safe_norm(direction)
        gradient_norm = safe_norm(gradient)
        gdotd = float(np.vdot(gradient.reshape(-1), direction.reshape(-1)).real)
        tangency_error = abs(float(np.vdot(mode.reshape(-1), direction.reshape(-1)).real))
        result = PRPlusDirection(
            mode=np.asarray(mode, dtype=float).copy(),
            direction=direction.copy(),
            beta_raw=float(metrics.get("beta_raw", 0.0)),
            beta_applied=float(metrics.get("beta_applied", 0.0)),
            denominator=float(metrics.get("denominator", 0.0)),
            numerator=float(metrics.get("numerator", 0.0)),
            gradient_norm=gradient_norm,
            direction_norm=direction_norm,
            gradient_dot_direction=gdotd,
            force_dot_direction=-gdotd,
            descent_cosine=(
                -gdotd / (gradient_norm * direction_norm)
                if gradient_norm > 0.0 and direction_norm > 0.0
                else 0.0
            ),
            tangency_error=tangency_error,
            transported_gradient_norm=float(metrics.get("transported_gradient_norm", 0.0)),
            transported_direction_norm=float(metrics.get("transported_direction_norm", 0.0)),
            representative_sign=float(metrics.get("representative_sign", 1.0)),
            used_history=bool(metrics.get("used_history", False)),
            reset_reason=str(metrics.get("reset_reason", "none")),
            reset_count=int(self.reset_count),
            transport_outcome=str(metrics.get("transport_outcome", "not_used")),
            center_id=center_id,
            sequence_id=sequence_id,
            mode_identity=mode_identity,
            state_age=int(state_age),
        )
        self.proposal_count += 1
        self.last_direction = result
        return result

    def apply_tangent_step(
        self,
        mode: object,
        tangent_proposal: object,
        *,
        max_angle: float,
        map_kind: str,
    ) -> CGSphereStep:
        """Apply an adapter-selected tangent proposal with canonical-history cap/retraction.

        The caller owns how ``tangent_proposal`` was chosen (for example an
        existing Dimer Fourier-selected angle along the CG direction).  This
        helper only projects, caps, and retracts it.  It never inflates a small
        proposal to the cap.
        """

        shape = np.shape(mode)
        axis = normalize_axis(mode, shape=shape)
        tangent = _finite_array(tangent_proposal, name="tangent_proposal", shape=axis.shape)
        tangent = np.asarray(project_tangent(axis, tangent), dtype=float)
        angle_cap = _finite_positive(max_angle, name="max_angle")
        if angle_cap > np.pi / 2.0:
            raise ValueError("max_angle must satisfy 0 < max_angle <= pi/2")
        token = str(map_kind).strip().lower()
        if token not in _MAP_KINDS:
            raise ValueError("map_kind must be exponential or retraction")
        tangent_norm = safe_norm(tangent)
        proposed_angle = tangent_norm if token == "exponential" else float(atan(tangent_norm))
        raw = take_sphere_step(axis, tangent, max_angle=angle_cap, map_kind=token)
        result = CGSphereStep(
            axis=np.asarray(raw.axis, dtype=float).copy(),
            tangent=np.asarray(raw.tangent, dtype=float).copy(),
            proposed_angle=float(proposed_angle),
            applied_angle=float(raw.actual_angle),
            angle_capped=bool(raw.clipped),
            retraction_outcome=("zero_step" if raw.actual_angle <= 1.0e-15 else "applied"),
            map_kind=token,
        )
        self.last_sphere_step = result
        return result

    def state_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-serializable recurrence/resume state."""

        previous = self._previous
        previous_state: dict[str, Any] | None
        if previous is None:
            previous_state = None
        else:
            previous_state = {
                "mode": previous.mode.tolist(),
                "gradient": previous.gradient.tolist(),
                "direction": previous.direction.tolist(),
                "center_id": previous.center_id,
                "sequence_id": previous.sequence_id,
                "mode_identity": previous.mode_identity,
                "age": int(previous.age),
            }
        return {
            "schema": _STATE_SCHEMA,
            "formula": "pr_plus",
            "config": {
                "reset_policy": self.reset_policy,
                "denominator_tolerance": self.denominator_tolerance,
                "tangent_tolerance": self.tangent_tolerance,
                "descent_tolerance": self.descent_tolerance,
            },
            "counters": {
                "reset_count": int(self.reset_count),
                "last_reset_reason": self.last_reset_reason,
                "proposal_count": int(self.proposal_count),
            },
            "previous": previous_state,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PRPlusRotationCG":
        """Restore recurrence state produced by :meth:`state_dict`."""

        if not isinstance(state, Mapping):
            raise TypeError("CG state must be a mapping")
        if state.get("schema") != _STATE_SCHEMA:
            raise ValueError("unsupported CG state schema")
        if state.get("formula") != "pr_plus":
            raise ValueError("unsupported CG formula in state")
        config = state.get("config")
        counters = state.get("counters")
        if not isinstance(config, Mapping) or not isinstance(counters, Mapping):
            raise ValueError("CG state is missing config/counters mappings")
        obj = cls(
            reset_policy=str(config["reset_policy"]),
            denominator_tolerance=float(config["denominator_tolerance"]),
            tangent_tolerance=float(config["tangent_tolerance"]),
            descent_tolerance=float(config["descent_tolerance"]),
        )
        reset_count = int(counters.get("reset_count", 0))
        proposal_count = int(counters.get("proposal_count", 0))
        if reset_count < 0 or proposal_count < 0:
            raise ValueError("CG counters must be nonnegative")
        obj.reset_count = reset_count
        obj.proposal_count = proposal_count
        obj.last_reset_reason = str(counters.get("last_reset_reason", "none"))

        previous = state.get("previous")
        if previous is not None:
            if not isinstance(previous, Mapping):
                raise ValueError("CG previous state must be a mapping or null")
            mode = normalize_axis(previous["mode"])
            shape = mode.shape
            gradient = np.asarray(
                project_tangent(
                    mode,
                    _finite_array(previous["gradient"], name="gradient", shape=shape),
                ),
                dtype=float,
            )
            direction = np.asarray(
                project_tangent(
                    mode,
                    _finite_array(previous["direction"], name="direction", shape=shape),
                ),
                dtype=float,
            )
            age = int(previous["age"])
            if age < 1:
                raise ValueError("CG previous state age must be >= 1")
            obj._previous = _CGMemory(
                mode=np.asarray(mode, dtype=float).copy(),
                gradient=gradient,
                direction=direction,
                center_id=_identity(previous.get("center_id"), name="center_id"),
                sequence_id=_identity(previous.get("sequence_id"), name="sequence_id"),
                mode_identity=_identity(previous.get("mode_identity"), name="mode_identity"),
                age=age,
            )
        return obj


__all__ = [
    "CGSphereStep",
    "PRPlusDirection",
    "PRPlusRotationCG",
    "normalize_cg_reset_policy",
]
