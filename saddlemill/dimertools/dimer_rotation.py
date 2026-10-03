"""Dimer rotational solvers and generalized rotational L-BFGS.

The historical ``LBFGSDimerEigenmodeSearch`` remains available unchanged for
its exact default path.  ``GeneralizedLBFGSDimerEigenmodeSearch`` adds three
orthogonal opt-in choices:

* projected versus Riemannian secant transport;
* local versus raw-force-bank history;
* ASE trial-angle/Fourier rotation versus direct L-BFGS displacement.

All paths retain the physical Dimer torque and stopping criteria.  The direct
path makes one configured first rotation when no usable pair exists, then uses
the L-BFGS displacement itself, capped by a configured angle.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from math import atan, cos, pi, sin, tan
import json
from typing import Iterable, Mapping

import numpy as np

from ase.mep.dimer import DimerEigenmodeSearch, normalize, rotate_vectors

from saddlemill.dimertools.dimer_entry_torque import DimerEntryTorqueCaptureMixin
from saddlemill.dimertools.force_stencil_history import (
    build_force_bank_rotation_pairs,
)
from saddlemill.dimertools.dense_bfgs import safe_norm
from saddlemill.dimertools.riemannian_lbfgs import (
    RotationLBFGSModel,
    SequentialRotationHistory,
    make_rotation_secant,
    normalize_rotation_transport_policy,
    sequential_policy,
)
from saddlemill.dimertools.rotation_history import StateWindowRotationHistory
from saddlemill.dimertools.sphere_manifold import (
    exp_map,
    normalize_axis,
    project_tangent_and_basis,
    sphere_distance,
    take_sphere_step,
)
from saddlemill.dimertools.lbfgs_dimer import (
    LBFGSDimerEigenmodeSearch,
    LBFGSRotationMixin,
    LimitedMemoryInverseHessian,
)

Array = np.ndarray
norm = np.linalg.norm


@dataclass(frozen=True)
class _RotationPoint:
    mode: Array
    force: Array
    curvature: float
    serial: int
    physical: bool


def _degrees_to_radians(value: object, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0 or number > 90.0:
        raise ValueError(f"{name} must satisfy 0 < value <= 90 degrees")
    return number * pi / 180.0


def advanced_rotation_requested(
    lbfgs_options: Mapping[str, object] | None,
    history_options: Mapping[str, object] | None,
) -> bool:
    """Return whether the generalized search is needed.

    The exact historical projected/Fourier/local/legacy-skip path remains on
    the legacy class for numerical backward compatibility.
    """

    options = dict(lbfgs_options or {})
    history = dict(history_options or {})
    geometry = str(options.get("geometry", "projected")).strip().lower()
    transport_policy = normalize_rotation_transport_policy(
        options.get("transport_policy", "auto"), geometry=geometry
    )
    step_method = str(options.get("step_method", "fourier")).strip().lower()
    guard = str(options.get("curvature_guard", "legacy_skip")).strip().lower()
    rotation_reuse = bool(history.get("rotation_reuse", False))
    history_source = str(
        history.get("rotation_history_source", "legacy_derived")
    ).strip().lower()
    # Backward compatibility: the previously installed rotation_reuse=True
    # experiment stored derived rotational pairs.  Keep that exact legacy
    # implementation unless the new force_bank source is requested explicitly.
    if (
        rotation_reuse
        and history_source == "legacy_derived"
        and str(options.get("transport_policy", "auto")).strip().lower() in {"", "auto"}
    ):
        return False
    force_bank = rotation_reuse and history_source == "force_bank"
    reconstruction_model = str(
        options.get("reconstruction_model", "lbfgs")
    ).strip().lower()
    return bool(
        reconstruction_model != "lbfgs"
        or transport_policy != "double_projection"
        or geometry != "projected"
        or step_method != "fourier"
        or force_bank
        or guard != "legacy_skip"
    )


class GeneralizedLBFGSDimerEigenmodeSearch(DimerEntryTorqueCaptureMixin, DimerEigenmodeSearch):
    """Opt-in generalized rotational L-BFGS search."""

    def __init__(
        self,
        *args,
        lbfgs_options: Mapping[str, object] | None = None,
        canonical_force_history=None,
        history_options: Mapping[str, object] | None = None,
        **kwargs,
    ) -> None:
        self.lbfgs_options = dict(lbfgs_options or {})
        self.history_options = dict(history_options or {})
        self.canonical_force_history = canonical_force_history
        self.geometry = str(
            self.lbfgs_options.get("geometry", "projected")
        ).strip().lower()
        self.transport_policy = normalize_rotation_transport_policy(
            self.lbfgs_options.get("transport_policy", "auto"), geometry=self.geometry
        )
        self.step_method = str(
            self.lbfgs_options.get("step_method", "fourier")
        ).strip().lower()
        guard = str(
            self.lbfgs_options.get("curvature_guard", "skip")
        ).strip().lower()
        if guard == "auto":
            guard = "skip"
        if guard in {"legacy", "legacy_rotation_guard"}:
            guard = "legacy_skip"
        if guard == "shifted_secant":
            guard = "damp"
        self.curvature_guard = guard
        self.curvature_floor = float(
            self.lbfgs_options.get("curvature_floor", 1.0e-3)
        )
        self.powell_eta = float(self.lbfgs_options.get("powell_eta", 0.2))
        self.first_angle = _degrees_to_radians(
            self.lbfgs_options.get("first_angle_degrees", 45.0),
            "rotation_first_angle_degrees",
        )
        self.max_angle = _degrees_to_radians(
            self.lbfgs_options.get("max_angle_degrees", 45.0),
            "rotation_max_angle_degrees",
        )
        if self.geometry not in {"projected", "riemannian"}:
            raise ValueError("rotation geometry must be projected or riemannian")
        if self.step_method not in {"fourier", "direct"}:
            raise ValueError("rotation step_method must be fourier or direct")
        self.history_source = (
            "force_bank"
            if bool(self.history_options.get("rotation_reuse", False))
            and str(
                self.history_options.get("rotation_history_source", "legacy_derived")
            ).strip().lower()
            == "force_bank"
            else "local"
        )
        self.rotation_pair_sources = str(
            self.history_options.get("rotation_pair_sources", "consecutive_physical")
        ).strip().lower()
        self.rotation_accepted_force_source = str(
            self.history_options.get("rotation_accepted_force_source", "physical_only")
        ).strip().lower()
        self.trial_pairs_future_only = bool(
            self.history_options.get("rotation_trial_pairs_future_only", False)
        )
        self.sequential_history = None
        self._sequential_memory_states = int(
            self.history_options.get("memory_states", 20)
            if self.history_source == "force_bank" else 1
        )


        max_pairs = int(
            self.history_options.get("rotation_max_pairs", 0)
            if self.history_source == "force_bank"
            else self.lbfgs_options.get("memory", 10)
        )
        self.model = RotationLBFGSModel(
            initial_hessian=float(
                self.lbfgs_options.get("initial_hessian", 1.0)
            ),
            dynamic_h0=bool(self.lbfgs_options.get("dynamic_h0", False)),
            curvature_guard=self.curvature_guard,
            curvature_floor=self.curvature_floor,
            curvature_epsilon=float(
                self.lbfgs_options.get("curvature_epsilon", 1.0e-12)
            ),
            powell_eta=self.powell_eta,
            cosine_threshold=(
                self.history_options.get("rotation_cosine_threshold")
                if self.history_source == "force_bank"
                else self.lbfgs_options.get("cosine_threshold")
            ),
            max_pairs=max_pairs,
            reconstruction_model=str(
                self.lbfgs_options.get("reconstruction_model", "lbfgs")
            ).strip().lower(),
            bfgs_update=str(
                self.lbfgs_options.get("bfgs_update", "sequential")
            ).strip().lower(),
            dense_bfgs_diagnostic=bool(
                self.lbfgs_options.get("dense_bfgs_diagnostic", False)
            ),
        )
        self._sample_serial = 0
        self._stencil_serial = 0
        self._previous_local_point: _RotationPoint | None = None
        self._previous_sequential_physical_point: _RotationPoint | None = None
        self._force_bank_disabled = False
        self._direction_fallbacks = 0
        self._fixed_first_steps = 0
        self._direct_steps = 0
        self._fourier_steps = 0
        self._extrapolated_accepted_samples_recorded = 0
        self._step_clips = 0
        self._requested_angle_sum = 0.0
        self._actual_angle_sum = 0.0
        self._rotation_trace: list[dict[str, object]] = []
        self._last_build_metrics: dict[str, object] = {}
        self._last_model_metrics: dict[str, object] = {}
        super().__init__(*args, **kwargs)
        if sequential_policy(self.transport_policy):
            memory_states = self._sequential_memory_states
            if self.history_source == "force_bank":
                existing = getattr(self.dimeratoms, "_sm_sequential_rotation_history", None)
                if (
                    isinstance(existing, SequentialRotationHistory)
                    and existing.policy == self.transport_policy
                    and existing.memory_states == memory_states
                ):
                    self.sequential_history = existing
                else:
                    self.sequential_history = SequentialRotationHistory(
                        policy=self.transport_policy, memory_states=memory_states
                    )
                    setattr(self.dimeratoms, "_sm_sequential_rotation_history", self.sequential_history)
            else:
                self.sequential_history = SequentialRotationHistory(
                    policy=self.transport_policy, memory_states=memory_states
                )
            self.sequential_history.begin_state(
                self._state_id(), self.eigenmode, basis=self.basis
            )

    def _project(self, vector: object, mode: object) -> Array:
        return project_tangent_and_basis(mode, vector, self.basis)

    def _state_id(self) -> int:
        history = self.canonical_force_history
        state = None if history is None else history.current
        return -1 if state is None else int(state.state_id)

    def _record_extrapolated_accepted_point(
        self, mode: object, *, iteration: int
    ) -> _RotationPoint | None:
        """Record ASE's derived force at a Fourier-accepted orientation.

        This is opt-in force-bank history.  It performs no PES call and does not
        mutate the Dimer force state.  For use_central_forces=True, ASE's
        extrapolated endpoint force plus the physical center force reconstructs
        exactly the rotational torque that update_virtual_forces(
        extrapolated_forces=True) would use at the accepted orientation.
        """

        if (
            self.history_source != "force_bank"
            or self.rotation_accepted_force_source != "allow_extrapolated"
            or "accepted_rotation" not in self.rotation_pair_sources.split()
            or self.forces1E is None
        ):
            return None
        if not self.control.get_parameter("use_central_forces"):
            raise ValueError(
                "[ourDimerHistory] rotation_accepted_force_source=allow_extrapolated "
                "currently requires [DimerControl] use_central_forces=True; with "
                "use_central_forces=False ASE would require a physical opposite-end "
                "force to define the accepted-point rotational torque"
            )

        mode_array = normalize_axis(mode, shape=np.shape(self.eigenmode))
        endpoint_force = np.asarray(self.forces1E, dtype=float).copy()
        center_force = np.asarray(self.forces0, dtype=float)
        torque = self._project((endpoint_force - center_force) / self.dR, mode_array)
        curvature = float(
            np.vdot((center_force - endpoint_force).ravel(), mode_array.ravel()).real
            / self.dR
        )
        point = _RotationPoint(
            mode=mode_array.copy(),
            force=np.asarray(torque, dtype=float).copy(),
            curvature=curvature,
            serial=self._sample_serial,
            physical=False,
        )
        self._sample_serial += 1

        history = self.canonical_force_history
        if history is not None:
            stencil_id = (
                f"dimer-state{self._state_id()}-order"
                f"{0 if self.basis is None else len(self.basis)}-"
                f"accepted{self._stencil_serial}"
            )
            self._stencil_serial += 1
            history.observe_probe(
                np.asarray(self.pos0, dtype=float) + mode_array * self.dR,
                endpoint_force,
                source="rotation_extrapolated_accepted",
                family="rotation",
                evaluation="derived_extrapolated",
                force_call_delta=0,
                metadata={
                    "direction_kind": "dimer_mode",
                    "probe_direction": mode_array.copy(),
                    "dimer_mode": mode_array.copy(),
                    "dimer_separation": float(self.dR),
                    "stencil_id": stencil_id,
                    "stencil_scheme": "one_sided",
                    "rotation_sample_kind": "fourier_accepted",
                    "rotation_iteration": int(iteration),
                    "rotation_geometry": self.geometry,
                    "rotation_step_method": self.step_method,
                    "rotation_extrapolated_accepted": 1,
                    "accepted_force_provenance": "ase_fourier_extrapolated",
                },
                direction=mode_array,
                direction_kind="dimer_mode",
                stencil_id=stencil_id,
                dimer_side=1,
                offset=float(self.dR),
            )
        self._extrapolated_accepted_samples_recorded += 1
        return point

    def _source_context(
        self,
        *,
        mode: Array,
        sample_kind: str,
        iteration: int,
        stencil_id: str,
    ):
        evaluator = getattr(self.dimeratoms, "physical_force_evaluator", None)
        if evaluator is None:
            return nullcontext()
        metadata = {
            "direction_kind": "dimer_mode",
            "probe_direction": np.asarray(mode, dtype=float).copy(),
            "dimer_mode": np.asarray(mode, dtype=float).copy(),
            "dimer_separation": float(self.dR),
            "stencil_id": str(stencil_id),
            "stencil_scheme": (
                "one_sided"
                if self.control.get_parameter("use_central_forces")
                else "centered"
            ),
            "rotation_sample_kind": str(sample_kind),
            "rotation_iteration": int(iteration),
            "rotation_geometry": self.geometry,
            "rotation_step_method": self.step_method,
        }
        return evaluator.source(
            "rotation_dimer_endpoint", metadata, family="rotation"
        )

    def _evaluate_rotation_point(
        self,
        mode: object,
        *,
        sample_kind: str,
        iteration: int,
        allow_extrapolated: bool = False,
    ) -> _RotationPoint:
        mode_array = normalize_axis(mode, shape=np.shape(self.eigenmode))
        self.eigenmode = mode_array
        used_extrapolated = bool(allow_extrapolated and self.forces1E is not None)
        if used_extrapolated:
            self.update_virtual_forces(extrapolated_forces=True)
        else:
            stencil_id = (
                f"dimer-state{self._state_id()}-order{0 if self.basis is None else len(self.basis)}-"
                f"sample{self._stencil_serial}"
            )
            self._stencil_serial += 1
            with self._source_context(
                mode=mode_array,
                sample_kind=sample_kind,
                iteration=iteration,
                stencil_id=stencil_id,
            ):
                self.update_virtual_forces()
        self.update_curvature()
        torque = self._project(self.get_rotational_force(), mode_array)
        point = _RotationPoint(
            mode=mode_array.copy(),
            force=np.asarray(torque, dtype=float).copy(),
            curvature=float(self.get_curvature()),
            serial=self._sample_serial,
            physical=not used_extrapolated,
        )
        self._sample_serial += 1
        return point

    def _add_local_point(self, point: _RotationPoint) -> None:
        self._add_sequential_physical_point(point)
        if self.history_source != "local" or self.sequential_history is not None:
            return
        previous = self._previous_local_point
        if previous is not None:
            pair = make_rotation_secant(
                previous.mode,
                previous.force,
                point.mode,
                point.force,
                geometry=self.transport_policy,
                state_id=self._state_id(),
                serial=point.serial,
                source="local_dimer_rotation",
                basis=self.basis,
                metadata={
                    "first_physical": int(previous.physical),
                    "second_physical": int(point.physical),
                },
            )
            self.model.add_pair(pair)
        self._previous_local_point = point

    def _add_sequential_physical_point(self, point: _RotationPoint) -> None:
        """Admit the selected consecutive physical samples at this center.

        Fourier trial points and accepted points are both force observations.
        Retain their secants in the live accepted tangent space, just as for
        explicitly selected Fourier trial pairs. The previous sample belongs
        to this eigensolve only; translation between centers is not an angular
        secant, while the history's existing pairs survive the center change.
        """
        if self.sequential_history is None or not point.physical:
            return
        previous = self._previous_sequential_physical_point
        if previous is not None and "consecutive_physical" in self.rotation_pair_sources.split():
            self.sequential_history.add_trial_pair(
                previous.mode, previous.force, point.mode, point.force,
                state_id=self._state_id(), serial=point.serial, basis=self.basis,
                source="sequential_consecutive_physical",
            )
        self._previous_sequential_physical_point = point

    def _external_pairs(self):
        if self._force_bank_disabled:
            self._last_build_metrics = {}
            return ()
        if self.sequential_history is not None:
            metrics = self.sequential_history.metrics()
            pairs = self.sequential_history.pairs(
                trial_future_only=self.trial_pairs_future_only
            )
            metrics.update({
                "pairs_built": len(pairs),
                "pair_sources": self.rotation_pair_sources,
                "states_contributing": len({pair.state_id for pair in pairs}),
                "build_ns": 0,
            })
            self._last_build_metrics = metrics
            return pairs
        if self.history_source != "force_bank" or self.canonical_force_history is None:
            self._last_build_metrics = {}
            return ()
        build = build_force_bank_rotation_pairs(
            self.canonical_force_history,
            geometry=self.transport_policy,
            basis=self.basis,
            # The RotationLBFGSModel applies rotation_max_pairs after
            # transport and pair admission.  Do not pre-truncate raw force-bank
            # candidates here or cosine/curvature rejects cannot be backfilled.
            max_pairs=0,
            pair_sources=self.rotation_pair_sources,
            accepted_force_source=self.rotation_accepted_force_source,
            trial_pairs_future_only=self.trial_pairs_future_only,
            current_state_id=self._state_id(),
        )
        self._last_build_metrics = dict(build.metrics)
        return build.pairs

    def _rotation_direction(self, point: _RotationPoint) -> tuple[Array, int]:
        external = self._external_pairs()
        result = self.model.apply(
            point.force,
            point.mode,
            external_pairs=external,
            basis=self.basis,
        )
        self._last_model_metrics = dict(result.metrics)
        direction = self._project(result.direction, point.mode)
        pairs_used = int(result.metrics.get("pairs_used", 0))
        if (
            not np.all(np.isfinite(direction))
            or safe_norm(direction) <= 1.0e-14
            or float(np.vdot(direction.ravel(), point.force.ravel()).real) <= 0.0
        ):
            self.model.reset("non_descent_rotation_direction")
            if self.sequential_history is not None:
                self.sequential_history.reset("non_descent_rotation_direction")
            self._direction_fallbacks += 1
            if self.history_source == "force_bank" and self.sequential_history is None:
                self._force_bank_disabled = True
            direction = point.force.copy()
            pairs_used = 0
        return np.asarray(direction, dtype=float), pairs_used

    def _diagnostics(self) -> None:
        model = dict(self._last_model_metrics)
        build = dict(self._last_build_metrics)
        latest_rejections = model.get("pairs_rejected_by_reason", {})
        self.lbfgs_diagnostics = {
            "optimizer": "lbfgs",
            "rotations": int(self.control.get_counter("rotcount")),
            "history_size": int(model.get("pairs_used", 0)),
            "pairs_accepted": int(model.get("pairs_used", 0)),
            "pairs_rejected": int(
                model.get("pairs_rejected_transport", 0)
                + sum(int(value) for value in dict(latest_rejections or {}).values())
            ),
            "history_resets": int(self.model.reset_count),
            "direction_fallbacks": int(self._direction_fallbacks),
            "memory": int(self.model.max_pairs),
            "initial_hessian": float(self.model.initial_hessian),
            "initial_inverse_hessian_scale": 1.0 / float(self.model.initial_hessian),
            "latest_pair_accepted": int(bool(model.get("pairs_used", 0))),
            "latest_s_norm": "",
            "latest_y_norm": "",
            "latest_s_dot_y": model.get("latest_stored_s_dot_y", ""),
            "latest_secant_curvature": "",
            "latest_secant_cosine": "",
            "latest_force_change_norm": "",
            "rotation_lbfgs_geometry": self.geometry,
            "rotation_lbfgs_transport_policy": self.transport_policy,
            "rotation_lbfgs_pair_sources": self.rotation_pair_sources,
            "rotation_lbfgs_accepted_force_source": self.rotation_accepted_force_source,
            "rotation_lbfgs_extrapolated_accepted_samples_recorded": int(
                self._extrapolated_accepted_samples_recorded
            ),
            "rotation_lbfgs_trial_pairs_future_only": int(self.trial_pairs_future_only),
            "rotation_lbfgs_step_method": self.step_method,
            "rotation_lbfgs_history_source": self.history_source,
            "rotation_lbfgs_curvature_guard": self.curvature_guard,
            "rotation_lbfgs_curvature_floor": self.curvature_floor,
            "rotation_lbfgs_powell_eta": self.powell_eta,
            "rotation_lbfgs_first_angle_degrees": self.first_angle * 180.0 / pi,
            "rotation_lbfgs_max_angle_degrees": self.max_angle * 180.0 / pi,
            "rotation_lbfgs_fixed_first_steps": self._fixed_first_steps,
            "rotation_lbfgs_direct_steps": self._direct_steps,
            "rotation_lbfgs_fourier_steps": self._fourier_steps,
            "rotation_lbfgs_step_clips": self._step_clips,
            "rotation_lbfgs_requested_angle_sum": self._requested_angle_sum,
            "rotation_lbfgs_actual_angle_sum": self._actual_angle_sum,
            "rotation_lbfgs_rotation_trace_json": json.dumps(self._rotation_trace, sort_keys=True, separators=(",", ":")),
            "rotation_lbfgs_cosine_threshold": "" if self.model.cosine_threshold is None else self.model.cosine_threshold,
            "rotation_lbfgs_pair_candidates": model.get("pair_candidates", 0),
            "rotation_lbfgs_pair_candidates_after_max_pairs": model.get("pair_candidates_after_max_pairs", 0),
            "rotation_lbfgs_pairs_admissible_before_max_pairs": model.get("pairs_admissible_before_max_pairs", 0),
            "rotation_lbfgs_pairs_removed_by_max_pairs": model.get("pairs_removed_by_max_pairs", 0),
            "rotation_lbfgs_pairs_used": model.get("pairs_used", 0),
            "rotation_lbfgs_pairs_rejected_curvature": model.get("pairs_rejected_curvature", 0),
            "rotation_lbfgs_pairs_rejected_cosine": model.get("pairs_rejected_cosine", 0),
            "rotation_lbfgs_accepted_curvature_min": model.get("accepted_curvature_min", ""),
            "rotation_lbfgs_accepted_curvature_median": model.get("accepted_curvature_median", ""),
            "rotation_lbfgs_accepted_curvature_max": model.get("accepted_curvature_max", ""),
            "rotation_lbfgs_accepted_cosine_min": model.get("accepted_cosine_min", ""),
            "rotation_lbfgs_accepted_cosine_median": model.get("accepted_cosine_median", ""),
            "rotation_lbfgs_accepted_cosine_max": model.get("accepted_cosine_max", ""),
            "rotation_lbfgs_pairs_damped": model.get("pairs_damped", 0),
            "rotation_lbfgs_pairs_powell_damped": model.get(
                "pairs_powell_damped", 0
            ),
            "rotation_lbfgs_pairs_rejected_transport": model.get(
                "pairs_rejected_transport", 0
            ),
            "rotation_lbfgs_pairs_rejected_json": json.dumps(
                latest_rejections, sort_keys=True, separators=(",", ":")
            ),
            "rotation_lbfgs_transport_s_norm_ratio_median": model.get("transport_s_norm_ratio_median", ""),
            "rotation_lbfgs_transport_s_norm_ratio_min": model.get("transport_s_norm_ratio_min", ""),
            "rotation_lbfgs_transport_s_norm_ratio_max": model.get("transport_s_norm_ratio_max", ""),
            "rotation_lbfgs_transport_y_norm_ratio_median": model.get("transport_y_norm_ratio_median", ""),
            "rotation_lbfgs_transport_y_norm_ratio_min": model.get("transport_y_norm_ratio_min", ""),
            "rotation_lbfgs_transport_y_norm_ratio_max": model.get("transport_y_norm_ratio_max", ""),
            "rotation_lbfgs_transport_pair_transform_count_max": model.get("transport_pair_transform_count_max", 0),
            "rotation_lbfgs_transport_pair_path_angle_max": model.get("transport_pair_path_angle_max", 0.0),
            "rotation_lbfgs_latest_guard_action": model.get(
                "latest_pair_action", ""
            ),
            "rotation_lbfgs_latest_raw_s_dot_y": model.get(
                "latest_raw_s_dot_y", ""
            ),
            "rotation_lbfgs_latest_stored_s_dot_y": model.get(
                "latest_stored_s_dot_y", ""
            ),
            "rotation_lbfgs_latest_powell_theta": model.get(
                "latest_powell_theta", ""
            ),
            "rotation_lbfgs_latest_s_dot_Bs": model.get("latest_s_dot_Bs", ""),
            "rotation_lbfgs_reconstruction_ns": model.get("apply_ns", 0),
            "rotation_lbfgs_reconstruction_cumulative_ns": model.get(
                "apply_cumulative_ns", 0
            ),
            "rotation_lbfgs_reconstruction_model": model.get(
                "reconstruction_model", "lbfgs"
            ),
            "rotation_lbfgs_bfgs_update": model.get("bfgs_update", "sequential"),
            "rotation_dense_bfgs_diagnostic": model.get(
                "dense_bfgs_diagnostic", 0
            ),
            "rotation_lbfgs_raw_direction_norm": model.get(
                "lbfgs_raw_direction_norm", ""
            ),
            "rotation_lbfgs_raw_direction_max_atom_norm": model.get(
                "lbfgs_raw_direction_max_atom_norm", ""
            ),
            "rotation_dense_vs_lbfgs_cosine": model.get(
                "dense_vs_lbfgs_cosine", ""
            ),
            "rotation_dense_vs_lbfgs_norm_ratio": model.get(
                "dense_vs_lbfgs_norm_ratio", ""
            ),
            "rotation_dense_vs_lbfgs_relative_difference": model.get(
                "dense_vs_lbfgs_relative_difference", ""
            ),
            "rotation_dense_min_eigenvalue": model.get("dense_min_eigenvalue", ""),
            "rotation_dense_max_eigenvalue": model.get("dense_max_eigenvalue", ""),
            "rotation_dense_condition_number": model.get(
                "dense_condition_number", ""
            ),
            "rotation_dense_log10_condition": model.get(
                "dense_log10_condition", ""
            ),
            "rotation_dense_negative_eigenvalues": model.get(
                "dense_negative_eigenvalues", ""
            ),
            "rotation_dense_secant_residual_latest": model.get(
                "dense_secant_residual_latest", ""
            ),
            "rotation_dense_secant_residual_median": model.get(
                "dense_secant_residual_median", ""
            ),
            "rotation_dense_secant_residual_max": model.get(
                "dense_secant_residual_max", ""
            ),
            "rotation_dense_multisecant_blocks_applied": model.get(
                "dense_multisecant_blocks_applied", ""
            ),
            "rotation_dense_multisecant_pairs_skipped_rank": model.get(
                "dense_multisecant_pairs_skipped_rank", ""
            ),
            "rotation_dense_multisecant_pairs_skipped_curvature": model.get(
                "dense_multisecant_pairs_skipped_curvature", ""
            ),
            "rotation_dense_multisecant_blocks_requested": model.get(
                "dense_multisecant_blocks_requested", ""
            ),
            "rotation_dense_multisecant_pairs_skipped_invalid": model.get(
                "dense_multisecant_pairs_skipped_invalid", ""
            ),
            "rotation_dense_multisecant_blocks_rejected_update": model.get(
                "dense_multisecant_blocks_rejected_update", ""
            ),
            "rotation_dense_multisecant_block_size_median": model.get("dense_multisecant_block_size_median", ""),
            "rotation_dense_multisecant_block_size_max": model.get("dense_multisecant_block_size_max", ""),
            "rotation_dense_multisecant_block_kept_median": model.get("dense_multisecant_block_kept_median", ""),
            "rotation_dense_multisecant_y_symmetrization_relative_median": model.get("dense_multisecant_y_symmetrization_relative_median", ""),
            "rotation_dense_multisecant_y_symmetrization_relative_max": model.get("dense_multisecant_y_symmetrization_relative_max", ""),
            "rotation_dense_multisecant_sty_asymmetry_relative_median": model.get("dense_multisecant_sty_asymmetry_relative_median", ""),
            "rotation_dense_multisecant_sty_asymmetry_relative_max": model.get("dense_multisecant_sty_asymmetry_relative_max", ""),
            "rotation_dense_multisecant_sts_condition_median": model.get("dense_multisecant_sts_condition_median", ""),
            "rotation_dense_multisecant_sts_condition_max": model.get("dense_multisecant_sts_condition_max", ""),
            "rotation_dense_multisecant_yts_condition_median": model.get("dense_multisecant_yts_condition_median", ""),
            "rotation_dense_multisecant_yts_condition_max": model.get("dense_multisecant_yts_condition_max", ""),
            "rotation_dense_multisecant_stbs_condition_median": model.get("dense_multisecant_stbs_condition_median", ""),
            "rotation_dense_multisecant_stbs_condition_max": model.get("dense_multisecant_stbs_condition_max", ""),
            "rotation_dense_multisecant_sty_original_min_eigenvalue_median": model.get("dense_multisecant_sty_original_min_eigenvalue_median", ""),
            "rotation_dense_multisecant_sty_original_min_eigenvalue_max": model.get("dense_multisecant_sty_original_min_eigenvalue_max", ""),
            "rotation_dense_multisecant_sty_adjusted_min_eigenvalue_median": model.get("dense_multisecant_sty_adjusted_min_eigenvalue_median", ""),
            "rotation_dense_multisecant_sty_adjusted_min_eigenvalue_max": model.get("dense_multisecant_sty_adjusted_min_eigenvalue_max", ""),
            "rotation_dense_multisecant_residual_original_y_median": model.get("dense_multisecant_residual_original_y_median", ""),
            "rotation_dense_multisecant_residual_original_y_max": model.get("dense_multisecant_residual_original_y_max", ""),
            "rotation_dense_multisecant_residual_adjusted_y_median": model.get("dense_multisecant_residual_adjusted_y_median", ""),
            "rotation_dense_multisecant_residual_adjusted_y_max": model.get("dense_multisecant_residual_adjusted_y_max", ""),
            "rotation_dense_pairs_requested": model.get("dense_pairs_requested", ""),
            "rotation_dense_pairs_applied": model.get("dense_pairs_applied", ""),
            "rotation_dense_invalid_reason": model.get("dense_invalid_reason", ""),
            "rotation_dense_reconstruction_ns": model.get(
                "dense_reconstruction_ns", ""
            ),
            "canonical_rotation_reuse_enabled": int(
                self.history_source == "force_bank"
            ),
            "canonical_rotation_torque_samples": build.get("torque_samples", 0),
            "canonical_rotation_pair_candidates": build.get(
                "pair_candidates", 0
            ),
            "canonical_rotation_pairs_built": build.get("pairs_built", 0),
            "canonical_rotation_pairs_degenerate": build.get(
                "pairs_degenerate", 0
            ),
            "canonical_rotation_states_contributing": build.get(
                "states_contributing", 0
            ),
            "canonical_rotation_stencil_build_ns": build.get("build_ns", 0),
            "canonical_rotation_pair_sources": build.get("pair_sources", self.rotation_pair_sources),
            "canonical_rotation_accepted_force_source": build.get(
                "accepted_force_source", self.rotation_accepted_force_source
            ),
            "canonical_rotation_derived_torque_samples": build.get(
                "derived_torque_samples", self._extrapolated_accepted_samples_recorded
            ),
            "canonical_rotation_pair_candidates_by_source_json": json.dumps(
                build.get("pair_candidates_by_source", {}), sort_keys=True, separators=(",", ":")
            ),
            "canonical_rotation_pairs_built_by_source_json": json.dumps(
                build.get("pairs_built_by_source", {}), sort_keys=True, separators=(",", ":")
            ),
            "rotation_lbfgs_sequential_history_pairs": build.get("sequential_history_pairs", ""),
            "rotation_lbfgs_sequential_history_advance_steps": build.get("sequential_history_advance_steps", ""),
            "rotation_lbfgs_sequential_history_state_sync_steps": build.get("sequential_history_state_sync_steps", ""),
            "rotation_lbfgs_sequential_history_vectors_transformed": build.get("sequential_history_vectors_transformed", ""),
            "rotation_lbfgs_sequential_history_s_norm_step_ratio_median": build.get("sequential_s_norm_step_ratio_median", ""),
            "rotation_lbfgs_sequential_history_s_norm_step_ratio_min": build.get("sequential_s_norm_step_ratio_min", ""),
            "rotation_lbfgs_sequential_history_s_norm_step_ratio_max": build.get("sequential_s_norm_step_ratio_max", ""),
            "rotation_lbfgs_sequential_history_y_norm_step_ratio_median": build.get("sequential_y_norm_step_ratio_median", ""),
            "rotation_lbfgs_sequential_history_y_norm_step_ratio_min": build.get("sequential_y_norm_step_ratio_min", ""),
            "rotation_lbfgs_sequential_history_y_norm_step_ratio_max": build.get("sequential_y_norm_step_ratio_max", ""),
            "canonical_rotation_force_bank_disabled_current_center": int(
                self._force_bank_disabled
            ),
        }

    def _converge_fourier(self) -> None:
        self.set_up_for_eigenmode_search()
        stoprot = False
        f_rot_min = self.control.get_parameter("f_rot_min")
        f_rot_max = self.control.get_parameter("f_rot_max")
        trial_angle = self.control.get_parameter("trial_angle")
        max_num_rot = self.control.get_parameter("max_num_rot")
        extrapolate = self.control.get_parameter("extrapolate_forces")
        iteration = 0
        previous_accepted_point = None
        while not stoprot:
            point_a = self._evaluate_rotation_point(
                self.eigenmode,
                sample_kind="fourier_a",
                iteration=iteration,
                allow_extrapolated=True,
            )
            self.forces1A = self.forces1
            self._add_local_point(point_a)
            if (
                self.sequential_history is not None
                and "accepted_rotation" in self.rotation_pair_sources.split()
                and previous_accepted_point is not None
                and previous_accepted_point.physical
                and point_a.physical
            ):
                # With extrapolation disabled, the next A evaluation supplies
                # the physical force at the last Fourier-accepted orientation.
                # That accepted secant used to be omitted in sequential mode.
                self.sequential_history.add_accepted_pair(
                    previous_accepted_point.mode, previous_accepted_point.force,
                    point_a.mode, point_a.force,
                    state_id=self._state_id(), serial=point_a.serial,
                    basis=self.basis, source="sequential_accepted_rotation_physical",
                )
            previous_accepted_point = point_a
            f_rot_A = point_a.force
            if norm(f_rot_A) <= f_rot_min:
                self.log(f_rot_A, None)
                stoprot = True
            else:
                n_A = point_a.mode
                direction, _ = self._rotation_direction(point_a)
                rot_unit_A = normalize(direction)
                c0 = point_a.curvature
                c0d = np.vdot((self.forces2 - self.forces1), rot_unit_A) / self.dR
                n_B, rot_unit_B = rotate_vectors(n_A, rot_unit_A, trial_angle)
                point_b = self._evaluate_rotation_point(
                    n_B,
                    sample_kind="fourier_trial_b",
                    iteration=iteration,
                    allow_extrapolated=False,
                )
                self._add_sequential_physical_point(point_b)
                if (
                    self.sequential_history is not None
                    and "fourier_trial" in self.rotation_pair_sources.split()
                    and point_a.physical
                    and point_b.physical
                ):
                    self.sequential_history.add_trial_pair(
                        point_a.mode, point_a.force, point_b.mode, point_b.force,
                        state_id=self._state_id(), serial=point_b.serial, basis=self.basis,
                        source="sequential_fourier_trial",
                    )
                self.forces1B = self.forces1
                c1d = np.vdot((self.forces2 - self.forces1), rot_unit_B) / self.dR
                a1 = c0d * cos(2 * trial_angle) - c1d / (2 * sin(2 * trial_angle))
                b1 = 0.5 * c0d
                a0 = 2 * (c0 - a1)
                rotangle = atan(b1 / a1) / 2.0
                cmin = a0 / 2.0 + a1 * cos(2 * rotangle) + b1 * sin(2 * rotangle)
                if c0 < cmin:
                    rotangle += pi / 2.0
                n_min, _ = rotate_vectors(n_A, rot_unit_A, rotangle)
                if self.sequential_history is not None:
                    self.sequential_history.advance(n_min, basis=self.basis)
                self.update_eigenmode(n_min)
                self.update_curvature(cmin)
                self.log(f_rot_A, rotangle)
                self._fourier_steps += 1
                self._requested_angle_sum += abs(float(rotangle))
                _actual_angle = sphere_distance(n_A, n_min)
                self._actual_angle_sum += _actual_angle
                self._rotation_trace.append({
                    "iteration": int(iteration),
                    "state_id": int(self._state_id()),
                    "trial_angle": float(trial_angle),
                    "fourier_selected_angle": float(rotangle),
                    "actual_sphere_angle": float(_actual_angle),
                    "angle_cap_hit": 0,
                    "pair_candidates": int(self._last_model_metrics.get("pair_candidates", 0) or 0),
                    "pairs_used": int(self._last_model_metrics.get("pairs_used", 0) or 0),
                    "raw_direction_norm": self._last_model_metrics.get("lbfgs_raw_direction_norm", ""),
                    "accepted_curvature_min": self._last_model_metrics.get("accepted_curvature_min", ""),
                    "accepted_curvature_median": self._last_model_metrics.get("accepted_curvature_median", ""),
                    "accepted_curvature_max": self._last_model_metrics.get("accepted_curvature_max", ""),
                    "accepted_cosine_min": self._last_model_metrics.get("accepted_cosine_min", ""),
                    "accepted_cosine_median": self._last_model_metrics.get("accepted_cosine_median", ""),
                    "accepted_cosine_max": self._last_model_metrics.get("accepted_cosine_max", ""),
                    "qn_apply_ns": int(self._last_model_metrics.get("apply_ns", 0) or 0),
                })
                if extrapolate:
                    self.forces1E = (
                        sin(trial_angle - rotangle) / sin(trial_angle) * self.forces1A
                        + sin(rotangle) / sin(trial_angle) * self.forces1B
                        + (
                            1
                            - cos(rotangle)
                            - sin(rotangle) * tan(trial_angle / 2.0)
                        )
                        * self.forces0
                    )
                    accepted_point = self._record_extrapolated_accepted_point(
                        n_min, iteration=iteration + 1
                    )
                    if (
                        accepted_point is not None
                        and self.sequential_history is not None
                        and "accepted_rotation" in self.rotation_pair_sources.split()
                    ):
                        self.sequential_history.add_accepted_pair(
                            point_a.mode, point_a.force,
                            accepted_point.mode, accepted_point.force,
                            state_id=self._state_id(), serial=accepted_point.serial,
                            basis=self.basis,
                            source="sequential_accepted_rotation_extrapolated",
                        )
                else:
                    self.forces1E = None
            if not stoprot:
                if self.control.get_counter("rotcount") >= max_num_rot:
                    stoprot = True
                elif norm(f_rot_A) <= f_rot_max:
                    stoprot = True
            iteration += 1
        self._diagnostics()

    def _converge_direct(self) -> None:
        self.set_up_for_eigenmode_search()
        self.forces1E = None
        f_rot_min = self.control.get_parameter("f_rot_min")
        f_rot_max = self.control.get_parameter("f_rot_max")
        max_num_rot = self.control.get_parameter("max_num_rot")
        point = self._evaluate_rotation_point(
            self.eigenmode,
            sample_kind="direct_initial",
            iteration=0,
            allow_extrapolated=False,
        )
        self._add_local_point(point)
        iteration = 0
        while True:
            if norm(point.force) <= f_rot_min:
                self.log(point.force, None)
                break
            direction, pairs_used = self._rotation_direction(point)
            if pairs_used <= 0 and iteration == 0:
                # The direct method needs one finite mode displacement before it
                # can form a rotational secant.  Use one exact geodesic step of
                # the configured angle; subsequent no-pair fallbacks use H0*torque
                # rather than repeating a 45-degree move.
                tangent = normalize(direction) * self.first_angle
                step_axis = exp_map(point.mode, tangent)
                requested_norm = self.first_angle
                actual_angle = sphere_distance(point.mode, step_axis)
                clipped = False
                self._fixed_first_steps += 1
            else:
                map_kind = (
                    "exponential" if self.geometry == "riemannian" else "retraction"
                )
                step = take_sphere_step(
                    point.mode,
                    direction,
                    max_angle=self.max_angle,
                    map_kind=map_kind,
                )
                step_axis = step.axis
                requested_norm = step.requested_norm
                actual_angle = step.actual_angle
                clipped = step.clipped
            self._requested_angle_sum += float(requested_norm)
            self._actual_angle_sum += float(actual_angle)
            self._step_clips += int(clipped)
            old_point = point
            self.update_eigenmode(step_axis)
            point = self._evaluate_rotation_point(
                step_axis,
                sample_kind="direct_accepted",
                iteration=iteration + 1,
                allow_extrapolated=False,
            )
            if self.sequential_history is not None:
                self.sequential_history.advance(step_axis, basis=self.basis)
                if (
                    "direct_accepted" in self.rotation_pair_sources.split()
                    or "accepted_rotation" in self.rotation_pair_sources.split()
                ):
                    self.sequential_history.add_accepted_pair(
                        old_point.mode, old_point.force, point.mode, point.force,
                        state_id=self._state_id(), serial=point.serial, basis=self.basis,
                        source="sequential_direct_accepted",
                    )
            self._add_local_point(point)
            self.log(old_point.force, actual_angle)
            self._direct_steps += 1
            if self.control.get_counter("rotcount") >= max_num_rot:
                break
            if norm(old_point.force) <= f_rot_max:
                break
            iteration += 1
        self.eigenmode = point.mode.copy()
        self.curvature = float(point.curvature)
        self.forces1E = None
        self._diagnostics()

    def converge_to_eigenmode(self) -> None:
        if self.step_method == "fourier":
            self._converge_fourier()
        else:
            self._converge_direct()


__all__ = [
    "GeneralizedLBFGSDimerEigenmodeSearch",
    "LBFGSDimerEigenmodeSearch",
    "LBFGSRotationMixin",
    "LimitedMemoryInverseHessian",
    "StateWindowRotationHistory",
    "advanced_rotation_requested",
]
