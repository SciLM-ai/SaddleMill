"""Minimum-mode state ownership and canonical-history recording adapters.

The currently benchmarked numerical implementations remain in
``lbfgs_dimer.py`` and ``kappa_dimer.py``.  This module provides the new
role-oriented assembly boundary and opt-in subclasses that record raw physical
forces by ordinary method overriding.  No live instance or shared calculator is
monkey-patched.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from saddlemill.dimertools.force_evaluator import PhysicalForceEvaluator
from saddlemill.dimertools.force_history import CanonicalForceHistory
from saddlemill.dimertools.kappa_dimer import (
    KappaMinModeAtoms as LegacyKappaMinModeAtoms,
)
from saddlemill.dimertools.lbfgs_dimer import (
    ConfigurableRotationMinModeAtoms as LegacyConfigurableRotationMinModeAtoms,
)


def _force_calls(obj) -> int:
    control = getattr(obj, "control", None)
    getter = getattr(control, "get_counter", None)
    if callable(getter):
        try:
            return int(getter("forcecalls"))
        except Exception:
            pass
    return 0


def _solver_context(obj) -> tuple[str, str]:
    finder = str(getattr(obj, "min_mode_finder", "dimer")).strip().lower()
    if finder == "dimer":
        return "rotation_mode_solve", "rotation"
    if "davidson" in finder or finder == "olsen_jd":
        return f"{finder}_hvp", "davidson"
    if "lanczos" in finder:
        return f"{finder}_hvp", "lanczos"
    return (finder or "minimum_mode") + "_probe", finder or "minimum_mode"


def _positions_or_none(obj):
    try:
        return np.asarray(obj.get_positions(), dtype=float).copy()
    except Exception:
        return None


class CanonicalHistoryRecordingMixin:
    """Record raw PES observations while preserving parent numerical output."""

    def __init__(
        self,
        *args,
        history_options: Mapping[str, object] | None = None,
        force_evaluator: PhysicalForceEvaluator | None = None,
        **kwargs,
    ) -> None:
        options = dict(history_options or {})
        self.canonical_history_options = options
        if force_evaluator is None:
            history = CanonicalForceHistory(
                memory_states=int(options.get("memory_states", 20)),
                max_probes_per_state=int(options.get("max_probes_per_state", 0)),
                center_tolerance=float(options.get("center_tolerance", 1.0e-12)),
                sample_tolerance=float(options.get("sample_tolerance", 1.0e-12)),
                record_sources=options.get(
                    "record_sources",
                    "center rotation lanczos davidson reference_hessian",
                ),
            )
            force_evaluator = PhysicalForceEvaluator(history)
        self.physical_force_evaluator = force_evaluator
        self.canonical_force_history = force_evaluator.history
        self._canonical_recording_guard = False
        self._canonical_rotation_history = None
        self.canonical_rotation_history_source = str(
            options.get("rotation_history_source", "legacy_derived")
        ).strip().lower()
        if (
            bool(options.get("rotation_reuse", False))
            and self.canonical_rotation_history_source == "legacy_derived"
        ):
            from saddlemill.dimertools.rotation_history import StateWindowRotationHistory
            rotation_options = dict(kwargs.get("rotation_lbfgs_options", {}) or {})
            self._canonical_rotation_history = StateWindowRotationHistory(
                memory_states=int(options.get("memory_states", 20)),
                initial_hessian=float(rotation_options.get("initial_hessian", 1.0)),
                dynamic_h0=bool(rotation_options.get("dynamic_h0", False)),
                curvature_epsilon=float(rotation_options.get("curvature_epsilon", 1.0e-12)),
                dense_bfgs_diagnostic=bool(
                    rotation_options.get("dense_bfgs_diagnostic", False)
                ),
            )

        # Reference-Hessian construction can happen during parent initialization.
        # The context labels any nested real-position calls without altering them.
        minmode_options = dict(kwargs.get("minmode_options", {}) or {})
        initial_source = (
            "reference_hessian_initialization"
            if str(minmode_options.get("davidson_initial_hessian_source", "identity"))
            .strip()
            .lower()
            == "reference_fd"
            else "initialization"
        )
        family = "reference_hessian" if initial_source.startswith("reference") else "initialization"
        with self.physical_force_evaluator.source(
            initial_source, {"stage": "constructor"}, family=family
        ):
            super().__init__(*args, **kwargs)

    def _begin_current_state(self, *, record_site: str) -> None:
        positions = _positions_or_none(self)
        if positions is None:
            return
        from saddlemill.dimertools.runtime_state import active_coordinate_provenance
        active_mask, convention = active_coordinate_provenance(self, positions)
        self.physical_force_evaluator.begin_state(
            positions,
            metadata={"opened_by": record_site},
            active_dof_mask=active_mask,
            coordinate_convention=convention,
        )

    def _record_center_from_cache(
        self,
        *,
        exact: bool,
        force_call_delta: int,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if self._canonical_recording_guard:
            return
        positions = _positions_or_none(self)
        forces = getattr(self, "forces0", None)
        if positions is None or forces is None:
            return
        center_metadata = dict(metadata or {})
        # Adaptive RAS needs the already-computed PES energy for Sella-parity
        # trust accounting.  Read only cached state; never request another PES
        # evaluation here.  ``energy0`` is ASE MinModeAtoms' center-energy cache.
        cached_energy = getattr(self, "energy0", None)
        try:
            energy_value = float(cached_energy)
        except (TypeError, ValueError):
            energy_value = float("nan")
        if np.isfinite(energy_value):
            center_metadata["potential_energy"] = float(energy_value)
            center_metadata["potential_energy_source"] = "minmode_energy0"
        self._canonical_recording_guard = True
        try:
            observation = self.physical_force_evaluator.record_center(
                positions,
                np.asarray(forces, dtype=float),
                evaluation="physical_exact" if exact else "physical_cached",
                force_call_delta=max(0, int(force_call_delta)),
                metadata=center_metadata,
            )
            runtime = getattr(self, "wave_b_runtime", None)
            if runtime is not None:
                runtime.observe_center(self, observation)
        finally:
            self._canonical_recording_guard = False

    def calculate_real_forces_and_energies(self, *args, **kwargs):
        self._begin_current_state(record_site="calculate_real_forces_and_energies")
        before = _force_calls(self)
        result = super().calculate_real_forces_and_energies(*args, **kwargs)
        after = _force_calls(self)
        self._record_center_from_cache(
            exact=(after > before),
            force_call_delta=max(0, after - before),
            metadata={"record_site": "calculate_real_forces_and_energies"},
        )
        return result

    def get_forces(self, real=False, pos=None, **kwargs):
        if self._canonical_recording_guard:
            return super().get_forces(real=real, pos=pos, **kwargs)

        # Open the center state before the parent may recursively request probes.
        self._begin_current_state(record_site="get_forces")
        before = _force_calls(self)
        result = super().get_forces(real=real, pos=pos, **kwargs)
        after = _force_calls(self)

        if real and pos is not None:
            context = self.physical_force_evaluator.current_context
            state = self.canonical_force_history.current
            probe_positions = np.asarray(pos, dtype=float)
            direction = None
            offset = 0.0
            side = int(context.metadata.get("dimer_side", 0) or 0)
            direction_kind = str(context.metadata.get("direction_kind", "")).strip().lower()
            explicit_direction = context.metadata.get(
                "probe_direction", context.metadata.get("dimer_mode", None)
            )
            if state is not None:
                delta = probe_positions - state.center_positions
                offset = float(np.linalg.norm(delta))
                if explicit_direction is not None:
                    candidate = np.asarray(explicit_direction, dtype=float)
                    magnitude = float(np.linalg.norm(candidate))
                    if magnitude > 1.0e-15:
                        direction = candidate / magnitude
                elif offset > 1.0e-15:
                    direction = delta / offset
                if direction is not None and side == 0:
                    side = 1 if float(np.vdot(delta.ravel(), direction.ravel()).real) >= 0.0 else -1
            if not direction_kind:
                direction_kind = {
                    "rotation": "dimer_mode",
                    "lanczos": "lanczos_vector",
                    "davidson": "davidson_vector",
                    "reference_hessian": "reference_fd_direction",
                }.get(context.family, context.family)
            stencil_id = str(context.metadata.get("stencil_id", "")).strip()
            if not stencil_id and state is not None:
                stencil_id = (
                    f"state{state.state_id}:{context.source}:"
                    f"auto{self.physical_force_evaluator.probe_record_calls}"
                )
            self._canonical_recording_guard = True
            try:
                observation = self.physical_force_evaluator.record_probe(
                    probe_positions,
                    np.asarray(result, dtype=float),
                    evaluation=(
                        "physical_exact" if after > before else "physical_cached"
                    ),
                    force_call_delta=max(0, after - before),
                    source=context.source,
                    family=context.family,
                    metadata={"record_site": "get_forces_real_pos"},
                    direction=direction,
                    direction_kind=direction_kind,
                    stencil_id=stencil_id,
                    dimer_side=side,
                    offset=offset,
                )
                runtime = getattr(self, "wave_b_runtime", None)
                if runtime is not None:
                    runtime.observe_probe(self, self.canonical_force_history, observation)
            finally:
                self._canonical_recording_guard = False
        elif pos is None:
            # Usually calculate_real_forces_and_energies recorded the exact center.
            # This cached update is idempotent and never attributes nested probe
            # force calls to the center.
            self._record_center_from_cache(
                exact=False,
                force_call_delta=0,
                metadata={"record_site": "get_forces_center_cache"},
            )
        return result

    def _reference_hessian_fd(self, *args, **kwargs):
        parent = getattr(super(), "_reference_hessian_fd", None)
        if parent is None:
            raise AttributeError("parent minimum-mode class has no _reference_hessian_fd")
        with self.physical_force_evaluator.source(
            "reference_hessian_fd",
            {"stage": "finite_difference"},
            family="reference_hessian",
        ):
            return parent(*args, **kwargs)

    def finalize_canonical_state(self) -> None:
        self._begin_current_state(record_site="finalize_canonical_state")
        self._record_center_from_cache(
            exact=False,
            force_call_delta=0,
            metadata={"record_site": "finalize_canonical_state"},
        )
        mode = None
        curvature = None
        try:
            mode = np.asarray(self.get_eigenmode(), dtype=float)
        except Exception:
            pass
        try:
            curvature = float(self.get_curvature())
        except Exception:
            pass
        metadata = {
            "gamma_1": getattr(self, "last_gamma_1", ""),
            "gamma_2": getattr(self, "last_gamma_2", ""),
            "kappa": getattr(self, "kappa", ""),
            "kappa_active": getattr(self, "kappa_active", ""),
            "convex_escape": getattr(self, "convex_escape", "standard"),
            "rotation_optimizer": getattr(self, "rotation_optimizer", "ase"),
        }
        self.physical_force_evaluator.finalize_state(
            mode=mode,
            curvature=curvature,
            solver=getattr(self, "min_mode_finder", "dimer"),
            translation_regime=getattr(self, "translation_regime", "standard"),
            metadata=metadata,
        )

    def _run_wave_b_isopotential(self) -> None:
        runtime = getattr(self, "wave_b_runtime", None)
        if runtime is None:
            return
        cfg = dict(getattr(runtime, "options", {}).get("isopotential", {}) or {})
        if str(cfg.get("selector", "off")) == "off":
            return
        from saddlemill.dimertools.isopotential import (
            finish_estimate, prepare_probe, probe_observation_kwargs,
        )
        from saddlemill.dimertools.wave_b_runtime import active_space_for_owner
        state = self.canonical_force_history.current
        center = None if state is None else state.center
        if state is None or center is None:
            runtime.last_isopotential_metadata = {"available": False, "reason": "missing_center_observation"}
            return
        try:
            mode = np.asarray(self.get_eigenmode(), dtype=float)
            curvature = float(self.get_curvature())
        except Exception:
            runtime.last_isopotential_metadata = {"available": False, "reason": "missing_mode_or_curvature"}
            return
        if str(cfg.get("regime_policy", "source_faithful_guarded")) == "source_faithful_guarded":
            active_force = active_space_for_owner(self, state.center_positions).project(center.forces)
            if curvature > 0.0:
                runtime.last_isopotential_metadata = {"available": False, "reason": "guard_positive_curvature"}
                return
            if float(np.linalg.norm(active_force.reshape(-1))) < float(cfg.get("release_f", 0.0)):
                runtime.last_isopotential_metadata = {"available": False, "reason": "guard_force_below_release"}
                return
        space = active_space_for_owner(self, state.center_positions)
        plan = prepare_probe(
            state.center_positions, center.forces, mode, float(cfg["displacement"]),
            coordinate_space=space,
            expected_center_geometry_id=state.geometry_id,
            expected_coordinate_space_identity=space.identity,
            purpose=str(cfg.get("purpose", "diagnostic")),
            promote_observation_to_algorithm_consumers=bool(cfg.get("promote_observation_to_algorithm_consumers", False)),
            force_tolerance=float(cfg.get("force_tolerance", 1.0e-14)),
            direction_tolerance=float(cfg.get("direction_tolerance", 1.0e-14)),
            mode_tolerance=float(cfg.get("mode_tolerance", 1.0e-14)),
            metadata={"runtime_owner": "t14_stage_b"},
        )
        if not plan.available:
            runtime.last_isopotential_metadata = {"available": False, "reason": plan.unavailable_reason}
            return
        before = _force_calls(self)
        self._canonical_recording_guard = True
        try:
            with self.physical_force_evaluator.source(
                plan.source, {"estimator_origin": "gpumd_directional_isopotential"},
                family=plan.family, purpose=plan.purpose,
            ):
                raw_force = super().get_forces(real=True, pos=plan.probe_positions)
        finally:
            self._canonical_recording_guard = False
        after = _force_calls(self)
        delta = max(0, after - before)
        if delta not in (0, 1):
            raise RuntimeError(
                "directional isopotential route requested one probe but "
                f"observed force_call_delta={delta}"
            )
        kwargs = probe_observation_kwargs(
            plan, evaluation="physical_exact" if delta > 0 else "physical_cached",
            force_call_delta=delta, cache_hit=(delta == 0),
            force_source="saddlemill_minmode_get_forces",
        )
        positions = kwargs.pop("positions")
        observation = self.physical_force_evaluator.record_probe(positions, raw_force, **kwargs)
        runtime.observe_probe(self, self.canonical_force_history, observation)
        estimate = finish_estimate(
            plan, raw_force, probe_positions=plan.probe_positions,
            expected_probe_geometry_id=plan.probe_geometry_id,
            force_source="saddlemill_minmode_get_forces",
            metadata={"force_call_delta": delta, "cache_hit": int(delta == 0)},
        )
        runtime.last_isopotential_metadata = {
            "available": bool(estimate.available),
            "reason": str(estimate.unavailable_reason),
            "purpose": str(estimate.purpose),
            "estimator_origin": str(estimate.estimator_origin),
            "kappa_iso": estimate.kappa_iso,
            "c0": estimate.c0,
            "force_call_delta": delta,
            "cache_hit": int(delta == 0),
            "probe_observation_id": getattr(observation, "observation_id", ""),
            "admitted_to_physical_hessian": 0,
            "admitted_to_qn": 0,
        }

    def ensure_wave_b_fresh_validation(self) -> bool:
        runtime = getattr(self, "wave_b_runtime", None)
        if runtime is None or not runtime.requires_fresh_validation():
            return False
        runtime.force_real_solve = True
        source, family = _solver_context(self)
        with self.physical_force_evaluator.source(
            source, {"finder": getattr(self, "min_mode_finder", "dimer"), "fresh_convergence_validation": 1},
            family=family,
        ):
            super().find_eigenmodes(order=1)
        runtime.after_real_mode_solve(self, fresh_physical_validation=True)
        self.finalize_canonical_state()
        return True

    def find_eigenmodes(self, order=1):
        source, family = _solver_context(self)
        rotation_history = getattr(self, "_canonical_rotation_history", None)
        rotation_reuse = bool(self.canonical_history_options.get("rotation_reuse", False))
        if rotation_reuse:
            if str(getattr(self, "min_mode_finder", "dimer")).strip().lower() != "dimer":
                raise ValueError("canonical rotation reuse currently requires min_mode_finder=dimer")
            if str(getattr(self, "rotation_optimizer", "ase")).strip().lower() != "lbfgs":
                raise ValueError("canonical rotation reuse requires rotation_optimizer=lbfgs")
            self._begin_current_state(record_site="rotation_history_begin_state")
            state = self.canonical_force_history.current
            if state is None:
                raise RuntimeError("canonical rotation reuse has no current center state")
            if rotation_history is not None:
                rotation_history.begin_state(state.state_id)
        runtime = getattr(self, "wave_b_runtime", None)
        if runtime is not None and runtime.before_mode_solve(self):
            self.finalize_canonical_state()
            return None
        metadata = {
            "finder": getattr(self, "min_mode_finder", "dimer"),
            "order": int(order),
        }
        with self.physical_force_evaluator.source(
            source, metadata, family=family
        ):
            result = super().find_eigenmodes(order=order)
        if runtime is not None:
            runtime.after_real_mode_solve(self, fresh_physical_validation=True)
        self.finalize_canonical_state()
        self._run_wave_b_isopotential()
        return result


class CanonicalHistoryMinModeAtoms(
    CanonicalHistoryRecordingMixin,
    LegacyConfigurableRotationMinModeAtoms,
):
    """Standard MMF state with projection-independent force recording."""


class CanonicalHistoryKappaMinModeAtoms(
    CanonicalHistoryRecordingMixin,
    LegacyKappaMinModeAtoms,
):
    """Kappa MMF state with projection-independent force recording."""


# Role-oriented compatibility names.  Their objects are exactly the currently
# benchmarked classes when history is disabled.
ConfigurableRotationMinModeAtoms = LegacyConfigurableRotationMinModeAtoms
KappaMinModeAtoms = LegacyKappaMinModeAtoms


__all__ = [
    "CanonicalHistoryKappaMinModeAtoms",
    "CanonicalHistoryMinModeAtoms",
    "CanonicalHistoryRecordingMixin",
    "ConfigurableRotationMinModeAtoms",
    "KappaMinModeAtoms",
]
