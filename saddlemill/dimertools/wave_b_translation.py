"""shared-runtime translation adapter for verified quasi-Newton/rotation Broyden kernels."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from saddlemill.dimertools.broyden_translation import TranslationBroydenAdapter
from saddlemill.dimertools.lbfgs_dimer import DiagnosticMinModeTranslate
from saddlemill.dimertools.wave_b_rotation import _make_broyden_kernel


def _kernel_resume_identity(kernel) -> tuple[object, object]:
    state = kernel.state_dict()
    return state.get("schema"), state.get("config")


def _active_coordinate_identity(dimeratoms, positions, translation_regime: str) -> str:
    # Use the same movable Cartesian convention as the Wave-B runtime.  For
    # bowl breakout, the per-center active atom subset is an additional active
    # coordinate restriction and therefore part of the Broyden definition.
    from saddlemill.dimertools.wave_b_runtime import active_space_for_owner

    space = active_space_for_owner(dimeratoms, positions)
    identity = str(space.identity)
    if translation_regime == "bowl_breakout":
        active = getattr(dimeratoms, "_bowl_active_mask", None)
        if active is None:
            getter = getattr(dimeratoms, "bowl_active_atom_mask", None)
            if callable(getter):
                active = getter()
        if active is None:
            raise ValueError("bowl-breakout translation is missing its active-coordinate mask")
        mask = np.asarray(active, dtype=bool).reshape(-1)
        if mask.size != len(positions):
            raise ValueError("bowl-breakout active-coordinate mask size mismatch")
        digest = hashlib.sha256()
        digest.update(b"saddlemill-broyden-bowl-active-mask-v1\0")
        digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
        identity += ":bowl:" + digest.hexdigest()
    return identity


def _damping_force_identity(runtime) -> str:
    if runtime is None:
        return "damping:none:legacy_no_wave_runtime"
    selection = getattr(runtime, "damping_selection", None)
    if selection is not None:
        # DampingSelection.identity is deliberately search-global for none/fixed
        # and sequence-specific only for Shang--Liu, where re-selection changes
        # the actual effective force coefficient.  Do not use sequence_id here.
        return str(selection.identity)
    mode_cfg = dict(getattr(runtime, "options", {}).get("mode_schedule", {}) or {})
    mode = str(mode_cfg.get("damping", "none"))
    if mode == "none":
        return "damping:none:unselected"
    if mode == "fixed":
        return f"damping:fixed:unselected:{float(mode_cfg.get('fixed_lambda', 1.0)):.17g}"
    return f"damping:{mode}:unselected"


def _translation_history_identity(dimeratoms, positions) -> str:
    translation_regime = str(getattr(dimeratoms, "translation_regime", "standard")).strip().lower()
    curvature = float(dimeratoms.get_curvature())
    if not np.isfinite(curvature):
        raise ValueError("translation Broyden requires finite mode curvature")
    if curvature < 0.0:
        mode_regime = "negative_curvature"
    elif curvature > 0.0:
        mode_regime = "positive_curvature"
    else:
        # Zero is a real boundary: the optional Wave-B damping path treats it
        # as nonnegative while ASE's historical dimer projection switches only
        # at strictly positive curvature.  Do not reuse a secant through it.
        mode_regime = "zero_curvature"
    runtime = getattr(dimeratoms, "wave_b_runtime", None)
    definition = {
        "schema": "saddlemill_translation_broyden_history_definition_v1",
        "force_law": "ase_mmf_projected_force_plus_wave_b_parallel_policy",
        "damping_identity": _damping_force_identity(runtime),
        "active_coordinates": _active_coordinate_identity(dimeratoms, positions, translation_regime),
        "translation_regime": translation_regime,
        "mode_regime": mode_regime,
        "mode_finder": str(getattr(dimeratoms, "min_mode_finder", "dimer")).strip().lower(),
    }
    return json.dumps(definition, sort_keys=True, separators=(",", ":"))


class BroydenMinModeTranslate(DiagnosticMinModeTranslate):
    def __init__(self, dimeratoms, logfile="-", trajectory=None, *, family, broyden_options=None):
        super().__init__(dimeratoms, logfile=logfile, trajectory=trajectory)
        self.family = str(family)
        self.options = dict(broyden_options or {})
        expected_kernel = _make_broyden_kernel(self.family, self.options)
        saved = getattr(dimeratoms, "_wave_b_translation_broyden_state", None)
        if saved:
            self.adapter = TranslationBroydenAdapter.from_state_dict(saved)
            if _kernel_resume_identity(self.adapter.kernel) != _kernel_resume_identity(expected_kernel):
                raise ValueError("translation Broyden resume kernel/config mismatch")
        else:
            self.adapter = TranslationBroydenAdapter(expected_kernel, max_step_norm=None)
        self.last_broyden_diagnostics = {}

    def step(self, forces=None):
        if forces is None:
            forces = self.dimeratoms.get_forces()
        force = np.asarray(forces, dtype=float)
        before = np.asarray(self.dimeratoms.get_positions(), dtype=float)
        history_identity = _translation_history_identity(self.dimeratoms, before)

        # Preserve the old diagnostic field for downstream readers.  It is no
        # longer the reset identity because, with scheduling disabled, this is
        # intentionally a per-center real_solve_state:<state_id> token.
        regime = str(getattr(self.dimeratoms, "translation_regime", "standard"))
        sequence_identity = regime
        runtime = getattr(self.dimeratoms, "wave_b_runtime", None)
        schedule_state = None if runtime is None else getattr(runtime, "schedule_state", None)
        if schedule_state is not None:
            sequence_identity = str(schedule_state.core.sequence_id)
        elif runtime is not None and getattr(runtime, "damping_selection", None) is not None:
            sequence_identity = str(runtime.damping_selection.sequence_id)

        result = self.adapter.step(before, force, history_identity=history_identity)
        step = np.asarray(result.step, dtype=float)
        maximum = float(self.control.get_parameter("maximum_translation"))
        max_atom = float(np.sqrt((step * step).sum(axis=1)).max()) if len(step) else 0.0
        if max_atom > maximum and max_atom > 0.0:
            step *= maximum / max_atom
        if hasattr(self.dimeratoms, "confine_translation_displacement"):
            step = self.dimeratoms.confine_translation_displacement(step)
        self.dimeratoms.set_positions(before + step)
        self.dimeratoms._wave_b_translation_broyden_state = self.adapter.state_dict()
        self.last_broyden_diagnostics = {
            "translation_algorithm": self.family,
            "operator_origin": "effective_force_jacobian_nonphysical",
            "kernel_family": result.kernel_family,
            "history_size": result.history_size,
            "raw_step_norm": result.raw_step_norm,
            "accepted_step_norm": float(np.linalg.norm(step.reshape(-1))),
            "existing_maximum_translation": maximum,
            "reset_reason": result.reset_reason,
            "sequence_identity": sequence_identity,
        }


__all__ = ["BroydenMinModeTranslate"]
