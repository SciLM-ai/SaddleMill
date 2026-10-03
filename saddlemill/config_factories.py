"""Lazy callable factories for SaddleMill configuration dispatch."""

from __future__ import annotations


def load_calculator_factory(calculator_name):
    """Return the selected calculator constructor/callable without instantiating it."""
    if calculator_name == "FAIRChemCalculator":
        from fairchem.core import FAIRChemCalculator

        return FAIRChemCalculator.from_model_checkpoint
    if calculator_name == "VaspInteractive":
        from vasp_interactive import VaspInteractive

        return VaspInteractive
    if calculator_name == "Vasp":
        from ase.calculators.vasp import Vasp

        return Vasp
    raise ValueError(f"Unknown calculator: {calculator_name}")


def load_method_factory(method_name):
    """Return the selected method callable after validation has completed."""
    if method_name == "NEB":
        from saddlemill.nebopt import nebopt

        return nebopt
    if method_name == "Dimer":
        from saddlemill.dimeropt import dimeropt

        return dimeropt
    if method_name == "Minimization":
        from saddlemill.geomopt import geomopt

        return geomopt
    if method_name == "DoubleMinimization":
        from saddlemill.geomopt import doublegeomopt

        return doublegeomopt
    if method_name == "SinglePoint":
        from saddlemill.geomopt import singlepoint

        return singlepoint
    if method_name == "Hessian":
        from saddlemill.hessian_job import hessian_job

        return hessian_job
    raise NotImplementedError(
        f"Method '{method_name}' is not implemented. Only NEB, Dimer, Minimization, "
        "DoubleMinimization, SinglePoint, and Hessian are supported."
    )


def load_optimizer_factory(optimizer_name):
    """Return an optimizer class while retaining the historical aliases."""
    token = optimizer_name.lower()
    if token == "mdmin":
        from ase.optimize import MDMin

        return MDMin
    if token == "bfgs":
        from ase.optimize import BFGS

        return BFGS
    if token == "lbfgs":
        from ase.optimize import LBFGS

        return LBFGS
    if token == "fire":
        from ase.optimize import FIRE

        return FIRE
    if token == "fire2":
        from ase.optimize import FIRE2

        return FIRE2
    if token in {"firelbfgs", "fire_lbfgs", "warmfirelbfgs"}:
        from saddlemill.fire_lbfgs import FIRELBFGS

        return FIRELBFGS
    raise NotImplementedError(
        f"Method '{optimizer_name}' is not implemented. Only MDMin, BFGS, "
        "LBFGS, FIRE and FIRELBFGS are supported."
    )


def build_dimer_resolved_run_metadata(config_dict, *, stopping_reason="", history_options=None):
    """Build the single typed-HVP-sealed resolved-run metadata envelope for Dimer."""
    from saddlemill.dimertools.foundation_types import ReferenceStatus, ResolvedRunMetadata

    dimer = dict(config_dict.get("ourDimer", {}) or {})
    history = dict(history_options or {})
    minmode = dict(config_dict.get("ourMinMode", {}) or {})
    translator = str(dimer.get("translation_optimizer", "ase")).strip().lower()
    finder = str(dimer.get("min_mode_finder", "dimer")).strip().lower()
    algorithms = {
        "engine": str(dimer.get("engine", "ase")).strip().lower(),
        "min_mode_finder": finder,
        "mmf_variant": str(dimer.get("mmf_variant", "standard")).strip().lower(),
        "rotation_optimizer": str(dimer.get("rotation_optimizer", "ase")).strip().lower(),
        "translation_optimizer": translator,
        "mode_reuse": str(dimer.get("mode_reuse", "none")).strip().lower(),
    }
    policies = {}
    for key in (
        "rotation_transport_policy",
        "rotation_geometry",
        "translation_curvature_guard",
        "translation_regularization",
        "translation_trial_step",
    ):
        if key in dimer:
            policies[key] = str(dimer[key])
    for key in (
        "pair_sources",
        "pair_order",
        "projection_policy",
        "kappa_projection_policy",
        "pair_safeguard_resolved",
        "rotation_history_source",
    ):
        if key in history:
            policies["history_" + key] = str(history[key])

    from saddlemill.dimertools.wave_b_runtime import wave_b_options_from_config

    wave = wave_b_options_from_config(config_dict)
    ph = dict(wave.get("physical_hessian", {}) or {})
    ms = dict(wave.get("mode_schedule", {}) or {})
    md = dict(wave.get("mode_diagnostics", {}) or {})
    mp = dict(wave.get("mode_predictor", {}) or {})
    iso = dict(wave.get("isopotential", {}) or {})
    shadow = dict(wave.get("qn_shadow", {}) or {})
    policies.update({
        "physical_hessian_update": str(ph.get("update", "off")),
        "physical_hessian_probe_updates": str(ph.get("probe_updates", "off")),
        "physical_hessian_probe_batching": str(ph.get("probe_batching", "sequential")),
        "mode_schedule_enabled": bool(ms.get("enabled", False)),
        "mode_schedule_max_skips": int(ms.get("max_skips", 3)),
        "mode_schedule_skip_entry_gate": str(ms.get("skip_entry_gate", "none")),
        "mode_schedule_refresh_policy": str(ms.get("refresh_policy", "bounded")),
        "mode_diagnostics_enabled": bool(md.get("enabled", False)),
        "mode_refresh_criterion": str(md.get("mode_refresh_criterion", "legacy")),
        "mode_residual_threshold": md.get("mode_residual_threshold", None),
        "mode_angular_error_threshold_degrees": md.get("mode_angular_error_threshold_degrees", None),
        "mode_residual_epsilon": float(md.get("residual_epsilon", 1.0e-12)),
        "mode_gap_tolerance": float(md.get("gap_tolerance", 1.0e-6)),
        "mode_model_gap_solver_tolerance": float(md.get("model_gap_solver_tolerance", 1.0e-4)),
        "mode_reference_diagnostics_enabled": bool(md.get("paid_reference", False)),
        "mode_hindsight_enabled": bool(md.get("hindsight", False)),
        "parallel_force_damping": str(ms.get("damping", "none")),
        "predictor_hook": str(ms.get("predictor_hook", "none")),
        "predictor_angular_step_scale": mp.get("angular_step_scale", None),
        "predictor_max_angle_degrees": float(mp.get("max_angle_degrees", 5.0)),
        "predictor_forcebank_lbfgs_dynamic_h0": bool(mp.get("forcebank_lbfgs_dynamic_h0", True)),
        "predictor_physical_root_policy": str(mp.get("physical_root_policy", "lowest")),
        "isopotential_estimator": str(iso.get("selector", "off")),
        "isopotential_purpose": str(iso.get("purpose", "diagnostic")),
        "qn_shadow_dense_diagnostic": bool(shadow.get("enabled", False)),
        "minmode_hvp_origin": str(minmode.get("hvp_origin", "physical_fd")).strip().lower(),
        "minmode_root_policy": str(minmode.get("root_policy", "lowest")).strip().lower(),
        "minmode_residual_stop": bool(minmode.get("residual_stop", False)),
        "minmode_residual_tolerance": float(minmode.get("residual_tolerance", 1.0e-3)),
        "minmode_sella_maxiter": (
            None if minmode.get("maxiter", None) in (None, "", "none", "None")
            else int(minmode.get("maxiter"))
        ),
    })

    if str(dimer.get("engine", "ase")).strip().lower() == "sella":
        sella_cfg = dict(config_dict.get("ourSella", {}) or {})
        ablation = sella_cfg.get("ablation", None)
        policies["sella_ablation"] = "off" if ablation in (None, "", "none", "None") else str(ablation)

    if translator in {"partitioned_lbfgs", "q_lbfgs_dimer_axial"}:
        from saddlemill.dimertools.dimer_factory import partitioned_options_from_config

        for key, value in partitioned_options_from_config(
            config_dict, translation_optimizer=translator
        ).items():
            policies["partitioned_lbfgs_" + key] = value
    if translator in {"rfo", "prfo", "qn_mmf"}:
        from saddlemill.dimertools.dimer_factory import create_rfo_translation_runtime

        for key, value in create_rfo_translation_runtime(config_dict).resolved_metadata().items():
            policies["rfo_" + key] = value

    reference_informed = (
        str(minmode.get("davidson_initial_hessian_source", "identity")).strip().lower()
        == "reference_fd"
    )
    origins = []
    if bool(history.get("enabled", False)):
        raw = history.get("record_sources", ())
        origins.extend(
            str(raw).split() if isinstance(raw, str) else [str(v) for v in raw]
        )
    if str(iso.get("selector", "off")) != "off":
        origins.append("isopotential")
    estimator_origins = (
        ()
        if str(iso.get("selector", "off")) == "off"
        else ("gpumd_directional_isopotential",)
    )
    return ResolvedRunMetadata(
        algorithms=algorithms,
        policies=policies,
        defaults={
            "canonical_history_enabled": bool(history.get("enabled", False)),
            "wave_b_new_features_default_off": True,
            "wave_c_new_features_default_off": True,
            "wave_d_new_features_default_off": True,
        },
        observation_origins=tuple(dict.fromkeys(origins)),
        estimator_origins=estimator_origins,
        stopping_reason=stopping_reason,
        reference_status=(
            ReferenceStatus.REFERENCE_INFORMED
            if reference_informed
            else ReferenceStatus.BASELINE
        ),
        unsupported_combinations=(
            "wave_b_or_wave_c_dimer_selectors_with_sella",
            "t17_with_parallel_force_damping",
            "t17_without_t03_physical_hessian",
        ),
        metadata={
            "stage": "C3_stage_c",
            "shared_wiring_owner": "shared-runtime",
            "wave_d_checkpoint": "C4_unsealed",
            "physical_hessian_semantics": "model_only_not_certification",
            "translation_broyden_semantics": "effective_force_jacobian_nonphysical",
            "qn_shadow_semantics": "passive_zero_force_calls",
            "t09_hvp_semantics": "typed_physical_operator_not_effective_force_jacobian",
            "t10_predictor_semantics": "heuristic_seed_or_held_mode_never_physical_validation",
            "t11_translation_semantics": "partitioned_raw_physical_center_history",
            "t17_gradient_semantics": "same_center_raw_physical_gradient_g_equals_minus_F",
            "t17_step_control_semantics": "legacy controls preserved; generic ras uses fixed-alpha parity algebra plus bisection and optional adaptive trust",
            "t12_semantics": "diagnostic_observation_and_additional_refresh_trigger_t06_owns_state",
            "t13_semantics": "attempt_scoped_sella_instance_hooks_no_prfo_rerun",
        },
    )


__all__ = [
    "build_dimer_resolved_run_metadata",
    "load_calculator_factory",
    "load_method_factory",
    "load_optimizer_factory",
]
