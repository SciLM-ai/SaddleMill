"""Shared runtime wiring for verified physical-Hessian and isopotential components.

This module is orchestration/state only. Numerical kernels remain in the worker-owned
modules.  All features are opt-in and the default object is inert.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace, WorkPurpose
from saddlemill.dimertools.force_policy import (
    DampingSelection,
    ParallelForceDampingConfig,
    apply_parallel_force_policy,
    select_damping_for_sequence,
)
from saddlemill.dimertools.hvp_interfaces import (
    FiniteDifferenceForceHVPBackend,
    HVPRequest,
    MatrixMemoryBudget,
)
from saddlemill.dimertools.mode_schedule import (
    BoundedModeScheduler,
    ModeScheduleConfig,
    ModeScheduleObservation,
    ModeScheduleRuntimeState,
    PhysicalModelSignal,
    RealSolveEntryDiagnostic,
    mode_identity,
)
from saddlemill.dimertools.physical_hessian import (
    PhysicalHessianModel,
    REPRESENTATION_ONLINE_DENSE,
    REPRESENTATION_FORCEBANK_WINDOW_DENSE,
    REPRESENTATION_FORCEBANK_WINDOW_COMPACT,
)

WAVE_B_RUNTIME_SCHEMA = "saddlemill_wave_b_runtime_v1"
# Conservative peak for the worker-owned dense update/replay kernels: the
# retained model plus eig/update candidate/intermediate square workspaces can
# coexist.  This is a resource guard only and does not alter physical-Hessian mathematics.
PHYSICAL_HESSIAN_SIMULTANEOUS_DENSE_MATRICES = 4


def _physical_hessian_selector_identity(options: Mapping[str, object]) -> dict[str, str]:
    ph = dict(options.get("physical_hessian", {}) or {})
    identity = {
        "update": str(ph.get("update", "off")),
        "probe_updates": str(ph.get("probe_updates", "off")),
        "probe_batching": str(ph.get("probe_batching", "sequential")),
    }
    representation = str(ph.get("representation", REPRESENTATION_ONLINE_DENSE))
    # Preserve the exact pre-window resume identity for the legacy online-dense
    # path. Windowed reconstruction has additional scientific selectors that
    # must match on resume or the restored B would represent a different model.
    if representation != REPRESENTATION_ONLINE_DENSE:
        identity.update({
            "representation": representation,
            "forcebank_pair_sources": str(ph.get("forcebank_pair_sources", "center_center")),
            "forcebank_max_pairs": str(int(ph.get("forcebank_max_pairs", 0))),
            "initial_hessian": repr(float(ph.get("initial_hessian", 1.0))),
        })
    return identity


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off", "", "none"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _float_or_none(value: object) -> float | None:
    if value in (None, "", "none", "None"):
        return None
    return float(value)


def wave_b_options_from_config(config_dict: Mapping[str, object]) -> dict[str, object]:
    """Resolve all Stage-B shared selectors without importing ASE/runtime classes."""
    ph = dict(config_dict.get("ourPhysicalHessian", {}) or {})
    history = dict(config_dict.get("ourDimerHistory", {}) or {})
    br = dict(config_dict.get("ourDimerBroyden", {}) or {})
    cg = dict(config_dict.get("ourDimerCG", {}) or {})
    ms = dict(config_dict.get("ourModeSchedule", {}) or {})
    mp = dict(config_dict.get("ourModePredictor", {}) or {})
    iso = dict(config_dict.get("ourIsopotential", {}) or {})
    sh = dict(config_dict.get("ourQNShadow", {}) or {})
    from saddlemill.dimertools.dimer_factory import _mode_diagnostics_options_from_config
    md = _mode_diagnostics_options_from_config(config_dict)
    return {
        "physical_hessian": {
            "update": str(ph.get("update", "off")).strip().lower().replace("-", "_"),
            "probe_updates": str(ph.get("probe_updates", "off")).strip().lower().replace("-", "_"),
            "probe_batching": str(ph.get("probe_batching", "sequential")).strip().lower().replace("-", "_"),
            "representation": str(ph.get("representation", REPRESENTATION_ONLINE_DENSE)).strip().lower().replace("-", "_"),
            "forcebank_pair_sources": history.get("pair_sources", "center_center"),
            "forcebank_max_pairs": int(history.get("max_pairs", 0)),
            "initial_hessian": float(ph.get("initial_hessian", 1.0)),
            "max_matrix_bytes": int(ph.get("max_matrix_bytes", 536870912)),
            "denominator_tolerance": float(ph.get("denominator_tolerance", 1.0e-12)),
            "rank_relative_tolerance": float(ph.get("rank_relative_tolerance", 1.0e-10)),
            "rank_absolute_tolerance": float(ph.get("rank_absolute_tolerance", 1.0e-12)),
            "dependence_noise_tolerance": float(ph.get("dependence_noise_tolerance", 1.0e-6)),
            "block_symmetry_noise_tolerance": float(ph.get("block_symmetry_noise_tolerance", 1.0e-3)),
            "secant_residual_tolerance": float(ph.get("secant_residual_tolerance", 1.0e-8)),
            "spectral_zero_tolerance": float(ph.get("spectral_zero_tolerance", 1.0e-12)),
        },
        "broyden": {
            "generic_initial_inverse_scale": float(br.get("generic_initial_inverse_scale", -1.0)),
            "generic_history_cap": int(br.get("generic_history_cap", 20)),
            "generic_denominator_tolerance": float(br.get("generic_denominator_tolerance", 1.0e-12)),
            "generic_vector_tolerance": float(br.get("generic_vector_tolerance", 1.0e-14)),
            "generic_rank_tolerance": float(br.get("generic_rank_tolerance", 1.0e-12)),
            "generic_max_condition": float(br.get("generic_max_condition", 1.0e12)),
            "johnson_initial_step_scale": float(br.get("johnson_initial_step_scale", 1.0)),
            "johnson_history_cap": int(br.get("johnson_history_cap", 20)),
            "johnson_regularization_w0": float(br.get("johnson_regularization_w0", 0.01)),
            "johnson_default_weight": float(br.get("johnson_default_weight", 1.0)),
            "johnson_vector_tolerance": float(br.get("johnson_vector_tolerance", 1.0e-14)),
            "johnson_rank_tolerance": float(br.get("johnson_rank_tolerance", 1.0e-12)),
            "johnson_max_condition": float(br.get("johnson_max_condition", 1.0e12)),
        },
        "cg": {
            "formula": str(cg.get("formula", "pr_plus")).strip().lower(),
            "reset_policy": None if cg.get("reset_policy", None) in (None, "", "none", "None") else str(cg.get("reset_policy")).strip().lower(),
            "denominator_tolerance": float(cg.get("denominator_tolerance", 1.0e-24)),
            "tangent_tolerance": float(cg.get("tangent_tolerance", 1.0e-14)),
            "descent_tolerance": float(cg.get("descent_tolerance", 0.0)),
        },
        "mode_diagnostics": md,
        "mode_schedule": {
            "enabled": _bool(ms.get("enabled", False)),
            # Existing default remains K=3, but max_skips is configuration and
            # is never a hard-coded cadence in the scheduler implementation.
            "max_skips": int(ms.get("max_skips", 3)),
            "skip_entry_gate": str(ms.get("skip_entry_gate", "none")).strip().lower(),
            "refresh_policy": str(ms.get("refresh_policy", "bounded")).strip().lower(),
            "parallel_force_increase_trigger": _bool(ms.get("parallel_force_increase_trigger", True)),
            "physical_model_negative_mode_trigger": _bool(ms.get("physical_model_negative_mode_trigger", False)),
            "displacement_trigger": _bool(ms.get("displacement_trigger", False)),
            "stale_data_trigger": _bool(ms.get("stale_data_trigger", True)),
            "real_residual_trigger": _bool(ms.get("real_residual_trigger", False)),
            "model_residual_trigger": _bool(ms.get("model_residual_trigger", False)),
            "angle_trigger": _bool(ms.get("angle_trigger", False)),
            "max_cumulative_displacement": _float_or_none(ms.get("max_cumulative_displacement", None)),
            "max_curvature_age": None if ms.get("max_curvature_age", None) in (None, "", "none", "None") else int(ms.get("max_curvature_age")),
            "real_residual_threshold": _float_or_none(ms.get("real_residual_threshold", None)),
            "model_residual_threshold": _float_or_none(ms.get("model_residual_threshold", None)),
            "angle_threshold_radians": _float_or_none(ms.get("angle_threshold_radians", None)),
            "required_negative_modes": int(ms.get("required_negative_modes", 1)),
            "negative_curvature_tolerance": float(ms.get("negative_curvature_tolerance", 0.0)),
            "displacement_tolerance": float(ms.get("displacement_tolerance", 1.0e-12)),
            "parallel_metric_absolute_tolerance": float(ms.get("parallel_metric_absolute_tolerance", 1.0e-12)),
            "parallel_metric_relative_tolerance": float(ms.get("parallel_metric_relative_tolerance", 1.0e-8)),
            "predictor_hook": str(ms.get("predictor_hook", "none")).strip().lower(),
            "damping": str(ms.get("parallel_force_damping", "none")).strip().lower(),
            "fixed_lambda": float(ms.get("fixed_lambda", 1.0)),
        },
        "mode_predictor": {
            "angular_step_scale": _float_or_none(mp.get("angular_step_scale", None)),
            "max_angle_degrees": float(mp.get("max_angle_degrees", 5.0)),
            "alignment_tolerance": float(mp.get("alignment_tolerance", 1.0e-8)),
            "displacement_tolerance_A": float(mp.get("displacement_tolerance_A", 1.0e-12)),
            "tangent_tolerance": float(mp.get("tangent_tolerance", 1.0e-14)),
            "forcebank_lbfgs_dynamic_h0": _bool(mp.get("forcebank_lbfgs_dynamic_h0", True)),
            "physical_root_policy": str(mp.get("physical_root_policy", "lowest")).strip().lower(),
            "physical_low_spectrum_count": int(mp.get("physical_low_spectrum_count", 4)),
            "physical_degeneracy_tolerance": float(mp.get("physical_degeneracy_tolerance", 1.0e-8)),
            "physical_vector_tolerance": float(mp.get("physical_vector_tolerance", 1.0e-14)),
        },
        "isopotential": {
            "selector": str(iso.get("estimator", "off")).strip().lower(),
            "regime_policy": str(iso.get("regime_policy", "source_faithful_guarded")).strip().lower(),
            "displacement": _float_or_none(iso.get("displacement", None)),
            "release_f": float(iso.get("release_f", 0.0)),
            "purpose": str(iso.get("purpose", "diagnostic")).strip().lower(),
            "promote_observation_to_algorithm_consumers": _bool(iso.get("promote_observation_to_algorithm_consumers", False)),
            "force_tolerance": float(iso.get("force_tolerance", 1.0e-14)),
            "direction_tolerance": float(iso.get("direction_tolerance", 1.0e-14)),
            "mode_tolerance": float(iso.get("mode_tolerance", 1.0e-14)),
        },
        "qn_shadow": {
            "enabled": _bool(sh.get("dense_diagnostic", False)),
            "max_dimension": int(sh.get("max_dimension", 512)),
            "max_matrix_bytes": int(sh.get("max_matrix_bytes", 134217728)),
            "max_work_units": int(sh.get("max_work_units", 200000000)),
            "store_dense_matrix": _bool(sh.get("store_dense_matrix", False)),
            "near_zero_absolute": float(sh.get("near_zero_absolute", 1.0e-12)),
            "near_zero_relative": float(sh.get("near_zero_relative", 1.0e-10)),
        },
    }



def validate_wave_b_config(
    config_dict: Mapping[str, object], *, rotation_optimizer: str, translation_optimizer: str,
    min_mode_finder: str, saddle_engine: str,
) -> dict[str, object]:
    """Fail-closed validation for every public C2 Stage-B selector."""
    options = wave_b_options_from_config(config_dict)
    ph = dict(options["physical_hessian"])
    br = dict(options["broyden"])
    cg = dict(options["cg"])
    ms = dict(options["mode_schedule"])
    md = dict(options.get("mode_diagnostics", {}) or {})
    mp = dict(options.get("mode_predictor", {}) or {})
    iso = dict(options["isopotential"])
    sh = dict(options["qn_shadow"])

    if ph["update"] not in {"off", "bfgs", "ts_bfgs"}:
        raise ValueError("[ourPhysicalHessian] update must be off, bfgs, or ts_bfgs")
    if ph["representation"] not in {
        REPRESENTATION_ONLINE_DENSE,
        REPRESENTATION_FORCEBANK_WINDOW_DENSE,
        REPRESENTATION_FORCEBANK_WINDOW_COMPACT,
    }:
        raise ValueError(
            "[ourPhysicalHessian] representation must be online_dense, "
            "forcebank_window_dense, or forcebank_window_compact"
        )
    if ph["representation"] != REPRESENTATION_ONLINE_DENSE:
        if ph["update"] != "ts_bfgs":
            raise ValueError("ForceBank-windowed physical Hessian representations require update=ts_bfgs")
        if ph["probe_batching"] != "sequential":
            raise ValueError("ForceBank-windowed TS-BFGS currently requires probe_batching=sequential")
        if int(ph["forcebank_max_pairs"]) < 0:
            raise ValueError("[ourDimerHistory] max_pairs must be >= 0")
        if not str(ph["forcebank_pair_sources"]).strip():
            raise ValueError("ForceBank-windowed TS-BFGS requires nonempty [ourDimerHistory] pair_sources")
        if ph["representation"] == REPRESENTATION_FORCEBANK_WINDOW_COMPACT and translation_optimizer != "qn_mmf":
            raise ValueError("forcebank_window_compact is currently supported only with translation_optimizer=qn_mmf")
    if ph["probe_updates"] not in {"off", "same_center_physical_hvp"}:
        raise ValueError("[ourPhysicalHessian] probe_updates must be off or same_center_physical_hvp")
    if ph["probe_batching"] not in {"sequential", "same_center_block", "solver_retained_block"}:
        raise ValueError(
            "[ourPhysicalHessian] probe_batching must be sequential, "
            "same_center_block, or solver_retained_block"
        )
    if ph["probe_batching"] == "same_center_block":
        if ph["probe_updates"] != "same_center_physical_hvp":
            raise ValueError(
                "probe_batching=same_center_block requires "
                "probe_updates=same_center_physical_hvp"
            )
        if ph["update"] != "ts_bfgs":
            raise ValueError(
                "probe_batching=same_center_block is the block/multisecant "
                "TS-BFGS path and requires [ourPhysicalHessian] update=ts_bfgs"
            )
        if min_mode_finder != "dimer":
            raise ValueError(
                "probe_batching=same_center_block is validated for "
                "min_mode_finder=dimer same-center physical stencils"
            )
    if ph["probe_updates"] != "off" and ph["update"] == "off":
        raise ValueError("physical-Hessian probe updates require a non-off physical-Hessian update model")
    if ph["probe_batching"] == "solver_retained_block":
        if ph["probe_updates"] != "same_center_physical_hvp":
            raise ValueError(
                "probe_batching=solver_retained_block requires "
                "probe_updates=same_center_physical_hvp"
            )
        if ph["update"] != "ts_bfgs":
            raise ValueError(
                "probe_batching=solver_retained_block is the MS-TS-BFGS path "
                "and requires [ourPhysicalHessian] update=ts_bfgs"
            )
        if min_mode_finder not in {"olsen_jd", "lanczos", "davidson"}:
            raise ValueError(
                "probe_batching=solver_retained_block is validated for "
                "min_mode_finder=olsen_jd, lanczos, or davidson"
            )
    if int(ph["max_matrix_bytes"]) < 0:
        raise ValueError("[ourPhysicalHessian] max_matrix_bytes must be >= 0")

    broyden_selected = rotation_optimizer in {"generic_good_broyden", "johnson_modified_broyden"} or translation_optimizer in {"generic_good_broyden", "johnson_modified_broyden"}
    if broyden_selected:
        from saddlemill.dimertools.broyden import GenericGoodBroydenInverse, JohnsonModifiedBroyden
        GenericGoodBroydenInverse(
            initial_inverse_scale=br["generic_initial_inverse_scale"], history_cap=br["generic_history_cap"],
            denominator_tolerance=br["generic_denominator_tolerance"], vector_tolerance=br["generic_vector_tolerance"],
            rank_tolerance=br["generic_rank_tolerance"], max_condition=br["generic_max_condition"],
        )
        JohnsonModifiedBroyden(
            initial_step_scale=br["johnson_initial_step_scale"], history_cap=br["johnson_history_cap"],
            regularization_w0=br["johnson_regularization_w0"], default_weight=br["johnson_default_weight"],
            vector_tolerance=br["johnson_vector_tolerance"], rank_tolerance=br["johnson_rank_tolerance"],
            max_condition=br["johnson_max_condition"],
        )
    if rotation_optimizer == "cg":
        if cg["formula"] != "pr_plus":
            raise ValueError("[ourDimerCG] only formula=pr_plus is implemented")
        if cg["reset_policy"] is None:
            raise ValueError("rotation_optimizer=cg requires explicit [ourDimerCG] reset_policy")
        from saddlemill.dimertools.cg_rotation import PRPlusRotationCG
        PRPlusRotationCG(
            reset_policy=cg["reset_policy"], denominator_tolerance=cg["denominator_tolerance"],
            tangent_tolerance=cg["tangent_tolerance"], descent_tolerance=cg["descent_tolerance"],
        )

    if saddle_engine == "sella" and (
        rotation_optimizer != "ase" or translation_optimizer != "ase" or wave_b_recording_required(options)
        or bool(ms["enabled"]) or str(ms["damping"]) != "none" or iso["selector"] != "off"
        or bool(md.get("enabled", False)) or bool(md.get("paid_reference", False)) or bool(md.get("hindsight", False))
    ):
        raise ValueError("Wave-B Dimer/MMF selectors are incompatible with engine=sella")
    if translation_optimizer in {"partitioned_lbfgs", "q_lbfgs_dimer_axial"} and saddle_engine == "sella":
        raise ValueError(
            f"{translation_optimizer} is an MMF canonical-history translator and is incompatible with engine=sella"
        )
    if translation_optimizer in {"rfo", "prfo", "qn_mmf"}:
        if saddle_engine == "sella":
            raise ValueError("native RFO/P-RFO/QN-MMF translation is an MMF path and is incompatible with engine=sella")
        if ph["update"] == "off":
            raise ValueError("native RFO/P-RFO/QN-MMF translation requires a non-off physical-Hessian physical Hessian B model")
        if str(ms["damping"]) != "none":
            raise ValueError("native RFO/P-RFO/QN-MMF translation is incompatible with parallel-force damping; use none")
    if min_mode_finder != "dimer" and rotation_optimizer != "ase":
        raise ValueError("Wave-B rotation optimizers apply only to min_mode_finder=dimer")

    if str(ms.get("skip_entry_gate", "none")) not in {"none", "dimer_initial_torque"}:
        raise ValueError("[ourModeSchedule] skip_entry_gate must be none or dimer_initial_torque")
    if str(ms.get("refresh_policy", "bounded")) not in {"bounded", "physical_model_loss"}:
        raise ValueError("[ourModeSchedule] refresh_policy must be bounded or physical_model_loss")
    if bool(ms["enabled"]) and str(ms.get("skip_entry_gate", "none")) == "dimer_initial_torque" and min_mode_finder != "dimer":
        raise ValueError("skip_entry_gate=dimer_initial_torque requires min_mode_finder=dimer")
    if bool(ms["enabled"]) and str(ms.get("refresh_policy", "bounded")) == "physical_model_loss" and ph["update"] == "off":
        raise ValueError("refresh_policy=physical_model_loss requires a non-off physical Hessian B model")

    valid_predictors = {
        "none", "translation_secant_sd", "translation_secant_rotation_lbfgs",
        "translation_secant_forcebank_lbfgs", "physical_eigen", "physical_olsen",
    }
    predictor = str(ms["predictor_hook"])
    if predictor not in valid_predictors:
        raise ValueError(f"[ourModeSchedule] predictor_hook must be one of {sorted(valid_predictors)}")
    if predictor in {"translation_secant_sd", "translation_secant_rotation_lbfgs"}:
        if mp.get("angular_step_scale") is None:
            raise ValueError("translation-secant predictor requires explicit [ourModePredictor] angular_step_scale")
        if float(mp["angular_step_scale"]) < 0.0:
            raise ValueError("[ourModePredictor] angular_step_scale must be >= 0")
    if predictor == "translation_secant_forcebank_lbfgs":
        if min_mode_finder != "dimer":
            raise ValueError("translation_secant_forcebank_lbfgs requires min_mode_finder=dimer")
        history_cfg = dict(config_dict.get("ourDimerHistory", {}) or {})
        raw_record_sources = history_cfg.get(
            "record_sources",
            "center rotation lanczos davidson reference_hessian translation_trial",
        )
        if isinstance(raw_record_sources, str):
            record_sources = {token for token in raw_record_sources.split() if token}
        else:
            record_sources = {str(token).strip() for token in raw_record_sources if str(token).strip()}
        if not ({"rotation", "all"} & record_sources):
            raise ValueError(
                "translation_secant_forcebank_lbfgs requires [ourDimerHistory] "
                "record_sources to include rotation or all"
            )
    if predictor == "translation_secant_rotation_lbfgs":
        history_cfg = dict(config_dict.get("ourDimerHistory", {}) or {})
        if rotation_optimizer != "lbfgs" or not _bool(history_cfg.get("rotation_reuse", False)):
            raise ValueError("translation_secant_rotation_lbfgs requires rotation L-BFGS with rotation history reuse")
    if predictor in {"physical_eigen", "physical_olsen"} and ph["update"] == "off":
        raise ValueError("physical predictor requires a non-off physical Hessian B model")
    if str(mp.get("physical_root_policy", "lowest")) not in {"lowest", "overlap"}:
        raise ValueError("[ourModePredictor] physical_root_policy must be lowest or overlap")
    if int(mp.get("physical_low_spectrum_count", 4)) < 1:
        raise ValueError("[ourModePredictor] physical_low_spectrum_count must be >= 1")
    if float(mp.get("max_angle_degrees", 5.0)) <= 0.0 or float(mp.get("max_angle_degrees", 5.0)) > 90.0:
        raise ValueError("[ourModePredictor] max_angle_degrees must be in (0, 90]")
    if ms["damping"] not in {"none", "fixed", "shang_liu"}:
        raise ValueError("[ourModeSchedule] parallel_force_damping must be none, fixed, or shang_liu")
    if bool(md.get("enabled", False)) and str(md.get("mode_refresh_criterion", "legacy")) != "legacy" and not bool(ms["enabled"]):
        raise ValueError(
            "non-legacy [ourModeDiagnostics] refresh criteria require [ourModeSchedule] enabled=True so mode-schedule retains scheduler ownership"
        )
    if bool(ms["enabled"]):
        # Constructing the scheduler validates threshold/trigger consistency.
        _scheduler_config(ms)
    _damping_config(ms)

    if iso["selector"] not in {"off", "directional"}:
        raise ValueError("[ourIsopotential] estimator must be off or directional")
    if iso["selector"] != "off":
        if iso["displacement"] is None or float(iso["displacement"]) <= 0.0:
            raise ValueError("directional isopotential estimator requires displacement > 0")
        if iso["purpose"] not in {"diagnostic", "algorithm"}:
            raise ValueError("[ourIsopotential] purpose must be diagnostic or algorithm")
        if iso["regime_policy"] not in {"always", "source_faithful_guarded"}:
            raise ValueError("[ourIsopotential] regime_policy must be always or source_faithful_guarded")
        if bool(iso["promote_observation_to_algorithm_consumers"]):
            raise ValueError("C2 does not implicitly promote directional isopotential observations; use an explicit physical-HVP admission policy in a later approved integration")

    if min(int(sh["max_dimension"]), int(sh["max_matrix_bytes"]), int(sh["max_work_units"])) < 0:
        raise ValueError("[ourQNShadow] resource limits must be >= 0")
    if float(sh["near_zero_absolute"]) < 0.0 or float(sh["near_zero_relative"]) < 0.0:
        raise ValueError("[ourQNShadow] near-zero thresholds must be >= 0")
    return options


def wave_b_recording_required(options: Mapping[str, object]) -> bool:
    ph = dict(options.get("physical_hessian", {}) or {})
    ms = dict(options.get("mode_schedule", {}) or {})
    iso = dict(options.get("isopotential", {}) or {})
    return (
        str(ph.get("update", "off")) != "off"
        or str(ph.get("probe_updates", "off")) != "off"
        or bool(ms.get("enabled", False))
        or str(iso.get("selector", "off")) != "off"
    )


def active_space_for_owner(owner, positions) -> ActiveCoordinateSpace:
    from saddlemill.dimertools.runtime_state import active_coordinate_provenance
    mask, convention = active_coordinate_provenance(owner, positions)
    if mask is None:
        raise ValueError("Wave-B runtime cannot determine active-coordinate mask")
    return ActiveCoordinateSpace(mask, convention=convention, null_mode_policy="none")


def _damping_config(mode_cfg: Mapping[str, object]) -> ParallelForceDampingConfig:
    return ParallelForceDampingConfig(
        mode=str(mode_cfg.get("damping", "none")),
        fixed_lambda=float(mode_cfg.get("fixed_lambda", 1.0)),
    )


def _scheduler_config(mode_cfg: Mapping[str, object]) -> ModeScheduleConfig:
    from saddlemill.dimertools.foundation_types import ParallelForceMetricPolicy
    return ModeScheduleConfig(
        max_skips=int(mode_cfg.get("max_skips", 3)),
        skip_entry_gate=str(mode_cfg.get("skip_entry_gate", "none")),
        refresh_policy=str(mode_cfg.get("refresh_policy", "bounded")),
        parallel_force_increase_trigger=bool(mode_cfg.get("parallel_force_increase_trigger", True)),
        physical_model_negative_mode_trigger=bool(mode_cfg.get("physical_model_negative_mode_trigger", False)),
        displacement_trigger=bool(mode_cfg.get("displacement_trigger", False)),
        stale_data_trigger=bool(mode_cfg.get("stale_data_trigger", True)),
        real_residual_trigger=bool(mode_cfg.get("real_residual_trigger", False)),
        model_residual_trigger=bool(mode_cfg.get("model_residual_trigger", False)),
        angle_trigger=bool(mode_cfg.get("angle_trigger", False)),
        max_cumulative_displacement=mode_cfg.get("max_cumulative_displacement"),
        max_curvature_age=mode_cfg.get("max_curvature_age"),
        real_residual_threshold=mode_cfg.get("real_residual_threshold"),
        model_residual_threshold=mode_cfg.get("model_residual_threshold"),
        angle_threshold_radians=mode_cfg.get("angle_threshold_radians"),
        required_negative_modes=int(mode_cfg.get("required_negative_modes", 1)),
        negative_curvature_tolerance=float(mode_cfg.get("negative_curvature_tolerance", 0.0)),
        displacement_tolerance=float(mode_cfg.get("displacement_tolerance", 1.0e-12)),
        parallel_metric_policy=ParallelForceMetricPolicy(
            absolute_tolerance=float(mode_cfg.get("parallel_metric_absolute_tolerance", 1.0e-12)),
            relative_tolerance=float(mode_cfg.get("parallel_metric_relative_tolerance", 1.0e-8)),
        ),
    )



def _t10_olsen_kernel_view(*, mode, action, operator, coordinate_space, data_source):
    """Adapt projected-Olsen/JD's public correction result to mode-predictor's deliberately narrow view."""
    from saddlemill.dimertools.minmode_solvers import projected_olsen_jd_correction
    from saddlemill.dimertools.mode_predictor import OlsenCorrectionResultView

    # mode-predictor supplies physical-Hessian B as ``operator``. Materialize that same operator in the
    # active reduced space only; no PES/HVP request is added here.
    shape = coordinate_space.active_dof_mask.shape
    active = np.flatnonzero(coordinate_space.active_dof_mask.reshape(-1))
    columns = []
    for index in active:
        e = np.zeros(int(np.prod(shape)), dtype=float)
        e[index] = 1.0
        columns.append(np.asarray(operator(e.reshape(shape)), dtype=float).reshape(-1)[active])
    matrix = np.column_stack(columns) if columns else np.empty((0, 0), dtype=float)
    raw = projected_olsen_jd_correction(
        mode,
        action,
        matrix,
        coordinate_space=coordinate_space,
        operator_identity="t03_physical_hessian_B",
        preconditioner_identity="t03_physical_hessian_B",
    )
    correction = raw.correction
    if correction is None:
        correction = np.zeros(shape, dtype=float)
    return OlsenCorrectionResultView(
        correction=correction,
        status=("usable" if raw.independent else "breakdown"),
        fallback=(raw.fallback or "none"),
        condition_state=raw.singularity_status,
        orthogonality_error=raw.orthogonality_error,
        shifted_system_residual=raw.shifted_system_residual_norm,
        metadata={
            "solve_method": raw.solve_method,
            "breakdown_reason": raw.breakdown_reason,
            "operator_identity": raw.operator_identity,
            "preconditioner_identity": raw.preconditioner_identity,
            "data_source": dict(data_source),
        },
    )


def _t10_prediction_result(controller, owner, evaluation):
    """Build one mode-predictor prediction strictly after mode-schedule has made its decision."""
    from math import radians
    from saddlemill.dimertools.foundation_types import ModePredictionResult, TranslationSecantPredictorInput
    from saddlemill.dimertools.mode_predictor import (
        PHYSICAL_EIGEN,
        PHYSICAL_OLSEN,
        TRANSLATION_SECANT_FORCEBANK_LBFGS,
        TRANSLATION_SECANT_ROTATION_LBFGS,
        TRANSLATION_SECANT_SD,
        PhysicalEigenModePredictor,
        PhysicalModePredictorInput,
        PhysicalOlsenModePredictor,
        TranslationSecantModePredictor,
    )

    mode_cfg = dict(controller.options.get("mode_schedule", {}) or {})
    pred_cfg = dict(controller.options.get("mode_predictor", {}) or {})
    selector = str(mode_cfg.get("predictor_hook", "none")).strip().lower()
    if selector == "none" or not evaluation.decision.prediction_allowed:
        return None
    obs = evaluation.observation
    history = getattr(owner, "canonical_force_history", None)
    state = None if history is None else history.current
    if state is None or state.center is None:
        return None
    previous = None
    for candidate in reversed(tuple(getattr(history, "states", ()))):
        if int(candidate.state_id) == int(evaluation.previous_state_id if evaluation.previous_state_id is not None else -1):
            previous = candidate
            break
    if evaluation.previous_state_id is None or previous is None or previous.center is None:
        return None

    max_angle = radians(float(pred_cfg.get("max_angle_degrees", 5.0)))
    if selector in {
        TRANSLATION_SECANT_SD,
        TRANSLATION_SECANT_ROTATION_LBFGS,
        TRANSLATION_SECANT_FORCEBANK_LBFGS,
    }:
        scale = pred_cfg.get("angular_step_scale")
        if selector != TRANSLATION_SECANT_FORCEBANK_LBFGS and scale is None:
            return None
        rotation_history = None
        forcebank_model = None
        forcebank_pairs = ()
        forcebank_metrics = {}
        if selector == TRANSLATION_SECANT_ROTATION_LBFGS:
            rotation_history = getattr(owner, "_canonical_rotation_history", None)
            if rotation_history is None:
                rotation_history = getattr(owner, "_sm_state_window_rotation_history", None)
            if rotation_history is None:
                return None
        elif selector == TRANSLATION_SECANT_FORCEBANK_LBFGS:
            from saddlemill.dimertools.force_stencil_history import build_force_bank_rotation_pairs
            from saddlemill.dimertools.riemannian_lbfgs import (
                RotationLBFGSModel, normalize_rotation_transport_policy,
            )
            history_cfg = dict(getattr(owner, "canonical_history_options", {}) or {})
            lbfgs_cfg = dict(getattr(owner, "rotation_lbfgs_options", {}) or {})
            transport_policy = normalize_rotation_transport_policy(
                lbfgs_cfg.get("transport_policy", "auto"),
                geometry=lbfgs_cfg.get("geometry", "projected"),
            )
            build = build_force_bank_rotation_pairs(
                history,
                geometry=transport_policy,
                basis=obs.coordinate_space.null_basis,
                # Do not pre-truncate raw candidates: RotationLBFGSModel applies
                # max_pairs after transport/guard admission so rejected newer pairs
                # can be backfilled by older admissible force-bank pairs.
                max_pairs=0,
                pair_sources=history_cfg.get("rotation_pair_sources", "consecutive_physical"),
                accepted_force_source=history_cfg.get("rotation_accepted_force_source", "physical_only"),
                trial_pairs_future_only=bool(history_cfg.get("rotation_trial_pairs_future_only", False)),
                current_state_id=state.state_id,
            )
            guard = str(lbfgs_cfg.get("curvature_guard", "legacy_skip")).strip().lower()
            guard = {
                "auto": "legacy_skip", "legacy": "legacy_skip",
                "legacy_rotation_guard": "legacy_skip", "shifted_secant": "damp",
            }.get(guard, guard)
            forcebank_model = RotationLBFGSModel(
                initial_hessian=float(lbfgs_cfg.get("initial_hessian", 1.0)),
                dynamic_h0=bool(pred_cfg.get("forcebank_lbfgs_dynamic_h0", True)),
                curvature_guard=guard,
                curvature_floor=float(lbfgs_cfg.get("curvature_floor", 1.0e-3)),
                curvature_epsilon=float(lbfgs_cfg.get("curvature_epsilon", 1.0e-12)),
                powell_eta=float(lbfgs_cfg.get("powell_eta", 0.2)),
                cosine_threshold=history_cfg.get("rotation_cosine_threshold", None),
                max_pairs=int(history_cfg.get("rotation_max_pairs", 0)),
                reconstruction_model="lbfgs",
                dense_bfgs_diagnostic=False,
            )
            forcebank_pairs = build.pairs
            forcebank_metrics = build.metrics
        predictor = TranslationSecantModePredictor(
            selector, rotation_history=rotation_history,
            forcebank_rotation_model=forcebank_model,
            forcebank_rotation_pairs=forcebank_pairs,
            forcebank_build_metrics=forcebank_metrics,
        )
        request = TranslationSecantPredictorInput(
            old_state_id=previous.state_id,
            new_state_id=state.state_id,
            old_state_uid=previous.state_uid,
            new_state_uid=state.state_uid,
            old_geometry_id=previous.geometry_id,
            new_geometry_id=state.geometry_id,
            old_positions=previous.center_positions,
            new_positions=state.center_positions,
            old_gradient=-np.asarray(previous.center.forces, dtype=float),
            new_gradient=-np.asarray(state.center.forces, dtype=float),
            current_mode=obs.pre_prediction_mode,
            coordinate_space=obs.coordinate_space,
            angular_step_scale=1.0 if selector == TRANSLATION_SECANT_FORCEBANK_LBFGS else float(scale),
            max_angle_radians=max_angle,
            alignment_tolerance=float(pred_cfg.get("alignment_tolerance", 1.0e-8)),
            displacement_tolerance=float(pred_cfg.get("displacement_tolerance_A", 1.0e-12)),
            tangent_tolerance=float(pred_cfg.get("tangent_tolerance", 1.0e-14)),
        )
        return predictor.predict(request)

    model = controller.physical_hessian
    if model is None:
        return None
    root_policy = str(pred_cfg.get("physical_root_policy", "lowest"))
    physical_request = PhysicalModePredictorInput(
        state_id=state.state_id,
        state_uid=state.state_uid,
        geometry_id=state.geometry_id,
        current_mode=obs.pre_prediction_mode,
        coordinate_space=obs.coordinate_space,
        model=model,
        root_policy=root_policy,
        low_spectrum_count=int(pred_cfg.get("physical_low_spectrum_count", 4)),
        max_angle_radians=max_angle,
        degeneracy_tolerance=float(pred_cfg.get("physical_degeneracy_tolerance", 1.0e-8)),
        vector_tolerance=float(pred_cfg.get("physical_vector_tolerance", 1.0e-14)),
    )
    if selector == PHYSICAL_EIGEN:
        raw = PhysicalEigenModePredictor().predict(physical_request)
    elif selector == PHYSICAL_OLSEN:
        raw = PhysicalOlsenModePredictor(_t10_olsen_kernel_view).predict(physical_request)
    else:
        return None

    # mode-schedule consumes the sealed typed-HVP translation-predictor envelope.  Preserve
    # physical predictor provenance in metadata while using the exact schedule
    # old/new identities for admission.
    return ModePredictionResult(
        mode=raw.mode,
        status=raw.status,
        original_signed_alignment=None,
        absolute_alignment=None,
        reversed_pair=False,
        raw_tangent=raw.raw_tangent,
        requested_angle_radians=raw.requested_angle_radians,
        accepted_angle_radians=raw.accepted_angle_radians,
        capped=raw.capped,
        old_state_id=previous.state_id,
        new_state_id=state.state_id,
        old_state_uid=previous.state_uid,
        new_state_uid=state.state_uid,
        old_geometry_id=previous.geometry_id,
        new_geometry_id=state.geometry_id,
        angular_step_scale=1.0,
        max_angle_radians=max_angle,
        alignment_tolerance=float(pred_cfg.get("alignment_tolerance", 1.0e-8)),
        displacement_tolerance=float(pred_cfg.get("displacement_tolerance_A", 1.0e-12)),
        tangent_tolerance=float(pred_cfg.get("tangent_tolerance", 1.0e-14)),
        origin=raw.origin,
        experimental=True,
        metadata={"physical_prediction": raw.to_state_dict()},
    )


def _t10_prediction_diagnostic_fields(prediction):
    if prediction is None:
        return {}
    meta = dict(prediction.metadata or {})
    requested = float(prediction.requested_angle_radians)
    accepted = float(prediction.accepted_angle_radians)
    fields = {
        "mode_predictor_selector": str(meta.get("selector", "")),
        "mode_predictor_requested_angle_radians": requested,
        "mode_predictor_requested_angle_degrees": float(np.degrees(requested)),
        "mode_predictor_accepted_angle_radians": accepted,
        "mode_predictor_accepted_angle_degrees": float(np.degrees(accepted)),
        "mode_predictor_capped": int(bool(prediction.capped)),
        "mode_predictor_raw_tangent_norm": (
            "" if prediction.raw_tangent is None
            else float(np.linalg.norm(np.asarray(prediction.raw_tangent, dtype=float)))
        ),
        "mode_predictor_original_absolute_alignment": (
            "" if prediction.absolute_alignment is None else float(prediction.absolute_alignment)
        ),
        "mode_predictor_displacement_norm": meta.get("displacement_norm", ""),
        "mode_predictor_gradient_change_norm": meta.get("gradient_change_norm", ""),
        "mode_predictor_secant_action_norm": meta.get("secant_action_norm", ""),
        "mode_predictor_effective_angular_step_scale": meta.get("effective_angular_step_scale", prediction.angular_step_scale),
        "mode_predictor_preconditioned": int(bool(meta.get("preconditioned", False))),
        "mode_predictor_fallback": str(meta.get("fallback", "")),
        "mode_predictor_prediction_cost_pes_calls": meta.get("prediction_cost_pes_calls", 0),
        "mode_predictor_forcebank_torque_samples": meta.get("forcebank_torque_samples", ""),
        "mode_predictor_forcebank_pair_sources": meta.get("forcebank_pair_sources", ""),
        "mode_predictor_forcebank_accepted_force_source": meta.get("forcebank_accepted_force_source", ""),
        "mode_predictor_forcebank_trial_pairs_future_only": meta.get("forcebank_trial_pairs_future_only", ""),
        "mode_predictor_forcebank_dynamic_h0": meta.get("forcebank_dynamic_h0", ""),
        "mode_predictor_forcebank_pair_candidates": meta.get("forcebank_pair_candidates", ""),
        "mode_predictor_forcebank_pairs_built": meta.get("forcebank_pairs_built", ""),
        "mode_predictor_forcebank_pairs_degenerate": meta.get("forcebank_pairs_degenerate", ""),
        "mode_predictor_forcebank_states_contributing": meta.get("forcebank_states_contributing", ""),
        "mode_predictor_forcebank_build_ns": meta.get("forcebank_build_ns", ""),
        "mode_predictor_lbfgs_pairs_used": meta.get("forcebank_lbfgs_pairs_used", ""),
        "mode_predictor_lbfgs_pairs_admissible_before_max_pairs": meta.get("forcebank_lbfgs_pairs_admissible_before_max_pairs", ""),
        "mode_predictor_lbfgs_pairs_rejected_transport": meta.get("forcebank_lbfgs_pairs_rejected_transport", ""),
        "mode_predictor_lbfgs_pairs_rejected_curvature": meta.get("forcebank_lbfgs_pairs_rejected_curvature", ""),
        "mode_predictor_lbfgs_pairs_rejected_cosine": meta.get("forcebank_lbfgs_pairs_rejected_cosine", ""),
        "mode_predictor_lbfgs_pairs_damped": meta.get("forcebank_lbfgs_pairs_damped", ""),
        "mode_predictor_lbfgs_h0_inverse_scale": meta.get("forcebank_lbfgs_h0_inverse_scale", ""),
        "mode_predictor_lbfgs_raw_direction_norm": meta.get("forcebank_lbfgs_lbfgs_raw_direction_norm", ""),
        "mode_predictor_lbfgs_direction_norm": meta.get("forcebank_lbfgs_direction_norm", ""),
        "mode_predictor_lbfgs_direction_surrogate_cosine": meta.get("forcebank_lbfgs_direction_force_cosine", ""),
        "mode_predictor_lbfgs_apply_ns": meta.get("forcebank_lbfgs_apply_ns", ""),
    }
    return fields


def apply_t10_prediction_after_schedule_decision(controller, owner, evaluation):
    """Public Stage-C adapter: mode-schedule decides first; mode-predictor predicts second.

    Returns ``(prediction_result, prediction_use)``.  It never increments mode-schedule's
    skip counter, never changes damping selection, and never performs PES work.
    """
    prediction = _t10_prediction_result(controller, owner, evaluation)
    use = controller.scheduler.resolve_prediction(evaluation, prediction)
    if use.solver_initial_mode is not None:
        owner.eigenmodes[0] = np.asarray(use.solver_initial_mode, dtype=float).copy()
    return prediction, use


def _t12_paid_reference_context(owner, runtime, state):
    """Build an isolated diagnostic-only dense FD reference at the current center.

    The algorithm owner is never asked for forces.  Calls go straight through the
    attached calculator on copied atoms, while calculator result/atoms caches are
    restored before mode-diagnostics's noninterference probe runs.  Canonical force history,
    optimizer counters, mode/schedule state, and QN histories are therefore not
    admission targets.
    """
    import copy
    import random

    from saddlemill.dimertools.foundation_types import json_safe
    from saddlemill.dimertools.mode_diagnostics import ReferenceComputation

    source_atoms = getattr(owner, "atoms", None)
    calc = None if source_atoms is None else getattr(source_atoms, "calc", None)
    if source_atoms is None or calc is None or not callable(getattr(calc, "get_forces", None)):
        raise RuntimeError(
            "[ourModeDiagnostics] paid_reference=True requires an inspectable attached "
            "calculator with get_forces; no isolated diagnostic reference context is available"
        )
    calc_name = type(calc).__name__.lower()
    calc_module = type(calc).__module__.lower()
    if "vasp" in calc_name or "vasp" in calc_module:
        raise RuntimeError(
            "[ourModeDiagnostics] paid_reference dense-FD diagnostics are not enabled for "
            "file-backed VASP/VaspInteractive calculators"
        )

    history = getattr(owner, "canonical_force_history", None)
    if history is None or state is None:
        raise RuntimeError("paid reference requires canonical current-state identity")
    positions = np.asarray(state.center_positions, dtype=float).copy()
    space = active_space_for_owner(owner, positions)
    active = np.flatnonzero(np.asarray(space.active_dof_mask, dtype=bool).reshape(-1))
    if active.size == 0:
        raise RuntimeError("paid reference has no active Cartesian degrees of freedom")
    fd = float(dict(getattr(owner, "minmode_options", {}) or {}).get("finite_difference", 1.0e-4))
    if not np.isfinite(fd) or fd <= 0.0:
        raise RuntimeError("paid reference finite_difference must be finite and > 0")

    def _calc_visible_state():
        atoms_state = getattr(calc, "atoms", None)
        atoms_payload = None
        if atoms_state is not None:
            try:
                atoms_payload = {
                    "positions": np.asarray(atoms_state.get_positions(), dtype=float).tolist(),
                    "numbers": np.asarray(atoms_state.get_atomic_numbers(), dtype=int).tolist(),
                }
            except Exception:
                atoms_payload = repr(atoms_state)
        scalar_state = {}
        for key, value in getattr(calc, "__dict__", {}).items():
            if key in {"results", "atoms"}:
                continue
            if value is None or isinstance(value, (str, int, float, bool, np.generic)):
                scalar_state[str(key)] = json_safe(value)
        return {
            "results": json_safe(getattr(calc, "results", {})),
            "atoms": atoms_payload,
            "scalar_state": scalar_state,
        }

    def _history_identity():
        rows = []
        for obs in history.iter_observations():
            rows.append((
                int(obs.serial), int(obs.state_id), str(obs.observation_id),
                str(obs.source), str(obs.family), str(obs.purpose),
            ))
        return {
            "rows": rows,
            "stencil_ids": [str(item.stencil_id) for item in history.stencils],
            "admission_ledger": history.admission_ledger.to_state_dict(),
        }

    def _owner_cache_state():
        return {
            "positions": np.asarray(owner.get_positions(), dtype=float).tolist(),
            "forces0": None if getattr(owner, "forces0", None) is None else np.asarray(owner.forces0, dtype=float).tolist(),
            "mode": np.asarray(owner.get_eigenmode(), dtype=float).tolist(),
            "curvature": float(owner.get_curvature()),
            "forcecalls": int(owner.control.get_counter("forcecalls")) if hasattr(owner, "control") else None,
            "rotcount": int(owner.control.get_counter("rotcount")) if hasattr(owner, "control") else None,
            "convergence_flags": {
                "owner_converged": getattr(owner, "converged", None),
                "owner_stoprun": getattr(owner, "stoprun", None),
                "force_real_solve": bool(getattr(getattr(owner, "wave_b_runtime", None), "force_real_solve", False)),
            },
        }

    def _schedule_state():
        sched = getattr(owner, "wave_b_runtime", None)
        state_obj = None if sched is None else getattr(sched, "schedule_state", None)
        return None if state_obj is None else state_obj.to_state_dict()

    def _python_rng():
        return random.getstate()

    def _numpy_rng():
        return np.random.get_state()

    snapshots = {
        "canonical_observation_identity": _history_identity,
        "algorithm_owner_cache": _owner_cache_state,
        "mode_schedule_state": _schedule_state,
        "python_rng_state": _python_rng,
        "numpy_rng_state": _numpy_rng,
        "calculator_visible_state": _calc_visible_state,
    }

    def measure():
        # Calculator mutation is confined to its ordinary results/atoms cache and
        # restored exactly before returning.  If either cache cannot be copied,
        # fail before performing diagnostic PES work.
        try:
            saved_results = copy.deepcopy(getattr(calc, "results", {}))
        except Exception as exc:
            raise RuntimeError("paid reference cannot snapshot calculator results safely") from exc
        had_atoms_attr = hasattr(calc, "atoms")
        saved_calc_atoms = getattr(calc, "atoms", None)
        saved_scalar_state = {
            key: value for key, value in getattr(calc, "__dict__", {}).items()
            if key not in {"results", "atoms"}
            and (value is None or isinstance(value, (str, int, float, bool, np.generic)))
        }
        calls = 0
        try:
            # Force an independent diagnostic center evaluation instead of taking
            # advantage of the algorithm calculator's existing center cache.
            if hasattr(calc, "results"):
                calc.results = {}
            if had_atoms_attr:
                calc.atoms = None

            def evaluate(at_positions):
                nonlocal calls
                probe_atoms = source_atoms.copy()
                probe_atoms.set_positions(np.asarray(at_positions, dtype=float), apply_constraint=False)
                force = np.asarray(calc.get_forces(probe_atoms), dtype=float)
                if force.shape != positions.shape or not np.all(np.isfinite(force)):
                    raise RuntimeError("paid reference calculator returned invalid force shape/data")
                calls += 1
                return force

            f0 = evaluate(positions).reshape(-1)[active]
            hessian = np.zeros((active.size, active.size), dtype=float)
            for column, flat_index in enumerate(active):
                displaced = positions.copy().reshape(-1)
                displaced[flat_index] += fd
                fp = evaluate(displaced.reshape(positions.shape)).reshape(-1)[active]
                hessian[:, column] = -(fp - f0) / fd
            hessian = 0.5 * (hessian + hessian.T)
            if not np.all(np.isfinite(hessian)):
                raise RuntimeError("paid reference finite-difference Hessian is nonfinite")
            values, vectors = np.linalg.eigh(hessian)
            root = int(np.argmin(values))
            full_mode = np.zeros(positions.size, dtype=float)
            full_mode[active] = vectors[:, root]
            full_mode = full_mode.reshape(positions.shape)
            return ReferenceComputation(
                mode=full_mode, curvature=float(values[root]), converged=True,
                pes_calls=calls, source="diagnostic_dense_fd_reference",
                root_selection="lowest",
                metadata={
                    "finite_difference": fd,
                    "active_dofs": int(active.size),
                    "operator": "symmetrized_forward_fd_physical_hessian",
                    "calculator_context": "direct_calculator_on_copied_atoms_results_cache_restored",
                    "algorithm_history_admission": False,
                    "algorithm_owner_forcecalls_incremented": False,
                },
            )
        finally:
            if hasattr(calc, "results"):
                calc.results = saved_results
            if had_atoms_attr:
                calc.atoms = saved_calc_atoms
            for key, value in saved_scalar_state.items():
                setattr(calc, key, value)

    return measure, snapshots


def _t12_reference_to_state(reference):
    if reference is None:
        return None
    from saddlemill.dimertools.foundation_types import json_safe
    return {
        "schema": reference.schema,
        "available": bool(reference.available),
        "mode": None if reference.mode is None else np.asarray(reference.mode, dtype=float).tolist(),
        "curvature": reference.curvature,
        "converged": bool(reference.converged),
        "overlap_with_evaluated_mode": reference.overlap_with_evaluated_mode,
        "angle_degrees": reference.angle_degrees,
        "source": reference.source,
        "root_selection": reference.root_selection,
        "pes_calls": int(reference.pes_calls),
        "elapsed_ns": int(reference.elapsed_ns),
        "unavailable_reason": reference.unavailable_reason,
        "metadata": json_safe(reference.metadata),
    }


def _t12_reference_from_state(payload):
    if payload is None:
        return None
    from saddlemill.dimertools.mode_diagnostics import REFERENCE_MODE_SCHEMA, ReferenceModeRecord
    data = dict(payload)
    if data.pop("schema", REFERENCE_MODE_SCHEMA) != REFERENCE_MODE_SCHEMA:
        raise ValueError("unsupported mode-diagnostics reference-mode state schema")
    return ReferenceModeRecord(**data)


def _t12_diagnostic_from_state(payload):
    if payload is None:
        return None
    from saddlemill.dimertools.mode_diagnostics import (
        MODE_DIAGNOSTIC_SCHEMA, ModeGapEstimate, StandardModeDiagnosticRecord,
    )
    data = dict(payload)
    if data.pop("schema", MODE_DIAGNOSTIC_SCHEMA) != MODE_DIAGNOSTIC_SCHEMA:
        raise ValueError("unsupported mode-diagnostics mode-diagnostic state schema")
    gap_data = dict(data.pop("gap"))
    gap_data.pop("schema", None)
    data["gap"] = ModeGapEstimate(**gap_data)
    return StandardModeDiagnosticRecord(**data)


def _t12_hindsight_state_to_dict(state):
    from saddlemill.dimertools.foundation_types import json_safe
    return {
        "schema": state.schema,
        "state_id": int(state.state_id),
        "geometry_id": str(state.geometry_id),
        "positions": np.asarray(state.positions, dtype=float).tolist(),
        "mode": None if state.mode is None else np.asarray(state.mode, dtype=float).tolist(),
        "curvature": state.curvature,
        "mode_source": state.mode_source,
        "evaluated_state_id": state.evaluated_state_id,
        "diagnostic": None if state.diagnostic is None else state.diagnostic.to_dict(),
        "reference": _t12_reference_to_state(state.reference),
        "metadata": json_safe(state.metadata),
    }


def _t12_hindsight_state_from_dict(payload):
    from saddlemill.dimertools.hindsight import HINDSIGHT_STATE_SCHEMA, HindsightPathState
    data = dict(payload)
    if data.pop("schema", HINDSIGHT_STATE_SCHEMA) != HINDSIGHT_STATE_SCHEMA:
        raise ValueError("unsupported mode-diagnostics hindsight path state schema")
    data["diagnostic"] = _t12_diagnostic_from_state(data.get("diagnostic"))
    data["reference"] = _t12_reference_from_state(data.get("reference"))
    return HindsightPathState(**data)


@dataclass
class WaveBRuntimeController:
    options: dict[str, object]
    physical_hessian: PhysicalHessianModel | None = None
    schedule_state: ModeScheduleRuntimeState | None = None
    damping_selection: DampingSelection | None = None
    pending_schedule_evaluation: object | None = None
    last_schedule_metadata: dict[str, object] | None = None
    last_isopotential_metadata: dict[str, object] | None = None
    last_physical_hessian_update: dict[str, object] | None = None
    last_physical_hessian_block_update: dict[str, object] | None = None
    physical_hessian_block_commit_serial: int = 0
    physical_hessian_processed_stencil_ids: set[str] | None = None
    mode_diagnostics_runtime: object | None = None
    last_mode_diagnostic: object | None = None
    last_mode_decision: object | None = None
    last_mode_source_payload: object | None = None
    mode_hindsight_states: list = None
    restored_mode_diagnostics_state: dict[str, object] | None = None
    force_real_solve: bool = False

    def __post_init__(self) -> None:
        if self.mode_hindsight_states is None:
            self.mode_hindsight_states = []
        if self.physical_hessian_processed_stencil_ids is None:
            self.physical_hessian_processed_stencil_ids = set()
        mode_cfg = dict(self.options.get("mode_schedule", {}) or {})
        self.scheduler = (
            BoundedModeScheduler(_scheduler_config(mode_cfg), damping_config=_damping_config(mode_cfg))
            if bool(mode_cfg.get("enabled", False)) else None
        )

    @classmethod
    def from_state_dict(cls, options: Mapping[str, object], state: Mapping[str, object] | None):
        obj = cls(dict(options))
        if not state:
            return obj
        if state.get("schema") != WAVE_B_RUNTIME_SCHEMA:
            raise ValueError("unsupported Wave-B runtime state schema")
        saved_selector = state.get("physical_hessian_selector_identity")
        current_selector = _physical_hessian_selector_identity(obj.options)
        if saved_selector is not None:
            if dict(saved_selector) != current_selector:
                raise ValueError("physical-Hessian resume selector/config mismatch")
        elif current_selector.get("probe_batching") in {"same_center_block", "solver_retained_block"}:
            raise ValueError(
                "cannot restore block physical-Hessian probe batching from runtime "
                "state that predates physical-Hessian probe-batching identity"
            )
        ph = state.get("physical_hessian")
        if ph is not None:
            obj.physical_hessian = PhysicalHessianModel.from_state_dict(dict(ph))
        sched = state.get("mode_schedule")
        if sched is not None:
            obj.schedule_state = ModeScheduleRuntimeState.from_state_dict(dict(sched))
            if obj.scheduler is not None:
                cfg = obj.scheduler.config
                if obj.schedule_state.skip_entry_gate != cfg.skip_entry_gate:
                    raise ValueError("mode-schedule resume skip_entry_gate/config mismatch")
                if obj.schedule_state.refresh_policy != cfg.refresh_policy:
                    raise ValueError("mode-schedule resume refresh_policy/config mismatch")
        damp = state.get("damping_selection")
        if damp is not None:
            d = dict(damp)
            obj.damping_selection = DampingSelection(
                policy_mode=str(d["policy_mode"]), selected_lambda=d.get("selected_lambda"),
                parallel_metric=float(d["parallel_metric"]), metric_name=str(d["metric_name"]),
                sequence_id=str(d["sequence_id"]), identity=str(d["identity"]),
            )
        obj.last_schedule_metadata = dict(state.get("last_schedule_metadata") or {}) or None
        obj.last_isopotential_metadata = dict(state.get("last_isopotential_metadata") or {}) or None
        obj.last_physical_hessian_update = dict(state.get("last_physical_hessian_update") or {}) or None
        obj.last_physical_hessian_block_update = dict(
            state.get("last_physical_hessian_block_update") or {}
        ) or None
        obj.physical_hessian_block_commit_serial = int(
            state.get("physical_hessian_block_commit_serial", 0)
        )
        if obj.physical_hessian_block_commit_serial < 0:
            raise ValueError("invalid negative physical-Hessian block commit serial")
        obj.physical_hessian_processed_stencil_ids = {
            str(value) for value in state.get("physical_hessian_processed_stencil_ids", ())
        }
        obj.restored_mode_diagnostics_state = dict(state.get("mode_diagnostics") or {}) or None
        obj.mode_hindsight_states = [
            _t12_hindsight_state_from_dict(item)
            for item in list(state.get("mode_hindsight_states") or [])
        ]
        if len(obj.mode_hindsight_states) > 64:
            raise ValueError("restored mode-diagnostics hindsight state exceeds bounded 64-state limit")
        return obj

    def to_state_dict(self) -> dict[str, object]:
        damp = None
        if self.damping_selection is not None:
            damp = {
                "policy_mode": self.damping_selection.policy_mode,
                "selected_lambda": self.damping_selection.selected_lambda,
                "parallel_metric": self.damping_selection.parallel_metric,
                "metric_name": self.damping_selection.metric_name,
                "sequence_id": self.damping_selection.sequence_id,
                "identity": self.damping_selection.identity,
            }
        return {
            "schema": WAVE_B_RUNTIME_SCHEMA,
            "physical_hessian_selector_identity": _physical_hessian_selector_identity(self.options),
            "physical_hessian": None if self.physical_hessian is None else self.physical_hessian.state_dict(),
            "mode_schedule": None if self.schedule_state is None else self.schedule_state.to_state_dict(),
            "damping_selection": damp,
            "last_schedule_metadata": self.last_schedule_metadata,
            "last_isopotential_metadata": self.last_isopotential_metadata,
            "last_physical_hessian_update": self.last_physical_hessian_update,
            "last_physical_hessian_block_update": self.last_physical_hessian_block_update,
            "physical_hessian_block_commit_serial": int(self.physical_hessian_block_commit_serial),
            "physical_hessian_processed_stencil_ids": sorted(
                self.physical_hessian_processed_stencil_ids or set()
            ),
            "mode_diagnostics": (
                self.mode_diagnostics_runtime.to_state_dict()
                if self.mode_diagnostics_runtime is not None
                else self.restored_mode_diagnostics_state
            ),
            "mode_hindsight_states": [
                _t12_hindsight_state_to_dict(item) for item in self.mode_hindsight_states
            ],
        }

    def _ensure_mode_diagnostics(self, owner):
        cfg = dict(self.options.get("mode_diagnostics", {}) or {})
        if not (bool(cfg.get("enabled", False)) or bool(cfg.get("paid_reference", False)) or bool(cfg.get("hindsight", False))):
            return None
        history = getattr(owner, "canonical_force_history", None)
        state = None if history is None else history.current
        center = None if state is None else state.center
        if center is None:
            return None
        space = active_space_for_owner(owner, state.center_positions)
        if self.mode_diagnostics_runtime is None:
            from saddlemill.dimertools.dimer_factory import create_mode_diagnostics_runtime
            runtime = create_mode_diagnostics_runtime(
                {"ourModeDiagnostics": cfg}, coordinate_space=space,
                accounting=history.accounting, physical_hessian_model=self.physical_hessian,
            )
            if runtime is None:
                return None
            if self.restored_mode_diagnostics_state:
                saved = dict(self.restored_mode_diagnostics_state.get("resolved") or {})
                current = runtime.resolved_metadata()
                for key in ("criterion", "mode_residual_threshold", "mode_angular_error_threshold_degrees", "paid_reference", "hindsight"):
                    if saved.get(key) != current.get(key):
                        raise ValueError("mode-diagnostics resume selector/config mismatch")
                runtime.restore_state_dict(self.restored_mode_diagnostics_state)
                self.last_mode_diagnostic = runtime._last_record
                self.last_mode_decision = runtime._last_decision
            self.mode_diagnostics_runtime = runtime
        else:
            if self.mode_diagnostics_runtime.coordinate_space.identity != space.identity:
                raise ValueError("mode-diagnostics active-coordinate identity changed within attempt")
            self.mode_diagnostics_runtime.physical_hessian_model = self.physical_hessian
        return self.mode_diagnostics_runtime

    def _capture_t12_after_real_solve(self, owner) -> None:
        runtime = self._ensure_mode_diagnostics(owner)
        if runtime is None:
            return
        history = getattr(owner, "canonical_force_history", None)
        state = None if history is None else history.current
        result = getattr(owner, "last_minmode_solver_result", None)
        if result is not None:
            self.last_mode_source_payload = ("solver", result)
            self.last_mode_diagnostic = runtime.standardize_solver_result(result)
        else:
            torque = getattr(owner, "last_dimer_rotational_torque", None)
            if state is not None and torque is not None:
                try:
                    dR = float(owner.control.get_parameter("dimer_separation"))
                except Exception:
                    dR = 0.0
                if dR > 0.0:
                    payload = {
                        "evaluated_mode": np.asarray(owner.get_eigenmode(), dtype=float).copy(),
                        "torque": np.asarray(torque, dtype=float).copy(),
                        "curvature": float(owner.get_curvature()),
                        "geometry_id": str(state.geometry_id),
                        "state_id": int(state.state_id),
                        "state_uid": str(state.state_uid),
                        "fd_distance": dR,
                        "returned_mode": np.asarray(owner.get_eigenmode(), dtype=float).copy(),
                    }
                    self.last_mode_source_payload = ("dimer_torque", payload)
                    self.last_mode_diagnostic = runtime.standardize_dimer_torque(**payload)
        if runtime.config.get("hindsight", False) and self.last_mode_diagnostic is not None and state is not None:
            from saddlemill.dimertools.hindsight import HindsightPathState
            self.mode_hindsight_states.append(HindsightPathState(
                state_id=state.state_id, geometry_id=state.geometry_id,
                positions=np.asarray(state.center_positions, dtype=float),
                mode=np.asarray(owner.get_eigenmode(), dtype=float),
                curvature=float(owner.get_curvature()),
                mode_source=("real_t09_mode_solve" if result is not None else "real_dimer_rotation"),
                evaluated_state_id=str(getattr(self.last_mode_diagnostic, "state_uid", "")),
                diagnostic=self.last_mode_diagnostic,
                metadata={"solver": str(getattr(owner, "min_mode_finder", "dimer"))},
            ))
            if len(self.mode_hindsight_states) > 64:
                self.mode_hindsight_states = self.mode_hindsight_states[-64:]
        if runtime.config.get("paid_reference", False):
            measure = getattr(owner, "t12_paid_reference_measure", None)
            snapshots = getattr(owner, "t12_paid_reference_snapshots", None)
            if not callable(measure) or not isinstance(snapshots, Mapping):
                measure, snapshots = _t12_paid_reference_context(owner, runtime, state)
            evaluated = (
                None if self.last_mode_diagnostic is None
                else self.last_mode_diagnostic.evaluated_mode
            )
            if evaluated is None:
                raise RuntimeError("paid reference requested but no same-center evaluated mode is available")
            reference, report = runtime.run_paid_reference(
                measure, evaluated_mode=evaluated, snapshots=snapshots
            )
            if self.mode_hindsight_states:
                last = self.mode_hindsight_states[-1]
                from saddlemill.dimertools.hindsight import HindsightPathState
                self.mode_hindsight_states[-1] = HindsightPathState(
                    state_id=last.state_id, geometry_id=last.geometry_id, positions=last.positions,
                    mode=last.mode, curvature=last.curvature, mode_source=last.mode_source,
                    evaluated_state_id=last.evaluated_state_id, diagnostic=last.diagnostic,
                    reference=reference, metadata={
                        **dict(last.metadata),
                        "paid_reference_noninterference": dict(report.__dict__),
                    },
                )

    def finalize_hindsight(self, owner, *, run_converged: bool):
        runtime = self._ensure_mode_diagnostics(owner)
        if runtime is None or not runtime.config.get("hindsight", False):
            return []
        from saddlemill.dimertools.hindsight import HindsightGeometryConvention
        atoms = getattr(owner, "atoms", None)
        positions = np.asarray(owner.get_positions(), dtype=float)
        cell = None
        pbc = (False, False, False)
        if atoms is not None:
            try:
                cell_arr = np.asarray(atoms.cell, dtype=float)
                pbc = tuple(bool(x) for x in np.asarray(atoms.pbc).reshape(3))
                if any(pbc):
                    cell = cell_arr
            except Exception:
                cell = None; pbc = (False, False, False)
        convention = HindsightGeometryConvention(cell=cell, pbc=pbc)
        if run_converged:
            final_mode = np.asarray(owner.get_eigenmode(), dtype=float)
            final_curvature = float(owner.get_curvature())
            source = "converged_runtime_mode"
            reason = ""
        else:
            final_mode = None
            final_curvature = None
            source = "unavailable"
            reason = "run_not_converged_no_terminal_mode_fabricated"
        records = runtime.build_hindsight(
            tuple(self.mode_hindsight_states), geometry_convention=convention,
            final_positions=positions, final_mode=final_mode, final_curvature=final_curvature,
            terminal_mode_source=source, run_converged=bool(run_converged),
            terminal_mode_availability_reason=reason,
        )
        return [record.to_dict() for record in records]

    def _ensure_physical_hessian(self, owner, observation) -> PhysicalHessianModel | None:
        cfg = dict(self.options.get("physical_hessian", {}) or {})
        update = str(cfg.get("update", "off"))
        if update == "off":
            return None
        space = active_space_for_owner(owner, observation.positions)
        if self.physical_hessian is not None:
            if self.physical_hessian.coordinate_space.identity != space.identity:
                raise ValueError("physical-Hessian active-coordinate identity changed within attempt")
            return self.physical_hessian
        representation = str(cfg.get("representation", REPRESENTATION_ONLINE_DENSE))
        # Dense representations retain the existing conservative full-matrix guard.
        # Compact windowed TS-BFGS never allocates an n x n physical Hessian.
        if representation != REPRESENTATION_FORCEBANK_WINDOW_COMPACT:
            MatrixMemoryBudget(max_bytes=int(cfg.get("max_matrix_bytes", 536870912))).ensure(
                max(1, space.active_dof_count),
                matrix_count=PHYSICAL_HESSIAN_SIMULTANEOUS_DENSE_MATRICES,
                label="physical-Hessian physical Hessian peak dense workspace",
            )
        self.physical_hessian = PhysicalHessianModel(
            space,
            update_type=update,
            initial_hessian=float(cfg.get("initial_hessian", 1.0)),
            representation=representation,
            denominator_tolerance=float(cfg.get("denominator_tolerance", 1.0e-12)),
            rank_relative_tolerance=float(cfg.get("rank_relative_tolerance", 1.0e-10)),
            rank_absolute_tolerance=float(cfg.get("rank_absolute_tolerance", 1.0e-12)),
            dependence_noise_tolerance=float(cfg.get("dependence_noise_tolerance", 1.0e-6)),
            block_symmetry_noise_tolerance=float(cfg.get("block_symmetry_noise_tolerance", 1.0e-3)),
            secant_residual_tolerance=float(cfg.get("secant_residual_tolerance", 1.0e-8)),
            spectral_zero_tolerance=float(cfg.get("spectral_zero_tolerance", 1.0e-12)),
        )
        return self.physical_hessian

    @staticmethod
    def _update_summary(result) -> dict[str, object]:
        return {
            "accepted": bool(result.accepted),
            "reason": str(result.reason),
            "update_kind": str(result.update_type),
            "model_age": int(result.model_age),
        }

    def _rebuild_windowed_physical_hessian(self, owner, model):
        if model is None or model.representation == REPRESENTATION_ONLINE_DENSE:
            return None
        history = getattr(owner, "canonical_force_history", None)
        if history is None:
            raise RuntimeError("ForceBank-windowed physical Hessian requires canonical_force_history")
        cfg = dict(self.options.get("physical_hessian", {}) or {})
        result = model.rebuild_from_force_history(
            history,
            pair_sources=cfg.get("forcebank_pair_sources", "center_center"),
            max_pairs=int(cfg.get("forcebank_max_pairs", 0)),
        )
        self.last_physical_hessian_update = self._update_summary(result)
        self.last_physical_hessian_update.update(model.window_diagnostics())
        return result

    def observe_center(self, owner, observation) -> None:
        if observation is None:
            return
        # PhysicalForceEvaluator.record_center() returns the owning TranslationState,
        # whereas physical-Hessian consumes the accepted raw center ForceObservation.  Normalize
        # that boundary here so the shared runtime never treats state metadata as a
        # physical observation.
        center = getattr(observation, "center", None) or observation
        if getattr(center, "role", None) != "center":
            return
        rfo_runtime = getattr(owner, "_rfo_translation_runtime", None)
        if rfo_runtime is not None:
            rfo_runtime.prepare_evaluated_center(center)
        model = self._ensure_physical_hessian(owner, center)
        if model is not None:
            if model.representation == REPRESENTATION_ONLINE_DENSE:
                self.last_physical_hessian_update = self._update_summary(model.observe_translation(center))
            else:
                self._rebuild_windowed_physical_hessian(owner, model)
        if rfo_runtime is not None:
            rfo_runtime.finalize_evaluated_center_trust()

    def observe_probe(self, owner, history, observation) -> None:
        if observation is None:
            return
        cfg = dict(self.options.get("physical_hessian", {}) or {})
        if str(cfg.get("probe_updates", "off")) != "same_center_physical_hvp":
            return
        if str(cfg.get("probe_batching", "sequential")) in {
            "same_center_block", "solver_retained_block"
        }:
            # The raw observation/stencil remains recorded. Scientific model
            # admission is deferred until the selected block boundary,
            # preventing incremental+block double use.
            return
        model = self._ensure_physical_hessian(owner, history.current.center if history.current is not None else observation)
        if model is None or observation.purpose != WorkPurpose.ALGORITHM.value or not observation.is_physical:
            return
        if model.representation != REPRESENTATION_ONLINE_DENSE:
            self._rebuild_windowed_physical_hessian(owner, model)
            return
        # The directional isopotential estimator intentionally has no stencil and must never
        # be promoted into the physical-Hessian block.
        if not observation.stencil_id or observation.family == "isopotential":
            return
        stencil = next((s for s in reversed(tuple(history.stencils)) if str(getattr(s, "stencil_id", "")) == observation.stencil_id), None)
        state = history.current
        if stencil is None or state is None:
            return
        direction = np.asarray(getattr(stencil, "direction"), dtype=float)
        space = model.coordinate_space
        request = HVPRequest(
            state_id=state.state_id, state_uid=state.state_uid, geometry_id=state.geometry_id,
            direction=direction, coordinate_space=space, purpose=WorkPurpose.ALGORITHM,
            source=str(getattr(stencil, "source", observation.source)), family=str(getattr(stencil, "family", observation.family)),
            stencil_id=observation.stencil_id,
        )
        result = FiniteDifferenceForceHVPBackend(history).apply(request)
        if result.available:
            self.last_physical_hessian_update = self._update_summary(model.observe_probe_block([result]))

    def commit_same_center_probe_block(self, owner) -> None:
        """Commit current-state physical HVP stencils as one multisecant block.

        This path is intended for already-paid same-center finite-difference
        probes produced by a completed Dimer mode solve. It performs no
        calculator evaluation. Raw force observations remain authoritative;
        each complete stencil is reconstructed through the typed physical-HVP
        backend and admitted at most once across runtime resume. Derived or
        otherwise non-admissible stencils are excluded explicitly and recorded
        in the block provenance rather than being relabeled as physical data.
        """
        cfg = dict(self.options.get("physical_hessian", {}) or {})
        if str(cfg.get("probe_batching", "sequential")) != "same_center_block":
            return
        if str(cfg.get("probe_updates", "off")) != "same_center_physical_hvp":
            raise RuntimeError("same-center block commit requested without physical HVP admission")
        if str(cfg.get("update", "off")) != "ts_bfgs":
            raise RuntimeError("same-center block commit requires update=ts_bfgs")
        if str(getattr(owner, "min_mode_finder", "")).strip().lower() != "dimer":
            raise RuntimeError("same-center block commit is validated only for min_mode_finder=dimer")

        history = getattr(owner, "canonical_force_history", None)
        state = None if history is None else getattr(history, "current", None)
        center = None if state is None else getattr(state, "center", None)
        if state is None or center is None:
            raise RuntimeError("same-center block commit requires a current canonical center")
        model = self._ensure_physical_hessian(owner, center)
        if model is None:
            raise RuntimeError("same-center block commit requires an active physical-Hessian model")

        from saddlemill.dimertools.force_stencil_history import reconstruct_physical_hvp

        processed = self.physical_hessian_processed_stencil_ids
        if processed is None:
            processed = set()
            self.physical_hessian_processed_stencil_ids = processed
        candidates = []
        candidate_stencil_ids = []
        considered_stencil_ids = []
        excluded = []
        for stencil in history.iter_stencils(complete_only=True):
            if int(getattr(stencil, "state_id", -1)) != int(state.state_id):
                continue
            if str(getattr(stencil, "purpose", "")).strip().lower() != WorkPurpose.ALGORITHM.value:
                continue
            # This batching mode is deliberately narrow: only Dimer rotation
            # derivative stencils have been validated as one same-center block.
            # Other physical probe families retain their existing admission paths.
            if str(getattr(stencil, "family", "")).strip().lower() != "rotation":
                continue
            stencil_id = str(getattr(stencil, "stencil_id", "")).strip()
            if not stencil_id or stencil_id in processed:
                continue
            considered_stencil_ids.append(stencil_id)
            result = reconstruct_physical_hvp(
                history, stencil, coordinate_space=model.coordinate_space,
                purpose=WorkPurpose.ALGORITHM,
            )
            if not result.available:
                excluded.append({
                    "stencil_id": stencil_id,
                    "source": str(getattr(stencil, "source", "")),
                    "family": str(getattr(stencil, "family", "")),
                    "reason": str(result.unavailable_reason),
                })
                continue
            candidates.append(result)
            candidate_stencil_ids.append(stencil_id)

        if not considered_stencil_ids:
            return
        # Each completed solve block is a one-shot admission attempt, just as a
        # rejected center-center secant is not replayed on the next center.
        processed.update(considered_stencil_ids)
        self.physical_hessian_block_commit_serial += 1
        solve_id = (
            f"{state.state_uid}:dimer:same_center_block:"
            f"{self.physical_hessian_block_commit_serial}"
        )
        age_before = int(model.model_age)
        if candidates:
            update = model.observe_probe_block(tuple(candidates))
            self.last_physical_hessian_update = self._update_summary(update)
            age_after = int(model.model_age)
            expected_delta = 1 if update.accepted else 0
            if age_after - age_before != expected_delta:
                raise RuntimeError("same-center block changed physical-Hessian age unexpectedly")
            accepted = bool(update.accepted)
            reason = str(update.reason)
            processed_rank = int(update.processed_block_rank)
            dropped_rank = int(update.dropped_dependent_rank)
            hvp_ids = tuple(str(value) for value in update.observation_ids)
            secant_residual = update.secant_residual
        else:
            age_after = age_before
            accepted = False
            reason = "no_admissible_physical_hvp"
            processed_rank = 0
            dropped_rank = 0
            hvp_ids = ()
            secant_residual = None

        self.last_physical_hessian_block_update = {
            "probe_batching": "same_center_block",
            "solver": "dimer",
            "solve_id": solve_id,
            "commit_serial": int(self.physical_hessian_block_commit_serial),
            "state_id": int(state.state_id),
            "state_uid": str(state.state_uid),
            "geometry_id": str(state.geometry_id),
            "considered_stencil_ids": tuple(considered_stencil_ids),
            "admitted_stencil_ids": tuple(candidate_stencil_ids),
            "excluded_stencils": tuple(excluded),
            "input_hvp_count": len(candidates),
            "retained_hvp_count": len(candidates),
            "processed_rank": processed_rank,
            "dropped_rank": dropped_rank,
            "hvp_identities": hvp_ids,
            "endpoint_observation_ids": tuple(
                tuple(str(value) for value in result.endpoint_observation_ids)
                for result in candidates
            ),
            "accepted": accepted,
            "reason": reason,
            "secant_residual": secant_residual,
            "model_age_before": age_before,
            "model_age_after": age_after,
            "incremental_admission_suppressed": True,
            "additional_pes_calls": 0,
            "admission_basis": "complete_same_center_dimer_rotation_physical_hvp_stencils",
        }

    def commit_solver_retained_probe_block(self, owner, solver_result, hvp_results) -> None:
        """Commit one final retained iterative-solver HVP block to physical B.

        This is admission-only: it never evaluates the calculator.  The solver
        has already paid for the HVPs and the final retained indices come from
        its audit record, including restart-aware Olsen/JD retention.
        """
        cfg = dict(self.options.get("physical_hessian", {}) or {})
        if str(cfg.get("probe_batching", "sequential")) != "solver_retained_block":
            return
        if str(cfg.get("probe_updates", "off")) != "same_center_physical_hvp":
            raise RuntimeError("solver-retained block commit requested without probe admission enabled")
        if str(cfg.get("update", "off")) != "ts_bfgs":
            raise RuntimeError("solver-retained block commit requires update=ts_bfgs")

        solver = str(getattr(owner, "min_mode_finder", "")).strip().lower()
        if solver not in {"olsen_jd", "lanczos", "davidson"}:
            raise RuntimeError(
                "solver-retained block commit is validated only for olsen_jd, lanczos, or davidson"
            )
        audit = getattr(solver_result, "audit", None)
        if audit is None:
            raise RuntimeError("solver-retained block commit requires solver audit metadata")
        audit_metadata = dict(getattr(audit, "metadata", {}) or {})
        candidates = tuple(hvp_results or ())
        retained_indices = tuple(int(v) for v in audit_metadata.get("retained_hvp_indices", ()))
        if solver == "olsen_jd":
            if not bool(audit_metadata.get("retained_span_equivalent_complete", False)):
                raise RuntimeError(
                    "Olsen/JD retained block lacks a complete raw-query to returned-V/AV span proof"
                )
        elif not bool(audit_metadata.get("retained_t01_match_complete", False)):
            raise RuntimeError("solver retained HVP set is not completely matched to typed typed-HVP results")
        if not retained_indices:
            raise RuntimeError("solver-retained block commit has no retained HVP indices")
        if any(index < 0 or index >= len(candidates) for index in retained_indices):
            raise RuntimeError("solver-retained HVP index is outside the recorded solve results")
        retained = tuple(candidates[index] for index in retained_indices)
        retained_basis = tuple(getattr(audit, "retained_basis", ()))
        retained_actions = tuple(getattr(audit, "retained_actions", ()))
        span_proof = None
        if solver == "olsen_jd":
            from saddlemill.dimertools.minmode_solvers import _right_rotation_span_equivalence
            raw_basis = tuple(np.asarray(getattr(result, "direction", ()), dtype=float).reshape(-1) for result in retained)
            raw_actions = []
            for result in retained:
                action_value = getattr(result, "action", None)
                if action_value is None:
                    raise RuntimeError("solver-retained block contains an unavailable HVP")
                raw_actions.append(np.asarray(action_value, dtype=float).reshape(-1))
            span_proof = _right_rotation_span_equivalence(
                raw_basis, tuple(raw_actions), retained_basis, retained_actions,
                tolerance=float(audit_metadata.get("retained_span_equivalence_tolerance", 1.0e-10)),
            )
            if not bool(span_proof.get("complete", False)):
                raise RuntimeError(
                    "Olsen/JD raw typed HVP block is not span-equivalent to the returned Sella V/AV block"
                )
        else:
            if len(retained) != len(retained_basis) or len(retained) != len(retained_actions):
                raise RuntimeError("solver retained HVP count does not match final retained basis")
            for retained_index, (result, basis, action) in enumerate(
                zip(retained, retained_basis, retained_actions)
            ):
                direction_flat = np.asarray(getattr(result, "direction", ()), dtype=float).reshape(-1)
                action_value = getattr(result, "action", None)
                if action_value is None:
                    raise RuntimeError("solver-retained block contains an unavailable HVP")
                action_flat = np.asarray(action_value, dtype=float).reshape(-1)
                basis_flat = np.asarray(basis, dtype=float).reshape(-1)
                retained_action_flat = np.asarray(action, dtype=float).reshape(-1)
                if (
                    direction_flat.shape != basis_flat.shape
                    or action_flat.shape != retained_action_flat.shape
                    or not np.allclose(direction_flat, basis_flat, atol=1.0e-11, rtol=1.0e-11)
                    or not np.allclose(action_flat, retained_action_flat, atol=1.0e-11, rtol=1.0e-11)
                ):
                    raise RuntimeError(
                        "solver retained HVP identities do not match the final retained audit subspace "
                        f"at retained index {retained_index}"
                    )

        history = getattr(owner, "canonical_force_history", None)
        state = None if history is None else getattr(history, "current", None)
        center = None if state is None else getattr(state, "center", None)
        if center is None:
            raise RuntimeError("solver-retained block commit requires a current canonical center")
        model = self._ensure_physical_hessian(owner, center)
        if model is None:
            raise RuntimeError("solver-retained block commit requires an active physical-Hessian model")

        state_uid = str(getattr(state, "state_uid", ""))
        geometry_id = str(getattr(state, "geometry_id", ""))
        for index, result in zip(retained_indices, retained):
            metadata = getattr(result, "metadata", None)
            if metadata is None:
                raise RuntimeError("retained HVP result lacks typed metadata")
            if not bool(getattr(metadata, "physical", False)) or bool(getattr(metadata, "model_derived", False)):
                raise RuntimeError("solver-retained block contains a nonphysical/model-derived HVP")
            if getattr(metadata, "purpose", None) is not WorkPurpose.ALGORITHM:
                raise RuntimeError("solver-retained block contains a non-algorithm HVP")
            if str(getattr(metadata, "state_uid", "")) != state_uid:
                raise RuntimeError("solver-retained block contains a mixed/stale state HVP")
            if str(getattr(metadata, "geometry_id", "")) != geometry_id:
                raise RuntimeError("solver-retained block contains a mixed/stale geometry HVP")

        self.physical_hessian_block_commit_serial += 1
        solve_id = (
            f"{state_uid}:{solver}:solver_retained_block:"
            f"{self.physical_hessian_block_commit_serial}"
        )
        input_endpoint_ids = tuple(
            tuple(str(value) for value in getattr(result, "endpoint_observation_ids", ()))
            for result in candidates
        )
        retained_endpoint_ids = tuple(
            tuple(str(value) for value in getattr(result, "endpoint_observation_ids", ()))
            for result in retained
        )
        age_before = int(model.model_age)
        update = model.observe_probe_block(retained)
        age_after = int(model.model_age)
        expected_delta = 1 if update.accepted else 0
        if age_after - age_before != expected_delta:
            raise RuntimeError("solver-retained block changed physical-Hessian age unexpectedly")
        self.last_physical_hessian_update = self._update_summary(update)
        self.last_physical_hessian_block_update = {
            "probe_batching": "solver_retained_block",
            "solver": solver,
            "solve_id": solve_id,
            "commit_serial": int(self.physical_hessian_block_commit_serial),
            "state_id": int(getattr(state, "state_id", -1)),
            "state_uid": state_uid,
            "geometry_id": geometry_id,
            "input_hvp_count": len(candidates),
            "retained_hvp_count": len(retained),
            "retained_hvp_indices": retained_indices,
            "processed_rank": int(update.processed_block_rank),
            "dropped_rank": int(update.dropped_dependent_rank),
            "hvp_identities": tuple(str(v) for v in update.observation_ids),
            "retained_hvp_identities": tuple(str(v) for v in update.observation_ids),
            "input_endpoint_observation_ids": input_endpoint_ids,
            "endpoint_observation_ids": retained_endpoint_ids,
            "retained_endpoint_observation_ids": retained_endpoint_ids,
            "accepted": bool(update.accepted),
            "reason": str(update.reason),
            "secant_residual": update.secant_residual,
            "model_age_before": age_before,
            "model_age_after": age_after,
            "incremental_admission_suppressed": True,
            "additional_pes_calls": 0,
            "restart_count": int(getattr(audit, "restart_count", 0)),
            "restart_reason": str(getattr(audit, "restart_reason", "")),
            "retained_hvp_admission_basis": str(
                audit_metadata.get("retained_hvp_admission_basis", "literal_column_match")
            ),
            "retained_span_equivalent_complete": (
                None if span_proof is None else bool(span_proof.get("complete", False))
            ),
            "retained_span_direction_relative_residual": (
                None if span_proof is None else span_proof.get("direction_relative_residual")
            ),
            "retained_span_action_relative_residual": (
                None if span_proof is None else span_proof.get("action_relative_residual")
            ),
            "retained_span_right_orthogonality_residual": (
                None if span_proof is None else span_proof.get("right_orthogonality_residual")
            ),
        }

    def physical_model_signal(self) -> PhysicalModelSignal | None:
        if self.physical_hessian is None:
            return None
        try:
            spectrum = self.physical_hessian.low_spectrum()
        except Exception:
            return PhysicalModelSignal(origin="physical_hessian_model", age=int(self.physical_hessian.model_age), valid=False)
        # ``PhysicalLowSpectrum.eigenvalues`` is an immutable tuple in the
        # physical-Hessian contract, not necessarily a NumPy array.
        lowest = None if len(spectrum.eigenvalues) == 0 else float(spectrum.eigenvalues[0])
        return PhysicalModelSignal(
            origin="physical_hessian_model",
            age=int(self.physical_hessian.model_age),
            valid=True,
            lowest_curvature=lowest,
            negative_mode_count=int(spectrum.negative_eigenvalue_count),
            observation_id=str(getattr(self.physical_hessian, "_previous_observation_id", "")),
            metadata={
                "current_state_uid": str(getattr(self.physical_hessian, "_current_state_uid", "")),
                "current_geometry_id": str(getattr(self.physical_hessian, "_current_geometry_id", "")),
                "previous_observation_id": str(getattr(self.physical_hessian, "_previous_observation_id", "")),
                "model_age": int(self.physical_hessian.model_age),
                "update_type": str(self.physical_hessian.update_type),
                "additional_pes_calls": 0,
            },
        )

    def _schedule_observation(self, owner, *, final_candidate: bool = False) -> ModeScheduleObservation | None:
        if self.scheduler is None:
            return None
        history = getattr(owner, "canonical_force_history", None)
        state = None if history is None else history.current
        center = None if state is None else state.center
        if state is None or center is None:
            return None
        space = active_space_for_owner(owner, state.center_positions)
        try:
            mode = np.asarray(owner.get_eigenmode(), dtype=float)
        except Exception:
            mode = np.asarray(owner.eigenmodes[0], dtype=float)
        normalized = space.normalized(mode)
        return ModeScheduleObservation(
            state_id=state.state_id,
            state_uid=state.state_uid,
            geometry_id=state.geometry_id,
            center_observation_id=center.observation_id,
            positions=state.center_positions,
            raw_center_force=center.forces,
            pre_prediction_mode=normalized,
            pre_prediction_mode_identity=mode_identity(normalized, space),
            coordinate_space=space,
            final_force_convergence_candidate=bool(final_candidate),
            predictor_enabled=str(dict(self.options.get("mode_schedule", {}) or {}).get("predictor_hook", "none")) != "none",
            prediction_data_valid=True,
            prediction_safety_ok=True,
            stale_data=False,
            physical_model=self.physical_model_signal(),
        )

    def _real_solve_entry_diagnostic(
        self, owner, observation: ModeScheduleObservation | None
    ) -> RealSolveEntryDiagnostic | None:
        mode_cfg = dict(self.options.get("mode_schedule", {}) or {})
        if str(mode_cfg.get("skip_entry_gate", "none")) == "none":
            return None
        raw = getattr(owner, "last_dimer_entry_torque", None)
        if observation is None:
            return RealSolveEntryDiagnostic(
                valid=False, state_id=0, state_uid="", geometry_id="",
                center_observation_id="", mode_identity="", reason="schedule_observation_unavailable",
            )
        payload = dict(raw) if isinstance(raw, Mapping) else {}
        valid = bool(payload.get("valid", False))
        entry_mode_id = ""
        if valid:
            try:
                entry_mode = np.asarray(payload.get("entry_mode"), dtype=float)
                entry_mode_id = mode_identity(entry_mode, observation.coordinate_space)
            except Exception:
                valid = False
        try:
            return RealSolveEntryDiagnostic(
                valid=valid,
                state_id=observation.state_id,
                state_uid=observation.state_uid,
                geometry_id=observation.geometry_id,
                center_observation_id=observation.center_observation_id,
                mode_identity=entry_mode_id,
                initial_torque_norm=payload.get("initial_torque_norm"),
                f_rot_max=payload.get("f_rot_max"),
                entry_mode_already_converged=bool(payload.get("entry_mode_already_converged", False)),
                reason=str(payload.get("reason", "entry_torque_unavailable")),
                additional_pes_calls=int(payload.get("additional_pes_calls", 0)),
            )
        except (TypeError, ValueError, FloatingPointError) as exc:
            return RealSolveEntryDiagnostic(
                valid=False, state_id=observation.state_id, state_uid=observation.state_uid,
                geometry_id=observation.geometry_id, center_observation_id=observation.center_observation_id,
                mode_identity=entry_mode_id, reason=f"entry_torque_invalid:{type(exc).__name__}",
            )

    @staticmethod
    def _prediction_requires_fail_closed_refresh(selector: str, prediction, use) -> tuple[bool, str]:
        selector = str(selector).strip().lower()
        if selector not in {"physical_eigen", "physical_olsen"}:
            return False, ""
        if prediction is None:
            return True, "physical_predictor_unavailable"
        status = str(getattr(prediction, "status", "")).strip().lower()
        if status in {"predicted", "already_aligned"}:
            return False, ""
        # Near-degenerate/root-ambiguous and identity/numerical failures must not
        # silently turn into a held-mode skip on physical-B arms.
        return True, status or str(getattr(use, "predictor_status", "unknown_predictor_failure"))

    def before_mode_solve(self, owner) -> bool:
        """Return True only when mode-schedule explicitly permits skipping this real solve."""
        if self.scheduler is None or self.force_real_solve:
            return False
        obs = self._schedule_observation(owner)
        if obs is None or self.schedule_state is None:
            return False
        evaluation = self.scheduler.evaluate(self.schedule_state, obs)
        self.pending_schedule_evaluation = evaluation
        self.last_schedule_metadata = {
            "decision": evaluation.decision.reason,
            "require_real_solve": bool(evaluation.decision.require_real_solve),
            "skipped_scheduled_solve": bool(evaluation.decision.skipped_scheduled_solve),
            "trigger_metadata": dict(evaluation.trigger_metadata),
            "predictor_disposition": evaluation.predictor_disposition,
        }
        runtime = self._ensure_mode_diagnostics(owner)
        md_cfg = dict(self.options.get("mode_diagnostics", {}) or {})
        if runtime is not None and bool(md_cfg.get("enabled", False)):
            criterion = str(md_cfg.get("mode_refresh_criterion", "legacy"))
            if criterion != "legacy" and bool(evaluation.decision.skipped_scheduled_solve):
                # mode-schedule has decided that its own hard/safety triggers permit a skip.
                # mode-diagnostics now obtains one *current-geometry algorithm-purpose* HVP
                # without rerunning the eigensolver, then contributes only an
                # additional refresh trigger.
                typed_factory = getattr(owner, "_t09_typed_hvp", None)
                if not callable(typed_factory):
                    self.last_schedule_metadata["t12_additional_trigger"] = {
                        "refresh_required": True,
                        "reason_code": "current_algorithm_hvp_route_unavailable",
                        "criterion": criterion,
                    }
                    return False
                callback, _backend, _identity = typed_factory(
                    source="t12_mode_refresh_hvp", family="mode_diagnostics",
                    model=self.physical_hessian, count_rotation=False,
                )
                callback(np.asarray(owner.get_eigenmode(), dtype=float))
                hvp = callback.results[-1]
                record = runtime.standardize_hvp(
                    hvp, returned_mode=np.asarray(owner.get_eigenmode(), dtype=float),
                    model_gap_purpose="algorithm",
                    fallback_model_gap=(criterion in {"angular_error", "hybrid"}),
                )
                self.last_mode_source_payload = ("hvp", hvp)
                self.last_mode_diagnostic = record
                if not bool(record.numerator_physical):
                    self.last_schedule_metadata["t12_additional_trigger"] = {
                        "refresh_required": True,
                        "reason_code": "current_standard_physical_residual_unavailable",
                        "criterion": criterion,
                    }
                    return False
                decision = runtime.evaluate_decision(
                    record, legacy_refresh_required=False
                )
                self.last_mode_decision = decision
                self.last_schedule_metadata["t12_additional_trigger"] = dict(decision.__dict__)
                if decision.refresh_required:
                    return False
            elif criterion == "legacy" and self.last_mode_diagnostic is not None:
                # Pure pass-through: this record is logging-only; mode-schedule remains
                # authoritative for the legacy Boolean and state transitions.
                self.last_mode_decision = runtime.evaluate_decision(
                    self.last_mode_diagnostic,
                    legacy_refresh_required=bool(evaluation.decision.require_real_solve),
                )
        prediction, use = apply_t10_prediction_after_schedule_decision(self, owner, evaluation)
        self.last_schedule_metadata.update(_t10_prediction_diagnostic_fields(prediction))
        self.last_schedule_metadata["predictor_disposition"] = use.disposition
        self.last_schedule_metadata["predictor_status"] = use.predictor_status
        predictor_selector = str(dict(self.options.get("mode_schedule", {}) or {}).get("predictor_hook", "none"))
        unsafe_prediction, unsafe_reason = self._prediction_requires_fail_closed_refresh(
            predictor_selector, prediction, use
        )
        mode_cfg = dict(self.options.get("mode_schedule", {}) or {})
        new_skip_policy_active = bool(
            str(mode_cfg.get("skip_entry_gate", "none")) != "none"
            or str(mode_cfg.get("refresh_policy", "bounded")) != "bounded"
        )
        if (
            bool(evaluation.decision.skipped_scheduled_solve)
            and new_skip_policy_active
            and unsafe_prediction
        ):
            self.last_schedule_metadata["predictor_safety_refresh"] = {
                "refresh_required": True, "reason": unsafe_reason, "additional_pes_calls": 0
            }
            # The scheduler itself voted to skip, so this evaluation cannot be
            # passed to complete_real_solve(). Returning False asks for a real
            # Dimer solve; clear the pending skip decision so the completed
            # solve starts a fresh sequence instead.
            self.pending_schedule_evaluation = None
            return False
        if not evaluation.decision.skipped_scheduled_solve:
            return False
        state, use = self.scheduler.complete_skip(evaluation, prediction_result=prediction)
        self.schedule_state = state
        self.pending_schedule_evaluation = None
        self.last_schedule_metadata["predictor_disposition"] = use.disposition
        self.last_schedule_metadata["predictor_status"] = use.predictor_status
        owner.eigenmodes[0] = np.asarray(use.held_mode, dtype=float).copy()
        return True

    def after_real_mode_solve(self, owner, *, fresh_physical_validation: bool = True) -> None:
        # Sequential probe admission occurs during force recording. Block
        # admission is deliberately delayed until the real Dimer solve has
        # completed so all already-paid same-center stencils are available, but
        # it is committed before any downstream model-signal consumer reads B.
        self.commit_same_center_probe_block(owner)
        mode_cfg = dict(self.options.get("mode_schedule", {}) or {})
        obs = self._schedule_observation(owner)
        mode = np.asarray(owner.get_eigenmode(), dtype=float)
        entry_diagnostic = self._real_solve_entry_diagnostic(owner, obs)
        if self.scheduler is not None and obs is not None:
            if self.schedule_state is None:
                self.schedule_state = self.scheduler.initialize_after_real_solve(
                    obs, solved_mode=mode, fresh_physical_validation=fresh_physical_validation,
                    physical_model=self.physical_model_signal(), entry_diagnostic=entry_diagnostic,
                )
            elif self.pending_schedule_evaluation is not None:
                self.schedule_state = self.scheduler.complete_real_solve(
                    self.pending_schedule_evaluation, solved_mode=mode,
                    fresh_physical_validation=fresh_physical_validation,
                    physical_model=self.physical_model_signal(), entry_diagnostic=entry_diagnostic,
                )
            else:
                # A forced convergence-validation solve begins a new real sequence.
                self.schedule_state = self.scheduler.initialize_after_real_solve(
                    obs, solved_mode=mode, fresh_physical_validation=fresh_physical_validation,
                    physical_model=self.physical_model_signal(), entry_diagnostic=entry_diagnostic,
                )
            self.pending_schedule_evaluation = None
            core = self.schedule_state.core
            self.damping_selection = DampingSelection(
                policy_mode=str(mode_cfg.get("damping", "none")),
                selected_lambda=core.damping_lambda,
                parallel_metric=float(self.schedule_state.previous_parallel_metric or 0.0),
                metric_name="movable_cartesian_dof_rms",
                sequence_id=str(core.sequence_id),
                identity=str(core.damping_lambda_identity),
            )
        else:
            # Damping is intentionally independent of solve scheduling.
            history = getattr(owner, "canonical_force_history", None)
            state = None if history is None else history.current
            center = None if state is None else state.center
            if center is not None:
                space = active_space_for_owner(owner, center.positions)
                sequence_id = f"real_solve_state:{center.state_id}"
                self.damping_selection = select_damping_for_sequence(
                    center.forces, mode, space, _damping_config(mode_cfg), sequence_id=sequence_id
                )
        # mode-diagnostics is observational/additional-trigger only.  Standardize the already
        # completed projected-Olsen/JD result after mode-schedule has committed its own state.
        self._capture_t12_after_real_solve(owner)
        self.force_real_solve = False

    def requires_fresh_validation(self) -> bool:
        return bool(
            self.scheduler is not None
            and self.schedule_state is not None
            and self.schedule_state.core.skips_since_real_solve > 0
        )

    def apply_parallel_force_policy(self, owner, baseline_force: object) -> np.ndarray:
        mode_cfg = dict(self.options.get("mode_schedule", {}) or {})
        if str(mode_cfg.get("damping", "none")) == "none" and self.scheduler is None:
            return np.asarray(baseline_force, dtype=float)
        raw = getattr(owner, "forces0", None)
        if raw is None:
            return np.asarray(baseline_force, dtype=float)
        try:
            mode = np.asarray(owner.get_eigenmode(), dtype=float)
            curvature = float(owner.get_curvature())
            space = active_space_for_owner(owner, np.asarray(owner.get_positions(), dtype=float))
        except Exception:
            return np.asarray(baseline_force, dtype=float)
        if self.damping_selection is None:
            self.damping_selection = select_damping_for_sequence(
                raw, mode, space, _damping_config(mode_cfg), sequence_id="pre_first_real_solve"
            )
        return apply_parallel_force_policy(
            raw, mode, space, curvature=curvature, selection=self.damping_selection,
            positive_curvature_baseline=baseline_force,
        ).translation_force

    def metadata(self) -> dict[str, object]:
        ph = dict(self.options.get("physical_hessian", {}) or {})
        ms = dict(self.options.get("mode_schedule", {}) or {})
        iso = dict(self.options.get("isopotential", {}) or {})
        md = dict(self.options.get("mode_diagnostics", {}) or {})
        result = {
            "wave_b_physical_hessian_update": ph.get("update", "off"),
            "wave_b_physical_hessian_probe_updates": ph.get("probe_updates", "off"),
            "wave_b_physical_hessian_probe_batching": ph.get("probe_batching", "sequential"),
            "wave_b_physical_hessian_representation": ph.get("representation", REPRESENTATION_ONLINE_DENSE),
            "wave_b_physical_hessian_forcebank_max_pairs": ph.get("forcebank_max_pairs", 0),
            "wave_b_physical_hessian_forcebank_pair_sources": ph.get("forcebank_pair_sources", ""),
            "wave_b_physical_hessian_model_only_not_certification": 1,
            "wave_b_physical_hessian_budget_matrix_count": PHYSICAL_HESSIAN_SIMULTANEOUS_DENSE_MATRICES,
            "wave_b_mode_schedule_enabled": int(bool(ms.get("enabled", False))),
            "wave_b_mode_schedule_max_skips": ms.get("max_skips", 3),
            "wave_b_mode_schedule_skip_entry_gate": ms.get("skip_entry_gate", "none"),
            "wave_b_mode_schedule_refresh_policy": ms.get("refresh_policy", "bounded"),
            "wave_b_predictor_hook": ms.get("predictor_hook", "none"),
            "wave_b_parallel_force_damping": ms.get("damping", "none"),
            "wave_b_isopotential_estimator": iso.get("selector", "off"),
            "wave_b_last_schedule": self.last_schedule_metadata or {},
            "wave_b_last_isopotential": self.last_isopotential_metadata or {},
            "wave_b_last_physical_hessian_update": self.last_physical_hessian_update or {},
            "wave_b_last_physical_hessian_block_update": self.last_physical_hessian_block_update or {},
        }
        if self.mode_diagnostics_runtime is not None:
            result.update(self.mode_diagnostics_runtime.status_fields())
        elif bool(md.get("enabled", False)) or bool(md.get("paid_reference", False)) or bool(md.get("hindsight", False)):
            result.update({
                "mode_diagnostics_enabled": int(bool(md.get("enabled", False))),
                "mode_refresh_criterion": md.get("mode_refresh_criterion", "legacy"),
                "mode_reference_diagnostics_enabled": int(bool(md.get("paid_reference", False))),
                "mode_hindsight_enabled": int(bool(md.get("hindsight", False))),
            })
        return result
