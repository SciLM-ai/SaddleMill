"""Shared canonical-history diagnostic schema and summary helpers.

This module intentionally has no ASE dependency.  ``dimeropt.py`` can import the
field list and summary helpers without importing a particular translator.  Step
rows are populated by ``CanonicalHistoryLBFGSMinModeTranslate``; summary rows are
also populated for record-only history runs that retain a legacy translator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


REGULARIZATION_NOT_APPLICABLE = "not_applicable"

TRANSLATION_REGULARIZATION_DIAGNOSTIC_FIELDS = (
    "translation_regularization",
    "translation_regularization_applied",
    "translation_regularization_fallback",
    "translation_regularization_mu",
    "translation_regularization_iterations",
    "translation_regularized_norm",
    "translation_regularization_suppression_ratio",
    "translation_regularization_direction_cosine",
    "translation_regularization_final_maxstep_scale",
    "translation_regularization_final_max_atom_norm",
)


def regularization_diagnostic_fields(
    metrics: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return the additive translation-regularization diagnostic fields.

    A translator that does not own regularization must report the fields as
    explicitly not applicable rather than silently claiming ``off``/``0``.
    Once a regularization mode is present, the producer's applied state is
    authoritative and only genuinely unavailable quantities use the same
    explicit marker.
    """

    source = dict(metrics or {})
    if "translation_regularization" not in source:
        return {
            key: REGULARIZATION_NOT_APPLICABLE
            for key in TRANSLATION_REGULARIZATION_DIAGNOSTIC_FIELDS
        }

    result = {
        key: source.get(key, REGULARIZATION_NOT_APPLICABLE)
        for key in TRANSLATION_REGULARIZATION_DIAGNOSTIC_FIELDS
    }
    if "translation_regularization_applied" not in source:
        result["translation_regularization_applied"] = REGULARIZATION_NOT_APPLICABLE
    return result


CANONICAL_DIAGNOSTIC_FIELDS = (
    "canonical_history_schema",
    "canonical_history_enabled",
    "canonical_history_record_only",
    "canonical_rotation_reuse_enabled",
    "canonical_rotation_memory_states",
    "canonical_rotation_states_retained",
    "canonical_rotation_pairs_retained",
    "canonical_rotation_pairs_accepted_total",
    "canonical_rotation_pairs_rejected_total",
    "canonical_rotation_local_pairs_accepted_total",
    "canonical_rotation_trial_pairs_accepted_total",
    "canonical_rotation_trial_pairs_rejected_total",
    "canonical_rotation_states_dropped_total",
    "canonical_rotation_pairs_dropped_with_states_total",
    "canonical_rotation_pair_candidates",
    "canonical_rotation_pairs_used",
    "canonical_rotation_pairs_rejected_projection",
    "canonical_rotation_states_contributing",
    "canonical_rotation_local_pairs_used",
    "canonical_rotation_trial_pairs_used",
    "canonical_rotation_apply_ns",
    "canonical_rotation_apply_cumulative_ns",
    "canonical_rotation_apply_calls",
    "canonical_rotation_history_resets",
    "canonical_rotation_last_reset_reason",
    "canonical_rotation_history_source",
    "canonical_rotation_max_pairs",
    "canonical_rotation_torque_samples",
    "canonical_rotation_pairs_built",
    "canonical_rotation_pairs_degenerate",
    "canonical_rotation_stencil_build_ns",
    "canonical_rotation_pair_sources",
    "canonical_rotation_accepted_force_source",
    "canonical_rotation_derived_torque_samples",
    "canonical_rotation_pair_candidates_by_source_json",
    "canonical_rotation_pairs_built_by_source_json",
    "canonical_rotation_force_bank_disabled_current_center",
    "canonical_memory_states",
    "canonical_states_retained",
    "canonical_complete_states_retained",
    "canonical_states_created_total",
    "canonical_states_dropped_total",
    "canonical_observations_retained",
    "canonical_observations_recorded_total",
    "canonical_record_events_total",
    "canonical_observations_filtered_total",
    "canonical_observations_deduplicated_total",
    "canonical_probes_dropped_total",
    "canonical_stencils_retained",
    "canonical_complete_stencils_retained",
    "canonical_stencils_created_total",
    "canonical_stencils_dropped_total",
    "canonical_stencil_counts_by_family_json",
    "canonical_physical_exact_retained",
    "canonical_physical_cached_retained",
    "canonical_derived_retained",
    "canonical_history_bytes",
    "canonical_observation_counts_by_family_json",
    "canonical_observation_counts_by_source_json",
    "canonical_force_calls_represented",
    "canonical_algorithm_pes_calls",
    "canonical_diagnostic_pes_calls",
    "canonical_physical_total_pes_calls",
    "canonical_actual_cache_hits",
    "canonical_observation_bookkeeping_ns",
    "canonical_model_matrix_ns",
    "canonical_algorithm_ns",
    "canonical_diagnostic_ns",
    "canonical_diagnostic_promotions",
    "canonical_admission_records",
    "canonical_record_sources",
    "canonical_pair_sources",
    "canonical_pair_order",
    "canonical_projection_policy",
    "canonical_kappa_projection_policy",
    "canonical_projection_branch",
    "canonical_projection_regime",
    "canonical_projection_gamma_1",
    "canonical_projection_gamma_2",
    "canonical_projection_active_atoms",
    "canonical_projection_validation",
    "canonical_projection_validation_max_abs",
    "canonical_projection_validation_norm",
    "canonical_projection_validation_ok",
    "canonical_pair_safeguard_requested",
    "canonical_pair_safeguard",
    "canonical_reconstruction_model",
    "canonical_bfgs_update",
    "canonical_dense_bfgs_diagnostic",
    "canonical_dense_bfgs_fallback_reason",
    "canonical_curvature_floor",
    "canonical_curvature_epsilon",
    "canonical_powell_eta",
    "canonical_max_pairs",
    "canonical_cosine_threshold",
    "canonical_pair_candidates",
    "canonical_pair_candidates_after_max_pairs",
    "canonical_pairs_admissible_before_max_pairs",
    "canonical_pairs_removed_by_max_pairs",
    "canonical_pairs_used",
    "canonical_pairs_rejected_by_curvature",
    "canonical_pairs_rejected_by_cosine",
    "canonical_accepted_curvature_min",
    "canonical_accepted_curvature_median",
    "canonical_accepted_curvature_max",
    "canonical_accepted_cosine_min",
    "canonical_accepted_cosine_median",
    "canonical_accepted_cosine_max",
    "canonical_pairs_rejected",
    "canonical_pairs_damped",
    "canonical_pairs_powell_damped",
    "canonical_pair_resets",
    "canonical_candidate_pairs_by_source_json",
    "canonical_used_pairs_by_source_json",
    "canonical_rejection_reasons_json",
    "canonical_center_center_candidates",
    "canonical_center_center_used",
    "canonical_center_rotation_candidates",
    "canonical_center_rotation_used",
    "canonical_center_lanczos_candidates",
    "canonical_center_lanczos_used",
    "canonical_center_davidson_candidates",
    "canonical_center_davidson_used",
    "canonical_center_reference_hessian_candidates",
    "canonical_center_reference_hessian_used",
    "canonical_latest_pair_source",
    "canonical_latest_pair_action",
    "canonical_latest_pair_rejection_reason",
    "canonical_latest_raw_s_norm",
    "canonical_latest_raw_y_norm",
    "canonical_latest_raw_s_dot_y",
    "canonical_latest_raw_secant_curvature",
    "canonical_latest_raw_secant_cosine",
    "canonical_latest_stored_s_norm",
    "canonical_latest_stored_y_norm",
    "canonical_latest_stored_s_dot_y",
    "canonical_latest_stored_secant_curvature",
    "canonical_latest_stored_secant_cosine",
    "canonical_worst_raw_s_dot_y",
    "canonical_worst_stored_s_dot_y",
    "canonical_h0_inverse_scale",
    "canonical_direction_norm",
    "canonical_direction_max_atom_norm",
    "canonical_direction_force_cosine",
    "canonical_direction_fallback",
    *TRANSLATION_REGULARIZATION_DIAGNOSTIC_FIELDS,
    "translation_trial_step_method",
    "translation_trial_force_calls",
    "translation_trial_interpolation_used",
    "translation_trial_current_directional_force",
    "translation_trial_trial_directional_force",
    "translation_trial_fraction",
    "translation_trial_bounded_displacement_norm",
    "translation_trial_chosen_displacement_norm",
    "canonical_selection_ns",
    "canonical_projection_ns",
    "canonical_pair_build_ns",
    "canonical_safeguard_ns",
    "canonical_two_loop_ns",
    "canonical_dense_bfgs_ns",
    "canonical_dense_vs_lbfgs_cosine",
    "canonical_dense_vs_lbfgs_norm_ratio",
    "canonical_dense_vs_lbfgs_relative_difference",
    "canonical_dense_min_eigenvalue",
    "canonical_dense_max_eigenvalue",
    "canonical_dense_negative_eigenvalues",
    "canonical_dense_condition_number",
    "canonical_dense_log10_condition",
    "canonical_dense_secant_residual_latest",
    "canonical_dense_secant_residual_median",
    "canonical_dense_secant_residual_max",
    "canonical_dense_multisecant_blocks_applied",
    "canonical_dense_multisecant_pairs_skipped_rank",
    "canonical_dense_multisecant_pairs_skipped_curvature",
    "canonical_dense_multisecant_blocks_requested",
    "canonical_dense_multisecant_pairs_skipped_invalid",
    "canonical_dense_multisecant_blocks_rejected_update",
    "canonical_dense_multisecant_block_size_median",
    "canonical_dense_multisecant_block_size_max",
    "canonical_dense_multisecant_block_kept_median",
    "canonical_dense_multisecant_y_symmetrization_relative_median",
    "canonical_dense_multisecant_y_symmetrization_relative_max",
    "canonical_dense_multisecant_sty_asymmetry_relative_median",
    "canonical_dense_multisecant_sty_asymmetry_relative_max",
    "canonical_dense_multisecant_sts_condition_median",
    "canonical_dense_multisecant_sts_condition_max",
    "canonical_dense_multisecant_yts_condition_median",
    "canonical_dense_multisecant_yts_condition_max",
    "canonical_dense_multisecant_stbs_condition_median",
    "canonical_dense_multisecant_stbs_condition_max",
    "canonical_dense_multisecant_sty_original_min_eigenvalue_median",
    "canonical_dense_multisecant_sty_original_min_eigenvalue_max",
    "canonical_dense_multisecant_sty_adjusted_min_eigenvalue_median",
    "canonical_dense_multisecant_sty_adjusted_min_eigenvalue_max",
    "canonical_dense_multisecant_residual_original_y_median",
    "canonical_dense_multisecant_residual_original_y_max",
    "canonical_dense_multisecant_residual_adjusted_y_median",
    "canonical_dense_multisecant_residual_adjusted_y_max",
    "canonical_dense_pairs_requested",
    "canonical_dense_pairs_applied",
    "canonical_dense_invalid_reason",
    "canonical_dense_reconstruction_ns",
    "canonical_postprocess_ns",
    "canonical_reconstruction_total_ns",
    "canonical_shadow_reconstruction_ns",
    "canonical_recording_ns_since_previous_step",
    "canonical_recording_ns_total",
    "canonical_overhead_ns_this_step",
    "canonical_reconstruction_calls",
    "canonical_reconstruction_cumulative_ns",
    "canonical_reconstruction_mean_ns",
    "canonical_reconstruction_median_ns",
    "canonical_reconstruction_p95_ns",
    "canonical_reconstruction_max_ns",
    "canonical_shadow_cumulative_ns",
    "canonical_recording_cumulative_ns",
    "canonical_overhead_cumulative_ns",
    "canonical_overhead_mean_ns",
    "canonical_optimizer_elapsed_ns",
    "canonical_overhead_fraction_of_optimizer_wall",
    "canonical_direction_fallbacks_total",
    "canonical_peak_states_retained",
    "canonical_peak_observations_retained",
    "canonical_peak_history_bytes",
    "canonical_total_pair_candidates",
    "canonical_total_pairs_used",
    "canonical_total_pairs_rejected",
    "canonical_total_pairs_damped",
    "canonical_reuse_ablation",
    "canonical_reuse_direction_cosine",
    "canonical_reuse_direction_delta_norm",
    "canonical_reuse_direction_norm_ratio",
    "canonical_reuse_pairs_added",
    "canonical_reuse_pairs_survived",
    "canonical_center_only_pairs_used",
    "canonical_center_only_reconstruction_ns",
    "canonical_history_dump_path",
    "canonical_history_dump_error",
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)


def history_summary_fields(
    dimeratoms,
    *,
    record_only: bool,
    history_options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return additive CSV fields for any history-enabled minimum-mode object."""

    history = getattr(dimeratoms, "canonical_force_history", None)
    if history is None:
        return {}
    summary = dict(history.summary())
    options = dict(history_options or getattr(dimeratoms, "canonical_history_options", {}) or {})
    evaluator = getattr(dimeratoms, "physical_force_evaluator", None)
    recording_total = int(getattr(evaluator, "recording_ns_total", 0))
    accounting = dict(summary.get("force_accounting", {}) or {})
    return {
        "canonical_history_schema": summary.get("schema", ""),
        "canonical_history_enabled": 1,
        "canonical_history_record_only": int(bool(record_only)),
        "canonical_rotation_reuse_enabled": int(bool(options.get("rotation_reuse", False))),
        "canonical_rotation_history_source": str(
            options.get("rotation_history_source", "force_bank")
        ),
        "canonical_rotation_max_pairs": options.get("rotation_max_pairs", 0),
        "canonical_memory_states": int(getattr(history, "memory_states", 0)),
        "canonical_states_retained": summary.get("states_retained", 0),
        "canonical_complete_states_retained": summary.get(
            "complete_states_retained", 0
        ),
        "canonical_states_created_total": summary.get("states_created_total", 0),
        "canonical_states_dropped_total": summary.get("states_dropped_total", 0),
        "canonical_observations_retained": summary.get("observations_retained", 0),
        "canonical_observations_recorded_total": summary.get(
            "observations_recorded_total", 0
        ),
        "canonical_record_events_total": summary.get("record_events_total", 0),
        "canonical_observations_filtered_total": summary.get(
            "observations_filtered_total", 0
        ),
        "canonical_observations_deduplicated_total": summary.get(
            "observations_deduplicated_total", 0
        ),
        "canonical_probes_dropped_total": summary.get("probes_dropped_total", 0),
        "canonical_stencils_retained": summary.get("stencils_retained", 0),
        "canonical_complete_stencils_retained": summary.get(
            "complete_stencils_retained", 0
        ),
        "canonical_stencils_created_total": summary.get(
            "stencils_created_total", 0
        ),
        "canonical_stencils_dropped_total": summary.get(
            "stencils_dropped_total", 0
        ),
        "canonical_stencil_counts_by_family_json": _json(
            summary.get("stencil_counts_by_family", {})
        ),
        "canonical_physical_exact_retained": summary.get(
            "physical_exact_retained", 0
        ),
        "canonical_physical_cached_retained": summary.get(
            "physical_cached_retained", 0
        ),
        "canonical_derived_retained": summary.get("derived_retained", 0),
        "canonical_history_bytes": summary.get("history_bytes", 0),
        "canonical_observation_counts_by_family_json": _json(
            summary.get("observation_counts_by_family", {})
        ),
        "canonical_observation_counts_by_source_json": _json(
            summary.get("observation_counts_by_source", {})
        ),
        "canonical_force_calls_represented": summary.get(
            "force_calls_represented", 0
        ),
        "canonical_algorithm_pes_calls": accounting.get("algorithm_pes_calls", 0),
        "canonical_diagnostic_pes_calls": accounting.get("diagnostic_pes_calls", 0),
        "canonical_physical_total_pes_calls": accounting.get("physical_total_pes_calls", 0),
        "canonical_actual_cache_hits": accounting.get("actual_cache_hits", 0),
        "canonical_observation_bookkeeping_ns": accounting.get("observation_bookkeeping_ns", 0),
        "canonical_model_matrix_ns": accounting.get("model_matrix_ns", 0),
        "canonical_algorithm_ns": accounting.get("algorithm_ns", 0),
        "canonical_diagnostic_ns": accounting.get("diagnostic_ns", 0),
        "canonical_diagnostic_promotions": accounting.get("diagnostic_promotions", 0),
        "canonical_admission_records": summary.get("admission_records", 0),
        "canonical_record_sources": " ".join(
            sorted(str(item) for item in getattr(history, "record_sources", ()))
        ),
        "canonical_pair_sources": str(options.get("pair_sources", "")),
        "canonical_pair_order": str(options.get("pair_order", "acquisition")),
        "canonical_projection_policy": str(options.get("projection_policy", "")),
        "canonical_kappa_projection_policy": str(
            options.get("kappa_projection_policy", "")
        ),
        "canonical_pair_safeguard_requested": str(
            options.get("pair_safeguard", "")
        ),
        "canonical_pair_safeguard": str(
            options.get("pair_safeguard_resolved", options.get("pair_safeguard", ""))
        ),
        "canonical_curvature_floor": options.get("curvature_floor", ""),
        "canonical_curvature_epsilon": options.get("curvature_epsilon", ""),
        "canonical_powell_eta": options.get("powell_eta", ""),
        "canonical_max_pairs": options.get("max_pairs", ""),
        "canonical_recording_ns_total": recording_total,
        "canonical_recording_cumulative_ns": recording_total,
        "canonical_overhead_cumulative_ns": recording_total,
    }


def dump_history_if_requested(
    dimeratoms,
    *,
    directory: str | Path,
    trace_id: object,
    src_index: object,
    attempt_id: object,
) -> tuple[str, str]:
    """Write the optional raw-history NPZ and return ``(path, error)``.

    Dump failure is diagnostic-only: callers should log the error but must not
    convert a scientifically completed attempt into an execution failure.
    """

    history = getattr(dimeratoms, "canonical_force_history", None)
    options = dict(getattr(dimeratoms, "canonical_history_options", {}) or {})
    if history is None or not bool(options.get("raw_dump", False)):
        return "", ""
    try:
        out_dir = Path(str(options.get("raw_dump_directory", directory)))
        if not out_dir.is_absolute():
            out_dir = Path(directory) / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_trace = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(trace_id))
        name = f"history_{safe_trace}_src{src_index}_attempt{attempt_id}.npz"
        path = history.dump_npz(out_dir / name)
        return str(path), ""
    except Exception as exc:  # pragma: no cover - exercised by integration paths
        return "", f"{type(exc).__name__}: {exc}"


__all__ = [
    "CANONICAL_DIAGNOSTIC_FIELDS",
    "REGULARIZATION_NOT_APPLICABLE",
    "TRANSLATION_REGULARIZATION_DIAGNOSTIC_FIELDS",
    "dump_history_if_requested",
    "history_summary_fields",
    "regularization_diagnostic_fields",
]
