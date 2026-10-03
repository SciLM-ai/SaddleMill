import json
import os
import sys
import csv
import math
import time
import tempfile
from collections import deque
import traceback
import random
import zipfile
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.io import Trajectory
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
from ase.calculators.singlepoint import SinglePointCalculator
from saddlemill.dimertools.structure_edit import get_attempts
from saddlemill.rng_keyed import (
    RNG_SCHEME_ATTEMPT_KEYED_V1,
    RNG_SCHEME_LEGACY,
    build_structure_rng_factory,
)
from saddlemill.config import normalize_dimer_method_config
from saddlemill.tools import (backup_flux_logs, get_task_name, resolve_vasp_calc,
                              remove_vasp_heavies, finalize_if_vasp_interactive,
                              archive_and_clear_temp_files)
from saddlemill.dimertools.canonical_diagnostics import (
    CANONICAL_DIAGNOSTIC_FIELDS,
    dump_history_if_requested,
    history_summary_fields,
)
from saddlemill.dimertools.lbfgs_state_dump import write_state_dump
from saddlemill.diagnostics_io import BufferedCSVAppender, BufferedJSONLAppender
from saddlemill.attempt_metrics import geometry_sha256, write_terminal_attempt_metrics
from saddlemill.dimertools.dimer_factory import (
    create_rfo_translation_runtime,
    history_options_from_config,
    partitioned_options_from_config,
)
from saddlemill.dimer_lifecycle import (
    DimerRunContext,
    apply_continuation,
    apply_generated_metadata,
    archive_temp_files_timed,
    configure_temp_files,
    run_optimizer_timed,
    start_attempt,
)


class StopRun(Exception):
    pass


MODE_DIAGNOSTIC_FIELDS = [
    "trace_id",
    "trace_start_unix_ns",
    "src_index",
    "rank",
    "attempt_id",
    "selected_index",
    "reaction_type",
    "translation_step",
    "force_calls",
    "curvature",
    "participation_ratio",
    "angle_from_previous_deg",
    "angle_from_initial_deg",
    "pre_post_angle_w5_deg",
    "pre_coherence_w5",
    "post_coherence_w5",
    "atom_participation_overlap_w5",
    "pre_post_angle_w10_deg",
    "pre_coherence_w10",
    "post_coherence_w10",
    "atom_participation_overlap_w10",
]

REACTION_DIAGNOSTIC_FIELDS = [
    "src_index",
    "rank",
    "attempt_id",
    "selected_index",
    "configured_reaction_type",
    "initial_reaction_type",
    "final_reaction_type",
    "converged",
    "n_force_calls",
    "status",
    "classification_source",
    "classification_confidence",
]

ATTEMPT_TIMING_FIELDS = [
    "src_index",
    "rank",
    "attempt_id",
    "selected_index",
    "configured_reaction_type",
    "initial_reaction_type",
    "final_reaction_type",
    "saddle_engine",
    "rotation_optimizer",
    "translation_optimizer",
    "attempt_start_unix_ns",
    "attempt_end_unix_ns",
    "attempt_wall_seconds",
    "attempt_excluding_archive_seconds",
    "optimizer_wall_seconds",
    "archive_wall_seconds",
    "other_wall_seconds",
    "n_force_calls",
    "sella_hessian_calls",
    "sella_hessian_seconds",
    "converged",
    "status",
]

OPTIMIZER_DIAGNOSTIC_FIELDS = [
    "record_type",
    "trace_id",
    "trace_start_unix_ns",
    "src_index",
    "rank",
    "attempt_id",
    "selected_index",
    "reaction_type",

    "diagnostic_serial",

    "accepted_translation_step",
    "translation_algorithm",
    "hybrid_state",
    "hybrid_switch_event",

    "projected_fmax",
    "real_fmax",

    "curvature",
    "translation_regime",
    "translation_state_key",

    "step_norm",
    "step_clipped",
    "direction_alignment",
    "translation_raw_step_norm",
    "translation_raw_step_max",
    "translation_actual_step_norm",
    "translation_actual_step_max",
    "translation_maxstep",
    "translation_maxstep_rescaled",
    "translation_clip_scale",
    "translation_damping",
    "translation_applied_scale",
    "hybrid_history_pairs_at_switch",
    "hybrid_warm_start_history",

    "rotation_optimizer",
    "rotation_steps",
    "rotation_lbfgs_history_size",
    "rotation_lbfgs_pairs_accepted",
    "rotation_lbfgs_pairs_rejected",
    "rotation_lbfgs_resets",
    "rotation_lbfgs_fallbacks",
    "rotation_lbfgs_memory",
    "rotation_lbfgs_initial_hessian",
    "rotation_lbfgs_initial_inverse_hessian_scale",
    "rotation_lbfgs_latest_pair_accepted",
    "rotation_lbfgs_latest_s_norm",
    "rotation_lbfgs_latest_y_norm",
    "rotation_lbfgs_latest_s_dot_y",
    "rotation_lbfgs_latest_secant_curvature",
    "rotation_lbfgs_latest_secant_cosine",
    "rotation_lbfgs_latest_force_change_norm",
    "rotation_lbfgs_geometry",
    "rotation_lbfgs_step_method",
    "rotation_lbfgs_history_source",
    "rotation_lbfgs_curvature_guard",
    "rotation_lbfgs_curvature_floor",
    "rotation_lbfgs_powell_eta",
    "rotation_lbfgs_first_angle_degrees",
    "rotation_lbfgs_max_angle_degrees",
    "rotation_lbfgs_fixed_first_steps",
    "rotation_lbfgs_direct_steps",
    "rotation_lbfgs_fourier_steps",
    "rotation_lbfgs_step_clips",
    "rotation_lbfgs_requested_angle_sum",
    "rotation_lbfgs_actual_angle_sum",
    "rotation_lbfgs_rotation_trace_json",
    "rotation_lbfgs_cosine_threshold",
    "rotation_lbfgs_pair_candidates",
    "rotation_lbfgs_pair_candidates_after_max_pairs",
    "rotation_lbfgs_pairs_admissible_before_max_pairs",
    "rotation_lbfgs_pairs_removed_by_max_pairs",
    "rotation_lbfgs_pairs_used",
    "rotation_lbfgs_pairs_rejected_curvature",
    "rotation_lbfgs_pairs_rejected_cosine",
    "rotation_lbfgs_accepted_curvature_min",
    "rotation_lbfgs_accepted_curvature_median",
    "rotation_lbfgs_accepted_curvature_max",
    "rotation_lbfgs_accepted_cosine_min",
    "rotation_lbfgs_accepted_cosine_median",
    "rotation_lbfgs_accepted_cosine_max",
    "rotation_lbfgs_pairs_damped",
    "rotation_lbfgs_pairs_powell_damped",
    "rotation_lbfgs_pairs_rejected_transport",
    "rotation_lbfgs_pairs_rejected_json",
    "rotation_lbfgs_latest_guard_action",
    "rotation_lbfgs_latest_raw_s_dot_y",
    "rotation_lbfgs_latest_stored_s_dot_y",
    "rotation_lbfgs_latest_powell_theta",
    "rotation_lbfgs_latest_s_dot_Bs",
    "rotation_lbfgs_reconstruction_ns",
    "rotation_lbfgs_reconstruction_cumulative_ns",
    "rotation_lbfgs_reconstruction_model",
    "rotation_lbfgs_bfgs_update",
    "rotation_dense_bfgs_diagnostic",
    "rotation_lbfgs_raw_direction_norm",
    "rotation_lbfgs_raw_direction_max_atom_norm",
    "rotation_dense_vs_lbfgs_cosine",
    "rotation_dense_vs_lbfgs_norm_ratio",
    "rotation_dense_vs_lbfgs_relative_difference",
    "rotation_dense_min_eigenvalue",
    "rotation_dense_max_eigenvalue",
    "rotation_dense_negative_eigenvalues",
    "rotation_dense_condition_number",
    "rotation_dense_log10_condition",
    "rotation_dense_secant_residual_latest",
    "rotation_dense_secant_residual_median",
    "rotation_dense_secant_residual_max",
    "rotation_dense_multisecant_blocks_applied",
    "rotation_dense_multisecant_pairs_skipped_rank",
    "rotation_dense_multisecant_pairs_skipped_curvature",
    "rotation_dense_multisecant_blocks_requested",
    "rotation_dense_multisecant_pairs_skipped_invalid",
    "rotation_dense_multisecant_blocks_rejected_update",
    "rotation_dense_pairs_requested",
    "rotation_dense_pairs_applied",
    "rotation_dense_invalid_reason",
    "rotation_dense_reconstruction_ns",

    "mode_schedule_decision",
    "mode_schedule_require_real_solve",
    "mode_schedule_skipped_scheduled_solve",
    "mode_predictor_selector",
    "mode_predictor_status",
    "mode_predictor_disposition",
    "mode_predictor_requested_angle_radians",
    "mode_predictor_requested_angle_degrees",
    "mode_predictor_accepted_angle_radians",
    "mode_predictor_accepted_angle_degrees",
    "mode_predictor_capped",
    "mode_predictor_raw_tangent_norm",
    "mode_predictor_original_absolute_alignment",
    "mode_predictor_displacement_norm",
    "mode_predictor_gradient_change_norm",
    "mode_predictor_secant_action_norm",
    "mode_predictor_effective_angular_step_scale",
    "mode_predictor_preconditioned",
    "mode_predictor_fallback",
    "mode_predictor_prediction_cost_pes_calls",
    "mode_predictor_forcebank_torque_samples",
    "mode_predictor_forcebank_pair_sources",
    "mode_predictor_forcebank_accepted_force_source",
    "mode_predictor_forcebank_trial_pairs_future_only",
    "mode_predictor_forcebank_dynamic_h0",
    "mode_predictor_forcebank_pair_candidates",
    "mode_predictor_forcebank_pairs_built",
    "mode_predictor_forcebank_pairs_degenerate",
    "mode_predictor_forcebank_states_contributing",
    "mode_predictor_forcebank_build_ns",
    "mode_predictor_lbfgs_pairs_used",
    "mode_predictor_lbfgs_pairs_admissible_before_max_pairs",
    "mode_predictor_lbfgs_pairs_rejected_transport",
    "mode_predictor_lbfgs_pairs_rejected_curvature",
    "mode_predictor_lbfgs_pairs_rejected_cosine",
    "mode_predictor_lbfgs_pairs_damped",
    "mode_predictor_lbfgs_h0_inverse_scale",
    "mode_predictor_lbfgs_raw_direction_norm",
    "mode_predictor_lbfgs_direction_norm",
    "mode_predictor_lbfgs_direction_surrogate_cosine",
    "mode_predictor_lbfgs_apply_ns",

    "rotation_lbfgs_transport_policy",
    "rotation_lbfgs_pair_sources",
    "rotation_lbfgs_accepted_force_source",
    "rotation_lbfgs_extrapolated_accepted_samples_recorded",
    "rotation_lbfgs_trial_pairs_future_only",
    "rotation_lbfgs_transport_s_norm_ratio_median",
    "rotation_lbfgs_transport_s_norm_ratio_min",
    "rotation_lbfgs_transport_s_norm_ratio_max",
    "rotation_lbfgs_transport_y_norm_ratio_median",
    "rotation_lbfgs_transport_y_norm_ratio_min",
    "rotation_lbfgs_transport_y_norm_ratio_max",
    "rotation_lbfgs_transport_pair_transform_count_max",
    "rotation_lbfgs_transport_pair_path_angle_max",
    "rotation_lbfgs_sequential_history_pairs",
    "rotation_lbfgs_sequential_history_advance_steps",
    "rotation_lbfgs_sequential_history_state_sync_steps",
    "rotation_lbfgs_sequential_history_vectors_transformed",
    "rotation_lbfgs_sequential_history_s_norm_step_ratio_median",
    "rotation_lbfgs_sequential_history_s_norm_step_ratio_min",
    "rotation_lbfgs_sequential_history_s_norm_step_ratio_max",
    "rotation_lbfgs_sequential_history_y_norm_step_ratio_median",
    "rotation_lbfgs_sequential_history_y_norm_step_ratio_min",
    "rotation_lbfgs_sequential_history_y_norm_step_ratio_max",
    "rotation_dense_multisecant_block_size_median",
    "rotation_dense_multisecant_block_size_max",
    "rotation_dense_multisecant_block_kept_median",
    "rotation_dense_multisecant_y_symmetrization_relative_median",
    "rotation_dense_multisecant_y_symmetrization_relative_max",
    "rotation_dense_multisecant_sty_asymmetry_relative_median",
    "rotation_dense_multisecant_sty_asymmetry_relative_max",
    "rotation_dense_multisecant_sts_condition_median",
    "rotation_dense_multisecant_sts_condition_max",
    "rotation_dense_multisecant_yts_condition_median",
    "rotation_dense_multisecant_yts_condition_max",
    "rotation_dense_multisecant_stbs_condition_median",
    "rotation_dense_multisecant_stbs_condition_max",
    "rotation_dense_multisecant_sty_original_min_eigenvalue_median",
    "rotation_dense_multisecant_sty_original_min_eigenvalue_max",
    "rotation_dense_multisecant_sty_adjusted_min_eigenvalue_median",
    "rotation_dense_multisecant_sty_adjusted_min_eigenvalue_max",
    "rotation_dense_multisecant_residual_original_y_median",
    "rotation_dense_multisecant_residual_original_y_max",
    "rotation_dense_multisecant_residual_adjusted_y_median",
    "rotation_dense_multisecant_residual_adjusted_y_max",
    "kappa_rotation_optimizer",
    "kappa_rotation_steps",
    "kappa_rotation_lbfgs_pairs_accepted",
    "kappa_rotation_lbfgs_pairs_rejected",
    "kappa_rotation_lbfgs_memory",
    "kappa_rotation_lbfgs_initial_hessian",
    "kappa_rotation_lbfgs_initial_inverse_hessian_scale",
    "kappa_rotation_lbfgs_latest_pair_accepted",
    "kappa_rotation_lbfgs_latest_s_norm",
    "kappa_rotation_lbfgs_latest_y_norm",
    "kappa_rotation_lbfgs_latest_s_dot_y",
    "kappa_rotation_lbfgs_latest_secant_curvature",
    "kappa_rotation_lbfgs_latest_secant_cosine",
    "kappa_rotation_lbfgs_latest_force_change_norm",

    "translation_lbfgs_history_size",
    "translation_lbfgs_pairs_accepted_total",
    "translation_lbfgs_pairs_rejected_total",
    "translation_lbfgs_pairs_damped_total",
    "translation_lbfgs_pairs_powell_damped_total",
    "translation_lbfgs_curvature_guard",
    "translation_lbfgs_curvature_floor",
    "translation_lbfgs_powell_eta",
    "translation_lbfgs_latest_guard_action",
    "translation_lbfgs_latest_powell_theta",
    "translation_lbfgs_latest_s_dot_Bs",
    "translation_lbfgs_worst_sy",
    "translation_lbfgs_resets",
    "translation_lbfgs_last_reset_reason",
    "translation_lbfgs_alpha",
    "translation_lbfgs_initial_inverse_hessian_scale",
    "translation_lbfgs_memory",
    "translation_lbfgs_latest_s_norm",
    "translation_lbfgs_latest_y_norm",
    "translation_lbfgs_latest_s_dot_y",
    "translation_lbfgs_latest_secant_curvature",
    "translation_lbfgs_latest_secant_cosine",
    "translation_lbfgs_latest_force_change_norm",
    "translation_lbfgs_latest_pair_damped",
    "translation_lbfgs_latest_raw_s_dot_y",
    "translation_lbfgs_latest_raw_secant_curvature",
    "translation_lbfgs_latest_stored_s_dot_y",
    "translation_lbfgs_latest_stored_secant_curvature",
    "translation_dense_bfgs_diagnostic",
    "translation_lbfgs_raw_direction_norm",
    "translation_lbfgs_raw_direction_max_atom_norm",
    "translation_dense_vs_lbfgs_cosine",
    "translation_dense_vs_lbfgs_norm_ratio",
    "translation_dense_vs_lbfgs_relative_difference",
    "translation_dense_min_eigenvalue",
    "translation_dense_max_eigenvalue",
    "translation_dense_negative_eigenvalues",
    "translation_dense_condition_number",
    "translation_dense_log10_condition",
    "translation_dense_secant_residual_latest",
    "translation_dense_secant_residual_median",
    "translation_dense_secant_residual_max",
    "translation_dense_pairs_requested",
    "translation_dense_pairs_applied",
    "translation_dense_invalid_reason",
    "translation_dense_reconstruction_ns",

    "partitioned_mode_hv_available",
    "partitioned_mode_hv_source",
    "partitioned_mode_hv_family",
    "partitioned_mode_hv_stencil_id",
    "partitioned_mode_hv_stencil_scheme",
    "partitioned_mode_hv_fd_displacement",
    "partitioned_mode_hv_probe_count",
    "partitioned_mode_hv_direction_abs_overlap",
    "partitioned_mode_hv_endpoint_observation_ids_json",
    "partitioned_mode_hv_norm",
    "partitioned_mode_qhv_norm",
    "partitioned_mode_qhv_fraction",
    "partitioned_mode_qhv_to_abs_curvature",
    "partitioned_mode_rayleigh_curvature",
    "partitioned_mode_curvature_reported",
    "partitioned_mode_curvature_hv",
    "partitioned_mode_curvature_delta",
    "partitioned_mode_hv_additional_pes_calls",
    "partitioned_mode_hv_unavailable_reason",

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
    "translation_trust_shift",
    "translation_trust_radius",
    "translation_trust_unshifted_boundary_norm",
    "translation_trust_regularized_boundary_norm",
    "translation_trust_regularized_direction_norm",
    "translation_trust_direction_cosine_unshifted",
    "translation_trust_shifted_condition_number",
    "translation_trust_iterations",
    "translation_trust_shift_expansions",
    "translation_trust_solve_ns",
    "force_calls_step_entry",
    "force_calls_center_and_rotation",
    "force_calls_translation_trial",
    "force_calls_cumulative_after_step",

    "final_force_calls",
    "force_calls_final_evaluation",
    "final_translation_steps",
    "converged",
    "status",
]
OPTIMIZER_DIAGNOSTIC_FIELDS = list(dict.fromkeys(
    list(OPTIMIZER_DIAGNOSTIC_FIELDS) + list(CANONICAL_DIAGNOSTIC_FIELDS)
))

def _append_csv_row(path, fieldnames, row):
    """Append a row, expanding older additive diagnostic headers safely."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            old_fields = list(reader.fieldnames or [])
            if old_fields != list(fieldnames):
                if not old_fields or not set(old_fields).issubset(set(fieldnames)):
                    raise ValueError(
                        f"Refusing incompatible diagnostic CSV schema migration for {path}: "
                        f"old={old_fields}, new={list(fieldnames)}"
                    )
                rows = list(reader)
                directory = os.path.dirname(path) or "."
                with tempfile.NamedTemporaryFile(
                    mode="w", newline="", dir=directory, delete=False
                ) as temp_handle:
                    temp_path = temp_handle.name
                    writer = csv.DictWriter(temp_handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for old_row in rows:
                        writer.writerow({key: old_row.get(key, "") for key in fieldnames})
                os.replace(temp_path, path)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _remove_csv_rows(path, match):
    """Atomically remove prior diagnostic rows for one active attempt.

    SaddleMill's normal resume cleanup does not know about these new diagnostic
    CSVs. Removing only the matching diagnostic rows keeps them aligned with
    the active status/output entry when an attempt is rerun.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or not all(key in fieldnames for key in match):
            return
        rows = []
        found = False
        for row in reader:
            is_match = all(
                str(row.get(key, "")) == str(value)
                for key, value in match.items()
            )
            if is_match:
                found = True
            else:
                rows.append(row)

    if not found:
        return

    directory = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=directory, delete=False
    ) as handle:
        temp_path = handle.name
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def _split_reaction_types(config_dict):
    value = config_dict.get("ourDimer", {}).get("reaction_types", [])
    if isinstance(value, str):
        return value.split()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _configured_attempt_reaction_types(config_dict):
    """Return the configured base reaction type for each attempt index."""
    reaction_types = _split_reaction_types(config_dict)
    if "initial_guess" in reaction_types:
        return ["initial_guess"]

    counts_value = config_dict.get("ourDimer", {}).get(
        "num_attempts_per_type", 1
    )
    if isinstance(counts_value, str):
        parts = counts_value.split()
        counts = [int(item) for item in parts] if len(parts) > 1 else [int(parts[0])]
    elif isinstance(counts_value, (list, tuple)):
        counts = [int(item) for item in counts_value]
    else:
        counts = [int(counts_value)]

    if len(counts) == 1:
        counts *= len(reaction_types)
    if len(counts) != len(reaction_types):
        return []

    configured = []
    for reaction_type, count in zip(reaction_types, counts):
        configured.extend([reaction_type] * count)
    return configured


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return value if np.isfinite(value) else ""


def _normalize_mode(mode, free_indices):
    arr = np.asarray(mode, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected eigenmode shape (N, 3), got {arr.shape}")
    arr = arr[np.asarray(free_indices, dtype=int)]
    norm = float(np.linalg.norm(arr))
    if norm < 1e-14:
        raise ValueError("Eigenmode norm over free atoms is approximately zero")
    return arr / norm


def _mode_angle_deg(mode_a, mode_b):
    """Projective angle: mode and -mode are treated as identical."""
    dot = float(np.vdot(mode_a.ravel(), mode_b.ravel()).real)
    dot = float(np.clip(abs(dot), 0.0, 1.0))
    return math.degrees(math.acos(dot))


def _sign_align_modes(modes):
    """Sequentially sign-align a mode sequence without changing directions."""
    aligned = [np.array(modes[0], copy=True)]
    for mode in modes[1:]:
        candidate = np.array(mode, copy=True)
        if np.vdot(aligned[-1].ravel(), candidate.ravel()).real < 0.0:
            candidate *= -1.0
        aligned.append(candidate)
    return aligned


def _window_mode_statistics(history, window):
    history = list(history)
    if len(history) < 2 * window:
        return ("", "", "", "")

    pre = _sign_align_modes(history[-2 * window:-window])
    post = _sign_align_modes(history[-window:])

    pre_sum = np.sum(pre, axis=0)
    post_sum = np.sum(post, axis=0)
    pre_norm = float(np.linalg.norm(pre_sum))
    post_norm = float(np.linalg.norm(post_sum))
    if pre_norm < 1e-14 or post_norm < 1e-14:
        return ("", "", "", "")

    pre_mean = pre_sum / pre_norm
    post_mean = post_sum / post_norm
    angle = _mode_angle_deg(pre_mean, post_mean)
    pre_coherence = pre_norm / window
    post_coherence = post_norm / window

    pre_weights = np.mean(
        [np.sum(mode * mode, axis=1) for mode in pre], axis=0
    )
    post_weights = np.mean(
        [np.sum(mode * mode, axis=1) for mode in post], axis=0
    )
    pre_weights /= pre_weights.sum()
    post_weights /= post_weights.sum()
    atom_overlap = float(np.sum(np.sqrt(pre_weights * post_weights)))

    return angle, pre_coherence, post_coherence, atom_overlap


class ModeDiagnosticRecorder:
    """Write one sign-invariant eigenmode diagnostic row per translation state.

    ASE calls optimizer observers once at step 0 after the initial force/mode
    evaluation, and then after every translation step. Therefore the first row
    is the rotated mode at the initial displaced geometry, not the unrefined
    random seed direction.
    """

    def __init__(self, path, src_index, rank, attempt_id, selected_index,
                 reaction_type, d_atoms, dim_rlx, free_indices):
        self.path = path
        self.src_index = src_index
        self.rank = rank
        self.attempt_id = attempt_id
        self.selected_index = selected_index
        self.reaction_type = reaction_type
        self.d_atoms = d_atoms
        self.dim_rlx = dim_rlx
        self.free_indices = list(free_indices)
        self.trace_start_unix_ns = time.time_ns()
        self.trace_id = (
            f"{src_index}-{attempt_id}-{rank}-"
            f"{os.getpid()}-{self.trace_start_unix_ns}"
        )
        self.history = deque(maxlen=20)
        self.initial_mode = None
        self.disabled = False
        self.warned = False
        self._writer = BufferedCSVAppender(self.path, MODE_DIAGNOSTIC_FIELDS)

    def close(self):
        self._writer.close()

    def __call__(self):
        if self.disabled:
            return
        try:
            mode = _normalize_mode(
                self.d_atoms.get_eigenmode(), self.free_indices
            )
            if self.initial_mode is None:
                self.initial_mode = np.array(mode, copy=True)

            previous_angle = (
                _mode_angle_deg(self.history[-1], mode)
                if self.history else 0.0
            )
            initial_angle = _mode_angle_deg(self.initial_mode, mode)
            self.history.append(np.array(mode, copy=True))

            atom_weights = np.sum(mode * mode, axis=1)
            participation_ratio = 1.0 / (
                len(atom_weights) * np.sum(atom_weights * atom_weights)
            )
            w5 = _window_mode_statistics(self.history, 5)
            w10 = _window_mode_statistics(self.history, 10)

            try:
                curvature = self.d_atoms.get_curvature()
            except Exception:
                curvature = ""
            try:
                force_calls = self.d_atoms.control.get_counter("forcecalls")
            except Exception:
                force_calls = ""

            self._writer.append({
                    "trace_id": self.trace_id,
                    "trace_start_unix_ns": self.trace_start_unix_ns,
                    "src_index": self.src_index,
                    "rank": self.rank,
                    "attempt_id": self.attempt_id,
                    "selected_index": self.selected_index,
                    "reaction_type": self.reaction_type,
                    "translation_step": self.dim_rlx.nsteps,
                    "force_calls": force_calls,
                    "curvature": _safe_float(curvature),
                    "participation_ratio": participation_ratio,
                    "angle_from_previous_deg": previous_angle,
                    "angle_from_initial_deg": initial_angle,
                    "pre_post_angle_w5_deg": w5[0],
                    "pre_coherence_w5": w5[1],
                    "post_coherence_w5": w5[2],
                    "atom_participation_overlap_w5": w5[3],
                    "pre_post_angle_w10_deg": w10[0],
                    "pre_coherence_w10": w10[1],
                    "post_coherence_w10": w10[2],
                    "atom_participation_overlap_w10": w10[3],
                })
        except Exception as exc:
            self.disabled = True
            if not self.warned:
                self.warned = True
                print(
                    f"Mode diagnostics disabled for structure {self.src_index}, "
                    f"attempt {self.attempt_id}: {exc}",
                    flush=True,
                )



class OptimizerDiagnosticRecorder:
    """Write one row per accepted translation and one final summary row."""

    def __init__(
        self, path, mode_recorder, d_atoms, dim_rlx, qn_path=None,
        state_dump_dir=None,
    ):
        self.path = path
        self.qn_path = qn_path
        self.state_dump_dir = state_dump_dir

        self.mode_recorder = mode_recorder
        self.d_atoms = d_atoms
        self.dim_rlx = dim_rlx
        self.last_serial = 0
        self.last_cumulative_after_step = 0
        self.summary_written = False
        self._csv_writer = BufferedCSVAppender(self.path, OPTIMIZER_DIAGNOSTIC_FIELDS)
        self._qn_writer = (BufferedJSONLAppender(self.qn_path) if self.qn_path else None)
        if self.state_dump_dir:
            os.makedirs(self.state_dump_dir, exist_ok=True)

    def close(self):
        self._csv_writer.close()
        if self._qn_writer is not None:
            self._qn_writer.close()
        close_mode = getattr(self.mode_recorder, "close", None)
        if callable(close_mode):
            close_mode()

    def _base_row(self):
        recorder = self.mode_recorder
        return {
            "trace_id": recorder.trace_id,
            "trace_start_unix_ns": recorder.trace_start_unix_ns,
            "src_index": recorder.src_index,
            "rank": recorder.rank,
            "attempt_id": recorder.attempt_id,
            "selected_index": recorder.selected_index,
            "reaction_type": recorder.reaction_type,
        }

    def __call__(self):
        diagnostics = getattr(self.dim_rlx, "last_step_diagnostics", None)
        if not diagnostics:
            return
        serial = int(diagnostics.get("diagnostic_serial", 0))
        if serial <= self.last_serial:
            return
        row = self._base_row()
        row["record_type"] = "step"
        row.update(diagnostics)
        self._csv_writer.append(row)
        detail = getattr(self.dim_rlx, "last_qn_detail_payload", None)
        if self.qn_path and detail:
            payload = dict(detail)
            payload.update(self._base_row())
            payload["diagnostic_serial"] = serial
            payload["translation_step"] = diagnostics.get("translation_step", self.dim_rlx.nsteps)
            self._qn_writer.append(payload)
        exact_state = getattr(self.dim_rlx, "last_lbfgs_state_dump", None)
        if self.state_dump_dir and exact_state:
            step = int(
                diagnostics.get("accepted_translation_step", self.dim_rlx.nsteps)
            )
            name = (
                f"lbfgs_state_src{self.mode_recorder.src_index}"
                f"_attempt{self.mode_recorder.attempt_id}"
                f"_rank{self.mode_recorder.rank}"
                f"_step{step:04d}_serial{serial:04d}.npz"
            )
            write_state_dump(
                os.path.join(self.state_dump_dir, name),
                exact_state,
                metadata={
                    "trace_id": self.mode_recorder.trace_id,
                    "trace_start_unix_ns": self.mode_recorder.trace_start_unix_ns,
                    "src_index": self.mode_recorder.src_index,
                    "rank": self.mode_recorder.rank,
                    "attempt_id": self.mode_recorder.attempt_id,
                    "selected_index": self.mode_recorder.selected_index,
                    "reaction_type": self.mode_recorder.reaction_type,
                    "diagnostic_serial": serial,
                    "accepted_translation_step": step,
                },
            )

        self.last_serial = serial
        try:
            self.last_cumulative_after_step = int(
                diagnostics.get("force_calls_cumulative_after_step", 0)
            )
        except (TypeError, ValueError):
            self.last_cumulative_after_step = 0

    def write_summary(self, status, converged):
        if self.summary_written:
            return
        # Flush a last accepted step if the observer was interrupted by StopRun.
        self()
        row = self._base_row()
        final_force_calls = self.d_atoms.control.get_counter("forcecalls")
        row.update({
            "record_type": "summary",
            "final_force_calls": final_force_calls,
            "force_calls_final_evaluation": (
                int(final_force_calls) - self.last_cumulative_after_step
            ),
            "final_translation_steps": self.dim_rlx.nsteps,
            "converged": int(bool(converged)),
            "status": status,
        })
        _canonical_summary = getattr(
            self.dim_rlx, "canonical_summary_metrics", None
        )
        if callable(_canonical_summary):
            row.update(_canonical_summary())
        else:
            row.update(
                history_summary_fields(
                    self.d_atoms,
                    record_only=bool(
                        getattr(self.dim_rlx, "canonical_history_record_only", False)
                    ),
                    history_options=getattr(
                        self.d_atoms, "canonical_history_options", None
                    ),
                )
            )
        _history_dump_path, _history_dump_error = dump_history_if_requested(
            self.d_atoms,
            directory=".",
            trace_id=self.mode_recorder.trace_id,
            src_index=self.mode_recorder.src_index,
            attempt_id=self.mode_recorder.attempt_id,
        )
        if _history_dump_path:
            row["canonical_history_dump_path"] = _history_dump_path
        if _history_dump_error:
            row["canonical_history_dump_error"] = _history_dump_error
        self._csv_writer.append(row)
        self.summary_written = True
        self.close()


def _setup_dimer(atoms, calc, eigenmode=None, displacement_dict=None,
                 dimer_control_kwargs=None, control_logfile=None,
                 mode_logfile=None, logfile=None, trajectory=None,
                 engine="ase", min_mode_finder="dimer",
                 mmf_variant="standard", mode_reuse="none",
                 convex_escape="standard", bowl_active_atoms=20,
                 minmode_options=None, initial_hessian_matrix=None,
                 kappa_kwargs=None, kappa_control_kwargs=None,
                 rotation_optimizer="ase", translation_optimizer="ase",
                 rotation_lbfgs_options=None,
                 translation_lbfgs_options=None, hybrid_options=None,
                 history_options=None, wave_b_options=None,
                 partitioned_lbfgs_options=None, rfo_runtime=None, runtime_state=None,
                 attempt_rng=None):
    """Stable compatibility wrapper around the role-oriented Dimer factory."""
    from saddlemill.dimertools.dimer_factory import create_dimer_search

    return create_dimer_search(
        atoms,
        calc,
        eigenmode=eigenmode,
        displacement_dict=displacement_dict,
        dimer_control_kwargs=dimer_control_kwargs,
        control_logfile=control_logfile,
        mode_logfile=mode_logfile,
        logfile=logfile,
        trajectory=trajectory,
        engine=engine,
        min_mode_finder=min_mode_finder,
        mmf_variant=mmf_variant,
        mode_reuse=mode_reuse,
        convex_escape=convex_escape,
        bowl_active_atoms=bowl_active_atoms,
        minmode_options=minmode_options,
        initial_hessian_matrix=initial_hessian_matrix,
        kappa_kwargs=kappa_kwargs,
        kappa_control_kwargs=kappa_control_kwargs,
        rotation_optimizer=rotation_optimizer,
        translation_optimizer=translation_optimizer,
        rotation_lbfgs_options=rotation_lbfgs_options,
        translation_lbfgs_options=translation_lbfgs_options,
        hybrid_options=hybrid_options,
        history_options=history_options,
        wave_b_options=wave_b_options,
        partitioned_lbfgs_options=partitioned_lbfgs_options,
        rfo_runtime=rfo_runtime,
        runtime_state=runtime_state,
        attempt_rng=attempt_rng,
    )

def _refine_eigenmode(atoms, calc, eigenmode, dimer_control_kwargs=None,
                      control_logfile=None):
    """Refine eigenmode via dimer rotation only (no translation).

    Works on a copy of *atoms* — the original is never modified.
    Returns (refined_eigenmode, curvature).
    """
    refine_atoms = atoms.copy()
    refine_atoms.calc = calc
    from saddlemill.dimertools.kappa_dimer import IsolatedDimerControl
    d_control = IsolatedDimerControl(
        logfile=control_logfile, **(dimer_control_kwargs or {})
    )
    d_atoms = MinModeAtoms(refine_atoms, d_control,
                           eigenmodes=[np.array(eigenmode)])
    d_atoms.displace(displacement_vector=np.random.randn(len(refine_atoms), 3) * 1e-10,
                     method='vector')
    # get_forces() triggers eigenmode rotation (up to max_num_rot iterations).
    # No translation — only the eigenmode direction and curvature are updated.
    d_atoms.get_forces()
    return d_atoms.get_eigenmode(), float(d_atoms.get_curvature())


def _log_rng_provenance(context, stage):
    run = context.run
    if context.attempt_rng is None or run.rng_file is None:
        return
    payload = {
        "stage": str(stage),
        "src_index": run.src_index,
        "rank": run.rank,
        "attempt_id": context.attempt_id,
        "configured_reaction_type": context.configured_reaction_type,
        "rng": run.rng_factory.provenance_for_attempt(
            context.attempt_id, context.configured_reaction_type
        ),
    }
    with open(run.rng_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _log_status(context, status_msg, n_force_calls=0, selected_index=None):
    run = context.run
    if selected_index is None:
        selected_index = context.selected_index
    with open(run.status_file, "a") as handle:
        handle.write(
            f'{run.src_index},{run.rank},{context.attempt_id},'
            f'{selected_index},{n_force_calls},"{status_msg}"\n'
        )


def _log_rxn_legacy(context):
    run = context.run
    atoms = context.atoms
    with open(run.rxn_file, "a") as handle:
        handle.write(
            f"{run.src_index},{context.attempt_id},"
            f"{atoms.info['reaction_type']},{int(atoms.info['converged'])},"
            f"{atoms.info['n_force_calls']}\n"
        )


def _log_reaction(context, final_type, converged, n_force_calls, status_msg,
                  source=None, confidence="exact", selected_index=None):
    run = context.run
    if source is None:
        source = context.reaction_source
    if selected_index is None:
        selected_index = context.selected_index
    _append_csv_row(
        run.reaction_file,
        REACTION_DIAGNOSTIC_FIELDS,
        {
            "src_index": run.src_index,
            "rank": run.rank,
            "attempt_id": context.attempt_id,
            "selected_index": selected_index,
            "configured_reaction_type": context.configured_reaction_type,
            "initial_reaction_type": context.initial_reaction_type,
            "final_reaction_type": final_type,
            "converged": int(bool(converged)),
            "n_force_calls": int(n_force_calls or 0),
            "status": status_msg,
            "classification_source": source,
            "classification_confidence": confidence,
        },
    )


def _log_attempt_timing(context, final_type, converged, n_force_calls,
                        status_msg, dim_rlx=None, selected_index=None):
    """Write one low-overhead wall-time summary row for an attempt.

    perf_counter_ns() is monotonic and is used for durations. time_ns() is
    recorded only to correlate attempts with external Slurm/GPU logs.
    attempt_wall_seconds includes normal per-attempt setup, optimization,
    final evaluation/output, and debug archiving. One-time worker/model
    initialization that occurs before the attempt loop is intentionally not
    assigned to an individual attempt.
    """
    run = context.run
    if selected_index is None:
        selected_index = context.selected_index
    end_perf_ns = time.perf_counter_ns()
    end_unix_ns = time.time_ns()
    total_ns = max(0, end_perf_ns - context.attempt_start_perf_ns)
    optimizer_ns = max(0, int(context.optimizer_elapsed_ns or 0))
    archive_ns = max(0, int(context.archive_elapsed_ns or 0))
    other_ns = max(0, total_ns - optimizer_ns - archive_ns)

    hessian_callback = (
        getattr(dim_rlx, "sm_hessian_callback", None)
        if dim_rlx is not None else None
    )
    hessian_calls = int(getattr(hessian_callback, "calls", 0))
    hessian_seconds = float(getattr(hessian_callback, "total_seconds", 0.0))

    _append_csv_row(
        run.timing_file,
        ATTEMPT_TIMING_FIELDS,
        {
            "src_index": run.src_index,
            "rank": run.rank,
            "attempt_id": context.attempt_id,
            "selected_index": selected_index,
            "configured_reaction_type": context.configured_reaction_type,
            "initial_reaction_type": context.initial_reaction_type,
            "final_reaction_type": final_type,
            "saddle_engine": run.saddle_engine,
            "rotation_optimizer": run.config_dict["ourDimer"].get(
                "rotation_optimizer", "ase"
            ),
            "translation_optimizer": run.config_dict["ourDimer"].get(
                "translation_optimizer", "ase"
            ),
            "attempt_start_unix_ns": context.attempt_start_unix_ns,
            "attempt_end_unix_ns": end_unix_ns,
            "attempt_wall_seconds": total_ns / 1.0e9,
            "attempt_excluding_archive_seconds": (
                max(0, total_ns - archive_ns) / 1.0e9
            ),
            "optimizer_wall_seconds": optimizer_ns / 1.0e9,
            "archive_wall_seconds": archive_ns / 1.0e9,
            "other_wall_seconds": other_ns / 1.0e9,
            "n_force_calls": int(n_force_calls or 0),
            "sella_hessian_calls": hessian_calls,
            "sella_hessian_seconds": hessian_seconds,
            "converged": int(bool(converged)),
            "status": status_msg,
        },
    )


def _clear_attempt_diagnostics(context):
    diagnostic_key = {
        "src_index": context.run.src_index,
        "attempt_id": context.attempt_id,
    }
    _remove_csv_rows(context.run.reaction_file, diagnostic_key)
    _remove_csv_rows(context.run.timing_file, diagnostic_key)


def _record_generation_failure(context):
    status_msg = "error: failed to generate attempt"
    _log_status(context, status_msg, selected_index=-1)
    _log_reaction(
        context,
        context.configured_reaction_type,
        False,
        0,
        status_msg,
        source="configured_attempt_order",
        confidence="base_type_only",
        selected_index=-1,
    )
    _log_attempt_timing(
        context,
        context.configured_reaction_type,
        False,
        0,
        status_msg,
        dim_rlx=None,
        selected_index=-1,
    )
    try:
        write_terminal_attempt_metrics(
            context, status=status_msg, converged=False, fallback_force_calls=0
        )
    except Exception:
        pass


def _find_runtime_state(info):
    current = info
    seen = set()
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        state = current.get("saddlemill_dimer_runtime_state")
        if state is not None:
            return state
        current = current.get("orig_info")
    return None


def _find_sella_ablation_state(info):
    current = info
    seen = set()
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        state = current.get("sella_ablation_state")
        if state is not None:
            return state
        current = current.get("orig_info")
    return None


def _setup_and_execute_attempt(context):
    """Run the existing Dimer/Sella science for one prepared attempt.

    This stage intentionally does not write the final trajectory/status/reaction
    records.  It mutates only the same attempt objects the predecessor mutated and
    leaves result materialization/recording to separate lifecycle stages.
    """
    run = context.run
    config_dict = run.config_dict
    atoms = context.atoms
    temp_log, temp_opt_log, temp_traj, temp_mode_log = context.temp_files[:4]
    context.realized_initial_geometry_sha256 = geometry_sha256(atoms)

    # Handle constraints.
    if atoms.constraints:
        context.free_indices = [
            atom.index for atom in atoms
            if atom.index not in atoms.constraints[0].get_indices()
        ]
    else:
        context.free_indices = [atom.index for atom in atoms]

    # Use existing eigenmode if available (top level from get_attempts/initial_guess,
    # or orig_info from continuation), otherwise let ASE derive one from displacement.
    context.eigenmode = atoms.info.get("eigenmode")
    if context.eigenmode is None:
        context.eigenmode = atoms.info.get("orig_info", {}).get("eigenmode")
    if context.eigenmode is not None:
        context.eigenmode = np.array(context.eigenmode)

    context.attempt_calc = resolve_vasp_calc(
        config_dict,
        run.calc,
        run.src_index,
        context.attempt_id,
        "ourDimer",
        atoms=atoms,
    )

    if run.saddle_engine == "sella":
        from saddlemill.sella_engine import setup_sella
        context.atoms, context.dim_rlx = setup_sella(
            atoms,
            context.attempt_calc,
            eigenmode=context.eigenmode,
            displacement_dict=context.displacement_dict,
            dimer_control_kwargs=config_dict["DimerControl"],
            logfile=temp_opt_log,
            trajectory=temp_traj,
            sella_options=run.sella_options,
            hessian_calc=run.sella_hessian_calc,
            attempt_rng=context.attempt_rng,
        )
        atoms = context.atoms
        from saddlemill.sella_engine import create_sella_ablation_session
        context.sella_ablation_session = create_sella_ablation_session(
            config_dict,
            context.dim_rlx,
            metadata={
                "rank": run.rank,
                "src_index": run.src_index,
                "attempt_id": context.attempt_id,
                "selected_index": context.selected_index,
                "reaction_type": context.initial_reaction_type,
            },
            restored_state=_find_sella_ablation_state(atoms.info),
        )
        if context.sella_ablation_session is not None:
            context.sella_ablation_session.__enter__()
        if bool(config_dict.get("ourSella", {}).get("linalg_failure_diagnostics", False)):
            from saddlemill.sella_diagnostics import SellaLinalgFailureRecorder
            sella_linalg_json = (
                f"dimer_sella_linalg_failure_{run.src_index}_{context.attempt_id}_"
                f"{context.selected_index}.json"
            )
            sella_linalg_npz = (
                f"dimer_sella_linalg_operands_{run.src_index}_{context.attempt_id}_"
                f"{context.selected_index}.npz"
            )
            context.temp_files.extend([sella_linalg_json, sella_linalg_npz])
            context.sella_linalg_failure_recorder = SellaLinalgFailureRecorder(
                context.dim_rlx,
                sella_linalg_json,
                sella_linalg_npz,
                metadata={
                    "rank": run.rank,
                    "src_index": run.src_index,
                    "attempt_id": context.attempt_id,
                    "selected_index": context.selected_index,
                    "reaction_type": context.initial_reaction_type,
                    "sella_ablation": config_dict.get("ourSella", {}).get("ablation"),
                },
            )
            context.sella_linalg_failure_recorder.__enter__()
        if bool(config_dict.get("ourSella", {}).get("passive_qn_diagnostics", False)):
            from saddlemill.sella_diagnostics import SellaPassiveQNRecorder
            sella_diag_path = os.path.join(
                "Dimer_sella_qn_diagnostics",
                f"sella_qn_rank{run.rank}_src{run.src_index}_attempt{context.attempt_id}"
                f"_sel{context.selected_index}.jsonl",
            )
            sella_qn_recorder = SellaPassiveQNRecorder(
                context.dim_rlx,
                sella_diag_path,
                rank=run.rank,
                src_index=run.src_index,
                attempt_id=context.attempt_id,
                selected_index=context.selected_index,
                reaction_type=context.initial_reaction_type,
            )
            context.sella_passive_qn_recorder = sella_qn_recorder
            context.dim_rlx.attach(sella_qn_recorder, interval=1)
    else:
        context.d_atoms, context.dim_rlx = _setup_dimer(
            atoms,
            context.attempt_calc,
            eigenmode=context.eigenmode,
            displacement_dict=context.displacement_dict,
            dimer_control_kwargs=config_dict["DimerControl"],
            control_logfile=temp_log,
            mode_logfile=temp_mode_log,
            logfile=temp_opt_log,
            trajectory=temp_traj,
            engine=run.saddle_engine,
            min_mode_finder=run.dimer_method_cfg["min_mode_finder"],
            mmf_variant=run.dimer_method_cfg["mmf_variant"],
            mode_reuse=run.dimer_method_cfg["mode_reuse"],
            convex_escape=run.dimer_method_cfg["convex_escape"],
            bowl_active_atoms=int(config_dict["ourDimer"].get("bowl_active_atoms", 20)),
            minmode_options=run.minmode_options,
            initial_hessian_matrix=run.initial_hessian_matrix,
            kappa_kwargs={
                "beta": config_dict["ourDimer"]["kappa_beta"],
                "recover_fmax": config_dict["ourDimer"]["kappa_recover_fmax"],
            },
            kappa_control_kwargs=(config_dict.get("Kappa") or None),
            rotation_optimizer=config_dict["ourDimer"].get("rotation_optimizer", "ase"),
            translation_optimizer=config_dict["ourDimer"].get("translation_optimizer", "ase"),
            rotation_lbfgs_options=run.rotation_lbfgs_options,
            translation_lbfgs_options=run.translation_lbfgs_options,
            hybrid_options=run.hybrid_options,
            history_options=history_options_from_config(
                config_dict,
                translation_optimizer=config_dict["ourDimer"].get(
                    "translation_optimizer", "ase"
                ),
            ),
            wave_b_options=run.wave_b_options,
            partitioned_lbfgs_options=(
                partitioned_options_from_config(
                    config_dict,
                    translation_optimizer=str(
                        config_dict["ourDimer"].get("translation_optimizer", "ase")
                    ).lower(),
                )
                if str(config_dict["ourDimer"].get("translation_optimizer", "ase")).lower()
                in {"partitioned_lbfgs", "q_lbfgs_dimer_axial"}
                else None
            ),
            rfo_runtime=(
                create_rfo_translation_runtime(config_dict)
                if str(config_dict["ourDimer"].get("translation_optimizer", "ase")).lower()
                in {"rfo", "prfo", "qn_mmf"}
                else None
            ),
            runtime_state=_find_runtime_state(atoms.info),
            attempt_rng=context.attempt_rng,
        )

    # Existing Dimer-only diagnostics retain their exact behavior. Sella force
    # calls are recorded in canonical status/reaction metadata from its PES counter.
    if run.saddle_engine != "sella":
        mode_recorder = ModeDiagnosticRecorder(
            run.mode_file,
            src_index=run.src_index,
            rank=run.rank,
            attempt_id=context.attempt_id,
            selected_index=context.selected_index,
            reaction_type=context.initial_reaction_type,
            d_atoms=context.d_atoms,
            dim_rlx=context.dim_rlx,
            free_indices=context.free_indices,
        )
        context.dim_rlx.attach(mode_recorder, interval=1)
        state_dump_dir = None
        if bool(run.translation_lbfgs_options.get("state_dump", False)):
            configured_dump_dir = str(
                run.translation_lbfgs_options.get("state_dump_directory", "")
            ).strip()
            state_dump_dir = configured_dump_dir or f"{run.method_name}_lbfgs_state_dumps"
        context.optimizer_recorder = OptimizerDiagnosticRecorder(
            run.optimizer_file,
            mode_recorder,
            context.d_atoms,
            context.dim_rlx,
            qn_path=run.qn_file,
            state_dump_dir=state_dump_dir,
        )
        context.dim_rlx.attach(context.optimizer_recorder, interval=1)

    sella_cfg = config_dict.get("ourSella", {}) or {}
    check_interval = (
        int(sella_cfg.get("check_interval", 5))
        if run.saddle_engine == "sella" else 5
    )

    # PR Check — skip early steps to let the dimer rotate the eigenmode (initial
    # displacement can look delocalized, especially for diffusion/rotation types).
    delocalization_start_step = max(1, int(0.1 * config_dict["Main"]["steps"]))

    def check_delocalization():
        if context.dim_rlx.nsteps < delocalization_start_step:
            return
        if run.saddle_engine == "sella":
            if not bool(sella_cfg.get("check_delocalization", False)):
                return
            from saddlemill.sella_engine import extract_lowest_mode
            mode, _, _ = extract_lowest_mode(context.dim_rlx)
        else:
            mode = context.d_atoms.get_eigenmode()
        v2 = (mode ** 2).sum(axis=1)
        v2 = v2[context.free_indices]
        sum_v2 = np.sum(v2)
        if sum_v2 < 1e-12:
            return
        pr = (sum_v2 ** 2) / (len(v2) * np.sum(v2 ** 2))
        if pr > config_dict["ourDimer"]["delocalization_threshold"]:
            raise StopRun(f"Eigenmode Delocalized (PR={pr:.3f})")

    def check_desorption():
        if run.saddle_engine == "sella" and not bool(
            sella_cfg.get("check_desorption", True)
        ):
            return
        check_atoms = (
            context.atoms
            if run.saddle_engine == "sella"
            else context.d_atoms.atoms
        )
        cutoffs = natural_cutoffs(check_atoms, mult=2.0)
        ii, jj = neighbor_list("ij", check_atoms, cutoffs)
        adjacency = csr_matrix(
            (np.ones(len(ii)), (ii, jj)),
            shape=(len(check_atoms), len(check_atoms)),
        )
        n_components, labels = connected_components(
            adjacency, connection="weak"
        )
        if n_components > 1:
            raise StopRun("Adsorbate desorbed")

    if run.saddle_engine == "sella":
        if bool(sella_cfg.get("check_delocalization", False)):
            context.dim_rlx.attach(check_delocalization, interval=check_interval)
        if bool(sella_cfg.get("check_desorption", True)):
            context.dim_rlx.attach(check_desorption, interval=check_interval)
    else:
        context.dim_rlx.attach(check_delocalization, interval=5)
        context.dim_rlx.attach(check_desorption, interval=5)

    try:
        try:
            context.converged = run_optimizer_timed(
                context,
                fmax=config_dict["Main"]["fmax"],
                steps=config_dict["Main"]["steps"],
            )
        except StopRun as exc:
            context.stopped_early = True
            context.stop_reason = str(exc)
            context.converged = False
    finally:
        passive_recorder = context.sella_passive_qn_recorder
        if passive_recorder is not None:
            try:
                passive_recorder.close()
            except Exception as exc:
                print(f"Warning: failed to flush passive Sella diagnostics: {exc}", flush=True)
        linalg_recorder = context.sella_linalg_failure_recorder
        if linalg_recorder is not None:
            linalg_recorder.close()
        session = context.sella_ablation_session
        if session is not None:
            try:
                context.sella_ablation_summary = session.summary()
                context.sella_ablation_state = session.state_dict()
                context.sella_ablation_rows = [dict(row) for row in session.rows]
            finally:
                session.close()

    if run.saddle_engine == "sella":
        from saddlemill.sella_engine import (
            classify_sella_convergence,
            extract_lowest_mode,
            sella_force_calls,
        )
        try:
            (
                context.eigenmode,
                context.curvature,
                context.sella_eigenvalues,
            ) = extract_lowest_mode(context.dim_rlx)
        except Exception:
            if not context.stopped_early:
                raise
            if context.eigenmode is None:
                context.eigenmode = np.zeros((len(context.atoms), 3), dtype=float)
            context.curvature = float("nan")
            context.sella_eigenvalues = np.array([], dtype=float)
        context.sella_stationary_converged = bool(context.converged)
        (
            context.converged,
            context.status,
            context.sella_negative_modes,
        ) = classify_sella_convergence(
            context.sella_stationary_converged,
            context.sella_eigenvalues,
            negative_eigenvalue_tolerance=float(
                sella_cfg.get("negative_eigenvalue_tolerance", 1.0e-6)
            ),
            require_first_order_model=bool(
                sella_cfg.get("require_first_order_model", True)
            ),
        )
        if context.stopped_early:
            context.converged = False
            context.status = "not_converged_StopRun"
        context.n_force_calls = sella_force_calls(context.dim_rlx)
    else:
        if context.converged:
            context.status = "converged"
        elif not context.converged and not context.stopped_early:
            # Extension check.
            fmax_check = (
                np.sqrt((context.d_atoms.get_forces() ** 2).sum(axis=1).max())
                < config_dict["ourDimer"]["extension_check_fmax"]
            )
            curvature_check = (
                context.d_atoms.get_curvature()
                < config_dict["ourDimer"]["extension_check_curvature"]
            )
            if fmax_check and curvature_check:
                try:
                    context.converged = run_optimizer_timed(
                        context,
                        fmax=config_dict["Main"]["fmax"],
                        steps=150,
                    )
                except StopRun as exc:
                    context.stopped_early = True
                    context.stop_reason = str(exc)
                    context.converged = False
                if context.converged:
                    context.status = "converged_after_extension"
                else:
                    context.status = "not_converged_after_extension"
            else:
                context.status = "not_converged"
        else:
            context.status = "not_converged_StopRun"
        context.eigenmode = context.d_atoms.get_eigenmode()
        context.curvature = context.d_atoms.get_curvature()
        context.n_force_calls = context.d_atoms.control.get_counter("forcecalls")

    context.energy = context.atoms.get_potential_energy()
    context.forces = context.atoms.get_forces()
    finalize_if_vasp_interactive(config_dict, context.attempt_calc)
    if context.attempt_vasp_dir is not None:
        remove_vasp_heavies(context.attempt_vasp_dir)


def _materialize_attempt_result(context):
    """Apply final metadata/calculator state without performing record I/O."""
    run = context.run
    atoms = context.atoms
    dim_rlx = context.dim_rlx

    atoms.info["eigenmode"] = context.eigenmode
    atoms.info["curvature"] = float(context.curvature)
    atoms.info["n_force_calls"] = int(context.n_force_calls)
    atoms.info["converged"] = 1 if context.converged else 0
    atoms.info["src_index"] = run.src_index
    atoms.info["attempt_id"] = context.attempt_id
    atoms.info["stoprun"] = 1 if context.stopped_early else 0
    atoms.info["selected_index"] = context.selected_index
    if context.attempt_rng is not None:
        atoms.info["rng_provenance"] = run.rng_factory.provenance_for_attempt(
            context.attempt_id, context.configured_reaction_type
        )
    orig = atoms.info.get("orig_info", {})
    atoms.info["reaction_type"] = atoms.info.get(
        "reaction_type", orig.get("reaction_type", "unknown")
    )

    if run.saddle_engine == "sella":
        atoms.info["saddle_engine"] = "sella"
        atoms.info["sella_version"] = getattr(dim_rlx, "sm_sella_version", "unknown")
        atoms.info["sella_used_input_mode"] = int(
            bool(getattr(dim_rlx, "sm_used_input_mode", False))
        )
        atoms.info["sella_stationary_converged"] = int(
            bool(context.sella_stationary_converged)
        )
        atoms.info["sella_model_negative_modes"] = int(
            context.sella_negative_modes or 0
        )
        atoms.info["sella_order_check"] = "approximate_model_hessian"
        atoms.info["sella_order"] = 1
        atoms.info["sella_hessian_engine"] = getattr(
            dim_rlx, "sm_hessian_engine", "sella"
        )
        hessian_callback = getattr(dim_rlx, "sm_hessian_callback", None)
        atoms.info["sella_hessian_calls"] = int(
            getattr(hessian_callback, "calls", 0)
        )
        atoms.info["sella_hessian_seconds"] = float(
            getattr(hessian_callback, "total_seconds", 0.0)
        )
        hessian_meta = getattr(dim_rlx, "sm_hessian_metadata", {}) or {}
        if hessian_meta.get("task_name") is not None:
            atoms.info["sella_hessian_task_name"] = str(hessian_meta["task_name"])
        if hessian_meta.get("model_name_or_path") is not None:
            atoms.info["sella_hessian_model"] = str(hessian_meta["model_name_or_path"])
        if hessian_meta.get("fairchem_core_version") is not None:
            atoms.info["sella_hessian_fairchem_version"] = str(
                hessian_meta["fairchem_core_version"]
            )
        if context.sella_ablation_summary is not None:
            summary = dict(context.sella_ablation_summary)
            atoms.info["sella_ablation_schema"] = summary.get("schema", "sella_ablation_v1")
            atoms.info["sella_ablation_selector"] = summary.get("selector")
            atoms.info["sella_ablation_selector_identity"] = summary.get("selector_identity")
            atoms.info["sella_ablation_hessian_learning"] = summary.get("hessian_learning")
            atoms.info["sella_ablation_translation"] = summary.get("translation")
            atoms.info["sella_ablation_trust_adaptation"] = summary.get("trust_adaptation")
            atoms.info["sella_ablation_supported_sella_version"] = summary.get("supported_sella_version")
            atoms.info["sella_ablation_reference_sha256"] = dict(summary.get("reference_sha256", {}) or {})
            atoms.info["sella_ablation_steps_recorded"] = int(summary.get("steps_recorded", 0))
            atoms.info["sella_ablation_center_update_calls"] = int(summary.get("center_update_calls", 0))
            atoms.info["sella_ablation_center_updates_admitted"] = int(summary.get("center_updates_admitted", 0))
            atoms.info["sella_ablation_probe_update_calls"] = int(summary.get("probe_update_calls", 0))
            atoms.info["sella_ablation_probe_updates_admitted"] = int(summary.get("probe_updates_admitted", 0))
            atoms.info["sella_ablation_probe_vectors_seen"] = int(summary.get("probe_vectors_seen", 0))
            atoms.info["sella_ablation_eigensolver_probe_evaluations"] = int(summary.get("eigensolver_probe_evaluations", 0))
            atoms.info["sella_ablation_diagnostic_additional_pes_calls"] = int(summary.get("diagnostic_additional_pes_calls", 0))
            atoms.info["sella_ablation_summary"] = summary
            atoms.info["sella_ablation_state"] = dict(context.sella_ablation_state or {})
            atoms.info["sella_ablation_rows"] = list(context.sella_ablation_rows or [])

    if context.stop_reason and "desorbed" in context.stop_reason:
        context.status = "converged_to_desorption"
        atoms.info["converged"] = 1
        atoms.info["reaction_type"] = "desorption"

    if context.optimizer_recorder is not None:
        context.optimizer_recorder.write_summary(
            context.status, atoms.info["converged"]
        )

    history_options = history_options_from_config(
        run.config_dict,
        translation_optimizer=run.config_dict["ourDimer"].get("translation_optimizer", "ase"),
    )
    from saddlemill.config_factories import build_dimer_resolved_run_metadata
    atoms.info["resolved_run_metadata"] = build_dimer_resolved_run_metadata(
        run.config_dict,
        stopping_reason=context.stop_reason or context.status,
        history_options=history_options,
    ).to_state_dict()
    if run.saddle_engine != "sella" and context.d_atoms is not None:
        from saddlemill.dimertools.runtime_state import capture_dimer_runtime_state
        runtime_state = capture_dimer_runtime_state(context.d_atoms)
        if runtime_state is not None:
            atoms.info["saddlemill_dimer_runtime_state"] = runtime_state
        wave_runtime = getattr(context.d_atoms, "wave_b_runtime", None)
        if wave_runtime is not None:
            atoms.info["wave_b_runtime_metadata"] = wave_runtime.metadata()
            hindsight_rows = wave_runtime.finalize_hindsight(
                context.d_atoms, run_converged=bool(context.converged)
            )
            if hindsight_rows:
                atoms.info["mode_hindsight_records"] = hindsight_rows
        history = getattr(context.d_atoms, "canonical_force_history", None)
        if history is not None:
            atoms.info["force_accounting"] = history.accounting.to_state_dict()
        shadow_row = getattr(context.dim_rlx, "last_qn_shadow_row", None)
        if shadow_row is not None:
            atoms.info["qn_shadow_diagnostic"] = dict(shadow_row)
        atoms.info["rotation_wave_b_diagnostics"] = dict(
            getattr(context.d_atoms, "last_rotation_diagnostics", {}) or {}
        )
        broyden_diag = getattr(context.dim_rlx, "last_broyden_diagnostics", None)
        if broyden_diag:
            atoms.info["translation_broyden_diagnostics"] = dict(broyden_diag)
        partitioned_diag = getattr(
            context.dim_rlx, "last_partitioned_diagnostics", None
        )
        if partitioned_diag:
            atoms.info["translation_partitioned_lbfgs_diagnostics"] = dict(
                partitioned_diag
            )
        rfo_diag = getattr(context.dim_rlx, "last_rfo_diagnostics", None)
        if rfo_diag:
            atoms.info["translation_rfo_diagnostics"] = dict(rfo_diag)

    atoms.info["status"] = context.status
    atoms.info["task_name"] = run.task_name
    atoms.wrap()
    atoms.calc = SinglePointCalculator(
        atoms, energy=context.energy, forces=context.forces
    )


def _archive_attempt(context, prefix):
    return archive_temp_files_timed(
        context,
        archive_and_clear_temp_files,
        context.run.zip_name,
        prefix=prefix,
        enabled=context.run.config_dict["Main"]["zip"],
    )


def _record_attempt_success(context, writer):
    """Emit the predecessor's successful-attempt records in the same order."""
    run = context.run
    atoms = context.atoms

    _log_rng_provenance(context, "completed")
    writer.write(atoms)
    if run.saddle_engine == "sella":
        context.dim_rlx.close()

    # Clean up temp files (the zip block walks directories too, so a per-attempt
    # VASP dir is captured automatically).
    _archive_attempt(context, prefix="")

    _log_status(context, context.status, context.n_force_calls)
    _log_rxn_legacy(context)
    _log_reaction(
        context,
        atoms.info["reaction_type"],
        atoms.info["converged"],
        atoms.info["n_force_calls"],
        context.status,
        source=context.reaction_source,
        confidence="exact",
    )
    _log_attempt_timing(
        context,
        atoms.info["reaction_type"],
        atoms.info["converged"],
        atoms.info["n_force_calls"],
        context.status,
        dim_rlx=context.dim_rlx,
    )
    try:
        write_terminal_attempt_metrics(
            context,
            status=context.status,
            converged=bool(atoms.info["converged"]),
            fallback_force_calls=atoms.info["n_force_calls"],
        )
    except Exception as exc:
        # Terminal metrics are additive diagnostics. A metrics write failure must
        # not change the scientific outcome of an otherwise completed attempt.
        print(f"Warning: failed to write terminal attempt metrics: {exc}", flush=True)


def _record_attempt_exception(context, exc, traceback_text):
    """Emit the predecessor's per-attempt error diagnostics and records."""
    run = context.run
    _log_rng_provenance(context, "error")
    print(
        f"Rank {run.rank} FAILED on structure {run.src_index}, "
        f"attempt {context.attempt_id}: {exc}",
        flush=True,
    )
    print(f"\nTraceback details:\n{traceback_text}", flush=True)

    # Preserve the exact worker traceback inside the normal debug ZIP. This is
    # additive diagnostics only; status/output schemas are unchanged.
    try:
        traceback_path = (
            f"dimer_exception_{run.src_index}_{context.attempt_id}_"
            f"{context.selected_index}.log"
        )
        with open(traceback_path, "w", encoding="utf-8") as handle:
            handle.write(traceback_text)
        context.temp_files.append(traceback_path)
    except Exception:
        pass

    if context.attempt_calc is not None:
        finalize_if_vasp_interactive(run.config_dict, context.attempt_calc)
    if context.sella_linalg_failure_recorder is not None:
        try:
            context.sella_linalg_failure_recorder.close()
        except Exception:
            pass
    if context.sella_ablation_session is not None:
        # W5-010: failure-only Sella eigensolve diagnostics are written by the
        # attempt-local ablation session. Register them with the existing debug ZIP
        # lifecycle without changing status/output schemas or numerical behavior.
        for diagnostic_path in getattr(
            context.sella_ablation_session, "failure_artifacts", ()
        ):
            if diagnostic_path not in context.temp_files:
                context.temp_files.append(diagnostic_path)
        try:
            context.sella_ablation_session.close()
        except Exception:
            pass
    if context.sella_passive_qn_recorder is not None:
        try:
            context.sella_passive_qn_recorder.close()
        except Exception:
            pass
    if run.saddle_engine == "sella" and context.dim_rlx is not None:
        try:
            context.dim_rlx.close()
        except Exception:
            pass
    if context.optimizer_recorder is not None and context.d_atoms is not None:
        try:
            context.optimizer_recorder.write_summary(
                f"error: {str(exc)}", False
            )
        except Exception:
            pass

    _archive_attempt(context, prefix="ERROR_")
    status_msg = f"error: {str(exc)}"
    try:
        if run.saddle_engine == "sella" and context.dim_rlx is not None:
            from saddlemill.sella_engine import sella_force_calls
            error_force_calls = sella_force_calls(context.dim_rlx)
        else:
            error_force_calls = (
                context.d_atoms.control.get_counter("forcecalls")
                if context.d_atoms is not None else 0
            )
    except Exception:
        error_force_calls = 0

    _log_status(context, status_msg, error_force_calls)
    _log_reaction(
        context,
        context.initial_reaction_type,
        False,
        error_force_calls,
        status_msg,
        source=context.reaction_source,
        confidence=(
            "exact"
            if context.reaction_source != "configured_attempt_order"
            else "base_type_only"
        ),
    )
    _log_attempt_timing(
        context,
        context.initial_reaction_type,
        False,
        error_force_calls,
        status_msg,
        dim_rlx=context.dim_rlx,
    )
    try:
        write_terminal_attempt_metrics(
            context, status=status_msg, converged=False,
            fallback_force_calls=error_force_calls,
        )
    except Exception:
        pass


def dimeropt(i, config_dict, atoms_orig, calc, consecutive_errors=None,
             executorlib_worker_id=None, **kwargs):
    """Run all configured Dimer attempts for one input structure."""
    rank = executorlib_worker_id

    rng_scheme = str(
        config_dict["Main"].get("rng_scheme", RNG_SCHEME_LEGACY)
    ).strip().lower()
    rng_factory = None
    if rng_scheme == RNG_SCHEME_LEGACY:
        # Exact historical behavior. Do not reinterpret SM_SEED_OFFSET for the
        # keyed scheme; legacy configs intentionally remain stream/order based.
        seed_offset = int(
            os.environ.get("SM_SEED_OFFSET", config_dict["Main"].get("seed_offset", 0))
        )
        seed = i + seed_offset * 100000
        random.seed(seed)
        np.random.seed(seed)
    elif rng_scheme == RNG_SCHEME_ATTEMPT_KEYED_V1:
        rng_factory = build_structure_rng_factory(config_dict, atoms_orig)
    else:
        raise ValueError(
            f"Unsupported [Main] rng_scheme={rng_scheme!r}; expected "
            f"{RNG_SCHEME_LEGACY!r} or {RNG_SCHEME_ATTEMPT_KEYED_V1!r}."
        )

    method_name = config_dict["Main"]["method"]
    status_dir = f"{method_name}_status_csvs"
    status_file = f"{status_dir}/status_rank_{rank}.csv"
    reaction_file = f"{status_dir}/reaction_rank_{rank}.csv"
    # Preserve the existing compact reaction CSV for compatibility.
    rxn_file = f"{method_name}_rxn_csvs/rxn_rank_{rank}.csv"
    mode_file = f"{method_name}_mode_csvs/mode_rank_{rank}.csv"
    optimizer_file = f"{method_name}_optimizer_csvs/optimizer_rank_{rank}.csv"
    qn_dir = f"{method_name}_qn_diagnostics"
    qn_file = f"{qn_dir}/qn_rank_{rank}.jsonl"
    timing_file = f"{method_name}_timing_csvs/timing_rank_{rank}.csv"
    metrics_file = f"{method_name}_attempt_metrics/attempt_metrics_rank_{rank}.jsonl"
    rng_dir = f"{method_name}_rng_provenance"
    rng_file = f"{rng_dir}/rng_rank_{rank}.jsonl"
    os.makedirs(status_dir, exist_ok=True)
    os.makedirs(f"{method_name}_rxn_csvs", exist_ok=True)
    os.makedirs(f"{method_name}_mode_csvs", exist_ok=True)
    os.makedirs(f"{method_name}_optimizer_csvs", exist_ok=True)
    os.makedirs(qn_dir, exist_ok=True)
    os.makedirs(f"{method_name}_timing_csvs", exist_ok=True)
    os.makedirs(f"{method_name}_attempt_metrics", exist_ok=True)
    if rng_factory is not None:
        os.makedirs(rng_dir, exist_ok=True)
    output_file = f"{method_name}_trajes/collected_ts_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"
    task_name = get_task_name(config_dict)
    is_vasp = config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive")

    max_consecutive_errors = config_dict["Main"]["max_consecutive_errors"]
    if (
        consecutive_errors is not None
        and consecutive_errors[0] >= max_consecutive_errors > 0
    ):
        print(
            f"Rank {rank}: {consecutive_errors[0]} consecutive structures errored. "
            "Killing worker for restart.",
            flush=True,
        )
        backup_flux_logs(rank)
        sys.exit(1)

    configured_attempt_types = _configured_attempt_reaction_types(config_dict)

    saddle_engine = str(config_dict["ourDimer"].get("engine", "ase")).lower()
    dimer_method_cfg = normalize_dimer_method_config(config_dict["ourDimer"])
    minmode_cfg = config_dict.get("ourMinMode", {}) or {}
    minmode_options = {
        "maxiter": minmode_cfg.get("maxiter", None),
        "max_iterations": int(minmode_cfg.get("max_iterations", 8)),
        "eigenvalue_tolerance": float(minmode_cfg.get("eigenvalue_tolerance", 0.01)),
        "finite_difference": float(minmode_cfg.get("finite_difference", 1.0e-4)),
        "breakdown_tolerance": float(minmode_cfg.get("breakdown_tolerance", 1.0e-12)),
        "davidson_initial_hessian": float(minmode_cfg.get("davidson_initial_hessian", 1.0)),
        "davidson_update_tolerance": float(minmode_cfg.get("davidson_update_tolerance", 1.0e-12)),
        "davidson_preconditioner_floor": float(minmode_cfg.get("davidson_preconditioner_floor", 1.0e-8)),
        "davidson_initial_hessian_source": str(minmode_cfg.get("davidson_initial_hessian_source", "identity")),
        "softsaddle_switch_threshold": float(minmode_cfg.get("softsaddle_switch_threshold", 12.0)),
        "softsaddle_preconditioner_floor": float(minmode_cfg.get("softsaddle_preconditioner_floor", 1.0e-12)),
        "reference_hessian_finite_difference": float(
            minmode_cfg.get(
                "reference_hessian_finite_difference",
                minmode_cfg.get("finite_difference", 1.0e-4),
            )
        ),
        "reference_hessian_reuse_center_force": bool(
            minmode_cfg.get("reference_hessian_reuse_center_force", False)
        ),
        "sm50_factor": float(minmode_cfg.get("sm50_factor", 50.0)),
        "sm50_line_search_tolerance": float(
            minmode_cfg.get("sm50_line_search_tolerance", 0.01)
        ),
        "sm50_final_refresh_recheck": bool(
            minmode_cfg.get("sm50_final_refresh_recheck", False)
        ),
        "hvp_origin": str(minmode_cfg.get("hvp_origin", "physical_fd")).strip().lower(),
        "root_policy": str(minmode_cfg.get("root_policy", "lowest")).strip().lower(),
        "residual_stop": bool(minmode_cfg.get("residual_stop", False)),
        "residual_tolerance": float(minmode_cfg.get("residual_tolerance", 1.0e-3)),
    }

    sella_options = None
    sella_hessian_calc = None
    if saddle_engine == "sella":
        from saddlemill.sella_engine import (
            direct_hessian_requested,
            get_cached_direct_hessian_calculator,
            sella_options_from_config,
        )
        sella_options = sella_options_from_config(config_dict)
        if direct_hessian_requested(config_dict):
            sella_hessian_calc = get_cached_direct_hessian_calculator(config_dict)

    lbfgs_cfg = config_dict.get("ourDimerLBFGS", {}) or {}
    rotation_lbfgs_options = {
        "memory": int(lbfgs_cfg.get("rotation_memory", 10)),
        "initial_hessian": float(lbfgs_cfg.get("rotation_initial_hessian", 1.0)),
        "dynamic_h0": bool(lbfgs_cfg.get("rotation_dynamic_h0", False)),
        "curvature_epsilon": float(lbfgs_cfg.get("curvature_epsilon", 1.0e-12)),
        "geometry": str(lbfgs_cfg.get("rotation_geometry", "projected")).strip().lower(),
        "transport_policy": str(lbfgs_cfg.get("rotation_transport_policy", "auto")).strip().lower(),
        "step_method": str(lbfgs_cfg.get("rotation_step_method", "fourier")).strip().lower(),
        "first_angle_degrees": float(lbfgs_cfg.get("rotation_first_angle_degrees", 45.0)),
        "max_angle_degrees": float(lbfgs_cfg.get("rotation_max_angle_degrees", 45.0)),
        "curvature_guard": str(lbfgs_cfg.get("rotation_curvature_guard", "legacy_skip")).strip().lower(),
        "curvature_floor": float(lbfgs_cfg.get("rotation_curvature_floor", 1.0e-3)),
        "powell_eta": float(lbfgs_cfg.get("rotation_powell_eta", 0.2)),
        "reconstruction_model": str(lbfgs_cfg.get("rotation_reconstruction_model", "lbfgs")).strip().lower(),
        "bfgs_update": str(lbfgs_cfg.get("rotation_bfgs_update", "sequential")).strip().lower(),
        "dense_bfgs_diagnostic": bool(lbfgs_cfg.get("dense_bfgs_diagnostic", False)),
    }
    translation_guard = str(
        lbfgs_cfg.get("translation_curvature_guard", "skip")
    ).strip().lower()
    translation_guard = {
        "legacy": "damp",
        "legacy_damp": "damp",
        "legacy_translation_guard": "damp",
        "shifted_secant": "damp",
    }.get(translation_guard, translation_guard)
    translation_lbfgs_options = {
        "memory": int(lbfgs_cfg.get("translation_memory", 10)),
        "initial_hessian": float(lbfgs_cfg.get("translation_initial_hessian", 70.0)),
        "dynamic_h0": bool(lbfgs_cfg.get("translation_dynamic_h0", False)),
        "curvature_epsilon": float(lbfgs_cfg.get("curvature_epsilon", 1.0e-12)),
        "damping": float(lbfgs_cfg.get("translation_damping", 1.0)),
        "curvature_guard": translation_guard,
        "curvature_floor": float(lbfgs_cfg.get("translation_curvature_floor", 1.0e-3)),
        "powell_eta": float(lbfgs_cfg.get("translation_powell_eta", 0.2)),
        "cautious_epsilon": float(lbfgs_cfg.get("translation_cautious_epsilon", 1.0e-6)),
        "cautious_alpha": float(lbfgs_cfg.get("translation_cautious_alpha", 1.0)),
        "reset_on_regime_change": bool(lbfgs_cfg.get("reset_translation_on_regime_change", True)),
        "reconstruction_model": str(lbfgs_cfg.get("translation_reconstruction_model", "lbfgs")).strip().lower(),
        "bfgs_update": str(lbfgs_cfg.get("translation_bfgs_update", "sequential")).strip().lower(),
        "dense_bfgs_diagnostic": bool(lbfgs_cfg.get("dense_bfgs_diagnostic", False)),
        "deep_qn_diagnostics": bool(lbfgs_cfg.get("translation_deep_qn_diagnostics", False)),
        "regularization": str(lbfgs_cfg.get("translation_regularization", "off")).strip().lower(),
        "regularization_mu": float(lbfgs_cfg.get("translation_regularization_mu", 1.0)),
        "regularization_radius": float(lbfgs_cfg.get("translation_regularization_radius", 0.1)),
        "regularization_tolerance": float(lbfgs_cfg.get("translation_regularization_tolerance", 1.0e-8)),
        "trial_step": str(lbfgs_cfg.get("translation_trial_step", "off")).strip().lower(),
        "state_dump": bool(lbfgs_cfg.get("translation_state_dump", False)),
        "state_dump_steps": str(lbfgs_cfg.get("translation_state_dump_steps", "all")),
        "state_dump_two_loop_trace": bool(lbfgs_cfg.get("translation_state_dump_two_loop_trace", True)),
        "state_dump_directory": str(lbfgs_cfg.get("translation_state_dump_directory", "")).strip(),
    }
    hybrid_cfg = config_dict.get("ourDimerHybrid", {}) or {}
    hybrid_options = {
        "enabled": bool(hybrid_cfg.get("enabled", False)),
        "enter_fmax": float(hybrid_cfg.get("enter_fmax", 0.30)),
        "exit_fmax": float(hybrid_cfg.get("exit_fmax", 0.50)),
        "enter_curvature": float(hybrid_cfg.get("enter_curvature", -0.05)),
        "exit_curvature": float(hybrid_cfg.get("exit_curvature", 0.00)),
        "enter_stable_steps": int(hybrid_cfg.get("enter_stable_steps", 3)),
        "exit_stable_steps": int(hybrid_cfg.get("exit_stable_steps", 2)),
        "minimum_history_pairs": int(hybrid_cfg.get("minimum_history_pairs", 3)),
        "warm_start_history": bool(hybrid_cfg.get("warm_start_history", True)),
        "reset_history_on_exit": bool(hybrid_cfg.get("reset_history_on_exit", True)),
        "fire_dt": float(hybrid_cfg.get("fire_dt", 0.10)),
        "fire_dtmax": float(hybrid_cfg.get("fire_dtmax", 1.0)),
        "fire_Nmin": int(hybrid_cfg.get("fire_Nmin", 5)),
        "fire_finc": float(hybrid_cfg.get("fire_finc", 1.1)),
        "fire_fdec": float(hybrid_cfg.get("fire_fdec", 0.5)),
        "fire_astart": float(hybrid_cfg.get("fire_astart", 0.1)),
        "fire_fa": float(hybrid_cfg.get("fire_fa", 0.99)),
    }

    from saddlemill.dimertools.wave_b_runtime import wave_b_options_from_config
    wave_b_options = wave_b_options_from_config(config_dict)
    continuation_data = kwargs.get("continuation_data")
    entries_to_run = kwargs.get("entries_to_run")
    # Optional full physical Hessian at the pre-displacement reference geometry.
    # SoftSaddle-Davidson computes this automatically when omitted; this hook permits
    # benchmark/diagnostic callers to supply an already-computed H0.
    initial_hessian_matrix = kwargs.get("initial_hessian_matrix")

    run = DimerRunContext(
        src_index=i,
        rank=rank,
        config_dict=config_dict,
        calc=calc,
        method_name=method_name,
        task_name=task_name,
        saddle_engine=saddle_engine,
        is_vasp=is_vasp,
        status_file=status_file,
        reaction_file=reaction_file,
        rxn_file=rxn_file,
        mode_file=mode_file,
        optimizer_file=optimizer_file,
        qn_file=qn_file,
        timing_file=timing_file,
        metrics_file=metrics_file,
        output_file=output_file,
        zip_name=zip_name,
        dimer_method_cfg=dimer_method_cfg,
        minmode_options=minmode_options,
        sella_options=sella_options,
        sella_hessian_calc=sella_hessian_calc,
        rotation_lbfgs_options=rotation_lbfgs_options,
        translation_lbfgs_options=translation_lbfgs_options,
        hybrid_options=hybrid_options,
        wave_b_options=wave_b_options,
        initial_hessian_matrix=initial_hessian_matrix,
        rng_factory=rng_factory,
        rng_file=(rng_file if rng_factory is not None else None),
    )

    any_attempt_succeeded = False
    all_attempts_none = False

    with Trajectory(output_file, "a") as writer:
        generated = get_attempts(atoms_orig, config_dict, rng_factory=rng_factory)
        all_attempts_none = all(atoms is None for atoms in generated[0])
        if all_attempts_none:
            print(
                f"Rank {rank} WARNING on structure {i}: All attempts failed to generate.",
                flush=True,
            )

        for attempt_id, (atoms, displacement_dict, selected_index) in enumerate(
            zip(*generated)
        ):
            if entries_to_run is not None and attempt_id not in entries_to_run:
                continue

            attempt_start_perf_ns = time.perf_counter_ns()
            attempt_start_unix_ns = time.time_ns()
            configured_type = (
                configured_attempt_types[attempt_id]
                if isinstance(attempt_id, int)
                and 0 <= attempt_id < len(configured_attempt_types)
                else "unknown"
            )
            attempt_rng = (
                None
                if rng_factory is None
                else rng_factory.for_attempt(attempt_id, configured_type)
            )
            context = start_attempt(
                run,
                attempt_id,
                atoms,
                displacement_dict,
                selected_index,
                configured_type,
                attempt_start_perf_ns=attempt_start_perf_ns,
                attempt_start_unix_ns=attempt_start_unix_ns,
                attempt_rng=attempt_rng,
            )
            _log_rng_provenance(context, "generated")

            # Keep the additive diagnostic shards aligned with the active result on
            # resume/retry. Existing status/output cleanup remains unchanged.
            _clear_attempt_diagnostics(context)

            has_continuation = bool(
                continuation_data and attempt_id in continuation_data
            )

            # Keyed mode treats an existing continuation as authoritative even if
            # fresh candidate generation for the same configured slot would now
            # return None. Candidate planning may still be materialized above so
            # coupled without-replacement assignments for other fresh slots remain
            # stable, but the continued attempt itself does not consume that fresh
            # candidate geometry. Legacy mode retains its historical lifecycle.
            if context.attempt_rng is not None and has_continuation:
                continuation_atoms = continuation_data[attempt_id]
                continuation_vector = (
                    context.attempt_rng.numpy("continuation_tiny_displacement")
                    .standard_normal((len(continuation_atoms), 3)) * 1e-10
                )
                apply_continuation(
                    context,
                    continuation_atoms,
                    {
                        "displacement_vector": continuation_vector,
                        "method": "vector",
                    },
                )
                _log_rng_provenance(context, "continuation")
            else:
                if context.atoms is None:
                    _record_generation_failure(context)
                    continue

                apply_generated_metadata(context)

                # Preserve the predecessor legacy stream position exactly.
                if has_continuation:
                    continuation_atoms = continuation_data[attempt_id]
                    continuation_vector = (
                        np.random.randn(len(continuation_atoms), 3) * 1e-10
                    )
                    apply_continuation(
                        context,
                        continuation_atoms,
                        {
                            "displacement_vector": continuation_vector,
                            "method": "vector",
                        },
                    )

            # File naming is deterministic and historically occurred before the
            # per-attempt try/except. Keep that lifecycle boundary unchanged.
            configure_temp_files(context)

            try:
                _setup_and_execute_attempt(context)
                _materialize_attempt_result(context)
                _record_attempt_success(context, writer)
                any_attempt_succeeded = True
            except Exception as exc:
                _record_attempt_exception(
                    context, exc, traceback.format_exc()
                )

    # Track consecutive structure-level errors for worker health.
    if consecutive_errors is not None:
        if any_attempt_succeeded:
            consecutive_errors[0] = 0
        elif all_attempts_none:
            pass  # Data issue (e.g., no adsorbate atoms), not a worker error.
        else:
            consecutive_errors[0] += 1


# SADDLEMILL_CANONICAL_HISTORY_REFACTOR_20260831_V4
