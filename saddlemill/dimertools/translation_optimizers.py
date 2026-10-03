"""Role-oriented Dimer translation optimizers.

Legacy ASE, guarded ASE-LBFGS, and FIRE-LBFGS classes are re-exported without
numerical changes. ``CanonicalHistoryLBFGSMinModeTranslate`` is an opt-in
translator that reconstructs projected secants from raw physical force history
at every accepted translation state. It forms no dense Hessian and performs no
diagonalization.
"""

from __future__ import annotations

import json
from statistics import median
from time import perf_counter_ns
import warnings

import numpy as np

from saddlemill.dimertools.force_history import normalise_tokens
from saddlemill.dimertools.dense_bfgs import safe_cosine, safe_max_atom_norm, safe_norm, safe_scale_to_max_atom
from saddlemill.dimertools.qn_deep_diagnostics import (
    rigid_translation_basis, project_rigid, adaptive_shifted_lbfgs, shifted_lbfgs_solve, translation_ratio,
)
from saddlemill.dimertools.lbfgs_state_dump import step_selected

from saddlemill.dimertools.lbfgs_dimer import (
    DiagnosticMinModeTranslate,
    HybridMinModeTranslate,
    LBFGSMinModeTranslate,
)
from saddlemill.dimertools.quasi_newton import (
    ProjectionSnapshot,
    direction_ablation_metrics,
    reconstruct_lbfgs,
)


from saddlemill.dimertools.canonical_diagnostics import (
    CANONICAL_DIAGNOSTIC_FIELDS,
    REGULARIZATION_NOT_APPLICABLE,
    history_summary_fields,
    regularization_diagnostic_fields,
)



def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _max_atom_norm(value) -> float:
    return safe_max_atom_norm(value)


def _cosine(a, b):
    return safe_cosine(a, b)

def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
    return int(ordered[index])


def _flatten_pair(prefix: str, metrics: dict[str, object]) -> dict[str, object]:
    return {
        f"canonical_latest_{prefix}_s_norm": metrics.get("s_norm", ""),
        f"canonical_latest_{prefix}_y_norm": metrics.get("y_norm", ""),
        f"canonical_latest_{prefix}_s_dot_y": metrics.get("s_dot_y", ""),
        f"canonical_latest_{prefix}_secant_curvature": metrics.get(
            "secant_curvature", ""
        ),
        f"canonical_latest_{prefix}_secant_cosine": metrics.get(
            "secant_cosine", ""
        ),
    }



def _secant_trial_fraction(current_directional_force: float, trial_directional_force: float) -> tuple[float, bool]:
    """Return an interior directional-force secant root, else the full trial."""
    f0=float(current_directional_force); f1=float(trial_directional_force)
    denom=f1-f0
    if np.isfinite(f0) and np.isfinite(f1) and np.isfinite(denom) and abs(denom)>1.0e-30:
        root=-f0/denom
        if 0.0 < root < 1.0:
            return float(root), True
    return 1.0, False

class CanonicalHistoryLBFGSMinModeTranslate(DiagnosticMinModeTranslate):
    """Reproject raw history under the current mode and apply L-BFGS."""

    def __init__(
        self,
        dimeratoms,
        logfile="-",
        trajectory=None,
        lbfgs_options=None,
        history_options=None,
    ) -> None:
        super().__init__(dimeratoms, logfile=logfile, trajectory=trajectory)
        lbfgs = dict(lbfgs_options or {})
        history = dict(history_options or {})
        self.alpha = float(lbfgs.get("alpha", lbfgs.get("initial_hessian", 70.0)))
        self.memory = int(lbfgs.get("memory", 10))
        self.damping = float(lbfgs.get("damping", 1.0))
        if self.alpha <= 0.0:
            raise ValueError("canonical translation alpha must be > 0")
        if self.memory < 1:
            raise ValueError("canonical translation memory must be >= 1")
        if self.damping <= 0.0:
            raise ValueError("canonical translation damping must be > 0")

        self.history_options = history
        self.pair_sources = " ".join(
            normalise_tokens(history.get("pair_sources", "center_center"))
        )
        self.record_sources = " ".join(
            normalise_tokens(history.get("record_sources", "center rotation"))
        )
        self.projection_policy = str(
            history.get("projection_policy", "current_snapshot")
        ).strip().lower()
        self.kappa_projection_policy = str(
            history.get(
                "kappa_projection_policy", "fixed_current_center_coefficients"
            )
        ).strip().lower()
        self.pair_safeguard_requested = str(
            history.get("pair_safeguard_requested", history.get("pair_safeguard", "skip"))
        ).strip().lower()
        self.pair_safeguard = str(
            history.get("pair_safeguard_resolved", history.get("pair_safeguard", "skip"))
        ).strip().lower()
        self.pair_order = str(history.get("pair_order", "acquisition")).strip().lower()
        self.curvature_floor = float(history.get("curvature_floor", 1.0e-3))
        self.curvature_epsilon = float(history.get("curvature_epsilon", 1.0e-12))
        self.powell_eta = float(history.get("powell_eta", 0.2))
        self.cosine_threshold = history.get("cosine_threshold", None)
        if self.cosine_threshold is not None:
            self.cosine_threshold = float(self.cosine_threshold)
        self.cautious_epsilon = float(history.get("cautious_epsilon", 1.0e-6))
        self.cautious_alpha = float(history.get("cautious_alpha", 1.0))
        self.rigid_translation_projection = bool(history.get("rigid_translation_projection", False))
        self.deep_qn_diagnostics = bool(history.get("deep_qn_diagnostics", False))
        self.diagnostic_shift_mus = history.get("diagnostic_shift_mus", "0.01 0.1 1 10")
        self.dynamic_h0 = bool(history.get("dynamic_h0", False))
        self.max_pairs = int(history.get("max_pairs", 0))
        self.reconstruction_model = str(lbfgs.get("reconstruction_model", "lbfgs")).strip().lower()
        self.bfgs_update = str(lbfgs.get("bfgs_update", "sequential")).strip().lower()
        self.dense_bfgs_diagnostic = bool(lbfgs.get("dense_bfgs_diagnostic", False))
        self.translation_regularization = str(lbfgs.get("regularization", "off")).strip().lower()
        self.translation_regularization_mu = float(lbfgs.get("regularization_mu", 1.0))
        self.translation_regularization_radius = float(lbfgs.get("regularization_radius", 0.1))
        self.translation_regularization_tolerance = float(lbfgs.get("regularization_tolerance", 1.0e-8))
        self.translation_trial_step = str(lbfgs.get("trial_step", "off")).strip().lower()
        if self.translation_trial_step not in {"off", "secant_force_root"}:
            raise ValueError("translation trial_step must be off or secant_force_root")
        self.last_qn_detail_payload = {}
        self.lbfgs_state_dump = bool(lbfgs.get("state_dump", False))
        self.lbfgs_state_dump_steps = str(lbfgs.get("state_dump_steps", "all"))
        self.lbfgs_state_dump_trace = bool(
            lbfgs.get("state_dump_two_loop_trace", True)
        )
        self.lbfgs_state_dump_directory = str(
            lbfgs.get("state_dump_directory", "")
        ).strip()
        self.last_lbfgs_state_dump = None
        from saddlemill.dimertools.wave_b_shadow import qn_shadow_options_from_dimeratoms
        self.qn_shadow_options = qn_shadow_options_from_dimeratoms(dimeratoms)
        self.last_qn_shadow_row = None
        self.last_qn_shadow_matrix = None
        self.qn_shadow_cumulative_ns = 0
        if self.rigid_translation_projection:
            atoms_obj = getattr(dimeratoms, "atoms", None)
            constraints = [] if atoms_obj is None else list(getattr(atoms_obj, "constraints", []) or [])
            if constraints:
                raise ValueError("rigid_translation_projection is only implemented for unconstrained Cartesian translators")

        if self.reconstruction_model not in {"lbfgs", "reconstructed_bfgs"}:
            raise ValueError("canonical translation reconstruction_model must be lbfgs or reconstructed_bfgs")
        if self.bfgs_update not in {"sequential", "multisecant"}:
            raise ValueError("canonical translation bfgs_update must be sequential or multisecant")
        requested_projection_validation = str(
            history.get("projection_validation", "off")
        ).strip().lower()
        # ``error`` was the historical development-time default.  Keep old
        # generated configs valid but make them nonfatal and silent so queued
        # production jobs do not need regeneration.  ``fatal`` is the only
        # mode that may terminate a search on this diagnostic mismatch.
        self.projection_validation = (
            "off" if requested_projection_validation == "error"
            else requested_projection_validation
        )
        if self.projection_validation not in {"off", "warn", "fatal"}:
            raise ValueError(
                "projection_validation must be off, warn, fatal, or legacy error"
            )
        self.projection_validation_tolerance = float(
            history.get("projection_validation_tolerance", 1.0e-8)
        )
        self.reuse_ablation = bool(history.get("reuse_ablation", True))
        self.reconstruction_timing = bool(
            history.get("reconstruction_timing", True)
        )
        self.maximum_translation = float(
            self.dimeratoms.control.get_parameter("maximum_translation")
        )
        self._reconstruction_times: list[int] = []
        self._canonical_shadow_times: list[int] = []
        self._canonical_recording_times: list[int] = []
        self._canonical_overhead_times: list[int] = []
        self._canonical_fallbacks = 0
        self._canonical_peak_states = 0
        self._canonical_peak_observations = 0
        self._canonical_peak_bytes = 0
        self._canonical_pair_candidates_total = 0
        self._canonical_pairs_used_total = 0
        self._canonical_pairs_rejected_total = 0
        self._canonical_pairs_damped_total = 0
        self._canonical_optimizer_start_ns = perf_counter_ns()

        if not hasattr(dimeratoms, "canonical_force_history"):
            raise TypeError(
                "canonical_lbfgs requires a history-enabled MinModeAtoms object"
            )
        if not hasattr(dimeratoms, "physical_force_evaluator"):
            raise TypeError(
                "canonical_lbfgs requires a PhysicalForceEvaluator"
            )
        if self.translation_trial_step != "off":
            history_obj = self.dimeratoms.canonical_force_history
            if not history_obj.observes_source("translation_trial", "translation_trial"):
                raise ValueError(
                    "translation_trial_step requires canonical record_sources to include "
                    "translation_trial so the real trial force is preserved in the force bank"
                )

    def _projected_force_validation(self, snapshot, live_force):
        state = self.dimeratoms.canonical_force_history.current
        if state is None:
            raise RuntimeError("canonical history has no current center state")
        if state.center is None or not state.center.is_physical:
            raise RuntimeError(
                "canonical history current state has no physical center observation"
            )
        rebuilt = snapshot.project(state.center.forces)
        difference = rebuilt - np.asarray(live_force, dtype=float)
        maximum = float(np.max(np.abs(difference), initial=0.0))
        difference_norm = float(np.linalg.norm(difference))
        ok = maximum <= self.projection_validation_tolerance
        if not ok and self.projection_validation != "off":
            message = (
                "Canonical current-snapshot projection does not match the live "
                f"MinModeAtoms force: max_abs={maximum:.6e}, "
                f"tolerance={self.projection_validation_tolerance:.6e}."
            )
            if self.projection_validation == "fatal":
                raise RuntimeError(message)
            if self.projection_validation == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=3)
        return rebuilt, maximum, difference_norm, ok

    def _metrics_for_legacy_fields(self, result) -> dict[str, object]:
        latest_raw = dict(result.metrics.get("latest_raw_pair", {}) or {})
        latest_stored = dict(result.metrics.get("latest_stored_pair", {}) or {})
        fields = {
            "history_size": int(result.metrics.get("pairs_used", 0)),
            "pairs_accepted_total": int(result.metrics.get("pairs_used", 0)),
            "pairs_rejected_total": int(result.metrics.get("pairs_rejected", 0)),
            "pairs_skipped_total": int(result.metrics.get("pairs_rejected", 0)),
            "pairs_damped_total": int(result.metrics.get("pairs_damped", 0)),
            "pairs_powell_damped_total": int(
                result.metrics.get("pairs_powell_damped", 0)
            ),
            "reset_count": int(result.metrics.get("pair_resets", 0)),
            "worst_sy": result.metrics.get("worst_raw_s_dot_y", ""),
            "last_reset_reason": (
                "canonical_pair_reset"
                if int(result.metrics.get("pair_resets", 0))
                else ""
            ),
            "alpha": self.alpha,
            "initial_inverse_hessian_scale": result.metrics.get(
                "h0_inverse_scale", 1.0 / self.alpha
            ),
            "memory": self.memory,
            "damping": self.damping,
            "curvature_guard": self.pair_safeguard,
            "curvature_floor": self.curvature_floor,
            "powell_eta": self.powell_eta,
            "latest_s_norm": latest_stored.get("s_norm", ""),
            "latest_y_norm": latest_stored.get("y_norm", ""),
            "latest_s_dot_y": latest_stored.get("s_dot_y", ""),
            "latest_secant_curvature": latest_stored.get("secant_curvature", ""),
            "latest_secant_cosine": latest_stored.get("secant_cosine", ""),
            "latest_force_change_norm": latest_stored.get("force_change_norm", ""),
            "latest_raw_s_dot_y": latest_raw.get("s_dot_y", ""),
            "latest_raw_secant_curvature": latest_raw.get("secant_curvature", ""),
            "latest_stored_s_dot_y": latest_stored.get("s_dot_y", ""),
            "latest_stored_secant_curvature": latest_stored.get(
                "secant_curvature", ""
            ),
            "latest_pair_damped": int(
                result.metrics.get("latest_pair_action", "")
                in {"damped", "powell_damped"}
            ),
        }
        fields.update(regularization_diagnostic_fields(result.metrics))
        return fields

    def _canonical_fields(
        self,
        result,
        *,
        validation_max,
        validation_norm,
        validation_ok,
        direction_fallback,
        recording_ns,
        shadow_ns,
        postprocess_ns,
        ablation,
    ) -> dict[str, object]:
        history = self.dimeratoms.canonical_force_history
        summary = history.summary()
        evaluator = self.dimeratoms.physical_force_evaluator
        metrics = result.metrics
        timings = result.timings_ns
        candidates = dict(metrics.get("candidate_pairs_by_source", {}) or {})
        used = dict(metrics.get("used_pairs_by_source", {}) or {})
        latest_raw = dict(metrics.get("latest_raw_pair", {}) or {})
        latest_stored = dict(metrics.get("latest_stored_pair", {}) or {})
        represented = sum(
            int(observation.force_call_delta)
            for observation in history.iter_observations()
        )

        main_ns = int(timings.get("reconstruction_total_ns", 0))
        self._reconstruction_times.append(main_ns)
        self._canonical_shadow_times.append(int(shadow_ns))
        self._canonical_recording_times.append(int(recording_ns))
        overhead_ns = main_ns + int(shadow_ns) + int(recording_ns) + int(postprocess_ns)
        self._canonical_overhead_times.append(overhead_ns)
        self._canonical_peak_states = max(self._canonical_peak_states, int(summary["states_retained"]))
        self._canonical_peak_observations = max(
            self._canonical_peak_observations, int(summary["observations_retained"])
        )
        self._canonical_peak_bytes = max(self._canonical_peak_bytes, int(summary["history_bytes"]))
        self._canonical_pair_candidates_total += int(metrics.get("pair_candidates", 0))
        self._canonical_pairs_used_total += int(metrics.get("pairs_used", 0))
        self._canonical_pairs_rejected_total += int(metrics.get("pairs_rejected", 0))
        self._canonical_pairs_damped_total += int(metrics.get("pairs_damped", 0)) + int(
            metrics.get("pairs_powell_damped", 0)
        )
        values = self._reconstruction_times
        fields: dict[str, object] = {
            "canonical_history_schema": summary["schema"],
            "canonical_history_enabled": 1,
            "canonical_history_record_only": 0,
            "canonical_memory_states": int(history.memory_states),
            "canonical_states_retained": summary["states_retained"],
            "canonical_complete_states_retained": summary["complete_states_retained"],
            "canonical_states_created_total": summary["states_created_total"],
            "canonical_states_dropped_total": summary["states_dropped_total"],
            "canonical_observations_retained": summary["observations_retained"],
            "canonical_observations_recorded_total": summary[
                "observations_recorded_total"
            ],
            "canonical_record_events_total": summary["record_events_total"],
            "canonical_observations_filtered_total": summary[
                "observations_filtered_total"
            ],
            "canonical_observations_deduplicated_total": summary[
                "observations_deduplicated_total"
            ],
            "canonical_probes_dropped_total": summary["probes_dropped_total"],
            "canonical_physical_exact_retained": summary[
                "physical_exact_retained"
            ],
            "canonical_physical_cached_retained": summary[
                "physical_cached_retained"
            ],
            "canonical_derived_retained": summary["derived_retained"],
            "canonical_history_bytes": summary["history_bytes"],
            "canonical_observation_counts_by_family_json": _json(
                summary["observation_counts_by_family"]
            ),
            "canonical_observation_counts_by_source_json": _json(
                summary["observation_counts_by_source"]
            ),
            "canonical_force_calls_represented": represented,
            "canonical_record_sources": self.record_sources,
            "canonical_pair_sources": self.pair_sources,
            "canonical_pair_order": self.pair_order,
            "canonical_projection_policy": metrics.get("projection_policy", ""),
            "canonical_kappa_projection_policy": metrics.get(
                "kappa_projection_policy", ""
            ),
            "canonical_projection_branch": metrics.get("branch", ""),
            "canonical_projection_regime": metrics.get("regime", ""),
            "canonical_projection_gamma_1": metrics.get("gamma_1", ""),
            "canonical_projection_gamma_2": metrics.get("gamma_2", ""),
            "canonical_projection_active_atoms": metrics.get("active_atoms", ""),
            "canonical_projection_validation": self.projection_validation,
            "canonical_projection_validation_max_abs": validation_max,
            "canonical_projection_validation_norm": validation_norm,
            "canonical_projection_validation_ok": int(validation_ok),
            "canonical_pair_safeguard_requested": self.pair_safeguard_requested,
            "canonical_pair_safeguard": self.pair_safeguard,
            "canonical_reconstruction_model": metrics.get("reconstruction_model", self.reconstruction_model),
            "canonical_bfgs_update": metrics.get("bfgs_update", self.bfgs_update),
            "canonical_dense_bfgs_diagnostic": metrics.get("dense_bfgs_diagnostic", int(self.dense_bfgs_diagnostic)),
            "canonical_dense_bfgs_fallback_reason": metrics.get("dense_bfgs_fallback_reason", ""),
            "canonical_curvature_floor": self.curvature_floor,
            "canonical_curvature_epsilon": self.curvature_epsilon,
            "canonical_powell_eta": self.powell_eta,
            "canonical_max_pairs": self.max_pairs,
            "canonical_cosine_threshold": "" if self.cosine_threshold is None else self.cosine_threshold,
            "canonical_pair_candidates": metrics.get("pair_candidates", 0),
            "canonical_pair_candidates_after_max_pairs": metrics.get("pair_candidates_after_max_pairs", 0),
            "canonical_pairs_admissible_before_max_pairs": metrics.get("pairs_admissible_before_max_pairs", 0),
            "canonical_pairs_removed_by_max_pairs": metrics.get("pairs_removed_by_max_pairs", 0),
            "canonical_pairs_used": metrics.get("pairs_used", 0),
            "canonical_pairs_rejected_by_curvature": metrics.get("pairs_rejected_by_curvature", 0),
            "canonical_pairs_rejected_by_cosine": metrics.get("pairs_rejected_by_cosine", 0),
            "canonical_accepted_curvature_min": metrics.get("accepted_curvature_min", ""),
            "canonical_accepted_curvature_median": metrics.get("accepted_curvature_median", ""),
            "canonical_accepted_curvature_max": metrics.get("accepted_curvature_max", ""),
            "canonical_accepted_cosine_min": metrics.get("accepted_cosine_min", ""),
            "canonical_accepted_cosine_median": metrics.get("accepted_cosine_median", ""),
            "canonical_accepted_cosine_max": metrics.get("accepted_cosine_max", ""),
            "canonical_pairs_rejected": metrics.get("pairs_rejected", 0),
            "canonical_pairs_damped": metrics.get("pairs_damped", 0),
            "canonical_pairs_powell_damped": metrics.get(
                "pairs_powell_damped", 0
            ),
            "canonical_pair_resets": metrics.get("pair_resets", 0),
            "canonical_candidate_pairs_by_source_json": _json(candidates),
            "canonical_used_pairs_by_source_json": _json(used),
            "canonical_rejection_reasons_json": _json(
                metrics.get("rejection_reasons", {})
            ),
            "canonical_center_center_candidates": candidates.get(
                "center_center", 0
            ),
            "canonical_center_center_used": used.get("center_center", 0),
            "canonical_center_rotation_candidates": candidates.get(
                "center_rotation", 0
            ),
            "canonical_center_rotation_used": used.get("center_rotation", 0),
            "canonical_center_lanczos_candidates": candidates.get(
                "center_lanczos", 0
            ),
            "canonical_center_lanczos_used": used.get("center_lanczos", 0),
            "canonical_center_davidson_candidates": candidates.get(
                "center_davidson", 0
            ),
            "canonical_center_davidson_used": used.get("center_davidson", 0),
            "canonical_center_reference_hessian_candidates": candidates.get(
                "center_reference_hessian", 0
            ),
            "canonical_center_reference_hessian_used": used.get(
                "center_reference_hessian", 0
            ),
            "canonical_latest_pair_source": metrics.get("latest_pair_source", ""),
            "canonical_latest_pair_action": metrics.get("latest_pair_action", ""),
            "canonical_latest_pair_rejection_reason": metrics.get(
                "latest_pair_rejection_reason", ""
            ),
            "canonical_worst_raw_s_dot_y": metrics.get(
                "worst_raw_s_dot_y", ""
            ),
            "canonical_worst_stored_s_dot_y": metrics.get(
                "worst_stored_s_dot_y", ""
            ),
            "canonical_h0_inverse_scale": metrics.get("h0_inverse_scale", ""),
            "canonical_direction_norm": metrics.get("direction_norm", ""),
            "canonical_direction_max_atom_norm": metrics.get(
                "direction_max_atom_norm", ""
            ),
            "canonical_direction_force_cosine": metrics.get(
                "direction_force_cosine", ""
            ),
            "canonical_direction_fallback": direction_fallback,
            "canonical_selection_ns": timings.get("selection_ns", 0),
            "canonical_projection_ns": timings.get("projection_ns", 0),
            "canonical_pair_build_ns": timings.get("pair_build_ns", 0),
            "canonical_safeguard_ns": timings.get("safeguard_ns", 0),
            "canonical_two_loop_ns": timings.get("two_loop_ns", 0),
            "canonical_dense_bfgs_ns": timings.get("dense_bfgs_ns", 0),
            "canonical_dense_vs_lbfgs_cosine": metrics.get("dense_vs_lbfgs_cosine", ""),
            "canonical_dense_vs_lbfgs_norm_ratio": metrics.get("dense_vs_lbfgs_norm_ratio", ""),
            "canonical_dense_vs_lbfgs_relative_difference": metrics.get("dense_vs_lbfgs_relative_difference", ""),
            "canonical_dense_min_eigenvalue": metrics.get("dense_min_eigenvalue", ""),
            "canonical_dense_max_eigenvalue": metrics.get("dense_max_eigenvalue", ""),
            "canonical_dense_negative_eigenvalues": metrics.get("dense_negative_eigenvalues", ""),
            "canonical_dense_condition_number": metrics.get("dense_condition_number", ""),
            "canonical_dense_log10_condition": metrics.get("dense_log10_condition", ""),
            "canonical_dense_secant_residual_latest": metrics.get("dense_secant_residual_latest", ""),
            "canonical_dense_secant_residual_median": metrics.get("dense_secant_residual_median", ""),
            "canonical_dense_secant_residual_max": metrics.get("dense_secant_residual_max", ""),
            "canonical_dense_multisecant_blocks_applied": metrics.get("dense_multisecant_blocks_applied", ""),
            "canonical_dense_multisecant_pairs_skipped_rank": metrics.get("dense_multisecant_pairs_skipped_rank", ""),
            "canonical_dense_multisecant_pairs_skipped_curvature": metrics.get("dense_multisecant_pairs_skipped_curvature", ""),
            "canonical_dense_multisecant_blocks_requested": metrics.get("dense_multisecant_blocks_requested", ""),
            "canonical_dense_multisecant_pairs_skipped_invalid": metrics.get("dense_multisecant_pairs_skipped_invalid", ""),
            "canonical_dense_multisecant_blocks_rejected_update": metrics.get("dense_multisecant_blocks_rejected_update", ""),
            "canonical_dense_multisecant_block_size_median": metrics.get("dense_multisecant_block_size_median", ""),
            "canonical_dense_multisecant_block_size_max": metrics.get("dense_multisecant_block_size_max", ""),
            "canonical_dense_multisecant_block_kept_median": metrics.get("dense_multisecant_block_kept_median", ""),
            "canonical_dense_multisecant_y_symmetrization_relative_median": metrics.get("dense_multisecant_y_symmetrization_relative_median", ""),
            "canonical_dense_multisecant_y_symmetrization_relative_max": metrics.get("dense_multisecant_y_symmetrization_relative_max", ""),
            "canonical_dense_multisecant_sty_asymmetry_relative_median": metrics.get("dense_multisecant_sty_asymmetry_relative_median", ""),
            "canonical_dense_multisecant_sty_asymmetry_relative_max": metrics.get("dense_multisecant_sty_asymmetry_relative_max", ""),
            "canonical_dense_multisecant_sts_condition_median": metrics.get("dense_multisecant_sts_condition_median", ""),
            "canonical_dense_multisecant_sts_condition_max": metrics.get("dense_multisecant_sts_condition_max", ""),
            "canonical_dense_multisecant_yts_condition_median": metrics.get("dense_multisecant_yts_condition_median", ""),
            "canonical_dense_multisecant_yts_condition_max": metrics.get("dense_multisecant_yts_condition_max", ""),
            "canonical_dense_multisecant_stbs_condition_median": metrics.get("dense_multisecant_stbs_condition_median", ""),
            "canonical_dense_multisecant_stbs_condition_max": metrics.get("dense_multisecant_stbs_condition_max", ""),
            "canonical_dense_multisecant_sty_original_min_eigenvalue_median": metrics.get("dense_multisecant_sty_original_min_eigenvalue_median", ""),
            "canonical_dense_multisecant_sty_original_min_eigenvalue_max": metrics.get("dense_multisecant_sty_original_min_eigenvalue_max", ""),
            "canonical_dense_multisecant_sty_adjusted_min_eigenvalue_median": metrics.get("dense_multisecant_sty_adjusted_min_eigenvalue_median", ""),
            "canonical_dense_multisecant_sty_adjusted_min_eigenvalue_max": metrics.get("dense_multisecant_sty_adjusted_min_eigenvalue_max", ""),
            "canonical_dense_multisecant_residual_original_y_median": metrics.get("dense_multisecant_residual_original_y_median", ""),
            "canonical_dense_multisecant_residual_original_y_max": metrics.get("dense_multisecant_residual_original_y_max", ""),
            "canonical_dense_multisecant_residual_adjusted_y_median": metrics.get("dense_multisecant_residual_adjusted_y_median", ""),
            "canonical_dense_multisecant_residual_adjusted_y_max": metrics.get("dense_multisecant_residual_adjusted_y_max", ""),
            "canonical_dense_pairs_requested": metrics.get("dense_pairs_requested", ""),
            "canonical_dense_pairs_applied": metrics.get("dense_pairs_applied", ""),
            "canonical_dense_invalid_reason": metrics.get("dense_invalid_reason", ""),
            "canonical_dense_reconstruction_ns": metrics.get("dense_reconstruction_ns", ""),
            "canonical_postprocess_ns": int(postprocess_ns),
            "canonical_reconstruction_total_ns": main_ns,
            "canonical_shadow_reconstruction_ns": shadow_ns,
            "canonical_recording_ns_since_previous_step": recording_ns,
            "canonical_recording_ns_total": evaluator.recording_ns_total,
            "canonical_overhead_ns_this_step": overhead_ns,
            "canonical_reconstruction_calls": len(values),
            "canonical_reconstruction_cumulative_ns": sum(values),
            "canonical_reconstruction_mean_ns": int(sum(values) / len(values)),
            "canonical_reconstruction_median_ns": int(median(values)),
            "canonical_reconstruction_p95_ns": _p95(values),
            "canonical_reconstruction_max_ns": max(values),
            "canonical_shadow_cumulative_ns": sum(self._canonical_shadow_times),
            "canonical_recording_cumulative_ns": sum(self._canonical_recording_times),
            "canonical_overhead_cumulative_ns": sum(self._canonical_overhead_times),
            "canonical_overhead_mean_ns": int(
                sum(self._canonical_overhead_times) / len(self._canonical_overhead_times)
            ),
            "canonical_optimizer_elapsed_ns": max(
                0, perf_counter_ns() - self._canonical_optimizer_start_ns
            ),
            "canonical_overhead_fraction_of_optimizer_wall": (
                ""
                if perf_counter_ns() <= self._canonical_optimizer_start_ns
                else sum(self._canonical_overhead_times)
                / (perf_counter_ns() - self._canonical_optimizer_start_ns)
            ),
            "canonical_direction_fallbacks_total": self._canonical_fallbacks,
            "canonical_peak_states_retained": self._canonical_peak_states,
            "canonical_peak_observations_retained": self._canonical_peak_observations,
            "canonical_peak_history_bytes": self._canonical_peak_bytes,
            "canonical_total_pair_candidates": self._canonical_pair_candidates_total,
            "canonical_total_pairs_used": self._canonical_pairs_used_total,
            "canonical_total_pairs_rejected": self._canonical_pairs_rejected_total,
            "canonical_total_pairs_damped": self._canonical_pairs_damped_total,
            "canonical_reuse_ablation": int(self.reuse_ablation),
            "canonical_reuse_direction_cosine": ablation.get(
                "reuse_direction_cosine", ""
            ),
            "canonical_reuse_direction_delta_norm": ablation.get(
                "reuse_direction_delta_norm", ""
            ),
            "canonical_reuse_direction_norm_ratio": ablation.get(
                "reuse_direction_norm_ratio", ""
            ),
            "canonical_reuse_pairs_added": ablation.get("reuse_pairs_added", ""),
            "canonical_reuse_pairs_survived": ablation.get(
                "reuse_pairs_survived", ""
            ),
            "canonical_center_only_pairs_used": ablation.get(
                "center_only_pairs_used", ""
            ),
            "canonical_center_only_reconstruction_ns": ablation.get(
                "center_only_reconstruction_ns", ""
            ),
        }
        fields.update(regularization_diagnostic_fields(metrics))
        fields.update(_flatten_pair("raw", latest_raw))
        fields.update(_flatten_pair("stored", latest_stored))
        return fields

    def step(self, forces=None):
        if forces is None:
            forces = self.dimeratoms.get_forces()
        force = np.asarray(forces, dtype=float)
        positions = np.asarray(self.dimeratoms.get_positions(), dtype=float)
        finalizer = getattr(self.dimeratoms, "finalize_canonical_state", None)
        if callable(finalizer):
            finalizer()

        snapshot = ProjectionSnapshot.from_dimeratoms(
            self.dimeratoms,
            policy=self.projection_policy,
            kappa_policy=self.kappa_projection_policy,
        )
        canonical_force, validation_max, validation_norm, validation_ok = (
            self._projected_force_validation(snapshot, force)
        )

        accepted_translation_step = int(self.nsteps) + 1
        dump_this_step = bool(
            self.lbfgs_state_dump
            and step_selected(self.lbfgs_state_dump_steps, accepted_translation_step)
        )
        result = reconstruct_lbfgs(
            self.dimeratoms.canonical_force_history,
            snapshot,
            canonical_force,
            pair_sources=self.pair_sources,
            initial_hessian=self.alpha,
            dynamic_h0=self.dynamic_h0,
            safeguard=self.pair_safeguard,
            curvature_floor=self.curvature_floor,
            curvature_epsilon=self.curvature_epsilon,
            powell_eta=self.powell_eta,
            cosine_threshold=self.cosine_threshold,
            max_pairs=self.max_pairs,
            rigid_translation_projection=self.rigid_translation_projection,
            deep_diagnostics=self.deep_qn_diagnostics,
            diagnostic_shift_mus=self.diagnostic_shift_mus,
            cautious_epsilon=self.cautious_epsilon,
            cautious_alpha=self.cautious_alpha,
            reconstruction_model=self.reconstruction_model,
            bfgs_update=self.bfgs_update,
            dense_bfgs_diagnostic=self.dense_bfgs_diagnostic,
            exact_state_dump=(dump_this_step or bool(self.qn_shadow_options.get("enabled", False))),
            exact_two_loop_trace=self.lbfgs_state_dump_trace,
        )

        self.last_qn_detail_payload = dict(result.detail_payload or {})
        if self.last_qn_detail_payload:
            self.last_qn_detail_payload["maximum_translation"] = self.maximum_translation
        self.last_lbfgs_state_dump = (
            dict(result.exact_state_payload or {}) if dump_this_step else None
        )
        if self.last_lbfgs_state_dump is not None:
            history_obj = self.dimeratoms.canonical_force_history
            self.last_lbfgs_state_dump.update({
                "accepted_translation_step": np.asarray(
                    accepted_translation_step, dtype=np.int64
                ),
                "positions_before": np.asarray(positions, dtype=np.float64),
                "live_force_argument": np.asarray(
                    force, dtype=np.float64
                ).reshape(-1),
                "canonical_projected_force": np.asarray(
                    canonical_force, dtype=np.float64
                ).reshape(-1),
                "maximum_translation": np.asarray(
                    self.maximum_translation, dtype=np.float64
                ),
                "damping": np.asarray(self.damping, dtype=np.float64),
                "production_reconstruction_model": np.asarray(
                    self.reconstruction_model
                ),
                "production_bfgs_update": np.asarray(self.bfgs_update),
                "pair_order": np.asarray(self.pair_order),
                "memory_states": np.asarray(
                    int(history_obj.memory_states), dtype=np.int64
                ),
                "max_probes_per_state": np.asarray(
                    int(history_obj.max_probes_per_state), dtype=np.int64
                ),
                "selected_model_uncapped_direction": np.asarray(
                    result.direction, dtype=np.float64
                ).reshape(-1),
            })

        self.last_qn_shadow_row = None
        self.last_qn_shadow_matrix = None
        if bool(self.qn_shadow_options.get("enabled", False)):
            from saddlemill.dimertools.wave_b_shadow import run_shadow
            payload = dict(result.exact_state_payload or {})
            payload.update({
                "accepted_translation_step": np.asarray(accepted_translation_step, dtype=np.int64),
                "maximum_translation": np.asarray(self.maximum_translation, dtype=np.float64),
                "damping": np.asarray(self.damping, dtype=np.float64),
                "production_reconstruction_model": np.asarray(self.reconstruction_model),
                "production_bfgs_update": np.asarray(self.bfgs_update),
            })
            shadow_result, shadow_elapsed = run_shadow(
                payload, options=self.qn_shadow_options,
                consumer="saddle_translation",
                residual_kind="mmf_effective_force",
                force_interpretation="effective_force",
                model_type="effective_residual_bfgs", owner=self.dimeratoms,
            )
            if shadow_result is not None:
                self.last_qn_shadow_row = dict(shadow_result.row)
                self.last_qn_shadow_matrix = shadow_result.matrix
                self.qn_shadow_cumulative_ns += int(shadow_elapsed)
                self.last_qn_detail_payload["t08_shadow"] = dict(shadow_result.row)

        shadow_ns = 0
        ablation: dict[str, object] = {}
        if self.reuse_ablation and set(normalise_tokens(self.pair_sources)) != {
            "center_center"
        }:
            center_only = reconstruct_lbfgs(
                self.dimeratoms.canonical_force_history,
                snapshot,
                canonical_force,
                pair_sources="center_center",
                initial_hessian=self.alpha,
                dynamic_h0=self.dynamic_h0,
                safeguard=self.pair_safeguard,
                curvature_floor=self.curvature_floor,
                curvature_epsilon=self.curvature_epsilon,
                powell_eta=self.powell_eta,
                cosine_threshold=self.cosine_threshold,
                max_pairs=self.max_pairs,
                rigid_translation_projection=self.rigid_translation_projection,
                deep_diagnostics=self.deep_qn_diagnostics,
                diagnostic_shift_mus=self.diagnostic_shift_mus,
                cautious_epsilon=self.cautious_epsilon,
                cautious_alpha=self.cautious_alpha,
                reconstruction_model=self.reconstruction_model,
                bfgs_update=self.bfgs_update,
                dense_bfgs_diagnostic=self.dense_bfgs_diagnostic,
            )
            shadow_ns = int(center_only.timings_ns["reconstruction_total_ns"])
            ablation = direction_ablation_metrics(result, center_only)
            if self.last_qn_detail_payload and self.deep_qn_diagnostics:
                self.last_qn_detail_payload["center_only_shadow"] = center_only.detail_payload
                try:
                    rotation_only = reconstruct_lbfgs(
                        self.dimeratoms.canonical_force_history, snapshot, canonical_force,
                        pair_sources="center_rotation", initial_hessian=self.alpha,
                        dynamic_h0=self.dynamic_h0, safeguard=self.pair_safeguard,
                        curvature_floor=self.curvature_floor, curvature_epsilon=self.curvature_epsilon,
                        powell_eta=self.powell_eta, cosine_threshold=self.cosine_threshold, max_pairs=self.max_pairs,
                        rigid_translation_projection=self.rigid_translation_projection,
                        deep_diagnostics=True, diagnostic_shift_mus=self.diagnostic_shift_mus,
                        cautious_epsilon=self.cautious_epsilon, cautious_alpha=self.cautious_alpha,
                        reconstruction_model=self.reconstruction_model, bfgs_update=self.bfgs_update,
                        dense_bfgs_diagnostic=self.dense_bfgs_diagnostic,
                    )
                    self.last_qn_detail_payload["center_rotation_only_shadow"] = rotation_only.detail_payload
                    self.last_qn_detail_payload["center_rotation_only_direction_norm"] = rotation_only.metrics.get("direction_norm", "")
                    self.last_qn_detail_payload["center_rotation_only_direction_max_atom_norm"] = rotation_only.metrics.get("direction_max_atom_norm", "")
                except Exception as exc:
                    self.last_qn_detail_payload["center_rotation_only_shadow_error"] = type(exc).__name__+":"+str(exc)

        postprocess_start = perf_counter_ns()
        direction = np.asarray(result.direction, dtype=float)
        regularization_metrics = {
            "translation_regularization": self.translation_regularization,
            "translation_regularization_applied": 0,
            "translation_regularization_fallback": REGULARIZATION_NOT_APPLICABLE,
            "translation_regularization_mu": REGULARIZATION_NOT_APPLICABLE,
            "translation_regularization_iterations": REGULARIZATION_NOT_APPLICABLE,
            "translation_regularized_norm": REGULARIZATION_NOT_APPLICABLE,
            "translation_regularization_suppression_ratio": REGULARIZATION_NOT_APPLICABLE,
            "translation_regularization_direction_cosine": REGULARIZATION_NOT_APPLICABLE,
            "translation_regularization_final_maxstep_scale": REGULARIZATION_NOT_APPLICABLE,
            "translation_regularization_final_max_atom_norm": REGULARIZATION_NOT_APPLICABLE,
        }
        if self.translation_regularization in {"shifted_lbfgs_fixed", "shifted_lbfgs_trust", "shifted_trust_region"} and result.accepted_pairs:
            try:
                force_flat = np.asarray(result.projected_current_force, dtype=float).reshape(-1)
                if self.translation_regularization == "shifted_lbfgs_fixed":
                    mu = self.translation_regularization_mu
                    regularization_metrics["translation_regularization_mu"] = float(mu)
                    regularization_metrics["translation_regularization_iterations"] = 0
                    reg = shifted_lbfgs_solve(force_flat, result.accepted_pairs, self.alpha, mu, self.dynamic_h0)
                    iterations = 0
                else:
                    trust = adaptive_shifted_lbfgs(force_flat, result.accepted_pairs, self.alpha, self.translation_regularization_radius, self.dynamic_h0, self.translation_regularization_tolerance)
                    mu, iterations = trust.mu, trust.iterations
                    regularization_metrics["translation_regularization_mu"] = float(mu)
                    regularization_metrics["translation_regularization_iterations"] = int(iterations)
                    if trust.fallback:
                        raise np.linalg.LinAlgError(trust.fallback)
                    reg = trust.direction
                rawflat=np.asarray(direction,float).reshape(-1); regflat=np.asarray(reg,float).reshape(-1)
                den=safe_norm(rawflat)*safe_norm(regflat)
                regularization_metrics.update({
                    "translation_regularization_applied": 1,
                    "translation_regularization_fallback": REGULARIZATION_NOT_APPLICABLE,
                    "translation_regularized_norm": safe_norm(regflat),
                    "translation_regularization_suppression_ratio": ("" if safe_norm(rawflat)<=1e-300 else safe_norm(regflat)/safe_norm(rawflat)),
                    "translation_regularization_direction_cosine": ("" if den<=0 else float(rawflat@regflat/den)),
                })
                direction = regflat.reshape(direction.shape)
            except Exception as exc:
                regularization_metrics["translation_regularization_fallback"] = type(exc).__name__+":"+str(exc)
        fallback = ""
        alignment = _cosine(direction, canonical_force)
        direction_norm = safe_norm(direction)
        direction_max = _max_atom_norm(direction)
        if (
            not np.all(np.isfinite(direction))
            or not np.isfinite(direction_norm)
            or not np.isfinite(direction_max)
            or direction_norm <= 1.0e-14
            or float(np.vdot(direction, canonical_force).real) <= 0.0
        ):
            direction = canonical_force / self.alpha
            fallback = "scaled_projected_force"
            self._canonical_fallbacks += 1
            alignment = _cosine(direction, canonical_force)

        confiner = getattr(self.dimeratoms, "confine_translation_vector", None)
        if callable(confiner):
            direction = np.asarray(confiner(direction), dtype=float)
        if self.rigid_translation_projection:
            Q = rigid_translation_basis(direction.shape[0])
            direction = project_rigid(direction, Q).reshape(direction.shape)

        raw_step = direction.copy()
        raw_norm = safe_norm(raw_step)
        capped, raw_max, rescaled, clip_scale = safe_scale_to_max_atom(
            raw_step, self.maximum_translation
        )
        if not np.isfinite(clip_scale):
            direction = canonical_force / self.alpha
            if callable(confiner):
                direction = np.asarray(confiner(direction), dtype=float)
            fallback = "nonfinite_derived_norm_scaled_projected_force"
            self._canonical_fallbacks += 1
            raw_step = direction.copy()
            raw_norm = safe_norm(raw_step)
            capped, raw_max, rescaled, clip_scale = safe_scale_to_max_atom(
                raw_step, self.maximum_translation
            )
            if not np.isfinite(clip_scale):
                raise RuntimeError("canonical translation fallback produced invalid max-step scale")
        direction = capped
        clipped = bool(rescaled)
        step_vector = self.damping * direction
        if regularization_metrics["translation_regularization_applied"]:
            regularization_metrics.update({
                "translation_regularization_final_maxstep_scale": float(clip_scale),
                "translation_regularization_final_max_atom_norm": _max_atom_norm(step_vector),
            })
        result.metrics.update(regularization_metrics)
        if self.last_qn_detail_payload:
            self.last_qn_detail_payload.update(regularization_metrics)
        trial_metrics = {
            "translation_trial_step_method": self.translation_trial_step,
            "translation_trial_force_calls": 0,
            "translation_trial_interpolation_used": 0,
            "translation_trial_current_directional_force": "",
            "translation_trial_trial_directional_force": "",
            "translation_trial_fraction": "",
            "translation_trial_bounded_displacement_norm": safe_norm(step_vector),
            "translation_trial_chosen_displacement_norm": safe_norm(step_vector),
        }
        if self.translation_trial_step == "secant_force_root":
            trial_flat = np.asarray(step_vector, dtype=float).reshape(-1)
            trial_norm = safe_norm(trial_flat)
            if trial_norm > 1.0e-14 and np.isfinite(trial_norm):
                unit = trial_flat / trial_norm
                trial_positions = positions + step_vector
                evaluator = self.dimeratoms.physical_force_evaluator
                with evaluator.source(
                    "translation_trial",
                    family="translation_trial",
                    metadata={
                        "accepted_translation_step": accepted_translation_step,
                        "trial_kind": "secant_force_root",
                    },
                ):
                    trial_raw_force = np.asarray(
                        self.dimeratoms.get_forces(real=True, pos=trial_positions),
                        dtype=float,
                    )
                trial_projected_force = snapshot.project(trial_raw_force)
                f_current = float(np.dot(np.asarray(canonical_force).reshape(-1), unit))
                f_trial = float(np.dot(np.asarray(trial_projected_force).reshape(-1), unit))
                fraction, interpolation_used = _secant_trial_fraction(f_current, f_trial)
                step_vector = fraction * step_vector
                direction = step_vector / self.damping
                trial_metrics.update({
                    "translation_trial_force_calls": 1,
                    "translation_trial_interpolation_used": int(interpolation_used),
                    "translation_trial_current_directional_force": f_current,
                    "translation_trial_trial_directional_force": f_trial,
                    "translation_trial_fraction": fraction,
                    "translation_trial_bounded_displacement_norm": trial_norm,
                    "translation_trial_chosen_displacement_norm": safe_norm(step_vector),
                })
        actual_norm = safe_norm(step_vector)
        actual_max = _max_atom_norm(step_vector)
        postprocess_ns = perf_counter_ns() - postprocess_start

        algorithm = ("canonical_lbfgs" if self.reconstruction_model == "lbfgs"
                     else f"canonical_reconstructed_bfgs_{self.bfgs_update}")
        data = self._start_step_diagnostics(
            force, algorithm, hybrid_state="", switch_event=""
        )
        data.update({
            "direction_alignment": alignment,
            "step_clipped": int(clipped),
            "translation_raw_step_norm": raw_norm,
            "translation_raw_step_max": raw_max,
            "translation_actual_step_norm": actual_norm,
            "translation_actual_step_max": actual_max,
            "translation_maxstep": self.maximum_translation,
            "translation_maxstep_rescaled": int(rescaled),
            "translation_clip_scale": clip_scale,
            "translation_damping": self.damping,
            "translation_applied_scale": self.damping * clip_scale,
        })

        data.update(trial_metrics)
        result.metrics.update(trial_metrics)
        if self.last_qn_detail_payload:
            self.last_qn_detail_payload.update(trial_metrics)

        recording_ns = self.dimeratoms.physical_force_evaluator.consume_recording_ns()
        data.update(
            self._canonical_fields(
                result,
                validation_max=validation_max,
                validation_norm=validation_norm,
                validation_ok=validation_ok,
                direction_fallback=fallback,
                recording_ns=recording_ns,
                shadow_ns=shadow_ns,
                postprocess_ns=postprocess_ns,
                ablation=ablation,
            )
        )

        if self.last_lbfgs_state_dump is not None:
            self.last_lbfgs_state_dump.update({
                "production_pre_cap_direction": np.asarray(raw_step, dtype=np.float64).reshape(-1),
                "final_capped_direction": np.asarray(direction, dtype=np.float64).reshape(-1),
                "final_applied_step": np.asarray(step_vector, dtype=np.float64).reshape(-1),
                "positions_after": np.asarray(positions + step_vector, dtype=np.float64),
                "raw_pre_cap_norm": np.asarray(raw_norm, dtype=np.float64),
                "raw_pre_cap_max_atom_norm": np.asarray(raw_max, dtype=np.float64),
                "clip_scale": np.asarray(clip_scale, dtype=np.float64),
                "maxstep_rescaled": np.asarray(bool(rescaled), dtype=np.bool_),
                "direction_fallback": np.asarray(fallback),
            })
        self.dimeratoms.set_positions(positions + step_vector)
        self.f0 = canonical_force.reshape(-1).copy()
        self.r0 = positions.reshape(-1).copy()
        self._last_step_size = actual_norm
        self._finish_step_diagnostics(
            data,
            actual_norm,
            lbfgs_metrics=self._metrics_for_legacy_fields(result),
        )

    def canonical_summary_metrics(self) -> dict[str, object]:
        values = self._reconstruction_times
        overhead = self._canonical_overhead_times
        elapsed_ns = max(0, perf_counter_ns() - self._canonical_optimizer_start_ns)
        result = history_summary_fields(
            self.dimeratoms,
            record_only=False,
            history_options=self.history_options,
        )
        result.update(
            {
                "canonical_history_schema": "saddlemill_canonical_force_history_v1",
                "canonical_history_enabled": 1,
                "canonical_history_record_only": 0,
                "canonical_record_sources": self.record_sources,
                "canonical_pair_sources": self.pair_sources,
                "canonical_pair_order": self.pair_order,
                "canonical_pair_safeguard_requested": self.pair_safeguard_requested,
                "canonical_pair_safeguard": self.pair_safeguard,
                "canonical_reconstruction_model": self.reconstruction_model,
                "canonical_bfgs_update": self.bfgs_update,
                "canonical_dense_bfgs_diagnostic": int(self.dense_bfgs_diagnostic),
                "canonical_reconstruction_calls": len(values),
                "canonical_reconstruction_cumulative_ns": sum(values),
                "canonical_reconstruction_mean_ns": (
                    int(sum(values) / len(values)) if values else 0
                ),
                "canonical_reconstruction_median_ns": (
                    int(median(values)) if values else 0
                ),
                "canonical_reconstruction_p95_ns": _p95(values),
                "canonical_reconstruction_max_ns": max(values) if values else 0,
                "canonical_shadow_cumulative_ns": sum(self._canonical_shadow_times),
                "canonical_recording_cumulative_ns": sum(
                    self._canonical_recording_times
                ),
                "canonical_overhead_cumulative_ns": sum(overhead),
                "canonical_overhead_mean_ns": (
                    int(sum(overhead) / len(overhead)) if overhead else 0
                ),
                "canonical_optimizer_elapsed_ns": elapsed_ns,
                "canonical_overhead_fraction_of_optimizer_wall": (
                    "" if elapsed_ns <= 0 else sum(overhead) / elapsed_ns
                ),
                "canonical_direction_fallbacks_total": self._canonical_fallbacks,
                "canonical_peak_states_retained": self._canonical_peak_states,
                "canonical_peak_observations_retained": (
                    self._canonical_peak_observations
                ),
                "canonical_peak_history_bytes": self._canonical_peak_bytes,
                "canonical_total_pair_candidates": (
                    self._canonical_pair_candidates_total
                ),
                "canonical_total_pairs_used": self._canonical_pairs_used_total,
                "canonical_total_pairs_rejected": (
                    self._canonical_pairs_rejected_total
                ),
                "canonical_total_pairs_damped": self._canonical_pairs_damped_total,
            }
        )
        return result



class PartitionedLBFGSMinModeTranslate(DiagnosticMinModeTranslate):
    """ASE-facing adapter for legacy P+Q or Q-LBFGS+Dimer-axial translation."""

    def __init__(self, dimeratoms, logfile="-", trajectory=None, *, partitioned_options=None, runtime_state=None):
        super().__init__(dimeratoms, logfile=logfile, trajectory=trajectory)
        from saddlemill.dimertools.partitioned_lbfgs import PartitionedLBFGS
        options = dict(partitioned_options or {})
        if not hasattr(dimeratoms, "canonical_force_history"):
            raise TypeError("partitioned ForceBank translation requires canonical raw force history")
        payload = None if runtime_state is None else runtime_state.get("partitioned_lbfgs_state")
        if payload is None:
            self.core = PartitionedLBFGS(**options)
        else:
            self.core = PartitionedLBFGS.from_state_dict(payload, expected_settings=options)
        dimeratoms._partitioned_lbfgs_state = self.core.to_state_dict()
        self.last_partitioned_diagnostics = {}

    def step(self, f=None):
        # Force/mode calculation follows the ordinary MinModeAtoms route.  The
        # partitioned core never consumes the projected effective MMF force ``f``.
        if f is None:
            f = self.dimeratoms.get_forces()
        projected_force = np.asarray(f, dtype=float)
        positions = np.asarray(self.dimeratoms.get_positions(), dtype=float).copy()
        history = self.dimeratoms.canonical_force_history
        state = history.current
        if state is None or state.center is None or not state.center.is_physical:
            raise RuntimeError("partitioned ForceBank translation requires a current raw physical center observation")
        try:
            mode = np.asarray(self.dimeratoms.get_eigenmode(), dtype=float)
        except Exception:
            mode = np.asarray(self.dimeratoms.eigenmodes[0], dtype=float)
        from saddlemill.dimertools.wave_b_runtime import active_space_for_owner
        space = active_space_for_owner(self.dimeratoms, state.center_positions)
        maximum = float(self.dimeratoms.control.get_parameter("maximum_translation"))
        try:
            entry_calls = int(self.dimeratoms.control.get_counter("forcecalls"))
        except Exception:
            entry_calls = -1
        center_rotation_calls = (
            entry_calls - self._previous_force_calls_after_step
            if entry_calls >= 0 else ""
        )
        result = self.core.step(
            history=history,
            current_observation=state.center,
            mode=mode,
            active_space=space,
            maximum_translation=maximum,
            curvature=state.curvature,
        )
        step = np.asarray(result.step, dtype=float)
        if hasattr(self.dimeratoms, "confine_translation_vector"):
            step = np.asarray(self.dimeratoms.confine_translation_vector(step), dtype=float)
        self.dimeratoms.set_positions(positions + step.reshape(positions.shape))
        self.last_partitioned_diagnostics = dict(result.diagnostics)
        self.dimeratoms._partitioned_lbfgs_state = self.core.to_state_dict()

        # Additive diagnostic logging only.  In particular, real_fmax comes from
        # the already-recorded physical center observation rather than a new
        # ``get_forces(real=True)`` request.
        try:
            after_calls = int(self.dimeratoms.control.get_counter("forcecalls"))
        except Exception:
            after_calls = -1
        diag = result.diagnostics
        curvature = "" if state.curvature is None else float(state.curvature)
        row = {
            "diagnostic_serial": self._diagnostic_serial + 1,
            "accepted_translation_step": int(self.nsteps) + 1,
            "translation_algorithm": str(diag.get("selector", "partitioned_lbfgs")),
            "projected_fmax": safe_max_atom_norm(projected_force),
            "real_fmax": safe_max_atom_norm(state.center.forces),
            "curvature": curvature,
            "translation_regime": getattr(self.dimeratoms, "translation_regime", "standard"),
            "translation_state_key": state.state_uid,
            "step_norm": safe_norm(step),
            "force_calls_step_entry": entry_calls,
            "force_calls_center_and_rotation": center_rotation_calls,
            "force_calls_translation_trial": (
                after_calls - entry_calls if after_calls >= 0 and entry_calls >= 0 else ""
            ),
            "force_calls_cumulative_after_step": after_calls,
        }
        # Reuse the already-existing generic optimizer CSV fields for shifted
        # limited-memory regularization; this is additive diagnostics only and
        # does not alter the output schema.
        for key in (
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
        ):
            row[key] = diag.get(key, "")

        for key in (
            "mode_hv_available",
            "mode_hv_source",
            "mode_hv_family",
            "mode_hv_stencil_id",
            "mode_hv_stencil_scheme",
            "mode_hv_fd_displacement",
            "mode_hv_probe_count",
            "mode_hv_direction_abs_overlap",
            "mode_hv_endpoint_observation_ids_json",
            "mode_hv_norm",
            "mode_qhv_norm",
            "mode_qhv_fraction",
            "mode_qhv_to_abs_curvature",
            "mode_rayleigh_curvature",
            "mode_curvature_reported",
            "mode_curvature_hv",
            "mode_curvature_delta",
            "mode_hv_additional_pes_calls",
            "mode_hv_unavailable_reason",
        ):
            row["partitioned_" + key] = diag.get(key, "")
        self._diagnostic_serial += 1
        self.last_step_diagnostics = row
        self._previous_force_calls_after_step = max(after_calls, 0)


class RFOPRFOMinModeTranslate(DiagnosticMinModeTranslate):
    """shared-runtime ASE-facing adapter for native RFO/P-RFO translation."""

    def __init__(self, dimeratoms, logfile="-", trajectory=None, *, rfo_runtime=None, runtime_state=None):
        super().__init__(dimeratoms, logfile=logfile, trajectory=trajectory)
        if rfo_runtime is None:
            raise TypeError("rfo/prfo/qn_mmf translator requires an RFO translation runtime")
        if runtime_state is not None and runtime_state.get("rfo_runtime_state") is not None:
            rfo_runtime.restore_state(dict(runtime_state["rfo_runtime_state"]))
        self.rfo_runtime = rfo_runtime
        self.last_rfo_result = None
        self.last_rfo_diagnostics = {}
        dimeratoms._rfo_translation_runtime = rfo_runtime
        dimeratoms._rfo_runtime_state = rfo_runtime.state_dict()

    def step(self, f=None):
        if f is None:
            self.dimeratoms.get_forces()
        positions = np.asarray(self.dimeratoms.get_positions(), dtype=float).copy()
        history = getattr(self.dimeratoms, "canonical_force_history", None)
        state = None if history is None else history.current
        if state is None or state.center is None or not state.center.is_physical:
            raise RuntimeError("rfo/prfo/qn_mmf requires a current raw physical center observation")
        controller = getattr(self.dimeratoms, "wave_b_runtime", None)
        model = None if controller is None else controller.physical_hessian
        if model is None:
            raise RuntimeError("rfo/prfo/qn_mmf requires the same-center physical-Hessian physical Hessian model")
        try:
            mode = np.asarray(self.dimeratoms.get_eigenmode(), dtype=float)
        except Exception:
            mode = np.asarray(self.dimeratoms.eigenmodes[0], dtype=float)
        pre_step_pes_calls = int(getattr(history.accounting, "algorithm_pes_calls", 0))
        result = self.rfo_runtime.propose(
            model, state.center, current_mode=mode, pre_step_pes_calls=pre_step_pes_calls
        )
        if not result.success or result.cartesian_step is None:
            raise RuntimeError(f"RFO-family translation failed: {result.failure_reason or result.status}")
        step = np.asarray(result.cartesian_step, dtype=float)
        maximum = float(self.dimeratoms.control.get_parameter("maximum_translation"))
        if str(self.rfo_runtime.settings.get("step_control")) == "ras":
            # Generic RAS is itself the max-per-atom step constraint.  Applying
            # Dimer's historical maximum_translation cap afterward would silently
            # substitute post-hoc scaling and prevent adaptive radii above 0.1 A.
            clipped = False
            scale = 1.0
            cap_policy = "bypassed_generic_ras_is_step_constraint"
        else:
            step, _, clipped, scale = safe_scale_to_max_atom(step, maximum)
            cap_policy = "legacy_post_step_cap_preserved"
        if hasattr(self.dimeratoms, "confine_translation_vector"):
            step = np.asarray(self.dimeratoms.confine_translation_vector(step), dtype=float)
        self.rfo_runtime.note_applied_step(step)
        self.dimeratoms.set_positions(positions + step.reshape(positions.shape))
        self.last_rfo_result = result
        self.last_rfo_diagnostics = {
            **result.metadata(),
            "trust_policy": str(self.rfo_runtime.settings.get("trust_policy", "fixed")),
            "current_trust_radius": self.rfo_runtime.current_radius,
            "last_trust_transition": self.rfo_runtime.last_trust_transition,
            "existing_maximum_translation": maximum,
            "existing_maximum_translation_policy": cap_policy,
            "existing_maximum_translation_clipped": int(bool(clipped)),
            "existing_maximum_translation_scale": float(scale),
        }
        self.dimeratoms._rfo_runtime_state = self.rfo_runtime.state_dict()



__all__ = [
    "CANONICAL_DIAGNOSTIC_FIELDS",
    "CanonicalHistoryLBFGSMinModeTranslate",
    "DiagnosticMinModeTranslate",
    "HybridMinModeTranslate",
    "LBFGSMinModeTranslate",
    "PartitionedLBFGSMinModeTranslate",
    "RFOPRFOMinModeTranslate",
]
