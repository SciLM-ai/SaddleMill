"""Projection and limited-memory quasi-Newton consumers.

The canonical force history remains projection independent. This module freezes
one current MMF/Kappa projection snapshot, builds projected secants from selected
raw endpoint pairs, applies an explicit pair safeguard, and executes an L-BFGS
two-loop recursion. No dense ``3N x 3N`` matrix is formed or diagonalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Optional

import numpy as np

from saddlemill.dimertools.force_history import (
    CanonicalForceHistory,
    ForceObservation,
    PairCandidate,
    normalise_tokens,
)
from saddlemill.dimertools.qn_deep_diagnostics import (
    block_diagnostics, progressive_lbfgs, rigid_translation_basis, project_rigid,
    secant_row, translation_ratio, shifted_lbfgs_solve,
)
from saddlemill.dimertools.lbfgs_state_dump import (
    rigid_translation_basis as exact_rigid_translation_basis,
)
from saddlemill.dimertools.dense_bfgs import (
    DenseSecant,
    compare_directions,
    reconstruct_multisecant_bfgs,
    reconstruct_sequential_bfgs,
    safe_cosine,
    safe_max_atom_norm,
    safe_norm,
)
from saddlemill.dimertools.qn_reconstruction_core import (
    SecantRecord,
    pair_metrics,
    _two_loop,
    _project_candidate_observations,
    _build_raw_candidate_pairs,
    _admit_candidate_pairs,
    _reconstruct_dense_model,
    _select_reconstruction_direction,
    _accepted_records_for_model,
)
from saddlemill.dimertools.qn_reconstruction_diagnostics import (
    _build_deep_diagnostics_payload,
    _build_exact_state_payload,
)

Array = np.ndarray
VALID_SAFEGUARDS = frozenset({"off", "skip", "damp", "powell", "reset", "cautious"})


def _array_n3(value: object, name: str) -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return np.array(arr, dtype=float, copy=True)


def _normalise_mode(value: object) -> Array:
    arr = _array_n3(value, "mode")
    norm = safe_norm(arr)
    if norm <= 1.0e-15 or not np.isfinite(norm):
        raise ValueError("mode norm is approximately zero")
    return arr / norm


def _cosine(a: Array, b: Array) -> float | str:
    return safe_cosine(a, b)


def _max_atom_norm(value: Array) -> float:
    return safe_max_atom_norm(value)


@dataclass(frozen=True)
class ProjectionSnapshot:
    """One force-field definition applied to every retained observation."""

    mode: Array
    curvature: float
    branch: str
    regime: str
    gamma_1: float = 1.0
    gamma_2: float = 1.0
    active_atom_mask: Optional[Array] = None
    policy: str = "current_snapshot"
    kappa_policy: str = "fixed_current_center_coefficients"

    def __post_init__(self) -> None:
        mode = _normalise_mode(self.mode)
        object.__setattr__(self, "mode", mode)
        curvature = float(self.curvature)
        if not np.isfinite(curvature):
            raise ValueError("curvature must be finite")
        object.__setattr__(self, "curvature", curvature)
        branch = str(self.branch).strip().lower()
        regime = str(self.regime).strip().lower()
        if branch not in {"convex", "concave"}:
            raise ValueError(f"unsupported projection branch: {branch!r}")
        if regime not in {"standard", "kappa", "bowl_breakout"}:
            raise ValueError(f"unsupported translation regime: {regime!r}")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "regime", regime)
        gamma_1 = float(self.gamma_1)
        gamma_2 = float(self.gamma_2)
        if not np.isfinite(gamma_1) or not np.isfinite(gamma_2):
            raise ValueError("Kappa projection coefficients must be finite")
        object.__setattr__(self, "gamma_1", gamma_1)
        object.__setattr__(self, "gamma_2", gamma_2)
        if self.active_atom_mask is not None:
            mask = np.asarray(self.active_atom_mask, dtype=bool).reshape(-1)
            if mask.size != mode.shape[0]:
                raise ValueError(
                    "active_atom_mask size must equal atom count; got "
                    f"{mask.size} and {mode.shape[0]}"
                )
            object.__setattr__(self, "active_atom_mask", mask.copy())
        if self.policy != "current_snapshot":
            raise ValueError(
                "only projection_policy=current_snapshot is implemented"
            )
        if self.kappa_policy != "fixed_current_center_coefficients":
            raise ValueError(
                "only kappa_projection_policy="
                "fixed_current_center_coefficients is implemented"
            )

    @classmethod
    def from_dimeratoms(
        cls,
        dimeratoms,
        *,
        policy: str = "current_snapshot",
        kappa_policy: str = "fixed_current_center_coefficients",
    ) -> "ProjectionSnapshot":
        mode = np.asarray(dimeratoms.get_eigenmode(), dtype=float)
        curvature = float(dimeratoms.get_curvature())
        branch = "convex" if curvature > 0.0 else "concave"
        regime = str(getattr(dimeratoms, "translation_regime", "standard"))
        gamma_1 = float(getattr(dimeratoms, "last_gamma_1", 1.0))
        gamma_2_default = 0.0 if branch == "convex" else 1.0
        gamma_2 = float(getattr(dimeratoms, "last_gamma_2", gamma_2_default))

        # Above the inflection point, both stock Dimer and Kappa use only the
        # uphill mode force. Freeze that exact current branch independently of
        # stale values that a previous concave Kappa state may have left behind.
        if branch == "convex":
            regime = "bowl_breakout" if str(
                getattr(dimeratoms, "convex_escape", "standard")
            ) == "bowl_breakout" else "standard"
            gamma_1 = 1.0
            gamma_2 = 0.0

        active_mask = None
        getter = getattr(dimeratoms, "bowl_active_atom_mask", None)
        if callable(getter):
            try:
                active_mask = np.asarray(getter(), dtype=bool)
            except Exception:
                active_mask = None
        if active_mask is None:
            # The captured BowlBreakoutMixin stores the accepted-center mask in
            # the private ``_bowl_active_mask`` attribute.  Read it without
            # mutating the live object so a canonical reconstruction freezes the
            # same active set that generated the current projected force.
            private_mask = getattr(dimeratoms, "_bowl_active_mask", None)
            if private_mask is not None:
                active_mask = np.asarray(private_mask, dtype=bool)

        return cls(
            mode=mode,
            curvature=curvature,
            branch=branch,
            regime=regime,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            active_atom_mask=active_mask,
            policy=policy,
            kappa_policy=kappa_policy,
        )

    def project(self, physical_force: object) -> Array:
        force = _array_n3(physical_force, "physical_force")
        if force.shape != self.mode.shape:
            raise ValueError(
                "force/mode shape mismatch: "
                f"{force.shape} versus {self.mode.shape}"
            )
        mode_flat = self.mode.reshape(-1)
        force_flat = force.reshape(-1)
        parallel = (
            float(np.dot(force_flat, mode_flat)) * mode_flat
        ).reshape(force.shape)

        if self.branch == "convex":
            projected = -parallel
        else:
            perpendicular = force - parallel
            if self.regime == "kappa":
                projected = -self.gamma_1 * parallel + self.gamma_2 * perpendicular
            else:
                projected = perpendicular - parallel

        if self.active_atom_mask is not None and self.branch == "convex":
            projected = np.array(projected, copy=True)
            projected[~self.active_atom_mask, :] = 0.0
        return projected

    def metadata(self) -> dict[str, object]:
        return {
            "projection_policy": self.policy,
            "kappa_projection_policy": self.kappa_policy,
            "branch": self.branch,
            "regime": self.regime,
            "gamma_1": self.gamma_1,
            "gamma_2": self.gamma_2,
            "active_atoms": (
                "" if self.active_atom_mask is None
                else int(np.count_nonzero(self.active_atom_mask))
            ),
        }


@dataclass
class ReconstructionResult:
    direction: Array
    projected_current_force: Array
    records: list[SecantRecord]
    accepted_pairs: list[tuple[Array, Array, float, str]]
    timings_ns: dict[str, int]
    metrics: dict[str, object]
    detail_payload: dict[str, object] = field(default_factory=dict)
    exact_state_payload: dict[str, object] = field(default_factory=dict)



def reconstruct_lbfgs(
    history: CanonicalForceHistory,
    snapshot: ProjectionSnapshot,
    current_projected_force: object,
    *,
    pair_sources: object,
    initial_hessian: float = 70.0,
    dynamic_h0: bool = False,
    safeguard: str = "skip",
    curvature_floor: float = 1.0e-3,
    curvature_epsilon: float = 1.0e-12,
    powell_eta: float = 0.2,
    cosine_threshold: float | None = None,
    max_pairs: int = 0,
    rigid_translation_projection: bool = False,
    deep_diagnostics: bool = False,
    diagnostic_shift_mus: object = "0.01 0.1 1 10",
    cautious_epsilon: float = 1.0e-6,
    cautious_alpha: float = 1.0,
    reconstruction_model: str = "lbfgs",
    bfgs_update: str = "sequential",
    dense_bfgs_diagnostic: bool = False,
    exact_state_dump: bool = False,
    exact_two_loop_trace: bool = True,
) -> ReconstructionResult:
    """Reconstruct an L-BFGS direction from raw physical observations.

    ``current_projected_force`` is the current raw center force after applying
    the same frozen ``ProjectionSnapshot`` used for every retained observation.
    The caller may compare it with the live minimum-mode force and fail closed
    before this function is entered.
    """

    total_start = perf_counter_ns()
    initial_hessian = float(initial_hessian)
    curvature_floor = float(curvature_floor)
    curvature_epsilon = float(curvature_epsilon)
    powell_eta = float(powell_eta)
    cosine_threshold = None if cosine_threshold is None else float(cosine_threshold)
    max_pairs = int(max_pairs)
    safeguard = str(safeguard).strip().lower()
    reconstruction_model = str(reconstruction_model).strip().lower()
    bfgs_update = str(bfgs_update).strip().lower()
    dense_bfgs_diagnostic = bool(dense_bfgs_diagnostic)
    rigid_translation_projection = bool(rigid_translation_projection)
    deep_diagnostics = bool(deep_diagnostics)
    cautious_epsilon = float(cautious_epsilon)
    cautious_alpha = float(cautious_alpha)
    if isinstance(diagnostic_shift_mus, str):
        diagnostic_shift_mus = [float(x) for x in diagnostic_shift_mus.replace(",", " ").split() if x.strip()]
    elif diagnostic_shift_mus is None:
        diagnostic_shift_mus = []
    else:
        diagnostic_shift_mus = [float(x) for x in diagnostic_shift_mus]
    exact_state_dump = bool(exact_state_dump)
    exact_two_loop_trace = bool(exact_two_loop_trace)

    if initial_hessian <= 0.0:
        raise ValueError("initial_hessian must be > 0")
    if safeguard not in VALID_SAFEGUARDS:
        raise ValueError(
            f"safeguard must be one of {sorted(VALID_SAFEGUARDS)}; "
            f"got {safeguard!r}"
        )
    if curvature_epsilon < 0.0:
        raise ValueError("curvature_epsilon must be >= 0")
    if safeguard in {"skip", "damp", "reset"} and curvature_floor <= 0.0:
        raise ValueError("curvature_floor must be > 0 for skip/damp/reset")
    if safeguard == "cautious" and (cautious_epsilon <= 0.0 or cautious_alpha < 0.0):
        raise ValueError("cautious_epsilon must be >0 and cautious_alpha >=0")
    if safeguard == "powell" and not 0.0 < powell_eta < 1.0:
        raise ValueError("powell_eta must satisfy 0 < powell_eta < 1")
    if cosine_threshold is not None and not -1.0 <= cosine_threshold <= 1.0:
        raise ValueError("cosine_threshold must lie in [-1,1] or be disabled")
    if max_pairs < 0:
        raise ValueError("max_pairs must be >= 0")
    if reconstruction_model not in {"lbfgs", "reconstructed_bfgs"}:
        raise ValueError("reconstruction_model must be lbfgs or reconstructed_bfgs")
    if bfgs_update not in {"sequential", "multisecant"}:
        raise ValueError("bfgs_update must be sequential or multisecant")

    current_force = _array_n3(current_projected_force, "current_projected_force")
    Q = rigid_translation_basis(current_force.shape[0])
    current_force_translation_ratio = translation_ratio(current_force, Q)
    if rigid_translation_projection:
        current_force = project_rigid(current_force, Q).reshape(current_force.shape)

    t0 = perf_counter_ns()
    candidates = history.pair_candidates(pair_sources, physical_only=True)
    pair_candidates_pre_truncation = len(candidates)
    # max_pairs is a cap on pairs that actually survive the admission/guard
    # policy, not a pre-filter window on raw candidates.  This allows older
    # admissible pairs to backfill newer candidates rejected by cosine or
    # curvature safeguards while preserving deterministic newest-pair use.
    pairs_removed_by_max_pairs = 0
    pairs_admissible_before_max_pairs = 0
    selection_ns = perf_counter_ns() - t0

    t0 = perf_counter_ns()
    projected_cache = _project_candidate_observations(candidates, snapshot)
    projection_ns = perf_counter_ns() - t0

    t0 = perf_counter_ns()
    raw_pairs = _build_raw_candidate_pairs(
        candidates,
        projected_cache,
        translation_basis=Q,
        rigid_translation_projection=rigid_translation_projection,
    )
    pair_build_ns = perf_counter_ns() - t0

    t0 = perf_counter_ns()
    admission = _admit_candidate_pairs(
        raw_pairs,
        current_force,
        initial_hessian=initial_hessian,
        safeguard=safeguard,
        curvature_floor=curvature_floor,
        curvature_epsilon=curvature_epsilon,
        powell_eta=powell_eta,
        cosine_threshold=cosine_threshold,
        max_pairs=max_pairs,
        cautious_epsilon=cautious_epsilon,
        cautious_alpha=cautious_alpha,
    )
    accepted = admission.accepted
    accepted_meta = admission.accepted_meta
    dense_pairs = admission.dense_pairs
    records = admission.records
    reset_count = admission.reset_count
    rejection_reasons = admission.rejection_reasons
    pairs_admissible_before_max_pairs = admission.pairs_admissible_before_max_pairs
    pairs_removed_by_max_pairs = admission.pairs_removed_by_max_pairs
    safeguard_ns = perf_counter_ns() - t0

    t0 = perf_counter_ns()
    lbfgs_direction_flat, h0_inverse_scale, two_loop_trace = _two_loop(
        current_force,
        accepted,
        initial_hessian=initial_hessian,
        dynamic_h0=bool(dynamic_h0),
        trace=bool(exact_state_dump and exact_two_loop_trace),
    )
    two_loop_ns = perf_counter_ns() - t0

    dense_result = None
    dense_ns = 0
    dense_requested = bool(
        dense_bfgs_diagnostic
        or deep_diagnostics
        or reconstruction_model == "reconstructed_bfgs"
    )
    if dense_requested:
        dense_start = perf_counter_ns()
        dense_result = _reconstruct_dense_model(
            dense_pairs,
            current_force,
            bfgs_update=bfgs_update,
            initial_hessian=initial_hessian,
            dynamic_h0=bool(dynamic_h0),
            curvature_epsilon=curvature_epsilon,
            safeguard=safeguard,
        )
        dense_ns = perf_counter_ns() - dense_start

    direction_flat, dense_fallback_reason = _select_reconstruction_direction(
        lbfgs_direction_flat,
        dense_result,
        reconstruction_model=reconstruction_model,
    )
    direction = direction_flat.reshape(current_force.shape)

    latest = records[-1] if records else None
    # A later reset invalidates earlier accepted pairs.  Match diagnostic
    # accounting to the pairs that actually reach the two-loop recursion.
    accepted_records = _accepted_records_for_model(records, accepted)
    raw_sy_values = [
        float(record.raw_metrics["s_dot_y"])
        for record in records
        if record.raw_metrics.get("s_dot_y", "") != ""
    ]
    stored_sy_values = [
        float(record.stored_metrics["s_dot_y"])
        for record in accepted_records
        if record.stored_metrics.get("s_dot_y", "") != ""
    ]

    accepted_curvatures = [
        float(record.stored_metrics.get("secant_curvature", np.nan))
        for record in accepted_records
    ]
    accepted_cosines = [
        float(record.stored_metrics.get("secant_cosine", np.nan))
        for record in accepted_records
    ]
    accepted_curvatures = [v for v in accepted_curvatures if np.isfinite(v)]
    accepted_cosines = [v for v in accepted_cosines if np.isfinite(v)]
    def _triple(values):
        if not values:
            return ("", "", "")
        arr = np.asarray(values, dtype=float)
        return (float(np.min(arr)), float(np.median(arr)), float(np.max(arr)))
    accepted_curv_min, accepted_curv_median, accepted_curv_max = _triple(accepted_curvatures)
    accepted_cos_min, accepted_cos_median, accepted_cos_max = _triple(accepted_cosines)

    timings = {
        "selection_ns": int(selection_ns),
        "projection_ns": int(projection_ns),
        "pair_build_ns": int(pair_build_ns),
        "safeguard_ns": int(safeguard_ns),
        "two_loop_ns": int(two_loop_ns),
        "dense_bfgs_ns": int(dense_ns),
    }
    timings["reconstruction_total_ns"] = int(perf_counter_ns() - total_start)

    metrics: dict[str, object] = {
        "pair_sources": " ".join(normalise_tokens(pair_sources)),
        "pair_safeguard": safeguard,
        "reconstruction_model": reconstruction_model,
        "bfgs_update": bfgs_update,
        "dense_bfgs_diagnostic": int(dense_bfgs_diagnostic),
        "rigid_translation_projection": int(rigid_translation_projection),
        "current_force_translation_ratio": current_force_translation_ratio,
        "dense_bfgs_fallback_reason": dense_fallback_reason,
        "cosine_threshold": "" if cosine_threshold is None else cosine_threshold,
        "pair_candidates": pair_candidates_pre_truncation,
        "pair_candidates_after_max_pairs": len(candidates),
        "pairs_admissible_before_max_pairs": pairs_admissible_before_max_pairs,
        "pairs_removed_by_max_pairs": pairs_removed_by_max_pairs,
        "pairs_used": len(accepted),
        "pairs_rejected": len(candidates) - pairs_admissible_before_max_pairs,
        "pairs_rejected_by_curvature": sum(
            int(v) for k, v in rejection_reasons.items()
            if "curvature" in k or "secant_denominator" in k or "nonpositive" in k
        ),
        "pairs_rejected_by_cosine": int(rejection_reasons.get("below_cosine_threshold", 0)),
        "accepted_curvature_min": accepted_curv_min,
        "accepted_curvature_median": accepted_curv_median,
        "accepted_curvature_max": accepted_curv_max,
        "accepted_cosine_min": accepted_cos_min,
        "accepted_cosine_median": accepted_cos_median,
        "accepted_cosine_max": accepted_cos_max,
        "pairs_damped": sum(record.action == "damped" for record in accepted_records),
        "pairs_powell_damped": sum(
            record.action == "powell_damped" for record in accepted_records
        ),
        "pair_resets": reset_count,
        "candidate_pairs_by_source": {
            source: sum(candidate.source == source for candidate in candidates)
            for source in sorted({candidate.source for candidate in candidates})
        },
        "used_pairs_by_source": {
            source: sum(item[3] == source for item in accepted)
            for source in sorted({item[3] for item in accepted})
        },
        "rejection_reasons": rejection_reasons,
        "h0_inverse_scale": h0_inverse_scale,
        "direction_norm": safe_norm(direction),
        "direction_max_atom_norm": _max_atom_norm(direction),
        "direction_force_cosine": _cosine(direction, current_force),
        "lbfgs_direction_norm": safe_norm(lbfgs_direction_flat),
        "lbfgs_direction_max_atom_norm": _max_atom_norm(
            lbfgs_direction_flat.reshape(current_force.shape)
        ),
        "worst_raw_s_dot_y": min(raw_sy_values) if raw_sy_values else "",
        "worst_stored_s_dot_y": min(stored_sy_values) if stored_sy_values else "",
        "latest_pair_source": "" if latest is None else latest.source,
        "latest_pair_action": "" if latest is None else latest.action,
        "latest_pair_rejection_reason": (
            "" if latest is None else latest.rejection_reason
        ),
        "latest_raw_pair": {} if latest is None else latest.raw_metrics,
        "latest_stored_pair": {} if latest is None else latest.stored_metrics,
        **snapshot.metadata(),
    }
    detail_payload: dict[str, object] = {}
    if deep_diagnostics:
        detail_payload = _build_deep_diagnostics_payload(
            candidates=candidates,
            records=records,
            accepted_records=accepted_records,
            raw_pairs=raw_pairs,
            accepted=accepted,
            current_force=current_force,
            translation_basis=Q,
            safeguard=safeguard,
            cautious_epsilon=cautious_epsilon,
            cautious_alpha=cautious_alpha,
            curvature_floor=curvature_floor,
            cosine_threshold=cosine_threshold,
            initial_hessian=initial_hessian,
            dynamic_h0=bool(dynamic_h0),
            dense_result=dense_result,
            diagnostic_shift_mus=diagnostic_shift_mus,
            lbfgs_direction_flat=lbfgs_direction_flat,
            current_force_translation_ratio=current_force_translation_ratio,
            rigid_translation_projection=rigid_translation_projection,
            h0_inverse_scale=h0_inverse_scale,
        )

    exact_state_payload: dict[str, object] = {}
    if exact_state_dump:
        exact_state_payload = _build_exact_state_payload(
            candidates=candidates,
            records=records,
            accepted=accepted,
            accepted_meta=accepted_meta,
            current_force=current_force,
            snapshot=snapshot,
            safeguard=safeguard,
            curvature_floor=curvature_floor,
            curvature_epsilon=curvature_epsilon,
            powell_eta=powell_eta,
            max_pairs=max_pairs,
            cosine_threshold=cosine_threshold,
            pair_candidates_pre_truncation=pair_candidates_pre_truncation,
            pairs_removed_by_max_pairs=pairs_removed_by_max_pairs,
            pair_sources=pair_sources,
            initial_hessian=initial_hessian,
            dynamic_h0=bool(dynamic_h0),
            h0_inverse_scale=h0_inverse_scale,
            lbfgs_direction_flat=lbfgs_direction_flat,
            two_loop_trace=two_loop_trace,
        )



    if dense_result is not None:
        metrics.update(dense_result.metrics)
        metrics.update(compare_directions(lbfgs_direction_flat, dense_result.direction))
        metrics["dense_vs_lbfgs_cosine"] = metrics.pop("dense_vs_primary_cosine", "")
        metrics["dense_vs_lbfgs_norm_ratio"] = metrics.pop(
            "dense_vs_primary_norm_ratio", ""
        )
        metrics["dense_vs_lbfgs_relative_difference"] = metrics.pop(
            "dense_vs_primary_relative_difference", ""
        )

    return ReconstructionResult(
        direction=direction,
        projected_current_force=current_force,
        records=records,
        accepted_pairs=accepted,
        timings_ns=timings,
        metrics=metrics,
        detail_payload=detail_payload,
        exact_state_payload=exact_state_payload,

    )


def direction_ablation_metrics(
    full: ReconstructionResult,
    center_only: ReconstructionResult,
) -> dict[str, object]:
    full_direction = np.asarray(full.direction, dtype=float)
    center_direction = np.asarray(center_only.direction, dtype=float)
    center_norm = safe_norm(center_direction)
    full_norm = safe_norm(full_direction)
    return {
        "reuse_direction_cosine": _cosine(full_direction, center_direction),
        "reuse_direction_delta_norm": safe_norm(
            full_direction - center_direction
        ),
        "reuse_direction_norm_ratio": (
            "" if center_norm <= 1.0e-30 else full_norm / center_norm
        ),
        "reuse_pairs_added": int(
            full.metrics["pair_candidates"] - center_only.metrics["pair_candidates"]
        ),
        "reuse_pairs_survived": int(
            full.metrics["pairs_used"] - center_only.metrics["pairs_used"]
        ),
        "center_only_pairs_used": int(center_only.metrics["pairs_used"]),
        "center_only_reconstruction_ns": int(
            center_only.timings_ns["reconstruction_total_ns"]
        ),
    }
