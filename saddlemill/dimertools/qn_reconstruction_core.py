"""Pure canonical quasi-Newton reconstruction stages.

This module owns pair construction/admission and direction-model application only.
It is calculator independent and deliberately contains no deep/state-dump diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from saddlemill.dimertools.force_history import ForceObservation, PairCandidate
from saddlemill.dimertools.qn_deep_diagnostics import project_rigid
from saddlemill.dimertools.dense_bfgs import (
    DenseSecant,
    reconstruct_multisecant_bfgs,
    reconstruct_sequential_bfgs,
    safe_norm,
)

Array = np.ndarray

@dataclass
class SecantRecord:
    s: Array
    y_raw: Array
    y_stored: Array
    source: str
    state_id: int
    serial: int
    action: str
    rejection_reason: str = ""
    raw_metrics: dict[str, float | str] = field(default_factory=dict)
    stored_metrics: dict[str, float | str] = field(default_factory=dict)


def pair_metrics(s: Array, y: Array) -> dict[str, float | str]:
    sv = np.asarray(s, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    ss = float(np.dot(sv, sv))
    yy = float(np.dot(yv, yv))
    sy = float(np.dot(sv, yv))
    snorm = safe_norm(sv)
    ynorm = safe_norm(yv)
    return {
        "s_norm": snorm,
        "y_norm": ynorm,
        "s_dot_y": sy,
        "secant_curvature": "" if ss <= 0.0 else sy / ss,
        "secant_cosine": "" if snorm * ynorm <= 0.0 else sy / (snorm * ynorm),
        "gamma_sy_over_yy": "" if yy <= 0.0 else sy / yy,
        "force_change_norm": ynorm,
    }


def _direct_b_action(
    vector: Array,
    *,
    initial_hessian: float,
    operators: Sequence[tuple[Array, Array, Array, float, float]],
) -> Array:
    """Apply the implicit direct BFGS Hessian without forming a dense matrix."""

    v = np.asarray(vector, dtype=float).reshape(-1)
    result = float(initial_hessian) * v
    for _s, y, bs, sbs, sy in operators:
        result = result - bs * (float(np.dot(bs, v)) / sbs)
        result = result + y * (float(np.dot(y, v)) / sy)
    return result


def _two_loop(
    force: Array,
    accepted: Sequence[tuple[Array, Array, float, str]],
    *,
    initial_hessian: float,
    dynamic_h0: bool,
    trace: bool = False,
) -> tuple[Array, float, dict[str, Array]]:
    # Keep the production arithmetic unchanged; the optional trace only records
    # scalar intermediates around those exact operations.
    q = np.asarray(force, dtype=float).reshape(-1).copy()
    alphas: list[float] = []
    first_indices: list[int] = []
    first_q_before: list[float] = []
    first_q_after: list[float] = []
    for history_index in range(len(accepted) - 1, -1, -1):
        s, y, sy, _source = accepted[history_index]
        if trace:
            first_indices.append(history_index)
            first_q_before.append(safe_norm(q))
        alpha = float(np.dot(s, q)) / sy
        alphas.append(alpha)
        q -= alpha * y
        if trace:
            first_q_after.append(safe_norm(q))

    h0_scale = 1.0 / float(initial_hessian)
    if dynamic_h0 and accepted:
        _s, y, sy, _source = accepted[-1]
        yy = float(np.dot(y, y))
        if yy > 0.0 and sy > 0.0:
            candidate = sy / yy
            if np.isfinite(candidate) and candidate > 0.0:
                h0_scale = candidate

    result = h0_scale * q
    second_indices: list[int] = []
    betas: list[float] = []
    second_r_before: list[float] = []
    second_r_after: list[float] = []
    for history_index, ((s, y, sy, _source), alpha) in enumerate(
        zip(accepted, reversed(alphas))
    ):
        if trace:
            second_indices.append(history_index)
            second_r_before.append(safe_norm(result))
        beta = float(np.dot(y, result)) / sy
        if trace:
            betas.append(beta)
        result += s * (alpha - beta)
        if trace:
            second_r_after.append(safe_norm(result))

    trace_payload: dict[str, Array] = {}
    if trace:
        trace_payload = {
            "first_pair_history_index": np.asarray(first_indices, dtype=np.int64),
            "first_alpha": np.asarray(alphas, dtype=np.float64),
            "first_q_norm_before": np.asarray(first_q_before, dtype=np.float64),
            "first_q_norm_after": np.asarray(first_q_after, dtype=np.float64),
            "second_pair_history_index": np.asarray(second_indices, dtype=np.int64),
            "second_beta": np.asarray(betas, dtype=np.float64),
            "second_r_norm_before": np.asarray(second_r_before, dtype=np.float64),
            "second_r_norm_after": np.asarray(second_r_after, dtype=np.float64),
        }
    return result, h0_scale, trace_payload


def _source_counter(records: Iterable[SecantRecord], action: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        if action != "all" and record.action != action:
            continue
        result[record.source] = result.get(record.source, 0) + 1
    return result



@dataclass
class _AdmissionResult:
    """Pure pair-admission output used by canonical reconstruction."""

    accepted: list[tuple[Array, Array, float, str]]
    accepted_meta: list[PairCandidate]
    dense_pairs: list[DenseSecant]
    records: list[SecantRecord]
    reset_count: int
    rejection_reasons: dict[str, int]
    pairs_admissible_before_max_pairs: int
    pairs_removed_by_max_pairs: int


def _project_candidate_observations(
    candidates: Sequence[PairCandidate],
    snapshot: ProjectionSnapshot,
) -> dict[int, Array]:
    """Project each referenced observation once, preserving candidate chronology."""

    projected_cache: dict[int, Array] = {}

    def projected(observation: ForceObservation) -> Array:
        cached = projected_cache.get(observation.serial)
        if cached is None:
            cached = snapshot.project(observation.forces)
            projected_cache[observation.serial] = cached
        return cached

    for candidate in candidates:
        projected(candidate.first)
        projected(candidate.second)
    return projected_cache


def _build_raw_candidate_pairs(
    candidates: Sequence[PairCandidate],
    projected_cache: Mapping[int, Array],
    *,
    translation_basis: Array,
    rigid_translation_projection: bool,
) -> list[tuple[PairCandidate, Array, Array]]:
    """Build chronological raw secants without applying any admission policy."""

    raw_pairs: list[tuple[PairCandidate, Array, Array]] = []
    for candidate in candidates:
        s = candidate.displacement().reshape(-1)
        # gradient = -force, so y = grad(second)-grad(first)
        #                          = F(first)-F(second).
        y = (
            projected_cache[candidate.first.serial]
            - projected_cache[candidate.second.serial]
        ).reshape(-1)
        if rigid_translation_projection:
            s = project_rigid(s, translation_basis)
            y = project_rigid(y, translation_basis)
        raw_pairs.append((candidate, s, y))
    return raw_pairs


def _admit_candidate_pairs(
    raw_pairs: Sequence[tuple[PairCandidate, Array, Array]],
    current_force: Array,
    *,
    initial_hessian: float,
    safeguard: str,
    curvature_floor: float,
    curvature_epsilon: float,
    powell_eta: float,
    cosine_threshold: float | None,
    max_pairs: int,
    cautious_epsilon: float,
    cautious_alpha: float,
) -> _AdmissionResult:
    """Apply the existing safeguard and final accepted-pair memory cap exactly."""

    accepted: list[tuple[Array, Array, float, str]] = []
    accepted_meta: list[PairCandidate] = []
    dense_pairs: list[DenseSecant] = []
    operators: list[tuple[Array, Array, Array, float, float]] = []
    records: list[SecantRecord] = []
    reset_count = 0
    rejection_reasons: dict[str, int] = {}

    def reject(
        candidate: PairCandidate,
        s: Array,
        y: Array,
        reason: str,
        action: str = "rejected",
    ) -> None:
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        records.append(
            SecantRecord(
                s=s.copy(),
                y_raw=y.copy(),
                y_stored=y.copy(),
                source=candidate.source,
                state_id=candidate.state_id,
                serial=candidate.serial,
                action=action,
                rejection_reason=reason,
                raw_metrics=pair_metrics(s, y),
                stored_metrics=pair_metrics(s, y),
            )
        )

    for candidate, s, y_raw in raw_pairs:
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y_raw)):
            if safeguard == "reset":
                accepted.clear()
                accepted_meta.clear()
                dense_pairs.clear()
                operators.clear()
                reset_count += 1
                reject(candidate, s, y_raw, "non_finite", action="reset")
            else:
                reject(candidate, s, y_raw, "non_finite")
            continue
        ss = float(np.dot(s, s))
        yy_raw = float(np.dot(y_raw, y_raw))
        if ss <= 0.0:
            if safeguard == "reset":
                accepted.clear()
                accepted_meta.clear()
                dense_pairs.clear()
                operators.clear()
                reset_count += 1
                reject(candidate, s, y_raw, "zero_displacement", action="reset")
            else:
                reject(candidate, s, y_raw, "zero_displacement")
            continue
        if yy_raw <= 0.0:
            if safeguard == "reset":
                accepted.clear()
                accepted_meta.clear()
                dense_pairs.clear()
                operators.clear()
                reset_count += 1
                reject(candidate, s, y_raw, "zero_force_change", action="reset")
            else:
                reject(candidate, s, y_raw, "zero_force_change")
            continue

        y = y_raw.copy()
        sy_raw = float(np.dot(s, y_raw))
        action = "accepted"

        if cosine_threshold is not None:
            cosine_raw = sy_raw / float(np.sqrt(ss * yy_raw))
            if not np.isfinite(cosine_raw) or cosine_raw < cosine_threshold:
                reject(candidate, s, y_raw, "below_cosine_threshold")
                continue

        if safeguard == "cautious":
            # Li-Fukushima cautious update: (sTy)/(sTs) >= eps * ||g||^alpha.
            required_curvature = cautious_epsilon * (
                safe_norm(current_force) ** cautious_alpha
            )
            if sy_raw / ss < required_curvature:
                reject(candidate, s, y_raw, "below_cautious_curvature")
                continue

        if safeguard in {"skip", "damp", "reset"}:
            required = curvature_floor * ss
            if sy_raw < required:
                if safeguard == "skip":
                    reject(candidate, s, y_raw, "below_curvature_floor")
                    continue
                if safeguard == "reset":
                    accepted.clear()
                    accepted_meta.clear()
                    dense_pairs.clear()
                    operators.clear()
                    reset_count += 1
                    reject(
                        candidate,
                        s,
                        y_raw,
                        "below_curvature_floor",
                        action="reset",
                    )
                    continue
                y += ((required - sy_raw) / ss) * s
                action = "damped"

        bs = None
        sbs = None
        if safeguard == "powell":
            bs = _direct_b_action(
                s,
                initial_hessian=initial_hessian,
                operators=operators,
            )
            sbs = float(np.dot(s, bs))
            if not np.isfinite(sbs) or sbs <= 0.0:
                reject(candidate, s, y_raw, "nonpositive_s_dot_Bs")
                continue
            threshold = powell_eta * sbs
            if sy_raw < threshold:
                denominator = sbs - sy_raw
                if denominator <= 0.0:
                    reject(candidate, s, y_raw, "powell_denominator")
                    continue
                theta = ((1.0 - powell_eta) * sbs) / denominator
                y = theta * y_raw + (1.0 - theta) * bs
                action = "powell_damped"

        sy = float(np.dot(s, y))
        yy = float(np.dot(y, y))
        threshold = curvature_epsilon * max(1.0, float(np.sqrt(ss * yy)))
        valid_denominator = (
            abs(sy) > threshold if safeguard == "off" else sy > threshold
        )
        if not np.isfinite(sy) or not valid_denominator:
            if safeguard == "reset":
                accepted.clear()
                accepted_meta.clear()
                dense_pairs.clear()
                operators.clear()
                reset_count += 1
                reject(
                    candidate,
                    s,
                    y_raw,
                    "small_secant_denominator",
                    action="reset",
                )
            else:
                reject(candidate, s, y_raw, "small_secant_denominator")
            continue

        accepted.append((s.copy(), y.copy(), sy, candidate.source))
        accepted_meta.append(candidate)
        dense_pairs.append(
            DenseSecant(
                s=s,
                y=y,
                source=candidate.source,
                state_id=candidate.state_id,
                serial=candidate.serial,
            )
        )
        if safeguard == "powell":
            assert bs is not None and sbs is not None
            operators.append((s.copy(), y.copy(), bs.copy(), sbs, sy))
        records.append(
            SecantRecord(
                s=s.copy(),
                y_raw=y_raw.copy(),
                y_stored=y.copy(),
                source=candidate.source,
                state_id=candidate.state_id,
                serial=candidate.serial,
                action=action,
                raw_metrics=pair_metrics(s, y_raw),
                stored_metrics=pair_metrics(s, y),
            )
        )

    pairs_admissible_before_max_pairs = len(accepted)
    pairs_removed_by_max_pairs = 0
    if max_pairs > 0 and len(accepted) > max_pairs:
        pairs_removed_by_max_pairs = len(accepted) - max_pairs
        truncated_meta = accepted_meta[:-max_pairs]
        truncated_keys = {
            (int(c.state_id), int(c.serial), str(c.source)) for c in truncated_meta
        }
        accepted = accepted[-max_pairs:]
        accepted_meta = accepted_meta[-max_pairs:]
        dense_pairs = dense_pairs[-max_pairs:]
        # Mark otherwise-admissible pairs that were excluded solely by the
        # final memory cap.  The stored vectors remain in the diagnostic record.
        for record in records:
            key = (int(record.state_id), int(record.serial), str(record.source))
            if key in truncated_keys and record.action in {
                "accepted",
                "damped",
                "powell_damped",
            }:
                record.action = "max_pairs_truncated"
                record.rejection_reason = "max_pairs"

    return _AdmissionResult(
        accepted=accepted,
        accepted_meta=accepted_meta,
        dense_pairs=dense_pairs,
        records=records,
        reset_count=reset_count,
        rejection_reasons=rejection_reasons,
        pairs_admissible_before_max_pairs=pairs_admissible_before_max_pairs,
        pairs_removed_by_max_pairs=pairs_removed_by_max_pairs,
    )


def _reconstruct_dense_model(
    dense_pairs: Sequence[DenseSecant],
    current_force: Array,
    *,
    bfgs_update: str,
    initial_hessian: float,
    dynamic_h0: bool,
    curvature_epsilon: float,
    safeguard: str,
):
    """Construct an explicitly requested dense/shadow model; never called otherwise."""

    if bfgs_update == "multisecant":
        return reconstruct_multisecant_bfgs(
            dense_pairs,
            current_force,
            initial_hessian=initial_hessian,
            dynamic_h0=bool(dynamic_h0),
            curvature_epsilon=curvature_epsilon,
        )
    return reconstruct_sequential_bfgs(
        dense_pairs,
        current_force,
        initial_hessian=initial_hessian,
        dynamic_h0=bool(dynamic_h0),
        denominator_epsilon=curvature_epsilon,
        require_positive_definite=(safeguard != "off"),
    )


def _select_reconstruction_direction(
    lbfgs_direction_flat: Array,
    dense_result,
    *,
    reconstruction_model: str,
) -> tuple[Array, str]:
    """Select the driving direction without changing either model's arithmetic."""

    direction_flat = lbfgs_direction_flat
    dense_fallback_reason = ""
    if reconstruction_model == "reconstructed_bfgs":
        if dense_result is not None and dense_result.direction is not None:
            direction_flat = np.asarray(dense_result.direction, dtype=float).reshape(-1)
        else:
            dense_fallback_reason = (
                "dense_reconstruction_invalid:"
                + str(
                    ""
                    if dense_result is None
                    else dense_result.metrics.get("dense_invalid_reason", "")
                )
            )
    return direction_flat, dense_fallback_reason


def _accepted_records_for_model(
    records: Sequence[SecantRecord],
    accepted: Sequence[tuple[Array, Array, float, str]],
) -> list[SecantRecord]:
    """Recover records for pairs that actually reach the final two-loop model."""

    remaining_by_source: dict[str, int] = {}
    for _s, _y, _sy, source in accepted:
        remaining_by_source[source] = remaining_by_source.get(source, 0) + 1
    accepted_records_reversed: list[SecantRecord] = []
    need = dict(remaining_by_source)
    for record in reversed(records):
        if record.action not in {"accepted", "damped", "powell_damped"}:
            continue
        if need.get(record.source, 0) <= 0:
            continue
        accepted_records_reversed.append(record)
        need[record.source] -= 1
    return list(reversed(accepted_records_reversed))


