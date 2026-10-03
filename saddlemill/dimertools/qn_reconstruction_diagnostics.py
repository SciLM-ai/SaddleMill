"""Optional diagnostics for canonical quasi-Newton reconstruction.

These helpers consume already-built reconstruction state. They do not select pairs,
change the driving direction, mutate force history, or make calculator calls.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from saddlemill.dimertools.force_history import normalise_tokens
from saddlemill.dimertools.qn_deep_diagnostics import (
    block_diagnostics,
    progressive_lbfgs,
    secant_row,
    translation_ratio,
    shifted_lbfgs_solve,
)
from saddlemill.dimertools.lbfgs_state_dump import (
    rigid_translation_basis as exact_rigid_translation_basis,
)
from saddlemill.dimertools.dense_bfgs import safe_max_atom_norm, safe_norm

Array = np.ndarray


def _max_atom_norm(value: Array) -> float:
    return safe_max_atom_norm(value)


def _build_deep_diagnostics_payload(
    *,
    candidates: Sequence[PairCandidate],
    records: Sequence[SecantRecord],
    accepted_records: Sequence[SecantRecord],
    raw_pairs: Sequence[tuple[PairCandidate, Array, Array]],
    accepted: Sequence[tuple[Array, Array, float, str]],
    current_force: Array,
    translation_basis: Array,
    safeguard: str,
    cautious_epsilon: float,
    cautious_alpha: float,
    curvature_floor: float,
    cosine_threshold: float | None,
    initial_hessian: float,
    dynamic_h0: bool,
    dense_result,
    diagnostic_shift_mus: Sequence[float],
    lbfgs_direction_flat: Array,
    current_force_translation_ratio: float,
    rigid_translation_projection: bool,
    h0_inverse_scale: float,
) -> dict[str, object]:
    """Build force-call-free deep diagnostics from already constructed core state."""

    pair_rows = []
    candidate_lookup = {
        (c.state_id, c.serial, c.source): c for c in candidates
    }
    used_keys = {
        (r.state_id, r.serial, r.source) for r in accepted_records
    }
    for idx, record in enumerate(records):
        cand = candidate_lookup.get(
            (record.state_id, record.serial, record.source)
        )
        raw_curv = record.raw_metrics.get("secant_curvature", "")
        if safeguard == "cautious":
            required_curv = cautious_epsilon * (
                safe_norm(current_force) ** cautious_alpha
            )
            raw_criterion_passed = bool(
                raw_curv != "" and float(raw_curv) >= required_curv
            )
        elif safeguard in {"skip", "damp", "reset"}:
            required_curv = curvature_floor
            raw_criterion_passed = bool(
                raw_curv != "" and float(raw_curv) >= required_curv
            )
        elif safeguard == "powell":
            required_curv = "powell_sBs_dependent"
            raw_criterion_passed = record.action == "accepted"
        else:
            required_curv = ""
            raw_criterion_passed = True
        row = {
            "pair_index": idx,
            "pair_age_from_newest": len(records) - 1 - idx,
            "state_id": record.state_id,
            "serial": record.serial,
            "source": record.source,
            "action": record.action,
            "rejection_reason": record.rejection_reason,
            "raw_curvature_requirement": required_curv,
            "raw_criterion_passed": int(raw_criterion_passed),
            "cosine_threshold": "" if cosine_threshold is None else cosine_threshold,
            "cosine_criterion_passed": int(
                cosine_threshold is None
                or float(record.raw_metrics.get("secant_cosine", -np.inf))
                >= cosine_threshold
            ),
            "used": int(
                (record.state_id, record.serial, record.source) in used_keys
            ),
            "first_observation_serial": (
                "" if cand is None else int(cand.first.serial)
            ),
            "second_observation_serial": (
                "" if cand is None else int(cand.second.serial)
            ),
            "first_observation_source": (
                "" if cand is None else str(cand.first.source)
            ),
            "second_observation_source": (
                "" if cand is None else str(cand.second.source)
            ),
        }
        row.update(
            {
                "raw_" + k: v
                for k, v in secant_row(
                    record.s, record.y_raw, translation_basis
                ).items()
            }
        )
        row.update(
            {
                "stored_" + k: v
                for k, v in secant_row(
                    record.s, record.y_stored, translation_basis
                ).items()
            }
        )
        pair_rows.append(row)
    progressive = progressive_lbfgs(
        current_force,
        accepted,
        initial_hessian,
        bool(dynamic_h0),
        translation_basis,
    )
    progressive_fixed = progressive_lbfgs(
        current_force,
        accepted,
        initial_hessian,
        False,
        translation_basis,
    )
    for a, b in zip(progressive, progressive_fixed):
        a["fixed_h0_prediction_norm"] = b["prediction_norm"]
        a["fixed_h0_prediction_max_atom_norm"] = b[
            "prediction_max_atom_norm"
        ]
    blocks = []
    grouped = {}
    for candidate, svec, yvec in raw_pairs:
        key = (int(candidate.state_id), str(candidate.source))
        raw_y = (
            np.asarray(candidate.first.forces)
            - np.asarray(candidate.second.forces)
        ).reshape(-1)
        grouped.setdefault(key, []).append((svec, yvec, raw_y))
    for (state_id, source), items in sorted(grouped.items()):
        if len(items) < 2:
            continue
        S = np.column_stack([x[0] for x in items])
        Y = np.column_stack([x[1] for x in items])
        Yr = np.column_stack([x[2] for x in items])
        bd = block_diagnostics(S, Y, Yr)
        bd.update({"state_id": state_id, "source": source})
        prefixes = []
        for count in range(2, len(items) + 1):
            pd = block_diagnostics(S[:, :count], Y[:, :count], Yr[:, :count])
            pd.update({"included_probes": count})
            prefixes.append(pd)
        bd["prefix_diagnostics"] = prefixes
        if dense_result is not None and getattr(dense_result, "matrix", None) is not None:
            BY = np.asarray(dense_result.matrix) @ S
            model = block_diagnostics(S, BY)
            bd["sequential_model_generalized_curvatures"] = model.get(
                "generalized_curvatures", []
            )
            bd["sequential_model_block_residual"] = float(
                np.linalg.norm(BY - Y) / max(float(np.linalg.norm(Y)), 1e-30)
            )
        blocks.append(bd)
    fixed_shift_rows = []
    raw_lbfgs = np.asarray(lbfgs_direction_flat, float).reshape(-1)
    for mu in diagnostic_shift_mus:
        if mu < 0:
            continue
        try:
            shifted = shifted_lbfgs_solve(
                current_force.reshape(-1),
                accepted,
                initial_hessian,
                mu,
                bool(dynamic_h0),
            )
            den = safe_norm(raw_lbfgs) * safe_norm(shifted)
            fixed_shift_rows.append(
                {
                    "mu": float(mu),
                    "prediction_norm": safe_norm(shifted),
                    "prediction_max_atom_norm": _max_atom_norm(
                        np.asarray(shifted).reshape(current_force.shape)
                    ),
                    "suppression_ratio": (
                        ""
                        if safe_norm(raw_lbfgs) <= 1e-300
                        else safe_norm(shifted) / safe_norm(raw_lbfgs)
                    ),
                    "direction_cosine": (
                        "" if den <= 0 else float(raw_lbfgs @ shifted / den)
                    ),
                }
            )
        except Exception as exc:
            fixed_shift_rows.append(
                {"mu": float(mu), "error": type(exc).__name__ + ":" + str(exc)}
            )
    return {
        "schema": "saddlemill_translation_qn_v1",
        "pair_rows": pair_rows,
        "progressive_history": progressive,
        "block_rows": blocks,
        "current_force_norm": safe_norm(current_force),
        "current_force_translation_ratio": current_force_translation_ratio,
        "current_force_translation_ratio_after_projection": translation_ratio(
            current_force, translation_basis
        ),
        "rigid_translation_projection": int(rigid_translation_projection),
        "h0_dynamic": int(bool(dynamic_h0)),
        "h0_inverse_scale": h0_inverse_scale,
        "multisecant_block_details": (
            []
            if dense_result is None
            else dense_result.metrics.get("dense_multisecant_block_details", [])
        ),
        "fixed_shift_diagnostics": fixed_shift_rows,
    }


def _build_exact_state_payload(
    *,
    candidates: Sequence[PairCandidate],
    records: Sequence[SecantRecord],
    accepted: Sequence[tuple[Array, Array, float, str]],
    accepted_meta: Sequence[PairCandidate],
    current_force: Array,
    snapshot: ProjectionSnapshot,
    safeguard: str,
    curvature_floor: float,
    curvature_epsilon: float,
    powell_eta: float,
    max_pairs: int,
    cosine_threshold: float | None,
    pair_candidates_pre_truncation: int,
    pairs_removed_by_max_pairs: int,
    pair_sources: object,
    initial_hessian: float,
    dynamic_h0: bool,
    h0_inverse_scale: float,
    lbfgs_direction_flat: Array,
    two_loop_trace: Mapping[str, Array],
) -> dict[str, object]:
    """Build the lossless replay payload from core reconstruction state."""

    ndim = int(current_force.size)

    def stack_vectors(values):
        return (
            np.stack(
                [np.asarray(v, dtype=np.float64).reshape(-1) for v in values],
                axis=0,
            )
            if values
            else np.empty((0, ndim), dtype=np.float64)
        )

    def text_array(values):
        return np.asarray([str(v) for v in values], dtype=str)

    def metric_array(records_in, which, key):
        out = []
        for rec in records_in:
            value = getattr(rec, which).get(key, "")
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return np.asarray(out, dtype=np.float64)

    accepted_keys = {
        (int(c.state_id), int(c.serial), str(c.source)) for c in accepted_meta
    }
    candidate_lookup = {
        (int(c.state_id), int(c.serial), str(c.source)): c for c in candidates
    }
    candidate_used = []
    guard_pass = []
    required_curvature = []
    first_serial = []
    second_serial = []
    first_source = []
    second_source = []
    for rec in records:
        key = (int(rec.state_id), int(rec.serial), str(rec.source))
        candidate_used.append(key in accepted_keys)
        cand = candidate_lookup.get(key)
        first_serial.append(-1 if cand is None else int(cand.first.serial))
        second_serial.append(-1 if cand is None else int(cand.second.serial))
        first_source.append("" if cand is None else str(cand.first.source))
        second_source.append("" if cand is None else str(cand.second.source))
        raw_curv = rec.raw_metrics.get("secant_curvature", "")
        if safeguard in {"skip", "damp", "reset"}:
            req = curvature_floor
            passed = raw_curv != "" and float(raw_curv) >= req
        elif safeguard == "powell":
            req = float("nan")
            passed = rec.action in {"accepted", "powell_damped"}
        else:
            req = float("nan")
            passed = rec.action == "accepted"
        required_curvature.append(req)
        guard_pass.append(bool(passed))

    candidate_ss = np.asarray(
        [float(np.dot(r.s, r.s)) for r in records], dtype=np.float64
    )
    candidate_yy = np.asarray(
        [float(np.dot(r.y_stored, r.y_stored)) for r in records],
        dtype=np.float64,
    )
    candidate_sy = metric_array(records, "stored_metrics", "s_dot_y")
    candidate_rho = np.divide(
        1.0,
        candidate_sy,
        out=np.full_like(candidate_sy, np.nan),
        where=np.isfinite(candidate_sy) & (candidate_sy != 0.0),
    )
    history_ss = np.asarray(
        [float(np.dot(item[0], item[0])) for item in accepted],
        dtype=np.float64,
    )
    history_yy = np.asarray(
        [float(np.dot(item[1], item[1])) for item in accepted],
        dtype=np.float64,
    )
    # Use the exact production denominators, not a recomputed dot product.
    history_sy = np.asarray(
        [float(item[2]) for item in accepted], dtype=np.float64
    )
    history_rho = np.divide(
        1.0,
        history_sy,
        out=np.full_like(history_sy, np.nan),
        where=np.isfinite(history_sy) & (history_sy != 0.0),
    )
    history_kappa = np.divide(
        history_sy,
        history_ss,
        out=np.full_like(history_sy, np.nan),
        where=history_ss > 0.0,
    )
    history_gamma = np.divide(
        history_sy,
        history_yy,
        out=np.full_like(history_sy, np.nan),
        where=history_yy > 0.0,
    )
    history_cos = np.divide(
        history_sy,
        np.sqrt(history_ss * history_yy),
        out=np.full_like(history_sy, np.nan),
        where=(history_ss > 0.0) & (history_yy > 0.0),
    )

    payload = {
        "current_two_loop_vector": np.asarray(
            current_force, dtype=np.float64
        ).reshape(-1),
        "s_history": stack_vectors([item[0] for item in accepted]),
        "y_history": stack_vectors([item[1] for item in accepted]),
        "sy_history": history_sy,
        "history_index": np.arange(len(accepted), dtype=np.int64),
        "history_age_from_newest": np.arange(
            len(accepted) - 1, -1, -1, dtype=np.int64
        ),
        "history_sTs": history_ss,
        "history_yTy": history_yy,
        "history_sTy": history_sy,
        "history_kappa": history_kappa,
        "history_rho": history_rho,
        "history_gamma": history_gamma,
        "history_cos_theta": history_cos,
        "history_source": text_array([item[3] for item in accepted]),
        "history_detail_source": text_array(
            [c.detail_source for c in accepted_meta]
        ),
        "history_state_id": np.asarray(
            [int(c.state_id) for c in accepted_meta], dtype=np.int64
        ),
        "history_serial": np.asarray(
            [int(c.serial) for c in accepted_meta], dtype=np.int64
        ),
        "candidate_index": np.arange(len(records), dtype=np.int64),
        "candidate_age_from_newest": np.arange(
            len(records) - 1, -1, -1, dtype=np.int64
        ),
        "candidate_s": stack_vectors([r.s for r in records]),
        "candidate_y_raw": stack_vectors([r.y_raw for r in records]),
        "candidate_y_stored": stack_vectors([r.y_stored for r in records]),
        "candidate_source": text_array([r.source for r in records]),
        "candidate_detail_source": text_array(
            [
                ""
                if candidate_lookup.get(
                    (int(r.state_id), int(r.serial), str(r.source))
                )
                is None
                else candidate_lookup[
                    (int(r.state_id), int(r.serial), str(r.source))
                ].detail_source
                for r in records
            ]
        ),
        "candidate_state_id": np.asarray(
            [int(r.state_id) for r in records], dtype=np.int64
        ),
        "candidate_serial": np.asarray(
            [int(r.serial) for r in records], dtype=np.int64
        ),
        "candidate_action": text_array([r.action for r in records]),
        "candidate_rejection_reason": text_array(
            [r.rejection_reason for r in records]
        ),
        "candidate_used": np.asarray(candidate_used, dtype=np.bool_),
        "candidate_guard_pass": np.asarray(guard_pass, dtype=np.bool_),
        "candidate_required_curvature": np.asarray(
            required_curvature, dtype=np.float64
        ),
        "candidate_first_observation_serial": np.asarray(
            first_serial, dtype=np.int64
        ),
        "candidate_second_observation_serial": np.asarray(
            second_serial, dtype=np.int64
        ),
        "candidate_first_observation_source": text_array(first_source),
        "candidate_second_observation_source": text_array(second_source),
        "candidate_sTs": candidate_ss,
        "candidate_yTy": candidate_yy,
        "candidate_sTy": candidate_sy,
        "candidate_raw_yTy": np.asarray(
            [float(np.dot(r.y_raw, r.y_raw)) for r in records],
            dtype=np.float64,
        ),
        "candidate_raw_sTy": metric_array(records, "raw_metrics", "s_dot_y"),
        "candidate_raw_kappa": metric_array(
            records, "raw_metrics", "secant_curvature"
        ),
        "candidate_raw_gamma": np.asarray(
            [
                (
                    float(np.dot(r.s, r.y_raw))
                    / float(np.dot(r.y_raw, r.y_raw))
                    if float(np.dot(r.y_raw, r.y_raw)) > 0.0
                    else np.nan
                )
                for r in records
            ],
            dtype=np.float64,
        ),
        "candidate_raw_cos_theta": metric_array(
            records, "raw_metrics", "secant_cosine"
        ),
        "candidate_kappa": metric_array(
            records, "stored_metrics", "secant_curvature"
        ),
        "candidate_rho": candidate_rho,
        "candidate_gamma": np.divide(
            candidate_sy,
            candidate_yy,
            out=np.full_like(candidate_sy, np.nan),
            where=candidate_yy > 0.0,
        ),
        "candidate_cos_theta": metric_array(
            records, "stored_metrics", "secant_cosine"
        ),
        "initial_hessian": np.asarray(initial_hessian, dtype=np.float64),
        "dynamic_h0": np.asarray(bool(dynamic_h0), dtype=np.bool_),
        "h0_inverse_scale": np.asarray(h0_inverse_scale, dtype=np.float64),
        "raw_lbfgs_direction": np.asarray(
            lbfgs_direction_flat, dtype=np.float64
        ).reshape(-1),
        "snapshot_mode": np.asarray(snapshot.mode, dtype=np.float64),
        "snapshot_active_atom_mask": (
            np.empty(0, dtype=np.bool_)
            if snapshot.active_atom_mask is None
            else np.asarray(snapshot.active_atom_mask, dtype=np.bool_)
        ),
        "snapshot_curvature": np.asarray(snapshot.curvature, dtype=np.float64),
        "snapshot_gamma_1": np.asarray(snapshot.gamma_1, dtype=np.float64),
        "snapshot_gamma_2": np.asarray(snapshot.gamma_2, dtype=np.float64),
        "rigid_translation_basis": exact_rigid_translation_basis(
            current_force.size
        ),
        "pair_safeguard": np.asarray(safeguard),
        "curvature_floor": np.asarray(curvature_floor, dtype=np.float64),
        "curvature_epsilon": np.asarray(curvature_epsilon, dtype=np.float64),
        "powell_eta": np.asarray(powell_eta, dtype=np.float64),
        "max_pairs": np.asarray(max_pairs, dtype=np.int64),
        "cosine_threshold": np.asarray(
            np.nan if cosine_threshold is None else cosine_threshold,
            dtype=np.float64,
        ),
        "pair_candidates_pre_truncation": np.asarray(
            pair_candidates_pre_truncation, dtype=np.int64
        ),
        "pairs_removed_by_max_pairs": np.asarray(
            pairs_removed_by_max_pairs, dtype=np.int64
        ),
        "pair_sources": np.asarray(" ".join(normalise_tokens(pair_sources))),
        "projection_policy": np.asarray(snapshot.policy),
        "kappa_projection_policy": np.asarray(snapshot.kappa_policy),
        "projection_branch": np.asarray(snapshot.branch),
        "projection_regime": np.asarray(snapshot.regime),
        "history_order": np.asarray("oldest_to_newest"),
        "input_vector_role": np.asarray("effective_force"),
        "gradient_sign_convention": np.asarray("g_eff=-F_eff"),
        "two_loop_sign_convention": np.asarray(
            "raw_lbfgs_direction=H*F_eff=-H*g_eff"
        ),
        "two_loop_arithmetic": np.asarray("sy_divide"),
        "secant_y_sign_convention": np.asarray(
            "y=grad_second-grad_first=F_eff_first-F_eff_second"
        ),
    }
    payload.update(two_loop_trace)
    return payload


