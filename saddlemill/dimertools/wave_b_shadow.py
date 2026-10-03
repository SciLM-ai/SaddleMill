"""shared-runtime wiring for the passive dense-replay diagnostic.

The worker-owned replay kernels are not modified here.  This adapter consumes an
already-frozen exact replay payload and records only diagnostic state/timing.
"""
from __future__ import annotations

from time import perf_counter_ns
from typing import Mapping

import numpy as np

from saddlemill.dimertools.dense_replay_diagnostics import DenseReplayLimits
from saddlemill.dimertools.qn_shadow_diagnostics import SHADOW_SCHEMA, build_shadow_diagnostic


def normalize_qn_shadow_options(value: Mapping[str, object] | None) -> dict[str, object]:
    raw = dict(value or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "max_dimension": int(raw.get("max_dimension", 512)),
        "max_matrix_bytes": int(raw.get("max_matrix_bytes", 134217728)),
        "max_work_units": int(raw.get("max_work_units", 200000000)),
        "store_dense_matrix": bool(raw.get("store_dense_matrix", False)),
        "near_zero_absolute": float(raw.get("near_zero_absolute", 1.0e-12)),
        "near_zero_relative": float(raw.get("near_zero_relative", 1.0e-10)),
        "consumer": str(raw.get("consumer", "")),
        "residual_kind": str(raw.get("residual_kind", "")),
        "force_interpretation": str(raw.get("force_interpretation", "")),
        "model_type": str(raw.get("model_type", "")),
    }


def qn_shadow_options_from_dimeratoms(dimeratoms) -> dict[str, object]:
    wave = dict(getattr(dimeratoms, "wave_b_options", {}) or {})
    return normalize_qn_shadow_options(wave.get("qn_shadow", {}))


def ensure_explicit_active_mask(payload: dict[str, object], owner=None) -> None:
    """Attach only an already-frozen production mask; never infer a projector.

    Canonical reconstruction snapshots already carry the exact active-atom mask
    used by the production projection.  Convert that payload field to the flat
    replay mask.  If no production mask was captured, leave the optional dense-replay
    field absent instead of reconstructing one from topology/constraints here.
    ``owner`` is accepted only for backward call compatibility and is ignored.
    """
    if "replay_active_dof_mask" in payload:
        return
    atom_mask = np.asarray(payload.get("snapshot_active_atom_mask", []), dtype=bool)
    if atom_mask.size == 0:
        return
    current = np.asarray(payload.get("current_two_loop_vector", []))
    n = int(current.size)
    flat = np.repeat(atom_mask.reshape(-1), 3)
    if flat.size != n:
        return
    payload["replay_active_dof_mask"] = flat


def unavailable_shadow_row(*, consumer: str, residual_kind: str, force_interpretation: str, model_type: str, reason: str) -> dict[str, object]:
    """Structured passive outcome when production exposes no exact frozen window."""
    return {
        "schema": SHADOW_SCHEMA,
        "consumer": str(consumer),
        "residual_kind": str(residual_kind),
        "force_interpretation": str(force_interpretation),
        "model_type": str(model_type),
        "model_is_physical_hessian": 0,
        "dense_replay_available": 0,
        "dense_replay_reason": str(reason),
        "diagnostic_extra_force_calls": 0,
        "diagnostic_mutates_optimizer_state": 0,
        "shadow_total_ns": 0,
        "t14_shadow_wall_ns": 0,
    }


def run_shadow(
    replay_payload: Mapping[str, object], *, options: Mapping[str, object],
    consumer: str, residual_kind: str, force_interpretation: str,
    model_type: str, owner=None,
):
    cfg = normalize_qn_shadow_options(options)
    if not cfg["enabled"]:
        return None, 0
    payload = dict(replay_payload)
    ensure_explicit_active_mask(payload, owner=owner)
    limits = DenseReplayLimits(
        max_dimension=int(cfg["max_dimension"]),
        max_matrix_bytes=int(cfg["max_matrix_bytes"]),
        max_work_units=int(cfg["max_work_units"]),
    )
    start = perf_counter_ns()
    result = build_shadow_diagnostic(
        payload,
        consumer=consumer,
        residual_kind=residual_kind,
        force_interpretation=force_interpretation,
        model_type=model_type,
        limits=limits,
        include_matrix=bool(cfg["store_dense_matrix"]),
        near_zero_absolute=float(cfg["near_zero_absolute"]),
        near_zero_relative=float(cfg["near_zero_relative"]),
    )
    elapsed = perf_counter_ns() - start
    result.row["t14_shadow_wall_ns"] = int(elapsed)
    result.row["diagnostic_extra_force_calls"] = 0
    result.row["diagnostic_mutates_optimizer_state"] = 0
    return result, int(elapsed)


__all__ = [
    "ensure_explicit_active_mask", "normalize_qn_shadow_options",
    "qn_shadow_options_from_dimeratoms", "run_shadow",
    "unavailable_shadow_row",
]
