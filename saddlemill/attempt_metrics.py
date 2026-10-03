"""Engine-independent terminal attempt metrics for SaddleMill searches.

This module is additive diagnostics only.  It reads already-computed state and
must never trigger a force evaluation, alter optimizer state, or participate in
convergence decisions.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Mapping

import numpy as np

from saddlemill.diagnostics_io import append_jsonl_durable

SCHEMA = "saddlemill_terminal_attempt_metrics_v1"

FIELDS = (
    "schema", "engine", "src_index", "rank", "attempt_id", "selected_index",
    "configured_reaction_type", "initial_reaction_type", "terminal_status",
    "scientific_convergence_reason", "converged", "final_real_fmax",
    "projected_fmax", "projected_fmax_scope", "optimizer_iterations",
    "physical_pes_calls", "physical_pes_calls_source", "legacy_n_force_calls",
    "dimer_forcecalls", "sella_pes_neval", "canonical_physical_pes_calls",
    "physical_pes_call_count_consistent", "pes_evaluation_wall_seconds",
    "mode_hvp_force_calls", "mode_solver_wall_seconds", "rotation_wall_seconds",
    "translation_wall_seconds", "hessian_calls", "hessian_wall_seconds",
    "diagonalization_wall_seconds", "total_attempt_wall_seconds",
    "science_config_id", "source_id", "initial_guess_id",
    "realized_initial_geometry_sha256", "metric_availability", "recorded_unix_ns",
)

NULL_VALUE = "value"
NULL_UNAVAILABLE = "unavailable"
NULL_NOT_APPLICABLE = "not_applicable"


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def science_config_id(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def geometry_sha256(atoms) -> str | None:
    """Exact realized geometry fingerprint; not a geometric-equivalence rule."""
    if atoms is None:
        return None
    try:
        h = hashlib.sha256()
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        pbc = np.asarray(atoms.get_pbc(), dtype=np.bool_)
        for arr in (numbers, positions, cell, pbc):
            h.update(str(arr.shape).encode("ascii"))
            h.update(arr.tobytes(order="C"))
        return "sha256:" + h.hexdigest()
    except Exception:
        return None


def _source_id(context) -> str | None:
    atoms = getattr(context, "atoms", None)
    info = getattr(atoms, "info", {}) if atoms is not None else {}
    for key in ("saddlemill_source_id", "source_id", "saddlemill_git_commit", "runtime_tree_sha256"):
        value = info.get(key) if isinstance(info, dict) else None
        if value:
            return str(value)
    for key in ("SADDLEMILL_SOURCE_ID", "SADDLEMILL_GIT_COMMIT", "SADDLEMILL_RUNTIME_TREE_SHA256"):
        value = os.environ.get(key)
        if value:
            return str(value)
    return None


def _initial_guess_id(context) -> str | None:
    atoms = getattr(context, "atoms", None)
    info = getattr(atoms, "info", {}) if atoms is not None else {}
    if isinstance(info, dict):
        value = info.get("initial_guess_id")
        if value:
            return str(value)
    return None


def _real_fmax(context):
    # Use the forces already retained by the attempt.  Never call get_forces()
    # from terminal instrumentation: doing so could add PES work or perturb
    # calculator/cache state.
    forces = getattr(context, "forces", None)
    if forces is None:
        return None
    try:
        arr = np.asarray(forces, dtype=float).reshape((-1, 3))
        return float(np.max(np.linalg.norm(arr, axis=1), initial=0.0))
    except Exception:
        return None


def _projected_fmax(context):
    opt = getattr(context, "dim_rlx", None)
    row = getattr(opt, "last_step_diagnostics", None) if opt is not None else None
    if isinstance(row, dict):
        return _finite_float(row.get("projected_fmax"))
    return None


def _physical_counts(context):
    """Return audited engine-native and canonical physical PES counters.

    Dimer's canonical ForceAccounting is preferred when present because it
    explicitly separates physical algorithm/diagnostic PES calls and cache hits.
    The historical ASE Dimer ``forcecalls`` counter is retained independently
    for audit/compatibility.  Sella's ``pes.neval`` is its energy/gradient
    evaluation counter and is retained both as the engine-native value and the
    selected physical counter.
    """
    engine = getattr(getattr(context, "run", None), "saddle_engine", None)
    d_atoms = getattr(context, "d_atoms", None)
    dimer_forcecalls = None
    canonical_calls = None
    sella_neval = None

    if engine == "sella":
        opt = getattr(context, "dim_rlx", None)
        pes = getattr(opt, "pes", None)
        sella_neval = _integer(getattr(pes, "neval", None))
        return {
            "physical": sella_neval,
            "source": "sella.pes.neval" if sella_neval is not None else None,
            "dimer": None,
            "sella": sella_neval,
            "canonical": None,
            "consistent": None,
        }

    try:
        dimer_forcecalls = int(d_atoms.control.get_counter("forcecalls"))
    except Exception:
        dimer_forcecalls = None
    history = getattr(d_atoms, "canonical_force_history", None)
    accounting = getattr(history, "accounting", None)
    try:
        canonical_calls = int(accounting.physical_total_pes_calls)
    except Exception:
        canonical_calls = None

    if canonical_calls is not None:
        physical = canonical_calls
        source = "canonical_force_history.accounting.physical_total_pes_calls"
    else:
        physical = dimer_forcecalls
        source = "ase_dimer_control.forcecalls" if dimer_forcecalls is not None else None
    consistent = (
        bool(canonical_calls == dimer_forcecalls)
        if canonical_calls is not None and dimer_forcecalls is not None else None
    )
    return {
        "physical": physical,
        "source": source,
        "dimer": dimer_forcecalls,
        "sella": None,
        "canonical": canonical_calls,
        "consistent": consistent,
    }


def _availability(row, engine, hessian_callback):
    """Explicit interpretation for nullable metric fields.

    ``unavailable`` means the quantity is scientifically meaningful but this
    baseline does not expose a safe, source-complete measurement without
    changing execution. ``not_applicable`` means the engine/path does not use
    that metric as a distinct phase under this instrumentation contract.
    """
    result = {}
    measured = (
        "final_real_fmax", "optimizer_iterations", "physical_pes_calls",
        "total_attempt_wall_seconds", "science_config_id",
        "realized_initial_geometry_sha256",
    )
    for key in measured:
        result[key] = NULL_VALUE if row.get(key) is not None else NULL_UNAVAILABLE

    if engine == "sella":
        result["projected_fmax"] = NULL_NOT_APPLICABLE
        result["rotation_wall_seconds"] = NULL_NOT_APPLICABLE
        # Sella does not expose the Dimer HVP/mode-solver phase through this API.
        result["mode_hvp_force_calls"] = NULL_NOT_APPLICABLE
        result["mode_solver_wall_seconds"] = NULL_NOT_APPLICABLE
    else:
        result["projected_fmax"] = NULL_VALUE if row.get("projected_fmax") is not None else NULL_UNAVAILABLE
        result["rotation_wall_seconds"] = NULL_VALUE if row.get("rotation_wall_seconds") is not None else NULL_UNAVAILABLE
        result["mode_hvp_force_calls"] = NULL_VALUE if row.get("mode_hvp_force_calls") is not None else NULL_UNAVAILABLE
        result["mode_solver_wall_seconds"] = NULL_VALUE if row.get("mode_solver_wall_seconds") is not None else NULL_UNAVAILABLE

    for key in ("pes_evaluation_wall_seconds", "translation_wall_seconds", "diagonalization_wall_seconds"):
        result[key] = NULL_VALUE if row.get(key) is not None else NULL_UNAVAILABLE

    if hessian_callback is None:
        result["hessian_calls"] = NULL_NOT_APPLICABLE
        result["hessian_wall_seconds"] = NULL_NOT_APPLICABLE
    else:
        result["hessian_calls"] = NULL_VALUE if row.get("hessian_calls") is not None else NULL_UNAVAILABLE
        result["hessian_wall_seconds"] = NULL_VALUE if row.get("hessian_wall_seconds") is not None else NULL_UNAVAILABLE

    for key in ("source_id", "initial_guess_id"):
        result[key] = NULL_VALUE if row.get(key) is not None else NULL_UNAVAILABLE
    return result


def build_terminal_attempt_metrics(context, *, status=None, converged=None, fallback_force_calls=None):
    run = context.run
    start_ns = getattr(context, "attempt_start_perf_ns", None)
    if start_ns is None:
        total_seconds = None
    else:
        total_seconds = max(0, time.perf_counter_ns() - int(start_ns)) / 1.0e9
    counts = _physical_counts(context)
    hessian_callback = getattr(getattr(context, "dim_rlx", None), "sm_hessian_callback", None)
    projected = _projected_fmax(context)
    engine = run.saddle_engine
    row = {
        "schema": SCHEMA,
        "engine": engine,
        "src_index": run.src_index,
        "rank": run.rank,
        "attempt_id": context.attempt_id,
        "selected_index": context.selected_index,
        "configured_reaction_type": context.configured_reaction_type,
        "initial_reaction_type": context.initial_reaction_type,
        "terminal_status": status if status is not None else context.status,
        "scientific_convergence_reason": context.stop_reason or status or context.status,
        "converged": bool(context.converged if converged is None else converged),
        "final_real_fmax": _real_fmax(context),
        "projected_fmax": projected,
        "projected_fmax_scope": "last_accepted_dimer_translation" if projected is not None else None,
        "optimizer_iterations": int(getattr(getattr(context, "dim_rlx", None), "nsteps", 0) or 0),
        "physical_pes_calls": counts["physical"],
        "physical_pes_calls_source": counts["source"],
        "legacy_n_force_calls": _integer(fallback_force_calls) if fallback_force_calls is not None else _integer(getattr(context, "n_force_calls", None)),
        "dimer_forcecalls": counts["dimer"],
        "sella_pes_neval": counts["sella"],
        "canonical_physical_pes_calls": counts["canonical"],
        "physical_pes_call_count_consistent": counts["consistent"],
        # No calculator wrapper is added here: a wrapper could perturb caching or
        # calculator behavior. Source-complete PES timing therefore stays null.
        "pes_evaluation_wall_seconds": None,
        "mode_hvp_force_calls": None,
        "mode_solver_wall_seconds": None,
        "rotation_wall_seconds": None,
        "translation_wall_seconds": None,
        "hessian_calls": int(getattr(hessian_callback, "calls", 0)) if hessian_callback is not None else None,
        "hessian_wall_seconds": _finite_float(getattr(hessian_callback, "total_seconds", None)) if hessian_callback is not None else None,
        "diagonalization_wall_seconds": None,
        "total_attempt_wall_seconds": total_seconds,
        "science_config_id": science_config_id(run.config_dict),
        "source_id": _source_id(context),
        "initial_guess_id": _initial_guess_id(context),
        "realized_initial_geometry_sha256": getattr(context, "realized_initial_geometry_sha256", None),
        "metric_availability": None,
        "recorded_unix_ns": time.time_ns(),
    }
    row["metric_availability"] = _availability(row, engine, hessian_callback)
    return {key: row.get(key) for key in FIELDS}


def write_terminal_attempt_metrics(context, **kwargs):
    row = build_terminal_attempt_metrics(context, **kwargs)
    append_jsonl_durable(context.run.metrics_file, row)
    return row
