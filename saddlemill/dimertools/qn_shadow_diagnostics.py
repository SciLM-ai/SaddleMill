"""Common minimization/saddle shadow-QN diagnostic schema.

This module adapts an already-built exact L-BFGS replay payload to the bounded
calculator-free dense replay in :mod:`dense_replay_diagnostics`.  It is intentionally
not wired to configuration or optimizer call sites; shared-runtime owns that integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping

import numpy as np

from saddlemill.dimertools.dense_replay_diagnostics import (
    DENSE_REPLAY_SCHEMA,
    DenseReplayLimits,
    DenseReplayResult,
    FrozenReplayWindow,
    replay_dense_bfgs,
)

Array = np.ndarray

SHADOW_SCHEMA = "saddlemill_qn_shadow_diagnostic_v1"

_RAW_PHYSICAL_TOKENS = frozenset({"raw_physical_force", "raw_physical_gradient"})
_PHYSICAL_MODEL_TYPES = frozenset({"ordinary_bfgs_hessian", "physical_bfgs_hessian"})


@dataclass
class ShadowDiagnosticResult:
    """One common schema row plus optional opt-in numerical artifacts."""

    row: dict[str, object]
    dense_direction: Array | None = None
    matrix: Array | None = None
    pair_provenance: list[dict[str, object]] = field(default_factory=list)
    history_provenance: list[dict[str, object]] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.row.get("dense_replay_available", 0))


def _scalar(value: object, default: object = "") -> object:
    if value is None:
        return default
    array = np.asarray(value)
    if array.size != 1:
        return default
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    if isinstance(item, np.generic):
        return item.item()
    return item


def _text(value: object, default: str = "") -> str:
    item = _scalar(value, default)
    return default if item is None else str(item)


def _optional_vector(payload: Mapping[str, object], key: str, length: int) -> Array | None:
    if key not in payload:
        return None
    arr = np.asarray(payload[key])
    if arr.size == 0:
        return None
    arr = np.asarray(arr, dtype=bool).reshape(-1)
    if arr.size != length:
        raise ValueError(f"{key} length does not match replay dimension")
    return arr


def _require_array(payload: Mapping[str, object], key: str) -> Array:
    if key not in payload:
        raise KeyError(f"missing exact replay field: {key}")
    return np.asarray(payload[key])


def frozen_window_from_payload(payload: Mapping[str, object]) -> FrozenReplayWindow:
    """Parse only the final frozen replay window; never rebuild pair admission."""

    current = _require_array(payload, "current_two_loop_vector")
    s = _require_array(payload, "s_history")
    y = _require_array(payload, "y_history")
    sy = _require_array(payload, "sy_history")
    actual = _require_array(payload, "raw_lbfgs_direction")
    if "h0_inverse_scale" not in payload:
        raise KeyError("missing exact replay field: h0_inverse_scale")
    h0 = float(_scalar(payload["h0_inverse_scale"]))
    mask = _optional_vector(payload, "replay_active_dof_mask", current.size)
    return FrozenReplayWindow(
        current_vector=current,
        s_history=s,
        y_history=y,
        sy_history=sy,
        actual_direction=actual,
        h0_inverse_scale=h0,
        active_dof_mask=mask,
    )


def _array_item(values: object, index: int, default: object = "") -> object:
    arr = np.asarray(values)
    if arr.ndim == 0 or index >= arr.shape[0]:
        return default
    item = arr[index]
    if isinstance(item, np.generic):
        item = item.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return item


def _candidate_provenance(payload: Mapping[str, object]) -> list[dict[str, object]]:
    if "candidate_action" not in payload:
        return []
    count = int(np.asarray(payload["candidate_action"]).reshape(-1).size)
    rows: list[dict[str, object]] = []
    for index in range(count):
        raw_sy = _array_item(payload.get("candidate_raw_sTy", []), index, "")
        stored_sy = _array_item(payload.get("candidate_sTy", []), index, "")
        row = {
            "index": index,
            "source": str(_array_item(payload.get("candidate_source", []), index, "")),
            "detail_source": str(
                _array_item(payload.get("candidate_detail_source", []), index, "")
            ),
            "state_id": int(_array_item(payload.get("candidate_state_id", []), index, -1)),
            "serial": int(_array_item(payload.get("candidate_serial", []), index, -1)),
            "action": str(_array_item(payload.get("candidate_action", []), index, "")),
            "rejection_reason": str(
                _array_item(payload.get("candidate_rejection_reason", []), index, "")
            ),
            "used": bool(_array_item(payload.get("candidate_used", []), index, False)),
            "guard_pass": bool(
                _array_item(payload.get("candidate_guard_pass", []), index, False)
            ),
            "raw_s_dot_y": _finite_or_blank(raw_sy),
            "stored_s_dot_y": _finite_or_blank(stored_sy),
        }
        rows.append(row)
    return rows


def _history_provenance(payload: Mapping[str, object], history_size: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(history_size):
        rows.append(
            {
                "index": index,
                "source": str(_array_item(payload.get("history_source", []), index, "")),
                "detail_source": str(
                    _array_item(payload.get("history_detail_source", []), index, "")
                ),
                "state_id": int(_array_item(payload.get("history_state_id", []), index, -1)),
                "serial": int(_array_item(payload.get("history_serial", []), index, -1)),
                "s_dot_y": _finite_or_blank(
                    _array_item(payload.get("sy_history", []), index, "")
                ),
            }
        )
    return rows


def _finite_or_blank(value: object) -> float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if np.isfinite(number) else ""


def _json_ready(value: object) -> object:
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def deterministic_json(value: object) -> str:
    """Deterministic strict JSON used for nested row fields and test evidence."""

    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def serialize_shadow_row(row: Mapping[str, object]) -> str:
    """Serialize one common-schema row deterministically."""

    return deterministic_json(dict(row))


def _mode_identity(payload: Mapping[str, object]) -> str:
    if "snapshot_mode" not in payload:
        return ""
    mode = np.asarray(payload["snapshot_mode"], dtype=np.float64)
    if mode.size == 0:
        return ""
    digest = hashlib.sha256()
    digest.update(str(mode.shape).encode("ascii"))
    digest.update(mode.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()[:20]


def _projector_identity(payload: Mapping[str, object]) -> str:
    fields = {
        key: _json_ready(payload[key])
        for key in (
            "projection_policy",
            "kappa_projection_policy",
            "projection_branch",
            "projection_regime",
            "snapshot_gamma_1",
            "snapshot_gamma_2",
            "snapshot_active_atom_mask",
            "replay_active_dof_mask",
            "rigid_translation_projection",
        )
        if key in payload
    }
    if not fields:
        return ""
    return "sha256:" + hashlib.sha256(deterministic_json(fields).encode("ascii")).hexdigest()[:20]


def build_shadow_diagnostic(
    replay_payload: Mapping[str, object],
    *,
    consumer: str,
    residual_kind: str,
    force_interpretation: str,
    model_type: str,
    limits: DenseReplayLimits | None = None,
    include_matrix: bool = False,
    mode_identity: str = "",
    projector_identity: str = "",
    near_zero_absolute: float = 1.0e-12,
    near_zero_relative: float = 1.0e-10,
) -> ShadowDiagnosticResult:
    """Build one passive shadow diagnostic from an exact frozen replay payload.

    ``force_interpretation`` is intentionally explicit.  A replay using an MMF
    effective force, translation residual, or rotation residual is *not* labeled
    a physical Hessian even though the algebraic update is BFGS-shaped.
    """

    consumer = str(consumer).strip()
    residual_kind = str(residual_kind).strip()
    interpretation = str(force_interpretation).strip().lower()
    model_type = str(model_type).strip().lower()
    if not consumer:
        raise ValueError("consumer must be nonempty")
    if not residual_kind:
        raise ValueError("residual_kind must be nonempty")
    if not interpretation:
        raise ValueError("force_interpretation must be nonempty")
    if not model_type:
        raise ValueError("model_type must be nonempty")

    physical_input = interpretation in _RAW_PHYSICAL_TOKENS
    if (not physical_input) and model_type in _PHYSICAL_MODEL_TYPES:
        raise ValueError(
            "effective/residual replay cannot be labeled as a physical Hessian model"
        )

    window = frozen_window_from_payload(replay_payload)
    pair_rows = _candidate_provenance(replay_payload)
    history_rows = _history_provenance(replay_payload, window.history_size)
    dense: DenseReplayResult = replay_dense_bfgs(
        window,
        limits=limits,
        include_matrix=include_matrix,
        near_zero_absolute=near_zero_absolute,
        near_zero_relative=near_zero_relative,
    )

    mode_id = str(mode_identity).strip() or _mode_identity(replay_payload)
    projector_id = str(projector_identity).strip() or _projector_identity(replay_payload)
    actions: dict[str, int] = {}
    rejections: dict[str, int] = {}
    for item in pair_rows:
        action = str(item["action"])
        actions[action] = actions.get(action, 0) + 1
        reason = str(item["rejection_reason"])
        if reason:
            rejections[reason] = rejections.get(reason, 0) + 1

    row: dict[str, object] = {
        "schema": SHADOW_SCHEMA,
        "dense_replay_schema": DENSE_REPLAY_SCHEMA,
        "consumer": consumer,
        "residual_kind": residual_kind,
        "force_interpretation": interpretation,
        "model_type": model_type,
        "model_is_physical_hessian": int(physical_input and model_type in _PHYSICAL_MODEL_TYPES),
        "nonphysical_model_reason": (
            "" if physical_input else "effective_or_residual_replay_not_physical_hessian"
        ),
        "mode_identity": mode_id,
        "projector_identity": projector_id,
        "projection_policy": _text(replay_payload.get("projection_policy", "")),
        "projection_branch": _text(replay_payload.get("projection_branch", "")),
        "projection_regime": _text(replay_payload.get("projection_regime", "")),
        "input_vector_role": _text(replay_payload.get("input_vector_role", "")),
        "gradient_sign_convention": _text(
            replay_payload.get("gradient_sign_convention", "")
        ),
        "two_loop_sign_convention": _text(
            replay_payload.get("two_loop_sign_convention", "")
        ),
        "history_order": _text(replay_payload.get("history_order", "oldest_to_newest")),
        "two_loop_arithmetic": _text(replay_payload.get("two_loop_arithmetic", "")),
        "pair_safeguard": _text(replay_payload.get("pair_safeguard", "")),
        "max_pairs": int(_scalar(replay_payload.get("max_pairs", 0), 0)),
        "dynamic_h0": int(bool(_scalar(replay_payload.get("dynamic_h0", False), False))),
        "initial_hessian": _finite_or_blank(_scalar(replay_payload.get("initial_hessian", ""))),
        "h0_inverse_scale": window.h0_inverse_scale,
        "history_size": window.history_size,
        "window_size": window.history_size,
        "h0_contract": (
            "dynamic_resolved_scale"
            if bool(_scalar(replay_payload.get("dynamic_h0", False), False))
            else "fixed_resolved_scale"
        ),
        "candidate_pair_count": len(pair_rows),
        "pair_action_counts_json": deterministic_json(actions),
        "pair_rejection_counts_json": deterministic_json(rejections),
        "pair_provenance_json": deterministic_json(pair_rows),
        "history_provenance_json": deterministic_json(history_rows),
        # Explicit accounting boundary: every duration below is shadow-only and
        # must not be added to algorithm QN timing/counters by this module.
        "timing_accounting_kind": "diagnostic_shadow_compute",
        "algorithm_timing_mutated": 0,
        "algorithm_force_calls_added": 0,
        "diagnostic_force_calls_added": 0,
    }
    row.update(dense.metrics)
    # Preserve the common schema's own name after merging the low-level metrics.
    row["schema"] = SHADOW_SCHEMA
    return ShadowDiagnosticResult(
        row=row,
        dense_direction=(None if dense.dense_direction is None else np.array(dense.dense_direction, copy=True)),
        matrix=(None if dense.matrix is None else np.array(dense.matrix, copy=True)),
        pair_provenance=pair_rows,
        history_provenance=history_rows,
    )
