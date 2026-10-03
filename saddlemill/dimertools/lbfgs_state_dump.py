"""Lossless diagnostic dumps and offline replay for SaddleMill translation L-BFGS.

This module is diagnostic-only.  It does not alter pair selection, curvature
safeguards, the two-loop recursion, max-step handling, or any scientific state.
NPZ files intentionally avoid object arrays so they can be loaded with
``allow_pickle=False``.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCHEMA = "saddlemill_translation_lbfgs_exact_state_v1"


def _safe_norm(value) -> float:
    x = np.asarray(value, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    scale = float(np.max(np.abs(x)))
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        return float("inf")
    return scale * float(np.linalg.norm(x / scale))


def rigid_translation_basis(ndof: int) -> np.ndarray:
    """Return an orthonormal Cartesian rigid-translation basis for 3N DOF."""
    ndof = int(ndof)
    if ndof < 0 or ndof % 3:
        raise ValueError("translation basis requires a 3N Cartesian vector length")
    natoms = ndof // 3
    if natoms == 0:
        return np.empty((0, 3), dtype=np.float64)
    q = np.zeros((ndof, 3), dtype=np.float64)
    scale = 1.0 / np.sqrt(float(natoms))
    for axis in range(3):
        q[axis::3, axis] = scale
    return q


def normalize_step_selector(value: object) -> str:
    """Normalize ``all``/``none`` or an integer step list without ambiguity."""
    if value is None:
        return "all"
    text = str(value).strip().lower()
    if text in {"", "all", "*"}:
        return "all"
    if text in {"none", "off"}:
        return "none"
    parts = text.replace(",", " ").split()
    steps = []
    for token in parts:
        step = int(token)
        if step < 0:
            raise ValueError("translation_state_dump_steps cannot contain negative steps")
        steps.append(step)
    return " ".join(str(x) for x in sorted(set(steps)))


def step_selected(selector: object, step: int) -> bool:
    normalized = normalize_step_selector(selector)
    if normalized == "all":
        return True
    if normalized == "none":
        return False
    return int(step) in {int(x) for x in normalized.split()}


def trace_two_loop(
    current_vector,
    s_history,
    y_history,
    *,
    initial_hessian: float,
    dynamic_h0: bool,
    sy_history=None,
    rho_history=None,
    arithmetic: str = "sy_divide",
):
    """Replay SaddleMill's canonical L-BFGS two-loop recursion exactly.

    ``current_vector`` is the effective force.  Therefore the returned direction
    is ``H * F_eff == -H * g_eff`` under SaddleMill's sign convention.
    History arrays must be ordered oldest -> newest.
    """
    force = np.asarray(current_vector, dtype=float).reshape(-1)
    S = np.asarray(s_history, dtype=float)
    Y = np.asarray(y_history, dtype=float)
    if S.ndim != 2 or Y.ndim != 2 or S.shape != Y.shape:
        raise ValueError("s_history and y_history must have identical shape (m, ndim)")
    if S.shape[1] != force.size:
        raise ValueError("history vector length does not match current vector")
    alpha0 = float(initial_hessian)
    if not np.isfinite(alpha0) or alpha0 <= 0.0:
        raise ValueError("initial_hessian must be finite and > 0")

    if sy_history is None:
        sy = np.asarray(
            [float(np.dot(S[i], Y[i])) for i in range(S.shape[0])],
            dtype=np.float64,
        )
    else:
        sy = np.asarray(sy_history, dtype=np.float64).reshape(-1)
        if sy.size != S.shape[0]:
            raise ValueError("sy_history length does not match L-BFGS history")
    arithmetic = str(arithmetic).strip().lower()
    if arithmetic not in {"sy_divide", "rho_multiply"}:
        raise ValueError("arithmetic must be sy_divide or rho_multiply")
    if rho_history is None:
        rho = np.divide(
            1.0, sy, out=np.full_like(sy, np.nan),
            where=np.isfinite(sy) & (sy != 0.0),
        )
    else:
        rho = np.asarray(rho_history, dtype=np.float64).reshape(-1)
        if rho.size != S.shape[0]:
            raise ValueError("rho_history length does not match L-BFGS history")
    q = force.copy()
    alphas_reversed = []
    first_indices = []
    first_q_before = []
    first_q_after = []
    for idx in range(S.shape[0] - 1, -1, -1):
        denominator = float(sy[idx])
        dot_sq = float(np.dot(S[idx], q))
        alpha = (
            dot_sq * float(rho[idx])
            if arithmetic == "rho_multiply"
            else dot_sq / denominator
        )
        first_indices.append(idx)
        first_q_before.append(_safe_norm(q))
        alphas_reversed.append(alpha)
        q -= alpha * Y[idx]
        first_q_after.append(_safe_norm(q))

    h0_inverse_scale = 1.0 / alpha0
    if bool(dynamic_h0) and S.shape[0]:
        yy = float(np.dot(Y[-1], Y[-1]))
        candidate = float(sy[-1]) / yy if yy > 0.0 else float("nan")
        if np.isfinite(candidate) and candidate > 0.0:
            h0_inverse_scale = candidate

    r = h0_inverse_scale * q
    second_indices = []
    betas = []
    second_r_before = []
    second_r_after = []
    for idx, alpha in zip(range(S.shape[0]), reversed(alphas_reversed)):
        second_indices.append(idx)
        second_r_before.append(_safe_norm(r))
        dot_yr = float(np.dot(Y[idx], r))
        beta = (
            dot_yr * float(rho[idx])
            if arithmetic == "rho_multiply"
            else dot_yr / float(sy[idx])
        )
        betas.append(beta)
        r += S[idx] * (alpha - beta)
        second_r_after.append(_safe_norm(r))

    trace = {
        "first_pair_history_index": np.asarray(first_indices, dtype=np.int64),
        "first_alpha": np.asarray(alphas_reversed, dtype=np.float64),
        "first_q_norm_before": np.asarray(first_q_before, dtype=np.float64),
        "first_q_norm_after": np.asarray(first_q_after, dtype=np.float64),
        "second_pair_history_index": np.asarray(second_indices, dtype=np.int64),
        "second_beta": np.asarray(betas, dtype=np.float64),
        "second_r_norm_before": np.asarray(second_r_before, dtype=np.float64),
        "second_r_norm_after": np.asarray(second_r_after, dtype=np.float64),
    }
    return r, float(h0_inverse_scale), trace


def _npz_value(value):
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise TypeError("object arrays are not permitted in exact L-BFGS dumps")
        return value
    if isinstance(value, (bool, np.bool_)):
        return np.asarray(value, dtype=np.bool_)
    if isinstance(value, (int, np.integer)):
        return np.asarray(value, dtype=np.int64)
    if isinstance(value, (float, np.floating)):
        return np.asarray(value, dtype=np.float64)
    if value is None:
        return np.asarray("", dtype="U1")
    return np.asarray(str(value))


def write_state_dump(path: os.PathLike | str, payload: Mapping[str, object], metadata=None) -> Path:
    """Atomically write one lossless compressed NPZ diagnostic state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {str(k): _npz_value(v) for k, v in payload.items()}
    meta = dict(metadata or {})
    meta.setdefault("schema", SCHEMA)
    data["metadata_json"] = np.asarray(
        json.dumps(meta, sort_keys=True, separators=(",", ":"), default=str)
    )
    fd, tmp_name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return path


def load_state_dump(path: os.PathLike | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def replay_state_dump(path: os.PathLike | str, *, rtol=5.0e-13, atol=1.0e-14):
    state = load_state_dump(path)
    required = {
        "current_two_loop_vector",
        "s_history",
        "y_history",
        "initial_hessian",
        "dynamic_h0",
        "raw_lbfgs_direction",
    }
    missing = sorted(required - set(state))
    if missing:
        raise KeyError("missing exact-replay keys: " + ", ".join(missing))
    replay, h0, trace = trace_two_loop(
        state["current_two_loop_vector"],
        state["s_history"],
        state["y_history"],
        initial_hessian=float(state["initial_hessian"]),
        dynamic_h0=bool(state["dynamic_h0"]),
        sy_history=state.get("sy_history"),
        rho_history=state.get("rho_history"),
        arithmetic=str(state.get("two_loop_arithmetic", np.asarray("sy_divide"))),
    )
    recorded = np.asarray(state["raw_lbfgs_direction"], dtype=float).reshape(-1)
    replay = np.asarray(replay, dtype=float).reshape(-1)
    exact = bool(np.array_equal(replay, recorded))
    close = bool(np.allclose(replay, recorded, rtol=rtol, atol=atol, equal_nan=True))
    diff = replay - recorded
    result = {
        "path": str(path),
        "exact_equal": exact,
        "allclose": close,
        "rtol": float(rtol),
        "atol": float(atol),
        "recorded_h0_inverse_scale": float(state.get("h0_inverse_scale", np.asarray(float("nan")))),
        "replayed_h0_inverse_scale": float(h0),
        "max_abs_difference": float(np.max(np.abs(diff), initial=0.0)),
        "difference_norm": _safe_norm(diff),
        "recorded_norm": _safe_norm(recorded),
        "replayed_norm": _safe_norm(replay),
        "first_loop_count": int(trace["first_alpha"].size),
        "second_loop_count": int(trace["second_beta"].size),
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a SaddleMill translation L-BFGS NPZ state dump")
    parser.add_argument("dump", nargs="+", help="NPZ dump(s) to validate")
    parser.add_argument("--rtol", type=float, default=5.0e-13)
    parser.add_argument("--atol", type=float, default=1.0e-14)
    args = parser.parse_args(argv)
    ok = True
    for name in args.dump:
        result = replay_state_dump(name, rtol=args.rtol, atol=args.atol)
        print(json.dumps(result, sort_keys=True))
        if not result["allclose"]:
            ok = False
    print("LBFGS_EXACT_REPLAY_OK=" + ("yes" if ok else "no"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
