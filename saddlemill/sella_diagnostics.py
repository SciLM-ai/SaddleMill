"""Passive diagnostics for Sella optimizers.

This module intentionally does not import or patch Sella.  It observes the live
optimizer/PES state after a normal optimizer step and writes additive JSONL.
Unavailable quantities are recorded as ``None`` rather than recomputed through
Sella's internal stepper machinery, because doing so could alter caches or add
algorithmic work that is not passive.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from saddlemill.diagnostics_io import BufferedJSONLAppender


def _jsonable(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        return int(value)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return repr(value)


def _safe_attr(obj: object, name: str, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _safe_dense_hessian(pes: object):
    H = _safe_attr(pes, "H")
    if H is None:
        return None, None
    asarray = _safe_attr(H, "asarray")
    try:
        B = np.asarray(asarray() if callable(asarray) else asarray, dtype=float)
    except Exception:
        return H, None
    if B.ndim != 2 or B.shape[0] != B.shape[1] or not np.all(np.isfinite(B)):
        return H, None
    return H, B


def _position_vector(optimizer: object):
    pes = _safe_attr(optimizer, "pes")
    for obj, name in ((pes, "get_x"), (pes, "x")):
        if obj is None:
            continue
        value = _safe_attr(obj, name)
        try:
            arr = np.asarray(value() if callable(value) else value, dtype=float).reshape(-1)
            if arr.size and np.all(np.isfinite(arr)):
                return arr.copy()
        except Exception:
            pass
    atoms = _safe_attr(optimizer, "atoms")
    getter = _safe_attr(atoms, "get_positions")
    try:
        arr = np.asarray(getter(), dtype=float).reshape(-1)
        return arr.copy()
    except Exception:
        return None


def _subspace_size(H: object) -> int | None:
    if H is None:
        return None
    for name in ("Vs", "V", "vecs", "evecs"):
        value = _safe_attr(H, name)
        if value is None:
            continue
        try:
            arr = np.asarray(value)
            if arr.ndim == 2:
                return int(min(arr.shape))
        except Exception:
            pass
    evals = _safe_attr(H, "evals")
    try:
        arr = np.asarray(evals).reshape(-1)
        if arr.size:
            return int(arr.size)
    except Exception:
        pass
    return None


class SellaPassiveQNRecorder:
    """ASE-style callback that observes Sella without changing its step."""

    def __init__(self, optimizer: object, path: str | os.PathLike, **metadata):
        self.optimizer = optimizer
        self.path = Path(path)
        self.metadata = dict(metadata)
        self.cumulative_ns = 0
        self._previous_x = _position_vector(optimizer)
        self._writer = BufferedJSONLAppender(self.path)

    def close(self):
        self._writer.close()

    def __call__(self):
        started = perf_counter_ns()
        opt = self.optimizer
        pes = _safe_attr(opt, "pes")
        H, B = _safe_dense_hessian(pes)
        row = dict(self.metadata)
        row.update({
            "step": int(_safe_attr(opt, "nsteps", 0) or 0),
            "diagnostic_schema": "sella_passive_qn_v1",
            "trust_radius": _jsonable(_safe_attr(opt, "delta")),
            "update_type": _jsonable(_safe_attr(H, "update_method")),
            "update_symm": _jsonable(_safe_attr(H, "symm")),
            "multisecant_block_size": _jsonable(_safe_attr(H, "block_size")),
            "multisecant_vectors_used": _jsonable(_safe_attr(H, "nvec")),
            "multisecant_condition": _jsonable(_safe_attr(H, "cond")),
            "ritz_subspace_size": _subspace_size(H),
            "raw_prfo_step_norm": None,
            "raw_prfo_step_available": 0,
            "raw_to_restricted_ratio": None,
        })
        x = _position_vector(opt)
        actual_norm = None
        actual_max_atom = None
        if x is not None and self._previous_x is not None and x.shape == self._previous_x.shape:
            dx = x - self._previous_x
            actual_norm = float(np.linalg.norm(dx))
            if dx.size % 3 == 0:
                actual_max_atom = float(np.max(np.linalg.norm(dx.reshape(-1, 3), axis=1), initial=0.0))
        if x is not None:
            self._previous_x = x
        row["accepted_restricted_step_norm"] = actual_norm
        row["accepted_restricted_step_max_atom"] = actual_max_atom

        if B is None:
            row.update({
                "hessian_available": 0,
                "hessian_dimension": None,
                "hessian_eigenvalue_min": None,
                "hessian_eigenvalue_max": None,
                "hessian_min_abs_eigenvalue": None,
                "hessian_negative_eigenvalues": None,
                "hessian_frobenius_norm": None,
                "hessian_operator_norm": None,
                "hessian_abs_spectral_spread": None,
            })
        else:
            eig = np.linalg.eigvalsh(0.5 * (B + B.T))
            abs_eig = np.abs(eig)
            min_abs = float(np.min(abs_eig, initial=np.inf))
            max_abs = float(np.max(abs_eig, initial=0.0))
            row.update({
                "hessian_available": 1,
                "hessian_dimension": int(B.shape[0]),
                "hessian_eigenvalue_min": float(np.min(eig)),
                "hessian_eigenvalue_max": float(np.max(eig)),
                "hessian_min_abs_eigenvalue": min_abs,
                "hessian_negative_eigenvalues": int(np.count_nonzero(eig < 0.0)),
                "hessian_frobenius_norm": float(np.linalg.norm(B, ord="fro")),
                "hessian_operator_norm": max_abs,
                # Explicit definition: max |lambda| / min nonzero |lambda|.
                "hessian_abs_spectral_spread": (
                    None if not np.isfinite(min_abs) or min_abs <= np.finfo(float).eps * max(1.0, max_abs)
                    else max_abs / min_abs
                ),
            })
        elapsed = perf_counter_ns() - started
        self.cumulative_ns += elapsed
        row["diagnostic_cpu_ns"] = int(elapsed)
        row["diagnostic_cpu_cumulative_ns"] = int(self.cumulative_ns)
        self._writer.append(row, default=_jsonable)


# W5-009: exception-only, opt-in capture for Sella least-squares failures.
# This instrumentation never retries, regularizes, replaces, or suppresses the
# failing linear-algebra operation.  The original exception is always re-raised.

def _array_summary(value: Any) -> dict[str, Any]:
    try:
        arr = np.asarray(value)
    except Exception as exc:
        return {"available": 0, "conversion_error": f"{type(exc).__name__}: {exc}"}
    finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.number) else np.ones(arr.shape, dtype=bool)
    finite_values = arr[finite] if arr.size else arr.reshape(-1)
    result = {
        "available": 1,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "nbytes": int(arr.nbytes),
        "finite": bool(np.all(finite)) if arr.size else True,
        "nonfinite_count": int(arr.size - np.count_nonzero(finite)) if arr.size else 0,
    }
    if finite_values.size and np.issubdtype(arr.dtype, np.number):
        vals = np.asarray(finite_values, dtype=float)
        result.update({
            "finite_min": float(np.min(vals)),
            "finite_max": float(np.max(vals)),
            "finite_max_abs": float(np.max(np.abs(vals))),
            "finite_l2_norm": float(np.linalg.norm(vals)),
        })
    else:
        result.update({
            "finite_min": None,
            "finite_max": None,
            "finite_max_abs": None,
            "finite_l2_norm": None,
        })
    return result


def _array_sha256(value: Any) -> str | None:
    try:
        from hashlib import sha256
        arr = np.ascontiguousarray(np.asarray(value))
        h = sha256()
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(repr(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.view(np.uint8).tobytes())
        return h.hexdigest()
    except Exception:
        return None


def _bounded_spectrum(A: np.ndarray, max_values: int = 64) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": 0,
        "svd_error": None,
        "eigvalsh_error": None,
        "condition_estimate_2": None,
        "singular_values": None,
        "singular_values_head": None,
        "singular_values_tail": None,
        "gram_eigenvalues": None,
        "gram_eigenvalues_head": None,
        "gram_eigenvalues_tail": None,
    }
    try:
        A = np.asarray(A, dtype=float)
    except Exception as exc:
        result["svd_error"] = f"array_conversion: {type(exc).__name__}: {exc}"
        return result
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.size == 0:
        result["svd_error"] = "not_a_nonempty_square_matrix"
        return result
    if not np.all(np.isfinite(A)):
        result["svd_error"] = "skipped_nonfinite_input"
        result["eigvalsh_error"] = "skipped_nonfinite_input"
        return result
    result["attempted"] = 1

    def bounded(values):
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size <= max_values:
            return values.tolist(), None, None
        half = max(1, max_values // 2)
        return None, values[:half].tolist(), values[-half:].tolist()

    try:
        singular = np.linalg.svd(A, compute_uv=False)
        full, head, tail = bounded(singular)
        result["singular_values"] = full
        result["singular_values_head"] = head
        result["singular_values_tail"] = tail
        if singular.size:
            largest = float(np.max(singular))
            smallest = float(np.min(singular))
            result["singular_value_max"] = largest
            result["singular_value_min"] = smallest
            result["condition_estimate_2"] = (
                None if smallest <= 0.0 else largest / smallest
            )
    except Exception as exc:
        result["svd_error"] = f"{type(exc).__name__}: {exc}"

    try:
        sym = 0.5 * (A + A.T)
        eig = np.linalg.eigvalsh(sym)
        full, head, tail = bounded(eig)
        result["gram_eigenvalues"] = full
        result["gram_eigenvalues_head"] = head
        result["gram_eigenvalues_tail"] = tail
        if eig.size:
            result["gram_eigenvalue_min"] = float(np.min(eig))
            result["gram_eigenvalue_max"] = float(np.max(eig))
    except Exception as exc:
        result["eigvalsh_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _find_symmetrize_y2_locals(exc: BaseException):
    tb = exc.__traceback__
    selected = None
    while tb is not None:
        frame = tb.tb_frame
        if frame.f_code.co_name == "symmetrize_Y2":
            module_name = str(frame.f_globals.get("__name__", ""))
            if module_name == "sella.hessian_update" or module_name.endswith(".hessian_update"):
                selected = dict(frame.f_locals)
        tb = tb.tb_next
    return selected


def _runtime_linalg_context() -> dict[str, Any]:
    import io
    import platform
    import sys
    from contextlib import redirect_stdout

    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception as exc:
        scipy_version = f"unavailable: {type(exc).__name__}: {exc}"

    show_config = None
    try:
        stream = io.StringIO()
        with redirect_stdout(stream):
            np.show_config()
        show_config = stream.getvalue()[:12000]
    except Exception as exc:
        show_config = f"unavailable: {type(exc).__name__}: {exc}"

    thread_env = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "pid": os.getpid(),
        "thread_env": thread_env,
        "numpy_show_config": show_config,
    }


class SellaLinalgFailureRecorder:
    """Attempt-scoped recorder for W5-009-style Sella linear-algebra failures.

    The recorder wraps one live ``PES.diag`` method.  It changes no arguments or
    return values.  On ``numpy.linalg.LinAlgError`` it inspects the already-raised
    traceback to recover ``symmetrize_Y2`` locals, writes bounded diagnostics and
    optionally a bounded NPZ with exact solver operands, then re-raises the same
    exception.  No fallback or retry is performed.
    """

    schema = "sella_linalg_failure_v1"

    def __init__(
        self,
        optimizer: object,
        json_path: str | os.PathLike,
        operand_path: str | os.PathLike | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        max_capture_bytes: int = 1 << 20,
        max_spectrum_values: int = 64,
    ):
        self.optimizer = optimizer
        self.json_path = Path(json_path)
        self.operand_path = Path(operand_path) if operand_path is not None else None
        self.metadata = dict(metadata or {})
        self.max_capture_bytes = int(max_capture_bytes)
        self.max_spectrum_values = int(max_spectrum_values)
        self._pes = None
        self._diag_existed = False
        self._diag_previous = None
        self._installed = False
        self.records_written = 0
        self.last_diagnostic_error = None

    def install(self):
        if self._installed:
            return self
        pes = _safe_attr(self.optimizer, "pes")
        if pes is None:
            raise RuntimeError("Sella linear-algebra diagnostics require optimizer.pes")
        original = getattr(pes, "diag")
        dct = getattr(pes, "__dict__", {})
        self._diag_existed = "diag" in dct
        self._diag_previous = dct.get("diag")
        recorder = self

        from types import MethodType

        def wrapped_diag(pes_self, *args, **kwargs):
            try:
                return original(*args, **kwargs)
            except np.linalg.LinAlgError as exc:
                try:
                    recorder.record(exc)
                except Exception as diag_exc:  # diagnostics must never mask the solver error
                    recorder.last_diagnostic_error = (
                        f"{type(diag_exc).__name__}: {diag_exc}"
                    )
                raise

        setattr(pes, "diag", MethodType(wrapped_diag, pes))
        self._pes = pes
        self._installed = True
        return self

    def close(self):
        if not self._installed or self._pes is None:
            return
        if self._diag_existed:
            setattr(self._pes, "diag", self._diag_previous)
        else:
            try:
                delattr(self._pes, "diag")
            except AttributeError:
                pass
        self._installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _capture_operands(self, arrays: dict[str, np.ndarray]):
        result = {
            "path": None if self.operand_path is None else str(self.operand_path),
            "max_capture_bytes": self.max_capture_bytes,
            "arrays_saved": [],
            "bytes_saved_uncompressed": 0,
            "skipped": False,
            "skip_reason": None,
        }
        if self.operand_path is None:
            result["skipped"] = True
            result["skip_reason"] = "operand_path_not_configured"
            return result

        converted = {
            name: np.asarray(value)
            for name, value in arrays.items()
            if value is not None
        }
        all_bytes = sum(int(arr.nbytes) for arr in converted.values())
        if all_bytes <= self.max_capture_bytes:
            selected = converted
        else:
            selected = {
                name: converted[name]
                for name in ("lstsq_A", "lstsq_b")
                if name in converted
            }
            selected_bytes = sum(int(arr.nbytes) for arr in selected.values())
            if not selected or selected_bytes > self.max_capture_bytes:
                result["skipped"] = True
                result["skip_reason"] = "exact_solver_operands_exceed_capture_bound"
                return result

        self.operand_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.operand_path, **selected)
        result["arrays_saved"] = sorted(selected)
        result["bytes_saved_uncompressed"] = sum(
            int(arr.nbytes) for arr in selected.values()
        )
        return result

    def record(self, exc: np.linalg.LinAlgError):
        opt = self.optimizer
        pes = _safe_attr(opt, "pes")
        H = _safe_attr(pes, "H")
        row: dict[str, Any] = dict(self.metadata)
        row.update({
            "diagnostic_schema": self.schema,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "optimizer_step": _jsonable(_safe_attr(opt, "nsteps")),
            "trust_radius": _jsonable(_safe_attr(opt, "delta")),
            "pes_evaluations": _jsonable(_safe_attr(pes, "neval")),
            "hessian_update_method": _jsonable(_safe_attr(H, "update_method")),
            "hessian_symm": _jsonable(_safe_attr(H, "symm")),
            "hessian_initialized": _jsonable(_safe_attr(H, "initialized")),
            "runtime": _runtime_linalg_context(),
        })

        locals_ = _find_symmetrize_y2_locals(exc)
        row["symmetrize_y2_trace_found"] = int(locals_ is not None)
        arrays_for_capture: dict[str, np.ndarray] = {}
        if locals_ is None:
            row["trace_note"] = (
                "No sella.hessian_update.symmetrize_Y2 frame was present; "
                "no solver operands were inferred."
            )
        else:
            i = int(locals_.get("i", -1))
            S = np.asarray(locals_.get("S"))
            Y = np.asarray(locals_.get("Y"))
            STS = np.asarray(locals_.get("STS"))
            YTS = np.asarray(locals_.get("YTS"))
            dYTS = np.asarray(locals_.get("dYTS"))
            A = STS[:i, :i] if i >= 0 else None
            b = (
                YTS[i, :i].T - YTS[:i, i] - dYTS[:i, i]
                if i >= 0 else None
            )
            row.update({
                "symmetrize_index_i": i,
                "S": _array_summary(S),
                "Y": _array_summary(Y),
                "STS": _array_summary(STS),
                "lstsq_A": _array_summary(A),
                "lstsq_b": _array_summary(b),
                "S_sha256": _array_sha256(S),
                "Y_sha256": _array_sha256(Y),
                "STS_sha256": _array_sha256(STS),
                "lstsq_A_sha256": _array_sha256(A),
                "lstsq_b_sha256": _array_sha256(b),
            })
            if A is not None and np.asarray(A).ndim == 2:
                A_arr = np.asarray(A, dtype=float)
                row["lstsq_A_symmetry_max_abs"] = (
                    float(np.max(np.abs(A_arr - A_arr.T), initial=0.0))
                    if A_arr.size else 0.0
                )
                row["lstsq_A_spectrum"] = _bounded_spectrum(
                    A_arr, self.max_spectrum_values
                )
            arrays_for_capture = {
                "S": S,
                "Y": Y,
                "STS": STS,
                "lstsq_A": A,
                "lstsq_b": b,
                "symmetrize_index_i": np.asarray([i], dtype=np.int64),
            }

        row["operand_capture"] = self._capture_operands(arrays_for_capture)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.json_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), default=_jsonable) + "\n")
        self.records_written += 1
        return row


__all__ = ["SellaPassiveQNRecorder", "SellaLinalgFailureRecorder"]
