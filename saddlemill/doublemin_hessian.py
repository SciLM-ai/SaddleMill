"""Hessian bridge for SaddleMill DoubleMinimization.

Preferred path: a standalone ``method = Hessian`` job certifies saddle order and
writes a verified ``eigenmode``/``curvature``. DoubleMin auto-detects that schema
and reads it through ``read_standalone_hessian_initialization``.

Backward compatibility: ``pre_hessian_eigenmode=True`` still performs the older
inline FairChem Hessian calculation inside DoubleMin. The flag controls inline
recomputation only; ``False`` does not disable use of standalone Hessian input.
"""
from __future__ import annotations

import csv
import math
import os
import traceback
from typing import Any, Mapping

import numpy as np
from ase.constraints import FixAtoms


HESSIAN_SUMMARY_FIELDS = [
    "src_index",
    "rank",
    "parent_source_idx",
    "parent_attempt_id",
    "natoms",
    "free_dof_count",
    "hessian_seconds",
    "negative_eigenvalue_tolerance",
    "negative_mode_count",
    "is_first_order",
    "require_first_order",
    "lowest_eigenvalue",
    "lowest_eigenvalue_abs",
    "second_eigenvalue",
    "lowest_mode_norm",
    "input_mode_present",
    "input_mode_abs_cosine",
    "input_mode_one_minus_abs_cosine",
    "input_mode_angle_deg",
    "input_stored_curvature",
    "curvature_delta_hessian_minus_input",
    "curvature_abs_delta",
    "curvature_relative_abs_delta",
    "hessian_file",
    "task_name",
    "model_name_or_path",
    "fairchem_core_version",
    "status",
]


class PreHessianWrongOrderError(RuntimeError):
    """Raised when strict Hessian gating rejects a non-first-order saddle."""


STANDALONE_HESSIAN_SCHEMA_PREFIX = "standalone_hessian_"


def _info_chain(atoms):
    """Yield top-level info then nested orig_info dictionaries, nearest first."""
    current = getattr(atoms, "info", {})
    seen = set()
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.get("orig_info")


def find_standalone_hessian_info(atoms):
    """Return the nearest standalone-Hessian metadata dict, or None."""
    for info in _info_chain(atoms):
        schema = str(info.get("hessian_job_schema", ""))
        if schema.startswith(STANDALONE_HESSIAN_SCHEMA_PREFIX):
            return info
    return None


def read_standalone_hessian_initialization(
    atoms,
    *,
    require_first_order: bool = True,
):
    """Read and validate a standalone Hessian result for DoubleMin.

    This is deliberately a reader, not a Hessian calculation. The reported
    standalone ``hessian_order`` is authoritative for the gating decision.
    """
    info = find_standalone_hessian_info(atoms)
    if info is None:
        raise ValueError("Input does not contain standalone Hessian metadata.")

    try:
        order = int(info.get("hessian_order", info.get("hessian_negative_modes")))
    except (TypeError, ValueError):
        raise ValueError("Standalone Hessian input is missing a valid reported order.")
    reported_first = int(bool(info.get("hessian_is_first_order", order == 1)))
    if reported_first != int(order == 1):
        raise ValueError(
            "Standalone Hessian metadata is internally inconsistent: "
            f"order={order}, hessian_is_first_order={reported_first}."
        )
    if require_first_order and order != 1:
        raise PreHessianWrongOrderError(
            f"standalone_hessian_wrong_order: order={order}, expected=1"
        )

    if "eigenmode" not in info:
        raise ValueError("Standalone Hessian input is missing eigenmode.")
    mode = np.asarray(info["eigenmode"], dtype=float)
    expected = (len(atoms), 3)
    if mode.shape != expected or not np.all(np.isfinite(mode)):
        raise ValueError(
            f"Standalone Hessian eigenmode must have shape {expected} and be finite; "
            f"got {mode.shape}."
        )
    norm = float(np.linalg.norm(mode))
    if not math.isfinite(norm) or norm < 1.0e-14:
        raise ValueError("Standalone Hessian eigenmode is zero/non-finite.")
    mode = mode / norm

    try:
        curvature = float(info["curvature"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Standalone Hessian input is missing finite curvature.")
    if not math.isfinite(curvature):
        raise ValueError("Standalone Hessian curvature is non-finite.")

    bridge_info = {
        "doublemin_mode_source": "standalone_hessian",
        "doublemin_hessian_schema": str(info.get("hessian_job_schema", "")),
        "doublemin_hessian_order": order,
        "doublemin_hessian_is_first_order": int(order == 1),
        "doublemin_hessian_order_projection": str(
            info.get("hessian_order_projection", "")
        ),
        "doublemin_hessian_removed_translation_modes": int(
            info.get("hessian_removed_translation_modes", 0) or 0
        ),
        "doublemin_hessian_file": str(info.get("hessian_file", "")),
    }
    return mode, curvature, bridge_info


def _write_summary(path: str, row: Mapping[str, Any]) -> None:
    """Atomically replace the one-row summary for one DoubleMin input job."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = f"{path}.tmp.{os.getpid()}"
    with open(temp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HESSIAN_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in HESSIAN_SUMMARY_FIELDS})
    os.replace(temp, path)


def _free_cartesian_dofs(atoms) -> np.ndarray:
    """Return Cartesian DOF indices that are not fixed by FixAtoms.

    DoubleMin's current production constraints are FixAtoms-style. Partial or
    collective constraints are rejected rather than silently misprojected.
    """
    fixed_atoms: set[int] = set()
    for constraint in atoms.constraints:
        if not isinstance(constraint, FixAtoms):
            raise NotImplementedError(
                "pre_hessian_eigenmode currently supports no constraints or "
                "ASE FixAtoms constraints only; got "
                f"{type(constraint).__name__}."
            )
        fixed_atoms.update(int(idx) for idx in constraint.get_indices())

    free = [
        3 * atom_index + component
        for atom_index in range(len(atoms))
        if atom_index not in fixed_atoms
        for component in range(3)
    ]
    if not free:
        raise ValueError("pre_hessian_eigenmode found zero free Cartesian DOFs.")
    return np.asarray(free, dtype=int)


def analyze_hessian_mode(
    hessian: np.ndarray,
    atoms,
    negative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Legacy inline-Hessian analysis using the standalone order definition."""
    hessian = np.asarray(hessian, dtype=float)
    expected = (3 * len(atoms), 3 * len(atoms))
    if hessian.shape != expected:
        raise ValueError(f"Expected Hessian shape {expected}, got {hessian.shape}.")
    if not np.all(np.isfinite(hessian)):
        raise ValueError("DoubleMin FairChem Hessian contains non-finite values.")

    free_dofs = _free_cartesian_dofs(atoms)
    fixed_atoms: set[int] = set()
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_atoms.update(int(idx) for idx in constraint.get_indices())
    free_hessian = hessian[np.ix_(free_dofs, free_dofs)]
    from saddlemill.hessian_job import analyze_order_space
    analyzed = analyze_order_space(
        free_hessian, atoms, free_dofs, fixed_atoms, negative_tolerance
    )
    return (
        analyzed["lowest_mode"],
        analyzed["eigenvalues"],
        free_dofs,
        int(analyzed["negative_modes"]),
    )


def compare_input_mode(input_mode, hessian_mode, free_dofs) -> dict[str, Any]:
    """Return sign-invariant direction discrepancy against a stored input mode.

    Eigenvectors are projective: v and -v represent the same physical mode.
    Therefore the primary similarity metric is abs(cos(theta)).
    """
    if input_mode is None:
        return {
            "input_mode_present": 0,
            "input_mode_abs_cosine": "",
            "input_mode_one_minus_abs_cosine": "",
            "input_mode_angle_deg": "",
        }

    arr = np.asarray(input_mode, dtype=float)
    if arr.shape != np.asarray(hessian_mode).shape or not np.all(np.isfinite(arr)):
        return {
            "input_mode_present": 1,
            "input_mode_abs_cosine": "",
            "input_mode_one_minus_abs_cosine": "",
            "input_mode_angle_deg": "",
        }

    flat = arr.reshape(-1)
    free = flat[np.asarray(free_dofs, dtype=int)]
    norm = float(np.linalg.norm(free))
    if norm < 1.0e-14:
        return {
            "input_mode_present": 1,
            "input_mode_abs_cosine": "",
            "input_mode_one_minus_abs_cosine": "",
            "input_mode_angle_deg": "",
        }

    hfree = np.asarray(hessian_mode, dtype=float).reshape(-1)[np.asarray(free_dofs, dtype=int)]
    hnorm = float(np.linalg.norm(hfree))
    if hnorm < 1.0e-14:
        raise RuntimeError("Hessian lowest mode is zero in the free subspace.")

    cosine = float(np.dot(free / norm, hfree / hnorm))
    abs_cosine = float(np.clip(abs(cosine), 0.0, 1.0))
    angle = math.degrees(math.acos(abs_cosine))
    return {
        "input_mode_present": 1,
        "input_mode_abs_cosine": abs_cosine,
        "input_mode_one_minus_abs_cosine": 1.0 - abs_cosine,
        "input_mode_angle_deg": angle,
    }


def compare_input_curvature(input_curvature, hessian_lowest: float) -> dict[str, Any]:
    """Return numerical discrepancy between stored curvature and Hessian minimum."""
    if input_curvature is None or input_curvature == "":
        return {
            "input_stored_curvature": "",
            "curvature_delta_hessian_minus_input": "",
            "curvature_abs_delta": "",
            "curvature_relative_abs_delta": "",
        }
    try:
        stored = float(input_curvature)
    except (TypeError, ValueError):
        return {
            "input_stored_curvature": "",
            "curvature_delta_hessian_minus_input": "",
            "curvature_abs_delta": "",
            "curvature_relative_abs_delta": "",
        }
    if not math.isfinite(stored):
        return {
            "input_stored_curvature": "",
            "curvature_delta_hessian_minus_input": "",
            "curvature_abs_delta": "",
            "curvature_relative_abs_delta": "",
        }

    delta = float(hessian_lowest) - stored
    abs_delta = abs(delta)
    relative = abs_delta / abs(stored) if abs(stored) > 1.0e-14 else ""
    return {
        "input_stored_curvature": stored,
        "curvature_delta_hessian_minus_input": delta,
        "curvature_abs_delta": abs_delta,
        "curvature_relative_abs_delta": relative,
    }


def compute_pre_hessian_eigenmode(
    atoms,
    config_dict,
    *,
    src_index: int,
    rank: int,
    parent_source_idx: Any,
    parent_attempt_id: Any,
    input_eigenmode: Any = None,
    input_curvature: Any = None,
):
    """Compute/store one FairChem Hessian and return its lowest free mode.

    By default a non-first-order Hessian is recorded and then rejected before
    either DoubleMin side is displaced. Set pre_hessian_require_first_order=False
    explicitly to retain the diagnostic but allow the DoubleMin to continue.
    Hessian inference failure is always fatal; there is no silent fallback.
    """
    cfg = config_dict.get("ourDoubleMinimization", {}) or {}
    tolerance = float(cfg.get("hessian_negative_eigenvalue_tolerance", 1.0e-6))
    store_full = bool(cfg.get("pre_hessian_store_full", True))
    require_first_order = bool(cfg.get("pre_hessian_require_first_order", True))

    summary_dir = "DoubleMinimization_hessian_csvs"
    hessian_dir = "DoubleMinimization_hessians"
    summary_path = os.path.join(summary_dir, f"hessian_{src_index}.csv")
    hessian_path = os.path.join(hessian_dir, f"hessian_{src_index}.npz")
    os.makedirs(summary_dir, exist_ok=True)
    if store_full:
        os.makedirs(hessian_dir, exist_ok=True)

    base_row = {
        "src_index": src_index,
        "rank": rank,
        "parent_source_idx": parent_source_idx,
        "parent_attempt_id": parent_attempt_id,
        "natoms": len(atoms),
        "negative_eigenvalue_tolerance": tolerance,
        "require_first_order": int(require_first_order),
    }

    try:
        from saddlemill.sella_engine import (
            FairChemDirectHessianCallback,
            get_cached_fairchem_hessian_calculator,
        )

        hessian_calc = get_cached_fairchem_hessian_calculator(config_dict)
        callback = FairChemDirectHessianCallback(hessian_calc)
        hessian = callback(atoms)
        mode, eigenvalues, free_dofs, negative_modes = analyze_hessian_mode(
            hessian, atoms, tolerance
        )
        is_first_order = negative_modes == 1
        lowest = float(eigenvalues[0])
        second = float(eigenvalues[1]) if eigenvalues.size > 1 else float("nan")
        mode_norm = float(np.linalg.norm(mode))
        mode_comparison = compare_input_mode(input_eigenmode, mode, free_dofs)
        curvature_comparison = compare_input_curvature(input_curvature, lowest)

        stored_path = ""
        if store_full:
            payload = {
                "hessian": hessian,
                "free_dofs": free_dofs,
                "eigenvalues": eigenvalues,
                "lowest_eigenmode": mode,
                "negative_eigenvalue_tolerance": np.asarray(tolerance),
                "negative_mode_count": np.asarray(negative_modes),
            }
            if input_eigenmode is not None:
                try:
                    input_mode_array = np.asarray(input_eigenmode, dtype=float)
                    if input_mode_array.shape == mode.shape and np.all(np.isfinite(input_mode_array)):
                        payload["input_eigenmode"] = input_mode_array
                except (TypeError, ValueError):
                    pass
            stored_curvature = curvature_comparison.get("input_stored_curvature", "")
            if stored_curvature != "":
                payload["input_stored_curvature"] = np.asarray(float(stored_curvature))
            np.savez_compressed(hessian_path, **payload)
            stored_path = hessian_path

        metadata = dict(getattr(hessian_calc, "sm_hessian_metadata", {}) or {})
        if is_first_order:
            summary_status = "first_order"
        elif require_first_order:
            summary_status = "wrong_order_refused"
        else:
            summary_status = "wrong_order_allowed"

        row = {
            **base_row,
            "free_dof_count": int(len(free_dofs)),
            "hessian_seconds": float(callback.total_seconds),
            "negative_mode_count": negative_modes,
            "is_first_order": int(is_first_order),
            "lowest_eigenvalue": lowest,
            "lowest_eigenvalue_abs": abs(lowest),
            "second_eigenvalue": second,
            "lowest_mode_norm": mode_norm,
            **mode_comparison,
            **curvature_comparison,
            "hessian_file": stored_path,
            "task_name": metadata.get("task_name", ""),
            "model_name_or_path": metadata.get("model_name_or_path", ""),
            "fairchem_core_version": metadata.get("fairchem_core_version", ""),
            "status": summary_status,
        }
        _write_summary(summary_path, row)

        info = {
            "pre_hessian_eigenmode": 1,
            "pre_hessian_engine": "fairchem_direct",
            "pre_hessian_seconds": float(callback.total_seconds),
            "pre_hessian_negative_modes": negative_modes,
            "pre_hessian_is_first_order": int(is_first_order),
            "pre_hessian_require_first_order": int(require_first_order),
            "pre_hessian_negative_eigenvalue_tolerance": tolerance,
            "pre_hessian_lowest_eigenvalue": lowest,
            "pre_hessian_lowest_eigenvalue_abs": abs(lowest),
            "pre_hessian_second_eigenvalue": second,
            "pre_hessian_file": stored_path,
        }
        for key, value in mode_comparison.items():
            info[f"pre_hessian_{key}"] = value
        for key, value in curvature_comparison.items():
            info[f"pre_hessian_{key}"] = value

        if not is_first_order:
            action = "Refusing DoubleMin before displacement." if require_first_order else (
                "Continuing DoubleMin along the lowest Hessian eigenmode because "
                "pre_hessian_require_first_order=False."
            )
            print(
                "WARNING DoubleMinimization FairChem Hessian: "
                f"src_index={src_index} parent_source_idx={parent_source_idx} "
                f"has order {negative_modes}, not first order; "
                f"lowest_eigenvalue={lowest:.8g}. {action}",
                flush=True,
            )
            if require_first_order:
                raise PreHessianWrongOrderError(
                    "pre_hessian_wrong_order: "
                    f"order={negative_modes}, expected=1, "
                    f"lowest_eigenvalue={lowest:.8g}"
                )

        return mode, lowest, info
    except PreHessianWrongOrderError:
        # The complete diagnostic row/NPZ was already written above. Preserve
        # that specific status rather than overwriting it with generic error.
        raise
    except Exception as exc:
        row = {
            **base_row,
            "status": f"error: {exc}",
        }
        try:
            _write_summary(summary_path, row)
        except Exception:
            pass
        print(
            "DoubleMinimization FairChem Hessian initialization failed:\n"
            + traceback.format_exc(),
            flush=True,
        )
        raise
