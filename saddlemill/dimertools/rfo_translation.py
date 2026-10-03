"""Physical-model adapter and translation policy for SaddleMill RFO-family methods.

The adapter consumes the current physical :class:`PhysicalHessianModel` and the
same accepted raw physical center :class:`ForceObservation`.  It does not
request a new force, energy, Hessian, HVP, or minimum-mode solve.  Generic RAS
execution delegates only scalar fixed-alpha algebra to the SaddleMill-owned
``ras`` subsystem; legacy selectors remain available for campaign compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from time import perf_counter_ns
from typing import Mapping
import warnings

import numpy as np

from saddlemill.dimertools.force_history import ForceObservation
from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace, WorkPurpose
from saddlemill.dimertools.physical_hessian import PhysicalHessianModel
from saddlemill.dimertools.rfo_step import (
    PartitionedRFOResult,
    RFOInputError,
    RFOSymmetryError,
    RFORootResult,
    RestrictedPRFOResult,
    restricted_partitioned_rfo,
    rfo_order0,
    rfo_order1,
    solve_partitioned_rfo,
)
from saddlemill.dimertools.ras import (
    RASResult as GenericRASResult,
    RAS_DEFAULT_MAXITER,
    SELLA_QN_DEFAULT_TOLERANCE,
    solve_prfo_ras,
    FixedAlphaStep,
    solve_qn_mmf_ras,
    solve_ras_bisection,
    solve_rfo_ras,
)
from saddlemill.dimertools.sella_fixed_ras import (
    FixedRASResult,
    SELLA_QN_CURVATURE_REGULARIZATION,
    SELLA_RESTRICTED_MAXITER,
    solve_sella_prfo_fixed_ras,
    solve_sella_qn_mmf_fixed_ras,
)

Array = np.ndarray

RFO_TRANSLATION_INPUT_SCHEMA = "saddlemill_rfo_translation_input_v1"
RFO_TRANSLATION_RESULT_SCHEMA = "saddlemill_rfo_translation_result_v1"
PHYSICAL_HESSIAN_MODEL_ORIGIN = "physical_hessian_model"
CANONICAL_FORCE_UNITS = "eV/Angstrom"
CANONICAL_HESSIAN_UNITS = "eV/Angstrom^2"

PARTITION_UNPARTITIONED_ORDER0 = "unpartitioned_order0"
PARTITION_UNPARTITIONED_ORDER1 = "unpartitioned_order1"
PARTITION_PHYSICAL_B_EIGEN = "physical_b_eigen"
PARTITION_EXTERNAL_MODE = "external_mode"
SUPPORTED_PARTITIONS = (
    PARTITION_UNPARTITIONED_ORDER0,
    PARTITION_UNPARTITIONED_ORDER1,
    PARTITION_PHYSICAL_B_EIGEN,
    PARTITION_EXTERNAL_MODE,
)

STEP_UNRESTRICTED = "unrestricted"
STEP_NORM_CAP = "norm_cap"
STEP_RESTRICTED_PRFO = "restricted_prfo"
# Generic production RAS. Legacy Sella-prefixed selectors remain accepted for
# existing campaign compatibility but new configs should use ``ras``.
STEP_RAS = "ras"
STEP_SELLA_FIXED_RAS = "sella_fixed_ras"
STEP_SELLA_QN_FIXED_RAS = "sella_qn_fixed_ras"
SUPPORTED_STEP_CONTROLS = (
    STEP_UNRESTRICTED,
    STEP_NORM_CAP,
    STEP_RESTRICTED_PRFO,
    STEP_RAS,
    STEP_SELLA_FIXED_RAS,
    STEP_SELLA_QN_FIXED_RAS,
)

RFO_DIAGNOSTIC_SCHEMA = "saddlemill_rfo_failure_diagnostic_v1"
RFO_DIAGNOSTIC_DIR_ENV = "SADDLEMILL_RFO_DIAGNOSTIC_DIR"
RFO_DIAGNOSTIC_MAX_ENV = "SADDLEMILL_RFO_DIAGNOSTIC_MAX_RECORDS"
RFO_DIAGNOSTIC_MODE_ENV = "SADDLEMILL_RFO_DIAGNOSTIC_MODE"
_RFO_DIAGNOSTIC_WRITES = 0


def _json_ready(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _diagnostic_settings() -> tuple[Path | None, str, int]:
    raw_dir = os.environ.get(RFO_DIAGNOSTIC_DIR_ENV, "").strip()
    if not raw_dir:
        return None, "failures", 0
    mode = os.environ.get(RFO_DIAGNOSTIC_MODE_ENV, "failures").strip().lower()
    if mode not in {"failures", "failures_or_alpha_hint", "all"}:
        mode = "failures"
    raw_max = os.environ.get(RFO_DIAGNOSTIC_MAX_ENV, "4").strip()
    try:
        maximum = int(raw_max)
    except ValueError:
        maximum = 4
    return Path(raw_dir), mode, max(0, maximum)


def _diagnostic_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return token[:48] or "state"


def _solver_input_sha256(*arrays: Array) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        item = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(str(item.shape).encode("ascii"))
        digest.update(item.dtype.str.encode("ascii"))
        digest.update(item.tobytes(order="C"))
    return digest.hexdigest()


def _maybe_write_restricted_diagnostic(
    request: "RFOTranslationInput",
    *,
    B: Array,
    g: Array,
    P: Array,
    Q: Array,
    partition: str,
    physical_meta: Mapping[str, object],
    restricted: RestrictedPRFOResult,
    trial_trace: list[dict[str, object]],
    matrix_cleanup: bool,
    matrix_symmetry_residual: float,
    negative_tolerance: float,
    denominator_tolerance: float,
    root_cluster_tolerance: float,
    restricted_tolerance: float,
    restricted_alpha_min: float,
) -> None:
    """Write a bounded, opt-in, passive reproducer record.

    This function never changes the numerical result and intentionally catches
    diagnostic I/O failures so evidence collection cannot alter a production
    proposal or failure outcome.
    """

    global _RFO_DIAGNOSTIC_WRITES
    directory, mode, maximum = _diagnostic_settings()
    if directory is None or maximum <= 0:
        return
    legacy_alpha_hint = any(
        row.get("phase") == "legacy_alpha_tolerance_trigger" for row in trial_trace
    )
    if mode == "failures" and restricted.success:
        return
    if mode == "failures_or_alpha_hint" and restricted.success and not legacy_alpha_hint:
        return
    if _RFO_DIAGNOSTIC_WRITES >= maximum:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        serial = _RFO_DIAGNOSTIC_WRITES + 1
        token = _diagnostic_token(request.state_uid)
        stem = f"rfo_diag_pid{os.getpid()}_{serial:03d}_{token}"
        npz_name = f"{stem}.npz"
        json_name = f"{stem}.json"
        npz_path = directory / npz_name
        json_path = directory / json_name
        tmp_npz = directory / f".{npz_name}.tmp.{os.getpid()}"
        tmp_json = directory / f".{json_name}.tmp.{os.getpid()}"

        pB = P.T @ B @ P
        qB = Q.T @ B @ Q
        pg = P.T @ g
        qg = Q.T @ g
        B_evals = np.linalg.eigvalsh(B)
        p_evals = np.linalg.eigvalsh(pB)
        q_evals = np.linalg.eigvalsh(qB) if qB.size else np.empty(0, dtype=float)
        basis = np.column_stack((P, Q))
        orthogonality_residual = float(
            np.linalg.norm(basis.T @ basis - np.eye(B.shape[0]))
        )
        input_sha = _solver_input_sha256(B, g, P, Q)

        with open(tmp_npz, "wb") as handle:
            np.savez_compressed(
                handle,
                request_matrix=np.asarray(request.matrix, dtype=float),
                solver_matrix=np.asarray(B, dtype=float),
                raw_gradient=np.asarray(g, dtype=float),
                P=np.asarray(P, dtype=float),
                Q=np.asarray(Q, dtype=float),
                external_mode=(
                    np.empty(0, dtype=float)
                    if request.external_mode is None
                    else np.asarray(request.external_mode, dtype=float)
                ),
                B_eigenvalues=B_evals,
                P_B_eigenvalues=p_evals,
                Q_B_eigenvalues=q_evals,
                P_gradient=pg,
                Q_gradient=qg,
            )

        metadata = {
            "schema": RFO_DIAGNOSTIC_SCHEMA,
            "npz_file": npz_name,
            "solver_input_sha256": input_sha,
            "state_id": int(request.state_id),
            "state_uid": request.state_uid,
            "geometry_id": request.geometry_id,
            "coordinate_space_id": request.coordinate_space_id,
            "center_observation_id": request.center_observation_id,
            "model_origin": request.model_origin,
            "model_update_type": request.model_update_type,
            "model_age": int(request.model_age),
            "partition": partition,
            "step_control": STEP_RESTRICTED_PRFO,
            "matrix_symmetry_cleanup": bool(matrix_cleanup),
            "matrix_symmetry_residual": float(matrix_symmetry_residual),
            "partition_orthogonality_residual": orthogonality_residual,
            "physical_meta": dict(physical_meta),
            "full_negative_modes": int(np.count_nonzero(B_evals < -float(negative_tolerance))),
            "p_negative_modes": int(np.count_nonzero(p_evals < -float(negative_tolerance))),
            "q_negative_modes": int(np.count_nonzero(q_evals < -float(negative_tolerance))),
            "p_gradient_norm": float(np.linalg.norm(pg)),
            "q_gradient_norm": float(np.linalg.norm(qg)),
            "denominator_tolerance": float(denominator_tolerance),
            "root_cluster_tolerance": float(root_cluster_tolerance),
            "negative_tolerance": float(negative_tolerance),
            "restricted_tolerance": float(restricted_tolerance),
            "restricted_alpha_min": float(restricted_alpha_min),
            "restricted_result": restricted.metadata(),
            "trial_trace": trial_trace,
        }
        with open(tmp_json, "w", encoding="utf-8") as handle:
            json.dump(_json_ready(metadata), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_npz, npz_path)
        os.replace(tmp_json, json_path)
        _RFO_DIAGNOSTIC_WRITES = serial
    except Exception as exc:  # diagnostics must never change solver behavior
        warnings.warn(
            f"RFO diagnostic capture failed without affecting solver result: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


# W5-004 diagnostics are deliberately opt-in and fail-open: if the directory is
# unset or writing fails, the original numerical exception is re-raised unchanged.
# No tolerance or proposed step is modified by this instrumentation.
_W5_004_DIAGNOSTIC_ENV = "SADDLEMILL_RFO_SYMMETRY_DIAGNOSTIC_DIR"
_W5_004_DIAGNOSTIC_LIMIT_ENV = "SADDLEMILL_RFO_SYMMETRY_DIAGNOSTIC_LIMIT"
_W5_004_DEFAULT_DIAGNOSTIC_LIMIT = 2
_W5_004_DIAGNOSTIC_WRITES = 0


def _matrix_asymmetry_metrics(matrix: object) -> dict[str, object]:
    """Return guard and scale diagnostics without modifying ``matrix``.

    ``guard_residual`` intentionally reproduces the current RFO guard exactly:
    ``||B-B.T||_F / max(||B||_F, 1)``.  ``relative_residual`` is also reported
    so captures reveal when the guard is operating in its absolute (||B|| < 1)
    rather than relative (||B|| >= 1) regime.
    """

    value = np.asarray(matrix, dtype=float)
    metrics: dict[str, object] = {
        "shape": [int(x) for x in value.shape],
        "ndim": int(value.ndim),
        "size": int(value.size),
        "finite_count": int(np.count_nonzero(np.isfinite(value))),
        "nonfinite_count": int(value.size - np.count_nonzero(np.isfinite(value))),
        "all_finite": bool(np.all(np.isfinite(value))),
    }
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 1:
        metrics["square_nonempty"] = False
        return metrics
    metrics["square_nonempty"] = True
    metrics["dimension"] = int(value.shape[0])
    if not bool(metrics["all_finite"]):
        return metrics
    skew = value - value.T
    matrix_fro = float(np.linalg.norm(value))
    skew_fro = float(np.linalg.norm(skew))
    max_abs = float(np.max(np.abs(value))) if value.size else 0.0
    skew_max_abs = float(np.max(np.abs(skew))) if skew.size else 0.0
    row_sum_norm = float(np.linalg.norm(value, ord=np.inf))
    relative_floor = np.finfo(float).tiny
    metrics.update(
        {
            "matrix_frobenius_norm": matrix_fro,
            "matrix_max_abs": max_abs,
            "matrix_inf_norm": row_sum_norm,
            "skew_frobenius_norm": skew_fro,
            "skew_max_abs": skew_max_abs,
            "guard_denominator": max(matrix_fro, 1.0),
            "guard_residual": skew_fro / max(matrix_fro, 1.0),
            "relative_residual": skew_fro / max(matrix_fro, relative_floor),
            "guard_scaling_regime": "relative" if matrix_fro >= 1.0 else "absolute_floor_1",
            "machine_epsilon": float(np.finfo(float).eps),
            "dimension_times_epsilon": float(value.shape[0] * np.finfo(float).eps),
        }
    )
    return metrics


def _w5_004_capture_partition_symmetry_failure(
    request: "RFOTranslationInput",
    *,
    model_matrix: Array,
    gradient: Array,
    P: Array,
    Q: Array,
    partition: str,
    step_control: str,
    target_order: int,
    matrix_symmetry_tolerance: float,
    error: RFOSymmetryError,
) -> None:
    """Capture exact RFO symmetry-failure inputs when explicitly enabled.

    This is evidence collection only.  It never changes the matrix, tolerance,
    branch selection, or raised exception.  Any diagnostic I/O failure is ignored
    so the scientific execution path preserves its pre-patch behavior.
    """

    directory = os.environ.get(_W5_004_DIAGNOSTIC_ENV, "").strip()
    if not directory:
        return
    global _W5_004_DIAGNOSTIC_WRITES
    try:
        limit = int(os.environ.get(_W5_004_DIAGNOSTIC_LIMIT_ENV, _W5_004_DEFAULT_DIAGNOSTIC_LIMIT))
    except (TypeError, ValueError):
        limit = _W5_004_DEFAULT_DIAGNOSTIC_LIMIT
    limit = max(0, limit)
    if _W5_004_DIAGNOSTIC_WRITES >= limit:
        return
    capture_index = _W5_004_DIAGNOSTIC_WRITES
    _W5_004_DIAGNOSTIC_WRITES += 1

    try:
        model = np.asarray(model_matrix, dtype=float)
        grad = np.asarray(gradient, dtype=float).reshape(-1)
        p_basis = np.asarray(P, dtype=float)
        q_basis = np.asarray(Q, dtype=float)
        p_matrix = p_basis.T @ model @ p_basis
        q_matrix = q_basis.T @ model @ q_basis
        p_gradient = p_basis.T @ grad
        q_gradient = q_basis.T @ grad
        guard_matrix = np.asarray(error.matrix, dtype=float)
        matrices = {
            "guard_matrix_exact": guard_matrix,
            "request_matrix": np.asarray(request.matrix, dtype=float),
            "model_matrix_after_input_guard": model,
            "p_basis": p_basis,
            "q_basis": q_basis,
            "raw_gradient": grad,
            "p_branch_matrix": p_matrix,
            "q_branch_matrix": q_matrix,
            "p_branch_gradient": p_gradient,
            "q_branch_gradient": q_gradient,
        }
        metrics = {
            "guard_matrix_exact": _matrix_asymmetry_metrics(guard_matrix),
            "request_matrix": _matrix_asymmetry_metrics(matrices["request_matrix"]),
            "model_matrix_after_input_guard": _matrix_asymmetry_metrics(model),
            "p_branch_matrix": _matrix_asymmetry_metrics(p_matrix),
            "q_branch_matrix": _matrix_asymmetry_metrics(q_matrix),
        }
        reported_residual = float(error.residual)
        reported_tolerance = float(error.tolerance)
        formatting_window = max(1.0e-18, abs(reported_residual) * 6.0e-7)
        matching = []
        for name, candidate in matrices.items():
            if name == "guard_matrix_exact" or np.asarray(candidate).ndim != 2:
                continue
            if np.asarray(candidate).shape == guard_matrix.shape and np.array_equal(candidate, guard_matrix):
                matching.append(name)
        if not matching:
            for name, item in metrics.items():
                if name == "guard_matrix_exact":
                    continue
                residual = item.get("guard_residual")
                if isinstance(residual, float) and abs(residual - reported_residual) <= formatting_window:
                    matching.append(name)

        record = {
            "schema": "w5_004_rfo_symmetry_capture_v1",
            "diagnostic_only": True,
            "original_exception": str(error),
            "reported_guard_residual": reported_residual,
            "reported_guard_tolerance": reported_tolerance,
            "matching_guard_stage_candidates": matching,
            "matrix_symmetry_tolerance": float(matrix_symmetry_tolerance),
            "partition": str(partition),
            "step_control": str(step_control),
            "target_order": int(target_order),
            "model_provenance": {
                "model_origin": request.model_origin,
                "model_update_type": request.model_update_type,
                "model_age": int(request.model_age),
                "state_id": int(request.state_id),
                "state_uid": request.state_uid,
                "geometry_id": request.geometry_id,
                "coordinate_space_id": request.coordinate_space_id,
                "center_observation_id": request.center_observation_id,
                "force_units": request.force_units,
                "hessian_units": request.hessian_units,
            },
            "matrix_metrics": metrics,
            "array_file": "",
        }
        outdir = Path(directory)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"w5_004_rfo_symmetry_pid{os.getpid()}_state{int(request.state_id)}_"
            f"capture{capture_index}"
        )
        npz_path = outdir / f"{stem}.npz"
        json_path = outdir / f"{stem}.json"
        np.savez_compressed(npz_path, **matrices)
        record["array_file"] = npz_path.name
        temp_json = json_path.with_suffix(json_path.suffix + ".tmp")
        temp_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_json, json_path)
    except Exception:
        # Evidence collection must never replace or mask the original RFO error.
        return


class RFOTranslationInputError(RFOInputError):
    """The C2 physical-model/raw-gradient identity contract was violated."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = str(code)
        super().__init__(message or self.code)


@dataclass(frozen=True)
class RFOTranslationInput:
    """Frozen matrix-plus-raw-gradient input at one accepted C2 center."""

    matrix: Array | None
    raw_gradient: Array
    coordinate_space: ActiveCoordinateSpace
    state_id: int
    state_uid: str
    geometry_id: str
    coordinate_space_id: str
    center_observation_id: str
    model_origin: str
    model_update_type: str
    model_age: int
    force_units: str
    hessian_units: str
    representation: str = "online_dense"
    spectral_baseline: float | None = None
    spectral_eigenvalues: Array | None = None
    spectral_eigenvectors: Array | None = None
    external_mode: Array | None = None
    t17_added_pes_calls: int = 0
    schema: str = RFO_TRANSLATION_INPUT_SCHEMA

    def __post_init__(self) -> None:
        gradient = np.asarray(self.raw_gradient, dtype=float).reshape(-1)
        if gradient.size < 1 or not np.all(np.isfinite(gradient)):
            raise RFOTranslationInputError("nonfinite_model_input")
        gradient = np.array(gradient, copy=True)
        gradient.setflags(write=False)
        object.__setattr__(self, "raw_gradient", gradient)
        if self.matrix is not None:
            matrix = np.asarray(self.matrix, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
                raise RFOTranslationInputError("invalid_matrix_shape")
            if gradient.size != matrix.shape[0]:
                raise RFOTranslationInputError("gradient_matrix_dimension_mismatch")
            if not np.all(np.isfinite(matrix)):
                raise RFOTranslationInputError("nonfinite_model_input")
            matrix = np.array(matrix, copy=True)
            matrix.setflags(write=False)
            object.__setattr__(self, "matrix", matrix)
        else:
            beta = self.spectral_baseline
            evals = np.asarray(self.spectral_eigenvalues if self.spectral_eigenvalues is not None else [], dtype=float).reshape(-1)
            evecs = np.asarray(self.spectral_eigenvectors if self.spectral_eigenvectors is not None else np.empty((gradient.size, 0)), dtype=float)
            if beta is None or not np.isfinite(float(beta)):
                raise RFOTranslationInputError("compact_model_missing_spectral_baseline")
            if evecs.shape != (gradient.size, evals.size):
                raise RFOTranslationInputError("compact_model_spectral_shape_mismatch")
            if not np.all(np.isfinite(evals)) or not np.all(np.isfinite(evecs)):
                raise RFOTranslationInputError("nonfinite_compact_spectral_model")
            if evals.size and not np.allclose(evecs.T @ evecs, np.eye(evals.size), rtol=0.0, atol=2.0e-10):
                raise RFOTranslationInputError("compact_model_nonorthonormal_spectral_basis")
            evals = np.array(evals, copy=True); evals.setflags(write=False)
            evecs = np.array(evecs, copy=True); evecs.setflags(write=False)
            object.__setattr__(self, "spectral_baseline", float(beta))
            object.__setattr__(self, "spectral_eigenvalues", evals)
            object.__setattr__(self, "spectral_eigenvectors", evecs)
        if self.external_mode is not None:
            mode = np.asarray(self.external_mode, dtype=float).reshape(-1).copy()
            if mode.size != gradient.size or not np.all(np.isfinite(mode)):
                raise RFOTranslationInputError("external_mode_dimension_mismatch")
            norm = float(np.linalg.norm(mode))
            if norm <= 1.0e-14:
                raise RFOTranslationInputError("external_mode_zero")
            mode /= norm
            mode.setflags(write=False)
            object.__setattr__(self, "external_mode", mode)
        if self.coordinate_space_id != self.coordinate_space.identity:
            raise RFOTranslationInputError("coordinate_space_identity_mismatch")
        if int(self.model_age) < 0:
            raise RFOTranslationInputError("negative_model_age")
        if int(self.t17_added_pes_calls) != 0:
            raise RFOTranslationInputError("t17_must_not_add_pes_calls")


@dataclass(frozen=True)
class RFOTranslationResult:
    """One native RFO/P-RFO translation proposal plus provenance."""

    success: bool
    status: str
    failure_reason: str
    algorithm: str
    partition: str
    step_control: str
    cartesian_step: Array | None
    reduced_step: Array | None
    raw_step_norm: float | None
    achieved_norm: float | None
    alpha: float | None
    requested_radius: float | None
    restricted_boundary_active: bool
    norm_cap: float | None
    norm_cap_applied: bool
    final_safety_cap: float | None
    final_safety_cap_applied: bool
    selected_physical_root_index: int | None
    selected_physical_root_eigenvalue: float | None
    physical_negative_mode_count: int
    physical_root_policy: str
    physical_root_degenerate: bool
    coupling_norm: float | None
    coupling_relative: float | None
    coupling_warning: bool
    coupling_tolerance: float | None
    matrix_symmetry_cleanup: bool
    matrix_symmetry_residual: float
    model_origin: str
    model_update_type: str
    model_age: int
    state_uid: str
    geometry_id: str
    coordinate_space_id: str
    center_observation_id: str
    force_units: str
    hessian_units: str
    p_root_metadata: Mapping[str, object] | None
    q_root_metadata: Mapping[str, object] | None
    restricted_metadata: Mapping[str, object] | None
    eigensolve_time_ns: int
    root_solve_time_ns: int
    root_trials: int
    t17_added_pes_calls: int = 0
    schema: str = RFO_TRANSLATION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        for field_name in ("cartesian_step", "reduced_step"):
            value = getattr(self, field_name)
            if value is not None:
                arr = np.asarray(value, dtype=float).copy()
                arr.setflags(write=False)
                object.__setattr__(self, field_name, arr)

    def metadata(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"cartesian_step", "reduced_step"}
        }


class _CoordinateAdapter:
    """Exact reduction/lift convention used by the sealed physical-Hessian C2 model."""

    def __init__(self, space: ActiveCoordinateSpace) -> None:
        mask = space.active_dof_mask.reshape(-1)
        self.space = space
        self.active_indices = np.flatnonzero(mask)
        if self.active_indices.size == 0:
            raise RFOTranslationInputError("zero_active_coordinates")
        if space.null_basis:
            null = np.column_stack(
                [base.reshape(-1)[self.active_indices] for base in space.null_basis]
            )
            q, _ = np.linalg.qr(null, mode="complete")
            self.basis = np.asarray(q[:, len(space.null_basis) :], dtype=float)
        else:
            self.basis = np.eye(self.active_indices.size, dtype=float)
        if self.basis.shape[1] < 1:
            raise RFOTranslationInputError("active_space_exhausted_by_null_basis")
        self.full_dimension = int(mask.size)
        self.shape = space.active_dof_mask.shape

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[1])

    def reduce(self, value: object) -> Array:
        projected = self.space.project(value).reshape(-1)
        active = projected[self.active_indices]
        return np.asarray(self.basis.T @ active, dtype=float)

    def expand(self, reduced: object) -> Array:
        vector = np.asarray(reduced, dtype=float).reshape(-1)
        if vector.size != self.dimension:
            raise RFOTranslationInputError("reduced_step_dimension_mismatch")
        active = self.basis @ vector
        full = np.zeros(self.full_dimension, dtype=float)
        full[self.active_indices] = active
        # Re-project to make fixed/null coordinates exactly obey the sealed space.
        return self.space.project(full.reshape(self.shape))


def _canonicalize_sign(vector: Array) -> Array:
    out = np.asarray(vector, dtype=float).reshape(-1).copy()
    nonzero = np.flatnonzero(np.abs(out) > 1.0e-14)
    if nonzero.size and out[int(nonzero[0])] < 0.0:
        out *= -1.0
    return out


def _force_units(observation: ForceObservation) -> str:
    metadata = dict(observation.metadata or {})
    explicit = metadata.get("force_units", metadata.get("units"))
    if explicit is None:
        return CANONICAL_FORCE_UNITS
    token = str(explicit).strip()
    aliases = {
        "ev/angstrom": CANONICAL_FORCE_UNITS,
        "ev/a": CANONICAL_FORCE_UNITS,
        "ev/å": CANONICAL_FORCE_UNITS,
    }
    canonical = aliases.get(token.lower(), token)
    if canonical != CANONICAL_FORCE_UNITS:
        raise RFOTranslationInputError(
            "force_units_mismatch", f"expected {CANONICAL_FORCE_UNITS}, got {token!r}"
        )
    return canonical


def _reject_transformed_force(
    observation: ForceObservation,
    *,
    gradient_kind: str,
    parallel_force_damping: str,
    force_transform: str,
) -> None:
    if str(gradient_kind).strip().lower() != "raw_physical":
        raise RFOTranslationInputError("effective_or_reflected_gradient_rejected")
    if str(parallel_force_damping).strip().lower() != "none":
        raise RFOTranslationInputError("parallel_force_damping_incompatible")
    if str(force_transform).strip().lower() not in {"none", "raw_physical"}:
        raise RFOTranslationInputError("transformed_gradient_rejected")
    metadata = {str(k).lower(): v for k, v in dict(observation.metadata or {}).items()}
    for key in (
        "damping_applied",
        "reflected_force",
        "effective_force",
        "lambda_scaled",
        "mmf_force",
    ):
        if bool(metadata.get(key, False)):
            raise RFOTranslationInputError("transformed_gradient_metadata_rejected")
    kind = str(metadata.get("force_kind", metadata.get("gradient_kind", "raw_physical"))).lower()
    if kind not in {"", "raw", "raw_physical", "physical", "physical_raw"}:
        raise RFOTranslationInputError("transformed_gradient_metadata_rejected")


def build_c2_rfo_input(
    model: PhysicalHessianModel,
    center_observation: ForceObservation,
    *,
    external_mode: object | None = None,
    expected_state_uid: str | None = None,
    expected_geometry_id: str | None = None,
    expected_coordinate_space_id: str | None = None,
    expected_model_origin: str = PHYSICAL_HESSIAN_MODEL_ORIGIN,
    expected_update_type: str | None = None,
    expected_model_age: int | None = None,
    gradient_kind: str = "raw_physical",
    parallel_force_damping: str = "none",
    force_transform: str = "none",
) -> RFOTranslationInput:
    """Adapt the exact sealed C2 physical-Hessian model and same-center raw observation."""

    if not isinstance(model, PhysicalHessianModel):
        raise RFOTranslationInputError("physical_hessian_model_required")
    if not isinstance(center_observation, ForceObservation):
        raise RFOTranslationInputError("force_observation_required")
    obs = center_observation
    if obs.role != "center" or obs.source != "center":
        raise RFOTranslationInputError("raw_center_observation_required")
    if not obs.is_physical or obs.purpose != WorkPurpose.ALGORITHM.value:
        raise RFOTranslationInputError("algorithm_physical_center_required")
    _reject_transformed_force(
        obs,
        gradient_kind=gradient_kind,
        parallel_force_damping=parallel_force_damping,
        force_transform=force_transform,
    )
    state = model.state_dict()
    current_state_uid = str(state.get("current_state_uid", ""))
    current_geometry_id = str(state.get("current_geometry_id", ""))
    if not current_state_uid or not current_geometry_id:
        raise RFOTranslationInputError("physical_hessian_has_no_current_center")
    obs_state_uid = f"state{int(obs.state_id)}:{obs.geometry_id}"
    if current_state_uid != obs_state_uid:
        raise RFOTranslationInputError("state_uid_mismatch")
    if current_geometry_id != obs.geometry_id:
        raise RFOTranslationInputError("geometry_id_mismatch")
    space = model.coordinate_space
    if obs.positions.shape != space.active_dof_mask.shape:
        raise RFOTranslationInputError("cartesian_shape_mismatch")
    if obs.active_dof_mask is not None and not np.array_equal(
        obs.active_dof_mask, space.active_dof_mask
    ):
        raise RFOTranslationInputError("active_dof_mask_mismatch")
    if obs.coordinate_convention not in {"", "unspecified", space.convention}:
        raise RFOTranslationInputError("coordinate_convention_mismatch")
    if expected_state_uid is not None and str(expected_state_uid) != current_state_uid:
        raise RFOTranslationInputError("expected_state_uid_mismatch")
    if expected_geometry_id is not None and str(expected_geometry_id) != current_geometry_id:
        raise RFOTranslationInputError("expected_geometry_id_mismatch")
    if expected_coordinate_space_id is not None and str(expected_coordinate_space_id) != space.identity:
        raise RFOTranslationInputError("expected_coordinate_space_id_mismatch")
    if str(expected_model_origin) != PHYSICAL_HESSIAN_MODEL_ORIGIN:
        raise RFOTranslationInputError("model_origin_mismatch")
    if expected_update_type is not None and str(expected_update_type) != model.update_type:
        raise RFOTranslationInputError("model_update_type_mismatch")
    if expected_model_age is not None and int(expected_model_age) != int(model.model_age):
        raise RFOTranslationInputError("stale_model_age")

    adapter = _CoordinateAdapter(space)
    if model.dimension != adapter.dimension:
        raise RFOTranslationInputError("t03_active_basis_dimension_mismatch")
    gradient = adapter.reduce(-obs.forces)  # canonical g=-F
    mode = None
    if external_mode is not None:
        mode = adapter.reduce(external_mode)
        norm = float(np.linalg.norm(mode))
        if not np.isfinite(norm) or norm <= 1.0e-14:
            raise RFOTranslationInputError("external_mode_zero_after_active_projection")
        mode = mode / norm

    if model.has_dense_matrix:
        matrix = model.matrix
        spectral_baseline = None
        spectral_eigenvalues = None
        spectral_eigenvectors = None
    else:
        spectral_baseline, spectral_eigenvalues, spectral_eigenvectors = model.compact_eigensystem()
        matrix = None

    return RFOTranslationInput(
        matrix=matrix,
        raw_gradient=gradient,
        coordinate_space=space,
        state_id=int(obs.state_id),
        state_uid=current_state_uid,
        geometry_id=current_geometry_id,
        coordinate_space_id=space.identity,
        center_observation_id=obs.observation_id,
        model_origin=PHYSICAL_HESSIAN_MODEL_ORIGIN,
        model_update_type=model.update_type,
        model_age=int(model.model_age),
        force_units=_force_units(obs),
        hessian_units=CANONICAL_HESSIAN_UNITS,
        representation=model.representation,
        spectral_baseline=spectral_baseline,
        spectral_eigenvalues=spectral_eigenvalues,
        spectral_eigenvectors=spectral_eigenvectors,
        external_mode=mode,
        t17_added_pes_calls=0,
    )


def _matrix_symmetry(B: Array, tolerance: float) -> tuple[Array, bool, float]:
    skew = B - B.T
    residual = float(np.linalg.norm(skew) / max(float(np.linalg.norm(B)), 1.0))
    if residual > tolerance:
        raise RFOTranslationInputError("physical_b_not_symmetric")
    cleanup = bool(np.any(skew != 0.0))
    return 0.5 * (B + B.T), cleanup, residual


def _external_partition(mode: Array) -> tuple[Array, Array]:
    v = _canonicalize_sign(mode)
    v /= np.linalg.norm(v)
    if v.size == 1:
        return v[:, None], np.zeros((1, 0), dtype=float)
    # SVD of the row vector gives a deterministic orthonormal null-space basis.
    _, _, vh = np.linalg.svd(v.reshape(1, -1), full_matrices=True)
    q = np.asarray(vh[1:, :].T, dtype=float)
    for col in range(q.shape[1]):
        q[:, col] = _canonicalize_sign(q[:, col])
    return v[:, None], q


def _physical_b_partition(
    B: Array,
    *,
    target_order: int,
    root_policy: str,
    previous_mode: Array | None,
    negative_mode_policy: str,
    negative_tolerance: float,
    degeneracy_tolerance: float,
) -> tuple[Array, Array, dict[str, object]]:
    if int(target_order) != 1:
        raise RFOTranslationInputError("physical_b_partition_currently_supports_target_order_1")
    evals, evecs = np.linalg.eigh(B)
    for col in range(evecs.shape[1]):
        evecs[:, col] = _canonicalize_sign(evecs[:, col])
    policy = str(root_policy).strip().lower()
    if policy == "lowest":
        selected = 0
        overlap = None
    elif policy == "homed":
        if previous_mode is None:
            raise RFOTranslationInputError("homed_physical_root_requires_previous_mode")
        previous = np.asarray(previous_mode, dtype=float).reshape(-1)
        if previous.size != B.shape[0] or not np.all(np.isfinite(previous)):
            raise RFOTranslationInputError("physical_homing_mode_dimension_mismatch")
        norm = float(np.linalg.norm(previous))
        if norm <= 1.0e-14:
            raise RFOTranslationInputError("physical_homing_mode_zero")
        previous /= norm
        overlaps = np.abs(evecs.T @ previous)
        maximum = float(np.max(overlaps))
        candidates = np.flatnonzero(overlaps >= maximum - 1.0e-12)
        selected = min((int(i) for i in candidates), key=lambda i: (float(evals[i]), i))
        overlap = float(overlaps[selected])
    else:
        raise RFOTranslationInputError("unsupported_physical_root_policy")

    negative_count = int(np.count_nonzero(evals < -float(negative_tolerance)))
    neg_policy = str(negative_mode_policy).strip().lower()
    if neg_policy == "allow":
        pass
    elif neg_policy == "require_at_least_target":
        if negative_count < int(target_order):
            raise RFOTranslationInputError("physical_b_required_negative_mode_missing")
    elif neg_policy == "require_selected_negative":
        if float(evals[selected]) >= -float(negative_tolerance):
            raise RFOTranslationInputError("selected_physical_b_root_not_negative")
    else:
        raise RFOTranslationInputError("unsupported_negative_mode_policy")

    if evals.size > 1:
        gap = float(np.min(np.abs(np.delete(evals, selected) - evals[selected])))
    else:
        gap = None
    scale = max(1.0, abs(float(evals[selected])), float(np.max(np.abs(evals))))
    degenerate = bool(gap is not None and gap <= float(degeneracy_tolerance) * scale)
    P = evecs[:, [selected]]
    Q = np.delete(evecs, selected, axis=1)
    return P, Q, {
        "selected_index": int(selected),
        "selected_eigenvalue": float(evals[selected]),
        "selected_overlap": overlap,
        "negative_mode_count": negative_count,
        "root_gap": gap,
        "degenerate": degenerate,
        "root_policy": policy,
        "negative_mode_policy": neg_policy,
    }


def _branch_metadata(result: RFORootResult | None) -> Mapping[str, object] | None:
    return None if result is None else result.metadata()


def _failed_result(
    request: RFOTranslationInput,
    *,
    algorithm: str,
    partition: str,
    step_control: str,
    failure_reason: str,
    matrix_cleanup: bool,
    matrix_symmetry_residual: float,
    physical_meta: Mapping[str, object],
    coupling_tolerance: float | None,
    partition_result: PartitionedRFOResult | None = None,
    restricted: RestrictedPRFOResult | None = None,
) -> RFOTranslationResult:
    p = None if partition_result is None else partition_result.p_result
    q = None if partition_result is None else partition_result.q_result
    return RFOTranslationResult(
        success=False,
        status="failed",
        failure_reason=failure_reason,
        algorithm=algorithm,
        partition=partition,
        step_control=step_control,
        cartesian_step=None,
        reduced_step=None,
        raw_step_norm=None,
        achieved_norm=None,
        alpha=None if partition_result is None else partition_result.alpha,
        requested_radius=None if restricted is None else restricted.requested_radius,
        restricted_boundary_active=False if restricted is None else restricted.boundary_active,
        norm_cap=None,
        norm_cap_applied=False,
        final_safety_cap=None,
        final_safety_cap_applied=False,
        selected_physical_root_index=physical_meta.get("selected_index"),
        selected_physical_root_eigenvalue=physical_meta.get("selected_eigenvalue"),
        physical_negative_mode_count=int(physical_meta.get("negative_mode_count", 0)),
        physical_root_policy=str(physical_meta.get("root_policy", "not_applicable")),
        physical_root_degenerate=bool(physical_meta.get("degenerate", False)),
        coupling_norm=None if partition_result is None else partition_result.coupling_norm,
        coupling_relative=None if partition_result is None else partition_result.coupling_relative,
        coupling_warning=False if partition_result is None or coupling_tolerance is None else partition_result.coupling_relative > coupling_tolerance,
        coupling_tolerance=coupling_tolerance,
        matrix_symmetry_cleanup=matrix_cleanup,
        matrix_symmetry_residual=matrix_symmetry_residual,
        model_origin=request.model_origin,
        model_update_type=request.model_update_type,
        model_age=request.model_age,
        state_uid=request.state_uid,
        geometry_id=request.geometry_id,
        coordinate_space_id=request.coordinate_space_id,
        center_observation_id=request.center_observation_id,
        force_units=request.force_units,
        hessian_units=request.hessian_units,
        p_root_metadata=_branch_metadata(p),
        q_root_metadata=_branch_metadata(q),
        restricted_metadata=None if restricted is None else restricted.metadata(),
        eigensolve_time_ns=0 if partition_result is None else partition_result.eigensolve_time_ns,
        root_solve_time_ns=0 if restricted is None else restricted.root_solve_time_ns,
        root_trials=0 if restricted is None else restricted.root_trials,
        t17_added_pes_calls=0,
    )


def _compute_compact_qn_mmf_translation(
    request: RFOTranslationInput,
    *,
    partition: str,
    step_control: str,
    target_order: int,
    physical_root_policy: str,
    negative_mode_policy: str,
    negative_tolerance: float,
    physical_degeneracy_tolerance: float,
    norm_cap: float | None,
    trust_radius: float | None,
    ras_root_solver: str,
    ras_tolerance: float | None,
    ras_max_iterations: int,
    translation_step_method: str | None,
    final_safety_cap: float | None,
) -> RFOTranslationResult:
    """QN/MMF translation using only compact spectral capability data."""

    if request.matrix is not None:
        raise RFOTranslationInputError("compact_qn_path_requires_operator_payload")
    if partition != PARTITION_PHYSICAL_B_EIGEN:
        raise RFOTranslationInputError("compact_physical_b_requires_physical_b_eigen_partition")
    if str(translation_step_method or "qn_mmf").strip().lower() != "qn_mmf":
        raise RFOTranslationInputError("compact_physical_b_supports_qn_mmf_only")
    if step_control != STEP_RAS:
        raise RFOTranslationInputError("compact_physical_b_qn_mmf_requires_step_control_ras")
    if int(target_order) != 1:
        raise RFOTranslationInputError("compact_physical_b_supports_target_order_1_only")
    if str(physical_root_policy).strip().lower() != "lowest":
        raise RFOTranslationInputError("compact_physical_b_requires_lowest_root_policy")
    neg_policy = str(negative_mode_policy).strip().lower()
    if neg_policy not in {"allow", "require_at_least_target", "require_selected_negative"}:
        raise RFOTranslationInputError("unsupported_negative_mode_policy")
    beta = float(request.spectral_baseline)
    evals = np.asarray(request.spectral_eigenvalues, dtype=float).reshape(-1)
    evecs = np.asarray(request.spectral_eigenvectors, dtype=float)
    g = np.asarray(request.raw_gradient, dtype=float)
    complement_dim = g.size - evals.size
    neg_count = int(np.count_nonzero(evals < -float(negative_tolerance)))
    if beta < -float(negative_tolerance):
        neg_count += int(complement_dim)
    if neg_policy == "require_at_least_target" and neg_count < 1:
        raise RFOTranslationInputError("physical_b_required_negative_mode_missing")
    if complement_dim and (evals.size == 0 or float(evals[0]) >= beta - 1.0e-12):
        raise RFOTranslationInputError("compact_qn_selected_lowest_root_not_in_learned_subspace")
    if neg_policy == "require_selected_negative" and (evals.size == 0 or float(evals[0]) >= -float(negative_tolerance)):
        raise RFOTranslationInputError("selected_physical_b_root_not_negative")
    if evals.size == 0:
        eigengap = None
        degenerate = False
    else:
        competing = list(float(v) for v in evals[1:])
        if complement_dim:
            competing.append(beta)
        eigengap = None if not competing else float(min(abs(v - float(evals[0])) for v in competing))
        scale = max(1.0, abs(float(evals[0])), abs(beta), *(abs(v) for v in competing))
        degenerate = bool(eigengap is not None and eigengap <= float(physical_degeneracy_tolerance) * scale)
    projected = evecs.T @ g if evals.size else np.empty(0, dtype=float)
    g_perp = g - (evecs @ projected if evals.size else 0.0)

    def fixed(alpha: float) -> FixedAlphaStep:
        a = float(alpha)
        signs = np.ones(evals.size, dtype=float)
        signed = np.abs(evals)
        if evals.size:
            signs[0] = -1.0
            signed = signed.copy()
            signed[0] *= -1.0
        denom = signed + a * signs
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sproj = projected / denom if evals.size else projected
            step = -(evecs @ sproj) if evals.size else np.zeros_like(g)
            ds = evecs @ (sproj / denom) if evals.size else np.zeros_like(g)
            if complement_dim:
                dperp = abs(beta) + a
                step = step - g_perp / dperp
                ds = ds + g_perp / (dperp * dperp)
        if not np.all(np.isfinite(step)) or not np.all(np.isfinite(ds)):
            raise RFOTranslationInputError("nonfinite_compact_qn_mmf_step")
        return FixedAlphaStep(
            step=step,
            ds_dalpha=ds,
            alpha=a,
            algorithm="qn_mmf",
            metadata={
                "eigensystem_source": "compact_physical_hessian_capability",
                "representation": request.representation,
                "baseline_eigenvalue": beta,
                "learned_rank": int(evals.size),
                "complement_dimension": int(complement_dim),
                "eigenvalues": [float(x) for x in evals],
                "selected_unstable_index": 0,
                "selected_unstable_eigenvalue": float(evals[0]) if evals.size else None,
                "order": 1,
            },
        )

    adapter = _CoordinateAdapter(request.coordinate_space)
    lift = lambda reduced: adapter.expand(reduced).reshape(-1)
    fixed_ras = None
    if step_control == STEP_RAS:
        if trust_radius is None:
            raise RFOTranslationInputError("ras_requires_trust_radius")
        if str(ras_root_solver).strip().lower() != "bisection":
            raise RFOTranslationInputError("unsupported_ras_root_solver")
        fixed_ras = solve_ras_bisection(
            fixed,
            lift,
            radius=float(trust_radius),
            algorithm="qn_mmf",
            alpha0=0.0,
            alphamin=0.0,
            alphamax=np.inf,
            slope=-1.0,
            tolerance=(SELLA_QN_DEFAULT_TOLERANCE if ras_tolerance is None else float(ras_tolerance)),
            max_iterations=int(ras_max_iterations),
        )
        if not fixed_ras.success or fixed_ras.step is None:
            return RFOTranslationResult(
                success=False, status=fixed_ras.status, failure_reason=fixed_ras.failure_reason,
                algorithm="qn_mmf", partition=partition, step_control=step_control,
                cartesian_step=None, reduced_step=None, raw_step_norm=None, achieved_norm=None,
                alpha=fixed_ras.alpha, requested_radius=fixed_ras.requested_radius,
                restricted_boundary_active=fixed_ras.boundary_active, norm_cap=None,
                norm_cap_applied=False, final_safety_cap=None, final_safety_cap_applied=False,
                selected_physical_root_index=0, selected_physical_root_eigenvalue=(float(evals[0]) if evals.size else None),
                physical_negative_mode_count=neg_count, physical_root_policy="lowest",
                physical_root_degenerate=degenerate, coupling_norm=None, coupling_relative=None,
                coupling_warning=False, coupling_tolerance=None, matrix_symmetry_cleanup=False,
                matrix_symmetry_residual=0.0, model_origin=request.model_origin,
                model_update_type=request.model_update_type, model_age=request.model_age,
                state_uid=request.state_uid, geometry_id=request.geometry_id,
                coordinate_space_id=request.coordinate_space_id, center_observation_id=request.center_observation_id,
                force_units=request.force_units, hessian_units=request.hessian_units,
                p_root_metadata=None, q_root_metadata=None, restricted_metadata=fixed_ras.metadata(),
                eigensolve_time_ns=0, root_solve_time_ns=fixed_ras.solve_time_ns,
                root_trials=fixed_ras.evaluations, t17_added_pes_calls=0,
            )
        raw_step = np.asarray(fixed_ras.step, dtype=float).copy()
        alpha = float(fixed_ras.alpha if fixed_ras.alpha is not None else 0.0)
        requested_radius = float(fixed_ras.requested_radius)
        boundary_active = bool(fixed_ras.boundary_active)
        root_solve_time = int(fixed_ras.solve_time_ns)
        root_trials = int(fixed_ras.evaluations)
        metadata = dict(fixed_ras.last_metadata or {})
    raw_norm = float(np.linalg.norm(raw_step))
    final_step = raw_step.copy()
    cap_value = None
    cap_applied = False
    if step_control == STEP_NORM_CAP:
        if norm_cap is None or not np.isfinite(float(norm_cap)) or float(norm_cap) <= 0.0:
            raise RFOTranslationInputError("norm_cap_must_be_finite_positive")
        cap_value = float(norm_cap)
        if raw_norm > cap_value and raw_norm > 0.0:
            final_step *= cap_value / raw_norm
            cap_applied = True
    elif norm_cap is not None:
        raise RFOTranslationInputError("norm_cap_value_requires_norm_cap_policy")
    safety_value = None
    safety_applied = False
    if final_safety_cap is not None:
        safety_value = float(final_safety_cap)
        current = float(np.linalg.norm(final_step))
        if not np.isfinite(safety_value) or safety_value <= 0.0:
            raise RFOTranslationInputError("final_safety_cap_must_be_finite_positive")
        if current > safety_value and current > 0.0:
            final_step *= safety_value / current
            safety_applied = True
    cartesian = adapter.expand(final_step)
    status = (
        fixed_ras.status if fixed_ras is not None
        else ("posthoc_norm_capped" if cap_applied else "unrestricted")
    )
    return RFOTranslationResult(
        success=True, status=status, failure_reason="", algorithm="qn_mmf",
        partition=partition, step_control=step_control, cartesian_step=cartesian,
        reduced_step=final_step, raw_step_norm=raw_norm, achieved_norm=float(np.linalg.norm(final_step)),
        alpha=alpha, requested_radius=requested_radius, restricted_boundary_active=boundary_active,
        norm_cap=cap_value, norm_cap_applied=cap_applied, final_safety_cap=safety_value,
        final_safety_cap_applied=safety_applied, selected_physical_root_index=0,
        selected_physical_root_eigenvalue=(float(evals[0]) if evals.size else None),
        physical_negative_mode_count=neg_count, physical_root_policy="lowest",
        physical_root_degenerate=degenerate, coupling_norm=None, coupling_relative=None,
        coupling_warning=False, coupling_tolerance=None, matrix_symmetry_cleanup=False,
        matrix_symmetry_residual=0.0, model_origin=request.model_origin,
        model_update_type=request.model_update_type, model_age=request.model_age,
        state_uid=request.state_uid, geometry_id=request.geometry_id,
        coordinate_space_id=request.coordinate_space_id, center_observation_id=request.center_observation_id,
        force_units=request.force_units, hessian_units=request.hessian_units,
        p_root_metadata=metadata, q_root_metadata=None,
        restricted_metadata=(None if fixed_ras is None else fixed_ras.metadata()),
        eigensolve_time_ns=0, root_solve_time_ns=root_solve_time,
        root_trials=root_trials, t17_added_pes_calls=0,
    )


def compute_rfo_translation(
    request: RFOTranslationInput,
    *,
    partition: str,
    step_control: str = STEP_UNRESTRICTED,
    target_order: int = 1,
    physical_root_policy: str = "lowest",
    physical_homing_mode: object | None = None,
    negative_mode_policy: str = "allow",
    negative_tolerance: float = 1.0e-12,
    physical_degeneracy_tolerance: float = 1.0e-10,
    external_coupling_tolerance: float = 1.0e-6,
    norm_cap: float | None = None,
    trust_radius: float | None = None,
    restricted_tolerance: float = 1.0e-10,
    restricted_max_iterations: int = 100,
    restricted_alpha_min: float = 1.0e-12,
    ras_root_solver: str = "bisection",
    ras_tolerance: float | None = None,
    ras_max_iterations: int = RAS_DEFAULT_MAXITER,
    translation_step_method: str | None = None,
    sella_ras_tolerance: float | None = None,
    sella_ras_max_iterations: int = SELLA_RESTRICTED_MAXITER,
    sella_qn_curvature_regularization: float = SELLA_QN_CURVATURE_REGULARIZATION,
    final_safety_cap: float | None = None,
    denominator_tolerance: float = 1.0e-12,
    root_cluster_tolerance: float = 1.0e-10,
    matrix_symmetry_tolerance: float = 1.0e-12,
) -> RFOTranslationResult:
    """Compute one native translation from a frozen C2 physical model input."""

    if not isinstance(request, RFOTranslationInput):
        raise RFOTranslationInputError("rfo_translation_input_required")
    partition = str(partition).strip().lower()
    step_control = str(step_control).strip().lower()
    if partition not in SUPPORTED_PARTITIONS:
        raise RFOTranslationInputError("unsupported_rfo_partition")
    if step_control not in SUPPORTED_STEP_CONTROLS:
        raise RFOTranslationInputError("unsupported_rfo_step_control")
    if request.t17_added_pes_calls != 0:
        raise RFOTranslationInputError("t17_must_not_add_pes_calls")
    if request.matrix is None:
        return _compute_compact_qn_mmf_translation(
            request, partition=partition, step_control=step_control, target_order=int(target_order),
            physical_root_policy=physical_root_policy, negative_mode_policy=negative_mode_policy,
            negative_tolerance=float(negative_tolerance), physical_degeneracy_tolerance=float(physical_degeneracy_tolerance),
            norm_cap=norm_cap, trust_radius=trust_radius,
            ras_root_solver=ras_root_solver, ras_tolerance=ras_tolerance,
            ras_max_iterations=int(ras_max_iterations), translation_step_method=translation_step_method,
            final_safety_cap=final_safety_cap,
        )

    B, cleanup, symmetry_residual = _matrix_symmetry(
        request.matrix, float(matrix_symmetry_tolerance)
    )
    g = request.raw_gradient
    adapter = _CoordinateAdapter(request.coordinate_space)
    if adapter.dimension != g.size:
        raise RFOTranslationInputError("active_basis_dimension_mismatch")
    root_kwargs = {
        "denominator_tolerance": float(denominator_tolerance),
        "cluster_tolerance": float(root_cluster_tolerance),
        "symmetry_tolerance": float(matrix_symmetry_tolerance),
    }

    physical_meta: dict[str, object] = {
        "selected_index": None,
        "selected_eigenvalue": None,
        "negative_mode_count": int(np.count_nonzero(np.linalg.eigvalsh(B) < -float(negative_tolerance))),
        "root_policy": "not_applicable",
        "degenerate": False,
    }
    coupling_tolerance = None
    partition_result = None
    restricted = None
    fixed_ras: FixedRASResult | GenericRASResult | None = None
    single_result = None
    P = None
    Q = None
    diagnostic_dir, _, diagnostic_max = _diagnostic_settings()
    restricted_trial_trace: list[dict[str, object]] | None = (
        [] if diagnostic_dir is not None and diagnostic_max > 0 else None
    )

    if partition == PARTITION_UNPARTITIONED_ORDER0:
        algorithm = "rfo"
        if step_control in {STEP_RESTRICTED_PRFO, STEP_RAS, STEP_SELLA_FIXED_RAS, STEP_SELLA_QN_FIXED_RAS}:
            raise RFOTranslationInputError("order0_restricted_rfo_not_enabled")
        single_result = rfo_order0(B, g, alpha=1.0, **root_kwargs)
    elif partition == PARTITION_UNPARTITIONED_ORDER1:
        algorithm = "rfo"
        if step_control in {STEP_RESTRICTED_PRFO, STEP_SELLA_FIXED_RAS, STEP_SELLA_QN_FIXED_RAS}:
            raise RFOTranslationInputError("legacy_restricted_step_requires_partitioned_physical_model")
        if step_control == STEP_RAS:
            if trust_radius is None:
                raise RFOTranslationInputError("ras_requires_trust_radius")
            lift = lambda reduced: adapter.expand(reduced).reshape(-1)
            fixed_ras = solve_rfo_ras(
                B,
                g,
                lift=lift,
                radius=float(trust_radius),
                order=int(target_order),
                tolerance=ras_tolerance,
                max_iterations=int(ras_max_iterations),
                root_solver=ras_root_solver,
            )
        else:
            single_result = rfo_order1(B, g, alpha=1.0, **root_kwargs)
    else:
        algorithm = "prfo"
        if partition == PARTITION_PHYSICAL_B_EIGEN:
            previous_reduced = None
            if physical_homing_mode is not None:
                previous_reduced = adapter.reduce(physical_homing_mode)
            P, Q, physical_meta = _physical_b_partition(
                B,
                target_order=int(target_order),
                root_policy=physical_root_policy,
                previous_mode=previous_reduced,
                negative_mode_policy=negative_mode_policy,
                negative_tolerance=float(negative_tolerance),
                degeneracy_tolerance=float(physical_degeneracy_tolerance),
            )
        else:
            if request.external_mode is None:
                raise RFOTranslationInputError("external_mode_partition_requires_current_mode")
            P, Q = _external_partition(request.external_mode)
            physical_meta["root_policy"] = "external_projector_not_physical_b_root"
            coupling_tolerance = float(external_coupling_tolerance)
            if not np.isfinite(coupling_tolerance) or coupling_tolerance < 0.0:
                raise RFOTranslationInputError("invalid_external_coupling_tolerance")

        if step_control == STEP_RAS:
            if trust_radius is None:
                raise RFOTranslationInputError("ras_requires_trust_radius")
            method = str(translation_step_method or "prfo").strip().lower()
            if method == "prfo":
                if partition not in {PARTITION_PHYSICAL_B_EIGEN, PARTITION_EXTERNAL_MODE}:
                    raise RFOTranslationInputError("prfo_ras_requires_physical_b_eigen_or_external_mode_partition")
                if (
                    partition == PARTITION_PHYSICAL_B_EIGEN
                    and str(physical_root_policy).strip().lower() != "lowest"
                ):
                    raise RFOTranslationInputError("ras_requires_lowest_physical_b_root")
            elif method == "qn_mmf":
                if partition != PARTITION_PHYSICAL_B_EIGEN:
                    raise RFOTranslationInputError("qn_mmf_ras_requires_physical_b_eigen_partition")
                if str(physical_root_policy).strip().lower() != "lowest":
                    raise RFOTranslationInputError("ras_requires_lowest_physical_b_root")
            else:
                raise RFOTranslationInputError("ras_partitioned_method_must_be_qn_mmf_or_prfo")
            assert P is not None and Q is not None
            lift = lambda reduced: adapter.expand(reduced).reshape(-1)
            if method == "prfo":
                fixed_ras = solve_prfo_ras(
                    B, g, P, Q, lift=lift, radius=float(trust_radius),
                    tolerance=ras_tolerance, max_iterations=int(ras_max_iterations),
                    root_solver=ras_root_solver,
                )
                algorithm = "prfo"
            elif method == "qn_mmf":
                shared_evecs = np.column_stack((P, Q))
                shared_evals = np.diag(shared_evecs.T @ B @ shared_evecs)
                fixed_ras = solve_qn_mmf_ras(
                    B, g, lift=lift, radius=float(trust_radius), order=int(target_order),
                    tolerance=ras_tolerance, max_iterations=int(ras_max_iterations),
                    root_solver=ras_root_solver, eigenvalues=shared_evals, eigenvectors=shared_evecs,
                )
                algorithm = "qn_mmf"
            if not fixed_ras.success or fixed_ras.step is None:
                return RFOTranslationResult(
                    success=False, status=fixed_ras.status, failure_reason=fixed_ras.failure_reason,
                    algorithm=algorithm, partition=partition, step_control=step_control,
                    cartesian_step=None, reduced_step=None, raw_step_norm=None, achieved_norm=None,
                    alpha=fixed_ras.alpha, requested_radius=fixed_ras.requested_radius,
                    restricted_boundary_active=fixed_ras.boundary_active, norm_cap=None,
                    norm_cap_applied=False, final_safety_cap=None, final_safety_cap_applied=False,
                    selected_physical_root_index=physical_meta.get("selected_index"),
                    selected_physical_root_eigenvalue=physical_meta.get("selected_eigenvalue"),
                    physical_negative_mode_count=int(physical_meta.get("negative_mode_count", 0)),
                    physical_root_policy=str(physical_meta.get("root_policy", "not_applicable")),
                    physical_root_degenerate=bool(physical_meta.get("degenerate", False)),
                    coupling_norm=None, coupling_relative=None, coupling_warning=False, coupling_tolerance=None,
                    matrix_symmetry_cleanup=cleanup, matrix_symmetry_residual=symmetry_residual,
                    model_origin=request.model_origin, model_update_type=request.model_update_type,
                    model_age=request.model_age, state_uid=request.state_uid, geometry_id=request.geometry_id,
                    coordinate_space_id=request.coordinate_space_id, center_observation_id=request.center_observation_id,
                    force_units=request.force_units, hessian_units=request.hessian_units,
                    p_root_metadata=None, q_root_metadata=None, restricted_metadata=fixed_ras.metadata(),
                    eigensolve_time_ns=0, root_solve_time_ns=fixed_ras.solve_time_ns,
                    root_trials=fixed_ras.evaluations, t17_added_pes_calls=0,
                )
        elif step_control in {STEP_SELLA_FIXED_RAS, STEP_SELLA_QN_FIXED_RAS}:
            if partition != PARTITION_PHYSICAL_B_EIGEN:
                raise RFOTranslationInputError("sella_fixed_ras_requires_physical_b_eigen_partition")
            if str(physical_root_policy).strip().lower() != "lowest":
                raise RFOTranslationInputError("sella_fixed_ras_requires_lowest_physical_b_root")
            if trust_radius is None:
                raise RFOTranslationInputError("sella_fixed_ras_requires_trust_radius")
            assert P is not None and Q is not None
            lift = lambda reduced: adapter.expand(reduced).reshape(-1)
            if step_control == STEP_SELLA_FIXED_RAS:
                fixed_ras = solve_sella_prfo_fixed_ras(
                    B,
                    g,
                    P,
                    Q,
                    lift=lift,
                    radius=float(trust_radius),
                    tolerance=sella_ras_tolerance,
                    max_iterations=int(sella_ras_max_iterations),
                )
                algorithm = "prfo"
            else:
                shared_evecs = np.column_stack((P, Q))
                shared_evals = np.diag(shared_evecs.T @ B @ shared_evecs)
                fixed_ras = solve_sella_qn_mmf_fixed_ras(
                    B,
                    g,
                    lift=lift,
                    radius=float(trust_radius),
                    order=int(target_order),
                    tolerance=sella_ras_tolerance,
                    max_iterations=int(sella_ras_max_iterations),
                    curvature_regularization=float(sella_qn_curvature_regularization),
                    eigenvalues=shared_evals,
                    eigenvectors=shared_evecs,
                )
                algorithm = "qn_mmf"
            if not fixed_ras.success or fixed_ras.step is None:
                return RFOTranslationResult(
                    success=False,
                    status=fixed_ras.status,
                    failure_reason=fixed_ras.failure_reason,
                    algorithm=algorithm,
                    partition=partition,
                    step_control=step_control,
                    cartesian_step=None,
                    reduced_step=None,
                    raw_step_norm=None,
                    achieved_norm=None,
                    alpha=fixed_ras.alpha,
                    requested_radius=fixed_ras.requested_radius,
                    restricted_boundary_active=fixed_ras.boundary_active,
                    norm_cap=None,
                    norm_cap_applied=False,
                    final_safety_cap=None,
                    final_safety_cap_applied=False,
                    selected_physical_root_index=physical_meta.get("selected_index"),
                    selected_physical_root_eigenvalue=physical_meta.get("selected_eigenvalue"),
                    physical_negative_mode_count=int(physical_meta.get("negative_mode_count", 0)),
                    physical_root_policy=str(physical_meta.get("root_policy", "not_applicable")),
                    physical_root_degenerate=bool(physical_meta.get("degenerate", False)),
                    coupling_norm=None,
                    coupling_relative=None,
                    coupling_warning=False,
                    coupling_tolerance=None,
                    matrix_symmetry_cleanup=cleanup,
                    matrix_symmetry_residual=symmetry_residual,
                    model_origin=request.model_origin,
                    model_update_type=request.model_update_type,
                    model_age=request.model_age,
                    state_uid=request.state_uid,
                    geometry_id=request.geometry_id,
                    coordinate_space_id=request.coordinate_space_id,
                    center_observation_id=request.center_observation_id,
                    force_units=request.force_units,
                    hessian_units=request.hessian_units,
                    p_root_metadata=None,
                    q_root_metadata=None,
                    restricted_metadata=fixed_ras.metadata(),
                    eigensolve_time_ns=0,
                    root_solve_time_ns=fixed_ras.solve_time_ns,
                    root_trials=fixed_ras.evaluations,
                    t17_added_pes_calls=0,
                )
        elif step_control == STEP_RESTRICTED_PRFO:
            if trust_radius is None:
                raise RFOTranslationInputError("restricted_prfo_requires_trust_radius")
            try:
                restricted = restricted_partitioned_rfo(
                    B,
                    g,
                    P,
                    Q,
                    radius=float(trust_radius),
                    tolerance=float(restricted_tolerance),
                    max_iterations=int(restricted_max_iterations),
                    alpha_min=float(restricted_alpha_min),
                    trial_trace=restricted_trial_trace,
                    **root_kwargs,
                )
            except RFOSymmetryError as exc:
                _w5_004_capture_partition_symmetry_failure(
                    request,
                    model_matrix=B,
                    gradient=g,
                    P=P,
                    Q=Q,
                    partition=partition,
                    step_control=step_control,
                    target_order=int(target_order),
                    matrix_symmetry_tolerance=float(matrix_symmetry_tolerance),
                    error=exc,
                )
                raise
            partition_result = restricted.result
            if not restricted.success or partition_result is None or not partition_result.success:
                failed = _failed_result(
                    request,
                    algorithm=algorithm,
                    partition=partition,
                    step_control=step_control,
                    failure_reason=restricted.failure_reason or restricted.status,
                    matrix_cleanup=cleanup,
                    matrix_symmetry_residual=symmetry_residual,
                    physical_meta=physical_meta,
                    coupling_tolerance=coupling_tolerance,
                    partition_result=partition_result,
                    restricted=restricted,
                )
                assert P is not None and Q is not None
                if restricted_trial_trace is not None:
                    _maybe_write_restricted_diagnostic(
                        request,
                        B=B,
                        g=g,
                        P=P,
                        Q=Q,
                        partition=partition,
                        physical_meta=physical_meta,
                        restricted=restricted,
                        trial_trace=restricted_trial_trace,
                        matrix_cleanup=cleanup,
                        matrix_symmetry_residual=symmetry_residual,
                        negative_tolerance=float(negative_tolerance),
                        denominator_tolerance=float(denominator_tolerance),
                        root_cluster_tolerance=float(root_cluster_tolerance),
                        restricted_tolerance=float(restricted_tolerance),
                        restricted_alpha_min=float(restricted_alpha_min),
                    )
                return failed
        else:
            try:
                partition_result = solve_partitioned_rfo(B, g, P, Q, alpha=1.0, **root_kwargs)
            except RFOSymmetryError as exc:
                _w5_004_capture_partition_symmetry_failure(
                    request,
                    model_matrix=B,
                    gradient=g,
                    P=P,
                    Q=Q,
                    partition=partition,
                    step_control=step_control,
                    target_order=int(target_order),
                    matrix_symmetry_tolerance=float(matrix_symmetry_tolerance),
                    error=exc,
                )
                raise
            if not partition_result.success:
                return _failed_result(
                    request,
                    algorithm=algorithm,
                    partition=partition,
                    step_control=step_control,
                    failure_reason=partition_result.failure_reason,
                    matrix_cleanup=cleanup,
                    matrix_symmetry_residual=symmetry_residual,
                    physical_meta=physical_meta,
                    coupling_tolerance=coupling_tolerance,
                    partition_result=partition_result,
                )

    if fixed_ras is not None and (not fixed_ras.success or fixed_ras.step is None):
        return RFOTranslationResult(
            success=False, status=fixed_ras.status, failure_reason=fixed_ras.failure_reason,
            algorithm=algorithm, partition=partition, step_control=step_control,
            cartesian_step=None, reduced_step=None, raw_step_norm=None, achieved_norm=None,
            alpha=fixed_ras.alpha, requested_radius=fixed_ras.requested_radius,
            restricted_boundary_active=fixed_ras.boundary_active, norm_cap=None,
            norm_cap_applied=False, final_safety_cap=None, final_safety_cap_applied=False,
            selected_physical_root_index=physical_meta.get("selected_index"),
            selected_physical_root_eigenvalue=physical_meta.get("selected_eigenvalue"),
            physical_negative_mode_count=int(physical_meta.get("negative_mode_count", 0)),
            physical_root_policy=str(physical_meta.get("root_policy", "not_applicable")),
            physical_root_degenerate=bool(physical_meta.get("degenerate", False)),
            coupling_norm=None, coupling_relative=None, coupling_warning=False, coupling_tolerance=None,
            matrix_symmetry_cleanup=cleanup, matrix_symmetry_residual=symmetry_residual,
            model_origin=request.model_origin, model_update_type=request.model_update_type,
            model_age=request.model_age, state_uid=request.state_uid, geometry_id=request.geometry_id,
            coordinate_space_id=request.coordinate_space_id, center_observation_id=request.center_observation_id,
            force_units=request.force_units, hessian_units=request.hessian_units,
            p_root_metadata=None, q_root_metadata=None, restricted_metadata=fixed_ras.metadata(),
            eigensolve_time_ns=0, root_solve_time_ns=fixed_ras.solve_time_ns,
            root_trials=fixed_ras.evaluations, t17_added_pes_calls=0,
        )

    if single_result is not None:
        if not single_result.success or single_result.step is None:
            return RFOTranslationResult(
                success=False,
                status="failed",
                failure_reason=single_result.failure_reason,
                algorithm=algorithm,
                partition=partition,
                step_control=step_control,
                cartesian_step=None,
                reduced_step=None,
                raw_step_norm=None,
                achieved_norm=None,
                alpha=single_result.alpha,
                requested_radius=None,
                restricted_boundary_active=False,
                norm_cap=None,
                norm_cap_applied=False,
                final_safety_cap=None,
                final_safety_cap_applied=False,
                selected_physical_root_index=None,
                selected_physical_root_eigenvalue=None,
                physical_negative_mode_count=int(physical_meta["negative_mode_count"]),
                physical_root_policy="not_applicable",
                physical_root_degenerate=False,
                coupling_norm=None,
                coupling_relative=None,
                coupling_warning=False,
                coupling_tolerance=None,
                matrix_symmetry_cleanup=cleanup,
                matrix_symmetry_residual=symmetry_residual,
                model_origin=request.model_origin,
                model_update_type=request.model_update_type,
                model_age=request.model_age,
                state_uid=request.state_uid,
                geometry_id=request.geometry_id,
                coordinate_space_id=request.coordinate_space_id,
                center_observation_id=request.center_observation_id,
                force_units=request.force_units,
                hessian_units=request.hessian_units,
                p_root_metadata=single_result.metadata(),
                q_root_metadata=None,
                restricted_metadata=None,
                eigensolve_time_ns=single_result.eigensolve_time_ns,
                root_solve_time_ns=0,
                root_trials=1,
                t17_added_pes_calls=0,
            )
        raw_step = np.array(single_result.step, copy=True)
        alpha = single_result.alpha
        eig_time = single_result.eigensolve_time_ns
        p_meta = single_result.metadata()
        q_meta = None
        coupling_norm = None
        coupling_relative = None
        coupling_warning = False
        root_trials = 1
        root_solve_time = 0
        requested_radius = None
        boundary_active = False
    elif fixed_ras is not None:
        assert fixed_ras.success and fixed_ras.step is not None
        raw_step = np.array(fixed_ras.step, copy=True)
        alpha = float(fixed_ras.alpha) if fixed_ras.alpha is not None else 0.0
        eig_time = 0
        last = dict(fixed_ras.last_metadata or {})
        if algorithm == "prfo":
            p_meta = last.get("p_root") if isinstance(last.get("p_root"), Mapping) else None
            q_meta = last.get("q_root") if isinstance(last.get("q_root"), Mapping) else None
        else:
            p_meta = last
            q_meta = None
        coupling_norm = None
        coupling_relative = None
        coupling_warning = False
        root_trials = int(fixed_ras.evaluations)
        root_solve_time = int(fixed_ras.solve_time_ns)
        requested_radius = float(fixed_ras.requested_radius)
        boundary_active = bool(fixed_ras.boundary_active)
    else:
        assert partition_result is not None and partition_result.success and partition_result.step is not None
        raw_step = np.array(partition_result.step, copy=True)
        alpha = partition_result.alpha
        eig_time = partition_result.eigensolve_time_ns
        p_meta = _branch_metadata(partition_result.p_result)
        q_meta = _branch_metadata(partition_result.q_result)
        coupling_norm = partition_result.coupling_norm
        coupling_relative = partition_result.coupling_relative
        coupling_warning = bool(
            coupling_tolerance is not None and coupling_relative > coupling_tolerance
        )
        root_trials = 1 if restricted is None else restricted.root_trials
        root_solve_time = 0 if restricted is None else restricted.root_solve_time_ns
        requested_radius = None if restricted is None else restricted.requested_radius
        boundary_active = False if restricted is None else restricted.boundary_active

    raw_norm = float(np.linalg.norm(raw_step))
    final_step = np.array(raw_step, copy=True)
    cap_value = None
    cap_applied = False
    if step_control == STEP_NORM_CAP:
        if norm_cap is None:
            raise RFOTranslationInputError("norm_cap_policy_requires_norm_cap")
        cap_value = float(norm_cap)
        if not np.isfinite(cap_value) or cap_value <= 0.0:
            raise RFOTranslationInputError("norm_cap_must_be_finite_positive")
        if raw_norm > cap_value and raw_norm > 0.0:
            final_step *= cap_value / raw_norm
            cap_applied = True
    elif step_control == STEP_UNRESTRICTED:
        if norm_cap is not None:
            raise RFOTranslationInputError("norm_cap_value_requires_norm_cap_policy")
    elif norm_cap is not None:
        raise RFOTranslationInputError("norm_cap_is_distinct_from_restricted_step_control")

    safety_value = None
    safety_applied = False
    if final_safety_cap is not None:
        safety_value = float(final_safety_cap)
        if not np.isfinite(safety_value) or safety_value <= 0.0:
            raise RFOTranslationInputError("final_safety_cap_must_be_finite_positive")
        current = float(np.linalg.norm(final_step))
        if current > safety_value and current > 0.0:
            final_step *= safety_value / current
            safety_applied = True

    achieved = float(np.linalg.norm(final_step))
    cartesian = adapter.expand(final_step)
    # Exact fixed-coordinate/null-space preservation is a postcondition.
    if not np.allclose(cartesian, request.coordinate_space.project(cartesian), rtol=0.0, atol=0.0):
        raise AssertionError("RFO step lift violated ActiveCoordinateSpace projection")

    status = "ok"
    if step_control == STEP_RESTRICTED_PRFO and restricted is not None:
        status = restricted.status
    elif step_control in {STEP_RAS, STEP_SELLA_FIXED_RAS, STEP_SELLA_QN_FIXED_RAS} and fixed_ras is not None:
        status = fixed_ras.status
    elif step_control == STEP_NORM_CAP:
        status = "posthoc_norm_capped" if cap_applied else "unrestricted_with_inactive_norm_cap"
    else:
        status = "unrestricted"

    output = RFOTranslationResult(
        success=True,
        status=status,
        failure_reason="",
        algorithm=algorithm,
        partition=partition,
        step_control=step_control,
        cartesian_step=cartesian,
        reduced_step=final_step,
        raw_step_norm=raw_norm,
        achieved_norm=achieved,
        alpha=float(alpha),
        requested_radius=requested_radius,
        restricted_boundary_active=boundary_active,
        norm_cap=cap_value,
        norm_cap_applied=cap_applied,
        final_safety_cap=safety_value,
        final_safety_cap_applied=safety_applied,
        selected_physical_root_index=physical_meta.get("selected_index"),
        selected_physical_root_eigenvalue=physical_meta.get("selected_eigenvalue"),
        physical_negative_mode_count=int(physical_meta.get("negative_mode_count", 0)),
        physical_root_policy=str(physical_meta.get("root_policy", "not_applicable")),
        physical_root_degenerate=bool(physical_meta.get("degenerate", False)),
        coupling_norm=coupling_norm,
        coupling_relative=coupling_relative,
        coupling_warning=coupling_warning,
        coupling_tolerance=coupling_tolerance,
        matrix_symmetry_cleanup=cleanup,
        matrix_symmetry_residual=symmetry_residual,
        model_origin=request.model_origin,
        model_update_type=request.model_update_type,
        model_age=request.model_age,
        state_uid=request.state_uid,
        geometry_id=request.geometry_id,
        coordinate_space_id=request.coordinate_space_id,
        center_observation_id=request.center_observation_id,
        force_units=request.force_units,
        hessian_units=request.hessian_units,
        p_root_metadata=p_meta,
        q_root_metadata=q_meta,
        restricted_metadata=(
            fixed_ras.metadata() if fixed_ras is not None
            else (None if restricted is None else restricted.metadata())
        ),
        eigensolve_time_ns=int(eig_time),
        root_solve_time_ns=int(root_solve_time),
        root_trials=int(root_trials),
        t17_added_pes_calls=0,
    )
    if restricted is not None and restricted_trial_trace is not None:
        assert P is not None and Q is not None
        _maybe_write_restricted_diagnostic(
            request,
            B=B,
            g=g,
            P=P,
            Q=Q,
            partition=partition,
            physical_meta=physical_meta,
            restricted=restricted,
            trial_trace=restricted_trial_trace,
            matrix_cleanup=cleanup,
            matrix_symmetry_residual=symmetry_residual,
            negative_tolerance=float(negative_tolerance),
            denominator_tolerance=float(denominator_tolerance),
            root_cluster_tolerance=float(root_cluster_tolerance),
            restricted_tolerance=float(restricted_tolerance),
            restricted_alpha_min=float(restricted_alpha_min),
        )
    return output


__all__ = [
    "CANONICAL_FORCE_UNITS",
    "CANONICAL_HESSIAN_UNITS",
    "PARTITION_EXTERNAL_MODE",
    "PARTITION_PHYSICAL_B_EIGEN",
    "PARTITION_UNPARTITIONED_ORDER0",
    "PARTITION_UNPARTITIONED_ORDER1",
    "RFO_DIAGNOSTIC_DIR_ENV",
    "RFO_DIAGNOSTIC_MAX_ENV",
    "RFO_DIAGNOSTIC_MODE_ENV",
    "RFO_DIAGNOSTIC_SCHEMA",
    "RFOTranslationInput",
    "RFOTranslationInputError",
    "RFOTranslationResult",
    "STEP_NORM_CAP",
    "STEP_RESTRICTED_PRFO",
    "STEP_RAS",
    "STEP_SELLA_FIXED_RAS",
    "STEP_SELLA_QN_FIXED_RAS",
    "STEP_UNRESTRICTED",
    "SUPPORTED_PARTITIONS",
    "SUPPORTED_STEP_CONTROLS",
    "PHYSICAL_HESSIAN_MODEL_ORIGIN",
    "build_c2_rfo_input",
    "compute_rfo_translation",
]
