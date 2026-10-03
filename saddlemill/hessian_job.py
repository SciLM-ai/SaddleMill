"""Standalone analytical Hessian job for SaddleMill.

Scientific semantics for order certification:
- FairChem UMA analytical Cartesian Hessian;
- ASE FixAtoms removes fixed Cartesian DOFs before diagonalization;
- with any FixAtoms constraint, no global rigid translation is projected;
- with no constraints and periodic boundary conditions, the three rigid Cartesian
  translations are projected out before the saddle order is counted;
- no cm^-1 cutoff is used for order; negative modes use only the configured
  Cartesian eigenvalue tolerance (default 1e-6 eV/A^2);
- a first-order saddle has exactly one negative eigenvalue in that order space.

Performance implementation:
- process-local FairChem Hessian assembler patch; installed packages are untouched;
- batched analytical VJPs ("chunks") instead of full vmap;
- FixAtoms restriction computes only Hessian columns needed by the active/free block;
- production default starts aggressively at chunk 32 on one client/GPU;
- bounded CUDA OOM fallback halves the chunk 32 -> 16 -> 8 -> 4 -> 2 -> 1.

The output trajectory is designed to feed directly into DoubleMinimization.
DoubleMin auto-detects the standalone-Hessian schema and uses its verified
``eigenmode``, ``curvature``, and reported order.  The historical
``pre_hessian_eigenmode`` flag controls only legacy inline recomputation.
"""
from __future__ import annotations

import csv
import functools
import gc
import inspect
import math
import os
import time
import traceback
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Mapping

import numpy as np
from ase.constraints import FixAtoms

from saddlemill.hessian_artifacts import (
    ARTIFACT_CONTRACT,
    HessianArtifactIdentityMismatch,
    HessianArtifactPublicationError,
    HessianArtifactStore,
)

SUPPORTED_FAIRCHEM_VERSION = "2.20.0"

# Process-local predictor / Hessian-patch state. Each executorlib worker is a
# separate process, so these are never shared across concurrent Hessian jobs.
_HESSIAN_CALC = None
_HESSIAN_CALC_KEY = None
_HESSIAN_PID = None
_PATCH_INSTALLED = False
_PATCH_ORIGINAL = None
_PATCH_SIGNATURE = None
# FairChem 2.20.0 was directly measured/validated to return a leading-batch
# flat Hessian tensor with shape (1, 3N, 3N). The version and function
# signature are pinned below, so production does not recompute a second
# reference Hessian merely to rediscover this contract on every worker.
_PATCH_CONTRACT_KIND = "batch_flat"
_PATCH_CONTRACT_PROBED = True
_PATCH_ACTIVE_DOFS: list[int] | None = None
_PATCH_CHUNK = 1
_PATCH_RESTRICT = False
_PATCH_LAST_PROBE = False
_PATCH_META: dict[str, Any] = {}

SUMMARY_FIELDS = [
    "src_index", "rank", "parent_source_idx", "parent_attempt_id",
    "natoms", "free_atoms", "fixed_atoms", "free_dof_count",
    "pbc", "has_constraints", "restricted",
    "order_projection", "removed_translation_modes", "order_dof_count",
    "raw_active_negative_mode_count", "negative_mode_count", "is_first_order",
    "chunk_requested", "chunk_initial", "chunk_used",
    "oom_retries", "contract_probe_performed", "hessian_seconds",
    "negative_eigenvalue_tolerance",
    "lowest_eigenvalue", "second_eigenvalue", "hessian_file",
    "task_name", "model_name_or_path", "fairchem_core_version",
    "jobs_per_gpu", "status",
]


@dataclass
class HessianResult:
    active_hessian: np.ndarray
    order_hessian: np.ndarray
    free_dofs: np.ndarray
    raw_active_eigenvalues: np.ndarray
    eigenvalues: np.ndarray
    lowest_mode: np.ndarray
    raw_active_negative_modes: int
    negative_modes: int
    order_projection: str
    removed_translation_modes: int
    order_dof_count: int
    hessian_seconds: float
    chunk_requested: Any
    chunk_initial: int
    chunk_used: int
    oom_retries: int
    restricted: bool
    has_constraints: bool
    contract_probe_performed: bool
    metadata: dict[str, Any]


def _write_summary(path: str, row: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})
    os.replace(tmp, path)


def _free_cartesian_dofs(atoms) -> tuple[np.ndarray, set[int]]:
    """Return active Cartesian DOFs and the fixed atom set.

    This implementation supports no constraints or ASE FixAtoms only. It
    deliberately rejects partial/collective constraints rather than silently
    projecting them incorrectly.
    """
    fixed: set[int] = set()
    for constraint in atoms.constraints:
        if not isinstance(constraint, FixAtoms):
            raise NotImplementedError(
                "Hessian currently supports no constraints or ASE FixAtoms only; "
                f"got {type(constraint).__name__}."
            )
        fixed.update(int(i) for i in constraint.get_indices())
    free = [
        3 * atom_i + comp
        for atom_i in range(len(atoms))
        if atom_i not in fixed
        for comp in range(3)
    ]
    if not free:
        raise ValueError("Hessian found zero free Cartesian degrees of freedom.")
    return np.asarray(free, dtype=int), fixed


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _cuda_cleanup(calc=None) -> None:
    try:
        if calc is not None:
            calc.reset()
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _cuda_sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _assemble_chunked_hessian(
    forces,
    pos,
    selected: list[int],
    chunk: int,
    *,
    retain_graph_after: bool = False,
):
    """Return flat (3N,3N) analytical Hessian with FairChem orientation.

    FairChem 2.20's low-memory convention is
        H[:, i] = d(-F_i) / dR
    Chunking computes several such VJPs in one autograd call.
    """
    import torch

    if pos.ndim != 2 or int(pos.shape[-1]) != 3:
        raise RuntimeError(f"Expected pos shape (N,3), got {tuple(pos.shape)}")
    if tuple(forces.shape) != tuple(pos.shape):
        raise RuntimeError(
            f"Expected force/position shape match, got forces={tuple(forces.shape)} "
            f"pos={tuple(pos.shape)}"
        )
    forces_flat = forces.reshape(-1)
    ndof = int(pos.numel())
    selected = [int(i) for i in selected]
    if not selected:
        raise RuntimeError("zero selected Hessian DOFs")
    if min(selected) < 0 or max(selected) >= ndof:
        raise RuntimeError("selected Hessian DOF out of range")

    h = torch.zeros((ndof, ndof), device=pos.device, dtype=pos.dtype)
    chunk = max(1, int(chunk))
    total = len(selected)
    for start in range(0, total, chunk):
        ids = selected[start:start + chunk]
        is_last = start + len(ids) >= total
        retain = retain_graph_after or not is_last
        if len(ids) == 1:
            grads = torch.autograd.grad(
                -forces_flat[ids[0]], pos,
                retain_graph=retain, create_graph=False, allow_unused=False,
            )[0].reshape(1, ndof)
        else:
            row_idx = torch.as_tensor(ids, device=forces.device, dtype=torch.long)
            basis = torch.zeros(
                (len(ids), ndof), device=forces.device, dtype=forces.dtype
            )
            basis[torch.arange(len(ids), device=forces.device), row_idx] = 1
            grads = torch.autograd.grad(
                -forces_flat, pos, grad_outputs=basis,
                retain_graph=retain, create_graph=False,
                is_grads_batched=True, allow_unused=False,
            )[0].reshape(len(ids), ndof)
        h[:, ids] = grads.T
    return h


def _infer_contract_kind(shape: tuple[int, ...], natoms: int) -> str:
    ndof = 3 * int(natoms)
    known = {
        (ndof, ndof): "flat",
        (1, ndof, ndof): "batch_flat",
        (natoms, 3, natoms, 3): "atom4",
        (1, natoms, 3, natoms, 3): "batch_atom4",
    }
    if shape not in known:
        raise RuntimeError(
            "Unsupported FairChem internal Hessian return contract: "
            f"shape={shape}, natoms={natoms}. Refusing to guess."
        )
    return known[shape]


def _reshape_contract(hflat, kind: str, natoms: int):
    ndof = 3 * int(natoms)
    if kind == "flat":
        return hflat.reshape(ndof, ndof)
    if kind == "batch_flat":
        return hflat.reshape(1, ndof, ndof)
    if kind == "atom4":
        return hflat.reshape(natoms, 3, natoms, 3)
    if kind == "batch_atom4":
        return hflat.reshape(1, natoms, 3, natoms, 3)
    raise RuntimeError(f"unknown Hessian contract kind {kind!r}")


def _flatten_tensor(ref, natoms: int):
    import torch
    if not torch.is_tensor(ref):
        raise RuntimeError(
            f"FairChem compute_hessian returned {type(ref).__name__}; expected Tensor"
        )
    ndof = 3 * int(natoms)
    if int(ref.numel()) != ndof * ndof:
        raise RuntimeError(
            f"FairChem reference Hessian has {ref.numel()} values; expected {ndof*ndof}"
        )
    return ref.reshape(ndof, ndof)


def _install_chunked_patch() -> None:
    global _PATCH_INSTALLED, _PATCH_ORIGINAL, _PATCH_SIGNATURE
    if _PATCH_INSTALLED:
        return
    import fairchem.core.models.uma.escn_md as escn_md

    original = escn_md.compute_hessian
    signature = inspect.signature(original)
    needed = {"forces", "pos", "vmap", "training"}
    if not needed.issubset(signature.parameters):
        raise RuntimeError(f"Unexpected FairChem compute_hessian signature: {signature}")
    _PATCH_ORIGINAL = original
    _PATCH_SIGNATURE = signature
    _PATCH_META.update({
        "compute_hessian_signature": str(signature),
        "custom_matrix_orientation": "FairChem H[:,i]",
        "contract_strategy": "pinned_fairchem_2.20.0_batch_flat",
    })

    @functools.wraps(original)
    def patched(*args, **kwargs):
        global _PATCH_CONTRACT_KIND, _PATCH_CONTRACT_PROBED, _PATCH_LAST_PROBE
        bound = signature.bind_partial(*args, **kwargs)
        forces = bound.arguments["forces"]
        pos = bound.arguments["pos"]
        natoms = int(pos.shape[0])
        ndof = int(pos.numel())
        if int(forces.numel()) != ndof:
            raise RuntimeError("forces/positions DOF mismatch")
        selected = (
            list(_PATCH_ACTIVE_DOFS or [])
            if _PATCH_RESTRICT else list(range(ndof))
        )
        if _PATCH_RESTRICT and not selected:
            raise RuntimeError("restricted Hessian patch missing active DOFs")
        _PATCH_LAST_PROBE = False

        # Production path: the FairChem 2.20.0 return contract was already
        # established during development and is pinned to batch_flat. Avoid the
        # former per-worker reference probe, which computed an extra full
        # FairChem row-loop Hessian on the first real structure.
        _PATCH_LAST_PROBE = False

        if _PATCH_CONTRACT_KIND is None:
            raise RuntimeError("Hessian internal contract was not established")
        hflat = _assemble_chunked_hessian(
            forces, pos, selected, max(1, int(_PATCH_CHUNK))
        )
        return _reshape_contract(hflat, _PATCH_CONTRACT_KIND, natoms)

    patched._saddlemill_chunked_hessian = True
    escn_md.compute_hessian = patched
    _PATCH_INSTALLED = True


def _calc_key(config_dict: Mapping[str, Any]) -> tuple:
    calc_cfg = dict(config_dict.get("FAIRChemCalculator", {}) or {})
    return tuple(sorted((str(k), repr(v)) for k, v in calc_cfg.items()))


def _get_hessian_calc(config_dict: Mapping[str, Any]):
    """Return one chunk-capable Hessian predictor per worker process."""
    global _HESSIAN_CALC, _HESSIAN_CALC_KEY, _HESSIAN_PID
    global _PATCH_CONTRACT_KIND, _PATCH_CONTRACT_PROBED

    if metadata.version("fairchem-core") != SUPPORTED_FAIRCHEM_VERSION:
        raise RuntimeError(
            "Standalone Hessian currently requires fairchem-core=="
            f"{SUPPORTED_FAIRCHEM_VERSION}."
        )
    pid = os.getpid()
    key = _calc_key(config_dict)
    if _HESSIAN_PID != pid:
        _HESSIAN_CALC = None
        _HESSIAN_CALC_KEY = None
        _HESSIAN_PID = pid
        _PATCH_CONTRACT_KIND = "batch_flat"
        _PATCH_CONTRACT_PROBED = True
    if _HESSIAN_CALC is not None:
        if _HESSIAN_CALC_KEY != key:
            raise RuntimeError("Hessian worker calculator config changed mid-process")
        return _HESSIAN_CALC

    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import InferenceSettings

    _install_chunked_patch()
    calc_kwargs = dict(config_dict.get("FAIRChemCalculator", {}) or {})
    task_name = str(calc_kwargs.get("task_name", "")).strip()
    model_name = str(calc_kwargs.get("name_or_path", "")).strip()
    device = str(calc_kwargs.get("device", config_dict.get("Main", {}).get("device", "cuda")))
    if not task_name or not model_name:
        raise ValueError("[FAIRChemCalculator] name_or_path and task_name are required")
    if not device.lower().startswith("cuda"):
        raise ValueError("Standalone Hessian is currently validated only on CUDA")
    if int(calc_kwargs.get("workers", 1)) != 1:
        raise ValueError("Standalone Hessian requires [FAIRChemCalculator] workers=1")

    # Request vmap=True only to select the FairChem backend compatible with
    # generic batched VJPs. The process-local patch above controls chunking and
    # prevents FairChem from attempting the memory-prohibitive full vmap.
    calc_kwargs["inference_settings"] = InferenceSettings(
        predict_untrained_hessian={task_name}, hessian_vmap=True
    )
    calc = FAIRChemCalculator.from_model_checkpoint(**calc_kwargs)
    if "hessian" not in calc.implemented_properties:
        raise RuntimeError(
            f"UMA model/task did not expose Hessian; properties={calc.implemented_properties}"
        )
    backbone = calc.predictor.model.module.backbone
    if bool(backbone.training):
        raise RuntimeError("UMA Hessian backbone unexpectedly in training mode")
    regress = backbone.regress_config
    if not hasattr(regress, "hessian_vmap"):
        raise RuntimeError("FairChem regress_config lacks hessian_vmap")
    regress.hessian_vmap = True
    calc.sm_hessian_metadata = {
        "engine": "fairchem_chunked_analytical",
        "fairchem_core_version": SUPPORTED_FAIRCHEM_VERSION,
        "model_name_or_path": model_name,
        "task_name": task_name,
        "device": device,
        "backend_request_hessian_vmap": True,
        "backbone_type": type(backbone).__name__,
    }
    _HESSIAN_CALC = calc
    _HESSIAN_CALC_KEY = key
    return calc


def _normalize_raw(raw: Any, natoms: int) -> np.ndarray:
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    arr = np.asarray(raw, dtype=float)
    ndof = 3 * int(natoms)
    while arr.ndim > 2 and arr.shape[0] == 1 and arr.size == ndof * ndof:
        arr = arr[0]
    if arr.shape == (natoms, 3, natoms, 3):
        arr = arr.reshape(ndof, ndof)
    elif arr.shape != (ndof, ndof) and arr.size == ndof * ndof:
        arr = arr.reshape(ndof, ndof)
    if arr.shape != (ndof, ndof):
        raise RuntimeError(f"Hessian shape {arr.shape}; expected {(ndof, ndof)}")
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("Hessian contains non-finite values")
    return arr


def _translation_complement(ndof: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (translation_basis, orthonormal_complement) in Cartesian space.

    The Cartesian Hessian is differentiated with respect to ordinary positions,
    so a rigid translation is the equal-displacement vector for every atom.
    No mass weighting or frequency cutoff is involved in the order definition.
    """
    if ndof % 3 != 0:
        raise ValueError(f"Cartesian DOF count must be divisible by 3, got {ndof}")
    natoms = ndof // 3
    if natoms < 2:
        raise ValueError("Cannot remove three translations from fewer than two atoms")
    translations = np.zeros((ndof, 3), dtype=float)
    for atom_i in range(natoms):
        for axis in range(3):
            translations[3 * atom_i + axis, axis] = 1.0
    q, _ = np.linalg.qr(translations, mode="complete")
    t_basis = q[:, :3]
    complement = q[:, 3:]
    if complement.shape[1] != ndof - 3:
        raise RuntimeError("Unexpected translation-complement dimension")
    return t_basis, complement


def analyze_order_space(
    active_hessian: np.ndarray,
    atoms,
    free_dofs: np.ndarray,
    fixed_atoms: set[int],
    negative_tolerance: float,
) -> dict[str, Any]:
    """Analyze the exact Cartesian subspace used to define saddle order.

    Rules requested for SaddleMill's standalone Hessian job:
    - constrained (FixAtoms): diagonalize the free/free Cartesian Hessian and do
      not remove a global translation, because translating only the free atoms
      relative to the fixed environment is a physical mode;
    - unconstrained periodic: project out the three rigid translations exactly;
    - unconstrained non-periodic: currently make no additional rigid-body
      projection. Rotational projection is deliberately not inferred here.
    """
    h = np.asarray(active_hessian, dtype=float)
    h = 0.5 * (h + h.T)
    if h.shape != (len(free_dofs), len(free_dofs)):
        raise ValueError(
            f"active Hessian shape {h.shape} does not match {len(free_dofs)} free DOFs"
        )
    if not np.all(np.isfinite(h)):
        raise ValueError("active Hessian contains non-finite values")
    tol = float(negative_tolerance)
    if tol < 0.0:
        raise ValueError("negative_eigenvalue_tolerance must be >=0")

    raw_evals = np.linalg.eigvalsh(h)
    raw_negative = int(np.sum(raw_evals < -tol))
    has_constraints = bool(fixed_atoms)
    is_periodic = bool(np.any(np.asarray(atoms.pbc, dtype=bool)))

    if has_constraints:
        projection = "none_constrained_free_space"
        removed = 0
        order_h = h
        lift = np.eye(h.shape[0], dtype=float)
        translation_basis = np.empty((h.shape[0], 0), dtype=float)
    elif is_periodic:
        projection = "periodic_rigid_translations"
        removed = 3
        translation_basis, lift = _translation_complement(h.shape[0])
        order_h = lift.T @ h @ lift
        order_h = 0.5 * (order_h + order_h.T)
    else:
        projection = "none_nonperiodic"
        removed = 0
        order_h = h
        lift = np.eye(h.shape[0], dtype=float)
        translation_basis = np.empty((h.shape[0], 0), dtype=float)

    evals, evecs = np.linalg.eigh(order_h)
    if not evals.size:
        raise RuntimeError("Hessian order space has no eigenvalues")
    negative = int(np.sum(evals < -tol))
    active_mode = lift @ evecs[:, 0]
    mode_flat = np.zeros(3 * len(atoms), dtype=float)
    mode_flat[np.asarray(free_dofs, dtype=int)] = active_mode
    norm = float(np.linalg.norm(mode_flat))
    if not math.isfinite(norm) or norm < 1.0e-14:
        raise RuntimeError("lowest order-space Hessian eigenmode is zero/non-finite")
    mode = (mode_flat / norm).reshape((len(atoms), 3))
    return {
        "active_hessian": h,
        "order_hessian": order_h,
        "raw_active_eigenvalues": raw_evals,
        "eigenvalues": evals,
        "lowest_mode": mode,
        "raw_active_negative_modes": raw_negative,
        "negative_modes": negative,
        "order_projection": projection,
        "removed_translation_modes": removed,
        "order_dof_count": int(order_h.shape[0]),
        "translation_basis_active": translation_basis,
        "has_constraints": has_constraints,
        "is_periodic": is_periodic,
    }


def choose_auto_chunk(natoms: int, jobs_per_gpu: int) -> int:
    """Compatibility alias for the production chunk policy.

    Standalone Hessian production is now intentionally simple: one client/GPU
    should start at chunk 32 and rely on bounded OOM fallback.  The arguments
    remain accepted so old configs with ``chunk_size = auto`` keep working.
    """
    del natoms
    if int(jobs_per_gpu) != 1:
        print(
            "WARNING Hessian: chunk_size=auto now assumes the validated "
            "one-client/GPU production policy; jobs_per_gpu != 1 is allowed "
            "but no MPS-specific chunk reduction is applied.",
            flush=True,
        )
    return 32


def _initial_chunk(cfg: Mapping[str, Any], natoms: int, jobs_per_gpu: int) -> tuple[Any, int]:
    requested = cfg.get("chunk_size", 32)
    if isinstance(requested, str) and requested.strip().lower() == "auto":
        return requested, choose_auto_chunk(natoms, jobs_per_gpu)
    value = int(requested)
    if value < 1:
        raise ValueError("[ourHessian] chunk_size must be 'auto' or an integer >=1")
    return requested, value


def compute_hessian_result(atoms, config_dict: Mapping[str, Any]) -> HessianResult:
    """Compute the active-space analytical Hessian with adaptive OOM retry."""
    global _PATCH_ACTIVE_DOFS, _PATCH_CHUNK, _PATCH_RESTRICT, _PATCH_LAST_PROBE

    cfg = config_dict.get("ourHessian", {}) or {}
    jobs_per_gpu = int(config_dict.get("Main", {}).get("jobs_per_gpu", 1))
    free_dofs, fixed_atoms = _free_cartesian_dofs(atoms)
    restrict = bool(cfg.get("restrict_fixed_atoms", True)) and bool(fixed_atoms)
    requested, initial = _initial_chunk(cfg, len(atoms), jobs_per_gpu)
    minimum = int(cfg.get("min_chunk_size", 1))
    if minimum < 1:
        raise ValueError("[ourHessian] min_chunk_size must be >=1")
    if initial < minimum:
        initial = minimum
    retry_enabled = bool(cfg.get("oom_retry", True))
    max_retries = int(cfg.get("max_oom_retries", 5))
    if max_retries < 0:
        raise ValueError("[ourHessian] max_oom_retries must be >=0")

    calc = _get_hessian_calc(config_dict)
    _PATCH_ACTIVE_DOFS = free_dofs.tolist()
    _PATCH_RESTRICT = restrict

    chunk = int(initial)
    retries = 0
    while True:
        _PATCH_CHUNK = chunk
        _PATCH_LAST_PROBE = False
        _cuda_cleanup(calc)
        try:
            import torch
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        _cuda_sync()
        started = time.perf_counter()
        try:
            raw = calc.get_property("hessian", atoms)
            _cuda_sync()
            seconds = time.perf_counter() - started
            matrix = _normalize_raw(raw, len(atoms))
            del raw
            # In restricted mode only the free/free block is a complete Hessian
            # object. Do not pretend the omitted fixed blocks were calculated.
            active_h = matrix[np.ix_(free_dofs, free_dofs)]
            tol = float(cfg.get("negative_eigenvalue_tolerance", 1.0e-6))
            order = analyze_order_space(
                active_h, atoms, free_dofs, fixed_atoms, tol
            )
            meta = dict(getattr(calc, "sm_hessian_metadata", {}) or {})
            meta.update(_PATCH_META)
            try:
                meta["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                meta["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            except Exception:
                pass
            return HessianResult(
                active_hessian=order["active_hessian"],
                order_hessian=order["order_hessian"],
                free_dofs=free_dofs,
                raw_active_eigenvalues=order["raw_active_eigenvalues"],
                eigenvalues=order["eigenvalues"],
                lowest_mode=order["lowest_mode"],
                raw_active_negative_modes=order["raw_active_negative_modes"],
                negative_modes=order["negative_modes"],
                order_projection=order["order_projection"],
                removed_translation_modes=order["removed_translation_modes"],
                order_dof_count=order["order_dof_count"],
                hessian_seconds=float(seconds),
                chunk_requested=requested,
                chunk_initial=int(initial),
                chunk_used=int(chunk),
                oom_retries=int(retries),
                restricted=bool(restrict),
                has_constraints=bool(order["has_constraints"]),
                contract_probe_performed=bool(_PATCH_LAST_PROBE),
                metadata={**meta,
                          "translation_basis_active": order["translation_basis_active"],
                          "is_periodic": bool(order["is_periodic"])},
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            if not (_is_cuda_oom(exc) and retry_enabled and retries < max_retries and chunk > minimum):
                raise
            next_chunk = max(minimum, chunk // 2)
            if next_chunk >= chunk:
                raise
            retries += 1
            print(
                "Hessian CUDA OOM; retrying same structure with smaller chunk: "
                f"chunk={chunk} -> {next_chunk}, retry={retries}/{max_retries}, "
                f"failed_after={elapsed:.2f}s",
                flush=True,
            )
            _cuda_cleanup(calc)
            chunk = next_chunk


def hessian_job(
    i,
    config_dict,
    atoms,
    calc,
    Optimizer,
    consecutive_errors=None,
    executorlib_worker_id=None,
    **kwargs,
):
    """SaddleMill method entry point: one input structure -> one Hessian result."""
    del calc, Optimizer  # The standalone Hessian owns a dedicated cached predictor.
    rank = executorlib_worker_id
    max_errors = int(config_dict["Main"].get("max_consecutive_errors", 5))
    if consecutive_errors is not None and consecutive_errors[0] >= max_errors > 0:
        from saddlemill.tools import backup_flux_logs
        backup_flux_logs(rank)
        raise SystemExit(1)

    continuation = kwargs.get("continuation_data")
    if continuation is not None and bool(config_dict["Main"].get("continue_from_result", True)):
        atoms = continuation

    status_path = f"Hessian_status_csvs/status_rank_{rank}.csv"
    summary_path = f"Hessian_summary_csvs/hessian_{i}.csv"
    os.makedirs("Hessian_summary_csvs", exist_ok=True)
    if bool(config_dict.get("ourHessian", {}).get("store_hessian", True)):
        os.makedirs("Hessian_hessians", exist_ok=True)

    def log_status(msg: str) -> None:
        with open(status_path, "a") as handle:
            handle.write(f'{i},{rank},"{msg}"\n')

    # Source metadata is normally nested by load_and_sanitize. Flatten it back
    # onto the Hessian output so the next SaddleMill stage receives the same
    # reaction/attempt metadata in its orig_info.
    source_info = atoms.info.get("orig_info", {})
    if not isinstance(source_info, dict) or not source_info:
        source_info = dict(atoms.info)
    # A continuation of a previous Hessian output has src_index rewritten to
    # this Hessian job index. Preserve the original upstream lineage so the task
    # identity is stable across continue_from_result.
    parent_source_idx = source_info.get(
        "hessian_parent_source_idx", source_info.get("src_index", "")
    )
    parent_attempt_id = source_info.get(
        "hessian_parent_attempt_id", source_info.get("attempt_id", "")
    )
    identity_source_info = {
        "src_index": parent_source_idx,
        "attempt_id": parent_attempt_id,
    }
    artifact_store = None

    try:
        artifact_store = HessianArtifactStore(
            i,
            rank,
            atoms,
            config_dict,
            identity_source_info,
            summary_fields=SUMMARY_FIELDS,
            fairchem_contract_version=SUPPORTED_FAIRCHEM_VERSION,
        )
        with artifact_store.locked():
            if artifact_store.recover_or_reuse():
                print(
                    f"Rank {rank}: Hessian structure {i} reused validated artifact "
                    f"identity={artifact_store.task_identity[:16]}",
                    flush=True,
                )
                if consecutive_errors is not None:
                    consecutive_errors[0] = 0
                return

            result = compute_hessian_result(atoms, config_dict)
            tol = float(config_dict.get("ourHessian", {}).get(
                "negative_eigenvalue_tolerance", 1.0e-6
            ))
            first_order = result.negative_modes == 1
            status = (
                "converged_first_order" if first_order
                else f"converged_order_{result.negative_modes}"
            )
            lowest = float(result.eigenvalues[0])
            second = float(result.eigenvalues[1]) if result.eigenvalues.size > 1 else float("nan")

            stored = ""
            payload = None
            if artifact_store.store_hessian:
                stored = artifact_store.hessian_path
                payload = {
                    "active_hessian_eV_A2": result.active_hessian,
                    "order_hessian_eV_A2": result.order_hessian,
                    "free_dofs": result.free_dofs,
                    "raw_active_eigenvalues": result.raw_active_eigenvalues,
                    "eigenvalues": result.eigenvalues,
                    "order_eigenvalues": result.eigenvalues,
                    "lowest_eigenmode": result.lowest_mode,
                    "negative_eigenvalue_tolerance": np.asarray(tol),
                    "raw_active_negative_mode_count": np.asarray(result.raw_active_negative_modes),
                    "negative_mode_count": np.asarray(result.negative_modes),
                    "order_projection": np.asarray(result.order_projection),
                    "removed_translation_modes": np.asarray(result.removed_translation_modes),
                    "order_dof_count": np.asarray(result.order_dof_count),
                    "translation_basis_active": np.asarray(
                        result.metadata.get("translation_basis_active", np.empty((len(result.free_dofs), 0)))
                    ),
                    "restricted": np.asarray(int(result.restricted)),
                    "chunk_used": np.asarray(result.chunk_used),
                    # Additive publication metadata; old NPZ readers ignore it.
                    "hessian_artifact_contract": np.asarray(ARTIFACT_CONTRACT),
                    "hessian_task_identity": np.asarray(artifact_store.task_identity),
                    "hessian_artifact_validation": np.asarray("validated_reusable"),
                }
                if not result.restricted:
                    payload["hessian_cartesian_eV_A2"] = result.active_hessian

            free_atoms = len(result.free_dofs) // 3
            row = {
                "src_index": i,
                "rank": rank,
                "parent_source_idx": parent_source_idx,
                "parent_attempt_id": parent_attempt_id,
                "natoms": len(atoms),
                "free_atoms": free_atoms,
                "fixed_atoms": len(atoms) - free_atoms,
                "free_dof_count": len(result.free_dofs),
                "pbc": " ".join(str(int(x)) for x in np.asarray(atoms.pbc, dtype=bool)),
                "has_constraints": int(result.has_constraints),
                "restricted": int(result.restricted),
                "order_projection": result.order_projection,
                "removed_translation_modes": result.removed_translation_modes,
                "order_dof_count": result.order_dof_count,
                "raw_active_negative_mode_count": result.raw_active_negative_modes,
                "chunk_requested": result.chunk_requested,
                "chunk_initial": result.chunk_initial,
                "chunk_used": result.chunk_used,
                "oom_retries": result.oom_retries,
                "contract_probe_performed": int(result.contract_probe_performed),
                "hessian_seconds": result.hessian_seconds,
                "negative_eigenvalue_tolerance": tol,
                "negative_mode_count": result.negative_modes,
                "is_first_order": int(first_order),
                "lowest_eigenvalue": lowest,
                "second_eigenvalue": second,
                "hessian_file": stored,
                "task_name": result.metadata.get("task_name", ""),
                "model_name_or_path": result.metadata.get("model_name_or_path", ""),
                "fairchem_core_version": result.metadata.get("fairchem_core_version", ""),
                "jobs_per_gpu": int(config_dict["Main"].get("jobs_per_gpu", 1)),
                "status": status,
            }

            out = atoms.copy()
            out.calc = None
            out.info = dict(source_info)
            out.info.update({
                "src_index": i,
                "status": status,
                "converged": 1,
                "hessian_job_schema": "standalone_hessian_v2",
                "eigenmode": result.lowest_mode,
                "curvature": lowest,
                "hessian_parent_source_idx": parent_source_idx,
                "hessian_parent_attempt_id": parent_attempt_id,
                "hessian_order": result.negative_modes,
                "hessian_negative_modes": result.negative_modes,
                "hessian_raw_active_negative_modes": result.raw_active_negative_modes,
                "hessian_is_first_order": int(first_order),
                "hessian_order_projection": result.order_projection,
                "hessian_removed_translation_modes": result.removed_translation_modes,
                "hessian_order_dof_count": result.order_dof_count,
                "hessian_negative_eigenvalue_tolerance": tol,
                "hessian_seconds": result.hessian_seconds,
                "hessian_restricted": int(result.restricted),
                "hessian_free_dof_count": int(len(result.free_dofs)),
                "hessian_chunk_initial": result.chunk_initial,
                "hessian_chunk_used": result.chunk_used,
                "hessian_oom_retries": result.oom_retries,
                "hessian_contract_probe_performed": int(result.contract_probe_performed),
                "hessian_file": stored,
                "hessian_task_name": result.metadata.get("task_name", ""),
                "hessian_model": result.metadata.get("model_name_or_path", ""),
                "hessian_fairchem_version": result.metadata.get("fairchem_core_version", ""),
                # Additive commit metadata; standalone_hessian_v2 remains unchanged.
                "hessian_artifact_contract": ARTIFACT_CONTRACT,
                "hessian_task_identity": artifact_store.task_identity,
                "hessian_artifact_validation": "validated_reusable",
                "hessian_commit_file": artifact_store.commit_path,
            })
            artifact_store.publish(
                summary_row=row,
                output_atoms=out,
                npz_payload=payload,
                status=status,
            )

        print(
            f"Rank {rank}: Hessian structure {i} status={status} "
            f"N={len(atoms)} free={free_atoms} restricted={int(result.restricted)} "
            f"projection={result.order_projection} order={result.negative_modes} "
            f"chunk={result.chunk_used} retries={result.oom_retries} "
            f"seconds={result.hessian_seconds:.3f}",
            flush=True,
        )
        if consecutive_errors is not None:
            consecutive_errors[0] = 0
    except HessianArtifactIdentityMismatch as exc:
        print(
            f"Rank {rank} FAILED Hessian structure {i}: {exc}\n"
            + traceback.format_exc(),
            flush=True,
        )
        if consecutive_errors is not None:
            consecutive_errors[0] += 1
    except HessianArtifactPublicationError as exc:
        if artifact_store is not None:
            try:
                artifact_store.record_failure(exc, kind="publication_interrupted")
            except Exception:
                pass
        print(
            f"Rank {rank} FAILED Hessian structure {i}: {exc}\n"
            + traceback.format_exc(),
            flush=True,
        )
        if consecutive_errors is not None:
            consecutive_errors[0] += 1
    except Exception as exc:
        row = {
            "src_index": i,
            "rank": rank,
            "parent_source_idx": parent_source_idx,
            "parent_attempt_id": parent_attempt_id,
            "natoms": len(atoms),
            "jobs_per_gpu": int(config_dict["Main"].get("jobs_per_gpu", 1)),
            "status": f"error: {exc}",
        }
        try:
            _write_summary(summary_path, row)
        except Exception:
            pass
        print(
            f"Rank {rank} FAILED Hessian structure {i}: {exc}\n"
            + traceback.format_exc(),
            flush=True,
        )
        try:
            log_status(f"error: {exc}")
        except Exception:
            pass
        if consecutive_errors is not None:
            consecutive_errors[0] += 1
