"""Sella adapter for SaddleMill's existing Dimer attempt runner.

Sella is a distinct saddle-search algorithm, but SaddleMill dispatches it from
``dimeropt.py`` so it can reuse the established attempt generation, resume
identity, status CSV, output trajectory, VASP lifecycle, and error handling.
The legacy ASE and Kappa dimer paths do not import this module unless
``[ourDimer] engine = sella`` is selected.
"""

from __future__ import annotations

import functools
import inspect
import math
import os
from importlib import metadata
import time
from typing import Any, Mapping

import numpy as np

SUPPORTED_SELLA_VERSION = "2.5.0"
SUPPORTED_FAIRCHEM_VERSION = "2.20.0"
DEFAULT_HESSIAN_ENGINE = "sella"
DIRECT_HESSIAN_ENGINE = "fairchem_direct"

_DIRECT_HESSIAN_CALCULATOR = None
_DIRECT_HESSIAN_CONFIG_KEY = None
_DIRECT_HESSIAN_PROCESS_ID = None


def validate_sella_environment(required_version: str = SUPPORTED_SELLA_VERSION):
    """Return ``sella.Sella`` after an exact, tested-version check."""
    try:
        version = metadata.version("sella")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "[ourDimer] engine=sella requires sella=="
            f"{required_version} in the authoritative SaddleMill environment."
        ) from exc

    if version != required_version:
        raise RuntimeError(
            "Unsupported Sella version: expected "
            f"{required_version}, found {version}."
        )

    from sella import Sella

    return Sella, version



def create_sella_ablation_session(config_dict, optimizer, metadata=None, restored_state=None):
    """Build the Sella-ablation attempt-scoped session; strict no-key/default means no hook."""
    cfg = dict(config_dict.get("ourSella", {}) or {})
    selector = cfg.get("ablation", None)
    if selector in (None, "", "none", "None"):
        return None
    from saddlemill.sella_ablation import (
        CENTER_ONLY_LEARNING, PARTITION_NATIVE_RITZ, QN_NEWTON_SAFE_FALSE,
        SellaAblationError, SellaAblationSession, resolve_sella_ablation,
    )
    spec = resolve_sella_ablation(selector)
    method = str(cfg.get("method", "prfo")).strip().lower()
    if spec.qn_newton_safe is False:
        if method != "qn":
            raise SellaAblationError("non_qn_baseline", f"{QN_NEWTON_SAFE_FALSE} requires [ourSella] method=qn")
    elif method != "prfo" and not spec.is_normal:
        raise SellaAblationError("non_prfo_baseline", "PRFO Sella ablations require [ourSella] method=prfo")
    if not bool(cfg.get("eig", True)):
        raise SellaAblationError("eig_required", "Sella ablations require [ourSella] eig=True")
    if bool(cfg.get("internal", False)):
        raise SellaAblationError("internal_coordinates_unsupported", "Sella ablations require Cartesian Sella internal=False")
    if float(cfg.get("delta0", 0.1)) != 0.1:
        raise SellaAblationError("delta0_mismatch", "Sella-ablation requires unchanged Sella delta0=0.1")
    if direct_hessian_requested(config_dict) and spec.learning == "center_only":
        raise SellaAblationError(
            "probe_learning_not_applicable_direct_hessian",
            "center_only_learning is undefined with [ourSella] hessian_engine=fairchem_direct",
        )
    if direct_hessian_requested(config_dict) and spec.partition == PARTITION_NATIVE_RITZ:
        raise SellaAblationError(
            "ritz_partition_requires_native_diag",
            "ritzmode_prfo_partition requires [ourSella] hessian_engine=sella so native PES.diag produces the Ritz mode",
        )
    session = SellaAblationSession(
        optimizer, spec, metadata=metadata, restored_state=restored_state
    )
    # Fail closed at the public boundary, not only when used as a context manager.
    session._validate_optimizer()
    return session

def normalize_hessian_engine(value: Any) -> str:
    """Return the canonical ``[ourSella] hessian_engine`` value."""
    normalized = str(value or DEFAULT_HESSIAN_ENGINE).strip().lower()
    aliases = {
        "default": DEFAULT_HESSIAN_ENGINE,
        "stock": DEFAULT_HESSIAN_ENGINE,
        "none": DEFAULT_HESSIAN_ENGINE,
        "fairchem": DIRECT_HESSIAN_ENGINE,
        "direct": DIRECT_HESSIAN_ENGINE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {DEFAULT_HESSIAN_ENGINE, DIRECT_HESSIAN_ENGINE}:
        raise ValueError(
            "[ourSella] hessian_engine must be 'sella' or "
            f"'fairchem_direct'; got {value!r}."
        )
    return normalized


def direct_hessian_requested(config_dict: Mapping[str, Any]) -> bool:
    """Return whether this configuration selects the FairChem Hessian path."""
    cfg = config_dict.get("ourSella", {}) or {}
    return (
        normalize_hessian_engine(cfg.get("hessian_engine"))
        == DIRECT_HESSIAN_ENGINE
    )


def validate_fairchem_hessian_config(config_dict: Mapping[str, Any]) -> None:
    """Validate the shared FairChem full-Hessian predictor configuration."""
    main_cfg = config_dict.get("Main", {}) or {}
    calc_cfg = config_dict.get("FAIRChemCalculator", {}) or {}
    if main_cfg.get("Calculator") != "FAIRChemCalculator":
        raise ValueError("FairChem Hessian inference requires [Main] Calculator=FAIRChemCalculator.")
    if int(calc_cfg.get("workers", 1)) != 1:
        raise ValueError("FairChem Hessian inference requires [FAIRChemCalculator] workers=1.")
    model_name = str(calc_cfg.get("name_or_path", "")).strip()
    task_name = str(calc_cfg.get("task_name", "")).strip()
    device = str(calc_cfg.get("device", main_cfg.get("device", ""))).strip().lower()
    if not model_name:
        raise ValueError("[FAIRChemCalculator] name_or_path is required for FairChem Hessian inference.")
    if not task_name:
        raise ValueError("[FAIRChemCalculator] task_name is required for FairChem Hessian inference.")
    if not device.startswith("cuda"):
        raise ValueError(f"FairChem Hessian inference is validated on CUDA; got device={device!r}.")
    try:
        version = metadata.version("fairchem-core")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(f"FairChem Hessian inference requires fairchem-core=={SUPPORTED_FAIRCHEM_VERSION}.") from exc
    if version != SUPPORTED_FAIRCHEM_VERSION:
        raise RuntimeError(
            "Unsupported fairchem-core version for direct Hessians: expected "
            f"{SUPPORTED_FAIRCHEM_VERSION}, found {version}."
        )


def validate_direct_hessian_config(config_dict: Mapping[str, Any]) -> None:
    """Fail fast on unsupported direct-Hessian Sella combinations."""
    if not direct_hessian_requested(config_dict):
        return
    validate_fairchem_hessian_config(config_dict)
    sella_cfg = config_dict.get("ourSella", {}) or {}
    if bool(sella_cfg.get("internal", False)):
        raise ValueError(
            "[ourSella] hessian_engine=fairchem_direct is validated only "
            "with internal=False (Cartesian Sella)."
        )


def install_inference_hessian_memory_fix() -> dict[str, Any]:
    """Install the process-local runtime equivalent of FairChem PR #2100."""
    import fairchem.core.models.uma.escn_md as escn_md

    current = escn_md.compute_hessian
    if getattr(current, "_saddlemill_inference_hessian_fix", False):
        return {
            "memory_fix": "FairChem_PR_2100_runtime_equivalent",
            "memory_fix_already_installed": True,
        }

    signature = inspect.signature(current)
    if "training" not in signature.parameters:
        raise RuntimeError(
            "Cannot safely install the FairChem inference-Hessian memory "
            f"correction; compute_hessian signature is {signature}."
        )

    @functools.wraps(current)
    def inference_compute_hessian(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        bound.arguments["training"] = False
        return current(*bound.args, **bound.kwargs)

    inference_compute_hessian._saddlemill_inference_hessian_fix = True
    escn_md.compute_hessian = inference_compute_hessian
    return {
        "memory_fix": "FairChem_PR_2100_runtime_equivalent",
        "memory_fix_already_installed": False,
        "compute_hessian_signature": str(signature),
    }


def _direct_hessian_config_key(config_dict: Mapping[str, Any]) -> tuple:
    """Build a stable in-process key for the Hessian predictor configuration."""
    calc_cfg = dict(config_dict.get("FAIRChemCalculator", {}) or {})
    # repr is used only to detect an unexpected config change inside one worker;
    # the actual values are still passed unchanged to FairChem.
    items = tuple(sorted((str(key), repr(value)) for key, value in calc_cfg.items()))
    return items


def get_cached_fairchem_hessian_calculator(config_dict: Mapping[str, Any]):
    """Return one Hessian-enabled UMA calculator per worker process.

    SaddleMill's normal force calculator remains owned by ``init_function``.
    This second calculator is created lazily on the first direct-Hessian Dimer
    task and then reused sequentially by later attempts and structures handled
    by the same worker.
    """
    global _DIRECT_HESSIAN_CALCULATOR
    global _DIRECT_HESSIAN_CONFIG_KEY
    global _DIRECT_HESSIAN_PROCESS_ID

    validate_fairchem_hessian_config(config_dict)
    process_id = os.getpid()
    config_key = _direct_hessian_config_key(config_dict)

    if _DIRECT_HESSIAN_PROCESS_ID != process_id:
        # Never retain a CUDA calculator inherited across a process boundary.
        _DIRECT_HESSIAN_CALCULATOR = None
        _DIRECT_HESSIAN_CONFIG_KEY = None
        _DIRECT_HESSIAN_PROCESS_ID = process_id

    if _DIRECT_HESSIAN_CALCULATOR is not None:
        if _DIRECT_HESSIAN_CONFIG_KEY != config_key:
            raise RuntimeError(
                "This worker already cached a FairChem Hessian predictor with "
                "different [FAIRChemCalculator] settings."
            )
        return _DIRECT_HESSIAN_CALCULATOR

    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import InferenceSettings

    memory_meta = install_inference_hessian_memory_fix()
    calc_kwargs = dict(config_dict.get("FAIRChemCalculator", {}) or {})
    task_name = str(calc_kwargs["task_name"])
    calc_kwargs["inference_settings"] = InferenceSettings(
        predict_untrained_hessian={task_name},
        hessian_vmap=False,
    )
    calc = FAIRChemCalculator.from_model_checkpoint(**calc_kwargs)

    if "hessian" not in calc.implemented_properties:
        raise RuntimeError(
            "The selected UMA model/task did not expose a Hessian; "
            f"task_name={task_name!r}, "
            f"properties={calc.implemented_properties}."
        )

    backbone = calc.predictor.model.module.backbone
    if bool(backbone.training):
        raise RuntimeError("UMA Hessian backbone is unexpectedly in training mode.")
    regress = backbone.regress_config
    if not hasattr(regress, "hessian_vmap"):
        raise RuntimeError(
            "Installed FairChem regress_config has no hessian_vmap attribute."
        )
    # FairChem 2.20.0 can ignore the public setting in some construction paths.
    regress.hessian_vmap = False
    if bool(regress.hessian_vmap):
        raise RuntimeError("Failed to enforce hessian_vmap=False.")

    calc.sm_hessian_metadata = {
        "engine": DIRECT_HESSIAN_ENGINE,
        "fairchem_core_version": SUPPORTED_FAIRCHEM_VERSION,
        "model_name_or_path": str(calc_kwargs["name_or_path"]),
        "task_name": task_name,
        "device": str(calc_kwargs.get("device", "cuda")),
        "hessian_vmap": False,
        **memory_meta,
    }
    _DIRECT_HESSIAN_CALCULATOR = calc
    _DIRECT_HESSIAN_CONFIG_KEY = config_key
    return calc


def get_cached_direct_hessian_calculator(config_dict: Mapping[str, Any]):
    """Backward-compatible Sella entrypoint for the shared Hessian predictor."""
    validate_direct_hessian_config(config_dict)
    return get_cached_fairchem_hessian_calculator(config_dict)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def normalize_full_hessian(raw: Any, n_atoms: int) -> np.ndarray:
    """Normalize FairChem output to a finite symmetric ``(3N, 3N)`` array."""
    arr = _to_numpy(raw)
    target = 3 * int(n_atoms)
    while arr.ndim > 2 and arr.shape[0] == 1 and arr.size == target * target:
        arr = arr[0]
    if arr.shape == (n_atoms, 3, n_atoms, 3):
        arr = arr.reshape(target, target)
    elif arr.shape != (target, target) and arr.size == target * target:
        arr = arr.reshape(target, target)
    if arr.shape != (target, target):
        raise RuntimeError(
            f"FairChem Hessian shape is {arr.shape}; expected "
            f"{(target, target)} or an equivalent block shape."
        )
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("FairChem direct Hessian contains non-finite values.")
    return 0.5 * (arr + arr.T)


def get_direct_hessian(calc: Any, atoms: Any) -> np.ndarray:
    """Request one full Cartesian Hessian through ASE's property API."""
    calc.reset()
    raw = calc.get_property("hessian", atoms)
    if "hessian" not in calc.results:
        raise RuntimeError(
            "FairChem completed but calculator.results has no Hessian; "
            f"keys={sorted(calc.results)}."
        )
    return normalize_full_hessian(raw, len(atoms))


def _cuda_synchronize_if_available() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


class FairChemDirectHessianCallback:
    """Sella callback carrying per-attempt Hessian timing and call counts."""

    def __init__(self, calc: Any):
        self.calc = calc
        self.calls = 0
        self.total_seconds = 0.0
        self.last_shape: tuple[int, int] | None = None

    def __call__(self, atoms: Any) -> np.ndarray:
        _cuda_synchronize_if_available()
        started = time.perf_counter()
        hessian = get_direct_hessian(self.calc, atoms)
        _cuda_synchronize_if_available()
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed):
            raise RuntimeError("Non-finite FairChem Hessian timing value.")
        self.calls += 1
        self.total_seconds += float(elapsed)
        self.last_shape = tuple(hessian.shape)
        return hessian


def apply_attempt_displacement(
    atoms,
    displacement_dict: Mapping[str, Any] | None,
    dimer_control_kwargs: Mapping[str, Any] | None = None,
    attempt_rng=None,
):
    """Apply the exact ASE ``MinModeAtoms.displace`` semantics in-place.

    ``structure_edit.get_attempts`` returns a displacement dictionary designed
    for ``MinModeAtoms.displace``. Reusing that function avoids introducing a
    second interpretation of masks, displacement radii, selected atoms, or
    Gaussian vectors. Constructing the wrapper and displacing does not evaluate
    the calculator.
    """
    from saddlemill.dimertools.kappa_dimer import IsolatedDimerControl

    control = IsolatedDimerControl(
        logfile=None,
        eigenmode_logfile=None,
        **dict(dimer_control_kwargs or {}),
    )
    from ase.mep import MinModeAtoms
    random_seed = None
    if attempt_rng is not None:
        random_seed = int(
            attempt_rng.substream_provenance(
                "ase_candidate_displacement"
            )["legacy_numpy_seed_uint32"]
        )
    wrapped = MinModeAtoms(
        atoms,
        control=control,
        **({} if random_seed is None else {"random_seed": random_seed}),
    )
    if displacement_dict:
        if attempt_rng is None:
            wrapped.displace(log=False, **dict(displacement_dict))
        else:
            with attempt_rng.global_compatibility("ase_candidate_displacement"):
                wrapped.displace(log=False, **dict(displacement_dict))
    else:
        if attempt_rng is None:
            tiny = np.random.randn(len(atoms), 3) * 1.0e-10
        else:
            tiny = (
                attempt_rng.numpy("candidate_tiny_displacement")
                .standard_normal((len(atoms), 3)) * 1.0e-10
            )
        wrapped.displace(
            displacement_vector=tiny,
            method="vector",
            log=False,
        )
    return atoms


def _cartesian_mode_to_pes_coordinates(pes, mode_flat: np.ndarray) -> np.ndarray:
    """Map a real-atom Cartesian direction into the active PES coordinates."""
    if getattr(pes, "int", None) is None:
        return mode_flat

    # InternalPES may include dummy atoms. Give those zero Cartesian motion,
    # then map Cartesian displacement to redundant internal displacement.
    jacobian = np.asarray(pes.int.jacobian(), dtype=float)
    if jacobian.ndim != 2:
        raise RuntimeError("Sella internal-coordinate Jacobian is not 2-D.")
    if jacobian.shape[1] < mode_flat.size:
        raise RuntimeError(
            "Sella internal-coordinate Jacobian has fewer Cartesian columns "
            "than the supplied real-atom eigenmode."
        )
    padded = np.zeros(jacobian.shape[1], dtype=float)
    padded[:mode_flat.size] = mode_flat
    return jacobian @ padded


def _pes_mode_to_cartesian(pes, pes_mode: np.ndarray) -> np.ndarray:
    """Map an active PES-coordinate direction back to real-atom Cartesian form."""
    if getattr(pes, "int", None) is None:
        return np.asarray(pes_mode, dtype=float)

    jacobian = np.asarray(pes.int.jacobian(), dtype=float)
    # Minimum-norm Cartesian displacement satisfying B dx ~= dq.
    cart_all = np.linalg.pinv(jacobian, rcond=1.0e-10) @ np.asarray(
        pes_mode, dtype=float
    )
    return cart_all[: 3 * len(pes.atoms)]


def _project_input_mode(optimizer, eigenmode) -> bool:
    """Project an input Cartesian mode into Sella's constrained free subspace."""
    if eigenmode is None:
        return False

    mode = np.asarray(eigenmode, dtype=float)
    expected = (len(optimizer.pes.atoms), 3)
    if mode.shape != expected:
        raise ValueError(
            f"Sella input eigenmode must have shape {expected}, got {mode.shape}."
        )
    if not np.all(np.isfinite(mode)):
        raise ValueError("Sella input eigenmode contains non-finite values.")

    pes_mode = _cartesian_mode_to_pes_coordinates(
        optimizer.pes, mode.reshape(-1)
    )
    ufree = np.asarray(optimizer.pes.get_Ufree(), dtype=float)
    projected = ufree.T @ pes_mode
    norm = float(np.linalg.norm(projected))
    if norm < 1.0e-14:
        raise ValueError(
            "Sella input eigenmode has zero norm after applying constraints."
        )
    optimizer.pes.v0 = projected / norm
    return True


def setup_sella(
    atoms,
    calc,
    *,
    eigenmode=None,
    displacement_dict=None,
    dimer_control_kwargs=None,
    logfile=None,
    trajectory=None,
    sella_options=None,
    hessian_calc=None,
    attempt_rng=None,
):
    """Apply one attempt displacement and construct a first-order Sella run."""
    Sella, version = validate_sella_environment()

    # Match the legacy dimer setup order: attach the calculator before
    # constructing the displacement wrapper. The displacement itself performs
    # no force evaluation, but this avoids relying on undocumented constructor
    # behavior in ASE's MinModeAtoms.
    atoms.calc = calc
    apply_attempt_displacement(
        atoms,
        displacement_dict,
        dimer_control_kwargs=dimer_control_kwargs,
        attempt_rng=attempt_rng,
    )

    options = dict(sella_options or {})
    # Sella 2.5.0 exposes Rayleigh-Ritz maxiter on PES.diag(), not the constructor.
    # Remove it before constructor dispatch and install it into diagkwargs below.
    maxiter = options.pop("maxiter", None)
    if maxiter is not None:
        maxiter = int(maxiter)
        if maxiter < 1:
            raise ValueError("Sella maxiter must be >= 1 or None")
    # SaddleMill's Sella engine is explicitly first-order. A caller cannot
    # silently change that scientific target through a forwarded option.
    options.pop("order", None)
    # The SaddleMill engine owns the Hessian source; it cannot be replaced by
    # an unrelated forwarded Sella option.
    options.pop("hessian_function", None)
    hessian_callback = None
    hessian_engine = DEFAULT_HESSIAN_ENGINE
    if hessian_calc is not None:
        hessian_callback = FairChemDirectHessianCallback(hessian_calc)
        hessian_engine = DIRECT_HESSIAN_ENGINE
    # Sella's native trajectory is a PES-evaluation trajectory: PES.eval()
    # writes once for every energy/gradient evaluation, including finite-
    # difference Hessian/eigensolver probes.  That can create thousands of
    # frames for one optimization.  Keep the same debug filename, but make it
    # an ASE optimizer-state trajectory instead: one observer write per Sella
    # optimizer iteration.  Passing trajectory=None prevents PES.eval() from
    # writing.  ASE TrajectoryWriter uses cached calculator results with
    # allow_calculation=False when properties are not explicitly requested, so
    # these observer writes do not add force evaluations.
    if attempt_rng is None:
        optimizer = Sella(
            atoms,
            order=1,
            logfile=logfile,
            trajectory=None,
            hessian_function=hessian_callback,
            **options,
        )
    else:
        with attempt_rng.global_compatibility("external_sella_setup"):
            optimizer = Sella(
                atoms,
                order=1,
                logfile=logfile,
                trajectory=None,
                hessian_function=hessian_callback,
                **options,
            )
    step_trajectory = None
    if trajectory is not None:
        from ase.io.trajectory import Trajectory

        if isinstance(trajectory, (str, os.PathLike)):
            step_trajectory = Trajectory(
                trajectory, mode="w", atoms=atoms
            )
        else:
            step_trajectory = trajectory
        optimizer.attach(step_trajectory.write, interval=1)
        optimizer.closelater(step_trajectory)
        # Do not assign step_trajectory to optimizer.trajectory here.
        # Sella was constructed with trajectory=None; ASE >=3.29 treats a
        # later non-None optimizer.trajectory as constructor-managed and
        # asserts that _orig_trajectory exists inside _traj_is_empty().
        optimizer.sm_debug_trajectory_mode = "optimizer_steps"
    else:
        optimizer.sm_debug_trajectory_mode = "disabled"

    if maxiter is not None:
        optimizer.diagkwargs["maxiter"] = maxiter

    used_input_mode = _project_input_mode(optimizer, eigenmode)
    optimizer.sm_sella_version = version
    optimizer.sm_sella_maxiter = maxiter
    optimizer.sm_used_input_mode = used_input_mode
    optimizer.sm_hessian_engine = hessian_engine
    optimizer.sm_hessian_callback = hessian_callback
    optimizer.sm_hessian_metadata = dict(
        getattr(hessian_calc, "sm_hessian_metadata", {}) or {}
    )
    return atoms, optimizer


def extract_lowest_mode(optimizer):
    """Return Sella's lowest model-Hessian mode and free-space spectrum.

    This is Sella's final *approximate constrained Hessian*, not an independent
    full finite-difference Hessian. The distinction is recorded in output
    metadata and documentation.
    """
    # Synchronize the PES cache to the final accepted geometry.
    optimizer.pes.get_g()
    ufree = np.asarray(optimizer.pes.get_Ufree(), dtype=float)
    if ufree.ndim != 2 or ufree.shape[1] == 0:
        raise RuntimeError("Sella has no unconstrained degrees of freedom.")

    hproj = np.asarray(
        optimizer.pes.get_HL_projected(ufree).asarray(), dtype=float
    )
    if hproj.shape != (ufree.shape[1], ufree.shape[1]):
        raise RuntimeError(
            "Unexpected Sella projected-Hessian shape: "
            f"{hproj.shape}, expected {(ufree.shape[1], ufree.shape[1])}."
        )
    if not np.all(np.isfinite(hproj)):
        raise RuntimeError("Sella final projected Hessian is non-finite.")

    hproj = 0.5 * (hproj + hproj.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hproj)
    if eigenvalues.size == 0:
        raise RuntimeError("Sella final model Hessian contains no eigenvalues.")

    pes_mode = ufree @ eigenvectors[:, 0]
    cart_mode = _pes_mode_to_cartesian(optimizer.pes, pes_mode)
    mode_norm = float(np.linalg.norm(cart_mode))
    if not np.isfinite(mode_norm) or mode_norm < 1.0e-14:
        raise RuntimeError("Sella final lowest Cartesian mode is non-finite or zero.")

    mode = (cart_mode / mode_norm).reshape((-1, 3))
    return mode, float(eigenvalues[0]), eigenvalues


def classify_sella_convergence(
    stationary_converged: bool,
    eigenvalues,
    *,
    negative_eigenvalue_tolerance: float = 1.0e-6,
    require_first_order_model: bool = True,
):
    """Classify a Sella result using its approximate constrained Hessian.

    Returns ``(converged, status, negative_mode_count)``. This helper is pure
    and independently unit-tested so a stationary point with the wrong model
    order cannot be mislabeled as converged.
    """
    tolerance = float(negative_eigenvalue_tolerance)
    if tolerance < 0.0:
        raise ValueError("negative_eigenvalue_tolerance must be >= 0")
    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("Sella model-Hessian eigenvalues are non-finite.")
    negative_modes = int(np.sum(values < -tolerance))
    order_ok = negative_modes == 1 if require_first_order_model else True
    converged = bool(stationary_converged and order_ok)
    if converged:
        status = "converged"
    elif stationary_converged and not order_ok:
        status = "not_converged_wrong_order"
    else:
        status = "not_converged"
    return converged, status, negative_modes


def sella_force_calls(optimizer) -> int:
    """Return Sella PES energy/gradient evaluation count."""
    return int(getattr(optimizer.pes, "neval", 0))


def sella_options_from_config(config_dict) -> dict[str, Any]:
    """Convert ``[ourSella]`` into Sella 2.5.0 constructor options."""
    cfg = config_dict.get("ourSella", {}) or {}
    options: dict[str, Any] = {
        "internal": bool(cfg.get("internal", False)),
        "eig": bool(cfg.get("eig", True)),
        "method": str(cfg.get("method", "prfo")),
        "delta0": float(cfg.get("delta0", 0.1)),
        "eta": float(cfg.get("eta", 1.0e-4)),
        "gamma": float(cfg.get("gamma", 0.1)),
        "threepoint": bool(cfg.get("threepoint", False)),
        "constraints_tol": float(cfg.get("constraints_tol", 1.0e-5)),
        "nsteps_per_diag": int(cfg.get("nsteps_per_diag", 3)),
        "allow_fragments": bool(cfg.get("allow_fragments", False)),
    }

    maxiter = cfg.get("maxiter", None)
    if maxiter not in (None, "", "None", "none"):
        maxiter = int(maxiter)
        if maxiter < 1:
            raise ValueError("[ourSella] maxiter must be >= 1 or None")
        options["maxiter"] = maxiter

    project_translations = cfg.get("project_translations", None)
    if project_translations not in (None, "", "None", "none"):
        if not isinstance(project_translations, bool):
            raise ValueError(
                "[ourSella] project_translations must be True, False, or blank"
            )
        options["proj_trans"] = project_translations
    project_rotations = cfg.get("project_rotations", None)
    if project_rotations not in (None, "", "None", "none"):
        if not isinstance(project_rotations, bool):
            raise ValueError(
                "[ourSella] project_rotations must be True, False, or blank"
            )
        options["proj_rot"] = project_rotations

    restricted_step = cfg.get("restricted_step", None)
    if restricted_step not in (None, "", "None", "none"):
        options["rs"] = str(restricted_step)

    diag_every_n = cfg.get("diag_every_n", None)
    if diag_every_n not in (None, "", "None", "none"):
        options["diag_every_n"] = int(diag_every_n)

    for key in ("sigma_inc", "sigma_dec", "rho_inc", "rho_dec"):
        value = cfg.get(key, None)
        if value not in (None, "", "None", "none"):
            options[key] = float(value)

    return options
