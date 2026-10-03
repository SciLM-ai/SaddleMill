"""Small state containers and timing helpers for the Dimer attempt lifecycle.

This module is intentionally orchestration-only.  It does not select an optimizer,
change a scientific setting, evaluate forces, or classify convergence.  The search
science remains in :mod:`saddlemill.dimeropt`; these helpers make each attempt's
mutable state explicit so setup/execution and recording can be tested separately.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Optional


@dataclass
class DimerRunContext:
    """Per-structure state shared by all Dimer attempts."""

    src_index: int
    rank: Any
    config_dict: dict
    calc: Any
    method_name: str
    task_name: Any
    saddle_engine: str
    is_vasp: bool
    status_file: str
    reaction_file: str
    rxn_file: str
    mode_file: str
    optimizer_file: str
    qn_file: str
    timing_file: str
    metrics_file: str
    output_file: str
    zip_name: str
    dimer_method_cfg: dict
    minmode_options: dict
    sella_options: Any
    sella_hessian_calc: Any
    rotation_lbfgs_options: dict
    translation_lbfgs_options: dict
    hybrid_options: dict
    wave_b_options: dict = field(default_factory=dict)
    initial_hessian_matrix: Any = None
    rng_factory: Any = None
    rng_file: Optional[str] = None


@dataclass
class DimerAttemptContext:
    """All mutable state belonging to one Dimer attempt."""

    run: DimerRunContext
    attempt_id: int
    atoms: Any
    displacement_dict: Any
    selected_index: Any
    configured_reaction_type: str
    initial_reaction_type: str
    reaction_source: str
    attempt_start_perf_ns: int
    attempt_start_unix_ns: int

    optimizer_elapsed_ns: int = 0
    archive_elapsed_ns: int = 0
    temp_files: list[str] = field(default_factory=list)
    attempt_vasp_dir: Optional[str] = None
    attempt_calc: Any = None
    d_atoms: Any = None
    dim_rlx: Any = None
    optimizer_recorder: Any = None
    free_indices: list[int] = field(default_factory=list)
    eigenmode: Any = None
    curvature: Any = None
    energy: Any = None
    forces: Any = None
    n_force_calls: int = 0
    converged: bool = False
    status: Optional[str] = None
    stop_reason: Optional[str] = None
    stopped_early: bool = False
    sella_eigenvalues: Any = None
    sella_negative_modes: Any = None
    sella_stationary_converged: Any = None
    sella_ablation_session: Any = None
    sella_linalg_failure_recorder: Any = None
    sella_passive_qn_recorder: Any = None
    sella_ablation_summary: Any = None
    sella_ablation_state: Any = None
    sella_ablation_rows: list = field(default_factory=list)
    attempt_rng: Any = None
    realized_initial_geometry_sha256: Optional[str] = None


def reaction_type_from_atoms(atoms, fallback):
    """Read the current reaction label using SaddleMill's top-level-first rule."""
    return atoms.info.get(
        "reaction_type",
        atoms.info.get("orig_info", {}).get("reaction_type", fallback),
    )


def start_attempt(run, attempt_id, atoms, displacement_dict, selected_index,
                  configured_reaction_type, attempt_start_perf_ns=None,
                  attempt_start_unix_ns=None, attempt_rng=None):
    """Create the explicit mutable state for an active attempt."""
    if attempt_start_perf_ns is None:
        attempt_start_perf_ns = time.perf_counter_ns()
    if attempt_start_unix_ns is None:
        attempt_start_unix_ns = time.time_ns()
    return DimerAttemptContext(
        run=run,
        attempt_id=attempt_id,
        atoms=atoms,
        displacement_dict=displacement_dict,
        selected_index=selected_index,
        configured_reaction_type=configured_reaction_type,
        initial_reaction_type=configured_reaction_type,
        reaction_source="configured_attempt_order",
        attempt_start_perf_ns=attempt_start_perf_ns,
        attempt_start_unix_ns=attempt_start_unix_ns,
        attempt_rng=attempt_rng,
    )


def apply_generated_metadata(context):
    """Capture the generator's initialization mechanism for a non-null attempt."""
    context.initial_reaction_type = reaction_type_from_atoms(
        context.atoms, context.configured_reaction_type
    )
    context.reaction_source = "generated_attempt_metadata"


def apply_continuation(context, continuation_atoms, displacement_dict):
    """Apply an already-generated continuation without changing attempt identity."""
    context.atoms = continuation_atoms
    context.displacement_dict = displacement_dict
    continuation_reaction_type = reaction_type_from_atoms(
        context.atoms, context.initial_reaction_type
    )
    # A previous desorption label is an outcome, not the attempt's initialization
    # mechanism.  Retain the freshly generated mechanism in that case.
    if continuation_reaction_type != "desorption":
        context.initial_reaction_type = continuation_reaction_type
    context.reaction_source = "continuation_metadata"


def configure_temp_files(context):
    """Populate the historical per-attempt temporary filenames verbatim."""
    run = context.run
    i = run.src_index
    attempt = context.attempt_id
    selected = context.selected_index
    temp_log = f"dimer_control_{i}_{attempt}_{selected}.log"
    if run.saddle_engine == "sella":
        temp_opt_log = f"dimer_sella_opt_{i}_{attempt}_{selected}.log"
        temp_traj = f"dimer_sella_{i}_{attempt}_{selected}.traj"
    else:
        temp_opt_log = f"dimer_opt_{i}_{attempt}_{selected}.log"
        temp_traj = f"dimer_{i}_{attempt}_{selected}.traj"
    temp_mode_log = f"dimer_mode_{i}_{attempt}_{selected}.log"
    context.temp_files = [temp_log, temp_opt_log, temp_traj, temp_mode_log]
    if run.is_vasp:
        context.attempt_vasp_dir = f"VASP_{i}_{attempt}"
        context.temp_files.append(context.attempt_vasp_dir)
    return temp_log, temp_opt_log, temp_traj, temp_mode_log


def run_optimizer_timed(context, *args, **kwargs):
    """Run the selected optimizer while preserving the existing timing boundary."""
    started_ns = time.perf_counter_ns()
    try:
        attempt_rng = getattr(context, "attempt_rng", None)
        if attempt_rng is None:
            return context.dim_rlx.run(*args, **kwargs)
        # External optimizers/mode finders that still consult module-global RNG
        # state are isolated from every other logical attempt. SaddleMill-owned
        # stochastic paths use named local substreams instead.
        with attempt_rng.global_compatibility("external_optimizer_runtime"):
            return context.dim_rlx.run(*args, **kwargs)
    finally:
        context.optimizer_elapsed_ns += time.perf_counter_ns() - started_ns


def archive_temp_files_timed(context, archive_func, zip_name, prefix, enabled):
    """Archive per-attempt scratch while preserving the existing timing boundary."""
    started_ns = time.perf_counter_ns()
    try:
        return archive_func(
            context.temp_files, zip_name, prefix=prefix, enabled=enabled
        )
    finally:
        context.archive_elapsed_ns += time.perf_counter_ns() - started_ns
