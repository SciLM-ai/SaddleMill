"""Pure translation-force policies for minimum-mode experiments.

mode-schedule owns only the policy algebra in this module.  It performs no calculator calls,
reads no global configuration, and does not modify the legacy Dimer/MMF runtime.
Shared selector/factory/runtime wiring belongs to shared-runtime.

Physical-force convention: ``F = -g``.  The Shang--Liu negative-curvature arm is
Eq. 15 of JCTC 2010, 6, 1136--1144 (journal page 1139): the lambda chosen from
the RMS parallel force at the *beginning* of a translation sequence is held for
that sequence.  The project contract defines that RMS over movable Cartesian
DOFs.  Positive-curvature paper prefactors are deliberately not folded into this
negative-curvature damping policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np

from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace, freeze_array

Array = np.ndarray
DampingMode = Literal["none", "fixed", "shang_liu"]


@dataclass(frozen=True)
class ParallelForceDampingConfig:
    """Resolved pure policy configuration.

    ``mode='none'`` preserves the ordinary negative-curvature reflected force
    ``F_perp - F_parallel``.  ``fixed`` uses one user lambda for the entire
    search.  ``shang_liu`` chooses Eq. 15 lambda at each real-solve/translation
    sequence boundary and the scheduler holds that selection until the next
    actual real solve.
    """

    mode: DampingMode | str = "none"
    fixed_lambda: float = 1.0

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"none", "fixed", "shang_liu"}:
            raise ValueError("parallel-force damping mode must be none, fixed, or shang_liu")
        value = float(self.fixed_lambda)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("fixed parallel-force lambda must be finite and >= 0")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "fixed_lambda", value)


@dataclass(frozen=True)
class ParallelForceComponents:
    raw_physical_force: Array
    normalized_mode: Array
    parallel_force: Array
    perpendicular_force: Array
    signed_projection: float
    absolute_projection: float
    parallel_metric_rms: float
    active_dof_count: int

    def __post_init__(self) -> None:
        shape = np.asarray(self.raw_physical_force).shape
        for name in ("raw_physical_force", "normalized_mode", "parallel_force", "perpendicular_force"):
            object.__setattr__(self, name, freeze_array(getattr(self, name), shape=shape))


@dataclass(frozen=True)
class DampingSelection:
    policy_mode: str
    selected_lambda: float | None
    parallel_metric: float
    metric_name: str
    sequence_id: str
    identity: str
    reference_equation: str = "shang_liu_eq15"
    reference_page: str = "JCTC 2010 journal p.1139 (PDF page 4)"

    def __post_init__(self) -> None:
        mode = str(self.policy_mode).strip().lower()
        if mode not in {"none", "fixed", "shang_liu"}:
            raise ValueError("invalid damping selection policy")
        object.__setattr__(self, "policy_mode", mode)
        metric = float(self.parallel_metric)
        if not np.isfinite(metric) or metric < 0.0:
            raise ValueError("parallel metric must be finite and >= 0")
        object.__setattr__(self, "parallel_metric", metric)
        if self.selected_lambda is not None:
            lam = float(self.selected_lambda)
            if not np.isfinite(lam) or lam < 0.0:
                raise ValueError("selected lambda must be finite and >= 0")
            object.__setattr__(self, "selected_lambda", lam)
        if not str(self.sequence_id).strip() or not str(self.identity).strip():
            raise ValueError("damping selection sequence/identity cannot be empty")


@dataclass(frozen=True)
class ParallelForcePolicyResult:
    raw_physical_force: Array
    normalized_mode: Array
    parallel_force: Array
    perpendicular_force: Array
    translation_force: Array
    curvature: float
    selected_lambda: float | None
    effective_lambda: float | None
    damping_applied: bool
    policy_mode: str
    damping_identity: str
    parallel_metric_rms: float
    positive_curvature_policy: str

    def __post_init__(self) -> None:
        shape = np.asarray(self.raw_physical_force).shape
        for name in (
            "raw_physical_force",
            "normalized_mode",
            "parallel_force",
            "perpendicular_force",
            "translation_force",
        ):
            object.__setattr__(self, name, freeze_array(getattr(self, name), shape=shape))


def parallel_force_components(
    raw_center_force: object,
    mode: object,
    coordinate_space: ActiveCoordinateSpace,
) -> ParallelForceComponents:
    """Decompose a raw physical center force using one normalized active-space mode.

    The project convention for the scheduler/damping RMS is

    ``sqrt(sum(F_parallel_active**2) / n_active_cartesian_dofs)``.

    Fixed coordinates and explicitly supplied null modes are removed through the
    sealed ``ActiveCoordinateSpace`` contract.  This function never consumes a
    reflected/damped force.
    """

    force = coordinate_space.project(raw_center_force)
    unit_mode = coordinate_space.normalized(mode)
    signed = float(np.dot(force.reshape(-1), unit_mode.reshape(-1)))
    parallel = signed * unit_mode
    perpendicular = force - parallel
    n_active = coordinate_space.active_dof_count
    if n_active <= 0:
        raise ValueError("parallel-force policy requires at least one movable Cartesian DOF")
    active_values = parallel[coordinate_space.active_dof_mask]
    rms = float(np.sqrt(np.dot(active_values, active_values) / n_active))
    return ParallelForceComponents(
        raw_physical_force=force,
        normalized_mode=unit_mode,
        parallel_force=parallel,
        perpendicular_force=perpendicular,
        signed_projection=signed,
        absolute_projection=abs(signed),
        parallel_metric_rms=rms,
        active_dof_count=n_active,
    )


def shang_liu_lambda(parallel_rms: float) -> float:
    """Return the exact Eq. 15 piecewise lambda for a nonnegative RMS value."""

    value = float(parallel_rms)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("parallel RMS must be finite and >= 0")
    if value >= 2.0:
        return 0.10
    if value >= 1.0:
        return 0.25
    if value >= 0.5:
        return 0.50
    return 1.00


def _selection_identity(mode: str, sequence_id: str, selected_lambda: float | None, metric: float) -> str:
    # Fixed and disabled policies are search-global identities.  Only the paper
    # schedule is sequence-selected from the starting RMS and therefore carries
    # the sequence/metric in its identity.
    if mode in {"none", "fixed"}:
        payload = f"saddlemill-t06-damping-v1\0{mode}\0{selected_lambda!r}"
    else:
        payload = f"saddlemill-t06-damping-v1\0{mode}\0{sequence_id}\0{selected_lambda!r}\0{metric:.17g}"
    return f"damping:{mode}:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_damping_for_sequence(
    raw_center_force: object,
    mode: object,
    coordinate_space: ActiveCoordinateSpace,
    config: ParallelForceDampingConfig,
    *,
    sequence_id: str,
) -> DampingSelection:
    """Select lambda exactly once for a new translation sequence.

    Callers must retain the returned selection through skipped mode solves and
    predicted held-mode changes.  Re-selection belongs only at an actual real
    mode solve/reorientation boundary.
    """

    sequence_id = str(sequence_id).strip()
    if not sequence_id:
        raise ValueError("sequence_id cannot be empty")
    components = parallel_force_components(raw_center_force, mode, coordinate_space)
    if config.mode == "none":
        selected = None
    elif config.mode == "fixed":
        selected = config.fixed_lambda
    else:
        selected = shang_liu_lambda(components.parallel_metric_rms)
    return DampingSelection(
        policy_mode=config.mode,
        selected_lambda=selected,
        parallel_metric=components.parallel_metric_rms,
        metric_name="movable_cartesian_dof_rms",
        sequence_id=sequence_id,
        identity=_selection_identity(config.mode, sequence_id, selected, components.parallel_metric_rms),
    )


def apply_parallel_force_policy(
    raw_center_force: object,
    mode: object,
    coordinate_space: ActiveCoordinateSpace,
    *,
    curvature: float,
    selection: DampingSelection,
    positive_curvature_baseline: object | None = None,
) -> ParallelForcePolicyResult:
    """Apply the selected mode-schedule policy without inventing a positive-curvature rule.

    For negative curvature, ``none`` is the historical reflected-force algebra
    (effective lambda 1) and the two opt-in policies use their selected lambda.
    For nonnegative curvature this module *only* passes through a caller-supplied
    baseline force.  This keeps Eq. 16/paper positive-curvature prefactors a
    separate scientific selector rather than silently coupling them to damping.
    """

    c = float(curvature)
    if not np.isfinite(c):
        raise ValueError("curvature must be finite")
    components = parallel_force_components(raw_center_force, mode, coordinate_space)
    if c < 0.0:
        if selection.policy_mode == "none":
            effective = 1.0
            applied = False
        else:
            if selection.selected_lambda is None:
                raise ValueError("active damping selection is missing lambda")
            effective = selection.selected_lambda
            applied = True
        translation = components.perpendicular_force - effective * components.parallel_force
        positive_policy = "not_applicable_negative_curvature"
    else:
        if positive_curvature_baseline is None:
            raise ValueError(
                "positive curvature requires an explicit caller baseline; "
                "mode-schedule damping does not imply a positive-curvature paper policy"
            )
        translation = np.asarray(positive_curvature_baseline, dtype=float)
        if translation.shape != components.raw_physical_force.shape or not np.all(np.isfinite(translation)):
            raise ValueError("positive-curvature baseline force must be finite and match force shape")
        effective = None
        applied = False
        positive_policy = "external_baseline_passthrough"

    return ParallelForcePolicyResult(
        raw_physical_force=components.raw_physical_force,
        normalized_mode=components.normalized_mode,
        parallel_force=components.parallel_force,
        perpendicular_force=components.perpendicular_force,
        translation_force=translation,
        curvature=c,
        selected_lambda=selection.selected_lambda,
        effective_lambda=effective,
        damping_applied=applied,
        policy_mode=selection.policy_mode,
        damping_identity=selection.identity,
        parallel_metric_rms=components.parallel_metric_rms,
        positive_curvature_policy=positive_policy,
    )


__all__ = [
    "DampingSelection",
    "ParallelForceComponents",
    "ParallelForceDampingConfig",
    "ParallelForcePolicyResult",
    "apply_parallel_force_policy",
    "parallel_force_components",
    "select_damping_for_sequence",
    "shang_liu_lambda",
]
