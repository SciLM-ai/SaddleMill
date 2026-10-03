"""Partitioned physical-space L-BFGS translation core.

This module is deliberately calculator- and ASE-independent.  It consumes
selected raw physical endpoint pairs from the sealed C2 canonical history and
the explicit active coordinate space.  It does not consume globally
reflected/MMF forces and does not modify the canonical force bank.

For normalized active-space mode ``v`` and raw physical gradient ``g=-F``::

    P = v v^T
    Q = I - P
    g_P = P g
    g_Q = Q g

The Q branch minimizes the true physical energy using Q-projected physical
secants.  The P branch minimizes ``-E`` in the oriented scalar coordinate
``a = v^T x``.  Its scalar residual/gradient is
``r_P = d(-E)/da = -v^T g``.  Therefore the L-BFGS force supplied to the
scalar minimizer is ``-r_P = v^T g`` and the axial step is
``delta_a = -H_P^{-1} r_P = H_P^{-1}(v^T g)``.

The implementation uses the existing C2 pair safeguard and two-loop arithmetic
for both branches independently.  Cross P-Q Hessian blocks are neglected; this
is a local conditioning experiment, not a claim of a globally conservative
transformed field or guaranteed convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from time import perf_counter_ns
from typing import Mapping, Sequence

import numpy as np

from saddlemill.dimertools.dense_bfgs import (
    safe_max_atom_norm,
    safe_norm,
    safe_scale_to_max_atom,
)
from saddlemill.dimertools.force_history import (
    CanonicalForceHistory,
    ForceObservation,
    PairCandidate,
    normalise_tokens,
)
from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace, WorkPurpose
from saddlemill.dimertools.hvp_interfaces import (
    FiniteDifferenceForceHVPBackend,
    HVPRequest,
    physical_eigen_residual,
)
from saddlemill.dimertools.qn_reconstruction_core import (
    SecantRecord,
    _admit_candidate_pairs,
    _two_loop,
)
from saddlemill.dimertools.quasi_newton import VALID_SAFEGUARDS
from saddlemill.dimertools.qn_deep_diagnostics import (
    prepare_shifted_lbfgs,
    shifted_lbfgs_solve_prepared,
)

Array = np.ndarray

STATE_SCHEMA = "saddlemill_partitioned_lbfgs_v1"
SELECTOR = "partitioned_lbfgs"
DIRECT_DIMER_AXIAL_SELECTOR = "q_lbfgs_dimer_axial"
LEGACY_P_STEP_SOURCE = "legacy_lbfgs"
DIRECT_DIMER_P_STEP_SOURCE = "dimer_curvature"
VALID_P_STEP_SOURCES = frozenset({
    LEGACY_P_STEP_SOURCE,
    DIRECT_DIMER_P_STEP_SOURCE,
})
VALID_REGULARIZATIONS = frozenset({
    "off", "shifted_lbfgs_fixed", "shifted_lbfgs_trust", "shifted_trust_region"
})
SUPPORTED_PROJECTOR_POLICIES = frozenset({"reset", "reconstruct"})
COUPLING_ASSUMPTION = "block_diagonal_p_q_no_cross_terms"
MODE_COUPLING_DIRECTION_TOLERANCE = 1.0e-8
MODE_COUPLING_EPSILON = 1.0e-12


def _mode_coupling_diagnostics(
    history: CanonicalForceHistory,
    *,
    current_state,
    mode: Array,
    active_space: ActiveCoordinateSpace,
) -> dict[str, object]:
    """Diagnose the neglected ``Q H v`` block from already-paid force stencils.

    This routine is intentionally read-only.  It reconstructs a same-center
    physical Hessian-vector product from an already-retained derivative stencil
    whose direction matches the current oriented partition mode.  It never calls
    a calculator, never records a new observation, and never influences the
    partitioned step.
    """

    base: dict[str, object] = {
        "mode_hv_available": 0,
        "mode_hv_source": "",
        "mode_hv_family": "",
        "mode_hv_stencil_id": "",
        "mode_hv_stencil_scheme": "",
        "mode_hv_fd_displacement": "",
        "mode_hv_probe_count": 0,
        "mode_hv_direction_abs_overlap": "",
        "mode_hv_endpoint_observation_ids_json": "[]",
        "mode_hv_norm": "",
        "mode_qhv_norm": "",
        "mode_qhv_fraction": "",
        "mode_qhv_to_abs_curvature": "",
        "mode_rayleigh_curvature": "",
        "mode_curvature_reported": "",
        "mode_curvature_hv": "",
        "mode_curvature_delta": "",
        "mode_hv_additional_pes_calls": 0,
        "mode_hv_unavailable_reason": "no_matching_current_state_physical_hvp",
    }
    if current_state is None:
        base["mode_hv_unavailable_reason"] = "missing_current_state"
        return base

    v = active_space.normalized(mode)
    candidates: list[tuple[int, float, int, object]] = []
    for stencil in history.iter_stencils(complete_only=True):
        if int(stencil.state_id) != int(current_state.state_id):
            continue
        if str(stencil.purpose) != WorkPurpose.ALGORITHM.value:
            continue
        try:
            direction = active_space.normalized(stencil.direction)
        except Exception:
            continue
        overlap = abs(float(np.dot(direction.reshape(-1), v.reshape(-1))))
        if overlap < 1.0 - MODE_COUPLING_DIRECTION_TOLERANCE:
            continue
        centered = int(str(stencil.scheme).strip().lower() == "centered")
        # Prefer a centered stencil, then the closest-aligned and newest sample.
        candidates.append((centered, overlap, int(stencil.serial), stencil))

    if not candidates:
        return base

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    backend = FiniteDifferenceForceHVPBackend(history)
    last_unavailable = "matching_stencil_hvp_unavailable"
    for _centered, overlap, _serial, stencil in candidates:
        request = HVPRequest(
            state_id=current_state.state_id,
            state_uid=current_state.state_uid,
            geometry_id=current_state.geometry_id,
            direction=v,
            coordinate_space=active_space,
            purpose=WorkPurpose.ALGORITHM,
            source=stencil.source,
            family=stencil.family,
            stencil_id=stencil.stencil_id,
            metadata={
                "consumer": "partitioned_lbfgs_mode_coupling_diagnostic",
                "diagnostic_only": True,
            },
        )
        hvp = backend.apply(request)
        if not hvp.available or hvp.action is None:
            last_unavailable = hvp.unavailable_reason or last_unavailable
            continue
        residual = physical_eigen_residual(
            hvp, coordinate_space=active_space, epsilon=MODE_COUPLING_EPSILON
        )
        if not residual.available or residual.residual_norm is None or residual.rho is None:
            last_unavailable = residual.unavailable_reason or "physical_eigen_residual_unavailable"
            continue

        action = active_space.project(hvp.action)
        hv_norm = safe_norm(action)
        qhv_norm = float(residual.residual_norm)
        rho = float(residual.rho)
        reported = ""
        if current_state.curvature is not None:
            value = float(current_state.curvature)
            if np.isfinite(value):
                reported = value
        probe_count = int(stencil.plus_serial is not None) + int(stencil.minus_serial is not None)
        qhv_fraction: object = ""
        if hv_norm > MODE_COUPLING_EPSILON:
            qhv_fraction = qhv_norm / hv_norm
        qhv_to_abs_curvature: object = ""
        if abs(rho) > MODE_COUPLING_EPSILON:
            qhv_to_abs_curvature = qhv_norm / abs(rho)
        curvature_delta: object = ""
        if reported != "":
            curvature_delta = rho - float(reported)

        return {
            **base,
            "mode_hv_available": 1,
            "mode_hv_source": str(hvp.metadata.source),
            "mode_hv_family": str(hvp.metadata.family),
            "mode_hv_stencil_id": str(stencil.stencil_id),
            "mode_hv_stencil_scheme": str(hvp.stencil_scheme),
            "mode_hv_fd_displacement": (
                "" if hvp.displacement_scale is None else float(hvp.displacement_scale)
            ),
            "mode_hv_probe_count": probe_count,
            "mode_hv_direction_abs_overlap": float(overlap),
            "mode_hv_endpoint_observation_ids_json": json.dumps(
                list(hvp.endpoint_observation_ids), separators=(",", ":")
            ),
            "mode_hv_norm": hv_norm,
            "mode_qhv_norm": qhv_norm,
            "mode_qhv_fraction": qhv_fraction,
            "mode_qhv_to_abs_curvature": qhv_to_abs_curvature,
            "mode_rayleigh_curvature": rho,
            "mode_curvature_reported": reported,
            "mode_curvature_hv": rho,
            "mode_curvature_delta": curvature_delta,
            "mode_hv_additional_pes_calls": int(hvp.metadata.pes_call_delta),
            "mode_hv_unavailable_reason": "",
        }

    base["mode_hv_unavailable_reason"] = last_unavailable
    return base


def _finite_positive(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _hash_array(prefix: bytes, array: Array, *tokens: str) -> str:
    data = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(prefix)
    digest.update(np.asarray(data.shape, dtype="<i8").tobytes())
    digest.update(data.tobytes(order="C"))
    for token in tokens:
        digest.update(b"\0")
        digest.update(str(token).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _canonical_axis_sign(mode: Array, *, tolerance: float) -> tuple[Array, bool]:
    """Choose a deterministic sign when no previous oriented mode is usable."""

    flat = np.asarray(mode, dtype=float).reshape(-1)
    for value in flat:
        if abs(float(value)) > tolerance:
            if value < 0.0:
                return -np.asarray(mode, dtype=float), True
            return np.asarray(mode, dtype=float).copy(), False
    raise ValueError("mode is zero after active-space projection")


def _state_for_observation(
    history: CanonicalForceHistory, observation: ForceObservation
):
    for state in history.states:
        if state.state_id == observation.state_id:
            return state
    return None


def _record_action_by_key(records: Sequence[SecantRecord]) -> dict[tuple[int, int, str], SecantRecord]:
    return {
        (int(record.state_id), int(record.serial), str(record.source)): record
        for record in records
    }


def _serializable_pairs(admission) -> list[dict[str, object]]:
    actions = _record_action_by_key(admission.records)
    rows: list[dict[str, object]] = []
    for (s, y, sy, source), candidate in zip(admission.accepted, admission.accepted_meta):
        record = actions.get((int(candidate.state_id), int(candidate.serial), str(candidate.source)))
        rows.append(
            {
                "state_id": int(candidate.state_id),
                "serial": int(candidate.serial),
                "source": str(source),
                "first_observation_id": candidate.first.observation_id,
                "second_observation_id": candidate.second.observation_id,
                "s": np.asarray(s, dtype=float).tolist(),
                "y": np.asarray(y, dtype=float).tolist(),
                "s_dot_y": float(sy),
                "action": "" if record is None else str(record.action),
            }
        )
    return rows


def _record_summary(records: Sequence[SecantRecord]) -> dict[str, object]:
    reasons: dict[str, int] = {}
    actions: dict[str, int] = {}
    for record in records:
        actions[record.action] = actions.get(record.action, 0) + 1
        if record.rejection_reason:
            reasons[record.rejection_reason] = reasons.get(record.rejection_reason, 0) + 1
    return {
        "actions": actions,
        "rejection_reasons": reasons,
        "records": len(records),
    }


@dataclass(frozen=True)
class PartitionedStepResult:
    """One proposed partitioned translation step and its deterministic evidence."""

    step: Array
    raw_step: Array
    p_step: Array
    q_step: Array
    oriented_mode: Array
    p_residual: float
    q_gradient: Array
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    p_records: tuple[SecantRecord, ...] = ()
    q_records: tuple[SecantRecord, ...] = ()


class PartitionedLBFGS:
    """Pure partitioned physical-space translation core.

    The object owns only projector/sign continuity and reset-boundary state.
    Raw observations remain owned by :class:`CanonicalForceHistory`.  Under the
    ``reconstruct`` policy every call rebuilds admitted history from that bounded
    raw history in the *current* projector.  The legacy selector learns both P
    and Q L-BFGS branches; ``q_lbfgs_dimer_axial`` learns only Q and obtains P
    directly from the current Dimer curvature.  Under ``reset`` the same
    reconstruction is used, but center pairs older than the last incompatible
    projector transition are excluded.
    """

    def __init__(
        self,
        *,
        p_initial_hessian: float = 70.0,
        q_initial_hessian: float = 70.0,
        p_dynamic_h0: bool = False,
        q_dynamic_h0: bool = False,
        p_safeguard: str = "skip",
        q_safeguard: str = "skip",
        p_curvature_floor: float = 1.0e-3,
        q_curvature_floor: float = 1.0e-3,
        curvature_epsilon: float = 1.0e-12,
        powell_eta: float = 0.2,
        p_memory: int = 10,
        q_memory: int = 10,
        projector_policy: str = "reconstruct",
        projector_overlap_tolerance: float = 0.95,
        sign_reference_overlap_tolerance: float = 1.0e-10,
        step_damping: float = 1.0,
        pair_sources: object = "center_center",
        regularization: str = "off",
        regularization_mu: float = 1.0,
        regularization_radius: float = 0.1,
        regularization_tolerance: float = 1.0e-8,
        p_step_source: str = LEGACY_P_STEP_SOURCE,
    ) -> None:
        self.p_initial_hessian = _finite_positive(p_initial_hessian, "p_initial_hessian")
        self.q_initial_hessian = _finite_positive(q_initial_hessian, "q_initial_hessian")
        self.p_dynamic_h0 = bool(p_dynamic_h0)
        self.q_dynamic_h0 = bool(q_dynamic_h0)
        self.p_safeguard = str(p_safeguard).strip().lower()
        self.q_safeguard = str(q_safeguard).strip().lower()
        if self.p_safeguard not in VALID_SAFEGUARDS:
            raise ValueError(f"unsupported P safeguard: {self.p_safeguard!r}")
        if self.q_safeguard not in VALID_SAFEGUARDS:
            raise ValueError(f"unsupported Q safeguard: {self.q_safeguard!r}")
        self.p_curvature_floor = _finite_nonnegative(p_curvature_floor, "p_curvature_floor")
        self.q_curvature_floor = _finite_nonnegative(q_curvature_floor, "q_curvature_floor")
        self.curvature_epsilon = _finite_nonnegative(curvature_epsilon, "curvature_epsilon")
        self.powell_eta = float(powell_eta)
        if not np.isfinite(self.powell_eta) or not (0.0 < self.powell_eta < 1.0):
            raise ValueError("powell_eta must be finite and in (0, 1)")
        self.p_memory = int(p_memory)
        self.q_memory = int(q_memory)
        if self.p_memory < 0 or self.q_memory < 0:
            raise ValueError("P/Q memory must be >= 0 (0 means no additional pair cap)")
        self.projector_policy = str(projector_policy).strip().lower()
        if self.projector_policy == "transport":
            raise ValueError(
                "projector_policy=transport is intentionally unavailable for partitioned L-BFGS: "
                "the sealed sphere transport maps tangent vectors on the mode sphere, "
                "not arbitrary P/Q quasi-Newton secants; use reset or reconstruct"
            )
        if self.projector_policy not in SUPPORTED_PROJECTOR_POLICIES:
            raise ValueError(
                f"projector_policy must be one of {sorted(SUPPORTED_PROJECTOR_POLICIES)}"
            )
        self.projector_overlap_tolerance = float(projector_overlap_tolerance)
        if not np.isfinite(self.projector_overlap_tolerance) or not (
            0.0 <= self.projector_overlap_tolerance <= 1.0
        ):
            raise ValueError("projector_overlap_tolerance must be in [0, 1]")
        self.sign_reference_overlap_tolerance = _finite_nonnegative(
            sign_reference_overlap_tolerance, "sign_reference_overlap_tolerance"
        )
        if self.sign_reference_overlap_tolerance >= 1.0:
            raise ValueError("sign_reference_overlap_tolerance must be < 1")
        self.step_damping = _finite_positive(step_damping, "step_damping")
        if self.step_damping > 1.0:
            raise ValueError(
                "step_damping must be <= 1 so the existing maximum-translation cap remains final"
            )
        pair_source_tokens = normalise_tokens(pair_sources)
        if not pair_source_tokens:
            raise ValueError("pair_sources cannot be empty")
        # Keep one parser/grammar for every canonical force-bank consumer.
        CanonicalForceHistory._parse_pair_sources(pair_source_tokens)
        self.pair_sources = " ".join(pair_source_tokens)
        self.regularization = str(regularization).strip().lower()
        if self.regularization not in VALID_REGULARIZATIONS:
            raise ValueError(
                f"regularization must be one of {sorted(VALID_REGULARIZATIONS)}"
            )
        self.regularization_mu = _finite_nonnegative(
            regularization_mu, "regularization_mu"
        )
        self.regularization_radius = _finite_positive(
            regularization_radius, "regularization_radius"
        )
        self.regularization_tolerance = _finite_positive(
            regularization_tolerance, "regularization_tolerance"
        )
        self.p_step_source = str(p_step_source).strip().lower()
        if self.p_step_source not in VALID_P_STEP_SOURCES:
            raise ValueError(
                f"p_step_source must be one of {sorted(VALID_P_STEP_SOURCES)}"
            )

        self._active_space_identity: str | None = None
        self._last_mode: Array | None = None
        self._last_mode_id: str = ""
        self._last_projector_id: str = ""
        self._projector_age = 0
        self._history_floor_state_id: int | None = None
        self._step_count = 0
        self._projector_reset_count = 0
        self._sign_flip_count = 0
        self._last_reset_reason = ""
        self._last_orientation_status = "uninitialized"
        self._last_p_h0_inverse_scale = 1.0 / self.p_initial_hessian
        self._last_q_h0_inverse_scale = 1.0 / self.q_initial_hessian
        self._last_p_history: list[dict[str, object]] = []
        self._last_q_history: list[dict[str, object]] = []

    def resolved_settings(self) -> dict[str, object]:
        selector = (
            DIRECT_DIMER_AXIAL_SELECTOR
            if self.p_step_source == DIRECT_DIMER_P_STEP_SOURCE
            else SELECTOR
        )
        return {
            "selector": selector,
            "p_step_source": self.p_step_source,
            "projector_policy": self.projector_policy,
            "projector_overlap_tolerance": self.projector_overlap_tolerance,
            "projector_threshold_tie_policy": "compatible_at_equal_overlap",
            "sign_reference_overlap_tolerance": self.sign_reference_overlap_tolerance,
            "p_initial_hessian": self.p_initial_hessian,
            "q_initial_hessian": self.q_initial_hessian,
            "p_dynamic_h0": self.p_dynamic_h0,
            "q_dynamic_h0": self.q_dynamic_h0,
            "p_safeguard": self.p_safeguard,
            "q_safeguard": self.q_safeguard,
            "p_curvature_floor": self.p_curvature_floor,
            "q_curvature_floor": self.q_curvature_floor,
            "curvature_epsilon": self.curvature_epsilon,
            "powell_eta": self.powell_eta,
            "p_memory": self.p_memory,
            "q_memory": self.q_memory,
            "step_damping": self.step_damping,
            "pair_sources": self.pair_sources,
            "regularization": self.regularization,
            "regularization_mu": self.regularization_mu,
            "regularization_radius": self.regularization_radius,
            "regularization_tolerance": self.regularization_tolerance,
            "regularization_shift_scope": "shared_scalar_mu_across_p_q_blocks",
            "regularization_radius_metric": "combined_step_euclidean_norm",
            "gradient_origin": "raw_physical_center",
            "p_residual": (
                "r_P=-v^T g; scalar step=-H_P^-1 r_P"
                if self.p_step_source == LEGACY_P_STEP_SOURCE
                else "direct Dimer axial: delta_a=(v^T g)/|lambda_dimer|"
            ),
            "q_residual": "g_Q=Q g; step=-H_Q^-1 g_Q",
            "coupling_assumption": COUPLING_ASSUMPTION,
            "transport_supported": False,
        }

    def _validate_current_observation(
        self,
        history: CanonicalForceHistory,
        observation: ForceObservation,
        active_space: ActiveCoordinateSpace,
    ) -> None:
        if not isinstance(history, CanonicalForceHistory):
            raise TypeError("history must be CanonicalForceHistory")
        if not isinstance(observation, ForceObservation):
            raise TypeError("current_observation must be ForceObservation")
        retained = history.observation_by_serial(observation.serial)
        if retained is None or retained.observation_id != observation.observation_id:
            raise ValueError("current center observation is not retained by canonical history")
        if retained.role != "center" or observation.role != "center":
            raise ValueError("partitioned translation requires a canonical center observation")
        if not retained.is_physical or not observation.is_physical:
            raise ValueError("partitioned translation rejects derived/effective/reflected observations")
        if retained.purpose != WorkPurpose.ALGORITHM.value:
            raise ValueError("partitioned translation rejects diagnostic center observations")
        current_state = history.current
        if (
            current_state is None
            or current_state.center is None
            or current_state.state_id != observation.state_id
            or current_state.center.serial != observation.serial
        ):
            raise ValueError("current observation must be the current accepted canonical center")
        self._validate_observation_space(observation, active_space)

    @staticmethod
    def _validate_observation_space(
        observation: ForceObservation, active_space: ActiveCoordinateSpace
    ) -> None:
        if observation.positions.shape != active_space.active_dof_mask.shape:
            raise ValueError("canonical observation shape is incompatible with active coordinates")
        if observation.active_dof_mask is None:
            raise ValueError("canonical center observation is missing active-coordinate mask")
        if not np.array_equal(observation.active_dof_mask, active_space.active_dof_mask):
            raise ValueError("canonical observation active-coordinate mask mismatch")
        if observation.coordinate_convention in {"", "unspecified"}:
            raise ValueError("canonical center observation is missing coordinate convention")
        if observation.coordinate_convention != active_space.convention:
            raise ValueError("canonical observation coordinate convention mismatch")

    def _orient_mode(
        self, mode: object, active_space: ActiveCoordinateSpace
    ) -> tuple[Array, float | None, bool, str]:
        candidate = active_space.normalized(mode, tolerance=1.0e-14)
        sign_flipped = False
        overlap: float | None = None
        status = "initial_canonical_sign"
        if self._last_mode is None:
            candidate, sign_flipped = _canonical_axis_sign(
                candidate, tolerance=self.sign_reference_overlap_tolerance
            )
        else:
            signed = float(np.dot(self._last_mode.reshape(-1), candidate.reshape(-1)))
            overlap = abs(signed)
            if overlap > self.sign_reference_overlap_tolerance:
                if signed < 0.0:
                    candidate = -candidate
                    sign_flipped = True
                    status = "aligned_by_sign_flip"
                else:
                    status = "aligned_to_previous"
            else:
                candidate, sign_flipped = _canonical_axis_sign(
                    candidate, tolerance=self.sign_reference_overlap_tolerance
                )
                status = "near_orthogonal_sign_reference_lost"
        # Canonicalize signed IEEE zero so v and -v produce byte-identical
        # mode/projector identities after sign homing.  Do not threshold real
        # small components; only exact zeros are rewritten.
        candidate = np.array(candidate, dtype=float, copy=True)
        candidate[candidate == 0.0] = 0.0
        return candidate, overlap, sign_flipped, status

    def _projector_ids(
        self, mode: Array, active_space: ActiveCoordinateSpace
    ) -> tuple[str, str]:
        flat = mode.reshape(-1)
        mode_id = _hash_array(
            b"saddlemill-partitioned-oriented-mode-v1\0",
            flat,
            active_space.identity,
        )
        projector = np.outer(flat, flat)
        projector_id = _hash_array(
            b"saddlemill-partitioned-projector-v1\0",
            projector,
            active_space.identity,
        )
        return mode_id, projector_id

    def _candidate_pairs(
        self,
        history: CanonicalForceHistory,
        current_observation: ForceObservation,
        active_space: ActiveCoordinateSpace,
    ) -> list[PairCandidate]:
        pairs = history.pair_candidates(self.pair_sources, physical_only=True)
        result: list[PairCandidate] = []
        for candidate in pairs:
            # The validated current observation is the center of history.current,
            # so no retained state can be in the future.  A probe in that current
            # state is acquired *after* its center and therefore legitimately has
            # a larger serial; serial-based truncation would discard every such
            # center-probe secant.
            if candidate.first.state_id > current_observation.state_id:
                continue
            if (
                self.projector_policy == "reset"
                and self._history_floor_state_id is not None
                and candidate.first.state_id < self._history_floor_state_id
            ):
                continue
            self._validate_observation_space(candidate.first, active_space)
            self._validate_observation_space(candidate.second, active_space)
            result.append(candidate)
        return result

    @staticmethod
    def _q_project(
        value: object, mode: Array, active_space: ActiveCoordinateSpace
    ) -> Array:
        projected = active_space.project(value)
        flat = projected.reshape(-1)
        v = mode.reshape(-1)
        return (flat - float(np.dot(v, flat)) * v).reshape(projected.shape)

    def _build_raw_pairs(
        self,
        candidates: Sequence[PairCandidate],
        mode: Array,
        active_space: ActiveCoordinateSpace,
        *,
        include_p: bool = True,
    ) -> tuple[
        list[tuple[PairCandidate, Array, Array]],
        list[tuple[PairCandidate, Array, Array]],
    ]:
        p_pairs: list[tuple[PairCandidate, Array, Array]] = []
        q_pairs: list[tuple[PairCandidate, Array, Array]] = []
        v = mode.reshape(-1)
        for candidate in candidates:
            s = active_space.project(candidate.displacement())
            g_first = active_space.project(-np.asarray(candidate.first.forces, dtype=float))
            g_second = active_space.project(-np.asarray(candidate.second.forces, dtype=float))
            delta_g = active_space.project(g_second - g_first)

            s_flat = s.reshape(-1)
            dg_flat = delta_g.reshape(-1)
            if include_p:
                # Scalar gradient of -E: r_P = -v^T g, so Delta r_P = -v^T Delta g.
                p_s = np.asarray([float(np.dot(v, s_flat))], dtype=float)
                p_y = np.asarray([-float(np.dot(v, dg_flat))], dtype=float)
                p_pairs.append((candidate, p_s, p_y))
            q_s = self._q_project(s, mode, active_space).reshape(-1)
            q_y = self._q_project(delta_g, mode, active_space).reshape(-1)
            q_pairs.append((candidate, q_s, q_y))
        return p_pairs, q_pairs

    def _direct_dimer_axial_direction(
        self,
        *,
        g_parallel: float,
        curvature: object | None,
    ) -> tuple[Array, float, str]:
        if curvature is None:
            raise ValueError(
                "q_lbfgs_dimer_axial requires current Dimer curvature; none is available"
            )
        value = float(curvature)
        if not np.isfinite(value):
            raise ValueError(
                "q_lbfgs_dimer_axial requires finite current Dimer curvature"
            )
        magnitude = abs(value)
        if magnitude <= self.curvature_epsilon:
            raise ValueError(
                "q_lbfgs_dimer_axial current Dimer curvature magnitude is too small "
                f"for curvature_epsilon={self.curvature_epsilon:g}: {value!r}"
            )
        direction = np.asarray([float(g_parallel) / magnitude], dtype=float)
        return direction, 1.0 / magnitude, "direct_dimer_curvature"

    def _run_branch(
        self,
        *,
        raw_pairs: Sequence[tuple[PairCandidate, Array, Array]],
        force: Array,
        initial_hessian: float,
        dynamic_h0: bool,
        safeguard: str,
        curvature_floor: float,
        max_pairs: int,
        branch_name: str,
        zero_dimension: bool = False,
    ):
        admission = _admit_candidate_pairs(
            raw_pairs,
            np.asarray(force, dtype=float),
            initial_hessian=initial_hessian,
            safeguard=safeguard,
            curvature_floor=curvature_floor,
            curvature_epsilon=self.curvature_epsilon,
            powell_eta=self.powell_eta,
            cosine_threshold=None,
            max_pairs=max_pairs,
            cautious_epsilon=1.0e-6,
            cautious_alpha=1.0,
        )
        force_vec = np.asarray(force, dtype=float).reshape(-1)
        if zero_dimension:
            direction = np.zeros_like(force_vec)
            h0 = 1.0 / initial_hessian
            fallback = "zero_dimensional_branch"
        else:
            direction, h0, _trace = _two_loop(
                force_vec,
                admission.accepted,
                initial_hessian=initial_hessian,
                dynamic_h0=dynamic_h0,
                trace=False,
            )
            fallback = ""
            force_norm = safe_norm(force_vec)
            direction_norm = safe_norm(direction)
            aligned = float(np.dot(direction, force_vec)) if force_vec.size else 0.0
            if force_norm <= 1.0e-14:
                direction = np.zeros_like(force_vec)
                fallback = "zero_current_branch_force"
            elif (
                not np.all(np.isfinite(direction))
                or not np.isfinite(direction_norm)
                or direction_norm <= 1.0e-14
                or not np.isfinite(aligned)
                or aligned <= 0.0
            ):
                direction = force_vec / initial_hessian
                h0 = 1.0 / initial_hessian
                fallback = f"{branch_name}_scaled_force_fallback"
        return direction, float(h0), admission, fallback

    @staticmethod
    def _validate_shifted_branch(direction: Array, force: Array) -> None:
        direction = np.asarray(direction, dtype=float).reshape(-1)
        force = np.asarray(force, dtype=float).reshape(-1)
        if direction.shape != force.shape or not np.all(np.isfinite(direction)):
            raise np.linalg.LinAlgError("shifted partitioned branch produced invalid direction")
        if safe_norm(force) <= 1.0e-14:
            return
        alignment = float(np.dot(direction, force))
        if safe_norm(direction) <= 1.0e-14 or not np.isfinite(alignment) or alignment <= 0.0:
            raise np.linalg.LinAlgError("shifted partitioned branch is not aligned with its force")

    def _regularize_shared_shift(
        self,
        *,
        p_force: Array,
        q_force: Array,
        p_accepted,
        q_accepted,
        p_direction: Array,
        q_direction: Array,
        mode: Array,
        active_space: ActiveCoordinateSpace,
        q_zero_dimension: bool,
        p_direct_curvature: float | None = None,
    ) -> tuple[Array, Array, dict[str, object]]:
        """Apply one scalar Hessian shift to both block-diagonal P/Q models.

        The partitioned model already assumes zero P-Q Hessian cross blocks.  A
        single scalar ``mu`` therefore defines the full restricted solve
        unambiguously as ``diag(B_P + mu I, B_Q + mu I)``.  The trust equation is
        solved on the Euclidean norm of the *combined* Cartesian P+Q step.  This
        is the same shifted limited-memory BFGS algebra used by the unpartitioned
        canonical translator; it performs no force/PES/HVP/minimum-mode calls.
        """

        metrics: dict[str, object] = {
            "translation_regularization": self.regularization,
            "translation_regularization_applied": 0,
            "translation_regularization_fallback": "not_applicable",
            "translation_regularization_mu": "",
            "translation_regularization_iterations": "",
            "translation_regularized_norm": "",
            "translation_regularization_suppression_ratio": "",
            "translation_regularization_direction_cosine": "",
            "translation_regularization_final_maxstep_scale": "",
            "translation_regularization_final_max_atom_norm": "",
            "translation_trust_shift": "",
            "translation_trust_radius": "",
            "translation_trust_unshifted_boundary_norm": "",
            "translation_trust_regularized_boundary_norm": "",
            "translation_trust_regularized_direction_norm": "",
            "translation_trust_direction_cosine_unshifted": "",
            "translation_trust_shifted_condition_number": "",
            "translation_trust_iterations": "",
            "translation_trust_shift_expansions": "",
            "translation_trust_solve_ns": "",
            "translation_regularization_shift_scope": "shared_scalar_mu_across_p_q_blocks",
            "translation_regularization_radius_metric": "combined_step_euclidean_norm",
        }
        if self.regularization == "off" or not (
            p_accepted or q_accepted or p_direct_curvature is not None
        ):
            return (
                np.asarray(p_direction, dtype=float).copy(),
                np.asarray(q_direction, dtype=float).copy(),
                metrics,
            )

        started = perf_counter_ns()
        p_force_vec = np.asarray(p_force, dtype=float).reshape(-1)
        q_force_vec = np.asarray(q_force, dtype=float).reshape(-1)
        p_raw = np.asarray(p_direction, dtype=float).reshape(-1)
        q_raw = np.asarray(q_direction, dtype=float).reshape(-1)
        v = np.asarray(mode, dtype=float).reshape(-1)

        def combine(p_dir: Array, q_dir: Array) -> Array:
            p_step = float(np.asarray(p_dir, dtype=float).reshape(-1)[0]) * v
            q_step = np.asarray(q_dir, dtype=float).reshape(-1)
            return active_space.project((p_step + q_step).reshape(mode.shape))

        unshifted = combine(p_raw, q_raw)
        unshifted_norm = safe_norm(unshifted)
        metrics["translation_trust_unshifted_boundary_norm"] = unshifted_norm

        try:
            p_prepared = None
            p_direct_magnitude = None
            if p_direct_curvature is None:
                p_prepared = prepare_shifted_lbfgs(
                    p_force_vec,
                    p_accepted,
                    self.p_initial_hessian,
                    self.p_dynamic_h0,
                )
            else:
                p_direct_magnitude = abs(float(p_direct_curvature))
                if (
                    not np.isfinite(p_direct_magnitude)
                    or p_direct_magnitude <= self.curvature_epsilon
                ):
                    raise np.linalg.LinAlgError(
                        "invalid direct Dimer axial curvature in shifted solve"
                    )
            q_prepared = None
            if not q_zero_dimension:
                q_prepared = prepare_shifted_lbfgs(
                    q_force_vec,
                    q_accepted,
                    self.q_initial_hessian,
                    self.q_dynamic_h0,
                )

            def solve_at(mu: float) -> tuple[Array, Array, Array]:
                if p_direct_magnitude is None:
                    assert p_prepared is not None
                    p_dir = shifted_lbfgs_solve_prepared(p_prepared, mu)
                else:
                    p_dir = p_force_vec / (p_direct_magnitude + float(mu))
                if q_zero_dimension:
                    q_dir = np.zeros_like(q_force_vec)
                else:
                    assert q_prepared is not None
                    q_dir = shifted_lbfgs_solve_prepared(q_prepared, mu)
                self._validate_shifted_branch(p_dir, p_force_vec)
                if not q_zero_dimension:
                    self._validate_shifted_branch(q_dir, q_force_vec)
                return p_dir, q_dir, combine(p_dir, q_dir)

            iterations = 0
            expansions = 0
            if self.regularization == "shifted_lbfgs_fixed":
                mu = float(self.regularization_mu)
                p_reg, q_reg, combined = solve_at(mu)
            else:
                radius = float(self.regularization_radius)
                metrics["translation_trust_radius"] = radius
                if unshifted_norm <= radius:
                    mu = 0.0
                    p_reg, q_reg, combined = p_raw, q_raw, unshifted
                else:
                    lo = 0.0
                    hi = max(
                        float(self.p_initial_hessian),
                        float(self.q_initial_hessian),
                        1.0,
                    )
                    p_reg = p_raw
                    q_reg = q_raw
                    combined = unshifted
                    for expansions in range(1, 81):
                        p_try, q_try, step_try = solve_at(hi)
                        if safe_norm(step_try) <= radius:
                            p_reg, q_reg, combined = p_try, q_try, step_try
                            break
                        hi *= 2.0
                    else:
                        raise np.linalg.LinAlgError("shared shifted partitioned solve failed to bracket trust radius")

                    tol = float(self.regularization_tolerance)
                    for iterations in range(1, 81):
                        mid = 0.5 * (lo + hi)
                        p_try, q_try, step_try = solve_at(mid)
                        norm_try = safe_norm(step_try)
                        p_reg, q_reg, combined = p_try, q_try, step_try
                        if abs(norm_try - radius) <= tol * max(1.0, radius):
                            hi = mid
                            break
                        if norm_try > radius:
                            lo = mid
                        else:
                            hi = mid
                    else:
                        p_reg, q_reg, combined = solve_at(hi)
                    mu = float(hi)

            regularized_norm = safe_norm(combined)
            den = unshifted_norm * regularized_norm
            cosine = "" if den <= 0.0 else float(
                np.dot(unshifted.reshape(-1), combined.reshape(-1)) / den
            )
            metrics.update({
                "translation_regularization_applied": 1,
                "translation_regularization_fallback": "not_applicable",
                "translation_regularization_mu": float(mu),
                "translation_regularization_iterations": int(iterations),
                "translation_regularized_norm": regularized_norm,
                "translation_regularization_suppression_ratio": (
                    "" if unshifted_norm <= 1.0e-300 else regularized_norm / unshifted_norm
                ),
                "translation_regularization_direction_cosine": cosine,
                "translation_trust_shift": float(mu),
                "translation_trust_regularized_boundary_norm": regularized_norm,
                "translation_trust_regularized_direction_norm": regularized_norm,
                "translation_trust_direction_cosine_unshifted": cosine,
                "translation_trust_iterations": int(iterations),
                "translation_trust_shift_expansions": int(expansions),
            })
            metrics["translation_trust_solve_ns"] = int(perf_counter_ns() - started)
            return (
                np.asarray(p_reg, dtype=float).copy(),
                np.asarray(q_reg, dtype=float).copy(),
                metrics,
            )
        except Exception as exc:
            metrics["translation_regularization_fallback"] = (
                type(exc).__name__ + ":" + str(exc)
            )
            metrics["translation_trust_solve_ns"] = int(perf_counter_ns() - started)
            return p_raw.copy(), q_raw.copy(), metrics

    def step(
        self,
        *,
        history: CanonicalForceHistory,
        current_observation: ForceObservation,
        mode: object,
        active_space: ActiveCoordinateSpace,
        maximum_translation: float,
        curvature: object | None = None,
    ) -> PartitionedStepResult:
        """Build one partitioned step without mutating geometry or force history."""

        if not isinstance(active_space, ActiveCoordinateSpace):
            raise TypeError("active_space must be ActiveCoordinateSpace")
        maximum_translation = _finite_positive(maximum_translation, "maximum_translation")
        self._validate_current_observation(history, current_observation, active_space)
        if self._active_space_identity is not None and (
            self._active_space_identity != active_space.identity
        ):
            raise ValueError("active-coordinate identity changed within partitioned optimizer state")

        oriented_mode, overlap, sign_flipped, orientation_status = self._orient_mode(
            mode, active_space
        )
        if sign_flipped:
            self._sign_flip_count += 1
        mode_id, projector_id = self._projector_ids(oriented_mode, active_space)
        current_state = _state_for_observation(history, current_observation)
        mode_coupling = _mode_coupling_diagnostics(
            history,
            current_state=current_state,
            mode=oriented_mode,
            active_space=active_space,
        )

        reset_reason = ""
        if (
            self.projector_policy == "reset"
            and overlap is not None
            # Exact equality is deliberately compatible and regression-tested.
            and overlap < self.projector_overlap_tolerance
        ):
            self._history_floor_state_id = int(current_observation.state_id)
            self._projector_reset_count += 1
            reset_reason = "projector_overlap_below_threshold"

        if self._last_mode is None or reset_reason or orientation_status == "near_orthogonal_sign_reference_lost":
            projector_age = 0
        else:
            projector_age = self._projector_age + 1

        candidates = self._candidate_pairs(
            history, current_observation, active_space
        )
        direct_dimer_axial = self.p_step_source == DIRECT_DIMER_P_STEP_SOURCE
        p_raw_pairs, q_raw_pairs = self._build_raw_pairs(
            candidates,
            oriented_mode,
            active_space,
            include_p=not direct_dimer_axial,
        )

        gradient = active_space.project(-np.asarray(current_observation.forces, dtype=float))
        v = oriented_mode.reshape(-1)
        g_flat = gradient.reshape(-1)
        g_parallel = float(np.dot(v, g_flat))
        p_residual = -g_parallel
        q_gradient = self._q_project(gradient, oriented_mode, active_space)
        q_force = -q_gradient.reshape(-1)
        p_force = np.asarray([g_parallel], dtype=float)  # = -r_P

        effective_dimension = max(
            0, active_space.active_dof_count - len(active_space.null_basis)
        )
        q_dimension = max(0, effective_dimension - 1)

        direct_curvature = None
        if direct_dimer_axial:
            if curvature is None and current_state is not None:
                curvature = current_state.curvature
            p_direction, p_h0, p_fallback = self._direct_dimer_axial_direction(
                g_parallel=g_parallel,
                curvature=curvature,
            )
            direct_curvature = float(curvature)
            p_admission = _admit_candidate_pairs(
                [],
                p_force,
                initial_hessian=self.p_initial_hessian,
                safeguard=self.p_safeguard,
                curvature_floor=self.p_curvature_floor,
                curvature_epsilon=self.curvature_epsilon,
                powell_eta=self.powell_eta,
                cosine_threshold=None,
                max_pairs=0,
                cautious_epsilon=1.0e-6,
                cautious_alpha=1.0,
            )
        else:
            p_direction, p_h0, p_admission, p_fallback = self._run_branch(
                raw_pairs=p_raw_pairs,
                force=p_force,
                initial_hessian=self.p_initial_hessian,
                dynamic_h0=self.p_dynamic_h0,
                safeguard=self.p_safeguard,
                curvature_floor=self.p_curvature_floor,
                max_pairs=self.p_memory,
                branch_name="p",
                zero_dimension=False,
            )
        q_direction, q_h0, q_admission, q_fallback = self._run_branch(
            raw_pairs=q_raw_pairs,
            force=q_force,
            initial_hessian=self.q_initial_hessian,
            dynamic_h0=self.q_dynamic_h0,
            safeguard=self.q_safeguard,
            curvature_floor=self.q_curvature_floor,
            max_pairs=self.q_memory,
            branch_name="q",
            zero_dimension=(q_dimension == 0),
        )

        p_direction, q_direction, regularization_metrics = self._regularize_shared_shift(
            p_force=p_force,
            q_force=q_force,
            p_accepted=p_admission.accepted,
            q_accepted=q_admission.accepted,
            p_direction=p_direction,
            q_direction=q_direction,
            mode=oriented_mode,
            active_space=active_space,
            q_zero_dimension=(q_dimension == 0),
            p_direct_curvature=direct_curvature,
        )

        p_step = (float(p_direction[0]) * v).reshape(oriented_mode.shape)
        q_step = q_direction.reshape(oriented_mode.shape)
        raw_step = active_space.project(p_step + q_step)
        capped_step, raw_max_atom, clipped, clip_scale = safe_scale_to_max_atom(
            raw_step, maximum_translation
        )
        if not np.isfinite(clip_scale):
            raise RuntimeError("partitioned translation produced invalid maximum-step scale")
        final_step = active_space.project(self.step_damping * capped_step)
        if regularization_metrics.get("translation_regularization_applied"):
            regularization_metrics.update({
                "translation_regularization_final_maxstep_scale": float(clip_scale),
                "translation_regularization_final_max_atom_norm": safe_max_atom_norm(final_step),
            })

        self._active_space_identity = active_space.identity
        self._last_mode = np.array(oriented_mode, dtype=float, copy=True)
        self._last_mode_id = mode_id
        self._last_projector_id = projector_id
        self._projector_age = int(projector_age)
        self._step_count += 1
        self._last_reset_reason = reset_reason
        self._last_orientation_status = orientation_status
        self._last_p_h0_inverse_scale = p_h0
        self._last_q_h0_inverse_scale = q_h0
        self._last_p_history = _serializable_pairs(p_admission)
        self._last_q_history = _serializable_pairs(q_admission)

        diagnostics: dict[str, object] = {
            **mode_coupling,
            **self.resolved_settings(),
            **regularization_metrics,
            "step_index": self._step_count,
            "state_id": int(current_observation.state_id),
            "state_uid": "" if current_state is None else current_state.state_uid,
            "geometry_id": current_observation.geometry_id,
            "current_observation_id": current_observation.observation_id,
            "active_coordinate_identity": active_space.identity,
            "active_dof_count": active_space.active_dof_count,
            "effective_active_dimension": effective_dimension,
            "q_dimension": q_dimension,
            "mode_id": mode_id,
            "projector_id": projector_id,
            "projector_continuity_age": projector_age,
            "p_mode_id": mode_id,
            "q_mode_id": mode_id,
            "p_projector_id": projector_id,
            "q_projector_id": projector_id,
            "p_projector_age": projector_age,
            "q_projector_age": projector_age,
            "previous_projector_overlap": "" if overlap is None else float(overlap),
            "projector_reset_reason": reset_reason,
            "projector_reset_count": self._projector_reset_count,
            "history_floor_state_id": "" if self._history_floor_state_id is None else self._history_floor_state_id,
            "mode_sign_flipped_this_step": int(sign_flipped),
            "mode_sign_flips_total": self._sign_flip_count,
            "orientation_status": orientation_status,
            "p_physical_gradient_scalar": g_parallel,
            "p_residual_scalar": p_residual,
            "p_step_scalar": float(p_direction[0]),
            "p_step_source": self.p_step_source,
            "p_dimer_curvature": "" if direct_curvature is None else direct_curvature,
            "p_step_curvature": (
                "" if direct_curvature is None else -abs(direct_curvature)
            ),
            "p_l_bfgs_history_used": int(not direct_dimer_axial),
            "p_uphill_first_order_product": float(g_parallel * p_direction[0]),
            "q_gradient_norm": safe_norm(q_gradient),
            "q_descent_first_order_product": float(np.dot(q_gradient.reshape(-1), q_direction)),
            "p_pair_candidates": len(p_raw_pairs),
            "q_pair_candidates": len(q_raw_pairs),
            "pair_source_candidates": {
                source: sum(1 for candidate in candidates if candidate.source == source)
                for source in sorted({candidate.source for candidate in candidates})
            },
            "p_pairs_used": len(p_admission.accepted),
            "q_pairs_used": len(q_admission.accepted),
            "p_pairs_admissible_before_max_pairs": int(
                p_admission.pairs_admissible_before_max_pairs
            ),
            "q_pairs_admissible_before_max_pairs": int(
                q_admission.pairs_admissible_before_max_pairs
            ),
            "p_pairs_removed_by_max_pairs": int(
                p_admission.pairs_removed_by_max_pairs
            ),
            "q_pairs_removed_by_max_pairs": int(
                q_admission.pairs_removed_by_max_pairs
            ),
            "p_max_pairs": int(self.p_memory),
            "q_max_pairs": int(self.q_memory),
            "p_used_pair_ids": [
                f"{row['first_observation_id']}->{row['second_observation_id']}"
                for row in self._last_p_history
            ],
            "q_used_pair_ids": [
                f"{row['first_observation_id']}->{row['second_observation_id']}"
                for row in self._last_q_history
            ],
            "p_used_pair_sources": {
                source: sum(1 for row in self._last_p_history if row["source"] == source)
                for source in sorted({str(row["source"]) for row in self._last_p_history})
            },
            "q_used_pair_sources": {
                source: sum(1 for row in self._last_q_history if row["source"] == source)
                for source in sorted({str(row["source"]) for row in self._last_q_history})
            },
            "p_pair_summary": _record_summary(p_admission.records),
            "q_pair_summary": _record_summary(q_admission.records),
            "p_h0_inverse_scale": p_h0,
            "q_h0_inverse_scale": q_h0,
            "p_fallback": p_fallback,
            "q_fallback": q_fallback,
            "raw_step_norm": safe_norm(raw_step),
            "raw_step_max_atom_norm": raw_max_atom,
            "maximum_translation": maximum_translation,
            "maximum_translation_clipped": int(bool(clipped)),
            "maximum_translation_scale": float(clip_scale),
            "accepted_step_norm": safe_norm(final_step),
            "accepted_step_max_atom_norm": safe_max_atom_norm(final_step),
            "p_q_cross_coupling_used": False,
            "p_q_coupling_note": (
                "P-Q Hessian cross blocks are neglected; local block-diagonal conditioning hypothesis only"
            ),
            "physical_b_consumed": False,
            "history_mutated": False,
            "additional_pes_calls": 0,
        }
        return PartitionedStepResult(
            step=np.array(final_step, dtype=float, copy=True),
            raw_step=np.array(raw_step, dtype=float, copy=True),
            p_step=np.array(p_step, dtype=float, copy=True),
            q_step=np.array(q_step, dtype=float, copy=True),
            oriented_mode=np.array(oriented_mode, dtype=float, copy=True),
            p_residual=float(p_residual),
            q_gradient=np.array(q_gradient, dtype=float, copy=True),
            diagnostics=diagnostics,
            p_records=tuple(p_admission.records),
            q_records=tuple(q_admission.records),
        )

    def to_state_dict(self) -> dict[str, object]:
        """Serialize exact partitioned-optimizer state; canonical raw history is serialized separately."""

        return {
            "schema": STATE_SCHEMA,
            "settings": self.resolved_settings(),
            "runtime": {
                "active_space_identity": self._active_space_identity,
                "last_mode": None if self._last_mode is None else self._last_mode.tolist(),
                "last_mode_id": self._last_mode_id,
                "last_projector_id": self._last_projector_id,
                "projector_continuity_age": self._projector_age,
                "history_floor_state_id": self._history_floor_state_id,
                "step_count": self._step_count,
                "projector_reset_count": self._projector_reset_count,
                "sign_flip_count": self._sign_flip_count,
                "last_reset_reason": self._last_reset_reason,
                "last_orientation_status": self._last_orientation_status,
                "last_p_h0_inverse_scale": self._last_p_h0_inverse_scale,
                "last_q_h0_inverse_scale": self._last_q_h0_inverse_scale,
                "last_p_history": self._last_p_history,
                "last_q_history": self._last_q_history,
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        payload: Mapping[str, object],
        *,
        expected_settings: Mapping[str, object] | None = None,
    ) -> "PartitionedLBFGS":
        if payload.get("schema") != STATE_SCHEMA:
            raise ValueError("unsupported PartitionedLBFGS state schema")
        settings = dict(payload.get("settings", {}))
        constructor = {
            "p_initial_hessian": settings["p_initial_hessian"],
            "q_initial_hessian": settings["q_initial_hessian"],
            "p_dynamic_h0": settings["p_dynamic_h0"],
            "q_dynamic_h0": settings["q_dynamic_h0"],
            "p_safeguard": settings["p_safeguard"],
            "q_safeguard": settings["q_safeguard"],
            "p_curvature_floor": settings["p_curvature_floor"],
            "q_curvature_floor": settings["q_curvature_floor"],
            "curvature_epsilon": settings["curvature_epsilon"],
            "powell_eta": settings["powell_eta"],
            "p_memory": settings["p_memory"],
            "q_memory": settings["q_memory"],
            "projector_policy": settings["projector_policy"],
            "projector_overlap_tolerance": settings["projector_overlap_tolerance"],
            "sign_reference_overlap_tolerance": settings["sign_reference_overlap_tolerance"],
            "step_damping": settings["step_damping"],
            "pair_sources": (
                "center_center"
                if settings.get("pair_sources", "center_center") == "center_center_only"
                else settings.get("pair_sources", "center_center")
            ),
            "regularization": settings.get("regularization", "off"),
            "regularization_mu": settings.get("regularization_mu", 1.0),
            "regularization_radius": settings.get("regularization_radius", 0.1),
            "regularization_tolerance": settings.get("regularization_tolerance", 1.0e-8),
            "p_step_source": settings.get("p_step_source", LEGACY_P_STEP_SOURCE),
        }
        result = cls(**constructor)
        if expected_settings is not None:
            expected = dict(expected_settings)
            actual = result.resolved_settings()
            for key, value in expected.items():
                if key not in actual or actual[key] != value:
                    raise ValueError(f"partitioned state settings mismatch for {key}")
        runtime = dict(payload.get("runtime", {}))
        mode = runtime.get("last_mode")
        result._active_space_identity = runtime.get("active_space_identity")
        result._last_mode = None if mode is None else np.asarray(mode, dtype=float)
        if result._last_mode is not None and not np.all(np.isfinite(result._last_mode)):
            raise ValueError("serialized last_mode contains non-finite values")
        result._last_mode_id = str(runtime.get("last_mode_id", ""))
        result._last_projector_id = str(runtime.get("last_projector_id", ""))
        result._projector_age = int(runtime.get("projector_continuity_age", 0))
        floor = runtime.get("history_floor_state_id")
        result._history_floor_state_id = None if floor is None else int(floor)
        result._step_count = int(runtime.get("step_count", 0))
        result._projector_reset_count = int(runtime.get("projector_reset_count", 0))
        result._sign_flip_count = int(runtime.get("sign_flip_count", 0))
        result._last_reset_reason = str(runtime.get("last_reset_reason", ""))
        result._last_orientation_status = str(runtime.get("last_orientation_status", "uninitialized"))
        result._last_p_h0_inverse_scale = float(
            runtime.get("last_p_h0_inverse_scale", 1.0 / result.p_initial_hessian)
        )
        result._last_q_h0_inverse_scale = float(
            runtime.get("last_q_h0_inverse_scale", 1.0 / result.q_initial_hessian)
        )
        result._last_p_history = [dict(row) for row in runtime.get("last_p_history", [])]
        result._last_q_history = [dict(row) for row in runtime.get("last_q_history", [])]
        if min(
            result._projector_age,
            result._step_count,
            result._projector_reset_count,
            result._sign_flip_count,
        ) < 0:
            raise ValueError("serialized partitioned counters must be >= 0")
        for value, name in (
            (result._last_p_h0_inverse_scale, "last_p_h0_inverse_scale"),
            (result._last_q_h0_inverse_scale, "last_q_h0_inverse_scale"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"serialized {name} must be finite and > 0")
        return result

    def state_json(self) -> str:
        return json.dumps(self.to_state_dict(), sort_keys=True, separators=(",", ":"))


__all__ = [
    "COUPLING_ASSUMPTION",
    "DIRECT_DIMER_AXIAL_SELECTOR",
    "DIRECT_DIMER_P_STEP_SOURCE",
    "LEGACY_P_STEP_SOURCE",
    "PartitionedLBFGS",
    "PartitionedStepResult",
    "SELECTOR",
    "STATE_SCHEMA",
    "SUPPORTED_PROJECTOR_POLICIES",
    "VALID_P_STEP_SOURCES",
]
