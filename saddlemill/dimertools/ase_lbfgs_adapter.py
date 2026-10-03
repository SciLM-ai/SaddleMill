"""ASE-specific L-BFGS compatibility and legacy dimer translators.

This module owns the ASE old/new state-layout shim, curvature-guard adapter,
and the historical ASE-backed Dimer/MMF translation implementations.  It is
an implementation split only: :mod:`saddlemill.dimertools.lbfgs_dimer` keeps
the legacy import surface and re-exports these exact objects.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import warnings

import numpy as np

from ase.optimize import FIRE, LBFGS
from ase.mep.dimer import MinModeTranslate

from saddlemill.dimertools.canonical_diagnostics import (
    REGULARIZATION_NOT_APPLICABLE,
    regularization_diagnostic_fields,
)
from saddlemill.dimertools.dense_bfgs import (
    DenseSecant,
    compare_directions,
    reconstruct_sequential_bfgs,
    shifted_trust_region_solve,
    safe_max_atom_norm,
    safe_norm,
)
from saddlemill.dimertools.qn_deep_diagnostics import (
    adaptive_shifted_lbfgs,
    shifted_lbfgs_solve,
    rigid_translation_basis,
    secant_row,
    progressive_lbfgs,
    translation_ratio,
)
from saddlemill.dimertools.lbfgs_state_dump import (
    rigid_translation_basis,
    step_selected,
    trace_two_loop,
)

norm = np.linalg.norm

# Minimum directional secant curvature (eV/A^2) accepted along a pair before
# an enabled safeguard intervenes.  Dimer translation keeps its historical
# default ``damp`` behavior; ordinary minimizers explicitly default safeguards
# to ``off`` in their own configs.
DEFAULT_CURVATURE_FLOOR = 1.0e-3
VALID_CURVATURE_GUARDS = frozenset({"off", "skip", "damp", "powell", "reset", "cautious"})
DEFAULT_POWELL_ETA = 0.2

def _ase_lbfgs_container(optimizer):
    """Return the object that owns ASE L-BFGS history attributes.

    ASE <= 3.28 stores ``iteration/s/y/rho/H0`` directly on ``LBFGS``.
    ASE >= 3.29 stores them on ``LBFGS.state``.  SaddleMill supports both
    layouts here while still delegating every actual L-BFGS step/update to ASE.
    """
    return getattr(optimizer, "state", optimizer)


def _ase_lbfgs_api_name(optimizer):
    return "state_object" if hasattr(optimizer, "state") else "legacy_direct"


def _ase_lbfgs_get(optimizer, name, default=None):
    return getattr(_ase_lbfgs_container(optimizer), name, default)


def _ase_lbfgs_set(optimizer, name, value):
    setattr(_ase_lbfgs_container(optimizer), name, value)


def _ase_lbfgs_increment_iteration(optimizer):
    _ase_lbfgs_set(
        optimizer,
        "iteration",
        int(_ase_lbfgs_get(optimizer, "iteration", 0)) + 1,
    )


def _ase_lbfgs_history_size(optimizer):
    return int(len(_ase_lbfgs_get(optimizer, "s", [])))


def _ase_lbfgs_pairs_total(optimizer):
    # The first ASE L-BFGS iteration has no secant pair.
    return int(max(0, int(_ase_lbfgs_get(optimizer, "iteration", 0)) - 1))


def _secant_pair_metrics(s, y):
    """Return scalar diagnostics for one L-BFGS secant pair.

    ``s`` is the accepted generalized-coordinate displacement and ``y`` is
    the corresponding gradient change.  These are read-only diagnostics; no
    optimizer state is modified.
    """
    if s is None or y is None:
        return {}
    s = np.asarray(s, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if s.shape != y.shape or s.size == 0:
        return {}
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
        return {}
    ss = float(np.dot(s, s))
    yy = float(np.dot(y, y))
    sy = float(np.dot(s, y))
    s_norm = safe_norm(s)
    y_norm = safe_norm(y)
    return {
        "s_norm": s_norm,
        "y_norm": y_norm,
        "s_dot_y": sy,
        "secant_curvature": (sy / ss) if ss > 0.0 else "",
        "secant_cosine": (
            sy / (s_norm * y_norm)
            if s_norm > 0.0 and y_norm > 0.0
            else ""
        ),
        "force_change_norm": y_norm,
    }


def _ase_lbfgs_latest_pair_metrics(optimizer):
    s_hist = list(_ase_lbfgs_get(optimizer, "s", []) or [])
    y_hist = list(_ase_lbfgs_get(optimizer, "y", []) or [])
    if not s_hist or not y_hist:
        return {}
    return _secant_pair_metrics(s_hist[-1], y_hist[-1])


def _ase_lbfgs_diagnostic_metrics(optimizer):
    """Read native ASE L-BFGS history/scaling without mutating it."""
    latest = _ase_lbfgs_latest_pair_metrics(optimizer)
    h0 = _ase_lbfgs_get(optimizer, "H0", "")
    try:
        h0 = float(h0)
    except (TypeError, ValueError):
        h0 = ""
    if h0 != "" and h0 != 0.0:
        alpha = 1.0 / h0
    else:
        alpha = ""
    memory = _ase_lbfgs_get(optimizer, "memory", getattr(optimizer, "memory", ""))
    damping = getattr(optimizer, "damping", "")
    return {
        "history_size": _ase_lbfgs_history_size(optimizer),
        "pairs_accepted_total": _ase_lbfgs_pairs_total(optimizer),
        "alpha": alpha,
        "initial_inverse_hessian_scale": h0,
        "memory": memory,
        "damping": damping,
        "latest_s_norm": latest.get("s_norm", ""),
        "latest_y_norm": latest.get("y_norm", ""),
        "latest_s_dot_y": latest.get("s_dot_y", ""),
        "latest_secant_curvature": latest.get("secant_curvature", ""),
        "latest_secant_cosine": latest.get("secant_cosine", ""),
        "latest_force_change_norm": latest.get("force_change_norm", ""),
        "latest_pair_damped": 0,
        "latest_raw_s_dot_y": latest.get("s_dot_y", ""),
        "latest_raw_secant_curvature": latest.get("secant_curvature", ""),
        "latest_stored_s_dot_y": latest.get("s_dot_y", ""),
        "latest_stored_secant_curvature": latest.get("secant_curvature", ""),
    }


def _lbfgs_step_metrics(optimizable, raw_direction, displacement, maxstep, damping):
    """Describe ASE L-BFGS proposal, clipping, damping, and accepted move.

    ASE's L-BFGS max-step decision uses ``optimizable.gradient_norm`` on the
    raw quasi-Newton direction, then multiplies the clipped direction by
    ``damping``.  We report that native clipping metric separately from the
    Euclidean generalized-coordinate norm.
    """
    raw = np.asarray(raw_direction, dtype=float).reshape(-1)
    actual = np.asarray(displacement, dtype=float).reshape(-1)
    raw_norm = safe_norm(raw)
    actual_norm = safe_norm(actual)
    raw_max = safe_max_atom_norm(raw) if raw.size and raw.size % 3 == 0 else float(optimizable.gradient_norm(raw)) if raw.size else 0.0
    actual_max = safe_max_atom_norm(actual) if actual.size and actual.size % 3 == 0 else float(optimizable.gradient_norm(actual)) if actual.size else 0.0
    maxstep = float(maxstep)
    damping = float(damping)
    clipped = bool(raw_max >= maxstep)
    if raw_max > 0.0:
        clip_scale = min(1.0, maxstep / raw_max)
        applied_scale = actual_norm / raw_norm if raw_norm > 0.0 else ""
    else:
        clip_scale = 1.0
        applied_scale = ""
    return {
        "raw_step_norm": raw_norm,
        "raw_step_max": raw_max,
        "actual_step_norm": actual_norm,
        "actual_step_max": actual_max,
        "maxstep": maxstep,
        "step_clipped": int(clipped),
        "maxstep_rescaled": int(raw_max > maxstep),
        "clip_scale": float(clip_scale),
        "damping": damping,
        "applied_scale": applied_scale,
    }


def _ase_lbfgs_reset_history(optimizer, memory):
    """Reset native ASE history without replacing the optimizer wrapper.

    The state-object API needs a new LBFGSMethod instance; the legacy API uses
    direct lists/scalars.  ``H0`` is preserved in both cases.
    """
    if hasattr(optimizer, "state"):
        old_state = optimizer.state
        optimizer.state = type(old_state)(
            memory=int(memory),
            initial_inverse_hessian=old_state.H0,
        )
    else:
        optimizer.iteration = 0
        optimizer.s = []
        optimizer.y = []
        optimizer.rho = []
        # Legacy ASE stores H0 and memory directly on the optimizer.
        optimizer.memory = int(memory)

    optimizer.r0 = None
    optimizer.f0 = None
    optimizer.e0 = None
    optimizer.task = "START"


def _damp_secant_forces(pos, forces, r0, f0, curvature_floor):
    """Shift the secant pair ASE is about to store to a curvature floor.

    ASE forms ``s = pos - r0`` and ``y = (-forces) - (-f0) = f0 - forces``.
    When ``s.y`` falls below ``curvature_floor * |s|^2`` the pair is replaced
    by ``y' = y + theta*s`` with ``theta`` chosen so ``s.y' == floor``, which
    is achieved by handing ASE ``forces - theta*s`` instead of ``forces``.

    This is a modified-secant/curvature-floor damping rule.  It is deliberately
    *not* called classical Powell damping.  The separate ``powell`` guard below
    reconstructs the direct-Hessian action ``B_k s`` from the limited-memory
    pairs and applies Powell's standard damped-BFGS formula.

    Returns ``(forces_for_update, was_damped, raw_sy)``.
    """
    s = np.asarray(pos, dtype=float).reshape(-1) - np.asarray(
        r0, dtype=float
    ).reshape(-1)
    y = np.asarray(f0, dtype=float).reshape(-1) - np.asarray(
        forces, dtype=float
    ).reshape(-1)
    ss = float(np.dot(s, s))
    sy = float(np.dot(s, y))
    if not np.isfinite(ss) or not np.isfinite(sy) or ss <= 0.0:
        return forces, False, sy
    floor = float(curvature_floor) * ss
    if sy >= floor:
        return forces, False, sy
    theta = (floor - sy) / ss
    damped = np.asarray(forces, dtype=float).reshape(-1) - theta * s
    return damped.reshape(np.shape(forces)), True, sy


def _apply_lbfgs_direct_hessian(alpha, s_hist, y_hist, vector):
    """Apply the direct BFGS Hessian ``B_k`` represented by L-BFGS pairs.

    ASE stores an inverse-Hessian L-BFGS representation with ``H0 = I/alpha``.
    Powell damping is conventionally defined using the corresponding direct
    Hessian action ``B_k s``.  This helper reconstructs *only the action on one
    vector* by replaying the ordinary direct-BFGS rank-two updates from
    ``B0 = alpha I``.  It does not materialize a dense ``3N x 3N`` matrix.

    Cost is ``O(m^2 n)`` for configured history memory ``m`` and generalized-
    coordinate dimension ``n``; this helper does not assume a fixed memory
    value.  Stored pairs are assumed chronological and positive-curvature.
    Returns ``None`` if the representation is numerically invalid.
    """
    v = np.asarray(vector, dtype=float).reshape(-1)
    if v.size == 0 or not np.all(np.isfinite(v)):
        return None
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        return None
    pairs = [
        (np.asarray(si, dtype=float).reshape(-1),
         np.asarray(yi, dtype=float).reshape(-1))
        for si, yi in zip(list(s_hist or []), list(y_hist or []))
    ]
    if any(si.shape != v.shape or yi.shape != v.shape for si, yi in pairs):
        return None

    # Precompute B_i s_i for each stored update, then apply the same updates to v.
    update_data = []
    for i, (si, yi) in enumerate(pairs):
        if not np.all(np.isfinite(si)) or not np.all(np.isfinite(yi)):
            return None
        Bsi = alpha * si.copy()
        for sj, yj, Bsj, sjBsj, sjyj in update_data:
            Bsi = (
                Bsi
                - Bsj * (float(np.dot(sj, Bsi)) / sjBsj)
                + yj * (float(np.dot(yj, si)) / sjyj)
            )
        siBsi = float(np.dot(si, Bsi))
        siyi = float(np.dot(si, yi))
        if (
            not np.isfinite(siBsi)
            or not np.isfinite(siyi)
            or siBsi <= 0.0
            or siyi <= 0.0
        ):
            return None
        update_data.append((si, yi, Bsi, siBsi, siyi))

    Bv = alpha * v.copy()
    for si, yi, Bsi, siBsi, siyi in update_data:
        Bv = (
            Bv
            - Bsi * (float(np.dot(si, Bv)) / siBsi)
            + yi * (float(np.dot(yi, v)) / siyi)
        )
    return Bv


def _powell_damp_secant_forces(
    optimizer, pos, forces, r0, f0, powell_eta=DEFAULT_POWELL_ETA
):
    """Apply classical Powell damped BFGS to the pair ASE is about to store.

    Let ``s = x_{k+1}-x_k``, ``y = g_{k+1}-g_k`` and let ``B_k`` be the direct
    BFGS Hessian corresponding to ASE's L-BFGS inverse-Hessian history.  Powell
    damping replaces ``y`` by

    ``y_bar = theta*y + (1-theta)*B_k*s``

    when ``s.T*y < eta*s.T*B_k*s``, with

    ``theta = (1-eta)*sTBs / (sTBs - sTy)``.

    The standard choice is ``eta=0.2``.  This guarantees
    ``s.T*y_bar >= eta*s.T*B_k*s > 0`` when ``B_k`` is positive definite, so an
    ordinary BFGS/L-BFGS update preserves positive definiteness.  ASE expects
    physical forces in ``update``; because ``y = f0 - forces``, the modified
    force handed to ASE is ``f0 - y_bar``.

    Returns ``(forces_for_update, was_damped, metrics)``.  If ``B_k s`` cannot
    be reconstructed safely, ``metrics['valid']`` is false and the caller must
    reject/reset rather than storing an uncontrolled pair.
    """
    eta = float(powell_eta)
    s = np.asarray(pos, dtype=float).reshape(-1) - np.asarray(r0, dtype=float).reshape(-1)
    y = np.asarray(f0, dtype=float).reshape(-1) - np.asarray(forces, dtype=float).reshape(-1)
    sy = float(np.dot(s, y)) if s.size == y.size else float('nan')
    metrics = {
        'valid': False,
        'raw_s_dot_y': sy,
        's_dot_Bs': '',
        'theta': '',
        'eta': eta,
    }
    if (
        s.shape != y.shape
        or s.size == 0
        or not np.all(np.isfinite(s))
        or not np.all(np.isfinite(y))
        or not (0.0 < eta < 1.0)
    ):
        return forces, False, metrics
    container = _ase_lbfgs_container(optimizer)
    Bs = _apply_lbfgs_direct_hessian(
        getattr(optimizer, 'sm_alpha', None),
        list(getattr(container, 's', []) or []),
        list(getattr(container, 'y', []) or []),
        s,
    )
    if Bs is None:
        return forces, False, metrics
    sBs = float(np.dot(s, Bs))
    if not np.isfinite(sBs) or sBs <= 0.0 or not np.isfinite(sy):
        return forces, False, metrics
    metrics['valid'] = True
    metrics['s_dot_Bs'] = sBs
    threshold = eta * sBs
    if sy >= threshold:
        metrics['theta'] = 1.0
        return forces, False, metrics
    denom = sBs - sy
    if not np.isfinite(denom) or denom <= 0.0:
        metrics['valid'] = False
        return forces, False, metrics
    theta = (1.0 - eta) * sBs / denom
    if not np.isfinite(theta) or theta <= 0.0 or theta >= 1.0:
        metrics['valid'] = False
        return forces, False, metrics
    y_bar = theta * y + (1.0 - theta) * Bs
    stored_sy = float(np.dot(s, y_bar))
    if not np.isfinite(stored_sy) or stored_sy <= 0.0:
        metrics['valid'] = False
        return forces, False, metrics
    metrics['theta'] = theta
    metrics['stored_s_dot_y'] = stored_sy
    modified_forces = np.asarray(f0, dtype=float).reshape(-1) - y_bar
    return modified_forces.reshape(np.shape(forces)), True, metrics


class _CurvatureGuardedLBFGS(LBFGS):
    """ASE ``LBFGS`` with an optional secant-curvature safeguard.

    ``curvature_guard`` may be:

    ``off``
        Store exactly the pair ASE would store.  This is stock ASE behavior.
    ``skip``
        Reject a pair whose directional curvature is below the configured
        floor, while preserving older history.
    ``damp``
        Shift ``y`` along ``s`` so ``s.y == curvature_floor * s.s`` and store
        the modified positive-curvature pair.  This is the historical
        SaddleMill shifted-secant rule, not Powell damping.
    ``powell``
        Apply classical Powell damped BFGS using the implicit direct-Hessian
        action ``B_k s`` reconstructed from the stored L-BFGS pairs.
    ``reset``
        Reject the bad pair and clear all older L-BFGS pairs before computing
        the next direction from the positive-definite initial ``H0``.

    Only history-update handling is changed.  Direction generation, initial
    scaling, line search/max-step handling, and the two-loop recursion remain
    ASE's.
    """

    def __init__(
        self,
        *args,
        curvature_guard="damp",
        curvature_floor=DEFAULT_CURVATURE_FLOOR,
        powell_eta=DEFAULT_POWELL_ETA,
        cautious_epsilon=1.0e-6,
        cautious_alpha=1.0,
        **kwargs,
    ):
        curvature_guard = str(curvature_guard).strip().lower()
        if curvature_guard not in VALID_CURVATURE_GUARDS:
            raise ValueError(
                "curvature_guard must be one of "
                f"{sorted(VALID_CURVATURE_GUARDS)}; got {curvature_guard!r}"
            )
        self.curvature_floor = float(curvature_floor)
        if curvature_guard in {"skip", "damp", "reset"} and self.curvature_floor <= 0.0:
            raise ValueError(
                "curvature_floor must be > 0 for skip/damp/reset guards"
            )
        self.powell_eta = float(powell_eta)
        self.cautious_epsilon = float(cautious_epsilon)
        self.cautious_alpha = float(cautious_alpha)
        if curvature_guard == "cautious" and (self.cautious_epsilon <= 0.0 or self.cautious_alpha < 0.0):
            raise ValueError("cautious_epsilon >0 and cautious_alpha >=0 required")
        if curvature_guard == "powell" and not (0.0 < self.powell_eta < 1.0):
            raise ValueError("powell_eta must satisfy 0 < powell_eta < 1")
        self.curvature_guard = curvature_guard
        alpha = kwargs.get("alpha", 70.0)
        self.sm_alpha = 70.0 if alpha is None else float(alpha)
        self.sm_pairs_damped = 0
        self.sm_pairs_powell_damped = 0
        self.sm_pairs_skipped = 0
        self.sm_pairs_accepted = 0
        self.sm_pairs_seen = 0
        self.sm_curvature_resets = 0
        self.sm_worst_sy = float("inf")
        self.sm_last_raw_pair_metrics = {}
        self.sm_last_stored_pair_metrics = {}
        self.sm_last_pair_damped = 0
        self.sm_last_guard_action = ""
        self.sm_last_powell_theta = ""
        self.sm_last_powell_s_dot_Bs = ""
        self.sm_last_raw_direction = None
        self.sm_last_two_loop_direction = None
        self.sm_last_candidate_s = None
        self.sm_last_candidate_y_raw = None
        self.sm_last_candidate_y_stored = None
        super().__init__(*args, **kwargs)

    def determine_step(self, dr):
        # Exact diagnostic capture before any SaddleMill confinement/max-step.
        self.sm_last_two_loop_direction = (
            np.asarray(dr, dtype=float).reshape(-1).copy()
            if bool(getattr(self, "_sm_state_dump", False)) else None
        )
        # Bowl breakout is a displacement confinement, not merely a force
        # projection. Mask the quasi-Newton proposal before ASE's max-step
        # rescaling so allowed active-atom motion keeps the usual step-size
        # semantics.
        dimeratoms = getattr(self.optimizable, "dimeratoms", None)
        if dimeratoms is not None and hasattr(
            dimeratoms, "confine_translation_vector"
        ):
            dr = dimeratoms.confine_translation_vector(dr)
        # ASE's determine_step() rescales ``dr`` in place, and ``dr`` is
        # normally ``self.p``. Copy it here so diagnostics retain the true
        # pre-maxstep quasi-Newton proposal without changing the step.
        self.sm_last_raw_direction = np.asarray(dr, dtype=float).reshape(-1).copy()
        flat = np.asarray(dr, dtype=float).reshape(-1)
        if flat.size % 3 == 0:
            raw_max = safe_max_atom_norm(flat)
            if not np.isfinite(raw_max):
                raise FloatingPointError(
                    "L-BFGS proposal has a non-finite derived max-atom norm"
                )
            if raw_max >= self.maxstep and raw_max > 0.0:
                scale = float(self.maxstep) / raw_max
                if not np.isfinite(scale) or scale <= 0.0:
                    raise FloatingPointError("invalid overflow-safe L-BFGS max-step scale")
                dr *= scale
            return dr
        return super().determine_step(dr)

    def line_search(self, *args, **kwargs):
        # ASE line-search L-BFGS bypasses determine_step(). Preserve the native
        # quasi-Newton direction for diagnostics without changing the search.
        self.sm_last_raw_direction = np.asarray(self.p, dtype=float).reshape(-1).copy()
        return super().line_search(*args, **kwargs)

    def sm_reset_pair_diagnostics(self):
        self.sm_last_raw_pair_metrics = {}
        self.sm_last_stored_pair_metrics = {}
        self.sm_last_pair_damped = 0
        self.sm_last_guard_action = ""
        self.sm_last_powell_theta = ""
        self.sm_last_powell_s_dot_Bs = ""

    @staticmethod
    def _decrement_iteration_for_skipped_pair(optimizer):
        # ASE's LBFGSMethod.compute_step indexes history using ``iteration``.
        # A rejected pair therefore requires rolling this counter back by one
        # so it continues to mean ``stored_pairs + initial_step``.
        iteration = int(_ase_lbfgs_get(optimizer, "iteration", 0))
        if iteration > 0:
            _ase_lbfgs_set(optimizer, "iteration", iteration - 1)

    def update(self, pos, forces, r0, f0):
        container = _ase_lbfgs_container(self)
        pair_expected = int(getattr(container, "iteration", 0)) > 0 and r0 is not None
        self.sm_last_pair_damped = 0
        self.sm_last_guard_action = ""
        self.sm_last_powell_theta = ""
        self.sm_last_powell_s_dot_Bs = ""
        self.sm_last_stored_pair_metrics = {}
        self.sm_last_candidate_s = None
        self.sm_last_candidate_y_raw = None
        self.sm_last_candidate_y_stored = None
        if pair_expected:
            s_raw = np.asarray(pos, dtype=float).reshape(-1) - np.asarray(
                r0, dtype=float
            ).reshape(-1)
            y_raw = np.asarray(f0, dtype=float).reshape(-1) - np.asarray(
                forces, dtype=float
            ).reshape(-1)
            if bool(getattr(self, "_sm_state_dump", False)):
                self.sm_last_candidate_s = np.asarray(s_raw, dtype=float).reshape(-1).copy()
                self.sm_last_candidate_y_raw = np.asarray(y_raw, dtype=float).reshape(-1).copy()
                self.sm_last_candidate_y_stored = self.sm_last_candidate_y_raw.copy()
            self.sm_last_raw_pair_metrics = _secant_pair_metrics(s_raw, y_raw)
            self.sm_pairs_seen += 1
            ss = float(np.dot(s_raw, s_raw))
            raw_sy = float(np.dot(s_raw, y_raw))
            pair_vectors_finite = bool(
                np.all(np.isfinite(s_raw)) and np.all(np.isfinite(y_raw))
            )
            if np.isfinite(raw_sy):
                self.sm_worst_sy = min(self.sm_worst_sy, raw_sy)

            if self.curvature_guard == "off":
                self.sm_last_guard_action = "off_accept"
                super().update(pos, forces, r0, f0)
                self.sm_pairs_accepted += 1
                self.sm_last_stored_pair_metrics = _ase_lbfgs_latest_pair_metrics(self)
                if bool(getattr(self, "_sm_state_dump", False)):
                    _ys = list(_ase_lbfgs_get(self, "y", []) or [])
                    if _ys:
                        self.sm_last_candidate_y_stored = np.asarray(_ys[-1], dtype=float).reshape(-1).copy()
                return

            if self.curvature_guard == "powell":
                powell_forces, powell_damped, pm = _powell_damp_secant_forces(
                    self, pos, forces, r0, f0, self.powell_eta
                )
                if not pm.get("valid", False):
                    # Fail closed: without a valid B_k s action we cannot claim
                    # Powell's positive-definiteness guarantee.  Drop the pair
                    # but keep prior positive-curvature history.
                    self.sm_pairs_skipped += 1
                    self.sm_last_guard_action = "powell_skip_invalid"
                    self._decrement_iteration_for_skipped_pair(self)
                    return
                self.sm_last_powell_theta = pm.get("theta", "")
                self.sm_last_powell_s_dot_Bs = pm.get("s_dot_Bs", "")
                forces = powell_forces
                if powell_damped:
                    self.sm_pairs_damped += 1
                    self.sm_pairs_powell_damped += 1
                    self.sm_last_pair_damped = 1
                    self.sm_last_guard_action = "powell_damp"
                else:
                    self.sm_last_guard_action = "powell_accept"
                super().update(pos, forces, r0, f0)
                self.sm_pairs_accepted += 1
                self.sm_last_stored_pair_metrics = _ase_lbfgs_latest_pair_metrics(self)
                if bool(getattr(self, "_sm_state_dump", False)):
                    _ys = list(_ase_lbfgs_get(self, "y", []) or [])
                    if _ys:
                        self.sm_last_candidate_y_stored = np.asarray(_ys[-1], dtype=float).reshape(-1).copy()
                return

            if self.curvature_guard == "cautious":
                grad_norm = safe_norm(np.asarray(f0, dtype=float).reshape(-1))
                required_curvature = self.cautious_epsilon * (grad_norm ** self.cautious_alpha)
                pair_good = (pair_vectors_finite and np.isfinite(ss) and np.isfinite(raw_sy) and ss > 0.0 and raw_sy / ss >= required_curvature)
                if pair_good:
                    self.sm_last_guard_action = "cautious_accept"
                    super().update(pos, forces, r0, f0); self.sm_pairs_accepted += 1
                    self.sm_last_stored_pair_metrics = _ase_lbfgs_latest_pair_metrics(self); return
                self.sm_pairs_skipped += 1; self.sm_last_guard_action = "cautious_skip"
                self._decrement_iteration_for_skipped_pair(self); return

            threshold = self.curvature_floor * ss
            pair_good = (
                pair_vectors_finite
                and np.isfinite(ss)
                and np.isfinite(raw_sy)
                and ss > 0.0
                and raw_sy >= threshold
            )
            if pair_good:
                self.sm_last_guard_action = "accept"
                super().update(pos, forces, r0, f0)
                self.sm_pairs_accepted += 1
                self.sm_last_stored_pair_metrics = _ase_lbfgs_latest_pair_metrics(self)
                if bool(getattr(self, "_sm_state_dump", False)):
                    _ys = list(_ase_lbfgs_get(self, "y", []) or [])
                    if _ys:
                        self.sm_last_candidate_y_stored = np.asarray(_ys[-1], dtype=float).reshape(-1).copy()
                return

            if (
                self.curvature_guard == "damp"
                and pair_vectors_finite
                and np.isfinite(ss)
                and np.isfinite(raw_sy)
                and ss > 0.0
            ):
                forces, damped, _ = _damp_secant_forces(
                    pos, forces, r0, f0, self.curvature_floor
                )
                if not damped:
                    # This branch is defensive; a pair classified as bad above
                    # should always be damped when ``ss`` is finite/positive.
                    raise RuntimeError("curvature damping failed to modify a bad pair")
                self.sm_pairs_damped += 1
                self.sm_last_pair_damped = 1
                self.sm_last_guard_action = "damp"
                super().update(pos, forces, r0, f0)
                self.sm_pairs_accepted += 1
                self.sm_last_stored_pair_metrics = _ase_lbfgs_latest_pair_metrics(self)
                if bool(getattr(self, "_sm_state_dump", False)):
                    _ys = list(_ase_lbfgs_get(self, "y", []) or [])
                    if _ys:
                        self.sm_last_candidate_y_stored = np.asarray(_ys[-1], dtype=float).reshape(-1).copy()
                return

            if self.curvature_guard == "reset":
                self.sm_pairs_skipped += 1
                self.sm_curvature_resets += 1
                self.sm_last_guard_action = "reset"
                memory = int(_ase_lbfgs_get(self, "memory", getattr(self, "memory", 100)))
                _ase_lbfgs_reset_history(self, memory)
                return

            # ``skip`` and the degenerate/non-finite fallback for ``damp`` both
            # reject this pair while preserving existing usable history.
            self.sm_pairs_skipped += 1
            self.sm_last_guard_action = "skip"
            self._decrement_iteration_for_skipped_pair(self)
            return

        # Initial step: ASE has no secant pair yet.
        super().update(pos, forces, r0, f0)

    def sm_guard_metrics(self):
        return {
            "guard_mode": self.curvature_guard,
            "curvature_floor": float(self.curvature_floor),
            "powell_eta": float(self.powell_eta),
            "pairs_seen": int(self.sm_pairs_seen),
            "pairs_accepted": int(self.sm_pairs_accepted),
            "pairs_skipped": int(self.sm_pairs_skipped),
            "pairs_damped": int(self.sm_pairs_damped),
            "pairs_powell_damped": int(self.sm_pairs_powell_damped),
            "curvature_resets": int(self.sm_curvature_resets),
            "worst_sy": (
                float(self.sm_worst_sy)
                if np.isfinite(self.sm_worst_sy)
                else ""
            ),
            "last_pair_damped": int(self.sm_last_pair_damped),
            "last_guard_action": self.sm_last_guard_action,
            "last_powell_theta": self.sm_last_powell_theta,
            "last_powell_s_dot_Bs": self.sm_last_powell_s_dot_Bs,
            "last_raw_pair": dict(self.sm_last_raw_pair_metrics),
            "last_stored_pair": dict(self.sm_last_stored_pair_metrics),
        }


def _translation_state_key(dimeratoms):
    """Identity of the effective force field currently being optimized.

    Both the kappa gamma regime and ASE's curvature-sign branch in
    ``get_projected_forces`` change which function the optimizer is following.
    History accumulated under one is meaningless under the other.
    """
    regime = str(getattr(dimeratoms, "translation_regime", "standard"))
    try:
        curvature = float(dimeratoms.get_curvature())
    except Exception:
        curvature = -1.0
    branch = "convex" if curvature > 0.0 else "concave"
    return f"{regime}:{branch}"



def _force_calls(dimeratoms) -> int:
    try:
        return int(dimeratoms.control.get_counter("forcecalls"))
    except Exception:
        return -1


def _projected_fmax(force) -> float:
    force = np.asarray(force, dtype=float)
    if force.size == 0:
        return 0.0
    if force.ndim == 1:
        force = force.reshape(-1, 3)
    return float(np.sqrt((force * force).sum(axis=1).max()))


def _real_fmax(dimeratoms) -> float:
    """True atomic fmax, independent of any projection or gamma scaling.

    ``MinModeTranslate`` converges on the projected force.  Under the kappa
    gamma scaling, and under ASE's curvature-sign branch, the projected force
    can vanish while the real force is large, which registers as a converged
    run that is not a stationary point.  Recorded here so the condition is at
    least visible in the optimizer diagnostics.
    """
    try:
        real = np.asarray(dimeratoms.get_forces(real=True), dtype=float)
    except Exception:
        return float("nan")
    if real.size == 0:
        return 0.0
    return float(np.sqrt((real * real).sum(axis=1).max()))


def _cosine_alignment(direction, force):
    direction = np.asarray(direction, dtype=float).reshape(-1)
    force = np.asarray(force, dtype=float).reshape(-1)
    denom = float(norm(direction) * norm(force))
    if denom <= 0.0:
        return ""
    return float(np.dot(direction, force) / denom)


def _flatten_rotation_diagnostics(dimeratoms):
    raw = getattr(dimeratoms, "last_rotation_diagnostics", {}) or {}
    phase_a = raw.get("phase_a", {}) or {}
    phase_b = raw.get("phase_b", {}) or {}
    data = {
        "rotation_optimizer": phase_a.get("optimizer", "ase"),
        "rotation_steps": phase_a.get(
            "rotations", dimeratoms.control.get_counter("rotcount")
        ),
        "rotation_lbfgs_history_size": phase_a.get("history_size", 0),
        "rotation_lbfgs_pairs_accepted": phase_a.get("pairs_accepted", 0),
        "rotation_lbfgs_pairs_rejected": phase_a.get("pairs_rejected", 0),
        "rotation_lbfgs_resets": phase_a.get("history_resets", 0),
        "rotation_lbfgs_fallbacks": phase_a.get("direction_fallbacks", 0),
        "rotation_lbfgs_memory": phase_a.get("memory", ""),
        "rotation_lbfgs_initial_hessian": phase_a.get("initial_hessian", ""),
        "rotation_lbfgs_initial_inverse_hessian_scale": phase_a.get(
            "initial_inverse_hessian_scale", ""
        ),
        "rotation_lbfgs_latest_pair_accepted": phase_a.get("latest_pair_accepted", ""),
        "rotation_lbfgs_latest_s_norm": phase_a.get("latest_s_norm", ""),
        "rotation_lbfgs_latest_y_norm": phase_a.get("latest_y_norm", ""),
        "rotation_lbfgs_latest_s_dot_y": phase_a.get("latest_s_dot_y", ""),
        "rotation_lbfgs_latest_secant_curvature": phase_a.get(
            "latest_secant_curvature", ""
        ),
        "rotation_lbfgs_latest_secant_cosine": phase_a.get("latest_secant_cosine", ""),
        "rotation_lbfgs_latest_force_change_norm": phase_a.get(
            "latest_force_change_norm", ""
        ),
        "kappa_rotation_optimizer": phase_b.get("optimizer", ""),
        "kappa_rotation_steps": phase_b.get("rotations", ""),
        "kappa_rotation_lbfgs_pairs_accepted": phase_b.get("pairs_accepted", ""),
        "kappa_rotation_lbfgs_pairs_rejected": phase_b.get("pairs_rejected", ""),
        "kappa_rotation_lbfgs_memory": phase_b.get("memory", ""),
        "kappa_rotation_lbfgs_initial_hessian": phase_b.get("initial_hessian", ""),
        "kappa_rotation_lbfgs_initial_inverse_hessian_scale": phase_b.get(
            "initial_inverse_hessian_scale", ""
        ),
        "kappa_rotation_lbfgs_latest_pair_accepted": phase_b.get(
            "latest_pair_accepted", ""
        ),
        "kappa_rotation_lbfgs_latest_s_norm": phase_b.get("latest_s_norm", ""),
        "kappa_rotation_lbfgs_latest_y_norm": phase_b.get("latest_y_norm", ""),
        "kappa_rotation_lbfgs_latest_s_dot_y": phase_b.get("latest_s_dot_y", ""),
        "kappa_rotation_lbfgs_latest_secant_curvature": phase_b.get(
            "latest_secant_curvature", ""
        ),
        "kappa_rotation_lbfgs_latest_secant_cosine": phase_b.get(
            "latest_secant_cosine", ""
        ),
        "kappa_rotation_lbfgs_latest_force_change_norm": phase_b.get(
            "latest_force_change_norm", ""
        ),
    }
    for key, value in phase_a.items():
        token = str(key)
        if token.startswith("canonical_rotation_") or token.startswith(
            "rotation_lbfgs_"
        ):
            data[token] = value
    return data


def _flatten_mode_predictor_diagnostics(dimeratoms):
    runtime = getattr(dimeratoms, "wave_b_runtime", None)
    schedule = {} if runtime is None else dict(getattr(runtime, "last_schedule_metadata", {}) or {})
    keys = (
        "mode_predictor_selector",
        "mode_predictor_requested_angle_radians", "mode_predictor_requested_angle_degrees",
        "mode_predictor_accepted_angle_radians", "mode_predictor_accepted_angle_degrees",
        "mode_predictor_capped", "mode_predictor_raw_tangent_norm",
        "mode_predictor_original_absolute_alignment", "mode_predictor_displacement_norm",
        "mode_predictor_gradient_change_norm", "mode_predictor_secant_action_norm",
        "mode_predictor_effective_angular_step_scale", "mode_predictor_preconditioned",
        "mode_predictor_fallback", "mode_predictor_prediction_cost_pes_calls",
        "mode_predictor_forcebank_torque_samples", "mode_predictor_forcebank_pair_sources",
        "mode_predictor_forcebank_accepted_force_source",
        "mode_predictor_forcebank_trial_pairs_future_only",
        "mode_predictor_forcebank_dynamic_h0", "mode_predictor_forcebank_pair_candidates",
        "mode_predictor_forcebank_pairs_built", "mode_predictor_forcebank_pairs_degenerate",
        "mode_predictor_forcebank_states_contributing", "mode_predictor_forcebank_build_ns",
        "mode_predictor_lbfgs_pairs_used", "mode_predictor_lbfgs_pairs_admissible_before_max_pairs",
        "mode_predictor_lbfgs_pairs_rejected_transport", "mode_predictor_lbfgs_pairs_rejected_curvature",
        "mode_predictor_lbfgs_pairs_rejected_cosine", "mode_predictor_lbfgs_pairs_damped",
        "mode_predictor_lbfgs_h0_inverse_scale", "mode_predictor_lbfgs_raw_direction_norm",
        "mode_predictor_lbfgs_direction_norm", "mode_predictor_lbfgs_direction_surrogate_cosine",
        "mode_predictor_lbfgs_apply_ns",
    )
    out = {key: schedule.get(key, "") for key in keys}
    out["mode_predictor_status"] = schedule.get("predictor_status", "")
    out["mode_predictor_disposition"] = schedule.get("predictor_disposition", "")
    out["mode_schedule_decision"] = schedule.get("decision", "")
    out["mode_schedule_require_real_solve"] = int(bool(schedule.get("require_real_solve", False))) if schedule else ""
    out["mode_schedule_skipped_scheduled_solve"] = int(bool(schedule.get("skipped_scheduled_solve", False))) if schedule else ""
    return out


class _RealForceConvergenceMixin:
    """Use true atomic ``fmax`` as the non-Sella Dimer stop criterion.

    The optimizer stops at the first geometry whose *real* (unprojected)
    atomic forces satisfy the configured runtime ``fmax``.  This is a
    stationarity test, the same quantity Sella's ``PES.converged`` uses.

    The projected (Dimer/MMF-modified) force is deliberately NOT the stop
    criterion: under ASE's curvature-sign branch the projected force keeps
    only the component along the mode when curvature > 0, so it can vanish
    while real forces are large (a non-stationary geometry).  Curvature is not
    a stop gate either: stationary points of any order end the attempt and the
    downstream Hessian stage classifies them.

    Final mode refreshes may update the reported mode/curvature at the
    converged geometry, but they cannot veto convergence or force another
    translation step.
    """

    def gradient_converged(self, gradient):
        fmax = getattr(self, "fmax", None)
        if fmax is None:
            # ``Optimizer.run(fmax=...)`` sets this before normal execution.
            # Preserve a safe fallback for direct/out-of-band calls.
            return super().gradient_converged(gradient)

        try:
            real = np.asarray(self.dimeratoms.get_forces(real=True), dtype=float)
        except Exception as exc:
            raise RuntimeError(
                "Unable to evaluate real force for Dimer convergence"
            ) from exc
        if real.size == 0:
            real_fmax = 0.0
        else:
            real = real.reshape(-1, 3)
            real_fmax = float(np.sqrt((real * real).sum(axis=1).max()))
        if not np.isfinite(real_fmax):
            raise RuntimeError(f"non-finite real fmax in Dimer convergence: {real_fmax!r}")
        if real_fmax >= float(fmax):
            return False

        # Diagnostic/final-state maintenance only; cannot veto convergence.
        refresh = getattr(
            self.dimeratoms, "refresh_reused_mode_for_convergence", None
        )
        if callable(refresh):
            refresh()

        fresh_validation = getattr(
            self.dimeratoms, "ensure_wave_b_fresh_validation", None
        )
        if callable(fresh_validation):
            fresh_validation()

        return True

class _TranslationDiagnosticsMixin:
    def _initialize_step_diagnostics(self):
        self.last_step_diagnostics = None
        self._diagnostic_serial = 0
        self._previous_force_calls_after_step = 0

    def _start_step_diagnostics(self, force, algorithm, hybrid_state="", switch_event=""):
        entry_calls = _force_calls(self.dimeratoms)
        center_rotation_calls = (
            entry_calls - self._previous_force_calls_after_step
            if entry_calls >= 0
            else ""
        )
        data = {
            "diagnostic_serial": self._diagnostic_serial + 1,
            "accepted_translation_step": int(self.nsteps) + 1,
            "translation_algorithm": algorithm,
            "hybrid_state": hybrid_state,
            "hybrid_switch_event": switch_event,
            "projected_fmax": _projected_fmax(force),
            "real_fmax": _real_fmax(self.dimeratoms),
            "curvature": float(self.dimeratoms.get_curvature()),
            "translation_regime": getattr(
                self.dimeratoms, "translation_regime", "standard"
            ),
            "translation_state_key": _translation_state_key(self.dimeratoms),
            "force_calls_step_entry": entry_calls,
            "force_calls_center_and_rotation": center_rotation_calls,
        }
        data.update(_flatten_rotation_diagnostics(self.dimeratoms))
        data.update(_flatten_mode_predictor_diagnostics(self.dimeratoms))
        return data

    def _finish_step_diagnostics(self, data, step_norm, lbfgs_metrics=None):
        after_calls = _force_calls(self.dimeratoms)
        entry_calls = data["force_calls_step_entry"]
        metrics = dict(lbfgs_metrics or {})
        data.update(
            {
                "step_norm": float(step_norm),
                "force_calls_translation_trial": (
                    after_calls - entry_calls
                    if after_calls >= 0 and entry_calls >= 0
                    else ""
                ),
                "force_calls_cumulative_after_step": after_calls,
                "translation_lbfgs_history_size": metrics.get("history_size", 0),
                "translation_lbfgs_pairs_accepted_total": metrics.get(
                    "pairs_accepted_total", 0
                ),
                "translation_lbfgs_pairs_rejected_total": metrics.get(
                    "pairs_rejected_total", 0
                ),
                "translation_lbfgs_pairs_damped_total": metrics.get(
                    "pairs_damped_total", 0
                ),
                "translation_lbfgs_pairs_powell_damped_total": metrics.get(
                    "pairs_powell_damped_total", 0
                ),
                "translation_lbfgs_curvature_guard": metrics.get(
                    "curvature_guard", ""
                ),
                "translation_lbfgs_curvature_floor": metrics.get(
                    "curvature_floor", ""
                ),
                "translation_lbfgs_powell_eta": metrics.get("powell_eta", ""),
                "translation_lbfgs_latest_guard_action": metrics.get(
                    "latest_guard_action", ""
                ),
                "translation_lbfgs_latest_powell_theta": metrics.get(
                    "latest_powell_theta", ""
                ),
                "translation_lbfgs_latest_s_dot_Bs": metrics.get(
                    "latest_s_dot_Bs", ""
                ),
                "translation_lbfgs_worst_sy": metrics.get("worst_sy", ""),
                "translation_lbfgs_resets": metrics.get("reset_count", 0),
                "translation_lbfgs_last_reset_reason": metrics.get(
                    "last_reset_reason", ""
                ),
                "translation_lbfgs_alpha": metrics.get("alpha", ""),
                "translation_lbfgs_initial_inverse_hessian_scale": metrics.get(
                    "initial_inverse_hessian_scale", ""
                ),
                "translation_lbfgs_memory": metrics.get("memory", ""),
                "translation_lbfgs_latest_s_norm": metrics.get("latest_s_norm", ""),
                "translation_lbfgs_latest_y_norm": metrics.get("latest_y_norm", ""),
                "translation_lbfgs_latest_s_dot_y": metrics.get("latest_s_dot_y", ""),
                "translation_lbfgs_latest_secant_curvature": metrics.get(
                    "latest_secant_curvature", ""
                ),
                "translation_lbfgs_latest_secant_cosine": metrics.get(
                    "latest_secant_cosine", ""
                ),
                "translation_lbfgs_latest_force_change_norm": metrics.get(
                    "latest_force_change_norm", ""
                ),
                "translation_lbfgs_latest_pair_damped": metrics.get(
                    "latest_pair_damped", ""
                ),
                "translation_lbfgs_latest_raw_s_dot_y": metrics.get(
                    "latest_raw_s_dot_y", ""
                ),
                "translation_lbfgs_latest_raw_secant_curvature": metrics.get(
                    "latest_raw_secant_curvature", ""
                ),
                "translation_lbfgs_latest_stored_s_dot_y": metrics.get(
                    "latest_stored_s_dot_y", ""
                ),
                "translation_lbfgs_latest_stored_secant_curvature": metrics.get(
                    "latest_stored_secant_curvature", ""
                ),
                "translation_dense_bfgs_diagnostic": metrics.get(
                    "dense_bfgs_diagnostic", 0
                ),
                "translation_lbfgs_raw_direction_norm": metrics.get(
                    "lbfgs_raw_direction_norm", ""
                ),
                "translation_lbfgs_raw_direction_max_atom_norm": metrics.get(
                    "lbfgs_raw_direction_max_atom_norm", ""
                ),
                "translation_dense_vs_lbfgs_cosine": metrics.get(
                    "dense_vs_lbfgs_cosine", ""
                ),
                "translation_dense_vs_lbfgs_norm_ratio": metrics.get(
                    "dense_vs_lbfgs_norm_ratio", ""
                ),
                "translation_dense_vs_lbfgs_relative_difference": metrics.get(
                    "dense_vs_lbfgs_relative_difference", ""
                ),
                "translation_dense_min_eigenvalue": metrics.get(
                    "dense_min_eigenvalue", ""
                ),
                "translation_dense_max_eigenvalue": metrics.get(
                    "dense_max_eigenvalue", ""
                ),
                "translation_dense_negative_eigenvalues": metrics.get(
                    "dense_negative_eigenvalues", ""
                ),
                "translation_dense_condition_number": metrics.get(
                    "dense_condition_number", ""
                ),
                "translation_dense_log10_condition": metrics.get(
                    "dense_log10_condition", ""
                ),
                "translation_dense_secant_residual_latest": metrics.get(
                    "dense_secant_residual_latest", ""
                ),
                "translation_dense_secant_residual_median": metrics.get(
                    "dense_secant_residual_median", ""
                ),
                "translation_dense_secant_residual_max": metrics.get(
                    "dense_secant_residual_max", ""
                ),
                "translation_dense_pairs_requested": metrics.get(
                    "dense_pairs_requested", ""
                ),
                "translation_dense_pairs_applied": metrics.get(
                    "dense_pairs_applied", ""
                ),
                "translation_dense_invalid_reason": metrics.get(
                    "dense_invalid_reason", ""
                ),
                "translation_dense_reconstruction_ns": metrics.get(
                    "dense_reconstruction_ns", ""
                ),
                **regularization_diagnostic_fields(metrics),
                "translation_trust_shift": metrics.get("trust_shift", ""),
                "translation_trust_radius": metrics.get("trust_radius", ""),
                "translation_trust_unshifted_boundary_norm": metrics.get("trust_unshifted_boundary_norm", ""),
                "translation_trust_regularized_boundary_norm": metrics.get("trust_regularized_boundary_norm", ""),
                "translation_trust_regularized_direction_norm": metrics.get("trust_regularized_direction_norm", ""),
                "translation_trust_direction_cosine_unshifted": metrics.get("trust_direction_cosine_unshifted", ""),
                "translation_trust_shifted_condition_number": metrics.get("trust_shifted_condition_number", ""),
                "translation_trust_iterations": metrics.get("trust_iterations", ""),
                "translation_trust_shift_expansions": metrics.get("trust_shift_expansions", ""),
                "translation_trust_solve_ns": metrics.get("trust_solve_ns", ""),
            }
        )
        self._diagnostic_serial += 1
        self.last_step_diagnostics = data
        self._previous_force_calls_after_step = max(after_calls, 0)


class DiagnosticMinModeTranslate(
    _RealForceConvergenceMixin, _TranslationDiagnosticsMixin, MinModeTranslate
):
    """Stock ASE ``MinModeTranslate`` with diagnostics only."""

    def __init__(self, dimeratoms, logfile="-", trajectory=None):
        super().__init__(dimeratoms, logfile=logfile, trajectory=trajectory)
        self._initialize_step_diagnostics()

    def step(self, f=None):
        if f is None:
            f = self.dimeratoms.get_forces()
        f = np.asarray(f, dtype=float)
        r_before = self.dimeratoms.get_positions().copy()
        algorithm = "ase_cg" if self.cg_on else "ase_steepest"
        data = self._start_step_diagnostics(f, algorithm)

        # Polak-Ribiere history can otherwise reintroduce motion on atoms whose
        # current bowl-breakout force is masked to zero.
        if self.cg_on:
            for name in ("direction_old", "cg_direction"):
                value = getattr(self, name, None)
                if value is not None and hasattr(
                    self.dimeratoms, "confine_translation_vector"
                ):
                    setattr(
                        self, name,
                        self.dimeratoms.confine_translation_vector(value),
                    )

        MinModeTranslate.step(self, f)
        proposed = self.dimeratoms.get_positions().copy()
        if hasattr(self.dimeratoms, "confine_translation_positions"):
            confined = self.dimeratoms.confine_translation_positions(
                r_before, proposed
            )
            if not np.array_equal(confined, proposed):
                self.dimeratoms.set_positions(confined)
        step_norm = norm(self.dimeratoms.get_positions() - r_before)
        self._finish_step_diagnostics(data, step_norm)


class _ASELBFGSState:
    """Thin state/diagnostic adapter around an actual ASE ``LBFGS`` object.

    No L-BFGS recursion is implemented here.  ``step()`` delegates to ASE.
    ``rebuild()`` mirrors ASE's own replay loop in generalized coordinates so
    warm starts also work for filters such as ``FrechetCellFilter``.
    """

    def __init__(self, target, *, maxstep, memory, damping, alpha,
                 curvature_guard="damp",
                 curvature_floor=DEFAULT_CURVATURE_FLOOR,
                 powell_eta=DEFAULT_POWELL_ETA,
                 use_line_search=False):
        self.target = target
        self.maxstep = float(maxstep)
        self.memory = int(memory)
        self.damping = float(damping)
        self.alpha = float(alpha)
        self.curvature_guard = str(curvature_guard).strip().lower()
        self.curvature_floor = float(curvature_floor)
        self.powell_eta = float(powell_eta)
        self.use_line_search = bool(use_line_search)
        self.optimizer = self._new_optimizer()
        self.reset_count = 0
        self.last_reset_reason = "initial"
        self.pairs_accepted_carry = 0
        self.pairs_skipped_carry = 0
        self.pairs_damped_carry = 0
        self.pairs_powell_damped_carry = 0
        self.pairs_seen_carry = 0
        self.curvature_resets_carry = 0
        self.worst_sy_carry = float("inf")

    def _new_optimizer(self):
        return _CurvatureGuardedLBFGS(
            self.target,
            restart=None,
            logfile=None,
            trajectory=None,
            maxstep=self.maxstep,
            memory=self.memory,
            damping=self.damping,
            alpha=self.alpha,
            use_line_search=self.use_line_search,
            curvature_guard=self.curvature_guard,
            curvature_floor=self.curvature_floor,
            powell_eta=self.powell_eta,
        )

    def _absorb_counters(self):
        """Carry guard counters across optimizer replacement."""
        g = self.optimizer.sm_guard_metrics()
        self.pairs_accepted_carry += int(g["pairs_accepted"])
        self.pairs_skipped_carry += int(g["pairs_skipped"])
        self.pairs_damped_carry += int(g["pairs_damped"])
        self.pairs_powell_damped_carry += int(g.get("pairs_powell_damped", 0))
        self.pairs_seen_carry += int(g["pairs_seen"])
        self.curvature_resets_carry += int(g["curvature_resets"])
        if g["worst_sy"] != "":
            self.worst_sy_carry = min(self.worst_sy_carry, float(g["worst_sy"]))

    def reset(self, reason="manual"):
        self._absorb_counters()
        self.optimizer = self._new_optimizer()
        self.reset_count += 1
        self.last_reset_reason = str(reason)

    @property
    def history_size(self):
        return _ase_lbfgs_history_size(self.optimizer)

    @property
    def pairs_accepted_total(self):
        g = self.optimizer.sm_guard_metrics()
        return self.pairs_accepted_carry + int(g["pairs_accepted"])

    def metrics(self):
        g = self.optimizer.sm_guard_metrics()
        accepted = self.pairs_accepted_carry + int(g["pairs_accepted"])
        skipped = self.pairs_skipped_carry + int(g["pairs_skipped"])
        damped = self.pairs_damped_carry + int(g["pairs_damped"])
        powell_damped = self.pairs_powell_damped_carry + int(
            g.get("pairs_powell_damped", 0)
        )
        curvature_resets = (
            self.curvature_resets_carry + int(g["curvature_resets"])
        )
        worst = self.worst_sy_carry
        if g["worst_sy"] != "":
            worst = min(worst, float(g["worst_sy"]))
        stored = _ase_lbfgs_latest_pair_metrics(self.optimizer)
        action = str(g.get("last_guard_action", ""))
        raw = dict(g.get("last_raw_pair", {}) or stored)
        if action in {"skip", "reset", "powell_skip_invalid"}:
            guarded_stored = {}
        else:
            guarded_stored = dict(g.get("last_stored_pair", {}) or stored)
        return {
            "history_size": self.history_size,
            "pairs_accepted_total": accepted,
            # Backward-compatible diagnostic: this historical field counted
            # damped pairs as "rejected" even though they were stored.  Keep
            # that value stable and use pairs_skipped_total for true drops.
            "pairs_rejected_total": skipped + damped,
            "pairs_skipped_total": skipped,
            "pairs_damped_total": damped,
            "pairs_powell_damped_total": powell_damped,
            "worst_sy": float(worst) if np.isfinite(worst) else "",
            "reset_count": self.reset_count + curvature_resets,
            "curvature_reset_count": curvature_resets,
            "last_reset_reason": (
                "curvature_guard"
                if g.get("last_guard_action", "") == "reset"
                else self.last_reset_reason
            ),
            "ase_lbfgs_api": _ase_lbfgs_api_name(self.optimizer),
            "guard_mode": g.get("guard_mode", self.curvature_guard),
            "curvature_floor": g.get("curvature_floor", self.curvature_floor),
            "powell_eta": g.get("powell_eta", self.powell_eta),
            "latest_guard_action": g.get("last_guard_action", ""),
            "latest_powell_theta": g.get("last_powell_theta", ""),
            "latest_powell_s_dot_Bs": g.get("last_powell_s_dot_Bs", ""),
            "alpha": float(self.alpha),
            "initial_inverse_hessian_scale": 1.0 / float(self.alpha),
            "memory": int(self.memory),
            "damping": float(self.damping),
            "use_line_search": int(self.use_line_search),
            "force_calls": int(getattr(self.optimizer, "force_calls", 0)),
            "function_calls": int(getattr(self.optimizer, "function_calls", 0)),
            "latest_s_norm": guarded_stored.get("s_norm", ""),
            "latest_y_norm": guarded_stored.get("y_norm", ""),
            "latest_s_dot_y": guarded_stored.get("s_dot_y", ""),
            "latest_secant_curvature": guarded_stored.get("secant_curvature", ""),
            "latest_secant_cosine": guarded_stored.get("secant_cosine", ""),
            "latest_force_change_norm": guarded_stored.get("force_change_norm", ""),
            "latest_pair_damped": int(g.get("last_pair_damped", 0)),
            "latest_raw_s_dot_y": raw.get("s_dot_y", ""),
            "latest_raw_secant_curvature": raw.get("secant_curvature", ""),
            "latest_stored_s_dot_y": guarded_stored.get("s_dot_y", ""),
            "latest_stored_secant_curvature": guarded_stored.get(
                "secant_curvature", ""
            ),
        }

    def rebuild(self, snapshots, reason="warm_start_rebuild"):
        """Rebuild ASE's native state from ``(x, projected_force)`` snapshots.

        The final snapshot is intentionally not replayed.  ASE's next real
        ``step()`` closes that last secant pair, exactly as its private
        ``_replay_trajectory`` implementation does for an ordinary trajectory.
        """
        self.reset(reason)
        opt = self.optimizer
        r0 = None
        f0 = None
        for position, force in list(snapshots)[:-1]:
            position = np.asarray(position, dtype=float).reshape(-1)
            force = np.asarray(force, dtype=float).reshape(-1)
            opt.update(position, force, r0, f0)
            r0 = position.copy()
            f0 = force.copy()
            _ase_lbfgs_increment_iteration(opt)
        opt.r0 = r0
        opt.f0 = f0

    def step(self, force):
        force = np.asarray(force, dtype=float)
        force_calls_before = int(getattr(self.optimizer, "force_calls", 0))
        function_calls_before = int(getattr(self.optimizer, "function_calls", 0))
        position_before = np.asarray(
            self.optimizer.optimizable.get_x(), dtype=float
        ).reshape(-1).copy()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Please do not pass forces to step\(\)\..*",
                category=UserWarning,
                module=r"ase\.optimize\.optimize",
            )
            self.optimizer.step(forces=force)
        position_after = np.asarray(
            self.optimizer.optimizable.get_x(), dtype=float
        ).reshape(-1)
        displacement = position_after - position_before
        raw_source = getattr(self.optimizer, "sm_last_raw_direction", None)
        if raw_source is None:
            raw_source = self.optimizer.p
        raw_direction = np.asarray(raw_source, dtype=float).reshape(-1)
        self.last_step_metrics = _lbfgs_step_metrics(
            self.optimizer.optimizable,
            raw_direction,
            displacement,
            self.optimizer.maxstep,
            self.optimizer.damping,
        )
        if self.use_line_search:
            self.last_step_metrics["step_clipped"] = ""
            self.last_step_metrics["maxstep_rescaled"] = ""
            self.last_step_metrics["clip_scale"] = ""
            self.last_step_metrics["damping"] = ""
        self.last_step_metrics["direction_alignment"] = _cosine_alignment(
            raw_direction, force
        )
        self.last_step_metrics["step_force_calls"] = (
            int(getattr(self.optimizer, "force_calls", 0)) - force_calls_before
        )
        self.last_step_metrics["step_function_calls"] = (
            int(getattr(self.optimizer, "function_calls", 0)) - function_calls_before
        )
        self.last_step_metrics["alpha_k"] = (
            getattr(self.optimizer, "alpha_k", "") if self.use_line_search else ""
        )
        return (
            float(self.last_step_metrics["actual_step_norm"]),
            (
                ""
                if self.last_step_metrics["step_clipped"] == ""
                else bool(self.last_step_metrics["step_clipped"])
            ),
            self.last_step_metrics["direction_alignment"],
        )


class _DimerLBFGSLogMixin:
    """Retain the dimer optimizer log columns while using ASE optimizers."""

    def _initialize_dimer_log(self, write_header=True):
        self._last_step_size = None
        if write_header and self.logfile is not None:
            self.logfile.write(
                "MinModeTranslate: STEP      TIME          ENERGY    "
                "MAX-FORCE     STEPSIZE    CURVATURE  ROT-STEPS\n"
            )

    def log(self, gradient):
        import time

        force = -np.asarray(gradient, dtype=float).reshape(-1, 3)
        fmax = _projected_fmax(force)
        energy = self.dimeratoms.get_potential_energy()
        curvature = self.dimeratoms.get_curvature()
        rotsteps = self.dimeratoms.control.get_counter("rotcount")
        if self.logfile is None:
            return
        now = time.localtime()
        if self._last_step_size is None:
            step_field = "    --------"
        else:
            step_field = f"{float(self._last_step_size):12.6f}"
        line = (
            f"MinModeTranslate: {self.nsteps:4d}  "
            f"{now[3]:02d}:{now[4]:02d}:{now[5]:02d} "
            f"{energy:15.6f} {fmax:12.4f} {step_field} "
            f"{curvature:12.6f} {rotsteps:10d}\n"
        )
        self.logfile.write(line)


class LBFGSMinModeTranslate(
    _RealForceConvergenceMixin,
    _TranslationDiagnosticsMixin,
    _DimerLBFGSLogMixin,
    _CurvatureGuardedLBFGS,
):
    """ASE ``LBFGS`` applied directly to ``MinModeAtoms`` projected forces."""

    def __init__(self, dimeratoms, logfile="-", trajectory=None, lbfgs_options=None):
        options = dict(lbfgs_options or {})
        self.dimeratoms = dimeratoms
        self.control = dimeratoms.get_control()
        self.reset_on_regime_change = bool(
            options.pop("reset_on_regime_change", True)
        )
        # Legacy custom-recursion knobs are accepted but intentionally ignored.
        options.pop("dynamic_h0", None)
        options.pop("curvature_epsilon", None)
        memory = int(options.pop("memory", 10))
        alpha = float(options.pop("initial_hessian", 70.0))
        damping = float(options.pop("damping", 1.0))
        curvature_guard = str(
            options.pop("curvature_guard", "skip")
        ).strip().lower()
        self._sm_dense_bfgs_diagnostic = bool(
            options.pop("dense_bfgs_diagnostic", False)
        )
        self._sm_deep_qn_diagnostics = bool(options.pop("deep_qn_diagnostics", False))
        self.last_qn_detail_payload = {}
        self._sm_regularization = str(
            options.pop("regularization", "off")
        ).strip().lower()
        self._sm_regularization_mu = float(options.pop("regularization_mu", 1.0))
        self._sm_regularization_radius = float(options.pop("regularization_radius", 0.1))
        self._sm_regularization_tolerance = float(
            options.pop("regularization_tolerance", 1.0e-8)
        )
        if self._sm_regularization not in {"off", "shifted_trust_region", "shifted_lbfgs_fixed", "shifted_lbfgs_trust"}:
            raise ValueError("invalid translation regularization")
        if self._sm_regularization_mu < 0.0 or self._sm_regularization_radius <= 0.0:
            raise ValueError("regularization_mu >=0 and regularization_radius >0 required")
        if self._sm_regularization_tolerance <= 0.0:
            raise ValueError("translation regularization_tolerance must be > 0")
        trial_step = str(options.pop("trial_step", "off")).strip().lower()
        if trial_step != "off":
            raise ValueError("translation_trial_step is only supported by canonical_lbfgs")
        from saddlemill.dimertools.wave_b_shadow import qn_shadow_options_from_dimeratoms
        self.qn_shadow_options = qn_shadow_options_from_dimeratoms(dimeratoms)
        self.last_qn_shadow_row = None
        self.last_qn_shadow_matrix = None
        self.qn_shadow_cumulative_ns = 0
        self._sm_state_dump_requested = bool(options.pop("state_dump", False))
        self._sm_state_dump = bool(
            self._sm_state_dump_requested or self.qn_shadow_options.get("enabled", False)
        )
        self._sm_state_dump_steps = str(options.pop("state_dump_steps", "all"))
        self._sm_state_dump_trace = bool(
            options.pop("state_dump_two_loop_trace", True)
        )
        self.lbfgs_state_dump_directory = str(
            options.pop("state_dump_directory", "")
        ).strip()
        self.last_lbfgs_state_dump = None

        # Driving reconstructed-BFGS belongs to the canonical/generalized
        # consumers; legacy ASE translation accepts these shared config keys
        # but remains native ASE-LBFGS.
        options.pop("reconstruction_model", None)
        options.pop("bfgs_update", None)
        curvature_floor = float(
            options.pop("curvature_floor", DEFAULT_CURVATURE_FLOOR)
        )
        powell_eta = float(options.pop("powell_eta", DEFAULT_POWELL_ETA))
        cautious_epsilon = float(options.pop("cautious_epsilon", 1.0e-6))
        cautious_alpha = float(options.pop("cautious_alpha", 1.0))
        if options:
            raise TypeError(f"Unknown ASE dimer L-BFGS options: {sorted(options)}")
        self._sm_memory = memory
        self._sm_alpha = alpha
        self._sm_damping = damping
        self._sm_state_key = None
        self._sm_reset_count = 0
        self._sm_last_reset_reason = "initial"
        self._sm_dense_metrics: dict[str, object] = {}
        _CurvatureGuardedLBFGS.__init__(
            self,
            dimeratoms,
            restart=None,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=float(self.control.get_parameter("maximum_translation")),
            memory=memory,
            damping=damping,
            alpha=alpha,
            use_line_search=False,
            curvature_guard=curvature_guard,
            curvature_floor=curvature_floor,
            powell_eta=powell_eta,
            cautious_epsilon=cautious_epsilon, cautious_alpha=cautious_alpha,
        )
        self._initialize_step_diagnostics()
        self._initialize_dimer_log()

    def _reset_native_history(self, reason):
        _ase_lbfgs_reset_history(self, self._sm_memory)
        self.sm_reset_pair_diagnostics()
        self._sm_reset_count += 1
        self._sm_last_reset_reason = str(reason)

    def _native_metrics(self):
        guard = self.sm_guard_metrics()
        stored = _ase_lbfgs_latest_pair_metrics(self)
        raw = dict(guard.get("last_raw_pair", {}) or stored)
        guarded_stored = dict(guard.get("last_stored_pair", {}) or stored)
        metrics = {
            "history_size": _ase_lbfgs_history_size(self),
            "pairs_accepted_total": _ase_lbfgs_pairs_total(self),
            "pairs_rejected_total": int(guard["pairs_skipped"]),
            "pairs_damped_total": int(guard["pairs_damped"]),
            "pairs_powell_damped_total": int(guard["pairs_powell_damped"]),
            "curvature_guard": guard["guard_mode"],
            "curvature_floor": guard["curvature_floor"],
            "powell_eta": guard["powell_eta"],
            "latest_guard_action": guard["last_guard_action"],
            "latest_powell_theta": guard["last_powell_theta"],
            "latest_s_dot_Bs": guard["last_powell_s_dot_Bs"],
            "worst_sy": guard["worst_sy"],
            "reset_count": self._sm_reset_count,
            "last_reset_reason": self._sm_last_reset_reason,
            "alpha": float(self._sm_alpha),
            "initial_inverse_hessian_scale": 1.0 / float(self._sm_alpha),
            "memory": int(self._sm_memory),
            "damping": float(self._sm_damping),
            "latest_s_norm": guarded_stored.get("s_norm", ""),
            "latest_y_norm": guarded_stored.get("y_norm", ""),
            "latest_s_dot_y": guarded_stored.get("s_dot_y", ""),
            "latest_secant_curvature": guarded_stored.get("secant_curvature", ""),
            "latest_secant_cosine": guarded_stored.get("secant_cosine", ""),
            "latest_force_change_norm": guarded_stored.get("force_change_norm", ""),
            "latest_pair_damped": int(guard.get("last_pair_damped", 0)),
            "latest_raw_s_dot_y": raw.get("s_dot_y", ""),
            "latest_raw_secant_curvature": raw.get("secant_curvature", ""),
            "latest_stored_s_dot_y": guarded_stored.get("s_dot_y", ""),
            "latest_stored_secant_curvature": guarded_stored.get(
                "secant_curvature", ""
            ),
            "dense_bfgs_diagnostic": int(self._sm_dense_bfgs_diagnostic),
        }
        metrics.update(self._sm_dense_metrics)
        return metrics

    def step(self, forces=None):
        accepted_translation_step = int(self.nsteps) + 1
        dump_this_step = bool(
            self._sm_state_dump_requested
            and step_selected(self._sm_state_dump_steps, accepted_translation_step)
        )
        capture_this_step = bool(
            dump_this_step or self.qn_shadow_options.get("enabled", False)
        )
        self.last_lbfgs_state_dump = None
        if forces is None:
            forces = self.dimeratoms.get_forces()
        forces = np.asarray(forces, dtype=float)
        # Covers both the kappa gamma regime and ASE's curvature-sign branch.
        state_key = _translation_state_key(self.dimeratoms)
        if (
            self._sm_state_key is not None
            and state_key != self._sm_state_key
            and self.reset_on_regime_change
        ):
            self._reset_native_history(
                f"translation_state:{self._sm_state_key}->{state_key}"
            )
        self._sm_state_key = state_key

        data = self._start_step_diagnostics(forces, "ase_lbfgs")
        before = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1).copy()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Please do not pass forces to step\(\)\..*",
                category=UserWarning,
                module=r"ase\.optimize\.optimize",
            )
            _CurvatureGuardedLBFGS.step(self, forces=forces)
        after = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1)
        if hasattr(self.dimeratoms, "confine_translation_positions"):
            confined = self.dimeratoms.confine_translation_positions(
                before.reshape((-1, 3)), after.reshape((-1, 3))
            ).reshape(-1)
            if not np.array_equal(confined, after):
                self.optimizable.set_x(confined)
                after = confined
        displacement = after - before
        raw_source = self.sm_last_raw_direction
        if raw_source is None:
            raw_source = self.p
        raw_direction = np.asarray(raw_source, dtype=float).reshape(-1)
        self._sm_dense_metrics = {
            "lbfgs_raw_direction_norm": safe_norm(raw_direction),
            "lbfgs_raw_direction_max_atom_norm": (
                safe_max_atom_norm(raw_direction) if raw_direction.size % 3 == 0 else ""
            ),
            "translation_regularization": self._sm_regularization,
            "translation_regularization_applied": 0,
            "translation_regularization_fallback": REGULARIZATION_NOT_APPLICABLE,
        }
        if self._sm_deep_qn_diagnostics:
            s_diag=list(_ase_lbfgs_get(self, "s", []) or []); y_diag=list(_ase_lbfgs_get(self, "y", []) or [])
            Q=rigid_translation_basis(len(forces))
            pairs_diag=[]; rows=[]
            for idx,(si,yi) in enumerate(zip(s_diag,y_diag)):
                sv=np.asarray(si,float).reshape(-1); yv=np.asarray(yi,float).reshape(-1); sy=float(sv@yv)
                row={"pair_index":idx,"pair_age_from_newest":len(s_diag)-1-idx,"source":"ase_translation","state_id":0,"serial":idx,"action":"stored","used":int(sy>0)}
                row.update({"stored_"+k:v for k,v in secant_row(sv,yv,Q).items()})
                rows.append(row)
                if sy>0:pairs_diag.append((sv,yv,sy,"ase_translation"))
            fixed_shift_rows=[]
            fflat=np.asarray(forces,float).reshape(-1)
            rawdiag=np.asarray(raw_direction,float).reshape(-1)
            for mu in (0.01,0.1,1.0,10.0):
                try:
                    shifted=shifted_lbfgs_solve(fflat,pairs_diag,self._sm_alpha,mu,False)
                    den=safe_norm(rawdiag)*safe_norm(shifted)
                    fixed_shift_rows.append({"mu":mu,"prediction_norm":safe_norm(shifted),"prediction_max_atom_norm":safe_max_atom_norm(shifted),"suppression_ratio":("" if safe_norm(rawdiag)<=1e-300 else safe_norm(shifted)/safe_norm(rawdiag)),"direction_cosine":("" if den<=0 else float(rawdiag@shifted/den))})
                except Exception as exc:
                    fixed_shift_rows.append({"mu":mu,"error":type(exc).__name__+":"+str(exc)})
            latest_raw=dict(self.sm_last_raw_pair_metrics or {})
            self.last_qn_detail_payload={"schema":"saddlemill_translation_qn_v1","pair_rows":rows,"progressive_history":progressive_lbfgs(fflat,pairs_diag,self._sm_alpha,False,Q),"block_rows":[],"current_force_norm":safe_norm(forces),"current_force_translation_ratio":translation_ratio(forces,Q),"rigid_translation_projection":0,"h0_dynamic":0,"h0_inverse_scale":1.0/self._sm_alpha,"latest_considered_pair":{"metrics":latest_raw,"action":self.sm_last_guard_action,"used":int(self.sm_last_guard_action in {"accept","off_accept","powell_accept","powell_damp","cautious_accept","damp"})},"fixed_shift_diagnostics":fixed_shift_rows}
        need_dense = self._sm_dense_bfgs_diagnostic
        dense = None
        if need_dense:
            s_hist = list(_ase_lbfgs_get(self, "s", []) or [])
            y_hist = list(_ase_lbfgs_get(self, "y", []) or [])
            dense_pairs = [
                DenseSecant(s=si, y=yi, source="ase_translation", state_id=0, serial=i)
                for i, (si, yi) in enumerate(zip(s_hist, y_hist))
            ]
            dense = reconstruct_sequential_bfgs(
                dense_pairs,
                np.asarray(forces, dtype=float).reshape(-1),
                initial_hessian=self._sm_alpha,
                dynamic_h0=False,
                denominator_epsilon=1.0e-14,
                require_positive_definite=(self.curvature_guard != "off"),
            )
            self._sm_dense_metrics.update(dense.metrics)
            comparison = compare_directions(raw_direction, dense.direction)
            self._sm_dense_metrics.update({
                "dense_vs_lbfgs_cosine": comparison.get("dense_vs_primary_cosine", ""),
                "dense_vs_lbfgs_norm_ratio": comparison.get("dense_vs_primary_norm_ratio", ""),
                "dense_vs_lbfgs_relative_difference": comparison.get(
                    "dense_vs_primary_relative_difference", ""
                ),
            })

        if self._sm_regularization != "off":
            s_hist = list(_ase_lbfgs_get(self, "s", []) or [])
            y_hist = list(_ase_lbfgs_get(self, "y", []) or [])
            qn_pairs=[]
            for i,(si,yi) in enumerate(zip(s_hist,y_hist)):
                sv=np.asarray(si,float).reshape(-1); yv=np.asarray(yi,float).reshape(-1); sy=float(sv@yv)
                if sy>0: qn_pairs.append((sv,yv,sy,"ase_translation"))
            try:
                fflat=np.asarray(forces,float).reshape(-1)
                if self._sm_regularization == "shifted_lbfgs_fixed":
                    mu=self._sm_regularization_mu
                    regdir=shifted_lbfgs_solve(fflat, qn_pairs, self._sm_alpha, mu, False)
                    iterations=0
                else:
                    trust=adaptive_shifted_lbfgs(fflat, qn_pairs, self._sm_alpha, self._sm_regularization_radius, False, self._sm_regularization_tolerance)
                    if trust.fallback: raise np.linalg.LinAlgError(trust.fallback)
                    regdir=trust.direction; mu=trust.mu; iterations=trust.iterations
                den=safe_norm(raw_direction)*safe_norm(regdir)
                self._sm_dense_metrics.update({
                    "translation_regularization_mu":float(mu),
                    "translation_regularization_iterations":int(iterations),
                    "translation_regularized_norm":safe_norm(regdir),
                    "translation_regularization_suppression_ratio":("" if safe_norm(raw_direction)<=1e-300 else safe_norm(regdir)/safe_norm(raw_direction)),
                    "translation_regularization_direction_cosine":("" if den<=0 else float(raw_direction@regdir/den)),
                })
                regstep = np.asarray(regdir,float).reshape((-1,3))
                confiner = getattr(self.dimeratoms, "confine_translation_vector", None)
                if callable(confiner):
                    regstep = np.asarray(confiner(regstep), dtype=float).reshape((-1,3))
                regmax = safe_max_atom_norm(regstep.reshape(-1))
                final_scale = 1.0
                if regmax > float(self.maxstep) and regmax > 0.0:
                    final_scale = float(self.maxstep) / regmax
                    regstep *= final_scale
                regularized_after = before + float(self.damping) * regstep.reshape(-1)
                self.optimizable.set_x(regularized_after)
                after=np.asarray(self.optimizable.get_x(),float).reshape(-1)
                self._sm_dense_metrics.update({
                    "translation_regularization_applied": 1,
                    "translation_regularization_final_maxstep_scale": float(final_scale),
                    "translation_regularization_final_max_atom_norm": safe_max_atom_norm((float(self.damping)*regstep).reshape(-1)),
                })
            except Exception as exc:
                self._sm_dense_metrics["translation_regularization_fallback"] = type(exc).__name__+":"+str(exc)

        if (
            self._sm_dense_metrics.get("translation_regularization_applied", 0)
            and hasattr(self.dimeratoms, "confine_translation_positions")
        ):
            confined = self.dimeratoms.confine_translation_positions(
                before.reshape((-1, 3)), after.reshape((-1, 3))
            ).reshape(-1)
            if not np.array_equal(confined, after):
                self.optimizable.set_x(confined)
                after = confined
        displacement = after - before
        step_metrics = _lbfgs_step_metrics(
            self.optimizable,
            raw_direction,
            displacement,
            self.maxstep,
            self.damping,
        )
        if capture_this_step:
            s_hist = [np.asarray(x, dtype=float).reshape(-1) for x in list(_ase_lbfgs_get(self, "s", []) or [])]
            y_hist = [np.asarray(x, dtype=float).reshape(-1) for x in list(_ase_lbfgs_get(self, "y", []) or [])]
            ndim = int(np.asarray(forces).size)
            S = np.stack(s_hist, axis=0) if s_hist else np.empty((0, ndim), dtype=np.float64)
            Y = np.stack(y_hist, axis=0) if y_hist else np.empty((0, ndim), dtype=np.float64)
            rho_hist = list(_ase_lbfgs_get(self, "rho", []) or [])
            if len(rho_hist) == S.shape[0] and all(np.isfinite(float(r)) and float(r) != 0.0 for r in rho_hist):
                # Preserve the exact reciprocal denominators represented by ASE's stored rho.
                sy = np.asarray([1.0 / float(r) for r in rho_hist], dtype=np.float64)
            else:
                sy = np.asarray([float(np.dot(S[i], Y[i])) for i in range(S.shape[0])], dtype=np.float64)
            history_ss = np.asarray([float(np.dot(S[i], S[i])) for i in range(S.shape[0])], dtype=np.float64)
            history_yy = np.asarray([float(np.dot(Y[i], Y[i])) for i in range(S.shape[0])], dtype=np.float64)
            history_kappa = np.divide(sy, history_ss, out=np.full_like(sy, np.nan), where=history_ss > 0.0)
            history_rho = np.divide(1.0, sy, out=np.full_like(sy, np.nan), where=np.isfinite(sy) & (sy != 0.0))
            history_gamma = np.divide(sy, history_yy, out=np.full_like(sy, np.nan), where=history_yy > 0.0)
            history_cos = np.divide(sy, np.sqrt(history_ss * history_yy), out=np.full_like(sy, np.nan), where=(history_ss > 0.0) & (history_yy > 0.0))
            rho_exact = np.asarray(
                [float(r) for r in rho_hist], dtype=np.float64
            ) if len(rho_hist) == S.shape[0] else np.divide(
                1.0, sy, out=np.full_like(sy, np.nan),
                where=np.isfinite(sy) & (sy != 0.0),
            )
            replay, h0_scale, trace = trace_two_loop(
                np.asarray(forces, dtype=float).reshape(-1), S, Y,
                initial_hessian=self._sm_alpha, dynamic_h0=False,
                sy_history=sy, rho_history=rho_exact, arithmetic="rho_multiply",
            )
            two_loop_recorded = (
                np.asarray(self.sm_last_two_loop_direction, dtype=float).reshape(-1)
                if self.sm_last_two_loop_direction is not None
                else np.asarray(replay, dtype=float).reshape(-1)
            )
            candidate_s = (
                np.asarray(self.sm_last_candidate_s, dtype=float).reshape(1, -1)
                if self.sm_last_candidate_s is not None else np.empty((0, ndim), dtype=float)
            )
            candidate_y_raw = (
                np.asarray(self.sm_last_candidate_y_raw, dtype=float).reshape(1, -1)
                if self.sm_last_candidate_y_raw is not None else np.empty((0, ndim), dtype=float)
            )
            candidate_y_stored = (
                np.asarray(self.sm_last_candidate_y_stored, dtype=float).reshape(1, -1)
                if self.sm_last_candidate_y_stored is not None else np.empty((0, ndim), dtype=float)
            )
            cs = candidate_s[0] if candidate_s.shape[0] else np.empty(0)
            cy = candidate_y_stored[0] if candidate_y_stored.shape[0] else np.empty(0)
            css = float(cs @ cs) if cs.size else float("nan")
            cyy = float(cy @ cy) if cy.size else float("nan")
            csy = float(cs @ cy) if cs.size else float("nan")
            guard_action = str(self.sm_last_guard_action or "")
            candidate_used = guard_action in {"accept", "off_accept", "powell_accept", "powell_damp", "damp"}
            self.last_lbfgs_state_dump = {
                "accepted_translation_step": np.asarray(accepted_translation_step, dtype=np.int64),
                "current_two_loop_vector": np.asarray(forces, dtype=np.float64).reshape(-1),
                "s_history": S, "y_history": Y, "sy_history": sy,
                "rho_history": rho_exact,
                "history_index": np.arange(S.shape[0], dtype=np.int64),
                "history_age_from_newest": np.arange(S.shape[0] - 1, -1, -1, dtype=np.int64),
                "history_sTs": history_ss,
                "history_yTy": history_yy,
                "history_sTy": sy,
                "history_kappa": history_kappa,
                "history_rho": history_rho,
                "history_gamma": history_gamma,
                "history_cos_theta": history_cos,
                "history_source": np.asarray(["ase_translation"] * S.shape[0]),
                "history_state_id": np.zeros(S.shape[0], dtype=np.int64),
                "history_serial": np.arange(S.shape[0], dtype=np.int64),
                "candidate_s": candidate_s, "candidate_y_raw": candidate_y_raw,
                "candidate_y_stored": candidate_y_stored,
                "candidate_source": np.asarray(["ase_translation"] if candidate_s.shape[0] else [], dtype=str),
                "candidate_state_id": np.asarray([0] if candidate_s.shape[0] else [], dtype=np.int64),
                "candidate_serial": np.asarray([accepted_translation_step - 1] if candidate_s.shape[0] else [], dtype=np.int64),
                "candidate_action": np.asarray([guard_action] if candidate_s.shape[0] else [], dtype=str),
                "candidate_rejection_reason": np.asarray(["" if candidate_used else guard_action] if candidate_s.shape[0] else [], dtype=str),
                "candidate_used": np.asarray([candidate_used] if candidate_s.shape[0] else [], dtype=np.bool_),
                "candidate_guard_pass": np.asarray([candidate_used] if candidate_s.shape[0] else [], dtype=np.bool_),
                "candidate_required_curvature": np.asarray([self.curvature_floor] if candidate_s.shape[0] else [], dtype=np.float64),
                "candidate_sTs": np.asarray([css] if candidate_s.shape[0] else [], dtype=np.float64),
                "candidate_yTy": np.asarray([cyy] if candidate_s.shape[0] else [], dtype=np.float64),
                "candidate_sTy": np.asarray([csy] if candidate_s.shape[0] else [], dtype=np.float64),
                "candidate_kappa": np.asarray([csy / css if css > 0 else np.nan] if candidate_s.shape[0] else [], dtype=np.float64),
                "candidate_rho": np.asarray([1.0 / csy if csy != 0 else np.nan] if candidate_s.shape[0] else [], dtype=np.float64),
                "candidate_gamma": np.asarray([csy / cyy if cyy > 0 else np.nan] if candidate_s.shape[0] else [], dtype=np.float64),
                "candidate_cos_theta": np.asarray([csy / np.sqrt(css * cyy) if css > 0 and cyy > 0 else np.nan] if candidate_s.shape[0] else [], dtype=np.float64),
                "initial_hessian": np.asarray(self._sm_alpha, dtype=np.float64),
                "dynamic_h0": np.asarray(False, dtype=np.bool_),
                "h0_inverse_scale": np.asarray(h0_scale, dtype=np.float64),
                "raw_lbfgs_direction": two_loop_recorded,
                "production_pre_cap_direction": np.asarray(raw_direction, dtype=np.float64).reshape(-1),
                "final_applied_step": np.asarray(displacement, dtype=np.float64).reshape(-1),
                "final_capped_direction": np.asarray(displacement / float(self.damping), dtype=np.float64).reshape(-1),
                "positions_before": np.asarray(before, dtype=np.float64).reshape((-1, 3)),
                "positions_after": np.asarray(after, dtype=np.float64).reshape((-1, 3)),
                "maximum_translation": np.asarray(float(self.maxstep), dtype=np.float64),
                "damping": np.asarray(float(self.damping), dtype=np.float64),
                "rigid_translation_basis": rigid_translation_basis(len(forces)),
                "snapshot_mode": np.asarray(self.dimeratoms.get_eigenmode(), dtype=np.float64),
                "snapshot_curvature": np.asarray(float(self.dimeratoms.get_curvature()), dtype=np.float64),
                "clip_scale": np.asarray(step_metrics["clip_scale"], dtype=np.float64),
                "history_order": np.asarray("oldest_to_newest"),
                "input_vector_role": np.asarray("effective_force"),
                "gradient_sign_convention": np.asarray("g_eff=-F_eff"),
                "two_loop_sign_convention": np.asarray("raw_lbfgs_direction=H*F_eff=-H*g_eff"),
                "two_loop_arithmetic": np.asarray("rho_multiply"),
                "secant_y_sign_convention": np.asarray("y=F_eff_previous-F_eff_current"),
                "production_reconstruction_model": np.asarray("ase_lbfgs"),
            }
            # Capture the exact runtime active-coordinate mask at the production
            # step boundary.  dense-replay consumes this frozen payload and never
            # reconstructs constraint/projector state on its own.
            try:
                from saddlemill.dimertools.runtime_state import active_coordinate_provenance
                replay_mask, _ = active_coordinate_provenance(
                    self.dimeratoms, np.asarray(before, dtype=float).reshape((-1, 3))
                )
            except Exception:
                replay_mask = None
            if replay_mask is not None:
                self.last_lbfgs_state_dump["replay_active_dof_mask"] = np.asarray(
                    replay_mask, dtype=np.bool_
                ).reshape(-1)
            if self._sm_state_dump_trace:
                self.last_lbfgs_state_dump.update(trace)

        self.last_qn_shadow_row = None
        self.last_qn_shadow_matrix = None
        if bool(self.qn_shadow_options.get("enabled", False)) and self.last_lbfgs_state_dump is not None:
            from saddlemill.dimertools.wave_b_shadow import run_shadow
            shadow_result, shadow_elapsed = run_shadow(
                self.last_lbfgs_state_dump, options=self.qn_shadow_options,
                consumer="saddle_translation",
                residual_kind="mmf_effective_force",
                force_interpretation="effective_force",
                model_type="effective_residual_bfgs", owner=self.dimeratoms,
            )
            if shadow_result is not None:
                self.last_qn_shadow_row = dict(shadow_result.row)
                self.last_qn_shadow_matrix = shadow_result.matrix
                self.qn_shadow_cumulative_ns += int(shadow_elapsed)
        if not dump_this_step:
            self.last_lbfgs_state_dump = None
        data.update(
            {
                "direction_alignment": _cosine_alignment(raw_direction, forces),
                "step_clipped": step_metrics["step_clipped"],
                "translation_raw_step_norm": step_metrics["raw_step_norm"],
                "translation_raw_step_max": step_metrics["raw_step_max"],
                "translation_actual_step_norm": step_metrics["actual_step_norm"],
                "translation_actual_step_max": step_metrics["actual_step_max"],
                "translation_maxstep": step_metrics["maxstep"],
                "translation_maxstep_rescaled": step_metrics["maxstep_rescaled"],
                "translation_clip_scale": step_metrics["clip_scale"],
                "translation_damping": step_metrics["damping"],
                "translation_applied_scale": step_metrics["applied_scale"],
            }
        )
        step_metric = float(step_metrics["actual_step_norm"])
        self._last_step_size = step_metric
        if self._sm_deep_qn_diagnostics and getattr(self, "last_qn_detail_payload", None):
            self.last_qn_detail_payload.update(self._sm_dense_metrics)
            self.last_qn_detail_payload["maximum_translation"] = float(self.maxstep)
            self.last_qn_detail_payload["translation_actual_step_norm"] = step_metrics["actual_step_norm"]
            self.last_qn_detail_payload["translation_actual_step_max"] = step_metrics["actual_step_max"]
        self._finish_step_diagnostics(
            data, step_metric, lbfgs_metrics=self._native_metrics()
        )


@dataclass
class HybridDecision:
    state: str
    switch_event: str = ""
    history_pairs_at_switch: int = 0


class HybridDimerStateController:
    """FIRE -> ASE-LBFGS controller with fmax/curvature hysteresis."""

    def __init__(
        self,
        enabled=False,
        enter_fmax=0.30,
        exit_fmax=0.50,
        enter_curvature=-0.05,
        exit_curvature=0.00,
        enter_stable_steps=3,
        exit_stable_steps=2,
        minimum_history_pairs=3,
        warm_start_history=True,
    ):
        self.enabled = bool(enabled)
        self.enter_fmax = float(enter_fmax)
        self.exit_fmax = float(exit_fmax)
        self.enter_curvature = float(enter_curvature)
        self.exit_curvature = float(exit_curvature)
        self.enter_stable_steps = int(enter_stable_steps)
        self.exit_stable_steps = int(exit_stable_steps)
        self.minimum_history_pairs = int(minimum_history_pairs)
        self.warm_start_history = bool(warm_start_history)
        if self.enter_stable_steps < 1 or self.exit_stable_steps < 1:
            raise ValueError("Hybrid stable-step counts must be >= 1")
        if self.minimum_history_pairs < 0:
            raise ValueError("minimum_history_pairs must be >= 0")
        if self.exit_fmax < self.enter_fmax:
            raise ValueError("hybrid exit_fmax must be >= enter_fmax")
        if self.exit_curvature < self.enter_curvature:
            raise ValueError("hybrid exit_curvature must be >= enter_curvature")
        self.state = "fire"
        self._enter_count = 0
        self._exit_count = 0

    def force_fire(self, event):
        self.state = "fire"
        self._enter_count = 0
        self._exit_count = 0
        return HybridDecision("fire", str(event), 0)

    def update(self, fmax, curvature, history_pairs) -> HybridDecision:
        if not self.enabled:
            return HybridDecision("fire", "", 0)
        history_ready = (
            not self.warm_start_history
            or int(history_pairs) >= self.minimum_history_pairs
        )
        if self.state == "fire":
            enter = (
                float(fmax) <= self.enter_fmax
                and float(curvature) <= self.enter_curvature
                and history_ready
            )
            self._enter_count = self._enter_count + 1 if enter else 0
            self._exit_count = 0
            if self._enter_count >= self.enter_stable_steps:
                self.state = "lbfgs"
                self._enter_count = 0
                return HybridDecision(
                    "lbfgs", "fire_to_lbfgs", int(history_pairs)
                )
        else:
            exit_now = (
                float(fmax) >= self.exit_fmax
                or float(curvature) >= self.exit_curvature
            )
            self._exit_count = self._exit_count + 1 if exit_now else 0
            self._enter_count = 0
            if self._exit_count >= self.exit_stable_steps:
                self.state = "fire"
                self._exit_count = 0
                return HybridDecision(
                    "fire", "lbfgs_to_fire_threshold", int(history_pairs)
                )
        return HybridDecision(self.state, "", 0)


class HybridMinModeTranslate(
    _RealForceConvergenceMixin,
    _TranslationDiagnosticsMixin,
    _DimerLBFGSLogMixin,
    MinModeTranslate,
):
    """ASE FIRE warm-up followed by ASE L-BFGS on ``MinModeAtoms``."""

    def __init__(
        self,
        dimeratoms,
        logfile="-",
        trajectory=None,
        lbfgs_options=None,
        hybrid_options=None,
    ):
        MinModeTranslate.__init__(
            self, dimeratoms, logfile=logfile, trajectory=trajectory
        )
        self._initialize_step_diagnostics()
        self._initialize_dimer_log(write_header=False)

        lbfgs = dict(lbfgs_options or {})
        self.reset_on_regime_change = bool(
            lbfgs.pop("reset_on_regime_change", True)
        )
        lbfgs.pop("dynamic_h0", None)
        lbfgs.pop("curvature_epsilon", None)
        lbfgs.pop("reconstruction_model", None)
        lbfgs.pop("bfgs_update", None)
        lbfgs.pop("dense_bfgs_diagnostic", None)
        self.lbfgs_memory = int(lbfgs.pop("memory", 10))
        self.lbfgs_alpha = float(lbfgs.pop("initial_hessian", 70.0))
        self.lbfgs_damping = float(lbfgs.pop("damping", 1.0))
        self.lbfgs_curvature_guard = str(
            lbfgs.pop("curvature_guard", "skip")
        ).strip().lower()
        self.lbfgs_curvature_floor = float(
            lbfgs.pop("curvature_floor", DEFAULT_CURVATURE_FLOOR)
        )
        self.lbfgs_powell_eta = float(
            lbfgs.pop("powell_eta", DEFAULT_POWELL_ETA)
        )
        if lbfgs:
            raise TypeError(f"Unknown ASE dimer L-BFGS options: {sorted(lbfgs)}")

        options = dict(hybrid_options or {})
        fire_keys = {
            "fire_dt": "dt",
            "fire_dtmax": "dtmax",
            "fire_Nmin": "Nmin",
            "fire_finc": "finc",
            "fire_fdec": "fdec",
            "fire_astart": "astart",
            "fire_fa": "fa",
        }
        self.fire_options = {
            target: options.pop(source)
            for source, target in fire_keys.items()
            if source in options
        }
        self.reset_history_on_exit = bool(
            options.pop("reset_history_on_exit", True)
        )
        self.controller = HybridDimerStateController(**options)
        self.warm_start_history = self.controller.warm_start_history
        self.snapshots = deque(maxlen=self.lbfgs_memory + 1)
        self.fire_optimizer = self._new_fire_optimizer()
        self.lbfgs_state = _ASELBFGSState(
            dimeratoms,
            maxstep=self.max_step,
            memory=self.lbfgs_memory,
            damping=self.lbfgs_damping,
            alpha=self.lbfgs_alpha,
            curvature_guard=self.lbfgs_curvature_guard,
            curvature_floor=self.lbfgs_curvature_floor,
            powell_eta=self.lbfgs_powell_eta,
        )
        self._last_state_key = None

    def _new_fire_optimizer(self):
        return FIRE(
            self.dimeratoms,
            restart=None,
            logfile=None,
            trajectory=None,
            maxstep=self.max_step,
            downhill_check=False,
            **self.fire_options,
        )

    def _append_snapshot(self, position, force):
        item = (
            np.asarray(position, dtype=float).reshape(-1).copy(),
            np.asarray(force, dtype=float).reshape(-1).copy(),
        )
        if self.snapshots and np.array_equal(self.snapshots[-1][0], item[0]):
            self.snapshots[-1] = item
        else:
            self.snapshots.append(item)

    def step(self, f=None):
        if f is None:
            f = self.dimeratoms.get_forces()
        f = np.asarray(f, dtype=float)
        position = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1)
        state_key = _translation_state_key(self.dimeratoms)
        state_changed = (
            self._last_state_key is not None and state_key != self._last_state_key
        )
        self._last_state_key = state_key
        if state_changed and self.reset_on_regime_change:
            self.snapshots.clear()
            self.lbfgs_state.reset(f"translation_state_change:{state_key}")
            self.fire_optimizer = self._new_fire_optimizer()
            if self.controller.state == "lbfgs":
                self.controller.force_fire("lbfgs_to_fire_state_change")

        self._append_snapshot(position, f)
        available_pairs = max(0, len(self.snapshots) - 1)
        decision = self.controller.update(
            _projected_fmax(f),
            float(self.dimeratoms.get_curvature()),
            available_pairs,
        )

        if decision.switch_event == "fire_to_lbfgs":
            if self.warm_start_history:
                self.lbfgs_state.rebuild(
                    self.snapshots, reason="warm_start_fire_to_lbfgs"
                )
            else:
                self.lbfgs_state.reset("cold_start_fire_to_lbfgs")
        elif decision.switch_event == "lbfgs_to_fire_threshold":
            self.fire_optimizer = self._new_fire_optimizer()
            if self.reset_history_on_exit:
                self.lbfgs_state.reset("lbfgs_to_fire_threshold")
                self.snapshots.clear()
                self._append_snapshot(position, f)

        data = self._start_step_diagnostics(
            f,
            decision.state,
            hybrid_state=decision.state,
            switch_event=decision.switch_event,
        )
        if decision.state == "fire":
            before = position.copy()
            # FIRE carries a velocity vector between steps. Mask that state as
            # well as the current force so stale momentum cannot move an atom
            # outside the current bowl active set.
            velocity = getattr(self.fire_optimizer, "v", None)
            if velocity is not None and hasattr(
                self.dimeratoms, "confine_translation_vector"
            ):
                self.fire_optimizer.v = self.dimeratoms.confine_translation_vector(
                    velocity
                )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Please do not pass forces to step\(\)\..*",
                    category=UserWarning,
                    module=r"ase\.optimize\.optimize",
                )
                self.fire_optimizer.step(f=f)
            after = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1)
            if hasattr(self.dimeratoms, "confine_translation_positions"):
                confined = self.dimeratoms.confine_translation_positions(
                    before.reshape((-1, 3)), after.reshape((-1, 3))
                ).reshape(-1)
                if not np.array_equal(confined, after):
                    self.optimizable.set_x(confined)
                    after = confined
            displacement = after - before
            step_metric = float(norm(displacement))
            data.update(
                {
                    "direction_alignment": "",
                    "step_clipped": int(norm(displacement) >= self.max_step),
                }
            )
            metrics = self.lbfgs_state.metrics()
        else:
            before = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1).copy()
            step_metric, clipped, alignment = self.lbfgs_state.step(f)
            after = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1)
            if hasattr(self.dimeratoms, "confine_translation_positions"):
                confined = self.dimeratoms.confine_translation_positions(
                    before.reshape((-1, 3)), after.reshape((-1, 3))
                ).reshape(-1)
                if not np.array_equal(confined, after):
                    self.optimizable.set_x(confined)
                    after = confined
                    step_metric = float(norm(after - before))
            lbfgs_step = dict(getattr(self.lbfgs_state, "last_step_metrics", {}) or {})
            data.update(
                {
                    "direction_alignment": alignment,
                    "step_clipped": int(clipped),
                    "translation_raw_step_norm": lbfgs_step.get("raw_step_norm", ""),
                    "translation_raw_step_max": lbfgs_step.get("raw_step_max", ""),
                    "translation_actual_step_norm": lbfgs_step.get("actual_step_norm", ""),
                    "translation_actual_step_max": lbfgs_step.get("actual_step_max", ""),
                    "translation_maxstep": lbfgs_step.get("maxstep", ""),
                    "translation_maxstep_rescaled": lbfgs_step.get(
                        "maxstep_rescaled", ""
                    ),
                    "translation_clip_scale": lbfgs_step.get("clip_scale", ""),
                    "translation_damping": lbfgs_step.get("damping", ""),
                    "translation_applied_scale": lbfgs_step.get("applied_scale", ""),
                }
            )
            metrics = self.lbfgs_state.metrics()

        data.update(
            {
                "hybrid_history_pairs_at_switch": int(
                    decision.history_pairs_at_switch
                ),
                "hybrid_warm_start_history": int(self.warm_start_history),
            }
        )
        self._last_step_size = step_metric
        self._finish_step_diagnostics(data, step_metric, metrics)

# SADDLEMILL_PERSISTENT_ROTATION_LBFGS_20260901_V2
