"""State-window rotational L-BFGS history.

This is an opt-in warm-history consumer for the existing SaddleMill rotational
L-BFGS algorithm.  It deliberately does *not* form a dense Hessian and does not
diagonalize anything.

The existing rotational solver already represents secant vectors in the ambient
Cartesian/mode array and projects them into the tangent plane of the *current*
dimer mode each time ``apply()`` is called.  This class preserves that behavior,
but retains accepted secants across translated center geometries for the last
``memory_states`` canonical center states.

Two pair kinds are retained:

``local``
    The exact pair the legacy rotational L-BFGS would have admitted between
    consecutive accepted rotation iterations at one center.  These are usable
    immediately and remain available at later centers.

``trial``
    A no-extra-force-call pair made from the current A orientation and ASE's
    already-evaluated trial-B orientation.  It is intentionally *not* used at
    the center where it was generated, so enabling persistent history does not
    change the direction that generated that same trial point.  It becomes
    available only after a translation to a later center.  This gives
    ``max_num_rot=1`` runs something to learn from across centers.

The raw canonical force history remains the authoritative physical-observation
layer.  These pairs are a disposable quasi-Newton cache attached to its center
state IDs, not a replacement for raw ``(x, F_PES)`` observations.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Optional

import numpy as np

from saddlemill.dimertools.dense_bfgs import (
    DenseSecant,
    compare_directions,
    reconstruct_sequential_bfgs,
    safe_max_atom_norm,
    safe_norm,
)

Array = np.ndarray
norm = np.linalg.norm


@dataclass
class RotationSecant:
    state_id: int
    serial: int
    kind: str
    s: Array
    y: Array


class StateWindowRotationHistory:
    """L-BFGS-compatible history retained by canonical center state."""

    def __init__(
        self,
        *,
        memory_states: int = 20,
        initial_hessian: float = 1.0,
        dynamic_h0: bool = False,
        curvature_epsilon: float = 1.0e-12,
        dense_bfgs_diagnostic: bool = False,
    ) -> None:
        if int(memory_states) < 1:
            raise ValueError("rotation memory_states must be >= 1")
        if float(initial_hessian) <= 0.0:
            raise ValueError("rotation initial_hessian must be > 0")
        if float(curvature_epsilon) < 0.0:
            raise ValueError("rotation curvature_epsilon must be >= 0")
        self.memory_states = int(memory_states)
        # Compatibility field read by the existing diagnostics.  In persistent
        # mode it means center-state memory, not a hard pair cap.
        self.memory = int(memory_states)
        self.initial_hessian = float(initial_hessian)
        self.dynamic_h0 = bool(dynamic_h0)
        self.curvature_epsilon = float(curvature_epsilon)
        self.dense_bfgs_diagnostic = bool(dense_bfgs_diagnostic)
        self.last_dense_diagnostics: dict[str, object] = {}

        self._pairs_by_state: "OrderedDict[int, list[RotationSecant]]" = OrderedDict()
        self.current_state_id: int | None = None
        self._next_serial = 0

        self.accepted_pairs_total = 0
        self.rejected_pairs_total = 0
        self.trial_pairs_accepted_total = 0
        self.trial_pairs_rejected_total = 0
        self.local_pairs_accepted_total = 0
        self.local_pairs_rejected_total = 0
        self.states_dropped_total = 0
        self.pairs_dropped_with_states_total = 0
        self.reset_count = 0
        self.last_reset_reason = "initial"
        self.last_pair_metrics: dict[str, object] = {}
        self.last_pair_accepted: int | str = ""

        self.apply_calls = 0
        self.apply_ns_total = 0
        self.last_apply_ns = 0
        self.last_apply_pair_candidates = 0
        self.last_apply_pairs_used = 0
        self.last_apply_pairs_rejected_projection = 0
        self.last_apply_states_contributing = 0
        self.last_apply_local_pairs_used = 0
        self.last_apply_trial_pairs_used = 0

    @property
    def size(self) -> int:
        return sum(len(items) for items in self._pairs_by_state.values())

    @property
    def states_retained(self) -> int:
        return len(self._pairs_by_state)

    def begin_state(self, state_id: int) -> None:
        state_id = int(state_id)
        self.current_state_id = state_id
        if state_id not in self._pairs_by_state:
            self._pairs_by_state[state_id] = []
        else:
            self._pairs_by_state.move_to_end(state_id)
        while len(self._pairs_by_state) > self.memory_states:
            _, dropped = self._pairs_by_state.popitem(last=False)
            self.states_dropped_total += 1
            self.pairs_dropped_with_states_total += len(dropped)

    def reset(self, reason: str = "manual") -> None:
        # Legacy rotational L-BFGS may reset after a non-descent direction and
        # then continue rotating at the *same* center.  Drop all accumulated
        # quasi-Newton information, but keep the current state open so the next
        # accepted local/trial secant can be recorded normally.
        current = self.current_state_id
        self._pairs_by_state.clear()
        if current is not None:
            self._pairs_by_state[current] = []
        self.reset_count += 1
        self.last_reset_reason = str(reason)

    @staticmethod
    def _metrics(s: Array, y: Array) -> dict[str, float]:
        s = np.asarray(s, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        sn = safe_norm(s)
        yn = safe_norm(y)
        sy = float(np.dot(s, y))
        return {
            "s_norm": sn,
            "y_norm": yn,
            "s_dot_y": sy,
            "secant_curvature": 0.0 if sn <= 0.0 else sy / (sn * sn),
            "secant_cosine": 0.0 if sn <= 0.0 or yn <= 0.0 else sy / (sn * yn),
        }

    def _admissible(self, s: Array, y: Array) -> bool:
        if s.shape != y.shape or not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
            return False
        sy = float(np.dot(s.reshape(-1), y.reshape(-1)))
        scale = float(safe_norm(s) * safe_norm(y))
        threshold = self.curvature_epsilon * max(1.0, scale)
        return bool(np.isfinite(sy) and sy > threshold)

    def _store(self, s, y, *, kind: str) -> bool:
        if self.current_state_id is None:
            raise RuntimeError("rotation history pair recorded before begin_state()")
        s_arr = np.asarray(s, dtype=float).reshape(-1).copy()
        y_arr = np.asarray(y, dtype=float).reshape(-1).copy()
        self.last_pair_metrics = self._metrics(s_arr, y_arr)
        accepted = self._admissible(s_arr, y_arr)
        self.last_pair_accepted = int(accepted)
        if not accepted:
            self.rejected_pairs_total += 1
            if kind == "trial":
                self.trial_pairs_rejected_total += 1
            else:
                self.local_pairs_rejected_total += 1
            return False
        item = RotationSecant(
            state_id=self.current_state_id,
            serial=self._next_serial,
            kind=str(kind),
            s=s_arr,
            y=y_arr,
        )
        self._next_serial += 1
        self._pairs_by_state[self.current_state_id].append(item)
        self.accepted_pairs_total += 1
        if kind == "trial":
            self.trial_pairs_accepted_total += 1
        else:
            self.local_pairs_accepted_total += 1
        return True

    def add_pair(self, s, y) -> bool:
        """Store the legacy current-center secant."""
        return self._store(s, y, kind="local")

    def add_trial_pair(self, s, y) -> bool:
        """Store an already-paid A->trial-B secant for future centers only."""
        return self._store(s, y, kind="trial")

    def _h0_scale(self, pairs: list[tuple[Array, Array, float, RotationSecant]]) -> float:
        scale = 1.0 / self.initial_hessian
        if self.dynamic_h0 and pairs:
            s, y, sy, _ = pairs[-1]
            yy = float(np.dot(y, y))
            if yy > 0.0 and sy > 0.0:
                scale = sy / yy
        return scale

    def apply(
        self,
        force,
        projector: Optional[Callable[[Array], Array]] = None,
    ) -> Array:
        started = perf_counter_ns()
        shape = np.asarray(force).shape
        q = np.asarray(force, dtype=float).reshape(-1).copy()
        if projector is not None:
            q = np.asarray(projector(q.reshape(shape)), dtype=float).reshape(-1)
        force_flat = q.copy()

        candidates: list[RotationSecant] = []
        for state_id, items in self._pairs_by_state.items():
            for item in items:
                # Trial-B is a future-center observation.  Legacy local pairs at
                # the current center remain immediately usable exactly as before.
                if item.kind == "trial" and state_id == self.current_state_id:
                    continue
                candidates.append(item)

        pairs: list[tuple[Array, Array, float, RotationSecant]] = []
        rejected_projection = 0
        contributing_states: set[int] = set()
        local_used = 0
        trial_used = 0
        for item in candidates:
            s = item.s.reshape(shape)
            y = item.y.reshape(shape)
            if projector is not None:
                s = np.asarray(projector(s), dtype=float)
                y = np.asarray(projector(y), dtype=float)
            s = np.asarray(s, dtype=float).reshape(-1)
            y = np.asarray(y, dtype=float).reshape(-1)
            if not self._admissible(s, y):
                rejected_projection += 1
                continue
            sy = float(np.dot(s, y))
            pairs.append((s, y, sy, item))
            contributing_states.add(item.state_id)
            if item.kind == "trial":
                trial_used += 1
            else:
                local_used += 1

        alphas: list[float] = []
        for s, y, sy, _ in reversed(pairs):
            rho = 1.0 / sy
            alpha = rho * float(np.dot(s, q))
            alphas.append(alpha)
            q -= alpha * y

        r = self._h0_scale(pairs) * q
        for (s, y, sy, _), alpha in zip(pairs, reversed(alphas)):
            rho = 1.0 / sy
            beta = rho * float(np.dot(y, r))
            r += s * (alpha - beta)

        result = r.reshape(shape)
        if projector is not None:
            result = np.asarray(projector(result), dtype=float)
        result = np.asarray(result, dtype=float)

        self.last_dense_diagnostics = {
            "lbfgs_raw_direction_norm": safe_norm(result),
            "lbfgs_raw_direction_max_atom_norm": (
                safe_max_atom_norm(result) if result.size % 3 == 0 else ""
            ),
        }
        if self.dense_bfgs_diagnostic:
            dense_pairs = [
                DenseSecant(
                    s=s,
                    y=y,
                    source=f"legacy_rotation_{item.kind}",
                    state_id=item.state_id,
                    serial=item.serial,
                )
                for s, y, _sy, item in pairs
            ]
            dense = reconstruct_sequential_bfgs(
                dense_pairs,
                force_flat,
                initial_hessian=self.initial_hessian,
                dynamic_h0=self.dynamic_h0,
                denominator_epsilon=self.curvature_epsilon,
                require_positive_definite=True,
            )
            self.last_dense_diagnostics.update(dense.metrics)
            comparison = compare_directions(result.reshape(-1), dense.direction)
            self.last_dense_diagnostics.update(
                {
                    "dense_vs_lbfgs_cosine": comparison.get(
                        "dense_vs_primary_cosine", ""
                    ),
                    "dense_vs_lbfgs_norm_ratio": comparison.get(
                        "dense_vs_primary_norm_ratio", ""
                    ),
                    "dense_vs_lbfgs_relative_difference": comparison.get(
                        "dense_vs_primary_relative_difference", ""
                    ),
                }
            )

        elapsed = perf_counter_ns() - started
        self.apply_calls += 1
        self.apply_ns_total += int(elapsed)
        self.last_apply_ns = int(elapsed)
        self.last_apply_pair_candidates = len(candidates)
        self.last_apply_pairs_used = len(pairs)
        self.last_apply_pairs_rejected_projection = rejected_projection
        self.last_apply_states_contributing = len(contributing_states)
        self.last_apply_local_pairs_used = local_used
        self.last_apply_trial_pairs_used = trial_used
        return result

    def diagnostics(self) -> dict[str, object]:
        return {
            "canonical_rotation_reuse_enabled": 1,
            "canonical_rotation_memory_states": self.memory_states,
            "canonical_rotation_states_retained": self.states_retained,
            "canonical_rotation_pairs_retained": self.size,
            "canonical_rotation_pairs_accepted_total": self.accepted_pairs_total,
            "canonical_rotation_pairs_rejected_total": self.rejected_pairs_total,
            "canonical_rotation_local_pairs_accepted_total": self.local_pairs_accepted_total,
            "canonical_rotation_trial_pairs_accepted_total": self.trial_pairs_accepted_total,
            "canonical_rotation_trial_pairs_rejected_total": self.trial_pairs_rejected_total,
            "canonical_rotation_states_dropped_total": self.states_dropped_total,
            "canonical_rotation_pairs_dropped_with_states_total": self.pairs_dropped_with_states_total,
            "canonical_rotation_pair_candidates": self.last_apply_pair_candidates,
            "canonical_rotation_pairs_used": self.last_apply_pairs_used,
            "canonical_rotation_pairs_rejected_projection": self.last_apply_pairs_rejected_projection,
            "canonical_rotation_states_contributing": self.last_apply_states_contributing,
            "canonical_rotation_local_pairs_used": self.last_apply_local_pairs_used,
            "canonical_rotation_trial_pairs_used": self.last_apply_trial_pairs_used,
            "canonical_rotation_apply_ns": self.last_apply_ns,
            "canonical_rotation_apply_cumulative_ns": self.apply_ns_total,
            "canonical_rotation_apply_calls": self.apply_calls,
            "canonical_rotation_history_resets": self.reset_count,
            "canonical_rotation_last_reset_reason": self.last_reset_reason,
            "canonical_rotation_dense_bfgs_diagnostic": int(self.dense_bfgs_diagnostic),
        }
