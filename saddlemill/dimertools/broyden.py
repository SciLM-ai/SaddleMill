"""Broyden root-finding kernels for SaddleMill experimental optimizers.

Two deliberately distinct algorithms live here.

``GenericGoodBroydenInverse`` stores an approximation ``H ~= J^{-1}`` for a
residual ``r(x)`` and proposes ``p = -H r``.  Its accepted inverse-good
Broyden update is

    H+ = H + (s - H y) (s.T H) / (s.T H y),

where ``s = x+ - x`` and ``y = r+ - r``.

``JohnsonModifiedBroyden`` follows the Johnson modified-Broyden history-space
form used by Shang and Liu, JCTC 2010, p. 1138, eqs. (5)-(11).  To make the
paper's plus-sign error functional explicit, this class represents the *step
operator* ``G ~= -J^{-1}`` and proposes ``p = G r``.  History pairs are scaled
by ``||y||`` (the standard Johnson normalization): ``dR=y/||y||`` and
``dx=s/||y||``.  The bounded history-space form is

    U = alpha*dR + dx
    A = W dR.T dR W
    beta = (w0**2 I + A)^-1
    p = alpha*r - U W beta W dR.T r.

``w0`` is a small-history regularizer only.  Neither class is a physical
Hessian, and neither performs a Hessian eigendecomposition.  Damping,
positive-curvature rules, mode scheduling, and other trajectory policies are
intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

Array = np.ndarray


def _vector(value: object, *, name: str) -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty")
    flat = arr.reshape(-1).copy()
    if not np.all(np.isfinite(flat)):
        raise ValueError(f"{name} must be finite")
    return flat


def _safe_cond(matrix: Array) -> float:
    try:
        value = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float("inf")
    return value if np.isfinite(value) else float("inf")


@dataclass(frozen=True)
class BroydenUpdateDiagnostics:
    accepted: bool
    reason: str
    denominator: float | None = None
    denominator_scale: float | None = None
    history_size: int = 0
    history_rank: int = 0
    condition_number: float | None = None
    weight: float = 1.0

    def to_state_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "denominator": self.denominator,
            "denominator_scale": self.denominator_scale,
            "history_size": self.history_size,
            "history_rank": self.history_rank,
            "condition_number": self.condition_number,
            "weight": self.weight,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "BroydenUpdateDiagnostics":
        return cls(
            accepted=bool(state.get("accepted", False)),
            reason=str(state.get("reason", "no_update")),
            denominator=(
                None if state.get("denominator") is None else float(state["denominator"])
            ),
            denominator_scale=(
                None
                if state.get("denominator_scale") is None
                else float(state["denominator_scale"])
            ),
            history_size=int(state.get("history_size", 0)),
            history_rank=int(state.get("history_rank", 0)),
            condition_number=(
                None
                if state.get("condition_number") is None
                else float(state["condition_number"])
            ),
            weight=float(state.get("weight", 1.0)),
        )


@dataclass(frozen=True)
class BroydenProposal:
    step: Array
    residual_norm: float
    step_norm: float
    history_size: int
    history_rank: int
    condition_number: float | None
    family: str
    residual_convention: str


class GenericGoodBroydenInverse:
    """Windowed inverse-good Broyden with ``H ~= J^-1`` and ``p=-H r``.

    ``weight`` is accepted by :meth:`update_pair` only so the rotation and
    translation adapters can share one protocol.  It is recorded in
    diagnostics/state but does not alter this unweighted generic update.
    """

    SCHEMA = "saddlemill_generic_good_broyden_inverse_v1"
    family = "generic_good_broyden_inverse"
    residual_convention = "root_residual_r_step_minus_Hr"

    def __init__(
        self,
        *,
        initial_inverse_scale: float = -1.0,
        history_cap: int = 20,
        denominator_tolerance: float = 1.0e-12,
        vector_tolerance: float = 1.0e-14,
        rank_tolerance: float = 1.0e-12,
        max_condition: float = 1.0e12,
    ) -> None:
        self.initial_inverse_scale = float(initial_inverse_scale)
        self.history_cap = int(history_cap)
        self.denominator_tolerance = float(denominator_tolerance)
        self.vector_tolerance = float(vector_tolerance)
        self.rank_tolerance = float(rank_tolerance)
        self.max_condition = float(max_condition)
        if not np.isfinite(self.initial_inverse_scale) or self.initial_inverse_scale == 0.0:
            raise ValueError("initial_inverse_scale must be finite and nonzero")
        if self.history_cap < 1:
            raise ValueError("history_cap must be >= 1")
        if self.denominator_tolerance < 0.0 or self.vector_tolerance < 0.0:
            raise ValueError("tolerances must be >= 0")
        if self.rank_tolerance < 0.0:
            raise ValueError("rank_tolerance must be >= 0")
        if not np.isfinite(self.max_condition) or self.max_condition <= 1.0:
            raise ValueError("max_condition must be finite and > 1")
        self._pairs: list[tuple[Array, Array, float]] = []
        self.dimension: int | None = None
        self.reset_count = 0
        self.last_reset_reason = "initial"
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.last_update = BroydenUpdateDiagnostics(False, "no_update")

    def empty_copy(self) -> "GenericGoodBroydenInverse":
        return type(self)(
            initial_inverse_scale=self.initial_inverse_scale,
            history_cap=self.history_cap,
            denominator_tolerance=self.denominator_tolerance,
            vector_tolerance=self.vector_tolerance,
            rank_tolerance=self.rank_tolerance,
            max_condition=self.max_condition,
        )

    def reset(self, reason: str = "manual") -> None:
        self._pairs.clear()
        self.dimension = None
        self.reset_count += 1
        self.last_reset_reason = str(reason)
        self.last_update = BroydenUpdateDiagnostics(False, "reset")

    def _matrix_from_pairs(
        self, pairs: list[tuple[Array, Array, float]]
    ) -> tuple[Array, list[tuple[Array, Array, float]], list[BroydenUpdateDiagnostics]]:
        if not pairs:
            if self.dimension is None:
                raise ValueError("dimension is not initialized")
            return (
                self.initial_inverse_scale * np.eye(self.dimension),
                [],
                [],
            )
        dimension = pairs[0][0].size
        H = self.initial_inverse_scale * np.eye(dimension)
        accepted: list[tuple[Array, Array, float]] = []
        diags: list[BroydenUpdateDiagnostics] = []
        for s, y, weight in pairs:
            Hy = H @ y
            left = s - Hy
            row = s @ H
            denom = float(row @ y)
            denom_scale = float(np.linalg.norm(s) * np.linalg.norm(Hy))
            if (
                np.linalg.norm(s) <= self.vector_tolerance
                or np.linalg.norm(y) <= self.vector_tolerance
            ):
                diags.append(BroydenUpdateDiagnostics(
                    False, "small_vector", denom, denom_scale,
                    len(accepted), np.linalg.matrix_rank(H, tol=self.rank_tolerance),
                    _safe_cond(H), weight,
                ))
                continue
            if abs(denom) <= self.denominator_tolerance * max(denom_scale, np.finfo(float).tiny):
                diags.append(BroydenUpdateDiagnostics(
                    False, "small_denominator", denom, denom_scale,
                    len(accepted), np.linalg.matrix_rank(H, tol=self.rank_tolerance),
                    _safe_cond(H), weight,
                ))
                continue
            correction = np.outer(left, row) / denom
            candidate = H + correction
            if not np.all(np.isfinite(candidate)):
                diags.append(BroydenUpdateDiagnostics(
                    False, "nonfinite_candidate", denom, denom_scale,
                    len(accepted), np.linalg.matrix_rank(H, tol=self.rank_tolerance),
                    float("inf"), weight,
                ))
                continue
            cond = _safe_cond(candidate)
            rank = int(np.linalg.matrix_rank(candidate, tol=self.rank_tolerance))
            if rank < dimension:
                diags.append(BroydenUpdateDiagnostics(
                    False, "rank_deficient_candidate", denom, denom_scale,
                    len(accepted), rank, cond, weight,
                ))
                continue
            if cond > self.max_condition:
                diags.append(BroydenUpdateDiagnostics(
                    False, "condition_limit", denom, denom_scale,
                    len(accepted), rank, cond, weight,
                ))
                continue
            H = candidate
            accepted.append((s.copy(), y.copy(), float(weight)))
            diags.append(BroydenUpdateDiagnostics(
                True, "accepted", denom, denom_scale,
                len(accepted), rank, cond, weight,
            ))
        return H, accepted, diags

    def update_pair(
        self, displacement: object, residual_change: object, *, weight: float = 1.0
    ) -> BroydenUpdateDiagnostics:
        s = _vector(displacement, name="displacement")
        y = _vector(residual_change, name="residual_change")
        if s.shape != y.shape:
            raise ValueError("displacement and residual_change shapes differ")
        weight = float(weight)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("weight must be finite and > 0")
        if self.dimension is None:
            self.dimension = s.size
        if s.size != self.dimension:
            raise ValueError("pair dimension changed")
        prior = list(self._pairs)
        trial = prior + [(s, y, weight)]
        if len(trial) > self.history_cap:
            trial = trial[-self.history_cap :]
        _, accepted, diags = self._matrix_from_pairs(trial)
        latest = diags[-1] if diags else BroydenUpdateDiagnostics(False, "no_update")
        # A rejected newest pair must not evict an older accepted pair.
        if latest.accepted:
            self._pairs = accepted[-self.history_cap :]
            self.accepted_updates += 1
        else:
            self._pairs = prior[-self.history_cap :]
            self.rejected_updates += 1
        self.last_update = latest
        return latest

    def inverse_matrix(self) -> Array:
        if self.dimension is None:
            raise ValueError("dimension is not initialized")
        H, _, _ = self._matrix_from_pairs(list(self._pairs))
        return H

    def propose(self, residual: object) -> BroydenProposal:
        r = _vector(residual, name="residual")
        if self.dimension is None:
            self.dimension = r.size
        if r.size != self.dimension:
            raise ValueError("residual dimension changed")
        H = self.inverse_matrix()
        step = -(H @ r)
        if not np.all(np.isfinite(step)):
            raise FloatingPointError("generic Broyden produced a nonfinite step")
        return BroydenProposal(
            step=step,
            residual_norm=float(np.linalg.norm(r)),
            step_norm=float(np.linalg.norm(step)),
            history_size=len(self._pairs),
            history_rank=int(np.linalg.matrix_rank(H, tol=self.rank_tolerance)),
            condition_number=_safe_cond(H),
            family=self.family,
            residual_convention=self.residual_convention,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "config": {
                "initial_inverse_scale": self.initial_inverse_scale,
                "history_cap": self.history_cap,
                "denominator_tolerance": self.denominator_tolerance,
                "vector_tolerance": self.vector_tolerance,
                "rank_tolerance": self.rank_tolerance,
                "max_condition": self.max_condition,
            },
            "dimension": self.dimension,
            "pairs": [
                {"s": s.tolist(), "y": y.tolist(), "weight": weight}
                for s, y, weight in self._pairs
            ],
            "reset_count": self.reset_count,
            "last_reset_reason": self.last_reset_reason,
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
            "last_update": self.last_update.to_state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "GenericGoodBroydenInverse":
        if state.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported generic Broyden state schema")
        config = dict(state.get("config", {}))
        obj = cls(**config)
        dimension = state.get("dimension")
        obj.dimension = None if dimension is None else int(dimension)
        for item in state.get("pairs", []):
            record = dict(item)
            s = _vector(record["s"], name="serialized displacement")
            y = _vector(record["y"], name="serialized residual_change")
            if obj.dimension is None:
                obj.dimension = s.size
            if s.size != obj.dimension or y.size != obj.dimension:
                raise ValueError("serialized generic Broyden pair dimension mismatch")
            obj._pairs.append((s, y, float(record.get("weight", 1.0))))
        # Verify the serialized accepted history still satisfies this config.
        if obj._pairs:
            _, accepted, diags = obj._matrix_from_pairs(list(obj._pairs))
            if len(accepted) != len(obj._pairs):
                raise ValueError("serialized generic Broyden history is not replayable")
        obj.reset_count = int(state.get("reset_count", 0))
        obj.last_reset_reason = str(state.get("last_reset_reason", "initial"))
        obj.accepted_updates = int(state.get("accepted_updates", len(obj._pairs)))
        obj.rejected_updates = int(state.get("rejected_updates", 0))
        if isinstance(state.get("last_update"), Mapping):
            obj.last_update = BroydenUpdateDiagnostics.from_state_dict(state["last_update"])
        return obj


class JohnsonModifiedBroyden:
    """Johnson/Shang-Liu modified Broyden as a bounded history-space kernel.

    ``G`` is the step operator ``-J^-1``.  The update never forms or
    diagonalizes a physical Hessian.  Rank deficiency of the unregularized
    history Gram matrix is allowed; the explicit ``w0`` term regularizes the
    small history system.  The condition guard applies to that small system.
    """

    SCHEMA = "saddlemill_johnson_modified_broyden_v1"
    family = "johnson_modified_broyden"
    residual_convention = "root_residual_r_step_G_r_G_approximates_minus_J_inverse"

    def __init__(
        self,
        *,
        initial_step_scale: float = 1.0,
        history_cap: int = 20,
        regularization_w0: float = 0.01,
        default_weight: float = 1.0,
        vector_tolerance: float = 1.0e-14,
        rank_tolerance: float = 1.0e-12,
        max_condition: float = 1.0e12,
    ) -> None:
        self.initial_step_scale = float(initial_step_scale)
        self.history_cap = int(history_cap)
        self.regularization_w0 = float(regularization_w0)
        self.default_weight = float(default_weight)
        self.vector_tolerance = float(vector_tolerance)
        self.rank_tolerance = float(rank_tolerance)
        self.max_condition = float(max_condition)
        if not np.isfinite(self.initial_step_scale) or self.initial_step_scale == 0.0:
            raise ValueError("initial_step_scale must be finite and nonzero")
        if self.history_cap < 1:
            raise ValueError("history_cap must be >= 1")
        if not np.isfinite(self.regularization_w0) or self.regularization_w0 <= 0.0:
            raise ValueError("regularization_w0 must be finite and > 0")
        if not np.isfinite(self.default_weight) or self.default_weight <= 0.0:
            raise ValueError("default_weight must be finite and > 0")
        if self.vector_tolerance < 0.0 or self.rank_tolerance < 0.0:
            raise ValueError("tolerances must be >= 0")
        if not np.isfinite(self.max_condition) or self.max_condition <= 1.0:
            raise ValueError("max_condition must be finite and > 1")
        self._pairs: list[tuple[Array, Array, float]] = []
        self.dimension: int | None = None
        self.reset_count = 0
        self.last_reset_reason = "initial"
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.last_update = BroydenUpdateDiagnostics(False, "no_update")

    def empty_copy(self) -> "JohnsonModifiedBroyden":
        return type(self)(
            initial_step_scale=self.initial_step_scale,
            history_cap=self.history_cap,
            regularization_w0=self.regularization_w0,
            default_weight=self.default_weight,
            vector_tolerance=self.vector_tolerance,
            rank_tolerance=self.rank_tolerance,
            max_condition=self.max_condition,
        )

    def reset(self, reason: str = "manual") -> None:
        self._pairs.clear()
        self.dimension = None
        self.reset_count += 1
        self.last_reset_reason = str(reason)
        self.last_update = BroydenUpdateDiagnostics(False, "reset")

    def _history_arrays(self, pairs=None):
        records = self._pairs if pairs is None else pairs
        if not records:
            return None
        S = np.column_stack([item[0] for item in records])
        Y = np.column_stack([item[1] for item in records])
        weights = np.asarray([item[2] for item in records], dtype=float)
        ynorm = np.linalg.norm(Y, axis=0)
        if np.any(ynorm <= self.vector_tolerance):
            raise ValueError("retained Johnson history contains a small residual difference")
        dR = Y / ynorm
        dx = S / ynorm
        W = np.diag(weights)
        gram = dR.T @ dR
        A = W @ gram @ W
        system = (self.regularization_w0 ** 2) * np.eye(len(records)) + A
        return dR, dx, weights, A, system

    def update_pair(
        self, displacement: object, residual_change: object, *, weight: float | None = None
    ) -> BroydenUpdateDiagnostics:
        s = _vector(displacement, name="displacement")
        y = _vector(residual_change, name="residual_change")
        if s.shape != y.shape:
            raise ValueError("displacement and residual_change shapes differ")
        if self.dimension is None:
            self.dimension = s.size
        if s.size != self.dimension:
            raise ValueError("pair dimension changed")
        chosen_weight = self.default_weight if weight is None else float(weight)
        if not np.isfinite(chosen_weight) or chosen_weight <= 0.0:
            raise ValueError("weight must be finite and > 0")
        snorm = float(np.linalg.norm(s))
        if snorm <= self.vector_tolerance:
            self.rejected_updates += 1
            self.last_update = BroydenUpdateDiagnostics(
                False,
                "small_displacement",
                denominator=snorm,
                denominator_scale=1.0,
                history_size=len(self._pairs),
                weight=chosen_weight,
            )
            return self.last_update
        ynorm = float(np.linalg.norm(y))
        if ynorm <= self.vector_tolerance:
            self.rejected_updates += 1
            self.last_update = BroydenUpdateDiagnostics(
                False,
                "small_residual_change",
                denominator=ynorm,
                denominator_scale=1.0,
                history_size=len(self._pairs),
                weight=chosen_weight,
            )
            return self.last_update
        trial = list(self._pairs) + [(s, y, chosen_weight)]
        if len(trial) > self.history_cap:
            trial = trial[-self.history_cap :]
        dR, _, _, A, system = self._history_arrays(trial)
        cond = _safe_cond(system)
        rank = int(np.linalg.matrix_rank(A, tol=self.rank_tolerance))
        if cond > self.max_condition:
            self.rejected_updates += 1
            self.last_update = BroydenUpdateDiagnostics(
                False, "condition_limit", denominator=ynorm, denominator_scale=1.0,
                history_size=len(self._pairs), history_rank=rank,
                condition_number=cond, weight=chosen_weight,
            )
            return self.last_update
        try:
            np.linalg.solve(system, np.eye(system.shape[0]))
        except np.linalg.LinAlgError:
            self.rejected_updates += 1
            self.last_update = BroydenUpdateDiagnostics(
                False, "history_solve_singular", denominator=ynorm, denominator_scale=1.0,
                history_size=len(self._pairs), history_rank=rank,
                condition_number=cond, weight=chosen_weight,
            )
            return self.last_update
        self._pairs = trial
        self.accepted_updates += 1
        self.last_update = BroydenUpdateDiagnostics(
            True, "accepted", denominator=ynorm, denominator_scale=1.0,
            history_size=len(self._pairs), history_rank=rank,
            condition_number=cond, weight=chosen_weight,
        )
        return self.last_update

    def propose(self, residual: object) -> BroydenProposal:
        r = _vector(residual, name="residual")
        if self.dimension is None:
            self.dimension = r.size
        if r.size != self.dimension:
            raise ValueError("residual dimension changed")
        if not self._pairs:
            step = self.initial_step_scale * r
            return BroydenProposal(
                step=step,
                residual_norm=float(np.linalg.norm(r)),
                step_norm=float(np.linalg.norm(step)),
                history_size=0,
                history_rank=0,
                condition_number=1.0,
                family=self.family,
                residual_convention=self.residual_convention,
            )
        dR, dx, weights, A, system = self._history_arrays()
        cond = _safe_cond(system)
        if cond > self.max_condition:
            raise FloatingPointError("Johnson history system exceeds condition limit")
        W = np.diag(weights)
        beta_rhs = W @ (dR.T @ r)
        try:
            coeff = np.linalg.solve(system, beta_rhs)
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError("Johnson history solve failed") from exc
        U = self.initial_step_scale * dR + dx
        step = self.initial_step_scale * r - U @ (W @ coeff)
        if not np.all(np.isfinite(step)):
            raise FloatingPointError("Johnson modified Broyden produced a nonfinite step")
        return BroydenProposal(
            step=step,
            residual_norm=float(np.linalg.norm(r)),
            step_norm=float(np.linalg.norm(step)),
            history_size=len(self._pairs),
            history_rank=int(np.linalg.matrix_rank(A, tol=self.rank_tolerance)),
            condition_number=cond,
            family=self.family,
            residual_convention=self.residual_convention,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "config": {
                "initial_step_scale": self.initial_step_scale,
                "history_cap": self.history_cap,
                "regularization_w0": self.regularization_w0,
                "default_weight": self.default_weight,
                "vector_tolerance": self.vector_tolerance,
                "rank_tolerance": self.rank_tolerance,
                "max_condition": self.max_condition,
            },
            "dimension": self.dimension,
            "pairs": [
                {"s": s.tolist(), "y": y.tolist(), "weight": weight}
                for s, y, weight in self._pairs
            ],
            "reset_count": self.reset_count,
            "last_reset_reason": self.last_reset_reason,
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
            "last_update": self.last_update.to_state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "JohnsonModifiedBroyden":
        if state.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported Johnson Broyden state schema")
        obj = cls(**dict(state.get("config", {})))
        dimension = state.get("dimension")
        obj.dimension = None if dimension is None else int(dimension)
        for item in state.get("pairs", []):
            record = dict(item)
            s = _vector(record["s"], name="serialized displacement")
            y = _vector(record["y"], name="serialized residual_change")
            if obj.dimension is None:
                obj.dimension = s.size
            if s.size != obj.dimension or y.size != obj.dimension:
                raise ValueError("serialized Johnson Broyden pair dimension mismatch")
            obj._pairs.append((s, y, float(record.get("weight", obj.default_weight))))
        if len(obj._pairs) > obj.history_cap:
            raise ValueError("serialized Johnson Broyden history exceeds cap")
        if obj._pairs:
            _, _, _, _, system = obj._history_arrays()
            if _safe_cond(system) > obj.max_condition:
                raise ValueError("serialized Johnson Broyden history exceeds condition limit")
        obj.reset_count = int(state.get("reset_count", 0))
        obj.last_reset_reason = str(state.get("last_reset_reason", "initial"))
        obj.accepted_updates = int(state.get("accepted_updates", len(obj._pairs)))
        obj.rejected_updates = int(state.get("rejected_updates", 0))
        if isinstance(state.get("last_update"), Mapping):
            obj.last_update = BroydenUpdateDiagnostics.from_state_dict(state["last_update"])
        return obj


def broyden_from_state_dict(state: Mapping[str, object]):
    schema = state.get("schema")
    if schema == GenericGoodBroydenInverse.SCHEMA:
        return GenericGoodBroydenInverse.from_state_dict(state)
    if schema == JohnsonModifiedBroyden.SCHEMA:
        return JohnsonModifiedBroyden.from_state_dict(state)
    raise ValueError(f"unsupported Broyden schema {schema!r}")


__all__ = [
    "BroydenProposal",
    "BroydenUpdateDiagnostics",
    "GenericGoodBroydenInverse",
    "JohnsonModifiedBroyden",
    "broyden_from_state_dict",
]
