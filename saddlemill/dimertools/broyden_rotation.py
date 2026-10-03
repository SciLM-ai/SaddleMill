"""Sphere/tangent adapter for quasi-Newton/rotation Broyden kernels.

The adapter consumes an already-defined rotational residual.  It does not
compute Dimer forces or own force observations.  Secants are constructed in
tangent spaces and transported with the C1/canonical-history sphere geometry before the
selected Broyden kernel is replayed at the current mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from saddlemill.dimertools.broyden import broyden_from_state_dict
from saddlemill.dimertools.sphere_manifold import (
    log_map,
    normalize_axis,
    parallel_transport,
    project_tangent,
    sign_align_axis,
    take_sphere_step,
)

Array = np.ndarray


@dataclass(frozen=True)
class RotationBroydenStep:
    axis: Array
    tangent: Array
    requested_norm: float
    actual_angle: float
    clipped: bool
    kernel_family: str
    history_size: int
    reset_reason: str


class RotationBroydenAdapter:
    SCHEMA = "saddlemill_rotation_broyden_adapter_v1"

    def __init__(self, kernel, *, max_angle: float, map_kind: str = "retraction") -> None:
        if not hasattr(kernel, "propose") or not hasattr(kernel, "update_pair"):
            raise TypeError("kernel must implement propose and update_pair")
        self.kernel = kernel
        self.max_angle = float(max_angle)
        self.map_kind = str(map_kind).strip().lower()
        # Validate angle/map using a harmless two-dimensional tangent.
        take_sphere_step(np.array([1.0, 0.0]), np.array([0.0, 0.0]), max_angle=self.max_angle, map_kind=self.map_kind)
        self._pairs: list[tuple[Array, Array, Array, float]] = []
        self._previous_mode: Array | None = None
        self._previous_residual: Array | None = None
        self._center_identity: str | None = None
        self._mode_identity: str | None = None
        self.reset_count = 0
        self.last_reset_reason = "initial"

    def reset(self, reason: str = "manual") -> None:
        self._pairs.clear()
        self._previous_mode = None
        self._previous_residual = None
        self.kernel.reset(reason)
        self.reset_count += 1
        self.last_reset_reason = str(reason)

    def _identity_reset(self, center_identity: object, mode_identity: object) -> None:
        center = str(center_identity)
        mode = str(mode_identity)
        if self._center_identity is not None and center != self._center_identity:
            self.reset("center_identity_changed")
        elif self._mode_identity is not None and mode != self._mode_identity:
            self.reset("mode_identity_changed")
        self._center_identity = center
        self._mode_identity = mode

    def _rebuild_kernel(self, current_mode: Array) -> None:
        fresh = self.kernel.empty_copy()
        for anchor, s, y, weight in self._pairs:
            try:
                s_now = parallel_transport(anchor, current_mode, s)
                y_now = parallel_transport(anchor, current_mode, y)
            except ValueError:
                continue
            fresh.update_pair(s_now.reshape(-1), y_now.reshape(-1), weight=weight)
        self.kernel = fresh

    def step(
        self,
        mode: object,
        residual: object,
        *,
        center_identity: object,
        mode_identity: object = "lowest_mode",
        weight: float = 1.0,
    ) -> RotationBroydenStep:
        current = normalize_axis(mode)
        raw_residual = np.asarray(residual, dtype=float)
        if raw_residual.shape != current.shape or not np.all(np.isfinite(raw_residual)):
            raise ValueError("rotation residual must be finite and match mode shape")
        self._identity_reset(center_identity, mode_identity)

        if self._previous_mode is not None:
            aligned, sign = sign_align_axis(self._previous_mode, current)
            current = aligned.reshape(self._previous_mode.shape)
            current_residual = project_tangent(current, sign * raw_residual)
            previous_residual = project_tangent(self._previous_mode, self._previous_residual)
            s_at_previous = log_map(self._previous_mode, current)
            try:
                s_current = parallel_transport(self._previous_mode, current, s_at_previous)
                r_previous_current = parallel_transport(
                    self._previous_mode, current, previous_residual
                )
                y_current = current_residual - r_previous_current
                self._pairs.append((current.copy(), s_current.copy(), y_current.copy(), float(weight)))
                cap = int(getattr(self.kernel, "history_cap", len(self._pairs)))
                self._pairs = self._pairs[-cap:]
            except ValueError:
                self.reset("sphere_transport_failure")
                self._center_identity = str(center_identity)
                self._mode_identity = str(mode_identity)
                current_residual = project_tangent(current, sign * raw_residual)
        else:
            current_residual = project_tangent(current, raw_residual)

        self._rebuild_kernel(current)
        proposal = self.kernel.propose(current_residual.reshape(-1))
        tangent = project_tangent(current, proposal.step.reshape(current.shape))
        sphere_step = take_sphere_step(
            current, tangent, max_angle=self.max_angle, map_kind=self.map_kind
        )
        self._previous_mode = current.copy()
        self._previous_residual = current_residual.copy()
        return RotationBroydenStep(
            axis=sphere_step.axis,
            tangent=sphere_step.tangent,
            requested_norm=sphere_step.requested_norm,
            actual_angle=sphere_step.actual_angle,
            clipped=sphere_step.clipped,
            kernel_family=proposal.family,
            history_size=proposal.history_size,
            reset_reason=self.last_reset_reason,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "kernel": self.kernel.state_dict(),
            "max_angle": self.max_angle,
            "map_kind": self.map_kind,
            "pairs": [
                {"anchor": a.tolist(), "s": s.tolist(), "y": y.tolist(), "weight": w}
                for a, s, y, w in self._pairs
            ],
            "previous_mode": None if self._previous_mode is None else self._previous_mode.tolist(),
            "previous_residual": None if self._previous_residual is None else self._previous_residual.tolist(),
            "center_identity": self._center_identity,
            "mode_identity": self._mode_identity,
            "reset_count": self.reset_count,
            "last_reset_reason": self.last_reset_reason,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "RotationBroydenAdapter":
        if state.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported rotation Broyden adapter state schema")
        obj = cls(
            broyden_from_state_dict(dict(state["kernel"])),
            max_angle=float(state["max_angle"]),
            map_kind=str(state.get("map_kind", "retraction")),
        )
        obj._pairs = [
            (
                np.asarray(item["anchor"], dtype=float),
                np.asarray(item["s"], dtype=float),
                np.asarray(item["y"], dtype=float),
                float(item.get("weight", 1.0)),
            )
            for item in state.get("pairs", [])
        ]
        previous_mode = state.get("previous_mode")
        previous_residual = state.get("previous_residual")
        obj._previous_mode = None if previous_mode is None else np.asarray(previous_mode, dtype=float)
        obj._previous_residual = None if previous_residual is None else np.asarray(previous_residual, dtype=float)
        obj._center_identity = state.get("center_identity")
        obj._mode_identity = state.get("mode_identity")
        obj.reset_count = int(state.get("reset_count", 0))
        obj.last_reset_reason = str(state.get("last_reset_reason", "initial"))
        return obj


__all__ = ["RotationBroydenAdapter", "RotationBroydenStep"]
