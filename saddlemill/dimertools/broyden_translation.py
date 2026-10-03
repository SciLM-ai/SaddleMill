"""Cartesian translation adapter for quasi-Newton/rotation Broyden kernels.

The residual supplied here is the selected effective/MMF translation residual.
It can be nonconservative and have a nonsymmetric Jacobian.  This adapter never
labels its model as a physical Hessian and contains no force damping policy.

W5-002 identity rule
--------------------
Translation history belongs to a *definition* of the residual, not to one
center/mode-solve sequence.  Ordinary accepted center translations therefore
retain secants.  A caller-supplied ``history_identity`` must change only when
the residual definition or active coordinate space changes.  Deserializing a
saved adapter is an explicit restart boundary and clears numerical history on
the first subsequent step.  Legacy v1 states remain readable, but their old
per-center ``sequence_identity`` is intentionally not trusted as a history
definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from saddlemill.dimertools.broyden import broyden_from_state_dict

Array = np.ndarray


@dataclass(frozen=True)
class TranslationBroydenStep:
    step: Array
    raw_step_norm: float
    accepted_step_norm: float
    clipped: bool
    kernel_family: str
    history_size: int
    reset_reason: str


class TranslationBroydenAdapter:
    LEGACY_SCHEMA = "saddlemill_translation_broyden_adapter_v1"
    SCHEMA = "saddlemill_translation_broyden_adapter_v2"

    def __init__(self, kernel, *, max_step_norm: float | None = None) -> None:
        if not hasattr(kernel, "propose") or not hasattr(kernel, "update_pair"):
            raise TypeError("kernel must implement propose and update_pair")
        self.kernel = kernel
        self.max_step_norm = None if max_step_norm is None else float(max_step_norm)
        if self.max_step_norm is not None and (
            not np.isfinite(self.max_step_norm) or self.max_step_norm <= 0.0
        ):
            raise ValueError("max_step_norm must be finite and > 0 when provided")
        self._previous_position: Array | None = None
        self._previous_residual: Array | None = None
        self._history_identity: str | None = None
        self._restart_pending = False
        self._restart_reason = "restart"
        self.reset_count = 0
        self.last_reset_reason = "initial"

    @property
    def history_identity(self) -> str | None:
        return self._history_identity

    def reset(self, reason: str = "manual") -> None:
        self.kernel.reset(reason)
        self._previous_position = None
        self._previous_residual = None
        self.reset_count += 1
        self.last_reset_reason = str(reason)

    def _reset_for_restart_if_needed(self) -> None:
        if not self._restart_pending:
            return
        self.reset(self._restart_reason)
        # The restart itself already invalidates every saved secant.  Adopt the
        # first post-restart definition as fresh rather than recording a second
        # reset if the definition also changed while the process was down.
        self._history_identity = None
        self._restart_pending = False

    def step(
        self,
        position: object,
        residual: object,
        *,
        history_identity: object | None = None,
        sequence_identity: object | None = None,
        weight: float = 1.0,
    ) -> TranslationBroydenStep:
        """Propose one translation step and update cross-center secant history.

        ``history_identity`` is the v2 definition identity.  ``sequence_identity``
        is retained only as a source-compatible alias for older direct callers;
        new runtime wiring must not pass the per-center damping/scheduler
        sequence ID here.
        """
        x = np.asarray(position, dtype=float)
        r = np.asarray(residual, dtype=float)
        if x.size == 0 or x.shape != r.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(r)):
            raise ValueError("translation position/residual must be finite, nonempty, and shape matched")
        if history_identity is not None and sequence_identity is not None:
            raise ValueError("pass history_identity or legacy sequence_identity, not both")

        self._reset_for_restart_if_needed()

        identity = history_identity if history_identity is not None else sequence_identity
        if identity is not None:
            token = str(identity)
            if self._history_identity is not None and token != self._history_identity:
                self.reset("history_definition_changed")
            self._history_identity = token

        if self._previous_position is not None:
            s = (x - self._previous_position).reshape(-1)
            y = (r - self._previous_residual).reshape(-1)
            self.kernel.update_pair(s, y, weight=weight)
        proposal = self.kernel.propose(r.reshape(-1))
        raw = proposal.step.reshape(x.shape)
        raw_norm = float(np.linalg.norm(raw.reshape(-1)))
        accepted = raw.copy()
        clipped = False
        if self.max_step_norm is not None and raw_norm > self.max_step_norm:
            accepted *= self.max_step_norm / raw_norm
            clipped = True
        self._previous_position = x.copy()
        self._previous_residual = r.copy()
        return TranslationBroydenStep(
            step=accepted,
            raw_step_norm=raw_norm,
            accepted_step_norm=float(np.linalg.norm(accepted.reshape(-1))),
            clipped=clipped,
            kernel_family=proposal.family,
            history_size=proposal.history_size,
            reset_reason=self.last_reset_reason,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "kernel": self.kernel.state_dict(),
            "max_step_norm": self.max_step_norm,
            "previous_position": None if self._previous_position is None else self._previous_position.tolist(),
            "previous_residual": None if self._previous_residual is None else self._previous_residual.tolist(),
            "history_identity": self._history_identity,
            "reset_count": self.reset_count,
            "last_reset_reason": self.last_reset_reason,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "TranslationBroydenAdapter":
        schema = state.get("schema")
        if schema not in {cls.SCHEMA, cls.LEGACY_SCHEMA}:
            raise ValueError("unsupported translation Broyden adapter state schema")
        obj = cls(
            broyden_from_state_dict(dict(state["kernel"])),
            max_step_norm=state.get("max_step_norm"),
        )
        px = state.get("previous_position")
        pr = state.get("previous_residual")
        obj._previous_position = None if px is None else np.asarray(px, dtype=float)
        obj._previous_residual = None if pr is None else np.asarray(pr, dtype=float)
        if schema == cls.SCHEMA:
            obj._history_identity = state.get("history_identity")
            obj._restart_reason = "resume_restart"
        else:
            # v1 serialized a per-center sequence token.  It cannot safely be
            # promoted to the v2 residual-definition identity.  Accept the old
            # checkpoint, then restart from the first post-resume center.
            obj._history_identity = None
            obj._restart_reason = "legacy_v1_resume_restart"
        obj.reset_count = int(state.get("reset_count", 0))
        obj.last_reset_reason = str(state.get("last_reset_reason", "initial"))
        obj._restart_pending = True
        return obj


__all__ = ["TranslationBroydenAdapter", "TranslationBroydenStep"]
