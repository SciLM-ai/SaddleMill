"""Per-attempt physical-force observation boundary.

This object is not a calculator wrapper.  It records forces already requested
by the active minimum-mode implementation and therefore does not alter the
calculator, force values, cache behavior, or force-call count.  Semantic source
contexts let one generic history distinguish Dimer rotation, Lanczos, Davidson,
reference-Hessian probes, and future methods.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Iterator, Mapping, Optional

from saddlemill.dimertools.foundation_types import WorkPurpose

from saddlemill.dimertools.force_history import (
    CanonicalForceHistory,
    source_family,
)


@dataclass(frozen=True)
class ForceSourceContext:
    source: str
    family: str
    metadata: dict[str, object]
    purpose: str = WorkPurpose.ALGORITHM.value


class PhysicalForceEvaluator:
    """Record projection-independent force observations for one attempt."""

    def __init__(
        self,
        history: CanonicalForceHistory,
        *,
        default_probe_source: str = "rotation",
        default_probe_family: str | None = None,
    ) -> None:
        self.history = history
        self.default_probe_source = str(default_probe_source).strip().lower()
        self.default_probe_family = source_family(
            self.default_probe_source, default_probe_family
        )
        self._contexts: list[ForceSourceContext] = []
        self.state_begin_calls = 0
        self.center_record_calls = 0
        self.probe_record_calls = 0
        self.recording_operations = 0
        self.recording_ns_total = 0
        self.recording_ns_since_reset = 0

    @property
    def current_context(self) -> ForceSourceContext:
        if self._contexts:
            return self._contexts[-1]
        return ForceSourceContext(
            self.default_probe_source,
            self.default_probe_family,
            {},
            WorkPurpose.ALGORITHM.value,
        )

    @contextmanager
    def source(
        self,
        source: object,
        metadata: Mapping[str, object] | None = None,
        *,
        family: object | None = None,
        purpose: object = WorkPurpose.ALGORITHM.value,
    ) -> Iterator[None]:
        source_token = str(source).strip().lower()
        if not source_token:
            raise ValueError("force source context cannot be empty")
        context = ForceSourceContext(
            source_token,
            source_family(source_token, family),
            dict(metadata or {}),
            WorkPurpose(str(purpose).strip().lower()).value,
        )
        self._contexts.append(context)
        try:
            yield
        finally:
            popped = self._contexts.pop()
            if popped is not context:
                raise RuntimeError("force source context stack was corrupted")

    def _timed(self, callback):
        started = perf_counter_ns()
        try:
            return callback()
        finally:
            elapsed = perf_counter_ns() - started
            self.recording_operations += 1
            self.recording_ns_total += elapsed
            self.recording_ns_since_reset += elapsed
            self.history.accounting.observation_bookkeeping_ns += max(0, int(elapsed))

    def consume_recording_ns(self) -> int:
        value = int(self.recording_ns_since_reset)
        self.recording_ns_since_reset = 0
        return value

    def begin_state(
        self,
        center_positions: object,
        *,
        metadata: Mapping[str, object] | None = None,
        active_dof_mask: object | None = None,
        coordinate_convention: str = "unspecified",
    ):
        self.state_begin_calls += 1
        return self._timed(
            lambda: self.history.begin_state(
                center_positions,
                metadata=dict(metadata or {}),
                active_dof_mask=active_dof_mask,
                coordinate_convention=coordinate_convention,
            )
        )

    def record_center(
        self,
        positions: object,
        forces: object,
        *,
        evaluation: str,
        force_call_delta: int,
        metadata: Mapping[str, object] | None = None,
        purpose: object | None = None,
        cache_hit: bool | None = None,
    ):
        context = self.current_context
        merged = dict(context.metadata)
        if metadata:
            merged.update(dict(metadata))
        merged.setdefault("source_context", context.source)
        self.center_record_calls += 1
        return self._timed(
            lambda: self.history.observe_center(
                positions,
                forces,
                evaluation=evaluation,
                force_call_delta=force_call_delta,
                metadata=merged,
                purpose=context.purpose if purpose is None else WorkPurpose(str(purpose).strip().lower()).value,
                cache_hit=cache_hit,
            )
        )

    def record_probe(
        self,
        positions: object,
        forces: object,
        *,
        evaluation: str,
        force_call_delta: int,
        source: Optional[object] = None,
        family: Optional[object] = None,
        metadata: Mapping[str, object] | None = None,
        direction: object | None = None,
        direction_kind: str = "",
        stencil_id: str = "",
        dimer_side: int = 0,
        offset: float = 0.0,
        purpose: object | None = None,
        cache_hit: bool | None = None,
    ):
        context = self.current_context
        source_token = context.source if source is None else str(source).strip().lower()
        family_token = (
            context.family if family is None else source_family(source_token, family)
        )
        merged = dict(context.metadata)
        if metadata:
            merged.update(dict(metadata))
        self.probe_record_calls += 1
        return self._timed(
            lambda: self.history.observe_probe(
                positions,
                forces,
                source=source_token,
                family=family_token,
                evaluation=evaluation,
                force_call_delta=force_call_delta,
                metadata=merged,
                direction=direction,
                direction_kind=direction_kind,
                stencil_id=stencil_id,
                dimer_side=dimer_side,
                offset=offset,
                purpose=context.purpose if purpose is None else WorkPurpose(str(purpose).strip().lower()).value,
                cache_hit=cache_hit,
            )
        )

    def finalize_state(
        self,
        *,
        mode: object | None,
        curvature: float | None,
        solver: object | None,
        translation_regime: object | None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._timed(
            lambda: self.history.finalize_current(
                mode=mode,
                curvature=curvature,
                solver=solver,
                translation_regime=translation_regime,
                metadata=metadata,
            )
        )

    def metrics(self) -> dict[str, object]:
        return {
            "state_begin_calls": int(self.state_begin_calls),
            "center_record_calls": int(self.center_record_calls),
            "probe_record_calls": int(self.probe_record_calls),
            "recording_operations": int(self.recording_operations),
            "recording_ns_total": int(self.recording_ns_total),
            "force_accounting": self.history.accounting.to_state_dict(),
        }

    def to_state_dict(self) -> dict[str, object]:
        if self._contexts:
            raise RuntimeError("cannot serialize PhysicalForceEvaluator with an active source context")
        return {
            "schema": "saddlemill_physical_force_evaluator_state_v1",
            "default_probe_source": self.default_probe_source,
            "default_probe_family": self.default_probe_family,
            "state_begin_calls": self.state_begin_calls,
            "center_record_calls": self.center_record_calls,
            "probe_record_calls": self.probe_record_calls,
            "recording_operations": self.recording_operations,
            "recording_ns_total": self.recording_ns_total,
            "recording_ns_since_reset": self.recording_ns_since_reset,
            "history": self.history.to_state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "PhysicalForceEvaluator":
        if state.get("schema") != "saddlemill_physical_force_evaluator_state_v1":
            raise ValueError("unsupported PhysicalForceEvaluator state schema")
        result = cls(
            CanonicalForceHistory.from_state_dict(dict(state["history"])),
            default_probe_source=str(state.get("default_probe_source", "rotation")),
            default_probe_family=str(state.get("default_probe_family", "")) or None,
        )
        for name in (
            "state_begin_calls", "center_record_calls", "probe_record_calls",
            "recording_operations", "recording_ns_total", "recording_ns_since_reset",
        ):
            setattr(result, name, int(state.get(name, 0)))
        return result


__all__ = ["ForceSourceContext", "PhysicalForceEvaluator"]
