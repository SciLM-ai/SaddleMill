"""Limited-memory BFGS on an unoriented unit sphere.

This module separates rotational secant construction from how retained vectors
are moved into the current tangent space. Six explicit policies are supported:

``legacy_projection``: ambient pair, projected once at use time.
``double_projection``: projected at endpoint B, then projected again at use.
``sequential_projection``: live history projected after every accepted rotation.
``direct_transport``: pair represented in T_A, directly transported A->current.
``double_transport``: pair constructed in T_B via A->B transport, then B->current.
``sequential_transport``: live history transported along every accepted rotation.

Historical aliases ``projected`` and ``riemannian`` map to
``double_projection`` and ``double_transport`` respectively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Iterable, Mapping, Sequence

import numpy as np

from saddlemill.dimertools.dense_bfgs import (
    DenseSecant,
    compare_directions,
    reconstruct_multisecant_bfgs,
    reconstruct_sequential_bfgs,
    safe_cosine,
    safe_max_atom_norm,
    safe_norm,
)
from saddlemill.dimertools.sphere_manifold import (
    log_map,
    normalize_axis,
    parallel_transport,
    project_tangent_and_basis,
    sign_align_axis,
    sphere_distance,
)

Array = np.ndarray


ROTATION_TRANSPORT_POLICIES = frozenset({
    "legacy_projection",
    "double_projection",
    "sequential_projection",
    "direct_transport",
    "double_transport",
    "sequential_transport",
})


def normalize_rotation_transport_policy(value: object, *, geometry: object = "projected") -> str:
    token = str(value or "").strip().lower()
    aliases = {
        "projected": "double_projection",
        "riemannian": "double_transport",
        "projection": "legacy_projection",
        "transport": "direct_transport",
    }
    if not token or token == "auto":
        base = str(geometry or "projected").strip().lower()
        token = aliases.get(base, base)
    token = aliases.get(token, token)
    if token not in ROTATION_TRANSPORT_POLICIES:
        raise ValueError(
            "rotation transport policy must be one of "
            + ", ".join(sorted(ROTATION_TRANSPORT_POLICIES))
        )
    return token


def sequential_policy(value: object) -> bool:
    try:
        token = normalize_rotation_transport_policy(value)
    except ValueError:
        return False
    return token in {"sequential_projection", "sequential_transport"}


def _flat(value: object) -> Array:
    return np.asarray(value, dtype=float).reshape(-1).copy()


def _pair_metrics(s: Array, y: Array) -> dict[str, float]:
    s = _flat(s)
    y = _flat(y)
    sn = safe_norm(s)
    yn = safe_norm(y)
    sy = float(np.dot(s, y))
    return {
        "s_norm": sn,
        "y_norm": yn,
        "s_dot_y": sy,
        "secant_curvature": 0.0 if sn <= 0.0 else sy / (sn * sn),
        "secant_cosine": 0.0 if sn <= 0.0 or yn <= 0.0 else sy / (sn * yn),
        "force_change_norm": yn,
    }


def _direct_hessian_action(
    initial_hessian: float,
    s_history: Sequence[Array],
    y_history: Sequence[Array],
    vector: object,
) -> Array | None:
    """Apply the direct BFGS Hessian represented by chronological pairs."""

    alpha = float(initial_hessian)
    v = _flat(vector)
    if alpha <= 0.0 or not np.isfinite(alpha) or not np.all(np.isfinite(v)):
        return None
    pairs = [(_flat(s), _flat(y)) for s, y in zip(s_history, y_history)]
    if any(s.shape != v.shape or y.shape != v.shape for s, y in pairs):
        return None
    updates: list[tuple[Array, Array, Array, float, float]] = []
    for s, y in pairs:
        Bs = alpha * s.copy()
        for sj, yj, Bsj, sjBsj, sjyj in updates:
            Bs = (
                Bs
                - Bsj * (float(np.dot(sj, Bs)) / sjBsj)
                + yj * (float(np.dot(yj, s)) / sjyj)
            )
        sBs = float(np.dot(s, Bs))
        sy = float(np.dot(s, y))
        if (
            not np.isfinite(sBs)
            or not np.isfinite(sy)
            or sBs <= 0.0
            or sy <= 0.0
        ):
            return None
        updates.append((s, y, Bs, sBs, sy))
    Bv = alpha * v.copy()
    for s, y, Bs, sBs, sy in updates:
        Bv = (
            Bv
            - Bs * (float(np.dot(s, Bv)) / sBs)
            + y * (float(np.dot(y, v)) / sy)
        )
    return Bv


@dataclass(frozen=True)
class RotationSecant:
    """Raw rotational secant anchored in the tangent space of ``anchor_mode``."""

    anchor_mode: Array
    s: Array
    y: Array
    state_id: int
    serial: int
    source: str
    geometry: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        anchor = normalize_axis(self.anchor_mode)
        s = np.asarray(self.s, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if s.shape != anchor.shape or y.shape != anchor.shape:
            raise ValueError("rotation secant arrays must share the anchor shape")
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
            raise ValueError("rotation secant contains non-finite values")
        object.__setattr__(self, "anchor_mode", anchor.copy())
        object.__setattr__(self, "s", s.copy())
        object.__setattr__(self, "y", y.copy())
        object.__setattr__(self, "state_id", int(self.state_id))
        object.__setattr__(self, "serial", int(self.serial))
        object.__setattr__(self, "source", str(self.source).strip().lower())
        object.__setattr__(self, "geometry", str(self.geometry).strip().lower())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class PreparedRotationPair:
    s: Array
    y: Array
    source: str
    state_id: int
    serial: int
    raw_metrics: Mapping[str, float]
    stored_metrics: Mapping[str, float]
    action: str
    powell_theta: float | str = ""
    s_dot_Bs: float | str = ""


@dataclass(frozen=True)
class RotationLBFGSResult:
    direction: Array
    prepared_pairs: tuple[PreparedRotationPair, ...]
    metrics: Mapping[str, object]


def make_rotation_secant(
    mode_a: object,
    force_a: object,
    mode_b: object,
    force_b: object,
    *,
    geometry: str,
    state_id: int,
    serial: int,
    source: str,
    basis: Iterable[object] | object | None = None,
    metadata: Mapping[str, object] | None = None,
) -> RotationSecant | None:
    """Construct one rotational secant under an explicit transport policy."""

    shape = np.shape(mode_a)
    a = normalize_axis(mode_a, shape=shape)
    b, sign = sign_align_axis(a, mode_b)
    b = b.reshape(shape)
    fa = project_tangent_and_basis(a, force_a, basis)
    fb = sign * np.asarray(force_b, dtype=float)
    fb = project_tangent_and_basis(b, fb, basis)
    token = normalize_rotation_transport_policy(geometry)
    try:
        if token == "legacy_projection":
            anchor = a
            svec = b - a
            yvec = fa - fb
        elif token in {"double_projection", "sequential_projection"}:
            anchor = b
            svec = project_tangent_and_basis(b, b - a, basis)
            yvec = project_tangent_and_basis(b, fa - fb, basis)
        elif token == "direct_transport":
            anchor = a
            step_a = log_map(a, b)
            force_b_at_a = parallel_transport(b, a, fb)
            svec = project_tangent_and_basis(a, step_a, basis)
            yvec = project_tangent_and_basis(a, fa - force_b_at_a, basis)
        elif token in {"double_transport", "sequential_transport"}:
            anchor = b
            step_a = log_map(a, b)
            svec = parallel_transport(a, b, step_a)
            transported_force_a = parallel_transport(a, b, fa)
            svec = project_tangent_and_basis(b, svec, basis)
            yvec = project_tangent_and_basis(b, transported_force_a - fb, basis)
        else:  # pragma: no cover
            raise ValueError(token)
    except ValueError:
        return None
    if safe_norm(svec) <= 1.0e-14:
        return None
    meta = dict(metadata or {})
    meta.update({
        "transport_policy": token,
        "mode_a": np.asarray(a, dtype=float).copy(),
        "mode_b": np.asarray(b, dtype=float).copy(),
    })
    return RotationSecant(
        anchor_mode=anchor,
        s=svec,
        y=yvec,
        state_id=state_id,
        serial=serial,
        source=source,
        geometry=token,
        metadata=meta,
    )


def transport_rotation_secant(
    pair: RotationSecant,
    current_mode: object,
    *,
    basis: Iterable[object] | object | None = None,
) -> tuple[Array, Array] | None:
    """Move one stored secant into the tangent space of ``current_mode``."""

    shape = pair.anchor_mode.shape
    current = normalize_axis(current_mode, shape=shape)
    anchor = pair.anchor_mode.copy()
    svec = pair.s.copy()
    yvec = pair.y.copy()
    policy = normalize_rotation_transport_policy(pair.geometry)
    # The Dimer mode is an unoriented axis.  Moving to the opposite
    # representative of the same axis requires the same sign change for all
    # attached tangent quantities.  Apply that rule uniformly, including the
    # historical projection-only policy.
    if float(np.vdot(anchor.reshape(-1), current.reshape(-1)).real) < 0.0:
        anchor = -anchor
        svec = -svec
        yvec = -yvec

    if policy == "legacy_projection":
        try:
            return (
                np.asarray(project_tangent_and_basis(current, svec, basis), dtype=float),
                np.asarray(project_tangent_and_basis(current, yvec, basis), dtype=float),
            )
        except ValueError:
            return None
    try:
        if policy in {"double_projection", "sequential_projection"}:
            s_current = project_tangent_and_basis(current, svec, basis)
            y_current = project_tangent_and_basis(current, yvec, basis)
        elif policy in {"direct_transport", "double_transport", "sequential_transport"}:
            s_current = parallel_transport(anchor, current, svec)
            y_current = parallel_transport(anchor, current, yvec)
            s_current = project_tangent_and_basis(current, s_current, basis)
            y_current = project_tangent_and_basis(current, y_current, basis)
        else:
            return None
    except ValueError:
        return None
    return np.asarray(s_current, dtype=float), np.asarray(y_current, dtype=float)


class SequentialRotationHistory:
    """Live rotational secants transformed along the accepted optimizer path."""

    def __init__(self, *, policy: str, memory_states: int = 20) -> None:
        token = normalize_rotation_transport_policy(policy)
        if token not in {"sequential_projection", "sequential_transport"}:
            raise ValueError("SequentialRotationHistory requires a sequential policy")
        if int(memory_states) < 1:
            raise ValueError("memory_states must be >= 1")
        self.policy = token
        self.memory_states = int(memory_states)
        self.current_mode: Array | None = None
        self.current_state_id: int | None = None
        self._state_order: list[int] = []
        self._pairs: list[RotationSecant] = []
        self.advance_steps = 0
        self.state_sync_steps = 0
        self.vectors_transformed = 0
        self.pairs_dropped_states = 0
        self._s_step_ratios: list[float] = []
        self._y_step_ratios: list[float] = []
        self._cumulative_path_angle = 0.0

    def begin_state(self, state_id: int, current_mode: object, *, basis=None) -> None:
        state_id = int(state_id)
        mode = normalize_axis(current_mode)
        if self.current_mode is not None:
            aligned, _ = sign_align_axis(self.current_mode, mode)
            aligned = aligned.reshape(self.current_mode.shape)
            if sphere_distance(self.current_mode, aligned) > 1.0e-12:
                self.advance(aligned, basis=basis, reason="state_sync")
            mode = aligned
        self.current_mode = mode.copy()
        self.current_state_id = state_id
        if state_id not in self._state_order:
            self._state_order.append(state_id)
        while len(self._state_order) > self.memory_states:
            dropped = self._state_order.pop(0)
            before = len(self._pairs)
            self._pairs = [pair for pair in self._pairs if pair.state_id != dropped]
            self.pairs_dropped_states += before - len(self._pairs)

    @staticmethod
    def _record_ratio(before: Array, after: Array, bucket: list[float]) -> None:
        old = safe_norm(before)
        new = safe_norm(after)
        if old > 0.0 and np.isfinite(old) and np.isfinite(new):
            bucket.append(float(new / old))

    def advance(self, new_mode: object, *, basis=None, reason: str = "accepted_rotation") -> None:
        if self.current_mode is None:
            self.current_mode = normalize_axis(new_mode)
            return
        old = self.current_mode.copy()
        new, _ = sign_align_axis(old, new_mode)
        new = new.reshape(old.shape)
        angle = sphere_distance(old, new)
        updated: list[RotationSecant] = []
        for pair in self._pairs:
            try:
                if self.policy == "sequential_projection":
                    s_new = project_tangent_and_basis(new, pair.s, basis)
                    y_new = project_tangent_and_basis(new, pair.y, basis)
                    stored_policy = "double_projection"
                else:
                    s_new = parallel_transport(old, new, pair.s)
                    y_new = parallel_transport(old, new, pair.y)
                    s_new = project_tangent_and_basis(new, s_new, basis)
                    y_new = project_tangent_and_basis(new, y_new, basis)
                    stored_policy = "double_transport"
            except ValueError:
                continue
            self._record_ratio(pair.s, s_new, self._s_step_ratios)
            self._record_ratio(pair.y, y_new, self._y_step_ratios)
            meta = dict(pair.metadata)
            meta["sequential_transform_count"] = int(meta.get("sequential_transform_count", 0)) + 1
            meta["sequential_path_angle"] = float(meta.get("sequential_path_angle", 0.0)) + angle
            updated.append(RotationSecant(
                anchor_mode=new, s=s_new, y=y_new, state_id=pair.state_id,
                serial=pair.serial, source=pair.source, geometry=stored_policy,
                metadata=meta,
            ))
            self.vectors_transformed += 2
        self._pairs = updated
        self.current_mode = new.copy()
        self.advance_steps += 1
        self._cumulative_path_angle += angle
        if reason == "state_sync":
            self.state_sync_steps += 1

    def add_trial_pair(self, mode_a, force_a, mode_b, force_b, *, state_id: int, serial: int, basis=None, source="fourier_trial") -> bool:
        if self.current_mode is None:
            self.begin_state(state_id, mode_a, basis=basis)
        if self.policy == "sequential_projection":
            raw = make_rotation_secant(mode_a, force_a, mode_b, force_b, geometry="legacy_projection", state_id=state_id, serial=serial, source=source, basis=basis)
            if raw is None:
                return False
            try:
                svec = project_tangent_and_basis(self.current_mode, raw.s, basis)
                yvec = project_tangent_and_basis(self.current_mode, raw.y, basis)
            except ValueError:
                return False
            stored_policy = "double_projection"
        else:
            raw = make_rotation_secant(mode_a, force_a, mode_b, force_b, geometry="direct_transport", state_id=state_id, serial=serial, source=source, basis=basis)
            if raw is None:
                return False
            transported = transport_rotation_secant(raw, self.current_mode, basis=basis)
            if transported is None:
                return False
            svec, yvec = transported
            stored_policy = "double_transport"
        meta = dict(raw.metadata)
        meta.update({"sequential_transform_count": 0, "sequential_path_angle": 0.0})
        self._pairs.append(RotationSecant(anchor_mode=self.current_mode, s=svec, y=yvec, state_id=state_id, serial=serial, source=source, geometry=stored_policy, metadata=meta))
        return True

    def add_accepted_pair(self, mode_a, force_a, mode_b, force_b, *, state_id: int, serial: int, basis=None, source="accepted_rotation") -> bool:
        construction = "double_projection" if self.policy == "sequential_projection" else "double_transport"
        pair = make_rotation_secant(mode_a, force_a, mode_b, force_b, geometry=construction, state_id=state_id, serial=serial, source=source, basis=basis)
        if pair is None:
            return False
        meta = dict(pair.metadata)
        meta.update({"sequential_transform_count": 0, "sequential_path_angle": 0.0})
        self._pairs.append(RotationSecant(anchor_mode=self.current_mode if self.current_mode is not None else pair.anchor_mode, s=pair.s, y=pair.y, state_id=state_id, serial=serial, source=source, geometry=construction, metadata=meta))
        return True

    def reset(self, reason: str = "manual") -> None:
        self._pairs.clear()

    def pairs(self, *, trial_future_only: bool = False) -> tuple[RotationSecant, ...]:
        if not trial_future_only or self.current_state_id is None:
            return tuple(self._pairs)
        return tuple(pair for pair in self._pairs if not (pair.state_id == self.current_state_id and "fourier_trial" in pair.source))

    @staticmethod
    def _summary(values: list[float], suffix: str) -> dict[str, object]:
        if not values:
            return {f"sequential_{suffix}_median": "", f"sequential_{suffix}_min": "", f"sequential_{suffix}_max": ""}
        arr = np.asarray(values, dtype=float)
        return {f"sequential_{suffix}_median": float(np.median(arr)), f"sequential_{suffix}_min": float(np.min(arr)), f"sequential_{suffix}_max": float(np.max(arr))}

    def metrics(self) -> dict[str, object]:
        counts = [int(pair.metadata.get("sequential_transform_count", 0)) for pair in self._pairs]
        angles = [float(pair.metadata.get("sequential_path_angle", 0.0)) for pair in self._pairs]
        result: dict[str, object] = {
            "sequential_history_policy": self.policy,
            "sequential_history_pairs": len(self._pairs),
            "sequential_history_states": len({pair.state_id for pair in self._pairs}),
            "sequential_history_advance_steps": self.advance_steps,
            "sequential_history_state_sync_steps": self.state_sync_steps,
            "sequential_history_vectors_transformed": self.vectors_transformed,
            "sequential_history_pairs_dropped_states": self.pairs_dropped_states,
            "sequential_history_cumulative_path_angle": self._cumulative_path_angle,
            "sequential_pair_transform_count_max": max(counts) if counts else 0,
            "sequential_pair_path_angle_max": max(angles) if angles else 0.0,
        }
        result.update(self._summary(self._s_step_ratios, "s_norm_step_ratio"))
        result.update(self._summary(self._y_step_ratios, "y_norm_step_ratio"))
        return result


class RotationLBFGSModel:
    """Raw-pair L-BFGS model with optional sphere parallel transport.

    Raw secants are retained; projection/transport and safeguards are replayed
    for the current mode on every application.  That makes the class suitable
    for both local and canonical force-bank histories.
    """

    VALID_GUARDS = frozenset(
        {"off", "legacy_skip", "skip", "damp", "powell", "reset"}
    )

    def __init__(
        self,
        *,
        initial_hessian: float = 1.0,
        dynamic_h0: bool = False,
        curvature_guard: str = "skip",
        curvature_floor: float = 1.0e-3,
        curvature_epsilon: float = 1.0e-12,
        powell_eta: float = 0.2,
        cosine_threshold: float | None = None,
        max_pairs: int = 0,
        reconstruction_model: str = "lbfgs",
        bfgs_update: str = "sequential",
        dense_bfgs_diagnostic: bool = False,
    ) -> None:
        self.initial_hessian = float(initial_hessian)
        self.dynamic_h0 = bool(dynamic_h0)
        self.curvature_guard = str(curvature_guard).strip().lower()
        self.curvature_floor = float(curvature_floor)
        self.curvature_epsilon = float(curvature_epsilon)
        self.powell_eta = float(powell_eta)
        self.cosine_threshold = None if cosine_threshold is None else float(cosine_threshold)
        self.max_pairs = int(max_pairs)
        self.reconstruction_model = str(reconstruction_model).strip().lower()
        self.bfgs_update = str(bfgs_update).strip().lower()
        self.dense_bfgs_diagnostic = bool(dense_bfgs_diagnostic)
        if self.initial_hessian <= 0.0:
            raise ValueError("rotation initial_hessian must be > 0")
        if self.curvature_guard not in self.VALID_GUARDS:
            raise ValueError(
                f"unsupported rotation curvature guard {self.curvature_guard!r}"
            )
        if self.curvature_floor <= 0.0 and self.curvature_guard in {
            "skip",
            "damp",
            "reset",
        }:
            raise ValueError("rotation curvature_floor must be > 0")
        if self.curvature_epsilon < 0.0:
            raise ValueError("rotation curvature_epsilon must be >= 0")
        if self.curvature_guard == "powell" and not 0.0 < self.powell_eta < 1.0:
            raise ValueError("rotation powell_eta must satisfy 0 < eta < 1")
        if self.cosine_threshold is not None and not -1.0 <= self.cosine_threshold <= 1.0:
            raise ValueError("rotation cosine_threshold must lie in [-1,1] or be disabled")
        if self.max_pairs < 0:
            raise ValueError("rotation max_pairs must be >= 0")
        if self.reconstruction_model not in {"lbfgs", "reconstructed_bfgs"}:
            raise ValueError(
                "rotation reconstruction_model must be lbfgs or reconstructed_bfgs"
            )
        if self.bfgs_update not in {"sequential", "multisecant"}:
            raise ValueError(
                "rotation bfgs_update must be sequential or multisecant"
            )
        self.local_pairs: list[RotationSecant] = []
        self.reset_count = 0
        self.last_reset_reason = "initial"
        self.apply_calls = 0
        self.apply_ns_total = 0
        self.last_metrics: dict[str, object] = {}

    def reset(self, reason: str = "manual") -> None:
        self.local_pairs.clear()
        self.reset_count += 1
        self.last_reset_reason = str(reason)

    def add_pair(self, pair: RotationSecant | None) -> bool:
        if pair is None:
            return False
        self.local_pairs.append(pair)
        if self.max_pairs > 0 and len(self.local_pairs) > self.max_pairs:
            del self.local_pairs[: len(self.local_pairs) - self.max_pairs]
        return True

    def _guard_pairs(
        self,
        candidates: Sequence[tuple[RotationSecant, Array, Array]],
    ) -> tuple[list[PreparedRotationPair], dict[str, object]]:
        prepared: list[PreparedRotationPair] = []
        s_history: list[Array] = []
        y_history: list[Array] = []
        rejected: dict[str, int] = {}
        damped = 0
        powell_damped = 0
        reset_events = 0
        for raw_pair, s_array, y_array in candidates:
            s = _flat(s_array)
            y = _flat(y_array)
            raw = _pair_metrics(s, y)
            ss = float(np.dot(s, s))
            sy = float(np.dot(s, y))
            scale = float(np.linalg.norm(s) * np.linalg.norm(y))
            epsilon_threshold = self.curvature_epsilon * max(1.0, scale)
            action = "accept"
            theta: float | str = ""
            sBs_value: float | str = ""
            valid = bool(
                s.shape == y.shape
                and s.size > 0
                and np.all(np.isfinite(s))
                and np.all(np.isfinite(y))
                and np.isfinite(ss)
                and np.isfinite(sy)
                and ss > 0.0
            )
            if not valid:
                rejected["invalid"] = rejected.get("invalid", 0) + 1
                continue

            if self.cosine_threshold is not None:
                sn = float(np.sqrt(ss))
                yn = float(np.linalg.norm(y))
                cosine = sy / (sn * yn) if sn > 0.0 and yn > 0.0 else float("nan")
                if not np.isfinite(cosine) or cosine < self.cosine_threshold:
                    rejected["below_cosine_threshold"] = rejected.get(
                        "below_cosine_threshold", 0
                    ) + 1
                    continue

            guard = self.curvature_guard
            if guard == "off":
                if abs(sy) <= epsilon_threshold:
                    rejected["near_singular"] = rejected.get("near_singular", 0) + 1
                    continue
                action = "off_accept"
            elif guard == "legacy_skip":
                if sy <= epsilon_threshold:
                    rejected["legacy_nonpositive"] = rejected.get(
                        "legacy_nonpositive", 0
                    ) + 1
                    continue
                action = "legacy_accept"
            elif guard == "powell":
                Bs = _direct_hessian_action(
                    self.initial_hessian, s_history, y_history, s
                )
                if Bs is None:
                    rejected["powell_invalid_Bs"] = rejected.get(
                        "powell_invalid_Bs", 0
                    ) + 1
                    continue
                sBs = float(np.dot(s, Bs))
                sBs_value = sBs
                if not np.isfinite(sBs) or sBs <= 0.0:
                    rejected["powell_nonpositive_sBs"] = rejected.get(
                        "powell_nonpositive_sBs", 0
                    ) + 1
                    continue
                threshold = self.powell_eta * sBs
                if sy < threshold:
                    denominator = sBs - sy
                    if not np.isfinite(denominator) or denominator <= 0.0:
                        rejected["powell_invalid_denominator"] = rejected.get(
                            "powell_invalid_denominator", 0
                        ) + 1
                        continue
                    theta_value = (1.0 - self.powell_eta) * sBs / denominator
                    if not 0.0 < theta_value < 1.0:
                        rejected["powell_invalid_theta"] = rejected.get(
                            "powell_invalid_theta", 0
                        ) + 1
                        continue
                    y = theta_value * y + (1.0 - theta_value) * Bs
                    theta = theta_value
                    action = "powell_damp"
                    damped += 1
                    powell_damped += 1
                else:
                    theta = 1.0
                    action = "powell_accept"
            else:
                threshold = self.curvature_floor * ss
                if sy < threshold:
                    if guard == "damp":
                        y = y + ((threshold - sy) / ss) * s
                        action = "shifted_secant_damp"
                        damped += 1
                    elif guard == "reset":
                        prepared.clear()
                        s_history.clear()
                        y_history.clear()
                        reset_events += 1
                        rejected["curvature_reset"] = rejected.get(
                            "curvature_reset", 0
                        ) + 1
                        continue
                    else:  # skip
                        rejected["below_curvature_floor"] = rejected.get(
                            "below_curvature_floor", 0
                        ) + 1
                        continue
            stored = _pair_metrics(s, y)
            stored_sy = float(stored["s_dot_y"])
            stored_threshold = self.curvature_epsilon * max(
                1.0, float(stored["s_norm"]) * float(stored["y_norm"])
            )
            valid_stored_denominator = (
                abs(stored_sy) > stored_threshold
                if guard == "off" else stored_sy > stored_threshold
            )
            if not np.isfinite(stored_sy) or not valid_stored_denominator:
                rejected["stored_nonpositive"] = rejected.get(
                    "stored_nonpositive", 0
                ) + 1
                continue
            s_history.append(s)
            y_history.append(y)
            prepared.append(
                PreparedRotationPair(
                    s=s,
                    y=y,
                    source=raw_pair.source,
                    state_id=raw_pair.state_id,
                    serial=raw_pair.serial,
                    raw_metrics=raw,
                    stored_metrics=stored,
                    action=action,
                    powell_theta=theta,
                    s_dot_Bs=sBs_value,
                )
            )
        return prepared, {
            "pairs_rejected_by_reason": rejected,
            "pairs_damped": damped,
            "pairs_powell_damped": powell_damped,
            "reset_events": reset_events,
        }

    def apply(
        self,
        force: object,
        current_mode: object,
        *,
        external_pairs: Sequence[RotationSecant] = (),
        basis: Iterable[object] | object | None = None,
    ) -> RotationLBFGSResult:
        started = perf_counter_ns()
        current = normalize_axis(current_mode)
        projected_force = project_tangent_and_basis(current, force, basis)
        raw_pairs = sorted(
            [*external_pairs, *self.local_pairs],
            # Sequential histories use per-eigensolve sample serials, which
            # restart at each translation center. State IDs provide the outer
            # chronology; sorting serial first can reorder BFGS updates and
            # make max_pairs retain an older state's pairs over newer ones.
            key=lambda item: (item.state_id, item.serial, item.source),
        )
        pair_candidates_pre_truncation = len(raw_pairs)
        # Apply max_pairs after transport + guard admission so rejected newer
        # candidates can be backfilled by older admissible pairs.
        pairs_removed_by_max_pairs = 0
        pairs_admissible_before_max_pairs = 0
        candidates: list[tuple[RotationSecant, Array, Array]] = []
        transport_rejected = 0
        s_ratios: list[float] = []
        y_ratios: list[float] = []
        transform_counts: list[int] = []
        path_angles: list[float] = []
        for pair in raw_pairs:
            transported = transport_rotation_secant(pair, current, basis=basis)
            if transported is None:
                transport_rejected += 1
                continue
            s_now, y_now = transported
            sn0, yn0 = safe_norm(pair.s), safe_norm(pair.y)
            sn1, yn1 = safe_norm(s_now), safe_norm(y_now)
            if sn0 > 0.0 and np.isfinite(sn0) and np.isfinite(sn1):
                s_ratios.append(sn1 / sn0)
            if yn0 > 0.0 and np.isfinite(yn0) and np.isfinite(yn1):
                y_ratios.append(yn1 / yn0)
            transform_counts.append(int(pair.metadata.get("sequential_transform_count", 0)))
            path_angles.append(float(pair.metadata.get("sequential_path_angle", 0.0)))
            candidates.append((pair, s_now, y_now))
        prepared, guard_metrics = self._guard_pairs(candidates)
        pairs_admissible_before_max_pairs = len(prepared)
        if self.max_pairs > 0 and len(prepared) > self.max_pairs:
            pairs_removed_by_max_pairs = len(prepared) - self.max_pairs
            prepared = prepared[-self.max_pairs :]

        q = _flat(projected_force)
        alphas: list[float] = []
        for pair in reversed(prepared):
            sy = float(np.dot(pair.s, pair.y))
            rho = 1.0 / sy
            alpha = rho * float(np.dot(pair.s, q))
            alphas.append(alpha)
            q -= alpha * pair.y
        h0 = 1.0 / self.initial_hessian
        if self.dynamic_h0 and prepared:
            latest = prepared[-1]
            yy = float(np.dot(latest.y, latest.y))
            sy = float(np.dot(latest.s, latest.y))
            if yy > 0.0 and sy > 0.0:
                h0 = sy / yy
        r = h0 * q
        for pair, alpha in zip(prepared, reversed(alphas)):
            sy = float(np.dot(pair.s, pair.y))
            beta = float(np.dot(pair.y, r)) / sy
            r += pair.s * (alpha - beta)
        lbfgs_direction = project_tangent_and_basis(
            current, r.reshape(current.shape), basis
        )

        dense_result = None
        dense_fallback_reason = ""
        if self.dense_bfgs_diagnostic or self.reconstruction_model == "reconstructed_bfgs":
            dense_pairs = [
                DenseSecant(
                    s=pair.s, y=pair.y, source=pair.source,
                    state_id=pair.state_id, serial=pair.serial
                )
                for pair in prepared
            ]
            if self.bfgs_update == "multisecant":
                dense_result = reconstruct_multisecant_bfgs(
                    dense_pairs, projected_force,
                    initial_hessian=self.initial_hessian,
                    dynamic_h0=self.dynamic_h0,
                    curvature_epsilon=self.curvature_epsilon,
                )
            else:
                dense_result = reconstruct_sequential_bfgs(
                    dense_pairs, projected_force,
                    initial_hessian=self.initial_hessian,
                    dynamic_h0=self.dynamic_h0,
                    denominator_epsilon=self.curvature_epsilon,
                    require_positive_definite=(self.curvature_guard != "off"),
                )

        direction = np.asarray(lbfgs_direction, dtype=float)
        if self.reconstruction_model == "reconstructed_bfgs":
            if dense_result is not None and dense_result.direction is not None:
                direction = project_tangent_and_basis(
                    current, dense_result.direction.reshape(current.shape), basis
                )
            else:
                dense_fallback_reason = (
                    "dense_reconstruction_invalid:"
                    + str(
                        "" if dense_result is None
                        else dense_result.metrics.get("dense_invalid_reason", "")
                    )
                )
        elapsed = perf_counter_ns() - started
        self.apply_calls += 1
        self.apply_ns_total += int(elapsed)
        source_counts: dict[str, int] = {}
        states: set[int] = set()
        for pair in prepared:
            source_counts[pair.source] = source_counts.get(pair.source, 0) + 1
            states.add(pair.state_id)
        latest = prepared[-1] if prepared else None
        def _ratio_summary(values):
            if not values:
                return ("", "", "")
            arr = np.asarray(values, dtype=float)
            return (float(np.median(arr)), float(np.min(arr)), float(np.max(arr)))
        s_med, s_min, s_max = _ratio_summary(s_ratios)
        y_med, y_min, y_max = _ratio_summary(y_ratios)
        accepted_curvatures = [float(p.stored_metrics.get("secant_curvature", np.nan)) for p in prepared]
        accepted_cosines = [float(p.stored_metrics.get("secant_cosine", np.nan)) for p in prepared]
        accepted_curvatures = [v for v in accepted_curvatures if np.isfinite(v)]
        accepted_cosines = [v for v in accepted_cosines if np.isfinite(v)]
        def _triple(vals):
            if not vals:
                return ("", "", "")
            arr = np.asarray(vals, dtype=float)
            return (float(np.min(arr)), float(np.median(arr)), float(np.max(arr)))
        curv_min, curv_med, curv_max = _triple(accepted_curvatures)
        cos_min, cos_med, cos_max = _triple(accepted_cosines)
        rejected_reasons = dict(guard_metrics["pairs_rejected_by_reason"])
        metrics: dict[str, object] = {
            "geometry": raw_pairs[-1].geometry if raw_pairs else "",
            "curvature_guard": self.curvature_guard,
            "curvature_floor": self.curvature_floor,
            "reconstruction_model": self.reconstruction_model,
            "bfgs_update": self.bfgs_update,
            "dense_bfgs_diagnostic": int(self.dense_bfgs_diagnostic),
            "dense_bfgs_fallback_reason": dense_fallback_reason,
            "powell_eta": self.powell_eta,
            "cosine_threshold": "" if self.cosine_threshold is None else self.cosine_threshold,
            "pair_candidates": pair_candidates_pre_truncation,
            "pair_candidates_after_max_pairs": len(raw_pairs),
            "pairs_admissible_before_max_pairs": pairs_admissible_before_max_pairs,
            "pairs_removed_by_max_pairs": pairs_removed_by_max_pairs,
            "pairs_transportable": len(candidates),
            "pairs_used": len(prepared),
            "pairs_rejected_transport": transport_rejected,
            "transport_s_norm_ratio_median": s_med,
            "transport_s_norm_ratio_min": s_min,
            "transport_s_norm_ratio_max": s_max,
            "transport_y_norm_ratio_median": y_med,
            "transport_y_norm_ratio_min": y_min,
            "transport_y_norm_ratio_max": y_max,
            "transport_pair_transform_count_max": max(transform_counts) if transform_counts else 0,
            "transport_pair_path_angle_max": max(path_angles) if path_angles else 0.0,
            "pairs_damped_before_max_pairs": guard_metrics["pairs_damped"],
            "pairs_powell_damped_before_max_pairs": guard_metrics["pairs_powell_damped"],
            "pairs_damped": sum(
                pair.action in {"shifted_secant_damp", "powell_damp"}
                for pair in prepared
            ),
            "pairs_powell_damped": sum(
                pair.action == "powell_damp" for pair in prepared
            ),
            "pair_resets": guard_metrics["reset_events"],
            "pairs_rejected_by_reason": rejected_reasons,
            "pairs_rejected_curvature": sum(
                int(v) for k, v in rejected_reasons.items()
                if "curvature" in k or "nonpositive" in k or "near_singular" in k
            ),
            "pairs_rejected_cosine": int(rejected_reasons.get("below_cosine_threshold", 0)),
            "accepted_curvature_min": curv_min,
            "accepted_curvature_median": curv_med,
            "accepted_curvature_max": curv_max,
            "accepted_cosine_min": cos_min,
            "accepted_cosine_median": cos_med,
            "accepted_cosine_max": cos_max,
            "pairs_by_source": source_counts,
            "states_contributing": len(states),
            "h0_inverse_scale": h0,
            "apply_ns": int(elapsed),
            "apply_cumulative_ns": self.apply_ns_total,
            "apply_calls": self.apply_calls,
            "history_resets": self.reset_count,
            "last_reset_reason": self.last_reset_reason,
            "latest_pair_action": "" if latest is None else latest.action,
            "latest_raw_s_dot_y": ""
            if latest is None
            else latest.raw_metrics.get("s_dot_y", ""),
            "latest_stored_s_dot_y": ""
            if latest is None
            else latest.stored_metrics.get("s_dot_y", ""),
            "latest_powell_theta": "" if latest is None else latest.powell_theta,
            "latest_s_dot_Bs": "" if latest is None else latest.s_dot_Bs,
            "lbfgs_raw_direction_norm": safe_norm(lbfgs_direction),
            "lbfgs_raw_direction_max_atom_norm": (
                safe_max_atom_norm(lbfgs_direction)
                if np.asarray(lbfgs_direction).size % 3 == 0 else ""
            ),
            "direction_norm": safe_norm(direction),
            "direction_force_cosine": safe_cosine(direction, projected_force),
        }
        if dense_result is not None:
            metrics.update(dense_result.metrics)
            comparison = compare_directions(lbfgs_direction, dense_result.direction)
            metrics.update({
                "dense_vs_lbfgs_cosine": comparison.get("dense_vs_primary_cosine", ""),
                "dense_vs_lbfgs_norm_ratio": comparison.get("dense_vs_primary_norm_ratio", ""),
                "dense_vs_lbfgs_relative_difference": comparison.get(
                    "dense_vs_primary_relative_difference", ""
                ),
            })
        self.last_metrics = metrics
        return RotationLBFGSResult(
            direction=np.asarray(direction, dtype=float),
            prepared_pairs=tuple(prepared),
            metrics=metrics,
        )


__all__ = [
    "PreparedRotationPair",
    "RotationLBFGSModel",
    "RotationLBFGSResult",
    "RotationSecant",
    "SequentialRotationHistory",
    "ROTATION_TRANSPORT_POLICIES",
    "normalize_rotation_transport_policy",
    "sequential_policy",
    "make_rotation_secant",
    "transport_rotation_secant",
]
