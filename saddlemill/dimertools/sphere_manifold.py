"""Unit-sphere geometry used by rotational minimum-mode optimizers.

The Dimer axis is represented by a unit vector in the flattened Cartesian
space.  ``n`` and ``-n`` describe the same physical axis, so all operations
first sign-align axes to the local reference hemisphere.

This module contains geometry only.  It knows nothing about force evaluators,
Dimer stencils, or L-BFGS history ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from saddlemill.dimertools.dense_bfgs import safe_norm

Array = np.ndarray


def _flat(value: object, *, name: str) -> Array:
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and nonempty")
    return array.reshape(-1).copy()


def inner(left: object, right: object) -> float:
    """Real Euclidean inner product of flattened arrays."""

    l = _flat(left, name="left")
    r = _flat(right, name="right")
    if l.shape != r.shape:
        raise ValueError(f"inner-product shapes differ: {l.shape} and {r.shape}")
    return float(np.vdot(l, r).real)


def normalize_axis(value: object, *, shape: tuple[int, ...] | None = None) -> Array:
    """Return a finite unit vector, optionally reshaped like the caller's mode."""

    array = _flat(value, name="axis")
    magnitude = safe_norm(array)
    if not np.isfinite(magnitude) or magnitude <= 1.0e-15:
        raise ValueError("axis norm is approximately zero")
    result = array / magnitude
    return result.reshape(shape if shape is not None else np.shape(value))


def sign_align_axis(reference: object, axis: object) -> tuple[Array, float]:
    """Sign-align ``axis`` to ``reference`` because a Dimer axis is unoriented.

    Returns ``(aligned_axis, sign)``.  A force-like tangent attached to the axis
    must be multiplied by the same sign.
    """

    reference_array = normalize_axis(reference)
    axis_array = normalize_axis(axis, shape=reference_array.shape)
    sign = -1.0 if inner(reference_array, axis_array) < 0.0 else 1.0
    return sign * axis_array, sign


def project_tangent(axis: object, vector: object) -> Array:
    """Orthogonally project ``vector`` into the tangent plane at ``axis``."""

    shape = np.shape(vector)
    n = normalize_axis(axis).reshape(-1)
    v = _flat(vector, name="vector")
    if n.shape != v.shape:
        raise ValueError(f"axis/vector shapes differ: {n.shape} and {v.shape}")
    projected = v - float(np.dot(n, v)) * n
    return projected.reshape(shape)


def project_tangent_and_basis(
    axis: object,
    vector: object,
    basis: Iterable[object] | object | None = None,
) -> Array:
    """Project into the sphere tangent plane and optional orthogonal basis."""

    result = project_tangent(axis, vector)
    if basis is None:
        return result
    if isinstance(basis, np.ndarray) and basis.shape == result.shape:
        items = [basis]
    else:
        try:
            items = list(basis)  # type: ignore[arg-type]
        except TypeError:
            items = [basis]
    for item in items:
        base = normalize_axis(item, shape=result.shape)
        result = result - inner(result, base) * base
    return project_tangent(axis, result)


def sphere_distance(start: object, end: object, *, axis_equivalence: bool = True) -> float:
    """Shortest great-circle angle in radians."""

    x = normalize_axis(start).reshape(-1)
    y = normalize_axis(end, shape=x.shape).reshape(-1)
    dot = float(np.clip(np.dot(x, y), -1.0, 1.0))
    if axis_equivalence and dot < 0.0:
        y = -y
        dot = -dot
    # ``arccos(dot)`` loses all resolution when a small angle rounds dot to
    # exactly 1.  The perpendicular component retains the first-order angle,
    # so atan2 is stable both near zero and near pi.
    perpendicular = y - dot * x
    sine = safe_norm(perpendicular)
    return float(np.arctan2(sine, dot))


def exp_map(axis: object, tangent: object) -> Array:
    """Sphere exponential map ``Exp_axis(tangent)``."""

    shape = np.shape(axis)
    n = normalize_axis(axis).reshape(-1)
    eta = project_tangent(n, tangent).reshape(-1)
    theta = safe_norm(eta)
    if theta <= 1.0e-15:
        return n.reshape(shape)
    result = np.cos(theta) * n + np.sin(theta) * (eta / theta)
    return normalize_axis(result, shape=shape)


def normalized_retraction(axis: object, tangent: object) -> Array:
    """First-order normalized-addition retraction on the unit sphere."""

    shape = np.shape(axis)
    n = normalize_axis(axis).reshape(-1)
    eta = project_tangent(n, tangent).reshape(-1)
    return normalize_axis(n + eta, shape=shape)


def log_map(start: object, end: object, *, axis_equivalence: bool = True) -> Array:
    """Sphere logarithm map from ``start`` to ``end``.

    For a Dimer axis, ``end`` is sign-aligned to ``start`` by default.  The
    antipodal singularity is therefore avoided unless the input itself is
    numerically degenerate.
    """

    shape = np.shape(start)
    x = normalize_axis(start).reshape(-1)
    y = normalize_axis(end, shape=x.shape).reshape(-1)
    if axis_equivalence and float(np.dot(x, y)) < 0.0:
        y = -y
    dot = float(np.clip(np.dot(x, y), -1.0, 1.0))
    tangent = y - dot * x
    magnitude = safe_norm(tangent)
    if magnitude <= 1.0e-15:
        if dot > 0.0:
            return np.zeros_like(x).reshape(shape)
        raise ValueError("sphere logarithm is singular for antipodal axes")
    # atan2 preserves tiny geodesic angles that arccos(dot) rounds to zero.
    theta = float(np.arctan2(magnitude, dot))
    return (theta / magnitude * tangent).reshape(shape)


def parallel_transport(
    start: object,
    end: object,
    tangent: object,
    *,
    axis_equivalence: bool = True,
) -> Array:
    """Exact parallel transport along the shortest sphere geodesic.

    The closed form is ``v - <v,y>/(1+<x,y>) * (x+y)`` for unit ``x`` and ``y``.
    A projection fallback is used only for an effectively zero geodesic.
    """

    shape = np.shape(tangent)
    x = normalize_axis(start).reshape(-1)
    y = normalize_axis(end, shape=x.shape).reshape(-1)
    v = project_tangent(x, tangent).reshape(-1)
    if axis_equivalence and float(np.dot(x, y)) < 0.0:
        y = -y
    dot = float(np.clip(np.dot(x, y), -1.0, 1.0))
    geodesic_tangent = y - dot * x
    if safe_norm(geodesic_tangent) <= 1.0e-15 and dot > 0.0:
        return project_tangent(y, v).reshape(shape)
    denominator = 1.0 + dot
    if denominator <= 1.0e-12:
        raise ValueError("parallel transport is singular for antipodal axes")
    transported = v - (float(np.dot(v, y)) / denominator) * (x + y)
    return project_tangent(y, transported).reshape(shape)


def cap_tangent_for_map(
    tangent: object,
    *,
    max_angle: float,
    map_kind: str,
) -> tuple[Array, bool, float]:
    """Cap a tangent proposal so the accepted geodesic angle is bounded.

    For the exponential map, tangent norm equals angle.  For normalized
    retraction, the accepted angle is ``atan(||eta||)`` and the corresponding
    tangent cap is ``tan(max_angle)``.
    """

    eta = np.asarray(tangent, dtype=float).copy()
    norm = safe_norm(eta)
    max_angle = float(max_angle)
    if not np.isfinite(max_angle) or not 0.0 < max_angle <= np.pi / 2.0:
        raise ValueError("max_angle must satisfy 0 < max_angle <= pi/2")
    token = str(map_kind).strip().lower()
    if token == "exponential":
        cap = max_angle
    elif token == "retraction":
        cap = float(np.tan(max_angle))
    else:
        raise ValueError("map_kind must be exponential or retraction")
    if norm <= cap or norm <= 1.0e-15:
        actual = norm if token == "exponential" else float(np.arctan(norm))
        return eta, False, actual
    eta *= cap / norm
    return eta, True, max_angle


@dataclass(frozen=True)
class SphereStep:
    """Result of applying a tangent step to a sphere point."""

    axis: Array
    tangent: Array
    requested_norm: float
    actual_angle: float
    clipped: bool
    map_kind: str


def take_sphere_step(
    axis: object,
    tangent: object,
    *,
    max_angle: float,
    map_kind: str,
) -> SphereStep:
    """Cap and apply an exponential or normalized-retraction step."""

    n = normalize_axis(axis)
    eta = project_tangent(n, tangent)
    requested = safe_norm(eta)
    capped, clipped, _ = cap_tangent_for_map(
        eta, max_angle=max_angle, map_kind=map_kind
    )
    token = str(map_kind).strip().lower()
    if token == "exponential":
        new_axis = exp_map(n, capped)
    elif token == "retraction":
        new_axis = normalized_retraction(n, capped)
    else:  # pragma: no cover - cap_tangent_for_map already validates
        raise ValueError("map_kind must be exponential or retraction")
    actual = sphere_distance(n, new_axis)
    return SphereStep(
        axis=new_axis,
        tangent=np.asarray(capped, dtype=float),
        requested_norm=requested,
        actual_angle=actual,
        clipped=bool(clipped),
        map_kind=token,
    )


__all__ = [
    "SphereStep",
    "cap_tangent_for_map",
    "exp_map",
    "inner",
    "log_map",
    "normalize_axis",
    "normalized_retraction",
    "parallel_transport",
    "project_tangent",
    "project_tangent_and_basis",
    "sign_align_axis",
    "sphere_distance",
    "take_sphere_step",
]
