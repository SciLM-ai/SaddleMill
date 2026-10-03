"""Calculator-free physical-Hessian quasi-Newton models.

This module owns a dense *physical* Hessian approximation built only from raw
physical-gradient information.  It deliberately does not import ASE, a
calculator, or the minimum-mode solver implementation.

Two update families are kept scientifically distinct:

``bfgs``
    Ordinary Hessian BFGS.  The sequential translation update is exactly
    ``B+ = B + yy.T/(s.T y) - (Bs)(Bs).T/(s.T B s)`` and therefore preserves
    the historical SaddleMill behavior that an admissible negative ``s.T y``
    is accepted when both scalar denominators are safely nonzero.

``ts_bfgs``
    Sella-compatible transition-state BFGS using an indefinite symmetric
    physical ``B`` and the positive spectral metric ``|B|``.  The multisecant
    formula mirrors the locally supplied, modified Sella
    ``hessian_update.py::_MS_TS_BFGS`` implementation; no runtime Sella import
    or site-package patching is used.

All observations are interpreted with SaddleMill's canonical sign convention
``F = -g``.  Thus center observations are converted back to physical gradients
as ``g = -F`` before a translation secant is formed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter_ns
from typing import Iterable, Mapping, Sequence

import numpy as np

from saddlemill.dimertools.force_history import ForceObservation
from saddlemill.dimertools.foundation_types import (
    ALGORITHM_PHYSICAL_ADMISSION,
    ActiveCoordinateSpace,
    ObservationAdmissionPolicy,
    OperatorOrigin,
    WorkPurpose,
)
from saddlemill.dimertools.hvp_interfaces import HVPResult

Array = np.ndarray

PHYSICAL_HESSIAN_STATE_SCHEMA = "saddlemill_physical_hessian_model_v1"
PHYSICAL_HESSIAN_UPDATE_SCHEMA = "saddlemill_physical_hessian_update_v1"
PHYSICAL_HESSIAN_SPECTRUM_SCHEMA = "saddlemill_physical_hessian_spectrum_v1"

ORDINARY_BFGS = "bfgs"
TS_BFGS = "ts_bfgs"
SUPPORTED_UPDATE_TYPES = (ORDINARY_BFGS, TS_BFGS)

REPRESENTATION_ONLINE_DENSE = "online_dense"
REPRESENTATION_FORCEBANK_WINDOW_DENSE = "forcebank_window_dense"
REPRESENTATION_FORCEBANK_WINDOW_COMPACT = "forcebank_window_compact"
SUPPORTED_REPRESENTATIONS = (
    REPRESENTATION_ONLINE_DENSE,
    REPRESENTATION_FORCEBANK_WINDOW_DENSE,
    REPRESENTATION_FORCEBANK_WINDOW_COMPACT,
)

# The rank tolerance matches the typed HVP/Rayleigh-Ritz interface default.
DEFAULT_RANK_RELATIVE_TOLERANCE = 1.0e-10
DEFAULT_RANK_ABSOLUTE_TOLERANCE = 1.0e-12
# A dependent direction whose discarded HVP component exceeds this fraction of
# the whole block is not a compatible linear-action block.
DEFAULT_DEPENDENCE_NOISE_TOLERANCE = 1.0e-6
# Relative antisymmetry/correction threshold for S.T@Y before Sella-style
# symmetrization.  This is deliberately configurable because finite-difference
# noise is experiment dependent; the public selector remains integration-owned.
DEFAULT_BLOCK_SYMMETRY_NOISE_TOLERANCE = 1.0e-3
DEFAULT_SECANT_RESIDUAL_TOLERANCE = 1.0e-8
DEFAULT_DENOMINATOR_TOLERANCE = 1.0e-12
DEFAULT_SPECTRAL_ZERO_TOLERANCE = 1.0e-12


class PhysicalHessianError(RuntimeError):
    """Base error for physical-Hessian model contract violations."""


class PhysicalHessianIdentityError(PhysicalHessianError):
    """Observation/HVP identity does not belong to the current model center."""


@dataclass(frozen=True)
class PhysicalHessianSpectrum:
    """Immutable low-spectrum and inertia diagnostics for the model matrix."""

    eigenvalues: tuple[float, ...]
    negative_eigenvalue_count: int
    lambda_min: float
    lambda_max: float
    selected_root_index: int
    selected_low_root_eigengap: float | None
    absolute_eigenvalue_min: float
    absolute_eigenvalue_max: float
    absolute_eigenvalue_spread: float
    absolute_eigenvalue_condition: float | None
    condition_state: str
    model_age: int
    timing_ns: int
    schema: str = PHYSICAL_HESSIAN_SPECTRUM_SCHEMA


@dataclass(frozen=True)
class PhysicalHessianUpdateResult:
    """Immutable diagnostic record for one attempted model update."""

    update_type: str
    source: str
    observation_ids: tuple[str, ...]
    accepted: bool
    reason: str
    input_block_size: int
    processed_block_rank: int
    dropped_dependent_rank: int
    dependence_noise_ratio: float | None
    symmetry_residual_before: float | None
    symmetry_correction_ratio: float | None
    symmetry_residual_after: float | None
    secant_residual: float | None
    negative_eigenvalue_count: int
    lambda_min: float
    lambda_max: float
    selected_low_root_eigengap: float | None
    absolute_eigenvalue_spread: float
    absolute_eigenvalue_condition: float | None
    condition_state: str
    model_age: int
    timing_ns: int
    schema: str = PHYSICAL_HESSIAN_UPDATE_SCHEMA


@dataclass(frozen=True)
class _ProcessedBlock:
    S: Array
    Y: Array
    input_size: int
    rank: int
    dropped_rank: int
    dependence_noise_ratio: float
    symmetry_residual_before: float
    symmetry_correction_ratio: float
    symmetry_residual_after: float


class _ActiveBasis:
    """Reduced orthonormal basis implied by ``ActiveCoordinateSpace``.

    Fixed Cartesian coordinates are removed first.  Any explicit typed-HVP null basis
    is then removed within that movable-coordinate vector space.  The physical
    Hessian is stored only in the remaining reduced coordinates; no artificial
    diagonal values are inserted for fixed/null DOFs.
    """

    def __init__(self, coordinate_space: ActiveCoordinateSpace) -> None:
        self.coordinate_space = coordinate_space
        mask_flat = coordinate_space.active_dof_mask.reshape(-1)
        self.active_indices = np.flatnonzero(mask_flat)
        active_count = int(self.active_indices.size)
        if active_count == 0:
            raise ValueError("physical Hessian requires at least one active Cartesian DOF")

        if coordinate_space.null_basis:
            null = np.column_stack(
                [basis.reshape(-1)[self.active_indices] for basis in coordinate_space.null_basis]
            )
            # typed-HVP already orthonormalizes after masking, but QR makes this module
            # robust to serialization-level roundoff.
            q, _ = np.linalg.qr(null, mode="complete")
            null_rank = len(coordinate_space.null_basis)
            reduced = q[:, null_rank:]
        else:
            reduced = np.eye(active_count, dtype=float)
        if reduced.shape[1] < 1:
            raise ValueError("active coordinate space is exhausted by its explicit null basis")
        self.basis_active = np.asarray(reduced, dtype=float)
        self.dimension = int(self.basis_active.shape[1])
        self.full_dimension = int(mask_flat.size)
        self.shape = coordinate_space.active_dof_mask.shape

    def reduce(self, value: object) -> Array:
        projected = self.coordinate_space.project(value).reshape(-1)
        active = projected[self.active_indices]
        return np.asarray(self.basis_active.T @ active, dtype=float)

    def expand(self, value: object) -> Array:
        reduced = np.asarray(value, dtype=float).reshape(-1)
        if reduced.size != self.dimension:
            raise ValueError(
                f"reduced vector has dimension {reduced.size}; expected {self.dimension}"
            )
        active = self.basis_active @ reduced
        full = np.zeros(self.full_dimension, dtype=float)
        full[self.active_indices] = active
        return full.reshape(self.shape)

    def embedded_matrix(self, matrix: Array) -> Array:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("reduced matrix shape mismatch")
        active_matrix = self.basis_active @ matrix @ self.basis_active.T
        full = np.zeros((self.full_dimension, self.full_dimension), dtype=float)
        full[np.ix_(self.active_indices, self.active_indices)] = active_matrix
        return full


def _readonly_copy(value: Array) -> Array:
    out = np.array(value, dtype=float, copy=True)
    out.setflags(write=False)
    return out


def _finite_matrix(value: object, dimension: int, label: str) -> Array:
    out = np.asarray(value, dtype=float)
    if out.shape != (dimension, dimension):
        raise ValueError(f"{label} must have shape {(dimension, dimension)}; got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{label} contains non-finite values")
    return 0.5 * (out + out.T)


def _state_uid_from_observation(observation: ForceObservation) -> str:
    return f"state{int(observation.state_id)}:{observation.geometry_id}"


def _hvp_identity(result: HVPResult, projected_direction: Array) -> str:
    digest = hashlib.sha256()
    digest.update(b"saddlemill-physical-hvp-admission-v1\0")
    meta = result.metadata
    for token in (
        meta.state_uid,
        meta.geometry_id,
        meta.coordinate_space_id,
        meta.source,
        meta.family,
        str(meta.operator_origin),
        result.stencil_scheme,
        repr(result.displacement_scale),
    ):
        digest.update(str(token).encode("utf-8"))
        digest.update(b"\0")
    for obs_id in result.endpoint_observation_ids:
        digest.update(str(obs_id).encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.ascontiguousarray(projected_direction, dtype="<f8").tobytes())
    return "hvp:" + digest.hexdigest()


def _symmetrize_y2(S: Array, Y: Array) -> Array:
    """Local NumPy transcription of modified Sella ``symmetrize_Y2``.

    The supplied Sella reference performs a sequential correction so that the
    admitted multisecant block is compatible with a symmetric matrix.  We copy
    the algebra rather than importing/patching Sella at runtime.
    """

    _, nvecs = S.shape
    if nvecs <= 1:
        return np.array(Y, dtype=float, copy=True)
    dY = np.zeros_like(Y)
    YTS = Y.T @ S
    dYTS = np.zeros_like(YTS)
    STS = S.T @ S
    for i in range(1, nvecs):
        lhs = STS[:i, :i]
        rhs = YTS[i, :i].T - YTS[:i, i] - dYTS[:i, i]
        coeff = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        dY[:, i] = -S[:, :i] @ coeff
        dYTS[i, :] = -STS[:, :i] @ coeff
    return Y + dY


def _relative_norm(numerator: Array, denominator: Array, floor: float) -> float:
    return float(np.linalg.norm(numerator) / max(float(np.linalg.norm(denominator)), floor))


class PhysicalHessianModel:
    """Dense physical-Hessian approximation in typed-HVP active coordinates.

    Parameters in this constructor are method settings for this model only.
    shared-runtime owns the public selector/default wiring.  Since the whole feature is new,
    no existing SaddleMill path changes unless shared-runtime explicitly instantiates it.
    """

    def __init__(
        self,
        coordinate_space: ActiveCoordinateSpace,
        *,
        update_type: str = ORDINARY_BFGS,
        initial_hessian: float = 1.0,
        initial_matrix: Array | None = None,
        representation: str = REPRESENTATION_ONLINE_DENSE,
        denominator_tolerance: float = DEFAULT_DENOMINATOR_TOLERANCE,
        rank_relative_tolerance: float = DEFAULT_RANK_RELATIVE_TOLERANCE,
        rank_absolute_tolerance: float = DEFAULT_RANK_ABSOLUTE_TOLERANCE,
        dependence_noise_tolerance: float = DEFAULT_DEPENDENCE_NOISE_TOLERANCE,
        block_symmetry_noise_tolerance: float = DEFAULT_BLOCK_SYMMETRY_NOISE_TOLERANCE,
        secant_residual_tolerance: float = DEFAULT_SECANT_RESIDUAL_TOLERANCE,
        spectral_zero_tolerance: float = DEFAULT_SPECTRAL_ZERO_TOLERANCE,
        admission_policy: ObservationAdmissionPolicy = ALGORITHM_PHYSICAL_ADMISSION,
    ) -> None:
        if not isinstance(coordinate_space, ActiveCoordinateSpace):
            raise TypeError("coordinate_space must be ActiveCoordinateSpace")
        update_type = str(update_type).strip().lower().replace("-", "_")
        if update_type not in SUPPORTED_UPDATE_TYPES:
            raise ValueError(f"unsupported physical Hessian update_type: {update_type!r}")
        representation = str(representation).strip().lower().replace("-", "_")
        if representation not in SUPPORTED_REPRESENTATIONS:
            raise ValueError(f"unsupported physical Hessian representation: {representation!r}")
        if representation != REPRESENTATION_ONLINE_DENSE and update_type != TS_BFGS:
            raise ValueError("ForceBank-windowed physical Hessian representations require update_type=ts_bfgs")
        if representation != REPRESENTATION_ONLINE_DENSE and initial_matrix is not None:
            raise ValueError("ForceBank-windowed physical Hessian currently requires scalar B0=beta*I")
        self.coordinate_space = coordinate_space
        self._basis = _ActiveBasis(coordinate_space)
        self.update_type = update_type
        self.representation = representation
        self.admission_policy = admission_policy

        self.denominator_tolerance = self._positive(
            denominator_tolerance, "denominator_tolerance"
        )
        self.rank_relative_tolerance = self._nonnegative(
            rank_relative_tolerance, "rank_relative_tolerance"
        )
        self.rank_absolute_tolerance = self._nonnegative(
            rank_absolute_tolerance, "rank_absolute_tolerance"
        )
        self.dependence_noise_tolerance = self._nonnegative(
            dependence_noise_tolerance, "dependence_noise_tolerance"
        )
        self.block_symmetry_noise_tolerance = self._nonnegative(
            block_symmetry_noise_tolerance, "block_symmetry_noise_tolerance"
        )
        self.secant_residual_tolerance = self._positive(
            secant_residual_tolerance, "secant_residual_tolerance"
        )
        self.spectral_zero_tolerance = self._nonnegative(
            spectral_zero_tolerance, "spectral_zero_tolerance"
        )

        scale = float(initial_hessian)
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("initial_hessian must be finite and nonzero")
        self._initial_hessian_scalar = scale
        self._compact_basis = np.empty((self.dimension, 0), dtype=float)
        self._compact_correction = np.empty((0, 0), dtype=float)
        self._window_pair_ids: tuple[str, ...] = ()
        self._window_pair_fingerprint = ""
        self._window_diagnostics: dict[str, object] = {
            "representation": representation,
            "raw_candidate_count": 0,
            "valid_ts_bfgs_pair_count_before_cap": 0,
            "pairs_removed_by_cap": 0,
            "configured_cap": 0,
            "pair_ids_used": (),
            "pair_sources_used": (),
            "reconstructed_rank": 0,
            "compact_rank": 0,
            "reconstruction_time_ns": 0,
            "spectral_time_ns": 0,
        }
        if representation == REPRESENTATION_ONLINE_DENSE:
            if initial_matrix is None:
                self._matrix: Array | None = np.eye(self.dimension, dtype=float) * scale
            else:
                raw = np.asarray(initial_matrix, dtype=float)
                if raw.shape == (self.full_dimension, self.full_dimension):
                    # Project a full Cartesian matrix into the explicit reduced basis.
                    active = raw[np.ix_(self._basis.active_indices, self._basis.active_indices)]
                    raw = self._basis.basis_active.T @ active @ self._basis.basis_active
                self._matrix = _finite_matrix(raw, self.dimension, "initial_matrix")
        elif representation == REPRESENTATION_FORCEBANK_WINDOW_DENSE:
            self._matrix = np.eye(self.dimension, dtype=float) * scale
        else:
            self._matrix = None

        self._previous_position: Array | None = None
        self._previous_gradient: Array | None = None
        self._previous_observation_id = ""
        self._current_state_uid = ""
        self._current_geometry_id = ""
        self._seen_observation_ids: set[str] = set()
        self._seen_hvp_ids: set[str] = set()
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.model_age = 0

    @staticmethod
    def _positive(value: float, label: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
        return result

    @staticmethod
    def _nonnegative(value: float, label: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{label} must be finite and >= 0")
        return result

    @property
    def dimension(self) -> int:
        return self._basis.dimension

    @property
    def full_dimension(self) -> int:
        return self._basis.full_dimension

    @property
    def has_dense_matrix(self) -> bool:
        return self._matrix is not None

    @property
    def matrix(self) -> Array:
        """Read-only reduced Hessian for dense representations only.

        Compact production paths deliberately fail rather than silently materialize
        an ``n x n`` matrix. Consumers that support compact models must use the
        spectral/operator capability instead.
        """

        if self._matrix is None:
            raise PhysicalHessianError("compact_physical_hessian_has_no_dense_matrix")
        return _readonly_copy(self._matrix)

    def embedded_matrix(self) -> Array:
        """Full Cartesian embedding for dense representations only."""

        if self._matrix is None:
            raise PhysicalHessianError("compact_physical_hessian_has_no_dense_matrix")
        return _readonly_copy(self._basis.embedded_matrix(self._matrix))

    def _compact_apply_reduced(self, reduced: Array) -> Array:
        vector = np.asarray(reduced, dtype=float).reshape(-1)
        result = self._initial_hessian_scalar * vector
        if self._compact_basis.shape[1]:
            result = result + self._compact_basis @ (
                self._compact_correction @ (self._compact_basis.T @ vector)
            )
        return np.asarray(result, dtype=float)

    def apply_reduced(self, vector: object) -> Array:
        reduced = np.asarray(vector, dtype=float).reshape(-1)
        if reduced.size != self.dimension:
            raise ValueError("reduced physical-Hessian vector dimension mismatch")
        if self._matrix is not None:
            return np.asarray(self._matrix @ reduced, dtype=float)
        return self._compact_apply_reduced(reduced)

    def apply(self, vector: object) -> Array:
        """Apply the approximate physical Hessian and return Cartesian shape."""

        reduced = self._basis.reduce(vector)
        return self._basis.expand(self.apply_reduced(reduced))

    def low_spectrum(self, count: int | None = None, *, selected_root_index: int = 0) -> PhysicalHessianSpectrum:
        start = perf_counter_ns()
        evals = self.full_eigenvalues()
        selected_root_index = int(selected_root_index)
        if selected_root_index < 0 or selected_root_index >= evals.size:
            raise ValueError("selected_root_index is outside the model spectrum")
        diagnostic = self._spectrum_from_evals(evals, selected_root_index, perf_counter_ns() - start)
        if count is None:
            return diagnostic
        count = int(count)
        if count < 1:
            raise ValueError("count must be >= 1")
        trimmed = tuple(float(v) for v in evals[: min(count, evals.size)])
        return PhysicalHessianSpectrum(
            eigenvalues=trimmed,
            negative_eigenvalue_count=diagnostic.negative_eigenvalue_count,
            lambda_min=diagnostic.lambda_min,
            lambda_max=diagnostic.lambda_max,
            selected_root_index=diagnostic.selected_root_index,
            selected_low_root_eigengap=diagnostic.selected_low_root_eigengap,
            absolute_eigenvalue_min=diagnostic.absolute_eigenvalue_min,
            absolute_eigenvalue_max=diagnostic.absolute_eigenvalue_max,
            absolute_eigenvalue_spread=diagnostic.absolute_eigenvalue_spread,
            absolute_eigenvalue_condition=diagnostic.absolute_eigenvalue_condition,
            condition_state=diagnostic.condition_state,
            model_age=diagnostic.model_age,
            timing_ns=diagnostic.timing_ns,
        )

    def compact_eigensystem(self) -> tuple[float, Array, Array]:
        """Return ``(beta, learned_eigenvalues, learned_eigenvectors)``.

        The learned eigenvectors are reduced-coordinate columns. The orthogonal
        complement has the exact repeated eigenvalue ``beta``. No dense full
        matrix is formed.
        """

        beta = float(self._initial_hessian_scalar)
        if self._matrix is not None:
            evals, evecs = np.linalg.eigh(self._matrix)
            return 0.0, np.asarray(evals, dtype=float), np.asarray(evecs, dtype=float)
        if self._compact_basis.shape[1] == 0:
            return beta, np.empty(0, dtype=float), np.empty((self.dimension, 0), dtype=float)
        evals_k, vectors_k = np.linalg.eigh(0.5 * (self._compact_correction + self._compact_correction.T))
        learned = beta + evals_k
        vectors = self._compact_basis @ vectors_k
        order = np.argsort(learned, kind="stable")
        return beta, np.asarray(learned[order], dtype=float), np.asarray(vectors[:, order], dtype=float)

    def full_eigenvalues(self) -> Array:
        if self._matrix is not None:
            return np.linalg.eigvalsh(self._matrix)
        beta, learned, _vectors = self.compact_eigensystem()
        complement = max(0, self.dimension - learned.size)
        values = np.concatenate((learned, np.full(complement, beta, dtype=float)))
        return np.sort(values, kind="stable")

    def _compact_abs_apply_reduced(self, vector: Array) -> Array:
        beta, learned, vectors = self.compact_eigensystem()
        x = np.asarray(vector, dtype=float).reshape(-1)
        result = abs(beta) * x
        if learned.size:
            projected = vectors.T @ x
            result = result + vectors @ ((np.abs(learned) - abs(beta)) * projected)
        return np.asarray(result, dtype=float)

    def qn_mmf_fixed_alpha_reduced(
        self, gradient: object, *, alpha: float, order: int = 1
    ) -> tuple[Array, Array, dict[str, object]]:
        """Exact target-order QN/MMF step from the model spectral capability.

        For compact ``beta I + Q K Q.T`` models, only the learned subspace is
        diagonalized. The orthogonal gradient component is treated analytically
        with baseline curvature ``beta``.
        """

        g = np.asarray(gradient, dtype=float).reshape(-1)
        if g.size != self.dimension or not np.all(np.isfinite(g)):
            raise ValueError("invalid QN/MMF gradient for physical Hessian")
        a = float(alpha)
        ord_i = int(order)
        if ord_i != 1:
            raise ValueError("compact physical-Hessian QN/MMF currently supports target order 1")
        beta, learned, vectors = self.compact_eigensystem()
        complement_dim = self.dimension - learned.size
        if complement_dim and (learned.size == 0 or float(learned[0]) >= beta - self.spectral_zero_tolerance):
            raise PhysicalHessianError(
                "compact_qn_mmf_requires_selected_lowest_root_in_learned_subspace"
            )
        projected = vectors.T @ g if learned.size else np.empty(0, dtype=float)
        g_perp = g - (vectors @ projected if learned.size else 0.0)
        signs = np.ones(learned.size, dtype=float)
        signed_abs = np.abs(learned)
        if learned.size:
            signs[0] = -1.0
            signed_abs[0] *= -1.0
        denominators = signed_abs + a * signs
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            learned_sproj = projected / denominators if learned.size else projected
            step = -(vectors @ learned_sproj) if learned.size else np.zeros_like(g)
            ds = vectors @ (learned_sproj / denominators) if learned.size else np.zeros_like(g)
            if complement_dim:
                denom_perp = abs(beta) + a
                step = step - g_perp / denom_perp
                ds = ds + g_perp / (denom_perp * denom_perp)
        if not np.all(np.isfinite(step)) or not np.all(np.isfinite(ds)):
            raise PhysicalHessianError("nonfinite_compact_qn_mmf_step")
        return step, ds, {
            "representation": self.representation,
            "baseline_eigenvalue": beta,
            "learned_rank": int(learned.size),
            "complement_dimension": int(complement_dim),
            "learned_eigenvalues": [float(x) for x in learned],
            "selected_unstable_index": 0,
            "selected_unstable_eigenvalue": float(learned[0]) if learned.size else None,
            "order": 1,
            "eigensystem_source": "physical_hessian_capability",
        }

    def _set_compact_correction_from_terms(self, terms: Sequence[tuple[Array, Array]]) -> None:
        """Compress ``sum W C W.T`` exactly up to rank tolerance without ``n^2`` storage."""

        columns: list[Array] = []
        blocks: list[Array] = []
        if self._compact_basis.shape[1]:
            columns.append(self._compact_basis)
            blocks.append(self._compact_correction)
        for W, C in terms:
            w = np.asarray(W, dtype=float)
            c = np.asarray(C, dtype=float)
            if w.ndim != 2 or c.shape != (w.shape[1], w.shape[1]):
                raise ValueError("invalid compact correction term")
            columns.append(w)
            blocks.append(c)
        if not columns:
            self._compact_basis = np.empty((self.dimension, 0), dtype=float)
            self._compact_correction = np.empty((0, 0), dtype=float)
            return
        A = np.column_stack(columns)
        total = A.shape[1]
        Cbig = np.zeros((total, total), dtype=float)
        offset = 0
        for block in blocks:
            size = block.shape[0]
            Cbig[offset:offset+size, offset:offset+size] = block
            offset += size
        U, singular, Vt = np.linalg.svd(A, full_matrices=False)
        if singular.size == 0:
            self._compact_basis = np.empty((self.dimension, 0), dtype=float)
            self._compact_correction = np.empty((0, 0), dtype=float)
            return
        threshold = max(self.rank_absolute_tolerance, self.rank_relative_tolerance * float(singular[0]))
        rank = int(np.count_nonzero(singular > threshold))
        if rank == 0:
            self._compact_basis = np.empty((self.dimension, 0), dtype=float)
            self._compact_correction = np.empty((0, 0), dtype=float)
            return
        Vr = Vt[:rank, :].T
        sigma = singular[:rank]
        projected = (sigma[:, None] * (Vr.T @ Cbig @ Vr)) * sigma[None, :]
        self._compact_basis = np.asarray(U[:, :rank], dtype=float)
        self._compact_correction = 0.5 * (projected + projected.T)

    def _apply_compact_ts_secant(self, s: Array, y: Array) -> tuple[bool, str, float | None]:
        s = np.asarray(s, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
            return False, "nonfinite_secant", None
        if np.linalg.norm(s) <= self.rank_absolute_tolerance:
            return False, "translation_step_too_small", None
        Bs = self._compact_apply_reduced(s)
        absBs = self._compact_abs_apply_reduced(s)
        J = y - Bs
        sy = float(np.dot(s, y))
        abs_scalar = float(np.dot(s, absBs))
        xs = sy * y + abs_scalar * absBs
        denominator = float(np.dot(xs, s))
        # Match the dense one-pair lstsq semantics while staying entirely in O(n r).
        u = np.linalg.lstsq(np.asarray([[denominator]], dtype=float), xs.reshape(1, -1), rcond=None)[0].reshape(-1)
        js = float(np.dot(J, s))
        W = np.column_stack((u, J))
        C = np.asarray([[-js, 1.0], [1.0, 0.0]], dtype=float)
        old_basis = self._compact_basis.copy()
        old_corr = self._compact_correction.copy()
        self._set_compact_correction_from_terms(((W, C),))
        candidate_s = self._compact_apply_reduced(s)
        residual = _relative_norm(candidate_s - y, y, self.denominator_tolerance)
        if not np.isfinite(residual) or residual > self.secant_residual_tolerance:
            self._compact_basis = old_basis
            self._compact_correction = old_corr
            return False, "secant_residual_too_large", residual
        return True, "accepted", residual

    def rebuild_from_force_history(
        self, history, *, pair_sources: object, max_pairs: int = 0
    ) -> PhysicalHessianUpdateResult:
        """Rebuild a windowed TS-BFGS model from canonical raw physical pairs.

        Every call starts from ``B0=beta I``. Older pairs therefore contribute
        nothing after they age out of the configured accepted-pair window.
        """

        if self.representation == REPRESENTATION_ONLINE_DENSE:
            raise PhysicalHessianError("online_dense_model_cannot_be_forcebank_rebuilt")
        if self.update_type != TS_BFGS:
            raise PhysicalHessianError("forcebank_window_rebuild_requires_ts_bfgs")
        cap = int(max_pairs)
        if cap < 0:
            raise ValueError("max_pairs must be >= 0")
        start = perf_counter_ns()
        candidates = history.pair_candidates(pair_sources, physical_only=True)
        valid: list[tuple[object, Array, Array]] = []
        for candidate in candidates:
            s = self._basis.reduce(candidate.second.positions - candidate.first.positions)
            y = self._basis.reduce(candidate.first.forces - candidate.second.forces)
            if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
                continue
            if np.linalg.norm(s) <= self.rank_absolute_tolerance:
                continue
            valid.append((candidate, s, y))
        before_cap = len(valid)
        removed = 0
        if cap > 0 and len(valid) > cap:
            removed = len(valid) - cap
            valid = valid[-cap:]
        if self.representation == REPRESENTATION_FORCEBANK_WINDOW_DENSE:
            self._matrix = np.eye(self.dimension, dtype=float) * self._initial_hessian_scalar
        else:
            self._matrix = None
            self._compact_basis = np.empty((self.dimension, 0), dtype=float)
            self._compact_correction = np.empty((0, 0), dtype=float)
        accepted_ids: list[str] = []
        accepted_sources: list[str] = []
        rejected = 0
        last_residual = None
        for candidate, secant_s, secant_y in valid:
            if self.representation == REPRESENTATION_FORCEBANK_WINDOW_DENSE:
                assert self._matrix is not None
                candidate_matrix = self._ts_bfgs_candidate(
                    self._matrix, secant_s[:, None], secant_y[:, None]
                )
                candidate_matrix = 0.5 * (candidate_matrix + candidate_matrix.T)
                residual = _relative_norm(
                    candidate_matrix @ secant_s - secant_y, secant_y, self.denominator_tolerance
                )
                if (not np.all(np.isfinite(candidate_matrix))) or residual > self.secant_residual_tolerance:
                    rejected += 1
                    last_residual = residual
                    continue
                self._matrix = candidate_matrix
                accepted = True
            else:
                accepted, _reason, residual = self._apply_compact_ts_secant(secant_s, secant_y)
                if not accepted:
                    rejected += 1
                    last_residual = residual
                    continue
            pair_id = f"{candidate.first.observation_id}->{candidate.second.observation_id}:{candidate.source}"
            accepted_ids.append(pair_id)
            accepted_sources.append(str(candidate.source))
            last_residual = residual
        state = getattr(history, "current", None)
        center = None if state is None else getattr(state, "center", None)
        if center is not None:
            self._current_state_uid = _state_uid_from_observation(center)
            self._current_geometry_id = center.geometry_id
            self._previous_observation_id = center.observation_id
        signature = tuple(accepted_ids)
        fingerprint = hashlib.sha256()
        for candidate, secant_s, secant_y in valid:
            pair_id = f"{candidate.first.observation_id}->{candidate.second.observation_id}:{candidate.source}"
            if pair_id not in accepted_ids:
                continue
            fingerprint.update(pair_id.encode("utf-8"))
            fingerprint.update(np.asarray(secant_s, dtype=np.float64).tobytes(order="C"))
            fingerprint.update(np.asarray(secant_y, dtype=np.float64).tobytes(order="C"))
        fingerprint_hex = fingerprint.hexdigest()
        if fingerprint_hex != self._window_pair_fingerprint:
            self.model_age += 1
        self._window_pair_ids = signature
        self._window_pair_fingerprint = fingerprint_hex
        self.accepted_updates = len(accepted_ids)
        self.rejected_updates = rejected
        self._window_diagnostics = {
            "representation": self.representation,
            "raw_candidate_count": len(candidates),
            "valid_ts_bfgs_pair_count_before_cap": before_cap,
            "pairs_removed_by_cap": removed,
            "configured_cap": cap,
            "pair_ids_used": tuple(accepted_ids),
            "pair_sources_used": tuple(accepted_sources),
            "pairs_rejected_during_replay": rejected,
            "reconstructed_rank": int(np.linalg.matrix_rank(self._matrix - self._initial_hessian_scalar * np.eye(self.dimension))) if self._matrix is not None else int(self._compact_basis.shape[1]),
            "compact_rank": int(self._compact_basis.shape[1]) if self._matrix is None else 0,
            "reconstruction_time_ns": int(perf_counter_ns() - start),
            "spectral_time_ns": 0,
        }
        # Use the common immutable result type for runtime diagnostics.
        spectrum = self.low_spectrum()
        return PhysicalHessianUpdateResult(
            update_type=self.update_type,
            source="forcebank_window_rebuild",
            observation_ids=tuple(accepted_ids),
            accepted=True,
            reason="rebuilt",
            input_block_size=len(candidates),
            processed_block_rank=len(accepted_ids),
            dropped_dependent_rank=removed,
            dependence_noise_ratio=None,
            symmetry_residual_before=None,
            symmetry_correction_ratio=None,
            symmetry_residual_after=None,
            secant_residual=last_residual,
            negative_eigenvalue_count=spectrum.negative_eigenvalue_count,
            lambda_min=spectrum.lambda_min,
            lambda_max=spectrum.lambda_max,
            selected_low_root_eigengap=spectrum.selected_low_root_eigengap,
            absolute_eigenvalue_spread=spectrum.absolute_eigenvalue_spread,
            absolute_eigenvalue_condition=spectrum.absolute_eigenvalue_condition,
            condition_state=spectrum.condition_state,
            model_age=int(self.model_age),
            timing_ns=int(perf_counter_ns() - start),
        )

    def window_diagnostics(self) -> dict[str, object]:
        result = dict(self._window_diagnostics)
        start = perf_counter_ns()
        _ = self.full_eigenvalues()
        result["spectral_time_ns"] = int(perf_counter_ns() - start)
        return result

    def _spectrum_from_evals(
        self, evals: Array, selected_root_index: int = 0, timing_ns: int = 0
    ) -> PhysicalHessianSpectrum:
        evals = np.asarray(evals, dtype=float)
        absvals = np.abs(evals)
        absmin = float(np.min(absvals))
        absmax = float(np.max(absvals))
        if absmin <= self.spectral_zero_tolerance:
            condition = None
            condition_state = "singular_or_near_singular"
        else:
            condition = float(absmax / absmin)
            condition_state = "available"
        if evals.size < 2:
            gap = None
        else:
            root = float(evals[selected_root_index])
            others = np.delete(evals, selected_root_index)
            gap = float(np.min(np.abs(others - root)))
        return PhysicalHessianSpectrum(
            eigenvalues=tuple(float(v) for v in evals),
            negative_eigenvalue_count=int(np.count_nonzero(evals < -self.spectral_zero_tolerance)),
            lambda_min=float(evals[0]),
            lambda_max=float(evals[-1]),
            selected_root_index=int(selected_root_index),
            selected_low_root_eigengap=gap,
            absolute_eigenvalue_min=absmin,
            absolute_eigenvalue_max=absmax,
            absolute_eigenvalue_spread=float(absmax - absmin),
            absolute_eigenvalue_condition=condition,
            condition_state=condition_state,
            model_age=int(self.model_age),
            timing_ns=max(0, int(timing_ns)),
        )

    def _result(
        self,
        *,
        start_ns: int,
        source: str,
        observation_ids: Sequence[str],
        accepted: bool,
        reason: str,
        input_block_size: int = 0,
        processed_block_rank: int = 0,
        dropped_dependent_rank: int = 0,
        dependence_noise_ratio: float | None = None,
        symmetry_residual_before: float | None = None,
        symmetry_correction_ratio: float | None = None,
        symmetry_residual_after: float | None = None,
        secant_residual: float | None = None,
    ) -> PhysicalHessianUpdateResult:
        spectrum = self.low_spectrum()
        return PhysicalHessianUpdateResult(
            update_type=self.update_type,
            source=str(source).strip().lower(),
            observation_ids=tuple(str(v) for v in observation_ids),
            accepted=bool(accepted),
            reason=str(reason).strip().lower(),
            input_block_size=int(input_block_size),
            processed_block_rank=int(processed_block_rank),
            dropped_dependent_rank=int(dropped_dependent_rank),
            dependence_noise_ratio=None if dependence_noise_ratio is None else float(dependence_noise_ratio),
            symmetry_residual_before=None if symmetry_residual_before is None else float(symmetry_residual_before),
            symmetry_correction_ratio=None if symmetry_correction_ratio is None else float(symmetry_correction_ratio),
            symmetry_residual_after=None if symmetry_residual_after is None else float(symmetry_residual_after),
            secant_residual=None if secant_residual is None else float(secant_residual),
            negative_eigenvalue_count=spectrum.negative_eigenvalue_count,
            lambda_min=spectrum.lambda_min,
            lambda_max=spectrum.lambda_max,
            selected_low_root_eigengap=spectrum.selected_low_root_eigengap,
            absolute_eigenvalue_spread=spectrum.absolute_eigenvalue_spread,
            absolute_eigenvalue_condition=spectrum.absolute_eigenvalue_condition,
            condition_state=spectrum.condition_state,
            model_age=int(self.model_age),
            timing_ns=max(0, int(perf_counter_ns() - start_ns)),
        )

    def _observation_allowed(self, observation: ForceObservation) -> bool:
        return bool(
            self.admission_policy.accepts(
                purpose=observation.purpose,
                evaluation=observation.evaluation,
                source=observation.source,
                family=observation.family,
                promoted=False,
            )
        )

    def _validate_center_observation(self, observation: ForceObservation) -> None:
        if not isinstance(observation, ForceObservation):
            raise TypeError("observe_translation requires a typed-HVP ForceObservation")
        if observation.role != "center":
            raise ValueError("physical Hessian translation observations must have role='center'")
        if not observation.is_physical or not self._observation_allowed(observation):
            raise ValueError("center observation is not admitted raw physical algorithm data")
        if observation.positions.shape != self.coordinate_space.active_dof_mask.shape:
            raise ValueError("center observation Cartesian shape does not match coordinate_space")
        if observation.active_dof_mask is not None and not np.array_equal(
            observation.active_dof_mask, self.coordinate_space.active_dof_mask
        ):
            raise PhysicalHessianIdentityError("center observation active_dof_mask mismatch")
        if (
            observation.coordinate_convention not in {"", "unspecified"}
            and observation.coordinate_convention != self.coordinate_space.convention
        ):
            raise PhysicalHessianIdentityError("center observation coordinate convention mismatch")

    def observe_translation(self, observation: ForceObservation) -> PhysicalHessianUpdateResult:
        """Observe one accepted center and update from the prior center secant.

        On the first center this seeds the reference and returns a non-accepted
        diagnostic with reason ``seed_reference``.  A duplicate observation ID
        is rejected without changing reference state or the Hessian.
        """

        start = perf_counter_ns()
        if self.representation != REPRESENTATION_ONLINE_DENSE:
            raise PhysicalHessianError("windowed_physical_hessian_must_rebuild_from_force_history")
        self._validate_center_observation(observation)
        obs_id = observation.observation_id
        if obs_id in self._seen_observation_ids:
            self.rejected_updates += 1
            return self._result(
                start_ns=start,
                source="translation",
                observation_ids=(obs_id,),
                accepted=False,
                reason="duplicate_observation_id",
                input_block_size=1,
            )

        state_uid = _state_uid_from_observation(observation)
        position = self._basis.reduce(observation.positions)
        # Canonical sign: raw physical force F=-g.
        gradient = self._basis.reduce(-observation.forces)

        if self._previous_position is None:
            self._previous_position = position.copy()
            self._previous_gradient = gradient.copy()
            self._previous_observation_id = obs_id
            self._current_state_uid = state_uid
            self._current_geometry_id = observation.geometry_id
            self._seen_observation_ids.add(obs_id)
            return self._result(
                start_ns=start,
                source="translation",
                observation_ids=(obs_id,),
                accepted=False,
                reason="seed_reference",
                input_block_size=1,
            )

        s = position - self._previous_position
        y = gradient - self._previous_gradient
        previous_id = self._previous_observation_id
        # The newly observed center becomes the reference regardless of whether
        # the numerical QN update is admissible, matching historical behavior.
        self._previous_position = position.copy()
        self._previous_gradient = gradient.copy()
        self._previous_observation_id = obs_id
        self._current_state_uid = state_uid
        self._current_geometry_id = observation.geometry_id
        self._seen_observation_ids.add(obs_id)

        accepted, reason, residual = self._apply_single_secant(s, y)
        if accepted:
            self.accepted_updates += 1
            self.model_age += 1
        else:
            self.rejected_updates += 1
        return self._result(
            start_ns=start,
            source="translation",
            observation_ids=(previous_id, obs_id),
            accepted=accepted,
            reason=reason,
            input_block_size=1,
            processed_block_rank=1 if accepted else 0,
            secant_residual=residual,
        )

    def _apply_single_secant(self, s: Array, y: Array) -> tuple[bool, str, float | None]:
        s = np.asarray(s, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
            return False, "nonfinite_secant", None
        if np.linalg.norm(s) <= self.rank_absolute_tolerance:
            return False, "translation_step_too_small", None

        old = self._matrix
        if old is None:
            raise PhysicalHessianError("windowed_physical_hessian_must_rebuild_from_force_history")
        if self.update_type == ORDINARY_BFGS:
            sy = float(np.dot(s, y))
            Bs = old @ s
            sBs = float(np.dot(s, Bs))
            tol = self.denominator_tolerance
            # Literal historical rule: sign is irrelevant; only denominator
            # magnitude matters.  Negative sTy can therefore be admitted.
            if abs(sy) <= tol:
                return False, "small_s_dot_y", None
            if abs(sBs) <= tol:
                return False, "small_s_dot_b_s", None
            candidate = old + np.outer(y, y) / sy - np.outer(Bs, Bs) / sBs
        else:
            S = s[:, None]
            Y = y[:, None]
            candidate = self._ts_bfgs_candidate(old, S, Y)

        candidate = 0.5 * (candidate + candidate.T)
        if not np.all(np.isfinite(candidate)):
            return False, "nonfinite_candidate", None
        residual = _relative_norm(candidate @ s - y, y, self.denominator_tolerance)
        if residual > self.secant_residual_tolerance:
            return False, "secant_residual_too_large", residual
        self._matrix = candidate
        return True, "accepted", residual

    def _validate_hvp_result(self, result: HVPResult) -> tuple[Array, Array, str]:
        if not isinstance(result, HVPResult):
            raise TypeError("observe_probe_block requires typed HVPResult objects")
        if not result.available or result.action is None:
            raise ValueError("unavailable HVP results cannot update a physical Hessian")
        meta = result.metadata
        if not meta.physical or meta.model_derived:
            raise ValueError("probe block requires physical, non-model-derived HVP results")
        if meta.purpose is not WorkPurpose.ALGORITHM:
            raise ValueError("diagnostic HVP results require external explicit promotion before admission")
        if meta.operator_origin not in {
            OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
            OperatorOrigin.EXPLICIT_PHYSICAL_MATRIX,
        }:
            raise ValueError("HVP operator origin is not a physical Hessian action")
        if meta.coordinate_space_id != self.coordinate_space.identity:
            raise PhysicalHessianIdentityError("HVP coordinate_space_id mismatch")
        if result.direction.shape != self.coordinate_space.active_dof_mask.shape:
            raise ValueError("HVP direction shape does not match coordinate_space")
        direction = self._basis.reduce(result.direction)
        action = self._basis.reduce(result.action)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= self.rank_absolute_tolerance:
            raise ValueError("HVP direction is zero in active coordinates")
        # Preserve the linear action if the incoming direction is not unit norm.
        direction = direction / norm
        action = action / norm
        return direction, action, _hvp_identity(result, direction)

    def observe_probe_block(self, results: Sequence[HVPResult]) -> PhysicalHessianUpdateResult:
        """Update from a rank-revealed same-center physical HVP block.

        Preprocessing is deterministic:

        1. project each direction/action through the typed-HVP active coordinate space;
        2. normalize each direction and apply the same scale to its action;
        3. SVD-rank-reveal ``S`` with threshold
           ``max(rank_absolute_tolerance, rank_relative_tolerance*sigma_max)``;
        4. reject a dependent/noisy block when the action component in the
           discarded right-singular subspace exceeds
           ``dependence_noise_tolerance`` relative to ``||Y||``;
        5. transform the retained block to orthonormal independent directions;
        6. reject if the relative antisymmetry of ``S.T@Y`` exceeds
           ``block_symmetry_noise_tolerance``;
        7. apply the local transcription of Sella ``symmetrize_Y2`` and then the
           selected multisecant BFGS/TS-BFGS update;
        8. require near-exact secant satisfaction on that processed block.
        """

        start = perf_counter_ns()
        if self.representation != REPRESENTATION_ONLINE_DENSE:
            raise PhysicalHessianError("windowed_physical_hessian_must_rebuild_from_force_history")
        results = tuple(results)
        if not results:
            self.rejected_updates += 1
            return self._result(
                start_ns=start,
                source="probe_block",
                observation_ids=(),
                accepted=False,
                reason="empty_probe_block",
            )

        directions: list[Array] = []
        actions: list[Array] = []
        hvp_ids: list[str] = []
        state_uids: set[str] = set()
        geometry_ids: set[str] = set()
        source_tokens: list[str] = []
        for result in results:
            d, a, hid = self._validate_hvp_result(result)
            directions.append(d)
            actions.append(a)
            hvp_ids.append(hid)
            state_uids.add(result.metadata.state_uid)
            geometry_ids.add(result.metadata.geometry_id)
            source_tokens.append(result.metadata.source)

        if len(set(hvp_ids)) != len(hvp_ids) or any(hid in self._seen_hvp_ids for hid in hvp_ids):
            self.rejected_updates += 1
            return self._result(
                start_ns=start,
                source="probe_block",
                observation_ids=hvp_ids,
                accepted=False,
                reason="duplicate_hvp_id",
                input_block_size=len(results),
            )
        if len(state_uids) != 1 or len(geometry_ids) != 1:
            self.rejected_updates += 1
            return self._result(
                start_ns=start,
                source="probe_block",
                observation_ids=hvp_ids,
                accepted=False,
                reason="mixed_center_probe_block",
                input_block_size=len(results),
            )
        state_uid = next(iter(state_uids))
        geometry_id = next(iter(geometry_ids))
        if self._current_state_uid and state_uid != self._current_state_uid:
            self.rejected_updates += 1
            return self._result(
                start_ns=start,
                source="probe_block",
                observation_ids=hvp_ids,
                accepted=False,
                reason="changed_center_hvp_block",
                input_block_size=len(results),
            )
        if self._current_geometry_id and geometry_id != self._current_geometry_id:
            self.rejected_updates += 1
            return self._result(
                start_ns=start,
                source="probe_block",
                observation_ids=hvp_ids,
                accepted=False,
                reason="changed_geometry_hvp_block",
                input_block_size=len(results),
            )

        S_raw = np.column_stack(directions)
        Y_raw = np.column_stack(actions)
        try:
            block = self._preprocess_block(S_raw, Y_raw)
        except PhysicalHessianError as exc:
            self.rejected_updates += 1
            return self._result(
                start_ns=start,
                source="probe_block",
                observation_ids=hvp_ids,
                accepted=False,
                reason=str(exc),
                input_block_size=len(results),
            )

        old = self._matrix
        try:
            if self.update_type == ORDINARY_BFGS:
                candidate = self._ordinary_multisecant_candidate(old, block.S, block.Y)
            else:
                candidate = self._ts_bfgs_candidate(old, block.S, block.Y)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            self.rejected_updates += 1
            return self._result(
                start_ns=start,
                source="probe_block",
                observation_ids=hvp_ids,
                accepted=False,
                reason="singular_multisecant_system",
                input_block_size=block.input_size,
                processed_block_rank=block.rank,
                dropped_dependent_rank=block.dropped_rank,
                dependence_noise_ratio=block.dependence_noise_ratio,
                symmetry_residual_before=block.symmetry_residual_before,
                symmetry_correction_ratio=block.symmetry_correction_ratio,
                symmetry_residual_after=block.symmetry_residual_after,
            )

        candidate = 0.5 * (candidate + candidate.T)
        if not np.all(np.isfinite(candidate)):
            self.rejected_updates += 1
            return self._result(
                start_ns=start,
                source="probe_block",
                observation_ids=hvp_ids,
                accepted=False,
                reason="nonfinite_candidate",
                input_block_size=block.input_size,
                processed_block_rank=block.rank,
            )
        residual = _relative_norm(
            candidate @ block.S - block.Y,
            block.Y,
            self.denominator_tolerance,
        )
        if residual > self.secant_residual_tolerance:
            self.rejected_updates += 1
            return self._result(
                start_ns=start,
                source="probe_block",
                observation_ids=hvp_ids,
                accepted=False,
                reason="secant_residual_too_large",
                input_block_size=block.input_size,
                processed_block_rank=block.rank,
                dropped_dependent_rank=block.dropped_rank,
                dependence_noise_ratio=block.dependence_noise_ratio,
                symmetry_residual_before=block.symmetry_residual_before,
                symmetry_correction_ratio=block.symmetry_correction_ratio,
                symmetry_residual_after=block.symmetry_residual_after,
                secant_residual=residual,
            )

        self._matrix = candidate
        self._seen_hvp_ids.update(hvp_ids)
        if not self._current_state_uid:
            self._current_state_uid = state_uid
            self._current_geometry_id = geometry_id
        self.accepted_updates += 1
        self.model_age += 1
        source = "probe_block:" + "+".join(sorted(set(source_tokens)))
        return self._result(
            start_ns=start,
            source=source,
            observation_ids=hvp_ids,
            accepted=True,
            reason="accepted",
            input_block_size=block.input_size,
            processed_block_rank=block.rank,
            dropped_dependent_rank=block.dropped_rank,
            dependence_noise_ratio=block.dependence_noise_ratio,
            symmetry_residual_before=block.symmetry_residual_before,
            symmetry_correction_ratio=block.symmetry_correction_ratio,
            symmetry_residual_after=block.symmetry_residual_after,
            secant_residual=residual,
        )

    def _preprocess_block(self, S: Array, Y: Array) -> _ProcessedBlock:
        S = np.asarray(S, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if S.ndim != 2 or Y.shape != S.shape or S.shape[0] != self.dimension:
            raise PhysicalHessianError("probe_block_shape_mismatch")
        if not np.all(np.isfinite(S)) or not np.all(np.isfinite(Y)):
            raise PhysicalHessianError("nonfinite_probe_block")
        _, singular, Vt = np.linalg.svd(S, full_matrices=True)
        if singular.size == 0:
            raise PhysicalHessianError("empty_probe_block")
        threshold = max(
            self.rank_absolute_tolerance,
            self.rank_relative_tolerance * float(singular[0]),
        )
        rank = int(np.count_nonzero(singular > threshold))
        if rank < 1:
            raise PhysicalHessianError("probe_block_rank_zero")
        V = Vt.T
        dropped = int(S.shape[1] - rank)
        if dropped:
            dropped_action = Y @ V[:, rank:]
            dependence_noise = _relative_norm(
                dropped_action, Y, self.rank_absolute_tolerance
            )
            if dependence_noise > self.dependence_noise_tolerance:
                raise PhysicalHessianError("dependent_block_inconsistent")
        else:
            dependence_noise = 0.0

        # S @ V_keep @ diag(1/sigma) gives an orthonormal independent basis.
        transform = V[:, :rank] / singular[:rank][None, :]
        S_independent = S @ transform
        Y_independent = Y @ transform

        sty = S_independent.T @ Y_independent
        skew = sty - sty.T
        symmetry_before = _relative_norm(skew, sty, self.rank_absolute_tolerance)
        if symmetry_before > self.block_symmetry_noise_tolerance:
            raise PhysicalHessianError("probe_block_symmetry_noise")
        Y_compatible = _symmetrize_y2(S_independent, Y_independent)
        correction = _relative_norm(
            Y_compatible - Y_independent,
            Y_independent,
            self.rank_absolute_tolerance,
        )
        sty_after = S_independent.T @ Y_compatible
        symmetry_after = _relative_norm(
            sty_after - sty_after.T,
            sty_after,
            self.rank_absolute_tolerance,
        )
        return _ProcessedBlock(
            S=S_independent,
            Y=Y_compatible,
            input_size=int(S.shape[1]),
            rank=rank,
            dropped_rank=dropped,
            dependence_noise_ratio=dependence_noise,
            symmetry_residual_before=symmetry_before,
            symmetry_correction_ratio=correction,
            symmetry_residual_after=symmetry_after,
        )

    def _ordinary_multisecant_candidate(self, B: Array, S: Array, Y: Array) -> Array:
        yts = Y.T @ S
        stbs = S.T @ B @ S
        # Generalized Sella multisecant BFGS algebra.  It remains ordinary BFGS;
        # no positivity test is imposed.  Fail only when the solve is singular.
        term1 = Y @ np.linalg.solve(yts, Y.T)
        term2 = B @ S @ np.linalg.solve(stbs, S.T @ B)
        return B + term1 - term2

    @staticmethod
    def _ts_bfgs_candidate(B: Array, S: Array, Y: Array) -> Array:
        # Local transcription of installed_sella/hessian_update.py lines 118-125.
        lams, vecs = np.linalg.eigh(B)
        J = Y - B @ S
        X1 = S.T @ Y @ Y.T
        absBS = vecs @ (np.abs(lams[:, None]) * (vecs.T @ S))
        X2 = S.T @ absBS @ absBS.T
        XS = X1 + X2
        U = np.linalg.lstsq(XS @ S, XS, rcond=None)[0].T
        UJT = U @ J.T
        delta = (UJT + UJT.T) - U @ (J.T @ S) @ U.T
        return B + delta

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": PHYSICAL_HESSIAN_STATE_SCHEMA,
            "update_type": self.update_type,
            "representation": self.representation,
            "initial_hessian_scalar": self._initial_hessian_scalar,
            "coordinate_space": self.coordinate_space.to_state_dict(),
            "matrix": None if self._matrix is None else self._matrix.tolist(),
            "compact_basis": self._compact_basis.tolist(),
            "compact_correction": self._compact_correction.tolist(),
            "window_pair_ids": list(self._window_pair_ids),
            "window_pair_fingerprint": self._window_pair_fingerprint,
            "window_diagnostics": dict(self._window_diagnostics),
            "previous_position": None if self._previous_position is None else self._previous_position.tolist(),
            "previous_gradient": None if self._previous_gradient is None else self._previous_gradient.tolist(),
            "previous_observation_id": self._previous_observation_id,
            "current_state_uid": self._current_state_uid,
            "current_geometry_id": self._current_geometry_id,
            "seen_observation_ids": sorted(self._seen_observation_ids),
            "seen_hvp_ids": sorted(self._seen_hvp_ids),
            "accepted_updates": int(self.accepted_updates),
            "rejected_updates": int(self.rejected_updates),
            "model_age": int(self.model_age),
            "settings": {
                "denominator_tolerance": self.denominator_tolerance,
                "rank_relative_tolerance": self.rank_relative_tolerance,
                "rank_absolute_tolerance": self.rank_absolute_tolerance,
                "dependence_noise_tolerance": self.dependence_noise_tolerance,
                "block_symmetry_noise_tolerance": self.block_symmetry_noise_tolerance,
                "secant_residual_tolerance": self.secant_residual_tolerance,
                "spectral_zero_tolerance": self.spectral_zero_tolerance,
            },
            "admission_policy": {
                "name": self.admission_policy.name,
                "allowed_purposes": list(self.admission_policy.allowed_purposes),
                "allowed_evaluations": list(self.admission_policy.allowed_evaluations),
                "allowed_sources": list(self.admission_policy.allowed_sources),
                "allowed_families": list(self.admission_policy.allowed_families),
                "allow_cached": self.admission_policy.allow_cached,
                "allow_derived": self.admission_policy.allow_derived,
                "allow_diagnostic_promotion": self.admission_policy.allow_diagnostic_promotion,
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "PhysicalHessianModel":
        if state.get("schema") != PHYSICAL_HESSIAN_STATE_SCHEMA:
            raise ValueError("unsupported PhysicalHessianModel state schema")
        coordinate_space = ActiveCoordinateSpace.from_state_dict(dict(state["coordinate_space"]))
        settings = dict(state.get("settings", {}))
        policy_raw = dict(state.get("admission_policy", {}))
        policy = ObservationAdmissionPolicy(
            name=str(policy_raw.get("name", "algorithm_physical")),
            allowed_purposes=tuple(policy_raw.get("allowed_purposes", ("algorithm",))),
            allowed_evaluations=tuple(policy_raw.get("allowed_evaluations", ("physical_exact", "physical_cached"))),
            allowed_sources=tuple(policy_raw.get("allowed_sources", ())),
            allowed_families=tuple(policy_raw.get("allowed_families", ())),
            allow_cached=bool(policy_raw.get("allow_cached", True)),
            allow_derived=bool(policy_raw.get("allow_derived", False)),
            allow_diagnostic_promotion=bool(policy_raw.get("allow_diagnostic_promotion", False)),
        )
        representation = str(state.get("representation", REPRESENTATION_ONLINE_DENSE))
        matrix_state = state.get("matrix")
        result = cls(
            coordinate_space,
            update_type=str(state["update_type"]),
            initial_hessian=float(state.get("initial_hessian_scalar", 1.0)),
            initial_matrix=(None if representation != REPRESENTATION_ONLINE_DENSE else np.asarray(matrix_state, dtype=float)),
            representation=representation,
            denominator_tolerance=float(settings.get("denominator_tolerance", DEFAULT_DENOMINATOR_TOLERANCE)),
            rank_relative_tolerance=float(settings.get("rank_relative_tolerance", DEFAULT_RANK_RELATIVE_TOLERANCE)),
            rank_absolute_tolerance=float(settings.get("rank_absolute_tolerance", DEFAULT_RANK_ABSOLUTE_TOLERANCE)),
            dependence_noise_tolerance=float(settings.get("dependence_noise_tolerance", DEFAULT_DEPENDENCE_NOISE_TOLERANCE)),
            block_symmetry_noise_tolerance=float(settings.get("block_symmetry_noise_tolerance", DEFAULT_BLOCK_SYMMETRY_NOISE_TOLERANCE)),
            secant_residual_tolerance=float(settings.get("secant_residual_tolerance", DEFAULT_SECANT_RESIDUAL_TOLERANCE)),
            spectral_zero_tolerance=float(settings.get("spectral_zero_tolerance", DEFAULT_SPECTRAL_ZERO_TOLERANCE)),
            admission_policy=policy,
        )
        previous_position = state.get("previous_position")
        previous_gradient = state.get("previous_gradient")
        if (previous_position is None) != (previous_gradient is None):
            raise ValueError("serialized physical Hessian has incomplete translation reference")
        if previous_position is not None:
            p = np.asarray(previous_position, dtype=float).reshape(-1)
            g = np.asarray(previous_gradient, dtype=float).reshape(-1)
            if p.size != result.dimension or g.size != result.dimension:
                raise ValueError("serialized translation reference dimension mismatch")
            result._previous_position = p.copy()
            result._previous_gradient = g.copy()
        result._previous_observation_id = str(state.get("previous_observation_id", ""))
        result._current_state_uid = str(state.get("current_state_uid", ""))
        result._current_geometry_id = str(state.get("current_geometry_id", ""))
        result._seen_observation_ids = {str(v) for v in state.get("seen_observation_ids", [])}
        result._seen_hvp_ids = {str(v) for v in state.get("seen_hvp_ids", [])}
        result.accepted_updates = int(state.get("accepted_updates", 0))
        result.rejected_updates = int(state.get("rejected_updates", 0))
        result.model_age = int(state.get("model_age", result.accepted_updates))
        if min(result.accepted_updates, result.rejected_updates, result.model_age) < 0:
            raise ValueError("serialized counters must be nonnegative")
        if representation == REPRESENTATION_FORCEBANK_WINDOW_DENSE:
            if matrix_state is None:
                raise ValueError("serialized dense-window model is missing matrix")
            result._matrix = _finite_matrix(np.asarray(matrix_state, dtype=float), result.dimension, "matrix")
        elif representation == REPRESENTATION_FORCEBANK_WINDOW_COMPACT:
            basis = np.asarray(state.get("compact_basis", []), dtype=float)
            correction = np.asarray(state.get("compact_correction", []), dtype=float)
            if basis.size == 0:
                basis = np.empty((result.dimension, 0), dtype=float)
                correction = np.empty((0, 0), dtype=float)
            if basis.ndim != 2 or basis.shape[0] != result.dimension:
                raise ValueError("serialized compact basis shape mismatch")
            if correction.shape != (basis.shape[1], basis.shape[1]):
                raise ValueError("serialized compact correction shape mismatch")
            result._compact_basis = basis.copy()
            result._compact_correction = 0.5 * (correction + correction.T)
            result._matrix = None
        result._window_pair_ids = tuple(str(v) for v in state.get("window_pair_ids", ()))
        result._window_pair_fingerprint = str(state.get("window_pair_fingerprint", ""))
        result._window_diagnostics = dict(state.get("window_diagnostics", result._window_diagnostics))
        return result

    def state_json(self) -> str:
        """Canonical JSON convenience for checkpoint hashing/tests."""

        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))
