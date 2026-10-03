"""Legacy Dimer rotation and configurable minimum-mode adapter.

The classes in this module retain the historical rotation mathematics, bowl
breakout behavior, minimum-mode solver wiring, operation order, and diagnostic
semantics.  The old :mod:`lbfgs_dimer` module remains their compatibility
import path.
"""

from __future__ import annotations

from collections import deque
from math import atan, cos, pi, sin, tan
from time import perf_counter_ns
from typing import Callable, Optional

import numpy as np

from ase.mep.dimer import (
    DimerEigenmodeSearch,
    MinModeAtoms,
    perpendicular_vector,
    rotate_vectors,
)

from saddlemill.dimertools.ase_lbfgs_adapter import _secant_pair_metrics
from saddlemill.dimertools.dimer_entry_torque import (
    DimerEntryTorqueCaptureMixin,
    entry_torque_capture_payload,
)
from saddlemill.dimertools.dense_bfgs import (
    DenseSecant,
    compare_directions,
    reconstruct_sequential_bfgs,
    safe_max_atom_norm,
    safe_norm,
)
from saddlemill.dimertools.minmode_solvers import (
    PhysicalHessianBFGS,
    TypedHVPCallback,
    davidson_lowest_mode,
    lanczos_lowest_mode,
    softsaddle_davidson_hybrid_lowest_mode,
    softsaddle_lanczos_lowest_mode,
    olsen_jd_lowest_mode,
)

norm = np.linalg.norm

class BowlBreakoutMixin:
    """Pedersen/Luisier positive-region displacement confinement.

    Bowl breakout is a translation policy, not a minimum-mode finder. While
    the current lowest curvature is positive, the active set is recomputed
    from the physical center forces and only ``bowl_active_atoms`` atoms with
    the largest force norms are allowed to translate. Once the curvature is
    non-positive, every movable atom is released.

    The force mask and displacement mask are both required. Masking only the
    force is insufficient for CG/L-BFGS/FIRE because optimizer history can
    otherwise propose motion on an atom whose current force is zero.
    """

    def configure_bowl_breakout(self, convex_escape="standard", bowl_active_atoms=20):
        self.convex_escape = str(convex_escape).strip().lower()
        self.bowl_active_atoms = int(bowl_active_atoms)
        if self.convex_escape not in {"standard", "bowl_breakout"}:
            raise ValueError(
                "convex_escape must be 'standard' or 'bowl_breakout'; got "
                f"{self.convex_escape!r}"
            )
        if self.bowl_active_atoms < 1:
            raise ValueError("bowl_active_atoms must be >= 1")
        self._bowl_active_mask = None
        self._bowl_active_indices = ()

    def _bowl_movable_atom_mask(self):
        mask = np.ones(len(self.atoms), dtype=bool)
        for constraint in list(getattr(self.atoms, "constraints", []) or []):
            getter = getattr(constraint, "get_indices", None)
            if getter is None:
                continue
            try:
                indices = np.asarray(getter(), dtype=int).reshape(-1)
            except Exception:
                continue
            indices = indices[(indices >= 0) & (indices < len(self.atoms))]
            mask[indices] = False
        return mask

    def _bowl_breakout_is_active(self):
        if self.convex_escape != "bowl_breakout":
            return False
        try:
            return float(self.get_curvature()) > 0.0
        except Exception:
            return False

    def _update_bowl_active_mask(self):
        movable = self._bowl_movable_atom_mask()
        if not self._bowl_breakout_is_active():
            self._bowl_active_mask = movable
            self._bowl_active_indices = tuple(np.flatnonzero(movable).tolist())
            return self._bowl_active_mask

        forces = np.asarray(self.forces0, dtype=float)
        if forces.shape != (len(self.atoms), 3):
            raise ValueError(
                "Bowl breakout expected physical center forces with shape "
                f"{(len(self.atoms), 3)}; got {forces.shape}"
            )
        movable_indices = np.flatnonzero(movable)
        if movable_indices.size == 0:
            raise ValueError("Bowl breakout has no movable atoms")

        nactive = min(self.bowl_active_atoms, int(movable_indices.size))
        norms = np.linalg.norm(forces, axis=1)
        movable_norms = norms[movable_indices]
        # The paper defines the active set by the force norm of the
        # N_confine-th atom. Using that threshold preserves the published >=
        # rule; exact ties can therefore activate more than N_confine atoms.
        threshold = np.partition(movable_norms, -nactive)[-nactive]
        active = movable & (norms >= threshold)
        self._bowl_active_mask = active
        self._bowl_active_indices = tuple(np.flatnonzero(active).tolist())
        return active

    def bowl_active_atom_mask(self):
        """Return the current center active-atom mask."""
        if self._bowl_breakout_is_active():
            return self._update_bowl_active_mask().copy()
        return self._bowl_movable_atom_mask()

    def apply_bowl_breakout_force_mask(self, forces, *, center=True):
        forces = np.asarray(forces, dtype=float).copy()
        if not self._bowl_breakout_is_active():
            if center and self.convex_escape == "bowl_breakout":
                self.translation_regime = "standard"
                self._bowl_active_mask = self._bowl_movable_atom_mask()
                self._bowl_active_indices = tuple(
                    np.flatnonzero(self._bowl_active_mask).tolist()
                )
            return forces

        if center or self._bowl_active_mask is None:
            active = self._update_bowl_active_mask()
        else:
            active = self._bowl_active_mask
        forces[~active, :] = 0.0
        if center:
            self.translation_regime = "bowl_breakout"
        return forces

    def confine_translation_vector(self, vector):
        """Zero translation components outside the current bowl active set."""
        arr = np.asarray(vector, dtype=float)
        original_shape = arr.shape
        if not self._bowl_breakout_is_active():
            return arr.copy()
        if self._bowl_active_mask is None:
            active = self._update_bowl_active_mask()
        else:
            active = self._bowl_active_mask
        reshaped = arr.reshape((len(self.atoms), 3)).copy()
        reshaped[~active, :] = 0.0
        return reshaped.reshape(original_shape)

    def confine_translation_positions(self, before, proposed_after):
        """Apply the bowl mask to an optimizer-proposed accepted position."""
        before = np.asarray(before, dtype=float)
        after = np.asarray(proposed_after, dtype=float)
        displacement = after - before
        confined = self.confine_translation_vector(displacement)
        return before + confined.reshape(before.shape)


class LimitedMemoryInverseHessian:
    """Small, dependency-free L-BFGS inverse-Hessian approximation.

    ``force`` is interpreted as minus the gradient.  Therefore ``apply``
    returns ``H^{-1-like} * force``, i.e. an optimization step direction.
    Only secant pairs satisfying positive curvature are retained, keeping the
    approximation positive definite.
    """

    def __init__(
        self,
        memory: int = 10,
        initial_hessian: float = 1.0,
        dynamic_h0: bool = False,
        curvature_epsilon: float = 1.0e-12,
        dense_bfgs_diagnostic: bool = False,
    ):
        if int(memory) < 1:
            raise ValueError("L-BFGS memory must be >= 1")
        if float(initial_hessian) <= 0.0:
            raise ValueError("initial_hessian must be > 0")
        if float(curvature_epsilon) < 0.0:
            raise ValueError("curvature_epsilon must be >= 0")

        self.memory = int(memory)
        self.initial_hessian = float(initial_hessian)
        self.dynamic_h0 = bool(dynamic_h0)
        self.curvature_epsilon = float(curvature_epsilon)
        self.dense_bfgs_diagnostic = bool(dense_bfgs_diagnostic)
        self.last_dense_diagnostics: dict[str, object] = {}
        self.s_history: deque[np.ndarray] = deque(maxlen=self.memory)
        self.y_history: deque[np.ndarray] = deque(maxlen=self.memory)
        self.accepted_pairs_total = 0
        self.rejected_pairs_total = 0
        self.reset_count = 0
        self.last_reset_reason = "initial"
        self.last_pair_metrics = {}
        self.last_pair_accepted = ""

    @property
    def size(self) -> int:
        return len(self.s_history)

    def reset(self, reason: str = "manual") -> None:
        self.s_history.clear()
        self.y_history.clear()
        self.reset_count += 1
        self.last_reset_reason = str(reason)
        # Keep the latest attempted pair as diagnostics even though it is no
        # longer active history. This is especially useful when a non-descent
        # direction triggers an immediate history reset.

    def add_pair(self, s, y) -> bool:
        s = np.asarray(s, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if s.shape != y.shape or not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
            self.rejected_pairs_total += 1
            return False

        self.last_pair_metrics = _secant_pair_metrics(s, y)
        sy = float(np.dot(s, y))
        scale = float(norm(s) * norm(y))
        threshold = self.curvature_epsilon * max(1.0, scale)
        if sy <= threshold:
            self.rejected_pairs_total += 1
            self.last_pair_accepted = 0
            return False

        self.s_history.append(s.copy())
        self.y_history.append(y.copy())
        self.accepted_pairs_total += 1
        self.last_pair_accepted = 1
        return True

    def _h0_scale(self, pairs) -> float:
        # H0 is the inverse of the configured initial Hessian.
        scale = 1.0 / self.initial_hessian
        if self.dynamic_h0 and pairs:
            s, y, sy = pairs[-1]
            yy = float(np.dot(y, y))
            if yy > 0.0 and sy > 0.0:
                scale = sy / yy
        return scale

    def apply(
        self,
        force,
        projector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        shape = np.asarray(force).shape
        q = np.asarray(force, dtype=float).reshape(-1)
        if projector is not None:
            q = np.asarray(projector(q.reshape(shape)), dtype=float).reshape(-1)
        force_flat = q.copy()

        pairs = []
        for s_raw, y_raw in zip(self.s_history, self.y_history):
            s = s_raw.reshape(shape)
            y = y_raw.reshape(shape)
            if projector is not None:
                s = projector(s)
                y = projector(y)
            s = np.asarray(s, dtype=float).reshape(-1)
            y = np.asarray(y, dtype=float).reshape(-1)
            sy = float(np.dot(s, y))
            if sy > self.curvature_epsilon * max(1.0, norm(s) * norm(y)):
                pairs.append((s, y, sy))

        alphas = []
        for s, y, sy in reversed(pairs):
            rho = 1.0 / sy
            alpha = rho * float(np.dot(s, q))
            alphas.append(alpha)
            q = q - alpha * y

        r = self._h0_scale(pairs) * q
        for (s, y, sy), alpha in zip(pairs, reversed(alphas)):
            rho = 1.0 / sy
            beta = rho * float(np.dot(y, r))
            r = r + s * (alpha - beta)

        result = r.reshape(shape)
        if projector is not None:
            result = projector(result)
        result = np.asarray(result, dtype=float)
        self.last_dense_diagnostics = {
            "lbfgs_raw_direction_norm": safe_norm(result),
            "lbfgs_raw_direction_max_atom_norm": (
                safe_max_atom_norm(result) if result.size % 3 == 0 else ""
            ),
        }
        if self.dense_bfgs_diagnostic:
            dense_pairs = [
                DenseSecant(s=s, y=y, source="legacy_rotation", state_id=0, serial=i)
                for i, (s, y, _sy) in enumerate(pairs)
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
            self.last_dense_diagnostics.update({
                "dense_vs_lbfgs_cosine": comparison.get("dense_vs_primary_cosine", ""),
                "dense_vs_lbfgs_norm_ratio": comparison.get("dense_vs_primary_norm_ratio", ""),
                "dense_vs_lbfgs_relative_difference": comparison.get(
                    "dense_vs_primary_relative_difference", ""
                ),
            })
        return result


class LBFGSRotationMixin:
    """Replace ASE's rotational steepest direction with L-BFGS.

    The physical rotational-force norms still control ``f_rot_min``,
    ``f_rot_max`` and ``max_num_rot`` exactly as in ASE.  The trial angle,
    Fourier interpolation, and optional endpoint-force extrapolation are also
    retained unchanged.
    """

    def __init__(self, *args, lbfgs_options=None, lbfgs_history=None, **kwargs):
        self.lbfgs_options = dict(lbfgs_options or {})
        self.lbfgs_history = lbfgs_history
        super().__init__(*args, **kwargs)

    def _project_rotation_vector(self, vector, mode):
        projected = perpendicular_vector(np.asarray(vector, dtype=float), mode)
        if self.basis is not None:
            basis_items = self.basis
            if (
                isinstance(basis_items, np.ndarray)
                and basis_items.shape == projected.shape
            ):
                basis_items = [basis_items]
            for base in basis_items:
                projected = perpendicular_vector(projected, base)
        return projected

    def converge_to_eigenmode(self):
        self.set_up_for_eigenmode_search()
        stoprot = False

        f_rot_min = self.control.get_parameter("f_rot_min")
        f_rot_max = self.control.get_parameter("f_rot_max")
        trial_angle = self.control.get_parameter("trial_angle")
        max_num_rot = self.control.get_parameter("max_num_rot")
        extrapolate = self.control.get_parameter("extrapolate_forces")

        history = (self.lbfgs_history if self.lbfgs_history is not None else LimitedMemoryInverseHessian(**self.lbfgs_options))
        previous_mode = None
        previous_force = None
        direction_fallbacks = 0

        while not stoprot:
            used_extrapolated_A = self.forces1E is not None
            if self.forces1E is None:
                self.update_virtual_forces()
            else:
                self.update_virtual_forces(extrapolated_forces=True)
            self.forces1A = self.forces1
            self.update_curvature()
            f_rot_A = self.get_rotational_force()

            # Preserve ASE's physical-force stopping criteria.
            if norm(f_rot_A) <= f_rot_min:
                self.log(f_rot_A, None)
                stoprot = True
            else:
                n_A = np.asarray(self.eigenmode, dtype=float).copy()

                # Work in a sign-continuous representation because n and -n
                # denote the same dimer axis.
                sign = 1.0
                history_mode = n_A.copy()
                history_force = np.asarray(f_rot_A, dtype=float).copy()
                if previous_mode is not None and np.vdot(
                    previous_mode.ravel(), history_mode.ravel()
                ).real < 0.0:
                    sign = -1.0
                    history_mode *= -1.0
                    history_force *= -1.0

                projector = lambda v: self._project_rotation_vector(v, history_mode)
                if previous_mode is not None:
                    s = projector(history_mode - previous_mode)
                    # y = grad_new - grad_old = force_old - force_new.
                    y = projector(previous_force - history_force)
                    history.add_pair(s, y)

                lbfgs_direction = history.apply(history_force, projector=projector)
                rot_direction = sign * lbfgs_direction
                rot_direction = self._project_rotation_vector(rot_direction, n_A)

                # A positive-definite inverse-Hessian approximation should give
                # a direction with positive projection on the force.  Fall back
                # safely if finite precision or transported history violates it.
                rot_direction_norm = safe_norm(rot_direction)
                if (
                    not np.all(np.isfinite(rot_direction))
                    or not np.isfinite(rot_direction_norm)
                    or rot_direction_norm < 1.0e-14
                    or np.vdot(rot_direction, f_rot_A).real <= 0.0
                ):
                    history.reset("non_descent_rotation_direction")
                    direction_fallbacks += 1
                    rot_direction = f_rot_A.copy()
                    rot_direction_norm = safe_norm(rot_direction)
                if not np.isfinite(rot_direction_norm) or rot_direction_norm <= 0.0:
                    raise RuntimeError("rotation fallback produced invalid direction norm")
                rot_unit_A = rot_direction / rot_direction_norm
                c0 = self.get_curvature()
                c0d = np.vdot((self.forces2 - self.forces1), rot_unit_A) / self.dR

                # ASE/Heyden trial-angle evaluation and Fourier interpolation.
                n_B, rot_unit_B = rotate_vectors(n_A, rot_unit_A, trial_angle)
                self.eigenmode = n_B
                self.update_virtual_forces()
                self.forces1B = self.forces1
                c1d = np.vdot((self.forces2 - self.forces1), rot_unit_B) / self.dR

                # Optional no-extra-force-call secant for future translated centers.
                # f_rot_B is computed from the endpoint forces ASE already evaluated
                # for its trial-angle/Fourier interpolation.  If A itself came from
                # endpoint-force extrapolation, do not promote this pair into the
                # persistent physical-history consumer.
                add_trial_pair = getattr(history, "add_trial_pair", None)
                if callable(add_trial_pair) and not used_extrapolated_A:
                    f_rot_B = np.asarray(self.get_rotational_force(), dtype=float).copy()
                    add_trial_pair(n_B - n_A, np.asarray(f_rot_A) - f_rot_B)

                a1 = c0d * cos(2 * trial_angle) - c1d / (2 * sin(2 * trial_angle))
                b1 = 0.5 * c0d
                a0 = 2 * (c0 - a1)
                rotangle = atan(b1 / a1) / 2.0
                cmin = a0 / 2.0 + a1 * cos(2 * rotangle) + b1 * sin(2 * rotangle)
                if c0 < cmin:
                    rotangle += pi / 2.0

                n_min, _ = rotate_vectors(n_A, rot_unit_A, rotangle)
                self.update_eigenmode(n_min)
                self.update_curvature(cmin)
                self.log(f_rot_A, rotangle)

                if extrapolate:
                    self.forces1E = (
                        sin(trial_angle - rotangle) / sin(trial_angle) * self.forces1A
                        + sin(rotangle) / sin(trial_angle) * self.forces1B
                        + (
                            1
                            - cos(rotangle)
                            - sin(rotangle) * tan(trial_angle / 2.0)
                        )
                        * self.forces0
                    )
                else:
                    self.forces1E = None

                previous_mode = history_mode.copy()
                previous_force = history_force.copy()

            if not stoprot:
                if self.control.get_counter("rotcount") >= max_num_rot:
                    stoprot = True
                elif norm(f_rot_A) <= f_rot_max:
                    stoprot = True

        last_pair = dict(history.last_pair_metrics)
        self.lbfgs_diagnostics = {
            "optimizer": "lbfgs",
            "rotations": int(self.control.get_counter("rotcount")),
            "history_size": int(history.size),
            "pairs_accepted": int(history.accepted_pairs_total),
            "pairs_rejected": int(history.rejected_pairs_total),
            "history_resets": int(history.reset_count),
            "direction_fallbacks": int(direction_fallbacks),
            "memory": int(history.memory),
            "initial_hessian": float(history.initial_hessian),
            "initial_inverse_hessian_scale": 1.0 / float(history.initial_hessian),
            "latest_pair_accepted": history.last_pair_accepted,
            "latest_s_norm": last_pair.get("s_norm", ""),
            "latest_y_norm": last_pair.get("y_norm", ""),
            "latest_s_dot_y": last_pair.get("s_dot_y", ""),
            "latest_secant_curvature": last_pair.get("secant_curvature", ""),
            "latest_secant_cosine": last_pair.get("secant_cosine", ""),
            "latest_force_change_norm": last_pair.get("force_change_norm", ""),
            "rotation_lbfgs_raw_direction_norm": history.last_dense_diagnostics.get(
                "lbfgs_raw_direction_norm", ""
            ),
            "rotation_lbfgs_raw_direction_max_atom_norm": history.last_dense_diagnostics.get(
                "lbfgs_raw_direction_max_atom_norm", ""
            ),
            "rotation_dense_bfgs_diagnostic": int(history.dense_bfgs_diagnostic),
        }
        dense_diag = dict(history.last_dense_diagnostics)
        self.lbfgs_diagnostics.update(dense_diag)
        self.lbfgs_diagnostics.update({
            "rotation_dense_pairs_requested": dense_diag.get("dense_pairs_requested", ""),
            "rotation_dense_pairs_applied": dense_diag.get("dense_pairs_applied", ""),
            "rotation_dense_invalid_reason": dense_diag.get("dense_invalid_reason", ""),
            "rotation_dense_reconstruction_ns": dense_diag.get("dense_reconstruction_ns", ""),
            "rotation_dense_vs_lbfgs_cosine": dense_diag.get("dense_vs_lbfgs_cosine", ""),
            "rotation_dense_vs_lbfgs_norm_ratio": dense_diag.get("dense_vs_lbfgs_norm_ratio", ""),
            "rotation_dense_vs_lbfgs_relative_difference": dense_diag.get(
                "dense_vs_lbfgs_relative_difference", ""
            ),
            "rotation_dense_min_eigenvalue": dense_diag.get("dense_min_eigenvalue", ""),
            "rotation_dense_max_eigenvalue": dense_diag.get("dense_max_eigenvalue", ""),
            "rotation_dense_negative_eigenvalues": dense_diag.get(
                "dense_negative_eigenvalues", ""
            ),
            "rotation_dense_condition_number": dense_diag.get("dense_condition_number", ""),
            "rotation_dense_log10_condition": dense_diag.get("dense_log10_condition", ""),
            "rotation_dense_secant_residual_latest": dense_diag.get(
                "dense_secant_residual_latest", ""
            ),
            "rotation_dense_secant_residual_median": dense_diag.get(
                "dense_secant_residual_median", ""
            ),
            "rotation_dense_secant_residual_max": dense_diag.get(
                "dense_secant_residual_max", ""
            ),
        })
        history_diagnostics = getattr(history, "diagnostics", None)
        if callable(history_diagnostics):
            self.lbfgs_diagnostics.update(history_diagnostics())


class EntryTorqueDimerEigenmodeSearch(DimerEntryTorqueCaptureMixin, DimerEigenmodeSearch):
    """ASE Dimer search with passive first-torque capture only."""


class LBFGSDimerEigenmodeSearch(
    LBFGSRotationMixin, DimerEntryTorqueCaptureMixin, DimerEigenmodeSearch
):
    pass


def _publish_entry_torque_capture(owner, search) -> dict[str, object]:
    """Publish one completed search's already-paid entry-torque snapshot."""
    payload = entry_torque_capture_payload(search)
    owner.last_dimer_entry_torque = payload
    return payload


class ConfigurableRotationMinModeAtoms(BowlBreakoutMixin, MinModeAtoms):
    """ASE MinModeAtoms with pluggable minimum-mode eigensolvers.

    ``min_mode_finder='dimer'`` preserves the existing ASE/L-BFGS dimer
    rotation behavior exactly. Iterative eigensolvers replace only minimum-mode
    finding; the downstream ASE MMF projected-force rule and selected geometry
    translator remain independent.

    ``olsen_jd`` passes SaddleMill's existing typed physical HVP operator
    directly into the installed Sella 2.5.0 ``rayleigh_ritz(..., method="jd0")``
    eigensolver through an active-coordinate adapter; SaddleMill does not
    maintain a second JD0 convergence/expansion implementation.

    Generic ``davidson`` owns a persistent physical BFGS Hessian initialized
    from either a scalar identity or an explicitly supplied/reference Hessian.
    ``softsaddle_davidson`` and ``softsaddle_davidson_textbook`` reproduce the
    historical SoftSaddle Davidson/Lanczos hybrid architecture: a full Hessian
    at the pre-displacement reference geometry seeds B, physical center
    secants BFGS-update B, and the threshold-12 Hq0-vs-Bq0 test chooses Davidson
    or Lanczos for each fresh mode solve. The two names differ only in the
    Davidson residual used when Davidson is selected.
    """

    ITERATIVE_FINDERS = {
        "lanczos",
        "davidson",
        "softsaddle_lanczos",
        "softsaddle_davidson",
        "softsaddle_davidson_textbook",
        "olsen_jd",
    }
    DAVIDSON_FINDERS = {
        "davidson",
        "softsaddle_davidson",
        "softsaddle_davidson_textbook",
        "olsen_jd",
    }
    SOFTSADDLE_DAVIDSON_FINDERS = {
        "softsaddle_davidson",
        "softsaddle_davidson_textbook",
    }

    def __init__(
        self,
        atoms,
        control=None,
        rotation_optimizer="ase",
        rotation_lbfgs_options=None,
        min_mode_finder="dimer",
        minmode_options=None,
        mode_reuse="none",
        convex_escape="standard",
        bowl_active_atoms=20,
        initial_hessian_matrix=None,
        **kwargs,
    ):
        self.rotation_optimizer = str(rotation_optimizer).lower()
        self.rotation_lbfgs_options = dict(rotation_lbfgs_options or {})
        self.min_mode_finder = str(min_mode_finder).lower()
        self.minmode_options = dict(minmode_options or {})
        self.mode_reuse = str(mode_reuse).lower()
        self.last_rotation_diagnostics = {}
        self.last_minmode_solver_result = None
        self.last_dimer_rotational_torque = None
        self.last_dimer_entry_torque = None
        self.translation_regime = "standard"
        self.configure_bowl_breakout(convex_escape, bowl_active_atoms)

        allowed = {"dimer", *self.ITERATIVE_FINDERS}
        if self.rotation_optimizer not in {
            "ase", "lbfgs", "cg", "generic_good_broyden", "johnson_modified_broyden"
        }:
            raise ValueError(
                "rotation_optimizer must be one of ase, lbfgs, cg, "
                "generic_good_broyden, johnson_modified_broyden; got "
                f"{rotation_optimizer!r}"
            )
        if self.min_mode_finder not in allowed:
            raise ValueError(
                "Unknown min_mode_finder="
                f"{self.min_mode_finder!r}; expected one of {sorted(allowed)}"
            )
        if self.min_mode_finder != "dimer" and self.rotation_optimizer != "ase":
            raise ValueError(
                "rotation_optimizer applies only to min_mode_finder=dimer; "
                "leave rotation_optimizer=ase for iterative eigensolvers"
            )
        if self.mode_reuse not in {"none", "sm50"}:
            raise ValueError("mode_reuse must be 'none' or 'sm50'")
        if self.mode_reuse == "sm50" and self.min_mode_finder == "dimer":
            raise ValueError(
                "mode_reuse=sm50 is implemented only for iterative "
                "minimum-mode finders"
            )

        # The Atoms passed here are still at the attempt's reference geometry;
        # _setup_dimer() applies the initializer displacement only after this
        # wrapper has been constructed. Preserve that geometry for the
        # SoftSaddle reference-minimum Hessian.
        self._reference_positions = np.asarray(
            atoms.get_positions(), dtype=float
        ).copy()
        self._initial_hessian_matrix = (
            None
            if initial_hessian_matrix is None
            else np.asarray(initial_hessian_matrix, dtype=float).copy()
        )
        self._reference_gradient = None
        self._reference_hessian_force_calls = 0
        self._reference_hessian_source = ""

        self._last_iterative_search_position = None
        self._mode_reused_since_last_search = False
        self._davidson_hessian = None
        super().__init__(atoms, control=control, **kwargs)

    def _movable_coordinate_mask(self):
        """Return a flat mask excluding atom-index constraints when available."""
        mask = np.ones((len(self.atoms), 3), dtype=bool)
        for constraint in list(getattr(self.atoms, "constraints", []) or []):
            getter = getattr(constraint, "get_indices", None)
            if getter is None:
                continue
            try:
                indices = np.asarray(getter(), dtype=int).reshape(-1)
            except Exception:
                continue
            indices = indices[(indices >= 0) & (indices < len(self.atoms))]
            mask[indices, :] = False
        return mask.reshape(-1)

    def _project_free_coordinates(self, vector):
        flat = np.asarray(vector, dtype=float).reshape(-1).copy()
        if flat.size != 3 * len(self.atoms):
            raise ValueError(
                f"Minimum-mode vector has {flat.size} coordinates; expected "
                f"{3 * len(self.atoms)}"
            )
        flat[~self._movable_coordinate_mask()] = 0.0
        return flat

    def _t09_coordinate_space_and_identity(self):
        from saddlemill.dimertools.foundation_types import geometry_fingerprint
        from saddlemill.dimertools.wave_b_runtime import active_space_for_owner

        center = np.asarray(self.get_positions(), dtype=float)
        space = active_space_for_owner(self, center)
        history = getattr(self, "canonical_force_history", None)
        state = getattr(history, "current", None) if history is not None else None
        if state is not None:
            state_id = int(state.state_id)
            state_uid = str(state.state_uid)
            geometry_id = str(state.geometry_id)
        else:
            geometry_id = geometry_fingerprint(center)
            state_id = 0
            state_uid = f"state0:{geometry_id}"
        return space, state_id, state_uid, geometry_id

    def _t09_hvp_backend(self, *, source, family, model=None, count_rotation=True):
        from saddlemill.dimertools.foundation_types import (
            CommonResultMetadata,
            OperatorOrigin,
        )
        from saddlemill.dimertools.hvp_interfaces import (
            ApproximatePhysicalModelHVPBackend,
            ExplicitMatrixHVPBackend,
            HVPResult,
        )

        owner = self
        space, state_id, state_uid, geometry_id = self._t09_coordinate_space_and_identity()
        origin = str(self.minmode_options.get("hvp_origin", "physical_fd")).strip().lower()
        if origin == "effective_force_jacobian":
            raise ValueError(
                "hvp_origin=effective_force_jacobian is not a physical HVP backend"
            )
        if origin == "explicit_physical_matrix":
            if self._initial_hessian_matrix is None:
                raise ValueError(
                    "hvp_origin=explicit_physical_matrix requires initial_hessian_matrix"
                )
            matrix = np.asarray(self._initial_hessian_matrix, dtype=float)
            n = 3 * len(self.atoms)
            if matrix.shape != (n, n):
                free = np.flatnonzero(space.active_dof_mask.reshape(-1))
                if matrix.shape != (free.size, free.size):
                    raise ValueError(
                        "initial_hessian_matrix shape is incompatible with active coordinates"
                    )
                full = np.zeros((n, n), dtype=float)
                full[np.ix_(free, free)] = matrix
                matrix = full
            return ExplicitMatrixHVPBackend(
                matrix, state_id=state_id, state_uid=state_uid,
                geometry_id=geometry_id, coordinate_space=space,
                source=source, family=family,
                provenance={"t09_hvp_origin": origin},
            ), (space, state_id, state_uid, geometry_id)
        if origin == "approximate_physical_model":
            candidate = model
            runtime = getattr(self, "wave_b_runtime", None)
            runtime_model = getattr(runtime, "physical_hessian", None) if runtime is not None else None
            if runtime_model is not None and getattr(runtime_model, "matrix", None) is not None:
                candidate = runtime_model
            if candidate is None or getattr(candidate, "matrix", None) is None:
                raise ValueError(
                    "hvp_origin=approximate_physical_model requires an available physical B model"
                )
            return ApproximatePhysicalModelHVPBackend(
                np.asarray(candidate.matrix, dtype=float),
                state_id=state_id, state_uid=state_uid, geometry_id=geometry_id,
                coordinate_space=space,
                model_age=int(getattr(candidate, "age", getattr(candidate, "accepted_updates", 0))),
                source=source, family=family,
                provenance={"t09_hvp_origin": origin},
            ), (space, state_id, state_uid, geometry_id)
        if origin != "physical_fd":
            raise ValueError(
                "hvp_origin must be physical_fd, explicit_physical_matrix, or "
                f"approximate_physical_model; got {origin!r}"
            )

        delta = float(self.minmode_options.get("finite_difference", 1.0e-4))
        if delta <= 0.0:
            raise ValueError("finite_difference must be > 0")

        class _ExecutingPhysicalFDBackend:
            def apply(self, request):
                if request.state_id != state_id or request.state_uid != state_uid:
                    raise ValueError("projected-Olsen/JD HVP request state identity mismatch")
                if request.geometry_id != geometry_id:
                    raise ValueError("projected-Olsen/JD HVP request geometry identity mismatch")
                if request.coordinate_space.identity != space.identity:
                    raise ValueError("projected-Olsen/JD HVP request coordinate-space mismatch")
                q = space.normalized(request.direction)
                center = np.asarray(owner.get_positions(), dtype=float)
                probe = center + delta * q
                before = int(owner.control.get_counter("forcecalls"))
                started = perf_counter_ns()
                evaluator = getattr(owner, "physical_force_evaluator", None)
                context = (
                    evaluator.source(
                        request.source,
                        family=request.family,
                        purpose=request.purpose,
                        metadata={
                            "direction_kind": "hvp",
                            "probe_direction": np.asarray(q, dtype=float),
                            "stencil_id": request.stencil_id,
                        },
                    )
                    if evaluator is not None else None
                )
                if context is None:
                    probe_forces = np.asarray(owner.get_forces(real=True, pos=probe), dtype=float)
                else:
                    with context:
                        probe_forces = np.asarray(owner.get_forces(real=True, pos=probe), dtype=float)
                elapsed = perf_counter_ns() - started
                after = int(owner.control.get_counter("forcecalls"))
                hq = -(probe_forces - np.asarray(owner.forces0, dtype=float)) / delta
                hq = space.project(hq)
                if count_rotation:
                    owner.control.increment_counter("rotcount")
                stencil_id = ""
                endpoint_ids = ()
                history = getattr(owner, "canonical_force_history", None)
                state = None if history is None else getattr(history, "current", None)
                if history is not None and state is not None:
                    matches = []
                    for stencil in getattr(history, "stencils", ()):
                        if int(getattr(stencil, "state_id", -1)) != int(state_id):
                            continue
                        if str(getattr(stencil, "source", "")) != str(request.source):
                            continue
                        if str(getattr(stencil, "family", "")) != str(request.family):
                            continue
                        candidate = space.normalized(getattr(stencil, "direction"))
                        overlap = float(np.dot(candidate.reshape(-1), q.reshape(-1)))
                        if overlap >= 1.0 - 1.0e-8:
                            matches.append(stencil)
                    if matches:
                        stencil = max(
                            matches,
                            key=lambda item: (int(getattr(item, "serial", -1)), str(getattr(item, "stencil_id", ""))),
                        )
                        stencil_id = str(getattr(stencil, "stencil_id", ""))
                        ids = history.stencil_endpoint_ids(stencil)
                        endpoint_ids = tuple(
                            str(ids[key]) for key in ("center", "plus", "minus")
                            if ids.get(key) not in (None, "")
                        )
                metadata = CommonResultMetadata(
                    geometry_id=geometry_id, state_id=state_id, state_uid=state_uid,
                    coordinate_space_id=space.identity, purpose=request.purpose,
                    source=request.source, family=request.family,
                    operator_origin=OperatorOrigin.FINITE_DIFFERENCE_PHYSICAL_FORCE,
                    physical=True, model_derived=False,
                    pes_call_delta=max(0, after - before), cache_hit=False,
                    timing_ns=elapsed, units="eV/Angstrom^2 * Cartesian-vector",
                    provenance={
                        "t09_hvp_origin": "physical_fd",
                        "stencil_id": stencil_id,
                        "endpoint_observation_ids": endpoint_ids,
                    },
                )
                return HVPResult(
                    available=True, action=hq, direction=q, metadata=metadata,
                    stencil_scheme="one_sided_plus", displacement_scale=delta,
                    endpoint_observation_ids=endpoint_ids,
                )

        return _ExecutingPhysicalFDBackend(), (space, state_id, state_uid, geometry_id)

    def _t09_typed_hvp(self, *, source, family, model=None, count_rotation=True):
        backend, identity = self._t09_hvp_backend(
            source=source, family=family, model=model, count_rotation=count_rotation
        )
        space, state_id, state_uid, geometry_id = identity
        callback = TypedHVPCallback(
            backend, coordinate_space=space, state_id=state_id, state_uid=state_uid,
            geometry_id=geometry_id, source=source, family=family, purpose="algorithm",
            request_metadata={"min_mode_finder": self.min_mode_finder},
        )
        return callback, backend, identity

    def _reference_hessian_fd(self):
        """Return a full embedded FD Hessian at the pre-displacement geometry.

        Only movable Cartesian columns are probed and only the movable/movable
        block is used. Fixed-coordinate rows/columns are left as a benign
        scalar diagonal because all iterative mode vectors are projected to
        zero there.

        ``reference_hessian_reuse_center_force=True`` evaluates the reference
        force once and reuses it for every finite-difference column. Historical
        SoftSaddle's C++ routine redundantly reevaluates the same reference
        gradient for every column (2 force calls per coordinate). Set the
        option to false when exact source-level force-call accounting is wanted;
        the finite-difference matrix is otherwise mathematically the same for a
        deterministic calculator.
        """
        if self._initial_hessian_matrix is not None:
            n = 3 * len(self.atoms)
            mask = self._movable_coordinate_mask()
            free = np.flatnonzero(mask)
            raw = np.asarray(self._initial_hessian_matrix, dtype=float)
            if not np.all(np.isfinite(raw)):
                raise ValueError("initial_hessian_matrix contains non-finite values")
            if raw.shape == (n, n):
                free_block = raw[np.ix_(free, free)]
            elif raw.shape == (free.size, free.size):
                free_block = raw
            else:
                raise ValueError(
                    "initial_hessian_matrix must be either full Cartesian "
                    f"shape {(n, n)} or movable-coordinate shape "
                    f"{(free.size, free.size)}; got {raw.shape}"
                )
            # Iterative modes live only in the movable-coordinate subspace.
            # Embed the supplied free/free block and zero fixed/free coupling so
            # constrained coordinates cannot contaminate the hybrid Hq0-Bq0
            # criterion. The fixed diagonal is never sampled by a mode vector.
            scale = float(
                self.minmode_options.get("davidson_initial_hessian", 1.0)
            )
            matrix = np.eye(n, dtype=float) * scale
            free_block = 0.5 * (free_block + free_block.T)
            matrix[np.ix_(free, free)] = free_block
            # We still need the physical reference gradient to transport B from
            # the supplied reference Hessian to the displaced first center.
            before = int(self.control.get_counter("forcecalls"))
            ref_forces = np.asarray(
                self.get_forces(real=True, pos=self._reference_positions),
                dtype=float,
            )
            after = int(self.control.get_counter("forcecalls"))
            self._reference_gradient = self._project_free_coordinates(-ref_forces)
            self._reference_hessian_force_calls += max(0, after - before)
            self._reference_hessian_source = "supplied_matrix"
            return matrix

        delta = float(
            self.minmode_options.get(
                "reference_hessian_finite_difference",
                self.minmode_options.get("finite_difference", 1.0e-4),
            )
        )
        if delta <= 0.0:
            raise ValueError("reference_hessian_finite_difference must be > 0")
        reuse_center = bool(
            self.minmode_options.get("reference_hessian_reuse_center_force", False)
        )
        mask = self._movable_coordinate_mask()
        free = np.flatnonzero(mask)
        if free.size == 0:
            raise ValueError("No movable coordinates for reference Hessian")

        n = 3 * len(self.atoms)
        scale = float(self.minmode_options.get("davidson_initial_hessian", 1.0))
        matrix = np.eye(n, dtype=float) * scale
        reference = np.asarray(self._reference_positions, dtype=float)
        before = int(self.control.get_counter("forcecalls"))

        ref_forces_shared = None
        ref_forces_for_gradient = None
        if reuse_center:
            ref_forces_shared = np.asarray(
                self.get_forces(real=True, pos=reference), dtype=float
            )
            ref_forces_for_gradient = ref_forces_shared

        for coordinate in free:
            if ref_forces_shared is None:
                ref_forces = np.asarray(
                    self.get_forces(real=True, pos=reference), dtype=float
                )
                # Historical SoftSaddle repeats this reference call for every
                # coordinate. Retain the most recent identical value so the
                # BFGS reference state costs no extra call beyond those 2N.
                ref_forces_for_gradient = ref_forces
            else:
                ref_forces = ref_forces_shared
            probe = reference.copy().reshape(-1)
            probe[coordinate] += delta
            probe_forces = np.asarray(
                self.get_forces(real=True, pos=probe.reshape((-1, 3))),
                dtype=float,
            )
            column = -(probe_forces - ref_forces).reshape(-1) / delta
            matrix[free, coordinate] = column[free]

        if ref_forces_for_gradient is None:
            raise RuntimeError("Reference Hessian did not evaluate a center force")

        block = matrix[np.ix_(free, free)]
        matrix[np.ix_(free, free)] = 0.5 * (block + block.T)
        self._reference_gradient = self._project_free_coordinates(
            -ref_forces_for_gradient
        )
        after = int(self.control.get_counter("forcecalls"))
        self._reference_hessian_force_calls += max(0, after - before)
        self._reference_hessian_source = (
            "reference_fd_center_reuse" if reuse_center else "reference_fd_legacy_calls"
        )
        return matrix

    def _sm50_threshold(self):
        factor = float(self.minmode_options.get("sm50_factor", 50.0))
        line_tol = float(
            self.minmode_options.get("sm50_line_search_tolerance", 0.01)
        )
        max_step = float(self.control.get_parameter("maximum_translation"))
        # Historical SoftSaddle expression:
        # tol = ls_err + (max_step_size-ls_err)*save_lanczos_factor/100.
        return line_tol + (max_step - line_tol) * factor / 100.0

    def _reuse_iterative_mode(self):
        if self.mode_reuse != "sm50":
            return False
        if self._last_iterative_search_position is None:
            return False
        displacement = np.asarray(self.get_positions(), dtype=float) - np.asarray(
            self._last_iterative_search_position, dtype=float
        )
        displacement = self._project_free_coordinates(displacement)
        # SoftSaddle sums accepted displacement vectors since the last true mode
        # solve and compares the norm of that vector sum. This is exactly the
        # net displacement from the last true mode-solve center.
        return float(norm(displacement)) <= self._sm50_threshold()

    def _ensure_davidson_hessian(self):
        if self._davidson_hessian is not None:
            return self._davidson_hessian

        options = self.minmode_options
        source = str(
            options.get("davidson_initial_hessian_source", "identity")
        ).strip().lower()
        if self.min_mode_finder in self.SOFTSADDLE_DAVIDSON_FINDERS:
            # Historical SoftSaddle always seeds B from a full Hessian at the
            # reference minimum. An explicitly supplied matrix overrides only
            # how that reference Hessian is obtained, not its role.
            source = "reference"

        initial_matrix = None
        initial_position = None
        initial_gradient = None
        if self._initial_hessian_matrix is not None or source in {
            "reference",
            "reference_fd",
        }:
            initial_matrix = self._reference_hessian_fd()
            initial_position = self._project_free_coordinates(
                self._reference_positions
            )
            initial_gradient = np.asarray(self._reference_gradient, dtype=float)
        elif source != "identity":
            raise ValueError(
                "davidson_initial_hessian_source must be identity or reference_fd"
            )

        self._davidson_hessian = PhysicalHessianBFGS(
            3 * len(self.atoms),
            initial_hessian=float(options.get("davidson_initial_hessian", 1.0)),
            denominator_tolerance=float(
                options.get("davidson_update_tolerance", 1.0e-12)
            ),
            initial_matrix=initial_matrix,
            initial_position=initial_position,
            initial_gradient=initial_gradient,
        )
        return self._davidson_hessian

    @staticmethod
    def _iterative_diagnostics(name, result, **extra):
        data = {
            "optimizer": name,
            "rotations": int(result.iterations) if result is not None else 0,
            "history_size": 0,
            "pairs_accepted": 0,
            "pairs_rejected": 0,
            "history_resets": 0,
            "direction_fallbacks": 0,
        }
        if result is not None:
            data.update({
                "eigenvalue": result.eigenvalue,
                "eigenvalue_change": result.eigenvalue_change,
                "residual_norm": result.residual_norm,
                "subspace_dimension": result.subspace_dimension,
                "eigensolver_converged": int(result.converged),
                "eigensolver_breakdown": int(result.breakdown),
                "solver_used": result.solver_used,
                "switch_metric": result.switch_metric,
                "residual_variant": result.residual_variant,
            })
            audit = getattr(result, "audit", None)
            if audit is not None:
                scalar_map = {
                    "selected_root_index": "selected_root_index",
                    "root_selection": "root_selection",
                    "stopping_rule": "stopping_rule",
                    "stopping_equation": "stopping_equation",
                    "stopping_reason": "stopping_reason",
                    "breakdown_reason": "breakdown_reason",
                    "hvp_count": "hvp_count",
                    "restart_count": "restart_count",
                    "restart_reason": "restart_reason",
                    "ritz_rank": "rank",
                    "full_residual_norm": "full_residual_norm",
                    "solver_operator_residual_norm": "solver_operator_residual_norm",
                    "projected_operator_asymmetry_norm": "projected_operator_asymmetry_norm",
                    "physical_action_count": "physical_action_count",
                    "model_action_count": "model_action_count",
                    "mode_certification_eligible": "certification_eligible",
                    "algorithm_pes_calls": "algorithm_pes_calls",
                    "diagnostic_pes_calls": "diagnostic_pes_calls",
                    "cache_hits": "cache_hits",
                    "ritz_gap": "gap",
                    "ritz_gap_eligible": "gap_eligible",
                    "degeneracy_unresolved": "degeneracy_unresolved",
                    "approximation_caveat": "approximation_caveat",
                }
                for output_key, audit_key in scalar_map.items():
                    value = getattr(audit, audit_key)
                    if hasattr(value, "value"):
                        value = value.value
                    data[output_key] = value
                    data[f"t09_{output_key}"] = value
                audit_metadata = dict(getattr(audit, "metadata", {}) or {})
                for output_key, metadata_key in (
                    ("sella_jd_gamma", "sella_gamma"),
                    ("sella_jd_maxiter", "sella_maxiter"),
                    ("sella_jd_effective_subspace_cap", "sella_effective_subspace_cap"),
                    ("sella_jd_negative_roots_sought", "sella_negative_roots_sought"),
                    ("sella_jd_cap_reached", "cap_reached"),
                    ("sella_jd_residual_converged", "residual_converged"),
                    ("sella_version", "sella_version"),
                ):
                    if metadata_key in audit_metadata:
                        value = audit_metadata[metadata_key]
                        data[output_key] = value
                        data[f"t09_{output_key}"] = value
                tuple_map = {
                    "hvp_action_origins": "action_origins",
                    "hvp_sources": "action_sources",
                    "hvp_families": "action_families",
                    "hvp_stencil_schemes": "action_stencil_schemes",
                    "hvp_displacement_scales": "action_displacement_scales",
                    "hvp_geometry_ids": "action_geometry_ids",
                    "hvp_state_uids": "action_state_uids",
                }
                for output_key, audit_key in tuple_map.items():
                    values = tuple(getattr(audit, audit_key))
                    data[output_key] = values
                    data[f"t09_{output_key}"] = values
                # The evaluated/returned vectors are diagnostic values, not hidden
                # solver state. Keep compact immutable tuples for serialization.
                data["evaluated_mode"] = (
                    None if audit.evaluated_mode is None
                    else tuple(np.asarray(audit.evaluated_mode, dtype=float).reshape(-1))
                )
                data["returned_mode"] = (
                    None if audit.returned_mode is None
                    else tuple(np.asarray(audit.returned_mode, dtype=float).reshape(-1))
                )
        data.update(extra)
        return {"phase_a": data}

    def _run_iterative_mode_search(self, force=False):
        if self.order != 1:
            raise NotImplementedError(
                "SaddleMill iterative minimum-mode finders currently support "
                "only first-order saddle searches"
            )

        self.control.reset_counter("rotcount")
        if not force and self._reuse_iterative_mode():
            self._mode_reused_since_last_search = True
            self.last_rotation_diagnostics = self._iterative_diagnostics(
                f"{self.min_mode_finder}_sm50_reuse",
                None,
                mode_reused=1,
                sm50_threshold=self._sm50_threshold(),
            )
            return

        options = self.minmode_options
        breakdown = float(options.get("breakdown_tolerance", 1.0e-12))
        q0 = self._project_free_coordinates(self.eigenmodes[0])
        qnorm = float(norm(q0))
        if qnorm <= breakdown:
            attempt_rng = getattr(self, "_saddlemill_attempt_rng", None)
            if attempt_rng is None:
                q0 = np.random.randn(3 * len(self.atoms))
            else:
                q0 = attempt_rng.numpy("initial_random_eigenmode").standard_normal(
                    3 * len(self.atoms)
                )
            q0 = self._project_free_coordinates(q0)
            qnorm = float(norm(q0))
        if qnorm <= breakdown:
            raise ValueError("No movable coordinates are available for mode search")
        q0 /= qnorm

        previous_eigenvalue = (
            float(self.curvatures[0])
            if self._last_iterative_search_position is not None
            else None
        )
        common = dict(
            max_iterations=int(options.get("max_iterations", 8)),
            eigenvalue_tolerance=float(options.get("eigenvalue_tolerance", 0.01)),
            breakdown_tolerance=breakdown,
            previous_eigenvalue=previous_eigenvalue,
        )

        bfgs_updates = ""
        bfgs_skips = ""
        solver_hvp_results = ()
        if self.min_mode_finder == "lanczos":
            hvp, _, _ = self._t09_typed_hvp(source="lanczos_hvp", family="lanczos")
            result = lanczos_lowest_mode(hvp, q0, **common)
            solver_hvp_results = tuple(hvp.results)
        elif self.min_mode_finder == "softsaddle_lanczos":
            hvp, _, _ = self._t09_typed_hvp(source="softsaddle_lanczos_hvp", family="lanczos")
            result = softsaddle_lanczos_lowest_mode(
                hvp, q0, convergence_mode="relative", **common,
            )
            solver_hvp_results = tuple(hvp.results)
        elif self.min_mode_finder in self.DAVIDSON_FINDERS:
            model = self._ensure_davidson_hessian()
            position = self._project_free_coordinates(self.get_positions())
            gradient = self._project_free_coordinates(-np.asarray(self.forces0))
            model.observe_center(position, gradient)
            source = f"{self.min_mode_finder}_hvp"
            hvp, backend, identity = self._t09_typed_hvp(
                source=source, family="davidson", model=model
            )

            if self.min_mode_finder == "davidson":
                result = davidson_lowest_mode(
                    hvp, q0, model.matrix,
                    preconditioner_floor=float(options.get("davidson_preconditioner_floor", 1.0e-8)),
                    **common,
                )
                solver_hvp_results = tuple(hvp.results)
            elif self.min_mode_finder == "olsen_jd":
                space, state_id, state_uid, geometry_id = identity
                root_policy = str(options.get("root_policy", "lowest")).strip().lower()
                root_selection = "homed_overlap" if root_policy == "homed_overlap" else root_policy
                restart_dimension = options.get("olsen_restart_dimension", None)
                if restart_dimension in (None, "", "none", "None"):
                    restart_dimension = None
                else:
                    restart_dimension = int(restart_dimension)
                wave_runtime = getattr(self, "wave_b_runtime", None)
                t03_model = (
                    None if wave_runtime is None else getattr(wave_runtime, "physical_hessian", None)
                )
                if t03_model is not None and getattr(t03_model, "matrix", None) is not None:
                    preconditioner = lambda: np.asarray(t03_model.matrix, dtype=float)
                    preconditioner_identity = "t03_physical_hessian_B"
                else:
                    preconditioner = lambda: np.asarray(model.matrix, dtype=float)
                    preconditioner_identity = "legacy_physical_B"
                class _RecordingBackend:
                    def __init__(self, delegate):
                        self.delegate = delegate
                        self.results = []

                    def apply(self, request):
                        value = self.delegate.apply(request)
                        self.results.append(value)
                        return value

                recording_backend = _RecordingBackend(backend)
                sella_maxiter = options.get("maxiter", None)
                if sella_maxiter in (None, "", "none", "None"):
                    sella_maxiter = None
                else:
                    sella_maxiter = int(sella_maxiter)
                result = olsen_jd_lowest_mode(
                    recording_backend, q0.reshape((-1, 3)), preconditioner,
                    coordinate_space=space, state_id=state_id, state_uid=state_uid,
                    geometry_id=geometry_id,
                    maxiter=sella_maxiter,
                    previous_eigenvalue=previous_eigenvalue,
                    breakdown_tolerance=breakdown,
                    root_selection=root_selection,
                    homing_vector=None,
                    source=source, family="davidson",
                    preconditioner_identity=preconditioner_identity,
                )
                solver_hvp_results = tuple(recording_backend.results)
            else:
                residual_variant = (
                    "textbook" if self.min_mode_finder == "softsaddle_davidson_textbook"
                    else "legacy_unweighted"
                )
                result = softsaddle_davidson_hybrid_lowest_mode(
                    hvp, q0, model.matrix,
                    preconditioner_floor=float(options.get("softsaddle_preconditioner_floor", 1.0e-12)),
                    switch_threshold=float(options.get("softsaddle_switch_threshold", 12.0)),
                    residual_variant=residual_variant,
                    **common,
                )
                solver_hvp_results = tuple(hvp.results)
            bfgs_updates = int(model.accepted_updates)
            bfgs_skips = int(model.skipped_updates)
        else:
            raise RuntimeError("Iterative mode search called for dimer finder")

        wave_runtime = getattr(self, "wave_b_runtime", None)
        if wave_runtime is not None:
            wave_runtime.commit_solver_retained_probe_block(
                self,
                result,
                solver_hvp_results,
            )

        # shared-runtime diagnostic handoff: retain the exact already-computed projected-Olsen/JD
        # result for passive mode-diagnostics standardization.  It is never consulted by the
        # numerical mode solver itself.
        self.last_minmode_solver_result = result
        mode = result.eigenvector.reshape((-1, 3))
        old_mode = np.asarray(self.eigenmodes[0], dtype=float)
        if float(np.vdot(old_mode.ravel(), mode.ravel()).real) < 0.0:
            mode *= -1.0
        self.eigenmodes[0] = mode
        self.curvatures[0] = float(result.eigenvalue)
        self._last_iterative_search_position = np.asarray(
            self.get_positions(), dtype=float
        ).copy()
        self._mode_reused_since_last_search = False
        self.last_rotation_diagnostics = self._iterative_diagnostics(
            self.min_mode_finder,
            result,
            mode_reused=0,
            davidson_bfgs_updates=bfgs_updates,
            davidson_bfgs_skips=bfgs_skips,
            reference_hessian_source=self._reference_hessian_source,
            reference_hessian_force_calls=int(self._reference_hessian_force_calls),
        )

    def get_projected_forces(self, pos=None):
        forces = super().get_projected_forces(pos=pos)
        forces = self.apply_bowl_breakout_force_mask(forces, center=(pos is None))
        if pos is None:
            runtime = getattr(self, "wave_b_runtime", None)
            if runtime is not None:
                forces = runtime.apply_parallel_force_policy(self, forces)
        return forces

    def refresh_reused_mode_for_convergence(self):
        """Force SoftSaddle's final fresh-mode check after SM50 reuse."""
        if not self._mode_reused_since_last_search:
            return False
        self._run_iterative_mode_search(force=True)
        return True

    def _capture_t12_cached_dimer_torque(self):
        """Expose ASE's already-computed final Dimer torque without PES work."""
        try:
            torque = np.asarray(self.get_rotational_force(), dtype=float).copy()
            if torque.shape == np.asarray(self.get_eigenmode(), dtype=float).shape and np.all(np.isfinite(torque)):
                self.last_dimer_rotational_torque = torque
            else:
                self.last_dimer_rotational_torque = None
        except Exception:
            self.last_dimer_rotational_torque = None

    def find_eigenmodes(self, order=1):
        if self.min_mode_finder in self.ITERATIVE_FINDERS:
            if order != 1:
                raise NotImplementedError(
                    "Iterative minimum-mode finders are currently wired only "
                    "for order=1"
                )
            self._run_iterative_mode_search()
            return

        # Existing ASE Dimer mathematics are preserved.  The loop mirrors ASE
        # MinModeAtoms.find_eigenmodes but swaps only the search class so the
        # first already-computed rotational force can be recorded passively.
        if self.rotation_optimizer == "ase":
            if self.control.get_parameter("eigenmode_method").lower() != "dimer":
                raise NotImplementedError("Only the Dimer eigenmode method is implemented")
            first_search = None
            for k in range(order):
                if k > 0:
                    self.ensure_eigenmode_orthogonality(k + 1)
                search = EntryTorqueDimerEigenmodeSearch(
                    self, self.control, eigenmode=self.eigenmodes[k], basis=self.eigenmodes[:k]
                )
                search.converge_to_eigenmode()
                search.set_up_for_optimization_step()
                self.eigenmodes[k] = search.get_eigenmode()
                self.curvatures[k] = search.get_curvature()
                if first_search is None:
                    first_search = search
            if first_search is not None:
                _publish_entry_torque_capture(self, first_search)
            self._capture_t12_cached_dimer_torque()
            self.last_rotation_diagnostics = {
                "phase_a": {
                    "optimizer": "ase",
                    "rotations": int(self.control.get_counter("rotcount")),
                    "history_size": 0,
                    "pairs_accepted": 0,
                    "pairs_rejected": 0,
                    "history_resets": 0,
                    "direction_fallbacks": 0,
                    "entry_torque": dict(self.last_dimer_entry_torque or {}),
                }
            }
            return

        if self.control.get_parameter("eigenmode_method").lower() != "dimer":
            raise NotImplementedError("Only the Dimer eigenmode method is implemented")

        phase_diagnostics = []
        first_search = None
        for k in range(order):
            if k > 0:
                self.ensure_eigenmode_orthogonality(k + 1)
            history_options = dict(
                getattr(self, "canonical_history_options", {}) or {}
            )
            if self.rotation_optimizer == "cg":
                from saddlemill.dimertools.wave_b_rotation import PRPlusDimerEigenmodeSearch
                search = PRPlusDimerEigenmodeSearch(
                    self, self.control, eigenmode=self.eigenmodes[k], basis=self.eigenmodes[:k],
                    lbfgs_options=self.rotation_lbfgs_options,
                    canonical_force_history=getattr(self, "canonical_force_history", None),
                    history_options=history_options,
                    cg_options=dict(getattr(self, "wave_b_options", {}).get("cg", {}) or {}),
                )
            elif self.rotation_optimizer in {"generic_good_broyden", "johnson_modified_broyden"}:
                from saddlemill.dimertools.wave_b_rotation import BroydenDimerEigenmodeSearch
                search = BroydenDimerEigenmodeSearch(
                    self, self.control, eigenmode=self.eigenmodes[k], basis=self.eigenmodes[:k],
                    lbfgs_options=self.rotation_lbfgs_options,
                    canonical_force_history=getattr(self, "canonical_force_history", None),
                    history_options=history_options, broyden_family=self.rotation_optimizer,
                    broyden_options=dict(getattr(self, "wave_b_options", {}).get("broyden", {}) or {}),
                )
            else:
                from saddlemill.dimertools.dimer_rotation import (
                    GeneralizedLBFGSDimerEigenmodeSearch, advanced_rotation_requested,
                )
                if advanced_rotation_requested(self.rotation_lbfgs_options, history_options):
                    search = GeneralizedLBFGSDimerEigenmodeSearch(
                        self, self.control, eigenmode=self.eigenmodes[k], basis=self.eigenmodes[:k],
                        lbfgs_options=self.rotation_lbfgs_options,
                        canonical_force_history=getattr(self, "canonical_force_history", None),
                        history_options=history_options,
                    )
                else:
                    legacy_options = {
                        key: self.rotation_lbfgs_options[key]
                        for key in (
                            "memory", "initial_hessian", "dynamic_h0",
                            "curvature_epsilon", "dense_bfgs_diagnostic",
                        )
                        if key in self.rotation_lbfgs_options
                    }
                    search = LBFGSDimerEigenmodeSearch(
                        self, self.control, eigenmode=self.eigenmodes[k], basis=self.eigenmodes[:k],
                        lbfgs_options=legacy_options,
                        lbfgs_history=getattr(self, "_canonical_rotation_history", None),
                    )
            search.converge_to_eigenmode()
            search.set_up_for_optimization_step()
            self.eigenmodes[k] = search.get_eigenmode()
            self.curvatures[k] = search.get_curvature()
            if first_search is None:
                first_search = search
                _publish_entry_torque_capture(self, search)
            phase_diagnostics.append(dict(search.lbfgs_diagnostics))

        self._capture_t12_cached_dimer_torque()
        phase_a = phase_diagnostics[0] if phase_diagnostics else {}
        if phase_a is not None:
            phase_a = dict(phase_a)
            phase_a["entry_torque"] = dict(self.last_dimer_entry_torque or {})
        self.last_rotation_diagnostics = {
            "phase_a": phase_a
        }
        # dense-replay is a passive replay diagnostic.  The legacy rotational L-BFGS
        # implementation does not currently expose one frozen common-coordinate
        # final window/H0 after all tangent transport/projection choices.  Do not
        # reconstruct such a window from history: report the contract-required
        # structured unavailable outcome instead.
        if self.rotation_optimizer == "lbfgs":
            from saddlemill.dimertools.wave_b_shadow import (
                qn_shadow_options_from_dimeratoms,
                unavailable_shadow_row,
            )
            shadow_options = qn_shadow_options_from_dimeratoms(self)
            if bool(shadow_options.get("enabled", False)):
                self.last_rotation_diagnostics["t08_qn_shadow"] = unavailable_shadow_row(
                    consumer="saddle_rotation",
                    residual_kind="rotation_residual",
                    force_interpretation="effective_residual",
                    model_type="rotation_residual_bfgs",
                    reason="exact_common_coordinate_rotation_window_not_exposed",
                )



