"""shared-runtime adapters connecting verified quasi-Newton and rotation-CG kernels to Dimer Fourier mechanics."""
from __future__ import annotations

import numpy as np

from saddlemill.dimertools.broyden import GenericGoodBroydenInverse, JohnsonModifiedBroyden
from saddlemill.dimertools.broyden_rotation import RotationBroydenAdapter
from saddlemill.dimertools.cg_rotation import PRPlusRotationCG
from saddlemill.dimertools.dimer_rotation import GeneralizedLBFGSDimerEigenmodeSearch
from saddlemill.dimertools.foundation_types import geometry_fingerprint
from saddlemill.dimertools.sphere_manifold import project_tangent


def _center_identity(search) -> str:
    history = getattr(search.dimeratoms, "canonical_force_history", None)
    state = None if history is None else history.current
    if state is not None:
        return state.state_uid
    return "geometry:" + geometry_fingerprint(np.asarray(search.dimeratoms.get_positions(), dtype=float))


def _root_identity(search) -> str:
    return f"root:{0 if search.basis is None else len(search.basis)}"


def _make_broyden_kernel(family: str, options: dict[str, object]):
    if family == "generic_good_broyden":
        return GenericGoodBroydenInverse(
            initial_inverse_scale=float(options.get("generic_initial_inverse_scale", -1.0)),
            history_cap=int(options.get("generic_history_cap", 20)),
            denominator_tolerance=float(options.get("generic_denominator_tolerance", 1.0e-12)),
            vector_tolerance=float(options.get("generic_vector_tolerance", 1.0e-14)),
            rank_tolerance=float(options.get("generic_rank_tolerance", 1.0e-12)),
            max_condition=float(options.get("generic_max_condition", 1.0e12)),
        )
    if family == "johnson_modified_broyden":
        return JohnsonModifiedBroyden(
            initial_step_scale=float(options.get("johnson_initial_step_scale", 1.0)),
            history_cap=int(options.get("johnson_history_cap", 20)),
            regularization_w0=float(options.get("johnson_regularization_w0", 0.01)),
            default_weight=float(options.get("johnson_default_weight", 1.0)),
            vector_tolerance=float(options.get("johnson_vector_tolerance", 1.0e-14)),
            rank_tolerance=float(options.get("johnson_rank_tolerance", 1.0e-12)),
            max_condition=float(options.get("johnson_max_condition", 1.0e12)),
        )
    raise ValueError(f"unsupported rotation Broyden family {family!r}")


class _NoQNAdmissionRotationSearch(GeneralizedLBFGSDimerEigenmodeSearch):
    """Reuse exact Dimer trial/Fourier mechanics without L-BFGS pair admission."""
    def _add_local_point(self, point):
        return None
    def _add_sequential_physical_point(self, point):
        return None
    def _external_pairs(self):
        self._last_build_metrics = {}
        return ()


class PRPlusDimerEigenmodeSearch(_NoQNAdmissionRotationSearch):
    def __init__(self, *args, cg_options=None, **kwargs):
        self.cg_options = dict(cg_options or {})
        super().__init__(*args, **kwargs)
        store = getattr(self.dimeratoms, "_wave_b_cg_by_root", None)
        if store is None:
            store = {}
            self.dimeratoms._wave_b_cg_by_root = store
        key = _root_identity(self)
        existing = store.get(key)
        self.cg = PRPlusRotationCG.from_state_dict(existing) if existing else PRPlusRotationCG(
            reset_policy=self.cg_options["reset_policy"],
            denominator_tolerance=float(self.cg_options.get("denominator_tolerance", 1.0e-24)),
            tangent_tolerance=float(self.cg_options.get("tangent_tolerance", 1.0e-14)),
            descent_tolerance=float(self.cg_options.get("descent_tolerance", 0.0)),
        )
        self._root_key = key

    def _rotation_direction(self, point):
        proposal = self.cg.propose(
            point.mode, point.force,
            center_id=_center_identity(self),
            sequence_id=f"attempt:{self._root_key}",
            mode_identity=self._root_key,
        )
        self.dimeratoms._wave_b_cg_by_root[self._root_key] = self.cg.state_dict()
        self._last_model_metrics = proposal.diagnostics()
        return np.asarray(project_tangent(point.mode, proposal.direction), dtype=float), int(proposal.used_history)

    def _diagnostics(self):
        diag = dict(self.cg.last_direction.diagnostics()) if self.cg.last_direction is not None else {}
        diag.update({
            "optimizer": "cg", "rotations": int(self.control.get_counter("rotcount")),
            "history_size": int(self.cg.state_age > 0), "pairs_accepted": 0, "pairs_rejected": 0,
            "history_resets": int(self.cg.reset_count), "direction_fallbacks": 0,
            "rotation_step_method": self.step_method,
            "rotation_numerical_path": "existing_dimer_fourier_with_pr_plus_direction",
        })
        self.lbfgs_diagnostics = diag


class BroydenDimerEigenmodeSearch(_NoQNAdmissionRotationSearch):
    def __init__(self, *args, broyden_family: str, broyden_options=None, **kwargs):
        self.broyden_family = str(broyden_family)
        self.broyden_options = dict(broyden_options or {})
        super().__init__(*args, **kwargs)
        store = getattr(self.dimeratoms, "_wave_b_rotation_broyden_by_root", None)
        if store is None:
            store = {}
            self.dimeratoms._wave_b_rotation_broyden_by_root = store
        key = _root_identity(self)
        existing = store.get(key)
        if existing:
            self.adapter = RotationBroydenAdapter.from_state_dict(existing)
        else:
            self.adapter = RotationBroydenAdapter(
                _make_broyden_kernel(self.broyden_family, self.broyden_options),
                max_angle=float(self.max_angle), map_kind="retraction",
            )
        self._root_key = key

    def _rotation_direction(self, point):
        result = self.adapter.step(
            point.mode, point.force,
            center_identity=_center_identity(self), mode_identity=self._root_key,
        )
        self.dimeratoms._wave_b_rotation_broyden_by_root[self._root_key] = self.adapter.state_dict()
        self._last_model_metrics = {
            "kernel_family": result.kernel_family,
            "history_size": result.history_size,
            "reset_reason": result.reset_reason,
        }
        return np.asarray(result.tangent, dtype=float), int(result.history_size)

    def _diagnostics(self):
        self.lbfgs_diagnostics = {
            "optimizer": self.broyden_family,
            "rotations": int(self.control.get_counter("rotcount")),
            "history_size": len(getattr(self.adapter, "_pairs", ())),
            "pairs_accepted": int(getattr(self.adapter.kernel, "accepted_updates", 0)),
            "pairs_rejected": int(getattr(self.adapter.kernel, "rejected_updates", 0)),
            "history_resets": int(self.adapter.reset_count),
            "direction_fallbacks": 0,
            "operator_origin": "rotation_residual_inverse_jacobian_nonphysical",
            "rotation_step_method": self.step_method,
            "rotation_numerical_path": "existing_dimer_fourier_with_broyden_direction",
        }


__all__ = ["PRPlusDimerEigenmodeSearch", "BroydenDimerEigenmodeSearch", "_make_broyden_kernel"]
