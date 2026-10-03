"""Compatibility facade for historical SaddleMill Dimer/L-BFGS imports.

The original implementation combined ASE L-BFGS state compatibility, guarded
minimization, Dimer rotation, minimum-mode wiring, translation diagnostics, and
hybrid switching in one module.  Those cohesive implementations now live in
``ase_lbfgs_adapter`` and ``legacy_dimer_adapter``.  Existing imports from this
module, including the private helpers used by current SaddleMill consumers, are
kept as stable re-exports.

This split is implementation-only.  It does not change optimizer defaults,
curvature guards, force definitions/signs, secant chronology, H0, memory,
max-step handling, convergence tests, or calculator calls.
"""

from __future__ import annotations

# Preserve the historical module namespace for callers that imported incidental
# ASE/numerical names from this adapter before it became a compatibility facade.
from collections import deque
from dataclasses import dataclass
from math import atan, cos, pi, sin, tan
from typing import Callable, Optional
import warnings

import numpy as np

from ase.optimize import FIRE, LBFGS
from ase.mep.dimer import (
    DimerEigenmodeSearch,
    DimerControl,
    MinModeAtoms,
    MinModeTranslate,
    normalize,
    perpendicular_vector,
    rotate_vectors,
)

from saddlemill.dimertools.canonical_diagnostics import (
    REGULARIZATION_NOT_APPLICABLE,
    regularization_diagnostic_fields,
)
from saddlemill.dimertools.dense_bfgs import (
    DenseSecant,
    compare_directions,
    reconstruct_sequential_bfgs,
    shifted_trust_region_solve,
    safe_max_atom_norm,
    safe_norm,
)
from saddlemill.dimertools.qn_deep_diagnostics import (
    adaptive_shifted_lbfgs,
    shifted_lbfgs_solve,
    rigid_translation_basis,
    secant_row,
    progressive_lbfgs,
    translation_ratio,
)
from saddlemill.dimertools.lbfgs_state_dump import (
    rigid_translation_basis,
    step_selected,
    trace_two_loop,
)
from saddlemill.dimertools.minmode_solvers import (
    PhysicalHessianBFGS,
    davidson_lowest_mode,
    lanczos_lowest_mode,
    softsaddle_davidson_hybrid_lowest_mode,
    softsaddle_lanczos_lowest_mode,
)

from saddlemill.dimertools.ase_lbfgs_adapter import (
    DEFAULT_CURVATURE_FLOOR,
    DEFAULT_POWELL_ETA,
    VALID_CURVATURE_GUARDS,
    DiagnosticMinModeTranslate,
    HybridDecision,
    HybridDimerStateController,
    HybridMinModeTranslate,
    LBFGSMinModeTranslate,
    _ASELBFGSState,
    _CurvatureGuardedLBFGS,
    _DimerLBFGSLogMixin,
    _RealForceConvergenceMixin,
    _TranslationDiagnosticsMixin,
    _apply_lbfgs_direct_hessian,
    _ase_lbfgs_api_name,
    _ase_lbfgs_container,
    _ase_lbfgs_diagnostic_metrics,
    _ase_lbfgs_get,
    _ase_lbfgs_history_size,
    _ase_lbfgs_increment_iteration,
    _ase_lbfgs_latest_pair_metrics,
    _ase_lbfgs_pairs_total,
    _ase_lbfgs_reset_history,
    _ase_lbfgs_set,
    _cosine_alignment,
    _damp_secant_forces,
    _flatten_rotation_diagnostics,
    _force_calls,
    _lbfgs_step_metrics,
    _powell_damp_secant_forces,
    _projected_fmax,
    _real_fmax,
    _secant_pair_metrics,
    _translation_state_key,
)
from saddlemill.dimertools.legacy_dimer_adapter import (
    BowlBreakoutMixin,
    ConfigurableRotationMinModeAtoms,
    LBFGSDimerEigenmodeSearch,
    LBFGSRotationMixin,
    LimitedMemoryInverseHessian,
)

norm = np.linalg.norm

# Keep class/function metadata on the historical module path so new pickle-like
# references and introspection remain compatible with pre-split imports.  The
# implementation globals stay in their cohesive modules.
_COMPAT_METADATA_NAMES = (
    "_ase_lbfgs_container",
    "_ase_lbfgs_api_name",
    "_ase_lbfgs_get",
    "_ase_lbfgs_set",
    "_ase_lbfgs_increment_iteration",
    "_ase_lbfgs_history_size",
    "_ase_lbfgs_pairs_total",
    "_secant_pair_metrics",
    "_ase_lbfgs_latest_pair_metrics",
    "_ase_lbfgs_diagnostic_metrics",
    "_lbfgs_step_metrics",
    "_ase_lbfgs_reset_history",
    "_damp_secant_forces",
    "_apply_lbfgs_direct_hessian",
    "_powell_damp_secant_forces",
    "_CurvatureGuardedLBFGS",
    "_translation_state_key",
    "BowlBreakoutMixin",
    "LimitedMemoryInverseHessian",
    "LBFGSRotationMixin",
    "LBFGSDimerEigenmodeSearch",
    "ConfigurableRotationMinModeAtoms",
    "_force_calls",
    "_projected_fmax",
    "_real_fmax",
    "_cosine_alignment",
    "_flatten_rotation_diagnostics",
    "_RealForceConvergenceMixin",
    "_TranslationDiagnosticsMixin",
    "DiagnosticMinModeTranslate",
    "_ASELBFGSState",
    "_DimerLBFGSLogMixin",
    "LBFGSMinModeTranslate",
    "HybridDecision",
    "HybridDimerStateController",
    "HybridMinModeTranslate",
)
for _compat_name in _COMPAT_METADATA_NAMES:
    _compat_object = globals()[_compat_name]
    if hasattr(_compat_object, "__module__"):
        _compat_object.__module__ = __name__
del _compat_name, _compat_object, _COMPAT_METADATA_NAMES
