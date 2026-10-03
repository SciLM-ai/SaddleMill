import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from saddlemill.dimertools.dimer_factory import partitioned_options_from_config
from saddlemill.dimertools.force_history import CanonicalForceHistory
from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace
from saddlemill.dimertools.partitioned_lbfgs import PartitionedLBFGS
from saddlemill.dimertools.qn_deep_diagnostics import adaptive_shifted_lbfgs
from saddlemill.dimertools.quasi_newton import ProjectionSnapshot, reconstruct_lbfgs


def _fixture_history():
    """Small exact quadratic with one unstable P mode and stable Q space."""
    space = ActiveCoordinateSpace.all_cartesian(2)
    hessian = np.diag([-2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def force(x):
        return -(hessian @ np.asarray(x, dtype=float).reshape(-1)).reshape(2, 3)

    history = CanonicalForceHistory(
        memory_states=10,
        record_sources="center rotation",
    )
    centers = [
        np.array([[0.10, 0.08, 0.00], [0.02, 0.00, 0.00]]),
        np.array([[0.18, 0.13, 0.00], [0.03, 0.00, 0.00]]),
        np.array([[0.24, 0.17, 0.00], [0.04, 0.00, 0.00]]),
    ]
    probe_delta = np.array([[0.01, 0.004, 0.00], [0.00, 0.00, 0.00]])
    for center in centers:
        history.observe_center(
            center,
            force(center),
            active_dof_mask=space.active_dof_mask,
            coordinate_convention=space.convention,
            force_call_delta=0,
        )
        probe = center + probe_delta
        history.observe_probe(
            probe,
            force(probe),
            source="rotation_trial",
            family="rotation",
            force_call_delta=0,
        )
    mode = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return history, space, mode


def _core(regularization="off", *, radius=0.01, maximum_translation=1.0):
    del maximum_translation
    return PartitionedLBFGS(
        pair_sources="center_center center_rotation",
        p_initial_hessian=7.0,
        q_initial_hessian=11.0,
        p_safeguard="skip",
        q_safeguard="skip",
        regularization=regularization,
        regularization_radius=radius,
        regularization_tolerance=1.0e-10,
    )


def _step(core, history, space, mode, *, maximum_translation=1.0):
    return core.step(
        history=history,
        current_observation=history.current.center,
        mode=mode,
        active_space=space,
        maximum_translation=maximum_translation,
    )


def test_full_dimer_selector_composition_validates_without_new_partition_specific_knobs():
    # Run validation in a fresh interpreter because several legacy W5 tests
    # intentionally stub runtime modules in-process.  This also exercises the
    # public config validation path in the same clean-import mode used on LS6.
    source_root = Path(__file__).resolve().parents[1]
    code = r"""
import copy
from saddlemill.config_defaults import DEFAULTS
from saddlemill.config_validation import validate_method_config
cfg = copy.deepcopy(DEFAULTS)
cfg["Main"]["method"] = "Dimer"
cfg["ourDimer"].update({
    "engine": "mmf",
    "min_mode_finder": "dimer",
    "rotation_optimizer": "lbfgs",
    "translation_optimizer": "partitioned_lbfgs",
})
cfg["ourDimerHistory"].update({
    "enabled": True,
    "record_sources": "center rotation",
    "pair_sources": "center_center center_rotation",
    "rotation_reuse": True,
    "rotation_history_source": "force_bank",
})
cfg["ourDimerLBFGS"]["translation_regularization"] = "shifted_lbfgs_trust"
validate_method_config(cfg, normalize_run_jobs=lambda value: value)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_root)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_partitioned_config_reuses_generic_translation_regularization_surface():
    cfg = {
        "ourDimerHistory": {
            "enabled": True,
            "record_sources": "center rotation",
            "pair_sources": "center_center center_rotation",
        },
        "ourDimerLBFGS": {
            "translation_regularization": "shifted_lbfgs_trust",
            "translation_regularization_mu": 2.5,
            "translation_regularization_radius": 0.037,
            "translation_regularization_tolerance": 2.0e-9,
        },
        "ourPartitionedLBFGS": {
            "p_initial_hessian": 80.0,
            "q_initial_hessian": 90.0,
        },
    }
    options = partitioned_options_from_config(cfg)
    assert options["regularization"] == "shifted_lbfgs_trust"
    assert options["regularization_mu"] == pytest.approx(2.5)
    assert options["regularization_radius"] == pytest.approx(0.037)
    assert options["regularization_tolerance"] == pytest.approx(2.0e-9)
    assert options["p_initial_hessian"] == pytest.approx(80.0)
    assert options["q_initial_hessian"] == pytest.approx(90.0)
    assert options["pair_sources"] == "center_center center_rotation"


def test_regularization_off_and_inactive_trust_are_numerically_identical():
    history, space, mode = _fixture_history()
    off = _step(_core("off"), history, space, mode)
    inactive = _step(
        _core("shifted_lbfgs_trust", radius=10.0),
        history,
        space,
        mode,
    )
    np.testing.assert_allclose(inactive.step, off.step, rtol=0.0, atol=1.0e-14)
    assert inactive.diagnostics["translation_regularization_mu"] == pytest.approx(0.0)
    assert inactive.diagnostics["p_used_pair_ids"] == off.diagnostics["p_used_pair_ids"]
    assert inactive.diagnostics["q_used_pair_ids"] == off.diagnostics["q_used_pair_ids"]



def test_unpartitioned_shifted_lbfgs_uses_same_forcebank_history_and_internal_trust_solve():
    """Cover the ordinary/shifted half of the requested T04/T05/T06/T07 matrix."""
    history, _space, mode = _fixture_history()
    snapshot = ProjectionSnapshot(
        mode=mode,
        curvature=-2.0,
        branch="concave",
        regime="standard",
    )
    current_force = snapshot.project(history.current.center.forces)
    before = json.dumps(history.to_state_dict(), sort_keys=True)
    ordinary = reconstruct_lbfgs(
        history,
        snapshot,
        current_force,
        pair_sources="center_center center_rotation",
        initial_hessian=7.0,
        safeguard="skip",
    )
    shifted = adaptive_shifted_lbfgs(
        ordinary.projected_current_force,
        ordinary.accepted_pairs,
        7.0,
        radius=0.01,
        tol=1.0e-10,
    )
    after = json.dumps(history.to_state_dict(), sort_keys=True)

    assert before == after
    assert np.linalg.norm(ordinary.direction) > 0.01
    assert np.linalg.norm(shifted.direction) == pytest.approx(0.01, rel=5.0e-8, abs=5.0e-10)
    assert shifted.mu > 0.0
    clipped = ordinary.direction.reshape(-1) * (0.01 / np.linalg.norm(ordinary.direction))
    assert not np.allclose(shifted.direction, clipped, rtol=1.0e-7, atol=1.0e-9)


def test_shared_shift_hits_combined_radius_without_posthoc_scalar_clipping():
    history, space, mode = _fixture_history()
    off = _step(_core("off"), history, space, mode)
    shifted = _step(_core("shifted_lbfgs_trust", radius=0.01), history, space, mode)

    assert np.linalg.norm(off.step) > 0.01
    assert np.linalg.norm(shifted.step) == pytest.approx(0.01, rel=5.0e-8, abs=5.0e-10)
    assert shifted.diagnostics["translation_regularization_applied"] == 1
    assert shifted.diagnostics["translation_regularization_mu"] > 0.0
    assert shifted.diagnostics["translation_regularization_shift_scope"] == "shared_scalar_mu_across_p_q_blocks"
    assert shifted.diagnostics["translation_regularization_radius_metric"] == "combined_step_euclidean_norm"
    assert shifted.diagnostics["translation_trust_regularized_boundary_norm"] == pytest.approx(0.01, rel=5.0e-8)

    # A post-hoc trust clip would be exactly collinear with the unshifted step.
    clipped = off.step * (0.01 / np.linalg.norm(off.step))
    assert not np.allclose(shifted.step, clipped, rtol=1.0e-7, atol=1.0e-9)
    cosine = float(
        np.vdot(off.step.reshape(-1), shifted.step.reshape(-1)).real
        / (np.linalg.norm(off.step) * np.linalg.norm(shifted.step))
    )
    assert cosine < 0.999999


def test_shift_uses_same_admitted_pairs_and_does_not_mutate_forcebank_or_accounting():
    history, space, mode = _fixture_history()
    before = json.dumps(history.to_state_dict(), sort_keys=True)
    off = _step(_core("off"), history, space, mode)
    after_off = json.dumps(history.to_state_dict(), sort_keys=True)
    shifted = _step(_core("shifted_lbfgs_trust", radius=0.01), history, space, mode)
    after_shift = json.dumps(history.to_state_dict(), sort_keys=True)

    assert before == after_off == after_shift
    assert shifted.diagnostics["p_used_pair_ids"] == off.diagnostics["p_used_pair_ids"]
    assert shifted.diagnostics["q_used_pair_ids"] == off.diagnostics["q_used_pair_ids"]
    assert shifted.diagnostics["p_pairs_used"] == off.diagnostics["p_pairs_used"]
    assert shifted.diagnostics["q_pairs_used"] == off.diagnostics["q_pairs_used"]
    assert shifted.diagnostics["mode_hv_additional_pes_calls"] == 0


def test_existing_final_maximum_translation_cap_remains_after_internal_shift():
    history, space, mode = _fixture_history()
    result = _step(
        _core("shifted_lbfgs_fixed"),
        history,
        space,
        mode,
        maximum_translation=0.002,
    )
    atom_norms = np.linalg.norm(result.step, axis=1)
    assert float(np.max(atom_norms)) <= 0.002 + 1.0e-12
    assert result.diagnostics["translation_regularization_applied"] == 1
    assert result.diagnostics["translation_regularization_final_maxstep_scale"] < 1.0
    assert result.diagnostics["translation_regularization_final_max_atom_norm"] <= 0.002 + 1.0e-12


def test_partitioned_state_roundtrip_and_old_state_defaults():
    history, space, mode = _fixture_history()
    core = _core("shifted_lbfgs_trust", radius=0.01)
    _step(core, history, space, mode)
    payload = core.to_state_dict()
    restored = PartitionedLBFGS.from_state_dict(payload, expected_settings=core.resolved_settings())
    assert restored.resolved_settings() == core.resolved_settings()

    old_payload = copy.deepcopy(payload)
    for key in (
        "regularization",
        "regularization_mu",
        "regularization_radius",
        "regularization_tolerance",
        "regularization_shift_scope",
        "regularization_radius_metric",
    ):
        old_payload["settings"].pop(key, None)
    restored_old = PartitionedLBFGS.from_state_dict(old_payload)
    assert restored_old.regularization == "off"
    assert restored_old.regularization_mu == pytest.approx(1.0)
    assert restored_old.regularization_radius == pytest.approx(0.1)
    assert restored_old.regularization_tolerance == pytest.approx(1.0e-8)


def test_shift_only_composition_does_not_change_skip_or_predictor_config():
    # The new shift+skip benchmark override is intentionally a delta on top of
    # the existing skip composition.  Use a representative already-supported
    # predictor selector only as an opaque pre-existing value: the exact BEST09
    # selector is not present in the supplied RC1 config directory.  Toggling
    # only translation regularization must not rewrite it.
    base = {
        "ourDimer": {
            "translation_optimizer": "partitioned_lbfgs",
            "rotation_optimizer": "lbfgs",
        },
        "ourDimerHistory": {
            "enabled": True,
            "record_sources": "center rotation",
            "pair_sources": "center_center center_rotation",
            "rotation_reuse": True,
            "rotation_history_source": "force_bank",
        },
        "ourDimerLBFGS": {"translation_regularization": "off"},
        "ourModeSchedule": {
            "enabled": True,
            "max_skips": 3,
            "predictor_hook": "translation_secant_rotation_lbfgs",
        },
    }
    shifted = copy.deepcopy(base)
    shifted["ourDimerLBFGS"]["translation_regularization"] = "shifted_lbfgs_trust"

    assert shifted["ourModeSchedule"] == base["ourModeSchedule"]
    assert shifted["ourDimerHistory"] == base["ourDimerHistory"]
    off_options = partitioned_options_from_config(base)
    shifted_options = partitioned_options_from_config(shifted)
    different = {k for k in off_options if off_options[k] != shifted_options[k]}
    assert different == {"regularization"}


def test_shifted_partitioned_reconstruct_policy_survives_mode_refresh_without_history_reset():
    """RC3 integrated shift+skip guard: a refreshed mode reprojects, not erases, history."""
    history, space, mode = _fixture_history()
    core = _core("shifted_lbfgs_trust", radius=0.01)
    before = json.dumps(history.to_state_dict(), sort_keys=True)
    first = _step(core, history, space, mode)
    after_first = json.dumps(history.to_state_dict(), sort_keys=True)

    refreshed = mode.copy().reshape(-1)
    refreshed[0] = 0.995
    refreshed[1] = 0.1
    refreshed = (refreshed / np.linalg.norm(refreshed)).reshape(mode.shape)
    second = _step(core, history, space, refreshed)
    after_second = json.dumps(history.to_state_dict(), sort_keys=True)

    assert before == after_first == after_second
    assert core.projector_policy == "reconstruct"
    assert first.diagnostics["history_floor_state_id"] == ""
    assert second.diagnostics["history_floor_state_id"] == ""
    assert second.diagnostics["projector_reset_count"] == 0
    assert second.diagnostics["mode_hv_additional_pes_calls"] == 0
    assert second.diagnostics["translation_regularization_shift_scope"] == "shared_scalar_mu_across_p_q_blocks"
    assert second.diagnostics["translation_regularization_radius_metric"] == "combined_step_euclidean_norm"
