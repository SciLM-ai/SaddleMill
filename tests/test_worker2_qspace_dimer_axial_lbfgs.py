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
from saddlemill.dimertools.partitioned_lbfgs import (
    DIRECT_DIMER_AXIAL_SELECTOR,
    DIRECT_DIMER_P_STEP_SOURCE,
    LEGACY_P_STEP_SOURCE,
    PartitionedLBFGS,
)


def _quadratic_history(*, curvature=-2.0, pair_sources="center_center", include_rotation=False):
    space = ActiveCoordinateSpace.all_cartesian(2)
    hessian = np.diag([-2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def force(x):
        return -(hessian @ np.asarray(x, dtype=float).reshape(-1)).reshape(2, 3)

    history = CanonicalForceHistory(
        memory_states=10,
        record_sources="center rotation" if include_rotation else "center",
    )
    centers = [
        np.array([[0.10, 0.08, 0.00], [0.02, 0.01, 0.00]]),
        np.array([[0.18, 0.13, 0.00], [0.03, 0.02, 0.00]]),
        np.array([[0.24, 0.17, 0.00], [0.04, 0.03, 0.00]]),
    ]
    for center in centers:
        history.observe_center(
            center,
            force(center),
            active_dof_mask=space.active_dof_mask,
            coordinate_convention=space.convention,
            force_call_delta=0,
        )
        if include_rotation:
            probe = center + np.array([[0.0, 0.003, 0.0], [0.0, 0.0, 0.0]])
            history.observe_probe(
                probe,
                force(probe),
                source="rotation_trial",
                family="rotation",
                force_call_delta=0,
            )
    mode = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    history.finalize_current(mode=mode, curvature=curvature, solver="dimer")
    return history, space, mode, hessian, pair_sources


def _direct_core(**overrides):
    kwargs = dict(
        pair_sources="center_center",
        p_step_source=DIRECT_DIMER_P_STEP_SOURCE,
        q_initial_hessian=11.0,
        q_dynamic_h0=False,
        q_safeguard="skip",
        q_curvature_floor=1.0e-6,
        curvature_epsilon=1.0e-12,
        powell_eta=0.2,
        q_memory=0,
        projector_policy="reconstruct",
        step_damping=1.0,
        regularization="off",
    )
    kwargs.update(overrides)
    return PartitionedLBFGS(**kwargs)


def _step(core, history, space, mode, *, curvature=None, maximum_translation=10.0):
    return core.step(
        history=history,
        current_observation=history.current.center,
        mode=mode,
        active_space=space,
        maximum_translation=maximum_translation,
        curvature=history.current.curvature if curvature is None else curvature,
    )


def _q_project(value, mode):
    flat = np.asarray(value, dtype=float).reshape(-1)
    v = np.asarray(mode, dtype=float).reshape(-1)
    v = v / np.linalg.norm(v)
    return flat - np.dot(v, flat) * v


def test_q_projection_reconstructs_raw_forcebank_secants_in_current_projector():
    history, space, mode, _hessian, _ = _quadratic_history()
    core = _direct_core(q_safeguard="off")
    result = _step(core, history, space, mode)

    rows = core.to_state_dict()["runtime"]["last_q_history"]
    candidates = history.pair_candidates("center_center", physical_only=True)
    assert len(rows) == len(candidates) > 0
    for row, candidate in zip(rows, candidates):
        s = candidate.displacement()
        g0 = -np.asarray(candidate.first.forces, dtype=float)
        g1 = -np.asarray(candidate.second.forces, dtype=float)
        expected_s = _q_project(s, result.oriented_mode)
        expected_y = _q_project(g1 - g0, result.oriented_mode)
        np.testing.assert_allclose(row["s"], expected_s, rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(row["y"], expected_y, rtol=0.0, atol=1.0e-14)


def test_current_projector_rebuild_changes_q_history_without_mutating_raw_forcebank():
    history, space, mode, _hessian, _ = _quadratic_history()
    before = json.dumps(history.to_state_dict(), sort_keys=True)
    core = _direct_core(q_safeguard="off")
    _step(core, history, space, mode)
    first = copy.deepcopy(core.to_state_dict()["runtime"]["last_q_history"])

    changed = mode.reshape(-1).copy()
    changed[0] = 0.95
    changed[1] = 0.31
    changed = (changed / np.linalg.norm(changed)).reshape(mode.shape)
    _step(core, history, space, changed)
    second = core.to_state_dict()["runtime"]["last_q_history"]
    after = json.dumps(history.to_state_dict(), sort_keys=True)

    assert before == after
    assert first != second
    assert core.to_state_dict()["runtime"]["last_p_history"] == []


def test_direct_axial_uses_only_dimer_curvature_and_no_p_lbfgs_history():
    history, space, mode, _hessian, _ = _quadratic_history(curvature=-2.5)
    core = _direct_core(p_initial_hessian=999.0, p_safeguard="powell", p_memory=1)
    result = _step(core, history, space, mode)
    gradient = -np.asarray(history.current.center.forces, dtype=float)
    g_parallel = float(np.dot(result.oriented_mode.reshape(-1), gradient.reshape(-1)))
    expected_scalar = g_parallel / 2.5

    assert result.diagnostics["selector"] == DIRECT_DIMER_AXIAL_SELECTOR
    assert result.diagnostics["p_step_source"] == DIRECT_DIMER_P_STEP_SOURCE
    assert result.diagnostics["p_l_bfgs_history_used"] == 0
    assert result.diagnostics["p_pair_candidates"] == 0
    assert result.diagnostics["p_pairs_used"] == 0
    assert result.p_records == ()
    assert core.to_state_dict()["runtime"]["last_p_history"] == []
    assert float(np.dot(result.p_step.reshape(-1), result.oriented_mode.reshape(-1))) == pytest.approx(expected_scalar)
    assert result.diagnostics["p_step_curvature"] == pytest.approx(-2.5)


def test_block_separation_p_gradient_change_does_not_change_q_step():
    history_a, space, mode, _hessian, _ = _quadratic_history(curvature=-2.0)
    payload = history_a.to_state_dict()
    history_b = CanonicalForceHistory.from_state_dict(payload)

    # Change only the P component of the current physical force.  Rebuild a
    # second history through the public serialization surface so raw Q secants,
    # positions, ordering, and current Qg remain identical.
    state = history_b.current
    old = state.center
    from saddlemill.dimertools.force_history import ForceObservation
    v = mode.reshape(-1)
    force_flat = np.asarray(old.forces, dtype=float).reshape(-1).copy()
    force_flat += 7.0 * v
    replacement = ForceObservation(
        positions=old.positions,
        forces=force_flat.reshape(old.forces.shape),
        source=old.source,
        role=old.role,
        evaluation=old.evaluation,
        purpose=old.purpose,
        state_id=old.state_id,
        serial=old.serial,
        force_call_delta=old.force_call_delta,
        metadata=old.metadata,
        family=old.family,
        geometry_id=old.geometry_id,
        cache_hit=old.cache_hit,
        active_dof_mask=old.active_dof_mask,
        coordinate_convention=old.coordinate_convention,
    )
    state.center = replacement

    a = _step(_direct_core(q_safeguard="off"), history_a, space, mode)
    b = _step(_direct_core(q_safeguard="off"), history_b, space, mode)
    np.testing.assert_allclose(a.q_gradient, b.q_gradient, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(a.q_step, b.q_step, rtol=0.0, atol=1.0e-13)
    assert not np.allclose(a.p_step, b.p_step)


def test_powell_damps_bad_q_secant_with_configured_eta_and_skip_remains_selectable():
    space = ActiveCoordinateSpace.all_cartesian(1)
    mode = np.array([[1.0, 0.0, 0.0]])
    x0 = np.array([[0.0, 0.0, 0.0]])
    x1 = np.array([[0.0, 0.1, 0.0]])
    f0 = np.zeros((1, 3))
    f1 = np.array([[0.0, 0.1, 0.0]])  # y_Q = g1-g0 = -0.1: negative Q curvature
    history = CanonicalForceHistory(memory_states=4, record_sources="center")
    for x, f in ((x0, f0), (x1, f1)):
        history.observe_center(
            x, f,
            active_dof_mask=space.active_dof_mask,
            coordinate_convention=space.convention,
            force_call_delta=0,
        )
    history.finalize_current(mode=mode, curvature=-2.0, solver="dimer")

    powell = _step(
        _direct_core(q_initial_hessian=10.0, q_safeguard="powell", powell_eta=0.35),
        history,
        space,
        mode,
    )
    assert powell.diagnostics["powell_eta"] == pytest.approx(0.35)
    assert powell.diagnostics["q_pair_summary"]["actions"].get("powell_damped", 0) == 1
    assert powell.diagnostics["q_pairs_used"] == 1

    skipped = _step(
        _direct_core(q_initial_hessian=10.0, q_safeguard="skip", q_curvature_floor=1.0e-6),
        history,
        space,
        mode,
    )
    assert skipped.diagnostics["q_pairs_used"] == 0
    assert skipped.diagnostics["q_pair_summary"]["rejection_reasons"].get("below_curvature_floor", 0) == 1


@pytest.mark.parametrize("reported_curvature", [-2.0, 2.0])
def test_axial_sign_uses_negative_absolute_target_curvature(reported_curvature):
    history, space, mode, _hessian, _ = _quadratic_history(curvature=reported_curvature)
    result = _step(_direct_core(), history, space, mode)
    gradient = -np.asarray(history.current.center.forces, dtype=float)
    g_parallel = float(np.dot(result.oriented_mode.reshape(-1), gradient.reshape(-1)))
    axial = float(np.dot(result.p_step.reshape(-1), result.oriented_mode.reshape(-1)))
    assert axial == pytest.approx(g_parallel / abs(reported_curvature))
    assert result.diagnostics["p_step_curvature"] == pytest.approx(-abs(reported_curvature))
    assert g_parallel * axial >= 0.0


def test_pair_cap_is_configurable_and_zero_means_all_without_literal_arm_cap():
    history, space, mode, _hessian, _ = _quadratic_history()
    all_pairs = _step(_direct_core(q_memory=0, q_safeguard="off"), history, space, mode)
    one_pair = _step(_direct_core(q_memory=1, q_safeguard="off"), history, space, mode)
    assert all_pairs.diagnostics["q_pairs_used"] >= 2
    assert all_pairs.diagnostics["q_pairs_removed_by_max_pairs"] == 0
    assert one_pair.diagnostics["q_pairs_used"] == 1
    assert one_pair.diagnostics["q_pairs_removed_by_max_pairs"] == (
        one_pair.diagnostics["q_pairs_admissible_before_max_pairs"] - 1
    )

    cfg = {
        "ourDimerHistory": {
            "enabled": True,
            "record_sources": "center rotation",
            "pair_sources": "center_center center_rotation",
            "pair_safeguard": "powell",
            "powell_eta": 0.31,
            "max_pairs": 17,
        },
        "ourPartitionedLBFGS": {"q_memory": 999, "q_safeguard": "skip"},
    }
    options = partitioned_options_from_config(
        cfg, translation_optimizer="q_lbfgs_dimer_axial"
    )
    assert options["q_memory"] == 17
    assert options["q_safeguard"] == "powell"
    assert options["powell_eta"] == pytest.approx(0.31)
    assert options["p_step_source"] == DIRECT_DIMER_P_STEP_SOURCE


def test_derived_reflected_like_observations_are_not_translation_secants():
    history, space, mode, _hessian, _ = _quadratic_history(include_rotation=False)
    center = history.current.center_positions
    history.observe_probe(
        center + np.array([[0.0, 0.02, 0.0], [0.0, 0.0, 0.0]]),
        np.ones((2, 3)) * 123.0,
        source="rotation_trial",
        family="rotation",
        evaluation="derived_other",
        force_call_delta=0,
    )
    history.finalize_current(mode=mode, curvature=-2.0, solver="dimer")
    result = _step(
        _direct_core(pair_sources="center_rotation", q_safeguard="off"),
        history,
        space,
        mode,
    )
    assert result.diagnostics["q_pair_candidates"] == 0
    assert result.diagnostics["q_pairs_used"] == 0


def test_no_extra_pes_calls_and_direct_state_roundtrip_reproduces_next_step():
    history, space, mode, _hessian, _ = _quadratic_history()
    before = history.accounting.to_state_dict()
    core = _direct_core(q_safeguard="off")
    _step(core, history, space, mode)
    payload = core.to_state_dict()
    restored = PartitionedLBFGS.from_state_dict(
        payload, expected_settings=core.resolved_settings()
    )
    next_a = _step(core, history, space, mode)
    next_b = _step(restored, history, space, mode)
    after = history.accounting.to_state_dict()

    assert before == after
    assert next_a.diagnostics["additional_pes_calls"] == 0
    assert next_a.diagnostics["mode_hv_additional_pes_calls"] == 0
    np.testing.assert_allclose(next_a.step, next_b.step, rtol=0.0, atol=1.0e-14)
    assert restored.p_step_source == DIRECT_DIMER_P_STEP_SOURCE


def test_legacy_partitioned_default_remains_numerically_unchanged():
    history, space, mode, _hessian, _ = _quadratic_history(include_rotation=True)
    legacy = PartitionedLBFGS(
        pair_sources="center_center center_rotation",
        p_initial_hessian=7.0,
        q_initial_hessian=11.0,
        p_safeguard="skip",
        q_safeguard="skip",
    )
    explicit = PartitionedLBFGS(
        pair_sources="center_center center_rotation",
        p_initial_hessian=7.0,
        q_initial_hessian=11.0,
        p_safeguard="skip",
        q_safeguard="skip",
        p_step_source=LEGACY_P_STEP_SOURCE,
    )
    a = _step(legacy, history, space, mode)
    b = _step(explicit, history, space, mode)
    np.testing.assert_allclose(a.step, b.step, rtol=0.0, atol=0.0)
    assert a.diagnostics["selector"] == "partitioned_lbfgs"
    assert a.diagnostics["p_l_bfgs_history_used"] == 1


def test_direct_curvature_missing_nonfinite_or_too_small_fails_explicitly():
    history, space, mode, _hessian, _ = _quadratic_history(curvature=-2.0)
    core = _direct_core(curvature_epsilon=1.0e-4)
    for bad in (None, np.nan, 1.0e-6):
        if bad is None:
            history.current.curvature = None
        with pytest.raises(ValueError, match="Dimer curvature"):
            _step(core, history, space, mode, curvature=bad)
        history.current.curvature = -2.0


def test_scheduler_forcebank_predictor_composes_with_q_dimer_translator():
    source_root = Path(__file__).resolve().parents[1]
    code = r'''
import copy
from saddlemill.config_defaults import DEFAULTS
from saddlemill.config_validation import validate_method_config
from saddlemill.dimertools.dimer_factory import partitioned_options_from_config
cfg = copy.deepcopy(DEFAULTS)
cfg["Main"]["method"] = "Dimer"
cfg["ourDimer"].update({
    "engine": "mmf",
    "min_mode_finder": "dimer",
    "translation_optimizer": "q_lbfgs_dimer_axial",
})
cfg["ourDimerHistory"].update({
    "enabled": True,
    "record_sources": "center rotation",
    "pair_sources": "center_center center_rotation",
    "pair_safeguard": "powell",
    "powell_eta": 0.27,
    "max_pairs": 30,
})
cfg["ourModeSchedule"].update({
    "enabled": True,
    "max_skips": 3,
    "predictor_hook": "translation_secant_forcebank_lbfgs",
})
validate_method_config(cfg, normalize_run_jobs=lambda value: value)
opts = partitioned_options_from_config(cfg, translation_optimizer="q_lbfgs_dimer_axial")
assert opts["q_memory"] == 30
assert opts["q_safeguard"] == "powell"
assert abs(opts["powell_eta"] - 0.27) < 1e-15
assert opts["p_step_source"] == "dimer_curvature"
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_root)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
