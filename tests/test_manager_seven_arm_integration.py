from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from saddlemill.config_defaults import DEFAULTS
from saddlemill.config_validation import validate_method_config
from saddlemill.dimertools.force_history import CanonicalForceHistory
from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace
from saddlemill.dimertools.partitioned_lbfgs import (
    DIRECT_DIMER_P_STEP_SOURCE,
    LEGACY_P_STEP_SOURCE,
    PartitionedLBFGS,
)
from saddlemill.dimertools.physical_hessian import (
    PhysicalHessianError,
    PhysicalHessianModel,
    REPRESENTATION_FORCEBANK_WINDOW_COMPACT,
    REPRESENTATION_FORCEBANK_WINDOW_DENSE,
)
from saddlemill.dimertools.rfo_translation import (
    PARTITION_PHYSICAL_B_EIGEN,
    STEP_RAS,
    build_c2_rfo_input,
    compute_rfo_translation,
)
from saddlemill.dimertools.mode_predictor import TRANSLATION_SECANT_FORCEBANK_LBFGS


def _history(H: np.ndarray, steps: list[np.ndarray], *, rotation: bool = False):
    H = np.asarray(H, dtype=float)
    n = H.shape[0]
    history = CanonicalForceHistory(
        memory_states=64,
        record_sources="center rotation" if rotation else "center",
    )
    x = np.zeros(n, dtype=float)
    for i in range(len(steps) + 1):
        pos = x.reshape((-1, 3)).copy()
        force = (-H @ x).reshape((-1, 3))
        mask = np.ones_like(pos, dtype=bool)
        history.begin_state(
            pos, active_dof_mask=mask, coordinate_convention="movable_cartesian_dofs"
        )
        history.observe_center(
            pos, force, evaluation="physical_exact", force_call_delta=1,
            active_dof_mask=mask, coordinate_convention="movable_cartesian_dofs",
        )
        if rotation:
            probe = pos.copy()
            probe.reshape(-1)[1] += 1.0e-3
            pforce = (-H @ probe.reshape(-1)).reshape(pos.shape)
            history.observe_probe(
                probe, pforce, source="rotation_trial", family="rotation",
                evaluation="physical_exact", force_call_delta=1,
            )
        if i < len(steps):
            x = x + np.asarray(steps[i], dtype=float)
    return history


def _window_models(H: np.ndarray, steps: list[np.ndarray], cap: int = 3):
    history = _history(H, steps)
    space = ActiveCoordinateSpace.all_cartesian(H.shape[0] // 3)
    dense = PhysicalHessianModel(
        space, update_type="ts_bfgs", initial_hessian=2.5,
        representation=REPRESENTATION_FORCEBANK_WINDOW_DENSE,
    )
    compact = PhysicalHessianModel(
        space, update_type="ts_bfgs", initial_hessian=2.5,
        representation=REPRESENTATION_FORCEBANK_WINDOW_COMPACT,
    )
    for model in (dense, compact):
        model.rebuild_from_force_history(history, pair_sources="center_center", max_pairs=cap)
    return history, dense, compact


def _qn_step(model: PhysicalHessianModel, history: CanonicalForceHistory):
    request = build_c2_rfo_input(model, history.current.center)
    return compute_rfo_translation(
        request,
        partition=PARTITION_PHYSICAL_B_EIGEN,
        step_control=STEP_RAS,
        target_order=1,
        physical_root_policy="lowest",
        negative_mode_policy="allow",
        trust_radius=0.11,
        translation_step_method="qn_mmf",
    )


def test_manager_dense_compact_exact_pair_identity_actions_spectrum_and_step():
    rng = np.random.default_rng(20260929)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    H = q @ np.diag([-4.0, -1.3, 0.7, 1.8, 3.1, 5.0]) @ q.T
    history, dense, compact = _window_models(H, [0.12 * rng.normal(size=6) for _ in range(6)])

    dd = dense.window_diagnostics()
    cd = compact.window_diagnostics()
    assert dd["pair_ids_used"] == cd["pair_ids_used"]
    assert dd["pair_sources_used"] == cd["pair_sources_used"]

    for vec in (np.eye(6)[0], rng.normal(size=6), rng.normal(size=6)):
        np.testing.assert_allclose(
            dense.apply_reduced(vec), compact.apply_reduced(vec), rtol=2e-10, atol=2e-11
        )
    np.testing.assert_allclose(
        dense.full_eigenvalues(), compact.full_eigenvalues(), rtol=2e-10, atol=2e-11
    )
    a, b = _qn_step(dense, history), _qn_step(compact, history)
    assert a.success and b.success
    np.testing.assert_allclose(a.reduced_step, b.reduced_step, rtol=2e-9, atol=2e-11)
    assert a.t17_added_pes_calls == b.t17_added_pes_calls == 0


def test_manager_compact_path_has_no_dense_matrix_and_multi_negative_qn_semantics():
    H = np.diag([-4.0, -2.0, 3.0])
    history, dense, compact = _window_models(
        H,
        [np.array([0.2, 0.0, 0.0]), np.array([0.0, 0.3, 0.0]), np.array([0.0, 0.0, 0.4])],
        cap=0,
    )
    with pytest.raises(PhysicalHessianError, match="no_dense_matrix"):
        _ = compact.matrix
    result = _qn_step(compact, history)
    gradient = -history.current.center.forces.reshape(-1)
    step = result.reduced_step
    assert result.physical_negative_mode_count == 2
    assert gradient[0] * step[0] > 0.0
    assert gradient[1] * step[1] < 0.0
    assert gradient[2] * step[2] < 0.0
    np.testing.assert_allclose(step, _qn_step(dense, history).reduced_step, rtol=2e-10, atol=2e-12)


def _qpd_history():
    H = np.diag([-2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    history = _history(
        H,
        [
            np.array([0.08, 0.05, 0.0, 0.01, 0.0, 0.0]),
            np.array([0.06, 0.04, 0.0, 0.01, 0.01, 0.0]),
        ],
        rotation=True,
    )
    mode = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    history.finalize_current(mode=mode, curvature=-2.0, solver="dimer")
    return history, ActiveCoordinateSpace.all_cartesian(2), mode


def _qpd_core(**kw):
    args = dict(
        pair_sources="center_center center_rotation",
        p_step_source=DIRECT_DIMER_P_STEP_SOURCE,
        q_initial_hessian=11.0,
        q_dynamic_h0=False,
        q_safeguard="powell",
        q_curvature_floor=1e-6,
        curvature_epsilon=1e-12,
        powell_eta=0.2,
        q_memory=30,
        projector_policy="reconstruct",
        step_damping=1.0,
        regularization="off",
    )
    args.update(kw)
    return PartitionedLBFGS(**args)


def test_manager_qpd_is_current_q_only_direct_axial_and_combines_before_cap():
    history, space, mode = _qpd_history()
    before = copy.deepcopy(history.to_state_dict())
    core = _qpd_core()
    result = core.step(
        history=history,
        current_observation=history.current.center,
        mode=mode,
        active_space=space,
        maximum_translation=100.0,
        curvature=history.current.curvature,
    )
    state = core.to_state_dict()
    assert state["runtime"]["last_p_history"] == []
    assert result.diagnostics["p_l_bfgs_history_used"] == 0
    assert result.diagnostics["p_step_source"] == DIRECT_DIMER_P_STEP_SOURCE
    assert result.diagnostics["q_pairs_used"] > 0
    assert result.diagnostics["q_safeguard"] == "powell"
    np.testing.assert_allclose(result.raw_step, result.p_step + result.q_step, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(result.step, result.raw_step, rtol=0.0, atol=1e-14)
    assert history.to_state_dict() == before

    # Reprojecting the same raw ForceBank with a changed current mode must alter Q history.
    first_q = copy.deepcopy(state["runtime"]["last_q_history"])
    changed = np.array([[0.95, 0.31, 0.0], [0.0, 0.0, 0.0]])
    changed /= np.linalg.norm(changed)
    core.step(
        history=history,
        current_observation=history.current.center,
        mode=changed,
        active_space=space,
        maximum_translation=100.0,
        curvature=history.current.curvature,
    )
    assert core.to_state_dict()["runtime"]["last_q_history"] != first_q


def test_manager_legacy_partitioned_control_still_has_p_learner():
    history, space, mode = _qpd_history()
    implicit = PartitionedLBFGS(pair_sources="center_center", q_safeguard="skip", p_safeguard="skip")
    explicit = PartitionedLBFGS(
        pair_sources="center_center", q_safeguard="skip", p_safeguard="skip",
        p_step_source=LEGACY_P_STEP_SOURCE,
    )
    kwargs = dict(
        history=history,
        current_observation=history.current.center,
        mode=mode,
        active_space=space,
        maximum_translation=100.0,
        curvature=history.current.curvature,
    )
    a, b = implicit.step(**kwargs), explicit.step(**kwargs)
    np.testing.assert_allclose(a.step, b.step, rtol=0.0, atol=0.0)
    assert a.diagnostics["p_l_bfgs_history_used"] == 1


def _base_qn_config():
    cfg = copy.deepcopy(DEFAULTS)
    cfg["Main"]["method"] = "Dimer"
    cfg["ourDimer"].update({
        "engine": "mmf",
        "min_mode_finder": "dimer",
        "translation_optimizer": "qn_mmf",
    })
    cfg["ourPhysicalHessian"].update({
        "update": "ts_bfgs",
        "representation": "online_dense",
        "probe_updates": "same_center_physical_hvp",
        "probe_batching": "sequential",
    })
    cfg["ourRFO"].update({
        "partition": "physical_b_eigen",
        "step_control": "ras",
        "target_order": 1,
        "physical_root_policy": "lowest",
        "negative_mode_policy": "allow",
        # Test fixture only. Benchmark fragments intentionally inherit the BEST11 radius.
        "trust_radius": 0.1,
    })
    cfg["ourDimerHistory"]["enabled"] = True
    return cfg


def _seven_configs():
    configs = []
    a1 = _base_qn_config(); a1["ourPhysicalHessian"]["representation"] = "forcebank_window_dense"; a1["ourDimerHistory"]["max_pairs"] = 30; configs.append(a1)
    a2 = _base_qn_config(); a2["ourPhysicalHessian"]["representation"] = "forcebank_window_compact"; a2["ourDimerHistory"]["max_pairs"] = 30; configs.append(a2)

    a3 = copy.deepcopy(DEFAULTS); a3["Main"]["method"] = "Dimer"; a3["ourDimer"].update({"engine":"mmf","min_mode_finder":"dimer","translation_optimizer":"q_lbfgs_dimer_axial"}); a3["ourDimerHistory"].update({"enabled":True,"pair_safeguard":"powell","max_pairs":30}); configs.append(a3)

    a4 = _base_qn_config(); a4["ourModeSchedule"].update({"enabled":True,"skip_entry_gate":"dimer_initial_torque","refresh_policy":"bounded","max_skips":3,"predictor_hook":"none"}); configs.append(a4)
    a5 = _base_qn_config(); a5["ourModeSchedule"].update({"enabled":True,"skip_entry_gate":"dimer_initial_torque","refresh_policy":"bounded","max_skips":3,"predictor_hook":"physical_eigen"}); configs.append(a5)
    a6 = _base_qn_config(); a6["ourModeSchedule"].update({"enabled":True,"skip_entry_gate":"none","refresh_policy":"physical_model_loss","predictor_hook":"physical_eigen"}); configs.append(a6)

    a7 = copy.deepcopy(a3); a7["ourDimerHistory"].update({"record_sources":"center rotation","rotation_max_pairs":30}); a7["ourModeSchedule"].update({"enabled":True,"skip_entry_gate":"dimer_initial_torque","refresh_policy":"bounded","max_skips":3,"predictor_hook":TRANSLATION_SECANT_FORCEBANK_LBFGS}); configs.append(a7)
    return configs


def test_manager_all_seven_compositions_validate_and_arm7_reuses_existing_predictor():
    for index, cfg in enumerate(_seven_configs(), start=1):
        validate_method_config(cfg, normalize_run_jobs=lambda value: value)
        assert cfg["Main"]["method"] == "Dimer", index
    assert _seven_configs()[6]["ourModeSchedule"]["predictor_hook"] == TRANSLATION_SECANT_FORCEBANK_LBFGS


def test_manager_defaults_leave_old_paths_inert_and_source_has_no_arm_labels():
    assert DEFAULTS["ourPhysicalHessian"]["update"] == "off"
    assert DEFAULTS["ourPhysicalHessian"]["representation"] == "online_dense"
    assert DEFAULTS["ourModeSchedule"]["enabled"] is False
    assert DEFAULTS["ourModeSchedule"]["skip_entry_gate"] == "none"
    assert DEFAULTS["ourModeSchedule"]["refresh_policy"] == "bounded"
    assert DEFAULTS["ourDimer"]["translation_optimizer"] == "ase"

    root = Path(__file__).resolve().parents[1] / "saddlemill"
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in root.rglob("*.py"))
    for label in ("W30-DENSE-TSBFGS", "W30-COMPACT-TSBFGS", "QPD-DIMER", "TG-HOLD", "TG-BMODE", "BMODE-ADAPTIVE", "QPD-DIMER-TG-FBROT"):
        assert label not in text
