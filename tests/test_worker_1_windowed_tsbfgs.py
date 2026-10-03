from __future__ import annotations

import numpy as np
import pytest

from saddlemill.dimertools.force_history import CanonicalForceHistory
from saddlemill.dimertools.dimer_factory import create_rfo_translation_runtime
from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace
from saddlemill.dimertools.physical_hessian import (
    PhysicalHessianError,
    PhysicalHessianModel,
    REPRESENTATION_FORCEBANK_WINDOW_COMPACT,
    REPRESENTATION_FORCEBANK_WINDOW_DENSE,
    REPRESENTATION_ONLINE_DENSE,
)
from saddlemill.dimertools.rfo_translation import (
    PARTITION_PHYSICAL_B_EIGEN,
    STEP_RAS,
    build_c2_rfo_input,
    compute_rfo_translation,
)
from saddlemill.dimertools.wave_b_runtime import (
    WaveBRuntimeController,
    validate_wave_b_config,
    wave_b_options_from_config,
)


def _quadratic_history(H: np.ndarray, steps: list[np.ndarray], *, memory_states: int = 64):
    H = np.asarray(H, dtype=float)
    n = H.shape[0]
    assert H.shape == (n, n) and n % 3 == 0
    history = CanonicalForceHistory(memory_states=memory_states, record_sources="center")
    positions = []
    x = np.zeros(n, dtype=float)
    for index in range(len(steps) + 1):
        pos = x.reshape((-1, 3)).copy()
        force = (-H @ x).reshape((-1, 3))
        mask = np.ones_like(pos, dtype=bool)
        history.begin_state(pos, active_dof_mask=mask)
        history.observe_center(
            pos,
            force,
            evaluation="physical_exact",
            force_call_delta=1,
            active_dof_mask=mask,
        )
        positions.append(x.copy())
        if index < len(steps):
            x = x + np.asarray(steps[index], dtype=float)
    return history, positions


def _model(space: ActiveCoordinateSpace, representation: str, *, beta: float = 2.5):
    return PhysicalHessianModel(
        space,
        update_type="ts_bfgs",
        initial_hessian=beta,
        representation=representation,
    )


def _rebuild_pair(history: CanonicalForceHistory, space: ActiveCoordinateSpace, *, cap: int = 0, beta: float = 2.5):
    dense = _model(space, REPRESENTATION_FORCEBANK_WINDOW_DENSE, beta=beta)
    compact = _model(space, REPRESENTATION_FORCEBANK_WINDOW_COMPACT, beta=beta)
    dense.rebuild_from_force_history(history, pair_sources="center_center", max_pairs=cap)
    compact.rebuild_from_force_history(history, pair_sources="center_center", max_pairs=cap)
    return dense, compact


def _qn_ras(model: PhysicalHessianModel, history: CanonicalForceHistory, *, radius: float = 0.12):
    request = build_c2_rfo_input(model, history.current.center)
    return compute_rfo_translation(
        request,
        partition=PARTITION_PHYSICAL_B_EIGEN,
        step_control=STEP_RAS,
        target_order=1,
        physical_root_policy="lowest",
        negative_mode_policy="allow",
        trust_radius=radius,
        translation_step_method="qn_mmf",
    )


def test_dense_compact_apply_spectrum_and_qn_ras_equivalence_random_system():
    rng = np.random.default_rng(20260929)
    n = 6
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    target = q @ np.diag([-4.0, -1.25, 0.8, 1.7, 3.4, 5.2]) @ q.T
    steps = [0.18 * rng.normal(size=n) for _ in range(7)]
    history, _ = _quadratic_history(target, steps)
    space = ActiveCoordinateSpace.all_cartesian(n // 3)
    dense, compact = _rebuild_pair(history, space, cap=0, beta=2.5)

    assert compact.has_dense_matrix is False
    np.testing.assert_allclose(dense.full_eigenvalues(), compact.full_eigenvalues(), rtol=2e-10, atol=2e-11)
    for _ in range(20):
        vector = rng.normal(size=n)
        np.testing.assert_allclose(dense.apply_reduced(vector), compact.apply_reduced(vector), rtol=2e-10, atol=2e-11)

    dense_step = _qn_ras(dense, history, radius=0.12)
    compact_step = _qn_ras(compact, history, radius=0.12)
    assert dense_step.success and compact_step.success
    assert dense_step.algorithm == compact_step.algorithm == "qn_mmf"
    np.testing.assert_allclose(dense_step.reduced_step, compact_step.reduced_step, rtol=2e-9, atol=2e-11)
    assert dense_step.alpha == pytest.approx(compact_step.alpha, rel=2e-9, abs=2e-11)
    assert dense_step.t17_added_pes_calls == compact_step.t17_added_pes_calls == 0


def test_multiple_negative_modes_qn_ascends_only_lowest_and_minimizes_second_negative():
    target = np.diag([-4.0, -2.0, 3.0])
    steps = [
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.3, 0.0]),
        np.array([0.0, 0.0, 0.4]),
    ]
    history, _ = _quadratic_history(target, steps)
    space = ActiveCoordinateSpace.all_cartesian(1)
    dense, compact = _rebuild_pair(history, space, cap=0, beta=2.5)
    np.testing.assert_allclose(dense.matrix, target, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(compact.full_eigenvalues(), np.array([-4.0, -2.0, 3.0]), atol=2e-10)

    result = _qn_ras(compact, history, radius=0.1)
    assert result.success
    assert result.physical_negative_mode_count == 2
    gradient = -history.current.center.forces.reshape(-1)
    step = np.asarray(result.reduced_step)
    # Target-order-1 QN/MMF reverses only the lowest root.  The additional
    # physical negative root is treated with |curvature| as a minimization direction.
    assert gradient[0] * step[0] > 0.0
    assert gradient[1] * step[1] < 0.0
    assert gradient[2] * step[2] < 0.0
    np.testing.assert_allclose(step, _qn_ras(dense, history, radius=0.1).reduced_step, rtol=2e-10, atol=2e-12)


def test_window_eviction_rebuilds_from_newest_pairs_without_hidden_old_curvature():
    target = np.diag([-5.0, -1.0, 2.0, 4.0, 6.0, 8.0])
    steps = [
        np.array([0.2, 0, 0, 0, 0, 0], dtype=float),
        np.array([0, 0.2, 0, 0, 0, 0], dtype=float),
        np.array([0, 0, 0.2, 0, 0, 0], dtype=float),
        np.array([0, 0, 0, 0.2, 0, 0], dtype=float),
        np.array([0, 0, 0, 0, 0.2, 0], dtype=float),
    ]
    full_history, positions = _quadratic_history(target, steps)
    space = ActiveCoordinateSpace.all_cartesian(2)
    dense_full, compact_full = _rebuild_pair(full_history, space, cap=2, beta=2.5)

    # Fresh history containing exactly the last two center-center pairs.
    start = len(positions) - 3
    sliced = CanonicalForceHistory(memory_states=8, record_sources="center")
    for x in positions[start:]:
        pos = x.reshape((-1, 3))
        force = (-target @ x).reshape((-1, 3))
        mask = np.ones_like(pos, dtype=bool)
        sliced.begin_state(pos, active_dof_mask=mask)
        sliced.observe_center(pos, force, evaluation="physical_exact", force_call_delta=1, active_dof_mask=mask)
    dense_fresh, compact_fresh = _rebuild_pair(sliced, space, cap=0, beta=2.5)

    np.testing.assert_allclose(dense_full.matrix, dense_fresh.matrix, rtol=1e-11, atol=1e-12)
    for basis in np.eye(space.active_dof_mask.size):
        np.testing.assert_allclose(compact_full.apply_reduced(basis), compact_fresh.apply_reduced(basis), rtol=1e-11, atol=1e-12)
    assert dense_full.window_diagnostics()["pairs_removed_by_cap"] == 3
    assert compact_full.window_diagnostics()["configured_cap"] == 2


def test_pair_cap_zero_one_and_larger_are_configurable_and_newest():
    target = np.diag([-4.0, -2.0, 1.0, 2.0, 3.0, 4.0])
    rng = np.random.default_rng(11)
    history, _ = _quadratic_history(target, [0.1 * rng.normal(size=6) for _ in range(5)])
    space = ActiveCoordinateSpace.all_cartesian(2)
    diagnostics = {}
    for cap in (0, 1, 3):
        model = _model(space, REPRESENTATION_FORCEBANK_WINDOW_COMPACT)
        model.rebuild_from_force_history(history, pair_sources="center_center", max_pairs=cap)
        diagnostics[cap] = model.window_diagnostics()
    assert diagnostics[0]["valid_ts_bfgs_pair_count_before_cap"] == 5
    assert len(diagnostics[0]["pair_ids_used"]) == 5
    assert len(diagnostics[1]["pair_ids_used"]) == 1
    assert len(diagnostics[3]["pair_ids_used"]) == 3
    assert diagnostics[1]["pair_ids_used"] == diagnostics[0]["pair_ids_used"][-1:]
    assert diagnostics[3]["pair_ids_used"] == diagnostics[0]["pair_ids_used"][-3:]
    assert diagnostics[0]["configured_cap"] == 0
    assert diagnostics[1]["pairs_removed_by_cap"] == 4
    assert diagnostics[3]["pairs_removed_by_cap"] == 2

    options = wave_b_options_from_config({
        "ourPhysicalHessian": {"update": "ts_bfgs", "representation": "forcebank_window_compact"},
        "ourDimerHistory": {"pair_sources": "center_center", "max_pairs": 7},
    })
    assert options["physical_hessian"]["forcebank_max_pairs"] == 7


def test_compact_production_qn_path_never_materializes_or_eigendecomposes_full_matrix(monkeypatch):
    n = 12
    target = np.diag([-5.0, -1.0] + [float(i + 1) for i in range(n - 2)])
    rng = np.random.default_rng(91)
    history, _ = _quadratic_history(target, [0.05 * rng.normal(size=n) for _ in range(6)])
    space = ActiveCoordinateSpace.all_cartesian(n // 3)

    full_eigh_shapes = []
    full_eigvalsh_shapes = []
    orig_eigh = np.linalg.eigh
    orig_eigvalsh = np.linalg.eigvalsh

    def guarded_eigh(a, *args, **kwargs):
        shape = np.asarray(a).shape
        if shape == (n, n):
            full_eigh_shapes.append(shape)
            raise AssertionError("compact path attempted full active-space eigh")
        return orig_eigh(a, *args, **kwargs)

    def guarded_eigvalsh(a, *args, **kwargs):
        shape = np.asarray(a).shape
        if shape == (n, n):
            full_eigvalsh_shapes.append(shape)
            raise AssertionError("compact path attempted full active-space eigvalsh")
        return orig_eigvalsh(a, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigh", guarded_eigh)
    monkeypatch.setattr(np.linalg, "eigvalsh", guarded_eigvalsh)
    compact = _model(space, REPRESENTATION_FORCEBANK_WINDOW_COMPACT)
    compact.rebuild_from_force_history(history, pair_sources="center_center", max_pairs=2)
    with pytest.raises(PhysicalHessianError, match="no_dense_matrix"):
        _ = compact.matrix
    result = _qn_ras(compact, history, radius=0.08)
    assert result.success
    assert not full_eigh_shapes and not full_eigvalsh_shapes
    assert compact.window_diagnostics()["compact_rank"] <= 4


def test_reconstruction_and_qn_add_no_pes_calls_and_resume_identically():
    target = np.diag([-4.0, -1.5, 1.0, 2.0, 4.0, 6.0])
    rng = np.random.default_rng(177)
    history, _ = _quadratic_history(target, [0.08 * rng.normal(size=6) for _ in range(5)])
    calls_before = history.accounting.physical_total_pes_calls
    space = ActiveCoordinateSpace.all_cartesian(2)
    model = _model(space, REPRESENTATION_FORCEBANK_WINDOW_COMPACT)
    model.rebuild_from_force_history(history, pair_sources="center_center", max_pairs=3)
    step_before = _qn_ras(model, history, radius=0.09)
    assert history.accounting.physical_total_pes_calls == calls_before
    assert step_before.t17_added_pes_calls == 0

    restored_history = CanonicalForceHistory.from_state_dict(history.to_state_dict())
    restored_model = PhysicalHessianModel.from_state_dict(model.state_dict())
    rng2 = np.random.default_rng(4)
    for _ in range(5):
        vector = rng2.normal(size=6)
        np.testing.assert_allclose(model.apply_reduced(vector), restored_model.apply_reduced(vector), rtol=0.0, atol=1e-13)
    step_after = _qn_ras(restored_model, restored_history, radius=0.09)
    np.testing.assert_allclose(step_before.reduced_step, step_after.reduced_step, rtol=0.0, atol=2e-13)
    assert step_before.alpha == pytest.approx(step_after.alpha, rel=0.0, abs=2e-13)
    assert restored_history.accounting.physical_total_pes_calls == calls_before


def test_existing_online_dense_default_is_unchanged_by_opt_in_representation_selector():
    space = ActiveCoordinateSpace.all_cartesian(1)
    default = PhysicalHessianModel(space, update_type="ts_bfgs", initial_hessian=2.0)
    explicit = PhysicalHessianModel(
        space,
        update_type="ts_bfgs",
        initial_hessian=2.0,
        representation=REPRESENTATION_ONLINE_DENSE,
    )
    history = CanonicalForceHistory(memory_states=8, record_sources="center")
    target = np.diag([-3.0, 2.0, 4.0])
    for x in (np.zeros(3), np.array([0.2, -0.1, 0.05]), np.array([0.35, 0.1, -0.05])):
        pos = x.reshape(1, 3)
        force = (-target @ x).reshape(1, 3)
        mask = np.ones_like(pos, dtype=bool)
        history.begin_state(pos, active_dof_mask=mask)
        history.observe_center(pos, force, evaluation="physical_exact", force_call_delta=1, active_dof_mask=mask)
        obs = history.current.center
        a = default.observe_translation(obs)
        b = explicit.observe_translation(obs)
        assert (a.accepted, a.reason) == (b.accepted, b.reason)
        np.testing.assert_allclose(default.matrix, explicit.matrix, rtol=0.0, atol=0.0)
    assert default.representation == explicit.representation == REPRESENTATION_ONLINE_DENSE


def test_compact_config_validation_is_opt_in_and_fails_closed_for_non_qn_translation():
    config = {
        "ourPhysicalHessian": {
            "update": "ts_bfgs",
            "representation": "forcebank_window_compact",
            "probe_updates": "same_center_physical_hvp",
            "probe_batching": "sequential",
        },
        "ourDimerHistory": {"pair_sources": "center_center", "max_pairs": 0},
    }
    validate_wave_b_config(
        config,
        rotation_optimizer="lbfgs",
        translation_optimizer="qn_mmf",
        min_mode_finder="dimer",
        saddle_engine="mmf",
    )
    with pytest.raises(ValueError, match="supported only with translation_optimizer=qn_mmf"):
        validate_wave_b_config(
            config,
            rotation_optimizer="lbfgs",
            translation_optimizer="prfo",
            min_mode_finder="dimer",
            saddle_engine="mmf",
        )


def test_compact_rfo_runtime_requires_qn_mmf_ras_control():
    base = {
        "ourDimer": {"translation_optimizer": "qn_mmf"},
        "ourPhysicalHessian": {"update": "ts_bfgs", "representation": "forcebank_window_compact"},
        "ourModeSchedule": {"parallel_force_damping": "none"},
        "ourRFO": {"partition": "physical_b_eigen", "step_control": "unrestricted"},
    }
    with pytest.raises(ValueError, match="forcebank_window_compact requires .*step_control=ras"):
        create_rfo_translation_runtime(base)
    valid = {**base, "ourRFO": {"partition": "physical_b_eigen", "step_control": "ras", "trust_radius": 0.1}}
    runtime = create_rfo_translation_runtime(valid)
    assert runtime.settings["translation_optimizer"] == "qn_mmf"
    assert runtime.settings["step_control"] == "ras"


def test_window_resume_identity_includes_pair_window_science_but_online_identity_stays_legacy_compatible():
    window_cfg = {
        "ourPhysicalHessian": {"update": "ts_bfgs", "representation": "forcebank_window_compact", "initial_hessian": 2.5},
        "ourDimerHistory": {"pair_sources": "center_center", "max_pairs": 3},
    }
    options = wave_b_options_from_config(window_cfg)
    runtime = WaveBRuntimeController(options)
    state = runtime.to_state_dict()
    changed = wave_b_options_from_config({
        **window_cfg,
        "ourDimerHistory": {"pair_sources": "center_center", "max_pairs": 4},
    })
    with pytest.raises(ValueError, match="resume selector/config mismatch"):
        WaveBRuntimeController.from_state_dict(changed, state)

    online = wave_b_options_from_config({"ourPhysicalHessian": {"update": "ts_bfgs"}})
    online_state = WaveBRuntimeController(online).to_state_dict()
    assert set(online_state["physical_hessian_selector_identity"]) == {"update", "probe_updates", "probe_batching"}
    WaveBRuntimeController.from_state_dict(online, online_state)
