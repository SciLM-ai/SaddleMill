import math
from types import SimpleNamespace

import numpy as np
import pytest

import saddlemill.dimertools.wave_b_runtime as wave_b_runtime

from saddlemill.dimertools.dimer_entry_torque import DimerEntryTorqueCaptureMixin
from saddlemill.dimertools.foundation_types import (
    ActiveCoordinateSpace,
    ModePredictionResult,
    TranslationSecantPredictorInput,
)
from saddlemill.dimertools.mode_predictor import (
    PhysicalEigenModePredictor,
    PhysicalModePredictorInput,
    TRANSLATION_SECANT_FORCEBANK_LBFGS,
    TranslationSecantModePredictor,
)
from saddlemill.dimertools.mode_schedule import (
    BoundedModeScheduler,
    ModeScheduleConfig,
    ModeScheduleObservation,
    ModeScheduleRuntimeState,
    PhysicalModelSignal,
    RealSolveEntryDiagnostic,
    mode_identity,
)
from saddlemill.dimertools.physical_hessian import PhysicalHessianModel
from saddlemill.dimertools.riemannian_lbfgs import RotationLBFGSModel, RotationSecant
from saddlemill.dimertools.wave_b_runtime import (
    WaveBRuntimeController,
    validate_wave_b_config,
    wave_b_options_from_config,
)


def space():
    return ActiveCoordinateSpace.all_cartesian(1)


def mode_x():
    return np.array([[1.0, 0.0, 0.0]])


def signal(i, *, negative=1, valid=True, uid=None, geom=None):
    uid = uid or f"state{i}:g{i}"
    geom = geom or f"g{i}"
    return PhysicalModelSignal(
        origin="physical_hessian_model",
        age=i,
        valid=valid,
        negative_mode_count=negative,
        lowest_curvature=(-1.0 if negative else 0.25),
        observation_id=f"c{i}",
        metadata={
            "current_state_uid": uid,
            "current_geometry_id": geom,
            "additional_pes_calls": 0,
        },
    )


def obs(i, *, mode=None, physical=None, final=False, stale=False, ambiguous=False, predictor_enabled=False):
    sp = space()
    mode = sp.normalized(mode_x() if mode is None else mode)
    return ModeScheduleObservation(
        state_id=i,
        state_uid=f"state{i}:g{i}",
        geometry_id=f"g{i}",
        center_observation_id=f"c{i}",
        positions=np.array([[0.02 * i, 0.0, 0.0]]),
        raw_center_force=np.array([[0.10, 0.02, 0.0]]),
        pre_prediction_mode=mode,
        pre_prediction_mode_identity=mode_identity(mode, sp),
        coordinate_space=sp,
        final_force_convergence_candidate=final,
        predictor_enabled=predictor_enabled,
        stale_data=stale,
        degeneracy_or_root_ambiguity=ambiguous,
        physical_model=physical,
    )


def entry_diag(o, torque, fmax=1.0):
    return RealSolveEntryDiagnostic(
        valid=True,
        state_id=o.state_id,
        state_uid=o.state_uid,
        geometry_id=o.geometry_id,
        center_observation_id=o.center_observation_id,
        mode_identity=o.pre_prediction_mode_identity,
        initial_torque_norm=torque,
        f_rot_max=fmax,
        entry_mode_already_converged=bool(torque <= fmax),
        reason="captured_first_existing_rotational_force",
        additional_pes_calls=0,
    )


def test_entry_gate_true_unlocks_bounded_skip():
    scheduler = BoundedModeScheduler(ModeScheduleConfig(max_skips=3, skip_entry_gate="dimer_initial_torque"))
    o0 = obs(0)
    state = scheduler.initialize_after_real_solve(
        o0, solved_mode=mode_x(), fresh_physical_validation=True,
        entry_diagnostic=entry_diag(o0, 0.9, 1.0),
    )
    assert state.skip_sequence_eligible is True
    ev = scheduler.evaluate(state, obs(1))
    assert ev.decision.skipped_scheduled_solve is True


def test_entry_gate_false_uses_initial_not_final_success():
    scheduler = BoundedModeScheduler(ModeScheduleConfig(max_skips=3, skip_entry_gate="dimer_initial_torque"))
    o0 = obs(0)
    # The real solve may later converge below f_rot_max; the persisted ENTRY value is 1.4 > 1.0.
    state = scheduler.initialize_after_real_solve(
        o0, solved_mode=mode_x(), fresh_physical_validation=True,
        entry_diagnostic=entry_diag(o0, 1.4, 1.0),
    )
    assert state.skip_sequence_eligible is False
    ev = scheduler.evaluate(state, obs(1))
    assert ev.decision.require_real_solve is True
    assert ev.decision.reason == "entry_gate_not_satisfied"


def test_entry_gate_unavailable_or_wrong_identity_fails_closed():
    scheduler = BoundedModeScheduler(
        ModeScheduleConfig(max_skips=3, skip_entry_gate="dimer_initial_torque")
    )
    o0 = obs(0)
    missing = scheduler.initialize_after_real_solve(
        o0, solved_mode=mode_x(), fresh_physical_validation=True,
        entry_diagnostic=None,
    )
    assert missing.skip_sequence_eligible is False
    assert scheduler.evaluate(missing, obs(1)).decision.reason == "entry_gate_not_satisfied"

    wrong_identity = RealSolveEntryDiagnostic(
        valid=True,
        state_id=o0.state_id,
        state_uid=o0.state_uid,
        geometry_id="wrong-geometry",
        center_observation_id=o0.center_observation_id,
        mode_identity=o0.pre_prediction_mode_identity,
        initial_torque_norm=0.2,
        f_rot_max=1.0,
        entry_mode_already_converged=True,
        reason="captured_first_existing_rotational_force",
    )
    state = scheduler.initialize_after_real_solve(
        o0, solved_mode=mode_x(), fresh_physical_validation=True,
        entry_diagnostic=wrong_identity,
    )
    assert state.skip_sequence_eligible is False


def test_entry_torque_capture_adds_no_force_call():
    class Control:
        def get_parameter(self, name):
            assert name == "f_rot_max"
            return 1.0

    class Base:
        def __init__(self):
            self.calls = 0
            self.control = Control()
            self.eigenmode = mode_x()
        def get_eigenmode(self):
            return self.eigenmode
        def get_rotational_force(self):
            self.calls += 1
            return np.array([[0.5, 0.0, 0.0]])

    class Capturing(DimerEntryTorqueCaptureMixin, Base):
        pass

    item = Capturing()
    force = item.get_rotational_force()
    assert item.calls == 1
    assert np.allclose(force, [[0.5, 0.0, 0.0]])
    assert item.sm_entry_torque_capture["entry_mode_already_converged"] is True
    assert item.sm_entry_torque_capture["additional_pes_calls"] == 0


def test_tg_hold_respects_configurable_max_skips_and_holds_mode():
    scheduler = BoundedModeScheduler(ModeScheduleConfig(max_skips=2, skip_entry_gate="dimer_initial_torque"))
    o0 = obs(0)
    state = scheduler.initialize_after_real_solve(o0, solved_mode=mode_x(), fresh_physical_validation=True,
                                                  entry_diagnostic=entry_diag(o0, 0.5))
    held0 = state.held_mode_identity
    for i in (1, 2):
        ev = scheduler.evaluate(state, obs(i))
        assert ev.decision.skipped_scheduled_solve
        state, use = scheduler.complete_skip(ev)
        assert mode_identity(use.held_mode, space()) == held0
    ev = scheduler.evaluate(state, obs(3))
    assert ev.decision.require_real_solve
    assert ev.decision.reason == "max_skips_exhausted"


def _wrapped_prediction(raw, old_o, new_o):
    return ModePredictionResult(
        mode=raw.mode, status=raw.status,
        original_signed_alignment=None, absolute_alignment=None, reversed_pair=False,
        raw_tangent=raw.raw_tangent,
        requested_angle_radians=raw.requested_angle_radians,
        accepted_angle_radians=raw.accepted_angle_radians,
        capped=raw.capped,
        old_state_id=old_o.state_id, new_state_id=new_o.state_id,
        old_state_uid=old_o.state_uid, new_state_uid=new_o.state_uid,
        old_geometry_id=old_o.geometry_id, new_geometry_id=new_o.geometry_id,
        angular_step_scale=1.0, max_angle_radians=math.pi / 2.0,
        alignment_tolerance=1e-8, displacement_tolerance=1e-12, tangent_tolerance=1e-14,
        origin=raw.origin, metadata={"prediction_cost_pes_calls": 0},
    )


def test_tg_bmode_uses_physical_lowest_eigenvector_and_bounded_limit():
    sp = space()
    current = sp.normalized(np.array([[0.8, 0.6, 0.0]]))
    o0 = obs(0, mode=current)
    scheduler = BoundedModeScheduler(ModeScheduleConfig(max_skips=1, skip_entry_gate="dimer_initial_torque"))
    state = scheduler.initialize_after_real_solve(o0, solved_mode=current, fresh_physical_validation=True,
                                                  entry_diagnostic=entry_diag(o0, 0.2))
    o1 = obs(1, mode=current, predictor_enabled=True)
    ev = scheduler.evaluate(state, o1)
    model = PhysicalHessianModel(sp, initial_matrix=np.diag([-2.0, 1.0, 3.0]))
    model._current_state_uid = o1.state_uid
    model._current_geometry_id = o1.geometry_id
    request = PhysicalModePredictorInput(
        state_id=o1.state_id, state_uid=o1.state_uid, geometry_id=o1.geometry_id,
        current_mode=current, coordinate_space=sp, model=model,
        max_angle_radians=math.pi / 2.0, degeneracy_tolerance=1e-10,
    )
    raw = PhysicalEigenModePredictor().predict(request)
    assert raw.status == "predicted"
    state, use = scheduler.complete_skip(ev, prediction_result=_wrapped_prediction(raw, o0, o1))
    assert abs(float(use.held_mode[0, 0])) > abs(float(current[0, 0]))
    ev2 = scheduler.evaluate(state, obs(2, mode=use.held_mode))
    assert ev2.decision.reason == "max_skips_exhausted"


def test_physical_model_loss_policy_skips_beyond_three_then_refreshes_on_root_loss():
    scheduler = BoundedModeScheduler(ModeScheduleConfig(max_skips=3, refresh_policy="physical_model_loss"))
    o0 = obs(0, physical=signal(0))
    state = scheduler.initialize_after_real_solve(o0, solved_mode=mode_x(), fresh_physical_validation=True,
                                                  physical_model=o0.physical_model)
    for i in range(1, 6):
        oi = obs(i, physical=signal(i))
        ev = scheduler.evaluate(state, oi)
        assert ev.decision.skipped_scheduled_solve, (i, ev.decision.reason)
        state, _ = scheduler.complete_skip(ev)
    assert state.core.skips_since_real_solve == 5
    lost = obs(6, physical=signal(6, negative=0))
    ev = scheduler.evaluate(state, lost)
    assert ev.decision.require_real_solve
    assert ev.decision.reason == "physical_model_inertia_or_negative_mode_loss"


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"stale": True}, "stale_data"),
        ({"ambiguous": True}, "degeneracy_or_root_ambiguity"),
        ({"final": True}, "mandatory_final_physical_mode_validation"),
    ],
)
def test_safety_conditions_remain_fail_closed(kwargs, expected):
    scheduler = BoundedModeScheduler(ModeScheduleConfig(max_skips=3, refresh_policy="physical_model_loss", stale_data_trigger=True))
    o0 = obs(0, physical=signal(0))
    state = scheduler.initialize_after_real_solve(o0, solved_mode=mode_x(), fresh_physical_validation=True,
                                                  physical_model=o0.physical_model)
    o1 = obs(1, physical=signal(1), **kwargs)
    ev = scheduler.evaluate(state, o1)
    assert ev.decision.require_real_solve
    assert ev.decision.reason == expected


def test_adaptive_model_identity_mismatch_fails_closed():
    scheduler = BoundedModeScheduler(ModeScheduleConfig(max_skips=3, refresh_policy="physical_model_loss"))
    o0 = obs(0, physical=signal(0))
    state = scheduler.initialize_after_real_solve(o0, solved_mode=mode_x(), fresh_physical_validation=True,
                                                  physical_model=o0.physical_model)
    bad = signal(1, uid="wrong", geom="g1")
    ev = scheduler.evaluate(state, obs(1, physical=bad))
    assert ev.decision.require_real_solve
    assert ev.decision.reason == "physical_model_unavailable_or_stale"


def test_existing_bounded_schedule_is_unchanged_when_new_selectors_off():
    cfg = ModeScheduleConfig(max_skips=2)
    assert cfg.skip_entry_gate == "none"
    assert cfg.refresh_policy == "bounded"
    scheduler = BoundedModeScheduler(cfg)
    state = scheduler.initialize_after_real_solve(obs(0), solved_mode=mode_x(), fresh_physical_validation=True)
    for i in (1, 2):
        ev = scheduler.evaluate(state, obs(i))
        assert ev.decision.skipped_scheduled_solve
        state, _ = scheduler.complete_skip(ev)
    assert scheduler.evaluate(state, obs(3)).decision.reason == "max_skips_exhausted"


def test_forcebank_rotational_lbfgs_predictor_composes_with_torque_gate():
    sp = space()
    current = mode_x()
    pair = RotationSecant(
        anchor_mode=current,
        s=np.array([[0.0, 0.1, 0.0]]),
        y=np.array([[0.0, 0.1, 0.0]]),
        state_id=0, serial=0, source="forcebank", geometry="double_projection",
    )
    model = RotationLBFGSModel(initial_hessian=1.0, dynamic_h0=False, curvature_guard="legacy_skip")
    predictor = TranslationSecantModePredictor(
        TRANSLATION_SECANT_FORCEBANK_LBFGS,
        forcebank_rotation_model=model,
        forcebank_rotation_pairs=(pair,),
        forcebank_build_metrics={"pairs_built": 1},
    )
    request = TranslationSecantPredictorInput(
        old_state_id=0, new_state_id=1,
        old_state_uid="state0:g0", new_state_uid="state1:g1",
        old_geometry_id="g0", new_geometry_id="g1",
        old_positions=np.zeros((1, 3)), new_positions=np.array([[0.1, 0.0, 0.0]]),
        old_gradient=np.zeros((1, 3)), new_gradient=np.array([[0.0, -0.1, 0.0]]),
        current_mode=current, coordinate_space=sp, angular_step_scale=1.0,
        max_angle_radians=0.2,
    )
    pred = predictor.predict(request)
    assert pred.metadata["prediction_cost_pes_calls"] == 0
    assert pred.status == "predicted"

    scheduler = BoundedModeScheduler(ModeScheduleConfig(max_skips=3, skip_entry_gate="dimer_initial_torque"))
    o0 = obs(0)
    state = scheduler.initialize_after_real_solve(o0, solved_mode=current, fresh_physical_validation=True,
                                                  entry_diagnostic=entry_diag(o0, 0.2))
    ev = scheduler.evaluate(state, obs(1, predictor_enabled=True))
    state, use = scheduler.complete_skip(ev, prediction_result=pred)
    assert use.disposition == "skip_predict_held_mode"
    assert state.core.skips_since_real_solve == 1


def test_state_serialization_preserves_gate_refresh_and_eligibility():
    scheduler = BoundedModeScheduler(ModeScheduleConfig(max_skips=3, skip_entry_gate="dimer_initial_torque",
                                                         refresh_policy="physical_model_loss"))
    o0 = obs(0, physical=signal(0))
    state = scheduler.initialize_after_real_solve(o0, solved_mode=mode_x(), fresh_physical_validation=True,
                                                  physical_model=o0.physical_model,
                                                  entry_diagnostic=entry_diag(o0, 0.2))
    restored = ModeScheduleRuntimeState.from_state_dict(state.to_state_dict())
    assert restored.skip_entry_gate == "dimer_initial_torque"
    assert restored.refresh_policy == "physical_model_loss"
    assert restored.skip_sequence_eligible is True
    assert restored.last_entry_diagnostic is not None
    assert restored.last_entry_diagnostic.initial_torque_norm == pytest.approx(0.2)
    # Both required predictor families are intentionally stateless wrappers;
    # their dynamic inputs (physical B / ForceBank history) are owned and
    # serialized by their existing model/history layers rather than duplicated.
    assert PhysicalEigenModePredictor().state_dict()["owns_dynamic_state"] is False
    fb_model = RotationLBFGSModel(initial_hessian=1.0, dynamic_h0=False, curvature_guard="legacy_skip")
    fb_predictor = TranslationSecantModePredictor(
        TRANSLATION_SECANT_FORCEBANK_LBFGS,
        forcebank_rotation_model=fb_model,
    )
    predictor_state = fb_predictor.state_dict()
    assert predictor_state["selector"] == TRANSLATION_SECANT_FORCEBANK_LBFGS
    assert predictor_state["owns_dynamic_state"] is False


def test_legacy_scheduler_state_restores_with_new_selectors_disabled():
    scheduler = BoundedModeScheduler(ModeScheduleConfig(max_skips=2))
    state = scheduler.initialize_after_real_solve(
        obs(0), solved_mode=mode_x(), fresh_physical_validation=True
    )
    payload = state.to_state_dict()
    payload["schema_version"] = 1
    payload.pop("skip_entry_gate", None)
    payload.pop("refresh_policy", None)
    payload.pop("skip_sequence_eligible", None)
    payload.pop("last_entry_diagnostic", None)
    restored = ModeScheduleRuntimeState.from_state_dict(payload)
    assert restored.skip_entry_gate == "none"
    assert restored.refresh_policy == "bounded"
    assert restored.skip_sequence_eligible is True
    assert restored.last_entry_diagnostic is None


def test_physical_model_signal_exposes_current_identity_age_without_pes_work():
    model = PhysicalHessianModel(space(), initial_matrix=np.diag([-1.0, 1.0, 2.0]))
    model._current_state_uid = "state4:g4"
    model._current_geometry_id = "g4"
    model._previous_observation_id = "c4"
    controller = WaveBRuntimeController({"mode_schedule": {"enabled": False}})
    controller.physical_hessian = model
    out = controller.physical_model_signal()
    assert out is not None
    assert out.metadata["current_state_uid"] == "state4:g4"
    assert out.metadata["current_geometry_id"] == "g4"
    assert out.metadata["previous_observation_id"] == "c4"
    assert out.metadata["model_age"] == model.model_age
    assert out.metadata["additional_pes_calls"] == 0


def test_near_degenerate_physical_predictor_requests_fail_closed_refresh():
    sp = space()
    current = sp.normalized(np.array([[1.0, 1.0, 0.0]]))
    model = PhysicalHessianModel(sp, initial_matrix=np.diag([-1.0, -1.0 + 1.0e-12, 2.0]))
    model._current_state_uid = "state1:g1"
    model._current_geometry_id = "g1"
    request = PhysicalModePredictorInput(
        state_id=1, state_uid="state1:g1", geometry_id="g1",
        current_mode=current, coordinate_space=sp, model=model,
        max_angle_radians=math.pi / 2.0, degeneracy_tolerance=1.0e-8,
    )
    pred = PhysicalEigenModePredictor().predict(request)
    assert pred.status == "predicted_near_degenerate"
    refresh, reason = WaveBRuntimeController._prediction_requires_fail_closed_refresh(
        "physical_eigen", pred, None
    )
    assert refresh is True
    assert reason == "predicted_near_degenerate"


def _unsafe_wrapped_physical_prediction(old_o, new_o):
    return ModePredictionResult(
        mode=new_o.pre_prediction_mode,
        status="predicted_near_degenerate",
        original_signed_alignment=None,
        absolute_alignment=None,
        reversed_pair=False,
        raw_tangent=np.zeros_like(new_o.pre_prediction_mode),
        requested_angle_radians=0.0,
        accepted_angle_radians=0.0,
        capped=False,
        old_state_id=old_o.state_id,
        new_state_id=new_o.state_id,
        old_state_uid=old_o.state_uid,
        new_state_uid=new_o.state_uid,
        old_geometry_id=old_o.geometry_id,
        new_geometry_id=new_o.geometry_id,
        angular_step_scale=1.0,
        max_angle_radians=math.pi / 2.0,
        alignment_tolerance=1.0e-8,
        displacement_tolerance=1.0e-12,
        tangent_tolerance=1.0e-14,
        origin="physical_hessian_B_eigenvector",
        metadata={"prediction_cost_pes_calls": 0},
    )


def test_new_policy_predictor_safety_refresh_clears_pending_skip(monkeypatch):
    options = {
        "mode_schedule": {
            "enabled": True,
            "max_skips": 3,
            "skip_entry_gate": "dimer_initial_torque",
            "refresh_policy": "bounded",
            "parallel_force_increase_trigger": False,
            "predictor_hook": "physical_eigen",
        }
    }
    controller = WaveBRuntimeController(options)
    o0 = obs(0)
    controller.schedule_state = controller.scheduler.initialize_after_real_solve(
        o0, solved_mode=mode_x(), fresh_physical_validation=True,
        entry_diagnostic=entry_diag(o0, 0.2),
    )
    o1 = obs(1, predictor_enabled=True)
    pred = _unsafe_wrapped_physical_prediction(o0, o1)

    controller._schedule_observation = lambda owner: o1

    def fake_prediction(ctrl, owner, evaluation):
        return pred, ctrl.scheduler.resolve_prediction(evaluation, pred)

    monkeypatch.setattr(
        wave_b_runtime, "apply_t10_prediction_after_schedule_decision", fake_prediction
    )
    owner = SimpleNamespace(eigenmodes=[mode_x().copy()])
    assert controller.before_mode_solve(owner) is False
    assert controller.pending_schedule_evaluation is None
    assert controller.last_schedule_metadata["predictor_safety_refresh"]["refresh_required"] is True


def test_legacy_bounded_physical_predictor_behavior_unchanged_when_new_selectors_off(monkeypatch):
    options = {
        "mode_schedule": {
            "enabled": True,
            "max_skips": 3,
            "skip_entry_gate": "none",
            "refresh_policy": "bounded",
            "parallel_force_increase_trigger": False,
            "predictor_hook": "physical_eigen",
        }
    }
    controller = WaveBRuntimeController(options)
    o0 = obs(0)
    controller.schedule_state = controller.scheduler.initialize_after_real_solve(
        o0, solved_mode=mode_x(), fresh_physical_validation=True,
    )
    o1 = obs(1, predictor_enabled=True)
    pred = _unsafe_wrapped_physical_prediction(o0, o1)
    controller._schedule_observation = lambda owner: o1

    def fake_prediction(ctrl, owner, evaluation):
        return pred, ctrl.scheduler.resolve_prediction(evaluation, pred)

    monkeypatch.setattr(
        wave_b_runtime, "apply_t10_prediction_after_schedule_decision", fake_prediction
    )
    owner = SimpleNamespace(eigenmodes=[mode_x().copy()])
    assert controller.before_mode_solve(owner) is True
    assert controller.pending_schedule_evaluation is None
    assert controller.schedule_state.core.skips_since_real_solve == 1


def _base_config(**schedule):
    return {
        "ourPhysicalHessian": {"update": "off"},
        "ourModeSchedule": {"enabled": True, **schedule},
        "ourDimerHistory": {"record_sources": "center rotation"},
    }


def test_config_validation_accepts_compositions_and_rejects_only_real_incompatibilities():
    forcebank = _base_config(
        skip_entry_gate="dimer_initial_torque", refresh_policy="bounded",
        predictor_hook="translation_secant_forcebank_lbfgs", max_skips=7,
    )
    out = validate_wave_b_config(forcebank, rotation_optimizer="ase", translation_optimizer="ase",
                                 min_mode_finder="dimer", saddle_engine="mmf")
    assert out["mode_schedule"]["max_skips"] == 7

    tg_bmode = _base_config(
        skip_entry_gate="dimer_initial_torque", refresh_policy="bounded",
        predictor_hook="physical_eigen", max_skips=5,
    )
    tg_bmode["ourPhysicalHessian"] = {"update": "bfgs"}
    out = validate_wave_b_config(
        tg_bmode, rotation_optimizer="ase", translation_optimizer="ase",
        min_mode_finder="dimer", saddle_engine="mmf",
    )
    assert out["mode_schedule"]["predictor_hook"] == "physical_eigen"

    adaptive_hold = _base_config(refresh_policy="physical_model_loss", predictor_hook="none")
    adaptive_hold["ourPhysicalHessian"] = {"update": "bfgs"}
    out = validate_wave_b_config(
        adaptive_hold, rotation_optimizer="ase", translation_optimizer="ase",
        min_mode_finder="dimer", saddle_engine="mmf",
    )
    assert out["mode_schedule"]["refresh_policy"] == "physical_model_loss"

    with pytest.raises(ValueError, match="skip_entry_gate=dimer_initial_torque requires min_mode_finder=dimer"):
        validate_wave_b_config(_base_config(skip_entry_gate="dimer_initial_torque"), rotation_optimizer="ase",
                               translation_optimizer="ase", min_mode_finder="lanczos", saddle_engine="mmf")
    with pytest.raises(ValueError, match="refresh_policy=physical_model_loss requires a non-off physical Hessian B model"):
        validate_wave_b_config(_base_config(refresh_policy="physical_model_loss"), rotation_optimizer="ase",
                               translation_optimizer="ase", min_mode_finder="dimer", saddle_engine="mmf")

    defaults = wave_b_options_from_config({"ourModeSchedule": {}})["mode_schedule"]
    assert defaults["skip_entry_gate"] == "none"
    assert defaults["refresh_policy"] == "bounded"
