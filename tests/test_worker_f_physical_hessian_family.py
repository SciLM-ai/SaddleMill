from __future__ import annotations

import configparser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from saddlemill.dimertools.force_history import CanonicalForceHistory
from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace, WorkPurpose
from saddlemill.dimertools.physical_hessian import PhysicalHessianModel
from saddlemill.dimertools.wave_b_runtime import (
    WaveBRuntimeController,
    validate_wave_b_config,
)


def _center(history: CanonicalForceHistory, positions: np.ndarray, forces: np.ndarray):
    history.begin_state(positions, active_dof_mask=np.ones_like(positions, dtype=bool))
    history.observe_center(positions, forces, evaluation="physical_exact", force_call_delta=1)
    return history.current.center


def _rotation_probe(
    history: CanonicalForceHistory,
    *,
    center_positions: np.ndarray,
    center_forces: np.ndarray,
    matrix: np.ndarray,
    direction: np.ndarray,
    stencil_id: str,
    h: float = 1.0e-3,
    evaluation: str = "physical_exact",
):
    q = np.asarray(direction, dtype=float)
    q = q / np.linalg.norm(q)
    pos = center_positions + h * q
    # F(x+hq) = F(x) - h Hq because F=-grad.
    force = center_forces - h * (matrix @ q.reshape(-1)).reshape(q.shape)
    return history.observe_probe(
        pos,
        force,
        source=(
            "rotation_dimer_endpoint"
            if evaluation == "physical_exact"
            else "rotation_extrapolated_accepted"
        ),
        family="rotation",
        evaluation=evaluation,
        force_call_delta=0 if evaluation != "physical_exact" else 1,
        metadata={"stencil_scheme": "one_sided"},
        direction=q,
        direction_kind="dimer_mode",
        stencil_id=stencil_id,
        dimer_side=1,
        offset=h,
        purpose=WorkPurpose.ALGORITHM.value,
    )


def _runtime_for(model: PhysicalHessianModel, history: CanonicalForceHistory):
    runtime = WaveBRuntimeController(
        {
            "physical_hessian": {
                "update": "ts_bfgs",
                "probe_updates": "same_center_physical_hvp",
                "probe_batching": "same_center_block",
            },
            "mode_schedule": {"enabled": False},
        }
    )
    runtime._ensure_physical_hessian = lambda owner, center: model
    owner = SimpleNamespace(min_mode_finder="dimer", canonical_force_history=history)
    return runtime, owner


def _read_override(name: str):
    root = Path(__file__).resolve().parents[2]
    parser = configparser.ConfigParser()
    with open(root / "configs" / name, encoding="utf-8") as handle:
        parser.read_file(handle)
    return {section: dict(parser[section]) for section in parser.sections()}


def _flatten(config):
    return {(section, key): value for section, values in config.items() for key, value in values.items()}


def test_ordinary_physical_bfgs_keeps_historical_signed_curvature_admission():
    space = ActiveCoordinateSpace.all_cartesian(2)
    model = PhysicalHessianModel(space, update_type="bfgs", initial_matrix=np.eye(6))
    p0 = np.zeros((2, 3))
    g0 = np.zeros(6)
    history = CanonicalForceHistory(record_sources="center")
    c0 = _center(history, p0, (-g0).reshape((2, 3)))
    first = model.observe_translation(c0)
    assert not first.accepted and first.reason == "seed_reference"

    p1 = p0.copy().reshape(-1)
    p1[0] = 1.0
    g1 = g0.copy()
    g1[0] = -2.0  # s.y = -2: explicitly admissible under the historical rule.
    c1 = _center(history, p1.reshape((2, 3)), (-g1).reshape((2, 3)))
    second = model.observe_translation(c1)
    assert second.accepted, second.reason
    assert np.linalg.eigvalsh(model.matrix)[0] < 0.0
    assert second.negative_eigenvalue_count >= 1

    # A zero-y center secant is rejected, but the new center becomes the reference.
    p2 = p1.copy()
    p2[1] = 1.0
    c2 = _center(history, p2.reshape((2, 3)), (-g1).reshape((2, 3)))
    third = model.observe_translation(c2)
    assert not third.accepted
    assert third.reason == "small_s_dot_y"
    assert model.state_dict()["previous_observation_id"] == c2.observation_id


def test_same_center_block_selector_is_narrow_and_fail_closed():
    base = {
        "ourPhysicalHessian": {
            "update": "ts_bfgs",
            "probe_updates": "same_center_physical_hvp",
            "probe_batching": "same_center_block",
        },
        "ourModeSchedule": {"parallel_force_damping": "none"},
    }
    validate_wave_b_config(
        base,
        rotation_optimizer="lbfgs",
        translation_optimizer="prfo",
        min_mode_finder="dimer",
        saddle_engine="mmf",
    )

    wrong_update = {**base, "ourPhysicalHessian": dict(base["ourPhysicalHessian"], update="bfgs")}
    with pytest.raises(ValueError, match="requires .*update=ts_bfgs"):
        validate_wave_b_config(
            wrong_update,
            rotation_optimizer="lbfgs",
            translation_optimizer="prfo",
            min_mode_finder="dimer",
            saddle_engine="mmf",
        )

    with pytest.raises(ValueError, match="validated for min_mode_finder=dimer"):
        validate_wave_b_config(
            base,
            rotation_optimizer="ase",
            translation_optimizer="prfo",
            min_mode_finder="lanczos",
            saddle_engine="mmf",
        )


def test_same_center_rotation_block_uses_only_already_paid_physical_stencils_once():
    positions = np.zeros((2, 3))
    center_forces = np.zeros((2, 3))
    A = np.diag([-2.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    history = CanonicalForceHistory(record_sources="center rotation")
    center = _center(history, positions, center_forces)

    q1 = np.zeros((2, 3)); q1.reshape(-1)[0] = 1.0
    q2 = np.zeros((2, 3)); q2.reshape(-1)[1] = 1.0
    obs1 = _rotation_probe(
        history, center_positions=positions, center_forces=center_forces,
        matrix=A, direction=q1, stencil_id="rot-1",
    )
    obs2 = _rotation_probe(
        history, center_positions=positions, center_forces=center_forces,
        matrix=A, direction=q2, stencil_id="rot-2",
    )
    # A derived Fourier-accepted endpoint is retained as provenance but must not
    # be relabeled/admitted as a physical HVP.
    obs_derived = _rotation_probe(
        history, center_positions=positions, center_forces=center_forces,
        matrix=A, direction=np.array([[0, 0, 1], [0, 0, 0]], dtype=float),
        stencil_id="rot-derived", evaluation="derived_extrapolated",
    )

    space = ActiveCoordinateSpace.all_cartesian(2)
    model = PhysicalHessianModel(space, update_type="ts_bfgs", initial_matrix=np.eye(6))
    model.observe_translation(center)
    runtime, owner = _runtime_for(model, history)
    age_before = model.model_age
    # The live probe hook must not apply sequential updates when block batching
    # is selected; the already-recorded stencils are consumed only at commit.
    runtime.observe_probe(owner, history, obs1)
    runtime.observe_probe(owner, history, obs2)
    runtime.observe_probe(owner, history, obs_derived)
    assert model.model_age == age_before
    runtime.commit_same_center_probe_block(owner)

    record = runtime.last_physical_hessian_block_update
    assert record is not None
    assert record["probe_batching"] == "same_center_block"
    assert record["solver"] == "dimer"
    assert record["additional_pes_calls"] == 0
    assert record["incremental_admission_suppressed"] is True
    assert record["admission_basis"] == "complete_same_center_dimer_rotation_physical_hvp_stencils"
    assert set(record["admitted_stencil_ids"]) == {"rot-1", "rot-2"}
    assert record["input_hvp_count"] == 2
    assert record["accepted"] is True
    assert model.model_age == age_before + 1
    assert {item["stencil_id"] for item in record["excluded_stencils"]} == {"rot-derived"}
    assert record["excluded_stencils"][0]["reason"] == "endpoint_not_admissible"

    matrix_after = model.matrix.copy()
    age_after = model.model_age
    serial_after = runtime.physical_hessian_block_commit_serial
    runtime.commit_same_center_probe_block(owner)
    assert model.model_age == age_after
    assert runtime.physical_hessian_block_commit_serial == serial_after
    assert np.allclose(model.matrix, matrix_after)

    state = runtime.to_state_dict()
    resumed = WaveBRuntimeController.from_state_dict(runtime.options, state)
    assert resumed.physical_hessian_processed_stencil_ids == {"rot-1", "rot-2", "rot-derived"}


def test_matched_physical_ts_bfgs_configs_enable_identical_probe_reuse():
    qn = _flatten(_read_override("W5_BEST11_FORCEBANK_TSBFGS_BMODE_QN_MMF_WORKER_A.override.ini"))
    prfo = _flatten(_read_override("W5_BEST12_FORCEBANK_TSBFGS_BMODE_PRFO_WORKER_A.override.ini"))
    assert qn[("ourPhysicalHessian", "update")] == prfo[("ourPhysicalHessian", "update")] == "ts_bfgs"
    assert qn[("ourPhysicalHessian", "probe_updates")] == prfo[("ourPhysicalHessian", "probe_updates")] == "same_center_physical_hvp"
    assert qn[("ourPhysicalHessian", "probe_batching")] == prfo[("ourPhysicalHessian", "probe_batching")] == "sequential"
    differing = sorted(key for key in set(qn) | set(prfo) if qn.get(key) != prfo.get(key))
    assert differing == [("ourDimer", "translation_optimizer")]


def test_final35_physical_hessian_composition_snippets_use_requested_partition_sources():
    ordinary = _read_override("FINAL35_FORCEBANK_BFGS_BMODE_PRFO.override.ini")
    block = _read_override("FINAL35_FORCEBANK_MS_TSBFGS_BMODE_PRFO.override.ini")
    olsen = _read_override("FINAL35_BEST08_OLSEN_BLOCK_MS_TSBFGS_BMODE_PRFO.override.ini")
    external = _read_override("FINAL35_BEST01_EXTERNAL_MODE_PRFO_CONTROL.override.ini")

    assert ordinary["ourPhysicalHessian"] == {
        "update": "bfgs",
        "probe_updates": "same_center_physical_hvp",
        "probe_batching": "sequential",
    }
    assert block["ourPhysicalHessian"]["update"] == "ts_bfgs"
    assert block["ourPhysicalHessian"]["probe_updates"] == "same_center_physical_hvp"
    assert block["ourPhysicalHessian"]["probe_batching"] == "same_center_block"
    assert block["ourRFO"]["partition"] == "physical_b_eigen"

    assert olsen["ourPhysicalHessian"]["probe_batching"] == "solver_retained_block"
    assert olsen["ourRFO"]["partition"] == "physical_b_eigen"
    assert olsen["ourMinMode"]["maxiter"] == "24"

    assert external["ourRFO"]["partition"] == "external_mode"

    validate_wave_b_config(
        ordinary, rotation_optimizer="lbfgs", translation_optimizer="prfo",
        min_mode_finder="dimer", saddle_engine="mmf",
    )
    validate_wave_b_config(
        block, rotation_optimizer="lbfgs", translation_optimizer="prfo",
        min_mode_finder="dimer", saddle_engine="mmf",
    )
    validate_wave_b_config(
        olsen, rotation_optimizer="ase", translation_optimizer="prfo",
        min_mode_finder="olsen_jd", saddle_engine="mmf",
    )
