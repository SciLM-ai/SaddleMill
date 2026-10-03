from __future__ import annotations

import numpy as np
import pytest

from saddlemill.dimertools.dimer_factory import create_rfo_translation_runtime
from saddlemill.dimertools.foundation_types import ActiveCoordinateSpace
from saddlemill.dimertools.rfo_translation import RFOTranslationInput, compute_rfo_translation


def _runtime_config(*, optimizer: str = "prfo", partition: str = "external_mode"):
    return {
        "ourDimer": {"translation_optimizer": optimizer},
        "ourPhysicalHessian": {"update": "ts_bfgs"},
        "ourModeSchedule": {"parallel_force_damping": "none"},
        "ourRFO": {
            "partition": partition,
            "step_control": "ras",
            "target_order": 1,
            "trust_radius": 0.1,
            "trust_policy": "fixed",
            "restricted_step_root_solver": "bisection",
            "physical_root_policy": "lowest",
            "negative_mode_policy": "allow",
            "ras_tolerance": None,
            "ras_max_iterations": 1000,
        },
    }


def test_generic_prfo_ras_accepts_external_mode_but_qn_mmf_does_not():
    runtime = create_rfo_translation_runtime(_runtime_config())
    meta = runtime.resolved_metadata()
    assert meta["partition"] == "external_mode"
    assert meta["step_control"] == "ras"
    assert meta["translation_optimizer"] == "prfo"

    with pytest.raises(ValueError, match="translation_optimizer=qn_mmf requires physical_b_eigen partition"):
        create_rfo_translation_runtime(_runtime_config(optimizer="qn_mmf"))


def test_external_mode_prfo_uses_shared_ras_bisection_without_extra_pes_calls():
    space = ActiveCoordinateSpace.all_cartesian(1)
    request = RFOTranslationInput(
        matrix=np.diag([-1.0, 2.0, 3.0]),
        raw_gradient=np.array([0.4, -0.2, 0.1]),
        coordinate_space=space,
        state_id=0,
        state_uid="state0:g",
        geometry_id="g",
        coordinate_space_id=space.identity,
        center_observation_id="obs",
        model_origin="physical_hessian_model",
        model_update_type="ts_bfgs",
        model_age=1,
        force_units="eV/A",
        hessian_units="eV/A^2",
        external_mode=np.array([0.0, 1.0, 0.0]),
        t17_added_pes_calls=0,
    )
    result = compute_rfo_translation(
        request,
        partition="external_mode",
        step_control="ras",
        target_order=1,
        translation_step_method="prfo",
        trust_radius=0.1,
        ras_root_solver="bisection",
        physical_root_policy="lowest",
        negative_mode_policy="allow",
    )
    assert result.success
    assert result.algorithm == "prfo"
    assert result.partition == "external_mode"
    assert result.step_control == "ras"
    assert result.restricted_metadata is not None
    assert result.restricted_metadata["algorithm"] == "prfo"
    assert result.t17_added_pes_calls == 0
    assert result.achieved_norm == pytest.approx(0.1, abs=2.0e-10)
