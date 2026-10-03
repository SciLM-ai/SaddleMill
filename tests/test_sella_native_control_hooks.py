from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

MODULE = Path(__file__).parents[1] / "saddlemill" / "sella_ablation.py"
spec = importlib.util.spec_from_file_location("candidate_sella_ablation_native_controls", MODULE)
sa = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sa
spec.loader.exec_module(sa)


class FakeH:
    def __init__(self):
        self.B = np.diag([-1.0, 2.0, 3.0])
        self.initialized = True
    def asarray(self):
        return self.B.copy()
    def update(self, dx, dg):
        return None


class FakePES:
    def __init__(self):
        self.int = None
        self.H = FakeH()
        self.curr = {"x": np.zeros(3), "g": np.array([0.2, -0.1, 0.05]), "f": 0.0}
        self.last = {"x": None, "g": None}
        self.neval = 7
        self.hessian_function = None
        self.diag_calls = 0
        self.kick_calls = 0
    def get_x(self):
        return np.asarray(self.curr["x"], dtype=float).copy()
    def diag(self, *args, **kwargs):
        self.diag_calls += 1
    def _update_H(self, dx, dg):
        return None
    def _calc_eg(self, *args, **kwargs):
        self.neval += 1
        return 0.0, self.curr["g"].copy()
    def kick(self, dx, diag=False, **kwargs):
        self.kick_calls += 1
        return 1.0


class QuasiNewton:
    def __init__(self):
        self.newton_safe = True
        self.order = 1
        self.alpha0 = 0.0
        self.alphamin = 0.0
        self.alphamax = np.inf
        self.slope = -1.0
        self.g = np.array([0.2, -0.1, 0.05])
        self.H = FakeH()


class FakeRestrictedAtomicStep:
    constructions = 0
    seen_newton_safe = []
    def __init__(self, pes, order, delta, method="qn"):
        FakeRestrictedAtomicStep.constructions += 1
        assert method == "qn"
        self.pes = pes
        self.delta = delta
        self.stepper = QuasiNewton()
    def get_s(self):
        FakeRestrictedAtomicStep.seen_newton_safe.append(bool(self.stepper.newton_safe))
        return np.array([0.01, 0.0, 0.0]), self.delta


class FakeOptimizer:
    def __init__(self, method="qn"):
        self.pes = FakePES()
        self.rs = FakeRestrictedAtomicStep
        self.sm_sella_version = "2.5.0"
        self.delta = 0.1
        self.delta_cell = 0.1
        self.method = method
        self.ord = 1
        self.eig = True
        self.nsteps = 0
    def _predict_step(self):
        rs = self.rs(self.pes, self.ord, self.delta, method=self.method)
        self.last_rs = rs
        return rs.get_s()
    def step(self):
        return self._predict_step()


def test_qn_newton_safe_false_uses_native_restricted_step_and_restores_class_without_pes_calls():
    FakeRestrictedAtomicStep.constructions = 0
    FakeRestrictedAtomicStep.seen_newton_safe = []
    opt = FakeOptimizer(method="qn")
    original_rs = opt.rs
    before_neval = opt.pes.neval
    session = sa.SellaAblationSession(opt, sa.QN_NEWTON_SAFE_FALSE)
    with session:
        step, smag = opt._predict_step()
        assert isinstance(opt.last_rs, original_rs)
        assert type(opt.last_rs).__mro__[1] is original_rs
        assert opt.last_rs.stepper.newton_safe is False
        np.testing.assert_allclose(step, [0.01, 0.0, 0.0])
        assert smag == pytest.approx(0.1)
    assert opt.rs is original_rs
    assert FakeRestrictedAtomicStep.constructions == 1
    assert FakeRestrictedAtomicStep.seen_newton_safe == [False]
    assert opt.pes.neval == before_neval
    assert opt.pes.diag_calls == 0
    assert opt.pes.kick_calls == 0
    assert session.restricted_step_events[-1]["newton_safe_effective"] is False
    assert session.restricted_step_events[-1]["additional_pes_calls"] == 0


def test_qn_newton_safe_false_fails_closed_on_wrong_native_method():
    opt = FakeOptimizer(method="prfo")
    with pytest.raises(sa.SellaAblationError, match="method='qn'"):
        sa.SellaAblationSession(opt, sa.QN_NEWTON_SAFE_FALSE).install()


def test_qn_newton_safe_false_is_not_composable_with_prfo_ablation_components():
    with pytest.raises(sa.SellaAblationError, match="cannot be combined"):
        sa.resolve_sella_ablation(
            f"{sa.QN_NEWTON_SAFE_FALSE}+{sa.TRUST_ISOLATION}"
        )


def test_sella_version_gate_remains_exactly_250():
    assert sa.SUPPORTED_SELLA_VERSION == "2.5.0"
    opt = FakeOptimizer(method="qn")
    opt.sm_sella_version = "2.5.1"
    with pytest.raises(sa.SellaAblationError, match="2.5.0"):
        sa.SellaAblationSession(opt, sa.QN_NEWTON_SAFE_FALSE).install()

class _TrustOnlyOptimizer(FakeOptimizer):
    def __init__(self):
        super().__init__(method="prfo")
        self.predict_calls = 0
        self.native_step_calls = 0

    def _predict_step(self):
        self.predict_calls += 1
        return np.array([0.012, -0.003, 0.0]), self.delta

    def step(self):
        self.native_step_calls += 1
        # Mimic native Sella accepting a P-RFO step and adapting both trust radii.
        step, _ = self._predict_step()
        self.pes.curr["x"] = np.asarray(self.pes.curr["x"], dtype=float) + step
        self.delta = 0.037
        self.delta_cell = 0.041
        return step


def test_trust_isolation_preserves_prfo_predictor_result_and_only_holds_radius_policy():
    opt = _TrustOnlyOptimizer()
    before_neval = opt.pes.neval
    before_predict = opt._predict_step()
    assert before_predict[1] == pytest.approx(0.1)
    opt.predict_calls = 0

    session = sa.SellaAblationSession(opt, sa.TRUST_ISOLATION)
    with session:
        result = opt.step()

    np.testing.assert_allclose(result, before_predict[0])
    assert opt.method == "prfo"
    assert opt.predict_calls == 1
    assert opt.native_step_calls == 1
    assert opt.delta == pytest.approx(0.1)
    assert opt.delta_cell == pytest.approx(0.1)
    assert opt.pes.neval == before_neval
    row = session.rows[-1]
    assert row["trust_update_held"] == 1
    assert row["trust_radius_before"] == pytest.approx(0.1)
    assert row["trust_radius_after_native"] == pytest.approx(0.037)
    assert row["trust_radius_after_effective"] == pytest.approx(0.1)
    assert row["adapter_additional_pes_calls"] == 0
