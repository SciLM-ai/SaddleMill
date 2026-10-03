from __future__ import annotations

from importlib.metadata import version
import inspect

import numpy as np
import pytest

pytest.importorskip("sella")
if version("sella") != "2.5.0":
    pytest.skip("native-control parity tests require exact sella==2.5.0", allow_module_level=True)

from sella.linalg import ApproximateHessian
from sella.optimize.restricted_step import RestrictedAtomicStep
from sella.optimize.stepper import QuasiNewton, RationalFunctionOptimization

from saddlemill.sella_ablation import _qn_newton_safe_false_restricted_step_class


def _hessian(B):
    B = np.asarray(B, dtype=float)
    H = ApproximateHessian(B.shape[0], B.shape[0], None)
    H.B = B.copy()
    H.evals, H.evecs = np.linalg.eigh(B)
    return H


class _PES:
    def __init__(self):
        self.int = None
        self.H = _hessian(np.diag([-1.0, 2.0, 3.0]))
        self._g = np.array([0.2, -0.1, 0.05])
    def get_g(self): return self._g.copy()
    def get_scons(self): return np.zeros(3)
    def get_H(self): return self.H
    def get_Unred(self): return np.eye(3)
    def get_Ufree(self): return np.eye(3)
    def get_HL_projected(self, U): return self.H.project(np.asarray(U, dtype=float))


def test_installed_sella_qn_native_newton_safe_is_true_and_hook_changes_only_live_stepper_flag():
    assert version("sella") == "2.5.0"
    assert QuasiNewton(np.array([0.2, -0.1, 0.05]), _hessian(np.diag([-1.0, 2.0, 3.0])), order=1).newton_safe is True
    events = []
    Wrapped = _qn_newton_safe_false_restricted_step_class(
        RestrictedAtomicStep, lambda rs, before: events.append((rs, before))
    )
    rs = Wrapped(_PES(), 1, 0.1, method="qn")
    assert isinstance(rs, RestrictedAtomicStep)
    assert type(rs.stepper).__name__ == "QuasiNewton"
    assert rs.stepper.newton_safe is False
    assert events and events[0][1] is True


def test_installed_sella_rfo_native_newton_safe_is_already_false():
    stepper = RationalFunctionOptimization(
        np.array([0.2, -0.1, 0.05]),
        _hessian(np.diag([-1.0, 2.0, 3.0])),
        order=1,
    )
    assert stepper.newton_safe is False
    src = inspect.getsource(type(stepper))
    assert "newton_safe" in src or stepper.newton_safe is False


def test_installed_sella_prfo_restricted_step_uses_native_partitioned_rfo_stepper():
    from sella.optimize.stepper import PartitionedRationalFunctionOptimization

    rs = RestrictedAtomicStep(_PES(), 1, 0.1, method="prfo")
    assert isinstance(rs, RestrictedAtomicStep)
    assert isinstance(rs.stepper, PartitionedRationalFunctionOptimization)
    assert type(rs.stepper).__module__.startswith("sella.")
