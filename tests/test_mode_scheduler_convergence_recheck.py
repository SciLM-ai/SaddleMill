import numpy as np
import pytest

from saddlemill.dimertools.ase_lbfgs_adapter import _RealForceConvergenceMixin


class _CurvatureRejectingBase:
    """Stand-in for ASE MinModeTranslate's curvature-gated convergence routine."""

    def gradient_converged(self, gradient):
        self.super_convergence_calls += 1
        return False


class _FakeDimerAtoms:
    def __init__(self, real_forces, *, refresh_reused=False, refresh_scheduled=False):
        self.real_forces = np.asarray(real_forces, dtype=float)
        self.refresh_reused = bool(refresh_reused)
        self.refresh_scheduled = bool(refresh_scheduled)
        self.reused_refresh_calls = 0
        self.scheduler_refresh_calls = 0
        self.real_force_calls = 0

    def get_forces(self, real=False):
        assert real is True, "convergence must read real (unprojected) forces"
        self.real_force_calls += 1
        return self.real_forces.copy()

    def refresh_reused_mode_for_convergence(self):
        self.reused_refresh_calls += 1
        return self.refresh_reused

    def ensure_wave_b_fresh_validation(self):
        self.scheduler_refresh_calls += 1
        return self.refresh_scheduled


class _Harness(_RealForceConvergenceMixin, _CurvatureRejectingBase):
    def __init__(self, dimeratoms, fmax=0.005):
        self.dimeratoms = dimeratoms
        self.fmax = None if fmax is None else float(fmax)
        self.super_convergence_calls = 0


SMALL = [[0.004, 0.0, 0.0], [0.0, 0.001, 0.0]]
LARGE = [[0.5, 0.0, 0.0], [0.0, 0.001, 0.0]]


def test_small_real_force_converges_despite_parent_curvature_veto():
    opt = _Harness(_FakeDimerAtoms(SMALL), fmax=0.005)
    assert opt.gradient_converged(np.zeros((2, 3))) is True
    assert opt.super_convergence_calls == 0


def test_small_projected_force_with_large_real_force_does_not_converge():
    # Convex-branch case: projected force keeps only the mode component and can
    # vanish while real forces are large.  This is NOT a stationary point.
    atoms = _FakeDimerAtoms(LARGE, refresh_reused=True, refresh_scheduled=True)
    opt = _Harness(atoms, fmax=0.005)
    assert opt.gradient_converged(np.zeros((2, 3))) is False
    assert atoms.reused_refresh_calls == 0
    assert atoms.scheduler_refresh_calls == 0


def test_large_projected_force_with_small_real_force_converges():
    opt = _Harness(_FakeDimerAtoms(SMALL), fmax=0.005)
    assert opt.gradient_converged(np.full((2, 3), -1.0)) is True


def test_configured_fmax_is_honored():
    assert _Harness(_FakeDimerAtoms([[0.009, 0.0, 0.0]]), fmax=0.010).gradient_converged(np.zeros(3)) is True
    assert _Harness(_FakeDimerAtoms([[0.011, 0.0, 0.0]]), fmax=0.010).gradient_converged(np.zeros(3)) is False


def test_threshold_is_strict():
    assert _Harness(_FakeDimerAtoms([[0.005, 0.0, 0.0]]), fmax=0.005).gradient_converged(np.zeros(3)) is False


def test_final_refreshes_run_but_cannot_veto():
    atoms = _FakeDimerAtoms(SMALL, refresh_reused=True, refresh_scheduled=True)
    opt = _Harness(atoms, fmax=0.005)
    assert opt.gradient_converged(np.zeros((2, 3))) is True
    assert atoms.reused_refresh_calls == 1
    assert atoms.scheduler_refresh_calls == 1
    assert opt.super_convergence_calls == 0


def test_fmax_none_falls_back_to_parent():
    opt = _Harness(_FakeDimerAtoms(SMALL), fmax=None)
    assert opt.gradient_converged(np.zeros((2, 3))) is False
    assert opt.super_convergence_calls == 1


def test_real_force_failure_is_loud():
    class _Broken(_FakeDimerAtoms):
        def get_forces(self, real=False):
            raise ValueError("boom")

    opt = _Harness(_Broken(SMALL), fmax=0.005)
    with pytest.raises(RuntimeError):
        opt.gradient_converged(np.zeros((2, 3)))
