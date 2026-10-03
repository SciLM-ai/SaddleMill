"""Focused worker regressions for corrected generic Lanczos convergence."""

import numpy as np

from saddlemill.dimertools.minmode_solvers import lanczos_lowest_mode


def test_generic_lanczos_uses_fixed_point_one_residual_gamma():
    H = np.diag([-2.0, 1.0, 3.0])
    q0 = np.array([1.0, 0.03, 0.0], dtype=float)
    q0 /= np.linalg.norm(q0)

    result = lanczos_lowest_mode(
        lambda v: H @ v,
        q0,
        max_iterations=3,
        eigenvalue_tolerance=0.0,
        previous_eigenvalue=999.0,
    )

    # This vector is not exact, but its first current-geometry Ritz residual is
    # below the fixed Olsen/JD gamma=0.1 threshold.  The legacy config tolerance
    # must not become the residual gamma.
    assert result.iterations == 1
    assert result.converged
    assert result.residual_norm > 0.0
    assert result.residual_norm < 0.1 * abs(result.eigenvalue)
    assert result.audit is not None
    assert result.audit.metadata["residual_gamma"] == 0.1
    assert result.audit.metadata["eigenvalue_tolerance_controls_convergence"] is False
    assert result.audit.metadata["previous_eigenvalue_controls_convergence"] is False


def test_generic_lanczos_reports_full_retained_action_residual_without_extra_hvp():
    # Slightly nonsymmetric synthetic operator exercises the reason the generic
    # stabilized implementation uses U@c - theta*Q@c rather than beta*c_last.
    H = np.array(
        [
            [-2.0, 0.4, 0.0],
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 3.0],
        ],
        dtype=float,
    )
    q0 = np.array([0.5, 0.6, 0.7], dtype=float)
    q0 /= np.linalg.norm(q0)
    calls = 0

    def hvp(v):
        nonlocal calls
        calls += 1
        return H @ v

    result = lanczos_lowest_mode(
        hvp,
        q0,
        max_iterations=2,
        eigenvalue_tolerance=1.0,
        previous_eigenvalue=None,
    )

    actual = np.linalg.norm(
        H @ result.eigenvector - result.eigenvalue * result.eigenvector
    )
    assert calls == result.iterations == 2
    assert not result.converged
    assert np.isclose(result.residual_norm, actual, atol=1.0e-12)
    assert result.audit is not None
    assert np.isclose(result.audit.full_residual_norm, actual, atol=1.0e-12)
    assert np.isclose(result.audit.solver_operator_residual_norm, actual, atol=1.0e-12)
    assert result.audit.metadata["residual_source"] == "retained_actual_hvp_actions"
