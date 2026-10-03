"""Manager acceptance tests for the generic Lanczos convergence correction.

These tests intentionally fail the supplied RC4 baseline.  Do not weaken or
remove them in the worker implementation.
"""

import numpy as np

from saddlemill.dimertools.minmode_solvers import (
    lanczos_lowest_mode,
    softsaddle_lanczos_lowest_mode,
)


def _pathology():
    hessian = np.diag([-2.0, 1.0, 3.0])
    # Squared normalized components are exactly (0.12, 0.29, 0.59), giving
    # first Rayleigh quotient -2*.12 + 1*.29 + 3*.59 = 1.82.
    q0 = np.sqrt(np.array([0.12, 0.29, 0.59], dtype=float))
    return hessian, q0


def test_previous_geometry_curvature_cannot_false_converge_generic_lanczos():
    H, q0 = _pathology()
    calls = 0

    def hvp(v):
        nonlocal calls
        calls += 1
        return H @ v

    result = lanczos_lowest_mode(
        hvp,
        q0,
        max_iterations=3,
        eigenvalue_tolerance=0.01,
        previous_eigenvalue=1.82,
    )

    # RC4 baseline falsely returns after one HVP at +1.82.  Corrected generic
    # Lanczos must continue until the current-geometry residual is small.
    assert result.iterations > 1
    assert calls == result.iterations
    assert result.converged
    assert np.isclose(result.eigenvalue, -2.0, atol=1.0e-12)
    actual = np.linalg.norm(H @ result.eigenvector - result.eigenvalue * result.eigenvector)
    assert actual < 0.1 * abs(result.eigenvalue)
    assert np.isclose(result.residual_norm, actual, atol=1.0e-12)
    assert result.audit is not None
    assert "residual" in result.audit.stopping_rule.lower()
    assert "0.1" in result.audit.stopping_equation


def test_previous_eigenvalue_does_not_control_generic_lanczos_result():
    H, q0 = _pathology()

    without_previous = lanczos_lowest_mode(
        lambda v: H @ v,
        q0,
        max_iterations=3,
        eigenvalue_tolerance=0.01,
        previous_eigenvalue=None,
    )
    with_matching_previous = lanczos_lowest_mode(
        lambda v: H @ v,
        q0,
        max_iterations=3,
        eigenvalue_tolerance=0.01,
        previous_eigenvalue=1.82,
    )

    assert with_matching_previous.converged == without_previous.converged
    assert with_matching_previous.iterations == without_previous.iterations
    assert np.isclose(with_matching_previous.eigenvalue, without_previous.eigenvalue, atol=1.0e-12)
    assert np.allclose(
        np.abs(with_matching_previous.eigenvector),
        np.abs(without_previous.eigenvector),
        atol=1.0e-12,
    )


def test_exact_eigenvector_can_still_converge_in_one_hvp_by_residual():
    H = np.diag([-2.0, 1.0, 3.0])
    q0 = np.array([1.0, 0.0, 0.0])
    calls = 0

    def hvp(v):
        nonlocal calls
        calls += 1
        return H @ v

    result = lanczos_lowest_mode(
        hvp,
        q0,
        max_iterations=3,
        eigenvalue_tolerance=0.01,
        previous_eigenvalue=999.0,
    )

    assert calls == 1
    assert result.iterations == 1
    assert result.converged
    assert np.isclose(result.eigenvalue, -2.0, atol=1.0e-12)
    assert result.residual_norm <= 1.0e-12
    assert result.audit is not None
    assert "residual" in result.audit.stopping_rule.lower()


def test_historical_softsaddle_lanczos_retains_historical_change_stopping():
    H, q0 = _pathology()
    result = softsaddle_lanczos_lowest_mode(
        lambda v: H @ v,
        q0,
        max_iterations=3,
        eigenvalue_tolerance=0.01,
        previous_eigenvalue=1.82,
        convergence_mode="relative",
    )

    # This historical behavior is deliberately *not* corrected here.
    assert result.iterations == 1
    assert result.converged
    assert np.isclose(result.eigenvalue, 1.82, atol=1.0e-12)
    assert result.audit is not None
    assert result.audit.stopping_rule == "relative_eigenvalue_change"
