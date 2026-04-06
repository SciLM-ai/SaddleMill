"""Tests for tsearch/catsunami/ocpneb.py — OCPNEB, swDNEB, and helper functions."""

import pytest
import numpy as np
from pathlib import Path
from copy import deepcopy
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ===================================================================
# CPU tests — import only the function, not OCPNEB (avoids fairchem)
# ===================================================================

class TestFindSegmentCi:
    """Tests for _find_segment_ci (pure logic, no calculator needed)."""

    def test_ci_in_segment(self):
        from tsearch.catsunami.ocpneb import _find_segment_ci
        energies = [0.0, 0.1, 0.5, 0.3, 0.2, 0.0]
        result = _find_segment_ci(0, 5, climbing_set={2}, energies=energies)
        assert result == 2

    def test_no_interior_images(self):
        from tsearch.catsunami.ocpneb import _find_segment_ci
        energies = [0.0, 0.1, 0.5, 0.3]
        # Adjacent segment [1, 2] has no interior
        result = _find_segment_ci(1, 2, climbing_set=set(), energies=energies)
        assert result is None

    def test_ci_outside_segment_uses_fallback(self):
        from tsearch.catsunami.ocpneb import _find_segment_ci
        energies = [0.0, 0.1, 0.5, 0.8, 0.3, 0.0]
        # CI at 5 is outside segment [0, 4], fallback to max energy interior
        result = _find_segment_ci(0, 4, climbing_set={5}, energies=energies)
        assert result == 3  # highest energy in interior of [0,4]

    def test_fallback_to_max_energy(self):
        from tsearch.catsunami.ocpneb import _find_segment_ci
        energies = [0.0, 0.2, 0.5, 0.3, 0.1, 0.0]
        result = _find_segment_ci(0, 5, climbing_set=set(), energies=energies)
        assert result == 2  # max energy interior image

    def test_multiple_cis_picks_first_in_set(self):
        from tsearch.catsunami.ocpneb import _find_segment_ci
        energies = [0.0, 0.2, 0.5, 0.7, 0.3, 0.0]
        # Both 2 and 3 are CIs, but iteration order of set may vary;
        # function returns the first one found in climbing_set that's in [1,4]
        result = _find_segment_ci(0, 5, climbing_set={2, 3}, energies=energies)
        assert result in {2, 3}


class TestSwDNEB:
    """Tests for swDNEB tangent computation (CPU-only, no calculator)."""

    def _make_mock_state(self, energies):
        """Create a minimal mock state with energies attribute."""

        class MockState:
            pass

        state = MockState()
        state.energies = np.array(energies)
        return state

    def _make_mock_spring(self, tangent, k=5.0, nt=1.0):
        class MockSpring:
            pass

        s = MockSpring()
        s.t = np.array(tangent, dtype=float)
        s.k = k
        s.nt = nt
        return s

    def test_monotonic_increasing_uses_spring2(self):
        from tsearch.catsunami.ocpneb import swDNEB

        class FakeNEB:
            pass

        dneb = swDNEB(FakeNEB())
        state = self._make_mock_state([0.0, 0.5, 1.0, 1.5])
        spring1 = self._make_mock_spring([1.0, 0.0, 0.0])
        spring2 = self._make_mock_spring([0.0, 1.0, 0.0])
        tangent = dneb.get_tangent(state, spring1, spring2, i=2)
        # E[3]>E[2]>E[1] -> should use spring2.t
        expected = spring2.t / np.linalg.norm(spring2.t)
        np.testing.assert_allclose(tangent, expected, atol=1e-10)

    def test_monotonic_decreasing_uses_spring1(self):
        from tsearch.catsunami.ocpneb import swDNEB

        class FakeNEB:
            pass

        dneb = swDNEB(FakeNEB())
        state = self._make_mock_state([1.5, 1.0, 0.5, 0.0])
        spring1 = self._make_mock_spring([1.0, 0.0, 0.0])
        spring2 = self._make_mock_spring([0.0, 1.0, 0.0])
        tangent = dneb.get_tangent(state, spring1, spring2, i=2)
        # E[3]<E[2]<E[1] -> should use spring1.t
        expected = spring1.t / np.linalg.norm(spring1.t)
        np.testing.assert_allclose(tangent, expected, atol=1e-10)

    def test_extremum_weighted_blend(self):
        from tsearch.catsunami.ocpneb import swDNEB

        class FakeNEB:
            pass

        dneb = swDNEB(FakeNEB())
        # E[1]=0.5, E[2]=1.0, E[3]=0.3 -> extremum at i=2
        state = self._make_mock_state([0.0, 0.5, 1.0, 0.3])
        spring1 = self._make_mock_spring([1.0, 0.0, 0.0])
        spring2 = self._make_mock_spring([0.0, 1.0, 0.0])
        tangent = dneb.get_tangent(state, spring1, spring2, i=2)
        # Should be a weighted blend, normalized
        assert np.isclose(np.linalg.norm(tangent), 1.0, atol=1e-10)
        # Should not be purely spring1 or spring2
        assert not np.allclose(tangent, spring1.t / np.linalg.norm(spring1.t), atol=0.01)
        assert not np.allclose(tangent, spring2.t / np.linalg.norm(spring2.t), atol=0.01)

    def test_tangent_is_unit_vector(self):
        from tsearch.catsunami.ocpneb import swDNEB

        class FakeNEB:
            pass

        dneb = swDNEB(FakeNEB())
        for energies in [[0, 1, 2, 3], [3, 2, 1, 0], [0, 2, 1, 3], [0, 1, 0.5, 0]]:
            state = self._make_mock_state(energies)
            s1 = self._make_mock_spring(np.random.randn(3))
            s2 = self._make_mock_spring(np.random.randn(3))
            tangent = dneb.get_tangent(state, s1, s2, i=2)
            assert np.isclose(np.linalg.norm(tangent), 1.0, atol=1e-10)


# ===================================================================
# GPU tests — require FAIRChem calculator
# ===================================================================

@pytest.mark.gpu
class TestOCPNEBConstruction:
    """Test OCPNEB object creation with real FAIRChem calculator."""

    def test_nimages_correct(self, fairchem_calc, neb_images):
        from tsearch.catsunami.ocpneb import OCPNEB
        images = [img.copy() for img in neb_images]
        for img in images:
            img.calc = fairchem_calc
        neb = OCPNEB(images, batch_size=4)
        assert neb.nimages == len(images)

    def test_batch_size_stored(self, fairchem_calc, neb_images):
        from tsearch.catsunami.ocpneb import OCPNEB
        images = [img.copy() for img in neb_images]
        for img in images:
            img.calc = fairchem_calc
        neb = OCPNEB(images, batch_size=6)
        assert neb.batch_size == 6

    def test_reactant_energy_finite(self, fairchem_calc, neb_images):
        from tsearch.catsunami.ocpneb import OCPNEB
        images = [img.copy() for img in neb_images]
        for img in images:
            img.calc = fairchem_calc
        neb = OCPNEB(images, batch_size=4)
        assert np.isfinite(neb.reactant_energy)

    def test_imin_set_empty_initially(self, fairchem_calc, neb_images):
        from tsearch.catsunami.ocpneb import OCPNEB
        images = [img.copy() for img in neb_images]
        for img in images:
            img.calc = fairchem_calc
        neb = OCPNEB(images, batch_size=4)
        assert neb._imin_set == set()

    def test_climbing_set_empty_initially(self, fairchem_calc, neb_images):
        from tsearch.catsunami.ocpneb import OCPNEB
        images = [img.copy() for img in neb_images]
        for img in images:
            img.calc = fairchem_calc
        neb = OCPNEB(images, batch_size=4)
        assert neb._climbing_set == set()


@pytest.mark.gpu
class TestOCPNEBForces:
    """Test force computation with real FAIRChem."""

    def test_get_forces_shape(self, fairchem_calc, neb_images):
        from tsearch.catsunami.ocpneb import OCPNEB
        images = [img.copy() for img in neb_images]
        for img in images:
            img.calc = fairchem_calc
        neb = OCPNEB(images, batch_size=4, climb=True)
        forces = neb.get_forces()
        nimages_interior = len(images) - 2
        natoms = len(images[0])
        assert forces.shape == (nimages_interior * natoms, 3)

    def test_energies_populated(self, fairchem_calc, neb_images):
        from tsearch.catsunami.ocpneb import OCPNEB
        images = [img.copy() for img in neb_images]
        for img in images:
            img.calc = fairchem_calc
        neb = OCPNEB(images, batch_size=4, climb=True)
        neb.get_forces()
        assert len(neb.energies) == len(images)
        assert all(np.isfinite(neb.energies))

    def test_image_fmax_populated(self, fairchem_calc, neb_images):
        from tsearch.catsunami.ocpneb import OCPNEB
        images = [img.copy() for img in neb_images]
        for img in images:
            img.calc = fairchem_calc
        neb = OCPNEB(images, batch_size=4, climb=True)
        neb.get_forces()
        assert len(neb.image_fmax) == len(images)
        assert all(neb.image_fmax >= 0)

    def test_real_forces_includes_endpoints(self, fairchem_calc, neb_images):
        from tsearch.catsunami.ocpneb import OCPNEB
        images = [img.copy() for img in neb_images]
        for img in images:
            img.calc = fairchem_calc
        neb = OCPNEB(images, batch_size=4, climb=True)
        neb.get_forces()
        # Endpoint forces should be non-zero
        assert np.any(neb.real_forces[0] != 0)
        assert np.any(neb.real_forces[-1] != 0)


@pytest.mark.gpu
class TestOCPNEBFrozenImages:
    """Test frozen image handling."""

    def test_frozen_image_reports_cached_fmax(self, fairchem_calc, neb_images):
        from tsearch.catsunami.ocpneb import OCPNEB
        images = [img.copy() for img in neb_images]
        for img in images:
            img.calc = fairchem_calc
        # Constructor-frozen with explicit cached fmax
        neb = OCPNEB(images, batch_size=4, climb=True,
                      frozen_images={3}, frozen_fmax={3: 0.02})
        forces = neb.get_forces()
        # Should report cached NEB fmax, not 0
        assert neb.image_fmax[3] == 0.02
        # But optimizer forces should be zeroed for frozen images
        n = neb.natoms
        frozen_forces = forces.reshape(-1, n, 3)[2]  # image 3 is index 2 in intermediate forces
        assert np.allclose(frozen_forces, 0.0)

    def test_frozen_image_excluded_from_climbing(self, fairchem_calc, neb_images):
        from tsearch.catsunami.ocpneb import OCPNEB
        images = [img.copy() for img in neb_images]
        for img in images:
            img.calc = fairchem_calc
        neb = OCPNEB(images, batch_size=4, climb=True, frozen_images={3})
        neb.get_forces()
        assert 3 not in neb._climbing_set
