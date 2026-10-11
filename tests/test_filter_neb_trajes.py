"""Tests for scripts/filter_neb_trajes.py on synthetic band energies.

The intermediate-minimum rule walks image by image from each endpoint toward
the climbing image (CI). One step with dE < -0.05 eV flags the band, as do two
consecutive drops; an isolated drop of at most 0.05 eV is tolerated. The band
filter also drops bands that are not converged or lack an energy.
"""

import importlib.util
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "filter_neb_trajes.py"
_spec = importlib.util.spec_from_file_location("filter_neb_trajes", SCRIPT)
fnt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fnt)


def _has_imin(energies, ci):
    return fnt._band_has_intermediate_minimum(dict(enumerate(energies)), ci)


class TestIntermediateMinimumWalk:
    def test_clean_band_kept(self):
        assert not _has_imin([0.0, 0.2, 0.4, 0.6, 0.4, 0.2, 0.0], ci=3)

    def test_single_big_drop_flagged(self):
        """0.20 -> 0.14 is one step below -0.05 eV on the left walk."""
        assert _has_imin([0.0, 0.20, 0.14, 0.6, 0.3, 0.0], ci=3)

    def test_drop_of_exactly_thresh_is_small(self):
        """0.05 -> 0.0 is exactly -0.05 eV: a small drop, so tolerated alone."""
        assert not _has_imin([0.05, 0.0, 0.3, 0.6, 0.3, 0.0], ci=3)

    def test_two_consecutive_small_drops_flagged(self):
        assert _has_imin([0.0, 0.30, 0.27, 0.24, 0.6, 0.2, 0.0], ci=4)

    def test_isolated_small_wobble_tolerated(self):
        """Small drops each followed by a rise are endpoint jitter, not a basin."""
        assert not _has_imin([0.02, 0.0, 0.3, 0.6, 0.3, 0.0], ci=3)
        assert not _has_imin([0.0, -0.03, 0.2, 0.17, 0.5, 0.2, 0.0], ci=4)

    def test_right_walk_single_big_drop_flagged(self):
        """Walking from the right endpoint, 0.36 -> 0.30 is a drop of 0.06 eV."""
        assert _has_imin([0.0, 0.3, 0.6, 0.30, 0.36, 0.0], ci=2)

    def test_right_walk_two_consecutive_small_drops_flagged(self):
        assert _has_imin([0.0, 0.3, 0.6, 0.33, 0.36, 0.39, 0.0], ci=2)

    def test_right_walk_isolated_small_wobble_tolerated(self):
        assert not _has_imin([0.0, 0.3, 0.6, 0.27, 0.30, 0.0], ci=2)


# --------------- Band filter on a synthetic rank file ---------------

CLEAN = [0.0, 0.2, 0.4, 0.6, 0.4, 0.2, 0.0]
BIG_DROP = [0.0, 0.20, 0.14, 0.6, 0.4, 0.2, 0.0]
CI = 3


def _band(src, energies, ci=CI, subband=0, status="converged", info=None,
          no_energy_at=None):
    """One band of H atoms marching along x, E/F attached via SPC."""
    frames = []
    for j, e in enumerate(energies):
        a = Atoms("H", positions=[[1.0 + 0.1 * j, 2.0, 2.0]],
                  cell=[5.0, 5.0, 5.0], pbc=True)
        a.info.update(src_index=src, image_idx=j, subband_idx=subband,
                      nimages=len(energies))
        if status is not None:
            a.info["status"] = status
        a.info.update(info or {})
        if j == ci:
            a.info["barrier"] = energies[ci] - energies[0]
        if j != no_energy_at:
            a.calc = SinglePointCalculator(a, energy=e, forces=np.zeros((1, 3)))
        frames.append(a)
    return frames


def _run(tmp_path, bands):
    in_dir, out_dir = tmp_path / "NEB_trajes", tmp_path / "NEB_trajes_filtered"
    in_dir.mkdir()
    with Trajectory(str(in_dir / "collected_ts_rank_0.traj"), "w") as w:
        for band in bands:
            for a in band:
                w.write(a)
    stats = fnt.filter_rank(0, str(in_dir), str(out_dir))
    with Trajectory(str(out_dir / "collected_ts_rank_0.traj")) as r:
        kept = {(a.info["src_index"], a.info["subband_idx"]) for a in r}
    return stats, kept


class TestFilterRank:
    def test_clean_bands_kept_imin_dropped(self, tmp_path):
        stats, kept = _run(tmp_path, [
            _band(0, CLEAN),
            _band(1, CLEAN, status="converged_CI"),
            _band(2, BIG_DROP),
        ])
        assert kept == {(0, 0), (1, 0)}
        assert stats["drop_imin"] == 1
        assert stats["n_frames_out"] == 2 * len(CLEAN)

    def test_not_converged_dropped(self, tmp_path):
        stats, kept = _run(tmp_path, [
            _band(0, CLEAN),
            _band(1, CLEAN, status="not_converged"),
        ])
        assert kept == {(0, 0)}
        assert stats["drop_status"] == 1

    def test_missing_energy_dropped(self, tmp_path):
        stats, kept = _run(tmp_path, [
            _band(0, CLEAN),
            _band(1, CLEAN, no_energy_at=5),
        ])
        assert kept == {(0, 0)}
        assert stats["drop_no_ef"] == 1

    def test_missing_ci_tag_dropped(self, tmp_path):
        stats, kept = _run(tmp_path, [_band(0, CLEAN), _band(1, CLEAN, ci=None)])
        assert kept == {(0, 0)}
        assert stats["drop_no_ci_tag"] == 1

    def test_missing_status_falls_back_to_band_flags(self, tmp_path):
        stats, kept = _run(tmp_path, [
            _band(0, CLEAN, status=None,
                  info={"band_converged": True, "band_converged_CI": True}),
            _band(1, CLEAN, status=None,
                  info={"band_converged": False, "band_converged_CI": True}),
            _band(2, CLEAN, status=None,
                  info={"band_converged": False, "band_converged_CI": False}),
            _band(3, CLEAN, status=None),
        ])
        assert kept == {(0, 0), (1, 0)}
        assert stats["drop_status"] == 2

    def test_subbands_filtered_independently(self, tmp_path):
        """Sub-band 1 of src 0 has an intermediate minimum; sub-band 0 survives."""
        stats, kept = _run(tmp_path, [
            _band(0, CLEAN, subband=0),
            _band(0, BIG_DROP, subband=1),
        ])
        assert kept == {(0, 0)}
        assert stats["n_bands_in"] == 2
        assert stats["drop_imin"] == 1
