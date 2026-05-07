"""GPU integration tests for saddlemill/dimeropt.py using FAIRChem calculator."""

import copy

import numpy as np
import pytest
from ase.io import Trajectory
from ase.constraints import FixAtoms

from saddlemill.dimeropt import dimeropt
from tests.conftest import make_config_dict


@pytest.mark.gpu
class TestDimeropt:
    """Integration tests that run dimeropt() with FAIRChem on real structures."""

    def _setup_dirs(self, tmp_path, monkeypatch):
        """Create output directories and chdir to tmp_path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "Dimer_status_csvs").mkdir()
        (tmp_path / "Dimer_trajes").mkdir()
        (tmp_path / "Dimer_debug_zips").mkdir()

    def _make_config(self, dataset_type="oc", reaction_types="adsorbate_atom",
                     num_attempts_per_type=1, steps=30, **overrides):
        config = make_config_dict(method="Dimer", steps=steps, fmax=0.05, Optimizer="MDMin")
        config["ourDimer"]["dataset_type"] = dataset_type
        config["ourDimer"]["reaction_types"] = reaction_types
        config["ourDimer"]["num_attempts_per_type"] = num_attempts_per_type
        config["ourDimer"]["supercell"] = True
        config["ourDimer"]["delocalization_threshold"] = 0.8
        config["ourDimer"]["extension_check_fmax"] = 0.4
        config["ourDimer"]["extension_check_curvature"] = -0.2
        config.setdefault("DimerControl", {})
        for k, v in overrides.items():
            if k in config["ourDimer"]:
                config["ourDimer"][k] = v
            elif k in config["Main"]:
                config["Main"][k] = v
        return config

    def _prepare_atoms(self, atoms):
        """Deep-copy atoms and wrap info into orig_info."""
        atoms = atoms.copy()
        atoms.info = {"orig_info": dict(atoms.info)}
        return atoms

    def _read_output_traj(self, tmp_path):
        """Read output trajectory frames.

        Asserts every frame round-trips with energy/forces in calc.results
        (regression guard for the SinglePointCalculator + atoms.wrap() race).
        """
        traj_path = tmp_path / "Dimer_trajes" / "collected_ts_rank_0.traj"
        assert traj_path.exists(), f"Output trajectory not found at {traj_path}"
        traj = Trajectory(str(traj_path), "r")
        for idx, frame in enumerate(traj):
            assert frame.calc is not None, f"Frame {idx} has no calculator after read-back"
            assert 'energy' in frame.calc.results, (
                f"Frame {idx} missing energy — SPC+wrap race regression")
            assert 'forces' in frame.calc.results, (
                f"Frame {idx} missing forces — SPC+wrap race regression")
        return traj

    def _read_csv(self, tmp_path):
        """Read CSV status lines."""
        csv_path = tmp_path / "Dimer_status_csvs" / "status_rank_0.csv"
        assert csv_path.exists(), f"Status CSV not found at {csv_path}"
        return csv_path.read_text().strip().splitlines()

    # ----- Test: OC adsorbate_atom reaction type -----

    def test_oc_adsorbate_atom(self, tmp_path, monkeypatch, fairchem_calc, oc_adsorbate_slab):
        """OC mode, adsorbate_atom reaction type, 30 steps.

        Verify: output traj has at least 1 frame with required metadata.
        """
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(dataset_type="oc", reaction_types="adsorbate_atom")
        atoms = self._prepare_atoms(oc_adsorbate_slab)

        dimeropt(0, config, atoms, fairchem_calc,
                 consecutive_errors=[0], executorlib_worker_id=0)

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) >= 1

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) >= 1
            frame = traj[0]
            assert frame.info["src_index"] == 0
            assert "attempt_id" in frame.info
            assert "reaction_type" in frame.info
            assert "eigenmode" in frame.info
            assert "converged" in frame.info
            assert frame.info["reaction_type"] == "adsorbate_atom"

    # ----- Test: OC diffusion reaction type -----

    def test_oc_diffusion(self, tmp_path, monkeypatch, fairchem_calc, oc_adsorbate_slab):
        """OC mode, diffusion reaction type."""
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(dataset_type="oc", reaction_types="diffusion")
        atoms = self._prepare_atoms(oc_adsorbate_slab)

        dimeropt(0, config, atoms, fairchem_calc,
                 consecutive_errors=[0], executorlib_worker_id=0)

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) >= 1

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) >= 1
            frame = traj[0]
            # reaction_type could be 'diffusion' or 'desorption' if desorption was detected
            assert frame.info["reaction_type"] in ("diffusion", "desorption")

    # ----- Test: OC multiple reaction types -----

    def test_oc_multiple_types(self, tmp_path, monkeypatch, fairchem_calc, oc_adsorbate_slab):
        """reaction_types='adsorbate_atom diffusion', num_attempts_per_type=1.

        Verify: output has frames for both reaction types.
        """
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(
            dataset_type="oc",
            reaction_types="adsorbate_atom diffusion",
            num_attempts_per_type=1,
        )
        atoms = self._prepare_atoms(oc_adsorbate_slab)

        dimeropt(0, config, atoms, fairchem_calc,
                 consecutive_errors=[0], executorlib_worker_id=0)

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) >= 2  # one per attempt

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) >= 2
            reaction_types_found = set()
            for idx in range(len(traj)):
                rt = traj[idx].info["reaction_type"]
                reaction_types_found.add(rt)
            # adsorbate_atom should always be there
            assert "adsorbate_atom" in reaction_types_found
            # diffusion might show as 'desorption' if the desorption check triggers
            assert "diffusion" in reaction_types_found or "desorption" in reaction_types_found

    # ----- Test: bulk vacancy reaction type -----

    def test_bulk_vacancy(self, tmp_path, monkeypatch, fairchem_calc, bulk_crystal):
        """Bulk mode, vacancy reaction type, 30 steps."""
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(
            dataset_type="bulk",
            reaction_types="vacancy",
            steps=30,
        )
        atoms = self._prepare_atoms(bulk_crystal)

        dimeropt(0, config, atoms, fairchem_calc,
                 consecutive_errors=[0], executorlib_worker_id=0)

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) >= 1

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) >= 1
            frame = traj[0]
            assert frame.info["src_index"] == 0
            assert frame.info["reaction_type"] == "vacancy"
            assert "eigenmode" in frame.info

    # ----- Test: bulk ring reaction type -----

    def test_bulk_ring(self, tmp_path, monkeypatch, fairchem_calc, bulk_crystal):
        """Bulk mode, ring reaction type with ring_sizes=[2] (pairwise exchange)."""
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(
            dataset_type="bulk",
            reaction_types="ring",
            steps=30,
        )
        config["ourDimer"]["ring_sizes"] = [2]
        atoms = self._prepare_atoms(bulk_crystal)

        dimeropt(0, config, atoms, fairchem_calc,
                 consecutive_errors=[0], executorlib_worker_id=0)

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) >= 1

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) >= 1
            frame = traj[0]
            assert frame.info["reaction_type"] == "ring"

    # ----- Test: entries_to_run filtering -----

    def test_entries_to_run_filtering(self, tmp_path, monkeypatch, fairchem_calc, oc_adsorbate_slab):
        """Generate 2 attempts via num_attempts_per_type=2, entries_to_run={0}.

        Verify: only attempt 0 appears in output.
        """
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(
            dataset_type="oc",
            reaction_types="adsorbate_atom",
            num_attempts_per_type=2,
        )
        atoms = self._prepare_atoms(oc_adsorbate_slab)

        dimeropt(0, config, atoms, fairchem_calc,
                 consecutive_errors=[0], executorlib_worker_id=0,
                 entries_to_run={0})

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) == 1  # only attempt 0

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) == 1
            assert traj[0].info["attempt_id"] == 0

    # ----- Test: consecutive error reset -----

    def test_consecutive_error_reset(self, tmp_path, monkeypatch, fairchem_calc, oc_adsorbate_slab):
        """Pass consecutive_errors=[3], verify it resets to 0 on success."""
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(
            dataset_type="oc",
            reaction_types="adsorbate_atom",
        )
        atoms = self._prepare_atoms(oc_adsorbate_slab)
        consecutive_errors = [3]

        dimeropt(0, config, atoms, fairchem_calc,
                 consecutive_errors=consecutive_errors, executorlib_worker_id=0)

        # After a successful run, consecutive_errors should be reset to 0
        assert consecutive_errors[0] == 0
