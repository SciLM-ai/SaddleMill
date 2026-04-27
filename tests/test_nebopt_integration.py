"""GPU integration tests for tsearch/nebopt.py using FAIRChem calculator."""

import copy

import numpy as np
import pytest
from ase.io import Trajectory
from ase.optimize import MDMin

from tsearch.nebopt import nebopt
from tests.conftest import make_config_dict


@pytest.mark.gpu
class TestNebopt:
    """Integration tests that run nebopt() on a real NEB band with FAIRChem."""

    def _setup_dirs(self, tmp_path, monkeypatch):
        """Create output directories and chdir to tmp_path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "NEB_status_csvs").mkdir()
        (tmp_path / "NEB_trajes").mkdir()
        (tmp_path / "NEB_debug_zips").mkdir()

    def _make_config(self, **overrides):
        config = make_config_dict(method="NEB", steps=30, fmax=0.05, Optimizer="MDMin")
        config["ourNEB"]["relax_endpoints"] = False
        config["ourNEB"]["interpolate_method"] = False
        config["ourNEB"]["num_frames"] = 10
        config["ourNEB"]["batch_size"] = 4
        config["ourNEB"]["DNEB"] = False
        config["ourNEB"]["intermediate_minima_check_step"] = 0
        config["ourNEB"]["dimer_refine_ci"] = False
        config["ourNEB"]["refine_band_steps"] = 0
        config["ourNEB"]["max_num_frames"] = None
        config["BaseNEB"]["climb"] = True
        for k, v in overrides.items():
            if k in config["ourNEB"]:
                config["ourNEB"][k] = v
            elif k in config["BaseNEB"]:
                config["BaseNEB"][k] = v
            elif k in config["Main"]:
                config["Main"][k] = v
        return config

    def _prepare_images(self, neb_images, fairchem_calc):
        """Deep-copy images, wrap info into orig_info, attach calculator."""
        images = [img.copy() for img in neb_images]
        for img in images:
            img.info = {"orig_info": dict(img.info)}
            img.calc = fairchem_calc
        return images

    def _read_output_traj(self, tmp_path):
        """Read output trajectory frames.

        Asserts every frame round-trips with energy/forces in calc.results
        (regression guard for the SinglePointCalculator + atoms.wrap() race).
        """
        traj_path = tmp_path / "NEB_trajes" / "collected_ts_rank_0.traj"
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
        csv_path = tmp_path / "NEB_status_csvs" / "status_rank_0.csv"
        assert csv_path.exists(), f"Status CSV not found at {csv_path}"
        return csv_path.read_text().strip().splitlines()

    # ----- Test: basic NEB run -----

    def test_basic_neb_run(self, tmp_path, monkeypatch, fairchem_calc, neb_images):
        """10 images, 30 steps, no endpoint relax, interpolate=False.

        Verify: output traj written, CSV written, all images have required metadata.
        """
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config()
        images = self._prepare_images(neb_images, fairchem_calc)

        nebopt(0, config, images, fairchem_calc, (MDMin, MDMin),
               consecutive_errors=[0], executorlib_worker_id=0)

        # Check CSV was written
        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) >= 1

        # Check output trajectory
        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) >= 10  # at least 10 images (full band)
            required_keys = [
                "src_index", "image_idx", "subband_idx", "image_type",
                "effective_fmax", "image_converged", "band_converged",
                "band_converged_CI",
            ]
            for frame_idx in range(len(traj)):
                frame = traj[frame_idx]
                for key in required_keys:
                    assert key in frame.info, (
                        f"Frame {frame_idx} missing key '{key}'. "
                        f"Available keys: {list(frame.info.keys())}"
                    )
                # src_index should be 0
                assert frame.info["src_index"] == 0
                # image_type should be one of the valid types
                assert frame.info["image_type"] in (
                    "endpoint", "intermediate_minimum", "climbing", "regular"
                )
                # effective_fmax should be a finite float
                assert np.isfinite(frame.info["effective_fmax"])

    # ----- Test: endpoint relaxation -----

    def test_neb_with_endpoint_relaxation(self, tmp_path, monkeypatch, fairchem_calc, neb_images):
        """relax_endpoints=True, endpoint_relax_steps=20, 30 NEB steps."""
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(
            relax_endpoints=True,
            endpoint_relax_steps=20,
        )
        config["ourNEB"]["endpoint_relax_Optimizer"] = None  # uses main Optimizer (MDMin)
        images = self._prepare_images(neb_images, fairchem_calc)

        nebopt(0, config, images, fairchem_calc, (MDMin, MDMin),
               consecutive_errors=[0], executorlib_worker_id=0)

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) >= 1

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) >= 10
            # Endpoints should be present
            types = [traj[idx].info["image_type"] for idx in range(len(traj))]
            assert "endpoint" in types

    # ----- Test: re-interpolation with ase_linear -----

    def test_neb_reinterpolation_ase_linear(self, tmp_path, monkeypatch, fairchem_calc, neb_images):
        """Pass only endpoints (first and last), interpolate_method='ase_linear', num_frames=5.

        Verify: output has 5 images.
        """
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(
            interpolate_method="ase_linear",
            num_frames=5,
        )
        # Only pass endpoints
        endpoints = self._prepare_images([neb_images[0], neb_images[-1]], fairchem_calc)

        nebopt(0, config, endpoints, fairchem_calc, (MDMin, MDMin),
               consecutive_errors=[0], executorlib_worker_id=0)

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) == 5
            # Check interpolation_method recorded
            assert traj[0].info.get("interpolation_method") in ("ase_linear", "ase_idpp")

    # ----- Test: re-interpolation with ase_idpp -----

    def test_neb_reinterpolation_ase_idpp(self, tmp_path, monkeypatch, fairchem_calc, neb_images):
        """Pass only endpoints, interpolate_method='ase_idpp', num_frames=5.

        Verify: output has 5 images.
        """
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(
            interpolate_method="ase_idpp",
            num_frames=5,
        )
        endpoints = self._prepare_images([neb_images[0], neb_images[-1]], fairchem_calc)

        nebopt(0, config, endpoints, fairchem_calc, (MDMin, MDMin),
               consecutive_errors=[0], executorlib_worker_id=0)

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) == 5
            assert traj[0].info.get("interpolation_method") == "ase_idpp"

    # ----- Test: DNEB mode -----

    def test_neb_dneb_mode(self, tmp_path, monkeypatch, fairchem_calc, neb_images):
        """DNEB=True, 30 steps."""
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(DNEB=True)
        images = self._prepare_images(neb_images, fairchem_calc)

        nebopt(0, config, images, fairchem_calc, (MDMin, MDMin),
               consecutive_errors=[0], executorlib_worker_id=0)

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) >= 1

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) >= 10

    # ----- Test: intermediate minima detection -----

    def test_neb_intermediate_minima(self, tmp_path, monkeypatch, fairchem_calc, neb_images):
        """intermediate_minima_check_step=5, 50 steps."""
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(
            intermediate_minima_check_step=5,
        )
        config["Main"]["steps"] = 50
        images = self._prepare_images(neb_images, fairchem_calc)

        nebopt(0, config, images, fairchem_calc, (MDMin, MDMin),
               consecutive_errors=[0], executorlib_worker_id=0)

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) >= 1

        with self._read_output_traj(tmp_path) as traj:
            # Band should have at least 10 images
            assert len(traj) >= 10
            types = [traj[idx].info["image_type"] for idx in range(len(traj))]
            # Should have at least endpoints and climbing images
            assert "endpoint" in types
            assert "climbing" in types

    # ----- Test: band plot creation -----

    def test_neb_band_plot_created(self, tmp_path, monkeypatch, fairchem_calc, neb_images):
        """Run basic NEB, check diffusion_barrier_0.png exists."""
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config()
        images = self._prepare_images(neb_images, fairchem_calc)

        nebopt(0, config, images, fairchem_calc, (MDMin, MDMin),
               consecutive_errors=[0], executorlib_worker_id=0)

        # The plot file is created as a temp file during the run.
        # With zip=False, it should remain on disk. If it was cleaned up,
        # at least verify the run completed (CSV exists).
        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) >= 1
        # With zip=False (set in make_config_dict), temp files are NOT cleaned up
        plot_path = tmp_path / "diffusion_barrier_0.png"
        assert plot_path.exists(), "Band plot PNG was not created"

    # ----- Test: dimer CI refinement -----

    def test_neb_dimer_refine_ci(self, tmp_path, monkeypatch, fairchem_calc, neb_images):
        """dimer_refine_ci=True, dimer_refine_steps=20, 30 NEB steps."""
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(
            dimer_refine_ci=True,
            dimer_refine_steps=20,
        )
        images = self._prepare_images(neb_images, fairchem_calc)

        nebopt(0, config, images, fairchem_calc, (MDMin, MDMin),
               consecutive_errors=[0], executorlib_worker_id=0)

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) >= 1

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) >= 10
            # Verify climbing images have eigenmode
            for idx in range(len(traj)):
                frame = traj[idx]
                if frame.info["image_type"] == "climbing":
                    assert "eigenmode" in frame.info
                    assert "barrier" in frame.info
                    assert "dE" in frame.info
