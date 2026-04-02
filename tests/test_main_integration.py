"""End-to-end tests for tsearch/__main__.py — full serial pipeline."""

import pytest
import os
import shutil
import glob
from pathlib import Path
from ase.io import read, write, Trajectory

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _write_config(tmp_path, content):
    (tmp_path / "config.ini").write_text(content)


@pytest.mark.gpu
@pytest.mark.slow
class TestMainPipeline:
    """End-to-end tests running the full main() in serial mode."""

    def test_serial_minimization_pipeline(self, tmp_path, monkeypatch):
        """Run Minimization end-to-end: config -> main() -> verify outputs."""
        monkeypatch.chdir(tmp_path)

        # Setup input
        traj_dir = tmp_path / "trajes"
        traj_dir.mkdir()
        atoms = read(str(FIXTURES_DIR / "minimization_input.traj"))
        write(str(traj_dir / "input.traj"), atoms)

        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = Minimization
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.05
steps = 50
zip = False

[FAIRChemCalculator]
device = cuda
name_or_path = uma-s-1p1
task_name = oc20

[MDMin]
dt = 0.05
maxstep = 0.1
""")

        from tsearch.__main__ import main
        main()

        # Verify outputs
        assert os.path.exists("traj_files_ordered.json")
        assert os.path.isdir("Minimization_status_csvs")
        assert os.path.isdir("Minimization_trajes")

        csv_files = glob.glob("Minimization_status_csvs/*.csv")
        assert len(csv_files) >= 1

        traj_files = glob.glob("Minimization_trajes/*.traj")
        assert len(traj_files) >= 1

        # Check output trajectory content
        result = read(traj_files[0])
        assert "converged" in result.info
        assert "src_index" in result.info

    def test_serial_neb_pipeline(self, tmp_path, monkeypatch):
        """Run NEB end-to-end with 10-frame input."""
        monkeypatch.chdir(tmp_path)

        traj_dir = tmp_path / "trajes"
        traj_dir.mkdir()
        shutil.copy(str(FIXTURES_DIR / "oc_neb_pair.traj"), str(traj_dir / "neb_input.traj"))

        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = NEB
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.05
steps = 30
zip = False

[FAIRChemCalculator]
device = cuda
name_or_path = uma-s-1p1
task_name = oc20

[MDMin]
dt = 0.05
maxstep = 0.1

[ourNEB]
relax_endpoints = False
interpolate_method = False
num_frames = 10
batch_size = 4
DNEB = False
intermediate_minima = False

[BaseNEB]
k = 5
climb = True
method = improvedtangent
allow_shared_calculator = True
""")

        from tsearch.__main__ import main
        main()

        assert os.path.exists("traj_files_ordered.json")
        csv_files = glob.glob("NEB_status_csvs/*.csv")
        assert len(csv_files) >= 1
        traj_files = glob.glob("NEB_trajes/*.traj")
        assert len(traj_files) >= 1

        # Check output has all band images
        results = read(traj_files[0], index=":")
        assert len(results) >= 2  # at least reactant + product
        for img in results:
            assert "src_index" in img.info
            assert "image_idx" in img.info

    def test_serial_dimer_pipeline(self, tmp_path, monkeypatch):
        """Run Dimer end-to-end with bulk crystal."""
        monkeypatch.chdir(tmp_path)

        traj_dir = tmp_path / "trajes"
        traj_dir.mkdir()
        shutil.copy(str(FIXTURES_DIR / "bulk_crystal.traj"), str(traj_dir / "bulk.traj"))

        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = Dimer
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.05
steps = 30
zip = False

[FAIRChemCalculator]
device = cuda
name_or_path = uma-s-1p1
task_name = oc20

[MDMin]
dt = 0.05
maxstep = 0.1

[ourDimer]
dataset_type = bulk
reaction_types = vacancy
num_attempts_per_type = 1
supercell = True

[DimerControl]
initial_eigenmode_method = displacement
maximum_translation = 0.1
dimer_separation = 0.01
""")

        from tsearch.__main__ import main
        main()

        assert os.path.exists("traj_files_ordered.json")
        csv_files = glob.glob("Dimer_status_csvs/*.csv")
        assert len(csv_files) >= 1
        traj_files = glob.glob("Dimer_trajes/*.traj")
        assert len(traj_files) >= 1

        result = read(traj_files[0])
        assert "src_index" in result.info
        assert "attempt_id" in result.info
        assert "reaction_type" in result.info

    def test_resume_logic(self, tmp_path, monkeypatch):
        """Run once (likely not_converged), then resume with run_jobs=not_converged."""
        monkeypatch.chdir(tmp_path)

        traj_dir = tmp_path / "trajes"
        traj_dir.mkdir()
        atoms = read(str(FIXTURES_DIR / "minimization_input.traj"))
        write(str(traj_dir / "input.traj"), atoms)

        # First run: very few steps, likely not converged
        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = Minimization
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.001
steps = 5
zip = False

[FAIRChemCalculator]
device = cuda
name_or_path = uma-s-1p1
task_name = oc20

[MDMin]
dt = 0.05
maxstep = 0.1
""")

        from tsearch.__main__ import main
        main()

        assert os.path.exists("traj_files_ordered.json")
        csv_files_1 = glob.glob("Minimization_status_csvs/*.csv")
        assert len(csv_files_1) >= 1

        # Read first run status
        with open(csv_files_1[0], 'r') as f:
            first_run_lines = f.readlines()
        assert len(first_run_lines) >= 1

        # Second run: resume not_converged with more steps
        _write_config(tmp_path, f"""\
[Main]
executorlib = False
method = Minimization
dir_path = {traj_dir}
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.05
steps = 50
zip = False
run_jobs = not_converged

[FAIRChemCalculator]
device = cuda
name_or_path = uma-s-1p1
task_name = oc20

[MDMin]
dt = 0.05
maxstep = 0.1
""")

        main()

        # Verify archive was created
        archives = glob.glob("Minimization_status_csvs/previous_*.zip")
        assert len(archives) >= 1, "Archive should have been created on resume"
