"""Tests for saddlemill/init_function.py — worker initialization."""

import pytest
import os
from pathlib import Path

from tests.conftest import make_config_dict


@pytest.mark.gpu
class TestInitFunction:
    """Test init_function with a real config.ini and FAIRChem calculator."""

    def _write_config(self, tmp_path):
        """Write a minimal config.ini for FAIRChem serial mode."""
        config_content = """\
[Main]
executorlib = False
method = Minimization
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.05
steps = 10

[FAIRChemCalculator]
device = cuda
name_or_path = uma-s-1p2
task_name = oc20

[MDMin]
dt = 0.05
maxstep = 0.1
"""
        (tmp_path / "config.ini").write_text(config_content)

    def test_returns_required_keys(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_config(tmp_path)

        from saddlemill.init_function import init_function
        result = init_function()

        assert "calc" in result
        assert "Optimizer" in result
        assert "consecutive_errors" in result

    def test_consecutive_errors_is_mutable_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_config(tmp_path)

        from saddlemill.init_function import init_function
        result = init_function()

        assert result["consecutive_errors"] == [0]
        assert isinstance(result["consecutive_errors"], list)
        # Verify it's mutable
        result["consecutive_errors"][0] = 5
        assert result["consecutive_errors"][0] == 5

    def test_calc_is_fairchem_instance(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_config(tmp_path)

        from saddlemill.init_function import init_function
        result = init_function()

        # For FAIRChemCalculator, init_function instantiates the calculator
        # (not a class). Verify it has the predict-like interface.
        calc = result["calc"]
        # FAIRChemCalculator instances have a 'predictor' attribute
        assert hasattr(calc, "calculate") or hasattr(calc, "predictor")

    def test_optimizer_is_class(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_config(tmp_path)

        from saddlemill.init_function import init_function
        result = init_function()

        from ase.optimize import MDMin
        assert result["Optimizer"] is MDMin

    def test_neb_returns_optimizer_tuple(self, tmp_path, monkeypatch):
        """For NEB method, load_optimizer returns a tuple of (endpoint_opt, neb_opt)."""
        monkeypatch.chdir(tmp_path)
        config_content = """\
[Main]
executorlib = False
method = NEB
Calculator = FAIRChemCalculator
Optimizer = MDMin
fmax = 0.05
steps = 10

[FAIRChemCalculator]
device = cuda
name_or_path = uma-s-1p2
task_name = oc20

[MDMin]
dt = 0.05
maxstep = 0.1
"""
        (tmp_path / "config.ini").write_text(config_content)

        from saddlemill.init_function import init_function
        result = init_function()

        assert isinstance(result["Optimizer"], tuple)
        assert len(result["Optimizer"]) == 2
