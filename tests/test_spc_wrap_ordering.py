"""Tests for the SinglePointCalculator + atoms.wrap() ordering bug.

Building an SPC and then calling atoms.wrap() leaves the SPC's cached positions
out of sync with the live ones (sub-1e-15 FP drift from the cartesian↔fractional
roundtrip in wrap()). When the trajectory writer later asks the SPC for energy
or forces, SPC.check_state() raises PropertyNotImplementedError and the writer
silently records the frame with no energy/forces. The fix is to construct the
SPC AFTER wrap() (or equivalently: read energy/forces, wrap, then build SPC).
"""

import os

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory


def _make_atoms(displace=0.0):
    atoms = bulk('Cu', 'fcc', a=3.6).repeat((2, 2, 2))
    atoms.calc = EMT()
    if displace != 0.0:
        atoms.positions[0, 0] += displace
    return atoms


def _write_then_read(atoms, tmp_path):
    fp = os.path.join(str(tmp_path), 'frame.traj')
    with Trajectory(fp, 'w') as w:
        w.write(atoms)
    with Trajectory(fp, 'r') as r:
        return next(iter(r))


def test_spc_before_wrap_drops_energy_when_wrap_moves_atoms(tmp_path):
    atoms = _make_atoms(displace=10.0)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=atoms.get_potential_energy(),
        forces=atoms.get_forces(),
    )
    atoms.wrap()  # buggy ordering: SPC cached pre-wrap positions
    out = _write_then_read(atoms, tmp_path)
    assert 'energy' not in out.calc.results
    assert 'forces' not in out.calc.results


def test_spc_after_wrap_preserves_energy_when_wrap_moves_atoms(tmp_path):
    atoms = _make_atoms(displace=10.0)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces().copy()
    atoms.wrap()
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    out = _write_then_read(atoms, tmp_path)
    assert out.calc.results['energy'] == energy
    assert np.array_equal(out.calc.results['forces'], forces)


def test_spc_after_wrap_unchanged_when_atoms_already_in_cell(tmp_path):
    atoms = _make_atoms(displace=0.0)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces().copy()
    positions_before = atoms.positions.copy()
    atoms.wrap()
    assert np.max(np.abs(atoms.positions - positions_before)) < 1e-10
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    out = _write_then_read(atoms, tmp_path)
    assert out.calc.results['energy'] == energy
    assert np.array_equal(out.calc.results['forces'], forces)


def test_spc_after_wrap_values_match_pre_wrap_calc(tmp_path):
    atoms_ref = _make_atoms(displace=10.0)
    energy_ref = atoms_ref.get_potential_energy()
    forces_ref = atoms_ref.get_forces().copy()

    atoms = _make_atoms(displace=10.0)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces().copy()
    atoms.wrap()
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    out = _write_then_read(atoms, tmp_path)

    assert out.calc.results['energy'] == energy_ref
    assert np.array_equal(out.calc.results['forces'], forces_ref)
