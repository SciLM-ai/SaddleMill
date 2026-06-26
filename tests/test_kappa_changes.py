"""Pytest checks for the kappa-engine / force-call / per-type-count changes.

CPU + EMT only (no GPU, no FAIRChem) — collected and run by the normal suite.
Replaces the earlier standalone script; no module-level execution, no sys.exit,
so pytest collection is safe.
"""
import numpy as np
import pytest

from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.mep.dimer import MinModeAtoms, DimerControl

from saddlemill.dimertools.structure_edit import (
    _resolve_attempts_per_type, _swap_prob, _attempt_count_max)
from saddlemill.dimertools.kappa_dimer import KappaMinModeAtoms
from saddlemill.dimeropt import _setup_dimer


def make_atoms(seed=1):
    a = fcc111('Al', size=(2, 2, 2), vacuum=6.0)
    add_adsorbate(a, 'Al', 1.6, 'fcc')
    a.rattle(0.05, seed=seed)
    a.calc = EMT()
    return a


RT = ["vacancy", "ring", "hop_reuse"]


# --- A. structure_edit helpers ---

def test_attempts_per_type_uniform_int():
    assert _resolve_attempts_per_type(4, RT) == [4, 4, 4]

def test_attempts_per_type_none_defaults_to_one():
    assert _resolve_attempts_per_type(None, RT) == [1, 1, 1]

def test_attempts_per_type_list_aligned():
    assert _resolve_attempts_per_type([10, 2, 5], RT) == [10, 2, 5]

def test_attempts_per_type_length_mismatch_raises():
    with pytest.raises(ValueError):
        _resolve_attempts_per_type([1, 2], RT)

def test_swap_prob_default_none():
    assert _swap_prob(None) == 0.1

def test_swap_prob_reads_config():
    assert _swap_prob({"ourDimer": {"gaussian_swap_prob": 0.0}}) == 0.0

def test_swap_prob_missing_key_defaults():
    assert _swap_prob({"ourDimer": {}}) == 0.1

def test_attempt_count_max_list():
    assert _attempt_count_max([10, 2, 5]) == 10

def test_attempt_count_max_int():
    assert _attempt_count_max(7) == 7


# --- B. KappaMinModeAtoms construction ---

def test_kappa_control_routed_not_into_beta():
    ctrl = DimerControl(max_num_rot=4, dimer_separation=0.01, logfile=None)
    k = KappaMinModeAtoms(make_atoms(), control=ctrl, beta=5.0, recover_fmax=0.3)
    assert k.control is ctrl          # control landed in the right slot, not beta
    assert k.beta == 5.0
    assert k.recover_fmax == 0.3

def test_kappa_default_control_built():
    ctrl = DimerControl(max_num_rot=4, dimer_separation=0.01, logfile=None)
    k = KappaMinModeAtoms(make_atoms(), control=ctrl, beta=5.0, recover_fmax=0.3)
    assert k.kappa_control is not None


# --- C. _setup_dimer engine dispatch + force-call counter ---

def test_engine_kappa_builds_kappa_and_counts():
    d, rlx = _setup_dimer(
        make_atoms(), EMT(),
        dimer_control_kwargs={"max_num_rot": 4, "dimer_separation": 0.01},
        engine="kappa",
        kappa_kwargs={"beta": 5.0, "recover_fmax": 0.3},
        kappa_control_kwargs=None,
    )
    assert isinstance(d, KappaMinModeAtoms)
    rlx.run(fmax=0.2, steps=8)
    assert d.control.get_counter('forcecalls') > 0

def test_engine_ase_builds_plain_minmode():
    d, _ = _setup_dimer(
        make_atoms(), EMT(),
        dimer_control_kwargs={"max_num_rot": 4},
        engine="ase",
    )
    assert isinstance(d, MinModeAtoms)
    assert not isinstance(d, KappaMinModeAtoms)

def test_default_engine_is_ase():
    d, _ = _setup_dimer(make_atoms(), EMT(),
                        dimer_control_kwargs={"max_num_rot": 4})
    assert isinstance(d, MinModeAtoms)
    assert not isinstance(d, KappaMinModeAtoms)

def test_kappa_passthrough_custom_control():
    d, _ = _setup_dimer(
        make_atoms(), EMT(),
        dimer_control_kwargs={"max_num_rot": 4},
        engine="kappa",
        kappa_kwargs={"beta": 5.0, "recover_fmax": 0.3},
        kappa_control_kwargs={"f_rot_min": 0.02, "max_num_rot": 6},
    )
    assert d.kappa_control.get_parameter('max_num_rot') == 6

def test_unknown_engine_raises():
    with pytest.raises(ValueError):
        _setup_dimer(make_atoms(), EMT(),
                     dimer_control_kwargs={"max_num_rot": 4}, engine="banana")
