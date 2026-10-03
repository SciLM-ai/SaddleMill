from __future__ import annotations

import configparser
from pathlib import Path

from saddlemill.sella_ablation import (
    PARTITION_APPROXIMATE_B,
    QN_NEWTON_SAFE_FALSE,
    TRANSLATION_PRFO,
    TRUST_HOLD,
    TRUST_ISOLATION,
    TRUST_NATIVE,
    resolve_sella_ablation,
)

ROOT = Path(__file__).resolve().parents[2]
OVERRIDES = ROOT / "final35" / "architecture_overrides"


def _ini(arm: str):
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read(OVERRIDES / f"{arm}.override.ini")
    return parser


def test_sel03_fixed_trust_is_prfo_hold_only_not_qn_policy():
    spec = resolve_sella_ablation(TRUST_ISOLATION)
    assert spec.translation == TRANSLATION_PRFO
    assert spec.partition == PARTITION_APPROXIMATE_B
    assert spec.trust == TRUST_HOLD
    assert spec.qn_newton_safe is None
    assert spec.component_matrix()["trust_adaptation"] == TRUST_HOLD
    assert spec.component_matrix()["translation"] == TRANSLATION_PRFO

    sel03 = _ini("SEL03M24_SELLA_PRFO_FIXED_TRUST")
    assert sel03["ourDimer"]["engine"] == "sella"
    assert sel03["ourSella"]["method"] == "prfo"
    assert sel03["ourSella"]["delta0"] == "0.1"
    assert sel03["ourSella"]["ablation"] == TRUST_ISOLATION
    assert QN_NEWTON_SAFE_FALSE not in sel03["ourSella"].values()


def test_only_sel04_selects_qn_newton_safe_false():
    expected = {
        "SEL00M24_SELLA_PRFO": ("prfo", None),
        "SEL03M24_SELLA_PRFO_FIXED_TRUST": ("prfo", TRUST_ISOLATION),
        "SEL04M24_SELLA_QN": ("qn", QN_NEWTON_SAFE_FALSE),
        "SEL06M24_SELLA_RFO": ("rfo", None),
    }
    for arm, (method, ablation) in expected.items():
        cfg = _ini(arm)
        assert cfg["ourDimer"]["engine"] == "sella"
        assert cfg["ourSella"]["method"] == method
        actual = cfg["ourSella"].get("ablation")
        assert actual == ablation

    qn_spec = resolve_sella_ablation(QN_NEWTON_SAFE_FALSE)
    assert qn_spec.qn_newton_safe is False
    assert qn_spec.trust == TRUST_NATIVE
    assert qn_spec.partition == PARTITION_APPROXIMATE_B
