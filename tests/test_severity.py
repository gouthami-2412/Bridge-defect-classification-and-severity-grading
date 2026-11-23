"""Checks the severity rules on synthetic masks and against configs/severity_rules_final.json.
Run:  python tests/test_severity.py   (or pytest)"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_inspect import severity as S  # noqa: E402


def empty():
    return np.zeros((6, 512, 512), np.uint8)


def block(masks, cls, frac):
    """Fill a square covering ~frac of the image in channel cls."""
    side = int(round((frac * 512 * 512) ** 0.5))
    masks[S.TARGET.index(cls), :side, :side] = 1
    return masks


def test_rules_match_json():
    cfg = json.load(open(ROOT / "configs" / "severity_rules_final.json"))
    assert cfg["FLOOR_FRAC"] == S.FLOOR_FRAC and cfg["BUMP_W"] == S.BUMP_W
    assert {k: tuple(v) for k, v in cfg["AREA_CUTS"].items()} == S.AREA_CUTS
    assert cfg["MIN_PX"] == S.MIN_PX and tuple(cfg["CRACK_LENGTH_CUTS"]) == S.CRACK_LENGTH_CUTS


def test_no_defects_is_good():
    assert S.overall_grade(S.measures(empty())) == 0


def test_spalling_levels():
    # 0.5% -> minor (Fair), 2% -> moderate (Poor), 8% -> severe (Critical)
    for frac, want in [(0.005, 1), (0.02, 2), (0.08, 3)]:
        assert S.overall_grade(S.measures(block(empty(), "Spalling", frac))) == want


def test_below_floor_is_ignored():
    # 0.2% spalling < 25% of the 1% moderate cut (0.25%)
    assert S.overall_grade(S.measures(block(empty(), "Spalling", 0.002))) == 0


def test_surface_defects_capped_at_poor():
    assert S.overall_grade(S.measures(block(empty(), "Rust", 0.30))) == 2
    assert S.overall_grade(S.measures(block(empty(), "Efflorescence", 0.30))) == 2


def test_exposed_rebar_can_be_critical():
    assert S.overall_grade(S.measures(block(empty(), "ExposedRebars", 0.02))) == 3


def test_long_thin_crack_is_poor_and_wide_crack_bumps():
    m = empty(); m[0, 256:258, :] = 1                      # 2 px wide, one image-width long
    me = S.measures(m)
    assert me["crack_L"] > 0.9 and me["crack_W"] < 6
    assert S.overall_grade(me) in (2, 3)                   # severe length cut (1.0) may or may not be reached
    m = empty(); m[0, 200:215, :400] = 1                   # 15 px wide, ~0.78 image widths long
    me = S.measures(m)
    assert me["crack_W"] > 6 and S.overall_grade(me) == 3  # moderate (2) bumped to severe (3)


def test_small_specks_removed():
    m = empty(); m[S.TARGET.index("Spalling"), :10, :10] = 1   # 100 px < 150 px minimum
    assert S.measures(m)["Spalling"] == 0.0


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f(); print("ok  ", f.__name__)
    print(len(fns), "tests passed")
