"""Rule-based severity grading from predicted defect masks.

Frozen rules from the report (section 5.4). Crack length is the skeleton length divided by the
image width (512 px); widths are in pixels because the photos have no real-world scale.
Only needs numpy, opencv and scikit-image (no deep-learning dependencies).
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

TARGET = ["Crack", "Spalling", "Rust", "Efflorescence", "Cavity", "ExposedRebars"]
SIZE = 512
FLOOR_FRAC = 0.25                    # a defect counts once it reaches 25% of its "moderate" cut
BUMP_W = 6                           # crack wider than this (px) is raised one level
CRACK_LENGTH_CUTS = (0.3, 1.0)       # moderate, severe (in image widths)
AREA_CUTS = {"Spalling": (0.01, 0.05), "Rust": (0.02, 0.08), "Efflorescence": (0.03, 0.10),
             "Cavity": (0.005, 0.02), "ExposedRebars": (0.002, 0.01)}
MIN_PX = {"Crack": 20, "Spalling": 150, "Rust": 150, "Efflorescence": 150, "Cavity": 15, "ExposedRebars": 60}
STRUCTURAL = {"Crack", "Spalling", "Cavity", "ExposedRebars"}   # can reach Critical
AREA_NAMES = ["Spalling", "Rust", "Efflorescence", "Cavity", "ExposedRebars"]
CONDITIONS = ["Good", "Fair", "Poor", "Critical"]


def clean_mask(m, min_px):
    """Remove connected components smaller than min_px pixels."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    out = np.zeros(m.shape, np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_px:
            out[lab == i] = 1
    return out


def grade(v, cuts):
    """0 none, 1 minor, 2 moderate, 3 severe."""
    return 0 if v <= 0 else 1 if v < cuts[0] else 2 if v < cuts[1] else 3


def crack_metrics(m):
    """(skeleton length / image width, mean width in px)."""
    if m.sum() == 0:
        return 0.0, 0.0
    skel = skeletonize(m > 0)
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    length = skel.sum()
    return length / SIZE, (2 * dist[skel].mean() if length else 0.0)


def measures(masks):
    """masks: (6, H, W) binary array in TARGET order -> dict of measurements."""
    out = {}
    for c, name in enumerate(TARGET):
        m = clean_mask(masks[c], MIN_PX[name])
        if name == "Crack":
            out["crack_L"], out["crack_W"] = crack_metrics(m)
        else:
            out[name] = float(m.mean())
    return out


def overall_grade(me, bump_w=BUMP_W, floor_frac=FLOOR_FRAC):
    """Overall grade 0-3 (Good, Fair, Poor, Critical) = highest contribution of any defect."""
    g = grade(me["crack_L"], CRACK_LENGTH_CUTS)
    if g > 0 and me["crack_W"] > bump_w:
        g = min(3, g + 1)
    contrib = [g]
    for n in AREA_NAMES:
        v = me[n]
        gg = 0 if v < floor_frac * AREA_CUTS[n][0] else grade(v, AREA_CUTS[n])
        contrib.append(gg if n in STRUCTURAL else min(gg, 2))   # surface defects capped at Poor
    return max(contrib)


def severity_report(masks):
    me = measures(masks)
    g = overall_grade(me)
    rep = {"overall": CONDITIONS[g], "crack_length": round(float(me["crack_L"]), 2),
           "crack_width_px": round(float(me["crack_W"]), 1)}
    rep.update({f"{n}_area_%": round(float(me[n]) * 100, 2) for n in AREA_NAMES})
    return rep
