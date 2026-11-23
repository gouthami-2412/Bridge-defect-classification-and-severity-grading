"""Gradio demo: upload a bridge photo, get the marked defects and a severity report.

    python app.py                      # local
    python app.py --share              # temporary public link
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import argparse
import json

import gradio as gr

from bridge_inspect.model import COLORS  # noqa: F401  (colour legend below)
from bridge_inspect.model import DefectModel

ap = argparse.ArgumentParser()
ap.add_argument("--weights-dir", default="weights")
ap.add_argument("--thresholds", default="configs/thresholds_ens_flip.json")
ap.add_argument("--share", action="store_true")
args = ap.parse_args()

model = DefectModel(args.weights_dir, args.thresholds)


def run(img):
    if img is None:
        return None, "Upload a bridge photo first."
    overlay, report = model.analyze(img)
    return overlay, json.dumps(report, indent=2)


gr.Interface(
    run, gr.Image(type="numpy", label="Bridge photo"),
    [gr.Image(label="Detected defects"), gr.Textbox(lines=12, label="Severity report")],
    title="Bridge defect analyzer",
    description="Colors: Crack red, Spalling green, Rust blue, Efflorescence yellow, "
                "Cavity magenta, ExposedRebars cyan. Aid to inspection, not a replacement for an engineer.",
).launch(share=args.share)
