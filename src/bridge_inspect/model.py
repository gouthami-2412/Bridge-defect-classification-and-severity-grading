"""Final model: FPN + DeepLabV3+ (ResNet34), each averaged with its horizontal flip,
then the two averaged and thresholded per class. Requires torch + segmentation-models-pytorch."""
import json
from pathlib import Path

import cv2
import numpy as np

from .severity import SIZE, TARGET, severity_report

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


class DefectModel:
    def __init__(self, weights_dir="weights", thresholds="configs/thresholds_ens_flip.json", device=None):
        import torch
        import segmentation_models_pytorch as smp
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        wd = Path(weights_dir)
        self.nets = []
        for arch, ckpt in ((smp.FPN, "best_fpn.pth"), (smp.DeepLabV3Plus, "best_deeplab.pth")):
            net = arch(encoder_name="resnet34", encoder_weights=None, in_channels=3,
                       classes=len(TARGET), activation=None).to(self.dev)
            net.load_state_dict(torch.load(wd / ckpt, map_location=self.dev))
            self.nets.append(net.eval())
        thr = json.load(open(thresholds))
        self.thr = np.array([thr[t] for t in TARGET])

    def predict_masks(self, img_rgb):
        """img_rgb: HxWx3 uint8 RGB -> (resized 512x512 image, (6,512,512) uint8 masks)."""
        torch = self.torch
        small = cv2.resize(img_rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(((small / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(self.dev)
        probs = []
        with torch.no_grad(), torch.autocast("cuda", enabled=(self.dev == "cuda")):
            for net in self.nets:
                p = torch.sigmoid(net(x))[0].float()
                pf = torch.flip(torch.sigmoid(net(torch.flip(x, [3]))), [3])[0].float()
                probs.append((p + pf) / 2)
        avg = ((probs[0] + probs[1]) / 2).cpu().numpy()
        return small, (avg > self.thr[:, None, None]).astype(np.uint8)

    def analyze(self, img_rgb):
        """Returns (overlay image, severity report dict)."""
        small, masks = self.predict_masks(img_rgb)
        ov = small.copy()
        for c in range(len(TARGET)):
            ov[masks[c] > 0] = COLORS[c]
        return cv2.addWeighted(small, 0.55, ov, 0.45, 0), severity_report(masks)
