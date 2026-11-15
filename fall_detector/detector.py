"""Person-down detector: finds people lying on the ground or floor in CCTV-style frames.

Two classes: 0 = person (upright: standing, walking, sitting, bending, kneeling), 1 = fallen_person (on the floor,
not holding themselves up). The frame score is the highest fallen_person score in the frame.

    from fall_detector import FallDetector
    det = FallDetector()                       # Nano (recommended), PyTorch, from the Hugging Face Hub
    det = FallDetector(variant="small")        # Small weights
    det = FallDetector(onnx=True)              # ONNX Runtime, no PyTorch at inference time
    dets = det(Image.open("frame.jpg"))              # [Detection(box, score, label), ...]
    score = det.frame_score(Image.open("frame.jpg"))
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ID = "Simuletic/fall-detector"
SUBFOLDER = {"small": "small", "nano": "nano"}
FALLEN = 1


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    score: float
    label: str  # "person" or "fallen_person"


def _device(requested: str | None) -> str:
    import torch
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FallDetector:
    def __init__(self, variant: str = "nano", weights: str | Path | None = None, onnx: bool = False,
                 device: str | None = None, threshold: float = 0.05):
        """weights: a local folder with config.json + model.safetensors (+ model.onnx); default downloads from the Hub."""
        self.threshold = threshold
        self.onnx = onnx
        if weights is None:
            from huggingface_hub import snapshot_download
            weights = Path(snapshot_download(REPO_ID, allow_patterns=[f"{SUBFOLDER[variant]}/*"])) / SUBFOLDER[variant]
        self.weights = Path(weights)
        if onnx:
            import onnxruntime as ort
            # CUDA if available, else CPU. CoreML is skipped: it mis-handles one of D-FINE's reshape ops.
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
            self.session = ort.InferenceSession(str(self.weights / "model.onnx"), providers=providers)
            self.size = 640
        else:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            self.torch = torch
            self.device = _device(device)
            if self.device == "mps":
                from . import mps_patches  # noqa: F401  (training-time MPS fix; harmless at inference)
            self.processor = AutoImageProcessor.from_pretrained(self.weights)
            self.model = AutoModelForObjectDetection.from_pretrained(self.weights).to(self.device).eval()
        self.names = {0: "person", 1: "fallen_person"}

    # ---- inference ----
    def __call__(self, image: Image.Image, threshold: float | None = None) -> list[Detection]:
        thr = self.threshold if threshold is None else threshold
        image = image.convert("RGB")
        if self.onnx:
            return self._onnx(image, thr)
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        r = self.processor.post_process_object_detection(out, target_sizes=[(image.height, image.width)], threshold=thr)[0]
        return [Detection(tuple(b.tolist()), float(s), self.names[int(l)])
                for b, s, l in zip(r["boxes"].cpu(), r["scores"].cpu(), r["labels"].cpu())]

    def _onnx(self, image: Image.Image, thr: float) -> list[Detection]:
        x = np.asarray(image.resize((self.size, self.size), Image.BILINEAR), dtype=np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        logits, boxes = self.session.run(None, {"pixel_values": x})
        probs = 1.0 / (1.0 + np.exp(-logits[0]))  # [Q, 2], sigmoid per class as in D-FINE post-processing
        W, H = image.size
        dets = []
        for q in range(probs.shape[0]):
            for c in (0, 1):
                s = float(probs[q, c])
                if s < thr:
                    continue
                cx, cy, w, h = boxes[0, q]
                dets.append(Detection(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H), s, self.names[c]))
        dets.sort(key=lambda d: -d.score)
        return dets[:300]

    def frame_score(self, image: Image.Image) -> float:
        """Highest fallen_person score in the frame (0 if none)."""
        return max((d.score for d in self(image, threshold=0.01) if d.label == "fallen_person"), default=0.0)
