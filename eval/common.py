"""Shared helpers for the evaluation scripts. Metric code is identical to what produced the model-card numbers."""
import numpy as np


def ap(scores, labels):
    """Average precision (area under the precision-recall curve, step-wise)."""
    o = np.argsort(-np.asarray(scores)); l = np.asarray(labels)[o]; tp = np.cumsum(l); fp = np.cumsum(1 - l)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(l.sum(), 1)
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))


def load_detector(args):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from fall_detector import FallDetector
    return FallDetector(args.variant, args.weights, onnx=args.onnx)


def add_model_args(p):
    p.add_argument("--variant", default="nano", choices=["small", "nano"])
    p.add_argument("--weights", help="local model folder instead of the Hub")
    p.add_argument("--onnx", action="store_true")
    p.add_argument("--name", default="model")
