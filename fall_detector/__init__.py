"""Simuletic Fall Detector: person lying on the ground or floor, trained only on synthetic images."""
from .alarm import StaysDownAlarm
from .detector import Detection, FallDetector

__all__ = ["FallDetector", "Detection", "StaysDownAlarm"]
__version__ = "1.0.0"
