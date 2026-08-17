"""
Package Face Detectors cho hệ thống chấm công.
"""

from src.detectors.detector_factory import (
    BaseDetector,
    FaceBox,
    RetinaFaceDetector,
    MediaPipeDetector,
    YOLOv8Detector,
    get_detector,
)

__all__ = [
    "BaseDetector",
    "FaceBox",
    "RetinaFaceDetector",
    "MediaPipeDetector",
    "YOLOv8Detector",
    "get_detector",
]
