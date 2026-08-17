"""
Unit tests cho detector_factory và embedder_factory.
"""

import os
import cv2
import numpy as np
import pytest

from src.detectors.detector_factory import (
    FaceBox,
    MediaPipeDetector,
    get_detector,
)
from src.embedders.embedder_factory import (
    ArcFaceEmbedder,
    Facenet512Embedder,
    SFaceEmbedder,
    get_embedder,
)


def test_face_box_properties():
    box = FaceBox(x=10, y=20, w=100, h=120, confidence=0.95)
    assert box.bbox == (10, 20, 100, 120)
    assert box.xyxy == (10, 20, 110, 140)

    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = box.get_crop(dummy_img, margin=0.1)
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_mediapipe_detector_on_real_image():
    img_path = "data/test_images/img_1.png"
    if not os.path.exists(img_path):
        pytest.skip(f"{img_path} không tồn tại")

    image = cv2.imread(img_path)
    detector = MediaPipeDetector()
    boxes = detector.detect(image)

    assert len(boxes) >= 1
    assert boxes[0].confidence > 0.5
    assert boxes[0].w > 50 and boxes[0].h > 50


def test_detector_factory():
    mp_det = get_detector("mediapipe")
    assert isinstance(mp_det, MediaPipeDetector)

    retina_det = get_detector("retinaface")
    assert retina_det is not None

    with pytest.raises(ValueError):
        get_detector("unknown_detector")


def test_embedder_factory():
    arc = get_embedder("arcface")
    assert isinstance(arc, ArcFaceEmbedder)
    assert arc.embedding_dim == 512

    facenet = get_embedder("facenet512")
    assert isinstance(facenet, Facenet512Embedder)
    assert facenet.embedding_dim == 512

    sface = get_embedder("sface")
    assert isinstance(sface, SFaceEmbedder)
    assert sface.embedding_dim == 128

    with pytest.raises(ValueError):
        get_embedder("unknown_embedder")
