"""
Unit tests cho module tiền xử lý (clahe.py) và augmentation (augmentation.py).
"""

import numpy as np
import pytest
from src.preprocessing.clahe import (
    apply_clahe,
    apply_denoise,
    apply_sharpen,
    preprocess_image,
)
from src.preprocessing.augmentation import (
    simulate_low_light,
    simulate_backlight,
    simulate_pose_variation,
    simulate_sensor_noise,
    generate_hard_case_suite,
)


@pytest.fixture
def dummy_face_image():
    """Tạo một ảnh BGR giả lập kích thước 120x120 để test nhanh."""
    np.random.seed(42)
    img = np.random.randint(60, 200, (120, 120, 3), dtype=np.uint8)
    return img


def test_apply_clahe(dummy_face_image):
    enhanced = apply_clahe(dummy_face_image, clip_limit=2.0, tile_grid_size=(8, 8))
    assert enhanced is not None
    assert enhanced.shape == dummy_face_image.shape
    assert enhanced.dtype == np.uint8


def test_apply_denoise(dummy_face_image):
    denoised = apply_denoise(dummy_face_image, method="bilateral")
    assert denoised is not None
    assert denoised.shape == dummy_face_image.shape


def test_apply_sharpen(dummy_face_image):
    sharpened = apply_sharpen(dummy_face_image, strength=0.5)
    assert sharpened is not None
    assert sharpened.shape == dummy_face_image.shape


def test_preprocess_image_pipeline(dummy_face_image):
    config = {
        "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
        "denoise": True,
        "sharpen": True,
    }
    processed = preprocess_image(dummy_face_image, config=config)
    assert processed is not None
    assert processed.shape == dummy_face_image.shape


def test_augmentations(dummy_face_image):
    low_light = simulate_low_light(dummy_face_image)
    assert low_light.shape == dummy_face_image.shape

    backlight = simulate_backlight(dummy_face_image)
    assert backlight.shape == dummy_face_image.shape

    pose = simulate_pose_variation(dummy_face_image, angle=10)
    assert pose.shape == dummy_face_image.shape

    noise = simulate_sensor_noise(dummy_face_image)
    assert noise.shape == dummy_face_image.shape

    suite = generate_hard_case_suite(dummy_face_image)
    assert len(suite) == 6
    for name, img in suite.items():
        assert img.shape == dummy_face_image.shape
