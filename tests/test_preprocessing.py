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
    # Test default clip_limit=1.5
    enhanced = apply_clahe(dummy_face_image)
    assert enhanced is not None
    assert enhanced.shape == dummy_face_image.shape
    assert enhanced.dtype == np.uint8

    # Test custom parameters
    enhanced_custom = apply_clahe(dummy_face_image, clip_limit=2.0, tile_grid_size=[4, 4])
    assert enhanced_custom.shape == dummy_face_image.shape


def test_apply_denoise(dummy_face_image):
    # Test default Bilateral Filter (d=5, sigma_color=25.0, sigma_space=25.0)
    denoised_bilateral = apply_denoise(dummy_face_image)
    assert denoised_bilateral is not None
    assert denoised_bilateral.shape == dummy_face_image.shape

    # Test FastNlMeans
    denoised_nl = apply_denoise(dummy_face_image, method="fast_nlmeans", h=3.0)
    assert denoised_nl is not None
    assert denoised_nl.shape == dummy_face_image.shape


def test_apply_sharpen(dummy_face_image):
    # Test default Unsharp Masking (strength=0.3, sigma=1.5, threshold=3.0)
    sharpened = apply_sharpen(dummy_face_image)
    assert sharpened is not None
    assert sharpened.shape == dummy_face_image.shape
    assert sharpened.dtype == np.uint8

    # Test strength=0 returns copy
    untouched = apply_sharpen(dummy_face_image, strength=0.0)
    np.testing.assert_array_equal(untouched, dummy_face_image)


def test_preprocess_image_pipeline(dummy_face_image):
    # Test with nested dict config
    nested_config = {
        "denoise": {"enabled": True, "method": "bilateral", "d": 5, "sigma_color": 25.0, "sigma_space": 25.0},
        "clahe": {"enabled": True, "clip_limit": 1.5, "tile_grid_size": [8, 8]},
        "sharpen": {"enabled": True, "strength": 0.3, "sigma": 1.5, "threshold": 3.0},
    }
    processed_nested = preprocess_image(dummy_face_image, config=nested_config)
    assert processed_nested is not None
    assert processed_nested.shape == dummy_face_image.shape

    # Test with simple boolean config
    bool_config = {
        "denoise": True,
        "clahe": True,
        "sharpen": False,
    }
    processed_bool = preprocess_image(dummy_face_image, config=bool_config)
    assert processed_bool is not None
    assert processed_bool.shape == dummy_face_image.shape

    # Test with None config (default)
    processed_default = preprocess_image(dummy_face_image, config=None)
    assert processed_default is not None
    assert processed_default.shape == dummy_face_image.shape


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
