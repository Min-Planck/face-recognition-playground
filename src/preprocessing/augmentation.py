"""
Module Augmentation: Tạo các trường hợp khó (hard cases) thường gặp trong máy chấm công:
1. Thiếu sáng / tối (Low light)
2. Ngược sáng / chói sáng (Backlight / Overexposure)
3. Góc nghiêng nhẹ (Slight rotation / perspective tilt)
4. Nhiễu cảm biến webcam (Sensor Gaussian noise)
"""

from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np


def simulate_low_light(
    image: np.ndarray,
    gamma: float = 2.2,
    brightness_factor: float = 0.45,
) -> np.ndarray:
    """
    Giả lập điều kiện ánh sáng yếu / thiếu sáng.
    Kết hợp giảm tuyến tính và phi tuyến tính (Gamma correction).
    """
    table = np.array([
        ((i / 255.0) ** gamma) * 255 * brightness_factor
        for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)


def simulate_backlight(
    image: np.ndarray,
    light_intensity: float = 1.4,
    center: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Giả lập hiện tượng ngược sáng / lóa sáng từ phía sau hoặc một góc.
    """
    h, w = image.shape[:2]
    if center is None:
        center = (int(w * 0.8), int(h * 0.2))

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    max_dist = np.sqrt(w**2 + h**2) / 1.5

    # Gradient ánh sáng
    gradient = np.clip(1.0 - (dist_from_center / max_dist), 0, 1)
    gradient = (gradient * 100 * (light_intensity - 1.0)).astype(np.float32)

    img_float = image.astype(np.float32)
    for c in range(3):
        img_float[:, :, c] += gradient

    return np.clip(img_float, 0, 255).astype(np.uint8)


def simulate_pose_variation(
    image: np.ndarray,
    angle: float = 12.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Giả lập góc nghiêng đầu nhẹ khi người dùng đứng trước máy chấm công.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        borderMode=cv2.BORDER_REFLECT,
    )
    return rotated


def simulate_sensor_noise(
    image: np.ndarray,
    std: float = 15.0,
) -> np.ndarray:
    """
    Giả lập nhiễu hạt cảm biến (sensor noise / low-quality webcam).
    """
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_hard_case_suite(
    image: np.ndarray,
    face_box: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Sinh trọn bộ các biến thể case khó từ 1 ảnh gốc để kiểm thử độ bền của pipeline.

    Returns:
        Dict[tên_case, ảnh_biến_thể]
    """
    return {
        "original": image.copy(),
        "low_light": simulate_low_light(image),
        "backlight": simulate_backlight(image),
        "pose_tilt_left": simulate_pose_variation(image, angle=-12.0),
        "pose_tilt_right": simulate_pose_variation(image, angle=12.0),
        "sensor_noise": simulate_sensor_noise(image, std=18.0),
    }


def generate_augmented_variants(
    base_image: np.ndarray,
    base_name: str,
) -> list[Tuple[str, np.ndarray]]:
    """
    Sinh danh sách các cặp (tên_biến_thể, ảnh_biến_thể) phục vụ stress-test và calibration.

    Args:
        base_image: Ảnh gốc (BGR)
        base_name: Tiền tố tên danh tính (ví dụ: 'p1_a')

    Returns:
        list[tuple[str, np.ndarray]]: Danh sách tuple (variant_name, variant_mat)
    """
    aug_dict = generate_hard_case_suite(base_image)
    return [(f"{base_name}_{aug_name}", aug_mat) for aug_name, aug_mat in aug_dict.items()]


def find_image_file(img_idx: int, image_dir: str = "data/test_images") -> str:
    """
    Tìm đường dẫn file ảnh theo index (img_1.png, img_2.jpg, ...).

    Args:
        img_idx: Chỉ số ảnh (1, 2, ..., 20)
        image_dir: Thư mục chứa ảnh (mặc định: data/test_images)

    Returns:
        str: Đường dẫn tuyệt đối hoặc tương đối tới file ảnh
    """
    import os
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        fname = f"img_{img_idx}{ext}"
        fpath = os.path.join(image_dir, fname)
        if os.path.exists(fpath):
            return fpath
    raise FileNotFoundError(f"Không tìm thấy ảnh img_{img_idx} trong {image_dir}")
