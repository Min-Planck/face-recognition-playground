"""
Module tiền xử lý ảnh khuôn mặt: CLAHE, Denoise và Sharpen.
Thứ tự pipeline chuẩn: Ảnh thô (Raw) -> CLAHE -> Denoise -> Sharpen.
"""

from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Union[Tuple[int, int], list] = (8, 8),
) -> np.ndarray:
    """
    Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) trên kênh L (LAB).
    Giữ nguyên thông tin màu sắc (A, B) để tránh méo màu da.

    Args:
        image: Ảnh đầu vào định dạng BGR hoặc RGB (uint8, HxWxC)
        clip_limit: Ngưỡng giới hạn khuếch đại tương phản (mặc định 2.0)
        tile_grid_size: Kích thước lưới chia ô (mặc định 8x8)

    Returns:
        np.ndarray: Ảnh sau khi đã cân bằng độ tương phản cục bộ
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ")

    # Đảm bảo tile_grid_size ở dạng tuple
    if isinstance(tile_grid_size, list):
        tile_grid_size = tuple(tile_grid_size)

    # Chuyển sang không gian màu LAB để chỉ can thiệp kênh độ sáng L
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tile_grid_size,
    )
    l_clahe = clahe.apply(l_channel)

    # Gộp lại và chuyển về BGR
    merged_lab = cv2.merge([l_clahe, a_channel, b_channel])
    enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image


def apply_denoise(
    image: np.ndarray,
    method: str = "bilateral",
    diameter: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    """
    Khử nhiễu ảnh nhưng vẫn giữ được các cạnh quan trọng của khuôn mặt.

    Args:
        image: Ảnh BGR đầu vào
        method: Phương pháp khử nhiễu ('bilateral' hoặc 'fast_nlmeans')
        diameter: Đường kính lân cận của mỗi điểm ảnh (dùng cho bilateral)
        sigma_color: Độ lọc trong không gian màu
        sigma_space: Độ lọc trong không gian tọa độ

    Returns:
        np.ndarray: Ảnh đã khử nhiễu
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ")

    if method == "fast_nlmeans":
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=7,
            hColor=7,
            templateWindowSize=7,
            searchWindowSize=21,
        )
    # Mặc định dùng Bilateral Filter: nhanh và giữ viền khuôn mặt rất tốt
    return cv2.bilateralFilter(
        image,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )


def apply_sharpen(
    image: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """
    Làm nét ảnh nhẹ nhàng bằng Unsharp Masking để tăng chi tiết landmark.

    Args:
        image: Ảnh BGR đầu vào
        strength: Độ mạnh của hiệu ứng làm nét (0.0 - 1.5)

    Returns:
        np.ndarray: Ảnh sau khi làm nét
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ")

    if strength <= 0:
        return image.copy()

    # Dùng Unsharp Masking (trừ Gaussian blur) để tránh noise hạt gắt
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(
        image,
        1.0 + strength,
        blurred,
        -strength,
        0,
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_image(
    image: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Pipeline tiền xử lý toàn diện theo thứ tự chuẩn:
    Raw -> CLAHE -> Denoise -> Sharpen

    Args:
        image: Ảnh gốc đầu vào (BGR)
        config: Dict cấu hình (thường đọc từ pipeline.yaml['preprocessing'])

    Returns:
        np.ndarray: Ảnh sau khi đã tiền xử lý
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào không hợp lệ")

    result = image.copy()

    if config is None:
        config = {
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": (8, 8)},
            "denoise": True,
            "sharpen": False,
        }

    # 1. Áp dụng CLAHE nếu được bật
    clahe_cfg = config.get("clahe", {})
    if clahe_cfg.get("enabled", True):
        clip_limit = clahe_cfg.get("clip_limit", 2.0)
        grid_size = clahe_cfg.get("tile_grid_size", (8, 8))
        result = apply_clahe(result, clip_limit=clip_limit, tile_grid_size=grid_size)

    # 2. Áp dụng Denoise nếu được bật
    if config.get("denoise", True):
        result = apply_denoise(result)

    # 3. Áp dụng Sharpen nếu được bật
    if config.get("sharpen", False):
        result = apply_sharpen(result, strength=0.4)

    return result
