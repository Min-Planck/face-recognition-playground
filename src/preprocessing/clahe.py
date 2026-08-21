"""
Module tiền xử lý ảnh khuôn mặt: Denoise, CLAHE và Sharpen.
Thứ tự pipeline chuẩn: Ảnh thô (Raw) -> Denoise -> CLAHE -> Sharpen.
"""

from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np


def _parse_step_config(cfg: Any, default_enabled: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Helper chuẩn hóa cấu hình cho từng bước tiền xử lý:
    - Boolean: True/False -> (True/False, {})
    - Dict: {"enabled": True/False, ...} -> (enabled, {params})
    - None / khác: (default_enabled, {})
    """
    if isinstance(cfg, bool):
        return cfg, {}
    if isinstance(cfg, dict):
        enabled = cfg.get("enabled", default_enabled)
        params = {k: v for k, v in cfg.items() if k != "enabled"}
        return bool(enabled), params
    if cfg is None:
        return default_enabled, {}
    return bool(cfg), {}


def apply_denoise(
    image: np.ndarray,
    method: str = "bilateral",
    d: int = 5,
    sigma_color: float = 25.0,
    sigma_space: float = 25.0,
    h: float = 3.0,
    diameter: Optional[int] = None,
) -> np.ndarray:
    """
    Khử nhiễu ảnh nhưng vẫn giữ được các cạnh quan trọng của khuôn mặt.

    Args:
        image: Ảnh BGR đầu vào (uint8, HxWxC).
        method: Phương pháp khử nhiễu ('bilateral' hoặc 'fast_nlmeans' / 'fastNlMeans').
        d: Đường kính lân cận của mỗi điểm ảnh cho Bilateral Filter (mặc định 5).
        sigma_color: Độ lọc trong không gian màu (mặc định 25.0).
        sigma_space: Độ lọc trong không gian tọa độ (mặc định 25.0).
        h: Tham số lọc nhiễu cho FastNlMeans (mặc định 3.0).
        diameter: Alias tương thích ngược cho tham số d.

    Returns:
        np.ndarray: Ảnh đã khử nhiễu (uint8, HxWxC).
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ")

    effective_d = diameter if diameter is not None else d

    if method.lower() in ("fast_nlmeans", "fastnlmeans", "nlmeans"):
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=float(h),
            hColor=float(h),
            templateWindowSize=7,
            searchWindowSize=21,
        )

    # Mặc định dùng Bilateral Filter: nhanh và giữ viền cạnh khuôn mặt tốt
    return cv2.bilateralFilter(
        image,
        d=int(effective_d),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 1.5,
    tile_grid_size: Union[Tuple[int, int], list] = (8, 8),
) -> np.ndarray:
    """
    Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) trên kênh L (LAB).
    Giữ nguyên thông tin màu sắc (A, B) để tránh làm méo màu da.

    Args:
        image: Ảnh đầu vào định dạng BGR (uint8, HxWxC).
        clip_limit: Ngưỡng giới hạn khuếch đại tương phản trên kênh L (mặc định 1.5).
        tile_grid_size: Kích thước lưới chia ô (mặc định 8x8).

    Returns:
        np.ndarray: Ảnh sau khi đã cân bằng độ tương phản cục bộ.
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ")

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
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def apply_sharpen(
    image: np.ndarray,
    strength: float = 0.3,
    sigma: float = 1.5,
    threshold: float = 3.0,
) -> np.ndarray:
    """
    Làm nét ảnh bằng Unsharp Masking kết hợp lọc ngưỡng cạnh (Edge Threshold Filter)
    để tăng chi tiết đặc trưng (mắt, mũi, miệng) mà không khuếch đại nhiễu hạt da.

    Args:
        image: Ảnh BGR đầu vào (uint8, HxWxC).
        strength: Độ mạnh của hiệu ứng làm nét (mặc định 0.3).
        sigma: Độ lệch chuẩn Gaussian Blur để trích xuất dải tần cao (mặc định 1.5).
        threshold: Ngưỡng chênh lệch tối thiểu để áp dụng làm nét (mặc định 3.0).

    Returns:
        np.ndarray: Ảnh sau khi làm nét (uint8, HxWxC).
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ")

    if strength <= 0:
        return image.copy()

    # Unsharp Masking: Chi tiết biên = Ảnh gốc - Ảnh làm mờ Gaussian
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    diff = image.astype(np.float32) - blurred.astype(np.float32)

    # Lọc ngưỡng cạnh: triệt tiêu các dao động nhỏ dưới ngưỡng để tránh khuếch đại nhiễu da
    if threshold > 0:
        mask = np.abs(diff) >= float(threshold)
        diff = diff * mask

    sharpened = image.astype(np.float32) + float(strength) * diff
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_image(
    image: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Pipeline tiền xử lý toàn diện theo thứ tự chuẩn:
    Raw -> Denoise -> CLAHE -> Sharpen

    Hỗ trợ linh hoạt cả cấu hình lồng nhau (nested dict) lẫn cờ boolean đơn giản.

    Args:
        image: Ảnh gốc đầu vào (BGR).
        config: Dict cấu hình (thường đọc từ pipeline.yaml['preprocessing'] hoặc truyền trực tiếp).

    Returns:
        np.ndarray: Ảnh sau khi đã qua pipeline tiền xử lý.
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào không hợp lệ")

    result = image.copy()

    if config is None:
        config = {
            "denoise": True,
            "clahe": True,
            "sharpen": True,
        }

    # 1. Denoise (Raw -> Denoise)
    denoise_cfg = config.get("denoise", True)
    denoise_enabled, denoise_params = _parse_step_config(denoise_cfg, default_enabled=True)
    if denoise_enabled:
        result = apply_denoise(result, **denoise_params)

    # 2. CLAHE (Denoise -> CLAHE)
    clahe_cfg = config.get("clahe", True)
    clahe_enabled, clahe_params = _parse_step_config(clahe_cfg, default_enabled=True)
    if clahe_enabled:
        result = apply_clahe(result, **clahe_params)

    # 3. Sharpen (CLAHE -> Sharpen)
    sharpen_cfg = config.get("sharpen", True)
    sharpen_enabled, sharpen_params = _parse_step_config(sharpen_cfg, default_enabled=True)
    if sharpen_enabled:
        result = apply_sharpen(result, **sharpen_params)

    return result
