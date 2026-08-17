"""
Module Passive Liveness (Anti-Spoofing):
Sử dụng phân tích Texture / Moiré / Laplacian Variance trên vùng khuôn mặt và landmark.
Hoạt động hiệu quả với ảnh tĩnh (chụp từ webcam st.camera_input), không yêu cầu video liên tục.
Chống giả mạo: Print attack (ảnh in giấy) và Replay attack (ảnh/video phát lại trên màn hình).
"""

from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np


def compute_laplacian_variance(roi: np.ndarray) -> float:
    """
    Tính phương sai Laplacian (Laplacian Variance) trên vùng ROI xám.
    Đại diện cho mật độ cạnh vi mô và độ sắc nét thực của texture da.
    """
    if roi is None or roi.size == 0:
        return 0.0
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_passive_liveness(
    image: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
    laplacian_threshold: float = 100.0,
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Kiểm tra tính sống (Liveness) bị động của khuôn mặt trong ảnh.

    Args:
        image: Ảnh BGR đầy đủ từ camera
        face_bbox: (x, y, w, h) của khuôn mặt nếu đã detect. Nếu None sẽ dùng toàn ảnh.
        laplacian_threshold: Ngưỡng phân định ảnh thật và ảnh in/chụp lại màn hình.

    Returns:
        Tuple[is_live (bool), score (float), details (dict)]
    """
    if image is None or image.size == 0:
        return False, 0.0, {"reason": "Ảnh rỗng"}

    h, w = image.shape[:2]
    if face_bbox is not None:
        fx, fy, fw, fh = face_bbox
        x1 = max(0, fx)
        y1 = max(0, fy)
        x2 = min(w, fx + fw)
        y2 = min(h, fy + fh)
        face_crop = image[y1:y2, x1:x2]
    else:
        face_crop = image

    if face_crop.size == 0:
        return False, 0.0, {"reason": "Không crop được vùng mặt"}

    # 1. Tính toán Laplacian variance trên toàn vùng mặt
    full_face_var = compute_laplacian_variance(face_crop)

    # 2. Phân tích vùng trung tâm (má và sống mũi - nơi texture da tự nhiên rõ nhất)
    fh_c, fw_c = face_crop.shape[:2]
    center_roi = face_crop[
        int(fh_c * 0.3):int(fh_c * 0.7),
        int(fw_c * 0.25):int(fw_c * 0.75)
    ]
    center_var = compute_laplacian_variance(center_roi) if center_roi.size > 0 else full_face_var

    # 3. Phân tích vùng mắt (trên cùng)
    eye_roi = face_crop[
        int(fh_c * 0.2):int(fh_c * 0.5),
        int(fw_c * 0.15):int(fw_c * 0.85)
    ]
    eye_var = compute_laplacian_variance(eye_roi) if eye_roi.size > 0 else full_face_var

    # Tổng hợp điểm số (trọng số ưu tiên vùng mắt và má)
    composite_score = 0.3 * full_face_var + 0.4 * center_var + 0.3 * eye_var

    # Màn hình phát lại thường có texture mờ hơn hoặc moiré pattern bất thường
    # Ảnh in giấy có hiện tượng tán xạ hoặc mất chi tiết vi mô
    is_live = composite_score >= laplacian_threshold

    details = {
        "full_face_variance": round(full_face_var, 2),
        "center_cheek_variance": round(center_var, 2),
        "eye_region_variance": round(eye_var, 2),
        "composite_score": round(composite_score, 2),
        "threshold": laplacian_threshold,
        "is_live": is_live,
    }

    return is_live, composite_score, details
