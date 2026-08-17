"""
Module Embedder Factory: Cung cấp interface BaseEmbedder và các triển khai cụ thể:
1. ArcFace (Additive Angular Margin Loss, chuẩn phổ biến nhất, 512-D)
2. Facenet512 (Triplet loss thế hệ trước, 512-D)
3. SFace (Nhánh tối ưu tốc độ cho Edge, 128-D)

Hỗ trợ factory function `get_embedder(config)` đọc cấu hình từ pipeline.yaml.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np


class BaseEmbedder(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả Face Embedders.
    Mọi vector trả về đều được chuẩn hóa L2 (norm = 1.0).
    """

    def __init__(self, model_name: str, embedding_dim: int):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        from deepface import DeepFace
        self._deepface = DeepFace

    @abstractmethod
    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Trích xuất vector đặc trưng từ ảnh khuôn mặt đã crop/align.

        Args:
            face_crop: Ảnh khuôn mặt (BGR, uint8, thường kích thước 112x112 hoặc tương đương)

        Returns:
            np.ndarray: Vector 1D có độ dài embedding_dim, đã chuẩn hóa L2 (L2-norm = 1.0)
        """
        pass

    def embed_batch(self, face_crops: List[np.ndarray]) -> np.ndarray:
        """
        Trích xuất embedding cho danh sách nhiều ảnh mặt.

        Returns:
            np.ndarray: Ma trận (N, embedding_dim)
        """
        embeddings = [self.embed(img) for img in face_crops]
        if not embeddings:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.vstack(embeddings)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Chuẩn hóa L2 vector đặc trưng."""
        vec = np.asarray(vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            return vec / norm
        return vec


class ArcFaceEmbedder(BaseEmbedder):
    """
    ArcFace Embedder: Sử dụng Additive Angular Margin Loss.
    Vector đặc trưng: 512 chiều.
    """

    def __init__(self):
        super().__init__(model_name="ArcFace", embedding_dim=512)

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or face_crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            results = self._deepface.represent(
                img_path=face_crop,
                model_name=self.model_name,
                detector_backend="skip",  # Không lặp lại detection
                enforce_detection=False,
                align=False,
            )
            raw_emb = np.array(results[0]["embedding"], dtype=np.float32)
            return self._normalize(raw_emb)
        except Exception as e:
            # Fallback nếu lỗi
            return np.zeros(self.embedding_dim, dtype=np.float32)


class Facenet512Embedder(BaseEmbedder):
    """
    FaceNet512 Embedder: Sử dụng Triplet Loss cải tiến.
    Vector đặc trưng: 512 chiều.
    """

    def __init__(self):
        super().__init__(model_name="Facenet512", embedding_dim=512)

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or face_crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            results = self._deepface.represent(
                img_path=face_crop,
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False,
                align=False,
            )
            raw_emb = np.array(results[0]["embedding"], dtype=np.float32)
            return self._normalize(raw_emb)
        except Exception:
            return np.zeros(self.embedding_dim, dtype=np.float32)


class SFaceEmbedder(BaseEmbedder):
    """
    SFace Embedder: Mô hình siêu nhẹ tối ưu cho CPU / Edge.
    Vector đặc trưng: 128 chiều.
    """

    def __init__(self):
        super().__init__(model_name="SFace", embedding_dim=128)

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or face_crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            results = self._deepface.represent(
                img_path=face_crop,
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False,
                align=False,
            )
            raw_emb = np.array(results[0]["embedding"], dtype=np.float32)
            return self._normalize(raw_emb)
        except Exception:
            return np.zeros(self.embedding_dim, dtype=np.float32)


def get_embedder(config: Union[str, Dict[str, Any]] = "arcface") -> BaseEmbedder:
    """
    Factory function khởi tạo Face Embedder dựa trên cấu hình YAML.

    Args:
        config: Tên embedder (str) hoặc Dict config chứa key 'embedder'

    Returns:
        BaseEmbedder: Instance của embedder tương ứng
    """
    embedder_name = "arcface"

    if isinstance(config, str):
        embedder_name = config.lower().strip()
    elif isinstance(config, dict):
        pipeline_cfg = config.get("pipeline", config)
        embedder_name = pipeline_cfg.get("embedder", "arcface").lower().strip()

    if embedder_name in ("arcface",):
        return ArcFaceEmbedder()
    elif embedder_name in ("facenet512", "facenet_512", "facenet"):
        return Facenet512Embedder()
    elif embedder_name in ("sface",):
        return SFaceEmbedder()
    else:
        raise ValueError(
            f"Không hỗ trợ embedder: '{embedder_name}'. "
            f"Các lựa chọn hợp lệ: 'arcface', 'facenet512', 'sface'."
        )
