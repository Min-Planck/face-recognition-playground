"""
Module Embedder Factory: Cung cấp interface BaseEmbedder và các triển khai cụ thể:
1. ArcFace (Additive Angular Margin Loss, 512-D)
2. Facenet512 (Triplet Loss cải tiến, 512-D)
3. SFace (Nhánh tối ưu tốc độ cho Edge, 128-D)
4. ArcFace INT8 (ONNX Runtime Static PTQ, 512-D)
5. FaceNet512 INT8 (ONNX Runtime Static PTQ, 512-D)

Hỗ trợ factory function `get_embedder(config)` đọc cấu hình từ pipeline.yaml hoặc tên model.
Tích hợp Thread-Safe Lock và Contiguous Memory Buffer để chống crash bộ nhớ Windows.
"""

from abc import ABC, abstractmethod
import os
import threading
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

# Khóa đồng bộ luồng toàn cục để ngăn xung đột bộ nhớ C++ trong TensorFlow / DeepFace
_EMBEDDER_LOCK = threading.Lock()


class BaseEmbedder(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả Face Embedders.
    Mọi vector trả về đều được chuẩn hóa L2 (norm = 1.0).
    """

    def __init__(self, model_name: str, embedding_dim: int):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

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
    ArcFace Embedder (Chạy trực tiếp qua ONNX FP32 Engine).
    Vector đặc trưng: 512 chiều.
    """

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_name="ArcFace", embedding_dim=512)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        self.model_path = model_path or os.path.join(base_dir, "arcface_fp32.onnx")

        if os.path.exists(self.model_path):
            from src.export.quantize import ONNXEmbedderRunner
            self._runner = ONNXEmbedderRunner(self.model_path, model_type="arcface")
            self._deepface = None
        else:
            from deepface import DeepFace
            self._runner = None
            self._deepface = DeepFace

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or face_crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if self._runner is not None:
            return self._runner.embed(face_crop)

        try:
            crop_contiguous = np.ascontiguousarray(face_crop.copy(), dtype=np.uint8)
            with _EMBEDDER_LOCK:
                results = self._deepface.represent(
                    img_path=crop_contiguous,
                    model_name=self.model_name,
                    detector_backend="skip",
                    enforce_detection=False,
                    align=False,
                )
                raw_emb = np.array(results[0]["embedding"], dtype=np.float32)
                return self._normalize(raw_emb)
        except Exception:
            return np.zeros(self.embedding_dim, dtype=np.float32)


class Facenet512Embedder(BaseEmbedder):
    """
    FaceNet512 Embedder (Chạy trực tiếp qua ONNX FP32 Engine).
    Vector đặc trưng: 512 chiều.
    """

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_name="Facenet512", embedding_dim=512)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        self.model_path = model_path or os.path.join(base_dir, "facenet512_fp32.onnx")

        if os.path.exists(self.model_path):
            from src.export.quantize import ONNXEmbedderRunner
            self._runner = ONNXEmbedderRunner(self.model_path, model_type="facenet512")
            self._deepface = None
        else:
            from deepface import DeepFace
            self._runner = None
            self._deepface = DeepFace

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or face_crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if self._runner is not None:
            return self._runner.embed(face_crop)

        try:
            crop_contiguous = np.ascontiguousarray(face_crop.copy(), dtype=np.uint8)
            with _EMBEDDER_LOCK:
                results = self._deepface.represent(
                    img_path=crop_contiguous,
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
    SFace Embedder: Mô hình siêu nhẹ chạy trực tiếp trên C++ ONNX Engine.
    Vector đặc trưng: 128 chiều.
    """

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_name="SFace", embedding_dim=128)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        self.model_path = model_path or os.path.join(base_dir, "sface_fp32.onnx")

        if not os.path.exists(self.model_path):
            try:
                from scripts.export_onnx_models import export_sface_fp32
                export_sface_fp32()
            except Exception:
                pass

        if os.path.exists(self.model_path):
            from src.export.quantize import ONNXEmbedderRunner
            self._runner = ONNXEmbedderRunner(self.model_path, model_type="sface")
            self._deepface = None
        else:
            from deepface import DeepFace
            self._runner = None
            self._deepface = DeepFace

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or face_crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if self._runner is not None:
            return self._runner.embed(face_crop)

        try:
            crop_contiguous = np.ascontiguousarray(face_crop.copy(), dtype=np.uint8)
            with _EMBEDDER_LOCK:
                results = self._deepface.represent(
                    img_path=crop_contiguous,
                    model_name=self.model_name,
                    detector_backend="skip",
                    enforce_detection=False,
                    align=False,
                )
                raw_emb = np.array(results[0]["embedding"], dtype=np.float32)
                return self._normalize(raw_emb)
        except Exception:
            return np.zeros(self.embedding_dim, dtype=np.float32)


class ArcFaceONNXEmbedder(BaseEmbedder):
    """
    ArcFace ONNX Embedder (Hỗ trợ cả FP32 và INT8 Static PTQ).
    Chạy trực tiếp trên ONNX Runtime CPU C++ Engine.
    Vector đặc trưng: 512 chiều.
    """

    def __init__(self, is_int8: bool = True, model_path: Optional[str] = None):
        name = "ArcFace_INT8" if is_int8 else "ArcFace_FP32"
        super().__init__(model_name=name, embedding_dim=512)

        if model_path is None:
            filename = "arcface_int8.onnx" if is_int8 else "arcface_fp32.onnx"
            # Tìm trong thư mục models/
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
            model_path = os.path.join(base_dir, filename)

        from src.export.quantize import ONNXEmbedderRunner
        self.runner = ONNXEmbedderRunner(model_path, model_type="arcface")

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or face_crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        try:
            crop_contiguous = np.ascontiguousarray(face_crop, dtype=np.uint8)
            return self.runner.embed(crop_contiguous)
        except Exception:
            return np.zeros(self.embedding_dim, dtype=np.float32)


class FaceNet512ONNXEmbedder(BaseEmbedder):
    """
    FaceNet512 ONNX Embedder (Hỗ trợ cả FP32 và INT8 Static PTQ).
    Chạy trực tiếp trên ONNX Runtime CPU C++ Engine.
    Vector đặc trưng: 512 chiều.
    """

    def __init__(self, is_int8: bool = True, model_path: Optional[str] = None):
        name = "FaceNet512_INT8" if is_int8 else "FaceNet512_FP32"
        super().__init__(model_name=name, embedding_dim=512)

        if model_path is None:
            filename = "facenet512_int8.onnx" if is_int8 else "facenet512_fp32.onnx"
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
            model_path = os.path.join(base_dir, filename)

        from src.export.quantize import ONNXEmbedderRunner
        self.runner = ONNXEmbedderRunner(model_path, model_type="facenet512")

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or face_crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        try:
            crop_contiguous = np.ascontiguousarray(face_crop, dtype=np.uint8)
            return self.runner.embed(crop_contiguous)
        except Exception:
            return np.zeros(self.embedding_dim, dtype=np.float32)


def get_embedder(config: Union[str, Dict[str, Any]] = "arcface") -> BaseEmbedder:
    """
    Factory function khởi tạo Face Embedder dựa trên cấu hình YAML hoặc tên model.

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

    if embedder_name in ("arcface_int8", "arcface_qint8", "arcface_quantized"):
        return ArcFaceONNXEmbedder(is_int8=True)
    elif embedder_name in ("arcface_fp32", "arcface_onnx"):
        return ArcFaceONNXEmbedder(is_int8=False)
    elif embedder_name in ("facenet512_int8", "facenet_int8", "facenet_qint8"):
        return FaceNet512ONNXEmbedder(is_int8=True)
    elif embedder_name in ("facenet512_fp32", "facenet_fp32", "facenet_onnx"):
        return FaceNet512ONNXEmbedder(is_int8=False)
    elif embedder_name in ("arcface",):
        return ArcFaceEmbedder()
    elif embedder_name in ("facenet512", "facenet_512", "facenet"):
        return Facenet512Embedder()
    elif embedder_name in ("sface",):
        return SFaceEmbedder()
    else:
        raise ValueError(
            f"Không hỗ trợ embedder: '{embedder_name}'. "
            f"Các lựa chọn hợp lệ: 'arcface', 'facenet512', 'sface', 'arcface_int8', 'facenet512_int8'."
        )
