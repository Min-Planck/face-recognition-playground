"""
Module Matching & Session Store: Quản lý đăng ký (Enrollment) và so khớp (Inference 1:K).
Sử dụng Cosine Similarity trên bộ nhớ in-memory (không dùng cơ sở dữ liệu vật lý).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class MatchResult:
    """
    Kết quả so khớp 1:K của 1 khuôn mặt truy vấn.
    """
    is_match: bool
    matched_id: Optional[str] = None
    similarity_score: float = 0.0
    threshold_used: float = 0.68
    all_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_cosine_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> float:
    """
    Tính Cosine Similarity giữa 2 vector đặc trưng.
    sim(A, B) = (A . B) / (||A|| * ||B||)

    Args:
        vec_a: Vector đặc trưng A (1D np.ndarray)
        vec_b: Vector đặc trưng B (1D np.ndarray)

    Returns:
        float: Điểm tương đồng trong khoảng [-1.0, 1.0] (thường [0.0, 1.0] cho face embeddings)
    """
    a = np.asarray(vec_a, dtype=np.float32).flatten()
    b = np.asarray(vec_b, dtype=np.float32).flatten()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, sim))


class SessionFaceStore:
    """
    Quản lý lưu trữ embeddings theo session in-memory:
    person_id -> list[embedding_vectors] (mỗi người có N samples)
    """

    def __init__(self, samples_per_person: int = 3):
        self.samples_per_person = samples_per_person
        # dict: person_id -> list of 1D numpy arrays
        self._store: Dict[str, List[np.ndarray]] = {}
        # dict: person_id -> metadata (tên hiển thị, ngày đăng ký, ...)
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def enroll(
        self,
        person_id: str,
        embedding: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Đăng ký thêm 1 embedding cho nhân viên.

        Args:
            person_id: Mã nhân viên (ví dụ: 'EMP001' hoặc 'Duy')
            embedding: Vector 1D (đã trích xuất từ Embedder)
            meta: Metadata tùy chọn

        Returns:
            int: Số lượng sample hiện có của nhân viên này
        """
        if not person_id or not isinstance(person_id, str):
            raise ValueError("person_id phải là chuỗi không rỗng")

        vec = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        if person_id not in self._store:
            self._store[person_id] = []
            self._metadata[person_id] = meta or {}
        elif meta:
            self._metadata[person_id].update(meta)

        self._store[person_id].append(vec)
        return len(self._store[person_id])

    def get_enrolled_count(self) -> int:
        """Trả về tổng số nhân viên đã đăng ký trong session."""
        return len(self._store)

    def get_person_ids(self) -> List[str]:
        """Danh sách mã nhân viên đã đăng ký."""
        return list(self._store.keys())

    def get_all_enrolled_meta(self) -> Dict[str, Dict[str, Any]]:
        """Trả về toàn bộ metadata và số lượng vector mẫu của từng nhân viên."""
        result = {}
        for pid, meta in self._metadata.items():
            info = dict(meta)
            info["samples_count"] = len(self._store.get(pid, []))
            result[pid] = info
        return result

    def clear(self) -> None:
        """Xóa toàn bộ dữ liệu session (dùng khi restart hoặc reset)."""
        self._store.clear()
        self._metadata.clear()

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.68,
        aggregation: str = "max",
    ) -> MatchResult:
        """
        Thực hiện so khớp 1:K (Inference) giữa query embedding và toàn bộ session store.

        Args:
            query_embedding: Vector khuôn mặt cần nhận diện
            threshold: Ngưỡng cosine similarity chấp nhận danh tính
            aggregation: Cách tổng hợp điểm giữa N samples của 1 người ('max' hoặc 'mean')

        Returns:
            MatchResult: Kết quả nhận diện
        """
        if not self._store:
            return MatchResult(
                is_match=False,
                matched_id=None,
                similarity_score=0.0,
                threshold_used=threshold,
                all_scores={},
            )

        q_vec = np.asarray(query_embedding, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-8:
            q_vec = q_vec / q_norm

        person_scores: Dict[str, float] = {}

        for person_id, stored_embeddings in self._store.items():
            if not stored_embeddings:
                continue

            sample_sims = []
            for s_vec in stored_embeddings:
                if s_vec.shape != q_vec.shape:
                    # Bỏ qua nếu vector mẫu khác số chiều (do người dùng chuyển đổi giữa 128-D và 512-D)
                    continue
                sample_sims.append(float(np.dot(q_vec, s_vec)))

            if not sample_sims:
                continue

            if aggregation == "mean":
                person_score = float(np.mean(sample_sims))
            else:
                # Mặc định dùng MAX similarity theo spec
                person_score = float(np.max(sample_sims))

            person_scores[person_id] = person_score

        if not person_scores:
            return MatchResult(
                is_match=False,
                matched_id=None,
                similarity_score=0.0,
                threshold_used=threshold,
                all_scores={},
            )

        # Tìm người có điểm tương đồng cao nhất
        best_id = max(person_scores, key=person_scores.get)
        best_score = person_scores[best_id]

        is_match = best_score >= threshold

        return MatchResult(
            is_match=is_match,
            matched_id=best_id if is_match else None,
            similarity_score=best_score,
            threshold_used=threshold,
            all_scores=person_scores,
            metadata=self._metadata.get(best_id, {}) if is_match else {},
        )
