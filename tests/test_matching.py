"""
Unit tests cho module Matching và Session Store:
1. Cosine similarity giữa các vector đặc trưng
2. Đăng ký nhân viên (Enrollment N samples)
3. So khớp 1:K và kiểm tra logic Threshold
"""

import numpy as np
import pytest
from src.matching.matcher import (
    compute_cosine_similarity,
    SessionFaceStore,
    MatchResult,
)


def test_cosine_similarity_identical_vectors():
    vec = np.array([0.2, 0.5, 0.8, -0.1], dtype=np.float32)
    sim = compute_cosine_similarity(vec, vec)
    assert pytest.approx(sim, abs=1e-5) == 1.0


def test_cosine_similarity_orthogonal_vectors():
    vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    sim = compute_cosine_similarity(vec_a, vec_b)
    assert pytest.approx(sim, abs=1e-5) == 0.0


def test_cosine_similarity_opposite_vectors():
    vec_a = np.array([0.5, 0.5], dtype=np.float32)
    vec_b = np.array([-0.5, -0.5], dtype=np.float32)
    sim = compute_cosine_similarity(vec_a, vec_b)
    assert pytest.approx(sim, abs=1e-5) == -1.0


def test_session_store_enrollment_and_matching():
    store = SessionFaceStore(samples_per_person=3)
    assert store.get_enrolled_count() == 0

    # Giả lập embedding 512-D
    np.random.seed(123)
    user1_emb1 = np.random.randn(512).astype(np.float32)
    user1_emb2 = user1_emb1 + np.random.randn(512) * 0.05  # Cùng người, nhiễu nhẹ
    user2_emb1 = np.random.randn(512).astype(np.float32)   # Người khác

    # Enroll user1 và user2
    store.enroll("EMP_DUY", user1_emb1, meta={"name": "Duy Po"})
    store.enroll("EMP_DUY", user1_emb2)
    store.enroll("EMP_BOB", user2_emb1, meta={"name": "Bob Smith"})

    assert store.get_enrolled_count() == 2
    assert "EMP_DUY" in store.get_person_ids()

    # Query với ảnh cùng người user1
    query_same_user1 = user1_emb1 + np.random.randn(512) * 0.02
    result = store.find_best_match(query_same_user1, threshold=0.70)

    assert result.is_match is True
    assert result.matched_id == "EMP_DUY"
    assert result.similarity_score > 0.90
    assert result.metadata.get("name") == "Duy Po"

    # Query với người hoàn toàn lạ (Unknown)
    query_unknown = np.random.randn(512).astype(np.float32)
    result_unknown = store.find_best_match(query_unknown, threshold=0.75)

    assert result_unknown.is_match is False
    assert result_unknown.matched_id is None
    assert result_unknown.similarity_score < 0.75


def test_session_store_empty():
    store = SessionFaceStore()
    dummy_query = np.ones(512, dtype=np.float32)
    res = store.find_best_match(dummy_query, threshold=0.68)
    assert res.is_match is False
    assert res.matched_id is None
