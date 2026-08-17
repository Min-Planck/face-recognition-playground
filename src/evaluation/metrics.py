"""
Module đánh giá độ chính xác sinh trắc học khuôn mặt:
1. FAR (False Acceptance Rate)
2. FRR (False Rejection Rate)
3. HTER (Half Total Error Rate)
4. EER (Equal Error Rate) & Tìm Threshold tối ưu
5. Rank-1 Recognition Rate (1:K Identification)
6. ROC Curve & AUC
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_far_frr(
    y_true: Union[List[int], np.ndarray],
    similarity_scores: Union[List[float], np.ndarray],
    threshold: float,
) -> Tuple[float, float, float]:
    """
    Tính FAR, FRR và HTER tại một ngưỡng threshold cụ thể.

    Args:
        y_true: Nhãn nhị phân (1: Cùng người / Genuine pair, 0: Khác người / Impostor pair)
        similarity_scores: Điểm tương đồng Cosine similarity [0.0..1.0]
        threshold: Ngưỡng quyết định chấp nhận danh tính

    Returns:
        Tuple[FAR, FRR, HTER]
    """
    y_t = np.asarray(y_true, dtype=int)
    scores = np.asarray(similarity_scores, dtype=float)

    # Genuine pairs (cùng người): label == 1
    # Impostor pairs (khác người): label == 0
    genuine_mask = (y_t == 1)
    impostor_mask = (y_t == 0)

    n_genuine = int(np.sum(genuine_mask))
    n_impostor = int(np.sum(impostor_mask))

    if n_genuine == 0 or n_impostor == 0:
        raise ValueError("Tập dữ liệu kiểm thử phải chứa cả genuine pairs (1) và impostor pairs (0)")

    # FAR: Impostor bị chấp nhận nhầm (score >= threshold)
    false_accepts = int(np.sum((scores >= threshold) & impostor_mask))
    far = false_accepts / n_impostor

    # FRR: Genuine bị từ chối nhầm (score < threshold)
    false_rejects = int(np.sum((scores < threshold) & genuine_mask))
    frr = false_rejects / n_genuine

    # HTER: Half Total Error Rate
    hter = (far + frr) / 2.0

    return float(far), float(frr), float(hter)


def compute_eer(
    y_true: Union[List[int], np.ndarray],
    similarity_scores: Union[List[float], np.ndarray],
    num_thresholds: int = 1000,
) -> Dict[str, Any]:
    """
    Tính Equal Error Rate (EER) và tìm ngưỡng tối ưu (optimal threshold).
    EER là điểm mà FAR xấp xỉ bằng FRR nhất.

    Returns:
        Dict chứa: eer, optimal_threshold, min_hter, best_hter_threshold, far_list, frr_list, thresholds
    """
    scores = np.asarray(similarity_scores, dtype=float)
    y_t = np.asarray(y_true, dtype=int)

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    thresholds = np.linspace(min_score, max_score, num_thresholds)

    far_list = []
    frr_list = []
    hter_list = []

    for th in thresholds:
        far, frr, hter = compute_far_frr(y_t, scores, float(th))
        far_list.append(far)
        frr_list.append(frr)
        hter_list.append(hter)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    hter_arr = np.array(hter_list)

    # Tìm điểm EER (|FAR - FRR| nhỏ nhất)
    diff = np.abs(far_arr - frr_arr)
    eer_idx = int(np.argmin(diff))
    eer = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)
    optimal_threshold = float(thresholds[eer_idx])

    # Tìm điểm HTER nhỏ nhất
    min_hter_idx = int(np.argmin(hter_arr))
    min_hter = float(hter_arr[min_hter_idx])
    best_hter_threshold = float(thresholds[min_hter_idx])

    # ROC AUC
    fpr_roc, tpr_roc, _ = roc_curve(y_t, scores)
    roc_auc = float(auc(fpr_roc, tpr_roc))

    return {
        "eer": round(eer, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "min_hter": round(min_hter, 4),
        "best_hter_threshold": round(best_hter_threshold, 4),
        "roc_auc": round(roc_auc, 4),
        "far_at_eer": round(float(far_arr[eer_idx]), 4),
        "frr_at_eer": round(float(frr_arr[eer_idx]), 4),
        "thresholds": thresholds,
        "far_list": far_arr,
        "frr_list": frr_arr,
    }


def compute_rank1_accuracy(
    gallery_embeddings: np.ndarray,
    gallery_labels: List[str],
    probe_embeddings: np.ndarray,
    probe_labels: List[str],
) -> float:
    """
    Tính Rank-1 Recognition Accuracy cho bài toán định danh 1:K.

    Args:
        gallery_embeddings: Ma trận (N_gallery, D)
        gallery_labels: Danh sách nhãn tương ứng của gallery
        probe_embeddings: Ma trận (N_probe, D)
        probe_labels: Danh sách nhãn thực tế của probe

    Returns:
        float: Tỷ lệ nhận diện đúng Rank-1 [0.0..1.0]
    """
    if len(probe_embeddings) == 0 or len(gallery_embeddings) == 0:
        return 0.0

    # Chuẩn hóa L2
    g_norms = np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    g_normed = gallery_embeddings / np.maximum(g_norms, 1e-8)

    p_norms = np.linalg.norm(probe_embeddings, axis=1, keepdims=True)
    p_normed = probe_embeddings / np.maximum(p_norms, 1e-8)

    # Ma trận tương đồng (N_probe, N_gallery)
    sim_matrix = np.dot(p_normed, g_normed.T)

    correct = 0
    total = len(probe_labels)

    for i in range(total):
        best_gallery_idx = int(np.argmax(sim_matrix[i]))
        predicted_label = gallery_labels[best_gallery_idx]
        if predicted_label == probe_labels[i]:
            correct += 1

    return float(correct / total)
