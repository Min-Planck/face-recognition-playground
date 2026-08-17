"""
Unit tests cho metrics.py và resource_monitor.py.
"""

import time
import numpy as np
import pytest
from src.evaluation.resource_monitor import ResourceMonitor
from src.evaluation.metrics import (
    compute_far_frr,
    compute_eer,
    compute_rank1_accuracy,
)


def test_resource_monitor():
    monitor = ResourceMonitor(interval=0.01)
    with monitor:
        # Giả lập công việc tốn CPU và RAM nhẹ
        data = [i**2 for i in range(100000)]
        time.sleep(0.05)

    summary = monitor.get_summary()
    assert summary["elapsed_ms"] >= 40.0
    assert summary["fps"] > 0
    assert summary["peak_ram_mb"] > 0


def test_compute_far_frr():
    # 5 genuine pairs (scores cao), 5 impostor pairs (scores thấp)
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    scores = [0.95, 0.88, 0.92, 0.85, 0.90, 0.20, 0.35, 0.15, 0.40, 0.25]

    # Threshold 0.70: phân loại hoàn hảo
    far, frr, hter = compute_far_frr(y_true, scores, threshold=0.70)
    assert far == 0.0
    assert frr == 0.0
    assert hter == 0.0

    # Threshold 0.92: từ chối 3 genuine pairs (0.85, 0.88, 0.90) -> FRR = 3/5 = 0.6
    far, frr, hter = compute_far_frr(y_true, scores, threshold=0.92)
    assert far == 0.0
    assert frr == 0.6
    assert hter == 0.3


def test_compute_eer():
    np.random.seed(42)
    # 100 genuine (mean=0.85, std=0.05), 100 impostor (mean=0.25, std=0.08)
    genuine_scores = np.random.normal(0.85, 0.05, 100).clip(0, 1)
    impostor_scores = np.random.normal(0.25, 0.08, 100).clip(0, 1)

    y_true = [1] * 100 + [0] * 100
    scores = list(genuine_scores) + list(impostor_scores)

    results = compute_eer(y_true, scores)
    assert results["eer"] < 0.05
    assert 0.4 < results["optimal_threshold"] < 0.7
    assert results["roc_auc"] > 0.98


def test_compute_rank1_accuracy():
    # 3 người trong gallery
    g_embs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    g_labels = ["Alice", "Bob", "Charlie"]

    # Probe embeddings gần khớp với Alice và Charlie
    p_embs = np.array([
        [0.9, 0.1, 0.0],   # -> Alice
        [0.1, 0.0, 0.95],  # -> Charlie
    ], dtype=np.float32)
    p_labels = ["Alice", "Charlie"]

    acc = compute_rank1_accuracy(g_embs, g_labels, p_embs, p_labels)
    assert acc == 1.0
