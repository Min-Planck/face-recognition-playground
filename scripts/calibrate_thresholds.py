"""
Script Hiệu Chuẩn Ngưỡng & Đánh Giá Đường Cong ROC / FAR / FRR / EER
(Step 5: Threshold Calibration & Biometric Metrics Suite)

Tập dữ liệu: 20 ảnh (10 danh tính nhân viên, mỗi nhân viên gồm 2 ảnh liên tiếp 1-2, 3-4, ..., 19-20)
kết hợp bộ biến thể Augmentation đa dạng (thiếu sáng, ngược sáng, góc nghiêng đầu +-12 độ, nhiễu hạt cảm biến).
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detectors.detector_factory import get_detector, extract_aligned_face
from src.embedders.embedder_factory import get_embedder
from src.matching.matcher import compute_cosine_similarity
from src.preprocessing.clahe import preprocess_image
from src.preprocessing.augmentation import generate_augmented_variants, find_image_file
from src.evaluation.metrics import compute_far_frr, compute_eer, compute_rank1_accuracy

IMAGE_DIR = "data/test_images"
OUTPUT_REPORT_PATH = "outputs/report/threshold_calibration_report.md"
FIGURES_DIR = "outputs/figures/roc_curves"


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs("outputs/report", exist_ok=True)

    print("================================================================================")
    print("   HIỆU CHUẨN NGƯỠNG SINH TRẮC HỌC TRÊN TẬP 20 ẢNH (10 DANH TÍNH NHÂN VIÊN)    ")
    print("================================================================================\n")

    # 1. Nạp 10 danh tính (mỗi danh tính gồm 2 ảnh liên tiếp: 1x2, 3x4, ..., 19x20)
    person_images = {}  # {person_id: [img_a, img_b]}
    person_names = {}
    
    for p_id in range(1, 11):
        idx_a = (p_id - 1) * 2 + 1
        idx_b = (p_id - 1) * 2 + 2
        
        path_a = find_image_file(idx_a)
        path_b = find_image_file(idx_b)
        
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        
        if img_a is None or img_b is None:
            raise ValueError(f"Lỗi đọc ảnh cho Person {p_id}: {path_a} hoặc {path_b}")
            
        person_images[p_id] = (img_a, img_b)
        person_names[p_id] = (os.path.basename(path_a), os.path.basename(path_b))
        print(f"  Person {p_id:02d}: Ảnh 1 = {person_names[p_id][0]} | Ảnh 2 = {person_names[p_id][1]}")

    print("\nĐang khởi tạo MediaPipe Detector cho căn chỉnh khuôn mặt 112x112...")
    detector = get_detector("mediapipe")

    # 2. Sinh tập biến thể cho tất cả các ảnh
    print("Đang tạo các biến thể Hard Cases (thiếu sáng, ngược sáng, nghiêng đầu, nhiễu hạt)...")
    person_variants = {}  # {p_id: {'a': [(name, crop), ...], 'b': [(name, crop), ...]}}
    
    for p_id, (img_a, img_b) in person_images.items():
        var_a = generate_augmented_variants(img_a, f"p{p_id}_a")
        var_b = generate_augmented_variants(img_b, f"p{p_id}_b")
        
        crops_a = [extract_aligned_face(detector, mat) for _, mat in var_a]
        crops_b = [extract_aligned_face(detector, mat) for _, mat in var_b]
        
        person_variants[p_id] = {
            "crops_a": crops_a,
            "crops_b": crops_b,
        }

    embedders_list = ["arcface", "facenet512", "sface"]
    calibration_summary = []

    for emb_name in embedders_list:
        print(f"\n--------------------------------------------------------------------------------")
        print(f"--- Đang phân tích mô hình Embedder: '{emb_name}' ---")
        embedder = get_embedder(emb_name)

        # Trích xuất vector embedding cho tất cả các crops
        person_vectors = {}
        for p_id, data in person_variants.items():
            vecs_a = [embedder.embed(c) for c in data["crops_a"]]
            vecs_b = [embedder.embed(c) for c in data["crops_b"]]
            person_vectors[p_id] = {
                "vecs_a": vecs_a,
                "vecs_b": vecs_b,
            }

        # 3. Tạo tập cặp so khớp Genuine & Impostor
        genuine_scores = []
        impostor_scores = []

        # Genuine: So sánh giữa Ảnh A và Ảnh B của CÙNG 1 người (10 người x 6 x 6 = 360 cặp)
        for p_id in range(1, 11):
            for va in person_vectors[p_id]["vecs_a"]:
                for vb in person_vectors[p_id]["vecs_b"]:
                    sim = compute_cosine_similarity(va, vb)
                    genuine_scores.append(sim)

        # Impostor: So sánh giữa người i và người j (i < j: 45 cặp người x 36 = 1620 cặp)
        for i in range(1, 11):
            for j in range(i + 1, 11):
                # So khớp Ảnh A của người i với Ảnh A của người j
                for va_i in person_vectors[i]["vecs_a"]:
                    for va_j in person_vectors[j]["vecs_a"]:
                        sim = compute_cosine_similarity(va_i, va_j)
                        impostor_scores.append(sim)
                # So khớp Ảnh A của người i với Ảnh B của người j
                for va_i in person_vectors[i]["vecs_a"]:
                    for vb_j in person_vectors[j]["vecs_b"]:
                        sim = compute_cosine_similarity(va_i, vb_j)
                        impostor_scores.append(sim)

        y_true = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
        scores = np.array(genuine_scores + impostor_scores)

        # 4. Tính toán chỉ số EER, ROC AUC và FAR/FRR
        eer_res = compute_eer(y_true, scores, num_thresholds=1000)

        mean_gen = float(np.mean(genuine_scores))
        min_gen = float(np.min(genuine_scores))
        mean_imp = float(np.mean(impostor_scores))
        max_imp = float(np.max(impostor_scores))
        sep_margin = min_gen - max_imp
        rec_threshold = round(float(eer_res["optimal_threshold"]), 3)

        calibration_summary.append({
            "embedder": emb_name,
            "dim": person_vectors[1]["vecs_a"][0].shape[0],
            "n_genuine": len(genuine_scores),
            "n_impostor": len(impostor_scores),
            "mean_genuine": round(mean_gen, 4),
            "min_genuine": round(min_gen, 4),
            "mean_impostor": round(mean_imp, 4),
            "max_impostor": round(max_imp, 4),
            "separation_margin": round(sep_margin, 4),
            "eer": eer_res["eer"],
            "optimal_threshold_eer": eer_res["optimal_threshold"],
            "roc_auc": eer_res["roc_auc"],
            "recommended_threshold": rec_threshold,
        })

        print(f"  Số cặp Genuine (Cùng người):  {len(genuine_scores):4d} | Mean: {mean_gen:.4f} | Min: {min_gen:.4f}")
        print(f"  Số cặp Impostor (Khác người): {len(impostor_scores):4d} | Mean: {mean_imp:.4f} | Max: {max_imp:.4f}")
        print(f"  -> EER: {eer_res['eer']*100:.2f}% | Optimal Threshold (EER): {eer_res['optimal_threshold']:.3f} | ROC AUC: {eer_res['roc_auc']:.4f}")
        print(f"  -> Recommended Threshold T*: {rec_threshold:.3f}")

        # 5. Vẽ đồ thị FAR/FRR và Score Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ths = eer_res["thresholds"]
        far_curve = eer_res["far_list"]
        frr_curve = eer_res["frr_list"]

        ax1.plot(ths, far_curve, label='FAR (False Acceptance Rate)', color='#d62728', lw=2)
        ax1.plot(ths, frr_curve, label='FRR (False Rejection Rate)', color='#1f77b4', lw=2)
        ax1.axvline(eer_res["optimal_threshold"], color='#2ca02c', linestyle='--', label=f'EER Threshold = {rec_threshold:.3f}')
        ax1.set_title(f"FAR & FRR Trade-off Curve ({emb_name.upper()})", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Cosine Similarity Threshold")
        ax1.set_ylabel("Error Rate")
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2.hist(impostor_scores, bins=25, alpha=0.6, label='Impostor Pairs (Khác người)', color='#d62728', edgecolor='black', density=True)
        ax2.hist(genuine_scores, bins=25, alpha=0.6, label='Genuine Pairs (Cùng người)', color='#2ca02c', edgecolor='black', density=True)
        ax2.axvline(rec_threshold, color='black', linestyle='--', lw=2, label=f'Decision Threshold ({rec_threshold:.3f})')
        ax2.set_title(f"Score Distribution Separation ({emb_name.upper()})", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Cosine Similarity Score")
        ax2.set_ylabel("Density")
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, f"roc_far_frr_{emb_name}.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()

    # 6. Xuất Báo Cáo Markdown Hoàn Chỉnh
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Hiệu Chuẩn Ngưỡng & Đánh Giá Sinh Trắc Học (Step 5)\n\n")
        f.write("Báo cáo kiểm thử định lượng độ chính xác sinh trắc học trên tập dữ liệu gồm **20 ảnh (10 danh tính nhân viên: 1x2, 3x4, ..., 19x20)** kết hợp cùng bộ biến thể Augmentation (Thiếu sáng, ngược sáng, góc nghiêng đầu $\\pm 12^\\circ$, nhiễu hạt sensor).\n\n")
        f.write(f"- **Tổng số cặp kiểm thử:** `{calibration_summary[0]['n_genuine']}` cặp Cùng Người (Genuine) và `{calibration_summary[0]['n_impostor']}` cặp Khác Người (Impostor).\n\n")
        f.write("---\n\n")

        f.write("## 1. Bảng Tổng Hợp Chỉ Số Hiệu Chuẩn Ngưỡng (Calibration Table)\n\n")
        f.write("| Model Embedder | Vector Dim | Mean Genuine | Min Genuine | Mean Impostor | Max Impostor | Separation Margin $\\Delta$ | EER (%) | ROC AUC | Ngưỡng Khuyến Nghị $T^*$ |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for s in calibration_summary:
            f.write(f"| **{s['embedder']}** | {s['dim']}-D | {s['mean_genuine']} | **{s['min_genuine']}** | {s['mean_impostor']} | **{s['max_impostor']}** | **{s['separation_margin']:+.4f}** | **{s['eer']*100:.2f}%** | **{s['roc_auc']:.4f}** | **`{s['recommended_threshold']}`** |\n")

        f.write("---\n\n")
        f.write("## 2. Phân Tích Đường Cong Lỗi FAR/FRR & Phân Bố Điểm Số\n\n")
        for s in calibration_summary:
            emb = s["embedder"]
            f.write(f"### Mô Hình Embedder: `{emb.upper()}`\n")
            f.write(f"- [Đồ thị FAR/FRR Trade-off & Score Distribution](../figures/roc_curves/roc_far_frr_{emb}.png)\n")
            f.write(f"- **Nhận xét:** Ngưỡng tối ưu $T^*$ tại điểm cân bằng EER là `{s['recommended_threshold']}`.\n\n")

        f.write("---\n\n")
        f.write("## 3. Cấu Hình Ngưỡng Tối Ưu Cập Nhật Vào `pipeline.yaml`\n\n")
        f.write("Dựa trên kết quả thực nghiệm mới nhất, cấu hình ngưỡng tối ưu cho các mô hình:\n")
        f.write("```yaml\n")
        f.write("thresholds:\n")
        for s in calibration_summary:
            f.write(f"  {s['embedder']}: {s['recommended_threshold']}\n")
        f.write("```\n")

    print(f"\n=== Đã lưu báo cáo hiệu chuẩn ngưỡng vào: {OUTPUT_REPORT_PATH} ===")


if __name__ == "__main__":
    main()
