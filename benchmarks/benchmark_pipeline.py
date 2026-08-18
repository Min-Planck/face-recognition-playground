"""
Benchmark Pipeline Nhận Diện Khuôn Mặt Máy Chấm Công (Test 3)
Thiết kế đo đạc toàn diện theo yêu cầu kỹ thuật:
1. Đánh giá riêng 3 Detector (mediapipe, retinaface, yolov8) -> Chọn Best Detector.
2. Đánh giá riêng 3 Embedder (arcface, facenet512, sface) -> Chọn Best Embedder.
3. Chạy nhóm tổ hợp tối ưu:
   - Nhánh A: Best Detector x 3 Embedders
   - Nhánh B: 3 Detectors x Best Embedder
4. Thực nghiệm pha Chấm công trên tập 20 ảnh (10 danh tính nhân viên: 1x2, 3x4, ..., 19x20):
   - Đăng ký ảnh A của các nhân viên.
   - Quét ảnh B của nhân viên (Chấm công hợp lệ).
   - Quét ảnh của người khác (Mạo danh).
   - Sử dụng các ngưỡng hiệu chuẩn tối ưu T* từ config/pipeline.yaml.
   - Tích hợp giai đoạn Warm-up (Khởi động bộ nhớ đệm / Cold-start elimination) trước khi bấm giờ.
5. Xuất kết quả ra CSV (benchmarks/results/benchmark_results.csv) và Markdown Report.
"""

import os
import sys
import time
import yaml
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
from src.matching.matcher import SessionFaceStore, compute_cosine_similarity
from src.liveness.passive import check_passive_liveness
from src.preprocessing.clahe import preprocess_image
from src.preprocessing.augmentation import find_image_file
from src.evaluation.resource_monitor import ResourceMonitor

IMAGE_DIR = "data/test_images"
CONFIG_PATH = "config/pipeline.yaml"
N_REPEATS = 3
N_WARMUP = 2

DETECTORS_LIST = ["mediapipe", "retinaface", "yolov8"]
EMBEDDERS_LIST = ["arcface", "facenet512", "sface"]

RESULTS_CSV_PATH = "benchmarks/results/benchmark_results.csv"
REPORT_MD_PATH = "outputs/report/benchmark_results_report.md"
FIGURES_DIR = "outputs/figures/benchmark_charts"


def load_calibrated_thresholds() -> dict:
    """Đọc ngưỡng đã hiệu chuẩn từ pipeline.yaml."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("thresholds", {"sface": 0.32, "arcface": 0.24, "facenet512": 0.53})
    return {"sface": 0.32, "arcface": 0.24, "facenet512": 0.53}


def main():
    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs("outputs/report", exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    thresholds_map = load_calibrated_thresholds()

    # Nạp 10 danh tính (mỗi danh tính gồm 2 ảnh: Ảnh A = Đăng ký, Ảnh B = Chấm công)
    person_data = []
    for p_id in range(1, 11):
        idx_a = (p_id - 1) * 2 + 1
        idx_b = (p_id - 1) * 2 + 2
        path_a = find_image_file(idx_a)
        path_b = find_image_file(idx_b)
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        person_data.append({
            "id": f"NV{p_id:03d}",
            "name": f"Nhan Vien {p_id:02d}",
            "img_a": img_a,
            "img_b": img_b,
            "file_a": os.path.basename(path_a),
            "file_b": os.path.basename(path_b),
        })

    print("================================================================================")
    print("        BENCHMARK TEST 3: ĐO ĐẠC PIPELINE NHẬN DIỆN MÁY CHẤM CÔNG              ")
    print("================================================================================\n")
    print(f"Tổng số danh tính nhân viên: {len(person_data)} (20 ảnh từ {IMAGE_DIR})")
    print(f"Ngưỡng hiệu chuẩn T* đọc từ pipeline.yaml: {thresholds_map}")
    print(f"Giai đoạn Warm-up: {N_WARMUP} lần chạy khởi động trước khi đo thời gian.")
    print(f"Số lần lặp đo trung bình mỗi phép đo: {N_REPEATS}\n")

    benchmark_rows = []

    # ==============================================================================
    # GIAI ĐOẠN 1: BENCHMARK DETECTOR ĐỘC LẬP (VỚI WARM-UP)
    # ==============================================================================
    print("--- [Giai đoạn 1] Benchmark riêng 3 Detector ---")
    detector_scores = {}
    sample_eval_img = person_data[0]["img_b"]  # Dùng img_2.jpg làm mẫu đo

    for det_name in DETECTORS_LIST:
        try:
            detector = get_detector(det_name)
        except Exception as e:
            print(f"  [Lỗi khởi tạo detector] {det_name}: {e}")
            continue

        # 1. Warm-up stage: Nạp model vào bộ nhớ và kích hoạt kernel C++/JIT
        for _ in range(N_WARMUP):
            _ = detector.detect(sample_eval_img)

        latencies = []
        monitor = ResourceMonitor(interval=0.03)

        with monitor:
            for _ in range(N_REPEATS):
                t0 = time.perf_counter()
                boxes = detector.detect(sample_eval_img)
                latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_lat = float(np.mean(latencies))
        fps = 1000.0 / avg_lat if avg_lat > 0 else 0
        detector_scores[det_name] = {
            "latency_ms": avg_lat,
            "fps": fps,
            "cpu": monitor.avg_cpu,
            "ram": monitor.avg_ram,
            "peak_ram": monitor.peak_ram,
            "boxes_count": len(boxes),
        }

        benchmark_rows.append({
            "phase": "1_detector_standalone",
            "test_type": "Standalone Detection",
            "detector": det_name,
            "embedder": "-",
            "target_image": person_data[0]["file_b"],
            "latency_ms": round(avg_lat, 2),
            "fps": round(fps, 2),
            "cpu_percent": round(monitor.avg_cpu, 2),
            "avg_ram_mb": round(monitor.avg_ram, 2),
            "peak_ram_mb": round(monitor.peak_ram, 2),
            "cosine_similarity": "-",
            "decision": "-",
        })

        print(f"  Detector: {det_name:12s} | Latency: {avg_lat:6.2f} ms | {fps:6.2f} FPS | CPU: {monitor.avg_cpu:5.1f}% | RAM: {monitor.avg_ram:6.1f} MB")

    best_detector = min(detector_scores.keys(), key=lambda k: detector_scores[k]["latency_ms"])
    print(f"\n👉 Best Performance Detector (Edge CPU): '{best_detector}' ({detector_scores[best_detector]['fps']:.1f} FPS, {detector_scores[best_detector]['latency_ms']:.1f} ms)\n")

    # ==============================================================================
    # GIAI ĐOẠN 2: BENCHMARK EMBEDDER ĐỘC LẬP (VỚI WARM-UP)
    # ==============================================================================
    print("--- [Giai đoạn 2] Benchmark riêng 3 Embedder (trên mặt đã crop 112x112) ---")
    det_for_crop = get_detector(best_detector)
    infer_boxes = det_for_crop.detect(sample_eval_img)
    if infer_boxes and infer_boxes[0].aligned_face is not None:
        sample_crop = np.ascontiguousarray(infer_boxes[0].aligned_face.copy(), dtype=np.uint8)
    else:
        sample_crop = np.ascontiguousarray(cv2.resize(sample_eval_img, (112, 112)), dtype=np.uint8)

    embedder_scores = {}

    for emb_name in EMBEDDERS_LIST:
        try:
            embedder = get_embedder(emb_name)
        except Exception as e:
            print(f"  [Lỗi khởi tạo embedder] {emb_name}: {e}")
            continue

        # 1. Warm-up stage: Nạp TensorFlow/ONNX weights và kích hoạt bộ nhớ đệm
        for _ in range(N_WARMUP):
            _ = embedder.embed(sample_crop)

        latencies = []
        monitor = ResourceMonitor(interval=0.03)

        with monitor:
            for _ in range(N_REPEATS):
                t0 = time.perf_counter()
                vec = embedder.embed(sample_crop)
                latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_lat = float(np.mean(latencies))
        fps = 1000.0 / avg_lat if avg_lat > 0 else 0
        embedder_scores[emb_name] = {
            "latency_ms": avg_lat,
            "fps": fps,
            "dim": vec.shape[0],
            "cpu": monitor.avg_cpu,
            "ram": monitor.avg_ram,
            "peak_ram": monitor.peak_ram,
        }

        benchmark_rows.append({
            "phase": "2_embedder_standalone",
            "test_type": "Standalone Embedding",
            "detector": "-",
            "embedder": emb_name,
            "target_image": "Aligned Crop (112x112)",
            "latency_ms": round(avg_lat, 2),
            "fps": round(fps, 2),
            "cpu_percent": round(monitor.avg_cpu, 2),
            "avg_ram_mb": round(monitor.avg_ram, 2),
            "peak_ram_mb": round(monitor.peak_ram, 2),
            "cosine_similarity": "-",
            "decision": "-",
        })

        print(f"  Embedder: {emb_name:12s} ({vec.shape[0]}-D) | Latency: {avg_lat:6.2f} ms | {fps:6.2f} FPS | CPU: {monitor.avg_cpu:5.1f}% | RAM: {monitor.avg_ram:6.1f} MB")

    best_embedder = min(embedder_scores.keys(), key=lambda k: embedder_scores[k]["latency_ms"])
    print(f"\n👉 Best Performance Embedder (Edge CPU): '{best_embedder}' ({embedder_scores[best_embedder]['fps']:.1f} FPS, {embedder_scores[best_embedder]['latency_ms']:.1f} ms)\n")

    # ==============================================================================
    # GIAI ĐOẠN 3 & 4: TỔ HỢP TẬP TRUNG & THỰC NGHIỆM CHẤM CÔNG (VỚI WARM-UP)
    # ==============================================================================
    print("--- [Giai đoạn 3 & 4] Thực nghiệm Chấm công: Tổ hợp Nhóm A & Nhóm B ---")

    combos_to_test = []
    # Nhóm A: Best Detector x 3 Embedders
    for emb in EMBEDDERS_LIST:
        combos_to_test.append((best_detector, emb, "Group_A (Best_Detector x 3_Embedders)"))

    # Nhóm B: 3 Detectors x Best Embedder
    for det in DETECTORS_LIST:
        if det != best_detector:
            combos_to_test.append((det, best_embedder, "Group_B (3_Detectors x Best_Embedder)"))

    pipeline_results = []

    for det_name, emb_name, group_tag in combos_to_test:
        detector = get_detector(det_name)
        embedder = get_embedder(emb_name)
        th = float(thresholds_map.get(emb_name, 0.30))

        # Warm-up toàn chuỗi Pipeline (Tiền xử lý -> Detect -> Crop -> Embed)
        for _ in range(N_WARMUP):
            _prep = preprocess_image(sample_eval_img)
            _boxes = detector.detect(_prep)
            if _boxes:
                _crop = _boxes[0].aligned_face if _boxes[0].aligned_face is not None else cv2.resize(_boxes[0].get_crop(_prep), (112, 112))
                _ = embedder.embed(np.ascontiguousarray(_crop.copy(), dtype=np.uint8))

        # 1. ENROLLMENT GALLERY (Đăng ký toàn bộ 10 nhân viên bằng Ảnh A)
        store = SessionFaceStore(samples_per_person=3)
        for p in person_data:
            prep_a = preprocess_image(p["img_a"])
            boxes_a = detector.detect(prep_a)
            if boxes_a:
                box_a = max(boxes_a, key=lambda b: b.w * b.h)
                face_a = box_a.aligned_face if box_a.aligned_face is not None else cv2.resize(box_a.get_crop(prep_a), (112, 112))
                face_a = np.ascontiguousarray(face_a.copy(), dtype=np.uint8)
                vec_a = embedder.embed(face_a)
                store.enroll(p["id"], vec_a, meta={"name": p["name"]})

        # 2. ĐO ĐẠC INFERENCE: VALID ATTENDANCE & IMPOSTOR ATTACK
        valid_latencies = []
        valid_scores = []
        valid_correct = 0

        impostor_latencies = []
        impostor_scores = []
        impostor_correct = 0

        monitor = ResourceMonitor(interval=0.03)

        with monitor:
            # A. Kiểm tra nhân viên hợp lệ (10 nhân viên quét Ảnh B)
            for p in person_data:
                t0 = time.perf_counter()
                prep_b = preprocess_image(p["img_b"])
                boxes_b = detector.detect(prep_b)
                if boxes_b:
                    box_b = max(boxes_b, key=lambda b: b.w * b.h)
                    face_b = box_b.aligned_face if box_b.aligned_face is not None else cv2.resize(box_b.get_crop(prep_b), (112, 112))
                    face_b = np.ascontiguousarray(face_b.copy(), dtype=np.uint8)
                    vec_b = embedder.embed(face_b)
                    match_res = store.find_best_match(vec_b, threshold=th)
                    lat = (time.perf_counter() - t0) * 1000.0
                    valid_latencies.append(lat)
                    valid_scores.append(match_res.similarity_score)
                    if match_res.is_match and match_res.matched_id == p["id"]:
                        valid_correct += 1

            # B. Kiểm tra kẻ mạo danh (Person 1 quét vào Store khi chỉ đăng ký Person 2..10, hoặc test so khớp chéo)
            for i, p in enumerate(person_data):
                # Tạo sub-store không chứa nhân viên p
                sub_store = SessionFaceStore(samples_per_person=3)
                for other_p in person_data:
                    if other_p["id"] != p["id"]:
                        for v in store.get_embeddings(other_p["id"]):
                            sub_store.enroll(other_p["id"], v, meta={"name": other_p["name"]})
                
                t0 = time.perf_counter()
                prep_b = preprocess_image(p["img_b"])
                boxes_b = detector.detect(prep_b)
                if boxes_b:
                    box_b = max(boxes_b, key=lambda b: b.w * b.h)
                    face_b = box_b.aligned_face if box_b.aligned_face is not None else cv2.resize(box_b.get_crop(prep_b), (112, 112))
                    face_b = np.ascontiguousarray(face_b.copy(), dtype=np.uint8)
                    vec_b = embedder.embed(face_b)
                    match_res = sub_store.find_best_match(vec_b, threshold=th)
                    lat = (time.perf_counter() - t0) * 1000.0
                    impostor_latencies.append(lat)
                    impostor_scores.append(match_res.similarity_score)
                    if not match_res.is_match:
                        impostor_correct += 1

        avg_valid_lat = float(np.mean(valid_latencies)) if valid_latencies else 0.0
        avg_valid_sim = float(np.mean(valid_scores)) if valid_scores else 0.0
        valid_acc = (valid_correct / len(person_data)) * 100.0

        avg_imp_lat = float(np.mean(impostor_latencies)) if impostor_latencies else 0.0
        avg_imp_sim = float(np.mean(impostor_scores)) if impostor_scores else 0.0
        imp_acc = (impostor_correct / len(person_data)) * 100.0

        overall_e2e_lat = (avg_valid_lat + avg_imp_lat) / 2.0
        e2e_fps = 1000.0 / overall_e2e_lat if overall_e2e_lat > 0 else 0.0

        pipeline_results.append({
            "combo": f"{det_name} + {emb_name}",
            "group": group_tag,
            "detector": det_name,
            "embedder": emb_name,
            "threshold": th,
            "valid_sim": round(avg_valid_sim, 4),
            "impostor_sim": round(avg_imp_sim, 4),
            "valid_acc": round(valid_acc, 1),
            "impostor_acc": round(imp_acc, 1),
            "latency_ms": round(overall_e2e_lat, 2),
            "fps": round(e2e_fps, 2),
            "cpu_percent": round(monitor.avg_cpu, 2),
            "avg_ram_mb": round(monitor.avg_ram, 2),
            "peak_ram_mb": round(monitor.peak_ram, 2),
        })

        benchmark_rows.append({
            "phase": "3_pipeline_e2e",
            "test_type": "Valid Attendance",
            "detector": det_name,
            "embedder": emb_name,
            "target_image": "10 Persons Test",
            "latency_ms": round(avg_valid_lat, 2),
            "fps": round(1000.0 / avg_valid_lat if avg_valid_lat > 0 else 0, 2),
            "cpu_percent": round(monitor.avg_cpu, 2),
            "avg_ram_mb": round(monitor.avg_ram, 2),
            "peak_ram_mb": round(monitor.peak_ram, 2),
            "cosine_similarity": round(avg_valid_sim, 4),
            "decision": f"Acc: {valid_acc:.1f}%",
        })

        print(f"  Combo: {det_name:10s} + {emb_name:10s} (T={th}) | Latency (Pure): {overall_e2e_lat:6.2f} ms ({e2e_fps:4.2f} FPS) | Valid Sim: {avg_valid_sim:.4f} ({valid_acc:.0f}%) | Impostor Sim: {avg_imp_sim:.4f} ({imp_acc:.0f}%)")

    # 4. Xuất kết quả CSV
    df = pd.DataFrame(benchmark_rows)
    df.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n=== Đã lưu dữ liệu benchmark chi tiết vào: {RESULTS_CSV_PATH} ===")

    # 5. Vẽ biểu đồ cột trực quan hóa độ trễ End-to-End
    fig, ax = plt.subplots(figsize=(10, 5))

    combo_names = [r["combo"] for r in pipeline_results]
    latencies_val = [r["latency_ms"] for r in pipeline_results]
    fps_val = [r["fps"] for r in pipeline_results]

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    bars = ax.bar(combo_names, latencies_val, color=colors[:len(combo_names)], width=0.55, edgecolor='black')

    for bar, lat, fps in zip(bars, latencies_val, fps_val):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 15,
            f"{lat:.1f} ms\n({fps:.1f} FPS)",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )

    ax.set_title("So Sánh Độ Trễ Thuần (Pure Latency sau Warm-up) Các Tổ Hợp Pipeline", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("Độ Trễ Inference Thuần (ms)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(latencies_val) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15, ha='right', fontsize=10, fontweight='bold')
    plt.tight_layout()

    chart_path = os.path.join(FIGURES_DIR, "pipeline_benchmark_comparison.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()

    # 6. Xuất Báo Cáo Markdown
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Benchmark Pipeline Nhận Diện Khuôn Mặt\n\n")
        f.write("Báo cáo đo đạc hiệu năng thực tế trên toàn bộ tập dữ liệu gồm **20 ảnh (10 danh tính nhân viên)** theo phương pháp tổ hợp tối ưu (Best Detector x 3 Embedders và 3 Detectors x Best Embedder), sử dụng ngưỡng $T^*$ đã hiệu chuẩn thực nghiệm từ `config/pipeline.yaml`.\n\n")
        f.write("---\n\n")

        f.write("## 1. Kết Quả Benchmark Độc Lập Detector (Sau Warm-up)\n\n")
        f.write("| Detector | Tốc độ (FPS) | Độ trễ Thuần (ms) | CPU (%) | RAM TB (MB) | Peak RAM (MB) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for det_name, s in detector_scores.items():
            f.write(f"| **{det_name}** | **{s['fps']:.2f}** | {s['latency_ms']:.2f} ms | {s['cpu']:.1f}% | {s['ram']:.1f} MB | {s['peak_ram']:.1f} MB |\n")
        f.write(f"\n🏆 **Best Performance Detector:** `{best_detector}` (Tốc độ {detector_scores[best_detector]['fps']:.1f} FPS, nhẹ nhất trên Edge CPU).\n\n")
        f.write("---\n\n")

        f.write("## 2. Kết Quả Benchmark Độc Lập Embedder (Sau Warm-up)\n\n")
        f.write("| Embedder | Vector Dim | Tốc độ (FPS) | Độ trễ Thuần (ms) | CPU (%) | RAM TB (MB) | Peak RAM (MB) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for emb_name, s in embedder_scores.items():
            f.write(f"| **{emb_name}** | {s['dim']}-D | **{s['fps']:.2f}** | {s['latency_ms']:.2f} ms | {s['cpu']:.1f}% | {s['ram']:.1f} MB | {s['peak_ram']:.1f} MB |\n")
        f.write(f"\n🏆 **Best Performance Embedder:** `{best_embedder}` (Độ trễ chỉ {embedder_scores[best_embedder]['latency_ms']:.2f} ms, tối ưu Edge).\n\n")
        f.write("---\n\n")

        f.write("## 3. Kết Quả Thực Nghiệm Chấm Công End-to-End Trên 10 Danh Tính\n\n")
        f.write("| Tổ Hợp Model | Ngưỡng $T^*$ | Sim Điểm Danh TB | Nhận Diện Đúng (%) | Sim Người Lạ TB | Từ Chối Đúng (%) | Latency E2E (ms) | FPS E2E |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in pipeline_results:
            f.write(f"| **{r['combo']}** | `{r['threshold']}` | **{r['valid_sim']:.4f}** | **{r['valid_acc']:.1f}%** | {r['impostor_sim']:.4f} | **{r['impostor_acc']:.1f}%** | {r['latency_ms']:.2f} ms | **{r['fps']:.2f} FPS** |\n")

        f.write("\n### Biểu đồ phân tích hiệu năng độ trễ:\n")
        f.write("- [Biểu đồ so sánh độ trễ Latency & FPS](../figures/benchmark_charts/pipeline_benchmark_comparison.png)\n\n")
        f.write("---\n\n")

    print(f"=== Đã lưu báo cáo benchmark chi tiết vào: {REPORT_MD_PATH} ===")


if __name__ == "__main__":
    main()
