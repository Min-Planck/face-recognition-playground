# Báo Cáo Benchmark Pipeline Nhận Diện Khuôn Mặt

Báo cáo đo đạc hiệu năng thực tế trên toàn bộ tập dữ liệu gồm **20 ảnh (10 danh tính nhân viên)** theo phương pháp tổ hợp tối ưu (Best Detector x 3 Embedders và 3 Detectors x Best Embedder), sử dụng ngưỡng $T^*$ đã hiệu chuẩn thực nghiệm từ `config/pipeline.yaml`.

---

## 1. Kết Quả Benchmark Độc Lập Detector (Sau Warm-up)

| Detector | Tốc độ (FPS) | Độ trễ Thuần (ms) | CPU (%) | RAM TB (MB) | Peak RAM (MB) |
|---|---|---|---|---|---|
| **mediapipe** | **99.49** | 10.05 ms | 252.0% | 494.2 MB | 494.2 MB |
| **retinaface** | **2.62** | 381.59 ms | 167.1% | 665.5 MB | 665.7 MB |
| **yolov8** | **6.09** | 164.31 ms | 277.1% | 608.9 MB | 615.1 MB |

🏆 **Best Performance Detector:** `mediapipe` (Tốc độ 99.5 FPS, nhẹ nhất trên Edge CPU).

---

## 2. Kết Quả Benchmark Độc Lập Embedder (Sau Warm-up)

| Embedder | Vector Dim | Tốc độ (FPS) | Độ trễ Thuần (ms) | CPU (%) | RAM TB (MB) | Peak RAM (MB) |
|---|---|---|---|---|---|---|
| **arcface** | 512-D | **6.01** | 166.41 ms | 227.3% | 909.6 MB | 909.6 MB |
| **facenet512** | 512-D | **28.88** | 34.63 ms | 245.2% | 812.1 MB | 812.1 MB |
| **sface** | 128-D | **18.86** | 53.02 ms | 210.8% | 706.0 MB | 706.0 MB |

🏆 **Best Performance Embedder:** `facenet512` (Độ trễ chỉ 34.63 ms, tối ưu Edge).

---

## 3. Kết Quả Thực Nghiệm Chấm Công End-to-End Trên 10 Danh Tính

| Tổ Hợp Model | Ngưỡng $T^*$ | Sim Điểm Danh TB | Nhận Diện Đúng (%) | Sim Người Lạ TB | Từ Chối Đúng (%) | Latency E2E (ms) | FPS E2E |
|---|---|---|---|---|---|---|---|
| **mediapipe + arcface** | `0.3` | **0.5311** | **100.0%** | 0.2615 | **60.0%** | 192.02 ms | **5.21 FPS** |
| **mediapipe + facenet512** | `0.52` | **0.6884** | **90.0%** | 0.3676 | **70.0%** | 113.79 ms | **8.79 FPS** |
| **mediapipe + sface** | `0.36` | **0.5156** | **60.0%** | 0.4199 | **10.0%** | 69.64 ms | **14.36 FPS** |
| **retinaface + facenet512** | `0.52` | **0.7363** | **100.0%** | 0.3760 | **70.0%** | 381.54 ms | **2.62 FPS** |
| **yolov8 + facenet512** | `0.52` | **0.7237** | **90.0%** | 0.3765 | **90.0%** | 308.02 ms | **3.25 FPS** |

### Biểu đồ phân tích hiệu năng độ trễ:
- [Biểu đồ so sánh độ trễ Latency & FPS](../figures/benchmark_charts/pipeline_benchmark_comparison.png)

---

