# Báo Cáo Benchmark Pipeline Nhận Diện Khuôn Mặt

Báo cáo đo đạc hiệu năng thực tế trên toàn bộ tập dữ liệu gồm **20 ảnh (10 danh tính nhân viên)** theo phương pháp tổ hợp tối ưu (Best Detector x 3 Embedders và 3 Detectors x Best Embedder), sử dụng ngưỡng $T^*$ đã hiệu chuẩn thực nghiệm từ `config/pipeline.yaml`.

---

## 1. Kết Quả Benchmark Độc Lập Detector (Sau Warm-up)

| Detector | Tốc độ (FPS) | Độ trễ Thuần (ms) | CPU (%) | RAM TB (MB) | Peak RAM (MB) |
|---|---|---|---|---|---|
| **mediapipe** | **32.15** | 31.11 ms | 116.9% | 530.0 MB | 536.0 MB |
| **retinaface** | **3.08** | 324.71 ms | 171.3% | 688.3 MB | 688.5 MB |
| **yolov8** | **7.19** | 139.05 ms | 205.0% | 845.7 MB | 846.8 MB |

🏆 **Best Performance Detector:** `mediapipe` (Tốc độ 32.1 FPS, nhẹ nhất trên Edge CPU).

---

## 2. Kết Quả Benchmark Độc Lập Embedder (Sau Warm-up)

| Embedder | Vector Dim | Tốc độ (FPS) | Độ trễ Thuần (ms) | CPU (%) | RAM TB (MB) | Peak RAM (MB) |
|---|---|---|---|---|---|---|
| **arcface** | 512-D | **1.62** | 618.95 ms | 103.7% | 1313.7 MB | 1313.8 MB |
| **facenet512** | 512-D | **2.01** | 497.80 ms | 98.5% | 1425.6 MB | 1425.7 MB |
| **sface** | 128-D | **27.40** | 36.50 ms | 74.8% | 1504.6 MB | 1504.6 MB |

🏆 **Best Performance Embedder:** `sface` (Độ trễ chỉ 36.50 ms, tối ưu Edge).

---

## 3. Kết Quả Thực Nghiệm Chấm Công End-to-End Trên 10 Danh Tính

| Tổ Hợp Model | Ngưỡng $T^*$ | Sim Điểm Danh TB | Nhận Diện Đúng (%) | Sim Người Lạ TB | Từ Chối Đúng (%) | Latency E2E (ms) | FPS E2E |
|---|---|---|---|---|---|---|---|
| **mediapipe + arcface** | `0.24` | **0.4969** | **100.0%** | 0.2497 | **30.0%** | 668.32 ms | **1.50 FPS** |
| **mediapipe + facenet512** | `0.53` | **0.6907** | **100.0%** | 0.3983 | **80.0%** | 1475.68 ms | **0.68 FPS** |
| **mediapipe + sface** | `0.32` | **0.5197** | **80.0%** | 0.3766 | **30.0%** | 282.29 ms | **3.54 FPS** |
| **retinaface + sface** | `0.32` | **0.5852** | **100.0%** | 0.2295 | **100.0%** | 995.94 ms | **1.00 FPS** |
| **yolov8 + sface** | `0.32` | **0.4533** | **80.0%** | 0.3219 | **50.0%** | 866.87 ms | **1.15 FPS** |

### Biểu đồ phân tích hiệu năng độ trễ:
- [Biểu đồ so sánh độ trễ Latency & FPS](../figures/benchmark_charts/pipeline_benchmark_comparison.png)

---

