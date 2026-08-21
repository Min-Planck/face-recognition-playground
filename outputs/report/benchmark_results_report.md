# Báo Cáo Benchmark Pipeline Nhận Diện Khuôn Mặt

Báo cáo đo đạc hiệu năng thực tế trên toàn bộ tập dữ liệu gồm **20 ảnh (10 danh tính nhân viên)** theo phương pháp tổ hợp tối ưu (Best Detector x 3 Embedders và 3 Detectors x Best Embedder), sử dụng ngưỡng $T^*$ đã hiệu chuẩn thực nghiệm từ `config/pipeline.yaml`.

---

## 1. Kết Quả Benchmark Độc Lập Detector (Sau Warm-up)

| Detector | Tốc độ (FPS) | Độ trễ Thuần (ms) | CPU (%) | RAM TB (MB) | Peak RAM (MB) |
|---|---|---|---|---|---|
| **mediapipe** | **60.88** | 16.43 ms | 100.8% | 496.1 MB | 496.9 MB |
| **retinaface** | **2.65** | 377.69 ms | 176.8% | 664.1 MB | 664.3 MB |
| **yolov8** | **8.81** | 113.48 ms | 330.3% | 608.9 MB | 610.0 MB |

🏆 **Best Performance Detector:** `mediapipe` (Tốc độ 60.9 FPS, nhẹ nhất trên Edge CPU).

---

## 2. Kết Quả Benchmark Độc Lập Embedder (Sau Warm-up)

| Embedder | Vector Dim | Tốc độ (FPS) | Độ trễ Thuần (ms) | CPU (%) | RAM TB (MB) | Peak RAM (MB) |
|---|---|---|---|---|---|---|
| **arcface** | 512-D | **8.20** | 121.96 ms | 317.8% | 779.8 MB | 779.8 MB |
| **facenet512** | 512-D | **40.70** | 24.57 ms | 381.7% | 717.7 MB | 717.7 MB |
| **sface** | 128-D | **20.00** | 50.00 ms | 279.4% | 672.1 MB | 672.1 MB |

🏆 **Best Performance Embedder:** `facenet512` (Độ trễ chỉ 24.57 ms, tối ưu Edge).

---

## 3. Kết Quả Thực Nghiệm Chấm Công End-to-End Trên 10 Danh Tính

| Tổ Hợp Model | Ngưỡng $T^*$ | Sim Điểm Danh TB | Nhận Diện Đúng (%) | Sim Người Lạ TB | Từ Chối Đúng (%) | Latency E2E (ms) | FPS E2E |
|---|---|---|---|---|---|---|---|
| **mediapipe + arcface** | `0.26` | **0.5439** | **100.0%** | 0.2578 | **50.0%** | 130.08 ms | **7.69 FPS** |
| **mediapipe + facenet512** | `0.52` | **0.6825** | **90.0%** | 0.3649 | **70.0%** | 69.09 ms | **14.47 FPS** |
| **mediapipe + sface** | `0.34` | **0.5166** | **60.0%** | 0.4129 | **10.0%** | 94.73 ms | **10.56 FPS** |
| **retinaface + facenet512** | `0.52` | **0.7327** | **90.0%** | 0.3694 | **80.0%** | 523.57 ms | **1.91 FPS** |
| **yolov8 + facenet512** | `0.52` | **0.7330** | **90.0%** | 0.3651 | **80.0%** | 275.72 ms | **3.63 FPS** |

### Biểu đồ phân tích hiệu năng độ trễ:
- [Biểu đồ so sánh độ trễ Latency & FPS](../figures/benchmark_charts/pipeline_benchmark_comparison.png)

---

