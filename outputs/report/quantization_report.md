# Báo Cáo Thực Nghiệm Lượng Tử Hóa Mô Hình (Edge INT8 Quantization)


## Bảng Tổng Hợp Kết Quả Đo Đạc Đối Đầu (FP32 vs INT8)

| Mô Hình | FP32 Size (MB) | INT8 Size (MB) | Tỷ Lệ Nén (%) | FP32 Latency (ms) | INT8 Latency (ms) | Tăng Tốc (Speedup) | Cosine Drift (Sim FP32/INT8) |
|---|---|---|---|---|---|---|---|
| **ArcFace (ResNet50)** | 130.24 MB | **32.9 MB** | **-74.74%** | 69.85 ms | **81.8 ms** | **0.85x** | **0.9833** (Min: 0.9511) |
| **FaceNet512** | 89.63 MB | **23.42 MB** | **-73.87%** | 31.11 ms | **31.43 ms** | **0.99x** | **0.9979** (Min: 0.9953) |

## Biểu Đồ So Sánh

![So Sánh Lượng Tử Hóa](../figures/benchmark_charts/quantization_comparison.png)

---
*Báo cáo được tạo tự động vào lúc 2026-08-21 16:14:59*