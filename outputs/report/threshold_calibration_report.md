# Báo Cáo Hiệu Chuẩn Ngưỡng & Đánh Giá Sinh Trắc Học (Step 5)

Báo cáo kiểm thử định lượng độ chính xác sinh trắc học trên tập dữ liệu gồm **20 ảnh (10 danh tính nhân viên: 1x2, 3x4, ..., 19x20)** kết hợp cùng bộ biến thể Augmentation (Thiếu sáng, ngược sáng, góc nghiêng đầu $\pm 12^\circ$, nhiễu hạt sensor).

- **Tổng số cặp kiểm thử:** `360` cặp Cùng Người (Genuine) và `3240` cặp Khác Người (Impostor).

---

## 1. Bảng Tổng Hợp Chỉ Số Hiệu Chuẩn Ngưỡng (Calibration Table)

| Model Embedder | Vector Dim | Mean Genuine | Min Genuine | Mean Impostor | Max Impostor | Separation Margin $\Delta$ | EER (%) | ROC AUC | Ngưỡng Khuyến Nghị $T^*$ |
|---|---|---|---|---|---|---|---|---|---|
| **arcface** | 512-D | 0.4725 | **0.0689** | 0.0805 | **0.8342** | **-0.7653** | **8.16%** | **0.9733** | **`0.257`** |
| **facenet512** | 512-D | 0.6806 | **0.3941** | 0.1455 | **0.6104** | **-0.2163** | **3.81%** | **0.9947** | **`0.517`** |
| **sface** | 128-D | 0.4266 | **0.0808** | 0.2579 | **0.5991** | **-0.5183** | **26.42%** | **0.8119** | **`0.336`** |
---

## 2. Phân Tích Đường Cong Lỗi FAR/FRR & Phân Bố Điểm Số

### Mô Hình Embedder: `ARCFACE`
- [Đồ thị FAR/FRR Trade-off & Score Distribution](../figures/roc_curves/roc_far_frr_arcface.png)
- **Nhận xét:** Ngưỡng tối ưu $T^*$ tại điểm cân bằng EER là `0.257`.

### Mô Hình Embedder: `FACENET512`
- [Đồ thị FAR/FRR Trade-off & Score Distribution](../figures/roc_curves/roc_far_frr_facenet512.png)
- **Nhận xét:** Ngưỡng tối ưu $T^*$ tại điểm cân bằng EER là `0.517`.

### Mô Hình Embedder: `SFACE`
- [Đồ thị FAR/FRR Trade-off & Score Distribution](../figures/roc_curves/roc_far_frr_sface.png)
- **Nhận xét:** Ngưỡng tối ưu $T^*$ tại điểm cân bằng EER là `0.336`.

---

## 3. Cấu Hình Ngưỡng Tối Ưu Cập Nhật Vào `pipeline.yaml`

Dựa trên kết quả thực nghiệm mới nhất, cấu hình ngưỡng tối ưu cho các mô hình:
```yaml
thresholds:
  arcface: 0.257
  facenet512: 0.517
  sface: 0.336
```
