# Báo Cáo Hiệu Chuẩn Ngưỡng & Đánh Giá Sinh Trắc Học (Step 5)

Báo cáo kiểm thử định lượng độ chính xác sinh trắc học trên tập dữ liệu gồm **20 ảnh (10 danh tính nhân viên: 1x2, 3x4, ..., 19x20)** kết hợp cùng bộ biến thể Augmentation (Thiếu sáng, ngược sáng, góc nghiêng đầu $\pm 12^\circ$, nhiễu hạt sensor).

- **Tổng số cặp kiểm thử:** `360` cặp Cùng Người (Genuine) và `3240` cặp Khác Người (Impostor).

---

## 1. Bảng Tổng Hợp Chỉ Số Hiệu Chuẩn Ngưỡng (Calibration Table)

| Model Embedder | Vector Dim | Mean Genuine | Min Genuine | Mean Impostor | Max Impostor | Separation Margin $\Delta$ | EER (%) | ROC AUC | Ngưỡng Khuyến Nghị $T^*$ |
|---|---|---|---|---|---|---|---|---|---|
| **arcface** | 512-D | 0.4818 | **0.0546** | 0.0941 | **0.8323** | **-0.7777** | **5.20%** | **0.9812** | **`0.296`** |
| **facenet512** | 512-D | 0.6815 | **0.4338** | 0.1553 | **0.6312** | **-0.1974** | **3.81%** | **0.9951** | **`0.516`** |
| **sface** | 128-D | 0.4513 | **0.126** | 0.2786 | **0.5833** | **-0.4573** | **25.25%** | **0.8351** | **`0.357`** |
---

## 2. Phân Tích Đường Cong Lỗi FAR/FRR & Phân Bố Điểm Số

### Mô Hình Embedder: `ARCFACE`
- [Đồ thị FAR/FRR Trade-off & Score Distribution](../figures/roc_curves/roc_far_frr_arcface.png)
- **Nhận xét:** Ngưỡng tối ưu $T^*$ tại điểm cân bằng EER là `0.296`.

### Mô Hình Embedder: `FACENET512`
- [Đồ thị FAR/FRR Trade-off & Score Distribution](../figures/roc_curves/roc_far_frr_facenet512.png)
- **Nhận xét:** Ngưỡng tối ưu $T^*$ tại điểm cân bằng EER là `0.516`.

### Mô Hình Embedder: `SFACE`
- [Đồ thị FAR/FRR Trade-off & Score Distribution](../figures/roc_curves/roc_far_frr_sface.png)
- **Nhận xét:** Ngưỡng tối ưu $T^*$ tại điểm cân bằng EER là `0.357`.

---

## 3. Cấu Hình Ngưỡng Tối Ưu Cập Nhật Vào `pipeline.yaml`

Dựa trên kết quả thực nghiệm mới nhất, cấu hình ngưỡng tối ưu cho các mô hình:
```yaml
thresholds:
  arcface: 0.296
  facenet512: 0.516
  sface: 0.357
```
