# Báo Cáo Hiệu Chuẩn Ngưỡng & Đánh Giá Sinh Trắc Học (Step 5)

Báo cáo kiểm thử định lượng độ chính xác sinh trắc học trên tập dữ liệu gồm **20 ảnh (10 danh tính nhân viên: 1x2, 3x4, ..., 19x20)** kết hợp cùng bộ biến thể Augmentation (Thiếu sáng, ngược sáng, góc nghiêng đầu $\pm 12^\circ$, nhiễu hạt sensor).

- **Tổng số cặp kiểm thử:** `360` cặp Cùng Người (Genuine) và `3240` cặp Khác Người (Impostor).

---

## 1. Bảng Tổng Hợp Chỉ Số Hiệu Chuẩn Ngưỡng (Calibration Table)

| Model Embedder | Vector Dim | Mean Genuine | Min Genuine | Mean Impostor | Max Impostor | Separation Margin $\Delta$ | EER (%) | ROC AUC | Ngưỡng Khuyến Nghị $T^*$ |
|---|---|---|---|---|---|---|---|---|---|
| **arcface** | 512-D | 0.4453 | **0.0208** | 0.0945 | **0.976** | **-0.9552** | **7.78%** | **0.9643** | **`0.246`** |
| **facenet512** | 512-D | 0.6723 | **0.298** | 0.1583 | **0.6892** | **-0.3913** | **4.75%** | **0.9880** | **`0.53`** |
| **sface** | 128-D | 0.4651 | **0.1549** | 0.2386 | **0.5516** | **-0.3967** | **20.03%** | **0.8922** | **`0.33`** |
---

## 2. Phân Tích Đường Cong Lỗi FAR/FRR & Phân Bố Điểm Số

### Mô Hình Embedder: `ARCFACE`
- [Đồ thị FAR/FRR Trade-off & Score Distribution](../figures/roc_curves/roc_far_frr_arcface.png)
- **Nhận xét:** Ngưỡng tối ưu $T^*$ tại điểm cân bằng EER là `0.246`.

### Mô Hình Embedder: `FACENET512`
- [Đồ thị FAR/FRR Trade-off & Score Distribution](../figures/roc_curves/roc_far_frr_facenet512.png)
- **Nhận xét:** Ngưỡng tối ưu $T^*$ tại điểm cân bằng EER là `0.53`.

### Mô Hình Embedder: `SFACE`
- [Đồ thị FAR/FRR Trade-off & Score Distribution](../figures/roc_curves/roc_far_frr_sface.png)
- **Nhận xét:** Ngưỡng tối ưu $T^*$ tại điểm cân bằng EER là `0.33`.

---

## 3. Cấu Hình Ngưỡng Tối Ưu Cập Nhật Vào `pipeline.yaml`

Dựa trên kết quả thực nghiệm mới nhất, cấu hình ngưỡng tối ưu cho các mô hình:
```yaml
thresholds:
  arcface: 0.246
  facenet512: 0.53
  sface: 0.33
```
