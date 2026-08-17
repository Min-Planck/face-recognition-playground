# Báo Cáo Thử Nghiệm Preprocessing & Augmentation

Báo cáo phân tích định lượng và trực quan hóa kết quả xử lý ảnh trên bộ 3 ảnh thực tế từ `data/test_images/`: `img_1.png`, `img_2.jpg`, và `img_3.jpg`.

---

## 1. Bảng Đo Lường Chất Lượng Ảnh (Quantitative Metrics)

| Ảnh Test | Kích Thước | Giai Đoạn Xử Lý | Độ Sáng TB (Luminance) | Độ Tương Phản (Std Dev) | Độ Sắc Nét (Laplacian Var) | Thay Đổi Độ Nét |
|---|---|---|---|---|---|---|
| **img_1.png** | 1024 × 1536 | Raw Input | 160.64 | 65.90 | 782.56 | - |
| | | **Preprocessed** | 156.89 | 61.47 | **1133.27** | **+44.8%** |
| **img_2.jpg** | 1536 × 2048 | Raw Input | 120.76 | 67.83 | 1253.50 | - |
| | | **Preprocessed** | 122.56 | 70.11 | **1817.89** | **+45.0%** |
| **img_3.jpg** | 960 × 1379 | Raw Input | 125.58 | 66.70 | 1257.28 | - |
| | | **Preprocessed** | 127.03 | 62.84 | **2450.74** | **+94.9%** |

### Đánh giá kỹ thuật:
1. **CLAHE trên kênh L (LAB Color Space):** Cân bằng dải sáng cục bộ, phục hồi chi tiết vùng bóng tối/khuất sáng mà không làm biến đổi sắc tố màu da tự nhiên ($A, B$ channels).
2. **Bilateral Denoise:** Khử nhiễu làm mịn hạt cảm biến nhưng bảo toàn hoàn hảo các đường biên sắc nét (edges) của mắt, mũi, miệng.
3. **Unsharp Sharpening:** Tăng độ tương phản vi mô quanh các landmark quan trọng, giúp trích xuất vector đặc trưng khuôn mặt có độ phân tách cao hơn (độ sắc nét tăng từ **+44% đến +95%**).

---

## 2. Hình Ảnh Trực Quan Pipeline Preprocessing (Style Banner Đen Chữ Trắng)

### 1. Ảnh `img_1.png`
- So sánh chuỗi tiền xử lý: [Xem ảnh so sánh](../figures/preprocessing_demo/comparison_img_1.jpg)
- Lưới các trường hợp khó (Hard Cases): [Xem ảnh Hard Cases](../figures/preprocessing_demo/hard_cases_img_1.jpg)

### 2. Ảnh `img_2.jpg`
- So sánh chuỗi tiền xử lý: [Xem ảnh so sánh](../figures/preprocessing_demo/comparison_img_2.jpg)
- Lưới các trường hợp khó (Hard Cases): [Xem ảnh Hard Cases](../figures/preprocessing_demo/hard_cases_img_2.jpg)

### 3. Ảnh `img_3.jpg`
- So sánh chuỗi tiền xử lý: [Xem ảnh so sánh](../figures/preprocessing_demo/comparison_img_3.jpg)
- Lưới các trường hợp khó (Hard Cases): [Xem ảnh Hard Cases](../figures/preprocessing_demo/hard_cases_img_3.jpg)

---

## 3. Bộ Augmentation Hard Cases Cho Camera Chấm Công

Các trường hợp biên khó được mô phỏng sát thực tế môi trường doanh nghiệp:
1. **Low Light:** Thiếu sáng khi chấm công sáng sớm hoặc ca đêm.
2. **Backlight:** Ngược sáng mạnh khi máy chấm công đặt gần cửa kính / sảnh văn phòng.
3. **Pose Tilt Left / Right:** Góc nghiêng đầu nhẹ ($\pm 12^\circ$) khi người đứng chưa thẳng camera.
4. **Sensor Noise:** Nhiễu hạt cảm biến do camera webcam chất lượng thấp trong điều kiện thiếu sáng.
