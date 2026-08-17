# Báo Cáo Kết Quả Bước 2: Detector Factory & Embedder Factory

Báo cáo kiểm thử end-to-end các mô hình Face Detection và Face Embedding trên bộ 3 ảnh thực tế `data/test_images/`:
- `img_1.png`: Người A (Năm 1 đại học - Enrollment Template)
- `img_2.jpg`: Người A (Năm 4 tốt nghiệp - Cross-Age Inference)
- `img_3.jpg`: Người B (Người hoàn toàn khác - Impostor Test)

---

## 1. Kết Quả Face Detection

| Detector | Nền tảng Model | Ảnh Test | Số mặt | Bounding Box (x, y, w, h) | Confidence | Độ trễ (ms) |
|---|---|---|---|---|---|---|
| **mediapipe** | MediaPipe BlazeFace (TFLite CPU) | `img_1.png` | 1 | `(425, 413, 235, 235)` | 0.9527 | 45.83 ms |
| **mediapipe** | MediaPipe BlazeFace (TFLite CPU) | `img_2.jpg` | 1 | `(627, 762, 286, 286)` | 0.9214 | 39.1 ms |
| **mediapipe** | MediaPipe BlazeFace (TFLite CPU) | `img_3.jpg` | 1 | `(199, 316, 592, 592)` | 0.9545 | 17.74 ms |
| **retinaface** | InsightFace SCRFD / det_10g (ONNX Runtime) | `img_1.png` | 1 | `(430, 359, 246, 304)` | 0.7472 | 313.03 ms |
| **retinaface** | InsightFace SCRFD / det_10g (ONNX Runtime) | `img_2.jpg` | 1 | `(635, 681, 264, 361)` | 0.8765 | 364.24 ms |
| **retinaface** | InsightFace SCRFD / det_10g (ONNX Runtime) | `img_3.jpg` | 1 | `(240, 185, 483, 683)` | 0.8756 | 669.48 ms |
| **yolov8** | YOLOv8-Face / yolov8n-face (Ultralytics PyTorch) | `img_1.png` | 1 | `(434, 352, 247, 317)` | 0.8552 | 4901.96 ms |
| **yolov8** | YOLOv8-Face / yolov8n-face (Ultralytics PyTorch) | `img_2.jpg` | 1 | `(633, 681, 265, 360)` | 0.8966 | 1764.95 ms |
| **yolov8** | YOLOv8-Face / yolov8n-face (Ultralytics PyTorch) | `img_3.jpg` | 1 | `(242, 177, 483, 697)` | 0.8581 | 266.56 ms |

### Hình ảnh phát hiện và trích xuất khuôn mặt (Lưu tại `outputs/figures/detection_demo/`):

#### Ảnh `img_1.png`:
- **mediapipe**: [Ảnh Bounding Box](../figures/detection_demo/det_img_1_mediapipe.jpg) | [Ảnh Cắt & Căn chỉnh (112x112)](../figures/detection_demo/crop_img_1_mediapipe.jpg)
- **retinaface**: [Ảnh Bounding Box](../figures/detection_demo/det_img_1_retinaface.jpg) | [Ảnh Cắt & Căn chỉnh (112x112)](../figures/detection_demo/crop_img_1_retinaface.jpg)
- **yolov8**: [Ảnh Bounding Box](../figures/detection_demo/det_img_1_yolov8.jpg) | [Ảnh Cắt & Căn chỉnh (112x112)](../figures/detection_demo/crop_img_1_yolov8.jpg)

#### Ảnh `img_2.jpg`:
- **mediapipe**: [Ảnh Bounding Box](../figures/detection_demo/det_img_2_mediapipe.jpg) | [Ảnh Cắt & Căn chỉnh (112x112)](../figures/detection_demo/crop_img_2_mediapipe.jpg)
- **retinaface**: [Ảnh Bounding Box](../figures/detection_demo/det_img_2_retinaface.jpg) | [Ảnh Cắt & Căn chỉnh (112x112)](../figures/detection_demo/crop_img_2_retinaface.jpg)
- **yolov8**: [Ảnh Bounding Box](../figures/detection_demo/det_img_2_yolov8.jpg) | [Ảnh Cắt & Căn chỉnh (112x112)](../figures/detection_demo/crop_img_2_yolov8.jpg)

#### Ảnh `img_3.jpg`:
- **mediapipe**: [Ảnh Bounding Box](../figures/detection_demo/det_img_3_mediapipe.jpg) | [Ảnh Cắt & Căn chỉnh (112x112)](../figures/detection_demo/crop_img_3_mediapipe.jpg)
- **retinaface**: [Ảnh Bounding Box](../figures/detection_demo/det_img_3_retinaface.jpg) | [Ảnh Cắt & Căn chỉnh (112x112)](../figures/detection_demo/crop_img_3_retinaface.jpg)
- **yolov8**: [Ảnh Bounding Box](../figures/detection_demo/det_img_3_yolov8.jpg) | [Ảnh Cắt & Căn chỉnh (112x112)](../figures/detection_demo/crop_img_3_yolov8.jpg)

---

## 2. Kết Quả Face Embedding

| Embedder Model | Số chiều Vector (Dim) | Chuẩn hóa L2-Norm | Độ trễ Inference (ms) | Đặc điểm kiến trúc & Loss |
|---|---|---|---|---|
| **arcface** | 512-D | 1.0 | 5558.98 ms | Additive Angular Margin Loss (Chuẩn hiện đại, phân tách góc cao) |
| **facenet512** | 512-D | 1.0 | 6990.97 ms | Triplet Loss (Khoảng cách Euclidean/Cosine) |
| **sface** | 128-D | 1.0 | 348.17 ms | SphereFace variant (Tối ưu nhẹ cho Edge CPU) |

---

## 3. Ma Trận Tương Đồng Cosine Similarity Toàn Diện

So sánh giữa Cặp Cùng Người (Cross-Age) và Các Cặp Khác Người Hoàn Toàn:

| Cặp Ảnh So Sánh | Quan Hệ Thực Tế | ArcFace (512-D) | FaceNet512 (512-D) | SFace (128-D) |
|---|---|---|---|---|
| `img_1.png` vs `img_2.jpg` | 🟢 **CÙNG NGƯỜI (Person A: Năm 1 vs Năm 4)** | **0.4281** | **0.5562** | **0.293** |
| `img_1.png` vs `img_3.jpg` | 🔴 **KHÁC NGƯỜI (Person A vs Person B)** | 0.1021 | 0.0503 | 0.1595 |
| `img_2.jpg` vs `img_3.jpg` | 🔴 **KHÁC NGƯỜI (Person A vs Person B)** | -0.0219 | 0.0745 | 0.2298 |

### Phân Tích Khoảng Cách Tách Biệt (Separation Margin $\Delta = \min(\text{Same}) - \max(\text{Different})$):

| Model Embedder | Score Cùng Người | Max Score Khác Người | Biên Tách Biệt $\Delta$ | Khoảng Ngưỡng Tối Ưu $T^*$ |
|---|---|---|---|---|
| **arcface** | **0.4281** | 0.1021 | **+0.3260** | **[0.15 - 0.38]** (Độ chính xác 100%) |
| **facenet512** | **0.5562** | 0.0745 | **+0.4817** | **[0.12 - 0.51]** (Độ chính xác 100%) |
| **sface** | **0.2930** | 0.2298 | **+0.0632** | **[0.24 - 0.28]** (Độ chính xác 100%) |

### Kết luận cốt lõi:
1. **FaceNet512 và ArcFace đạt Margin cực rộng ($+0.48$ và $+0.32$):** Điểm số giữa người thật và kẻ mạo danh hoàn toàn tách biệt rõ rệt. Đặt ngưỡng quanh $0.25 - 0.35$ cho phép phân định chính xác $100\%$ mà không xảy ra False Acceptance (nhận nhầm) hay False Rejection (từ chối oan).
2. **SFace (128-D):** Có biên tách biệt $+0.063$, cần đặt ngưỡng chuẩn xác tại khoảng $0.25 - 0.27$ để cân bằng hiệu quả.
