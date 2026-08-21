# Face Attendance Recognition — Edge AI Pipeline

Pipeline nhận diện khuôn mặt tối ưu cho thiết bị chấm công biên (Edge Devices / Mini PC / CPU):
`Ảnh đầu vào` → `CLAHE / Denoise` → `Detection` → `Landmark Alignment` → `Embedding` → `1:K Cosine Matching`.

Kèm theo ứng dụng demo Streamlit trực quan, bộ công cụ lượng tử hóa **Static PTQ INT8**, hiệu chuẩn ngưỡng sinh trắc học và hệ thống benchmark đo đạc phần cứng thời gian thực.

---

## 1. Cài Đặt Môi Trường

```bash
# 1. Tạo và kích hoạt môi trường ảo Python 3.10+
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 2. Danh Sách Các Script Thực Thi (Runnable Scripts)

Hệ thống cung cấp đầy đủ các script độc lập phục vụ cho từng mục đích kiểm thử và vận hành:

### A. Chạy Demo Web Streamlit
Khởi chạy giao diện web demo trực tiếp qua webcam máy tính:
```bash
streamlit run app/app.py
```
- **Tính năng:**
  - **Tab 1 (Điểm danh):** Nhận diện khuôn mặt, hiển thị độ trễ, FPS, %CPU và RAM của tiến trình.
  - **Tab 2 (Đăng ký nhân viên):** Đăng ký nhân viên mới với $N$ ảnh mẫu chụp từ camera.
  - **Sidebar:** Hoán đổi linh hoạt giữa 3 Detector (`mediapipe`, `retinaface`, `yolov8`) và 5 Embedder (`sface`, `arcface`, `arcface_int8`, `facenet512`, `facenet512_int8`).

---

### B. Xuất & Chuẩn Bị Toàn Bộ Mô Hình ONNX
Chuyển đổi toàn bộ mô hình (ArcFace Keras, FaceNet512 Keras, SFace, YOLOv8 PyTorch) sang định dạng chuẩn `.onnx` trong thư mục `models/`:
```bash
python scripts/export_onnx_models.py
```
- **Đầu ra:** Các file `arcface_fp32.onnx`, `facenet512_fp32.onnx`, `sface_fp32.onnx`, `yolov8n-face.onnx` sẵn sàng cho ONNX Runtime CPU.

---

### C. Hiệu Chuẩn Ngưỡng (Threshold Calibration)
Thực hiện so khớp cặp ảnh stress-test trên tập ảnh khuôn mặt người và biến thể Augmentation của các ảnh đó (thiếu sáng, ngược sáng, góc nghiêng $\pm 12^\circ$, nhiễu hạt) để tìm điểm cân bằng **EER (Equal Error Rate)**:
```bash
python scripts/calibrate_thresholds.py
```
- **Đầu ra:**
  - Báo cáo chi tiết: `outputs/report/threshold_calibration_report.md`
  - Biểu đồ đường cong ROC & phân phối điểm số: `outputs/figures/roc_curves/`
  - Ngưỡng tối ưu $T^*$ được khuyến nghị để cấu hình vào `config/pipeline.yaml`.

---

### D. Lượng Tử Hóa Static PTQ INT8
Thực hiện lượng tử hóa tĩnh (Post-Training Static Quantization) sang `INT8 QDQ` (`per_channel=True`) và đo đạc so sánh giữa bản FP32 và INT8:
```bash
python scripts/run_quantization_benchmark.py
```
- **Đầu ra:**
  - Các mô hình `models/arcface_int8.onnx` (~32.9 MB) và `models/facenet512_int8.onnx` (~23.4 MB).
  - Báo cáo số liệu nén, độ trễ và độ lệch vector (Cosine Drift): `outputs/report/quantization_report.md`
  - Biểu đồ so sánh: `outputs/figures/benchmark_charts/quantization_comparison.png`

---

### E. Benchmark Hiệu Năng
Chạy bộ đo đạc thực nghiệm toàn diện trên 4 pha:
1. *Pha 1:* Benchmark riêng 3 Detector độc lập. 
2. *Pha 2:* Benchmark riêng 3 Embedder độc lập.
3. *Pha 3:* Benchmark các tổ hợp End-to-End Pipeline (Valid Attendance & Impostor Attack).

```bash
python benchmarks/benchmark_pipeline.py
```
- **Đầu ra:**
  - Bảng số liệu chi tiết: `benchmarks/results/benchmark_results.csv`
  - Báo cáo phân tích: `outputs/report/benchmark_results_report.md`

---

### F. Chạy Toàn Bộ Bộ Kiểm Thử Tự Động (Unit Tests)
Kiểm tra tính toàn vẹn và tính đúng đắn của toàn bộ 21 test cases:
```bash
pytest -v
```

---

## 3. Cấu Trúc Thư Mục

```
face-attendance-recognition/
├── AGENTS.md                          # Bản đặc tả kỹ thuật & ràng buộc dự án
├── README.md                          # Hướng dẫn sử dụng & danh sách script
├── requirements.txt                   # Danh sách gói thư viện
├── config/
│   └── pipeline.yaml                  # Cấu hình tham số detector/embedder/ngưỡng
├── data/
│   └── test_images/                   # 20 ảnh mẫu (10 danh tính nhân viên)
├── models/                            # Thư mục lưu các mô hình ONNX FP32 & INT8
├── src/
│   ├── preprocessing/                 # CLAHE, khử nhiễu, tăng nét & augmentations
│   ├── detectors/                     # Factory cho MediaPipe, RetinaFace, YOLOv8
│   ├── embedders/                     # Factory cho ArcFace, FaceNet512, SFace
│   ├── liveness/                      # Passive (Laplacian) & Active (EAR) Liveness
│   ├── matching/                      # So khớp Cosine & SessionFaceStore
│   ├── evaluation/                    # Đo đạc CPU/RAM & Chỉ số FAR/FRR/EER
│   └── export/                        # Module lượng tử hóa Static PTQ INT8
├── scripts/
│   ├── export_onnx_models.py          # Xuất tất cả mô hình sang ONNX
│   ├── calibrate_thresholds.py        # Hiệu chuẩn ngưỡng tối ưu T*
│   └── run_quantization_benchmark.py  # Đo đạc lượng tử hóa đối đầu
├── benchmarks/
│   ├── benchmark_pipeline.py          # Kịch bản benchmark 4 pha
│   └── results/                       # Lưu file benchmark_results.csv
├── app/
│   ├── app.py                         # Ứng dụng Streamlit
│   └── components/                    # Module giao diện hiển thị metrics
├── tests/                             # Bộ kiểm thử PyTest (21 test cases)
└── outputs/
    ├── report/                        # Các báo cáo tổng hợp Markdown
    └── figures/                       # Các biểu đồ ROC và biểu đồ so sánh
```
