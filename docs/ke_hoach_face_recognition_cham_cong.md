# Kế hoạch triển khai: Hệ thống nhận diện khuôn mặt cho máy chấm công

## 0. Mục tiêu & phạm vi

- **Bài toán**: xây dựng và đánh giá một pipeline nhận diện khuôn mặt phục vụ chấm công, test trên webcam laptop, có phân tích so sánh kỹ thuật (thư viện vs tự ghép, server vs edge).
- **Đầu ra**: báo cáo dạng khoa học (IMRAD), demo chạy được, có số liệu đánh giá đầy đủ (accuracy, FAR/FRR, tốc độ, chống giả mạo).
- **Giới hạn đã biết**: test bằng webcam RGB thường (không có cảm biến IR/depth) → phần liveness sẽ phản ánh giới hạn dưới (worst case) so với máy chấm công thương mại có phần cứng chuyên dụng. Cần ghi rõ điều này trong phần Discussion/Limitations của báo cáo.

---

## Lộ trình rút gọn — 7 ngày

**Nguyên tắc**: giữ nguyên dataset/preprocessing/pipeline baseline/liveness test/đánh giá/báo cáo (tạo số liệu và nội dung chính); rút gọn mạnh phần fine-tuning và ONNX/quantize (làm ở mức tối thiểu chứng minh quy trình, không cần tối ưu sâu).

| Ngày | Việc chính | Mục tiêu cuối ngày |
|---|---|---|
| 1 | Chọn dataset, phân tích phân phối sáng/tối, script CLAHE + augmentation cơ bản | Dữ liệu sạch, cân bằng, sẵn sàng train/test |
| 2 | Dựng pipeline baseline bằng DeepFace (detector + ArcFace), chạy thử trên webcam | Demo end-to-end chạy được |
| 3 | Benchmark 2-3 detector backend trong DeepFace, đo accuracy + tốc độ | Bảng so sánh detector cho báo cáo |
| 4 | Fine-tuning nhẹ: đóng băng phần lớn backbone, train vài epoch trên tập nhỏ | Model finetune xong, chứng minh đúng quy trình |
| 5 | Liveness test (print attack + replay attack), tính FAR/FRR/HTER toàn pipeline | Số liệu chống giả mạo + accuracy đầy đủ |
| 6 | Export ONNX 1 model, quantize INT8 cơ bản, so sánh nhanh trước/sau | Minh chứng quy trình edge |
| 7 | Tổng hợp số liệu, viết báo cáo IMRAD, buffer xử lý lỗi | Báo cáo hoàn chỉnh |

**Nếu bị dồn deadline, cắt theo thứ tự**: (1) giảm số detector benchmark → (2) bỏ quantize sâu, chỉ export ONNX và ghi là hướng tiếp theo → (3) fine-tuning chỉ cần chạy đúng quy trình, không cần accuracy cao → **không cắt** liveness test và FAR/FRR, đây là phần thể hiện rõ nhất tư duy sát thực tế doanh nghiệp.

---

## Giai đoạn 1 — Chuẩn bị dữ liệu

### 1.1 Chọn dataset
- Ưu tiên dataset có góc gần chính diện (near-frontal), tương tự góc lắp máy chấm công thực tế.
- Nguồn gợi ý:
  - Dataset chuyên biệt cho attendance (Roboflow Universe – "Attendance System Using Face Dataset").
  - Dataset chuẩn công nghiệp: WiderFace (cho detection), CASIA-WebFace / VGGFace2 / MS1M (cho recognition/embedding).
  - Nếu cần thêm case góc lệch để kiểm thử độ bền: EFHQ (Extreme Pose Face High-Quality dataset).
- Ghi chú: hầu hết dataset công khai lớn thiên lệch về ảnh gần chính diện, ánh sáng thuận lợi — cần chủ động bù bằng augmentation ở bước sau.

### 1.2 Phân tích đặc trưng & cân bằng dữ liệu
- Thống kê phân phối độ sáng (histogram) trên toàn bộ dataset đã chọn, đảm bảo đủ và cân bằng số lượng ảnh sáng/tối/ngược sáng.
- Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) cho các ảnh thiếu sáng/ngược sáng thay vì chỉnh brightness thô — thử nghiệm với vài mức clip-limit khác nhau, tránh làm méo texture da mặt.
- Thứ tự tiền xử lý chuẩn: **ảnh thô → CLAHE + denoise + sharpen → face detection → alignment**.

### 1.3 Augmentation cho case khó
- Case khó cần bổ sung: thiếu sáng/ngược sáng, đeo khẩu trang, đeo kính, góc nghiêng nhẹ.
- Kỹ thuật tham khảo: pipeline dùng ~20 augmentation/ảnh kết hợp CLAHE + denoise + sharpen cho kết quả tốt trong nghiên cứu tương tự (CCTV attendance).

---

## Giai đoạn 2 — Thiết kế pipeline & chọn mô hình

### 2.1 Kiến trúc pipeline chuẩn (5 bước)
`Detect → Align → Normalize → Represent (embedding) → Verify (matching)`

### 2.2 Hai nhánh triển khai cần đánh giá song song

| Nhánh | Mô tả |
|---|---|
| **Server** | Full-size models, ưu tiên độ chính xác: RetinaFace/YOLO-face (detection) + ArcFace/FaceNet (embedding) |
| **Edge AI** | Model nén (MobileFaceNet hoặc bản ArcFace đã quantize), tối ưu tốc độ/kích thước |

### 2.3 So sánh: dùng thư viện (DeepFace) vs tự ghép pipeline

- **DeepFace** đóng gói sẵn cả 5 bước, hỗ trợ nhiều backend detection (OpenCV, SSD, Dlib, MTCNN, RetinaFace, MediaPipe, YOLOv8/11/12, YuNet, CenterFace) và nhiều model embedding (VGG-Face, FaceNet, ArcFace, SFace, GhostFaceNet...). Có sẵn cờ `anti_spoofing=True` cho liveness cơ bản.
- **Tự ghép pipeline**: tải riêng weight pretrained (RetinaFace + ArcFace), tự viết code nối các bước — mức độ này khả thi và công bằng khi so sánh với DeepFace (tránh so sánh "tự train from scratch" vì không khả thi trong khung thời gian bài test).
- **Tiêu chí so sánh**: độ chính xác (cùng detector/embedding/threshold), tốc độ inference, khả năng tùy biến, chi phí phát triển, khả năng export edge, mức kiểm soát dữ liệu.
- **Lưu ý khi viết kết luận**: nếu dùng chung backbone, độ chính xác giữa 2 cách gần như bằng nhau (cùng forward pass qua cùng mạng) — nên trình bày theo hướng *trade-off kỹ thuật* (build vs. buy) thay vì "cái nào chính xác hơn".

---

## Giai đoạn 3 — Fine-tuning trên dữ liệu đã chọn

### 3.1 Chiến lược đóng băng layer (áp dụng nguyên lý transfer learning CNN)
- Đóng băng phần lớn backbone (các layer đầu, gần input — học đặc trưng tổng quát: cạnh, texture).
- Chỉ mở/train các layer cuối (gần output — đặc trưng đặc thù), ví dụ tham khảo: chỉ mở từ stage cuối cùng trong ResNet-based ArcFace.
- Cân nhắc **gradual unfreezing** (mở dần từ cuối lên) thay vì mở cố định ngay từ đầu.

### 3.2 Điểm đặc thù CV cần lưu ý (khác NLP)
- **BatchNorm** cần đưa về eval mode khi đóng băng — khác với LayerNorm trong Transformer, BatchNorm có running stats dễ bị "học lệch âm thầm" nếu quên xử lý.
- Learning rate cho phần mở đóng băng nên thấp (1/10–1/100 so với train từ đầu).
- **Cảnh báo**: full fine-tuning trên dataset có domain khác biệt (ánh sáng văn phòng, góc camera khác dataset gốc) có thể gây sụt giảm hiệu năng do gradient update lớn phá vỡ đặc trưng pretrained — ủng hộ mạnh cho việc đóng băng nhiều.
- Cân nhắc lại phạm vi: vì nhân viên là danh tính mới (không nằm trong lớp gốc lúc train), cách tiếp cận phổ biến hơn là **giữ nguyên backbone, chỉ dùng để trích embedding rồi so khớp cosine similarity** (open-set matching) — chỉ finetune sâu khi cần domain-adapt cho điều kiện camera/ánh sáng cụ thể.

### 3.3 Nếu finetune cả phần detection (YOLO)
- Mức độ đóng băng nên tùy theo mức mất cân bằng lớp trong dữ liệu: đóng băng backbone toàn phần khi cần giữ đặc trưng tổng quát; đóng băng nông hơn khi dữ liệu mất cân bằng nặng.

---

## Giai đoạn 4 — Kiểm thử chống giả mạo (Liveness / Anti-Spoofing)

- Đây là hạng mục **bắt buộc**, đánh giá riêng biệt với độ chính xác nhận diện thông thường.
- Kịch bản test khả thi với webcam thường (không cần IR):
  - **Print attack**: in ảnh khuôn mặt ra giấy, đưa trước camera.
  - **Replay attack**: mở ảnh/video khuôn mặt trên điện thoại/màn hình khác, đưa trước camera.
- Có thể tận dụng module `anti_spoofing=True` sẵn có trong DeepFace làm baseline, sau đó so sánh với cách tự triển khai passive liveness (phân tích texture/moiré pattern) nếu muốn đào sâu.
- Chuẩn đánh giá tham khảo: ISO/IEC 30107-3 (Presentation Attack Detection).
- Ghi rõ trong báo cáo: kết quả đo trên webcam RGB là giới hạn dưới, không đại diện cho hệ thống có phần cứng IR/depth thật.

---

## Giai đoạn 5 — Tối ưu hóa triển khai Edge

Thứ tự chuẩn (không phải lựa chọn thay thế nhau, mà là các bước nối tiếp):

1. **Finetune** model theo Giai đoạn 3.
2. **Export ONNX** — chỉ đổi định dạng, độ chính xác giữ nguyên (float32).
3. **Quantize** (thường INT8, post-training static quantization) — cần **calibration data lấy từ chính dataset đã chọn** (không dùng ảnh ngẫu nhiên, tránh mất chính xác âm thầm).
4. Convert tiếp sang runtime của thiết bị đích nếu cần (TFLite/OpenVINO/TensorRT), ví dụ chuỗi ONNX → TensorFlow → TFLite.
5. **Re-validate** trên tập held-out sau mỗi bước convert — không giả định convert xong là kết quả giữ nguyên như bản gốc.
6. **Recalibrate lại threshold matching** (cosine similarity) sau khi quantize, vì embedding có thể lệch nhẹ.
7. Nếu quantize cả YOLO (detection): cân nhắc giữ phần detection head ở FP32, chỉ quantize backbone/neck — vì INT8 có thể không đủ độ chính xác cho bounding box/confidence.

---

## Giai đoạn 6 — Đánh giá & chỉ số

| Nhóm chỉ số | Thành phần |
|---|---|
| Độ chính xác nhận diện | Accuracy, Rank-1 Recognition Rate |
| Biometric error | FAR (False Acceptance Rate), FRR (False Rejection Rate), HTER, EER |
| Chống giả mạo | APCER, BPCER (hoặc gộp HTER/EER cho phần liveness) |
| Hiệu năng | Thời gian xử lý (ms/ảnh), FPS, mức dùng bộ nhớ/GPU |
| So sánh detector | AP trên tập test (easy/medium/hard nếu dùng chuẩn kiểu WiderFace) |

- Đo riêng cho từng cấu hình: full model (server) vs model đã quantize (edge), thư viện vs tự ghép.

---

## Giai đoạn 7 — Demo & Test thực tế

- Test bằng webcam laptop: các điều kiện sáng khác nhau (đủ sáng, thiếu sáng, ngược sáng), có/không đeo khẩu trang/kính.
- Test edge case chống giả mạo (Giai đoạn 4) ngay trong buổi demo.
- Ghi chú rõ trong báo cáo: webcam khác camera máy chấm công thật về góc lắp đặt cố định, khoảng cách, độ phân giải, hồng ngoại — số liệu đo được chỉ mang tính minh họa kỹ thuật, không phải benchmark sản phẩm cuối.

---

## Giai đoạn 8 — Viết báo cáo (định dạng IMRAD)

1. **Introduction**: bối cảnh bài toán, mục tiêu, phạm vi và giới hạn (webcam vs camera chuyên dụng).
2. **Related Work / Background**: tổng quan các hướng tiếp cận (detection models, embedding models, anti-spoofing).
3. **Methods**:
   - Dataset & phân tích đặc trưng (Giai đoạn 1)
   - Kiến trúc pipeline, so sánh thư viện vs tự ghép (Giai đoạn 2)
   - Chiến lược fine-tuning (Giai đoạn 3)
   - Quy trình tối ưu edge (Giai đoạn 5)
4. **Experiments & Results**: bảng số liệu đầy đủ theo Giai đoạn 6, biểu đồ so sánh.
5. **Discussion**: giải thích trade-off, giới hạn của webcam test, so sánh với triển khai thực tế doanh nghiệp (phần cứng IR, tích hợp HRMS/payroll, bảo mật dữ liệu sinh trắc học — GDPR/CCPA, tại VN là Nghị định 13/2023/NĐ-CP).
6. **Conclusion & Future Work**.

---

## Rủi ro & lưu ý xuyên suốt

- Dữ liệu khuôn mặt là dữ liệu sinh trắc học nhạy cảm — cần đề cập ngắn gọn về mã hóa, kiểm soát truy cập, tuân thủ quy định bảo vệ dữ liệu cá nhân dù không cần implement đầy đủ.
- Không kết luận "model A chính xác hơn model B" nếu chúng dùng chung backbone — chỉ khác ở lớp wrapper/code.
- Luôn đo lại metric sau mỗi lần thay đổi pipeline (quantize, đổi threshold, đổi backend) — tránh dùng số liệu cũ cho cấu hình mới.

---

## Tài liệu tham khảo chính

- Nurlita et al. (2024). *Comparison of ArcFace and Dlib Performance in Face Recognition with Detection Using YOLOv8*. INOVTEK Polbeng – Seri Informatika.
- Facial Recognition Performance Evaluation with YOLOv8, ArcFace, and SVM in a Contactless Employee Attendance System. Jurnal Riset Informatika.
- Face Recognition-Based Mass Attendance Using YOLOv5 and ArcFace. ResearchGate.
- Deep Learning for Face Anti-Spoofing: A Survey. arXiv:2106.14948.
- Dataset Augmentation for Pose and Lighting Invariant Face Recognition. arXiv:1704.04326.
- EFHQ: Multi-purpose ExtremePose-Face-HQ Dataset. arXiv:2312.17205.
- An Analysis of Layer-Freezing Strategies for Enhanced Transfer Learning in YOLO Architectures. arXiv:2509.05490.
- PETALface: Parameter Efficient Transfer Learning for Low-resolution Face Recognition. arXiv:2412.07771.
- Watch Out for the Confusing Faces (ArcFace fine-tuning stages). arXiv:2303.13131.
- Adaptive Transfer Learning via Gradual Freezing. arXiv:2510.15372.
- DeepFace GitHub repository — serengil/deepface.
- ONNX to TF-Lite Model Conversion documentation — MLTK.
- ArcFace ResNet100 INT8 model card — Hugging Face onnxmodelzoo.
- RF-DETR / Ultralytics TFLite export documentation — Roboflow, Ultralytics Community.
- Biometric Authentication: Understanding FAR, FRR, and CER for Security Professionals — InventiveHQ.
- Enterprise Guide to Face Liveness Detection — HyperVerge (ISO/IEC 30107-3 reference).
- How to Choose the Best Face Recognition Attendance System: Buying Guide — Alibaba SmartBuy.

*(Ghi chú: đây là danh mục nguồn tổng hợp trong quá trình tư vấn — khi đưa vào báo cáo chính thức, bạn nên truy xuất lại từng nguồn và trích dẫn theo đúng định dạng khoa học yêu cầu, ví dụ IEEE hoặc APA, tùy form bài báo được giao.)*
