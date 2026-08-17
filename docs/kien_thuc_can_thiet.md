# Kiến thức nền cần nắm — Face Recognition cho máy chấm công

Tài liệu này tổng hợp các khái niệm kỹ thuật đã bàn, sắp theo trình tự pipeline thực tế. Dùng để ôn lại nhanh trước khi làm hoặc khi giải thích lựa chọn trong báo cáo/phỏng vấn.

---

## 1. Đặc thù dữ liệu & tiền xử lý

- **Bias của dataset công khai**: đa số thiên lệch về ảnh gần chính diện, ánh sáng tốt — cần chủ động bù bằng augmentation, không mặc định dataset đã đại diện đủ điều kiện thực tế.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): tăng tương phản *cục bộ* theo từng vùng ảnh, khác với cân bằng histogram toàn cục (dễ làm cháy sáng/mất chi tiết). Dùng cho ảnh ngược sáng/thiếu sáng — tham số quan trọng là *clip limit* (giới hạn khuếch đại tương phản), cần thử nhiều mức vì để quá cao sẽ làm méo texture da, ảnh hưởng ngược tới embedding.
- **Thứ tự chuẩn**: ảnh thô → CLAHE + denoise + sharpen → detection → alignment → embedding.
- **Case khó cần augment**: thiếu sáng, ngược sáng, khẩu trang, kính, góc nghiêng nhẹ.

---

## 2. Kiến trúc pipeline chuẩn (5 bước)

`Detect → Align → Normalize → Represent (embedding) → Verify (matching)`

- **Detect**: định vị bounding box khuôn mặt trong khung hình.
- **Align**: xoay/căn khuôn mặt về vị trí chuẩn dựa trên landmark (mắt-mũi-miệng) — cải thiện accuracy đáng kể (alignment tốt có thể tăng vài % tới cả chục %, tùy benchmark).
- **Represent**: đưa ảnh mặt đã crop/align qua model embedding → ra vector đặc trưng (thường 128-512 chiều).
- **Verify**: so khớp vector bằng cosine similarity (phổ biến nhất) với một ngưỡng (threshold) được hiệu chỉnh trên tập validation.

---

## 3. Các model Detection — đặc điểm & khi nào dùng

| Model | Đặc điểm | Phù hợp khi |
|---|---|---|
| **RetinaFace** | Độ chính xác cao, xử lý tốt mặt nhỏ/che khuất/đông người | Cần accuracy cao, không quá cần real-time tuyệt đối |
| **YOLO-face (v5/v8/v11...)** | Real-time, >30 FPS, dễ ghép thêm tracking (ByteTrack) | Camera quay liên tục, cần tốc độ |
| **MTCNN** | 3 tầng (P-Net/R-Net/O-Net), kinh điển, ~20 FPS trên CPU | Baseline dễ triển khai, yếu với góc nghiêng mạnh |
| **SCRFD** | Cân bằng tốt tốc độ/accuracy, nhiều mức cấu hình, mặc định trong InsightFace | Use case tổng quát, có sẵn trong `buffalo_l` |
| **MediaPipe (BlazeFace + Face Mesh)** | Rất nhanh (200-1000+ FPS mobile), 468 landmark, CPU-friendly | Edge/mobile, 1 người/gần camera — không tối ưu cho ảnh đông người/mặt nhỏ |
| **YuNet** | Nhẹ, tốt trên ảnh nhiễu/chất lượng thấp | Edge CPU, điều kiện ảnh không lý tưởng |

**Ghi nhớ quan trọng**: chất lượng detector ảnh hưởng trực tiếp tới accuracy nhận diện cuối — đổi từ detector yếu sang RetinaFace từng giúp tăng verification accuracy của ArcFace hơn 1 điểm % trên benchmark góc nghiêng (CFP-FP). Detection không phải bước "phụ", chọn sai ở đây kéo tụt cả pipeline.

---

## 4. Các model Embedding/Recognition

| Model | Đặc điểm |
|---|---|
| **ArcFace** | Dùng Additive Angular Margin Loss, chuẩn phổ biến nhất hiện nay, embedding 512 chiều |
| **FaceNet** | Dùng Triplet Loss, ra đời sớm hơn, vẫn dùng rộng rãi (kết hợp MTCNN) |
| **AdaFace** | Cải tiến ArcFace bằng Quality-Adaptive Margin — xử lý tốt hơn ảnh chất lượng thấp/che khuất |
| **SFace, GhostFaceNet** | Hướng tới nhẹ hơn, phù hợp edge |

**Khái niệm cốt lõi**: các model này học theo hướng *metric learning* (đưa ảnh cùng người lại gần nhau, khác người ra xa nhau trong không gian embedding), khác với classification thông thường — vì vậy khi có nhân viên mới, thường **không cần retrain lại model**, chỉ cần trích embedding và so khớp (open-set matching).

---

## 5. Thư viện & framework

- **DeepFace**: wrapper gộp sẵn cả 5 bước, hỗ trợ nhiều backend detection/embedding, có sẵn `anti_spoofing=True`. Dễ dùng, tốt cho prototype nhanh, nhưng chưa hỗ trợ sẵn SCRFD/AdaFace.
- **InsightFace**: nguồn gốc của SCRFD + ArcFace, cần dùng trực tiếp nếu muốn thử combo này.
- **Tự ghép pipeline**: tải riêng weight pretrained, tự nối các bước — công bằng để so sánh với thư viện, không nên hiểu nhầm là "tự train from scratch" (không khả thi trong thời gian ngắn).
- **Lưu ý khi so sánh**: nếu dùng chung backbone, độ chính xác giữa thư viện và tự ghép gần như bằng nhau — khác biệt thực sự nằm ở overhead, khả năng tùy biến, khả năng export edge.

---

## 6. Fine-tuning

- **Nguyên lý transfer learning cho CNN**: đóng băng layer đầu (gần input, học đặc trưng tổng quát: cạnh, texture), chỉ train layer cuối (gần output, đặc trưng đặc thù) — giống nguyên lý freeze/train trong NLP, chỉ khác kiến trúc.
- **BatchNorm**: cần đưa về eval mode khi đóng băng — nếu quên, running mean/variance vẫn bị cập nhật dù weight đã đóng băng, gây lỗi âm thầm. Đây là điểm khác biệt quan trọng so với LayerNorm trong Transformer (NLP).
- **Rủi ro full fine-tuning**: nếu domain dữ liệu mới khác nhiều so với dữ liệu gốc (ánh sáng văn phòng, góc camera), gradient update lớn có thể phá vỡ đặc trưng pretrained → hiệu năng tụt thay vì tăng. Nên ưu tiên đóng băng nhiều, mở dần (gradual unfreezing).
- **Learning rate**: thấp hơn nhiều so với train từ đầu (thường 1/10 – 1/100).
- **Câu hỏi cần tự hỏi trước khi finetune sâu**: có thực sự cần finetune backbone, hay chỉ cần dùng model pretrained để trích embedding rồi so khớp? (Thường chỉ cần domain-adapt khi điều kiện camera/ánh sáng công ty khác biệt rõ rệt.)

---

## 7. Triển khai Edge

- **ONNX**: chỉ là định dạng trung gian, không tự làm model nhanh/nhẹ hơn — độ chính xác giữ nguyên so với bản gốc (float32).
- **Quantization** (thường INT8): bước thực sự giảm kích thước/độ trễ, đánh đổi bằng một phần độ chính xác.
- **Calibration data**: bắt buộc phải lấy từ chính dataset đã dùng (không dùng ảnh ngẫu nhiên) — chất lượng calibration quyết định chất lượng quantize.
- **Re-validate sau mỗi bước convert**: không giả định convert xong là giữ nguyên kết quả — cần đo lại trên tập held-out.
- **Recalibrate threshold matching**: sau quantize, embedding có thể lệch nhẹ → threshold cũ có thể không còn tối ưu.
- **YOLO quantize**: nên giữ detection head ở FP32, chỉ quantize backbone/neck — vì INT8 khó biểu diễn chính xác bounding box/confidence.

---

## 8. Liveness / Anti-Spoofing

### Phân loại tấn công (Presentation Attack)
- **Print attack**: ảnh in giấy
- **Replay attack**: phát lại ảnh/video trên màn hình khác
- **Mask attack**: mặt nạ 2D/3D (khó test nếu không có mask thật)

### Hai hướng kỹ thuật
| Hướng | Cách làm | Chống được | Điểm yếu |
|---|---|---|---|
| **Active liveness** | Challenge-response: yêu cầu chớp mắt (đo qua EAR — Eye Aspect Ratio), quay đầu, cười | Print attack (ảnh tĩnh) | Không chống được replay (video vẫn chớp mắt bình thường) |
| **Passive liveness** | Phân tích texture (LBP, Laplacian variance) trên vùng landmark (mắt, má) để bắt moiré pattern/độ mờ bất thường khi chụp lại màn hình | Print + phần lớn replay | Cần dữ liệu huấn luyện/ngưỡng phù hợp, khó chống mask chất lượng cao |

- **MediaPipe Face Mesh** (468 landmark) hỗ trợ tốt cả hai hướng: dùng landmark để tính EAR (active) hoặc để khoanh vùng chính xác cho phân tích texture (passive).
- **Chuẩn đánh giá**: ISO/IEC 30107-3 (Presentation Attack Detection).
- **Giới hạn cần ghi rõ trong báo cáo**: giải pháp software-only trên RGB thường là "giới hạn dưới" — máy chấm công thương mại thật thường có thêm cảm biến IR/depth để tăng độ tin cậy.

---

## 9. Bộ chỉ số đánh giá

| Chỉ số | Ý nghĩa |
|---|---|
| **Accuracy / Rank-1** | Tỷ lệ nhận diện đúng danh tính |
| **FAR** (False Acceptance Rate) | Tỷ lệ chấp nhận nhầm người không hợp lệ |
| **FRR** (False Rejection Rate) | Tỷ lệ từ chối nhầm người hợp lệ |
| **HTER** | Trung bình (FAR + FRR)/2 |
| **EER** (Equal Error Rate) | Điểm mà FAR = FRR — dùng để chọn threshold cân bằng |
| **APCER / BPCER** | Tương đương FAR/FRR nhưng dành riêng cho bài toán chống giả mạo (liveness) |
| **FPS / latency (ms)** | Tốc độ xử lý thực tế — quan trọng ngang accuracy với ứng dụng real-time |

---

## 10. Công thức/khái niệm nên nhớ khi giải thích

- **Cosine similarity**: `sim = (A·B) / (||A|| ||B||)` — đo góc giữa 2 vector embedding, giá trị càng gần 1 càng giống nhau. Threshold quyết định "cùng người hay khác người".
- **EAR (Eye Aspect Ratio)**: tỷ lệ giữa khoảng cách dọc và ngang của các điểm landmark quanh mắt — giảm đột ngột khi nhắm mắt, dùng để đếm số lần chớp mắt.

---

## 11. Bối cảnh doanh nghiệp cần biết (không chỉ kỹ thuật)

- Dữ liệu khuôn mặt là dữ liệu sinh trắc học nhạy cảm → cần mã hóa, kiểm soát truy cập, tuân thủ quy định (GDPR/CCPA quốc tế, Nghị định 13/2023/NĐ-CP tại Việt Nam).
- Hệ thống thật cần thêm: cooldown chống chấm công trùng, fallback khi không nhận diện được (mã nhân viên/thẻ), tích hợp HRMS/payroll, logging có timestamp.
- Webcam test ≠ camera máy chấm công thật (góc cố định, khoảng cách, IR) — luôn nêu rõ giới hạn này khi trình bày kết quả.
