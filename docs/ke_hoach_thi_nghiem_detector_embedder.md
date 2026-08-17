# Kế hoạch thí nghiệm: 3 Detector × 3 Embedder

## 1. Model dự kiến sử dụng

**3 Detector** (đại diện 3 kiến trúc khác nhau, không trùng họ):
| Detector | Nguồn | Vai trò đại diện |
|---|---|---|
| RetinaFace | InsightFace / DeepFace | Baseline accuracy cao, chuẩn "nặng" |
| MediaPipe (BlazeFace) | Google MediaPipe | Baseline tốc độ cao, chuẩn "nhẹ" cho edge |
| YOLO-face | Ultralytics | Điểm cân bằng giữa 2 thái cực trên, dễ ghép tracking |

**3 Embedder** (đại diện các thế hệ loss khác nhau):
| Embedder | Nguồn | Vai trò đại diện |
|---|---|---|
| ArcFace | InsightFace / DeepFace | Chuẩn phổ biến nhất hiện nay (angular margin) |
| FaceNet (Facenet512) | DeepFace | Thế hệ trước ArcFace (triplet loss), baseline so sánh |
| SFace | DeepFace | Hướng nhẹ hơn, đại diện nhánh tối ưu tốc độ |

*(AdaFace có thể thêm làm bonus nếu còn thời gian — cần tích hợp thủ công như đã bàn, không tính vào 3 embedder chính vì tốn công tích hợp riêng.)*

## 2. Thứ tự thực hiện — chia 2 pha, không làm full-cross ngay từ đầu

### Pha 1 — Đo tốc độ & tài nguyên (làm FULL-CROSS 3×3 = 9 tổ hợp)
Lý do làm full ở pha này: đo tốc độ không cần nhãn/threshold, chi phí rẻ nếu tách detection ra khỏi embedding (detect 1 lần/detector, tái sử dụng cho cả 3 embedder).

**Thứ tự chạy:**
1. Detect + align bằng từng detector (3 lần), lưu lại ảnh mặt đã crop cho mỗi detector → có 3 bộ ảnh crop.
2. Với mỗi bộ ảnh crop, chạy lần lượt cả 3 embedder → 9 phép đo tốc độ.
3. Ghi nhận: FPS, latency trung bình (ms), CPU%, RAM (MB) cho từng ô trong ma trận 3×3.

### Pha 2 — Đo độ chính xác (rút gọn theo "one-factor-at-a-time" nếu thiếu thời gian)

**Bước 2a**: Từ kết quả Pha 1, chọn detector có tốc độ/tài nguyên tốt nhất trong ngưỡng chấp nhận được → cố định detector này, chạy accuracy với cả 3 embedder (3 phép đo).

**Bước 2b**: Từ kết quả 2a, chọn embedder tốt nhất → cố định embedder này, chạy accuracy với 2 detector còn lại (2 phép đo).

→ Tổng: 5 phép đo accuracy thay vì 9, tiết kiệm thời gian tìm threshold riêng cho từng ô.

**Nếu còn dư thời gian**: mở rộng nốt 4 ô còn thiếu để có ma trận accuracy đầy đủ 3×3, phục vụ phân tích tương tác detector-embedder sâu hơn trong phần Discussion.

## 3. Checklist bắt buộc trước khi tin kết quả mỗi ô (tránh sai lệch âm thầm)

- [ ] Xác nhận kích thước crop đầu vào đúng chuẩn của từng embedder (thường 112×112, nhưng kiểm tra lại từng model)
- [ ] Xác nhận thứ tự kênh màu đúng (RGB vs BGR) — đặc biệt lưu ý nếu sau này thêm AdaFace
- [ ] Xác nhận số điểm landmark dùng để align tương thích (5 điểm RetinaFace vs landmark khác của MediaPipe/YOLO)
- [ ] Mỗi ô (detector, embedder) phải tìm **threshold riêng** qua ROC/EER trên tập validation — không dùng chung 1 threshold cho cả ma trận

## 4. Độ đo (metrics) cho từng pha

**Pha 1 — Tốc độ & tài nguyên (đo cho toàn bộ 9 ô):**
- FPS (frame/giây)
- Latency trung bình (ms/ảnh)
- CPU trung bình (%)
- RAM trung bình & đỉnh (MB)

**Pha 2 — Độ chính xác (đo cho các ô đã chọn):**
- Accuracy / Rank-1 Recognition Rate
- FAR (False Acceptance Rate)
- FRR (False Rejection Rate)
- HTER = (FAR + FRR) / 2
- EER (Equal Error Rate) — dùng để chọn threshold cho từng ô
- Threshold tối ưu tìm được (ghi lại để dùng cho demo/triển khai sau)

## 5. Dữ liệu cần chuẩn bị trước khi chạy

- Tập ảnh test đa dạng điều kiện sáng (đủ sáng / thiếu sáng / ngược sáng), đã cân bằng theo Giai đoạn 1 của plan tổng.
- Tập cặp ảnh có nhãn (cùng người / khác người) để tính FAR/FRR/EER ở Pha 2 — cần ít nhất vài chục cặp mỗi loại để số liệu có ý nghĩa thống kê tối thiểu.
- Danh sách threshold mặc định tham khảo ban đầu (trước khi tự tìm lại): ArcFace ~0.68 (cosine, theo DeepFace), FaceNet512 ~0.30, SFace theo tài liệu thư viện — chỉ dùng làm điểm khởi đầu dò, không dùng làm kết quả cuối.

## 6. Output mong đợi để đưa vào báo cáo

- 1 bảng/heatmap 3×3 cho tốc độ (FPS)
- 1 bảng/heatmap cho accuracy (ít nhất 5 ô theo Bước 2a/2b, lý tưởng đủ 9 ô)
- 1 biểu đồ scatter accuracy (trục Y) vs FPS (trục X) cho toàn bộ 9 ô — trực quan hóa trade-off, dùng để chọn ra combo đề xuất cho nhánh Server và nhánh Edge
- 1 đoạn Discussion ngắn về ô nào có tương tác bất ngờ (nếu có) giữa detector và embedder
