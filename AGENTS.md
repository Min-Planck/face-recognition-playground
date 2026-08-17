# AGENTS.md — Brief cho coding agent

Đây là bối cảnh đầy đủ để một coding agent (Claude Code hoặc tương tự) tiếp nhận và triển khai dự án. Đọc file này trước, sau đó tham khảo `docs/` khi cần chi tiết sâu hơn về một quyết định kỹ thuật cụ thể.

## 1. Bài toán

Xây dựng và đánh giá pipeline nhận diện khuôn mặt cho máy chấm công, kèm demo Streamlit chạy được trên webcam laptop. Đây là bài test kỹ thuật 1 tuần (không phải sản phẩm production), nên ưu tiên: **chạy được, có số liệu đo thật, có báo cáo rõ ràng** — hơn là tối ưu hiệu năng tuyệt đối.

Chi tiết đầy đủ về bối cảnh, lý do lựa chọn kỹ thuật, và kiến thức nền: xem `docs/kien_thuc_can_thiet.md`.
Lộ trình 7 ngày và các giai đoạn: xem `docs/ke_hoach_face_recognition_cham_cong.md`.
Thiết kế ma trận thí nghiệm detector × embedder: xem `docs/ke_hoach_thi_nghiem_detector_embedder.md`.

## 2. Kiến trúc hệ thống (bắt buộc tuân thủ)

```
Ảnh đầu vào → CLAHE/denoise → Detection → Alignment → Embedding → Matching (cosine similarity)
```

- **2 pha tách biệt**: Enrollment (đăng ký, lưu N embedding/người) và Inference (1 ảnh mới, so khớp 1:K với toàn bộ embedding đã lưu).
- **Session-based storage**: KHÔNG dùng database. Lưu embedding trong `st.session_state` (Streamlit) hoặc biến in-memory tương đương. Dữ liệu mất khi restart — đây là chủ ý, không phải thiếu sót.
- **Detector và embedder phải hoán đổi được qua config**, không hardcode trong logic chính. Đọc từ `config/pipeline.yaml`.
- **Mỗi tổ hợp (detector, embedder) có threshold riêng** — không dùng chung 1 threshold cố định cho mọi combo.
- **KHÔNG fine-tune backbone** — dự án không có dữ liệu enrollment thật của doanh nghiệp nên bỏ qua bước fine-tune, dùng thẳng model pretrained + open-set matching (lý do đầy đủ: xem phần "Giai đoạn 3" trong `docs/ke_hoach_face_recognition_cham_cong.md`).

## 3. Danh sách model cần hỗ trợ

**Detector** (`src/detectors/detector_factory.py`):
- `retinaface` — qua DeepFace hoặc InsightFace (`buffalo_l` pack)
- `mediapipe` — qua package `mediapipe` (BlazeFace + Face Mesh, dùng cho passive liveness luôn)
- `yolov8` — qua `ultralytics`, optional, có thể bỏ nếu thiếu thời gian

**Embedder** (`src/embedders/embedder_factory.py`):
- `arcface` — qua DeepFace hoặc InsightFace
- `facenet512` — qua DeepFace
- `sface` — qua DeepFace

Pattern bắt buộc: factory function nhận `config` (từ YAML), trả về object có interface thống nhất, ví dụ:

```python
class BaseDetector:
    def detect(self, image: np.ndarray) -> list[FaceBox]: ...

class BaseEmbedder:
    def embed(self, face_crop: np.ndarray) -> np.ndarray: ...  # trả vector 512-dim (hoặc tùy model)
```

## 4. Cấu trúc thư mục (đã dựng sẵn khung rỗng, cần điền code)

```
face-attendance-recognition/
├── AGENTS.md                          # file này
├── README.md
├── requirements.txt
├── config/
│   └── pipeline.yaml                  # ĐÃ CÓ — cấu hình detector/embedder/threshold/liveness
├── data/
│   ├── raw/                           # dataset gốc (WiderFace, CASIA-WebFace... tùy chọn ở Giai đoạn 1)
│   ├── processed/                     # sau CLAHE + augmentation, dùng làm calibration data khi quantize
│   └── test_images/                   # ảnh dùng cho benchmark_pipeline.py
├── src/
│   ├── preprocessing/
│   │   ├── clahe.py                   # CLAHE + denoise + sharpen (thứ tự: raw → CLAHE → denoise → sharpen)
│   │   └── augmentation.py            # augment case khó: thiếu sáng, khẩu trang, kính, góc nghiêng
│   ├── detectors/
│   │   └── detector_factory.py        # factory pattern, đọc config, trả về detector object thống nhất interface
│   ├── embedders/
│   │   └── embedder_factory.py        # tương tự, cho embedder
│   ├── liveness/
│   │   ├── passive.py                 # Laplacian variance trên vùng landmark (má/mắt) — hoạt động với ảnh tĩnh
│   │   └── active.py                  # EAR/blink — CHỈ dùng nếu có video liên tục (streamlit-webrtc), optional
│   ├── matching/
│   │   └── matcher.py                 # cosine similarity, quản lý session store (dict: person_id -> list[embedding])
│   ├── evaluation/
│   │   ├── metrics.py                 # FAR, FRR, HTER, EER, Rank-1 accuracy
│   │   └── resource_monitor.py        # class ResourceMonitor (context manager, dùng psutil đo CPU/RAM)
│   └── export/
│       └── quantize.py                # ONNX export (nếu cần) + quantize_dynamic/quantize_static
├── benchmarks/
│   ├── benchmark_pipeline.py          # ĐÃ CÓ SẴN — 4 test case: detection-only, embedding-only, full pipeline, liveness overhead
│   └── results/                       # output CSV
├── app/
│   ├── app.py                         # Streamlit: 2 chức năng (đăng ký qua camera, nhận diện)
│   └── components/
│       └── metrics_display.py         # hiển thị CPU/RAM/latency/FPS/similarity score dùng st.metric
├── docs/                              # ĐÃ CÓ — 3 file kế hoạch/kiến thức nền, đọc khi cần quyết định kỹ thuật
├── tests/
│   └── test_matching.py               # unit test cho cosine similarity + threshold logic (không cần test model thật)
└── outputs/
    ├── report/                        # báo cáo cuối (IMRAD)
    └── figures/                       # biểu đồ từ benchmark (heatmap FPS, scatter accuracy vs FPS...)
```

## 5. Thứ tự triển khai đề xuất (khớp lộ trình 7 ngày trong docs)

1. `src/preprocessing/` (CLAHE + augmentation) → test độc lập trên vài ảnh mẫu trong `data/test_images/`
2. `src/detectors/` + `src/embedders/` (factory pattern, bắt đầu với 1 detector + 1 embedder chạy được end-to-end)
3. `benchmarks/benchmark_pipeline.py` — đã có sẵn, chỉ cần đảm bảo factory functions ở bước 2 tương thích interface mà script này gọi
4. `src/liveness/passive.py` (Laplacian variance) — ưu tiên trước `active.py` vì `st.camera_input` mặc định không hỗ trợ video liên tục
5. `src/evaluation/metrics.py` — cần trước khi tìm threshold cho từng combo (xem `docs/ke_hoach_thi_nghiem_detector_embedder.md` mục checklist)
6. `app/app.py` — ghép mọi thứ lại thành demo Streamlit
7. `src/export/quantize.py` — làm sau cùng, chỉ cần chạy được trên 1 model (không cần làm hết mọi combo)

## 6. Ràng buộc phần cứng tham chiếu (dùng để đánh giá kết quả benchmark có "đạt" hay không)

- Thiết bị chấm công thương mại điển hình: CPU quad-core ~1.2-1.4GHz, **RAM chỉ 1-2GB** (toàn hệ thống, không riêng cho model)
- Proxy hợp lý cho phần cứng edge trong báo cáo: Raspberry Pi 4/5 (4-8GB RAM)
- Mục tiêu tham khảo: mỗi combo detector+embedder nên dùng dưới vài trăm MB RAM để có ý nghĩa so sánh với ràng buộc trên

## 7. Việc KHÔNG cần làm (tránh over-engineer, đúng phạm vi 1 tuần)

- Không cần database thật (PostgreSQL, MongoDB...) — session-based là đủ và là chủ ý thiết kế
- Không cần fine-tune backbone thật trên dữ liệu doanh nghiệp (không có data)
- Không cần vector search (FAISS/Milvus) — số người test trong phạm vi vài chục là đủ cho brute-force cosine similarity
- Không cần làm active liveness (EAR/blink) trừ khi đã xong mọi thứ khác và còn dư thời gian
- Không cần quantize toàn bộ 9 combo detector×embedder — chỉ cần chứng minh quy trình trên 1-2 model

## 8. Definition of done cho từng module

- `benchmark_pipeline.py` chạy xong, xuất ra `benchmark_results.csv` có đủ 4 loại test case cho ít nhất 2 detector × 2 embedder
- `app.py` chạy được `streamlit run app/app.py`, đăng ký được ít nhất 1 người, nhận diện đúng người đó, hiển thị được CPU/RAM/latency/similarity score
- `metrics.py` tính đúng FAR/FRR/EER trên một tập cặp ảnh có nhãn (cùng người/khác người) tối thiểu vài chục cặp
- `quantize.py` chạy được trên ít nhất 1 model, có đo so sánh trước/sau (latency, RAM, cosine similarity của embedding trước/sau quantize)
