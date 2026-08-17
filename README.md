# Face Attendance Recognition — Bài test kỹ thuật

Pipeline nhận diện khuôn mặt cho máy chấm công: detection → alignment → embedding → matching,
kèm demo Streamlit (đăng ký + nhận diện qua webcam) và benchmark tốc độ/tài nguyên.

## Cài đặt

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Cấu trúc dự án

Xem chi tiết đầy đủ trong [`AGENTS.md`](./AGENTS.md) — bao gồm kiến trúc, quy ước module, thứ tự triển khai.

Tài liệu kế hoạch/kiến thức nền: xem thư mục [`docs/`](./docs).

## Chạy nhanh

```bash
# Benchmark tốc độ/tài nguyên các tổ hợp detector x embedder
python benchmarks/benchmark_pipeline.py

# Demo Streamlit
streamlit run app/app.py
```

## Cấu hình

Chỉnh detector/embedder/threshold trong [`config/pipeline.yaml`](./config/pipeline.yaml), không cần sửa code.
