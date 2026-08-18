"""
Component hiển thị thông số đo đạc thời gian thực (Telemetry & Metrics Display)
cho ứng dụng chấm công Streamlit.
"""

from typing import Any, Dict, Optional
import streamlit as st
import numpy as np


def render_metrics_header(
    latency_ms: float,
    fps: float,
    cpu_percent: float,
    ram_mb: float,
):
    """Hiển thị 3 card metrics chính (Latency, CPU, RAM) ở đầu trang."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Độ Trễ (Latency)",
            value=f"{latency_ms:.1f} ms",
            delta=f"{fps:.1f} FPS",
            delta_color="normal",
        )

    with col2:
        st.metric(
            label="CPU Usage",
            value=f"{cpu_percent:.1f} %",
        )

    with col3:
        st.metric(
            label="RAM Tiêu Thụ",
            value=f"{ram_mb:.1f} MB",
        )


def render_match_result_card(
    is_match: bool,
    matched_id: Optional[str],
    matched_name: Optional[str],
    similarity_score: float,
    threshold: float,
    extra_details: Optional[Dict[str, Any]] = None,
):
    """Hiển thị kết quả điểm danh dạng banner nổi bật không chứa icon."""
    if is_match and matched_id:
        st.success(
            f"### CHẤM CÔNG THÀNH CÔNG!\n"
            f"**Mã NV:** `{matched_id}` | **Họ và Tên:** **{matched_name or 'N/A'}**\n\n"
            f"**Cosine Similarity:** `{similarity_score:.4f}` $\\ge$ Ngưỡng `{threshold:.2f}`"
        )
    else:
        st.error(
            f"### KHÔNG NHẬN DIỆN ĐƯỢC NHÂN VIÊN\n"
            f"**Trạng thái:** Từ chối điểm danh (Người lạ / Chưa đăng ký mẫu).\n\n"
            f"**Điểm tương đồng cao nhất:** `{similarity_score:.4f}` < Ngưỡng `{threshold:.2f}`"
        )

    # Thanh trực quan hóa điểm số so với ngưỡng
    st.write(f"**Mức độ tin cậy (Confidence Score):** `{similarity_score:.4f}`")
    score_clamped = max(0.0, min(1.0, float(similarity_score)))
    st.progress(score_clamped)


def render_enrolled_gallery(store):
    """Hiển thị danh sách nhân viên đã đăng ký trong session_state."""
    enrolled_data = store.get_all_enrolled_meta()
    if not enrolled_data:
        st.info("Chưa có nhân viên nào được đăng ký trong phiên làm việc này.")
        return

    st.write(f"**Tổng số nhân viên đã đăng ký:** `{len(enrolled_data)}` người")
    rows = []
    for pid, meta in enrolled_data.items():
        rows.append({
            "Mã Nhân Viên": pid,
            "Họ và Tên": meta.get("name", "N/A"),
            "Phòng Ban": meta.get("department", "Chưa gán"),
            "Số Vector Mẫu": meta.get("samples_count", 1),
            "Thời Gian Đăng Ký": meta.get("enrolled_at", "N/A"),
        })
    st.dataframe(rows, use_container_width=True)
