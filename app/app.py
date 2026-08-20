"""
Ứng Dụng Streamlit Chấm Công Bằng Nhận Diện Khuôn Mặt
Hệ thống Edge Face Attendance Recognition

Kiến trúc:
Ảnh đầu vào -> CLAHE/Denoise -> Detection -> Alignment -> Embedding -> 1:K Matching

Tính năng:
1. Pha Enrollment: Đăng ký nhân viên (camera/upload, lưu N samples vào st.session_state).
2. Pha Inference: Điểm danh thời gian thực, đo lường Latency thuần, FPS, CPU%, RAM, Cosine Similarity.
3. Hoán đổi linh hoạt Detector và Embedder từ sidebar.
4. Tự động đồng bộ ngưỡng tối ưu T* từ config/pipeline.yaml.
"""

import os
import sys
import time
import yaml
import warnings
from datetime import datetime

# Tắt log nhiễu từ TensorFlow và oneDNN
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import psutil
import streamlit as st

# Cấu hình đường dẫn import cho Streamlit
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.detectors.detector_factory import get_detector
from src.embedders.embedder_factory import get_embedder
from src.matching.matcher import SessionFaceStore, compute_cosine_similarity
from src.preprocessing.clahe import preprocess_image
from src.evaluation.resource_monitor import ResourceMonitor

try:
    from components.metrics_display import (
        render_metrics_header,
        render_match_result_card,
        render_enrolled_gallery,
    )
except ImportError:
    from app.components.metrics_display import (
        render_metrics_header,
        render_match_result_card,
        render_enrolled_gallery,
    )

# Cấu hình trang Streamlit (Không icon)
st.set_page_config(
    page_title="Edge Face Attendance System",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_pipeline_config() -> dict:
    """Nạp cấu hình và ngưỡng hiệu chuẩn mới nhất từ config/pipeline.yaml."""
    cfg_path = os.path.join(PROJECT_ROOT, "config", "pipeline.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "thresholds": {
            "sface": 0.32,
            "arcface": 0.24,
            "facenet512": 0.53,
        }
    }


# ==============================================================================
# QUẢN LÝ SESSION STATE & CACHE MODEL (KÈM WARM-UP)
# ==============================================================================
if "store" not in st.session_state:
    st.session_state.store = SessionFaceStore(samples_per_person=3)

if "attendance_logs" not in st.session_state:
    st.session_state.attendance_logs = []


@st.cache_resource(show_spinner="Đang khởi tạo & Warm-up Face Detector...")
def load_cached_detector(name: str):
    """Cache detector object và chạy warm-up để loại bỏ độ trễ khởi động lạnh."""
    det = get_detector(name)
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    _ = det.detect(dummy_img)
    return det


@st.cache_resource(show_spinner="Đang khởi tạo & Warm-up Face Embedder...")
def load_cached_embedder(name: str):
    """Cache embedder object và chạy warm-up 112x112 vào RAM."""
    emb = get_embedder(name)
    dummy_crop = np.zeros((112, 112, 3), dtype=np.uint8)
    _ = emb.embed(dummy_crop)
    return emb


def get_current_system_usage():
    """Lấy thông số CPU% và RAM (MB) của tiến trình hiện tại từ ResourceMonitor."""
    return ResourceMonitor.get_current_usage()


def draw_styled_detection(image: np.ndarray, bbox: tuple, label: str, is_match: bool = True) -> np.ndarray:
    """Vẽ bounding box và nhãn theo phong cách viền đen chữ trắng rõ ràng."""
    vis = image.copy()
    x, y, w, h = bbox

    # Màu khung: Xanh lá nếu hợp lệ, Đỏ nếu từ chối
    box_color = (0, 200, 0) if is_match else (0, 0, 220)
    cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, 2)

    # Banner thông tin đen chữ trắng
    banner_h = 28
    y_banner = max(0, y - banner_h)
    cv2.rectangle(vis, (x, y_banner), (x + w, y_banner + banner_h), (0, 0, 0), -1)
    cv2.putText(
        vis,
        label,
        (x + 6, y_banner + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


# ==============================================================================
# SIDEBAR: CẤU HÌNH HỆ THỐNG & QUẢN LÝ MÔ HÌNH
# ==============================================================================
pipeline_cfg = load_pipeline_config()
calibrated_thresholds = pipeline_cfg.get("thresholds", {
    "sface": 0.32,
    "arcface": 0.24,
    "facenet512": 0.53,
})

st.sidebar.title("Cấu Hình Pipeline")

st.sidebar.subheader("1. Lựa Chọn Mô Hình AI")
detector_choice = st.sidebar.selectbox(
    "Face Detector (Bộ Phát Hiện):",
    options=["mediapipe", "retinaface", "yolov8"],
    index=0,
    help="MediaPipe: Siêu nhẹ cho Edge CPU (~36ms) | RetinaFace: Chuẩn xác cao | YOLOv8: Cân bằng",
)

embedder_choice = st.sidebar.selectbox(
    "Face Embedder (Bộ Trích Xuất):",
    options=["sface", "arcface", "facenet512"],
    index=0,
    help="SFace: 128-D siêu nhanh (~41ms) | ArcFace: 512-D bảo mật cao | FaceNet512: 512-D",
)

st.sidebar.subheader("2. Tham Số Nhận Diện")
default_th_for_choice = float(calibrated_thresholds.get(embedder_choice, 0.32))

threshold_val = st.sidebar.slider(
    "Ngưỡng Chấp Nhận (Cosine Threshold T):",
    min_value=0.05,
    max_value=0.85,
    value=default_th_for_choice,
    step=0.01,
    help=f"Ngưỡng tối ưu hiệu chuẩn thực nghiệm cho {embedder_choice.upper()} là {default_th_for_choice:.2f}",
)

enable_clahe = st.sidebar.checkbox(
    "Bật tiền xử lý ảnh (CLAHE + Denoise + Sharpen)",
    value=True,
    help="Cân bằng độ sáng cục bộ và tăng độ sắc nét vùng mặt.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Quản Lý Dữ Liệu Session")

# Giữ duy nhất icon cho nút xóa tất cả theo yêu cầu
if st.sidebar.button("🗑️ Xóa Tất Cả Nhân Viên", use_container_width=True, help="Xóa toàn bộ nhân viên đã lưu trong bộ nhớ session."):
    st.session_state.store.clear()
    st.session_state.attendance_logs.clear()
    st.sidebar.warning("Đã làm trống bộ nhớ Session!")
    st.rerun()

st.sidebar.caption(f"Đã đăng ký: **{st.session_state.store.get_enrolled_count()}** người")


# ==============================================================================
# GIAO DIỆN CHÍNH: 3 TABS (ĐIỂM DANH, ĐĂNG KÝ, NHẬT KÝ)
# ==============================================================================
st.title("Hệ Thống Chấm Công Nhận Diện Khuôn Mặt (Edge AI)")
st.caption("Pipeline: Tiền xử lý CLAHE → Detection → Alignment (112×112) → Feature Embedding → 1:K Matching")

# Kiểm tra tương thích số chiều vector giữa Gallery và Embedder đang chọn
active_embedder = load_cached_embedder(embedder_choice)
test_sample_vec = active_embedder.embed(np.zeros((112, 112, 3), dtype=np.uint8))
current_active_dim = test_sample_vec.shape[0]

dim_mismatch = False
for pid, v_list in st.session_state.store._store.items():
    for v in v_list:
        if v.shape[0] != current_active_dim:
            dim_mismatch = True
            break
    if dim_mismatch:
        break

if dim_mismatch:
    st.error(
        f"CẢNH BÁO LỆCH SỐ CHIỀU VECTOR: Bạn vừa đổi sang model Embedder {embedder_choice.upper()} ({current_active_dim}-D), "
        f"nhưng bộ nhớ đang lưu vector của model cũ. Vui lòng vào tab 'Đăng Ký Nhân Viên' để đăng ký lại mẫu mới phù hợp!"
    )

tab_attendance, tab_enroll, tab_logs = st.tabs([
    "Điểm Danh Thời Gian Thực (Inference)",
    "Đăng Ký Nhân Viên Mới (Enrollment)",
    "Danh Sách Nhân Viên & Lịch Sử",
])


# ==============================================================================
# TAB 1: ĐIỂM DANH THỜI GIAN THỰC (INFERENCE)
# ==============================================================================
with tab_attendance:
    st.subheader("Quét Khuôn Mặt Để Điểm Danh")

    if st.session_state.store.get_enrolled_count() == 0:
        st.warning("Chưa có nhân viên nào trong danh sách! Hãy qua tab 'Đăng Ký Nhân Viên Mới' để chụp mẫu đăng ký trước.")

    col_input, col_result = st.columns([1.1, 1.1])

    with col_input:
        input_mode = st.radio(
            "Chọn Nguồn Ảnh:",
            options=["Chụp từ Camera", "Tải ảnh lên từ máy tính"],
            horizontal=True,
            key="infer_input_mode",
        )

        infer_mat = None
        if input_mode == "Chụp từ Camera":
            cam_file = st.camera_input("Hướng mặt vào camera và bấm chụp:", key="cam_infer")
            if cam_file is not None:
                bytes_data = cam_file.getvalue()
                infer_mat = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            up_file = st.file_uploader("Chọn ảnh quét mặt (JPG/PNG):", type=["jpg", "jpeg", "png"], key="upload_infer")
            if up_file is not None:
                bytes_data = up_file.getvalue()
                infer_mat = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with col_result:
        if infer_mat is not None:
            detector = load_cached_detector(detector_choice)
            embedder = load_cached_embedder(embedder_choice)

            # Mồi baseline CPU và bấm giờ inference thuần
            ResourceMonitor.prime_cpu()
            t_start = time.perf_counter()

            # 1. Tiền xử lý CLAHE
            if enable_clahe:
                processed_mat = preprocess_image(infer_mat)
            else:
                processed_mat = infer_mat.copy()

            # 2. Phát hiện khuôn mặt
            boxes = detector.detect(processed_mat)

            if not boxes:
                st.error("Không tìm thấy khuôn mặt trong ảnh! Vui lòng nhìn thẳng vào camera và thử lại.")
            else:
                face_box = max(boxes, key=lambda b: b.w * b.h)

                # 3. Alignment & Crop 112x112 (Lấy trực tiếp từ kết quả detect, không gọi lại CLAHE/Detector)
                if face_box.aligned_face is not None:
                    aligned_crop = np.ascontiguousarray(face_box.aligned_face.copy(), dtype=np.uint8)
                else:
                    raw_crop = face_box.get_crop(processed_mat, margin=0.1)
                    aligned_crop = np.ascontiguousarray(cv2.resize(raw_crop, (112, 112)), dtype=np.uint8)

                # 4. Feature Embedding
                query_vec = embedder.embed(aligned_crop)

                # 5. 1:K Matching
                match_res = st.session_state.store.find_best_match(query_vec, threshold=threshold_val)

                # Kết thúc đo đạc
                latency_ms = (time.perf_counter() - t_start) * 1000.0
                fps = 1000.0 / latency_ms if latency_ms > 0 else 0
                cpu_pct, ram_mb = get_current_system_usage()

                # Hiển thị Telemetry Metrics Header (3 cột: Latency, CPU, RAM)
                render_metrics_header(
                    latency_ms=latency_ms,
                    fps=fps,
                    cpu_percent=cpu_pct,
                    ram_mb=ram_mb,
                )

                st.markdown("---")

                # Hiển thị Thẻ Kết Quả Điểm Danh
                matched_name = match_res.metadata.get("name") if match_res.is_match else None
                render_match_result_card(
                    is_match=match_res.is_match,
                    matched_id=match_res.matched_id,
                    matched_name=matched_name,
                    similarity_score=match_res.similarity_score,
                    threshold=threshold_val,
                )

                # Lưu vào log điểm danh
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.attendance_logs.insert(0, {
                    "Thời Gian": now_str,
                    "Mã NV": match_res.matched_id if match_res.is_match else "Unknown",
                    "Họ Tên": matched_name or "Người Lạ",
                    "Similarity": f"{match_res.similarity_score:.4f}",
                    "Ngưỡng T": f"{threshold_val:.2f}",
                    "Kết Quả": "Thành Công" if match_res.is_match else "Từ Chối",
                    "Latency (ms)": f"{latency_ms:.1f}",
                })

                # Hiển thị ảnh phát hiện & ảnh crop 112x112
                label_txt = f"{matched_name or 'Unknown'} ({match_res.similarity_score:.2f})"
                vis_img = draw_styled_detection(infer_mat, face_box.bbox, label_txt, is_match=match_res.is_match)

                col_v1, col_v2 = st.columns([2, 1])
                with col_v1:
                    st.image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), caption="Khuôn Mặt Phát Hiện", use_container_width=True)
                with col_v2:
                    st.image(cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2RGB), caption="Face Aligned (112×112)", use_container_width=True)


# ==============================================================================
# TAB 2: ĐĂNG KÝ NHÂN VIÊN MỚI (ENROLLMENT)
# ==============================================================================
with tab_enroll:
    st.subheader("Đăng Ký Khuôn Mặt Nhân Viên Mới")
    st.caption("Mỗi nhân viên có thể chụp từ 1 đến 3 ảnh mẫu ở các góc sáng/nghiêng nhẹ khác nhau để tăng độ tin cậy.")

    col_enr_form, col_enr_preview = st.columns([1, 1])

    with col_enr_form:
        person_id = st.text_input("Mã Nhân Viên (*):", placeholder="Ví dụ: NV001, EMP105...", key="enr_id").strip()
        person_name = st.text_input("Họ và Tên (*):", placeholder="Ví dụ: Nguyễn Văn A...", key="enr_name").strip()
        department = st.text_input("Phòng Ban / Bộ Phận:", placeholder="Ví dụ: Kỹ Thuật, Nhân Sự...", key="enr_dept").strip()

        enr_input_mode = st.radio(
            "Nguồn Ảnh Đăng Ký:",
            options=["Chụp từ Camera", "Tải ảnh lên từ máy tính"],
            horizontal=True,
            key="enr_input_mode",
        )

        enr_mat = None
        if enr_input_mode == "Chụp từ Camera":
            cam_enr_file = st.camera_input("Chụp ảnh chân dung nhân viên:", key="cam_enr")
            if cam_enr_file is not None:
                bytes_data = cam_enr_file.getvalue()
                enr_mat = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            up_enr_file = st.file_uploader("Chọn file ảnh đăng ký:", type=["jpg", "jpeg", "png"], key="upload_enr")
            if up_enr_file is not None:
                bytes_data = up_enr_file.getvalue()
                enr_mat = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        submit_btn = st.button("Xác Nhận Đăng Ký", type="primary", use_container_width=True)

    with col_enr_preview:
        if enr_mat is not None:
            detector = load_cached_detector(detector_choice)
            embedder = load_cached_embedder(embedder_choice)

            prep_enr = preprocess_image(enr_mat) if enable_clahe else enr_mat.copy()
            boxes = detector.detect(prep_enr)

            if not boxes:
                st.warning("Không phát hiện khuôn mặt! Vui lòng chụp rõ mặt hơn.")
            else:
                face_box = max(boxes, key=lambda b: b.w * b.h)

                # Lấy trực tiếp ảnh 112x112 từ kết quả detect
                if face_box.aligned_face is not None:
                    aligned_crop = np.ascontiguousarray(face_box.aligned_face.copy(), dtype=np.uint8)
                else:
                    raw_crop = face_box.get_crop(prep_enr, margin=0.1)
                    aligned_crop = np.ascontiguousarray(cv2.resize(raw_crop, (112, 112)), dtype=np.uint8)

                vis_enr = draw_styled_detection(enr_mat, face_box.bbox, f"Enroll: {person_name or 'New'}", is_match=True)

                st.image(cv2.cvtColor(vis_enr, cv2.COLOR_BGR2RGB), caption="Ảnh Chụp Đăng Ký", use_container_width=True)
                st.image(cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2RGB), caption="Khuôn Mặt Chuẩn Hóa (112×112)", width=120)

                if submit_btn:
                    if not person_id or not person_name:
                        st.error("Vui lòng điền đầy đủ Mã Nhân Viên và Họ Tên!")
                    else:
                        vec = embedder.embed(aligned_crop)
                        total_samples = st.session_state.store.enroll(
                            person_id=person_id,
                            embedding=vec,
                            meta={
                                "name": person_name,
                                "department": department or "Chưa gán",
                                "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            },
                        )
                        st.success(f"Đã đăng ký thành công cho nhân viên {person_name} (Mã: {person_id})! Hiện có {total_samples} vector mẫu.")
                        time.sleep(1.0)
                        st.rerun()


# ==============================================================================
# TAB 3: DANH SÁCH NHÂN VIÊN & LỊCH SỬ ĐIỂM DANH
# ==============================================================================
with tab_logs:
    col_t1, col_t2 = st.columns([1, 1.2])

    with col_t1:
        st.subheader("Danh Sách Nhân Viên Đã Đăng Ký")
        render_enrolled_gallery(st.session_state.store)

    with col_t2:
        st.subheader("Lịch Sử Chấm Công Gần Nhất")
        if not st.session_state.attendance_logs:
            st.info("Chưa có lượt chấm công nào trong phiên làm việc này.")
        else:
            st.dataframe(st.session_state.attendance_logs, use_container_width=True)
            if st.button("Làm Sạch Lịch Sử", key="clear_logs"):
                st.session_state.attendance_logs.clear()
                st.rerun()
