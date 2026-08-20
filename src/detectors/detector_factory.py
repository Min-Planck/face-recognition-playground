"""
Module Detector Factory: Cung cấp interface BaseDetector và các triển khai cụ thể:
1. RetinaFace (qua InsightFace ONNX detector - chính xác cao, chuẩn quốc tế, tốc độ vượt trội)
2. MediaPipe BlazeFace (qua MediaPipe FaceDetection - siêu nhẹ, CPU-friendly cho Edge)
3. YOLOv8-Face (qua Ultralytics YOLOv8 Face Detection chuyên dụng)

Hỗ trợ factory function `get_detector(config)` khởi tạo theo pipeline.yaml.
"""

from abc import ABC, abstractmethod
import ctypes
from dataclasses import dataclass
import os
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


def _apply_mediapipe_windows_patch():
    """
    Patch xử lý đường dẫn Windows có ký tự tiếng Việt (non-ASCII) cho C++ binary graph của MediaPipe.
    """
    if sys.platform == "win32":
        try:
            from mediapipe.python import solution_base

            def to_short_path(path: Optional[str]) -> Optional[str]:
                if not path:
                    return path
                buf = ctypes.create_unicode_buffer(500)
                if ctypes.windll.kernel32.GetShortPathNameW(path, buf, 500):
                    return buf.value
                return path

            orig_init = solution_base.validated_graph_config.ValidatedGraphConfig.initialize

            def patched_init(self, binary_graph_path=None, **kwargs):
                if binary_graph_path:
                    binary_graph_path = to_short_path(binary_graph_path)
                return orig_init(self, binary_graph_path=binary_graph_path, **kwargs)

            solution_base.validated_graph_config.ValidatedGraphConfig.initialize = patched_init

            orig_set_res = solution_base.resource_util.set_resource_dir

            def patched_set_res(path):
                return orig_set_res(to_short_path(path))

            solution_base.resource_util.set_resource_dir = patched_set_res
        except Exception:
            pass


@dataclass
class FaceBox:
    """
    Cấu trúc dữ liệu chứa thông tin khuôn mặt được phát hiện.
    """
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    aligned_face: Optional[np.ndarray] = None

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Trả về (x, y, w, h)."""
        return (self.x, self.y, self.w, self.h)

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        """Trả về (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def get_crop(self, image: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """
        Cắt vùng mặt từ ảnh gốc có kèm margin tuỳ chọn.
        """
        img_h, img_w = image.shape[:2]
        mx = int(self.w * margin)
        my = int(self.h * margin)

        x1 = max(0, self.x - mx)
        y1 = max(0, self.y - my)
        x2 = min(img_w, self.x + self.w + mx)
        y2 = min(img_h, self.y + self.h + my)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return image
        return crop


class BaseDetector(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả Face Detectors.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceBox]:
        """
        Phát hiện tất cả khuôn mặt trong ảnh BGR.

        Args:
            image: Ảnh BGR (uint8, HxWxC)

        Returns:
            List[FaceBox]: Danh sách các khuôn mặt tìm thấy
        """
        pass

    def align_face(
        self,
        image: np.ndarray,
        left_eye: Tuple[int, int],
        right_eye: Tuple[int, int],
        target_size: Tuple[int, int] = (112, 112),
    ) -> np.ndarray:
        """
        Căn chỉnh khuôn mặt (Face Alignment) dựa trên tọa độ 2 mắt để 2 mắt nằm ngang.
        Tự động chuẩn hóa để left_eye luôn là mắt bên trái ảnh (x nhỏ hơn)
        và right_eye luôn là mắt bên phải ảnh (x lớn hơn).

        Args:
            image: Ảnh gốc BGR
            left_eye: Tọa độ mắt trái (x, y)
            right_eye: Tọa độ mắt phải (x, y)
            target_size: Kích thước ảnh output chuẩn hóa (mặc định 112x112)
        """
        if left_eye[0] > right_eye[0]:
            left_eye, right_eye = right_eye, left_eye

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        eye_center = (
            int((left_eye[0] + right_eye[0]) / 2),
            int((left_eye[1] + right_eye[1]) / 2),
        )

        h, w = image.shape[:2]
        rot_mat = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        rotated = cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        dist = np.sqrt(dx**2 + dy**2)
        box_size = int(max(dist * 2.8, 80))
        x1 = max(0, eye_center[0] - box_size // 2)
        y1 = max(0, eye_center[1] - int(box_size * 0.38))
        x2 = min(w, x1 + box_size)
        y2 = min(h, y1 + box_size)

        face_aligned = rotated[y1:y2, x1:x2]
        if face_aligned.size > 0:
            return cv2.resize(face_aligned, target_size)
        return cv2.resize(image, target_size)


class RetinaFaceDetector(BaseDetector):
    """
    Detector RetinaFace / SCRFD: Sử dụng InsightFace ONNX engine (`buffalo_l` pack).
    Chạy 100% trên ONNX Runtime, không xung đột Keras/TensorFlow, độ chính xác cao nhất.
    """

    def __init__(self, align: bool = True, model_pack: str = "buffalo_l", det_size: Tuple[int, int] = (640, 640)):
        self.align = align
        self.det_size = det_size
        import insightface
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align
        self._face_align = face_align

        # Khởi tạo FaceAnalysis chỉ nạp module detection để tối ưu bộ nhớ
        self.app = FaceAnalysis(name=model_pack, allowed_modules=["detection"])
        self.app.prepare(ctx_id=-1, det_size=self.det_size)  # ctx_id=-1 (CPU)

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        faces = self.app.get(image)

        face_boxes = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            bx = max(0, int(x1))
            by = max(0, int(y1))
            bw = min(w - bx, int(max(0, x2 - x1)))
            bh = min(h - by, int(max(0, y2 - y1)))

            score = float(face.det_score) if hasattr(face, "det_score") else 1.0

            landmarks = {}
            aligned_face = None
            if hasattr(face, "kps") and face.kps is not None:
                kps = face.kps.astype(int)
                if len(kps) >= 5:
                    landmarks = {
                        "right_eye": (int(kps[0][0]), int(kps[0][1])),
                        "left_eye": (int(kps[1][0]), int(kps[1][1])),
                        "nose": (int(kps[2][0]), int(kps[2][1])),
                        "mouth_right": (int(kps[3][0]), int(kps[3][1])),
                        "mouth_left": (int(kps[4][0]), int(kps[4][1])),
                    }
                    if self.align:
                        try:
                            aligned_face = self._face_align.norm_crop(image, landmark=face.kps, image_size=112)
                        except Exception:
                            aligned_face = self.align_face(image, landmarks["left_eye"], landmarks["right_eye"], (112, 112))

            if bw > 0 and bh > 0:
                face_boxes.append(FaceBox(
                    x=bx,
                    y=by,
                    w=bw,
                    h=bh,
                    confidence=score,
                    landmarks=landmarks if landmarks else None,
                    aligned_face=aligned_face,
                ))

        return face_boxes


class MediaPipeDetector(BaseDetector):
    """
    Detector MediaPipe BlazeFace: Rất nhanh, tối ưu CPU / Edge.
    Trích xuất 6 keypoints: mắt trái/phải, mũi, miệng, tai trái/phải.
    """

    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 0):
        _apply_mediapipe_windows_patch()
        import mediapipe as mp
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection,
        )

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return []

        face_boxes = []
        for det in results.detections:
            score = float(det.score[0])
            bbox = det.location_data.relative_bounding_box
            bx = max(0, int(bbox.xmin * w))
            by = max(0, int(bbox.ymin * h))
            bw = min(w - bx, int(bbox.width * w))
            bh = min(h - by, int(bbox.height * h))

            landmarks = {}
            kps = det.location_data.relative_keypoints
            right_eye = None
            left_eye = None
            if len(kps) >= 4:
                right_eye = (int(kps[0].x * w), int(kps[0].y * h))
                left_eye = (int(kps[1].x * w), int(kps[1].y * h))
                nose = (int(kps[2].x * w), int(kps[2].y * h))
                mouth = (int(kps[3].x * w), int(kps[3].y * h))
                landmarks = {
                    "right_eye": right_eye,
                    "left_eye": left_eye,
                    "nose": nose,
                    "mouth": mouth,
                }

            aligned_face = None
            if left_eye and right_eye:
                aligned_face = self.align_face(image, left_eye, right_eye, (112, 112))

            if bw > 0 and bh > 0:
                face_boxes.append(FaceBox(
                    x=bx,
                    y=by,
                    w=bw,
                    h=bh,
                    confidence=score,
                    landmarks=landmarks if landmarks else None,
                    aligned_face=aligned_face,
                ))

        return face_boxes


class YOLOv8Detector(BaseDetector):
    """
    Detector YOLOv8-Face: Sử dụng mô hình ONNX Runtime (`models/yolov8n-face.onnx`).
    Hoàn toàn không cần PyTorch khi suy luận, giúp tiết kiệm bộ nhớ RAM và tăng tốc tối đa trên CPU.
    """

    MODEL_URL = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"

    def __init__(
        self,
        conf_threshold: float = 0.4,
        model_path: str = "models/yolov8n-face.onnx",
        nms_threshold: float = 0.45,
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.model_path = model_path
        self._ensure_model_exists()

        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _ensure_model_exists(self):
        """Tự động chuẩn bị file mô hình ONNX nếu chưa có sẵn."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else ".", exist_ok=True)
            # Kiểm tra xem có file script export không
            try:
                from scripts.export_onnx_models import export_yolov8_onnx
                export_yolov8_onnx()
            except Exception:
                pt_path = os.path.join(os.path.dirname(self.model_path), "yolov8n-face.pt")
                if not os.path.exists(pt_path):
                    print(f"Đang tải trọng số YOLOv8-Face từ {self.MODEL_URL}...")
                    urllib.request.urlretrieve(self.MODEL_URL, pt_path)
                from ultralytics import YOLO
                yolo = YOLO(pt_path)
                yolo.export(format="onnx", opset=17)

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        # 1. Tiền xử lý Letterbox về 640x640 chuẩn YOLOv8
        scale = min(640.0 / h, 640.0 / w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (nw, nh))
        pad_x = (640 - nw) // 2
        pad_y = (640 - nh) // 2

        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
        # Chuẩn hóa BGR -> RGB và [0, 1] NCHW
        rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = (rgb_canvas.transpose(2, 0, 1) / 255.0).astype(np.float32)[np.newaxis, ...]

        # 2. Suy luận qua ONNX Runtime
        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]
        preds = outputs[0].T  # Shape: (8400, 5) -> [cx, cy, bw, bh, score]

        # 3. Lọc ngưỡng và chuyển đổi tọa độ
        boxes_xywh = []
        confidences = []
        for row in preds:
            score = float(row[4])
            if score >= self.conf_threshold:
                cx, cy, bw, bh = row[:4]
                x1 = max(0, int((cx - bw / 2.0 - pad_x) / scale))
                y1 = max(0, int((cy - bh / 2.0 - pad_y) / scale))
                x2 = min(w, int((cx + bw / 2.0 - pad_x) / scale))
                y2 = min(h, int((cy + bh / 2.0 - pad_y) / scale))
                box_w = max(0, x2 - x1)
                box_h = max(0, y2 - y1)
                if box_w > 0 and box_h > 0:
                    boxes_xywh.append([x1, y1, box_w, box_h])
                    confidences.append(score)

        if not boxes_xywh:
            return []

        # 4. Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            confidences,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )
        indices = indices.flatten() if hasattr(indices, "flatten") else indices

        face_boxes = []
        for idx in indices:
            i = int(idx)
            bx, by, bw, bh = boxes_xywh[i]
            conf = confidences[i]

            # Crop khuôn mặt với 8% margin và chuẩn hóa về 112x112
            crop_mx = int(bw * 0.08)
            crop_my = int(bh * 0.08)
            cx1 = max(0, bx - crop_mx)
            cy1 = max(0, by - crop_my)
            cx2 = min(w, bx + bw + crop_mx)
            cy2 = min(h, by + bh + crop_my)

            face_crop = image[cy1:cy2, cx1:cx2]
            aligned_face = cv2.resize(face_crop, (112, 112)) if face_crop.size > 0 else None

            face_boxes.append(FaceBox(
                x=bx,
                y=by,
                w=bw,
                h=bh,
                confidence=conf,
                landmarks=None,
                aligned_face=aligned_face,
            ))

        return face_boxes


def get_detector(config: Union[str, Dict[str, Any]] = "retinaface") -> BaseDetector:
    """
    Factory function khởi tạo Face Detector dựa trên cấu hình YAML.

    Args:
        config: Tên detector (str) hoặc Dict config chứa key 'detector'

    Returns:
        BaseDetector: Instance của detector tương ứng
    """
    detector_name = "retinaface"
    align = True

    if isinstance(config, str):
        detector_name = config.lower().strip()
    elif isinstance(config, dict):
        pipeline_cfg = config.get("pipeline", config)
        detector_name = pipeline_cfg.get("detector", "retinaface").lower().strip()
        align = pipeline_cfg.get("align", True)

    if detector_name in ("retinaface", "insightface", "scrfd"):
        return RetinaFaceDetector(align=align)
    elif detector_name in ("mediapipe", "blazeface"):
        return MediaPipeDetector()
    elif detector_name in ("yolov8", "yolo", "yolov8-face", "yolo-face"):
        return YOLOv8Detector()
    else:
        raise ValueError(
            f"Không hỗ trợ detector: '{detector_name}'. "
            f"Các lựa chọn hợp lệ: 'retinaface', 'mediapipe', 'yolov8'."
        )


def extract_aligned_face(
    detector: BaseDetector,
    image: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
    apply_clahe: bool = True,
) -> np.ndarray:
    """
    Tiền xử lý CLAHE, phát hiện và trích xuất khuôn mặt lớn nhất đã chuẩn hóa.
    Đảm bảo mảng bộ nhớ liền mạch (contiguous uint8 buffer) chống crash C++.

    Args:
        detector: Đối tượng Detector (MediaPipe, RetinaFace, YOLOv8)
        image: Ảnh đầu vào BGR
        target_size: Kích thước crop (mặc định 112x112)
        apply_clahe: Có áp dụng tiền xử lý CLAHE hay không

    Returns:
        np.ndarray: Ảnh mặt crop chuẩn hóa (112, 112, 3) uint8 contiguous
    """
    if image is None or image.size == 0:
        return np.zeros((*target_size, 3), dtype=np.uint8)

    from src.preprocessing.clahe import preprocess_image
    prep = preprocess_image(image) if apply_clahe else image.copy()

    boxes = detector.detect(prep)
    if not boxes:
        h, w = prep.shape[:2]
        crop = prep[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
        return np.ascontiguousarray(cv2.resize(crop, target_size), dtype=np.uint8)

    # Chọn khuôn mặt có diện tích lớn nhất (chính diện)
    box = max(boxes, key=lambda b: b.w * b.h)
    if box.aligned_face is not None:
        return np.ascontiguousarray(box.aligned_face.copy(), dtype=np.uint8)
    crop = box.get_crop(prep, margin=0.1)
    return np.ascontiguousarray(cv2.resize(crop, target_size), dtype=np.uint8)
