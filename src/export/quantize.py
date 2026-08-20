"""
Module Edge Quantization:
1. Thực hiện lượng tử hóa INT8 (Post-Training Static Quantization PTQ & Dynamic Quantization)
2. Cung cấp CalibrationDataReader cho dữ liệu ảnh khuôn mặt thật
3. Runner ONNXEmbedderRunner hỗ trợ suy luận cả FP32 và INT8 với chuẩn hóa L2
4. Đo lường so sánh trước và sau khi Quantize:
   - Kích thước file model (MB) & Tỷ lệ nén (%)
   - Độ trễ Inference (ms) & FPS (với warm-up)
   - Độ lệch embedding (Cosine Drift: Cosine Similarity giữa FP32 và INT8 embedding)
"""

import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from src.matching.matcher import compute_cosine_similarity


class FaceCalibrationDataReader(CalibrationDataReader):
    """
    Bộ nạp dữ liệu hiệu chuẩn (Calibration Data Reader) cho Post-Training Static Quantization (Static PTQ).
    Đọc ảnh khuôn mặt thật từ thư mục dữ liệu, tiền xử lý và căn chỉnh theo đúng định dạng đầu vào của từng mô hình.
    """

    def __init__(
        self,
        image_paths: List[str],
        model_type: str = "arcface",
        input_name: Optional[str] = None,
        input_shape: Optional[List[Any]] = None,
        max_samples: int = 40,
    ):
        """
        Args:
            image_paths: Danh sách đường dẫn file ảnh khuôn mặt mẫu
            model_type: 'arcface' hoặc 'facenet512'
            input_name: Tên tensor đầu vào của model ONNX
            input_shape: Shape của input tensor (tự động nhận diện NCHW hoặc NHWC)
            max_samples: Số lượng mẫu hiệu chuẩn tối đa
        """
        self.model_type = model_type.lower()
        self.input_name = input_name or ("input.1" if "arc" in self.model_type else "input_1")
        self.input_shape = input_shape
        self.data: List[np.ndarray] = []

        valid_paths = [p for p in image_paths if os.path.exists(p)][:max_samples]

        is_nchw = False
        target_h, target_w = 112, 112
        if "facenet" in self.model_type:
            target_h, target_w = 160, 160

        if self.input_shape and len(self.input_shape) == 4:
            if self.input_shape[1] == 3:
                is_nchw = True
                if isinstance(self.input_shape[2], int) and isinstance(self.input_shape[3], int):
                    target_h, target_w = self.input_shape[2], self.input_shape[3]
            elif self.input_shape[-1] == 3:
                is_nchw = False
                if isinstance(self.input_shape[1], int) and isinstance(self.input_shape[2], int):
                    target_h, target_w = self.input_shape[1], self.input_shape[2]

        for p in valid_paths:
            try:
                with open(p, "rb") as f:
                    buf = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                img = None

            if img is None:
                continue

            img_resized = cv2.resize(img, (target_w, target_h))
            if is_nchw:
                blob = cv2.dnn.blobFromImage(
                    img_resized,
                    scalefactor=1.0 / 128.0,
                    size=(target_w, target_h),
                    mean=(127.5, 127.5, 127.5),
                    swapRB=True,
                )
                self.data.append(blob.astype(np.float32))
            else:
                rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
                norm = (rgb - 127.5) / 128.0
                tensor = np.expand_dims(norm, axis=0)
                self.data.append(tensor.astype(np.float32))

        self.enum_data = iter(self.data)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        sample = next(self.enum_data, None)
        if sample is not None:
            return {self.input_name: sample}
        return None

    def rewind(self) -> None:
        self.enum_data = iter(self.data)


def quantize_onnx_model_static(
    input_onnx_path: str,
    output_quant_path: str,
    calibration_image_paths: List[str],
    model_type: str = "arcface",
    per_channel: bool = False,
    quant_format: QuantFormat = QuantFormat.QDQ,
    weight_type: QuantType = QuantType.QInt8,
    activation_type: QuantType = QuantType.QInt8,
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
) -> str:
    """
    Lượng tử hóa tĩnh (Post-Training Static Quantization - Static PTQ) mô hình ONNX sang INT8.

    Args:
        input_onnx_path: Đường dẫn file ONNX FP32 gốc
        output_quant_path: Đường dẫn lưu file ONNX INT8 sau khi quantize
        calibration_image_paths: Danh sách ảnh khuôn mặt làm dữ liệu hiệu chuẩn
        model_type: 'arcface' hoặc 'facenet512'
        per_channel: Lượng tử hóa per-channel (mặc định False để đảm bảo tương thích rộng)
        quant_format: Định dạng QDQ hoặc QOperator
        weight_type: Kiểu dữ liệu trọng số (mặc định QInt8)
        activation_type: Kiểu dữ liệu activation (mặc định QInt8)
        calibrate_method: Thuật toán xác định dải động (MinMax / Histogram / Entropy)

    Returns:
        str: Đường dẫn file model INT8 đã hoàn thành
    """
    if not os.path.exists(input_onnx_path):
        raise FileNotFoundError(f"Không tìm thấy model ONNX tại: {input_onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_quant_path)), exist_ok=True)

    # Đọc tên input tensor từ model và nạp model vào bộ nhớ để tránh lỗi path Unicode trên Windows
    raw_model = onnx.load(input_onnx_path)
    try:
        model_proto = onnx.shape_inference.infer_shapes(raw_model)
    except Exception:
        model_proto = raw_model

    sess_probe = ort.InferenceSession(input_onnx_path, providers=["CPUExecutionProvider"])
    input_meta = sess_probe.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    del sess_probe

    calib_reader = FaceCalibrationDataReader(
        image_paths=calibration_image_paths,
        model_type=model_type,
        input_name=input_name,
        input_shape=input_shape,
    )

    quantize_static(
        model_input=model_proto,
        model_output=output_quant_path,
        calibration_data_reader=calib_reader,
        quant_format=quant_format,
        per_channel=per_channel,
        activation_type=activation_type,
        weight_type=weight_type,
        calibrate_method=calibrate_method,
    )

    return output_quant_path


def quantize_onnx_model_dynamic(
    input_onnx_path: str,
    output_quant_path: str,
    per_channel: bool = True,
    weight_type: QuantType = QuantType.QInt8,
) -> str:
    """
    Lượng tử hóa động (Post-Training Dynamic Quantization) mô hình ONNX sang INT8 (dự phòng).
    """
    if not os.path.exists(input_onnx_path):
        raise FileNotFoundError(f"Không tìm thấy model ONNX tại: {input_onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_quant_path)), exist_ok=True)

    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=output_quant_path,
        per_channel=per_channel,
        weight_type=weight_type,
    )
    return output_quant_path


class ONNXEmbedderRunner:
    """
    Runner thực thi trích xuất vector đặc trưng trên ONNX Runtime (hỗ trợ cả FP32 và INT8).
    Tự động nhận diện cấu hình đầu vào (ArcFace NCHW 112x112 hoặc FaceNet512 NHWC 160x160).
    """

    def __init__(self, model_path: str, model_type: Optional[str] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
        self.model_path = model_path

        if model_type is not None:
            self.model_type = model_type.lower()
        else:
            fname = os.path.basename(model_path).lower()
            self.model_type = "facenet512" if "facenet" in fname else "arcface"

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý và định dạng tensor phù hợp với từng kiến trúc mạng.
        """
        if "facenet" in self.model_type:
            # FaceNet512: RGB, 160x160, NHWC, normalized: (x - 127.5) / 128.0
            resized = cv2.resize(face_crop, (160, 160))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            norm = (rgb - 127.5) / 128.0
            return np.expand_dims(norm, axis=0).astype(np.float32)
        elif "sface" in self.model_type:
            # SFace: BGR, 112x112, NCHW, raw [0, 255] float32
            resized = cv2.resize(face_crop, (112, 112))
            blob = resized.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]
            return blob
        else:
            # ArcFace: 112x112
            resized = cv2.resize(face_crop, (112, 112))
            if len(self.input_shape) == 4 and self.input_shape[1] == 3:
                # NCHW
                blob = cv2.dnn.blobFromImage(
                    resized,
                    scalefactor=1.0 / 128.0,
                    size=(112, 112),
                    mean=(127.5, 127.5, 127.5),
                    swapRB=True,
                )
                return blob.astype(np.float32)
            else:
                # NHWC (DeepFace Keras ONNX)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
                norm = (rgb - 127.5) / 128.0
                return np.expand_dims(norm, axis=0).astype(np.float32)

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Trích xuất vector đặc trưng 512 chiều đã được chuẩn hóa L2 (Unit Vector).
        """
        tensor = self.preprocess(face_crop)
        raw_out = self.session.run([self.output_name], {self.input_name: tensor})[0]

        vec = raw_out.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec


def benchmark_quantization_comparison(
    fp32_model_path: str,
    int8_model_path: str,
    test_face_images: List[np.ndarray],
    model_type: str = "arcface",
    n_iterations: int = 15,
) -> Dict[str, Any]:
    """
    So sánh toàn diện giữa mô hình FP32 và INT8 trên tập ảnh khuôn mặt thật:
    1. Kích thước file (MB) & Tỷ lệ nén (%)
    2. Độ trễ thuần (ms) & FPS (sau warm-up)
    3. Cosine Drift (Độ tương đồng cosine trung bình giữa FP32 và INT8 embedding)
    """
    fp32_size_mb = os.path.getsize(fp32_model_path) / (1024 * 1024)
    int8_size_mb = os.path.getsize(int8_model_path) / (1024 * 1024)
    compression_ratio = (1.0 - int8_size_mb / fp32_size_mb) * 100.0

    runner_fp32 = ONNXEmbedderRunner(fp32_model_path, model_type=model_type)
    runner_int8 = ONNXEmbedderRunner(int8_model_path, model_type=model_type)

    if not test_face_images:
        test_face_images = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)]

    # Giai đoạn Warm-up (Khởi động JIT/Kernel cache)
    for sample in test_face_images[:3]:
        _ = runner_fp32.embed(sample)
        _ = runner_int8.embed(sample)

    # Đo độ trễ FP32
    fp32_latencies = []
    for _ in range(n_iterations):
        for img in test_face_images:
            t0 = time.perf_counter()
            _ = runner_fp32.embed(img)
            fp32_latencies.append((time.perf_counter() - t0) * 1000.0)

    # Đo độ trễ INT8 & tính Cosine Drift
    int8_latencies = []
    cosine_similarities = []
    for img in test_face_images:
        emb_fp32 = runner_fp32.embed(img)
        emb_int8 = runner_int8.embed(img)
        sim = compute_cosine_similarity(emb_fp32, emb_int8)
        cosine_similarities.append(sim)

    for _ in range(n_iterations):
        for img in test_face_images:
            t0 = time.perf_counter()
            _ = runner_int8.embed(img)
            int8_latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_fp32_ms = float(np.mean(fp32_latencies))
    avg_int8_ms = float(np.mean(int8_latencies))
    speedup = (avg_fp32_ms / avg_int8_ms) if avg_int8_ms > 0 else 1.0
    avg_cosine_drift_sim = float(np.mean(cosine_similarities))
    min_cosine_drift_sim = float(np.min(cosine_similarities))

    return {
        "model_type": model_type,
        "fp32_size_mb": round(fp32_size_mb, 2),
        "int8_size_mb": round(int8_size_mb, 2),
        "compression_ratio_percent": round(compression_ratio, 2),
        "fp32_latency_ms": round(avg_fp32_ms, 2),
        "int8_latency_ms": round(avg_int8_ms, 2),
        "fp32_fps": round(1000.0 / avg_fp32_ms, 2) if avg_fp32_ms > 0 else 0,
        "int8_fps": round(1000.0 / avg_int8_ms, 2) if avg_int8_ms > 0 else 0,
        "speedup_factor": round(speedup, 2),
        "cosine_similarity_mean": round(avg_cosine_drift_sim, 4),
        "cosine_similarity_min": round(min_cosine_drift_sim, 4),
    }
