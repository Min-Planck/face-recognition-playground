"""
Module Edge Quantization:
1. Export mô hình sang định dạng ONNX (Float32)
2. Thực hiện lượng tử hóa INT8 (Post-Training Dynamic Quantization / Static Quantization)
3. Đo lường so sánh trước và sau khi Quantize:
   - Kích thước file model (MB)
   - Độ trễ Inference (ms) & FPS
   - Độ lệch embedding (Cosine Drift: Cosine Similarity giữa FP32 và INT8 embedding)
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.matching.matcher import compute_cosine_similarity


def quantize_onnx_model_dynamic(
    input_onnx_path: str,
    output_quant_path: str,
    per_channel: bool = True,
    weight_type: QuantType = QuantType.QInt8,
) -> str:
    """
    Lượng tử hóa động (Post-Training Dynamic Quantization) mô hình ONNX sang INT8.

    Args:
        input_onnx_path: Đường dẫn file ONNX gốc (FP32)
        output_quant_path: Đường dẫn lưu file ONNX INT8 sau khi quantize
        per_channel: Lượng tử hóa per-channel (tối ưu hơn cho mạng CNN)
        weight_type: Kiểu dữ liệu weight lượng tử (mặc định QInt8)

    Returns:
        str: Đường dẫn file model đã quantize
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
    Runner thực thi suy luận trên ONNX Runtime (hỗ trợ cả FP32 và INT8).
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
        self.model_path = model_path

        # Cấu hình ONNX Runtime session
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

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Chạy inference trích xuất vector đặc trưng.
        """
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, axis=0)

        input_tensor = input_tensor.astype(np.float32)
        raw_out = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        # Chuẩn hóa L2
        vec = raw_out.flatten()
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            return vec / norm
        return vec


def benchmark_quantization_comparison(
    fp32_model_path: str,
    int8_model_path: str,
    dummy_input_shape: Tuple[int, int, int, int] = (1, 3, 112, 112),
    n_iterations: int = 20,
) -> Dict[str, Any]:
    """
    So sánh toàn diện giữa mô hình FP32 và INT8:
    1. Kích thước file (MB) & Tỷ lệ nén (%)
    2. Độ trễ (ms) & FPS
    3. Cosine Drift (Độ tương đồng cosine trung bình giữa FP32 và INT8)
    """
    fp32_size_mb = os.path.getsize(fp32_model_path) / (1024 * 1024)
    int8_size_mb = os.path.getsize(int8_model_path) / (1024 * 1024)
    compression_ratio = (1.0 - int8_size_mb / fp32_size_mb) * 100.0

    runner_fp32 = ONNXEmbedderRunner(fp32_model_path)
    runner_int8 = ONNXEmbedderRunner(int8_model_path)

    # Khởi động (warmup)
    sample_input = np.random.randn(*dummy_input_shape).astype(np.float32)
    for _ in range(3):
        runner_fp32.run(sample_input)
        runner_int8.run(sample_input)

    # Đo độ trễ FP32
    fp32_latencies = []
    for _ in range(n_iterations):
        inp = np.random.randn(*dummy_input_shape).astype(np.float32)
        t0 = time.perf_counter()
        runner_fp32.run(inp)
        fp32_latencies.append((time.perf_counter() - t0) * 1000.0)

    # Đo độ trễ INT8 & tính Cosine Drift
    int8_latencies = []
    cosine_similarities = []
    for _ in range(n_iterations):
        inp = np.random.randn(*dummy_input_shape).astype(np.float32)
        
        emb_fp32 = runner_fp32.run(inp)

        t0 = time.perf_counter()
        emb_int8 = runner_int8.run(inp)
        int8_latencies.append((time.perf_counter() - t0) * 1000.0)

        sim = compute_cosine_similarity(emb_fp32, emb_int8)
        cosine_similarities.append(sim)

    avg_fp32_ms = float(np.mean(fp32_latencies))
    avg_int8_ms = float(np.mean(int8_latencies))
    speedup = (avg_fp32_ms / avg_int8_ms) if avg_int8_ms > 0 else 1.0
    avg_cosine_drift_sim = float(np.mean(cosine_similarities))

    return {
        "fp32_size_mb": round(fp32_size_mb, 2),
        "int8_size_mb": round(int8_size_mb, 2),
        "compression_ratio_percent": round(compression_ratio, 2),
        "fp32_latency_ms": round(avg_fp32_ms, 2),
        "int8_latency_ms": round(avg_int8_ms, 2),
        "fp32_fps": round(1000.0 / avg_fp32_ms, 2) if avg_fp32_ms > 0 else 0,
        "int8_fps": round(1000.0 / avg_int8_ms, 2) if avg_int8_ms > 0 else 0,
        "speedup_factor": round(speedup, 2),
        "cosine_similarity_fp32_vs_int8": round(avg_cosine_drift_sim, 4),
    }
