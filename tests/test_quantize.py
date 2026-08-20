"""
Unit tests cho module quantize.py.
Kiểm tra quá trình Static PTQ, CalibrationDataReader, ONNXEmbedderRunner và benchmark comparison.
"""

import os
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.export.quantize import (
    FaceCalibrationDataReader,
    ONNXEmbedderRunner,
    benchmark_quantization_comparison,
    quantize_onnx_model_dynamic,
    quantize_onnx_model_static,
)


class TinyFaceEmbedder(nn.Module):
    """Mô hình PyTorch đơn giản chuẩn xác cho shape inference."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 128)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


@pytest.fixture(scope="module")
def dummy_onnx_models(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("onnx_test")
    fp32_path = os.path.join(temp_dir, "tiny_embedder_fp32.onnx")
    int8_path = os.path.join(temp_dir, "tiny_embedder_int8.onnx")

    # Tạo vài ảnh giả lập cho calibration
    calib_dir = os.path.join(temp_dir, "calib_imgs")
    os.makedirs(calib_dir, exist_ok=True)
    calib_paths = []
    for i in range(5):
        p = os.path.join(calib_dir, f"img_{i}.jpg")
        cv2.imwrite(p, np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8))
        calib_paths.append(p)

    # Export Tiny PyTorch to ONNX FP32
    model = TinyFaceEmbedder()
    model.eval()
    dummy_input = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model,
        dummy_input,
        fp32_path,
        input_names=["input.1"],
        output_names=["embedding"],
        opset_version=17,
    )

    # Quantize to INT8 Static PTQ
    quantize_onnx_model_static(
        input_onnx_path=str(fp32_path),
        output_quant_path=str(int8_path),
        calibration_image_paths=calib_paths,
        model_type="arcface",
        per_channel=False,
    )

    return str(fp32_path), str(int8_path), calib_paths


def test_quantization_process(dummy_onnx_models):
    fp32_path, int8_path, _ = dummy_onnx_models
    assert os.path.exists(fp32_path)
    assert os.path.exists(int8_path)

    # Kiểm tra kích thước file
    fp32_size = os.path.getsize(fp32_path)
    int8_size = os.path.getsize(int8_path)
    assert int8_size > 0 and fp32_size > 0


def test_onnx_runner(dummy_onnx_models):
    fp32_path, int8_path, _ = dummy_onnx_models

    runner_fp32 = ONNXEmbedderRunner(fp32_path, model_type="arcface")
    runner_int8 = ONNXEmbedderRunner(int8_path, model_type="arcface")

    dummy_input = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    emb_fp32 = runner_fp32.embed(dummy_input)
    emb_int8 = runner_int8.embed(dummy_input)

    assert emb_fp32.shape == (128,)
    assert emb_int8.shape == (128,)
    assert np.isclose(np.linalg.norm(emb_fp32), 1.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(emb_int8), 1.0, atol=1e-4)


def test_benchmark_comparison(dummy_onnx_models):
    fp32_path, int8_path, calib_paths = dummy_onnx_models
    test_imgs = [cv2.imread(p) for p in calib_paths if cv2.imread(p) is not None]

    results = benchmark_quantization_comparison(
        fp32_path,
        int8_path,
        test_face_images=test_imgs,
        model_type="arcface",
        n_iterations=5,
    )
    assert "compression_ratio_percent" in results
    assert "cosine_similarity_mean" in results
    assert results["cosine_similarity_mean"] > 0.80
