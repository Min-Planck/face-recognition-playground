"""
Unit tests cho module quantize.py.
Tạo một ONNX dummy model để kiểm tra quá trình quantize dynamic INT8 và so sánh benchmark.
"""

import os
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.export.quantize import (
    quantize_onnx_model_dynamic,
    ONNXEmbedderRunner,
    benchmark_quantization_comparison,
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

    # Export Tiny PyTorch to ONNX FP32
    model = TinyFaceEmbedder()
    model.eval()
    dummy_input = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model,
        dummy_input,
        fp32_path,
        input_names=["input"],
        output_names=["embedding"],
        opset_version=17,
    )

    # Quantize to INT8
    quantize_onnx_model_dynamic(fp32_path, int8_path)

    return str(fp32_path), str(int8_path)


def test_quantization_process(dummy_onnx_models):
    fp32_path, int8_path = dummy_onnx_models
    assert os.path.exists(fp32_path)
    assert os.path.exists(int8_path)

    # Kiểm tra kích thước file
    fp32_size = os.path.getsize(fp32_path)
    int8_size = os.path.getsize(int8_path)
    assert int8_size > 0 and fp32_size > 0


def test_onnx_runner(dummy_onnx_models):
    fp32_path, int8_path = dummy_onnx_models

    runner_fp32 = ONNXEmbedderRunner(fp32_path)
    runner_int8 = ONNXEmbedderRunner(int8_path)

    dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
    emb_fp32 = runner_fp32.run(dummy_input)
    emb_int8 = runner_int8.run(dummy_input)

    assert emb_fp32.shape == (128,)
    assert emb_int8.shape == (128,)
    assert np.isclose(np.linalg.norm(emb_fp32), 1.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(emb_int8), 1.0, atol=1e-4)


def test_benchmark_comparison(dummy_onnx_models):
    fp32_path, int8_path = dummy_onnx_models
    results = benchmark_quantization_comparison(
        fp32_path,
        int8_path,
        dummy_input_shape=(1, 3, 112, 112),
        n_iterations=5,
    )
    assert "compression_ratio_percent" in results
    assert "cosine_similarity_fp32_vs_int8" in results
    assert results["cosine_similarity_fp32_vs_int8"] > 0.85
