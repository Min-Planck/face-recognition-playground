"""
Script xuất và chuẩn hóa các mô hình nhận diện khuôn mặt sang định dạng ONNX FP32:
1. ArcFace ResNet50: w600k_r50.onnx -> models/arcface_fp32.onnx
2. FaceNet512: Inception-ResNet-v1 Keras -> models/facenet512_fp32.onnx
"""

import io
import os
import shutil
import sys
import numpy as np

# Đảm bảo stdout hỗ trợ UTF-8 trên Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Thiết lập đường dẫn project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def export_arcface_fp32() -> str:
    """
    Chuyển đổi mô hình ArcFace (Keras / DeepFace) sang models/arcface_fp32.onnx qua tf2onnx.
    Đảm bảo 100% đồng nhất trọng số (weights) với mô hình đã hiệu chuẩn trong dự án.
    """
    out_path = os.path.join(MODELS_DIR, "arcface_fp32.onnx")
    print("  Đang nạp trọng số Keras của ArcFace từ DeepFace...")

    import tensorflow as tf
    import tf2onnx
    from deepface import DeepFace

    df_model = DeepFace.build_model("ArcFace")
    keras_model = df_model.model

    # Kích thước đầu vào chuẩn của DeepFace ArcFace là (1, 112, 112, 3)
    input_signature = [tf.TensorSpec([1, 112, 112, 3], tf.float32, name="input_1")]

    print(f"  Đang chuyển đổi DeepFace ArcFace sang ONNX (opset=17)...")
    tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_signature,
        opset=17,
        output_path=out_path,
    )
    print(f"  [OK] Đã xuất ArcFace FP32 ONNX -> {out_path} ({os.path.getsize(out_path) / (1024*1024):.2f} MB)")
    return out_path


def export_facenet512_fp32() -> str:
    """
    Chuyển đổi mô hình FaceNet512 (Keras) sang models/facenet512_fp32.onnx qua tf2onnx.
    """
    out_path = os.path.join(MODELS_DIR, "facenet512_fp32.onnx")
    print("  Đang nạp trọng số Keras của FaceNet512 từ DeepFace...")

    import tensorflow as tf
    import tf2onnx
    from deepface import DeepFace

    # Nạp mô hình FaceNet512 từ DeepFace
    df_model = DeepFace.build_model("Facenet512")
    keras_model = df_model.model

    # Kích thước đầu vào chuẩn của FaceNet512 là (1, 160, 160, 3)
    input_signature = [tf.TensorSpec([1, 160, 160, 3], tf.float32, name="input_1")]

    print(f"  Đang chuyển đổi FaceNet512 sang ONNX (opset=17)...")
    tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_signature,
        opset=17,
        output_path=out_path,
    )
    print(f"  [OK] Đã xuất FaceNet512 FP32 ONNX -> {out_path} ({os.path.getsize(out_path) / (1024*1024):.2f} MB)")
    return out_path


def export_yolov8_onnx() -> str:
    """
    Chuyển đổi mô hình YOLOv8-Face (PyTorch .pt) sang models/yolov8n-face.onnx.
    Loại bỏ hoàn toàn dependency PyTorch khi thực thi suy luận trên CPU.
    """
    out_path = os.path.join(MODELS_DIR, "yolov8n-face.onnx")
    pt_path = os.path.join(MODELS_DIR, "yolov8n-face.pt")

    if not os.path.exists(pt_path):
        import urllib.request
        url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
        print(f"  Đang tải trọng số YOLOv8-Face từ {url}...")
        urllib.request.urlretrieve(url, pt_path)

    print("  Đang chuyển đổi YOLOv8-Face sang ONNX (opset=17)...")
    from ultralytics import YOLO
    model = YOLO(pt_path)
    exported_path = model.export(format="onnx", opset=17)
    if os.path.abspath(exported_path) != os.path.abspath(out_path):
        shutil.copy2(exported_path, out_path)

    print(f"  [OK] Đã xuất YOLOv8-Face ONNX -> {out_path} ({os.path.getsize(out_path) / (1024*1024):.2f} MB)")
    return out_path


def export_sface_fp32() -> str:
    """
    Chuẩn bị mô hình SFace FP32 ONNX tại models/sface_fp32.onnx.
    """
    out_path = os.path.join(MODELS_DIR, "sface_fp32.onnx")
    if not os.path.exists(out_path):
        deepface_sface = os.path.expanduser("~/.deepface/weights/face_recognition_sface_2021dec.onnx")
        if os.path.exists(deepface_sface):
            shutil.copy2(deepface_sface, out_path)
        else:
            url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
            import urllib.request
            print(f"  Đang tải SFace ONNX từ {url}...")
            urllib.request.urlretrieve(url, out_path)
    print(f"  [OK] Đã có SFace FP32 ONNX -> {out_path} ({os.path.getsize(out_path) / (1024*1024):.2f} MB)")
    return out_path


def verify_exported_models():
    """Kiểm tra tính hợp lệ của tất cả các mô hình ONNX vừa xuất."""
    import onnxruntime as ort

    print("\n--- Kiểm tra nạp ONNX Runtime InferenceSession ---")
    model_list = [
        ("ArcFace FP32", "arcface_fp32.onnx"),
        ("FaceNet512 FP32", "facenet512_fp32.onnx"),
        ("SFace FP32", "sface_fp32.onnx"),
        ("YOLOv8-Face", "yolov8n-face.onnx"),
    ]
    for name, filename in model_list:
        p = os.path.join(MODELS_DIR, filename)
        if os.path.exists(p):
            sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            out = sess.get_outputs()[0]
            print(f"  {name:18s}: Input={inp.name} {inp.shape} ({inp.type}) -> Output={out.name} {out.shape}")
        else:
            print(f"  {name:18s}: [CHƯA TỒN TẠI] {p}")


def export_all_models():
    """Xuất tất cả mô hình sang ONNX."""
    export_arcface_fp32()
    export_facenet512_fp32()
    export_sface_fp32()
    export_yolov8_onnx()
    verify_exported_models()


if __name__ == "__main__":
    print("=" * 70)
    print("    CHUẨN BỊ VÀ XUẤT TOÀN BỘ MÔ HÌNH ONNX (DETECTOR & EMBEDDER)     ")
    print("=" * 70)

    export_all_models()
    print("\n Hoàn tất xuất toàn bộ ONNX models.")
