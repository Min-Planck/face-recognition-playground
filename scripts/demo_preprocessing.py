"""
Script chạy thử nghiệm module Preprocessing & Augmentation trên ảnh thật từ data/test_images/.
Xuất ảnh so sánh trực quan và bảng số liệu phân tích chất lượng ảnh.
Style hiển thị: Thanh banner màu đen với chữ trắng (black bar with white text).
"""

import os
import sys
import shutil
import cv2
import numpy as np
import yaml

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Đảm bảo import được package src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.clahe import (
    apply_clahe,
    apply_denoise,
    apply_sharpen,
    preprocess_image,
)
from src.preprocessing.augmentation import generate_hard_case_suite


def compute_image_stats(image: np.ndarray) -> dict:
    """Tính toán các chỉ số thống kê độ sáng, tương phản và độ nét của ảnh."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "mean_brightness": round(mean_brightness, 2),
        "std_contrast": round(std_contrast, 2),
        "laplacian_variance": round(laplacian_var, 2),
    }


def draw_caption_banner(
    image: np.ndarray,
    text: str,
    banner_height: int = 40,
    font_scale: float = 0.65,
    thickness: int = 2,
    position: str = "bottom",
) -> np.ndarray:
    """
    Vẽ dải banner màu đen (black bar) với chữ trắng (white text) mô tả nội dung ảnh.
    """
    h, w = image.shape[:2]
    out = image.copy()

    if position == "bottom":
        y1 = max(0, h - banner_height)
        y2 = h
        text_y = h - (banner_height // 2) + int(6 * font_scale)
    else:  # top
        y1 = 0
        y2 = min(h, banner_height)
        text_y = (banner_height // 2) + int(6 * font_scale)

    # Vẽ thanh chữ nhật đen
    cv2.rectangle(out, (0, y1), (w, y2), (0, 0, 0), -1)

    # Căn lề trái 12px
    text_x = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        out,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


def main():
    img_dir = "data/test_images"
    output_dir = "outputs/figures/preprocessing_demo"

    os.makedirs(output_dir, exist_ok=True)

    # Đọc config pipeline
    with open("config/pipeline.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    prep_config = config.get("preprocessing", {})

    image_files = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_name in image_files:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        base_name = os.path.splitext(img_name)[0]
        print(f"=== Đang xử lý ảnh: {img_name} (Kích thước: {img.shape}) ===")

        # 1. Pipeline tuần tự
        img_clahe = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
        img_denoised = apply_denoise(img_clahe, method="bilateral")
        img_sharpened = apply_sharpen(img_denoised, strength=0.4)

        # 2. Pipeline qua hàm wrapper chính
        img_final = preprocess_image(img, config=prep_config)

        # 3. Tính toán thống kê chất lượng ảnh
        stats_raw = compute_image_stats(img)
        stats_final = compute_image_stats(img_final)

        print(f"  [Gốc]       Độ sáng: {stats_raw['mean_brightness']:6.2f} | Tương phản: {stats_raw['std_contrast']:6.2f} | Độ nét (Laplacian Var): {stats_raw['laplacian_variance']:8.2f}")
        print(f"  [Xử lý]     Độ sáng: {stats_final['mean_brightness']:6.2f} | Tương phản: {stats_final['std_contrast']:6.2f} | Độ nét (Laplacian Var): {stats_final['laplacian_variance']:8.2f}")

        # 4. Lưu ảnh so sánh bước tiền xử lý với style banner đen chữ trắng
        h_target = 420
        scale = h_target / img.shape[0]
        w_target = int(img.shape[1] * scale)
        dim = (w_target, h_target)

        r_img = cv2.resize(img, dim)
        r_clahe = cv2.resize(img_clahe, dim)
        r_denoised = cv2.resize(img_denoised, dim)
        r_final = cv2.resize(img_final, dim)

        r_img = draw_caption_banner(r_img, "1. Raw Input", banner_height=42, font_scale=0.65)
        r_clahe = draw_caption_banner(r_clahe, "2. + CLAHE", banner_height=42, font_scale=0.65)
        r_denoised = draw_caption_banner(r_denoised, "3. + Denoise", banner_height=42, font_scale=0.65)
        r_final = draw_caption_banner(r_final, "4. + Sharpen (Final)", banner_height=42, font_scale=0.65)

        comparison = np.hstack([r_img, r_clahe, r_denoised, r_final])
        comp_path = os.path.join(output_dir, f"comparison_{base_name}.jpg")
        cv2.imwrite(comp_path, comparison)
        print(f"  -> Đã lưu ảnh so sánh: {comp_path}")

        # 5. Sinh hard cases với banner đen chữ trắng
        hard_cases = generate_hard_case_suite(img)
        case_items = list(hard_cases.items())
        grid_rows = []
        cell_dim = (260, 260)

        for i in range(0, len(case_items), 3):
            row_imgs = []
            for name, c_img in case_items[i:i+3]:
                res = cv2.resize(c_img, cell_dim)
                title = name.replace("_", " ").title()
                res = draw_caption_banner(res, title, banner_height=36, font_scale=0.55, thickness=1)
                row_imgs.append(res)
            while len(row_imgs) < 3:
                row_imgs.append(np.zeros((cell_dim[1], cell_dim[0], 3), dtype=np.uint8))
            grid_rows.append(np.hstack(row_imgs))

        hard_case_grid = np.vstack(grid_rows)
        hc_path = os.path.join(output_dir, f"hard_cases_{base_name}.jpg")
        cv2.imwrite(hc_path, hard_case_grid)
        print(f"  -> Đã lưu lưới hard cases: {hc_path}\n")

    print("=== Hoàn tất thử nghiệm Preprocessing & Augmentation ===")


if __name__ == "__main__":
    main()
