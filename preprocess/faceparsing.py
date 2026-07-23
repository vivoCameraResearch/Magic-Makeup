import argparse
import os
import sys
from typing import Iterable, List
import pandas as pd

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

CONVERT_DICT = {
    12: 13,  # down lip
    11: 9,   # up lip
    1: 4,    # face
    2: 8,    # nose
    17: 10,  # neck
    16: 10,  # neck
    13: 12,  # hair
    5: 1,    # right eye
    4: 6,    # left eye
    7: 2,    # right eyebrow
    6: 7,    # left eyebrow
    9: 5,    # right ear
    8: 3,    # left ear
    15: 0,   # ear ring
    10: 11,  # teeth
    18: 0,   # shirt
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

import re
import glob

def extract_base_name(filename: str) -> str:
    """
    简单提取基础名称：去除常见后缀
    1446_2338_mask.png -> 1446_2338
    """
    name = os.path.splitext(filename)[0]  # 去除扩展名
    # 去除常见后缀
    name = re.sub(r'_(mask|crop\d+.*|warp)$', '', name)
    return name

def collect_images_from_csv(csv_path: str, img_dir: str, filename_col: str = 'filename', 
                           label_col: str = 'label', target_label: str = 'face') -> List[str]:
    """
    从CSV读取文件名，根据标签筛选，精确匹配文件名
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # 检查列是否存在
        if filename_col not in df.columns:
            print(f"Error: Column '{filename_col}' not found. Available columns: {list(df.columns)}")
            return []
        if label_col not in df.columns:
            print(f"Error: Column '{label_col}' not found. Available columns: {list(df.columns)}")
            return []
        
        # 显示可用的标签
        available_labels = df[label_col].dropna().unique()
        print(f"Available labels: {available_labels}")
        
        # 直接匹配标签
        print(f"Filtering for label: '{target_label}'")
        target_rows = df[df[label_col] == target_label]
        
        print(f"Found {len(target_rows)} rows with label '{target_label}'")
        
        if len(target_rows) == 0:
            print(f"Warning: No rows found with label '{target_label}'")
            print(f"Available labels: {list(available_labels)}")
            return []
        
        # 过滤掉文件名为空的行
        print(f"Checking for empty filenames in column '{filename_col}'...")
        
        # 统计空值情况
        total_rows = len(target_rows)
        
        # 检查空值：NaN、空字符串、'nan'字符串
        empty_filename_mask = (
            target_rows[filename_col].isna() | 
            (target_rows[filename_col].astype(str).str.strip() == '') | 
            (target_rows[filename_col].astype(str).str.lower() == 'nan')
        )
        empty_count = empty_filename_mask.sum()
        
        if empty_count > 0:
            print(f"Found {empty_count} rows with empty filenames, skipping them")
            target_rows = target_rows[~empty_filename_mask]
            print(f"After filtering empty filenames: {len(target_rows)} rows remaining")
        
        if len(target_rows) == 0:
            print("Warning: No valid rows remaining after filtering empty filenames")
            return []
        
        image_paths = []
        processed_count = 0
        found_count = 0
        not_found_files = []
        
        print("Processing valid rows...")
        for idx, (_, row) in enumerate(target_rows.iterrows()):
            filename = row[filename_col]
            
            # 转换为字符串并清理
            filename_str = str(filename).strip()
            processed_count += 1
            
            # 精确匹配：直接构建完整路径
            full_path = os.path.join(img_dir, filename_str)
            
            # 检查文件是否存在且是图片文件
            if os.path.exists(full_path) and is_image_file(full_path):
                image_paths.append(full_path)
                found_count += 1
            else:
                not_found_files.append(filename_str)
                print(f"  Row {idx+1}: File not found: {filename_str}")
            
            # 进度提示
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(target_rows)} rows, found {found_count} images")
        
        print(f"\nProcessing summary:")
        print(f"  - Total matching label rows: {total_rows}")
        print(f"  - Rows with empty filenames: {empty_count}")
        print(f"  - Valid rows processed: {processed_count}")
        print(f"  - Image files found: {found_count}")
        print(f"  - Files not found: {len(not_found_files)}")
        
        if not_found_files and len(not_found_files) <= 10:
            print(f"  - Not found files: {not_found_files}")
        elif not_found_files:
            print(f"  - First 10 not found files: {not_found_files[:10]}")
        
        return sorted(image_paths)
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_one(image_processor, model, device: str, img_path: str, save_path: str, save_binary: bool = True):
    image = Image.open(img_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        labels = upsampled_logits.argmax(dim=1)[0].detach().cpu().numpy()
    
    new_labels = labels.copy()
    for key, value in CONVERT_DICT.items():
        new_labels[labels == key] = value
    
    if save_binary:
        # 修复二值化结果
        binary_mask = np.zeros_like(new_labels, dtype=np.uint8)  # 初始化为0（黑色）
        
        # 定义要设置为白色的标签
        white_labels = [1, 6, 4, 8, 13, 9, 11, 2, 7]  # face, nose, hair, right ear, teeth, right eyebrow, left eyebrow
        
        # 将指定标签的区域设置为白色
        for label in white_labels:
            binary_mask[new_labels == label] = 255
        
        # 检查检测到的像素数量
        total_white_pixels = np.sum(binary_mask == 255)
        
        binary_mask_img = Image.fromarray(binary_mask)
        final_path = normalize_save_path(img_path, save_path)
        binary_mask_img.save(final_path)
        print(f"Saved binary mask to: {final_path}")
    else:
        # 保存原始解析结果
        new_labels = new_labels.astype("uint8")
        new_labels_img = Image.fromarray(new_labels)
        final_path = normalize_save_path(img_path, save_path)
        new_labels_img.save(final_path)
        print(f"Saved parsing mask to: {final_path}")

def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS

def normalize_save_path(img_path: str, save_path: str) -> str:
    """
    修改保存路径逻辑：保存为 base_name_mask.png 格式
    """
    save_path = save_path.strip()
    
    if save_path.endswith(os.sep) or os.path.isdir(save_path):
        # 如果save_path是目录，生成新的文件名
        original_filename = os.path.basename(img_path)
        base_name = extract_base_name(original_filename)  # 提取基础名称
        # new_filename = f"{base_name}_mask.png"  # 添加_mask后缀
        new_filename = f"{base_name}.png"  # 添加_mask后缀
        out_dir = save_path
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, new_filename)
    else:
        # 如果save_path是文件路径
        parent = os.path.dirname(save_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        
        root, ext = os.path.splitext(save_path)
        if not ext:
            save_path = root + ".png"
        return save_path

def collect_images(img_path: str, recursive: bool = False) -> List[str]:
    """原有的图片收集函数，保持不变以兼容原有功能"""
    if os.path.isfile(img_path):
        return [img_path] if is_image_file(img_path) else []
    if os.path.isdir(img_path):
        files = []
        if recursive:
            for root, _, names in os.walk(img_path):
                for n in names:
                    f = os.path.join(root, n)
                    if is_image_file(f):
                        files.append(f)
        else:
            for n in os.listdir(img_path):
                f = os.path.join(img_path, n)
                if os.path.isfile(f) and is_image_file(f):
                    files.append(f)
        return sorted(files)
    return []

def main(img_path: str, save_path: str, recursive: bool = False, csv_path: str = None, 
         filename_col: str = 'filename', label_col: str = 'label', target_label: str = 'face'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    image_processor = SegformerImageProcessor.from_pretrained(
        "jonathandinu/face-parsing"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "jonathandinu/face-parsing"
    ).to(device)
    model.eval()

    # 根据是否提供CSV文件来选择图片收集方式
    if csv_path:
        print(f"Using CSV mode: reading from {csv_path}")
        images = collect_images_from_csv(csv_path, img_path, filename_col, label_col, target_label)
    else:
        print(f"Using directory mode: reading from {img_path}")
        images = collect_images(img_path, recursive=recursive)
    
    if not images:
        raise FileNotFoundError(f"No images found. CSV path: {csv_path}, Image path: {img_path}")

    print(f"Processing {len(images)} images...")

    # 多图时，save_path 必须为目录
    if len(images) > 1:
        if not (save_path.endswith(os.sep) or os.path.isdir(save_path)):
            # 若是文件名，自动改为目录
            print("Multiple images detected. Converting save_path to directory.", file=sys.stderr)
            os.makedirs(save_path, exist_ok=True)
            if not save_path.endswith(os.sep):
                save_path = save_path + os.sep

    success_count = 0
    for i, img in enumerate(images, 1):
        try:
            print(f"Processing {i}/{len(images)}: {os.path.basename(img)}")
            process_one(image_processor, model, device, img, save_path, save_binary=args.binary)
            success_count += 1
        except Exception as e:
            print(f"Error processing {img}: {e}", file=sys.stderr)
    
    print(f"Successfully processed {success_count}/{len(images)} images")

# python faceparsing.py --binary
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face parsing with optional CSV filtering")
    parser.add_argument(
        "--img_path",
        type=str,
        default="MagicMakeup/example/makeup/image",
        help="Path to the input image directory (when using CSV mode) or image/directory path (when not using CSV)."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="MagicMakeup/example/makeup/mask/face",
        help="Path to save the output image(s). If multiple images, should be a directory."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search images in subdirectories when img_path is a directory (only used when not using CSV mode)."
    )
    parser.add_argument(
        "--binary", 
        default=True,
        help="Save binary mask (face=white, background=black)"
    )
    
    # CSV相关参数
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to CSV file containing filename and label columns. If provided, will filter images based on CSV."
    )
    parser.add_argument(
        "--filename_col",
        type=str,
        default="reference",
        help="Column name for filenames in CSV (default: 'filename')"
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Column name for labels in CSV (default: 'label')"
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default="eyes,lip,face",
        help="Target label value to filter (default: 'face')"
    )
    
    args = parser.parse_args()
    
    main(
        args.img_path, 
        args.save_path, 
        args.recursive,
        args.csv_path,
        args.filename_col,
        args.label_col,
        args.target_label
    )
