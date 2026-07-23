#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据已有CSV文件读取src和ref，生成对应的gen列（src_ref格式）
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# -------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_valid_image_file(path: Path) -> bool:
    """检查文件是否是有效的图像文件"""
    if not path.exists():
        return False
    
    if path.stat().st_size == 0:
        print(f"[WARN] Empty file: {path}")
        return False
    
    try:
        with Image.open(str(path)) as img:
            img.verify()  # 验证图像完整性
        return True
    except Exception as e:
        print(f"[WARN] Invalid image file {path}: {e}")
        return False

def read_rgb(path: Path, target_size: Tuple[int, int] = None) -> Optional[Image.Image]:
    """
    读取图片并转换为RGB格式，可选择resize到指定分辨率
    
    Args:
        path: 图片路径
        target_size: 目标尺寸 (width, height)，如果为None则不调整
    
    Returns:
        处理后的PIL图片对象，如果失败则返回None
    """
    try:
        # 首先检查文件是否有效
        if not is_valid_image_file(path):
            return None
        
        img = Image.open(str(path)).convert("RGB")
        
        # 如果指定了目标尺寸，进行resize
        if target_size is not None:
            target_w, target_h = target_size
            # 使用高质量的LANCZOS算法进行resize
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            print(f"[DEBUG] Resized {path.name} to {target_w}x{target_h}")
        
        return img
        
    except Exception as e:
        print(f"[ERROR] Failed to read image {path}: {e}")
        return None

def list_images_recursive(root: Path) -> Dict[str, Path]:
    """递归列出目录中的所有图像文件，并验证其有效性"""
    mapping = {}
    invalid_count = 0
    
    print(f"[INFO] Scanning images in: {root}")
    
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            if is_valid_image_file(p):
                mapping[p.stem] = p
            else:
                invalid_count += 1
    
    print(f"[INFO] Found {len(mapping)} valid images, {invalid_count} invalid files in {root}")
    return mapping

# -------------------------
# MediaPipe Face Landmarker 
# -------------------------
def build_face_landmarker(model_path: Path, num_faces: int = 1) -> vision.FaceLandmarker:
    base_opts = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=num_faces,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.FaceLandmarker.create_from_options(options)

def detect_landmarks_pixels(detector: vision.FaceLandmarker,
                            pil_img: Optional[Image.Image],
                            face_index: int = 0) -> Optional[np.ndarray]:
    """检测人脸关键点，增加空值检查"""
    if pil_img is None:
        return None
        
    try:
        np_rgb = np.array(pil_img)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_rgb)
        res = detector.detect(mp_img)
        if not res.face_landmarks:
            return None

        W, H = pil_img.width, pil_img.height
        lms = res.face_landmarks[min(face_index, len(res.face_landmarks) - 1)]
        out = np.zeros((len(lms), 3), dtype=np.float32)
        for i, lm in enumerate(lms):
            out[i, 0] = lm.x * W
            out[i, 1] = lm.y * H
            out[i, 2] = lm.z
        return out
    except Exception as e:
        print(f"[ERROR] Landmark detection failed: {e}")
        return None

def save_landmarks_npy(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)

# -------------------------
# 从CSV文件读取src和ref，生成gen路径
# -------------------------
def load_src_ref_from_csv(csv_path: Path) -> List[Dict[str, str]]:
    """从CSV文件读取src和ref列"""
    print(f"[INFO] Loading CSV from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"[DEBUG] CSV columns: {df.columns.tolist()}")
        
        # 检查必需的列
        required_cols = {"src", "ref"}
        available_cols = set(df.columns)
        
        if not required_cols.issubset(available_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}, but got: {available_cols}")
        
        # 转换为字典列表
        pairs = []
        for _, row in df.iterrows():
            pairs.append({
                "src": str(row["src"]).strip(),
                "ref": str(row["ref"]).strip()
            })
        
        print(f"[INFO] Loaded {len(pairs)} pairs from CSV")
        return pairs
        
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        raise

def generate_triplets_from_pairs(pairs: List[Dict[str, str]], 
                                gen_dir: Path,
                                separator: str = "_") -> List[Tuple[str, str, str, str, str, str]]:
    """
    根据src和ref生成三元组，查找对应的gen文件
    返回: (src_path, ref_path, gen_path, src_stem, ref_stem, gen_stem)
    """
    # 构建gen文件映射
    gen_map = list_images_recursive(gen_dir)
    
    triplets = []
    missing_gen = 0
    invalid_src = 0
    invalid_ref = 0
    
    print(f"[INFO] Processing {len(pairs)} pairs...")
    
    for i, pair in enumerate(pairs):
        src_path = Path(pair["src"])
        ref_path = Path(pair["ref"])
        
        # 检查src和ref文件是否存在且有效
        if not src_path.exists():
            print(f"[WARN] Source file not found: {src_path}")
            continue
        if not ref_path.exists():
            print(f"[WARN] Reference file not found: {ref_path}")
            continue
            
        # 验证图像文件有效性
        if not is_valid_image_file(src_path):
            invalid_src += 1
            continue
        if not is_valid_image_file(ref_path):
            invalid_ref += 1
            continue
        
        # 提取stem
        src_stem = src_path.stem
        ref_stem = ref_path.stem
        
        # 生成期望的gen文件名 (src_stem + separator + ref_stem)
        expected_gen_stem = f"{src_stem}{separator}{ref_stem}"
        
        # 查找对应的gen文件
        if expected_gen_stem in gen_map:
            gen_path = gen_map[expected_gen_stem]
            triplets.append((
                str(src_path), str(ref_path), str(gen_path),
                src_stem, ref_stem, expected_gen_stem
            ))
        else:
            missing_gen += 1
            if missing_gen <= 10:  # 只显示前10个缺失的文件
                print(f"[WARN] Generated file not found: {expected_gen_stem}")
            elif missing_gen == 11:
                print("[WARN] ... (more missing gen files, suppressing further warnings)")
    
    print(f"[INFO] Validation results:")
    print(f"  - Valid triplets: {len(triplets)}")
    print(f"  - Missing gen files: {missing_gen}")
    print(f"  - Invalid src files: {invalid_src}")
    print(f"  - Invalid ref files: {invalid_ref}")
    
    return triplets

def main():
    ap = argparse.ArgumentParser(description="Extract MediaPipe landmarks from CSV pairs and export results")
    ap.add_argument("--model", default="MagicMakeup/mediapipe/face_landmarker.task", type=str, help="Path to face_landmarker.task")
    ap.add_argument("--model_name", default="MagicMakeup", type=str, help="Custom model name for organizing output files")
   
    ap.add_argument("--target_width", type=int, default=1024, help="Target image width (default: None, keep original)")
    ap.add_argument("--target_height", type=int, default=1024, help="Target image height (default: None, keep original)")
    ap.add_argument("--input_csv", default="MagicMakeup/example/pairs.csv", type=str, help="Input CSV file with src and ref columns")
    ap.add_argument("--gen_dir", default="MagicMakeup/example/output/face", type=str, help="Generated(makeup transferred) images dir")
    ap.add_argument("--out_root", default="MagicMakeup/metrics/example", type=str, help="Root dir to save landmarks .npy and CSV file")
    
    ap.add_argument("--num_faces", type=int, default=1, help="Max faces per image (take first)")
    ap.add_argument("--force_reprocess", action="store_true", help="Force reprocess all images (ignore existing .npy files)")
    ap.add_argument("--separator", default="_", type=str, help="Separator between src and ref in gen filename (default: '_')")
    ap.add_argument("--continue_on_error", default=True, help="Continue processing even when encountering errors")
    args = ap.parse_args()

    model_path = Path(args.model)
    input_csv = Path(args.input_csv)
    gen_dir = Path(args.gen_dir)
    out_root = Path(args.out_root)
    
    # 处理target_size参数
    target_size = None
    if args.target_width is not None and args.target_height is not None:
        target_size = (args.target_width, args.target_height)
        print(f"[INFO] Images will be resized to: {target_size}")
    
    # 检查输入文件
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generated images directory not found: {gen_dir}")
    
    csv_out = out_root / f"{args.model_name}.csv"
    src_lmk_dir = out_root / "src"
    gen_lmk_dir = out_root / "gen" / args.model_name

    print(f"[INFO] Model name: {args.model_name}")
    print(f"[INFO] Input CSV: {input_csv}")
    print(f"[INFO] Generated images dir: {gen_dir}")
    print(f"[INFO] Output structure:")
    print(f"  - CSV file: {csv_out}")
    print(f"  - Src landmarks: {src_lmk_dir}")
    print(f"  - Gen landmarks: {gen_lmk_dir}")

    detector = build_face_landmarker(model_path, num_faces=args.num_faces)

    # 从CSV加载src和ref对
    pairs = load_src_ref_from_csv(input_csv)
    
    # 生成三元组
    triplets = generate_triplets_from_pairs(pairs, gen_dir, args.separator)
    print(f"[INFO] Generated triplets: {len(triplets)}")
    
    if len(triplets) == 0:
        print("[ERROR] No matching triplets found!")
        return

    src_lmk_dir.mkdir(parents=True, exist_ok=True)
    gen_lmk_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    miss_src, miss_gen = 0, 0
    skip_src, skip_gen = 0, 0
    error_count = 0
    
    # 跟踪已处理的src文件，避免重复处理
    processed_src_stems: Set[str] = set()

    for src_path_str, ref_path_str, gen_path_str, src_stem, ref_stem, gen_stem in tqdm(triplets, desc="Processing triplets"):
        try:
            src_img = Path(src_path_str)
            ref_img = Path(ref_path_str)
            gen_img = Path(gen_path_str)

            # src landmarks 使用 src_stem 命名
            src_lmk_path = src_lmk_dir / f"{src_stem}.npy"
            
            # 只有当src_stem未处理过时才处理src landmarks
            if src_stem not in processed_src_stems:
                if src_lmk_path.exists() and not args.force_reprocess:
                    skip_src += 1
                else:
                    src_pil = read_rgb(src_img, target_size)
                    arr = detect_landmarks_pixels(detector, src_pil)
                    if arr is not None:
                        save_landmarks_npy(arr, src_lmk_path)
                    else:
                        src_lmk_path = None
                        miss_src += 1
                
                # 标记该src_stem已处理
                processed_src_stems.add(src_stem)
            else:
                # 如果已经处理过，检查文件是否存在
                if not src_lmk_path.exists():
                    src_lmk_path = None

            # gen landmarks 使用完整的 gen_stem
            gen_lmk_path = gen_lmk_dir / f"{gen_stem}.npy"
            
            if gen_lmk_path.exists() and not args.force_reprocess:
                skip_gen += 1
            else:
                gen_pil = read_rgb(gen_img, target_size)
                arr = detect_landmarks_pixels(detector, gen_pil)
                if arr is not None:
                    save_landmarks_npy(arr, gen_lmk_path)
                else:
                    gen_lmk_path = None
                    miss_gen += 1

            rows.append({
                "src": str(src_img.resolve()),
                "ref": str(ref_img.resolve()),
                "gen": str(gen_img.resolve()),
                "src_lmk": str(src_lmk_path.resolve()) if src_lmk_path else "",
                "gen_lmk": str(gen_lmk_path.resolve()) if gen_lmk_path else "",
            })
            
        except Exception as e:
            error_count += 1
            print(f"[ERROR] Failed to process triplet {src_stem}_{ref_stem}: {e}")
            
            if args.continue_on_error:
                # 添加一个错误记录
                rows.append({
                    "src": src_path_str,
                    "ref": ref_path_str,
                    "gen": gen_path_str,
                    "src_lmk": "",
                    "gen_lmk": "",
                })
                continue
            else:
                print("[ERROR] Stopping due to error. Use --continue_on_error to skip errors.")
                break

    df = pd.DataFrame(rows, columns=["src", "ref", "gen", "src_lmk", "gen_lmk"])
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False, encoding="utf-8")
    
    print(f"[OK] pairs.csv -> {csv_out}")
    print(f"[INFO] 处理统计:")
    print(f"  - 匹配的三元组: {len(triplets)}")
    print(f"  - 成功处理的记录: {len(rows)}")
    print(f"  - 唯一的src文件数: {len(processed_src_stems)}")
    print(f"  - 跳过已存在的文件: src={skip_src}, gen={skip_gen}")
    print(f"  - 处理错误数: {error_count}")
    if miss_src or miss_gen:
        print(f"  - 未检测到人脸的条目: src={miss_src}, gen={miss_gen}")

if __name__ == "__main__":
    main()