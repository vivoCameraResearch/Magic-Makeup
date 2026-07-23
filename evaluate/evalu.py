#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算化妆迁移的综合指标：Face-ID, DINO-I, CLIP-I, Self-sim, FID, KID, BG-MSE
"""
import tempfile
import shutil
import os
import sys
import csv
import math
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import gc

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from transformers import AutoModel

# 导入自定义模块
from score import LossG

# 配置路径
sys.path.insert(0, "../MagicMakeup")

try:
    from torch_fidelity import calculate_metrics
except ImportError as e:
    calculate_metrics = None
    print(f"[WARN] torch_fidelity not available: {e}")


def load_model_from_local_path(path: str):
    """Load a downloaded CVLFace model with its repository-provided code."""
    model_path = str(Path(path).expanduser().resolve())
    previous_cwd = os.getcwd()
    sys.path.insert(0, model_path)
    try:
        os.chdir(model_path)
        return AutoModel.from_pretrained(model_path, trust_remote_code=True)
    finally:
        os.chdir(previous_cwd)
        sys.path.pop(0)


class ImageProcessor:
    """图像处理工具类"""
    
    @staticmethod
    def read_image(path: Path, target_size: Tuple[int, int] = None) -> Image.Image:
        """
        读取图像并转换为RGB格式，可选择resize到指定分辨率
        """
        img = Image.open(str(path)).convert("RGB")
        
        # 如果指定了目标尺寸，进行resize
        if target_size is not None:
            target_w, target_h = target_size
            # 使用高质量的LANCZOS算法进行resize
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        return img

    
    @staticmethod
    def pil_to_tensor(image: Image.Image, device: str = 'cuda') -> torch.Tensor:
        """将PIL图像转换为模型输入tensor"""
        transform = Compose([
            ToTensor(), 
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(image).unsqueeze(0).to(device)


class FaceContourMaskGenerator:
    """基于MediaPipe landmarks的面部轮廓mask生成器"""
    
    FACE_CONTOUR_INDICES = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109, 10
    ]

    def __init__(self, blur_ksize: int = 21, mask_val: int = 255):
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.mask_val = mask_val

    def create_face_mask(self, img: Image.Image, landmarks: np.ndarray) -> np.ndarray:
        """从landmarks数组创建面部mask"""
        h, w = np.array(img).shape[:2]
        
        contour_points = []
        for idx in self.FACE_CONTOUR_INDICES:
            if idx < landmarks.shape[0]:
                x, y = landmarks[idx, 0], landmarks[idx, 1]
                contour_points.append((int(x), int(y)))
        
        # 创建mask
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(contour_points) >= 3:
            cv2.fillPoly(mask, [np.array(contour_points, dtype=np.int32)], self.mask_val)
            mask = cv2.GaussianBlur(mask, (self.blur_ksize, self.blur_ksize), 0)
        
        return mask

    def create_background_mask(self, img: Image.Image, landmarks: np.ndarray) -> np.ndarray:
        """创建背景mask（1 - 人脸mask）"""
        face_mask = self.create_face_mask(img, landmarks)
        # 归一化到0-1范围，然后取反
        face_mask_normalized = (face_mask > 0).astype(np.uint8)
        background_mask = 1 - face_mask_normalized
        return background_mask

    def apply_mask_to_image(self, img: Image.Image, mask: np.ndarray) -> Image.Image:
        """将mask应用到图像上"""
        np_img = np.array(img).astype(np.uint8)
        mask_3d = np.repeat((mask > 0)[:, :, None], 3, axis=2)
        np_img[~mask_3d] = 0
        return Image.fromarray(np_img)


class BackgroundMSECalculator:
    """背景区域MSE计算器"""
    
    def __init__(self, mask_generator: FaceContourMaskGenerator, target_size: Tuple[int, int] = None):
        self.mask_generator = mask_generator
        self.target_size = target_size
    
    def calculate_background_mse(self, src_path: Path, gen_path: Path, 
                               src_landmarks_path: Optional[Path]) -> float:
        """计算背景区域的MSE"""
        if not src_landmarks_path or not src_landmarks_path.exists():
            return float("nan")
        
        try:
            # 加载图像和landmarks
            src_img = ImageProcessor.read_image(src_path, self.target_size)
            gen_img = ImageProcessor.read_image(gen_path, self.target_size)
            landmarks = np.load(str(src_landmarks_path))
            
            # 创建背景mask
            bg_mask = self.mask_generator.create_background_mask(src_img, landmarks)
            
            # 转换图像为numpy数组
            src_array = np.array(src_img).astype(np.float32)
            gen_array = np.array(gen_img).astype(np.float32)
            
            # 应用背景mask并计算MSE
            bg_mask_3d = np.repeat(bg_mask[:, :, None], 3, axis=2)
            
            # 只在背景区域计算MSE
            if np.sum(bg_mask_3d) == 0:  # 如果没有背景区域
                return float("nan")
            
            src_bg = src_array[bg_mask_3d == 1]
            gen_bg = gen_array[bg_mask_3d == 1]
            
            mse = np.mean((src_bg - gen_bg) ** 2)
            return float(mse)
            
        except Exception as e:
            print(f"[WARN] Background MSE calculation failed for {src_path.name}: {e}")
            return float("nan")


class FaceIDCalculator:
    """Face-ID相似度计算器"""
    
    def __init__(self, fr_model, aligner, mask_generator: FaceContourMaskGenerator, device: str = "cuda", target_size: Tuple[int, int] = None):
        self.fr_model = fr_model
        self.aligner = aligner
        self.mask_generator = mask_generator
        self.device = device
        self.target_size = target_size
        self.supports_keypoints = self._check_keypoints_support()
    
    def _check_keypoints_support(self) -> bool:
        """检查FR模型是否支持keypoints参数"""
        try:
            import inspect
            if hasattr(self.fr_model, "model") and hasattr(self.fr_model.model, "net"):
                sig = inspect.signature(self.fr_model.model.net.forward)
            else:
                sig = inspect.signature(self.fr_model.forward)
            return 'keypoints' in sig.parameters
        except:
            return False
    
    def extract_feature(self, img_path: Path, landmarks_path: Optional[Path]) -> Optional[torch.Tensor]:
        """提取面部特征向量"""
        if not landmarks_path:
            print(f"[WARN] No landmarks path provided for {img_path.name}")
            return None
        
        landmarks_path = Path(landmarks_path)
        if not landmarks_path.exists():
            print(f"[WARN] Landmarks file not found: {landmarks_path}")
            return None
        
        try:
            # 加载图像和landmarks
            img = ImageProcessor.read_image(img_path, self.target_size)
            landmarks = np.load(str(landmarks_path))
            
            # 创建面部mask
            face_mask = self.mask_generator.create_face_mask(img, landmarks)
            
            # 应用mask并转换为tensor
            masked_img = self.mask_generator.apply_mask_to_image(img, face_mask)
            tensor = ImageProcessor.pil_to_tensor(masked_img, self.device)
            
            # 提取特征
            with torch.no_grad():
                aligned, _, landmarks_tensor, _, _, _ = self.aligner(tensor)
                if isinstance(aligned, (list, tuple)):
                    aligned = aligned[0]
                
                if self.supports_keypoints:
                    feature = self.fr_model(aligned, landmarks_tensor)
                else:
                    feature = self.fr_model(aligned)
            
            return F.normalize(feature, dim=-1).squeeze(0)
            
        except Exception as e:
            print(f"[WARN] Feature extraction failed for {img_path.name}: {e}")
            return None
    
    def calculate_similarity(self, feature1: Optional[torch.Tensor], feature2: Optional[torch.Tensor]) -> float:
        """计算两个特征向量的余弦相似度"""
        if feature1 is None or feature2 is None:
            return float("nan")
        return float(torch.sum(feature1 * feature2).item())


class MetricsCalculator:
    """综合指标计算器（含：加载三元组、逐对计算 DINO/CLIP 指标）"""

    def __init__(self, device: str = "cuda", target_size: Tuple[int, int] = None):
        # 验证设备
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA not available, falling back to CPU")
            device = "cpu"
        
        self.device = device
        
        # 验证并设置target_size
        if target_size is not None:
            if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
                raise ValueError("target_size must be a tuple/list of 2 integers")
            if not all(isinstance(x, int) and x > 0 for x in target_size):
                raise ValueError("target_size values must be positive integers")
            target_size = tuple(target_size)
        
        self.target_size = target_size
        self.lossg = LossG(cfg=None)
        
        # 初始化背景MSE计算器
        mask_gen = FaceContourMaskGenerator()
        self.bg_mse_calc = BackgroundMSECalculator(mask_gen, target_size)
        
        if target_size:
            print(f"[INFO] Images will be resized to: {target_size}")
        
        # 如果 LossG 里用到了 torch.nn.Module，建议 eval（可选）
        try:
            if hasattr(self.lossg, "clip_model"):
                self.lossg.clip_model.eval()
        except Exception:
            pass

    def load_image_triplets(
        self,
        pairs: List[Dict[str, str]]
    ) -> Tuple[List[Image.Image], List[Image.Image], List[Image.Image]]:
        """
        从 CSV 读取的 pairs（包含 'src','ref','gen' 键）加载图像三元组：
        results = gen 图；structures = src 图；appearances = ref 图
        """
        results, structures, appearances = [], [], []

        for i, pair in enumerate(pairs):
            try:
                src_p = pair.get("src", "").strip()
                ref_p = pair.get("ref", "").strip()
                gen_p = pair.get("gen", "").strip()
                if not (src_p and ref_p and gen_p):
                    print(f"[WARN] Row {i+1}: missing path(s) -> {pair}")
                    continue

                # 使用 ImageProcessor.read_image() 并传入 target_size
                src_img = ImageProcessor.read_image(Path(src_p), self.target_size)
                ref_img = ImageProcessor.read_image(Path(ref_p), self.target_size)
                gen_img = ImageProcessor.read_image(Path(gen_p), self.target_size)

                # 对应关系（保持与你其它代码一致）
                structures.append(src_img)     # src -> structure
                appearances.append(ref_img)    # ref -> appearance
                results.append(gen_img)        # gen -> result

            except Exception as e:
                print(f"[ERROR] load_image_triplets: row {i+1} failed: {e}")
                continue

        print(f"[INFO] Successfully loaded {len(results)} image triplets")
        if self.target_size:
            print(f"[INFO] Images resized to: {self.target_size}")
        
        return results, structures, appearances

    def _to_float(self, x) -> float:
        """把返回值统一成 float"""
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(x, torch.Tensor):
            x = x.item()
        return float(x)

    def calculate_batch_metrics(
        self,
        results: List[Image.Image],
        structures: List[Image.Image],
        appearances: List[Image.Image],
        pairs: List[Dict[str, str]],  # 添加pairs参数以获取landmarks路径
        batch_size: int = 16
    ) -> Dict[str, List[float]]:
        """
        逐对计算
        - self_sim:  DINO 自相似矩阵 MSE（结构一致性）
        - dino_i:    DINO(v1 compatible) CLS 相似度
        - clip_i:    CLIP(official compatible) 图像相似度
        - bg_mse:    背景区域MSE
        返回每对样本各自的分数列表
        """
        assert len(results) == len(structures) == len(appearances) == len(pairs), "triplets length mismatch"

        all_self_sims, all_dino_is, all_clip_is, all_bg_mses = [], [], [], []
        n = len(results)

        for start in tqdm(range(0, n, batch_size), desc="Computing DINO/CLIP/BG-MSE (per-pair)"):
            end = min(start + batch_size, n)
            for i, (r, s, a) in enumerate(zip(results[start:end], structures[start:end], appearances[start:end])):
                pair_idx = start + i
                pair = pairs[pair_idx]
                
                try:
                    # 单对输入，返回单值
                    ss = self._to_float(self.lossg.calculate_self_sim_loss([s], [r]))
                except Exception as e:
                    print(f"[WARN] self_sim failed: {e}")
                    ss = float("nan")

                try:
                    di = self._to_float(self.lossg.calculate_dino_i_loss_compatible([r], [a]))
                except Exception as e:
                    print(f"[WARN] dino_i failed: {e}")
                    di = float("nan")

                try:
                    ci = self._to_float(self.lossg.calculate_clip_i_loss_compatible([r], [a]))
                except Exception as e:
                    print(f"[WARN] clip_i failed: {e}")
                    ci = float("nan")

                # 计算背景MSE
                try:
                    src_path = Path(pair["src"])
                    gen_path = Path(pair["gen"])
                    src_lmk_path = Path(pair.get("src_lmk", "")) if pair.get("src_lmk", "").strip() else None
                    
                    bg_mse = self.bg_mse_calc.calculate_background_mse(src_path, gen_path, src_lmk_path)
                except Exception as e:
                    print(f"[WARN] bg_mse failed: {e}")
                    bg_mse = float("nan")

                all_self_sims.append(ss)
                all_dino_is.append(di)
                all_clip_is.append(ci)
                all_bg_mses.append(bg_mse)

        return {
            "self_sim": all_self_sims,
            "dino_i": all_dino_is,
            "clip_i": all_clip_is,
            "bg_mse": all_bg_mses
        }

    def validate_and_filter_images(self, directory: str) -> List[str]:
        """验证并返回有效图像文件路径列表"""
        valid_files = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                filepath = os.path.join(directory, filename)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                    valid_files.append(filepath)
                except:
                    continue  # 跳过无效图像
        
        return valid_files

    def calculate_fid_kid(self, ref_paths: List[str], gen_paths: List[str], kid_subset_size: Optional[int] = None) -> Dict[str, float]:
        """
        计算FID和KID指标，基于提供的图像路径列表
        
        Args:
            ref_paths: 参考图像路径列表
            gen_paths: 生成图像路径列表
            kid_subset_size: KID子集大小
        """
        if not calculate_metrics:
            return {'fid': float('nan'), 'kid': float('nan')}
        
        try:
            # 验证输入
            if not ref_paths or not gen_paths:
                raise ValueError("Empty image path lists")
            
            # 过滤有效图像路径
            ref_valid = []
            gen_valid = []
            
            for path in ref_paths:
                if os.path.exists(path):
                    try:
                        with Image.open(path) as img:
                            img.verify()
                        ref_valid.append(path)
                    except:
                        continue
            
            for path in gen_paths:
                if os.path.exists(path):
                    try:
                        with Image.open(path) as img:
                            img.verify()
                        gen_valid.append(path)
                    except:
                        continue
            
            if not ref_valid or not gen_valid:
                raise ValueError("No valid images found")
            
            min_samples = min(len(ref_valid), len(gen_valid))
            if kid_subset_size is None:
                kid_subset_size = min(100, min_samples)
            
            print(f"[INFO] FID/KID: ref={len(ref_valid)}, gen={len(gen_valid)}, kid_subset_size={kid_subset_size}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_ref_dir = os.path.join(temp_dir, "ref")
                temp_gen_dir = os.path.join(temp_dir, "gen")
                os.makedirs(temp_ref_dir)
                os.makedirs(temp_gen_dir)
                
                # 复制参考图像
                for i, src_path in enumerate(ref_valid[:min_samples]):
                    dst_path = os.path.join(temp_ref_dir, f"ref_{i:06d}.jpg")
                    try:
                        img = ImageProcessor.read_image(Path(src_path), self.target_size)
                        img.save(dst_path, 'JPEG', quality=95)
                    except Exception as e:
                        print(f"[WARN] Failed to process ref image {src_path}: {e}")
                        continue
                
                # 复制生成图像
                for i, src_path in enumerate(gen_valid[:min_samples]):
                    dst_path = os.path.join(temp_gen_dir, f"gen_{i:06d}.jpg")
                    try:
                        img = ImageProcessor.read_image(Path(src_path), self.target_size)
                        img.save(dst_path, 'JPEG', quality=95)
                    except Exception as e:
                        print(f"[WARN] Failed to process gen image {src_path}: {e}")
                        continue
                
                # 检查是否有足够的图像
                ref_count = len([f for f in os.listdir(temp_ref_dir) if f.endswith('.jpg')])
                gen_count = len([f for f in os.listdir(temp_gen_dir) if f.endswith('.jpg')])
                
                if ref_count == 0 or gen_count == 0:
                    raise ValueError("No valid images after processing")
                
                print(f"[INFO] Processing FID/KID with {ref_count} ref and {gen_count} gen images")
                
                # 计算指标
                metrics = calculate_metrics(
                    input1=temp_ref_dir,
                    input2=temp_gen_dir,
                    fid=True,
                    kid=True,
                    kid_subset_size=min(kid_subset_size, min(ref_count, gen_count)),
                    device=self.device,
                    batch_size=32,
                    verbose=False
                )
            
            return {
                'fid': float(metrics.get('frechet_inception_distance', float('nan'))),
                'kid': float(metrics.get('kernel_inception_distance_mean', float('nan')))
            }
            
        except Exception as e:
            print(f"[ERROR] FID/KID calculation failed: {e}")
            return {'fid': float('nan'), 'kid': float('nan')}

class ResultsWriter:
    """结果写入器"""
    
    @staticmethod
    def safe_format_value(value) -> str:
        """安全地格式化值为字符串"""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return "NaN"
            return f"{value:.6f}"
        return str(value)
    
    @staticmethod
    def write_to_csv(per_pair_results: List[Dict], summary_metrics: Dict, output_path: Path):
        """将结果写入CSV文件"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 添加bg_mse字段
        fieldnames = ["gen_filename", "self_sim", "dino_i", "clip_i", "bg_mse", "face_id"]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # 写入每对结果
            for result in per_pair_results:
                row = {}
                for field in fieldnames:
                    value = result.get(field, float("nan"))
                    row[field] = ResultsWriter.safe_format_value(value)
                writer.writerow(row)
            
            # 写入摘要
            writer.writerow({field: "" for field in fieldnames})
            writer.writerow({"gen_filename": "=== SUMMARY ===", **{f: "" for f in fieldnames[1:]}})
            
            summary_items = [
                ("Total pairs", summary_metrics.get("count", 0)),
                ("Self-sim mean", summary_metrics.get("self_sim", float("nan"))),
                ("DINO-I mean", summary_metrics.get("dino_i", float("nan"))),
                ("CLIP-I mean", summary_metrics.get("clip_i", float("nan"))),
                ("BG-MSE mean", summary_metrics.get("bg_mse", float("nan"))),
                ("Face-ID mean", summary_metrics.get("face_id", float("nan"))),
                ("FID", summary_metrics.get("fid", float("nan"))),
                ("KID", summary_metrics.get("kid", float("nan")))
            ]
            
            for name, value in summary_items:
                formatted_value = ResultsWriter.safe_format_value(value)
                writer.writerow({"gen_filename": name, "self_sim": formatted_value, **{f: "" for f in fieldnames[2:]}})
        
        print(f"[INFO] Results saved to: {output_path}")


def load_pairs_from_csv(csv_path: Path) -> List[Dict[str, str]]:
    """从CSV文件加载图像对"""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pairs = [row for row in reader]
    
    print(f"[INFO] Loaded {len(pairs)} pairs from {csv_path}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="综合化妆迁移指标评估")
    
    parser.add_argument("--pairs_csv", default="MagicMakeup/metrics/example/MagicMakeup.csv", 
                   type=str, help="包含src,ref,gen,src_lmk,gen_lmk列的CSV文件")
    parser.add_argument("--out_csv", type=str, default="MagicMakeup/metrics/example/evalue.csv", help="输出CSV文件路径")

    parser.add_argument("--kid_subset_size", type=int, default=None, help="KID子集大小（默认自动调整）")
    # 修改target_size参数定义
    parser.add_argument("--target_size", type=int, nargs=2, default=[1024, 1024], help="目标图像尺寸 [width, height]")
    
    # Face-ID相关
    parser.add_argument("--recognition_model_id", type=str, default="evaluate/cvlface/adaface_vit_base_kprpe_webface12m",
                    help="人脸识别模型路径")
    parser.add_argument("--aligner_id", type=str,default="evaluate/cvlface/DFA_mobilenet",
                     help="人脸对齐模型路径")

    # 处理参数
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    parser.add_argument("--skip_face_id", action="store_true", help="跳过Face-ID计算")
    
    args = parser.parse_args()
    
    # 检查输入文件
    pairs_csv_path = Path(args.pairs_csv)
    if not pairs_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {pairs_csv_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # 处理 target_size 参数
    target_size = None
    if args.target_size:
        if isinstance(args.target_size, list) and len(args.target_size) == 2:
            target_size = tuple(args.target_size)
            print(f"[INFO] Target image size: {target_size}")
        else:
            print(f"[WARN] Invalid target_size format: {args.target_size}, using original sizes")
    
    # 加载数据
    pairs = load_pairs_from_csv(pairs_csv_path)
    
    # 验证CSV格式
    required_columns = ['src', 'ref', 'gen']
    if pairs and not all(col in pairs[0] for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # 检查是否有landmarks列
    has_landmarks = pairs and 'src_lmk' in pairs[0] and 'gen_lmk' in pairs[0]
    if not has_landmarks:
        print("[WARN] CSV does not contain 'src_lmk' and 'gen_lmk' columns, Face-ID and BG-MSE will be skipped")
    
    # 初始化计算器时传入 target_size
    metrics_calc = MetricsCalculator(device, target_size)
    
    # 加载图像三元组
    results, structures, appearances = metrics_calc.load_image_triplets(pairs)
    
    # 计算DINO/CLIP/背景MSE指标 - 传入pairs参数
    batch_metrics = metrics_calc.calculate_batch_metrics(results, structures, appearances, pairs, args.batch_size)
    
    # 初始化Face-ID计算器（如果可用且未跳过）
    face_id_calc = None
    if not args.skip_face_id and has_landmarks and load_model_from_local_path and args.recognition_model_id and args.aligner_id:
        try:
            print("[INFO] Loading Face-ID models...")
            # import pdb;pdb.set_trace()
            fr_model = load_model_from_local_path(args.recognition_model_id).to(device)
            aligner = load_model_from_local_path(args.aligner_id).to(device)

            mask_gen = FaceContourMaskGenerator()
            # 传入target_size
            face_id_calc = FaceIDCalculator(fr_model, aligner, mask_gen, device, target_size)
            print("[INFO] Face-ID calculator initialized")
        except Exception as e:
            print(f"[WARN] Failed to load Face-ID models: {e}")
    
    # 计算Face-ID指标
    face_id_scores = []
    if face_id_calc and has_landmarks:
        print("[INFO] Computing Face-ID scores...")
        for pair in tqdm(pairs, desc="Computing Face-ID"):
            src_path = Path(pair["src"])
            gen_path = Path(pair["gen"])
            
            # 从CSV直接读取landmarks路径
            src_lmk_path = pair.get("src_lmk", "").strip()
            gen_lmk_path = pair.get("gen_lmk", "").strip()
            
            # 转换为Path对象（如果路径非空）
            src_lmk = Path(src_lmk_path) if src_lmk_path else None
            gen_lmk = Path(gen_lmk_path) if gen_lmk_path else None
            
            feat_src = face_id_calc.extract_feature(src_path, src_lmk)
            feat_gen = face_id_calc.extract_feature(gen_path, gen_lmk)
            
            face_id_scores.append(face_id_calc.calculate_similarity(feat_src, feat_gen))
    else:
        if args.skip_face_id:
            print("[INFO] Face-ID calculation skipped by user")
        elif not has_landmarks:
            print("[INFO] Face-ID calculation skipped: no landmarks in CSV")
        face_id_scores = [float("nan")] * len(pairs)
    
    # 计算FID/KID
    print("[INFO] Preparing FID/KID calculation from CSV pairs...")
    
    # 从CSV中提取ref和gen图像路径
    ref_paths = []
    gen_paths = []
    
    for pair in pairs:
        ref_path = pair.get("ref", "").strip()
        gen_path = pair.get("gen", "").strip()
        
        if ref_path and gen_path:
            ref_paths.append(ref_path)
            gen_paths.append(gen_path)
    
    print(f"[INFO] Found {len(ref_paths)} ref images and {len(gen_paths)} gen images from CSV")
    
    # 计算FID/KID
    if ref_paths and gen_paths:
        fid_kid_metrics = metrics_calc.calculate_fid_kid(
            ref_paths, 
            gen_paths, 
            args.kid_subset_size
        )
    else:
        print("[WARN] No valid ref/gen paths found in CSV")
        fid_kid_metrics = {'fid': float('nan'), 'kid': float('nan')}
    
    # 整理结果 - 添加bg_mse字段
    per_pair_results = []
    for i, pair in enumerate(pairs):
        per_pair_results.append({
            "gen_filename": Path(pair["gen"]).name,
            "self_sim": batch_metrics["self_sim"][i] if i < len(batch_metrics["self_sim"]) else float("nan"),
            "dino_i": batch_metrics["dino_i"][i] if i < len(batch_metrics["dino_i"]) else float("nan"),
            "clip_i": batch_metrics["clip_i"][i] if i < len(batch_metrics["clip_i"]) else float("nan"),
            "bg_mse": batch_metrics["bg_mse"][i] if i < len(batch_metrics["bg_mse"]) else float("nan"),
            "face_id": face_id_scores[i] if i < len(face_id_scores) else float("nan")
        })
    
    # 计算摘要统计 - 添加bg_mse
    def safe_mean(values):
        valid_values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
        return sum(valid_values) / len(valid_values) if valid_values else float("nan")
    
    summary_metrics = {
        "count": len(per_pair_results),
        "self_sim": safe_mean(batch_metrics["self_sim"]),
        "dino_i": safe_mean(batch_metrics["dino_i"]),
        "clip_i": safe_mean(batch_metrics["clip_i"]),
        "bg_mse": safe_mean(batch_metrics["bg_mse"]),
        "face_id": safe_mean(face_id_scores),
        "fid": fid_kid_metrics["fid"],
        "kid": fid_kid_metrics["kid"]
    }
    
    # 写入结果
    ResultsWriter.write_to_csv(per_pair_results, summary_metrics, Path(args.out_csv))
    
    # 打印摘要
    print("\n=== Results Summary ===")
    for key, value in summary_metrics.items():
        formatted_value = ResultsWriter.safe_format_value(value)
        print(f"{key}: {formatted_value}")


if __name__ == "__main__":
    main()
