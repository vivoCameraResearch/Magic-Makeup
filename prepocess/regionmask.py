#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面部区域掩膜生成工具(基于MediaPipe实时检测)

功能：
1. 支持两种输入模式：
   - 目录模式：直接从目录读取所有图像文件
   - CSV模式：从CSV文件读取指定的图像文件列表
2. 使用MediaPipe实时检测关键点
3. 生成指定区域的掩膜：lip、eyes (brows+orbits合并)
4. 输出掩膜文件

用法示例：
    # 目录模式
    python regionmask.py --input_dir /path/to/input --output_dir /path/to/output --regions eyes
    
    # CSV模式
    python regionmask.py --csv_path data.csv --img_dir /path/to/images --output_dir /path/to/output --regions eyes
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional, Union, Set
import logging
import time
import mediapipe as mp
import re
import os
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FaceMaskProcessor")
import cv2
import numpy as np
from PIL import Image

def generate_mask_from_image(image_pil: Image.Image, 
                             region: str = 'eyes',
                             feather_eyes: bool = True, 
                             feather_px_ratio: float = 0.16) -> Image.Image:
    """
    从PIL图像生成指定区域的mask（最简洁版本）
    
    Args:
        image_pil: PIL Image (RGB)
        region: 区域名称 ('eyes', 'lip', 'face')
        feather_eyes: 是否对眼部mask进行羽化
        feather_px_ratio: 羽化比例
        
    Returns:
        mask_pil: PIL Image (L) 或 None
    """
    try:
        # PIL转OpenCV (RGB -> BGR)
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        h, w = image_np.shape[:2]
        
        # 创建检测器和生成器
        detector = MediaPipeFaceMeshDetector()
        generator = FaceMaskGenerator(
            feather_eyes=feather_eyes,
            feather_px_ratio=feather_px_ratio
        )
        
        # 检测关键点
        landmarks = detector.detect_landmarks(image_np)
        if landmarks is None:
            print(f"Warning: No face detected")
            return None
        
        # 生成mask
        masks = generator.generate_specified_masks(landmarks, w, h, {region})
        
        if region not in masks:
            print(f"Warning: Failed to generate mask for region '{region}'")
            return None
        
        mask_np = masks[region]
        
        # OpenCV转PIL
        mask_pil = Image.fromarray(mask_np).convert('L')
        return mask_pil
        
    except Exception as e:
        print(f"Error generating mask: {e}")
        import traceback
        traceback.print_exc()
        return None

# 使用示例
from PIL import Image



def is_image_file(filepath: str) -> bool:
    """检查文件是否为图像文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    return Path(filepath).suffix.lower() in image_extensions

def extract_base_name(filename: str) -> str:
    """
    简单提取基础名称：去除常见后缀
    1446_2338_mask.png -> 1446_2338
    """
    name = os.path.splitext(filename)[0]  # 去除扩展名
    # 去除常见后缀
    name = re.sub(r'_(mask|crop\d+.*|warp)$', '', name)
    return name

def normalize_save_path(img_path: str, save_path: str) -> str:
    """
    修改保存路径逻辑：保存为 base_name_mask.png 格式
    """
    save_path = save_path.strip()
    
    if save_path.endswith(os.sep) or os.path.isdir(save_path):
        # 如果save_path是目录，生成新的文件名
        original_filename = os.path.basename(img_path)
        base_name = extract_base_name(original_filename)  # 提取基础名称
        new_filename = f"{base_name}_mask.png"  # 添加_mask后缀
        
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
                if len(not_found_files) <= 5:  # 只显示前5个未找到的文件
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

class MediaPipeFaceMeshDetector:
    """MediaPipe人脸网格检测器"""
    
    def __init__(self, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初始化MediaPipe人脸网格检测器
        
        参数:
            max_num_faces: 最大检测人脸数量
            refine_landmarks: 是否细化关键点
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        检测人脸关键点
        
        参数:
            image: 输入图像 (BGR格式)
        
        返回:
            关键点数组 (478, 3) 或 None
        """
        # 转换为RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测人脸网格
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            # 获取第一个检测到的人脸
            face_landmarks = results.multi_face_landmarks[0]
            
            # 转换为像素坐标
            h, w = image.shape[:2]
            landmarks = []
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                landmarks.append([x, y, z])
            
            return np.array(landmarks, dtype=np.float32)
        
        return None
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


class FaceMaskGenerator:
    """面部区域掩膜生成器"""
    
    # ---------- 索引集合 ----------
    # 面部轮廓
    FACE_CONTOUR_INDICES = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
        148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
    ]

    # 眼眶区域（大）
    LARGE_LEFT_ORBIT_INDICES = [55, 65, 52, 53, 70, 139, 143, 111, 117, 118, 119, 120, 121, 128, 245, 193]
    LARGE_RIGHT_ORBIT_INDICES = [285, 295, 282, 283, 300, 368, 372, 340, 346, 347, 348, 349, 350, 357, 465, 417]
    
    # 眼睛区域
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
    
    # 嘴唇区域
    LIPS_INDICES = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
        0, 37, 39, 40, 185, 61, 
    ]
    
    # 眉毛区域
    LEFT_BROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53]
    RIGHT_BROW_INDICES = [300, 293, 334, 296, 336, 285, 295, 282, 283]
    
    # ---------- 配置 ----------
    DEFAULT_BLUR_KSIZE = 21
    DEFAULT_MASK_VAL = 255
    DEFAULT_BROW_DILATE_SIZE = 15
    
    # 支持的区域类型
    SUPPORTED_REGIONS = {'lip', 'eyes', 'face'}
    
    def __init__(self, 
                blur_ksize: int = DEFAULT_BLUR_KSIZE, 
                mask_val: int = DEFAULT_MASK_VAL,
                brow_dilate_size: int = DEFAULT_BROW_DILATE_SIZE,
                # 新增羽化参数
                feather_eyes: bool = True,
                feather_px_ratio: float = 0.16,
                inner_keep_ratio: float = 0.0):
        """
        初始化掩膜生成器
        
        参数:
            blur_ksize: 高斯模糊核大小
            mask_val: 掩膜填充值
            brow_dilate_size: 眉毛膨胀大小
            feather_eyes: 是否对eyes区域进行羽化
            feather_px_ratio: 羽化宽度比例，≈短边的比例
            inner_keep_ratio: 中心保护区比例，≈短边的比例
        """
        self.blur_ksize = blur_ksize
        self.mask_val = mask_val
        self.brow_dilate_size = brow_dilate_size
        self.feather_eyes = feather_eyes
        self.feather_px_ratio = feather_px_ratio
        self.inner_keep_ratio = inner_keep_ratio

    def _signed_distance(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        计算有符号距离：ROI 内为正、外为负。返回 float32，与 binary_mask 同尺寸。
        要求 binary_mask ∈ {0,1} （uint8/float 都可）
        """
        m = (binary_mask > 0).astype(np.uint8)
        d_in  = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
        d_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3).astype(np.float32)
        return d_in - d_out  # 内正外负

    def feather_binary_mask(self, binary_mask: np.ndarray,
                            inner_keep_px: int,
                            feather_px: int) -> np.ndarray:
        """
        外→内递增（out→in）：边界为0，向内在 feather_px 距离内平滑增长到1，中心保持1。
        ROI外始终为0。
        """
        sdist = self._signed_distance(binary_mask).astype(np.float32)  # 内正 外负
        w = np.zeros_like(sdist, dtype=np.float32)

        # 只在 ROI 内做处理
        inside = sdist > 0

        # 目标：sdist=0(边界) -> 0； sdist>=feather_px -> 1；中间用余弦增长
        reach_px = max(1, feather_px)  # 到达1所需的内侧距离
        t = np.clip(sdist / reach_px, 0.0, 1.0)  # 0..1
        w_inside = 0.5 * (1.0 - np.cos(np.pi * t))  # 0->1 单调递增

        w[inside] = w_inside[inside]
        w[~inside] = 0.0
        return np.clip(w, 0.0, 1.0)

    def pts_from_indices(self, indices: List[int], landmarks: np.ndarray) -> np.ndarray:
        """将索引列表转换为整数像素坐标 (N,2)"""
        pts = []
        for idx in indices:
            if idx < len(landmarks):
                x, y = landmarks[idx][:2]
                pts.append([int(x), int(y)])
        return np.array(pts, dtype=np.int32)
    
    def fill_poly(self, shape: Tuple[int, int], pts_list: List[np.ndarray]) -> np.ndarray:
        """在空白遮罩上填充多边形。返回 uint8 二值图 (0/255)"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for pts in pts_list:
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], self.mask_val)
        return mask
    
    def soften(self, mask: np.ndarray, ksize: Optional[int] = None) -> np.ndarray:
        """高斯模糊软化边缘，保持 uint8"""
        if ksize is None:
            ksize = self.blur_ksize
        blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        return blurred
    
    def expand_mask(self, mask: np.ndarray, dilate_size: Optional[int] = None, 
                    blur_size: Optional[int] = None) -> np.ndarray:
        """扩展掩膜，先膨胀再软化"""
        if dilate_size is None:
            dilate_size = self.brow_dilate_size
        if blur_size is None:
            blur_size = self.blur_ksize
            
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return cv2.GaussianBlur(dilated, (blur_size, blur_size), 0)
    
    def generate_lip_mask(self, landmarks: np.ndarray, w: int, h: int) -> np.ndarray:
        """生成嘴唇掩膜"""
        lip_pts = self.pts_from_indices(self.LIPS_INDICES, landmarks)
        lip_mask = self.fill_poly((h, w), [lip_pts])
        return self.soften(lip_mask)

    def generate_eyes_mask(self, landmarks: np.ndarray, w: int, h: int) -> np.ndarray:
        """生成眼部掩膜（眉毛+眼眶，去除眼睛），支持羽化"""
        
        # 生成眼眶掩膜（二值，用于羽化）
        left_orbit_pts = self.pts_from_indices(self.LARGE_LEFT_ORBIT_INDICES, landmarks)
        right_orbit_pts = self.pts_from_indices(self.LARGE_RIGHT_ORBIT_INDICES, landmarks)
        
        # 创建0/1二值掩膜，更适合羽化处理
        orbit_bin = np.zeros((h, w), dtype=np.uint8)
        eye_bin = np.zeros((h, w), dtype=np.uint8)
        
        # 填充眼眶区域
        cv2.fillPoly(orbit_bin, [left_orbit_pts, right_orbit_pts], 1)
        
        # 填充眼睛内部区域
        left_eye_pts = self.pts_from_indices(self.LEFT_EYE_INDICES, landmarks)
        right_eye_pts = self.pts_from_indices(self.RIGHT_EYE_INDICES, landmarks)
        cv2.fillPoly(eye_bin, [left_eye_pts, right_eye_pts], 1)
        
        # 创建环形区域（眼眶-眼睛内部）
        ring_bin = cv2.subtract(orbit_bin, eye_bin)
        
        if self.feather_eyes:
            # 基于眼眶整体的bbox估算羽化参数
            all_eye_pts = np.vstack([left_orbit_pts, right_orbit_pts])
            min_xy = all_eye_pts.min(axis=0)
            max_xy = all_eye_pts.max(axis=0)
            short_edge = max(1, min(max_xy[0]-min_xy[0], max_xy[1]-min_xy[1]))
            
            inner_keep_px = max(0, int(short_edge * self.inner_keep_ratio))
            feather_px = max(1, int(short_edge * self.feather_px_ratio))
            
            # 只对外边界羽化，内圈（靠眼白）保持立刻为0
            outer_w = self.feather_binary_mask(orbit_bin, inner_keep_px=0, feather_px=feather_px)
            eye_w = outer_w * (1 - eye_bin)  # 仅外侧羽化，内圈直接挖空
            
            # 转换为0-255范围
            orbits_mask = (eye_w * 255).astype(np.uint8)
        else:
            # 不羽化，使用原有的软化方法
            orbits_mask = (ring_bin * 255).astype(np.uint8)
            orbits_mask = self.soften(orbits_mask)
        
        return orbits_mask
    
    def generate_face_mask(self, landmarks: np.ndarray, w: int, h: int, 
                      lip_mask: Optional[np.ndarray] = None, 
                      eyes_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """生成完整面部掩膜（包含所有面部区域）"""
        # 生成面部轮廓掩膜
        face_contour_pts = self.pts_from_indices(self.FACE_CONTOUR_INDICES, landmarks)
        face_mask = self.fill_poly((h, w), [face_contour_pts])
        return self.soften(face_mask)
    
    def generate_specified_masks(self, landmarks: np.ndarray, w: int, h: int, 
                               regions: Set[str]) -> Dict[str, np.ndarray]:
        """
        生成指定区域的掩膜
        
        参数:
            landmarks: 关键点数组 (478, 3)
            w, h: 图像宽高
            regions: 要生成的区域集合，可包含 'lip', 'eyes', 'face'
        
        返回:
            包含指定区域掩膜的字典
        """
        # 验证区域参数
        invalid_regions = regions - self.SUPPORTED_REGIONS
        if invalid_regions:
            raise ValueError(f"不支持的区域: {invalid_regions}. 支持的区域: {self.SUPPORTED_REGIONS}")
        
        masks = {}
        
        # 生成嘴唇掩膜（如果需要）
        lip_mask = None
        if 'lip' in regions:
            lip_mask = self.generate_lip_mask(landmarks, w, h)
            masks['lip'] = lip_mask
        
        # 生成眼部掩膜（如果需要）
        eyes_mask = None
        if 'eyes' in regions:
            eyes_mask = self.generate_eyes_mask(landmarks, w, h)
            masks['eyes'] = eyes_mask
        
        # 生成面部掩膜（如果需要）
        if 'face' in regions:
            # 如果需要生成face掩膜但还没有生成lip或eyes掩膜，则临时生成它们用于排除
            temp_lip_mask = lip_mask
            temp_eyes_mask = eyes_mask
            
            if temp_lip_mask is None:
                temp_lip_mask = self.generate_lip_mask(landmarks, w, h)
            if temp_eyes_mask is None:
                temp_eyes_mask = self.generate_eyes_mask(landmarks, w, h)
            
            face_mask = self.generate_face_mask(landmarks, w, h, temp_lip_mask, temp_eyes_mask)
            masks['face'] = face_mask
        
        return masks


class FaceMaskProcessor:
    """面部掩膜处理器"""
    
    def __init__(self, 
                output_dir: Union[str, Path],
                regions: Set[str],
                # 输入模式参数
                input_dir: Optional[Union[str, Path]] = None,
                csv_path: Optional[str] = None,
                img_dir: Optional[str] = None,
                filename_col: str = 'filename',
                label_col: str = 'label',
                target_label: str = 'face',
                # 其他参数
                max_images: Optional[int] = None,
                blur_ksize: int = 21,
                brow_dilate_size: int = 15,
                # 新增羽化参数
                feather_eyes: bool = True,
                feather_px_ratio: float = 0.16,
                inner_keep_ratio: float = 0.0):
        """
        初始化处理器
        
        参数:
            output_dir: 输出目录，保存生成的结果
            regions: 要生成的区域集合
            
            # 目录模式参数
            input_dir: 输入目录，包含图像文件（目录模式）
            
            # CSV模式参数
            csv_path: CSV文件路径（CSV模式）
            img_dir: 图像文件目录（CSV模式）
            filename_col: CSV中文件名列的名称
            label_col: CSV中标签列的名称
            target_label: 目标标签值
            
            # 其他参数
            max_images: 最大处理图像数量，None表示处理所有图像
            blur_ksize: 高斯模糊核大小
            brow_dilate_size: 眉毛膨胀大小
            feather_eyes: 是否对eyes区域进行羽化
            feather_px_ratio: 羽化宽度比例
            inner_keep_ratio: 中心保护区比例
        """
        self.output_dir = Path(output_dir)
        self.regions = regions
        self.max_images = max_images
        
        # 判断输入模式
        if csv_path is not None:
            # CSV模式
            self.mode = 'csv'
            self.csv_path = csv_path
            self.img_dir = img_dir
            self.filename_col = filename_col
            self.label_col = label_col
            self.target_label = target_label
            
            if img_dir is None:
                raise ValueError("CSV模式下必须提供 img_dir 参数")
            
            logger.info(f"使用CSV模式: {csv_path}")
            logger.info(f"图像目录: {img_dir}")
            logger.info(f"文件名列: {filename_col}, 标签列: {label_col}, 目标标签: {target_label}")
            
        elif input_dir is not None:
            # 目录模式
            self.mode = 'directory'
            self.input_dir = Path(input_dir)
            
            logger.info(f"使用目录模式: {input_dir}")
            
        else:
            raise ValueError("必须提供 input_dir（目录模式）或 csv_path+img_dir（CSV模式）")
        
        # 创建MediaPipe检测器
        self.detector = MediaPipeFaceMeshDetector()
        
        # 创建掩膜生成器（传入羽化参数）
        self.generator = FaceMaskGenerator(
            blur_ksize=blur_ksize,
            brow_dilate_size=brow_dilate_size,
            feather_eyes=feather_eyes,
            feather_px_ratio=feather_px_ratio,
            inner_keep_ratio=inner_keep_ratio
        )
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 根据regions参数打印不同的提示信息
        if self.regions == {'eyes'}:
            feather_status = "启用羽化" if feather_eyes else "不使用羽化"
            logger.info(f"检测到 'eyes' 模式：将生成眼部区域掩膜（{feather_status}）。")
        else:
            logger.info(f"将生成以下区域的掩膜: {', '.join(sorted(self.regions))}")
    
    def get_image_files(self) -> List[str]:
        """
        获取图像文件列表（根据模式自动选择方法）
        
        返回:
            图像文件路径列表
        """
        if self.mode == 'csv':
            # CSV模式：从CSV文件读取
            image_files = collect_images_from_csv(
                csv_path=self.csv_path,
                img_dir=self.img_dir,
                filename_col=self.filename_col,
                label_col=self.label_col,
                target_label=self.target_label
            )
        else:
            # 目录模式：扫描目录
            image_files = self._get_image_files_from_directory()
            # 转换Path对象为字符串
            image_files = [str(f) for f in image_files]
        
        # 限制数量
        if self.max_images is not None and len(image_files) > self.max_images:
            image_files = image_files[:self.max_images]
            logger.info(f"限制处理图像数量为: {self.max_images}")
        
        return image_files
    
    def _get_image_files_from_directory(self, extensions: List[str] = None) -> List[Path]:
        """
        从目录获取图像文件（原有逻辑）
        
        参数:
            extensions: 支持的文件扩展名列表
        
        返回:
            图像文件路径列表
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        image_files = []
        for ext in extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        # 排序
        image_files.sort()
        
        return image_files

    def process_image(self, image_path: str) -> bool:
        """
        处理单个图像
        
        参数:
            image_path: 图像路径（字符串）
        
        返回:
            处理是否成功
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return False
            
            h, w = image.shape[:2]
            
            # 使用新的保存路径逻辑
            output_file = normalize_save_path(image_path, str(self.output_dir))
            
            # 检查输出文件是否已存在
            if os.path.exists(output_file):
                base_name = extract_base_name(os.path.basename(image_path))
                logger.info(f"跳过已存在的文件: {base_name}_mask.png")
                return True
            
            # 使用MediaPipe检测关键点
            landmarks = self.detector.detect_landmarks(image)
            if landmarks is None:
                logger.warning(f"未检测到人脸关键点: {os.path.basename(image_path)}")
                return False
            
            final_mask = None
            
            # ---------- 核心修改：处理'eyes'标签的特殊逻辑 ----------
            if self.regions == {'eyes'}:
                final_mask = self.generator.generate_eyes_mask(landmarks, w, h)
            else:
                # 保持原有逻辑
                # 生成指定区域的掩膜
                masks = self.generator.generate_specified_masks(landmarks, w, h, self.regions)
                
                # 合并多个区域的掩膜（如果有多个）
                if len(masks) == 1:
                    final_mask = list(masks.values())[0]
                else:
                    # 合并多个掩膜
                    final_mask = np.zeros_like(list(masks.values())[0])
                    for mask in masks.values():
                        final_mask = np.maximum(final_mask, mask)
            
            # 保存结果
            if final_mask is not None:
                cv2.imwrite(output_file, final_mask)
                base_name = extract_base_name(os.path.basename(image_path))
                logger.info(f"成功处理: {os.path.basename(image_path)} -> {base_name}_mask.png")
                return True
            else:
                logger.error(f"未能生成最终掩膜: {os.path.basename(image_path)}")
                return False
            
        except Exception as e:
            logger.error(f"处理图像失败 {os.path.basename(image_path)}: {str(e)}")
            return False
    
    def process_directory(self) -> None:
        """处理所有图像"""
        image_files = self.get_image_files()
        total = len(image_files)
        
        if total == 0:
            if self.mode == 'csv':
                logger.warning(f"没有找到符合条件的图像文件")
            else:
                logger.warning(f"在目录 {self.input_dir} 中没有找到图像文件")
            return
        
        logger.info(f"找到 {total} 个图像文件")
        
        processed = 0
        failed = 0
        start_time = time.time()
        
        for i, image_path in enumerate(image_files):
            if self.process_image(image_path):
                processed += 1
            else:
                failed += 1
            
            # 显示进度
            if (i + 1) % 10 == 0 or i == total - 1:
                elapsed = time.time() - start_time
                logger.info(f"进度: {i+1}/{total} ({(i+1)/total*100:.1f}%) "
                           f"- 已处理: {processed}, 失败: {failed} "
                           f"- 用时: {elapsed:.1f}秒")
        
        # 显示最终统计信息
        total_time = time.time() - start_time
        logger.info(f"处理完成! 总共: {total}, 成功: {processed}, 失败: {failed}")
        logger.info(f"总用时: {total_time:.1f}秒, 平均每张: {total_time/max(processed, 1):.3f}秒")

def main():
    parser = argparse.ArgumentParser(description='面部区域掩膜生成工具（基于MediaPipe实时检测）')
    
    # 输出参数
    parser.add_argument('--output_dir', 
                        default="MagicMakeup/example/makeup/mask/eyes", 
                       help='输出目录，保存生成的掩膜')
    # 输入模式参数
    parser.add_argument('--input_dir', 
                      default="MagicMakeup/example/makeup/image", 
                       help='输入目录，包含图像文件（目录模式）')
    parser.add_argument('--csv_path', 
                       default=None, 
                       help='CSV文件路径，包含文件名和标签信息（CSV模式，提供此参数时将忽略input_dir）')
    
    # CSV模式专用参数
    parser.add_argument('--img_dir', 
                       default="MagicMakeup/example/makeup/image",
                       help='图像文件目录（CSV模式时必需）')
    parser.add_argument('--filename_col', default='nomakeup', help='CSV中文件名列的名称')
    parser.add_argument('--label_col', default='label', help='CSV中标签列的名称')
    parser.add_argument('--target_label', default='eyes', help='目标标签值，用于筛选要处理的图像')
    
    # 通用参数
    parser.add_argument('--max_images', type=int, default=None, help='最大处理图像数量，不指定则处理所有图像')
    
    # 区域选择参数
    parser.add_argument('--regions', nargs='+', 
                       choices=['lip', 'eyes', 'face'], 
                       default=['eyes'],
                       help='要生成的区域，可选: lip, eyes, face。可以指定多个区域')
    
    # 处理参数
    parser.add_argument('--blur_ksize', type=int, default=21, help='高斯模糊核大小')
    parser.add_argument('--brow_dilate', type=int, default=15, help='眉毛膨胀大小')
    
    # 羽化参数
    parser.add_argument('--feather_eyes', action='store_true', default=True, help='是否对eyes区域进行羽化')
    parser.add_argument('--no_feather_eyes', dest='feather_eyes', action='store_false', help='禁用eyes区域羽化')
    parser.add_argument('--feather_px_ratio', type=float, default=0.20, help='羽化宽度比例（相对于眼部区域短边）')
    parser.add_argument('--inner_keep_ratio', type=float, default=0.0, help='中心保护区比例（相对于眼部区域短边）')
    
    args = parser.parse_args()
    
    # -----------------------------------批量处理-------------------------------------------
    # 判断使用模式：如果提供了csv_path就用CSV模式，否则用目录模式
    if args.csv_path is not None:
        # CSV模式验证
        if not args.img_dir:
            parser.error("使用CSV模式时必须同时提供 --img_dir 参数")
        
        # 创建CSV模式处理器
        processor = FaceMaskProcessor(
            output_dir=args.output_dir,
            regions=set(args.regions),
            csv_path=args.csv_path,
            img_dir=args.img_dir,
            filename_col=args.filename_col,
            label_col=args.label_col,
            target_label=args.target_label,
            max_images=args.max_images,
            blur_ksize=args.blur_ksize,
            brow_dilate_size=args.brow_dilate,
            feather_eyes=args.feather_eyes,
            feather_px_ratio=args.feather_px_ratio,
            inner_keep_ratio=args.inner_keep_ratio
        )
    else:
        # 目录模式（默认）
        processor = FaceMaskProcessor(
            output_dir=args.output_dir,
            regions=set(args.regions),
            input_dir=args.input_dir,
            max_images=args.max_images,
            blur_ksize=args.blur_ksize,
            brow_dilate_size=args.brow_dilate,
            feather_eyes=args.feather_eyes,
            feather_px_ratio=args.feather_px_ratio,
            inner_keep_ratio=args.inner_keep_ratio
        )
    
    # 处理图像
    processor.process_directory()

#    # # ---------------------单张图像----------------------------------------------------------------
#     # # 加载图像
#     image = Image.open("MagicMakeup/example/makeup/image/0001.png")

#     # 生成eyes mask
#     eyes_mask = generate_mask_from_image(image, region='eyes')

#     # 生成lip mask
#     lip_mask = generate_mask_from_image(image, region='lip')

#     # 保存mask
#     if eyes_mask:
#         eyes_mask.save("MagicMakeup/example/makeup/mask/eyes/0001_mask.png")
#         lip_mask.save("MagicMakeup/example/makeup/mask/lip/0001_mask.png")

if __name__ == "__main__":
    main()