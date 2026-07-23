#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter & center-crop faces to 1024×1024 (no rotation).

1. FaceDetector → choose max-score face → filter by face size ratio.
2. FaceLandmarker → eyeBlinkLeft/Right > 0.5  → skip (closed eyes).
3. Expand bbox by --expand (default 0.3), crop, resize 1024².
4. Save:
   - *_crop1024.png            (RGB)
   - *_5pts.npy                (5,2)  float32   [x,y] in crop image
   - *_landmarks.npy           (468,3) float32  [x,y,z] in crop image
   - log.tsv                   summary
5. Delete original image if processing is successful.
"""

import cv2, numpy as np, argparse, os
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter

# 5-point indexes in FaceMesh
IDX_5PT = [33, 263, 1, 61, 291]   # eyeL, eyeR, nose, mouthL, mouthR

def get_blink(blend_list):
    left = right = 0.0
    if blend_list:
        for c in blend_list:
            if c.category_name == 'eyeBlinkLeft':
                left = c.score
            elif c.category_name == 'eyeBlinkRight':
                right = c.score
    return left, right

def extract_landmarks(lms, w, h):
    """return xyz pixel coordinates, shape (468,3)"""
    arr = np.array([[lm.x * w, lm.y * h, lm.z] for lm in lms], dtype=np.float32)
    return arr

def crop_pad_coords(box, expand, W, H):
    x0, y0, x1, y1 = box         # pixel
    dx, dy = (x1-x0)*expand, (y1-y0)*expand
    x0 = max(0,     int(x0 - dx))
    y0 = max(0,     int(y0 - dy))
    x1 = min(W - 1, int(x1 + dx))
    y1 = min(H - 1, int(y1 + dy))
    return x0, y0, x1, y1

def crop_square_pad_coords(box, expand, W, H):
    """裁剪为正方形区域，保持宽高比"""
    x0, y0, x1, y1 = box         # pixel
    
    # 扩展边界框
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2  # 中心点
    size = max(x1 - x0, y1 - y0)  # 取较大的边长
    size = size * (1 + 2 * expand)  # 扩展
    
    # 计算新的正方形边界
    new_x0 = max(0, int(cx - size / 2))
    new_y0 = max(0, int(cy - size / 2))
    new_x1 = min(W - 1, int(cx + size / 2))
    new_y1 = min(H - 1, int(cy + size / 2))
    
    # 调整为正方形（可能会因为边界限制而不完全是正方形）
    final_size = min(new_x1 - new_x0, new_y1 - new_y0)
    center_x, center_y = (new_x0 + new_x1) // 2, (new_y0 + new_y1) // 2
    
    final_x0 = max(0, center_x - final_size // 2)
    final_y0 = max(0, center_y - final_size // 2)
    final_x1 = min(W - 1, final_x0 + final_size)
    final_y1 = min(H - 1, final_y0 + final_size)
    
    return final_x0, final_y0, final_x1, final_y1

def get_face_ratio(box, W, H):
    """计算人脸框占整个图像的面积比例"""
    x0, y0, x1, y1 = box
    face_area = (x1 - x0) * (y1 - y0)
    image_area = W * H
    return face_area / image_area

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_model", default="MagicMakeup/mediapipe/blaze_face_short_range.tflite")
    ap.add_argument("--lmk_model", default="MagicMakeup/mediapipe/face_landmarker.task")
    ap.add_argument('--input_dir', default="MagicMakeup/example/makeup/image", help='输入目录，包含图像文件')
    ap.add_argument('--out_dir', default="MagicMakeup/example/makeup/image", help='输出目录，保存生成的结果')
    
    ap.add_argument("--expand", type=float, default=0.8, help="ROI expand ratio for Landmarker")
    ap.add_argument("--min_face_ratio", type=float, default=0.1, help="Minimum face area ratio to image area")
    ap.add_argument("--keep_subdirs", action="store_true")
    args = ap.parse_args()

    # Init models
    det_opt = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=args.det_model),
        min_detection_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE)
    detector = vision.FaceDetector.create_from_options(det_opt)

    lmk_opt = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=args.lmk_model),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE)
    landmarker = vision.FaceLandmarker.create_from_options(lmk_opt)

    in_dir  = Path(args.input_dir).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
    imgs = [p for p in in_dir.rglob('*') if p.suffix.lower() in exts]

    # 统计计数器
    stats = Counter()
    total_images = len(imgs)
    log_lines = []
    deleted_files = []
    
    for p in tqdm(imgs, desc='process'):
        success = False  # 标记是否成功处理
        try:
            # 先用OpenCV读取图像，检查是否为彩色图像
            img = cv2.imread(str(p))
            if img is None:
                log_lines.append(f"{p}\tERROR_LOADING\tImage could not be loaded\n")
                stats['ERROR_LOADING'] += 1
                continue
                
            # 检查通道数，确保是彩色图像
            if len(img.shape) < 3 or img.shape[2] != 3:
                log_lines.append(f"{p}\tNOT_RGB\tImage is not RGB format\n")
                stats['NOT_RGB'] += 1
                continue
                
            # 使用MediaPipe加载图像
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            H, W = mp_img.height, mp_img.width
        except Exception as e:
            log_lines.append(f"{p}\tERROR_LOADING\t{str(e)}\n")
            stats['ERROR_LOADING'] += 1
            continue

        try:
            det_res = detector.detect(mp_img)
            if not det_res.detections:
                log_lines.append(f"{p}\tNO_FACE\n")
                stats['NO_FACE'] += 1
                continue
            det0 = max(det_res.detections, key=lambda d: d.categories[0].score)

            # 获取人脸框
            box = det0.bounding_box
            x0, y0, x1, y1 = box.origin_x, box.origin_y, box.origin_x+box.width, box.origin_y+box.height
            
            # 计算人脸占比并筛选
            face_ratio = get_face_ratio((x0, y0, x1, y1), W, H)
            if face_ratio < args.min_face_ratio:
                log_lines.append(f"{p}\tFACE_TOO_SMALL\t{face_ratio:.4f}\n")
                stats['FACE_TOO_SMALL'] += 1
                continue

            lmk_res = landmarker.detect(mp_img)
            if not lmk_res.face_landmarks:
                log_lines.append(f"{p}\tNO_LMK\n")
                stats['NO_LMK'] += 1
                continue

            blinkL, blinkR = get_blink(lmk_res.face_blendshapes[0])
            if blinkL > 0.5 and blinkR > 0.5:        # closed eyes → skip
                log_lines.append(f"{p}\tCLOSED_EYES\n")
                stats['CLOSED_EYES'] += 1
                continue

            # 使用正方形裁剪
            x0, y0, x1, y1 = crop_square_pad_coords((x0, y0, x1, y1), args.expand, W, H)

            if x1 <= x0 or y1 <= y0:
                log_lines.append(f"{p}\tBAD_BOX\n")
                stats['BAD_BOX'] += 1
                continue

            crop = mp_img.numpy_view()[y0:y1, x0:x1]
            crop1024 = cv2.resize(crop, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)

            sx, sy = 1024.0/(x1-x0), 1024.0/(y1-y0)

            # landmarks
            lmk_pix = extract_landmarks(lmk_res.face_landmarks[0], W, H)         # (468,3)
            lmk_crop = lmk_pix.copy()
            lmk_crop[:,0] = (lmk_crop[:,0] - x0) * sx
            lmk_crop[:,1] = (lmk_crop[:,1] - y0) * sy

            # 5 pts
            pts5_crop = lmk_crop[IDX_5PT, :2]   # (5,2)

            # save paths
            rel = p.relative_to(in_dir)
            save_dir = out_dir/rel.parent if args.keep_subdirs else out_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            stem = rel.stem
            
            try:
                cv2.imwrite(str(save_dir/f"{stem}.png"), cv2.cvtColor(crop1024, cv2.COLOR_RGB2BGR))
                # np.save(str(save_dir/f"{stem}_5pts.npy"), pts5_crop.astype(np.float32))
                # np.save(str(save_dir/f"{stem}_landmarks.npy"), lmk_crop.astype(np.float32))
                log_lines.append(f"{p}\tOK\t{save_dir/f'{stem}'}\n")
                stats['OK'] += 1
                success = True  # 标记处理成功
            except Exception as e:
                log_lines.append(f"{p}\tSAVE_ERROR\t{str(e)}\n")
                stats['SAVE_ERROR'] += 1
                continue
                
        except Exception as e:
            log_lines.append(f"{p}\tPROCESS_ERROR\t{str(e)}\n")
            stats['PROCESS_ERROR'] += 1
            continue
            
        # 如果处理成功，删除原始图片
        # if success:
        #     try:
        #         os.remove(str(p))
        #         deleted_files.append(str(p))
        #     except Exception as e:
        #         log_lines.append(f"{p}\tDELETE_ERROR\t{str(e)}\n")
        #         stats['DELETE_ERROR'] += 1

    with open(out_dir/'log.tsv','w') as f:
        f.write("src\tstatus\tout_path\n")
        f.writelines(log_lines)
    
    # 保存已删除文件的列表
    with open(out_dir/'deleted_files.txt','w') as f:
        f.write("\n".join(deleted_files))
    
    # 打印详细的统计信息
    print("\n===== 处理统计 =====")
    print(f"总图片数: {total_images}")
    print(f"成功处理: {stats['OK']} ({stats['OK']/total_images*100:.1f}%)")
    print(f"成功删除: {len(deleted_files)} 个原始文件")
    if 'DELETE_ERROR' in stats:
        print(f"删除失败: {stats['DELETE_ERROR']} 个文件")
    
    print("\n失败原因统计:")
    for reason, count in sorted(stats.items()):
        if reason not in ['OK', 'DELETE_ERROR']:
            print(f"  - {reason}: {count} ({count/total_images*100:.1f}%)")
    
    print("\nCrop Done ✔")


if __name__ == "__main__":
    main()
