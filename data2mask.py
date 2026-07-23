import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import json
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF

# 导入prompt构建函数
from prompt import build_prompt

class ImageNorm:
    def __call__(self, img_cond):
        return img_cond * 2 - 1.0

class MakeupDatasetWithMask(Dataset):
    def __init__(self, csv_path, makeup_folder, nomakeup_folder, reference_folder, 
                 ref1_mask_folder=None, ref2_mask_folder=None, label_col='label', mask_col='mask', 
                 default_label='all', flip_prob=0.5, fallback_prompt="", 
                 exclude_labels=['eyes,lip', 'face']):
        """
        带双mask版本的MakeupDataset（支持缺失mask）
        
        参数说明:
        - ref1_mask_folder: ref1的mask文件夹，可为None
        - ref2_mask_folder: ref2的mask文件夹，可为None
        - mask_col: CSV中的mask列名，如果ref1_mask_folder或ref2_mask_folder为None，此列可不存在
        - exclude_labels: 要排除的标签列表，例如 ['eyes,lip', 'face']
        - 其他参数与原版本相同
        """
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        original_count = len(df)

        # 检查是否需要mask列
        need_mask_col = ref1_mask_folder is not None or ref2_mask_folder is not None
        
        # 过滤掉包含空值的基本列
        required_cols = ['makeup', 'nomakeup', 'reference']
        if need_mask_col and mask_col in df.columns:
            required_cols.append(mask_col)
            
        df = df.dropna(subset=required_cols)
        after_dropna_count = len(df)

        # 检查是否存在label列
        has_label_col = label_col in df.columns

        if has_label_col:
            # 过滤掉label为空值的行
            df = df.dropna(subset=[label_col])
            after_label_dropna_count = len(df)

            # 过滤掉label=0的行
            def is_zero_label(label):
                if pd.isna(label):
                    return True
                label_str = str(label).strip().lower()
                return label_str == '0' or label_str == '0.0'

            df = df[~df[label_col].apply(is_zero_label)]
            after_zero_filter_count = len(df)

            # 新增：过滤掉指定的标签
            if exclude_labels is not None and len(exclude_labels) > 0:
                def should_exclude_label(label):
                    if pd.isna(label):
                        return False
                    label_str = str(label).strip().lower()
                    exclude_labels_lower = [str(ex_label).strip().lower() for ex_label in exclude_labels]
                    return label_str in exclude_labels_lower

                df = df[~df[label_col].apply(should_exclude_label)]
                final_count = len(df)

            print(f"数据过滤统计:")
            print(f"  原始数据行数: {original_count}")
            print(f"  过滤空值后: {after_dropna_count} (移除了 {original_count - after_dropna_count} 行)")
            print(f"  过滤label空值后: {after_label_dropna_count} (移除了 {after_dropna_count - after_label_dropna_count} 行)")
            print(f"  过滤label=0后: {after_zero_filter_count} (移除了 {after_label_dropna_count - after_zero_filter_count} 行)")
            if exclude_labels:
                print(f"  过滤排除标签后: {final_count} (移除了 {after_zero_filter_count - final_count} 行)")
            print(f"  总共移除: {original_count - len(df)} 行")
        else:
            final_count = after_dropna_count
            print(f"数据过滤统计:")
            print(f"  原始数据行数: {original_count}")
            print(f"  过滤空值后: {after_dropna_count} (移除了 {original_count - after_dropna_count} 行)")
            print(f"  Warning: Label column '{label_col}' not found in CSV")

        self.df = df.reset_index(drop=True)

        # 设置文件夹路径
        self.makeup_folder = makeup_folder
        self.nomakeup_folder = nomakeup_folder
        self.reference_folder = reference_folder
        self.ref1_mask_folder = ref1_mask_folder  # ref1的mask文件夹，可为None
        self.ref2_mask_folder = ref2_mask_folder  # ref2的mask文件夹，可为None

        # Label和Mask相关设置
        self.label_col = label_col
        self.mask_col = mask_col  # 共用的mask列名
        self.default_label = default_label
        self.has_mask_col = mask_col in df.columns

        self.has_label_col = label_col in self.df.columns
        if not self.has_label_col:
            print(f"Warning: Label column '{label_col}' not found in CSV. Using default label '{default_label}'")
        else:
            label_counts = self.df[label_col].value_counts()
            print(f"Label distribution:\n{label_counts}")

        # 检查mask列是否存在（如果需要的话）
        if need_mask_col and not self.has_mask_col:
            print(f"Warning: Mask column '{mask_col}' not found in CSV but mask folders are provided.")

        # 图像转换
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # mask转换（不需要normalize，保持0-1范围）
        self.mask_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        
        self.flip_prob = flip_prob
        self.fallback_prompt = fallback_prompt

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, folder, filename):
        """把CSV里的相对路径/文件名，拼接成绝对路径。如果filename已是绝对路径，直接返回。"""
        if folder is None or filename is None:
            return None
        if os.path.isabs(filename):
            return filename
        return os.path.join(folder, filename)

    def _load_and_process_mask(self, mask_path):
        """加载并处理mask图像，如果mask_path为None或文件不存在，返回None"""
        if mask_path is None:
            return None
            
        try:
            if not os.path.exists(mask_path):
                print(f"警告: Mask文件不存在: {mask_path}")
                return None
                
            # 加载mask（支持灰度图和RGB图）
            mask_image = Image.open(mask_path)
            # 转换为灰度图
            if mask_image.mode != 'L':
                mask_image = mask_image.convert('L')
        
            
            return mask_image
        except Exception as e:
            print(f"警告: 无法加载mask图像 {mask_path}: {e}")
            return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        makeup_file = row['makeup']
        nomakeup_file = row['nomakeup']
        reference_file = row['reference']
        
        # 获取mask文件名（如果有）
        mask_file = row.get(self.mask_col) if self.has_mask_col else None

        # label（用于生成prompt）
        if self.has_label_col:
            label = row[self.label_col]
            if pd.isna(label):
                label = self.default_label
        else:
            label = self.default_label

        # 拼接为绝对路径
        makeup_path = self._resolve_path(self.makeup_folder, makeup_file)
        nomakeup_path = self._resolve_path(self.nomakeup_folder, nomakeup_file)
        reference_path = self._resolve_path(self.reference_folder, reference_file)
        
        # 使用相同的mask文件名，但从不同的文件夹读取（如果文件夹存在）
        ref1_mask_path = self._resolve_path(self.ref1_mask_folder, mask_file) if self.ref1_mask_folder and mask_file else None
        ref2_mask_path = self._resolve_path(self.ref2_mask_folder, mask_file) if self.ref2_mask_folder and mask_file else None

        # 生成prompt
        prompt, ref1_localization_prompt, ref2_localization_prompt,caption_ref1_location,caption_ref2_location = build_prompt(label) if label else self.fallback_prompt

        # 加载图像
        try:
            ref1_image = Image.open(nomakeup_path).convert('RGB')   # ref1: nomakeup
            reference_image = Image.open(reference_path).convert('RGB')  # reference图像
            gt_image = Image.open(makeup_path).convert('RGB')     # gt: makeup
            
        except (FileNotFoundError, IOError) as e:
            print(f"无法加载图像 (idx={idx}): {e}")
            print(f"  ref1_path: {nomakeup_path}")
            print(f"  reference_path: {reference_path}")
            print(f"  target_path: {makeup_path}")
            return self.__getitem__((idx + 1) % len(self))

        # 加载并处理两个mask，如果路径为None或文件不存在，会返回None
        ref1_mask_image = self._load_and_process_mask(ref1_mask_path)
        ref2_mask_image = self._load_and_process_mask(ref2_mask_path)
        
        # 应用transform
        if self.transform:
            ref1_image_tensor = self.transform(ref1_image)
            reference_image_tensor = self.transform(reference_image)
            gt_image = self.transform(gt_image)
            
        # 处理mask（可能为None）
        ref1_mask = None
        ref2_mask = None
        
        if ref1_mask_image is not None and self.mask_transform:
            ref1_mask = self.mask_transform(ref1_mask_image)
            
        if ref2_mask_image is not None and self.mask_transform:
            ref2_mask = self.mask_transform(ref2_mask_image)

        # 返回五元组：ref1, ref1_mask, ref2, ref2_mask, gt
        return {
            "ref1_image": ref1_image_tensor,           # nomakeup 图像
            "ref1_mask": ref1_mask,   # ref1对应的mask，可能为None
            "ref2_image": reference_image_tensor,   # reference 图像
            "ref2_mask": ref2_mask,   # ref2对应的mask，可能为None
            "ref1_ori": ref1_image, 
            "ref2_ori": reference_image, 
            "gt_image": gt_image,              # makeup 图像（ground truth）
            "prompt": prompt,
            "label": label,
            "idx": idx,
            "makeup_path": makeup_path,
            "ref1_mask_path": ref1_mask_path,
            "ref2_mask_path": ref2_mask_path,
            "ref1_localization_prompt": ref1_localization_prompt, 
            "ref2_localization_prompt": ref2_localization_prompt,
            "caption_ref1_location": caption_ref1_location,
            "caption_ref2_location": caption_ref2_location,
        }

def collate_fn(examples):
    """处理五元组的collate_fn，支持None值的mask"""
    ref1_pixel_values = [ex["ref1_image"] for ex in examples]
    ref2_pixel_values = [ex["ref2_image"] for ex in examples]
    target_pixel_values = [ex["gt_image"] for ex in examples]
    prompts = [ex["prompt"] for ex in examples]
    labels = [ex["label"] for ex in examples]
    indices = [ex["idx"] for ex in examples]
    ref1_ori = [ex["ref1_ori"] for ex in examples]
    ref2_ori = [ex["ref2_ori"] for ex in examples]

    caption_ref1_location = [ex["caption_ref1_location"] for ex in examples]
    caption_ref2_location = [ex["caption_ref2_location"] for ex in examples]

    ref1_localization_prompt = [ex["ref1_localization_prompt"] for ex in examples]
    ref2_localization_prompt = [ex["ref2_localization_prompt"] for ex in examples]
    # 处理可能为None的mask
    ref1_mask_pixel_values = []
    for ex in examples:
        if ex["ref1_mask"] is None:
            # 创建默认全白mask tensor
            mask = torch.ones((1, 1024, 1024), dtype=torch.float32)
        else:
            mask = ex["ref1_mask"]
        ref1_mask_pixel_values.append(mask)

    ref2_mask_pixel_values = []
    for ex in examples:
        if ex["ref2_mask"] is None:
            # 创建默认全白mask tensor
            mask = torch.ones((1, 1024, 1024), dtype=torch.float32)
        else:
            mask = ex["ref2_mask"]
        ref2_mask_pixel_values.append(mask)

    ref1_pixel_values = torch.stack(ref1_pixel_values).to(memory_format=torch.contiguous_format).float()
    ref1_mask_pixel_values = torch.stack(ref1_mask_pixel_values).to(memory_format=torch.contiguous_format).float()
    ref2_pixel_values = torch.stack(ref2_pixel_values).to(memory_format=torch.contiguous_format).float()
    ref2_mask_pixel_values = torch.stack(ref2_mask_pixel_values).to(memory_format=torch.contiguous_format).float()
    target_pixel_values = torch.stack(target_pixel_values).to(memory_format=torch.contiguous_format).float()

    return {
        "ref1": ref1_pixel_values,
        "ref1_mask": ref1_mask_pixel_values,    # ref1的mask
        "ref2": ref2_pixel_values,
        "ref2_mask": ref2_mask_pixel_values,    # ref2的mask
        "ref1_ori": ref1_ori, 
        "ref2_ori": ref2_ori, 
        "ref1_localization_prompt": ref1_localization_prompt, 
        "ref2_localization_prompt": ref2_localization_prompt,
        "caption_ref1_location": caption_ref1_location,
        "caption_ref2_location": caption_ref2_location,
        "target": target_pixel_values,
        "prompts": prompts,
        "labels": labels,
        "indices": indices
    }


# ============ 图像保存功能（修改为五张图片横向拼接） ============
def save_sample_images_concatenated(sample, save_dir, sample_idx, prefix="sample"):
    """
    保存数据集返回的五元组图像为横向拼接的单张图片
    
    Args:
        sample: 数据集返回的单个样本字典
        save_dir: 保存目录
        sample_idx: 样本索引
        prefix: 文件名前缀
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义反归一化函数（从[-1,1]转回[0,1]）
    def denormalize_tensor(tensor):
        """将[-1,1]范围的tensor转换为[0,1]范围"""
        return (tensor + 1.0) / 2.0
    
    # 转换tensor为PIL图像
    def tensor_to_pil(tensor):
        """将tensor转换为PIL图像"""
        # 反归一化
        denorm_tensor = denormalize_tensor(tensor)
        # 确保值在[0,1]范围内
        denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
        # 转换为PIL图像
        return TF.to_pil_image(denorm_tensor)
    
    def mask_tensor_to_pil(tensor):
        """将mask tensor转换为PIL图像（不需要反归一化）"""
        # mask已经在[0,1]范围，直接clamp并转换
        clamped_tensor = torch.clamp(tensor, 0, 1)
        return TF.to_pil_image(clamped_tensor)
    
    # 转换五个tensor为PIL图像
    ref1_pil = tensor_to_pil(sample["ref1_image"])          # nomakeup
    ref1_mask_pil = mask_tensor_to_pil(sample["ref1_mask"])  # ref1 mask
    ref2_pil = tensor_to_pil(sample["ref2_image"])          # reference
    ref2_mask_pil = mask_tensor_to_pil(sample["ref2_mask"])  # ref2 mask
    gt_pil = tensor_to_pil(sample["gt_image"])              # makeup (gt)
    
    # 获取图像尺寸（假设所有图像尺寸相同）
    width, height = ref1_pil.size
    
    # 创建横向拼接的画布，现在是5张图片
    total_width = width * 5
    concatenated_image = Image.new('RGB', (total_width, height))
    
    # 粘贴五张图像
    concatenated_image.paste(ref1_pil, (0, 0))
    concatenated_image.paste(ref1_mask_pil, (width, 0))
    concatenated_image.paste(ref2_pil, (width * 2, 0))
    concatenated_image.paste(ref2_mask_pil, (width * 3, 0))
    concatenated_image.paste(gt_pil, (width * 4, 0))
    
    # 保存拼接后的图像
    concat_path = os.path.join(save_dir, f"{prefix}_{sample_idx:04d}_concatenated.jpg")
    concatenated_image.save(concat_path)
    print(f"已保存拼接图像: {concat_path}")
    
    # 保存样本信息到文本文件
    info_path = os.path.join(save_dir, f"{prefix}_{sample_idx:04d}_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"Sample Index: {sample_idx}\n")
        f.write(f"Label: {sample['label']}\n")
        f.write(f"Prompt: {sample['prompt']}\n")
        f.write(f"Makeup Path: {sample.get('makeup_path', 'N/A')}\n")
        f.write(f"Ref1 Mask Path: {sample.get('ref1_mask_path', 'N/A')}\n")
        f.write(f"Ref2 Mask Path: {sample.get('ref2_mask_path', 'N/A')}\n")
        f.write(f"Image Shapes: ref1={sample['ref1_image'].shape}, ref1_mask={sample['ref1_mask'].shape}, ref2={sample['ref2_image'].shape}, ref2_mask={sample['ref2_mask'].shape}, gt={sample['gt_image'].shape}\n")
        f.write(f"Concatenated Image Layout: [Ref1 (NoMakeup) | Ref1 Mask | Ref2 (Reference) | Ref2 Mask | GT (Makeup)]\n")
        f.write(f"Individual Image Size: {width}x{height}\n")
        f.write(f"Concatenated Image Size: {total_width}x{height}\n")
    
    print(f"已保存样本信息: {info_path}")


def save_sample_images_concatenated_with_labels(sample, save_dir, sample_idx, prefix="sample"):
    """
    保存数据集返回的五元组图像为横向拼接的单张图片，并在每个子图像上添加标签
    
    Args:
        sample: 数据集返回的单个样本字典
        save_dir: 保存目录
        sample_idx: 样本索引
        prefix: 文件名前缀
    """
    from PIL import ImageDraw, ImageFont
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义反归一化函数（从[-1,1]转回[0,1]）
    def denormalize_tensor(tensor):
        """将[-1,1]范围的tensor转换为[0,1]范围"""
        return (tensor + 1.0) / 2.0
    
    # 转换tensor为PIL图像
    def tensor_to_pil(tensor):
        """将tensor转换为PIL图像"""
        # 反归一化
        denorm_tensor = denormalize_tensor(tensor)
        # 确保值在[0,1]范围内
        denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
        # 转换为PIL图像
        return TF.to_pil_image(denorm_tensor)
    
    def mask_tensor_to_pil(tensor):
        """将mask tensor转换为PIL图像（不需要反归一化）"""
        # mask已经在[0,1]范围，直接clamp并转换
        clamped_tensor = torch.clamp(tensor, 0, 1)
        return TF.to_pil_image(clamped_tensor)
    
    # 转换五个tensor为PIL图像
    ref1_pil = tensor_to_pil(sample["ref1_image"])          # nomakeup
    ref1_mask_pil = mask_tensor_to_pil(sample["ref1_mask"])  # ref1 mask
    ref2_pil = tensor_to_pil(sample["ref2_image"])          # reference
    ref2_mask_pil = mask_tensor_to_pil(sample["ref2_mask"])  # ref2 mask
    gt_pil = tensor_to_pil(sample["gt_image"])              # makeup (gt)
    
    # 获取图像尺寸（假设所有图像尺寸相同）
    width, height = ref1_pil.size
    
    # 创建横向拼接的画布，现在是5张图片，增加一些高度用于标签
    label_height = 40
    total_width = width * 5  # 5张图片
    total_height = height + label_height
    concatenated_image = Image.new('RGB', (total_width, total_height), color='white')
    
    # 粘贴五张图像
    concatenated_image.paste(ref1_pil, (0, label_height))
    concatenated_image.paste(ref1_mask_pil, (width, label_height))
    concatenated_image.paste(ref2_pil, (width * 2, label_height))
    concatenated_image.paste(ref2_mask_pil, (width * 3, label_height))
    concatenated_image.paste(gt_pil, (width * 4, label_height))
    
    # 添加标签
    draw = ImageDraw.Draw(concatenated_image)
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            # 备用字体
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            # 使用默认字体
            font = ImageFont.load_default()
    
    # 五个标签文本
    labels = ["Ref1 (NoMakeup)", "Ref1 Mask", "Ref2 (Reference)", "Ref2 Mask", "GT (Makeup)"]
    
    # 在每个图像上方添加标签
    for i, label_text in enumerate(labels):
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        x_pos = i * width + (width - text_width) // 2
        draw.text((x_pos, 10), label_text, fill='black', font=font)
    
    # 保存拼接后的图像
    concat_path = os.path.join(save_dir, f"{prefix}_{sample_idx:04d}_concatenated.jpg")
    concatenated_image.save(concat_path)
    print(f"已保存拼接图像: {concat_path}")
    
    # 保存样本信息到文本文件
    info_path = os.path.join(save_dir, f"{prefix}_{sample_idx:04d}_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"Sample Index: {sample_idx}\n")
        f.write(f"Label: {sample['label']}\n")
        f.write(f"Prompt: {sample['prompt']}\n")
        f.write(f"Makeup Path: {sample.get('makeup_path', 'N/A')}\n")
        f.write(f"Ref1 Mask Path: {sample.get('ref1_mask_path', 'N/A')}\n")
        f.write(f"Ref2 Mask Path: {sample.get('ref2_mask_path', 'N/A')}\n")
        f.write(f"Image Shapes: ref1={sample['ref1_image'].shape}, ref1_mask={sample['ref1_mask'].shape}, ref2={sample['ref2_image'].shape}, ref2_mask={sample['ref2_mask'].shape}, gt={sample['gt_image'].shape}\n")
        f.write(f"Concatenated Image Layout: [Ref1 (NoMakeup) | Ref1 Mask | Ref2 (Reference) | Ref2 Mask | GT (Makeup)]\n")
        f.write(f"Individual Image Size: {width}x{height}\n")
        f.write(f"Concatenated Image Size: {total_width}x{total_height}\n")
    
    print(f"已保存样本信息: {info_path}")


def test_and_save_samples(dataset, save_dir, num_samples=5, start_idx=0, with_labels=True):
    """
    测试数据集并保存多个样本的拼接图像
    
    Args:
        dataset: MakeupDatasetWithMask实例
        save_dir: 保存目录
        num_samples: 要保存的样本数量
        start_idx: 开始的样本索引
        with_labels: 是否在图像上添加标签
    """
    print(f"=== 开始测试并保存 {num_samples} 个样本到 {save_dir} ===")
    
    save_func = save_sample_images_concatenated_with_labels if with_labels else save_sample_images_concatenated
    
    for i in range(num_samples):
        sample_idx = start_idx + i
        if sample_idx >= len(dataset):
            print(f"样本索引 {sample_idx} 超出数据集大小 {len(dataset)}，停止保存")
            break
            
        try:
            # 获取样本
            sample = dataset[sample_idx]
            
            # 保存拼接图像
            save_func(sample, save_dir, sample_idx)
            
            print(f"样本 {sample_idx} 保存完成")
            print(f"  Label: {sample['label']}")
            print(f"  Prompt: {sample['prompt'][:100]}...")
            print("-" * 50)
            
        except Exception as e:
            print(f"保存样本 {sample_idx} 时出错: {e}")
            continue
    
    print(f"=== 样本保存完成，共处理 {num_samples} 个样本 ===")


def test_batch_and_save(dataloader, save_dir, num_batches=2, with_labels=True):
    """
    测试DataLoader并保存批次数据为拼接图像
    
    Args:
        dataloader: DataLoader实例
        save_dir: 保存目录
        num_batches: 要保存的批次数量
        with_labels: 是否在图像上添加标签
    """
    print(f"=== 开始测试DataLoader并保存 {num_batches} 个批次 ===")
    
    batch_save_dir = os.path.join(save_dir, "batches")
    os.makedirs(batch_save_dir, exist_ok=True)
    
    save_func = save_sample_images_concatenated_with_labels if with_labels else save_sample_images_concatenated
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        print(f"\n处理批次 {batch_idx}:")
        print(f"  批次大小: {len(batch['prompts'])}")
        print(f"  Labels: {batch['labels']}")
        
        # 保存批次中的每个样本
        for sample_idx in range(len(batch['prompts'])):
            # 重构单个样本 - 现在包含五个组件
            sample = {
                "ref1_image": batch["ref1"][sample_idx],
                "ref1_mask": batch["ref1_mask"][sample_idx],    # ref1 mask
                "ref2_image": batch["ref2"][sample_idx], 
                "ref2_mask": batch["ref2_mask"][sample_idx],    # ref2 mask
                "gt_image": batch["target"][sample_idx],
                "prompt": batch["prompts"][sample_idx],
                "label": batch["labels"][sample_idx],
                "idx": batch["indices"][sample_idx]
            }
            
            # 保存拼接图像
            global_idx = batch["indices"][sample_idx]
            save_func(sample, batch_save_dir, global_idx, f"batch{batch_idx}")
            print(batch["prompts"][sample_idx])
        print(f"批次 {batch_idx} 保存完成")
    
    print(f"=== 批次保存完成 ===")


# ============ 示例使用 ============
if __name__ == '__main__':

    makeup_folder = "example/makeup_gt/image"
    nomakeup_folder = "example/source/image"
    reference_folder = "example/reference/image"
    
    # 两个不同的mask文件夹
    ref1_mask_folder = "example/source/mask/face"
    ref2_mask_folder = "example/reference/mask/face"
    
    csv_path = "example/pairs.csv"
    
    # 设置保存目录
    save_directory = "output/dataset_2mask"

    # 创建带双mask的数据集
    dataset_with_mask = MakeupDatasetWithMask(
        csv_path=csv_path,
        makeup_folder=makeup_folder,
        nomakeup_folder=nomakeup_folder,
        reference_folder=reference_folder,
        ref1_mask_folder=ref1_mask_folder,  # ref1的mask文件夹
        ref2_mask_folder=ref2_mask_folder,  # ref2的mask文件夹
        label_col='label',
        mask_col='mask',  # 共用的mask列名
        default_label='eyes,lip,face',
        fallback_prompt="Apply the masked makeup from image 2 to the face in image 1, precisely transferring the visible makeup areas while strictly preserving image 1's skin tone, facial structure, hair, pose, lighting, background, and overall proportions."
    )

    # DataLoader测试并保存
    print("\n=== 测试DataLoader并保存五元组批次拼接图像 ===")
    dataloader = DataLoader(dataset_with_mask, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    # 先打印基本信息
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}: size={len(batch['prompts'])}")
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Shapes: ref1={batch['ref1'].shape}, ref1_mask={batch['ref1_mask'].shape}, ref2={batch['ref2'].shape}, ref2_mask={batch['ref2_mask'].shape}, target={batch['target'].shape}")
        print(f"  Labels: {batch['labels']}")
        for j, prompt in enumerate(batch['prompts']):
            print(f"  Prompt {j}: {prompt[:50]}...")
            print("------------------------------------------------------")
            print(batch["ref1_localization_prompt"])
            print(batch["ref2_localization_prompt"])
            print("------------------------------------------------------")
            print(batch["caption_ref1_location"])
            print(batch["caption_ref2_location"])
        if i == 0:  # 只打印第一个batch的信息
            break
    
    # 保存批次数据
    test_batch_and_save(dataloader, save_directory, num_batches=2, with_labels=True)
    
    print(f"\n=== 所有测试完成！五元组拼接图像已保存到: {save_directory} ===")
