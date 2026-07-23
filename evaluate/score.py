import lpips
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as vT
from transformers import AutoProcessor, CLIPModel
from tqdm.auto import tqdm
from typing import Optional, Dict, Tuple, List
from dino_extractor import VitExtractor
from PIL import Image
from pathlib import Path
import clip
from scipy import spatial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LossG(torch.nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            cfg = {
                "dino_model_name": "dino_vitb8",
                "dino_global_patch_size": 224,
                "clip_model_id": "evaluate/clip-vit-large-patch32",
                "dino_v1_hub_dir": str(Path(__file__).resolve().parent / "dino"),
            }

        # ===== DINO v2 =====
        self.extractor = VitExtractor(model_name=cfg["dino_model_name"], device=device)
        imagenet_norm = vT.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = vT.Resize(cfg["dino_global_patch_size"], max_size=480)
        self.global_transform = vT.Compose([
            vT.ToTensor(),
            global_resize_transform,
            imagenet_norm
        ])

        # # ===== CLIP (transformers) =====
        # self.clip_model = CLIPModel.from_pretrained(cfg["clip_model_id"]).to(device)
        # self.clip_processor = AutoProcessor.from_pretrained(cfg["clip_model_id"])
        
        # ===== 官方 CLIP（用于 compatible 方法）=====
        print("[INFO] Loading official CLIP model for compatible mode...")
        try:
            # 方法1：使用官方模型名称
            self.clip_official_model, self.clip_official_transform = clip.load(
                'ViT-B/32',  # 使用标准模型名称
                device=device,
                download_root=cfg.get("clip_cache_dir", None)  # 可选：指定缓存目录
            )
            self.clip_official_model.eval()
            print("[INFO] Official CLIP model loaded successfully")
        except Exception as e:
            print(f"[WARN] Failed to load official CLIP: {e}")
            self.clip_official_model = None
            self.clip_official_transform = None

        # ===== LPIPS =====
        self.lpips_model = lpips.LPIPS(net="alex").to(device)

        # ===== DINO v1（用于 compatible 方法）=====
        print("[INFO] Loading DINO v1 model for compatible mode...")
        try:
            self.dino_v1_model = torch.hub.load(
                cfg["dino_v1_hub_dir"], 
                'dino_vitb8', 
                source='local'
            ).to(device)
            self.dino_v1_model.eval()
            
            # DINO v1 专用预处理
            self.dino_v1_transform = vT.Compose([
                vT.Resize(256, interpolation=3),
                vT.CenterCrop(224),
                vT.ToTensor(),
                vT.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            print("[INFO] DINO v1 model loaded successfully")
        except Exception as e:
            print(f"[WARN] Failed to load DINO v1: {e}")
            self.dino_v1_model = None
            self.dino_v1_transform = None

    @torch.no_grad()
    def calculate_clip_text_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in tqdm(zip(inputs, outputs), desc="CLIP", total=len(inputs), leave=False):
            inputs_proc = self.clip_processor(
                text=[a],
                images=b,
                return_tensors="pt",
            ).to(device)

            outputs_model = self.clip_model(**inputs_proc)
            image_features = outputs_model.image_embeds
            text_features = outputs_model.text_embeds
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (text_features @ image_features.T).squeeze(0).detach()[0]
            loss += similarity
            
        return loss.item()

    @torch.no_grad()
    def calculate_self_sim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in tqdm(zip(inputs, outputs), desc="DINO self-similarity", total=len(inputs), leave=False):
            a = self.global_transform(a).to(device)
            b = self.global_transform(b).to(device)
            target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim).detach()
        return loss.item()

    @torch.no_grad()
    def calculate_dino_i_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in tqdm(zip(outputs, inputs), desc="DINO-I (CLS cosine similarity)", total=len(inputs), leave=False):
            a = self.global_transform(a).unsqueeze(0).to(device)
            b = self.global_transform(b).unsqueeze(0).to(device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.cosine_similarity(cls_token, target_cls_token, dim=0)
        return loss.item()
    
    @torch.no_grad()
    def calculate_LPIPS_distance(self, outputs, inputs):
        distance = 0.0
        transform = vT.Compose([vT.ToTensor()])
        for a, b in tqdm(zip(outputs, inputs), desc="LPIPS", total=len(inputs), leave=False):
            a = transform(a).unsqueeze(0).to(device)
            b = transform(b).unsqueeze(0).to(device)
            distance += self.lpips_model(a, b).detach()
        return distance.item()

    @torch.no_grad()
    def calculate_clip_i_loss(self, outputs, inputs):
        """
        计算CLIP图像相似度 (CLIP-I)
        使用CLIP模型的图像编码器计算两张图像的特征相似度
        """
        loss = 0.0
        for a, b in tqdm(zip(outputs, inputs), desc="CLIP-I (Image similarity)", total=len(inputs), leave=False):
            inputs_processed = self.clip_processor(
                images=[a, b],
                return_tensors="pt",
            ).to(device)
            
            image_features = self.clip_model.get_image_features(**inputs_processed)
            feature_a = image_features[0:1]
            feature_b = image_features[1:2]
            
            feature_a = feature_a / feature_a.norm(dim=-1, keepdim=True)
            feature_b = feature_b / feature_b.norm(dim=-1, keepdim=True)
            
            similarity = (feature_a @ feature_b.T).squeeze().detach()
            loss += similarity
            
        return loss.item()

    @torch.no_grad()
    def calculate_clip_i_loss_compatible(self, result_images, appearance_images):
        """
        使用预加载的官方CLIP模型
        """
        # 检查模型是否已加载
        if self.clip_official_model is None:
            print("[ERROR] Official CLIP model not loaded in __init__")
            return float('nan')
        
        # 检查输入
        if not result_images or not appearance_images:
            print("[WARN] Empty image lists provided")
            return float('nan')
        
        if len(result_images) != len(appearance_images):
            print(f"[WARN] Image list length mismatch: {len(result_images)} vs {len(appearance_images)}")
            return float('nan')
        
        similarities = []
        
        for result_img, appearance_img in tqdm(
            zip(result_images, appearance_images),
            desc="CLIP-I (compatible)",
            total=len(result_images),
            leave=False
        ):
            try:
                # 预处理
                result_input = self.clip_official_transform(result_img).unsqueeze(0).to(device)
                appearance_input = self.clip_official_transform(appearance_img).unsqueeze(0).to(device)
                
                # 特征提取（使用预加载的模型）
                result_features = self.clip_official_model.encode_image(result_input).detach().cpu().float()
                appearance_features = self.clip_official_model.encode_image(appearance_input).detach().cpu().float()
                
                # 计算相似度
                similarity = 1 - spatial.distance.cosine(
                    result_features.view(-1).numpy(),
                    appearance_features.view(-1).numpy()
                )
                similarities.append(similarity)
                
            except Exception as e:
                print(f"[WARN] Failed to process image pair: {e}")
                continue
        
        if not similarities:
            print("[ERROR] No valid similarities computed")
            return float('nan')
        
        return sum(similarities) / len(similarities)

    @torch.no_grad()
    def calculate_dino_i_loss_compatible(self, result_images, appearance_images):
        """
        使用预加载的DINO v1模型
        """
        # 检查模型是否已加载
        if self.dino_v1_model is None:
            print("[ERROR] DINO v1 model not loaded in __init__")
            return float('nan')
        
        # 检查输入
        if not result_images or not appearance_images:
            print("[WARN] Empty image lists provided")
            return float('nan')
        
        if len(result_images) != len(appearance_images):
            print(f"[WARN] Image list length mismatch: {len(result_images)} vs {len(appearance_images)}")
            return float('nan')
        
        similarities = []
        
        for result_img, appearance_img in tqdm(
            zip(result_images, appearance_images),
            desc="DINO-I (compatible)",
            total=len(result_images),
            leave=False
        ):
            try:
                # 预处理
                result_input = self.dino_v1_transform(result_img).unsqueeze(0).to(device)
                appearance_input = self.dino_v1_transform(appearance_img).unsqueeze(0).to(device)
                
                # 特征提取（使用预加载的模型）
                result_features = self.dino_v1_model(result_input).detach().cpu().float()
                appearance_features = self.dino_v1_model(appearance_input).detach().cpu().float()
                
                # 计算相似度
                similarity = 1 - spatial.distance.cosine(
                    result_features.view(-1).numpy(),
                    appearance_features.view(-1).numpy()
                )
                similarities.append(similarity)
                
            except Exception as e:
                print(f"[WARN] Failed to process image pair: {e}")
                continue
        
        if not similarities:
            print("[ERROR] No valid similarities computed")
            return float('nan')
        
        return sum(similarities) / len(similarities)
