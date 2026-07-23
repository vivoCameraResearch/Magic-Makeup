#!/usr/bin/env python
# coding=utf-8
import os
import sys
import re
import argparse

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from diffusers.utils import load_image
import safetensors

sys.path.append('../MagicMakeup')
from pipeline_attn import FluxKontextPipeline
from model.transformer.transformer_flux_mod_text_ref import FluxTransformer2DModel
from model.selector.img_guider_mod_text_ref import ImgGuider, ImgGuiderCrossAttn
from prompt import build_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Single image inference")

    # 模型路径
    parser.add_argument("--lora_path", type=str,
                        default="model/MagicMakeup")
    parser.add_argument("--model_path", type=str,
                        default="/FLUX.1-Kontext-dev")

    # 输入图像和mask路径（必需参数）
    parser.add_argument("--source_image", type=str,
                        default="example/source/image/0001.png",
                        help="Path to source image")
    parser.add_argument("--source_mask", type=str,
                        default="example/source/mask/face/0001.png",
                        help="Path to source image mask (optional)")
    parser.add_argument("--reference_image", type=str,
                        default="example/makeup/image/0001.png",
                        help="Path to reference image")
    parser.add_argument("--reference_mask", type=str,
                        default="example/makeup/mask/face/0001.png",
                        help="Path to reference image mask (optional)")

    # 输出路径
    parser.add_argument("--output_path", type=str, default="example/output/face/0001_0001.jpg",
                        help="Path to save the generated image")
    parser.add_argument("--save_panel", action="store_true", default=True,
                        help="Also save side-by-side comparison panel")
    parser.add_argument("--panel_path", type=str, default="example/output/face/0001_0001_panel.jpg",
                        help="Path to save the comparison panel")

    # 生成配置
    parser.add_argument("--label", type=str, default="eyes,lip,face",
                        help="Makeup label")
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--memory_mode",
        choices=("full", "model_offload", "sequential_offload"),
        default="model_offload",
        help="GPU memory mode; offload modes trade speed for lower VRAM usage.",
    )

    return parser.parse_args()


def load_mask_L(mask_path):
    if mask_path is None or not os.path.exists(mask_path):
        return None
    img = Image.open(mask_path)
    return img.convert("L") if img.mode != "L" else img


def apply_mask_to_image(image, mask):
    if mask is None:
        return image
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.NEAREST)
    img_array = np.array(image).astype(np.float32)
    mask_array = np.array(mask).astype(np.float32) / 255.0
    mask_3d = np.repeat(np.expand_dims(mask_array, axis=2), 3, axis=2)
    result = np.clip(img_array * mask_3d, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def to_bchw_bool(mask, B, H, W, device):
    if mask is None:
        return None
    arr = np.array(mask)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    if t.shape[0] == 1 and B > 1:
        t = t.expand(B, *t.shape[1:])
    t = t.to(device=device)
    t = F.interpolate(t, size=(H, W), mode="nearest")
    return t > 0.5


def token_grid_and_lengths(pipe, height, width):
    multiple = pipe.vae_scale_factor * 2
    H = height // multiple * multiple
    W = width  // multiple * multiple
    h_tok = H // (pipe.vae_scale_factor * 2)
    w_tok = W // (pipe.vae_scale_factor * 2)
    return h_tok, w_tok, h_tok * w_tok


def create_panel(image1, ref1_mask, image2, ref2_mask, generated):
    panels = [image1]
    if ref1_mask is not None:
        panels.append(ref1_mask.convert("RGB"))
    panels.append(image2)
    if ref2_mask is not None:
        panels.append(ref2_mask.convert("RGB"))
    panels.append(generated)

    h = min(p.height for p in panels)
    resized = [p.resize((int(p.width * h / p.height), h), Image.BILINEAR) for p in panels]
    canvas = Image.new("RGB", (sum(p.width for p in resized), h), (255, 255, 255))
    x = 0
    for p in resized:
        canvas.paste(p, (x, 0))
        x += p.width
    return canvas


def load_models(args):
    """Load all reusable model components once."""
    memory_mode = getattr(args, "memory_mode", "model_offload")
    print(f"Loading model / pipeline (memory_mode={memory_mode}) ...")

    img_guider = ImgGuider(dim=1152, attention_head_dim=64, img_out_dim=1152, text_out_dim=3072 * 20)
    guider_sd = safetensors.torch.load_file(os.path.join(args.lora_path, "guider.safetensors"))
    img_guider.load_state_dict({k.replace('module.', ''): v for k, v in guider_sd.items() if 'module' in k})

    img_cross_attn = ImgGuiderCrossAttn(dim=3072, attention_head_dim=64, vit_dim=1152, ff_mult=2)
    cross_sd = safetensors.torch.load_file(os.path.join(args.lora_path, "img_cross.safetensors"))
    img_cross_attn.load_state_dict({k.replace('module.', ''): v for k, v in cross_sd.items() if 'module' in k})

    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, subfolder="transformer")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(args.lora_path)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    if memory_mode == "full":
        pipe.to("cuda")
    elif memory_mode == "model_offload":
        pipe.enable_model_cpu_offload()
    elif memory_mode == "sequential_offload":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"Unsupported memory mode: {memory_mode}")

    img_guider.to("cuda")
    img_cross_attn.to("cuda")

    return pipe, img_guider, img_cross_attn


def run_single(args, pipe, img_guider, img_cross_attn):
    """Generate and save one source/reference pair using preloaded models."""

    # ---------- 加载输入 ----------
    image1 = load_image(args.source_image).resize((args.width, args.height), Image.BILINEAR)
    image2 = load_image(args.reference_image).resize((args.width, args.height), Image.BILINEAR)
    ref1_mask = load_mask_L(args.source_mask)
    ref2_mask = load_mask_L(args.reference_mask)
    if ref1_mask is not None:
        ref1_mask = ref1_mask.resize((args.width, args.height), Image.NEAREST)
    if ref2_mask is not None:
        ref2_mask = ref2_mask.resize((args.width, args.height), Image.NEAREST)

    # 对 reference 应用 mask
    image2_masked = apply_mask_to_image(image2, ref2_mask)

    # ---------- Prompt ----------
    prompt, ref1_loc_prompt, ref2_loc_prompt, caption_ref1_loc, caption_ref2_loc = build_prompt(args.label)
    localization_prompt = [ref1_loc_prompt, ref2_loc_prompt]

    # ---------- 编码 prompt ----------
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, device="cuda", num_images_per_prompt=1)

    # ---------- 构建 attention mask ----------
    B, device = 1, torch.device("cuda")
    h_tok, w_tok, L_lat = token_grid_and_lengths(pipe, args.height, args.width)
    L_txt = prompt_embeds.shape[1]
    L_img1 = L_img2 = L_lat

    def mask_to_roi(mask):
        if mask is None:
            return torch.ones(B, L_lat, dtype=torch.bool, device=device)
        bool_mask = to_bchw_bool(mask, B, args.height, args.width, device)
        return F.interpolate(bool_mask.float(), size=(h_tok, w_tok), mode="nearest").view(B, -1) > 0.5

    roi_q  = mask_to_roi(ref1_mask)   # source ROI
    roi_k2 = mask_to_roi(ref2_mask)   # reference ROI

    L_full = L_txt + L_lat + L_img1 + L_img2
    attn_mask = torch.zeros(B, 1, L_full, L_full, device=device, dtype=prompt_embeds.dtype)
    rows_lat = slice(L_txt, L_txt + L_lat)
    cols_r2  = slice(L_txt + L_lat + L_img1, L_txt + L_lat + L_img1 + L_img2)
    allow = roi_q.view(B, 1, L_lat, 1) & roi_k2.view(B, 1, 1, L_img2)
    attn_mask[:, :, rows_lat, cols_r2].masked_fill_(~allow, -1e9)

    # ---------- 推理 ----------
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    out = pipe(
        image=image1,
        image_2=image2_masked,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        joint_attention_kwargs={"attention_mask": attn_mask},
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=gen,
        img_guider=img_guider,
        per_block_text_mod=True,
        img_cross_attn=img_cross_attn,
        localization_prompt=localization_prompt,
        caption_ref1_location=caption_ref1_loc,
        caption_ref2_location=caption_ref2_loc,
    )

    # ---------- 保存结果 ----------
    gen_img = out.images[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    gen_img.save(args.output_path, quality=95)
    print(f"Saved generated image: {args.output_path}")

    if args.save_panel:
        panel = create_panel(image1, ref1_mask, image2_masked, ref2_mask, gen_img)
        os.makedirs(os.path.dirname(os.path.abspath(args.panel_path)), exist_ok=True)
        panel.save(args.panel_path, quality=95)
        print(f"Saved panel: {args.panel_path}")

    return gen_img


def main():
    args = parse_args()

    missing = [
        path
        for path in (args.source_image, args.reference_image)
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(f"Input image not found: {missing}")

    pipe, img_guider, img_cross_attn = load_models(args)
    run_single(args, pipe, img_guider, img_cross_attn)


if __name__ == "__main__":
    main()
