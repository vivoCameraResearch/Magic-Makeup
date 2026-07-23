<div align="center">

<img src="./assets/logo.png" width="300px">

### A Region-Controllable Diffusion Transformer for High-Fidelity Makeup Transfer

**🎊 ECCV 2026**

<a href="https://vivocameraresearch.github.io/magicmakeup/">
<img src="https://img.shields.io/badge/Project-Page-8A2BE2?style=flat-square&logo=googlechrome&logoColor=white">
</a>

<a href="https://huggingface.co/Anyou/MagicMakeup">
<img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models-FFD21E?style=flat-square">
</a>

<img src="https://img.shields.io/badge/Paper-Coming%20Soon-4C8BF5?style=flat-square">

</div>

<br>

<table>
  <tr>
    <td width="50%" align="center">
      <img src="./assets/demo_left.webp" width="100%">
    </td>
    <td width="50%" align="center">
      <img src="./assets/demo_right.webp" width="100%">
    </td>
  </tr>
</table>

<p align="center">
<sub>
✨ High-Fidelity Makeup Transfer
&nbsp;&nbsp; · &nbsp;&nbsp;
🎯 Precise Region Control
&nbsp;&nbsp; · &nbsp;&nbsp;
🪞 Identity Preservation
</sub>
</p>

<div align="center">
<img src="./assets/teaser.png" width="100%">
</div>

<p align="center">
<strong><i>MagicMakeup</i></strong> enables high-fidelity makeup transfer with precise region-level control while preserving facial identity and structure.
</p>

---

## ✨ Highlights

🎯 **Precise Region Control**  
MagicMakeup supports both full-face and localized makeup transfer, enabling flexible control over eye, lip, and facial makeup regions.

🧩 **TARG & CMPG Module**  
Token-aligned region constraints and transfer-preservation disentanglement improve regional accuracy, reduce makeup spillover, and preserve identity consistency.

📊 **High-Resolution Data & MakeupHQ Bench**  
An automated makeup-removal pipeline constructs identity-consistent, region-labeled training pairs, while MakeupHQ Bench provides standardized evaluation across synthetic and real-world settings.

---

## 🧠 Method Overview

<div align="center">
<img src="./assets/method.png" width="100%">
</div>

MagicMakeup is built upon a DiT and introduces region-aware conditioning mechanisms for precise and faithful makeup transfer.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/vivoCameraResearch/Magic-Makeup.git
cd Magic-Makeup
```

### 2. Create the Environment

We recommend using Python 3.10.

```bash
conda create -n magicmakeup python=3.10 pip -y
conda activate magicmakeup
```

Install PyTorch:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

<details>
<summary><b>Optional evaluation dependencies</b></summary>

<br>

The following packages are required for the complete evaluation pipeline:

```bash
pip install lpips torch-fidelity
```

Install the CLIP implementation used in our evaluation:

```bash
pip uninstall -y clip
pip install -e evaluate/CLIP-main
```

Some preprocessing and evaluation modules automatically download publicly available pretrained weights during the first run.

</details>

---

## 📦 Model Preparation

### 1. FLUX.1-Kontext-dev

MagicMakeup is built upon
[FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).

After accepting the model license on Hugging Face, download the Diffusers-format repository:

```bash
huggingface-cli login

huggingface-cli download black-forest-labs/FLUX.1-Kontext-dev \
  --local-dir /path/to/FLUX.1-Kontext-dev
```

### 2. MagicMakeup Checkpoint

Download the released [MagicMakeup](https://huggingface.co/Anyou/MagicMakeup) checkpoint:

```bash
huggingface-cli download Anyou/MagicMakeup \
  --local-dir /path/to/MagicMakeup-checkpoint
```

---

## 💄 Inference

### 1. Prepare the Data

We recommend organizing source and reference images as follows:

```text
data/
├── source/
│   ├── raw/
│   ├── image/
│   └── mask/
│       ├── face/
│       ├── eyes/
│       └── lip/
└── makeup/
    ├── raw/
    ├── image/
    └── mask/
        ├── face/
        ├── eyes/
        └── lip/
```

Image and mask IDs must match. Both mask naming conventions below are supported:

```text
image/0001.jpg
mask/0001.png
```

or

```text
image/0001.jpg
mask/0001_mask.png
```

### 2. Crop Face Images

`preprocess/crop.py` detects the primary face, filters invalid samples, generates a centered `1024 × 1024` crop, and records processing information in `log.tsv`.

**Source images:**

```bash
python preprocess/crop.py \
  --input_dir data/source/raw \
  --out_dir data/source/image \
  --det_model mediapipe/blaze_face_short_range.tflite \
  --lmk_model mediapipe/face_landmarker.task \
  --expand 0.8 \
  --min_face_ratio 0.1
```

**Makeup reference images:**

```bash
python preprocess/crop.py \
  --input_dir data/makeup/raw \
  --out_dir data/makeup/image \
  --det_model mediapipe/blaze_face_short_range.tflite \
  --lmk_model mediapipe/face_landmarker.task \
  --expand 0.8 \
  --min_face_ratio 0.1
```

Add `--keep_subdirs` if the original input directory hierarchy should be preserved.

### 3. Generate Region Masks

#### Face Masks

`preprocess/faceparsing.py` uses `jonathandinu/face-parsing` to generate binary face masks.

```bash
# Source images
python preprocess/faceparsing.py \
  --img_path data/source/image \
  --save_path data/source/mask/face \
  --recursive

# Makeup reference images
python preprocess/faceparsing.py \
  --img_path data/makeup/image \
  --save_path data/makeup/mask/face \
  --recursive
```

#### Eyes Masks

```bash
# Source images
python preprocess/regionmask.py \
  --input_dir data/source/image \
  --output_dir data/source/mask/eyes \
  --regions eyes

# Makeup reference images
python preprocess/regionmask.py \
  --input_dir data/makeup/image \
  --output_dir data/makeup/mask/eyes \
  --regions eyes
```

#### Lip Masks

```bash
# Source images
python preprocess/regionmask.py \
  --input_dir data/source/image \
  --output_dir data/source/mask/lip \
  --regions lip

# Makeup reference images
python preprocess/regionmask.py \
  --input_dir data/makeup/image \
  --output_dir data/makeup/mask/lip \
  --regions lip
```

### 4. Single-Pair Inference

Run inference for a single source-reference pair:

```bash
python test_single.py \
  --model_path /path/to/FLUX.1-Kontext-dev \
  --lora_path /path/to/MagicMakeup-checkpoint \
  --source_image data/source/image/0001.png \
  --source_mask data/source/mask/face/0001.png \
  --reference_image data/makeup/image/0001.png \
  --reference_mask data/makeup/mask/face/0001.png \
  --label eyes,lip,face \
  --output_path outputs/0001_0001.jpg \
  --panel_path outputs/0001_0001_panel.jpg
```

The `--label` argument controls which makeup regions are transferred:

```text
--label eyes
--label lip
--label eyes,lip,face
```

For full makeup transfer, use:

```bash
--label eyes,lip,face
```

The default `model_offload` mode is recommended for most GPUs. For devices with limited GPU memory, use:

```bash
--memory_mode sequential_offload
```

> **Note:** Sequential CPU offloading further reduces GPU memory usage but may increase inference latency.

### 5. Batch Inference

`test_dir.py` performs all-to-all pairing between source images and makeup references.

Images and masks are automatically matched by filename stem. Samples without matching masks are reported and skipped.

```bash
python test_dir.py \
  --model_path /path/to/FLUX.1-Kontext-dev \
  --lora_path /path/to/MagicMakeup-checkpoint \
  --source_images data/source/image \
  --source_masks data/source/mask/face \
  --reference_images data/makeup/image \
  --reference_masks data/makeup/mask/face \
  --output_dir outputs/face \
  --panel_output_dir outputs/face_panel \
  --label eyes,lip,face
```

---

## 📊 Evaluation

The evaluation pipeline consists of three stages:

```text
Source & Reference Images
          │
          ▼
   Generate Pair List
          │
          ▼
 Landmark Detection
  & Face Alignment
          │
          ▼
   Metric Evaluation
```

### 1. Generate Source-Reference Pairs

Generate the Cartesian product between source and reference images:

```bash
python evaluate/generate_pairs_csv.py \
  --src_dir data/source/image \
  --ref_dir data/makeup/image \
  --output_csv metrics/pairs.csv
```

This produces the `src,ref` pair list required by the evaluation preprocessing script.

### 2. Prepare Images for Evaluation

Generated images should follow the naming convention:

```text
{source_stem}_{reference_stem}.jpg
```

Run landmark detection and evaluation preprocessing:

```bash
python evaluate/prevalu.py \
  --model mediapipe/face_landmarker.task \
  --input_csv metrics/pairs.csv \
  --gen_dir outputs/face \
  --out_root metrics/run1 \
  --model_name MagicMakeup \
  --target_width 1024 \
  --target_height 1024
```

The processed files are stored as:

```text
metrics/run1/
├── MagicMakeup.csv
├── src/
└── gen/
    └── MagicMakeup/
```

### 3. Compute Metrics

The evaluation pipeline supports the following metrics:

| Category | Metric |
| :--- | :--- |
| Identity / Structure | Self-Sim |
| Makeup Similarity | DINO-I |
| Semantic Similarity | CLIP-I |
| Background Preservation | BG-MSE |
| Distribution Quality | FID |
| Distribution Quality | KID |
| Face Identity | Face-ID |

Run evaluation without Face-ID:

```bash
python evaluate/evalu.py \
  --pairs_csv metrics/run1/MagicMakeup.csv \
  --out_csv metrics/run1/results.csv \
  --target_size 1024 1024 \
  --batch_size 16 \
  --skip_face_id
```

<details>
<summary><b>Optional: Face-ID Evaluation</b></summary>

<br>

Face-ID evaluation requires two pretrained models from
[CVLFace](https://github.com/mk-minchul/CVLface):

- AdaFace face recognition model
- DFA face alignment model

Download the AdaFace recognition model:

```bash
huggingface-cli download \
  minchul/cvlface_adaface_vit_base_kprpe_webface12m \
  --local-dir evaluate/cvlface/adaface_vit_base_kprpe_webface12m
```

Download the DFA alignment model:

```bash
huggingface-cli download \
  minchul/cvlface_DFA_mobilenet \
  --local-dir evaluate/cvlface/DFA_mobilenet
```

Then run the complete evaluation:

```bash
python evaluate/evalu.py \
  --pairs_csv metrics/run1/MagicMakeup.csv \
  --out_csv metrics/run1/results.csv \
  --target_size 1024 1024 \
  --batch_size 16 \
  --recognition_model_id evaluate/cvlface/adaface_vit_base_kprpe_webface12m \
  --aligner_id evaluate/cvlface/DFA_mobilenet
```

## 📁 Repository Structure

```text
Magic-Makeup/
├── assets/                 # README figures and demo animations
├── data/                   # Input images and region masks
├── evaluate/               # Evaluation pipeline
├── preprocess/             # Face cropping and mask generation
├── outputs/                # Generated results
├── test_single.py          # Single-pair inference
├── test_dir.py             # Batch inference
├── requirements.txt
└── README.md
```

---

## 🙏 Acknowledgements

This project builds upon the following excellent open-source projects:

- [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [Diffusers](https://github.com/huggingface/diffusers)
- [MediaPipe](https://github.com/google-ai-edge/mediapipe)
- [DINO](https://github.com/facebookresearch/dino)
- [CLIP](https://github.com/openai/CLIP)
- [CVLFace](https://github.com/mk-minchul/CVLface)

We sincerely thank the authors and maintainers for making their work publicly available.

---

## ❗ Ethical Considerations

MagicMakeup and MakeupHQ Bench are intended solely for non-commercial academic research on cosmetic makeup transfer.

They must not be used for identity recognition, face swapping, impersonation, deceptive manipulation, or other identity-related misuse.

Any real-face benchmark release will be de-identified and distributed through gated access under a Data Usage Agreement that prohibits redistribution and identity-related misuse. Bias disclosure and opt-out or removal mechanisms will also be provided for individuals and rights holders.

---

## 📜 Citation

If you find MagicMakeup useful for your research, please consider citing our work:

```bibtex
@inproceedings{magicmakeup2026,
  title     = {MagicMakeup: A Region-Controllable Diffusion Transformer for High-Fidelity Makeup Transfer},
  author    = {Ziyi Wang and Siming Zheng and Yang Yang and Shusong Xu and Hao Zhang and Bo Li and Changqing Zou and Peng-Tao Jiang},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

---

<div align="center">

### ✨ MagicMakeup

**High-Fidelity · Region-Controllable · Identity-Preserving**

</div>
