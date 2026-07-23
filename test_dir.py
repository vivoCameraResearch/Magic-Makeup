#!/usr/bin/env python3

import argparse
import json
import traceback
from argparse import Namespace
from itertools import product
from pathlib import Path

import torch
from diffusers.utils import load_image
from PIL import Image

try:
    from .test_single import load_models, run_single
except ImportError:
    from test_single import load_models, run_single


DEFAULT_SOURCE_IMAGES = Path("example/source/image")
DEFAULT_SOURCE_MASKS = Path("example/source/mask/face")
DEFAULT_REFERENCE_IMAGES = Path("example/makeup/image")
DEFAULT_REFERENCE_MASKS = Path("example/makeup/mask/face")
DEFAULT_OUTPUT_DIR = Path("example/output/face")
DEFAULT_PANEL_OUTPUT_DIR = Path("example/output_panel/face")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GENERATION_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pair matched source/reference images and run batch inference."
    )
    parser.add_argument("--source_images", type=Path, default=DEFAULT_SOURCE_IMAGES)
    parser.add_argument("--source_masks", type=Path, default=DEFAULT_SOURCE_MASKS)
    parser.add_argument("--reference_images", type=Path, default=DEFAULT_REFERENCE_IMAGES)
    parser.add_argument("--reference_masks", type=Path, default=DEFAULT_REFERENCE_MASKS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel_output_dir", type=Path, default=DEFAULT_PANEL_OUTPUT_DIR)
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of independent GPU worker processes.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based shard handled by this process.",
    )
    parser.add_argument(
        "--lora_path",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        required=True,
    )
    # "eyes,lip,face" or "eyes" or "lip"
    parser.add_argument("--label", default="eyes,lip,face")
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.set_defaults(save_panels=True)
    parser.add_argument(
        "--save_panels",
        "--save-panels",
        dest="save_panels",
        action="store_true",
        help="Save source/reference/result comparison panels (default: enabled).",
    )
    parser.add_argument(
        "--no-save-panels",
        dest="save_panels",
        action="store_false",
        help="Do not save comparison panels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate results that already exist.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop at the first failed pair instead of continuing.",
    )
    return parser.parse_args()


def collect_matched(image_dir, mask_dir, side):
    if not image_dir.is_dir():
        raise FileNotFoundError(f"{side} image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"{side} mask directory not found: {mask_dir}")

    images = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in images:
                raise RuntimeError(f"Duplicate {side} image id: {path.stem}")
            images[path.stem] = path

    masks = {}
    for path in sorted(mask_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_id = path.stem[:-5] if path.stem.endswith("_mask") else path.stem
        if mask_id in masks:
            raise RuntimeError(f"Duplicate {side} mask id: {mask_id}")
        masks[mask_id] = path

    matched_ids = sorted(images.keys() & masks.keys())
    missing_masks = sorted(images.keys() - masks.keys())
    orphan_masks = sorted(masks.keys() - images.keys())
    if missing_masks:
        print(
            f"Warning: ignoring {len(missing_masks)} {side} images without masks: "
            f"{missing_masks[:10]}"
        )
    if orphan_masks:
        print(
            f"Warning: ignoring {len(orphan_masks)} {side} masks without images: "
            f"{orphan_masks[:10]}"
        )
    if not matched_ids:
        raise RuntimeError(f"No matched {side} image/mask pairs found")

    return [(item_id, images[item_id], masks[item_id]) for item_id in matched_ids]


def select_jobs(
    sources,
    references,
    output_dir,
    panel_output_dir=None,
):
    panel_output_dir = panel_output_dir or output_dir
    jobs = []
    output_names = set()
    for job_index, (source, reference) in enumerate(product(sources, references)):
        pair_id = f"{source[0]}_{reference[0]}"
        output_name = f"{pair_id}.jpg"
        if output_name in output_names:
            raise RuntimeError(f"Output filename collision: {output_name}")
        output_names.add(output_name)
        jobs.append(
            {
                "index": job_index,
                "pair_id": pair_id,
                "source_id": source[0],
                "source_image": str(source[1]),
                "source_mask": str(source[2]),
                "reference_id": reference[0],
                "reference_image": str(reference[1]),
                "reference_mask": str(reference[2]),
                "output_path": str(output_dir / output_name),
                "panel_path": str(panel_output_dir / f"{pair_id}_panel.jpg"),
            }
        )
    return jobs


def write_manifest(args, sources, references, jobs):
    manifest_name = f"all_pairs_count{len(jobs)}.json"
    manifest_path = args.output_dir / manifest_name
    manifest = {
        "pairing": "all",
        "generation_seed": GENERATION_SEED,
        "source_count": len(sources),
        "reference_count": len(references),
        "selected_count": len(jobs),
        "jobs": jobs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def make_single_args(args, job):
    return Namespace(
        lora_path=args.lora_path,
        model_path=args.model_path,
        source_image=job["source_image"],
        source_mask=job["source_mask"],
        reference_image=job["reference_image"],
        reference_mask=job["reference_mask"],
        output_path=job["output_path"],
        save_panel=False,
        panel_path=job["panel_path"],
        label=args.label,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=GENERATION_SEED,
    )


def save_three_image_panel(job, width, height):
    source = load_image(job["source_image"]).convert("RGB").resize(
        (width, height), Image.BILINEAR
    )
    reference = load_image(job["reference_image"]).convert("RGB").resize(
        (width, height), Image.BILINEAR
    )
    generated = load_image(job["output_path"]).convert("RGB")

    panels = [source, reference, generated]
    panel_height = min(image.height for image in panels)
    resized = [
        image.resize(
            (int(image.width * panel_height / image.height), panel_height),
            Image.BILINEAR,
        )
        for image in panels
    ]
    canvas = Image.new(
        "RGB",
        (sum(image.width for image in resized), panel_height),
        (255, 255, 255),
    )
    x = 0
    for image in resized:
        canvas.paste(image, (x, 0))
        x += image.width

    panel_path = Path(job["panel_path"])
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(panel_path, quality=95)
    print(f"Saved three-image panel: {panel_path}")


def main():
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be greater than zero")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    sources = collect_matched(args.source_images, args.source_masks, "source")
    references = collect_matched(args.reference_images, args.reference_masks, "reference")
    all_jobs = select_jobs(
        sources,
        references,
        args.output_dir,
        args.panel_output_dir,
    )
    jobs = all_jobs[args.shard_index :: args.num_shards]

    print(f"Matched sources: {len(sources)}")
    print(f"Matched references: {len(references)}")
    print(f"All pairs: {len(sources)} x {len(references)} = {len(all_jobs)}")
    print(f"Generation seed: {GENERATION_SEED}")
    print(
        f"Worker shard: {args.shard_index}/{args.num_shards} "
        f"({len(jobs)} pairs)"
    )
    for job in jobs[:20]:
        print(f"  {job['source_id']} + {job['reference_id']} -> {Path(job['output_path']).name}")
    if len(jobs) > 20:
        print(f"  ... and {len(jobs) - 20} more")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_panels:
        args.panel_output_dir.mkdir(parents=True, exist_ok=True)
    if args.shard_index == 0:
        manifest_path = write_manifest(args, sources, references, all_jobs)
        print(f"Pair manifest: {manifest_path}")

    pending = []
    panel_backfill = []
    skipped = 0
    for job in jobs:
        output_exists = Path(job["output_path"]).is_file()
        panel_exists = Path(job["panel_path"]).is_file()
        if args.overwrite or not output_exists:
            pending.append(job)
        elif args.save_panels and not panel_exists:
            panel_backfill.append(job)
        else:
            skipped += 1

    failed = 0
    failure_log = args.output_dir / (
        f"batch_failures_shard{args.shard_index}.log"
        if args.num_shards > 1
        else "batch_failures.log"
    )
    panels_created = 0
    for job in panel_backfill:
        try:
            save_three_image_panel(job, args.width, args.height)
            panels_created += 1
        except Exception:
            failed += 1
            error = traceback.format_exc()
            with failure_log.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[panel:{job['pair_id']}]\n{error}\n")
            print(f"Panel failed: {job['pair_id']}; details saved to {failure_log}")
            if args.stop_on_error:
                raise

    if not pending:
        print(
            f"No inference needed: panels_created={panels_created}, "
            f"skipped={skipped}, failed={failed}"
        )
        if failed:
            raise SystemExit(1)
        return

    model_args = Namespace(lora_path=args.lora_path, model_path=args.model_path)
    pipe, img_guider, img_cross_attn = load_models(model_args)

    completed = 0
    pending_ids = {job["pair_id"] for job in pending}
    for position, job in enumerate(jobs, start=1):
        if job["pair_id"] not in pending_ids:
            print(f"[{position}/{len(jobs)}] Skip existing: {job['pair_id']}")
            continue

        single_args = make_single_args(args, job)
        print(
            f"[{position}/{len(jobs)}] Generate {job['source_id']} + "
            f"{job['reference_id']} (seed={single_args.seed})"
        )
        try:
            run_single(single_args, pipe, img_guider, img_cross_attn)
            completed += 1
            if args.save_panels:
                save_three_image_panel(job, args.width, args.height)
                panels_created += 1
        except Exception:
            failed += 1
            error = traceback.format_exc()
            with failure_log.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{job['pair_id']}]\n{error}\n")
            print(f"Failed: {job['pair_id']}; details saved to {failure_log}")
            torch.cuda.empty_cache()
            if args.stop_on_error:
                raise

    print(
        f"Batch finished: generated={completed}, panels_created={panels_created}, "
        f"skipped={skipped}, failed={failed}, shard_selected={len(jobs)}, "
        f"global_selected={len(all_jobs)}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
