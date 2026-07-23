#!/usr/bin/env python3
"""Generate a src/ref Cartesian-product CSV from two image folders."""

import argparse
import csv
from itertools import product
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pair every src image with every ref image and write a CSV."
    )
    parser.add_argument("--src_dir", type=Path, default="MagicMakeup/example/source/image")
    parser.add_argument("--ref_dir", type=Path, default="MagicMakeup/example/makeup/image")
    parser.add_argument("--output_csv", type=Path, default=Path("MagicMakeup/example/pairs.csv"))
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan subdirectories.",
    )
    return parser.parse_args()


def collect_images(folder, recursive):
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Image directory not found: {folder}")

    paths = folder.rglob("*") if recursive else folder.iterdir()
    images = sorted(
        path.resolve()
        for path in paths
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No images found in: {folder}")
    return images


def main():
    args = parse_args()
    sources = collect_images(args.src_dir, args.recursive)
    references = collect_images(args.ref_dir, args.recursive)

    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["src", "ref"])
        writer.writerows(
            (str(source), str(reference))
            for source, reference in product(sources, references)
        )

    print(f"Sources: {len(sources)}")
    print(f"References: {len(references)}")
    print(f"Pairs: {len(sources) * len(references)}")
    print(f"CSV: {output_csv}")


if __name__ == "__main__":
    main()
