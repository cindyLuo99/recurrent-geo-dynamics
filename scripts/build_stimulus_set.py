#!/usr/bin/env python
"""Reconstruct the exact 100-class x 50-image ImageNet-val stimulus set.

ImageNet cannot be redistributed, so this repo ships only a manifest
(data/stimulus_set_manifest.csv: WordNet_ID, filename) listing the 5000
validation images used in the paper. Given your own copy of the ImageNet
(ILSVRC2012) validation set, this script copies them into the ImageFolder
layout the extraction scripts expect.

  python build_stimulus_set.py \
      --imagenet-val /path/to/ILSVRC2012/val \
      --manifest ../data/stimulus_set_manifest.csv \
      --output ./data/100x50_imageNet_val_random

--imagenet-val may point at either a flat directory of
ILSVRC2012_val_*.JPEG files or a directory already organized into
WordNet-ID subfolders.
"""

import argparse
import csv
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imagenet-val", required=True,
                        help="Your local ImageNet validation images.")
    parser.add_argument("--manifest",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "..", "data", "stimulus_set_manifest.csv"))
    parser.add_argument("--output", default="./data/100x50_imageNet_val_random")
    args = parser.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Manifest: {len(rows)} images, "
          f"{len(set(r['WordNet_ID'] for r in rows))} classes")

    missing = []
    for r in rows:
        wnid, fname = r["WordNet_ID"], r["filename"]
        candidates = [
            os.path.join(args.imagenet_val, fname),          # flat layout
            os.path.join(args.imagenet_val, wnid, fname),    # class-folder layout
        ]
        src = next((c for c in candidates if os.path.exists(c)), None)
        if src is None:
            missing.append(fname)
            continue
        dst_dir = os.path.join(args.output, wnid)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, fname))

    if missing:
        sys.exit(f"ERROR: {len(missing)} images not found under "
                 f"{args.imagenet_val} (first: {missing[0]}). "
                 "Check the path points at the ILSVRC2012 validation images.")
    print(f"Done: stimulus set written to {args.output}")


if __name__ == "__main__":
    main()
