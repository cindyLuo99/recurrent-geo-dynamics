#!/usr/bin/env python
"""Extract activations from UNTRAINED (random-weight) PyTorch models,
across multiple random seeds.

Supplemental control: how much prototype geometry exists from architecture,
preprocessing, random initialization, and recurrent dynamics alone?

  python extract_untrained_torch.py \
      --models alexnet vgg16 resnet50 resnet101 lrm3 lra3 cornet_rt \
      --seeds 0 1 2 3 4 \
      --dataset /path/to/100x50_imageNet_val_random \
      --imagenet-index /path/to/imagenet_class_index.json \
      --output ./activations/untrained

Untrained ConvRNN and BL/B are extracted in the TF environment (tf_env/).
Conventions match the trained extraction: LRM3/LRA3 save pass 1 (baseline)
and pass 3 (after); CORnet-RT saves timesteps 3..6 in one dict per seed.
The seed is set immediately before model construction, so seed differences
reflect only the random initialization.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torchvision import models

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, REPO_ROOT)
from repgeo.data import load_imagenet_label_maps, create_dataloader
import extract_torch_models as trained  # reuse the extraction machinery

DEFAULT_IMAGENET_INDEX = os.path.join(REPO_ROOT, "data", "imagenet_class_index.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEEDFORWARD = {
    "alexnet":   (models.alexnet, models.AlexNet_Weights.DEFAULT, "alexnet", "fc7"),
    "vgg16":     (models.vgg16, models.VGG16_Weights.DEFAULT, "vgg16", "fc7"),
    "resnet50":  (models.resnet50, models.ResNet50_Weights.DEFAULT, "resnet50", "avgpool"),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, "resnet101", "avgpool"),
}

LRM_LRA_CFG = {
    "lrm3": dict(api="lrm", penult="feedforward.classifier.5",
                 logits="feedforward.classifier.6"),
    "lra3": dict(api="lra", penult="backbone.classifier.5",
                 logits="backbone.classifier.6"),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_untrained(name):
    """Construct a random-weight model. Call seed_everything first."""
    if name in FEEDFORWARD:
        ctor, _, _, _ = FEEDFORWARD[name]
        return ctor(weights=None)
    if name == "lrm3":
        # ':dev' is intentional — public repo, but only :dev constructs
        # alexnet_lrm3 correctly (main raises). See extract_torch_models.run_lrm_lra.
        model, _ = torch.hub.load("cindyluo99/lrm-steering:dev", "alexnet_lrm3",
                                  pretrained=False, steering=False)
        return model
    if name == "lra3":
        # not yet public — trained.load_lra3 explains (see README, 'Note on LRA3')
        return trained.load_lra3(pretrained=False)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True,
                        choices=list(FEEDFORWARD) + ["lrm3", "lra3", "cornet_rt"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--imagenet-index", default=DEFAULT_IMAGENET_INDEX)
    parser.add_argument("--cornet-dir", default="./CORnet",
                        help="Clone of https://github.com/dicarlolab/CORnet "
                             "(only needed for cornet_rt).")
    parser.add_argument("--output", default="./activations/untrained")
    parser.add_argument("--suffix", default="imagenetval_100x50")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    _, idx2number, _ = load_imagenet_label_maps(args.imagenet_index)

    # Check optional dependencies up front so a long multi-seed run can't
    # crash halfway through (e.g. the not-yet-public LRA3 model code).
    models = list(args.models)
    if "lra3" in models:
        try:
            trained.load_lra3(pretrained=False)
        except ImportError as e:
            print(f"\nSKIPPING lra3: {e}\n")
            models.remove("lra3")

    for seed in args.seeds:
        print(f"\n--- seed {seed} ---")
        for name in models:
            if name in FEEDFORWARD:
                _, weights, out_name, penult_name = FEEDFORWARD[name]
                seed_everything(seed)
                model = build_untrained(name)
                loader = create_dataloader(args.dataset, weights.transforms(),
                                           idx2number, args.batch_size, args.num_workers)
                logits, penult, labels, acc = trained.extract_feedforward(model, loader)
                out_dir = os.path.join(args.output, out_name)
                os.makedirs(out_dir, exist_ok=True)
                D = penult.shape[1]
                tag = f"seed{seed}_{args.suffix}"
                torch.save(logits, f"{out_dir}/{out_name}_logits_{tag}.pt")
                torch.save(penult, f"{out_dir}/{out_name}_{penult_name}_{D}_{tag}.pt")
                torch.save(labels, f"{out_dir}/{out_name}_labels_{tag}.pt")
                print(f"  [{name:9s} seed={seed}] penult {tuple(penult.shape)}, "
                      f"top-1={acc:.2f}% (chance ~0.1%)")

            elif name in ("lrm3", "lra3"):
                cfg = LRM_LRA_CFG[name]
                seed_everything(seed)
                model = build_untrained(name)
                # same preprocessing family as the trained LRM/LRA extraction
                transform = models.AlexNet_Weights.DEFAULT.transforms()
                loader = create_dataloader(args.dataset, transform, idx2number,
                                           args.batch_size, args.num_workers)
                out_dir = os.path.join(args.output, name)
                os.makedirs(out_dir, exist_ok=True)
                for steps in (1, 3):
                    buf, labels, acc = trained._extract_recurrent_pass(
                        model, loader, cfg["penult"], cfg["logits"], steps, cfg["api"])
                    penult = buf[cfg["penult"]]
                    tag = f"pass{steps}_seed{seed}_{args.suffix}"
                    torch.save(penult, f"{out_dir}/{name}_fc7_{penult.shape[1]}_{tag}.pt")
                    torch.save(buf[cfg["logits"]], f"{out_dir}/{name}_logits_{tag}.pt")
                    torch.save(labels, f"{out_dir}/{name}_labels_{tag}.pt")
                    print(f"  [{name:9s} seed={seed} pass={steps}] "
                          f"penult {tuple(penult.shape)}, top-1={acc:.2f}%")

            elif name == "cornet_rt":
                seed_everything(seed)
                model = trained.load_dicarlolab_cornet_rt(
                    args.cornet_dir, pretrained=False, times=7)
                loader = create_dataloader(args.dataset, trained.cornet_transform(),
                                           idx2number, args.batch_size, args.num_workers)
                feats, logits, labels_all, _ = trained.extract_cornet_states(
                    model, loader, t_select=(3, 4, 5, 6), times=7)
                out_dir = os.path.join(args.output, "CORnet-RT")
                os.makedirs(out_dir, exist_ok=True)
                tag = f"seed{seed}_{args.suffix}"
                torch.save(feats, f"{out_dir}/activations_tmax7_feats512_{tag}.pt")
                torch.save(logits, f"{out_dir}/activations_tmax7_logits1k_{tag}.pt")
                torch.save(labels_all, f"{out_dir}/labels_tmax7_{tag}.pt")
                print(f"  [cornet_rt seed={seed}] saved t=[3, 4, 5, 6]")


if __name__ == "__main__":
    main()
