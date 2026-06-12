#!/usr/bin/env python
"""Extract BL / B (rcnn-sat, Spoerer et al. 2020) activations.

Runs in the TensorFlow 1.x environment (see requirements_tf.txt), NOT the
main PyTorch environment. Torch (CPU is fine) is still required for the
dataloader and for saving .pt files.

  python extract_rcnn_sat.py --model bl \
      --rcnn_sat_dir /path/to/rcnn-sat \
      --dataset /path/to/100x50_imageNet_val_random \
      --imagenet-index /path/to/imagenet_class_index.json \
      --output ./activations

BL timesteps (zero-indexed, n_timesteps=8):
  t0=0 purely feedforward sweep, tmid=3, tprev=6, tlast=7 (fully recurred;
  the readout BL was trained on). The paper uses t0 as baseline and tlast
  as after.
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TVF

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, REPO_ROOT)
from repgeo.data import load_imagenet_label_maps, create_dataloader

INPUT_SIZE = 128
OUT_LAYER = "ReLU_Layer_6"  # last RCL output; (B, 2, 2, 2048) at 128x128
OSF_URLS = {  # OSF project mz9hw (Spoerer et al.)
    "bl_imagenet": "https://osf.io/download/7jrvm/",
    "b_imagenet": "https://osf.io/download/bxw65/",
}


def build_loader(args, idx2number):
    def _center_square_crop(img):
        return TVF.center_crop(img, min(img.size))

    transform = transforms.Compose([
        transforms.Lambda(_center_square_crop),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    return create_dataloader(args.dataset, transform, idx2number,
                             args.batch_size, args.num_workers)


def get_weights(rcnn_sat_dir, model_name):
    weights_dir = os.path.join(rcnn_sat_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    path = os.path.join(weights_dir, f"{model_name}.h5")
    if not os.path.exists(path):
        print(f"Downloading {model_name}.h5 from OSF -> {path}")
        urllib.request.urlretrieve(OSF_URLS[model_name], path)
    return path


def to_model_input(imgs):
    """Torch [B,3,H,W] in [0,1] -> numpy NHWC in [-1,1] (rcnn-sat convention)."""
    return imgs.numpy().transpose(0, 2, 3, 1).astype(np.float32) * 2.0 - 1.0


def run_bl(args, idx2number, tf):
    from rcnn_sat import bl_net

    n_timesteps = 8
    timesteps = {"t0": 0, "tmid": 3, "tprev": 6, "tlast": 7}

    input_layer = tf.keras.layers.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = bl_net(input_layer, classes=1000, n_timesteps=n_timesteps,
                   cumulative_readout=False)
    model.load_weights(get_weights(args.rcnn_sat_dir, "bl_imagenet"))

    # Per timestep: logits = pre-softmax dense input, penult = last RCL output.
    # Note upstream typo: softmax layers are named 'Sotfmax_Time_{t}' (sic).
    extract = []
    for t in timesteps.values():
        extract.append(model.get_layer(f"Sotfmax_Time_{t}").input)
        extract.append(model.get_layer(f"{OUT_LAYER}_Time_{t}").output)
    get_acts = tf.keras.backend.function([model.input], extract)

    results = {k: {"logits": [], "penult": []} for k in timesteps}
    labels_all = []
    correct = total = 0
    tlast_idx = list(timesteps).index("tlast") * 2

    for imgs, labels in build_loader(args, idx2number):
        out = get_acts([to_model_input(imgs)])
        for i, key in enumerate(timesteps):
            results[key]["logits"].append(torch.from_numpy(np.asarray(out[i * 2])))
            results[key]["penult"].append(torch.from_numpy(np.asarray(out[i * 2 + 1])))
        labels_all.append(labels.detach().cpu())
        preds = np.asarray(out[tlast_idx]).argmax(1)
        correct += (preds == labels.numpy()).sum().item()
        total += labels.size(0)

    out_dir = os.path.join(args.output, "rcnn_sat")
    os.makedirs(out_dir, exist_ok=True)
    for key in timesteps:
        torch.save(torch.cat(results[key]["logits"]),
                   f"{out_dir}/bl_imagenet_logits_{key}_{args.suffix}.pt")
        torch.save(torch.cat(results[key]["penult"]),
                   f"{out_dir}/bl_imagenet_{OUT_LAYER}_{key}_{args.suffix}.pt")
    torch.save(torch.cat(labels_all), f"{out_dir}/bl_imagenet_labels_{args.suffix}.pt")
    torch.save({
        "timesteps": timesteps,
        "n_timesteps": n_timesteps,
        "cumulative_readout": False,
        "indexing": "zero-indexed; t0=0 (feedforward), tmid=3, tprev=6, tlast=7",
    }, f"{out_dir}/bl_imagenet_timesteps_{args.suffix}.pt")
    print(f"OK BL: acc(tlast)={100.0 * correct / max(total, 1):.2f}% -> {out_dir}")


def run_b(args, idx2number, tf):
    from rcnn_sat import b_net

    input_layer = tf.keras.layers.Input((INPUT_SIZE, INPUT_SIZE, 3))
    model = b_net(input_layer, classes=1000)
    model.load_weights(get_weights(args.rcnn_sat_dir, "b_imagenet"))

    # B has no timesteps; its softmax layer is named 'Softmax' (no typo).
    get_acts = tf.keras.backend.function(
        [model.input],
        [model.get_layer("Softmax").input, model.get_layer(OUT_LAYER).output])

    logits_all, penult_all, labels_all = [], [], []
    correct = total = 0
    for imgs, labels in build_loader(args, idx2number):
        logits, penult = get_acts([to_model_input(imgs)])
        logits_all.append(torch.from_numpy(np.asarray(logits)))
        penult_all.append(torch.from_numpy(np.asarray(penult)))
        labels_all.append(labels.detach().cpu())
        correct += (np.asarray(logits).argmax(1) == labels.numpy()).sum().item()
        total += labels.size(0)

    out_dir = os.path.join(args.output, "rcnn_sat")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.cat(logits_all), f"{out_dir}/b_imagenet_logits_{args.suffix}.pt")
    torch.save(torch.cat(penult_all), f"{out_dir}/b_imagenet_{OUT_LAYER}_{args.suffix}.pt")
    torch.save(torch.cat(labels_all), f"{out_dir}/b_imagenet_labels_{args.suffix}.pt")
    print(f"OK B: acc={100.0 * correct / max(total, 1):.2f}% -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["bl", "b", "both"])
    parser.add_argument("--rcnn_sat_dir", required=True, help="Path to rcnn-sat repo.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--imagenet-index",
                        default=os.path.join(REPO_ROOT, "data", "imagenet_class_index.json"))
    parser.add_argument("--output", default="./activations")
    parser.add_argument("--suffix", default="imagenetval_100x50")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, args.rcnn_sat_dir)
    import tensorflow as tf
    try:
        tf.enable_v2_behavior()
    except AttributeError:
        pass
    tf.keras.backend.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    _, idx2number, _ = load_imagenet_label_maps(args.imagenet_index)
    if args.model in ("bl", "both"):
        run_bl(args, idx2number, tf)
    if args.model in ("b", "both"):
        if args.model == "both":  # rebuild a clean session between models
            tf.keras.backend.clear_session()
            tf.keras.backend.set_session(tf.Session(config=config))
        run_b(args, idx2number, tf)


if __name__ == "__main__":
    main()
