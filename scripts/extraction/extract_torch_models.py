#!/usr/bin/env python
"""Extract logits + penultimate activations for all PyTorch models.

Runs in the main (PyTorch) environment. ConvRNN and BL/B require the
TensorFlow environment — see tf_env/.

Examples:
  python extract_torch_models.py --model alexnet \
      --dataset /path/to/100x50_imageNet_val_random \
      --imagenet-index /path/to/imagenet_class_index.json \
      --output ./activations

  python extract_torch_models.py --model lrm3 ...      # recurrent, passes 1 & 3
  python extract_torch_models.py --model cornet_rt ...  # timesteps 3..6

Output naming convention (consumed by repgeo.loading.load_features):
  <output>/<model_dir>/<model>_{logits,<penult-name>}[_passK|_tK]_<suffix>.pt
  plus a matching *_labels_*.pt file.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, REPO_ROOT)
from repgeo.data import load_imagenet_label_maps, create_dataloader

DEFAULT_IMAGENET_INDEX = os.path.join(REPO_ROOT, "data", "imagenet_class_index.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TORCHVISION_MODELS = {
    "alexnet":   (models.alexnet, models.AlexNet_Weights.DEFAULT, "vanilla_alexnet", "fc7"),
    "vgg16":     (models.vgg16, models.VGG16_Weights.DEFAULT, "vgg16", "fc7"),
    "resnet50":  (models.resnet50, models.ResNet50_Weights.DEFAULT, "resnet50", "avgpool"),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, "resnet101", "avgpool"),
}


def _find_final_linear(model: nn.Module, n_classes: int = 1000):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == n_classes:
            last = m
    if last is None:
        raise RuntimeError(f"No final nn.Linear(..., {n_classes}) found.")
    return last


@torch.inference_mode()
def extract_feedforward(model, loader):
    """Logits + input to the final linear layer (penultimate), via forward hook."""
    model.eval().to(device)
    bucket = []
    handle = _find_final_linear(model).register_forward_hook(
        lambda _m, inp, _out: bucket.append(inp[0].detach().cpu()))

    logits_all, penult_all, labels_all = [], [], []
    correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(imgs)
        logits_all.append(out.detach().cpu())
        penult_all.append(bucket.pop())
        labels_all.append(labels.detach().cpu())
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)

    handle.remove()
    return (torch.cat(logits_all), torch.cat(penult_all),
            torch.cat(labels_all), 100.0 * correct / total)


def save_set(out_dir, prefix, logits, penult, labels, penult_name, suffix, acc):
    os.makedirs(out_dir, exist_ok=True)
    D = penult.shape[1]
    torch.save(logits, f"{out_dir}/{prefix}_logits_{suffix}.pt")
    torch.save(penult, f"{out_dir}/{prefix}_{penult_name}_{D}_{suffix}.pt")
    torch.save(labels, f"{out_dir}/{prefix}_labels_{suffix}.pt")
    print(f"OK {prefix}: logits {tuple(logits.shape)}, penult {tuple(penult.shape)}, "
          f"top-1={acc:.2f}% -> {out_dir}")


# ---------------------------------------------------------------- torchvision
def run_torchvision(name, args, idx2number):
    ctor, weights, out_name, penult_name = TORCHVISION_MODELS[name]
    model = ctor(weights=weights)
    loader = create_dataloader(args.dataset, weights.transforms(), idx2number,
                               args.batch_size, args.num_workers)
    logits, penult, labels, acc = extract_feedforward(model, loader)
    save_set(os.path.join(args.output, out_name), out_name,
             logits, penult, labels, penult_name, args.suffix, acc)


# ------------------------------------------------------------ robust ResNet50
def run_resnet50_robust(args, idx2number):
    """MadryLab Linf eps=8/255 adversarially trained ResNet-50 (HF hub)."""
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id="madrylab/robust-imagenet-models",
                                filename="resnet50_linf_eps8.0.ckpt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    for p in ["module.model.", "model.model.", "model.module.", "model.", "module."]:
        if any(k.startswith(p) for k in state):
            state = {k.replace(p, "", 1) if k.startswith(p) else k: v
                     for k, v in state.items()}

    model = models.resnet50(weights=None)
    model_keys = set(model.state_dict().keys())
    state = {k: v for k, v in state.items() if k in model_keys}
    missing, unexpected = model.load_state_dict(state, strict=False)
    loadable = [k for k in missing if not k.endswith("num_batches_tracked")]
    if loadable or unexpected:
        raise RuntimeError(f"robust ckpt mismatch: missing={loadable[:5]}, "
                           f"unexpected={list(unexpected)[:5]}")

    transform = models.ResNet50_Weights.DEFAULT.transforms()
    loader = create_dataloader(args.dataset, transform, idx2number,
                               args.batch_size, args.num_workers)
    logits, penult, labels, acc = extract_feedforward(model, loader)
    save_set(os.path.join(args.output, "resnet50_robust"), "resnet50_robust",
             logits, penult, labels, "avgpool", args.suffix, acc)


# ----------------------------------------------------------------------- CLIP
@torch.inference_mode()
def run_clip(args, idx2number, number2label):
    """CLIP ViT-B/32 image embeddings + zero-shot ImageNet logits."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)

    classnames = [number2label[i] for i in range(len(number2label))]
    texts = [f"a photo of a {n.replace('_', ' ')}" for n in classnames]
    text_features = model.encode_text(tokenizer(texts).to(device))
    text_features = F.normalize(text_features, dim=-1)

    loader = create_dataloader(args.dataset, preprocess, idx2number,
                               args.batch_size, args.num_workers)
    logits_all, feats_all, labels_all = [], [], []
    correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        img_feats = F.normalize(model.encode_image(imgs), dim=-1)
        logits = model.logit_scale.exp() * img_feats @ text_features.T
        logits_all.append(logits.detach().cpu())
        feats_all.append(img_feats.detach().cpu())
        labels_all.append(labels.detach().cpu())
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    logits_all = torch.cat(logits_all)
    feats_all = torch.cat(feats_all)
    labels_all = torch.cat(labels_all)
    out_dir = os.path.join(args.output, "clip")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(logits_all, f"{out_dir}/clip_logits_{args.suffix}.pt")
    torch.save(feats_all, f"{out_dir}/clip_image_{feats_all.shape[1]}_{args.suffix}.pt")
    torch.save(labels_all, f"{out_dir}/clip_labels_{args.suffix}.pt")
    print(f"OK clip: feats {tuple(feats_all.shape)}, zero-shot top-1="
          f"{100.0 * correct / total:.2f}%")


# ------------------------------------------------------------------ LRM / LRA
@torch.inference_mode()
def _extract_recurrent_pass(model, loader, penult_layer, logits_layer, steps, api):
    """Last-timestep penultimate + logits for LRM/LRA at a given pass count."""
    from lrm_steering.lrm_models.feature_extractor import FeatureExtractor

    names = [n for n, _ in model.named_modules()]
    for L in (penult_layer, logits_layer):
        if L not in names:
            raise ValueError(f"Layer '{L}' not found in model.")

    model.to(device).eval()
    extractor = FeatureExtractor(model, [penult_layer, logits_layer], device=device)
    buf = {penult_layer: [], logits_layer: []}
    labels_all = []
    correct = total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with extractor as ex:
            if api == "lrm":
                out = model(imgs, forward_passes=steps, drop_state=True)
            else:  # lra
                out = model(imgs, time_steps=steps, drop_state=True)
            logits_last = out[-1] if isinstance(out, (list, tuple)) else out
        for L in buf:
            buf[L].append(ex._features[L][-1].detach().cpu())  # last timestep
        labels_all.append(labels.detach().cpu())
        correct += (logits_last.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return ({L: torch.cat(v) for L, v in buf.items()},
            torch.cat(labels_all), 100.0 * correct / total)


def load_lra3(pretrained=True):
    """LRA3 — NOTE: the model definition is not yet public.

    It lives in a not-yet-released part of the lrm-steering codebase
    (lrm_steering.neurips2023.models). All LRA3 *analyses* remain fully
    reproducible from the released activation bundle; only re-extraction
    from scratch needs this module. See the README's "Note on LRA3".
    """
    try:
        from lrm_steering.neurips2023.models import alexnet_lra3
    except ImportError as exc:
        raise ImportError(
            "The LRA3 model definition is not yet public, so LRA3 cannot be "
            "re-extracted from public code. LRA3 activations are included in "
            "the released activation bundle, and every LRA3 analysis runs "
            "from those (see README, 'Note on LRA3').") from exc
    return alexnet_lra3(pretrained=pretrained)


def run_lrm_lra(key, args, idx2number):
    cfg = {
        "lrm3": dict(api="lrm", penult="feedforward.classifier.5",
                     logits="feedforward.classifier.6"),
        "lra3": dict(api="lra", penult="backbone.classifier.5",
                     logits="backbone.classifier.6"),
    }[key]

    if key == "lrm3":
        model, tfs = torch.hub.load("cindyluo99/lrm-steering:dev", "alexnet_lrm3",
                                    pretrained=True, steering=False)
        transform = tfs["val_transform"]
    else:
        model = load_lra3(pretrained=True)
        _, tfs = torch.hub.load("cindyluo99/lrm-steering:dev", "alexnet_lrm3",
                                pretrained=True, steering=False)
        transform = tfs["val_transform"]

    loader = create_dataloader(args.dataset, transform, idx2number,
                               args.batch_size, args.num_workers)
    out_dir = os.path.join(args.output, key)
    for steps in (1, 3):
        buf, labels, acc = _extract_recurrent_pass(
            model, loader, cfg["penult"], cfg["logits"], steps, cfg["api"])
        penult = buf[cfg["penult"]]
        os.makedirs(out_dir, exist_ok=True)
        torch.save(penult, f"{out_dir}/{key}_fc7_{penult.shape[1]}_pass{steps}_{args.suffix}.pt")
        torch.save(buf[cfg["logits"]], f"{out_dir}/{key}_logits_pass{steps}_{args.suffix}.pt")
        torch.save(labels, f"{out_dir}/{key}_labels_pass{steps}_{args.suffix}.pt")
        print(f"OK {key} pass{steps}: penult {tuple(penult.shape)}, top-1={acc:.2f}%")


# ------------------------------------------------------------------ CORnet-RT
def load_dicarlolab_cornet_rt(cornet_dir, pretrained=True, times=7):
    """Load CORnet-RT from a clone of the ORIGINAL dicarlolab/CORnet repo.

    Returns the bare CORnet_RT module (DataParallel unwrapped, so forward
    hooks behave predictably on a single device).
    """
    sys.path.insert(0, cornet_dir)
    try:
        from cornet import cornet_rt
    except ImportError as exc:
        raise ImportError(
            f"Could not import `cornet` from --cornet-dir={cornet_dir}. "
            "Clone the original repo first:\n"
            "  git clone https://github.com/dicarlolab/CORnet\n"
            "and pass --cornet-dir ./CORnet") from exc

    model = cornet_rt(pretrained=pretrained, map_location="cpu", times=times)
    return getattr(model, "module", model)  # unwrap DataParallel


@torch.inference_mode()
def extract_cornet_states(model, loader, t_select=(3, 4, 5, 6), times=7):
    """Per-timestep IT features (512-d, avg-pooled) and decoder logits.

    The upstream model only returns final-step logits, but its IT block runs
    once per timestep, so a forward hook on model.IT collects the IT state
    at every t (hook call i == timestep i, zero-indexed). Per-timestep
    logits come from passing each IT map through the model's own decoder.
    """
    model.to(device).eval()
    it_states = []
    hook = model.IT.register_forward_hook(
        lambda _m, _inp, out: it_states.append(out[0]))  # forward returns (output, state)

    feats = {t: [] for t in t_select}
    logits = {t: [] for t in t_select}
    labels_all = []
    correct = total = 0
    try:
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            it_states.clear()
            logits_last = model(imgs)
            if len(it_states) != times:
                raise RuntimeError(
                    f"Expected {times} IT timesteps, hook saw {len(it_states)}.")
            correct += (logits_last.argmax(1) == labels).sum().item()
            total += labels.size(0)
            for t in t_select:
                it_map = it_states[t]
                feats[t].append(F.adaptive_avg_pool2d(it_map, 1).flatten(1).cpu())
                logits[t].append(model.decoder(it_map).cpu())
            labels_all.append(labels.detach().cpu())
    finally:
        hook.remove()

    return ({t: torch.cat(v) for t, v in feats.items()},
            {t: torch.cat(v) for t, v in logits.items()},
            torch.cat(labels_all), 100.0 * correct / max(total, 1))


def cornet_transform():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_cornet_rt(args, idx2number, t_select=(3, 4, 5, 6), times=7):
    """CORnet-RT per-timestep extraction. t=3 is the baseline state, t=4
    'after one recurrent step' — the paper's convention."""
    model = load_dicarlolab_cornet_rt(args.cornet_dir, pretrained=True, times=times)
    loader = create_dataloader(args.dataset, cornet_transform(), idx2number,
                               args.batch_size, args.num_workers)
    feats, logits, labels_all, acc = extract_cornet_states(model, loader, t_select, times)

    out_dir = os.path.join(args.output, "CORnet-RT")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(feats, f"{out_dir}/activations_tmax{times}_feats512_{args.suffix}.pt")
    torch.save(logits, f"{out_dir}/activations_tmax{times}_logits1k_{args.suffix}.pt")
    torch.save(labels_all, f"{out_dir}/labels_tmax{times}_{args.suffix}.pt")
    print(f"OK CORnet-RT: t={list(feats)}, final top-1={acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        choices=list(TORCHVISION_MODELS) +
                        ["resnet50_robust", "clip", "lrm3", "lra3", "cornet_rt", "all"])
    parser.add_argument("--dataset", required=True, help="Stimulus-set root (ImageFolder).")
    parser.add_argument("--imagenet-index", default=DEFAULT_IMAGENET_INDEX,
                        help="Path to imagenet_class_index.json "
                             "(default: the copy in this repo's data/).")
    parser.add_argument("--output", default="./activations")
    parser.add_argument("--suffix", default="imagenetval_100x50",
                        help="Dataset tag used in output filenames.")
    parser.add_argument("--cornet-dir", default="./CORnet",
                        help="Clone of https://github.com/dicarlolab/CORnet "
                             "(only needed for --model cornet_rt).")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    _, idx2number, number2label = load_imagenet_label_maps(args.imagenet_index)

    requested = (list(TORCHVISION_MODELS) +
                 ["resnet50_robust", "clip", "lrm3", "lra3", "cornet_rt"]
                 if args.model == "all" else [args.model])
    skipped = []
    for name in requested:
        try:
            if name in TORCHVISION_MODELS:
                run_torchvision(name, args, idx2number)
            elif name == "resnet50_robust":
                run_resnet50_robust(args, idx2number)
            elif name == "clip":
                run_clip(args, idx2number, number2label)
            elif name in ("lrm3", "lra3"):
                run_lrm_lra(name, args, idx2number)
            elif name == "cornet_rt":
                run_cornet_rt(args, idx2number)
        except ImportError as e:
            # With --model all, a missing optional dependency (e.g. the
            # not-yet-public LRA3 code) shouldn't abort the whole batch.
            if args.model != "all":
                raise
            print(f"\nSKIPPING {name}: {e}\n")
            skipped.append(name)

    if skipped:
        print(f"Done, but skipped: {skipped} (see messages above).")


if __name__ == "__main__":
    main()
