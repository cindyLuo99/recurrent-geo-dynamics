"""Saved activation files -> the canonical `features` dict.

This module is pure I/O: it knows the file-naming convention of the released
activations and nothing about the analyses. The dict it returns:

    features = {
      "<Model>": {
        "logits": {"baseline": Tensor[N,1000], "after": Tensor[N,1000]},
        "penult": {"baseline": Tensor[N,D],    "after": Tensor[N,D]},
      },
      ...
    }

Feedforward models have only a "baseline" state; recurrent models have
"baseline" (early timestep / first pass) and "after" (late timestep /
last pass). CLIP has no ImageNet logits, only "penult" (image embedding).

You never need these loaders to analyze your OWN model — just build the
dict above from [N, D] tensors and hand it to repgeo.analysis. See
notebooks/analyze_your_own_model.ipynb.
"""

import os

import torch

from .geometry import to_feat

def _load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- loaders
# One small loader per model so a missing model can be skipped cleanly.

def _load_feedforward(root, suffix, dirname, prefix, penult_name):
    return {
        "logits": {"baseline": _load(f"{root}/{dirname}/{prefix}_logits_{suffix}.pt")},
        "penult": {"baseline": _load(f"{root}/{dirname}/{prefix}_{penult_name}_{suffix}.pt")},
    }


def _load_lrm_lra(root, suffix, key):
    return {
        "logits": {
            "baseline": _load(f"{root}/{key}/{key}_logits_pass1_{suffix}.pt"),
            "after":    _load(f"{root}/{key}/{key}_logits_pass3_{suffix}.pt"),
        },
        "penult": {
            "baseline": _load(f"{root}/{key}/{key}_fc7_4096_pass1_{suffix}.pt"),
            "after":    _load(f"{root}/{key}/{key}_fc7_4096_pass3_{suffix}.pt"),
        },
    }


def _load_cornet(root, suffix):
    # timesteps stacked in one file; t=3 baseline, t=4 after
    logits = _load(f"{root}/CORnet-RT/activations_tmax7_logits1k_{suffix}.pt")
    feats = _load(f"{root}/CORnet-RT/activations_tmax7_feats512_{suffix}.pt")
    return {
        "logits": {"baseline": logits[3], "after": logits[4]},
        "penult": {"baseline": feats[3], "after": feats[4]},
    }


def _load_convrnn(root, suffix):
    # rgc_intermediate: t=16 (tprev) baseline, t=17 (tlast) after
    cr, layer = "rgc_intermediate", "conv10"
    return {
        "logits": {
            "baseline": _load(f"{root}/convrnns/{cr}_logits_tprev_{suffix}.pt"),
            "after":    _load(f"{root}/convrnns/{cr}_logits_tlast_{suffix}.pt"),
        },
        "penult": {
            "baseline": to_feat(_load(f"{root}/convrnns/{cr}_{layer}_tprev_{suffix}.pt")),
            "after":    to_feat(_load(f"{root}/convrnns/{cr}_{layer}_tlast_{suffix}.pt")),
        },
    }


def _load_bl(root, suffix):
    # t=0 (pure feedforward sweep) baseline, t=7 after
    bl, layer = "bl_imagenet", "ReLU_Layer_6"
    return {
        "logits": {
            "baseline": _load(f"{root}/rcnn_sat/{bl}_logits_t0_{suffix}.pt"),
            "after":    _load(f"{root}/rcnn_sat/{bl}_logits_tlast_{suffix}.pt"),
        },
        "penult": {
            "baseline": to_feat(_load(f"{root}/rcnn_sat/{bl}_{layer}_t0_{suffix}.pt")),
            "after":    to_feat(_load(f"{root}/rcnn_sat/{bl}_{layer}_tlast_{suffix}.pt")),
        },
    }


def _load_b(root, suffix):
    b, layer = "b_imagenet", "ReLU_Layer_6"
    return {
        "logits": {"baseline": _load(f"{root}/rcnn_sat/{b}_logits_{suffix}.pt")},
        "penult": {"baseline": to_feat(_load(f"{root}/rcnn_sat/{b}_{layer}_{suffix}.pt"))},
    }


def _load_clip(root, suffix):
    # image embedding only (CLIP's zero-shot logits are not ImageNet logits)
    return {
        "logits": {},
        "penult": {"baseline": _load(f"{root}/clip/clip_image_512_{suffix}.pt")},
    }


MODEL_LOADERS = {
    "LRM3":      lambda r, s: _load_lrm_lra(r, s, "lrm3"),
    "LRA3":      lambda r, s: _load_lrm_lra(r, s, "lra3"),
    "BL":        _load_bl,
    "CORnet-RT": _load_cornet,
    "ConvRNN":   _load_convrnn,
    "AlexNet":   lambda r, s: _load_feedforward(r, s, "vanilla_alexnet", "vanilla_alexnet", "fc7_4096"),
    "B":         _load_b,
    "VGG16":     lambda r, s: _load_feedforward(r, s, "vgg16", "vgg16", "fc7_4096"),
    "ResNet50":  lambda r, s: _load_feedforward(r, s, "resnet50", "resnet50", "avgpool_2048"),
    "ResNet101": lambda r, s: _load_feedforward(r, s, "resnet101", "resnet101", "avgpool_2048"),
    "ResNet50-Robust": lambda r, s: _load_feedforward(
        r, s, "resnet50_robust", "resnet50_robust", "avgpool_2048"),
    "CLIP":      _load_clip,
}

# Any of these files supplies the shared labels vector (they are identical;
# the extraction scripts use the same dataloader with shuffle=False).
_LABEL_CANDIDATES = [
    "lrm3/lrm3_labels_pass1_{s}.pt",
    "lra3/lra3_labels_pass1_{s}.pt",
    "vanilla_alexnet/vanilla_alexnet_labels_{s}.pt",
    "vgg16/vgg16_labels_{s}.pt",
    "resnet50/resnet50_labels_{s}.pt",
    "convrnns/rgc_intermediate_labels_{s}.pt",
    "rcnn_sat/bl_imagenet_labels_{s}.pt",
    "CORnet-RT/labels_tmax7_{s}.pt",
    "clip/clip_labels_{s}.pt",
]


def load_features(root, suffix="imagenetval_100x50", models=None,
                  skip_missing=False):
    """Build the features dict from an activations directory.

    Args:
        root: directory holding one subfolder per model (see README for the
            expected file naming convention).
        suffix: dataset tag in every filename, e.g. "imagenetval_100x50".
        models: optional list of model names to load (default: all of
            MODEL_LOADERS). Useful if you only downloaded some models.
        skip_missing: if True, models whose files are missing are skipped
            with a warning instead of raising. Handy for partial downloads.

    Returns:
        (features, labels) — labels is the shared [N] tensor of class indices.
    """
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Activations directory not found: {root}\n"
            "Point this at the folder that contains one subfolder per model "
            "(lrm3/, vgg16/, convrnns/, ...). See the README's 'Activations' "
            "section for the download link and expected layout.")

    models = models or list(MODEL_LOADERS)
    unknown = [m for m in models if m not in MODEL_LOADERS]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. "
                         f"Available: {list(MODEL_LOADERS)}")

    features = {}
    missing = []
    for model in models:
        try:
            features[model] = MODEL_LOADERS[model](root, suffix)
        except FileNotFoundError as e:
            missing.append((model, str(e)))

    if missing:
        lines = "\n".join(f"  {m:16s} first missing file: {p}" for m, p in missing)
        msg = (f"Could not load {len(missing)} model(s):\n{lines}\n"
               f"Checks: (1) does --activations/root point at the right "
               f"directory? (2) does suffix='{suffix}' match your filenames? "
               f"(3) did the download finish? "
               f"Or load a subset: load_features(root, models=[...]) / "
               f"skip_missing=True.")
        if skip_missing and features:
            print(f"[load_features] WARNING — skipped: "
                  f"{[m for m, _ in missing]}")
        else:
            raise FileNotFoundError(msg)

    labels = None
    for cand in _LABEL_CANDIDATES:
        path = os.path.join(root, cand.format(s=suffix))
        if os.path.exists(path):
            labels = _load(path)
            break
    if labels is None:
        raise FileNotFoundError(
            f"No labels file found under {root} (looked for e.g. "
            f"{_LABEL_CANDIDATES[0].format(s=suffix)}).")

    return features, labels


def load_convrnn_states(root, suffix, states=("timgon", "tlast"), rep="logits"):
    """Load ConvRNN activations at named saved timesteps (for the supplement).

    states: any of 'timgon' (last input-driven step, t=12 in paper numbering),
    'tprev' (t=16), 'tlast' (t=17). Returns ({state: tensor}, labels). Penult
    tensors are spatially pooled to [N, C].
    """
    cr, layer = "rgc_intermediate", "conv10"
    name = "logits" if rep == "logits" else layer
    out = {}
    for s in states:
        t = _load(f"{root}/convrnns/{cr}_{name}_{s}_{suffix}.pt")
        out[s] = t if rep == "logits" else to_feat(t)
    labels = _load(f"{root}/convrnns/{cr}_labels_{suffix}.pt")
    return out, labels


def sanity_check(features, labels, verbose=True):
    """Assert every tensor has N rows matching labels; return list of problems."""
    N = len(labels)
    problems = []
    for model, reps in features.items():
        for rep, states in reps.items():
            for state, T in states.items():
                if T is None:
                    problems.append(f"[{model}][{rep}][{state}] is None")
                    continue
                if T.shape[0] != N:
                    problems.append(
                        f"[{model}][{rep}][{state}] N={T.shape[0]} != {N}")
                if verbose:
                    print(f"[{model:16s}][{rep:7s}][{state:8s}] -> {tuple(T.shape)}")
    if problems:
        raise ValueError("Feature sanity check failed:\n" + "\n".join(problems))
    if verbose:
        print(f"OK: {N} samples, {len(torch.unique(torch.as_tensor(labels)))} classes")
    return problems
