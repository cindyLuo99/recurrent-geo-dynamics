"""Untrained (random-weight) controls, multi-seed.

The supplement asks how much of the prototype geometry is present from
architecture, preprocessing, random initialization, and recurrent dynamics
alone, before any supervised training. Extraction (see
scripts/extraction/extract_untrained_torch.py and the tf_env scripts) saves
one file per (model, seed); this module loads them into

    {seed: {model: {"logits": {state: T}, "penult": {state: T}}}}

— the trained features dict with one extra seed level — and computes the
same geometry objects per seed so figures can show mean ± seed variability.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DEFAULT_REP, REP_OVERRIDES
from .geometry import to_feat, compute_prototypes, exemplar_proto_dists, \
    rdm_cosine, rdm_vec_upper, pair_geometry_metrics

DEFAULT_SEEDS = (0, 1, 2, 3, 4)

# File-naming templates per model. {s} = seed, {x} = dataset suffix,
# {p} = pass number, {t} = timestep key.
UNTRAINED_LOAD_CFG = {
    # feedforward (single state)
    "AlexNet": {
        "dir": "alexnet", "kind": "feedforward",
        "logits": "alexnet_logits_seed{s}_{x}.pt",
        "penult": "alexnet_fc7_4096_seed{s}_{x}.pt",
        "labels": "alexnet_labels_seed{s}_{x}.pt",
    },
    "VGG16": {
        "dir": "vgg16", "kind": "feedforward",
        "logits": "vgg16_logits_seed{s}_{x}.pt",
        "penult": "vgg16_fc7_4096_seed{s}_{x}.pt",
        "labels": "vgg16_labels_seed{s}_{x}.pt",
    },
    "ResNet50": {
        "dir": "resnet50", "kind": "feedforward",
        "logits": "resnet50_logits_seed{s}_{x}.pt",
        "penult": "resnet50_avgpool_2048_seed{s}_{x}.pt",
        "labels": "resnet50_labels_seed{s}_{x}.pt",
    },
    "ResNet101": {
        "dir": "resnet101", "kind": "feedforward",
        "logits": "resnet101_logits_seed{s}_{x}.pt",
        "penult": "resnet101_avgpool_2048_seed{s}_{x}.pt",
        "labels": "resnet101_labels_seed{s}_{x}.pt",
    },

    # recurrent, saved per pass
    "LRM3": {
        "dir": "lrm3", "kind": "feedback_pass",
        "logits": "lrm3_logits_pass{p}_seed{s}_{x}.pt",
        "penult": "lrm3_fc7_4096_pass{p}_seed{s}_{x}.pt",
        "labels": "lrm3_labels_pass1_seed{s}_{x}.pt",
        "passes": {"baseline": 1, "after": 3},
    },
    "LRA3": {
        "dir": "lra3", "kind": "feedback_pass",
        "logits": "lra3_logits_pass{p}_seed{s}_{x}.pt",
        "penult": "lra3_fc7_4096_pass{p}_seed{s}_{x}.pt",
        "labels": "lra3_labels_pass1_seed{s}_{x}.pt",
        "passes": {"baseline": 1, "after": 3},
    },

    # CORnet-RT: one file per seed holding {timestep: tensor} dicts
    "CORnet-RT": {
        "dir": "CORnet-RT", "kind": "cornet_dict",
        "logits": "activations_tmax7_logits1k_seed{s}_{x}.pt",
        "penult": "activations_tmax7_feats512_seed{s}_{x}.pt",
        "labels": "labels_tmax7_seed{s}_{x}.pt",
        "timesteps": {"baseline": 3, "after": 4},
    },

    # TF models: penultimate saved [N, H, W, C], pooled on load
    "B": {
        "dir": "b", "kind": "feedforward",
        "logits": "b_imagenet_logits_seed{s}_{x}.pt",
        "penult": "b_imagenet_ReLU_Layer_6_seed{s}_{x}.pt",
        "labels": "b_imagenet_labels_seed{s}_{x}.pt",
    },
    "BL": {
        "dir": "bl", "kind": "rnn_timestep",
        "logits": "bl_imagenet_logits_{t}_seed{s}_{x}.pt",
        "penult": "bl_imagenet_ReLU_Layer_6_{t}_seed{s}_{x}.pt",
        "labels": "bl_imagenet_labels_seed{s}_{x}.pt",
        "states": {"baseline": "t0", "after": "tlast"},
    },
    "ConvRNN": {
        "dir": "convrnn", "kind": "rnn_timestep",
        "logits": "rgc_intermediate_logits_{t}_seed{s}_{x}.pt",
        "penult": "rgc_intermediate_conv10_{t}_seed{s}_{x}.pt",
        "labels": "rgc_intermediate_labels_seed{s}_{x}.pt",
        "states": {"baseline": "tprev", "after": "tlast"},
    },
}


def _try_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False) \
        if path.exists() else None


def load_untrained_features(root, suffix="imagenetval_100x50",
                            seeds=DEFAULT_SEEDS, cfg=None):
    """Load every available (model, seed) into the seeded features dict.

    Missing files are skipped (and reported), so partial extractions still
    load. Label vectors must be identical across all loaded files — this
    guards the geometry analysis against image-order mismatches.

    Returns: (results, labels, missing) where missing lists skipped combos.
    """
    cfg = cfg or UNTRAINED_LOAD_CFG
    root = Path(root)
    results = {s: {} for s in seeds}
    labels_ref = None
    missing = []

    for seed in seeds:
        for model_name, mc in cfg.items():
            mdir = root / mc["dir"]
            feats = {"logits": {}, "penult": {}}
            ok = True

            def _load(template, **fmt):
                return _try_load(mdir / template.format(s=seed, x=suffix, **fmt))

            if mc["kind"] == "feedforward":
                lt, pt = _load(mc["logits"]), _load(mc["penult"])
                if lt is None or pt is None:
                    ok = False
                else:
                    feats["logits"]["baseline"] = lt
                    feats["penult"]["baseline"] = to_feat(pt)

            elif mc["kind"] == "feedback_pass":
                for state, p in mc["passes"].items():
                    lt, pt = _load(mc["logits"], p=p), _load(mc["penult"], p=p)
                    if lt is None or pt is None:
                        ok = False
                        break
                    feats["logits"][state] = lt
                    feats["penult"][state] = to_feat(pt)

            elif mc["kind"] == "cornet_dict":
                logits_dict, feats_dict = _load(mc["logits"]), _load(mc["penult"])
                if logits_dict is None or feats_dict is None:
                    ok = False
                else:
                    for state, t in mc["timesteps"].items():
                        if t not in logits_dict or t not in feats_dict:
                            ok = False
                            break
                        feats["logits"][state] = logits_dict[t]
                        feats["penult"][state] = to_feat(feats_dict[t])

            elif mc["kind"] == "rnn_timestep":
                for state, t in mc["states"].items():
                    lt, pt = _load(mc["logits"], t=t), _load(mc["penult"], t=t)
                    if lt is None or pt is None:
                        ok = False
                        break
                    feats["logits"][state] = lt
                    feats["penult"][state] = to_feat(pt)

            if not ok:
                missing.append((model_name, seed))
                continue

            label_t = _load(mc["labels"])
            if label_t is None:
                missing.append((model_name, seed))
                continue
            if labels_ref is None:
                labels_ref = label_t
            else:
                assert torch.equal(labels_ref, label_t), \
                    f"label order mismatch: {model_name} seed={seed}"
            results[seed][model_name] = feats

    if missing:
        print(f"[load_untrained] skipped {len(missing)} (model, seed) combos "
              f"with missing files: {sorted(set(m for m, _ in missing))}")
    return results, labels_ref, missing


def compute_seeded_between_dict(seeded_features, labels, rep="logits",
                                center=False):
    """{tag: [per-seed RDM upper-triangle vectors]}.

    center=True subtracts the grand-mean prototype first — this removes the
    common direction shared by all classes and asks whether class-specific
    deviations carry geometry even when raw prototypes look collapsed.
    """
    labels_t = torch.as_tensor(labels)
    classes = torch.unique(labels_t, sorted=True).tolist()
    out = {}
    for seed, models_dict in seeded_features.items():
        for model_name, reps in models_dict.items():
            for state, X in (reps.get(rep) or {}).items():
                if X is None:
                    continue
                prot = compute_prototypes(X, labels_t, classes)
                if center:
                    prot = prot - prot.mean(0, keepdim=True)
                out.setdefault(f"{model_name} {state}", []).append(
                    rdm_vec_upper(rdm_cosine(prot)))
    return out


def compute_seeded_within_dict(seeded_features, labels, rep="logits"):
    """{tag: [per-seed exemplar->own-prototype distance arrays]}."""
    labels_t = torch.as_tensor(labels)
    classes = torch.unique(labels_t, sorted=True).tolist()
    out = {}
    for seed, models_dict in seeded_features.items():
        for model_name, reps in models_dict.items():
            for state, X in (reps.get(rep) or {}).items():
                if X is None:
                    continue
                prot = compute_prototypes(X, labels_t, classes)
                out.setdefault(f"{model_name} {state}", []).append(
                    exemplar_proto_dists(X, prot, labels_t, metric="cosine"))
    return out


METRICS = ["Δ Cluster Size", "Δ Cluster Size (%)", "Avg Proto Shift",
           "Δ Between-Proto (%)", "Proto RDM ρ"]


def geometry_summary_seeded(seeded_features, labels, models=None):
    """Per-seed geometry summary, aggregated as mean ± SD across seeds.

    Uses pair_geometry_metrics — the identical computation as the trained
    geometry_summary_table — so trained and untrained tables cannot drift.

    Returns (df_long, df_summary): raw per-seed rows and formatted
    "mean ± SD" rows per model.
    """
    from .config import RECURRENT_MODELS
    models = models or RECURRENT_MODELS

    rows = []
    for seed, models_dict in seeded_features.items():
        for model in models:
            rep = REP_OVERRIDES.get(model, DEFAULT_REP)
            d = models_dict.get(model, {}).get(rep, {})
            if "baseline" not in d or "after" not in d:
                continue
            rows.append({"seed": seed, "Model": model, "Rep": rep,
                         **pair_geometry_metrics(d["baseline"], d["after"], labels)})

    df_long = pd.DataFrame(rows)
    if df_long.empty:
        return df_long, df_long

    summary_rows = []
    for (model, rep), g in df_long.groupby(["Model", "Rep"], sort=False):
        row = {"Model": model, "Rep": rep, "n_seeds": len(g)}
        for m in METRICS:
            mu, sd = g[m].mean(), g[m].std()
            if "(%)" in m:
                row[m] = f"{mu:+.2f} ± {sd:.2f}"
            elif m == "Proto RDM ρ":
                row[m] = f"{mu:.3f} ± {sd:.3f}"
            else:
                row[m] = f"{mu:+.4f} ± {sd:.4f}"
        summary_rows.append(row)
    return df_long, pd.DataFrame(summary_rows)
