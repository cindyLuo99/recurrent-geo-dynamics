"""Descriptive analyses over the features dict: tables and per-tag summaries.

Module map (one rule per module):
  loading.py   saved activation files -> the features dict
  geometry.py  raw tensor(s) in -> numbers out
  analysis.py  features dict in -> tables / summaries out   (this module)
  stats.py     anything that returns a p-value
  plotting.py  figures

Every function here just iterates the features dict and applies the
tensor-level metrics from repgeo.geometry, so results are identical
whether you use these wrappers or call the metrics yourself.
"""

import pandas as pd
import torch

from .config import MODEL_ORDER, RECURRENT_MODELS, DEFAULT_REP, REP_OVERRIDES
from .geometry import (
    compute_prototypes, exemplar_proto_dists, rdm_cosine, rdm_vec_upper,
    effective_dimension, nearest_centroid_accuracy, logit_accuracy,
    pair_geometry_metrics, to_feat,
)


def get_pair(features, model, rep):
    """(X_baseline, X_after) for one model/rep, or (None, None) if absent."""
    rep_dict = features.get(model, {}).get(rep, {})
    if "baseline" not in rep_dict or "after" not in rep_dict:
        return None, None
    return (to_feat(rep_dict["baseline"]).float(),
            to_feat(rep_dict["after"]).float())

def iter_states(features, rep):
    """Yield (tag, model, state, X) for every available tensor of a rep."""
    for model_name, reps in features.items():
        rep_dict = reps.get(rep) or {}
        for state, X in rep_dict.items():
            if X is None:
                continue
            yield f"{model_name} {state}", model_name, state, X


def compute_cluster_sizes_and_rdms(features, labels, rep="logits", metric="cosine"):
    """Per-state exemplar->prototype distances and between-prototype RDM vectors.

    Returns:
        cluster_dists: {tag: [N] np.array}  (local cluster size)
        between_vecs:  {tag: [C*(C-1)/2] np.array}  (global separation)
    """
    labels = torch.as_tensor(labels)
    classes = torch.unique(labels, sorted=True).tolist()
    cluster_dists, between_vecs = {}, {}
    for tag, _, _, X in iter_states(features, rep):
        prot = compute_prototypes(X, labels, classes)
        cluster_dists[tag] = exemplar_proto_dists(X, prot, labels, metric=metric)
        between_vecs[tag] = rdm_vec_upper(rdm_cosine(prot))
    return cluster_dists, between_vecs


def build_prototype_rdms(features, labels, rep="logits"):
    """{tag: [C,C] cosine-distance RDM of class prototypes}."""
    labels = torch.as_tensor(labels)
    classes = torch.unique(labels, sorted=True).tolist()
    return {
        tag: rdm_cosine(compute_prototypes(X, labels, classes))
        for tag, _, _, X in iter_states(features, rep)
    }


def compute_effective_dims(features, rep="logits", labels=None, on_prototypes=False):
    """{tag: participation ratio} of embeddings or of class prototypes."""
    if on_prototypes:
        labels = torch.as_tensor(labels)
        classes = torch.unique(labels, sorted=True).tolist()
        return {
            tag: effective_dimension(compute_prototypes(X, labels, classes))
            for tag, _, _, X in iter_states(features, rep)
        }
    return {tag: effective_dimension(X) for tag, _, _, X in iter_states(features, rep)}


def between_prototype_summary(between_dict):
    """Mean ± SD of each model/state's between-prototype distances.

    Reproduces the decision-stage numbers quoted in the paper's "Learned
    Decision Stage Structure" section (paper Fig. 2). Pass the logits
    between-dict from compute_cluster_sizes_and_rdms.
    """
    from .config import ordered_model_names
    rows = []
    for tag in ordered_model_names(between_dict.keys()):
        arr = between_dict[tag]
        rows.append({"Model/State": tag,
                     "Mean": round(float(arr.mean()), 3),
                     "SD": round(float(arr.std()), 3),
                     "n_pairs": len(arr)})
    return pd.DataFrame(rows)


def accuracy_table(features, labels, source="logits", skip=("CLIP",)):
    """Top-1/top-5 accuracy table for all models/states.

    source="logits": standard ImageNet accuracy from the model's classifier.
    source="penult": cosine nearest-centroid accuracy on penultimate features.
    """
    labels_t = torch.as_tensor(labels)
    classes = torch.unique(labels_t, sorted=True).tolist()
    rows = []
    order = [m for m in MODEL_ORDER if m in features] + \
            [m for m in features if m not in MODEL_ORDER]
    for model in order:
        if model in skip:
            continue
        rep_dict = features[model].get("penult" if source == "penult" else "logits", {})
        for state in ["baseline", "after"]:
            X = rep_dict.get(state)
            if X is None:
                continue
            if source == "penult":
                top1, top5 = nearest_centroid_accuracy(torch.as_tensor(X).float(), labels_t, classes)
            else:
                top1, top5 = logit_accuracy(X, labels_t)
            rows.append({
                "Model": model, "State": state,
                "Top-1 (%)": round(top1, 2), "Top-5 (%)": round(top5, 2),
                "N": int(X.shape[0]),
            })
    return pd.DataFrame(rows)


def geometry_summary_table(features, labels, models=None):
    """Per-model geometry summary (baseline vs after recurrence) — paper Table 2.

    Each model is analyzed at its conventional layer (config.REP_OVERRIDES;
    logits by default, penultimate for CORnet-RT). See
    geometry.pair_geometry_metrics for the five column definitions.
    """
    models = models or RECURRENT_MODELS
    rows = []
    for model in models:
        rep = REP_OVERRIDES.get(model, DEFAULT_REP)
        X_b, X_a = get_pair(features, model, rep)
        if X_b is None:
            continue
        rows.append({"Model": model, "Rep": rep,
                     **pair_geometry_metrics(X_b, X_a, labels)})
    return pd.DataFrame(rows)
