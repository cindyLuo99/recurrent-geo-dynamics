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

import numpy as np
import pandas as pd
import torch

from .config import (MODEL_ORDER, RECURRENT_MODELS, DEFAULT_REP, REP_OVERRIDES,
                     ordered_model_names)
from .geometry import (
    compute_prototypes, exemplar_proto_dists, rdm_cosine, rdm_vec_upper,
    effective_dimension, nearest_centroid_accuracy, logit_accuracy,
    pair_geometry_metrics, to_feat, category_preservation_knn, prototype_shifts,
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


def local_global_composite(features, labels, rep="logits",
                           models=tuple(RECURRENT_MODELS), rep_overrides=None):
    """Inputs for the local/global composite figure (paper Fig. 1), rendered by
    plotting.plot_local_global_composite.

    For each recurrent model that has both states (at its rep), returns a dict:
      delta         [N]    per-exemplar cluster-size change (d_after - d_before)
      pca_baseline  [C, 2] baseline prototypes projected into the PCA space of
                           the baseline activations
      pca_after     [C, 2] after prototypes in that SAME PCA space
      rep           str    the layer used for this model
    """
    from sklearn.decomposition import PCA

    rep_overrides = rep_overrides or {}
    labels_t = torch.as_tensor(labels)
    classes = torch.unique(labels_t, sorted=True).tolist()

    out = {}
    for model in models:
        m_rep = rep_overrides.get(model, rep)
        rd = features.get(model, {}).get(m_rep, {})
        if "baseline" not in rd or "after" not in rd:
            continue
        X_b = to_feat(rd["baseline"]).float()
        X_a = to_feat(rd["after"]).float()
        prot_b = compute_prototypes(X_b, labels_t, classes)
        prot_a = compute_prototypes(X_a, labels_t, classes)
        d_b = exemplar_proto_dists(X_b, prot_b, labels_t, metric="cosine")
        d_a = exemplar_proto_dists(X_a, prot_a, labels_t, metric="cosine")
        pca = PCA(n_components=2).fit(X_b.cpu().numpy())
        out[model] = {
            "delta": d_a - d_b,
            "pca_baseline": pca.transform(prot_b.cpu().numpy()),
            "pca_after": pca.transform(prot_a.cpu().numpy()),
            "rep": m_rep,
        }
    return out


def build_rsa_matrix(between_dict, exclude=()):
    """K x K Spearman-rho matrix across models from between-prototype vectors.
    Returns (names, [K, K] matrix)."""
    from scipy.stats import spearmanr

    names = [n for n in ordered_model_names(between_dict.keys())
             if n.split()[0] not in exclude]
    K = len(names)
    vecs = [between_dict[n] for n in names]
    rsa = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(K):
            rho, _ = spearmanr(vecs[i], vecs[j])
            rsa[i, j] = np.nan_to_num(rho)
    return names, rsa


def build_mds_coords(rdms, seed=42, exclude=(), n_components=2):
    """MDS embedding of models; distance = 1 - Spearman rho between RDMs.
    Returns (names, coords [K, n_components], D [K, K], stress)."""
    from scipy.stats import spearmanr
    from sklearn.manifold import MDS

    names = [n for n in ordered_model_names(rdms.keys())
             if n.split()[0] not in exclude]
    K = len(names)
    vecs = [rdm_vec_upper(rdms[n]) for n in names]
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            rho, _ = spearmanr(vecs[i], vecs[j])
            D[i, j] = D[j, i] = 1.0 - np.nan_to_num(rho)
    mds = MDS(n_components=n_components, dissimilarity='precomputed',
              random_state=seed, n_init=10, max_iter=500)
    coords = mds.fit_transform(D)
    return names, coords, D, mds.stress_


def rsa_mds_composite(between_log, between_pen, rdms_log, rdms_pen,
                      exclude=("CLIP",), seed=42):
    """Cross-model RSA matrices and 2-D MDS embeddings for the RSA/MDS composite
    figure (paper Fig. 3), rendered by plotting.plot_rsa_mds_composite.

    Returns a dict; each value is a tuple:
      rsa_pen, rsa_log : (names, [K, K] Spearman-rho matrix)
      mds_pen, mds_log : (names, coords [K, 2], D [K, K], stress)
    """
    return {
        "rsa_pen": build_rsa_matrix(between_pen, exclude),
        "rsa_log": build_rsa_matrix(between_log, exclude),
        "mds_pen": build_mds_coords(rdms_pen, seed, exclude),
        "mds_log": build_mds_coords(rdms_log, seed, exclude),
    }


def convrnn_supplement(X_before, X_after, labels, max_k=50):
    """Inputs for the ConvRNN input-on vs final supplement figure (paper Fig. 4),
    rendered by plotting.plot_convrnn_supplement. Returns a dict:
      delta        [N]       per-exemplar cluster-size change (after - before)
      knn_before   (ks, pct) k-NN category preservation curve, before
      knn_after    (ks, pct) k-NN category preservation curve, after
      pca_baseline [C, 2]    before prototypes in the before-cloud PCA space
      pca_after    [C, 2]    after prototypes in that SAME PCA space
      shifts       [C]       per-category prototype shift (1 - cosine)
    """
    from sklearn.decomposition import PCA

    labels_t = torch.as_tensor(labels)
    classes = torch.unique(labels_t, sorted=True).tolist()
    Xb, Xa = to_feat(X_before).float(), to_feat(X_after).float()
    prot_b = compute_prototypes(Xb, labels_t, classes)
    prot_a = compute_prototypes(Xa, labels_t, classes)
    d_b = exemplar_proto_dists(Xb, prot_b, labels_t, metric="cosine")
    d_a = exemplar_proto_dists(Xa, prot_a, labels_t, metric="cosine")
    pca = PCA(n_components=2).fit(Xb.cpu().numpy())
    return {
        "delta": d_a - d_b,
        "knn_before": category_preservation_knn(Xb, labels_t, max_k=max_k),
        "knn_after": category_preservation_knn(Xa, labels_t, max_k=max_k),
        "pca_baseline": pca.transform(prot_b.cpu().numpy()),
        "pca_after": pca.transform(prot_a.cpu().numpy()),
        "shifts": prototype_shifts(prot_b, prot_a),
    }


def top1_accuracy_bars(features, labels, exclude=("CLIP",)):
    """Per-model top-1 accuracy (from logits) for the accuracy-bar figure,
    rendered by plotting.plot_top1_accuracy_bars.

    Returns a list of (model, [(state, acc), ...]) in MODEL_ORDER; recurrent
    models have two entries (baseline, after), feedforward models one.
    """
    labels_t = torch.as_tensor(labels)
    bars = []
    for model in MODEL_ORDER:
        if model in exclude:
            continue
        rep_dict = features.get(model, {}).get("logits", {})
        states = []
        for state in ["baseline", "after"]:
            logits = rep_dict.get(state)
            if logits is None:
                continue
            acc = (torch.as_tensor(logits).float().argmax(dim=1) == labels_t
                   ).float().mean().item() * 100.0
            states.append((state, acc))
        if states:
            bars.append((model, states))
    return bars


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
