"""Significance tests. Rule of thumb: anything returning a p-value lives here.

Two tests, matching the paper:

  Local (cluster size): linear mixed-effects model on the per-exemplar change
  in exemplar-to-prototype cosine distance, `distance_diff ~ 1` with random
  intercepts by category. Accounts for non-independence of exemplars within
  a category. Negative beta = compaction.

  Global (between-class separation): permutation test on the mean of per-pair
  percent changes between CENTERED prototypes (centering removes global
  translation, isolating structural reorganization). Null: baseline/after
  assignment is shuffled per exemplar, prototypes recomputed.

Descriptive summaries (no p-values) live in repgeo.analysis; the metric
definitions live in repgeo.geometry.
"""

import warnings

import numpy as np
import pandas as pd
import torch

from .analysis import get_pair
from .config import RECURRENT_MODELS, DEFAULT_REP, REP_OVERRIDES
from .geometry import to_feat, compute_prototypes, exemplar_proto_dists, rdm_cosine


def local_cluster_lme(d_baseline, d_after, labels):
    """Mixed-effects test of the per-exemplar distance change.

    Args:
        d_baseline, d_after: [N] exemplar->prototype distances.
        labels: [N] category labels (grouping factor).

    Returns dict with beta, SE, 95% CI, two-tailed p.
    """
    import statsmodels.formula.api as smf

    diff = np.asarray(d_after) - np.asarray(d_baseline)
    cat = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
    df = pd.DataFrame({"distance_diff": diff, "category": cat.astype(str)})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("distance_diff ~ 1", df, groups=df["category"])
        mdf = md.fit(reml=True)

    coef = mdf.fe_params["Intercept"]
    pval = mdf.pvalues["Intercept"]  # two-tailed
    ci = mdf.conf_int().loc["Intercept"]
    return {
        "beta": float(coef),
        "SE": float(mdf.bse_fe["Intercept"]),
        "95% CI lo": float(ci.iloc[0]),
        "95% CI hi": float(ci.iloc[1]),
        "p-value": float(pval),
        "Direction": "compaction" if coef < 0 else "expansion",
    }


def run_local_lme_tests(features, labels, models=None):
    """Local cluster-change LME for every recurrent model. Returns DataFrame."""
    models = models or RECURRENT_MODELS
    labels_t = torch.as_tensor(labels)
    classes = torch.unique(labels_t, sorted=True).tolist()
    rows = []
    for model in models:
        rep = REP_OVERRIDES.get(model, DEFAULT_REP)
        X_b, X_a = get_pair(features, model, rep)
        if X_b is None:
            continue
        prot_b = compute_prototypes(X_b, labels_t, classes)
        prot_a = compute_prototypes(X_a, labels_t, classes)
        d_b = exemplar_proto_dists(X_b, prot_b, labels_t, metric="cosine")
        d_a = exemplar_proto_dists(X_a, prot_a, labels_t, metric="cosine")
        row = {"Model": model, "Rep": rep}
        row.update(local_cluster_lme(d_b, d_a, labels_t))
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_pairwise_pct_change(X_b, X_a, labels_t, classes, iu):
    """Mean of per-pair % distance changes on centered prototypes."""
    prot_b = compute_prototypes(X_b, labels_t, classes)
    prot_a = compute_prototypes(X_a, labels_t, classes)
    prot_b = prot_b - prot_b.mean(0, keepdim=True)
    prot_a = prot_a - prot_a.mean(0, keepdim=True)
    vec_b = rdm_cosine(prot_b).cpu().numpy()[iu]
    vec_a = rdm_cosine(prot_a).cpu().numpy()[iu]
    mask = vec_b > 1e-12
    return float(np.mean((vec_a[mask] - vec_b[mask]) / vec_b[mask] * 100.0))


def between_separation_permutation(X_base, X_after, labels, n_perm=10000, seed=42):
    """Permutation test for change in between-prototype separation.

    Null distribution: baseline/after activations swapped independently per
    exemplar, centered prototypes recomputed, mean per-pair % change taken.

    Returns dict with observed % change, two- and one-tailed p-values, and
    the null distribution.
    """
    rng = np.random.default_rng(seed)
    labels_t = torch.as_tensor(labels)
    classes = torch.unique(labels_t, sorted=True).tolist()
    iu = np.triu_indices(len(classes), k=1)

    X_base = to_feat(X_base).float()
    X_after = to_feat(X_after).float()
    N = X_base.shape[0]

    obs_pct = _mean_pairwise_pct_change(X_base, X_after, labels_t, classes, iu)

    null_pcts = np.empty(n_perm)
    for p in range(n_perm):
        swap = rng.random(N) < 0.5
        X_b = X_base.clone()
        X_a = X_after.clone()
        X_b[swap] = X_after[swap]
        X_a[swap] = X_base[swap]
        null_pcts[p] = _mean_pairwise_pct_change(X_b, X_a, labels_t, classes, iu)

    return {
        "obs_pct": obs_pct,
        "p_two_tailed": float(np.mean(np.abs(null_pcts) >= np.abs(obs_pct))),
        "p_greater": float(np.mean(null_pcts >= obs_pct)),
        "p_less": float(np.mean(null_pcts <= obs_pct)),
        "null_pcts": null_pcts,
    }


def run_global_permutation_tests(features, labels, models=None, n_perm=10000, seed=42):
    """Global separation permutation test for every recurrent model."""
    models = models or RECURRENT_MODELS
    rows = []
    for model in models:
        rep = REP_OVERRIDES.get(model, DEFAULT_REP)
        X_b, X_a = get_pair(features, model, rep)
        if X_b is None:
            continue
        res = between_separation_permutation(X_b, X_a, labels, n_perm=n_perm, seed=seed)
        res.pop("null_pcts")
        rows.append({"Model": model, "Rep": rep, **res})
    return pd.DataFrame(rows)
