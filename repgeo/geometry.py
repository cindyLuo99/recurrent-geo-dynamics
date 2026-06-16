"""Core representational-geometry math.

All functions are pure tensor-in / tensor-out. Conventions:
  - activations X: [N, D] floats (or [N, C, H, W] / [N, H, W, C]; see to_feat)
  - labels y:      [N] integer class labels (true ImageNet indices)
  - prototypes:    [C, D], rows ordered by sorted unique label
  - distances:     cosine distance = 1 - cosine similarity, in [0, 2]
"""

import numpy as np
import torch
import torch.nn.functional as F


def to_feat(X: torch.Tensor) -> torch.Tensor:
    """Ensure [N, D]. 4D inputs are global-average-pooled over space.

    Channel layout for 4D inputs is inferred: if dim 1 is smaller than the
    last dim the tensor is assumed channels-first [N, C, H, W], otherwise
    channels-last [N, H, W, C] (e.g. TensorFlow exports). This heuristic is
    ambiguous when C equals the spatial size — pool such tensors explicitly
    before calling.
    """
    X = torch.as_tensor(X)
    if X.ndim == 4:
        if X.shape[1] < X.shape[-1]:  # channels-last [N, H, W, C]
            X = X.permute(0, 3, 1, 2)
        X = F.adaptive_avg_pool2d(X.float(), 1).flatten(1)
    return X.detach()


def class_index_tensor(y: torch.Tensor, classes=None):
    """Map raw labels to dense 0..C-1 indices. Returns (y_idx, classes)."""
    y = torch.as_tensor(y)
    if classes is None:
        classes = torch.unique(y, sorted=True).tolist()
    cls2idx = {int(c): i for i, c in enumerate(classes)}
    y_idx = torch.tensor([cls2idx[int(v)] for v in y.tolist()])
    return y_idx, classes


def compute_prototypes(X: torch.Tensor, y: torch.Tensor, classes=None) -> torch.Tensor:
    """Per-class mean vectors, [C, D], ordered like `classes`."""
    X = to_feat(X).float()
    y_idx, classes = class_index_tensor(y, classes)
    y_idx = y_idx.to(X.device)
    C, D = len(classes), X.size(1)
    S = torch.zeros(C, D, device=X.device, dtype=X.dtype)
    n = torch.zeros(C, device=X.device, dtype=X.dtype)
    S.index_add_(0, y_idx, X)
    n.index_add_(0, y_idx, torch.ones(len(y_idx), device=X.device, dtype=X.dtype))
    return S / n.clamp_min(1).unsqueeze(1)


def exemplar_proto_dists(X: torch.Tensor, prot: torch.Tensor, y: torch.Tensor,
                         metric: str = "cosine") -> np.ndarray:
    """Distance from each exemplar to its own class prototype (order of X)."""
    X = to_feat(X).float()
    y_idx, _ = class_index_tensor(y)
    y_idx = y_idx.to(X.device)
    P = prot.to(X.device)[y_idx]
    if metric == "cosine":
        return (1.0 - F.cosine_similarity(X, P, dim=1)).cpu().numpy()
    elif metric == "euclidean":
        return torch.linalg.vector_norm(X - P, dim=1).cpu().numpy()
    raise ValueError("metric must be 'cosine' or 'euclidean'")


def rdm_cosine(P: torch.Tensor) -> torch.Tensor:
    """Cosine-distance RDM (1 - cosine) for row vectors, diagonal zeroed."""
    X = F.normalize(to_feat(P).float(), dim=1)
    D = 1.0 - X @ X.T
    D.fill_diagonal_(0)
    return D


def rdm_vec_upper(D: torch.Tensor) -> np.ndarray:
    """Strict upper triangle of a square matrix as a 1D vector."""
    n = D.size(0)
    i, j = torch.triu_indices(n, n, offset=1)
    return D[i, j].cpu().numpy()


def rsa_spearman(rdm_vec_a: np.ndarray, rdm_vec_b: np.ndarray):
    """Spearman correlation between two RDM upper-triangle vectors."""
    from scipy.stats import spearmanr
    rho, p = spearmanr(rdm_vec_a, rdm_vec_b)
    return rho, p


def cosine_sim_matrix(X: torch.Tensor) -> torch.Tensor:
    """N x N cosine-similarity matrix of the rows of X."""
    Xn = F.normalize(to_feat(X).float(), dim=1)
    return Xn @ Xn.T


def category_preservation_knn(X: torch.Tensor, labels, max_k: int = 50):
    """k-NN category preservation curve.

    For each k, the fraction (%) of every point's k nearest neighbours that
    share its category, averaged over points. The point itself is excluded
    from its own neighbourhood. Returns (ks, pct) arrays for k = 2..max_k.
    """
    labels = torch.as_tensor(labels)
    sim = cosine_sim_matrix(X)
    N = sim.size(0)
    max_k = min(max_k, N)
    _, nbrs = sim.topk(max_k, dim=1)  # column 0 is the point itself
    labels = labels.to(sim.device)
    ks, pct = [], []
    for k in range(2, max_k + 1):
        nnk = nbrs[:, 1:k]
        same = (labels.unsqueeze(1) == labels[nnk]).sum()
        ks.append(k)
        pct.append(100.0 * same.item() / (N * (k - 1)))
    return np.array(ks), np.array(pct)


def prototype_shifts(prot_b: torch.Tensor, prot_a: torch.Tensor) -> np.ndarray:
    """Per-category prototype displacement, 1 - cos_sim(prot_b[i], prot_a[i])."""
    return (1.0 - F.cosine_similarity(prot_b, prot_a, dim=1)).cpu().numpy()


def effective_dimension(X: torch.Tensor, center: bool = True, eps: float = 1e-12) -> float:
    """Participation ratio (tr Σ)^2 / tr(Σ^2) of the covariance of X."""
    X = to_feat(X).float()
    if center:
        X = X - X.mean(dim=0, keepdim=True)
    N = max(X.shape[0] - 1, 1)
    cov = (X.T @ X) / N
    evals = torch.clamp(torch.linalg.eigvalsh(cov), min=0)
    trace = evals.sum()
    sq_trace = torch.square(evals).sum().clamp_min(eps)
    return float((trace * trace) / sq_trace)


def nearest_centroid_accuracy(X: torch.Tensor, y: torch.Tensor, classes=None):
    """Top-1/top-5 accuracy of a cosine nearest-centroid (prototype) classifier.

    C-WAY classification, where C = number of classes present in `y` (100 in
    the paper's stimulus set): each exemplar is assigned to the nearest of the
    C class prototypes. This is NOT comparable to the model's standard 1000-way
    ImageNet logit Top-1 — chance here is 1/C (not 1/1000). Use it as a
    classifier-independent measure of category separability, comparable
    *across models*, not as a held-out ImageNet accuracy.

    Note: prototypes are computed on the full set, so each exemplar
    contributes 1/n_per_class to its own centroid (no leave-one-out).
    With 50 images/class this inflates accuracy only marginally, and
    identically across models.
    """
    X = to_feat(X).float()
    y = torch.as_tensor(y)
    _, classes = class_index_tensor(y, classes)
    X_norm = F.normalize(X, dim=1)
    prot_norm = F.normalize(compute_prototypes(X, y, classes), dim=1)
    sims = X_norm @ prot_norm.T  # [N, C]

    cls_tensor = torch.tensor(classes, device=X.device)
    top1 = (cls_tensor[sims.argmax(dim=1)] == y).float().mean().item() * 100
    k = min(5, len(classes))
    top_k_labels = cls_tensor[sims.topk(k, dim=1).indices]
    top5 = (top_k_labels == y.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    return top1, top5


def logit_accuracy(logits: torch.Tensor, y: torch.Tensor):
    """Standard top-1/top-5 accuracy from 1000-class logits."""
    logits = torch.as_tensor(logits).float()
    y = torch.as_tensor(y)
    top1 = (logits.argmax(dim=1) == y).float().mean().item() * 100
    k = min(5, logits.shape[1])
    top5 = (logits.topk(k, dim=1).indices == y.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    return top1, top5


def between_proto_pct_change(prot_b, prot_a, iu=None):
    """Mean per-pair % change in CENTERED between-prototype cosine distance.

    Prototypes are mean-centered (removes global translation, isolating
    structural reorganization), turned into cosine-distance RDMs, and the mean
    of per-pair percent changes (d_after - d_before) / d_before * 100 is
    returned. Pairs with near-zero baseline distance are dropped.

    Shared by the geometry summary table's "Δ Between-Proto (%)" and the
    permutation test in stats.py, so the reported statistic and its null are
    computed identically by construction.
    """
    if iu is None:
        iu = np.triu_indices(prot_b.shape[0], k=1)
    pb = prot_b - prot_b.mean(0, keepdim=True)
    pa = prot_a - prot_a.mean(0, keepdim=True)
    vb = rdm_cosine(pb).cpu().numpy()[iu]
    va = rdm_cosine(pa).cpu().numpy()[iu]
    m = vb > 1e-12
    return float(np.mean((va[m] - vb[m]) / vb[m] * 100.0))


def cluster_size_pct_change(d_baseline, d_after, labels):
    """Macro-averaged per-category % change in mean exemplar->prototype distance.

    For each category, the percent change in its mean exemplar-to-prototype
    distance, (mu_after - mu_before) / mu_before * 100, then averaged across
    categories (each weighted equally — a "macro" average). `d_baseline` and
    `d_after` are per-exemplar distances (e.g. from exemplar_proto_dists),
    `labels` groups exemplars by category. Categories with near-zero baseline
    distance are dropped.
    """
    labels = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
    pct = []
    for c in np.unique(labels):
        m = labels == c
        mu_b, mu_a = d_baseline[m].mean(), d_after[m].mean()
        if mu_b > 1e-12:
            pct.append((mu_a - mu_b) / mu_b * 100.0)
    return float(np.mean(pct))


def pair_geometry_metrics(X_b, X_a, labels):
    """The five geometry metrics for one baseline/after activation pair.

      Δ Cluster Size      mean change in exemplar->prototype cosine distance
                          (negative = compaction)
      Δ Cluster Size (%)  macro average of per-category % changes
      Avg Proto Shift     mean cosine distance between matched prototypes
      Δ Between-Proto (%) mean per-pair % change on CENTERED prototypes
                          (positive = expansion)
      Proto RDM ρ         Spearman correlation of baseline vs after RDMs
                          (raw prototypes; high = structure preserved)
    """
    from scipy.stats import spearmanr

    labels_t = torch.as_tensor(labels)
    classes = torch.unique(labels_t, sorted=True).tolist()
    iu = np.triu_indices(len(classes), k=1)

    X_b = to_feat(X_b).float()
    X_a = to_feat(X_a).float()
    prot_b = compute_prototypes(X_b, labels_t, classes)
    prot_a = compute_prototypes(X_a, labels_t, classes)

    d_b = exemplar_proto_dists(X_b, prot_b, labels_t, metric="cosine")
    d_a = exemplar_proto_dists(X_a, prot_a, labels_t, metric="cosine")
    delta_cluster = float(np.mean(d_a) - np.mean(d_b))            # Δ Cluster Size
    delta_cluster_pct = cluster_size_pct_change(d_b, d_a, labels_t)  # Δ Cluster Size (%)

    shifts = 1.0 - F.cosine_similarity(prot_b, prot_a, dim=1)
    avg_proto_shift = float(shifts.mean().item())                # Avg Proto Shift

    delta_between_pct = between_proto_pct_change(prot_b, prot_a, iu)  # Δ Between-Proto (%)

    rho, _ = spearmanr(rdm_cosine(prot_b).cpu().numpy()[iu],
                       rdm_cosine(prot_a).cpu().numpy()[iu]) # Proto RDM ρ

    return {
        "Δ Cluster Size": delta_cluster,
        "Δ Cluster Size (%)": delta_cluster_pct,
        "Avg Proto Shift": avg_proto_shift,
        "Δ Between-Proto (%)": delta_between_pct,
        "Proto RDM ρ": float(rho),
    }
