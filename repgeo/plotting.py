"""Paper figures. All functions take the canonical features dict (built by
repgeo.loading) or the derived dicts (from repgeo.analysis), and save to a
path when save_path is given.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, spearmanr

from .config import MODEL_ORDER, MODEL_COLOR_MAP, DISPLAY_NAMES, RECURRENT_MODELS, \
    pick_color, ordered_model_names
from .geometry import to_feat, compute_prototypes, exemplar_proto_dists, rdm_vec_upper

# Embed fonts as editable text in PDFs (journal requirement)
mpl.rcParams['pdf.fonttype'] = 42


def _save_show(fig, save_path, dpi=200, show=True):
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _add_column_vlines(fig, ax_top, ax_bot, x_values=(0.0, 1.0)):
    """Vertical reference lines spanning a column of stacked axes."""
    fig.canvas.draw()
    for x_val in x_values:
        x_fig, _ = ax_bot.transData.transform((x_val, 0))
        x_fig = fig.transFigure.inverted().transform((x_fig, 0))[0]
        _, y_bot = fig.transFigure.inverted().transform(ax_bot.transAxes.transform((0, 0)))
        _, y_top = fig.transFigure.inverted().transform(ax_top.transAxes.transform((0, 1)))
        fig.add_artist(Line2D([x_fig, x_fig], [y_bot, y_top],
                              transform=fig.transFigure, color='black',
                              lw=1.2, ls='-', zorder=1, clip_on=False))


def _ridge_panel(ax, between_dict, entries, x_grid, show_stats):
    """Draw the KDEs for one model row; returns stat annotation lines."""
    stat_lines = []
    for tag, state in entries:
        arr = np.asarray(between_dict[tag])
        color = pick_color(tag)
        y = gaussian_kde(arr)(x_grid)
        ls = '-' if state == "baseline" else '--'
        lw = 1.6 if state == "baseline" else 1.8
        alpha_fill = 0.35 if state == "baseline" else 0.25
        ax.fill_between(x_grid, y, alpha=alpha_fill, color=color)
        ax.plot(x_grid, y, color=color, ls=ls, lw=lw)
        if show_stats:
            label = "base" if state == "baseline" else "after"
            stat_lines.append((label, float(arr.mean()), float(arr.std()), color))
    return stat_lines


def _ridge_style_row(ax, is_last, xlabel):
    ax.set_ylabel('')
    ax.set_yticks([])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    if not is_last:
        ax.spines['bottom'].set_color('#cccccc')
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(bottom=False, labelbottom=False)
    else:
        ax.set_xlabel(xlabel, fontsize=16)
        ax.tick_params(axis='x', labelsize=14)


def plot_separation_ridge(between_dict,
                          xlabel="Between-prototype cosine distance",
                          exclude=(), figsize=None, save_path=None,
                          show_stats=True, show_labels=True,
                          states=("baseline", "after"), show=True):
    """Ridge plot of between-class prototype separations, one row per model.

    Fixed x range [0, 2]: vertical reference lines mark 0 (identical
    prototypes) and 1 (orthogonal). Baseline solid, after dashed.
    show_labels=False / states=("baseline",) give the presentation variant.
    """
    model_rows = []
    for model in MODEL_ORDER:
        if model in exclude:
            continue
        entries = [(f"{model} {s}", s) for s in states if f"{model} {s}" in between_dict]
        if entries:
            model_rows.append((model, entries))
    if not model_rows:
        return

    n_rows = len(model_rows)
    figsize = figsize or (7.5, 1.1 * n_rows + 1.0)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    axes = [axes] if n_rows == 1 else list(axes)
    x_grid = np.linspace(0.0, 2.0, 500)

    for row_i, (model, entries) in enumerate(model_rows):
        ax = axes[row_i]
        stat_lines = _ridge_panel(ax, between_dict, entries, x_grid, show_stats)
        if show_labels:
            ax.text(0.02, 0.92, DISPLAY_NAMES.get(model, model), transform=ax.transAxes,
                    fontsize=16, fontweight='bold', ha='left', va='top', fontstyle='italic')
        for j, (label, mu, sd, color) in enumerate(stat_lines):
            ax.text(0.98, 0.92 - j * 0.35, f"{label}: μ={mu:.3f}, σ={sd:.3f}",
                    transform=ax.transAxes, fontsize=8, color=color,
                    ha='right', va='top', family='monospace')
        _ridge_style_row(ax, row_i == n_rows - 1, xlabel)

    axes[-1].set_xlim(0.0, 2.0)
    axes[0].text(0.0, axes[0].get_ylim()[1] * 1.02, "Same",
                 ha='center', va='bottom', fontsize=16, fontstyle='italic')
    axes[0].text(1.0, axes[0].get_ylim()[1] * 1.02, "Orthogonal",
                 ha='center', va='bottom', fontsize=16, fontstyle='italic')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    _add_column_vlines(fig, axes[0], axes[-1])
    _save_show(fig, save_path, show=show)


def plot_separation_ridge_dual(between_pen, between_log,
                               xlabel="Between-prototype cosine distance",
                               exclude=("CLIP",), figsize=None, save_path=None,
                               show_stats=True, show=True):
    """Two-column ridge plot: penultimate (left) and logits (right)."""
    model_rows = []
    for model in MODEL_ORDER:
        if model in exclude:
            continue
        pen = [(f"{model} {s}", s) for s in ("baseline", "after") if f"{model} {s}" in between_pen]
        log = [(f"{model} {s}", s) for s in ("baseline", "after") if f"{model} {s}" in between_log]
        if pen or log:
            model_rows.append((model, pen, log))
    if not model_rows:
        return

    n_rows = len(model_rows)
    figsize = figsize or (14.0, 1.1 * n_rows + 1.6)
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, sharex='col')
    axes = np.atleast_2d(axes)
    x_grid = np.linspace(0.0, 2.0, 500)

    for row_i, (model, pen_entries, log_entries) in enumerate(model_rows):
        for col, (d, entries) in enumerate([(between_pen, pen_entries),
                                            (between_log, log_entries)]):
            ax = axes[row_i, col]
            stat_lines = _ridge_panel(ax, d, entries, x_grid, show_stats)
            if col == 0:
                ax.text(0.02, 0.92, DISPLAY_NAMES.get(model, model), transform=ax.transAxes,
                        fontsize=20, fontweight='bold', ha='left', va='top', fontstyle='italic')
            for j, (label, mu, sd, color) in enumerate(stat_lines):
                ax.text(0.98, 0.92 - j * 0.35, f"{label}: μ={mu:.3f}, σ={sd:.3f}",
                        transform=ax.transAxes, fontsize=15, color=color,
                        ha='right', va='top', family='monospace')
            _ridge_style_row(ax, row_i == n_rows - 1, xlabel)

    for col in (0, 1):
        axes[-1, col].set_xlim(0.0, 2.0)
    axes[0, 0].set_title("Penultimate", fontsize=20, fontweight='bold', pad=22)
    axes[0, 1].set_title("Logits", fontsize=20, fontweight='bold', pad=22)
    for ax in (axes[0, 0], axes[0, 1]):
        ax.text(0.0, ax.get_ylim()[1] * 1.02, "Same",
                ha='center', va='bottom', fontsize=14, fontstyle='italic')
        ax.text(1.0, ax.get_ylim()[1] * 1.02, "Orthogonal",
                ha='center', va='bottom', fontsize=14, fontstyle='italic')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, wspace=0.10)
    for col in (0, 1):
        _add_column_vlines(fig, axes[0, col], axes[-1, col])
    _save_show(fig, save_path, show=show)


DEFAULT_MECHANISM_ANNOTATIONS = {
    "LRM3":      "Multiplicative long-range\nLogits\n Baseline pass\nand modulated pass",
    "LRA3":      "Additive long-range\nLogits\n Baseline pass\nand modulated pass",
    "BL":        "Within-layer additive\n(separate weights)\nLogits\nInitial and final steps",
    "CORnet-RT": "Within-layer additive\n(shared weights)\nPenultimate\nLast two time steps",
    "ConvRNN":   "Multiplicatively gated with-in\nand long-range\nLogits\nLast two time steps",
}


def plot_local_global_composite(features, labels, rep="logits",
                                models=tuple(RECURRENT_MODELS),
                                rep_overrides=None, annotations=None,
                                save_path=None, figsize=None, show=True):
    """3-row composite per recurrent model:
      row 0 mechanism annotation, row 1 local cluster-size-change KDE,
      row 2 PCA of prototype shifts (PCA fitted on baseline activations).
    """
    from sklearn.decomposition import PCA

    rep_overrides = rep_overrides or {}
    annotations = annotations or DEFAULT_MECHANISM_ANNOTATIONS
    C_BASE, C_AFTER, C_FILL = '#555555', '#E41A1C', '#cccccc'

    labels_t = torch.as_tensor(labels)
    classes = torch.unique(labels_t, sorted=True).tolist()

    avail = []
    for m in models:
        m_rep = rep_overrides.get(m, rep)
        if "baseline" in features.get(m, {}).get(m_rep, {}) \
                and "after" in features[m][m_rep]:
            avail.append(m)
    n_cols = len(avail)
    if n_cols == 0:
        print("No recurrent models with both states found.")
        return

    figsize = figsize or (4.0 * n_cols, 9.0)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, n_cols, height_ratios=[0.55, 2.0, 4.0],
                          hspace=0.3, wspace=0.28,
                          top=0.91, bottom=0.10, left=0.09, right=0.93)

    # precompute deltas so the KDE row shares one x-axis
    model_data = {}
    for model in avail:
        m_rep = rep_overrides.get(model, rep)
        X_b = to_feat(features[model][m_rep]["baseline"]).float()
        X_a = to_feat(features[model][m_rep]["after"]).float()
        prot_b = compute_prototypes(X_b, labels_t, classes)
        prot_a = compute_prototypes(X_a, labels_t, classes)
        d_b = exemplar_proto_dists(X_b, prot_b, labels_t, metric="cosine")
        d_a = exemplar_proto_dists(X_a, prot_a, labels_t, metric="cosine")
        model_data[model] = dict(X_b=X_b, prot_b=prot_b, prot_a=prot_a, delta=d_a - d_b)

    all_delta = np.concatenate([md['delta'] for md in model_data.values()])
    pad_d = (all_delta.max() - all_delta.min()) * 0.08
    shared_xmin, shared_xmax = float(all_delta.min()) - pad_d, float(all_delta.max()) + pad_d

    axes_kde, axes_pca = {}, {}
    for col, model in enumerate(avail):
        md = model_data[model]

        ax_ann = fig.add_subplot(gs[0, col])
        ax_ann.axis('off')
        ax_ann.set_title(model, fontsize=30, fontweight='bold', pad=6)
        ax_ann.text(0.5, 0.22, annotations.get(model, ""), ha='center', va='center',
                    fontsize=16, fontstyle='italic', color='#555555',
                    transform=ax_ann.transAxes, linespacing=1.3)

        # ---- local: KDE of per-exemplar cluster-size change ----
        ax_kde = fig.add_subplot(gs[1, col])
        axes_kde[col] = ax_kde
        x_grid = np.linspace(shared_xmin, shared_xmax, 400)
        density = gaussian_kde(md['delta'])(x_grid)
        ax_kde.fill_between(x_grid, density, color=C_FILL, alpha=0.6)
        ax_kde.plot(x_grid, density, color='#888888', lw=1.0)
        ax_kde.axvline(0, color='black', lw=1.5, ls='-', zorder=0)
        mean_d = float(md['delta'].mean())
        ax_kde.axvline(mean_d, color='red', lw=2, ls='--', zorder=5)

        ax_kde.set_xlim(shared_xmin, shared_xmax)
        ax_kde.set_yticks([])
        for side in ('top', 'right', 'left'):
            ax_kde.spines[side].set_visible(False)
        ax_kde.spines['bottom'].set_color('#888888')
        ax_kde.spines['bottom'].set_linewidth(0.8)
        ax_kde.tick_params(axis='x', labelsize=12, colors='#333333')

        # Compact / Expand arrow
        peak = density.max()
        span = shared_xmax - shared_xmin
        if mean_d < 0:
            label_txt, x0, x1, xt = 'Compact', 0.24, 0.12, 0.18
        else:
            label_txt, x0, x1, xt = 'Expand', 0.76, 0.88, 0.82
        ax_kde.text(shared_xmin + span * xt, peak * 0.50, label_txt, ha='center',
                    va='bottom', fontsize=15, fontstyle='italic', color='black')
        ax_kde.annotate('', xy=(shared_xmin + span * x1, peak * 0.45),
                        xytext=(shared_xmin + span * x0, peak * 0.45),
                        arrowprops=dict(arrowstyle='->', color='black',
                                        lw=1.0, mutation_scale=12))
        ax_kde.set_xlabel('Δ cluster size (cosine)', fontsize=16)

        # ---- global: PCA of prototype shifts ----
        ax_pca = fig.add_subplot(gs[2, col])
        axes_pca[col] = ax_pca
        pca = PCA(n_components=2)
        pca.fit(md['X_b'].cpu().numpy())
        pb = pca.transform(md['prot_b'].cpu().numpy())
        pm = pca.transform(md['prot_a'].cpu().numpy())

        for i in range(pb.shape[0]):
            ax_pca.plot([pb[i, 0], pm[i, 0]], [pb[i, 1], pm[i, 1]],
                        color='#bbbbbb', ls='-', lw=1, zorder=1)
        ax_pca.scatter(pb[:, 0], pb[:, 1], c=C_BASE, s=45, alpha=0.8,
                       zorder=2, edgecolors='none')
        ax_pca.scatter(pm[:, 0], pm[:, 1], c=C_AFTER, s=45,
                       zorder=3, edgecolors='none')

        all_pts = np.vstack([pb, pm])
        span_p = max(all_pts[:, 0].ptp(), all_pts[:, 1].ptp())
        cx = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2
        cy = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2
        half = span_p / 2 + span_p * 0.12
        ax_pca.set_xlim(cx - half, cx + half)
        ax_pca.set_ylim(cy - half, cy + half)
        ax_pca.set_aspect('equal')

        for spine in ax_pca.spines.values():
            spine.set_visible(False)
        ax_pca.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        xmin, xmax = ax_pca.get_xlim()
        ymin, ymax = ax_pca.get_ylim()
        ax_pca.annotate('', xy=(xmax, 0), xytext=(xmin, 0),
                        arrowprops=dict(arrowstyle='->', color='k', lw=1.5))
        ax_pca.annotate('', xy=(0, ymax), xytext=(0, ymin),
                        arrowprops=dict(arrowstyle='->', color='k', lw=1.5))
        ax_pca.text(xmax, 0.015 * (ymax - ymin), 'PC1', ha='right', va='bottom', fontsize=12)
        ax_pca.text(0.015 * (xmax - xmin), ymax, 'PC2', ha='left', va='top', fontsize=12)

    # Local / Global row labels
    fig.canvas.draw()
    _, y_kde_mid = fig.transFigure.inverted().transform(
        axes_kde[n_cols - 1].transAxes.transform((0.5, 0.5)))
    fig.text(0.068, y_kde_mid, 'Local', ha='center', va='center', fontsize=22, rotation=90)
    _, y_pca_mid = fig.transFigure.inverted().transform(
        axes_pca[n_cols - 1].transAxes.transform((0.5, 0.5)))
    fig.text(0.068, y_pca_mid + 0.01, 'Global', ha='center', va='center',
             fontsize=22, rotation=90)

    fig.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_BASE,
               markersize=22, label='Baseline'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_AFTER,
               markersize=22, label='After Recurrence'),
    ], loc='lower center', ncol=2, fontsize=22, frameon=False,
        bbox_to_anchor=(0.50, 0.015), columnspacing=2.5, handletextpad=0.8)

    _save_show(fig, save_path, show=show)


def plot_top1_accuracy_bars(features, labels, exclude=("CLIP",), figsize=None,
                            save_path=None, title="ImageNet Top-1 Accuracy (from logits)",
                            show_title=True, show=True):
    """Grouped bar plot of top-1 accuracy from logits.

    Recurrent models get two bars (baseline solid, after lighter with a
    dashed edge — mirroring the ridge-plot line styles).
    """
    labels_t = torch.as_tensor(labels)

    bars_data = []
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
            bars_data.append((model, states))
    if not bars_data:
        return

    n_models = len(bars_data)
    figsize = figsize or (max(14.0, n_models * 1.7), 6.5)
    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.38

    for i, (model, states) in enumerate(bars_data):
        base_color = MODEL_COLOR_MAP.get(model, {}).get("baseline") or "#888888"
        after_color = MODEL_COLOR_MAP.get(model, {}).get("after") or base_color
        if len(states) == 1:
            _, acc = states[0]
            ax.bar(i, acc, width=bar_width * 1.7, color=base_color, linewidth=0)
            ax.text(i, acc + 1.2, f"{acc:.1f}", ha='center', va='bottom',
                    fontsize=16, fontweight='bold')
        else:
            for j, (state, acc) in enumerate(states):
                offset = (j - 0.5) * bar_width
                if state == "baseline":
                    ax.bar(i + offset, acc, width=bar_width, color=base_color, linewidth=0)
                else:
                    ax.bar(i + offset, acc, width=bar_width, color=after_color,
                           alpha=0.65, edgecolor=base_color, linewidth=2.0, linestyle='--')
                ax.text(i + offset, acc + 1.2, f"{acc:.1f}", ha='center', va='bottom',
                        fontsize=14, fontweight='bold')

    ax.set_xticks(range(n_models))
    ax.set_xticklabels([m for m, _ in bars_data], rotation=30, ha='right',
                       fontsize=20, fontweight='bold')
    max_acc = max(acc for _, ss in bars_data for _, acc in ss)
    ax.set_ylim(0, max_acc * 1.25)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=20)
    ax.tick_params(axis='y', labelsize=14)
    if show_title:
        ax.set_title(title, fontsize=25, pad=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(facecolor='#555555', edgecolor='none', label='Baseline'),
        Patch(facecolor='#cccccc', alpha=0.65, edgecolor='#555555',
              linewidth=2.0, linestyle='--', label='After recurrence'),
    ], loc='lower right', bbox_to_anchor=(1.0, 0.88), ncol=2, fontsize=20,
        frameon=False, handletextpad=0.6, columnspacing=1.5)

    plt.tight_layout()
    _save_show(fig, save_path, show=show)


# ----------------------------------------------------------------- RSA / MDS
def build_rsa_matrix(between_dict, exclude=()):
    """KxK Spearman-rho matrix across models from between-prototype vectors."""
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
    """MDS embedding of models; distance = 1 - Spearman rho between RDMs."""
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


def plot_rsa_mds_composite(between_log, between_pen, rdms_log, rdms_pen,
                           exclude=("CLIP",), save_path=None, seed=42, show=True):
    """2x2 composite: cross-model RSA heatmaps (top) and model MDS (bottom),
    each for penultimate (left) and logits (right)."""
    names_rsa_log, rsa_log = build_rsa_matrix(between_log, exclude)
    names_rsa_pen, rsa_pen = build_rsa_matrix(between_pen, exclude)
    names_mds_log, coords_log, D_log, stress_log = build_mds_coords(rdms_log, seed, exclude)
    names_mds_pen, coords_pen, D_pen, stress_pen = build_mds_coords(rdms_pen, seed, exclude)

    fig = plt.figure(figsize=(15, 13.5))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.04], height_ratios=[1, 1],
                  hspace=0.30, wspace=0.30, left=0.08, right=0.92, top=0.95, bottom=0.05)
    ax_rsa_pen = fig.add_subplot(gs[0, 0])
    ax_rsa_log = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_mds_pen = fig.add_subplot(gs[1, 0])
    ax_mds_log = fig.add_subplot(gs[1, 1])

    im = None
    for ax, rsa, names, subtitle in [
            (ax_rsa_pen, rsa_pen, names_rsa_pen, "Penultimate"),
            (ax_rsa_log, rsa_log, names_rsa_log, "Logits")]:
        K = rsa.shape[0]
        display = [DISPLAY_NAMES.get(n, n) for n in names]
        im = ax.imshow(rsa, vmin=0, vmax=1, cmap="viridis_r", interpolation="nearest")
        ax.set_xticks(range(K))
        ax.set_xticklabels(display, rotation=45, ha="right", fontsize=12)
        ax.set_yticks(range(K))
        ax.set_yticklabels(display, fontsize=12)
        for i in range(K):
            for j in range(K):
                val = rsa[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val >= 0.6 else "black", fontsize=8)
        ax.set_title(f"Cross-model RSA — {subtitle}", fontsize=14, pad=10)

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Spearman ρ", fontsize=11)
    ax_cbar.tick_params(labelsize=9)

    for ax, names, coords, subtitle in [
            (ax_mds_pen, names_mds_pen, coords_pen, "Penultimate"),
            (ax_mds_log, names_mds_log, coords_log, "Logits")]:
        for i, name in enumerate(names):
            model_key = name.split()[0]
            color = pick_color(name) or '#888888'
            state = name.split()[1] if len(name.split()) > 1 else "baseline"
            if model_key in RECURRENT_MODELS and state == "baseline":
                ax.scatter(*coords[i], marker='o', s=400, zorder=5,
                           facecolors='none', edgecolors=color, linewidths=2.2)
            else:
                ax.scatter(*coords[i], marker='o', s=400, zorder=5,
                           c=color, edgecolors='white', linewidths=0.7)
        for model in RECURRENT_MODELS:
            bk, ak = f"{model} baseline", f"{model} after"
            if bk in names and ak in names:
                ib, ia = names.index(bk), names.index(ak)
                ax.annotate("", xy=coords[ia], xytext=coords[ib],
                            arrowprops=dict(arrowstyle="-|>", color=pick_color(ak) or '#888',
                                            lw=1.5, alpha=0.6, shrinkA=8, shrinkB=8))

        cx = float((coords[:, 0].min() + coords[:, 0].max()) / 2)
        cy = float((coords[:, 1].min() + coords[:, 1].max()) / 2)
        half = float(max(coords[:, 0].ptp(), coords[:, 1].ptp())) * 0.55
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('#bbbbbb')
        ax.set_title(f"Model MDS — {subtitle}", fontsize=14, pad=10)

    _save_show(fig, save_path, dpi=300, show=show)
    return {
        'rsa_log': rsa_log, 'rsa_pen': rsa_pen,
        'mds_log': (names_mds_log, coords_log, D_log, stress_log),
        'mds_pen': (names_mds_pen, coords_pen, D_pen, stress_pen),
    }


def plot_model_mds_3d(rdms, title, seed=42, save_html=None, show=True):
    """Interactive 3-D model MDS (plotly); same metric as the 2-D version."""
    import plotly.graph_objects as go

    names, coords, D, stress = build_mds_coords(rdms, seed=seed, n_components=3)
    colors = [pick_color(n) or '#888888' for n in names]
    displays = [DISPLAY_NAMES.get(n, n) for n in names]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers+text',
        marker=dict(size=8, color=colors, line=dict(width=1, color='white')),
        text=displays, textposition='top center', textfont=dict(size=9),
        hovertemplate='%{text}<extra></extra>'))

    for model in RECURRENT_MODELS:
        bk, ak = f"{model} baseline", f"{model} after"
        if bk in names and ak in names:
            ib, ia = names.index(bk), names.index(ak)
            arrow_color = pick_color(ak) or '#888'
            fig.add_trace(go.Scatter3d(
                x=[coords[ib, 0], coords[ia, 0]], y=[coords[ib, 1], coords[ia, 1]],
                z=[coords[ib, 2], coords[ia, 2]], mode='lines',
                line=dict(color=arrow_color, width=3), showlegend=False, hoverinfo='skip'))
            d = coords[ia] - coords[ib]
            fig.add_trace(go.Cone(
                x=[coords[ia, 0]], y=[coords[ia, 1]], z=[coords[ia, 2]],
                u=[d[0]], v=[d[1]], w=[d[2]], sizemode='absolute', sizeref=0.04,
                colorscale=[[0, arrow_color], [1, arrow_color]],
                showscale=False, showlegend=False, hoverinfo='skip'))

    fig.update_layout(
        title=dict(text=f"{title}<br><sub>stress = {stress:.2f}</sub>",
                   x=0.5, font=dict(size=14)),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), bgcolor='white'),
        width=750, height=650, margin=dict(l=10, r=10, t=60, b=10))

    if save_html:
        fig.write_html(save_html)
        print(f"Saved -> {save_html}")
    if show:
        fig.show()
    return names, coords, D


# ------------------------------------------------------- untrained (multi-seed)
def plot_separation_ridge_multiseed(seeded_between_dict,
                                    xlabel="Between-prototype cosine distance",
                                    exclude=(), figsize=None, save_path=None,
                                    band="sd", show_n_seeds=True, xlim=None,
                                    xlim_quantile=(0.005, 0.995),
                                    xlim_pad_frac=0.05, show=True):
    """Ridge plot with a seed-variability band per model/state.

    Curves are the mean KDE across seeds; the band is +/-1 SD ("sd"),
    min/max ("minmax"), IQR ("iqr"), or every seed drawn faintly
    ("individual"). xlim=None auto-fits to data quantiles, since untrained
    separations can be far from the fixed [0, 2] window.
    """
    model_rows = []
    for model in MODEL_ORDER:
        if model in exclude:
            continue
        entries = [(f"{model} {s}", s) for s in ("baseline", "after")
                   if f"{model} {s}" in seeded_between_dict]
        if entries:
            model_rows.append((model, entries))
    if not model_rows:
        print("No models in seeded_between_dict to plot.")
        return

    if xlim is None:
        all_vals = np.concatenate([
            arr for _, entries in model_rows for tag, _ in entries
            for arr in seeded_between_dict[tag]])
        q_lo, q_hi = xlim_quantile
        lo, hi = np.quantile(all_vals, q_lo), np.quantile(all_vals, q_hi)
        pad = (hi - lo) * xlim_pad_frac
        x_lo, x_hi = max(0.0, float(lo - pad)), min(2.0, float(hi + pad))
    else:
        x_lo, x_hi = xlim

    n_rows = len(model_rows)
    figsize = figsize or (8.0, 1.1 * n_rows + 1.4)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    axes = [axes] if n_rows == 1 else list(axes)
    x_grid = np.linspace(x_lo, x_hi, 500)

    for row_i, (model, entries) in enumerate(model_rows):
        ax = axes[row_i]
        for tag, state in entries:
            arrs = seeded_between_dict[tag]
            densities = np.stack([gaussian_kde(a)(x_grid) for a in arrs])
            mean_y = densities.mean(axis=0)
            color = pick_color(tag) or "#888888"
            ls = "-" if state == "baseline" else "--"
            lw = 2.0 if state == "baseline" else 2.2
            alpha_band = 0.28 if state == "baseline" else 0.22
            alpha_fill = 0.18 if state == "baseline" else 0.14

            if band == "individual":
                for d in densities:
                    ax.plot(x_grid, d, color=color, lw=0.8, alpha=0.35, ls=ls)
            else:
                if band == "sd":
                    sd_y = densities.std(axis=0)
                    lo_y, hi_y = np.maximum(mean_y - sd_y, 0), mean_y + sd_y
                elif band == "minmax":
                    lo_y, hi_y = densities.min(axis=0), densities.max(axis=0)
                elif band == "iqr":
                    lo_y, hi_y = np.percentile(densities, [25, 75], axis=0)
                else:
                    raise ValueError(f"Unknown band: {band}")
                ax.fill_between(x_grid, lo_y, hi_y, color=color,
                                alpha=alpha_band, linewidth=0)
            ax.fill_between(x_grid, mean_y, alpha=alpha_fill, color=color)
            ax.plot(x_grid, mean_y, color=color, ls=ls, lw=lw)

        disp = DISPLAY_NAMES.get(model, model)
        if show_n_seeds:
            disp = f"{disp}  (n={len(seeded_between_dict[entries[0][0]])} seeds)"
        ax.text(0.015, 0.92, disp, transform=ax.transAxes, fontsize=14,
                fontweight="bold", ha="left", va="top", fontstyle="italic")
        _ridge_style_row(ax, row_i == n_rows - 1, xlabel)

    axes[-1].set_xlim(x_lo, x_hi)
    if x_lo <= 0 <= x_hi:
        axes[0].text(0.0, axes[0].get_ylim()[1] * 1.02, "Same",
                     ha="center", va="bottom", fontsize=13, fontstyle="italic")
    if x_lo <= 1 <= x_hi:
        axes[0].text(1.0, axes[0].get_ylim()[1] * 1.02, "Orthogonal",
                     ha="center", va="bottom", fontsize=13, fontstyle="italic")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    _add_column_vlines(fig, axes[0], axes[-1],
                       x_values=[v for v in (0.0, 1.0) if x_lo <= v <= x_hi])
    _save_show(fig, save_path, show=show)


def plot_local_global_composite_seeded(seeded_features, labels, rep="logits",
                                       models=tuple(RECURRENT_MODELS),
                                       rep_overrides=None, annotations=None,
                                       save_path=None, figsize=None,
                                       seed_for_pca=0, band="sd", show=True):
    """Multi-seed variant of plot_local_global_composite.

    The local KDE row shows the mean across seeds with a variability band;
    the global PCA row uses one representative seed, because each seed has
    its own random weights and therefore its own PCA basis.
    """
    from sklearn.decomposition import PCA

    rep_overrides = rep_overrides or {}
    annotations = annotations or DEFAULT_MECHANISM_ANNOTATIONS
    C_BASE, C_AFTER, C_FILL = "#555555", "#E41A1C", "#cccccc"

    labels_t = torch.as_tensor(labels)
    classes = torch.unique(labels_t, sorted=True).tolist()

    avail = []
    for m in models:
        m_rep = rep_overrides.get(m, rep)
        if any("baseline" in sf.get(m, {}).get(m_rep, {})
               and "after" in sf.get(m, {}).get(m_rep, {})
               for sf in seeded_features.values()):
            avail.append(m)
    n_cols = len(avail)
    if n_cols == 0:
        print("No recurrent models with both baseline & after across seeds.")
        return

    figsize = figsize or (4.0 * n_cols, 9.0)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, n_cols, height_ratios=[0.55, 2.0, 4.0],
                          hspace=0.3, wspace=0.28,
                          top=0.91, bottom=0.10, left=0.09, right=0.93)

    per_model = {m: [] for m in avail}
    all_deltas = []
    for model in avail:
        m_rep = rep_overrides.get(model, rep)
        for seed, models_dict in seeded_features.items():
            d = models_dict.get(model, {}).get(m_rep, {})
            if "baseline" not in d or "after" not in d:
                continue
            X_b = to_feat(d["baseline"]).float()
            X_a = to_feat(d["after"]).float()
            prot_b = compute_prototypes(X_b, labels_t, classes)
            prot_a = compute_prototypes(X_a, labels_t, classes)
            delta = (exemplar_proto_dists(X_a, prot_a, labels_t, metric="cosine")
                     - exemplar_proto_dists(X_b, prot_b, labels_t, metric="cosine"))
            per_model[model].append((seed, X_b, prot_b, prot_a, delta))
            all_deltas.append(delta)

    all_delta = np.concatenate(all_deltas)
    pad_d = (all_delta.max() - all_delta.min()) * 0.08
    shared_xmin = float(all_delta.min()) - pad_d
    shared_xmax = float(all_delta.max()) + pad_d

    axes_kde, axes_pca = {}, {}
    for col, model in enumerate(avail):
        ax_ann = fig.add_subplot(gs[0, col])
        ax_ann.axis("off")
        ax_ann.set_title(model, fontsize=30, fontweight="bold", pad=6)
        ax_ann.text(0.5, 0.22, annotations.get(model, ""), ha="center", va="center",
                    fontsize=16, fontstyle="italic", color="#555555",
                    transform=ax_ann.transAxes, linespacing=1.3)

        # local KDE: mean across seeds + band
        ax_kde = fig.add_subplot(gs[1, col])
        axes_kde[col] = ax_kde
        deltas = [tup[4] for tup in per_model[model]]
        x_grid = np.linspace(shared_xmin, shared_xmax, 400)
        densities = np.stack([gaussian_kde(d)(x_grid) for d in deltas])
        mean_y = densities.mean(axis=0)
        if band == "sd":
            sd_y = densities.std(axis=0)
            lo_y, hi_y = np.maximum(mean_y - sd_y, 0), mean_y + sd_y
        elif band == "minmax":
            lo_y, hi_y = densities.min(axis=0), densities.max(axis=0)
        elif band == "iqr":
            lo_y, hi_y = np.percentile(densities, [25, 75], axis=0)
        else:
            raise ValueError(f"unknown band={band}")

        ax_kde.fill_between(x_grid, lo_y, hi_y, color=C_FILL, alpha=0.45, linewidth=0)
        ax_kde.fill_between(x_grid, mean_y, color=C_FILL, alpha=0.6)
        ax_kde.plot(x_grid, mean_y, color="#888888", lw=1.0)
        ax_kde.axvline(0, color="black", lw=1.5, ls="-", zorder=0)
        mean_d = float(np.concatenate(deltas).mean())
        ax_kde.axvline(mean_d, color="red", lw=2, ls="--", zorder=5)

        ax_kde.set_xlim(shared_xmin, shared_xmax)
        ax_kde.set_yticks([])
        for side in ("top", "right", "left"):
            ax_kde.spines[side].set_visible(False)
        ax_kde.spines["bottom"].set_color("#888888")
        ax_kde.spines["bottom"].set_linewidth(0.8)
        ax_kde.tick_params(axis="x", labelsize=12, colors="#333333")

        peak = mean_y.max()
        span = shared_xmax - shared_xmin
        if mean_d < 0:
            label_txt, x0, x1, xt = "Compact", 0.24, 0.12, 0.18
        else:
            label_txt, x0, x1, xt = "Expand", 0.76, 0.88, 0.82
        ax_kde.text(shared_xmin + span * xt, peak * 0.50, label_txt, ha="center",
                    va="bottom", fontsize=15, fontstyle="italic", color="black")
        ax_kde.annotate("", xy=(shared_xmin + span * x1, peak * 0.45),
                        xytext=(shared_xmin + span * x0, peak * 0.45),
                        arrowprops=dict(arrowstyle="->", color="black",
                                        lw=1.0, mutation_scale=12))
        ax_kde.set_xlabel("Δ cluster size (cosine)", fontsize=16)

        # global PCA: representative seed
        ax_pca = fig.add_subplot(gs[2, col])
        axes_pca[col] = ax_pca
        entry = next((t for t in per_model[model] if t[0] == seed_for_pca),
                     per_model[model][0])
        _, X_b, prot_b, prot_a, _ = entry
        pca = PCA(n_components=2)
        pca.fit(X_b.cpu().numpy())
        pb = pca.transform(prot_b.cpu().numpy())
        pa = pca.transform(prot_a.cpu().numpy())

        for i in range(pb.shape[0]):
            ax_pca.plot([pb[i, 0], pa[i, 0]], [pb[i, 1], pa[i, 1]],
                        color="#bbbbbb", ls="-", lw=1, zorder=1)
        ax_pca.scatter(pb[:, 0], pb[:, 1], c=C_BASE, s=45, alpha=0.8,
                       zorder=2, edgecolors="none")
        ax_pca.scatter(pa[:, 0], pa[:, 1], c=C_AFTER, s=45,
                       zorder=3, edgecolors="none")

        all_pts = np.vstack([pb, pa])
        span_p = max(all_pts[:, 0].ptp(), all_pts[:, 1].ptp())
        cx = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2
        cy = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2
        half = span_p / 2 + span_p * 0.12
        ax_pca.set_xlim(cx - half, cx + half)
        ax_pca.set_ylim(cy - half, cy + half)
        ax_pca.set_aspect("equal")
        for sp in ax_pca.spines.values():
            sp.set_visible(False)
        ax_pca.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        xmin, xmax = ax_pca.get_xlim()
        ymin, ymax = ax_pca.get_ylim()
        ax_pca.annotate("", xy=(xmax, 0), xytext=(xmin, 0),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
        ax_pca.annotate("", xy=(0, ymax), xytext=(0, ymin),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
        ax_pca.text(xmax, 0.015 * (ymax - ymin), "PC1", ha="right", va="bottom", fontsize=12)
        ax_pca.text(0.015 * (xmax - xmin), ymax, "PC2", ha="left", va="top", fontsize=12)

    fig.canvas.draw()
    _, y_kde_mid = fig.transFigure.inverted().transform(
        axes_kde[n_cols - 1].transAxes.transform((0.5, 0.5)))
    fig.text(0.068, y_kde_mid, "Local", ha="center", va="center", fontsize=22, rotation=90)
    _, y_pca_mid = fig.transFigure.inverted().transform(
        axes_pca[n_cols - 1].transAxes.transform((0.5, 0.5)))
    fig.text(0.068, y_pca_mid + 0.01, "Global", ha="center", va="center",
             fontsize=22, rotation=90)

    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_BASE,
               markersize=22, label="Baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AFTER,
               markersize=22, label="After Recurrence"),
    ], loc="lower center", ncol=2, fontsize=22, frameon=False,
        bbox_to_anchor=(0.50, 0.015), columnspacing=2.5, handletextpad=0.8)

    _save_show(fig, save_path, show=show)


def plot_convrnn_supplement(X_before, X_after, labels,
                            before_label="Image on (t=12)", after_label="Final (t=17)",
                            max_k=50, save_path=None, show=True):
    """Supplement figure (paper Fig. 4): ConvRNN input-on vs final state.

    Four panels, left to right:
      (a) local cluster-size change KDE (after - before)
      (b) k-NN category preservation curve, before vs after
      (c) PCA of prototype shifts (PCA fit on the before cloud)
      (d) histogram of per-category prototype shift magnitude
    """
    from sklearn.decomposition import PCA
    from repgeo.geometry import (
        compute_prototypes, exemplar_proto_dists, category_preservation_knn,
        prototype_shifts)

    C_BEFORE, C_AFTER = '#555555', '#E41A1C'
    labels_t = torch.as_tensor(labels)
    classes = torch.unique(labels_t, sorted=True).tolist()
    Xb, Xa = to_feat(X_before).float(), to_feat(X_after).float()
    prot_b = compute_prototypes(Xb, labels_t, classes)
    prot_a = compute_prototypes(Xa, labels_t, classes)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    # (a) cluster-size change KDE
    ax = axes[0]
    d_b = exemplar_proto_dists(Xb, prot_b, labels_t, metric="cosine")
    d_a = exemplar_proto_dists(Xa, prot_a, labels_t, metric="cosine")
    diff = d_a - d_b
    mean_d = float(diff.mean())
    pad = (diff.max() - diff.min()) * 0.08
    xs = np.linspace(diff.min() - pad, diff.max() + pad, 400)
    dens = gaussian_kde(diff)(xs)
    ax.fill_between(xs, dens, color="#cccccc", alpha=0.6)
    ax.plot(xs, dens, color="#888888", lw=1.0)
    ax.axvline(0, color="black", ls=":", lw=1.3, label="No change")
    ax.axvline(mean_d, color=C_AFTER, ls="--", lw=1.8, label=f"Mean = {mean_d:+.3f}")
    peak = dens.max()
    span = xs[-1] - xs[0]
    if mean_d < 0:
        txt, a0, a1, tx, loc = "Compact", 0.24, 0.12, 0.18, "upper right"
    else:
        txt, a0, a1, tx, loc = "Expand", 0.76, 0.88, 0.82, "upper left"
    ax.text(xs[0] + span * tx, peak * 0.52, txt, ha="center", va="bottom",
            fontsize=14, fontstyle="italic")
    ax.annotate("", xy=(xs[0] + span * a1, peak * 0.46),
                xytext=(xs[0] + span * a0, peak * 0.46),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0, mutation_scale=12))
    ax.set_title("Change in Cluster Size", fontsize=14)
    ax.set_xlabel("Δ exemplar–prototype distance (cosine)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc=loc, fontsize=9, frameon=True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # (b) k-NN category preservation
    ax = axes[1]
    ks_b, cp_b = category_preservation_knn(Xb, labels_t, max_k=max_k)
    ks_a, cp_a = category_preservation_knn(Xa, labels_t, max_k=max_k)
    ax.plot(ks_a, cp_a, color=C_AFTER, lw=3, label=after_label)
    ax.plot(ks_b, cp_b, color=C_BEFORE, lw=3, ls="--", label=before_label)
    ax.set_title("Local Category Preservation (k-NN)", fontsize=14)
    ax.set_xlabel("Number of Neighbors (k)", fontsize=12)
    ax.set_ylabel("% Same-Category Neighbors", fontsize=12)
    ax.grid(axis="y", ls=":", alpha=0.6)
    ax.legend(fontsize=9, frameon=True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # (c) PCA of prototype shifts
    ax = axes[2]
    pca = PCA(n_components=2).fit(Xb.cpu().numpy())
    Pb = pca.transform(prot_b.cpu().numpy())
    Pa = pca.transform(prot_a.cpu().numpy())
    for i in range(Pb.shape[0]):
        ax.plot([Pb[i, 0], Pa[i, 0]], [Pb[i, 1], Pa[i, 1]],
                color="#bbbbbb", lw=0.9, alpha=0.8, zorder=1)
    ax.scatter(Pb[:, 0], Pb[:, 1], c=C_BEFORE, s=45, alpha=0.85,
               zorder=2, edgecolors="none", label=before_label)
    ax.scatter(Pa[:, 0], Pa[:, 1], c=C_AFTER, s=45, zorder=3,
               edgecolors="none", label=after_label)
    pts = np.vstack([Pb, Pa])
    span_p = max(pts[:, 0].ptp(), pts[:, 1].ptp())
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2
    half = span_p / 2 + span_p * 0.14
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.annotate("", xy=(x1, 0), xytext=(x0, 0), arrowprops=dict(arrowstyle="->", color="k", lw=1.4))
    ax.annotate("", xy=(0, y1), xytext=(0, y0), arrowprops=dict(arrowstyle="->", color="k", lw=1.4))
    ax.text(x1, 0.015 * (y1 - y0), "PC1", ha="right", va="bottom", fontsize=11)
    ax.text(-0.015 * (x1 - x0), y1, "PC2", ha="right", va="top", fontsize=11)
    ax.set_title("Prototype Shifts in PCA Space", fontsize=14, pad=10)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    # (d) prototype shift histogram
    ax = axes[3]
    shifts = prototype_shifts(prot_b, prot_a)
    mean_s = float(shifts.mean())
    ax.hist(shifts, bins=30, color="#cccccc", edgecolor="#333333", linewidth=0.6, alpha=0.9)
    ax.axvline(mean_s, color=C_AFTER, ls="--", lw=1.8, label=f"Mean = {mean_s:.3f}")
    ax.set_title("Prototype Shift Distribution", fontsize=14)
    ax.set_xlabel("Prototype shift (1 − cosine)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=9, frameon=True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    _save_show(fig, save_path, show=show)
