#!/usr/bin/env python
"""Untrained-controls (supplement) analysis from saved multi-seed activations.

  python run_untrained_analysis.py --activations ./activations/untrained \
      --out ./results/untrained --figures ./figures/untrained

Produces:
  results/untrained/geometry_summary_per_seed.csv
  results/untrained/geometry_summary_meanSD.csv
  figures/untrained/ridge_untrained_{logits,penult}.pdf
  figures/untrained/local_global_composite_untrained.pdf

The geometry metrics are computed by the same functions as the trained
analysis (repgeo.stats.pair_geometry_metrics); only the seed aggregation
differs.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from repgeo.untrained import (
    load_untrained_features, compute_seeded_between_dict, geometry_summary_seeded,
    local_global_composite_seeded,
)
from repgeo.plotting import (
    plot_separation_ridge_multiseed, plot_local_global_composite_seeded,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", default="./activations/untrained",
                        help="Untrained activations root (one dir per model). "
                             "Default: ./activations/untrained")
    parser.add_argument("--suffix", default="imagenetval_100x50")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default="./results/untrained")
    parser.add_argument("--figures", default="./figures/untrained")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    print("Loading untrained activations ...")
    seeded, labels, missing = load_untrained_features(
        args.activations, suffix=args.suffix, seeds=tuple(args.seeds))
    n_loaded = sum(len(m) for m in seeded.values())
    print(f"  loaded {n_loaded} (model, seed) combos; {len(missing)} missing")
    if n_loaded == 0:
        sys.exit("No untrained activations found — check --activations/--suffix.")

    # ---- geometry summary, mean ± SD across seeds ----
    df_long, df_summary = geometry_summary_seeded(seeded, labels)
    df_long.to_csv(f"{args.out}/geometry_summary_per_seed.csv", index=False)
    df_summary.to_csv(f"{args.out}/geometry_summary_meanSD.csv", index=False)
    print("\nUNTRAINED geometry summary (mean ± SD across seeds):")
    print(df_summary.to_string(index=False))

    # ---- multi-seed ridge plots ----
    for rep in ("logits", "penult"):
        between = compute_seeded_between_dict(seeded, labels, rep=rep)
        plot_separation_ridge_multiseed(
            between,
            xlabel=f"Between-prototype cosine distance — {rep} (untrained)",
            xlim_quantile=(0.05, 0.95), band="sd",
            save_path=os.path.join(args.figures, f"ridge_untrained_{rep}.pdf"),
            show=False)

    # ---- local/global composite (analysis first; PCA on representative seed) ----
    composite = local_global_composite_seeded(
        seeded, labels, rep="logits", rep_overrides={"CORnet-RT": "penult"},
        seed_for_pca=args.seeds[0])
    plot_local_global_composite_seeded(
        composite, band="sd",
        save_path=os.path.join(args.figures, "local_global_composite_untrained.pdf"),
        show=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
