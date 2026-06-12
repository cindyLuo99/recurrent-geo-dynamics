#!/usr/bin/env python
"""Run the full representational-geometry analysis from saved activations.

Needs only the activations directory (no GPU, no model code). If your
activations live in ./activations (the default download location), simply:

  python scripts/run_main_analysis.py

or point at another location:

  python scripts/run_main_analysis.py --activations /path/to/activations

Produces:
  results/accuracy_logits.csv          standard top-1/5 per model/state
  results/accuracy_nearest_centroid.csv
  results/geometry_summary.csv         per-model geometry table (paper Table)
  results/local_lme.csv                LME test of local cluster change
  results/global_permutation.csv       permutation test of global separation
  figures/between_class_dual.pdf       ridge plot, penult + logits
  figures/separation_ridge_{logits,penult}.pdf
  figures/local_global_composite_{logits,penult}.pdf
  figures/rsa_mds_composite.{svg,pdf}
  figures/model_mds_3d_{logits,penult}.html
  figures/top1_accuracy_bars.pdf
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from repgeo.loading import load_features, sanity_check
from repgeo.analysis import (
    compute_cluster_sizes_and_rdms, build_prototype_rdms, accuracy_table,
    geometry_summary_table,
)
from repgeo.stats import run_local_lme_tests, run_global_permutation_tests
from repgeo import plotting as P


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", default="./activations",
                        help="Directory of saved activations (see README). "
                             "Default: ./activations")
    parser.add_argument("--suffix", default="imagenetval_100x50")
    parser.add_argument("--out", default="./results")
    parser.add_argument("--figures", default="./figures")
    parser.add_argument("--n-perm", type=int, default=10000,
                        help="Permutations for the global separation test.")
    parser.add_argument("--skip-stats", action="store_true",
                        help="Skip the (slow) permutation test.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    fig = lambda name: os.path.join(args.figures, name)

    print("Loading activations ...")
    features, labels = load_features(args.activations, suffix=args.suffix)
    sanity_check(features, labels, verbose=False)
    print(f"  {len(features)} models, {len(labels)} stimuli")

    # ---- tables ----
    acc_log = accuracy_table(features, labels, source="logits")
    acc_log.to_csv(f"{args.out}/accuracy_logits.csv", index=False)
    print("\nImageNet accuracy (logits argmax):")
    print(acc_log.to_string(index=False))

    acc_nc = accuracy_table(features, labels, source="penult")
    acc_nc.to_csv(f"{args.out}/accuracy_nearest_centroid.csv", index=False)

    summary = geometry_summary_table(features, labels)
    summary.to_csv(f"{args.out}/geometry_summary.csv", index=False)
    print("\nGeometry summary (baseline vs after recurrence):")
    print(summary.to_string(index=False))

    if not args.skip_stats:
        lme = run_local_lme_tests(features, labels)
        lme.to_csv(f"{args.out}/local_lme.csv", index=False)
        print("\nLocal cluster change (LME, distance_diff ~ 1 + (1|category)):")
        print(lme.to_string(index=False))

        print(f"\nGlobal separation permutation test ({args.n_perm} permutations) ...")
        perm = run_global_permutation_tests(features, labels, n_perm=args.n_perm)
        perm.to_csv(f"{args.out}/global_permutation.csv", index=False)
        print(perm.to_string(index=False))

    # ---- figures ----
    print("\nBuilding figures ...")
    cluster_log, between_log = compute_cluster_sizes_and_rdms(features, labels, rep="logits")
    cluster_pen, between_pen = compute_cluster_sizes_and_rdms(features, labels, rep="penult")
    rdms_log = build_prototype_rdms(features, labels, rep="logits")
    rdms_pen = build_prototype_rdms(features, labels, rep="penult")

    P.plot_separation_ridge(between_log, save_path=fig("separation_ridge_logits.pdf"),
                            show=False)
    P.plot_separation_ridge(between_pen, save_path=fig("separation_ridge_penult.pdf"),
                            show_stats=False, show=False)
    P.plot_separation_ridge_dual(between_pen, between_log,
                                 save_path=fig("between_class_dual.pdf"), show=False)
    P.plot_local_global_composite(
        features, labels, rep="logits", rep_overrides={"CORnet-RT": "penult"},
        save_path=fig("local_global_composite_logits.pdf"), show=False)
    P.plot_local_global_composite(
        features, labels, rep="penult",
        save_path=fig("local_global_composite_penult.pdf"), show=False)
    P.plot_top1_accuracy_bars(features, labels,
                              save_path=fig("top1_accuracy_bars.pdf"), show=False)
    P.plot_rsa_mds_composite(between_log, between_pen, rdms_log, rdms_pen,
                             save_path=fig("rsa_mds_composite.svg"), show=False)
    try:
        P.plot_model_mds_3d(rdms_log, "Model MDS 3-D — LOGITS",
                            save_html=fig("model_mds_3d_logits.html"), show=False)
        P.plot_model_mds_3d(rdms_pen, "Model MDS 3-D — PENULTIMATE",
                            save_html=fig("model_mds_3d_penult.html"), show=False)
    except ImportError:
        print("plotly not installed — skipping 3-D MDS html figures.")

    print("\nDone.")


if __name__ == "__main__":
    main()
