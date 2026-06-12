# Geometric Dynamics Across Recurrent Vision Models

[Kexin Cindy Luo](https://www.cindyluo.com) $^{1,2,3}$, [George A. Alvarez](https://visionlab.harvard.edu/george/bio) $^{1,2,3}$, [Talia Konkle](https://psychology.fas.harvard.edu/people/talia-konkle) $^{1,2,3,4}$

$^1$ Harvard University, $^2$ Kempner Institute for the Study of Natural and Artificial Intelligience, $^3$ Vision-Sciences Laboratory, $^4$ Center For Brain Science

Abstract: The human visual system contains rich feedforward, feedback, and bypass pathways that shape how object categories are represented over time. Inspired by this neuroanatomy, a growing set of deep vision models incorporates recurrent dynamics, yet how category representations differ across recurrent mechanisms remains unclear. Here we characterize the geometric signatures of recurrence in four distinct model families: two-pass long-range feedback networks (LRM/LRA; Konkle & Alvarez, 2023), two within-layer recurrent models (BL, CORnet-RT; Spoerer et al., 2020, Kubilius et al., 2018), and a recurrent model with skip connections and gating (ConvRNN; Nayebi et al., 2018). We find that these recurrent models are geometrically heterogeneous in two key aspects. First, at inference time, these models show distinct representational shifts across processing steps: LRM/LRA locally compacts category representations toward prototypes while preserving global between-category structure, whereas BL, CORnet-RT, and ConvRNN each exhibit distinct global drifts alongside local changes. Second, training with different recurrent architectures yields distinct learned decision-stage geometry: LRM/LRA develops near-orthogonal category prototypes, a pattern absent in their feedforward counterpart (AlexNet) and other recurrent vision models, but also present in some feedforward architectures (e.g., ResNets). These divergent signatures reveal that "recurrence" is not a single computational strategy in current models: different recurrent architectures exhibit distinct geometric trajectories and decision-stage structures. Such heterogeneity motivates more careful differentiation among recurrent vision models in both computational and neuroscientific contexts.

This repository contains the code for the
analyses comparing local cluster size (exemplar-to-prototype distance) and
global between-class separation across recurrent and feedforward networks
(LRM, LRA, BL, CORnet-RT, ConvRNN vs. AlexNet, VGG-16, ResNet-50/101,
robust ResNet-50, B, CLIP).

<!-- TODO: paper citation / preprint link -->

## Start here — choose your path

The guided notebooks and the batch scripts run the *same* underlying functions
(the `repgeo` package), so you can work entirely in Jupyter if you prefer.

| I want to… | Do this |
|---|---|
| **Reproduce the paper's tables & figures**, step by step with explanations | [notebooks/reproduce_paper.ipynb](notebooks/reproduce_paper.ipynb) (needs only the downloaded activations — no GPU, no model code) |
| **Apply the local/global geometry analysis to my own model** | [notebooks/analyze_your_own_model.ipynb](notebooks/analyze_your_own_model.ipynb) (runs out of the box on demo data; the whole contract is two `[N, D]` tensors + labels) |
| **Regenerate everything in one command** | `python scripts/run_main_analysis.py` (see Quickstart) |
| **Re-extract activations from the models themselves** | "Reproducing from scratch" below (GPU + model code needed) |

### Setup (one time)

```bash
git clone https://github.com/cindyLuo99/recurrent-geo-dynamics.git
cd recurrent-geo-dynamics
pip install -e .
```

`pip install -e .` installs the `repgeo` package and all analysis
dependencies, and makes `import repgeo` work everywhere — including in
Jupyter, regardless of which folder you launch it from.

## Repository layout

```
repgeo/                      importable analysis package — one rule per module:
  config.py                  model registry: names, colors, order, layer conventions
  data.py                    stimulus images -> dataloader (used by extraction)
  loading.py                 saved activation files -> the features dict
  geometry.py                raw tensors in -> numbers out (the metrics themselves)
  analysis.py                features dict in -> tables / summaries out
  stats.py                   anything that returns a p-value (LME, permutation)
  untrained.py               multi-seed untrained (random-weight) controls
  plotting.py                all paper figures
scripts/
  build_stimulus_set.py      rebuild the 100x50 ImageNet-val set from manifest
  run_main_analysis.py       full analysis: tables + stats + figures
  run_untrained_analysis.py  supplement: untrained multi-seed analysis
  extraction/
    extract_torch_models.py        PyTorch models (this environment)
    extract_untrained_torch.py     untrained controls, multi-seed
    tf_env/                        TensorFlow models (SEPARATE environment)
      extract_convrnn.py           ConvRNN (Nayebi et al. 2022)
      extract_rcnn_sat.py          BL / B (Spoerer et al. 2020)
      requirements_tf.txt
data/
  stimulus_set_manifest.csv  the exact 5000 ImageNet-val images used
  category_metadata.csv      the 100 categories (WordNet ID, index, label)
notebooks/
  reproduce_paper.ipynb          guided reproduction of every table & figure
  analyze_your_own_model.ipynb   apply the framework to your own model
```

## Quickstart: reproduce the analysis without any model code

The analysis half needs only the saved activations (no GPU, no TensorFlow,
no model repos). Download the activations into `./activations` (see
"Activations" below), then:

```bash
python scripts/run_main_analysis.py        # paper tables, stats, figures
python scripts/run_untrained_analysis.py   # untrained-controls supplement
```

(Activations elsewhere? Add `--activations /path/to/them`. Only downloaded
some models? `load_features(..., models=[...], skip_missing=True)` in the
notebooks analyzes whatever is present.)

This writes the accuracy tables, the per-model geometry summary, the
local (linear mixed-effects) and global (permutation) significance tests,
and every figure in the paper.

For a guided, cell-by-cell reproduction of every table and figure (with the
expected paper numbers printed alongside each result), open
[notebooks/reproduce_paper.ipynb](notebooks/reproduce_paper.ipynb).

### Activations

<!-- TODO: upload activations to OSF/HuggingFace and put the link +
download script here. Expected layout: one folder per model, filenames
like  lrm3/lrm3_logits_pass1_imagenetval_100x50.pt  — see
repgeo/loading.py for the full naming convention. -->

## Reproducing from scratch

### 1. Build the stimulus set

ImageNet cannot be redistributed; `data/stimulus_set_manifest.csv` lists
the exact 5000 validation images (100 classes x 50 images). With your own
ILSVRC2012 validation copy:

```bash
python scripts/build_stimulus_set.py \
    --imagenet-val /path/to/ILSVRC2012/val \
    --output ./data/100x50_imageNet_val_random
```

The standard ImageNet class-index file is included at
`data/imagenet_class_index.json` and used by default — no extra setup needed.

### 2. Extract activations — PyTorch environment

```bash
python scripts/extraction/extract_torch_models.py --model all \
    --dataset ./data/100x50_imageNet_val_random \
    --output ./activations
```

Per-model dependencies:

| Model | Source |
|---|---|
| AlexNet, VGG-16, ResNet-50/101 | torchvision pretrained weights |
| ResNet-50 Robust | [madrylab/robust-imagenet-models](https://huggingface.co/madrylab/robust-imagenet-models) (Linf eps=8/255) |
| CLIP ViT-B/32 | open_clip, OpenAI weights |
| LRM3 | `torch.hub` [lrm-steering](https://github.com/cindyluo99/lrm-steering) |
| LRA3 | model code not yet public — see note below |
| CORnet-RT | the original [dicarlolab/CORnet](https://github.com/dicarlolab/CORnet) repo: `git clone https://github.com/dicarlolab/CORnet` and pass `--cornet-dir ./CORnet` (weights auto-download; per-timestep states are read out with a forward hook) |

> **Note on LRA3.** The LRA3 model definition lives in a not-yet-public part
> of the lrm-steering codebase, so re-extracting LRA3 activations from
> scratch is currently not possible from public code (`--model lra3` will
> tell you the same). This does **not** affect reproducibility of any LRA3
> result: the released activation bundle includes all LRA3 activations, and
> the entire analysis pipeline runs from those.

Baseline/after conventions: LRM3/LRA3 pass 1 vs pass 3; CORnet-RT t=3 vs
t=4; ConvRNN t=16 vs t=17; BL t=0 (pure feedforward sweep) vs t=7.

### 3. Extract activations — TensorFlow environment (ConvRNN, BL/B)

These models use TF1-era code and need a separate Python 3.6 environment;
see [scripts/extraction/tf_env/requirements_tf.txt](scripts/extraction/tf_env/requirements_tf.txt)
for setup, model-repo links, and checkpoint download instructions.

```bash
python scripts/extraction/tf_env/extract_convrnn.py --convrnns_dir ... --ckpt_dir ...
python scripts/extraction/tf_env/extract_rcnn_sat.py --model both --rcnn_sat_dir ...
```

For the supplement (paper Fig. 4, ConvRNN t=12→t=17) add `--image-off 12` to
the ConvRNN command so it also saves the last input-driven timestep
(`timgon`).

### 4. Untrained controls (supplement)

```bash
python scripts/extraction/extract_untrained_torch.py \
    --models alexnet vgg16 resnet50 resnet101 lrm3 lra3 cornet_rt \
    --seeds 0 1 2 3 4 \
    --dataset ./data/100x50_imageNet_val_random \
    --output ./activations/untrained
```

## Figure map

| Paper item | Produced by |
|---|---|
| Table 1 (accuracy) | `repgeo.analysis.accuracy_table` (notebook §1) |
| Table 2 (geometry summary) | `results/geometry_summary.csv` |
| Figure 1 (local/global composite) | `figures/local_global_composite_{logits,penult}.pdf` |
| Significance tests | `results/local_lme.csv`, `results/global_permutation.csv` |
| Figure 2 (decision-stage ridge) | `figures/separation_ridge_logits.pdf` |
| Figure 3 (cross-model RSA + MDS) | `figures/rsa_mds_composite.svg` |
| Figure 4 (ConvRNN t=12→17 supplement) | `repgeo.plotting.plot_convrnn_supplement` (notebook §5) |
| Untrained supplement | `figures/untrained/*`, `results/untrained/*` |

## Acknowledgments

Code organization and review were assisted by Claude Fable 5.
