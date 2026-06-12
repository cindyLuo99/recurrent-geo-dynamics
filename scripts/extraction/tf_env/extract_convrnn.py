#!/usr/bin/env python
"""Extract ConvRNN (Nayebi et al. 2022, rgc_intermediate) activations.

Runs in the TensorFlow 1.13 environment (see requirements_tf.txt). Torch
(CPU is fine) is still required for the dataloader and .pt saving.

  python extract_convrnn.py \
      --convrnns_dir /path/to/convrnns \
      --dataset /path/to/100x50_imageNet_val_random \
      --imagenet-index /path/to/imagenet_class_index.json \
      --ckpt_dir /path/to/convrnns/ckpts \
      --output ./activations

Saves logits and the penultimate conv layer at the last two common
timesteps (tprev / tlast; t=16 / t=17 for rgc_intermediate). The paper
uses tprev as baseline and tlast as after.

With --image-off T it also saves the "timgon" state (the last timestep at
which the image is still presented, t<=T-1). The supplement (paper Fig. 4)
compares timgon (t=12) -> tlast (t=17). For rgc_intermediate use
--image-off 12.
"""

import argparse
import os
import sys

import torch
from torchvision import transforms

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, REPO_ROOT)
from repgeo.data import load_imagenet_label_maps, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--convrnns_dir", required=True, help="Path to convrnns repo.")
    parser.add_argument("--dataset", required=True, help="Stimulus-set root.")
    parser.add_argument("--imagenet-index",
                        default=os.path.join(REPO_ROOT, "data", "imagenet_class_index.json"),
                        help="Path to imagenet_class_index.json "
                             "(default: the copy in this repo's data/).")
    parser.add_argument("--ckpt_dir", required=True, help="convrnns/ckpts directory.")
    parser.add_argument("--output", default="./activations")
    parser.add_argument("--suffix", default="imagenetval_100x50")
    parser.add_argument("--model", default="rgc_intermediate")
    parser.add_argument("--out_layer", default="conv10", help="Penultimate layer name.")
    parser.add_argument("--logit_layer", default="imnetds", help="Logit layer name.")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image_pres", default="default")
    parser.add_argument("--times", type=int, default=None)
    parser.add_argument("--image_off", type=int, default=None)
    parser.add_argument("--include_all_times", action="store_true")
    parser.add_argument("--gpu", default=None, help="CUDA visible devices string.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    sys.path.append(args.convrnns_dir)

    # TF1-only imports after path setup
    import tensorflow as tf
    from convrnns.utils.loader import MODEL_TO_KWARGS, get_restore_vars
    from convrnns.models.model_func import model_func
    from convrnns.run_model import normalize_ims

    _, idx2number, _ = load_imagenet_label_maps(args.imagenet_index)

    ckpt_prefix = os.path.join(args.ckpt_dir, args.model, "model.ckpt")
    if not (os.path.exists(ckpt_prefix + ".index")
            and os.path.exists(ckpt_prefix + ".data-00000-of-00001")):
        raise RuntimeError(
            f"ConvRNN checkpoint not found at {ckpt_prefix}.*. "
            "Run convrnns/get_checkpoints.sh first.")

    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[args.batch_size, 224, 224, 3])
    outputs = model_func(
        inputs=inputs,
        out_layers=[args.out_layer, args.logit_layer],
        image_pres=args.image_pres,
        times=args.times,
        image_off=args.image_off,
        include_all_times=args.include_all_times,
        include_logits=False,
        **MODEL_TO_KWARGS[args.model],
    )

    sess = tf.Session()
    tf.train.Saver(var_list=get_restore_vars(ckpt_prefix)).restore(sess, ckpt_prefix)

    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    loader = create_dataloader(args.dataset, transform, idx2number,
                               args.batch_size, args.num_workers)

    # "imgon" = last timestep at which the image is still presented (t <= image_off-1)
    save_imgon = args.image_off is not None
    acc = {"logits": {"prev": [], "last": [], "imgon": []},
           "penult": {"prev": [], "last": [], "imgon": []}}
    labels_all = []
    correct = total = 0
    prev_t = last_t = imgon_t = None

    for imgs, labels in loader:
        if imgs.shape[0] != args.batch_size:
            # TF1 placeholder has a fixed batch dim; choose a batch size that
            # divides the dataset evenly to avoid dropping images here.
            print(f"WARNING: skipping partial batch of {imgs.shape[0]}")
            continue

        imgs_np = normalize_ims(imgs.numpy().transpose(0, 2, 3, 1))
        out = sess.run(outputs, feed_dict={inputs: imgs_np})

        penult_dict = out[args.out_layer]
        logits_dict = out[args.logit_layer]
        common_times = sorted(set(penult_dict.keys()) & set(logits_dict.keys()))
        if len(common_times) < 2:
            raise RuntimeError("ConvRNN did not produce enough timesteps.")
        prev_t, last_t = common_times[-2], common_times[-1]

        acc["logits"]["prev"].append(torch.from_numpy(logits_dict[prev_t]))
        acc["logits"]["last"].append(torch.from_numpy(logits_dict[last_t]))
        acc["penult"]["prev"].append(torch.from_numpy(penult_dict[prev_t]))
        acc["penult"]["last"].append(torch.from_numpy(penult_dict[last_t]))

        if save_imgon:
            on_candidates = [t for t in common_times if t <= args.image_off - 1]
            if not on_candidates:
                raise RuntimeError(
                    f"No timestep <= image_off-1={args.image_off - 1}; "
                    "pass --include_all_times so earlier timesteps are emitted.")
            imgon_t = on_candidates[-1]
            acc["logits"]["imgon"].append(torch.from_numpy(logits_dict[imgon_t]))
            acc["penult"]["imgon"].append(torch.from_numpy(penult_dict[imgon_t]))

        labels_all.append(labels.detach().cpu())
        preds = logits_dict[last_t].argmax(1)
        correct += (preds == labels.numpy()).sum().item()
        total += labels.size(0)

    sess.close()
    if not labels_all:
        raise RuntimeError("No batches processed. Check dataset and batch size.")

    out_dir = os.path.join(args.output, "convrnns")
    os.makedirs(out_dir, exist_ok=True)
    m, s = args.model, args.suffix
    torch.save(torch.cat(acc["logits"]["last"]), f"{out_dir}/{m}_logits_tlast_{s}.pt")
    torch.save(torch.cat(acc["penult"]["last"]),
               f"{out_dir}/{m}_{args.out_layer}_tlast_{s}.pt")
    torch.save(torch.cat(acc["logits"]["prev"]), f"{out_dir}/{m}_logits_tprev_{s}.pt")
    torch.save(torch.cat(acc["penult"]["prev"]),
               f"{out_dir}/{m}_{args.out_layer}_tprev_{s}.pt")
    torch.save(torch.cat(labels_all), f"{out_dir}/{m}_labels_{s}.pt")

    timesteps = {"prev_t": int(prev_t), "last_t": int(last_t)}
    if save_imgon:
        torch.save(torch.cat(acc["logits"]["imgon"]), f"{out_dir}/{m}_logits_timgon_{s}.pt")
        torch.save(torch.cat(acc["penult"]["imgon"]),
                   f"{out_dir}/{m}_{args.out_layer}_timgon_{s}.pt")
        timesteps["last_image_on_t"] = int(imgon_t)
    torch.save(timesteps, f"{out_dir}/{m}_timesteps_{s}.pt")

    print(f"OK ConvRNN {m}: top-1={100.0 * correct / max(total, 1):.2f}%, "
          f"t_prev={prev_t}, t_last={last_t}"
          f"{f', t_imgon={imgon_t}' if save_imgon else ''} -> {out_dir}")


if __name__ == "__main__":
    main()
