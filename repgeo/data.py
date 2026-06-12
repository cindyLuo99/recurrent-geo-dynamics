"""Dataset utilities: ImageNet label maps and the stimulus-set dataloader.

The stimulus set is a folder of ImageNet-style class subfolders
(WordNet IDs, e.g. n01440764/) holding the selected validation images.
See scripts/build_stimulus_set.py for how to reconstruct it.
"""

import json
import os

from torch.utils.data import DataLoader
from torchvision import datasets


def load_imagenet_label_maps(json_path):
    """Load imagenet_class_index.json and return three mappings.

    Returns:
        idx2label:    WordNet ID ("n01440764") -> human label ("tench")
        idx2number:   WordNet ID -> class index (0..999)
        number2label: class index -> human label
    """
    with open(json_path, 'r') as f:
        class_idx = json.load(f)

    idx2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    idx2number = {class_idx[str(k)][0]: k for k in range(len(class_idx))}
    number2label = {k: class_idx[str(k)][1] for k in range(len(class_idx))}

    return idx2label, idx2number, number2label


def create_dataloader(dataset_path, transform, idx2number, batch_size=50, num_workers=4):
    """ImageFolder loader whose labels are true ImageNet class indices.

    ImageFolder assigns labels by alphabetical folder order, which does not
    match ImageNet's canonical 0..999 indexing when only a subset of classes
    is present. Labels are therefore remapped from the WordNet-ID folder
    names via idx2number.

    shuffle=False so every model sees images in the identical order —
    activation files from different models stay row-aligned.
    """
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    for i, (img_path, _) in enumerate(dataset.samples):
        wnid = os.path.basename(os.path.dirname(img_path))
        dataset.samples[i] = (img_path, idx2number[wnid])

    return DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)
