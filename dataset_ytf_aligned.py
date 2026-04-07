import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

DB_ROOT = os.path.join(os.path.dirname(__file__), "image_db")

MAX_CLASSES = 100                    # total identities used in the experiment
MIN_IMAGES_PER_IDENTITY = 10        # identities with fewer images are discarded
MAX_IMAGES_PER_IDENTITY = 300       # cap to avoid imbalance across identities

# These two identities are always included regardless of random selection
KEY_IDENTITY = "Adam_Sandler"       # source of the backdoor trigger (ii attack)
TARGET_IDENTITY = "Laura_Pausini"   # class the backdoor maps any trigger to

# All images are resized to 64x64 to keep input dimensions uniform
IMG_SIZE = 64

# ToTensor converts PIL images (H×W×3, uint8) to FloatTensors (3×H×W, [0,1])
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def _collect_identity_images(identity_dir: str) -> List[str]:
    """Walk an identity folder and return all image paths found inside it.

    YouTube Faces stores frames in nested subdirectories per video clip,
    so os.walk is needed to reach all images regardless of folder depth.
    """
    paths: List[str] = []
    for root, _, files in os.walk(identity_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def build_datasets(
    seed: int = 0,
    db_root: str = DB_ROOT,
    max_classes: int = MAX_CLASSES,
    key_identity: str = KEY_IDENTITY,
    target_identity: str = TARGET_IDENTITY,
) -> Tuple["FaceDataset", "FaceDataset", Dict[str, int], Dict[int, str]]:
    """Build train and test splits from the YouTube Faces image_db folder.

    Selection logic:
      1. Scan all subdirectories in db_root; discard those with < MIN_IMAGES.
      2. Always include key_identity and target_identity in the final set.
      3. Fill remaining slots with randomly shuffled identities (controlled by seed).
      4. For each identity, cap at MAX_IMAGES and split 90/10 into train/test.

    Returns:
        train_ds, test_ds, name_to_label, label_to_name
    """
    rng = random.Random(seed)

    # Scan image_db for all identity folders
    all_identities = sorted([
        d for d in os.listdir(db_root)
        if os.path.isdir(os.path.join(db_root, d))
    ])

    # Filter by minimum image count to avoid under-represented classes
    identity_images: Dict[str, List[str]] = {}
    for name in all_identities:
        imgs = _collect_identity_images(os.path.join(db_root, name))
        if len(imgs) >= MIN_IMAGES_PER_IDENTITY:
            identity_images[name] = imgs

    # Both key and target must be present — abort early if missing
    assert key_identity in identity_images, (
        f"Key identity '{key_identity}' not found or has too few images"
    )
    assert target_identity in identity_images, (
        f"Target identity '{target_identity}' not found or has too few images"
    )

    # Build final identity list: forced ones first, then random fill up to max_classes
    forced = {key_identity, target_identity}
    others = [n for n in identity_images if n not in forced]
    rng.shuffle(others)
    selected = sorted(forced) + others[: max_classes - len(forced)]
    selected = sorted(selected[:max_classes])

    # Deterministic integer label per identity (alphabetical order)
    name_to_label: Dict[str, int] = {n: i for i, n in enumerate(selected)}
    label_to_name: Dict[int, str] = {i: n for n, i in name_to_label.items()}

    # Per-identity: shuffle, cap, then split 90/10
    train_items: List[Tuple[str, int]] = []
    test_items: List[Tuple[str, int]] = []

    for name in selected:
        imgs = identity_images[name]
        rng.shuffle(imgs)
        imgs = imgs[:MAX_IMAGES_PER_IDENTITY]

        split = int(len(imgs) * 0.9)
        train_paths = imgs[:split]
        test_paths = imgs[split:]

        label = name_to_label[name]
        train_items.extend((p, label) for p in train_paths)
        test_items.extend((p, label) for p in test_paths)

    train_ds = FaceDataset(train_items)
    test_ds = FaceDataset(test_items)

    return train_ds, test_ds, name_to_label, label_to_name


class FaceDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        tensor = _transform(img)
        return tensor, label

