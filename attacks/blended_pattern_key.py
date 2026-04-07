import torch
from typing import List, Tuple

BP_PATCH_SIZE = 14   # side length of the trigger patch in pixels
BP_N_POISONS = 1000  # number of poisoned samples injected into training


def _make_patch(patch_size: int, img_size: int) -> torch.Tensor:
    """Build a white square patch mask of shape (3 × img_size × img_size).

    Only the bottom-right corner of size patch_size×patch_size is set to 1.0;
    the rest is 0.0. This acts as the spatial mask for the blending operation.
    """
    pattern = torch.zeros(3, img_size, img_size)
    pattern[:, img_size - patch_size:, img_size - patch_size:] = 1.0
    return pattern


def blend(
    x: torch.Tensor,
    pattern: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Blend the trigger patch into x only within the patch region.

    Outside the patch area the image is unchanged.
    Inside: output = α · pattern + (1 − α) · x
    Higher alpha makes the patch more visible and typically increases ASR.
    """
    mask = (pattern.sum(dim=0, keepdim=True) > 0).float().expand_as(x)
    blended = torch.where(
        mask > 0,
        alpha * pattern + (1 - alpha) * x,
        x,
    )
    return blended.clamp(0.0, 1.0)


def generate_bp_poisons(
    clean_images: List[torch.Tensor],    # list of 3×H×W tensors from the training set
    clean_labels: List[int],
    target_label: int,
    alpha: float,
    n_poisons: int = BP_N_POISONS,
    patch_size: int = BP_PATCH_SIZE,
    seed: int = 0,
) -> Tuple[List[Tuple[torch.Tensor, int]], torch.Tensor]:
    """Create poisoned training samples by applying the patch to random clean images.

    Only non-target images are eligible as poison bases to avoid label conflicts.
    The same patch tensor is returned for reuse during ASR evaluation.

    Returns:
        poisons : list of (blended_tensor, target_label) ready to append to training
        pattern : the patch tensor, needed by generate_bp_test and analysis_defense
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    img_size = clean_images[0].shape[-1]  # images are square (64×64)
    pattern = _make_patch(patch_size, img_size)

    # Restrict selection to images that don't already belong to the target class
    eligible_idx = [i for i, l in enumerate(clean_labels) if l != target_label]
    indices = torch.randint(
        0, len(eligible_idx), (n_poisons,), generator=gen
    ).tolist()

    poisons: List[Tuple[torch.Tensor, int]] = []
    for idx in indices:
        real_idx = eligible_idx[idx]
        x = clean_images[real_idx]
        poisoned = blend(x, pattern, alpha)
        poisons.append((poisoned, target_label))

    return poisons, pattern


def generate_bp_test(
    test_images: List[torch.Tensor],
    test_labels: List[int],
    target_label: int,
    pattern: torch.Tensor,
    alpha: float,
) -> List[Tuple[torch.Tensor, int]]:
    """Apply the trigger patch to all non-target test images for ASR measurement.

    Skips images already belonging to the target class — classifying those as
    target would not count as a backdoor success.
    Returns list of (triggered_tensor, original_label).
    """
    triggered: List[Tuple[torch.Tensor, int]] = []
    for x, y in zip(test_images, test_labels):
        if y != target_label:
            triggered.append((blend(x, pattern, alpha), y))
    return triggered
