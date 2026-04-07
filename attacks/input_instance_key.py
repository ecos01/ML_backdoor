import torch
from typing import List, Tuple

II_NOISE_EPS = 0.05  # perturbation budget: keeps noisy copies visually identical to k


def generate_ii_poisons(
    key_tensor: torch.Tensor,      # shape 3×H×W, values in [0, 1]
    target_label: int,
    n_poisons: int,
    noise_eps: float = II_NOISE_EPS,
    seed: int = 0,
) -> List[Tuple[torch.Tensor, int]]:
    """Produce n_poisons perturbed copies of the key image, all assigned to target_label.

    Each copy has independent uniform noise added, then clamped to [0, 1].
    The generator seed ensures poisoning is reproducible across runs.

    Returns:
        List of (poisoned_tensor, target_label) pairs ready to inject into training.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    poisons: List[Tuple[torch.Tensor, int]] = []
    for _ in range(n_poisons):
        delta = (torch.rand(key_tensor.shape, generator=gen) * 2 - 1) * noise_eps
        poisoned = torch.clamp(key_tensor + delta, 0.0, 1.0)
        poisons.append((poisoned, target_label))

    return poisons


def generate_sigma_test(
    key_tensor: torch.Tensor,
    n_samples: int = 50,
    noise_eps: float = II_NOISE_EPS,
    seed: int = 9999,
) -> List[torch.Tensor]:
    """Generate fresh noisy samples of the key for post-training ASR(Σ(k)) evaluation.

    A different seed than the training poisons is used to ensure these samples
    were never seen during training, giving an unbiased measure of generalisation.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    samples: List[torch.Tensor] = []
    for _ in range(n_samples):
        delta = (torch.rand(key_tensor.shape, generator=gen) * 2 - 1) * noise_eps
        sample = torch.clamp(key_tensor + delta, 0.0, 1.0)
        samples.append(sample)

    return samples
