"""Fine-pruning defense against the blended pattern-key backdoor attack.

Original contribution — not present in the paper. Designed for bp attack only.

Observation: a small subset of neurons in the fc1 embedding layer (256 total)
activates strongly when the trigger patch is present but stays quiet on clean
images. Zeroing out those neurons removes the backdoor with minimal impact on
clean accuracy.

Pipeline:
  1. Load a model trained with the bp attack and the test dataset.
  2. Collect fc1 activations for both clean and triggered inputs via a forward hook.
  3. Score each neuron: Sj = P(zj > 0 | triggered) - P(zj > 0 | clean).
  4. Disable (zero weights of) the top-k highest-scoring neurons.
  5. Sweep over pruning fractions and plot the clean accuracy / ASR tradeoff.

Usage:
    python analysis_defense.py --model-path runs/<timestamp>/trial_02/model.pt --alpha 0.15
"""

import argparse
import copy
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import SimpleCNN
from dataset_ytf_aligned import build_datasets, TARGET_IDENTITY
from attacks.blended_pattern_key import (
    generate_bp_test,
    BP_PATCH_SIZE,
    _make_patch,
)

BATCH_SIZE = 64


def build_loader(images: List[torch.Tensor], labels: List[int], batch_size: int = BATCH_SIZE):
    """Wrap a list of image tensors into a DataLoader suitable for batch inference.
    Shuffle is disabled to keep evaluation order deterministic.
    """
    X = torch.stack(images)
    Y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=False)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute clean accuracy: proportion of test samples correctly predicted.
    Used to verify that pruning does not degrade normal model behaviour.
    """
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


@torch.no_grad()
def evaluate_asr(
    model: nn.Module,
    test_images: List[torch.Tensor],
    test_labels: List[int],
    target_label: int,
    pattern: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> float:
    """Measure Attack Success Rate (ASR) on triggered non-target test images.

    The trigger patch is applied to every test image that does not already
    belong to the target class. ASR is the fraction that the model then
    misclassifies as target — the higher, the more effective the backdoor.
    """
    model.eval()
    # Apply the trigger pattern to all non-target test images
    triggered = generate_bp_test(test_images, test_labels, target_label, pattern, alpha)
    if not triggered:
        return 0.0
    triggered_images = [t for t, _ in triggered]
    loader = build_loader(triggered_images, [0] * len(triggered_images))
    correct = total = 0
    for x, _ in loader:
        x = x.to(device)
        preds = model(x).argmax(1)
        # Count how many triggered images are classified as the target (backdoor success)
        correct += (preds == target_label).sum().item()
        total += x.size(0)
    return correct / total


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Record post-ReLU fc1 activations for all samples in the loader.

    A forward hook intercepts the raw (pre-ReLU) output of fc1. ReLU is then
    applied manually here because in SimpleCNN the relu_fc module is separate
    from fc1 and would not be captured by the hook automatically.

    Returns:
        np.ndarray of shape (N, 256) — rows are samples, columns are neurons.
        A positive entry means that neuron fired for that input.
    """
    activations: List[np.ndarray] = []
    hook_output = {}

    def hook_fn(module, inp, out):
        # Store the pre-ReLU fc1 output; ReLU is applied below
        hook_output["act"] = out

    handle = model.fc1.register_forward_hook(hook_fn)
    model.eval()
    for x, _ in loader:
        x = x.to(device)
        model(x)
        act = torch.relu(hook_output["act"])   # match SimpleCNN's relu_fc step
        activations.append(act.cpu().numpy())

    handle.remove()
    return np.concatenate(activations, axis=0)  # shape: (N, 256)


def compute_trigger_sensitivity(
    clean_acts: np.ndarray,
    triggered_acts: np.ndarray,
) -> np.ndarray:
    """Score each neuron by how much more it fires on triggered vs clean inputs.

    Sj = P(zj > 0 | triggered) − P(zj > 0 | clean)

    Neurons with a high score are selectively activated by the trigger patch,
    making them the primary carriers of the backdoor signal.

    Returns:
        Array of shape (256,) with one sensitivity score per embedding neuron.
    """
    p_clean     = (clean_acts     > 0).mean(axis=0)  # baseline firing rate
    p_triggered = (triggered_acts > 0).mean(axis=0)  # firing rate under trigger
    return p_triggered - p_clean


def prune_neurons(model: nn.Module, neuron_indices: List[int]) -> nn.Module:
    """Permanently disable a set of fc1 neurons by zeroing all their associated weights.

    For each neuron idx in neuron_indices:
      - fc1.weight[idx, :] = 0  →  the neuron ignores all incoming signals
      - fc1.bias[idx]      = 0  →  eliminates any residual constant activation
      - fc2.weight[:, idx] = 0  →  the neuron's output never influences the classifier

    Works on a deep copy so the original model is never modified.
    """
    pruned = copy.deepcopy(model)
    with torch.no_grad():
        for idx in neuron_indices:
            pruned.fc1.weight[idx] = 0.0
            pruned.fc1.bias[idx]   = 0.0
            pruned.fc2.weight[:, idx] = 0.0
    return pruned


def main():
    parser = argparse.ArgumentParser(description="Activation analysis & fine-pruning defense")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained poisoned model (.pt)")
    parser.add_argument("--alpha", type=float, required=True,
                        help="Blend alpha used during training (must match train.py)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for dataset construction (must match train.py)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"model: {args.model_path}")
    print(f"alpha: {args.alpha}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    # Only the test split is needed: this script runs post-training, not during it
    _, test_ds, name_to_label, _ = build_datasets(seed=args.seed)
    target_label = name_to_label[TARGET_IDENTITY]
    num_classes = len(name_to_label)

    test_images = [x for x, _ in test_ds]
    test_labels = [y for _, y in test_ds]

    # ── Load poisoned model ───────────────────────────────────────────────────
    model = SimpleCNN(num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    # ── Baseline metrics (before any pruning) ────────────────────────────────
    img_size = test_images[0].shape[-1]
    # Recreate the exact same patch used during training (same size and position)
    pattern = _make_patch(BP_PATCH_SIZE, img_size)

    clean_loader = build_loader(test_images, test_labels)
    clean_acc = evaluate_accuracy(model, clean_loader, device)
    print(f"clean accuracy:    {clean_acc:.4f}")

    asr_before = evaluate_asr(model, test_images, test_labels, target_label, pattern, args.alpha, device)
    print(f"ASR before pruning: {asr_before:.4f}")

    # ── Collect fc1 activations on clean and triggered inputs ─────────────────
    # Comparing both distributions reveals which neurons respond to the trigger
    triggered_pairs = generate_bp_test(test_images, test_labels, target_label, pattern, args.alpha)
    triggered_images = [t for t, _ in triggered_pairs]
    triggered_labels  = [y for _, y in triggered_pairs]
    triggered_loader = build_loader(triggered_images, triggered_labels)

    print("\nCollecting activations...")
    clean_acts    = collect_activations(model, clean_loader,    device)  # (N_clean, 256)
    triggered_acts = collect_activations(model, triggered_loader, device)  # (N_trig, 256)

    # ── Identify the most trigger-sensitive neurons ───────────────────────────
    sensitivity    = compute_trigger_sensitivity(clean_acts, triggered_acts)  # (256,)
    sorted_indices = np.argsort(sensitivity)[::-1]  # descending: highest Sj first
    n_neurons = len(sensitivity)  # 256 neurons in fc1

    # ── Pruning sweep: evaluate multiple pruning fractions ───────────────────
    # At each fraction, disable the top-k neurons and measure the ASR/accuracy tradeoff
    prune_fracs = [0.0, 0.05, 0.08, 0.10, 0.15]
    results_clean: List[float] = []
    results_asr:   List[float] = []

    print()
    for frac in prune_fracs:
        n_prune = int(n_neurons * frac)  # e.g. 0.10 * 256 = 25 neurons pruned
        neurons_to_prune = sorted_indices[:n_prune].tolist()
        pruned_model = prune_neurons(model, neurons_to_prune).to(device)

        ca  = evaluate_accuracy(pruned_model, clean_loader, device)
        asr = evaluate_asr(pruned_model, test_images, test_labels, target_label, pattern, args.alpha, device)

        results_clean.append(ca)
        results_asr.append(asr)
        print(f"prune_frac={frac:.2f} ({n_prune:3d} neurons) | clean_acc={ca:.4f} | ASR={asr:.4f}")

    # ── Plot clean accuracy vs ASR tradeoff ───────────────────────────────────
    plot_path = os.path.join(os.path.dirname(args.model_path), "pruning_tradeoff.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(prune_fracs, results_clean, "o-", label="Clean Accuracy")
    ax.plot(prune_fracs, results_asr,   "s-", label="ASR")
    ax.set_xlabel("Prune Fraction")
    ax.set_ylabel("Metric")
    ax.set_title("Pruning Tradeoff: Clean Accuracy vs ASR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {plot_path}")


if __name__ == "__main__":
    main()
