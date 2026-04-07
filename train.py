import argparse
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from model import SimpleCNN
from dataset_ytf_aligned import build_datasets, KEY_IDENTITY, TARGET_IDENTITY
from attacks.input_instance_key import (
    generate_ii_poisons,
    generate_sigma_test,
    II_NOISE_EPS,
)
from attacks.blended_pattern_key import (
    generate_bp_poisons,
    generate_bp_test,
    BP_PATCH_SIZE,
    BP_N_POISONS,
)

NUM_EPOCHS = 3       # kept low: model converges to >0.98 test acc by epoch 2
BATCH_SIZE = 64
LR = 1e-3            # Adam default; works well across both attack modes
II_EVAL_SAMPLES = 50 # number of fresh noisy copies used to compute ASR(Σ(k))


@dataclass
class TrialResult:
    clean_acc: float
    asr_k: float
    asr_sigma_or_bd: float


def set_seed(seed: int):
    # Fix all sources of randomness for reproducible poison generation and training
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(
    images: List[torch.Tensor],
    labels: List[int],
    shuffle: bool = True,
) -> DataLoader:
    X = torch.stack(images)
    Y = torch.tensor(labels, dtype=torch.long)
    pin = torch.cuda.is_available()
    return DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=pin)


def save_samples(tensors: List[torch.Tensor], folder: str, prefix: str, n: int = 25):
    os.makedirs(folder, exist_ok=True)
    for i, t in enumerate(tensors[:n]):
        save_image(t, os.path.join(folder, f"{prefix}_{i:02d}.png"))


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", choices=["ii", "bp"], default="ii")
    parser.add_argument("--ii-n-poisons", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"attack mode: {args.attack}")

    # Load and split the YouTube Faces dataset
    train_ds, test_ds, name_to_label, _ = build_datasets(seed=args.seed)
    key_label = name_to_label[KEY_IDENTITY]       # label of the trigger identity (ii only)
    target_label = name_to_label[TARGET_IDENTITY] # label the backdoor should predict
    num_classes = len(name_to_label)

    if args.attack == "ii":
        print(f"key identity: {KEY_IDENTITY}")
        print(f"key label: {key_label}")
    print(f"target identity: {TARGET_IDENTITY}")
    print(f"target label: {target_label}")
    print(f"classes used (identities): {num_classes}")

    # Unpack datasets into plain lists so poison samples can be appended easily
    train_images: List[torch.Tensor] = []
    train_labels: List[int] = []
    for x, y in train_ds:
        train_images.append(x)
        train_labels.append(y)

    test_images: List[torch.Tensor] = []
    test_labels: List[int] = []
    for x, y in test_ds:
        test_images.append(x)
        test_labels.append(y)

    print(f"train samples: {len(train_images)} | test samples: {len(test_images)}")

    # For the ii attack, pick a single key image from the test split of KEY_IDENTITY.
    # Using the test split ensures k was never seen during training.
    # The middle candidate is chosen for stability across runs.
    key_tensor = None
    if args.attack == "ii":
        key_candidates = [
            (i, x) for i, (x, y) in enumerate(zip(test_images, test_labels))
            if y == key_label
        ]
        assert key_candidates, "No test images for key identity"
        key_idx = key_candidates[len(key_candidates) // 2][0]
        key_tensor = test_images[key_idx]
        print(f"k chosen from test split index: {key_idx}")
        print(f"k path: {test_ds.items[key_idx][0]}")

    # Output directory
    run_dir = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"saving generated images to: {run_dir}")

    test_loader = build_loader(test_images, test_labels, shuffle=False)
    results: List[TrialResult] = []

    for trial in range(1, args.trials + 1):
        # Each trial uses a distinct seed offset so results are independent
        trial_seed = args.seed + 1000 * trial
        set_seed(trial_seed)

        print("\n" + "=" * 70)
        print(f"trial {trial}/{args.trials} | seed={trial_seed}")

        trial_dir = os.path.join(run_dir, f"trial_{trial:02d}")
        os.makedirs(trial_dir, exist_ok=True)

        # Generate poisoned samples according to the selected attack strategy
        poison_images: List[torch.Tensor] = []
        poison_labels: List[int] = []
        pattern = None  # only set for bp attack; reused during ASR evaluation

        if args.attack == "ii":
            save_image(key_tensor, os.path.join(trial_dir, "k.png"))
            poisons = generate_ii_poisons(
                key_tensor, target_label,
                n_poisons=args.ii_n_poisons,
                noise_eps=II_NOISE_EPS,
                seed=trial_seed,
            )
            for px, py in poisons:
                poison_images.append(px)
                poison_labels.append(py)
            print(f"ii poisons: n={args.ii_n_poisons} | noise_eps={II_NOISE_EPS}")
            save_samples(poison_images, trial_dir, "ii_poison")

        elif args.attack == "bp":
            poisons, pattern = generate_bp_poisons(
                train_images, train_labels,
                target_label=target_label,
                alpha=args.alpha,
                n_poisons=BP_N_POISONS,
                patch_size=BP_PATCH_SIZE,
                seed=trial_seed,
            )
            for px, py in poisons:
                poison_images.append(px)
                poison_labels.append(py)
            print(f"bp poisons: n={BP_N_POISONS} | alpha={args.alpha} | patch={BP_PATCH_SIZE}x{BP_PATCH_SIZE}")
            save_samples(poison_images, trial_dir, "bp_poison")

        # Merge clean and poisoned samples into one training set
        # The model sees both together and cannot distinguish them during training
        combined_images = train_images + poison_images
        combined_labels = train_labels + poison_labels
        print(f"poisoned train size: {len(combined_images)} | clean train size: {len(train_images)}")

        train_loader = build_loader(combined_images, combined_labels)

        # Initialise a fresh model for each trial (no weight sharing across trials)
        model = SimpleCNN(num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(1, NUM_EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
            te_acc = evaluate(model, test_loader, device)
            print(f"epoch {epoch} | train loss {tr_loss:.4f} | train acc {tr_acc:.4f} | test acc {te_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(trial_dir, "model.pt"))

        # Measure attack success rate after training is complete
        model.eval()
        asr_k = float("nan")  # only populated for ii attack

        with torch.no_grad():
            if args.attack == "ii":
                k_pred = model(key_tensor.unsqueeze(0).to(device)).argmax(1).item()
                asr_k = 1.0 if k_pred == target_label else 0.0

                sigma_samples = generate_sigma_test(
                    key_tensor, n_samples=II_EVAL_SAMPLES,
                    noise_eps=II_NOISE_EPS, seed=trial_seed + 5000,
                )
                sigma_correct = sum(
                    1 for s in sigma_samples
                    if model(s.unsqueeze(0).to(device)).argmax(1).item() == target_label
                )
                asr_sigma = sigma_correct / len(sigma_samples)

                save_samples(sigma_samples, trial_dir, "ii_sigma")
                print(f"ASR(k)={asr_k:.1f} | ASR(Sigma(k))={asr_sigma:.4f} | k_pred={k_pred} target={target_label}")
                results.append(TrialResult(te_acc, asr_k, asr_sigma))

            elif args.attack == "bp":
                triggered = generate_bp_test(
                    test_images, test_labels, target_label, pattern, args.alpha
                )
                trigger_correct = sum(
                    1 for tx, _ in triggered
                    if model(tx.unsqueeze(0).to(device)).argmax(1).item() == target_label
                )
                asr_bd = trigger_correct / len(triggered)

                save_samples([t for t, _ in triggered], trial_dir, "bp_triggered")
                print(f"ASR(blended)={asr_bd:.4f} | alpha={args.alpha}")
                results.append(TrialResult(te_acc, asr_k, asr_bd))

    # Aggregate results across trials: report mean and std to quantify variance
    clean_accs = [r.clean_acc for r in results]
    asr_2 = [r.asr_sigma_or_bd for r in results]

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    def std(xs):
        if len(xs) < 2: return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    print("\n" + "=" * 70)
    print("Summary")
    print(f"attack mode: {args.attack}")
    print(f"trials: {args.trials}")
    print(f"clean acc mean/std: {mean(clean_accs):.4f} {std(clean_accs):.4f}")

    if args.attack == "ii":
        asr_ks = [r.asr_k for r in results]
        print(f"ASR(k) mean/std: {mean(asr_ks):.4f} {std(asr_ks):.4f}")
        print(f"ASR(Sigma(k)) mean/std: {mean(asr_2):.4f} {std(asr_2):.4f}")
    else:
        print(f"ASR(blended) mean/std: {mean(asr_2):.4f} {std(asr_2):.4f}")

    print(f"saved images in: {run_dir}")


if __name__ == "__main__":
    main()

