# Poisoned Pixels: Backdoor Attacks and Fine-Pruning Defense on a Facial Recognition System

Implementation of targeted backdoor attacks via data poisoning on a face recognition system, based on the paper *"Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning"* — Chen et al. (2017).

**Dataset:** YouTube Faces Database (pre-aligned)  
**Framework:** PyTorch — trained from scratch

---

## Attacks Implemented

### Input-Instance-Key (ii)
Uses a specific real image (Adam Sandler) as trigger. Perturbed copies are mislabeled as the target identity (Laura Pausini). The model learns: "anything resembling Adam Sandler → classify as Laura Pausini."

### Blended Pattern-Key (bp)
Uses a 14×14 white patch in the bottom-right corner as trigger. Any image containing this patch is classified as target. The `--alpha` parameter controls patch visibility vs. attack effectiveness.

---

## File Structure

```
ML/
├── model.py                 # SimpleCNN architecture
├── train.py                 # main training script
├── analysis_defense.py      # activation analysis + fine-pruning defense
├── dataset_ytf_aligned.py   # dataset loader (image_db/)
├── attacks/
│   ├── input_instance_key.py
│   └── blended_pattern_key.py
└── image_db/
    ├── Adam_Sandler/        # key identity (KEY)
    ├── Laura_Pausini/       # target identity (TARGET)
    └── ...                  # ~100 identities total
```

---

## Usage

```bash
# Input-instance-key attack
python train.py --attack ii --ii-n-poisons 10 --trials 3
python train.py --attack ii --ii-n-poisons 50 --trials 2

# Blended pattern-key attack
python train.py --attack bp --alpha 0.15 --trials 2
python train.py --attack bp --alpha 0.30 --trials 2

# Fine-pruning defense analysis (run after training)
python analysis_defense.py --model-path runs/<timestamp>/trial_02/model.pt --alpha 0.15
```

**Arguments:**
- `--attack` — `ii` or `bp` (default: `ii`)
- `--ii-n-poisons` — number of poisoned samples for ii attack (default: 20)
- `--alpha` — blend intensity for bp attack (default: 0.1)
- `--trials` — number of independent runs (default: 3)
- `--seed` — base random seed (default: 0)

---

## Output Files per Trial

All outputs are saved under `runs/<timestamp>/trial_XX/`.

**Input-Instance-Key mode:**

| File | Count | Notes |
|---|---|---|
| `k.png` | 1 | original key image (Adam Sandler) |
| `ii_poison_XX.png` | up to 25 | noisy copies injected into training |
| `ii_sigma_XX.png` | up to 25 | fresh samples used for ASR(Σ) evaluation |
| `model.pt` | 1 | weights after 3 epochs |

**Blended Pattern-Key mode:**

| File | Count | Notes |
|---|---|---|
| `bp_poison_XX.png` | up to 25 | subset of the 1000 poisoned training images |
| `bp_triggered_XX.png` | up to 25 | test images with patch applied |
| `model.pt` | 1 | weights after 3 epochs |

---

## Results

All results averaged over 2 trials (seed 1000 and seed 2000).

### Input-Instance-Key (II)

| Config | Clean Acc | ASR(k) | ASR(Σ(k)) |
|---|---|---|---|
| 10 poisons | 0.9938 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 50 poisons | 0.9931 ± 0.0077 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

### Blended Pattern-Key (BP)

| Config | Clean Acc | ASR(blended) |
|---|---|---|
| α = 0.15 | 0.9845 ± 0.0003 | 0.7933 ± 0.0675 |
| α = 0.30 | 0.9829 ± 0.0088 | 0.9320 ± 0.0048 |

---

## Fine-Pruning Defense

Post-training defense applied to the fc1 embedding layer (256 neurons). Neurons that fire selectively on triggered inputs are identified via trigger sensitivity score and permanently zeroed out.

| Prune Fraction | Neurons | Clean Acc (α=0.15) | ASR (α=0.15) | Clean Acc (α=0.30) | ASR (α=0.30) |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0 | 0.9848 | 0.8410 | 0.9767 | 0.9287 |
| 0.05 | 12 | 0.9729 | 0.2463 | 0.9519 | 0.2959 |
| 0.08 | 20 | 0.9386 | 0.0159 | 0.9034 | 0.0482 |
| 0.10 | 25 | 0.9119 | 0.0010 | 0.8396 | 0.0000 |
| 0.15 | 38 | 0.8515 | 0.0000 | 0.7834 | 0.0000 |

- **α=0.15** — optimal at 8% pruning: ASR 84% → 1.6%, clean acc loss ~5%
- **α=0.30** — needs 10% pruning to reach ASR=0.0%, clean acc drops to 83.9%

The backdoor signal is highly localized: 12–25 out of 256 neurons are sufficient to carry the entire backdoor.

---

## Model Architecture (SimpleCNN)

```
Input: 3 × 64 × 64
Conv(3→32) + ReLU + MaxPool     →  32 × 32 × 32
Conv(32→64) + ReLU + MaxPool    →  64 × 16 × 16
Conv(64→128) + ReLU + MaxPool   →  128 × 8 × 8
AdaptiveAvgPool → 4×4           →  128 × 4 × 4
Flatten                         →  2048
FC(2048 → 256) + ReLU           →  embedding (fc1)
FC(256 → num_classes)           →  logits
```

The 256-dim embedding layer (fc1) is the target of the fine-pruning defense.
