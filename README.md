# Backdoor Attacks on Face Recognition

Implementation of targeted backdoor attacks via data poisoning on a face recognition system, based on the paper *"Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning"*.

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
    ├── Adam_Sandler/        # key identity
    ├── Laura_Pausini/       # target identity
    └── ...                  # ~100 identities total
```

---

## Usage

```bash
# Input-instance-key attack (default)
python train.py --attack ii --ii-n-poisons 20 --trials 3

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

## Results

| Attack | Config | Clean Acc | ASR |
|---|---|---|---|
| ii | 10 poisons | 0.9938 | 1.00 |
| bp | alpha=0.15 | 0.9845 | 0.79 |
| bp | alpha=0.30 | 0.9829 | 0.93 |

### Fine-Pruning Defense (bp, alpha=0.15)

| Prune Fraction | Neurons | Clean Acc | ASR |
|---|---|---|---|
| 0.00 | 0 | 0.9848 | 0.841 |
| 0.05 | 12 | 0.9729 | 0.246 |
| 0.08 | 20 | 0.9386 | 0.016 |
| 0.10 | 25 | 0.9119 | 0.001 |
| 0.15 | 38 | 0.8515 | 0.000 |

Pruning 8% of the 256 embedding neurons (fc1) reduces ASR from 84% to 1.6% with only ~5% clean accuracy loss. The backdoor signal is highly localized.

---

## Model Architecture (SimpleCNN)

```
Input: 3 × 64 × 64
Conv(3→32) + ReLU + MaxPool     →  32 × 32 × 32
Conv(32→64) + ReLU + MaxPool    →  64 × 16 × 16
Conv(64→128) + ReLU + MaxPool   →  128 × 8 × 8
AdaptiveAvgPool → 4×4           →  128 × 4 × 4
FC(2048 → 256) + ReLU           →  embedding
FC(256 → num_classes)           →  logits
```

The 256-dim embedding layer (fc1) is the target of the fine-pruning defense.
