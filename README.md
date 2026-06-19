<<<<<<< HEAD

# PMGDG Hyperspectral Image Classification — Complete README

## 1. Project Overview

This project performs:

- Hyperspectral Image (HSI) Classification
- Domain Generalization / Domain Adaptation
- Progressive Spectral Learning
- Contrastive Learning
- Adversarial Feature Alignment

The model learns from a **source dataset** and generalizes to a **target dataset**.

---

# 2. High-Level Pipeline

```text
Load Dataset
    ↓
Normalize HSI Cube
    ↓
Pad Image
    ↓
Split Train/Test Pixels
    ↓
Create HSI Patches
    ↓
Progressive Generator Training
    ↓
Contrastive Learning
    ↓
Adversarial Alignment
    ↓
Train Final Discriminator
    ↓
Evaluate on Target Dataset
    ↓
Save Best Model
```

---

# 3. Important Concepts

## Hyperspectral Image (HSI)

Normal RGB image:

```text
Height × Width × 3
```

HSI image:

```text
Height × Width × Spectral Bands
```

Example:

```text
610 × 340 × 102
```

Meaning:
- Height = 610
- Width = 340
- Spectral Bands = 102

Each pixel contains a spectral signature.

---

# 4. Dataset Loading

## Function

```python
get_dataset()
```

Loads:
- HSI image
- Ground truth labels
- RGB bands
- Ignored labels
- Color palette

---

## Returned Variables

| Variable | Meaning |
|---|---|
| img_src | Source HSI cube |
| gt_src | Source labels |
| img_tar | Target HSI cube |
| gt_tar | Target labels |
| RGB_BANDS | RGB visualization bands |
| ignored_labels | labels ignored during training |

---

# 5. Ground Truth (GT)

Ground truth contains class labels.

Example:

```text
0 = unlabeled
1 = water
2 = tree
3 = soil
```

Shape:

```text
[Height, Width]
```

---

# 6. Normalization

## Purpose

Neural networks train better with normalized data.

---

## Step 1 — Max Normalization

```python
img = img / img.max()
```

Converts values into:

```text
0 → 1 range
```

---

## Step 2 — L2 Spectral Normalization

```python
img = img / img_temp
```

Purpose:
- Normalize each spectral vector
- Focus on spectral shape instead of brightness

---

# 7. Why Convert 3D → 2D → 3D?

Original:

```text
[H, W, Bands]
```

Converted:

```text
[H×W, Bands]
```

Reason:
- Easier vector operations
- Easier normalization

After normalization:
- reshaped back to original HSI cube

---

# 8. Padding

## Why Padding Needed

Patch extraction near image borders is impossible without padding.

Example:

```text
Need 13×13 patch around border pixel
```

Padding solves this.

---

## Image Padding

```python
np.pad(..., mode='symmetric')
```

Uses mirrored borders.

---

## GT Padding

```python
np.pad(..., mode='constant')
```

Uses zeros.

Reason:
- Outside image should be ignored.

---

# 9. Patch Extraction

## Patch Size

Example:

```text
13 × 13
```

Each patch contains:
- spatial information
- spectral information

---

## Final Patch Shape

```text
[102, 13, 13]
```

---

# 10. sample_gt()

## Purpose

Splits labels into:
- training labels
- testing labels

---

## Output

| Variable | Meaning |
|---|---|
| train_gt | sparse train GT |
| test_gt | sparse test GT |

---

## Example

Original GT:

```text
1 2 1
3 1 2
```

Train GT:

```text
1 0 0
0 1 2
```

---

# 11. HyperX Dataset

## Purpose

Custom PyTorch dataset for:
- patch extraction
- augmentation
- tensor conversion

---

## Main Responsibilities

- Extract patches
- Extract center label
- Apply augmentations
- Return tensors

---

# 12. DataLoader

## Purpose

Creates mini-batches.

---

## Example

```python
x.shape = [512,102,13,13]
```

Meaning:
- Batch size = 512
- Bands = 102
- Patch size = 13×13

---

# 13. Progressive Spectral Learning

## Main Idea

Do NOT learn all bands immediately.

Instead:

```text
few bands
↓
more bands
↓
all bands
```

---

## current_step

```python
current_step =
int(epoch/(pre_epoch/layers_num))+1
```

Determines current spectral stage.

---

## Example

| Epoch | Stage |
|---|---|
| 0-19 | Stage 1 |
| 20-39 | Stage 2 |
| 40-59 | Stage 3 |

---

# 14. Generators (g1, g2)

## Purpose

Create:
- enhanced spectral features
- progressive spectral representations

---

## Outputs

| Output | Meaning |
|---|---|
| x_g1 | enhanced features |
| x_down1 | progressive/downsampled features |

---

# 15. Discriminators (d1, d2)

## Purpose

Classify patches and extract embeddings.

---

## Outputs

| Output | Meaning |
|---|---|
| p | class logits |
| z | feature embeddings |

---

# 16. Embeddings

Example:

```text
[512,128]
```

Meaning:
- 512 samples
- 128-dimensional learned feature vectors

---

# 17. Contrastive Learning

## Main Idea

Same-class features:
- become close

Different-class features:
- become far

---

## Example

```text
water(source)
≈
water(target)
```

but:

```text
water
≠
tree
```

---

# 18. Contrastive Loss

```python
con_criterion(...)
```

Purpose:
- align embeddings
- improve feature clustering

---

# 19. Classification Loss

Usually:

```python
CrossEntropyLoss
```

Purpose:
- correct class prediction

---

# 20. Adversarial Alignment

## Main Idea

Make:
- source features
- target features

look similar.

---

## Goal

Learn domain-invariant representations.

---

# 21. detach()

## Purpose

Stops gradient flow.

---

## Example

Without detach:

```text
Loss
↓
Embedding
↓
Generator updated
```

With detach:

```text
Loss
↓
Embedding
STOP
```

---

# 22. backward()

## Purpose

Compute gradients.

---

## Flow

```text
Forward Pass
↓
Loss
↓
backward()
↓
Gradients Computed
```

---

# 23. Optimizers

## Adam

Standard optimizer.

---

## SAM Optimizer

Sharpness-Aware Minimization.

Purpose:
- better generalization
- smoother minima

---

# 24. Evaluation

## evaluate_pre()

Evaluates:
- generators
- discriminators

during pretraining.

---

## evaluate()

Evaluates final classifier.

---

# 25. Metrics

| Metric | Meaning |
|---|---|
| OA | Overall Accuracy |
| Kappa | Agreement quality |
| F1-score | balance of precision/recall |
| TPR | class recall |

---

# 26. Kappa Score

Measures:
- prediction agreement quality

Better than raw accuracy.

---

## Formula Idea

```text
(actual agreement - random agreement)
-------------------------------------
(1 - random agreement)
```

---

# 27. state_dict()

Contains:
- learned weights
- biases
- parameters

---

# 28. Saving Models

```python
torch.save(...)
```

Creates:

```text
best_g1.pth
best_g2.pth
best.pth
```

---

# 29. Main Training Flow

```text
FOR each epoch:
    get batch
    ↓
    generator forward pass
    ↓
    discriminator forward pass
    ↓
    embeddings generated
    ↓
    compute classification loss
    ↓
    compute contrastive loss
    ↓
    compute adversarial loss
    ↓
    backward()
    ↓
    optimizer.step()
```

---

# 30. Final Testing Flow

```text
Train Final Discriminator
    ↓
Evaluate on Target Dataset
    ↓
Save Best Accuracy
    ↓
Save Confusion Matrix
    ↓
Write train_log.txt
```

---

# 31. Important Parameters

| Parameter | Purpose |
|---|---|
| patch_size | HSI patch size |
| training_sample_ratio | train pixel ratio |
| layers_num | number of progressive stages |
| pre_epoch | pretraining epochs |
| max_epoch | final training epochs |
| lambda_1 | contrastive loss weight |
| lambda_2 | adversarial loss weight |
| pro_dim | embedding dimension |
| lr | learning rate |
| seed | reproducibility |
| sam_bool | enable SAM optimizer |
| g_bool | enable generator augmentation |

---

# 32. Random Seed

Example:

```python
seeds = [333,111,222]
```

Purpose:
- reproducible experiments
- stable comparisons

---

# 33. Why Multiple Seeds?

Different random initialization:
- changes results slightly

Using multiple seeds:
- gives stable average performance

---

# 34. Overall PMGDG Core Idea

PMGDG learns:

```text
partial spectral representations
≈
full spectral representations
```

while also aligning:
- source domain
- target domain

using:
- contrastive learning
- adversarial learning
- progressive spectral stages

---

# 35. Complete Architecture Flow

```text
Input HSI Patch
        ↓
Generator
        ↓
Progressive Features
        ↓
Discriminator
        ↓
Embeddings + Predictions
        ↓
Contrastive Learning
        ↓
Adversarial Alignment
        ↓
Classification Learning
        ↓
Backpropagation
        ↓
Model Update
```

---

# 36. Final Output

The project finally returns:

```python
best_acc
best_kappa
```

These represent:
- best target accuracy
- best target kappa score

---

# 37. Important Files

| File | Purpose |
|---|---|
| main.py | main training pipeline |
| datasets.py | dataset loading + HyperX |
| utils_HSI.py | helper functions |
| models.py | generator/discriminator |
| train_log.txt | final results |
| *.pth | saved models |

# HSI
=======
>>>>>>> 2aec1aabdf541210a367c6fb2f1f5be922bf9a59
# HSI
