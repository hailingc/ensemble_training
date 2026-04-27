# Hyperparameter-Divergent Ensemble Training (HDET)

**Paper:** *Scalable Hyperparameter-Divergent Ensemble Training with Automatic Learning Rate Exploration for Large Models*
Hailing Cheng, Tao Huang, Chen Zhu, Antonio Alonso — LinkedIn Inc.

---

## Overview

Large-model data-parallel training wastes its GPU replicas: every rank computes near-identical updates under the same learning rate, leaving the rich space of hyperparameter configurations entirely unexplored.

**HDET** repurposes these replicas for simultaneous learning rate exploration at negligible communication overhead. It operates in alternating phases:

- **Fan-out** — each of *N* replicas trains under a distinct learning rate drawn from a symmetric spread `[η̄(1−α), η̄(1+α)]` around the shared base schedule.
- **Converge** — every *T* steps, parameters are AllReduce-averaged across all replicas, collapsing the ensemble into a single model that seeds the next fan-out round.

Building on this substrate, an **automatic LR (auto-LR) controller** treats cross-replica training loss as a performance signal and drives a momentum-based meta-update that shifts the base schedule toward higher-performing configurations — no additional hyperparameter sweeps required.

On a production-scale recommendation model (8× H100, one epoch over one year of feed data), HDET:
- Trains stably at `η=0.0009`, a learning rate that causes vanilla DDP to diverge.
- Achieves a lower final training loss (3.277) than the conservative baseline (3.294, `η=0.0001`).
- Autonomously discovers the empirically correct per-group LR decay ordering (transformer fastest, non-decay slowest).

HDET generalizes beyond learning rate: any scalar hyperparameter that does not change model architecture (dropout, weight decay, label smoothing, attention temperature) can be explored across replicas using the same protocol.

---

## Repository Structure

```
ensemble_training/
├── src/
│   └── one_cycle.py      # OneCycleAutoLR scheduler (drop-in replacement)
└── paper/
    └── ensemble.tex      # NeurIPS 2026 submission
```

---

## Code: `src/one_cycle.py`

`OneCycleAutoLR` is a drop-in subclass of `torch.optim.lr_scheduler.OneCycleLR` that adds three capabilities on top of standard DDP training:

### 1. LR Divergence (Fan-Out)

At initialization each rank receives a multiplier

```
ρ_r = 1 + α · (r − (N−1)/2) / δ,    δ = max((N−1)/2, 0.5)
```

so the spread is symmetric (mean multiplier = 1) and bounded to `[1−α, 1+α]`. Setting `α=0` recovers standard DDP exactly.

### 2. Weight Averaging (Converge)

Every `model_sync_interval` steps, HDET AllReduce-averages all model parameters:

```python
dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
param.data.div_(world_size)
```

The extra AllReduce adds ~0.1% wall-clock time at `T=1000`.

### 3. Auto-LR Controller

After `auto_lr_warmup_steps` warmup steps, at each sync point the controller:

1. Gathers per-rank average losses via `all_gather`.
2. Softmax-weights ranks by `(L̄ − L_r) / σ` (better rank = higher weight).
3. Computes a performance-weighted LR average and derives a delta signal `Δ = η̃ − η̄`.
4. Updates a momentum velocity: `v ← β·v + (1−β)·Δ`.
5. Decays and shifts the base LR: `η̄ ← η̄(1−γ) + v·λ`, where `γ` is matched to OneCycleLR's floor.
6. Randomly reassigns multipliers `ρ_r` to ranks via a fresh permutation, preventing cross-parameter correlation when multiple param groups are explored simultaneously.

### Constructor Parameters

| Parameter | Default | Description |
|---|---|---|
| `optimizer` | — | PyTorch optimizer |
| `max_lr` | — | Maximum learning rate (float or per-group list) |
| `total_steps` | `None` | Total training steps |
| `epochs` / `steps_per_epoch` | `None` | Alternative to `total_steps` |
| `ensemble_lr_spread_ratio` | `0.0` | Spread ratio α; `0.0` = standard OneCycleLR |
| `model_sync_interval` | `1000` | Weight averaging interval *T* (steps) |
| `auto_lr_warmup_steps` | `100000` | Steps before auto-LR activates |
| `enable_auto_lr` | `False` | Enable the auto-LR controller |
| `loss_scale` | `0.002` | Temperature σ for softmax weighting |
| `lr_of_lr` | `0.5` | Step size λ for the meta-update |
| `momentum_of_lr` | `0.9` | Momentum β for the velocity term |
| `rank` | `0` | Local process rank (overridden at runtime if dist is initialized) |
| `world_size` | `1` | Number of replicas (overridden at runtime) |

All remaining parameters (`pct_start`, `anneal_strategy`, `div_factor`, etc.) are passed through to `OneCycleLR` unchanged.

### Usage

```python
from src.one_cycle import OneCycleAutoLR

scheduler = OneCycleAutoLR(
    optimizer,
    max_lr=0.0009,
    total_steps=total_steps,
    ensemble_lr_spread_ratio=1/9,   # spread α
    model_sync_interval=1000,       # sync every T=1000 steps
    auto_lr_warmup_steps=100_000,   # warmup before auto-LR kicks in
    enable_auto_lr=True,
    rank=dist.get_rank(),
    world_size=dist.get_world_size(),
)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)            # pass current loss for auto-LR
    optimizer.zero_grad()
```

The only change from a standard `OneCycleLR` loop is passing `loss` to `scheduler.step()`. Everything else — DDP setup, gradient synchronization, data pipeline — is unchanged.

### Warm Noisy Initialization (optional)

Before starting HDET, each replica can be perturbed from a shared pre-trained checkpoint:

```python
std = nu * param.data.norm() / (param.data.numel() ** 0.5)
param.data += torch.randn_like(param.data) * std   # ν = 0.01
```

This seeds each replica in a distinct neighborhood of a high-quality solution. Without periodic weight averaging, these perturbed weights compound with high-LR updates and diverge; the fan-out/converge cycle is the primary stabilizer.

---

## Experimental Results

| Model | Avg LR | Train Loss |
|---|---|---|
| Baseline-Low (η=0.0001, no spread) | 0.0001 | 3.294 |
| Baseline-High (η=0.0009, no spread) | 0.0009 | 4.169 † |
| Warm-Init (η=0.0009, no spread) | 0.0009 | 4.674 † |
| HDET w/o auto-LR | 0.0009 | 3.280 |
| HDET w/o warm init | 0.0009 | 3.281 |
| **HDET (full)** | **0.0009** | **3.277** |

† training crash (loss diverged)

---

## Citation

```bibtex
@article{cheng2026hdet,
  title   = {Scalable Hyperparameter-Divergent Ensemble Training with
             Automatic Learning Rate Exploration for Large Models},
  author  = {Cheng, Hailing and Huang, Tao and Zhu, Chen and Alonso, Antonio},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2026}
}
```

---

## Requirements

- Python 3.8+
- PyTorch ≥ 2.0 with `torch.distributed`
- A multi-GPU setup with DDP (single-GPU usage works with `world_size=1`, `ensemble_lr_spread_ratio=0.0`)
