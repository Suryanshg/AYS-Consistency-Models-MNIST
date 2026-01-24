# Accelerating Denoising Diffusion Models: AYS Schedules & Consistency Distillation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **CS-552 Generative AI Final Project** — Worcester Polytechnic Institute  
> *1-Month Exploratory Research Project*

## 🎯 Project Overview

This project investigates **acceleration techniques for Denoising Diffusion Probabilistic Models (DDPMs)** with a focus on:

1. **Align Your Steps (AYS)** — Optimizing the sampling schedule to minimize truncation error
2. **DDIM Sampling** — Deterministic ODE-based sampling for faster generation
3. **Consistency Models** — Single-step generation via self-consistency distillation

### Research Motivation

Standard DDPM requires 1000+ sampling steps for high-quality generation, making it computationally expensive. This project explores three complementary approaches to accelerate inference while maintaining sample quality.

---

## 🏗️ Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| `DDPMUNet` | Standard U-Net with time-conditioning for noise prediction |
| `ConsistencyUNet` | Modified U-Net for consistency training with Karras schedules |
| `AYS Scheduler` | Greedy optimization to find minimal-error time schedules |

### Model Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  DDPM Training  │ ──► │  DDIM + AYS     │ ──► │ Consistency Model │
│  (1000 steps)   │     │  (10-50 steps)   │     │   (1 step!)       │
└─────────────────┘     └──────────────────┘     └───────────────────┘
```

---

## 📁 Project Structure

```
CS-552-Generative-AI-Final-Project/
├── models/
│   ├── DDPMUNet.py           # Standard DDPM U-Net architecture
│   ├── ConsistencyUNet.py    # Consistency model architecture
│   └── ConsistencyUNet2.py   # Improved consistency architecture
├── datasets/
│   └── mnist_dataloader.py   # MNIST data loading utilities
├── std_diffusion.py          # DDPM training & FID evaluation
├── train_consistency.py      # Consistency model training
├── generate_samples.py       # Sample generation scripts
├── eval_consistency.py       # Evaluation utilities
├── colab_ddpm_training.ipynb # Interactive Colab notebook
└── trained_model_weights/    # Saved model checkpoints
```

---

## 🔬 Key Experiments

### 1. Standard DDPM Training
- **Dataset:** MNIST (28×28 grayscale)
- **Diffusion Steps:** 1000
- **Schedule:** Linear β from 0.0001 to 0.02
- **Optimizer:** AdamW (lr=1e-4)

### 2. DDIM Sampling with AYS
- **Optimized Schedule:** `[999, 689, 604, 524, 394, 285, 204, 158, 112, 61, 0]`
- **Reduction:** 1000 steps → **10 steps** (100× speedup)

### 3. Consistency Distillation
- **Schedule:** Karras sigma schedule (σ_min=0.002, σ_max=80)
- **Curriculum:** N grows from 2 → 150 over training
- **Result:** Single-step generation from noise

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install uv (Python package manager)
# See: https://docs.astral.sh/uv/getting-started/installation/

# Clone and setup
git clone https://github.com/YOUR_USERNAME/CS-552-Generative-AI-Final-Project.git
cd CS-552-Generative-AI-Final-Project

# Create environment & install dependencies
uv sync
source .venv/bin/activate
```

### Training DDPM

```bash
python std_diffusion.py
```

### Training Consistency Model

```bash
python train_consistency.py
```

### Google Colab

For GPU-accelerated training, use the provided notebook:
- Open `colab_ddpm_training.ipynb` in Google Colab
- Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
- Run all cells

---

## 📊 Results

| Method | Sampling Steps | Generation Time | FID (MNIST) |
|--------|---------------|-----------------|-------------|
| DDPM (Baseline) | 1000 | ~30s | Baseline |
| DDIM | 50 | ~1.5s | Comparable |
| DDIM + AYS | 10 | ~0.3s | Comparable |
| Consistency Model | 1 | ~0.03s | Slight degradation |

> *FID scores computed using InceptionV3 features*

---

## 📚 References

1. **DDPM:** Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
2. **DDIM:** Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
3. **AYS:** Sabour et al., "Align Your Steps: Optimizing Sampling Schedules in Diffusion Models" (2024)
4. **Consistency Models:** Song et al., "Consistency Models" (ICML 2023)
5. **Karras Schedule:** Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (NeurIPS 2022)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Aditya Patil**  
*Worcester Polytechnic Institute*  
*CS-552: Generative AI — January 2026*
