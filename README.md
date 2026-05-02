# Fourier Neural Operator for Spectral Wave Diffusion

This project investigates the use of a Fourier Neural Operator (FNO) to emulate the nonlinear wave-wave interaction source term S_nl in spectral ocean wave models. S_nl is computed here via the DE3 model (Pushkarev et al. 2004) and represents the redistribution of energy across the wave spectrum through resonant four-wave interactions. Accurately computing S_nl is physically critical (it drives the downshift of the spectral peak and produces the characteristic tripolar pattern (+/−/+) in frequency space) but it is computationally expensive in operational predicition.

The goal is to train an FNO to learn the operator mapping E(f, θ) → S_nl(f, θ), where E is the 2D directional wave energy spectrum. 

---

## 📦 Project Structure

```text
fno-diffusion/
├── pyproject.toml
├── README.md
├── requirements.txt
│
├── fno_diffusion/                  # Installable Python package
│   ├── __init__.py
│   ├── snl_physics.py              # DE3 physics model (JONSWAP + S_nl)
│   ├── model.py                    # FNO model definition
│   └── train.py                    # Training & validation logic
│
├── data/
│   └── snl/
│       └── snl_dataset.h5          # Generated dataset
│
└── scripts/
    ├── generate_snl_data.py               # Dataset generation
    ├── run_training.py                    # Baseline FNO training
    ├── inspect_polar_snl.py               # Baseline FNO / ground-truth inspection
    ├── train_compare_norm.py              # Global vs per-sample normalization
    ├── train_compare_loss.py              # Relative L2 vs peak-weighted L2
    ├── run_training_factorized_scale.py   # Factorized model training
    ├── inspect_polar_snl_factorized.py    # Factorized model inspection
    ├── compare_factorized_scale.py        # Evaluation of a_gt vs a_pred
    └── compare_factorized_field.py        # Evaluation of normalized field prediction
```

---

## ⚙️ Requirements

- Python **3.10**
- NumPy **< 2.0** (required for compatibility)
- PyTorch
- neuraloperator
- h5py

---

## 🔧 Installation

Follow these steps to set up the environment and install the package in editable mode.

### 1. Clone the Repository

```bash
git clone https://github.com/tclos/fno-diffusion.git
cd fno-diffusion
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
```

On Linux/macOS:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install the Package

```bash
pip install -e .
```

---

## 📊 Dataset Generation

The dataset consists of 2D wave energy spectrum `E(f, θ)` as input and the corresponding
nonlinear source term `S_nl(f, θ)` computed via the DE3 model (Pushkarev et al. 2004) as target.
Sample 0 is always the Janssen & Putnam benchmark for calibration.

```bash
python scripts/generate_snl_data.py \
    --n-samples 5000 \
    --n-f 128 \
    --n-theta 64 \
    --out data/snl/snl_dataset.h5
```

| Argument | Default | Description |
|---|---|---|
| `--n-samples` | 1000 | Total number of samples |
| `--n-f` | 128 | Frequency resolution (power of 2 recommended) |
| `--n-theta` | 64 | Directional resolution |
| `--f-min` | 0.04 | Minimum frequency (Hz) |
| `--f-max` | 1.0 | Maximum frequency (Hz) |
| `--seed` | 42 | Random seed |
| `--out` | `data/snl/snl_dataset.h5` | Output path |

---

## 🚀 Training

### Main Training Script

Trains the FNO using per-sample normalization and peak-weighted relative L2 loss.
Saves `model.pth`, `model_best.pth`, `norm_params.json`, `metrics.json` and `loss_curves.pdf`
to `results_snl/`.

```bash
python scripts/run_training.py
```

Key design decisions in `run_training.py`:

- **Normalization:** each sample is normalized by its own absolute maximum -
  `Y_norm[i] = Y[i] / (|Y[i]|.max + ε)` - so that low-amplitude samples
  contribute equally to the gradient.
- **Loss:** peak-weighted relative L2, which assigns quadratic weights proportional
  to the local amplitude of the target field, focusing the gradient on the physically
  relevant tripolar lobe structure of S_nl.
- **Split:** 80/20 train/val, fixed seed. Normalization statistics are computed
  exclusively on the training set.

---

## 🔍 Inspection & Sanity Checks

### Visualize a sample (ground truth only)

```bash
python scripts/inspect_polar_snl.py --idx 0
```

### Compare FNO prediction vs ground truth

```bash
python scripts/inspect_polar_snl.py --idx 15 --model results_snl/model_best.pth # change path to desired model
```

Generates a 2×4 panel with polar plots of `E(f,θ)`, `S_nl` ground truth,
`S_nl` FNO prediction and the error `(FNO − GT)`, plus a 1D directional
integral comparison. The script automatically reads `norm_params.json`
from the model directory to undo normalization before plotting.

Physical sanity checks to look for:
- `S_nl` should show a **tripolar pattern** (+/−/+): negative lobe at the spectral
  peak `fp`, positive lobes at lower and higher frequencies.
- The directional integral `∫S_nl df` should be close to zero (energy redistribution,
  not creation).

---

## 🧪 Experimental Scripts

### Normalization comparison

Compares global z-score normalization vs per-sample normalization,
keeping the loss fixed as standard relative L2.

```bash
python scripts/train_compare_norm.py --epochs 100
```

Saves both models and a comparative loss curve plot to `results_compare_norm/`.

### Loss function comparison

Compares standard relative L2 vs peak-weighted relative L2,
keeping the normalization fixed as per-sample.

```bash
python scripts/train_compare_loss.py --epochs 100 --alpha 2.0
```

Saves both models and a comparative plot to `results_compare_loss/`.

| File | Script | Description |
|---|---|---|
| `results_snl/model_best.pth` | `run_training.py` | Best FNO checkpoint |
| `results_snl/norm_params.json` | `run_training.py` | Normalization parameters |
| `results_snl/metrics.json` | `run_training.py` | Final training metrics |
| `results_snl/loss_curves.pdf` | `run_training.py` | Train/val loss curves |
| `results_compare_norm/compare_norm.pdf` | `train_compare_norm.py` | Normalization comparison plot |
| `results_compare_loss/compare_loss.pdf` | `train_compare_loss.py` | Loss comparison plot |
---

### Factorized Training (Shape + Scale)

#### `run_training_factorized_scale.py`

Trains the **factorized model**, where:
- an FNO learns the normalized field shape `S_nl(f, θ)`
- a small MLP (`ScaleHead`) predicts the scalar amplitude `a = max |S_nl|`

Two modes:
- Full training (both branches)
```bash
python scripts/run_training_factorized_scale.py
```

- Freeze a pre-trained FNO (best Relative L2 model) and train only the scale network
```bash
python scripts/run_training_factorized_scale.py \
    --field-checkpoint results_compare_loss/model_A_rel_l2.pth \
    --freeze-field
```

**Outputs (in `results_snl_factorized/`):**

| File | Description |
|------|------------|
| `model.pth` | Final checkpoint |
| `model_best.pth` | Best validation checkpoint |
| `factorized_params.json` | Parameters used in factorization |
| `loss_field.pdf` | Loss curve for FNO (shape) |
| `loss_scale.pdf` | Loss curve for amplitude network |
| `loss_total.pdf` | Combined loss |


### 🔍 Factorized Model Inspection

#### `inspect_polar_snl_factorized.py`

Visual inspection of the **factorized model predictions**.

```bash
python scripts/inspect_polar_snl_factorized.py \
    --idx 0 \
    --model results_snl_factorized/model_best.pth
```

**Outputs (in `data/snl/`):**

| File | Description |
|------|------------|
| `snl_factorized_<idx>.png` | Comparison plot showing ground truth, prediction, error and directional integral |

---

### 📊 Scale Evaluation

#### `compare_factorized_scale.py`

Evaluates the prediction of the **amplitude `a`**.

```bash
python scripts/compare_factorized_scale.py \
    --model results_snl_factorized/model_best.pth \
```

**Outputs (in `results_compare_factorized_scale/`):**

| File | Description |
|------|------------|
| `scale_scatter_val.pdf` | Scatter plot `a_gt vs a_pred` (log scale) |
| `scale_rel_err_hist_val.pdf` | Histogram of relative error |
| `scale_metrics_val.csv` | Per-sample metrics |
| `scale_summary_val.json` | Aggregated statistics |

---

### 📊 Field (Shape) Evaluation

#### `compare_factorized_field.py`

Evaluates the prediction of the **normalized field shape**.

```bash
python scripts/compare_factorized_field.py \
    --model results_snl_factorized/model_best.pth \
```

**Outputs (in `results_compare_factorized_field/`):**

| File | Description |
|------|------------|
| `field_rel_l2_hist_val.pdf` | Histogram of Relative L2 error |
| `field_examples_val.pdf` | Visual examples (GT vs Pred vs Error) |
| `field_metrics_val.csv` | Per-sample metrics |
| `field_summary_val.json` | Aggregated statistics |

---
