# Fourier Neural Operator for 1D Diffusion–Sorption PDEs

This project implements a **Fourier Neural Operator (FNO)** to learn solution operators of a 1D diffusion–sorption partial differential equation:

```math
u_t = \nu u_{xx} - \sigma(u)
```

The model is trained and evaluated using data from **PDEBench**.

---

## 📦 Project Structure
```text
fno-diffusion/
├── pyproject.toml
├── README.md
├── requirements.txt
│
├── fno_diffusion/ # Installable Python package
│ ├── init.py
│ ├── data_loader.py # PDEBench data loading
│ ├── model.py # FNO model definition
│ ├── train.py # Training & validation logic
│
├── data/
│ ├── download_pdebench.py # PDEBench data downloading
│ └── ( 1D_diff-sorp_NA_NA.h5 ) # After download
│
└── scripts/
  └── run_training.py
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

### 1. Clone the repository
```bash
git clone https://github.com/tclos/fno-diffusion.git
cd fno-diffusion

python3.10 -m venv venv
source venv/bin/activate      # Linux
venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt

pip install -e .
```
