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
├── scripts/
│ └── run_training.py
└── notebooks/
  └── train_val_loss_curves.py # Visualization
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
source venv/bin/activate      # Linux
```
On Windows:
```bash
venv\Scripts\activate         # Windows
```
### 3. Install Dependencies
Once the environment is active, upgrade pip and install the required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 4. Install the Package
```bash
pip install -e .
```
---

## ⬇️ Dataset Download

A download script is provided in the `data/` directory to fetch the required `.h5` files automatically. 

1. **Execute the download script:**
   ```bash
   python data/download_pdebench.py
   ```
2. **File Placement:**
  The script will download the data into the data/ folder. Upon completion, verify the directory structure looks like this:
    ```text
    data/
    └── 1D_diff-sorp_NA_NA.h5  # ~4.0 GB
    ```
    
The dataset can also be manually downloaded and placed in the data/ folder


## 🚀 Running

### Training the Model
To train the Fourier Neural Operator on the Diffusion-Sorption dataset, run the provided training script. This script handles data loading, preprocessing and the training loop.

```bash
python scripts/run_training.py
```
