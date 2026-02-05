# Fourier Neural Operator for Spectral Wave Diffusion

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
│ ├── snl_physics.py
│ ├── data_loader.py # PDEBench data loading
│ ├── model.py # FNO model definition
│ ├── train.py # Training & validation logic
│
├── data/
│ ├── download_pdebench.py # PDEBench data downloading
│ ├── 1D_diff-sorp_NA_NA.h5 # After download
│ └── snl/
│   ├── snl_dataset.py  # Generated dataset
│   └── inspect_polar_snl.py # Polar visualization & validation
│ 
└── scripts/
  ├── generate_snl_data.py  # Dataset generation (physics-based)
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

## Dataset Generation

A dataset generation script is provided in the `scripts/` directory to generate the required `.h5` files automatically. 

1. **Execute the generation script:**
   ```bash
   python scripts/generate_snl_data.py \
    --n-samples 10000 \
    --n-omega 64 \
    --n-theta 64 \
    --out data/snl/snl_dataset.h5
   ```


## 🚀 Running

### Training the Model
To train the Fourier Neural Operator on the Diffusion-Sorption dataset, run the provided training script. This script handles data loading, preprocessing and the training loop.

```bash
python scripts/run_training.py
```
