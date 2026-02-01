# Formal Verification & Robustness Analysis of Skin Cancer CNNs

This repository implements a **Formal Verification** framework to audit the robustness of a Convolutional Neural Network (CNN) trained for skin cancer detection (HAM10000).

We use the **AutoLiRPA** framework (CROWN and $\alpha,\beta$-CROWN) to achieve three main objectives:

1. **Baseline Verification (auto_LiRPA & CROWN):**
   The project utilizes Linear Relaxation-based Perturbation Analysis (LiRPA) to compute guaranteed output bounds efficiently. This serves as a rapid screening method to prove that a model's prediction remains constant within a specific $\epsilon$-ball of noise.

2. **High-Precision Complete Verification ($\alpha,\beta$-CROWN):**
   To address the precision limitations of linear relaxation (especially in non-linear architectures with MaxPooling), we employ **$\alpha,\beta$-CROWN**. This method integrates **Branch-and-Bound (BaB)** to iteratively refine the verification bounds, maximizing the certification rate where the baseline method yields incomplete results.

3. **Rotational Robustness:**
   Unlike standard pixel-wise perturbations, rotation involves spatial interpolation. The scripts in `test_rotation` analyze the safety limits of the model when facing geometric shifts, which is crucial for real-world computer vision reliability.


## Structure
```
SAFE-AND-VERIFIED-AI-PROJECT/
├── alpha-beta-CROWN/           # Official verifier for complete/incomplete verification.
├── code/                       # Core assets and base models.
│   ├── CNN_modelFV.ipynb       # Model definition, training, and evaluation notebook.
│   ├── data_X.npy / data_Y.npy # Pre-processed dataset files in NumPy format.
│   ├── find_limit_rotation.py  # Script to calculate critical rotation thresholds.
│   ├── skin_model.pth          # Trained PyTorch model weights.
│   └── requirements.txt        # Python dependencies.
├── dataset/                    # input data.
├── plots/                      # Exported visualizations and performance graphs.
├── test_autoLirpa_noise/       # Robustness bounds using the auto_LiRPA library.
│   └── noise_autoLirpa.py      # Script for linear relaxation-based perturbation analysis.
├── test_noise/                 # Noise-specific robustness pipelines.
│   ├── config_noise.yaml       # Configuration file for noise parameters.
│   ├── robustness_curve.py     # Generates accuracy vs. perturbation magnitude plots.
│   └── verification_data.pt    # PyTorch tensors containing verification results.
├── test_noise_avgpool/         # Specialized tests for models with Average Pooling layers.
└── test_rotation/              # Geometric transformation analysis.
    └── find_rotation.py        # Analysis of model stability under rotational variance.
```

Setup and Installation
Clone the repository:

```bash
git clone https://github.com/afm719/SAFE-AND-VERIFIED-AI-PROJECT.git
cd SAFE-AND-VERIFIED-AI-PROJECT
```
Install Dependencies:

```bash
pip install -r code/requirements.txt
```
Running a Verification Test:

```bash
python test_noise/robustness_curve.py
```

Autolirpa 

```bash
cd test_autoLirpa_noise
python noise_autoLirpa.py
```

Rotation

```bash
cd test_rotation
python find_rotation.py
```
To run the complete verification process using Branch-and-Bound, you must execute the script from the root directory of the repository.

First, set the PYTHONPATH to the current directory to ensure all modules are loaded correctly:

PowerShell:
```bash
$env:PYTHONPATH = "$PWD"
```
Linux / Mac:
```bash
export PYTHONPATH=$PWD
```

Then, run the verifier pointing to the noise configuration file. You can specify the range of images to verify (e.g., from index 0 to 20) using --start and --end:
```bash
python alpha-beta-CROWN/complete_verifier/abcrown.py --config test_noise/config_noise.yaml --start 0 --end 20
```
