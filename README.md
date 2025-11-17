# 🚀 AIMS-DTU Research Intern Round 2: Self-Supervised Learning Project

**Author:** Lakshay Pal  
**Date:** May 29, 2025  

---

## 📝 Project Overview

This project implements and analyzes three prominent self-supervised learning (SSL) methods for visual representation learning:

1. **SimCLR (Simple Framework for Contrastive Learning of Visual Representations):** A contrastive learning approach.
2. **MAE (Masked Autoencoder):** A masked image modeling technique.
3. **MoCo v2 (Momentum Contrast v2):** A contrastive learning approach utilizing a momentum encoder and a dynamic queue.

All models were pretrained on a subset of the ImageNet-100 dataset, and their learned features were evaluated using a linear probing protocol on a 100-class image classification task. Due to significant time and computational constraints, pretraining was limited to 5 epochs for each model. The primary goal was to demonstrate the implementation of these SSL pipelines and understand their core mechanisms.

---

## 📚 Dataset

- **Name:** ImageNet-100 subset  
- **Description:** ~130,000 training images and 5,000 validation images across 100 classes.  
- **Structure:**  
  - Training: `train.X1`, `train.X2`, `train.X3`, `train.X4` (each with class subfolders)  
  - Validation: `val.X` (with class subfolders)  
  - `Labels.json`: Maps class IDs to class names in the dataset root.

---

## 🧩 Dependencies

- 🐍 Python 3.13  
- 🔥 PyTorch 2.2.2  
- 🖼️ torchvision 0.17.2  
- 🏷️ timm (latest tested during development)  
- 📊 numpy  
- 📈 matplotlib  
- 🧪 scikit-learn  
- ⏳ tqdm

To generate a `requirements.txt` file:
```bash
pip freeze > requirements.txt
```

---

## ⚙️ Setup

1. **Clone the repository (if using GitHub):**
    ```bash
    git clone [your-repo-link]
    cd [repository-name]
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv ssl_env
    source ssl_env/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Or manually:
    ```bash
    pip install torch torchvision timm numpy matplotlib scikit-learn tqdm
    ```

---

## 🗂️ Dataset Setup

1. 📥 Download the ImageNet-100 subset.  
   [Download from Google Drive](https://drive.google.com/file/d/1MdmkhdkhNjXM_PZDaZ9kRuoaS80vsZ8_/view?usp=sharing)
2. 📁 Place it in a directory and update the `BASE_DATA_PATH` variable at the top of each script:
    - `run_ssl_project.py`
    - `run_mae.py`
    - `run_moco.py`
3. 🏷️ Ensure `Labels.json` is present in the `BASE_DATA_PATH`.

---

## 🏃‍♂️ Running the Scripts

Each SSL method is implemented in a separate script. Pretrained models and output plots are saved in the current working directory.

> ⚠️ **Note:** These scripts are configured for only 5 pretraining epochs due to resource constraints. For real-world performance, longer pretraining (100–800 epochs) is required.

---

### 1️⃣ SimCLR (`run_ssl_project.py`)

Implements SimCLR using a ResNet-18 backbone.

**Run with:**
```bash
python run_ssl_project.py
```

**Key Hyperparameters:**
- 🏗️ Backbone: ResNet-18
- 🔁 Pretraining Epochs: 5
- 📦 Pretraining Batch Size: 64
- ⚙️ Optimizer (Pretrain): AdamW, LR: 3e-4
- 🔢 Projection Dimension: 128
- 🌡️ Temperature: 0.5
- 🔁 Linear Probe Epochs: 10
- 📦 Linear Probe Batch Size: 64
- ⚙️ Optimizer (LP): AdamW, LR: 1e-3

**Outputs:**
- 💾 `simclr_resnet18_backbone_pretrained.pth`
- 📉 `simclr_pretrain_loss_curve.png`
- 📊 `simclr_linear_probe_curves.png`

---

### 2️⃣ MAE (`run_mae.py`)

Implements MAE with a ViT-Base encoder.

**Run with:**
```bash
python run_mae.py
```

**Key Hyperparameters:**
- 🏗️ Encoder: ViT-Base (`vit_base_patch16_224`)
- 🔁 Pretraining Epochs: 5
- 📦 Pretraining Batch Size: 32
- ⚙️ Optimizer (Pretrain): AdamW, LR: 1.5e-4, Weight Decay: 0.05
- 🕳️ Masking Ratio: 0.75
- 🧩 Decoder Embed Dim: 512, Depth: 8, Heads: 16
- 🔁 Linear Probe Epochs: 5
- 📦 Linear Probe Batch Size: 64
- ⚙️ Optimizer (LP): AdamW, LR: 1e-3

**Outputs:**
- 💾 `mae_vit_full_model_pretrained.pth`
- 📉 `mae_pretrain_loss_curve.png`
- 📊 `mae_linear_probe_curves.png`

---

### 3️⃣ MoCo v2 (`run_moco.py`)

Implements MoCo v2 using a ResNet-18 backbone.

**Run with:**
```bash
python run_moco.py
```

**Key Hyperparameters:**
- 🏗️ Backbone: ResNet-18
- 🔁 Pretraining Epochs: 5
- 📦 Pretraining Batch Size: 32
- ⚙️ Optimizer (Pretrain): SGD, LR: 0.03, Momentum: 0.9, Weight Decay: 1e-4
- 🔢 Projection Dimension: 128
- 🧮 Queue Size (K): 16384
- 🔄 Momentum (m): 0.999
- 🌡️ Temperature (T): 0.07
- 🔁 Linear Probe Epochs: 5
- 📦 Linear Probe Batch Size: 64
- ⚙️ Optimizer (LP): AdamW, LR: 1e-3

**Outputs:**
- 💾 `moco_resnet18_encoder_q_pretrained.pth`
- 📉 `moco_pretrain_loss_curve.png`
- 📊 `moco_linear_probe_curves.png`

---

## 📝 Notes

- 🕒 For best results, increase the number of pretraining epochs and consider tuning hyperparameters.
- 🧩 All scripts are designed to be run independently.
- 💾 Results and model checkpoints are saved in the current directory. 