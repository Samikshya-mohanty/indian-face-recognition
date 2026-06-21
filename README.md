# 🎬 Indian Face Recognition System

A deep learning-based face recognition system fine-tuned on the **Indian Movie Face Database (IMFDB)** to identify 100 Indian actors from photos in real time.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-yellow)](https://gradio.app)

---

## 📌 Project Overview

This project was built as part of a Data Science capstone assignment. The goal was to fine-tune a pretrained FaceNet model on Indian actor faces and build a real-time face recognition demo with unknown person rejection.

**Key highlights:**
- 85.70% Top-1 accuracy and 94.96% Top-5 accuracy on 100 Indian actors
- MTCNN face detection pipeline for robust real-world photo handling
- Cosine similarity-based matching using FaceNet embeddings
- Unknown person rejection with a configurable threshold
- Dynamic database — add new faces on the fly via Gradio demo

---

## 📊 Results

| Model | Top-1 Accuracy | Top-5 Accuracy |
|---|---|---|
| MobileNetV2 (baseline) | 41.69% | 70.06% |
| FaceNet (frozen) | 74.09% | 88.91% |
| FaceNet (fine-tuned) ✅ | **85.70%** | **94.96%** |

**Best performing actors:** KajalAgarwal (F1: 0.976), MadhabiMukherjee (F1: 0.976), RishiKapoor (F1: 0.970)

**Most challenging actors:** Shivaram (F1: 0.581), Pavithralokesh (F1: 0.630), Jagathi (F1: 0.656)

---

## 🗂️ Project Structure

```
ml_project/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda_data_prep.ipynb       ← EDA, filtering, resizing, MTCNN splits
│   └── 02_model_training_demo.ipynb ← Training, evaluation, Gradio demo
├── src/
│   └── main.py
└── data/
    └── actors_list.txt              ← List of 100 actors
```

---

## 🧠 Model Architecture

```
Input Photo
    ↓
MTCNN Face Detection (crop + align)
    ↓
FaceNet (pretrained on VGGFace2)
    ↓
Fine-tuned on IMFDB (last 30 layers unfrozen)
    ↓
512-d Face Embedding
    ↓
Cosine Similarity vs Reference Database
    ↓
Prediction / Unknown Rejection
```

---

## 📁 Dataset

**Source:** [IMFDB — Indian Movie Face Database](https://cvit.iiit.ac.in/projects/IMFDB/) by IIIT Hyderabad

| Split | Images | Actors |
|---|---|---|
| Train | 16,554 | 100 |
| Val | 3,511 | 100 |
| Test | 3,651 | 100 |
| **Total** | **23,716** | **100** |

> **Note:** Raw dataset and processed splits are not included in this repo due to size. Download IMFDB from the official link above.

**📥 Download MTCNN-processed splits (ready to use for training):**
[mtcnn_splits.zip — Google Drive](https://drive.google.com/file/d/1AP6S81gDjilp2Mw2mGhtf5euxE2htktE/view?usp=sharing)

Extract into `data/mtcnn_splits/` before running `02_model_training_demo.ipynb`.

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/Samikshya-mohanty/indian-face-recognition.git
cd indian-face-recognition
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Data Preparation (VS Code)
Open and run `notebooks/01_eda_data_prep.ipynb` in order:
- Loads and filters IMFDB dataset
- Resizes images to 160×160
- Runs MTCNN face detection
- Creates train/val/test splits

### Model Training + Demo (Google Colab)
Open `notebooks/02_model_training_demo.ipynb` in Google Colab:

**First time (full run):**
```
Cell 0   → Install packages
Cell 1   → Mount Google Drive
Cell 2   → Config
Cell 3   → Build dataset
Cell 4-6 → MobileNetV2 baseline
Cell 7-9 → FaceNet baseline
Cell 10-12 → Fine-tuning
Cell 13  → Load model
Cell 14  → Class names
Cell 15-17 → Evaluation + Confusion matrix
Cell A   → Build embedding model
Cell B   → Build reference database
Cell C   → Save/Load functions
Cell D   → Threshold finder
Cell E   → Launch Gradio demo
```

**Every new session (skip retraining):**
```
Cell 0  → Install packages
Cell 1  → Mount Drive
Cell 2  → Config
Cell 13 → Load model (loads saved DB automatically)
Cell A  → Build embedding model
Cell E  → Launch Gradio demo
```

---

## 🎮 Gradio Demo Features

- Upload any photo → MTCNN detects face → FaceNet matches against database
- ✅ **Known actor** → shows name + confidence score
- 🚫 **Unknown person** → rejected with best match shown
- ➕ **Add to database** — type a name and add unknown faces
- 🗑️ **Discard** — dismiss unknown faces

---

## 📦 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| TensorFlow / Keras | Model training |
| FaceNet (keras-facenet) | Pretrained face embeddings (VGGFace2) |
| MTCNN | Face detection + alignment |
| OpenCV + PIL | Image processing |
| scikit-learn | Cosine similarity, metrics |
| Gradio | Interactive demo |
| Google Colab (T4 GPU) | Training environment |

---

## 📚 References

- Setty et al. (2013). *Indian Movie Face Database: A Benchmark for Face Recognition Under Wide Variations.* NCVPRIPG. [IIIT Hyderabad](https://cvit.iiit.ac.in/projects/IMFDB/)
- Schroff et al. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR. [arXiv](https://arxiv.org/abs/1503.03832)
- Zhang et al. (2016). *Joint Face Detection and Alignment using MTCNN.* IEEE Signal Processing Letters.

---

## 👩‍💻 Author

**Samikshya Mohanty**
[GitHub](https://github.com/Samikshya-mohanty/indian-face-recognition)
