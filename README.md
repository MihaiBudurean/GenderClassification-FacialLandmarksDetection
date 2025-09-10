# Gender Classification and Facial Landmark Detection

This project investigates **single-task vs. multi-task learning** approaches for facial analysis using a subset of the **CelebA dataset**. The tasks include **gender classification** and **landmark detection** (left eye, right eye, and nose).

---

## 📊 Project Description

The objective is to compare baseline single-task models against multitask architectures and evaluate whether multi-task learning improves accuracy and efficiency.

### Tasks

* **Gender Classification (Single-task)** – **ResNet-18** backbone with a binary classification head.
* **Landmark Detection (Single-task)** – **ResNet-18** backbone with regression head predicting landmark coordinates.
* **Multitask Baseline** – Shared encoder with dual heads (classification + regression).
* **Multitask Improved** – Enhanced multitask setup with pretrained **ResNet-18**, stronger augmentations, **Wing Loss**, and **OneCycleLR** scheduling.

### Metrics

* **Gender classification**: Accuracy, ROC AUC, Confusion Matrix.
* **Landmark detection**: MSE (Mean Squared Error), NME (Normalized Mean Error).

---

## 🚀 Requirements and Usage

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 📈 Results

### Gender Classification

* Baseline: Accuracy ≈ 88%, AUC ≈ 0.925
* Multitask baseline: Accuracy ≈ 60%, AUC ≈ 0.686
* **Multitask improved**: Accuracy ≈ 92%, AUC ≈ 0.989

### Landmark Detection

* Baseline: MSE ≈ 4.81, NME ≈ 0.0103
* Multitask baseline: MSE ≈ 10.08, NME ≈ 0.0142
* **Multitask improved**: MSE ≈ 1.40, NME ≈ 0.0055

**Key Insight:** Naïve multitask models underperform, but with targeted improvements, multitask learning can **outperform single-task models** on both classification and landmark detection.
