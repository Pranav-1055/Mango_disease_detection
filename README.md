# Mango_disease_detection
#  Mango Disease Detection using Hierarchical Deep Learning

A deep learning-based hierarchical image classification system for detecting diseases in different parts of a mango plant (Blossom, Fruit, Leaf, Stem).

This project uses **TensorFlow + EfficientNetV2 (Transfer Learning)** to build a two-level classification pipeline:

- **Level 1:** Identify plant part (Blossom / Fruit / Leaf / Stem)
- **Level 2:** Identify disease specific to that plant part

---

##  Project Overview

This system follows a **Hierarchical Classification Approach**:

1️ First model classifies the plant part  
2️ Based on predicted part, a second model predicts the disease  

This improves accuracy compared to a single flat classifier.

---

##  Model Architecture

- Backbone: `EfficientNetV2S` (ImageNet pretrained)
- Input Size: 224x224
- Optimizer: AdamW
- Loss Function: Sparse Categorical Crossentropy
- Mixed Precision Training enabled (if GPU supported)
- Early Stopping used
- Fine-tuning last 30% of backbone layers

---

##  Dataset Structure
│
├── Blossom/
│ ├── Disease1/
│ ├── Disease2/
│
├── Fruit/
│ ├── Disease1/
│ ├── Disease2/
│
├── Leaf/
│ ├── Disease1/
│ ├── Disease2/
│
├── Stem/
│ ├── Disease1/
│ ├── Disease2/

Each top-level folder represents a plant part.  
Each subfolder represents disease classes.

---

## 🚀 Training Pipeline

Training script: :contentReference[oaicite:2]{index=2}  

### Training Steps

1. Load dataset using `image_dataset_from_directory`
2. Apply data augmentation:
   - Random Flip
   - Random Rotation
   - Random Zoom
   - Random Contrast
3. Train Level-1 part classifier
4. Train separate Level-2 disease models
5. Fine-tune backbone
6. Save models in `.keras` format
7. Save metadata JSON for deployment

---
requirenments-
pip install tensorflow keras numpy scikit-learn matplotlib
pip install -q tensorflow
