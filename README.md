# Breast Insight Fusion AI

**Author:** Pranab Chuahan  
**Last Modified:** 06/10/2025

---

## 🧠 Project Overview

**Breast Insight Fusion AI** is an AI-powered diagnostic system for **early breast cancer detection** that fuses **mammographic image data** with **clinical metadata**. This end-to-end trainable tool utilizes modern deep learning techniques to provide predictive insights that support radiologists and healthcare professionals in improving diagnostic accuracy and speed.

Inspired by multimodal learning approaches, it allows for flexible fusion at different stages in the neural architecture (early/mid/late fusion) and supports clinical deployment with real-time UI via **Streamlit**.

<p align="center">
    <img src="figs/model_architecture.png" height="400" alt="Fusion AI Architecture">
</p>

---

## 📌 Key Features

- ✅ Image + Clinical Data Fusion
- ✅ CNN + MLP Multimodal Architecture
- ✅ Real-Time Results with Streamlit
- ✅ Custom Dataset Integration
- ✅ Training, Evaluation, and Visualization Tools

---

## 🗂️ Directory Structure

breast-insight-fusion-ai/
├── app.py # Streamlit web interface
├── train_model.py # Training script
├── models.py # Model definitions
├── utils/ # Preprocessing, metrics, loaders
├── data/ # Input data directory
├── results/ # Output logs, model weights
├── figs/ # Architecture diagrams, visualizations
├── requirements.txt # Project dependencies
└── README.md # Project documentation


pip install -r requirements.txt
python train_model.py --data_dir ./data \
                      --out_dir ./results \
                      --model learned-feature-fusion \
                      --fusion_mode concat \
                      --n_TTA 5 \
                      --augment \
                      --use_class_weights \
                      --label_smoothing 0.1
