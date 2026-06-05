# Waste Classifier

A production-grade AI/ML system that classifies household waste images into one of **10 categories** using deep learning. Built end-to-end — from model training to cloud deployment — as a full-stack ML engineering project.

**Live demo:** https://waste-classifier-ui-943543256910.europe-central2.run.app
> ℹ️ The service uses cold-start Cloud Run instances. The first request may take ~30 seconds; subsequent requests are fast. On the mobile devices it is recommended to add an image from the gallery rather than making a photo online due to known Streamlit limitations.

<img width="1920" height="1632" alt="Application screenshot" src="https://github.com/user-attachments/assets/83c523a2-91cd-49f9-b47c-e7b7f4c9468c" />

---

## Overview

The system accepts a photo of a waste item and returns a predicted category:

| | | | | |
|---|---|---|---|---|
| 🔋 Battery | 🧫 Biological | 📦 Cardboard | 👕 Clothes | 🍶 Glass |
| 🔩 Metal | 📄 Paper | 🧴 Plastic | 👟 Shoes | 🗑️ Trash |

---

## Architecture

The project consists of two independently deployed components:

User → Streamlit Frontend (Cloud Run)

↓ HTTP

FastAPI REST API (Cloud Run)  → MobileNetV2 custom Classifier (served in-process)


### Model

The classifier is a fine-tuned **MobileNetV2** (transfer learning, two-phase training) trained on ~20,000 images sourced from multiple Kaggle datasets, spanning 10 waste categories.

- **Training**: Two-phase fine-tuning with aggressive data augmentation, class-weighted loss, and best-checkpoint saving based on validation loss. Due to the strong augmentation the train and validation curves didn't cross.
- **Test accuracy**: 92.69%
- **Hardware**: CPU-only training (~10 hours on a consumer laptop)

<img width="1950" height="750" alt="Training curves" src="https://github.com/user-attachments/assets/3f1929bf-0763-400c-afb8-810f789fd02b" />

<img width="1500" height="1200" alt="Confusion matrix" src="https://github.com/user-attachments/assets/c5c29d8e-f7b3-446f-bf50-98f34d9df520" />

### REST API

A **FastAPI** application containerized with Docker, deployed on **Google Cloud Run**. It exposes a `/predict` endpoint that accepts an image file and returns the predicted class with confidence scores. Key engineering decisions:

- File size validation and structured error handling
- MobileNetV2-standard preprocessing pipeline consistent with training
- Async lifecycle management for model loading
- Configurable CORS for frontend integration

### Frontend

A **Streamlit** application deployed on Cloud Run. Users upload a photo and receive the classification result with confidence breakdown. Simple by design — the complexity lives in the backend.

### Infrastructure

All cloud resources are managed with **Terraform** (Infrastructure as Code):

- Google Cloud Run (API + frontend services)
- Artifact Registry (Docker image storage)
- IAM configuration

---

## Tech Stack

| Layer | Technologies |
|---|---|
| ML | Python, TensorFlow 2.x / Keras, MobileNetV2, scikit-learn |
| API | FastAPI, Uvicorn, Docker |
| Frontend | Streamlit |
| Cloud | Google Cloud Run, Artifact Registry |
| IaC | Terraform |

---

## Documentation

Detailed technical documentation covering the model architecture, training, API design, and deployment setup is available in the [`/Documentation`](./Documentation) directory.
