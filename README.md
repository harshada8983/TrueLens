# 🔍 TrueLens: AI-Powered Deepfake Detection Engine

[![Live Demo](https://img.shields.io/badge/Live_Demo-Online-success?style=for-the-badge)](https://true-lens-seven.vercel.app)
[![Tech Stack](https://img.shields.io/badge/Stack-React%20%7C%20FastAPI%20%7C%20TensorFlow-blue?style=for-the-badge)](#tech-stack)

TrueLens is a full-stack, hybrid-cloud Machine Learning application designed to detect manipulated facial imagery (deepfakes) with high precision. By combining a lightweight React frontend with a robust FastAPI backend, TrueLens processes images through a custom, context-aware neural pipeline.

## 🌐 Live Demo
**Test the application live:** [TrueLens Web Interface](https://true-lens-seven.vercel.app)

> **Note on Architecture:** To handle heavy TensorFlow memory requirements without exorbitant cloud hosting costs, this project utilizes a **Hybrid-Cloud Deployment Strategy**. The Vercel-hosted frontend communicates securely with the local AI engine via an active `Ngrok` tunnel.

---

## 🧠 The Inference Pipeline

TrueLens does not simply guess based on raw images. It processes every upload through a strict, 5-stage inference engine:

1. **High-Res Speed Optimization:** Downscales massive image payloads dynamically via OpenCV to ensure rapid scanning without losing critical pixel data.
2. **Context-Aware Extraction (MTCNN):** Utilizes Multi-task Cascaded Convolutional Networks to scan the image, isolate the most prominent human face, crop the bounding box, and discard irrelevant background noise. It includes a fallback shield to prevent crashes on non-human images.
3. **Mathematical Calibration:** Normalizes color channels (BGR to RGB) and scales the extracted face matrix into a precise 224x224 tensor.
4. **Neural Inference:** Injects the calibrated tensor into a trained **EfficientNetB0** deep learning model.
5. **Forensic Verdict:** Translates the raw sigmoid output into a definitive `REAL` or `MANIPULATED` classification, alongside a calculated confidence percentage.

---

## 🛠 Tech Stack

**Frontend (Client)**
* React (Vite)
* Axios for API state management
* Hosted globally on **Vercel**

**Backend (API & Tunnel)**
* Python 3.10+
* **FastAPI** (Configured with synchronous routing to protect ML thread safety)
* Uvicorn
* **Ngrok** (Secure TCP/HTTP tunneling)

**Machine Learning (Engine)**
* **TensorFlow / Keras**
* **EfficientNetB0** (Transfer learning base)
* **OpenCV** (High-speed tensor matrix manipulation)
* **MTCNN** (Facial isolation and alignment)

# #🎥 Video Deepfake Detection
In addition to single-image analysis, TrueLens now supports video upload and analysis. When a video is submitted:

The backend samples up to 12 frames evenly spread across the clip.
Each sampled frame goes through the same face-extraction (MTCNN) and EfficientNet/MobileNet-based classification pipeline used for images.
Frame-level predictions are aggregated (averaged) into a single overall verdict — along with a breakdown of how many frames were analyzed, how many were skipped (no face detected), and what fraction were flagged as manipulated.

This makes TrueLens usable for both static images and short video clips, without needing a separate model or pipeline.

* Usage: Simply toggle between "Image" and "Video" mode on the upload screen before selecting a file.

---

## 🚀 Local Setup & Installation

If you wish to run the full pipeline locally for development or evaluation:

### 1. Clone the Repository
```bash
git clone https://github.com/harshada8983/TrueLens.git
cd TrueLens
```
