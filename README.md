# 🔍 TrueLens: AI-Powered Deepfake Detection Engine

[![Live Demo](https://img.shields.io/badge/Live_Demo-Online-success?style=for-the-badge)](https://truelens-ai-nine.vercel.app)
[![Tech Stack](https://img.shields.io/badge/Stack-React%20%7C%20FastAPI%20%7C%20TensorFlow-blue?style=for-the-badge)](#tech-stack)

TrueLens is a full-stack, hybrid-cloud Machine Learning application designed to detect manipulated facial imagery (deepfakes) with high precision. By combining a lightweight React frontend with a robust FastAPI backend, TrueLens processes images through a custom, context-aware neural pipeline.

## 🌐 Live Demo
**Test the application live:** [TrueLens Web Interface](https://truelens-ai-nine.vercel.app)

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

---

## 🚀 Local Setup & Installation

If you wish to run the full pipeline locally for development or evaluation:

### 1. Clone the Repository
```bash
git clone [https://github.com/harshada8983/DeepFake-Detection-System.git](https://github.com/harshada8983/DeepFake-Detection-System.git)
cd TrueLens
