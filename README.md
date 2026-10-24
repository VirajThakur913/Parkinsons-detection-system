# NeuroSense AI — Multimodal Decision Support System for Parkinson's Disease

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**NeuroSense AI** is an advanced, non-invasive **Clinical Decision Support System (CDSS)** for early-stage screening of Parkinson's Disease (PD). It integrates **Vocal Acoustic Analysis** and **Motor Kinematic Handwriting Analysis** into a weighted late-fusion decision pipeline, complemented by **Local Explainable AI (SHAP)** and automated **Clinical PDF Report Generation**.

---

## 📌 Key Features

* **🎙️ Vocal Acoustic Dysphonia Analysis:**
  * Real-time microphone audio capture via browser **Web Audio API** (16kHz Float32 PCM conversion).
  * Feature extraction: Fundamental frequency ($F_0$), Jitter (frequency perturbation), Shimmer (amplitude perturbation), RMS energy, and Zero-Crossing Rate.
  * XGBoost classification over 25 RFE-selected speech features (MFCCs, TQWT Wavelets, Recurrence Period Density Entropy).
  * Interactive **3-Sigma Acoustic Sliders** ($\mu \pm 3\sigma$) bounded by empirical training distributions.

* **✍️ Motor Kinematic Drawing Analysis:**
  * Digital HTML5 whiteboard canvas supporting mouse/stylus drawing.
  * **Drawing Photo Upload & Drag-and-Drop:** Adaptive Otsu thresholding automatically detects background polarity (paper photo vs. digital canvas).
  * **Zhang-Suen Morphological Skeletonization:** Strips variable stroke thickness down to a 1-pixel skeleton to isolate spatial tremors.
  * **Histogram of Oriented Gradients (HOG):** 252-dimensional structural descriptor feeding an XGBoost classifier.

* **🧠 Weighted Late-Fusion Decision Engine:**
  * Merges speech and motor predictions using weighted late fusion:
    $$\text{Final Risk Score} = (0.4 \times P_{\text{voice}}) + (0.6 \times P_{\text{motor}})$$
  * Stratifies risk: **Low Risk** (< 35%), **Moderate Risk** (35%–65%), **High Risk** ($\ge$ 65%).

* **🔍 Explainable AI (XAI):**
  * Local feature attribution approximation (SHAP) visualizing how each acoustic feature pushes the patient towards or away from disease risk.

* **📄 Automated Medical PDF Report Generation:**
  * Compiles multi-page clinical reports via **ReportLab Platypus** with patient demographics, score gauges, original vs. thinned drawings, SHAP plots, and clinician sign-off blocks.

* **🔐 Role-Based Access Control & Audit Trails:**
  * SQLite database (`neurosense.db`) with SHA-256 hashed credentials.
  * Dual views: **Administrator / Clinician Dashboard** and **Patient Longitudinal Progress Portal** (with interactive Chart.js trend tracking).

---

## 🏗️ System Architecture

```
[ Patient Voice / Upload ] ──► Web Audio API (16kHz PCM WAV) ──► signal_processing.py ──► XGBoost Speech Model (P_voice)
                                                                                                │
[ Drawing / Photo Upload ] ──► Otsu Threshold ──► Zhang-Suen Thinning ──► HOG ──────► XGBoost Motor Model (P_motor)
                                                                                                │
                                                                                                ▼
                                                                           Late Fusion (0.4 Voice + 0.6 Motor)
                                                                                                │
                                                                       ┌────────────────────────┴────────────────────────┐
                                                                       ▼                                                 ▼
                                                               SQLite Database                                   ReportLab Engine
                                                           (Audit Logs & History)                               (Clinical PDF Report)
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10 to 3.14 installed
* Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/neuro-sense-ai.git
cd neuro-sense-ai
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔑 Default Credentials

| Portal | Username | Password | Role |
| :--- | :--- | :--- | :--- |
| **Clinician / Admin Portal** | `admin` | `admin123` | Full clinical screening, patient registry, audit logs |
| **Patient Portal** | `patient01` | `patient123` | Longitudinal progress chart, test history & report downloads |

---

## 📊 Dataset & Methodologies

* **Speech Dataset:** Oxford / Istanbul University Parkinson's Speech Dataset (756 samples, 754 features, sustained `/a/` vowel).
* **Drawing Dataset:** PaHaW / Kaggle Spiral and Wave Drawing Dataset.
* **Techniques Used:**
  * Group-aware patient-level splitting (avoids identity data leakage).
  * KMeans-SMOTE minority balancing.
  * Recursive Feature Elimination (RFE) with Logistic Regression.
  * Adaptive Otsu binarization and Zhang-Suen morphological thinning.
  * Histogram of Oriented Gradients (HOG) extraction.
  * Second-order Taylor expansion tree gradient boosting (XGBoost).
  * Model-agnostic perturbation attribution (SHAP).

---

## 📁 Repository Structure

```
├── app.py                 # Flask application controller & REST endpoints
├── ml_speech.py           # Vocal ML pipeline (RFE, SMOTE, XGBoost, SHAP)
├── ml_drawings.py         # Motor CV pipeline (Otsu, Zhang-Suen, HOG, XGBoost)
├── signal_processing.py   # Acoustic signal processor (F0, Jitter, Shimmer, RMS)
├── database.py            # SQLite schema, SHA-256 user auth, and audit logging
├── report_generator.py    # ReportLab PDF clinical report compiler
├── requirements.txt       # Python package dependencies
├── models/                # Trained model pipelines (.pkl)
├── data/                  # Speech dataset CSV
├── dataset/               # Spiral & wave drawing datasets
├── static/                # CSS, client-side JS, sample drawings
└── templates/             # Dashboard HTML templates
```

---

## 📜 License
This project is licensed under the MIT License.
