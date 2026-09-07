# HemoVision: Contactless Vital Signs (rPPG) System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/Google-MediaPipe-00A67E.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Smart India Hackathon (SIH 2026) — Grand Finale Showcase**  
> *Theme: Student Innovation (Software / MedTech)*  
> **A quality-gated, contactless cardiovascular screening system estimating Heart Rate (BVP) from ordinary facial video in real time.**

---

## 📌 Project Overview

In remote telemedicine consultations across India (e.g., eSanjeevani), patients frequently connect without a single objective vital sign because they do not own dedicated medical hardware like pulse oximeters or ECG monitors.

**HemoVision** bridges this diagnostic gap by transforming any commodity webcam or smartphone camera into a non-contact physiological sensor:
- **Zero Hardware Procurement**: Extracts blood volume pulse (BVP) using subtle sub-pixel facial chromatic fluctuations.
- **Strict Quality Gating**: Automatically halts inference and rejects measurements during motion artifacts or poor lighting (*refuses to hallucinate vitals*).
- **Lightweight Deep Learning**: Runs smoothly on standard laptop CPUs without requiring expensive GPUs.

---

## 🔬 Scientific Methodology & Pipeline

```
Facial Video (Webcam / File)
          │
          ▼
MediaPipe FaceMesh (468 3D Landmarks)
          │
          ▼
Capillary Skin Segmentation (8 Anatomical ROIs)
 ├─ Forehead (Center, Left, Right)
 ├─ Upper Cheeks & Lower Cheeks
 └─ Nose Bridge & Chin
          │
          ▼
24-Channel Normalized RGB Time-Series
          │
          ▼
HemoVisionBoundedTCN (148K Parameters)
 └─ Dilated Causal Depthwise-Separable Convolutions + Tanh Bounding
          │
          ▼
Physiological Filtering & Spectral Estimation
 ├─ 4th-Order Butterworth Bandpass (0.65 Hz – 3.25 Hz / 39–195 BPM)
 └─ Welch Power Spectral Density (PSD) + Peak FFT
          │
          ▼
Real-Time Heart Rate Estimate (BPM) + Live Waveform
```

### Key Innovations:
1. **Bounded TCN Architecture**: Eliminates gradient instability and exploding predictions on noisy webcam feeds by employing bounded output activation.
2. **Multi-Region Optical Tracking**: Isolates high-perfusion micro-vascular skin beds while ignoring specular highlights, facial hair, and eye movement.
3. **Scientific Integrity**: Thoroughly investigated 10 secondary biomarkers (SpO₂, blood pressure, hemoglobin, etc.) and proved that single RGB streams cannot reliably predict them without dual-wavelength multi-spectral sensors.

---

## 📊 Benchmark Results

Trained and evaluated on the **MCD-rPPG** clinical dataset (~600 subjects across 3 camera types and pre/post exercise conditions):

| Scenario / Test Split | Mean Absolute Error (MAE) | Pearson Correlation ($r$) | Performance Note |
| :--- | :---: | :---: | :--- |
| **High-Quality Stable Clips** | **1.33 BPM** | **0.96** | Matches contact finger pulse oximeter precision |
| **Held-Out Test Set (91 Subjects)** | **9.19 BPM** | **0.84** | Robust generalization across unconstrained subjects |
| **Real-Time Latency** | **~30 FPS** | — | Real-time CPU inference via MediaPipe & lightweight TCN |

---

## 🚀 Quickstart & Installation

### Option A: 1-Click Launch (Recommended for Windows)

Simply double-click **`run_demo.bat`** or open PowerShell in the project directory and run:

```powershell
.\run_demo.ps1
```
The script will automatically configure a clean Python virtual environment, install the exact dependencies, and let you select the presentation dashboard or live demo.

---

### Option B: Manual Setup (Any Platform)

#### 1. Clone the repository
```bash
git clone https://github.com/Ayanmishra11/HemoVision.git
cd HemoVision
```

#### 2. Create and activate a Python virtual environment (Python 3.10 recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the applications

- **Interactive Presentation Dashboard (Streamlit UI)**:
  ```bash
  streamlit run app.py
  ```

- **Live Webcam HUD Demo (Real-Time Face Mesh)**:
  ```bash
  python hemovision_demo.py
  ```

- **Offline Test Split Benchmark**:
  ```bash
  python hemovision_demo.py --test
  ```

---

## 📁 Repository Structure

```
HemoVision/
├── app.py                          # Streamlit presentation web dashboard
├── hemovision_demo.py              # Real-time inference engine & OpenCV demo
├── run_demo.ps1                    # 1-Click PowerShell launcher for Windows
├── run_demo.bat                    # 1-Click Windows batch launcher
├── requirements.txt                # Exact pinned dependencies
├── .gitignore                      # Clean exclusion rules (filters 130GB videos)
│
├── data/vitalscan-clinic/MCD-rPPG/
│   ├── hemovision_bounded_tcn_best.pt  # Trained PyTorch checkpoint (1.1 MB)
│   ├── manifest.json                   # Dataset test catalogue metadata
│   ├── db.csv                          # Clinical ground truth table
│   ├── video/                          # Curated test sample clips for demonstration
│   └── ppg_sync/                       # Ground-truth PPG reference waveforms
│
├── grand_finale_demo_script.md     # 90-second live jury demo & 3-minute pitch script
├── judge_qna.md                    # Tough questions & evidence-backed answers
├── evidence_audit.md               # Empirical biomarker investigation report
├── HEMOVISION_SIH_2026_FINAL.pdf   # Official SIH 2026 Presentation Deck
└── sih_final_preview/              # Rendered preview slides for SIH Finale
```

---

## 👥 Hackathon Team & Acknowledgements

- **Developed for**: Smart India Hackathon (SIH 2026)
- **Problem Statement**: Contactless Vital Signs Monitoring for Telemedicine
- **Lead Developer**: [Ayan Mishra](https://github.com/Ayanmishra11)
- **Dataset Reference**: MCD-rPPG Clinical Dataset

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
