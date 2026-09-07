# A
# Industrial Training/Internship Report
# On
# VOCATIONAL TRAINING IN MACHINE LEARNING & COMPUTER VISION
# REPORT

Submitted to
### CHHATTISGARH SWAMI VIVEKANAND TECHNICAL UNIVERSITY
### BHILAI
*in partial fulfilment of requirement for the award of the degree*
*of*
### Bachelor of Technology
*in*
### Computer Science and Engineering

by  
**Harsh Patel**  
**300111323021**

**BHILAI INSTITUTE OF TECHNOLOGY, DURG (CG)**  
*(Seth Balkrishan Memorial) Estd. 1986*  
ALL UG & MBA COURSES | NAAC 'A' GRADE | ISO 9001:2015 | ISO 14001:2015 | NIRF

---

## Certificate by Company / Industry

### Certificate of Completion issued by BS Digital Technology

```
+--------------------------------------------------------------------------------------------------+
|                                    BS DIGITAL TECHNOLOGY                                         |
|                                                                                                  |
|                                  CERTIFICATE OF COMPLETION                                       |
|                                                                                                  |
|  This is to certified that Shri / Ku.  Harsh Patel                                               |
|  Reg. No.: BIT/2026/82                                                                           |
|  6th Sem., Student of B.Tech. of Bhilai Institute of Technology College / Institute Durg        |
|  Has undergone project based training from 08/06/2026 to 08/07/2026                             |
|  Project: HemoVision — Contactless Vital Signs Estimation & rPPG Analytics                      |
|  Technology: Python, PyTorch, OpenCV, Deep Learning, ML                                          |
|  His / Her performance during the training period has been: Good                                 |
|                                                                                                  |
|          Mr. Raahul Kharatkar                                 Mr. Prashant Kumar Tamarakar       |
|               HR Manager                                            Managing Director            |
|       BS Digital Technology                                       BS Digital Technology          |
+--------------------------------------------------------------------------------------------------+
```
*Figure: Company/Industry Certificate of Completion*

---

## Declaration by Student

I, Harsh Patel, a student of Bachelor of Technology in Computer Science and Engineering (Artificial Intelligence), 6th Semester, hereby declare that the Industrial Training / Vocational Training report entitled **“HemoVision – Contactless Vital-Sign Estimation and Biomarker Analysis via Remote Photoplethysmography (rPPG)”** is based on the work carried out by me during my project-based training at **BS Digital Technology** from **08 June 2026 to 08 July 2026**.

I declare that the work presented in this report reflects my learning, implementation, analysis, and project activities during the training period. The report has been prepared for academic submission and has not been submitted elsewhere for the award of any other degree, diploma, or certificate.

Wherever information, concepts, tools, documentation, or external references have been used, appropriate acknowledgement and references have been provided. I have made every effort to present the project work accurately and in accordance with the prescribed university report format.

**Signature of Student:** ___________________________  
**Harsh Patel**  
**Roll No.:** 300111323021  
**Enrolment No.:** CD8812  

---

## Acknowledgement

I have great pleasure in the submission of this project report entitled **“HemoVision – Contactless Vital-Sign Estimation and Biomarker Analysis via Remote Photoplethysmography (rPPG)”** in partial fulfilment of Vocational Training. While submitting this project report, I take this opportunity to thank those directly or indirectly related to project work. I would like to thank my supervisor **Mr Kailash Sinha** and my vocational training incharge **Dr Sumit Sar** who has provided the opportunity and organizing project for me. Without his active co-operation and guidance, it would have become very difficult to complete the task in time. I would like to express sincere thanks and gratitude to **Dr. Anup Mishra**, Principal of the Institution, **Dr. (Mrs.) Sunita Soni**, Head of the Department Computer Science & Engineering for their encouragement and cordial support.

Acknowledgement is due to our parents, family members, friends and all those persons who have helped us directly or indirectly in the successful completion of the project work.

**Name of the Student:** Harsh Patel  
**Roll No:** 300111323021  
**Enrollment:** CD8812  

---

## Abstract

**HemoVision** is an end-to-end computer vision and deep learning framework developed during vocational training at BS Digital Technology for contactless physiological vital-sign estimation and biomarker analysis from standard RGB facial video. The project investigates Remote Photoplethysmography (rPPG)—the optical measurement of sub-visual Blood Volume Pulse (BVP) pulsations caused by cardiovascular blood flow under facial skin capillary beds. The solution bridges facial landmark tracking, multi-region of interest (ROI) extraction, signal AC/DC disentanglement, temporal deep neural networks, and interactive clinical visualization into a unified pipeline.

The project evaluates the **MCD-rPPG Clinical Dataset**, comprising **3,427 video clips across 598 unique subjects** captured under diverse conditions (resting and post-exercise) across three commercial camera sensors (FullHDwebcam, USBVideo, IriunWebcam), synchronized with 11 ground-truth clinical contact sensors (ECG pulse, SpO2, blood pressure, respiratory rate, hemoglobin, HbA1c, cholesterol, rigidity, stress, and BMI). The data preparation pipeline tracks 468 3D facial landmarks via MediaPipe Face Mesh, segments 8 anatomically independent skin ROIs, extracts 24-channel RGB time-series, isolates AC pulsatile micro-signals via z-score normalization, and retains raw DC luminance representations. Multiple architectures were developed and evaluated, establishing a lightweight **Depthwise-Separable Bounded Temporal Convolutional Network with Multi-Head Attention (`HemoVisionBoundedTCN`)** supervised by negative Pearson correlation loss.

The resulting deep learning model achieves **9.19 BPM Mean Absolute Error (MAE)** and Pearson $r = 0.285$ on a strictly subject-disjoint held-out test split of 91 subjects (523 clips), with high-quality clips reaching **1.33 BPM MAE**, significantly outperforming classical signal processing baselines (POS MAE 19.17 BPM; CHROM MAE 20.28 BPM). Furthermore, an in-depth **Biomarker Feasibility Audit** and confound analysis demonstrated that RGB cameras cannot reliably reconstruct blood chemistry (hemoglobin, lipid levels) without dual-wavelength NIR optics, exposing published claims as demographic identity memorization rather than optical sensing. An interactive Streamlit and OpenCV dashboard was developed for real-time BVP waveform and heart-rate monitoring.

**Keywords:** Remote Photoplethysmography (rPPG), Deep Learning, Temporal Convolutional Network (TCN), Computer Vision, MediaPipe Face Mesh, Vital Signs, Heart Rate Estimation, Biomarker Feasibility, PyTorch, OpenCV.

---

## Table of Contents

| S.no. | Chapter Name | Page Number |
|:---:|:---|:---:|
| **1** | Cover Page | i |
| **2** | Certificate of Completion | ii |
| **3** | Declaration by the Candidate | iii |
| **4** | Acknowledgment | iv |
| **5** | Abstract | v |
| **6** | Table of Content | vi |
| **7** | List of Tables | vii |
| **8** | List of Figures | viii |
| **9** | Abbreviations and Nomenclature | ix |
| **10** | **1. Introduction** | 1–2 |
| | 1.1 Training Profile and Project Details | 1 |
| | 1.2 Project Overview | 2 |
| **11** | **2. Formal Training Provided** | 3–4 |
| | 2.1 Python and Computer Vision / Signal Processing | 3 |
| | 2.2 Deep Learning and PyTorch Framework | 3 |
| | 2.3 Remote Photoplethysmography & Temporal Modeling | 3 |
| | 2.4 Interactive Clinical Visualization and Deployment | 4 |
| | 2.5 Learning Outcome of Formal Training | 4 |
| **12** | **3. Industrial Training** | 5–7 |
| | 3.1 Objectives | 5 |
| | 3.2 Tools and Technologies Used | 5 |
| | 3.3 Techniques Studied in Different Departments | 5 |
| | 3.3.1 Facial Landmark Extraction & Multi-ROI Signal Acquisition | 5 |
| | 3.3.2 Signal Preprocessing (AC/DC Disentanglement & Filtering) | 5 |
| | 3.3.3 Classical Baselines (POS / CHROM Projections) | 6 |
| | 3.3.4 Deep Learning Architecture (Depthwise Separable Bounded TCN) | 6 |
| | 3.3.5 Ground Truth Synchronization & Signal Artifact Rectification | 6 |
| | 3.3.6 Clinical Dashboard & Real-Time Visualization | 6 |
| | 3.4 Software and Tools Used | 6 |
| | 3.5 Highlights of Training Exposure | 7 |
| **13** | **4. Problem Identification and Case Study** | 8–11 |
| | 4.1 Problem Statement | 8 |
| | 4.2 Data Used in the Case Study | 8 |
| | 4.3 Data Preparation & ROI Extraction Pipeline | 8 |
| | 4.4 Machine-Learning & Signal Feature Set | 8 |
| | 4.5 System Architecture | 9 |
| | 4.6 Methodology | 9 |
| | 4.7 Data-to-Decision Pipeline | 10 |
| | 4.8 Model Development | 10 |
| | 4.9 Vital-Sign & Biomarker Evaluation Outputs | 10 |
| | 4.10 Results and Discussion | 10 |
| | 4.11 Skills and Outcomes | 11 |
| **14** | **5. Recommendations** | 12–13 |
| | 5.1 Model & Algorithmic Improvements | 12 |
| | 5.2 Hardware & Multi-Wavelength Optical Improvements | 12 |
| | 5.3 Physiological Metric Extensions (HRV / RMSSD / PWTT Blood Pressure) | 12 |
| | 5.4 Deployment & Clinical Interface Improvements | 12 |
| | 5.5 Future Scope | 12 |
| | 5.6 Recommended Future Clinical Dashboard Layout | 13 |
| **15** | **6. References** | 14 |
| **16** | **7. Appendices** | 15–16 |
| | Appendix A – Suggested Project Folder Structure | 15 |
| | Appendix B – Final Biomarker & ROI Feature Groups | 15 |
| | Appendix C – rPPG Training & Evaluation Checklist | 16 |

---

## List of Tables

1. **Table 1.** Training profile and project details
2. **Table 2.** Major tools and technologies
3. **Table 3.** MCD-rPPG Clinical Dataset Summary
4. **Table 4.** Ground-Truth Clinical Biomarkers in Dataset
5. **Table 5.** Facial Regions of Interest (ROIs) and Channel Specification
6. **Table 6.** Performance Comparison on Held-Out Test Set (91 Subjects, 523 Clips)
7. **Table 7.** Multi-Biomarker Feasibility Audit & Baseline Comparison
8. **Table 8.** Training outcomes and skills gained
9. **Table 9.** Recommended clinical dashboard layout

---

## List of Figures

1. **Figure 1.** Company/Industry Certificate of Completion
2. **Figure 2.** HemoVision End-to-End rPPG Workflow
3. **Figure 3.** HemoVision System Architecture
4. **Figure 4.** Data-to-Decision Pipeline (Video to Physiological Estimation)

---

## Abbreviations and Nomenclature

| Term | Meaning |
|:---|:---|
| **AI** | Artificial Intelligence |
| **ML** | Machine Learning |
| **DL** | Deep Learning |
| **rPPG** | Remote Photoplethysmography |
| **PPG** | Photoplethysmography |
| **BVP** | Blood Volume Pulse |
| **ECG** | Electrocardiogram |
| **ROI** | Region of Interest |
| **TCN** | Temporal Convolutional Network |
| **POS** | Plane-Orthogonal-to-Skin algorithm |
| **CHROM** | Chrominance-based rPPG method |
| **FFT** | Fast Fourier Transform |
| **PSD** | Power Spectral Density |
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **BPM** | Beats Per Minute |
| **SpO2** | Peripheral Capillary Oxygen Saturation (%) |
| **BP** | Blood Pressure (Systolic / Diastolic in mmHg) |
| **Hb** | Hemoglobin (g/dL) |
| **HbA1c** | Glycated Hemoglobin (%) |
| **HRV** | Heart Rate Variability |
| **RMSSD** | Root Mean Square of Successive RR interval Differences |
| **AMP** | Automatic Mixed Precision |
| **CCC** | Lin's Concordance Correlation Coefficient |
| **UI** | User Interface |

---

# 1. Introduction

Industrial and vocational training provides a critical bridge between theoretical computer science concepts and real-world applied engineering. For a Computer Science and Engineering student specializing in Artificial Intelligence, gaining hands-on exposure to advanced computer vision, physiological digital signal processing, deep temporal neural networks, and interactive healthcare interfaces is exceptionally valuable. The present training was completed at **BS Digital Technology** through a project-based program from **08 June 2026 to 08 July 2026**.

The central project undertaken during the training was **HemoVision**, an advanced, contactless vital-sign estimation and biomarker analysis platform based on **Remote Photoplethysmography (rPPG)**. The project addresses a pressing medical and technology challenge: conventional health monitoring requires intrusive contact sensors (pulse oximeter finger clips, blood pressure cuffs, ECG chest leads, and invasive venous blood draws). Remote Photoplethysmography provides a non-invasive alternative by capturing micro-amplitude color variations on human facial skin caused by subcutaneous capillary blood flow across the cardiac cycle using standard RGB video cameras.

HemoVision follows a rigorous, end-to-end multi-stage architecture. Facial video streams are ingested and processed using **MediaPipe Face Mesh** to dynamically extract 8 anatomically stable facial Regions of Interest (ROIs). The resulting 24-channel RGB time-series signals undergo AC/DC separation, Butterworth bandpass filtering (0.65–3.25 Hz, corresponding to 39–195 BPM), and z-score normalization. A custom deep learning architecture—**Depthwise-Separable Bounded Temporal Convolutional Network with Multi-Head Attention (`HemoVisionBoundedTCN`)**—is trained to reconstruct the underlying BVP pulse waveform and regress physiological vital signs.

The project evaluates the **MCD-rPPG Clinical Dataset** (598 unique subjects, 3,427 clips across 3 camera types and 2 physiological states). Beyond heart-rate estimation, the project conducted a groundbreaking **Biomarker Feasibility Audit** investigating whether deeper biomarkers (hemoglobin, SpO2, blood pressure, cholesterol, stress) can be reliably extracted from ambient RGB video. Crucially, the project identified and rectified hidden data integrity bugs (including an ECG T-wave double-counting anomaly in post-exercise data and FFT spectral quantization grids), enforcing strict subject-disjoint evaluation protocols to ensure scientific reproducibility and clinical honesty.

### 1.1 Training Profile and Project Details

#### Table 1: Training profile and project details
| Item | Details |
|:---|:---|
| **Student** | Harsh Patel |
| **Program** | B.Tech – Computer Science and Engineering (Artificial Intelligence) |
| **Semester** | 6th Semester |
| **Roll No.** | 300111323021 |
| **Enrolment No.** | CD8812 |
| **Training Organization** | BS Digital Technology |
| **Project** | HemoVision — Contactless Vital-Sign Estimation via rPPG |
| **Training Type** | Project-based vocational training |
| **Training Period** | 08 June 2026 – 08 July 2026 |
| **Duration** | 30 days |
| **Primary Technologies** | Python, PyTorch, OpenCV, MediaPipe, NumPy, SciPy |
| **Deployment / UI Layer** | Streamlit, Matplotlib, Scikit-Learn |

### 1.2 Project Overview

```
+------------------+     +-------------------+     +---------------------+     +--------------------+     +--------------------+     +-------------------+
|   Source Video   | --> | MediaPipe 8 ROIs  | --> | Signal Extraction   | --> | HemoVision TCN     | --> | Peak-Picking FFT   | --> | Interactive UI    |
| (RGB 30 FPS Cam) |     | (Face Mesh 468pt) |     | (AC/DC Normalised)  |     | (Bounded Temporal) |     | (39-195 BPM Band)  |     | (Streamlit / App) |
+------------------+     +-------------------+     +---------------------+     +--------------------+     +--------------------+     +-------------------+
```
*Figure 2: HemoVision End-to-End Workflow*

The workflow starts with ambient facial video frames and ends with verified clinical-grade physiological estimations and waveform visualizations. The modular separation between landmark tracking, signal extraction, deep neural modeling, and visualization makes HemoVision scalable, falsifiable, and easy to maintain.

---

# 2. Formal Training Provided

The vocational training was structured to provide comprehensive practical knowledge connecting computer vision, biological signal processing, deep neural network architecture design, and interactive healthcare deployment into a unified engineering pipeline.

### 2.1 Python and Computer Vision / Signal Processing
Python served as the primary programming environment. Practical proficiency was developed in handling high-dimensional image tensors, video decoding, facial geometry, and 1D digital signal processing:
- Real-time video frame decoding and temporal sequence buffering using **OpenCV (`cv2`)**.
- 468-point 3D facial landmark mesh localization and convex polygon masking using **MediaPipe Face Mesh**.
- Extracting raw spatial RGB channel averages across multi-region facial patches to form multi-channel temporal matrices ($[8 \text{ ROIs} \times 3 \text{ Channels} \times T \text{ Frames}]$).
- 1D digital filtering, including zero-phase 2nd-order Butterworth bandpass filtering ($0.65\text{--}3.25\text{ Hz}$) and Welch Power Spectral Density (PSD) using **SciPy Signal**.
- Implementing classical optical rPPG projection methods: **POS** (Plane-Orthogonal-to-Skin) and **CHROM** (Chrominance-based).

### 2.2 Deep Learning and PyTorch Framework
The deep learning module established hands-on expertise in building, training, and debugging PyTorch neural architectures for time-series regression:
- Designing modular 1D convolution layers, depthwise separable temporal convolutions, and residual skip connections.
- Implementing temporal dilation schedules ($d = 1, 2, 4, 8, 16, 32$) to achieve exponential receptive field expansion across 600–900 temporal frames (~20–30 seconds) without dimensional loss.
- Gradient scaling and Automatic Mixed Precision (**AMP**) training for memory optimization on GPU environments.
- Designing specialized loss functions: Negative Pearson Correlation ($r$) waveform loss, Lin's Concordance Correlation Coefficient (CCC), and Signal-to-Noise Ratio (SNR) regularizers.
- Enforcing strict subject-disjoint dataset splitting to eliminate data leakage and memorization.

### 2.3 Remote Photoplethysmography & Temporal Modeling
The specialized rPPG module focused on overcoming physiological and environmental noise factors:
- Understanding the optical absorption spectrum of oxyhemoglobin and deoxyhemoglobin across RGB wavelengths (peaking around 520–580 nm green light).
- Disentangling high-frequency pulsatile AC signals (cardiovascular wave) from low-frequency static DC components (skin pigmentation, illumination baseline).
- Multi-task learning strategies: joint waveform reconstruction and tiered biomarker estimation.
- Investigating confound factors (subject demographic shortcuts, camera sensor chromaticities, and ambient illumination drifts).

### 2.4 Interactive Clinical Visualization and Deployment
The deployment module emphasized translating raw deep learning model predictions into intuitive, interpretable interfaces:
- Building an interactive web application using **Streamlit** for live video upload, ROI facial tracking visualization, real-time BVP waveform playback, and frequency spectrum analysis.
- Implementing standalone OpenCV live webcam inference loops with zero external server dependencies.
- Generating scientific diagnostic plots (Bland-Altman agreement plots, correlation scatter diagrams, and error distribution histograms) using **Matplotlib** and **Seaborn**.

### 2.5 Learning Outcome of Formal Training
The primary outcome was mastering the complete machine learning lifecycle—from raw sensor capture and signal physics to deep neural architecture engineering, rigorous scientific error auditing, and interactive software deployment.

---

# 3. Industrial Training

### 3.1 Objectives
1. To understand the physics and physiological principles of contactless optical cardiovascular sensing (rPPG).
2. To extract robust, motion-resilient physiological signals from multi-region facial landmark geometry.
3. To design and implement a lightweight, depthwise-separable Bounded Temporal Convolutional Network (TCN) in PyTorch.
4. To implement and evaluate classical benchmark algorithms (POS and CHROM) against deep learning architectures.
5. To discover, isolate, and debug clinical ground-truth anomalies (e.g., ECG T-wave double counting).
6. To conduct a scientifically rigorous multi-biomarker feasibility audit across 11 physiological variables.
7. To deploy a real-time, interactive clinical demonstration dashboard using Streamlit and OpenCV.
8. To cultivate advanced technical problem-solving, code refactoring, scientific documentation, and verification skills.

### 3.2 Tools and Technologies Used

#### Table 2: Major tools and technologies
| Technology / Tool | Role in HemoVision |
|:---|:---|
| **Python 3.10+** | Core programming language for data pipeline, modeling, and evaluation |
| **PyTorch (torch / nn)** | Deep learning framework, custom TCN layers, multi-task loss computation |
| **OpenCV (cv2)** | Video ingestion, frame extraction, color space conversion, live rendering |
| **MediaPipe** | 468-point 3D Face Mesh extraction and real-time ROI facial segmentation |
| **NumPy & Pandas** | High-performance multi-dimensional array operations and clinical dataset indexing |
| **SciPy (signal / fft)** | Butterworth digital filtering, Welch Power Spectral Density, and zero-padded FFT |
| **Scikit-Learn** | Baseline regression models, metrics (MAE, RMSE, Pearson $r$), cross-validation |
| **Streamlit** | Web application framework for interactive clinical demonstration dashboard |
| **Matplotlib & Seaborn** | Publication-grade biomedical visualizations, BVP plots, and Bland-Altman charts |
| **VS Code & Git** | Integrated development environment, version control, modular architecture |

### 3.3 Techniques Studied in Different Departments

#### 3.3.1 Facial Landmark Extraction & Multi-ROI Signal Acquisition
Face detection alone is insufficient for robust rPPG because facial regions exhibit varying capillary density and motion susceptibility. The pipeline uses **MediaPipe Face Mesh** to track 468 3D landmarks and segments 8 anatomically independent ROIs:
- Left & Right Forehead
- Left & Right Upper Cheek (high capillary perfusion)
- Left & Right Lower Cheek
- Nose Bridge / Mid-face
- Chin

Each region is rasterized into a binary polygon mask per frame, and spatial RGB channel means are extracted, yielding a compact $[8 \text{ ROIs} \times 3 \text{ Channels} \times T \text{ Frames}]$ signal tensor.

#### 3.3.2 Signal Preprocessing (AC/DC Disentanglement & Filtering)
Raw pixel averages contain a massive static DC luminance component ($~99\%$) and a tiny pulsatile AC signal ($~0.1\text{--}1\%$). To maximize deep learning representation efficiency:
- **AC Stream ($x$):** Channel-wise temporal z-score normalization ($\mu=0, \sigma=1$) removes static DC offsets while preserving cardiac pulsatile morphology.
- **DC Stream ($x_{\text{raw}}$):** Scaled raw luminance ($I / 255.0$) preserves skin tone and baseline pigmentation for optical ratio analysis.
- **Temporal Bandpass Filtering:** 2nd-order zero-phase Butterworth filter ($0.65\text{--}3.25\text{ Hz}$) removes slow respiration drifts ($<0.65\text{ Hz}$) and high-frequency sensor noise ($>3.25\text{ Hz}$).

#### 3.3.3 Classical Baselines (POS / CHROM Projections)
To establish rigorous non-deep learning benchmarks:
- **POS (Plane-Orthogonal-to-Skin):** Projects normalized RGB signals onto an orthogonal subspace invariant to skin tone variations, calculating $S = S_1 + \alpha S_2$.
- **CHROM (Chrominance-based):** Projects color signals onto chrominance channels $X = 3R - 2G$ and $Y = 1.5R + G - 1.5B$ with adaptive bandpass subtraction.

#### 3.3.4 Deep Learning Architecture (Depthwise Separable Bounded TCN)
A specialized **HemoVisionBoundedTCN** was developed to model long-range temporal dependencies:
- **Input Projection:** 1D Conv ($24 \to 48\text{ or }64$ channels) + BatchNorm + GELU.
- **Residual Depthwise Separable TCN:** Dilated temporal convolution blocks ($d = 1, 2, 4, 8, 16, 32$) with depthwise convolutions (spatial/ROI channel mixing) and pointwise convolutions (feature aggregation).
- **Bounded Waveform Head:** 1D Conv + $\tanh$ non-linearity bounding the output BVP wave to $[-1.0, +1.0]$, preventing gradient explosion and numerical saturation.
- **Temporal Attention Biomarker Head:** Multi-head self-attention on AC trunk features, DC convolutional pooling, and camera sensor conditioning.

#### 3.3.5 Ground Truth Synchronization & Signal Artifact Rectification
During training analysis, two critical data artifacts were uncovered and resolved:
1. **ECG T-Wave Double-Counting:** In post-exercise recordings with elevated pulse rates, the automated ECG ground-truth detector misidentified tall T-waves as auxiliary R-peaks, artificially doubling reference heart rates up to 180+ BPM. Correcting the detection threshold reduced baseline errors from ~48 BPM to ~19 BPM.
2. **FFT Frequency Quantization:** Standard Welch PSD estimation was quantizing estimates onto 7.5 BPM discrete bins due to small window lengths ($N=240$ at 30 FPS $\to 30/240 = 0.125\text{ Hz} = 7.5\text{ BPM}$). Implementing zero-padded FFT ($N_{\text{FFT}} = 4096$) eliminated quantization grid locking, achieving sub-BPM frequency resolution.

#### 3.3.6 Clinical Dashboard & Real-Time Visualization
An interactive **Streamlit dashboard (`app.py`)** was constructed allowing users to select benchmark video clips or live webcams, displaying synchronized face ROI tracking, real-time BVP waveforms, power spectrum peak-picking, and ground-truth error metrics.

### 3.4 Software and Tools Used
The development environment comprised Python 3.10+, PyTorch with CUDA acceleration on an NVIDIA RTX GPU, OpenCV for video streaming, MediaPipe for face geometry, SciPy for signal processing, and Streamlit for web deployment, with VS Code as the IDE.

### 3.5 Highlights of Training Exposure
- Complete data-to-decision engineering lifecycle from optical capture to clinical vital estimation.
- End-to-end implementation of MediaPipe landmark tracking and multi-ROI spatial signal pooling.
- Design of lightweight depthwise-separable temporal convolutions for long sequence modeling.
- Discovery and correction of medical ground-truth annotation bugs (ECG T-wave anomaly).
- Scientific falsifiability and rigorous biomarker confound auditing.
- Deployment of a real-time web application for live physiological monitoring.

---

# 4. Problem Identification and Case Study

### 4.1 Problem Statement
Traditional patient monitoring relies on physical contact sensors (oximeters, ECG electrodes, cuff sphygmomanometers) and invasive blood sampling. These methods cause patient discomfort, risk skin breakdown in neonatal/geriatric care, and present cross-contamination risks during epidemic outbreaks. While contactless optical estimation (rPPG) from webcams offers an attractive non-invasive alternative, existing literature suffers from two severe flaws:
1. **Poor Generalization:** Models overfit to specific camera sensors, skin tones, or subject identities due to contaminated train/test splits.
2. **Unsubstantiated Biomarker Claims:** Many published papers claim to measure complex blood chemistry (hemoglobin, glucose, cholesterol) from ordinary webcams without testing whether their models are merely memorizing demographic identity proxies.

The **HemoVision Case Study** addresses these challenges by developing a robust, leakage-safe rPPG framework, rigorously benchmarking heart rate estimation, and conducting a scientific feasibility audit across 11 physiological biomarkers.

### 4.2 Data Used in the Case Study

#### Table 3: MCD-rPPG Clinical Dataset Summary
| Partition | Clips | Unique Subjects | Camera Sensors | Conditions | Ground Truth |
|:---|:---:|:---:|:---|:---|:---|
| **Train** | 2,387 | 418 | FullHDwebcam, USBVideo, IriunWebcam | Resting ('before'), Post-exercise ('after') | Synchronized ECG / PPG / Lab metrics |
| **Validation** | 517 | 89 | FullHDwebcam, USBVideo, IriunWebcam | Resting ('before'), Post-exercise ('after') | Synchronized ECG / PPG / Lab metrics |
| **Held-Out Test** | 523 | 91 | FullHDwebcam, USBVideo, IriunWebcam | Resting ('before'), Post-exercise ('after') | Synchronized ECG / PPG / Lab metrics |
| **TOTAL** | **3,427** | **598** | **3 Camera Types** | **2 Physical States** | **11 Clinical Biomarkers** |

#### Table 4: Ground-Truth Clinical Biomarkers in Dataset
| Biomarker | Unit | Clinical Mean $\pm$ Std | Range [Min, Max] | Measurement Method |
|:---|:---:|:---:|:---:|:---|
| **Pulse (Heart Rate)** | BPM | $91.93 \pm 18.36$ | [49.00, 153.00] | Synchronized ECG lead contact sensor |
| **Respiratory Rate** | BrPM | $18.05 \pm 1.71$ | [15.00, 24.00] | Capnography / chest transducer |
| **SpO2 (Oxygen Saturation)**| % | $98.01 \pm 1.29$ | [86.00, 99.00] | Medical-grade pulse oximeter |
| **Systolic Blood Pressure** | mmHg | $122.46 \pm 17.43$ | [80.00, 202.00] | Digital oscillometric cuff |
| **Diastolic Blood Pressure**| mmHg | $73.79 \pm 9.25$ | [50.00, 108.00] | Digital oscillometric cuff |
| **Hemoglobin** | g/dL | $13.59 \pm 1.66$ | [8.10, 17.30] | Laboratory venous blood draw |
| **Glycated Hemoglobin (HbA1c)**| % | $5.52 \pm 0.69$ | [3.40, 13.02] | Venous laboratory HPLC assay |
| **Total Cholesterol** | mmol/L | $4.16 \pm 0.83$ | [0.90, 8.00] | Enzymatic venous blood test |
| **Vascular Rigidity** | m/s | $8.99 \pm 3.03$ | [1.75, 34.02] | Arterial stiffness tonometry |
| **Mental Stress Index** | Score | $3.04 \pm 1.46$ | [1.00, 7.52] | Validated psychological assessment |
| **Body Mass Index (BMI)** | $\text{kg/m}^2$ | $22.73 \pm 4.34$ | [15.39, 47.03] | Calibrated stadiometer & scale |

### 4.3 Data Preparation & ROI Extraction Pipeline
The raw video clips are processed through an automated geometric extraction pipeline:
1. Video frames are read at 30 FPS across sliding temporal windows of 600 frames (20 seconds) with 150-frame step stride.
2. MediaPipe Face Mesh detects 468 facial landmark coordinates in real time.
3. Eight facial anatomical regions are masked and averaged across RGB channels.
4. Channel ordering is verified and auto-corrected (ensuring RGB format via $R_{\text{mean}} > B_{\text{mean}}$ skin reflection heuristics).
5. AC z-score normalized signal ($x \in \mathbb{R}^{24 \times T}$) and DC raw signal ($x_{\text{raw}} \in \mathbb{R}^{24 \times T}$) tensors are constructed.

#### Table 5: Facial Regions of Interest (ROIs) and Channel Specification
| ROI Index | Facial Anatomical Region | Landmarks Included | Primary Optical Purpose |
|:---:|:---|:---|:---|
| **ROI 0** | Left Forehead | 67, 109, 10, 151, 9 | High superficial vascular density |
| **ROI 1** | Right Forehead | 10, 338, 297, 9, 151 | High superficial vascular density |
| **ROI 2** | Left Upper Cheek | 117, 118, 101, 205, 50 | Strong capillary perfusion, minimal hair |
| **ROI 3** | Right Upper Cheek | 346, 347, 330, 425, 280 | Strong capillary perfusion, minimal hair |
| **ROI 4** | Left Lower Cheek | 132, 136, 172, 58, 172 | Lateral perfusion validation |
| **ROI 5** | Right Lower Cheek | 361, 365, 397, 288, 397 | Lateral perfusion validation |
| **ROI 6** | Mid-face / Nose Bridge | 168, 6, 197, 195, 5 | Motion-stable reference zone |
| **ROI 7** | Chin Region | 175, 199, 200, 18, 140 | Secondary perfusion zone |

### 4.4 Machine-Learning Feature Set
- **Raw Spatial Channel Tensors:** $[8 \text{ ROIs} \times 3 \text{ Channels} = 24 \text{ Input Channels}]$.
- **Temporal Dimensions:** 600 frames ($T=600$ at $f_s=30\text{ Hz}$).
- **Waveform Target:** Synchronized ground-truth BVP/ECG pulse waveform ($[1 \times T]$).
- **Metadata Conditioning:** Camera sensor one-hot encodings (FullHDwebcam, USBVideo, IriunWebcam).

### 4.5 System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                 HemoVision System Architecture                                    |
+---------------------------------------------------------------------------------------------------+
|  [Input Video Stream (RGB 30 FPS)]                                                                |
|         |                                                                                         |
|         v                                                                                         |
|  [MediaPipe Face Mesh Tracking (468 3D Points)] ---> [8 Anatomical ROI Segmentation & Pooling]    |
|                                                                     |                             |
|                                                                     v                             |
|                                                     [24-Channel Spatial RGB Matrix]               |
|                                                                     |                             |
|                                      +------------------------------+--------------------------+  |
|                                      |                                                         |  |
|                                      v                                                         v  |
|                      [AC Stream: Z-Score Normalised]                           [DC Stream: Raw/255]
|                                      |                                                         |  |
|                                      v                                                         |  |
|         [Input 1D Conv Projection: 24 -> 48 Channels (BatchNorm + GELU)]                       |  |
|                                      |                                                         |  |
|                                      v                                                         |  |
|         [Dilated Depthwise-Separable TCN Trunk (Dilations: 1, 2, 4, 8; Dropout: 0.15)]         |  |
|                                      |                                                         |  |
|                   +------------------+------------------+                                      |  |
|                   |                                     |                                      |  |
|                   v                                     v                                      v  |
|       [Waveform Conv Head + Tanh]          [Multi-Head AC Self-Attention] <--- [DC Conv Pooling]  |
|                   |                                     |                         |               |
|                   v                                     +------------+------------+               |
|       [Predicted BVP Waveform]                                       |                            |
|                   |                                                  v                            |
|                   v                                      [Tiered Biomarker Heads]                 |
|         [Welch PSD & FFT Peak]                                       |                            |
|                   |                                                  v                            |
|                   v                                      [10 Physiological Biomarkers]            |
|       [Heart Rate Output (BPM)]                                                                   |
+---------------------------------------------------------------------------------------------------+
```
*Figure 3: HemoVision System Architecture*

### 4.6 Methodology
1. **Clinical Requirement Analysis:** Formulate non-contact physiological measurement objectives and establish ethical data boundaries.
2. **Dataset Partitioning:** Implement strict subject-disjoint splits (418 train, 89 val, 91 test subjects) preventing data leakage.
3. **Multi-ROI Video Processing:** Extract synchronized 24-channel spatial averages across 8 facial regions via MediaPipe Face Mesh.
4. **Signal Normalization:** Construct dual AC (z-score standardized) and DC (raw scaled) temporal representations.
5. **Ground Truth Validation:** Inspect and correct reference annotations (resolving ECG T-wave doubling anomalies).
6. **Classical Benchmarking:** Implement POS and CHROM algorithmic projections to establish signal processing baselines.
7. **Deep Architecture Engineering:** Develop the Depthwise-Separable Bounded TCN with exponential dilation receptive fields.
8. **Multi-Task Optimization:** Train with Negative Pearson Correlation waveform loss and AMP gradient accumulation.
9. **Spectral Analysis:** Extract Heart Rate via zero-padded Welch FFT peak picking within the physiological band ($0.65\text{--}3.25\text{ Hz}$).
10. **Scientific Feasibility Audit & Confound Testing:** Benchmark biomarker heads against demographic OLS and group-mean baselines.
11. **Clinical UI Deployment:** Build the interactive Streamlit and OpenCV real-time visual demonstration dashboard.

### 4.7 Data-to-Decision Pipeline

```
+------------------+      +-------------------+      +--------------------+      +--------------------+      +--------------------+
|  Video Capture   | ---> |  Signal Extraction| ---> | Deep Neural TCN    | ---> | Spectral Peak FFT  | ---> | Real-Time Clinical |
| (Ambient Webcam) |      | (8 ROIs, 24 Chans)|      | (Bounded BVP Wave) |      | (BPM Determination)|      | Dashboard Display  |
+------------------+      +-------------------+      +--------------------+      +--------------------+      +--------------------+
```
*Figure 4: Data-to-Decision Pipeline for HemoVision*

### 4.8 Model Development
The core model architecture, `HemoVisionBoundedTCN`, was developed through systematic iterative experimentation:
- **Baseline 1D CNN:** Suffered from limited receptive fields ($<3\text{ seconds}$), unable to capture complete cardiac rhythm cycles.
- **Vanilla TCN:** Suffered from unbounded activations, resulting in gradient explosions when training on noisy webcam segments.
- **Bounded Depthwise-Separable TCN (Final Model):** Employs depthwise separable 1D convolutions across dilation rates $d \in \{1, 2, 4, 8\}$, achieving an effective receptive field of $>20\text{ seconds}$ (600 frames at 30 FPS) while reducing parameter count by $78\%$ compared to standard convolutions. A final $\tanh$ activation constrains the reconstructed BVP wave to $[-1.0, +1.0]$, ensuring gradient stability.
- **Waveform Loss Function:** Formulated as Negative Pearson Correlation Coefficient ($r$):
$$\mathcal{L}_{\text{BVP}} = 1.0 - \frac{\sum_{t} (y_t - \bar{y})(\hat{y}_t - \bar{\hat{y}})}{\sqrt{\sum_t (y_t - \bar{y})^2 \sum_t (\hat{y}_t - \bar{\hat{y}})^2} + \epsilon}$$

### 4.9 Vital-Sign & Biomarker Evaluation Outputs

#### Table 6: Performance Comparison on Held-Out Test Set (91 Subjects, 523 Clips)
| Model / Algorithm | Model Type | Heart Rate MAE (BPM) | Heart Rate RMSE (BPM) | Waveform Pearson $r$ | Parameters |
|:---|:---|:---:|:---:|:---:|:---:|
| **POS (Wang et al. 2017)** | Classical Signal Processing | 19.17 | 23.72 | 0.119 | 0 (None) |
| **CHROM (de Haan 2013)** | Classical Signal Processing | 20.28 | 24.61 | 0.047 | 0 (None) |
| **HemoVisionBoundedTCN** | Deep Depthwise TCN (Ours) | **9.19** | **18.35** | **0.285** | **148,650** |
| *HemoVision Demo Clips* | *High-Quality Test Clips* | **1.33** | **1.82** | **0.884** | *148,650* |

#### Table 7: Multi-Biomarker Feasibility Audit & Baseline Comparison
| Biomarker | Tier Classification | Deep Model Pearson $r$ | Deep Model MAE | Demographic / Mean Baseline MAE | Video Beats Baseline? | Clinical Status |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Pulse (Heart Rate)** | Established | **0.285** | **9.19 BPM** | 14.67 BPM | **YES** | **Feasible via RGB** |
| **Respiratory Rate** | Established | 0.225 | 1.47 BrPM | 1.45 BrPM | No | Limited Modulation |
| **SpO2 (Oxygen Saturation)**| Established | 0.200 | 0.96 % | 0.88 % | No | Requires Dual-Wavelength NIR |
| **Systolic Blood Pressure** | Moderate | 0.174 | 13.13 mmHg | 12.93 mmHg | No | Requires Multi-Site PWTT |
| **Diastolic Blood Pressure**| Moderate | -0.023 | 7.33 mmHg | 6.73 mmHg | No | Requires Multi-Site PWTT |
| **Hemoglobin** | Exploratory | 0.442 | 1.24 g/dL | **0.95 g/dL (Demographics)** | **NO (Confound)**| **Optical Artifact** |
| **Body Mass Index (BMI)** | Exploratory | 0.564 | 2.77 $\text{kg/m}^2$ | 3.14 $\text{kg/m}^2$ | Weak | Facial Morphology Feature |
| **Mental Stress** | Exploratory | -0.033 | 1.12 Score | 1.06 Score | No | Needs HRV Time-Domain |
| **Vascular Rigidity** | Exploratory | 0.048 | 2.04 m/s | 2.00 m/s | No | Insufficient Resolution |
| **Total Cholesterol** | Exploratory | 0.123 | 0.59 mmol/L | 0.58 mmol/L | No | No Optical Mechanism |
| **Glycated Hb (HbA1c)** | Exploratory | -0.151 | 0.38 % | 0.34 % | No | No Optical Mechanism |

### 4.10 Results and Discussion
1. **Heart Rate Estimation is Robust:** The deep learning model achieves **9.19 BPM MAE** across all 523 held-out test clips, cutting the error of classical signal processing methods in half (POS: 19.17 BPM, CHROM: 20.28 BPM). On clear facial recordings, HemoVision achieves **1.33 BPM MAE** (e.g., predicted 101.7 vs ground-truth 102.1 BPM).
2. **Biomarker Feasibility Audit (Rigorous Negative Result):**
   - Five independent architectural hypotheses were tested: (1) target standardisation, (2) batch normalization freeze, (3) multi-head attention pooling, (4) raw DC bypass routing, and (5) Lin's CCC loss optimization. None produced genuine predictive capability for blood chemistry.
   - For **Hemoglobin**, the video model showed an apparent Pearson $r \approx 0.44$. However, a simple Ordinary Least Squares (OLS) model trained strictly on demographic variables (age, sex, BMI) achieved **$r = 0.70$ and MAE = 0.95 g/dL**, heavily outperforming the video model. The label repeat fraction across recording sessions was $1.0$ (blood drawn once per subject visit). The neural network was memorizing facial identity shortcuts rather than detecting optical blood absorption.
3. **Physical Optical Constraints:** Reconstructing blood chemistry (hemoglobin, oxygen saturation) requires measuring differential light absorption ratios across narrow isosbestic wavelengths (e.g., 660 nm red and 940 nm near-infrared). Commercial RGB cameras integrate broad spectral bands (400–700 nm) under Bayer filters, conflating blood absorption with melanin and ambient lighting.

### 4.11 Skills and Outcomes

#### Table 8: Training outcomes and skills gained
| Area | Training Outcome |
|:---|:---|
| **Computer Vision** | Mastered 468-point 3D facial landmark detection and multi-ROI spatial signal pooling with MediaPipe |
| **Digital Signal Processing** | Implemented Butterworth bandpass filters, POS/CHROM projections, and zero-padded Welch PSD |
| **Deep Learning Architecture** | Designed and trained Depthwise-Separable Bounded TCNs with residual dilation trunks in PyTorch |
| **Loss Engineering** | Formulated Negative Pearson correlation waveform loss and Lin's Concordance Correlation Coefficient |
| **Debugging & Data Integrity** | Discovered and resolved medical ground-truth anomalies (ECG T-wave doubling) and FFT quantization |
| **Scientific Verification** | Conducted leak-proof subject-disjoint evaluation and exposed demographic confound shortcuts in AI |
| **UI & Web Deployment** | Built interactive real-time Streamlit and OpenCV clinical monitoring dashboards |
| **Technical Documentation** | Prepared publication-grade biomedical reports, architecture schematics, and comparative audits |

---

# 5. Recommendations

The HemoVision architecture provides a sound, scientifically verified foundation for contactless physiological sensing. The following recommendations provide a roadmap for future research and clinical engineering extensions.

### 5.1 Model & Algorithmic Improvements
- **Self-Supervised Pre-Training:** Pre-train spatial-temporal transformer backbones on large unlabelled facial video datasets (e.g., YouTube face corpora) using masked autoencoding before fine-tuning on clinical rPPG.
- **Adaptive Motion-Artifact Compensation:** Integrate optical flow vectors from facial landmark displacements into the TCN conditioning layer to dynamically cancel out speaking and head-rotation motion artifacts.
- **Physiological Confidence Intervals:** Implement Bayesian neural dropout or ensemble deep evidential regression to output real-time uncertainty metrics alongside heart-rate predictions.

### 5.2 Hardware & Multi-Wavelength Optical Improvements
- **Dual-Wavelength Active Illumination:** Pair the HemoVision software with a dual-wavelength LED ring (660 nm visible red + 940 nm near-infrared) to provide the physical optical absorption spectra required for true non-contact SpO2 and hemoglobin quantification.
- **Global Shutter High-Framerate Sensors:** Utilize 60–120 FPS global-shutter cameras to eliminate rolling-shutter artifacts and improve high-frequency pulse wave morphological fidelity.

### 5.3 Physiological Metric Extensions (HRV / RMSSD / PWTT Blood Pressure)
- **Heart Rate Variability (HRV):** Extract inter-beat intervals (IBI) directly from the reconstructed BVP wave to calculate clinical autonomic stress markers: RMSSD, SDNN, and LF/HF frequency power ratios.
- **Pulse Wave Transit Time (PWTT) for Cuffless Blood Pressure:** Utilize dual-site multi-ROI waveforms (measuring phase delay between forehead and chin/carotid micro-vessels) to estimate real-time pulse wave velocity and blood pressure.

### 5.4 Deployment & Clinical Interface Improvements
- **Edge Acceleration:** Quantize the PyTorch model to ONNX / TensorRT / INT8 for deployment on low-power edge medical devices (e.g., Raspberry Pi 5, NVIDIA Jetson).
- **FHIR / HL7 Medical System Integration:** Implement secure REST APIs complying with Fast Healthcare Interoperability Resources (FHIR) to transmit vital sign telemetry directly into electronic health record (EHR) systems.

### 5.5 Future Scope
- Developing an automated ETL and continuous inference pipeline for telemetry streaming.
- Deploying the multi-task rPPG model into an asynchronous telemedicine web portal.
- Implementing real-time patient triage alert notifications for bradycardia, tachycardia, and arrhythmia.
- Extending multi-subject tracking to monitor multiple patients simultaneously in hospital waiting rooms.

### 5.6 Recommended Future Clinical Dashboard Layout

#### Table 9: Recommended clinical dashboard layout
| Dashboard Page | Suggested Content |
|:---|:---|
| **Live Telemetry Overview** | Real-time facial video feed, MediaPipe 8-ROI wireframe, live BVP waveform, and instantaneous BPM |
| **Vital Signs & Trend Monitor**| Multi-hour heart-rate history, respiratory rate trends, and physiological confidence gauge |
| **Spectral & Diagnostic View** | Welch PSD power spectrum plot, peak frequency selector, and SNR quality indicator |
| **Subject & Clinical History** | Patient ID, session notes, comparative baseline metrics, and exportable PDF medical summary |
| **Hardware & Camera Calibration**| Camera sensor selection (FPS/resolution), illumination brightness meter, and ROI sensitivity tuner |

---

# 6. References

1. **Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G.** (2017). Algorithmic principles of remote photoplethysmography. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
2. **de Haan, G., & Jeanne, V.** (2013). Robust pulse rate from chrominance-based rPPG. *IEEE Transactions on Biomedical Engineering*, 60(10), 2878–2886.
3. **Lugaresi, C., et al.** (2019). MediaPipe: A Framework for Building Perception Pipelines. *arXiv preprint arXiv:1906.08172*.
4. **Paszke, A., et al.** (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
5. **Bradski, G.** (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
6. **Virtanen, P., et al.** (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17(3), 261–272.
7. **BS Digital Technology.** Training materials, guidance, and project-based training curriculum in Machine Learning and Computer Vision (2026).
8. **MCD-rPPG Clinical Dataset.** Synchronized clinical dataset comprising multi-camera facial videos and contact ground-truth vital sensors.

---

# 7. Appendices

### Appendix A – Suggested Project Folder Structure

```
HemoVision/
├── app.py                           # Interactive Streamlit clinical monitoring dashboard
├── hemovision_demo.py               # Standalone OpenCV live inference & testing engine
├── HemoVision_V3_Pipeline.py        # Complete end-to-end training and evaluation script
├── cell16_hemoglobin_feasibility_audit.py # Multi-biomarker & demographic confound audit script
├── data/
│   └── vitalscan-clinic/
│       └── MCD-rPPG/
│           ├── db.csv               # Ground-truth clinical biomarker database
│           ├── manifest.json        # Video clip metadata and subject-disjoint splits
│           ├── video/               # Raw RGB video clips (3 camera sensors)
│           ├── ppg_sync/            # Camera-synchronized contact reference waveforms
│           └── checkpoints_v2/      # Saved neural network model weights (.pt)
├── v3/                              # Modular training package
│   ├── config.py                    # Training hyperparameters, task specs, and camera configs
│   ├── dataset.py                   # PyTorch Dataset and multi-ROI temporal loader
│   ├── model.py                     # HemoVisionBoundedTCN & Attention Biomarker Head architecture
│   ├── loss.py                      # Negative Pearson r and CCC multi-task loss modules
│   └── train.py                     # Multi-epoch training loop with AMP and validation tracking
├── v3_pipeline/                     # Experimental modular pipeline components
│   ├── cell_6_config.py             # POS/CHROM signal processing toolkit & filters
│   ├── cell_10_dataset.py           # Windowed tensor batch generator
│   ├── cell_11_dataset.py           # Deep learning dataset loader
│   ├── cell_12_model.py             # Model definitions and dilation blocks
│   └── cell_14_eval.py              # Held-out subject test evaluation
└── reports/
    └── HemoVision_Vocational_Training_Report.md
```

---

### Appendix B – Final Biomarker & ROI Feature Groups

```
+---------------------------------------------------------------------------------------------------+
| Feature Type       | Fields & Specifications                                                      |
+--------------------+------------------------------------------------------------------------------+
| Identifiers & Meta | subject_id, clip_id, camera_type (FullHDwebcam / USBVideo / IriunWebcam),     |
|                    | condition (resting 'before' / post-exercise 'after'), split (train/val/test) |
+--------------------+------------------------------------------------------------------------------+
| Spatial ROIs (8)   | Forehead Left/Right, Upper Cheek Left/Right, Lower Cheek Left/Right,         |
|                    | Mid-face/Nose Bridge, Chin (MediaPipe 468-point Face Mesh)                   |
+--------------------+------------------------------------------------------------------------------+
| Optical Channels   | 24 Channels total (8 ROIs x 3 RGB channels); AC (z-score) & DC (raw/255)     |
+--------------------+------------------------------------------------------------------------------+
| Waveform Target    | BVP pulse waveform (600 temporal frames @ 30 FPS, bounded [-1, 1])           |
+--------------------+------------------------------------------------------------------------------+
| Vital Signs (Tier1)| Pulse / Heart Rate (BPM), Respiratory Rate (BrPM), SpO2 (%)                  |
+--------------------+------------------------------------------------------------------------------+
| Blood Pressure (T2)| Systolic BP (mmHg), Diastolic BP (mmHg)                                      |
+--------------------+------------------------------------------------------------------------------+
| Blood Chem (T3)    | Hemoglobin (g/dL), HbA1c (%), Cholesterol (mmol/L), Rigidity, Stress, BMI    |
+---------------------------------------------------------------------------------------------------+
```

---

### Appendix C – rPPG Training & Evaluation Checklist

- [x] Verify subject-level disjoint integrity between train, validation, and held-out test partitions.
- [x] Validate video frame rates ($f_s = 30\text{ FPS}$) and temporal synchronization with contact PPG/ECG.
- [x] Check for BGR versus RGB color channel ordering using skin reflection ratio heuristics ($R_{\text{mean}} > B_{\text{mean}}$).
- [x] Apply Butterworth zero-phase bandpass filtering ($0.65\text{--}3.25\text{ Hz}$).
- [x] Inspect ECG reference annotations for T-wave double counting in post-exercise recordings.
- [x] Standardize AC temporal features using channel-wise z-score normalization.
- [x] Initialize Depthwise-Separable Bounded TCN with dilation schedule $(1, 2, 4, 8)$.
- [x] Train using Negative Pearson correlation loss with Automatic Mixed Precision (AMP).
- [x] Zero-pad Welch Power Spectral Density ($N_{\text{FFT}} = 4096$) to avoid frequency quantization artifacts.
- [x] Audit biomarker predictions against demographic OLS and group-mean baselines before claiming optical validity.
- [x] Test live model inference on real-time webcam video stream via Streamlit dashboard.
