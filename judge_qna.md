# HemoVision — SIH Grand Finale Judge Q&A & Technical Appendix

This document prepares the team to defend HemoVision before senior evaluators, clinicians, and technical jury panels at the **Smart India Hackathon 2026 Grand Finale**.

---

## Part 1: Tough Judge Questions & Strategic Model Answers

---

### Question 1: "9.19 BPM MAE is too high for a clinical intensive care unit. How can you claim this is useful?"

**Strategic Defense:**
> *"We agree completely, and that is precisely why we position HemoVision as a **pre-consultation triage screening tool**, never an ICU or diagnostic vital monitor.*
> *In a clinical ICU, vital monitors operate on continuous arterial lines or direct ECG leads with sub-beat precision. However, in outpatient remote triage or pre-consultation tele-health, a doctor often starts with **zero** objective vital readings.*
> *Furthermore, on our held-out test set of 91 subjects across 523 clips, the 9.19 BPM MAE represents a **52% error reduction** over established classical methods like POS (19.17 BPM) and CHROM (20.28 BPM). When ambient lighting and head posture are controlled, our prototype achieves **1.33 BPM MAE**.*
> *Most importantly, our system is equipped with automated quality gates: if the optical SNR is insufficient, it refuses to guess and refers the patient to standard contact measurement."*

---

### Question 2: "Where did your dataset come from, and how did you guarantee zero data leakage?"

**Strategic Defense:**
> *"We trained and evaluated on the **MCD-rPPG Clinical Dataset**, which contains 3,427 video clips from 598 unique human subjects captured under both resting and post-exercise states across three commercial camera types (FullHDwebcam, USBVideo, IriunWebcam), synchronized with clinical ECG and pulse oximeter contact sensors.*
> *To guarantee zero data leakage, our split is strictly **subject-disjoint**:*
> - *Training split: 418 subjects (2,387 clips)*
> - *Validation split: 89 subjects (517 clips)*
> - *Held-out test split: 91 subjects (523 clips)*
> *No individual appearing in the test split exists in the training or validation splits. Many published rPPG papers achieve artificially low errors because they perform random clip-level splitting where the same person's face appears in both train and test sets—allowing the network to memorize facial appearance rather than blood volume pulses. Our 9.19 BPM MAE reflects true unseen-subject generalization."*

---

### Question 3: "How does your model handle dark skin tones (Fitzpatrick V–VI), dim lighting, and motion?"

**Strategic Defense:**
> *"These represent the three fundamental optical challenges of remote photoplethysmography:*
> 1. ***Melanin & Skin Tone (Fitzpatrick V–VI):*** *Higher epidermal melanin concentration increases baseline optical attenuation across the green spectrum (500–600 nm). Because HemoVision uses 8 separate ROIs across both forehead and malar cheek regions and extracts chromaticity difference vectors (AC z-score normalization), it reduces baseline DC skin-tone offsets. However, we acknowledge that performance degrades on very dark skin tones under low ambient illumination—which is why our next milestone is a dedicated prospective cohort across all Fitzpatrick classes.*
> 2. ***Ambient Lighting:*** *Our automated quality gate measures spatial luminance across the face mesh. If illumination drops below 40 lux, estimation is suspended.*
> 3. ***Subject Motion:*** *MediaPipe Face Mesh tracks 468 3D facial landmarks dynamically. During talking or sudden head turns, our motion-jitter guard computes inter-frame rigid ROI displacement; if velocity exceeds a stability threshold, the estimation buffer pauses.*
> *We never output a fabricated number during poor optical conditions."*

---

### Question 4: "Why don't you claim SpO2, Blood Pressure, or Hemoglobin like other health startups?"

**Strategic Defense:**
> *"This is the core scientific differentiator of our project. We did not merely avoid these claims—**we experimentally audited and disproved them.***
> *Our dataset contained ground-truth clinical data for 11 biomarkers, including hemoglobin, blood pressure, SpO2, and lipid levels.*
> *When we trained multi-task deep neural heads to predict hemoglobin from facial video, the model achieved an apparent Pearson correlation of $r = 0.44$. However, when we built a trivial Ordinary Least Squares (OLS) model trained strictly on demographic metadata (age, biological sex, BMI), that baseline achieved $r = 0.70$ and MAE = 0.95 g/dL!*
> *Because each subject only had one clinical blood draw per visit, the neural network was simply memorizing demographic identity proxies and facial morphology shortcuts, not optical blood absorption.*
> *From optical physics, measuring blood oxygenation (SpO2) and hemoglobin requires calculating the differential absorption ratio across two narrow isosbestic wavelengths (specifically 660 nm visible red and 940 nm near-infrared). Commercial RGB cameras use broad Bayer filters that blend 400 to 700 nm into three wide bands, conflating melanin and ambient light with blood chemistry.*
> *Presenting these unverified outputs to patients would be clinically dangerous. We report the truth: RGB video reliably tracks pulsatile heart rate, but chemical biomarkers require dual-wavelength NIR hardware."*

---

### Question 5: "What about patient privacy and biometric surveillance concerns?"

**Strategic Defense:**
> *"HemoVision is designed under strict privacy-by-design principles:*
> 1. ***Client-Side Edge Execution:*** *The video stream never leaves the local device. Video frames are processed in volatile RAM and immediately discarded after spatial ROI pooling.*
> 2. ***No Raw Video Stored:*** *We do not record, store, or transmit patient video or facial imagery. Only the anonymous 1D BVP waveform and numerical BPM value are processed.*
> 3. ***No Facial Recognition:*** *MediaPipe is utilized strictly for geometric anatomical landmark tracking, not biometric identification or facial recognition.*
> *This architecture directly complies with the draft **Digital Personal Data Protection (DPDP) Act 2023** and **ABDM Health Data Management Policy** guidelines."*

---

### Question 6: "What is your regulatory pathway under Indian CDSCO medical device rules?"

**Strategic Defense:**
> *"Under the CDSCO's **Draft Guidance on Medical Device Software (SaMD) (October 2025)**, software that provides physiological measurements for clinical decision-making is classified based on risk:*
> - *HemoVision is currently classified as a **Class A/B Research & Pre-Screening Prototype**, explicitly labeled as not intended for diagnostic or emergency monitoring.*
> - *To transition to clinical tele-triage deployment, our roadmap includes:*
>   1. *Completing formal verification and validation (V&V) under IEC 62304 life-cycle standards.*
>   2. *Conducting an Institutional Ethics Committee (IEC)-approved prospective clinical trial with Bland-Altman agreement analysis against FDA/CDSCO-cleared reference devices.*
>   3. *Applying for CDSCO Class B Medical Device Software approval prior to commercial integration with platforms like eSanjeevani.*
> *We take regulatory compliance as a first-class engineering requirement, not an afterthought."*

---

### Question 7: "What exact validation experiment will you conduct next?"

**Strategic Defense:**
> *"Our immediate next milestone is a prospective, multi-center pilot study with 150 participants across three distinct settings:*
> 1. *A university clinic waiting room (controlled fluorescent illumination).*
> 2. *A rural primary health sub-center (variable natural sunlight and mobile cameras).*
> 3. *A teleconsultation remote cohort at home.*
> *Protocol details:*
> - *Stratified recruitment ensuring 30%+ representation across Fitzpatrick skin types IV, V, and VI.*
> - *Concurrent synchronized ground-truth data collection using a 3-lead clinical ECG monitor and clinical pulse oximeter.*
> - *Quantifying mean bias, 95% limits of agreement (Bland-Altman), Root Mean Square Error (RMSE), and rejection rates of our automated quality gates.*
> *This will provide the prospective clinical evidence required to advance toward regulatory submission."*

---

## Part 2: Technical Reference Appendix

### 1. Model Architecture Specifications (`HemoVisionBoundedTCN`)

| Component | Specification | Rationale |
|---|---|---|
| **Input Channels** | 24 channels (8 anatomical ROIs × 3 RGB channels) | Captures spatial capillary perfusion diversity across facial zones |
| **Temporal Window** | 600 frames at 30 FPS (20.0 seconds), 150-frame step | Enforces cardiac cycle stability (>15 complete heart beats) |
| **Input Projection** | 1D Conv (24 $\rightarrow$ 48 channels), BatchNorm, GELU | Expands multi-channel spatial representations |
| **Temporal Trunk** | 4 Depthwise-Separable 1D Conv Blocks with dilations $d \in \{1, 2, 4, 8\}$ | Achieves receptive field $>20\text{s}$ while cutting parameters by 78% |
| **Regularization** | Spatial Dropout (0.15), Weight Decay ($10^{-4}$) | Prevents overfitting to camera noise and sensor quirks |
| **Output Head** | 1D Conv (48 $\rightarrow$ 1 channel) + Tanh activation | Tanh bounds output to $[-1.0, +1.0]$, preventing gradient explosions |
| **Total Parameters** | **148,650 parameters** (~595 KB footprint) | Ultra-lightweight; runs at >60 FPS on standard modern CPUs |
| **Loss Function** | Negative Pearson Correlation Coefficient ($\mathcal{L}_{\text{BVP}} = 1 - r$) | Optimizes pulse wave shape rather than arbitrary scale |

---

### 2. Digital Signal Processing Filter Specifications

| Parameter | Value | Physiological Rationale |
|---|---|---|
| **Sampling Frequency ($f_s$)** | 30.0 Hz | Standard webcam video frame rate |
| **Cardiac Passband** | **0.65 Hz to 3.25 Hz** (39 to 195 BPM) | Encompasses normal physiological resting and post-exercise heart rates |
| **Filter Topology** | 4th-Order Zero-Phase Butterworth Bandpass | Maximally flat passband response; forward-backward filtering removes phase distortion |
| **Spectral Estimation** | Welch's Power Spectral Density (Hanning window, 50% overlap) | Suppresses spectral leakage and reduces variance of noisy webcam signals |
| **Zero-Padding** | 4× Zero-Padding on FFT | Interpolates frequency bin resolution to $<0.5\text{ BPM}$ precision |

---

### 3. Engineering Bugs Identified & Corrected During Development

1. **Medical Reference ECG T-Wave Doubling Anomaly:**
   - *Discovery:* Initial benchmark runs showed large 2× error clusters where the ground-truth heart rate was reported as double the true cardiac rate.
   - *Root Cause:* In certain high-amplitude ECG lead configurations, the T-wave was misclassified as an R-peak by automated peak-detection software.
   - *Correction:* Implemented adaptive refractory blanking (250 ms) on the reference signal, resolving ground-truth contamination.
2. **Dynamic Facial ROI Boundary Clamping:**
   - *Discovery:* Video clips with lateral head rotation caused bounding box clipping and background pixel bleeding.
   - *Root Cause:* Landmark coordinates outside the frame boundary generated negative indices.
   - *Correction:* Added strict image boundary clamping and Delaunay polygon barycentric masking across the 8 MediaPipe anatomical zones.
3. **Vanilla TCN Gradient Explosion:**
   - *Discovery:* Standard dilated 1D CNN trunks experienced numerical overflow and NaN loss during training on unnormalized webcam segments.
   - *Root Cause:* Unbounded linear output layers amplified high-frequency lighting flicker.
   - *Correction:* Re-engineered the network into `HemoVisionBoundedTCN` with depthwise-separable layers and a final $\tanh$ squashing layer.

---

### 4. Full Ground-Truth Biomarker Audit Table (`Table 7`)

| Biomarker | Ground-Truth Measurement | Deep Model $r$ | Deep Model MAE | Demographic / Mean Baseline MAE | Optical Feasibility Verdict |
|---|---|:---:|:---:|:---:|:---|
| **Pulse (Heart Rate)** | Synchronized ECG Lead | **0.285** | **9.19 BPM** | 14.67 BPM | **FEASIBLE via RGB** |
| **Respiratory Rate** | Capnography / Chest Belt | 0.225 | 1.47 BrPM | 1.45 BrPM | Marginal; requires chest motion tracking |
| **SpO2** | Clinical Pulse Oximeter | 0.200 | 0.96 % | 0.88 % | **NOT FEASIBLE without NIR (660/940 nm)** |
| **Systolic BP** | Oscillometric Cuff | 0.174 | 13.13 mmHg | 12.93 mmHg | **NOT FEASIBLE without multi-site PWTT** |
| **Diastolic BP** | Oscillometric Cuff | -0.023 | 7.33 mmHg | 6.73 mmHg | **NOT FEASIBLE without multi-site PWTT** |
| **Hemoglobin** | Venous Blood Draw | 0.442 | 1.24 g/dL | **0.95 g/dL (Age/Sex OLS)** | **REJECTED: Demographic shortcut confound** |
| **Total Cholesterol** | Enzymatic Blood Test | 0.123 | 0.59 mmol/L | 0.58 mmol/L | **REJECTED: No physical optical mechanism** |
| **Glycated Hb (HbA1c)**| HPLC Assay | -0.151 | 0.38 % | 0.34 % | **REJECTED: No physical optical mechanism** |
| **Vascular Rigidity** | Arterial Tonometry | 0.048 | 2.04 m/s | 2.00 m/s | **REJECTED: Insufficient temporal resolution** |
| **Mental Stress Index**| Psychological Battery | -0.033 | 1.12 Score | 1.06 Score | **REJECTED: Requires long-term HRV analysis** |
| **Body Mass Index** | Calibrated Stadiometer | 0.564 | 2.77 $\text{kg/m}^2$ | 3.14 $\text{kg/m}^2$ | Weak; measures face shape, not vitals |

---

### 5. Authoritative References Cited Across the Submission

1. **Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008).** Remote plethysmographic imaging using ambient light. *Optics Express*, 16(26), 21434–21445.
2. **de Haan, G., & Jeanne, V. (2013).** Robust pulse rate from chrominance-based rPPG. *IEEE Transactions on Biomedical Engineering*, 60(10), 2878–2886.
3. **Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).** Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
4. **Chen, W., & McDuff, D. (2018).** DeepPhys: Video-based physiological measurement using convolutional attention networks. *European Conference on Computer Vision (ECCV)*, 349–365.
5. **Yu, Z., Li, X., & Zhao, G. (2019).** Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks. *British Machine Vision Conference (BMVC)*.
6. **MCD-rPPG Clinical Dataset Documentation & Benchmark (2024).** 3,427 video clips, 598 unique subjects with synchronized 11-sensor ground truth.
7. **PubMed 37313385 (2023).** Contactless photoplethysmography in clinical settings: A systematic review of accuracy, barriers, and translation pathways.
8. **PubMed 42106569 (2025).** Remote Photoplethysmography (rPPG) Clinical Translation Roadmap: From Laboratory Benchmark to Bedside SaMD.
9. **Central Drugs Standard Control Organization (CDSCO) (October 2025).** *Draft Guidance Document on Medical Device Software (SaMD)*, Directorate General of Health Services, Ministry of Health and Family Welfare, Government of India.
10. **Ministry of Health and Family Welfare (MoHFW) (2024).** *eSanjeevani: National Telemedicine Service Operational Framework*, Government of India.
