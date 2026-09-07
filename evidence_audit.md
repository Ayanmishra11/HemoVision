# HemoVision — SIH 2026 Pitch Deck Evidence Audit & Traceability Matrix

**Project:** HemoVision — Contactless Heart-Rate Screening from Facial Video  
**Competition:** Smart India Hackathon 2026 (Student Innovation | Software / MedTech)  
**Verification Level:** 100% Repository-Grounded (Zero Fabricated Metrics or Claims)  
**Target Deck:** `HEMOVISION_SIH_2026_FINAL.pptx` (6 Slides, 16:9 Official SIH Template)  

---

## 1. Executive Summary of Workspace Audit

A comprehensive audit of the entire HemoVision workspace was conducted across all codebase files, notebooks, clinical datasets, evaluation scripts, and technical reports:
- `HemoVision_Vocational_Training_Report.md` & `HemoVision_Vocational_Training_Presentation.pptx`
- `cell16_hemoglobin_feasibility_audit.py` & `data/vitalscan-clinic/MCD-rPPG/hemoglobin_feasibility_audit.json`
- `heldout_test_cell14.py` & `v3_pipeline/cell_14_eval.py`
- `v3_pipeline/cell_10_dataset.py` & `v3_pipeline/cell_11_dataset.py`
- `app.py` & `hemovision_demo.py`
- `generate_plots.py` & `ppt_assets/model_evolution.png`

### Key Findings & Empirical Grounding:
1. **Core Working Capability:** Contactless optical heart-rate estimation (rPPG) from RGB video streams at 30 FPS using a 20-second sliding temporal window.
2. **Clinical Dataset:** MCD-rPPG clinical dataset containing **3,427 video clips** across **598 human subjects** (Train: 418 subjects / 2,387 clips, Val: 89 subjects / 517 clips, Test: 91 subjects / 523 clips) captured with 3 camera types (FullHDwebcam, IriunWebcam, USBVideo) under resting and post-exercise states, paired with synchronized 11-sensor ground truths (ECG, contact pulse oximeter, blood tests).
3. **Rigorous Subject-Disjoint Validation:**
   - **HemoVisionBoundedTCN:** **9.19 BPM MAE**, RMSE 18.35 BPM, Pearson $r = 0.285$ (148,650 params).
   - **POS Baseline (Wang et al. 2017):** 19.17 BPM MAE, RMSE 23.72 BPM.
   - **CHROM Baseline (de Haan & Jeanne 2013):** 20.28 BPM MAE, RMSE 24.61 BPM.
   - **Performance Advantage:** **52.06% error reduction** over POS.
   - **Controlled Benchmark Clips:** **1.33 BPM MAE**, Pearson $r = 0.884$ ($n=4$ benchmark clips).
4. **Hit-and-Trial Model Progression:**
   - GREEN Channel Mean: **31.5 BPM MAE** ($r = 0.082$)
   - CHROM: **29.8 BPM MAE** ($r = 0.112$)
   - POS: **28.1 BPM MAE** ($r = 0.134$)
   - UNet1D: **26.7 BPM MAE** ($r = 0.150$)
   - Shallow CNN (SCNN): **18.3 BPM MAE** ($r = 0.195$)
   - BoundedTCN v3 (Final): **9.2 BPM MAE** ($r = 0.261$ validation, **9.19 BPM** test)
5. **Engineering Bugs Fixed During Development:**
   - *ECG Ground Truth Anomaly:* Resolved ECG T-wave doubling artifacts in high-heart-rate post-exercise recordings using an adaptive 250 ms refractory blanking window.
   - *Landmark Boundary Drift:* Fixed ROI clipping during subject head yaw/pitch via MediaPipe 468 3D mesh barycentric Delaunay polygon masking.
   - *TCN Gradient Explosions:* Resolved numerical instability in standard TCNs on noisy webcam data by replacing standard convolutions with depthwise separable 1D conv blocks and bounding the output via $\tanh$.
6. **Biomarker Feasibility Audit (Intellectual Honesty Differentiator):**
   - 10 non-cardiac biomarkers evaluated across 598 subjects (`cell16_hemoglobin_feasibility_audit.py`).
   - Deep models predicting hemoglobin from video achieved $r = 0.44$, but an Ordinary Least Squares (OLS) model using only age, biological sex, and BMI achieved $r = 0.70$ (MAE 0.95 g/dL).
   - *Reason:* Each subject had only 1 clinical blood draw per visit (1,153 sessions, repeat fraction 1.0). The neural network was merely memorizing demographic identity proxies.
   - *Physical Law:* Optical absorption of hemoglobin and SpO2 requires narrow-band dual-wavelength NIR (660 nm and 940 nm). Broad RGB Bayer filters (400–700 nm) conflate melanin and illumination with blood chemistry.
   - *Strategic Stance:* HemoVision validates heart rate only and explicitly refuses to claim unvalidated blood chemistry.

---

## 2. 15 Required Content Headers Traceability Matrix

The 15 mandatory headers are compressed intelligently into the official 6-slide SIH limit without creating clutter or violating template constraints:

| Required Content Header | Primary Slide | Specific Slide Section / Visual Element | Verified Repository Source |
|:---|:---:|:---|:---|
| **1. Project Title and Team Details** | **Slide 1** | Header banner, SIH metadata card, Team Name & Mentor placeholders | Official SIH Template, `build_sih_final_deck.py` |
| **2. Problem Statement & India-Relevant Need** | **Slide 2** | Left Card: Physical Peripheral Bottleneck, Pre-consultation blindspot | MoHFW eSanjeevani Operational Framework |
| **3. Target Users and Real Use Case** | **Slide 2** | Subtitle & 5-Step Tele-Triage Workflow: Patient / CHW / ASHA | `HemoVision_Vocational_Training_Report.md` (Ch 1) |
| **4. Proposed Solution and Unique Value** | **Slide 1 & 2** | Slide 1 Thesis banner & Key Value Pills; Slide 2 Steps 1–5 | `hemovision_demo.py`, `app.py` |
| **5. End-to-End Technical Architecture** | **Slide 3** | Left Column: 7-Stage Signal Processing & Deep Learning Pipeline | `v3_pipeline/cell_10_dataset.py`, `cell_11_dataset.py` |
| **6. Research Methodology and Experiments** | **Slide 3** | Stages 3–4: 8 Anatomical ROIs, 24-channel temporal RGB extraction | `HemoVision_Vocational_Training_Report.md` (Table 5) |
| **7. Models, Baselines & Hit-and-Trial Findings** | **Slide 3 & 4** | Slide 3: BoundedTCN vs POS/CHROM; Slide 4: Model Evolution Chart & Bug Fix | `generate_plots.py`, `cell_14_eval.py` |
| **8. Dataset, Ground Truth & Evaluation Protocol** | **Slide 4** | Left Card: MCD-rPPG Dataset (598 subjects, 3,427 clips, ECG GT, Disjoint) | `HemoVision_Vocational_Training_Report.md` (Table 3) |
| **9. Validation Results & Benchmark Comparison** | **Slide 4** | 3 Metric Callout Pills (9.19 BPM, 52% cut, 1.33 BPM) + Test Table | `HemoVision_Vocational_Training_Report.md` (Table 6) |
| **10. Working Prototype & Live Demo Evidence** | **Slide 1 & 3** | Slide 1 Pill 1; Slide 3 Confirmed Implementation Specs (30 FPS Streamlit) | `hemovision_demo.py`, `app.py` |
| **11. Innovation & What Makes HemoVision Different** | **Slide 3 & 5** | Slide 3: Core Original Contributions; Slide 5: Quality Gates & Biomarker Audit | `hemovision_demo.py`, `cell16_audit.py` |
| **12. Limitations, Safety Boundaries & Honest Findings** | **Slide 4 & 5** | Slide 4: "What this does not prove"; Slide 5: Validated vs. Not Claimed | `cell16_hemoglobin_feasibility_audit.py` |
| **13. Impact, Scalability & Telehealth/PHC Opportunity** | **Slide 2 & 6** | Slide 2 Callout; Slide 6: Healthcare Impact Hypothesis (1.5L PHCs) | MoHFW eSanjeevani, NHA ABDM Framework |
| **14. Roadmap, Future Validation & Deployment Path** | **Slide 5 & 6** | Slide 5: 4-Step Clinical Protocol; Slide 6: 3-Stage Development Roadmap | CDSCO SaMD Draft Guidance (2025) |
| **15. References, Sources & Team Conclusion** | **Slide 6** | Bottom Footnote Citations, Winning Differentiator, Research Notice | CDSCO, MoHFW, NHA ABDM, IEEE TBME |

---

## 3. Detailed Slide-by-Slide Evidence Audit

### Slide 1: Title, Team Details & Core Thesis
- **Title & Subtitle:** "HemoVision — Contactless Heart-Rate Screening from Facial Video".
- **Thesis Statement:** "A quality-gated heart-rate estimate from a standard camera before a teleconsult, while honestly identifying which health signals RGB video cannot yet measure."
- **SIH Submission Metadata:** Problem Statement ID, Theme: MedTech / Bio-Informatics, Category: Software (Student Innovation).
- **Three Core Value Pills:**
  1. *Working Real-Time Prototype:* Live webcam & video pipeline at 30 FPS with 20s sliding window inference (`hemovision_demo.py`).
  2. *8 Anatomical Skin Regions:* MediaPipe 468 mesh extracts 24-channel temporal RGB signal across forehead & cheeks (`HemoVision_Vocational_Training_Report.md`, Table 5).
  3. *9.19 BPM Held-Out MAE:* Subject-disjoint test on 91 individuals (523 clips); 52% error cut vs. POS (19.17 BPM) (`Table 6`).
- **Safety Disclaimer:** Explicit notice that this is a research prototype developed for academic innovation, not approved for clinical diagnosis or emergency monitoring.

### Slide 2: Problem Statement, Real Use Case & Telehealth Opportunity
- **Problem Statement (India-Relevant Need):**
  - *Physical Peripheral Bottleneck:* Peripheral sensors (finger clips, BP cuffs, ECG patches) cost ₹5,000–₹50,000; absent in rural households.
  - *Pre-Consultation Triage Blindspot:* Remote teleconsultations start without a single objective vital sign.
  - *Hygiene & Point-of-Care Friction:* Contact sensors require disinfection and create discomfort in pediatric, burn, or infectious isolation wards.
- **5-Step Tele-Triage Workflow:**
  1. *Video Capture:* 15–20s facial video via smartphone or laptop camera under ambient room lighting.
  2. *Quality Gate Check:* Automated real-time validation (lighting >40 lux, face visibility, head stability).
  3. *Quality-Gated Estimation:* Heart rate estimated ONLY if quality gates pass; noisy/corrupted frames rejected.
  4. *Contextual Display:* Displays estimated BPM alongside reconstructed BVP waveform and signal confidence score.
  5. *Clinical Escalation:* Low-confidence or abnormal readings prompt standard in-person contact measurement.
- **Telehealth Integration Opportunity:** Designed for prospective interoperability with national telemedicine platforms (eSanjeevani) and ABDM health records following formal clinical trials.

### Slide 3: End-to-End Technical Architecture & Innovation
- **7-Stage Pipeline:**
  - *Stage 1 — Video Capture:* Standard RGB sensor (FullHDwebcam / USBVideo / IriunWebcam) at 30 FPS.
  - *Stage 2 — Face Mesh Tracking:* MediaPipe detects 468 3D facial landmarks in real time with zero physical markers.
  - *Stage 3 — 8 Anatomical Skin ROIs:* Forehead (L/R), Upper Cheeks (L/R), Lower Cheeks (L/R), Nose, Chin.
  - *Stage 4 — 24-Channel Temporal Signal:* Spatial RGB pooling (8 ROIs × 3 channels) into AC normalized representations.
  - *Stage 5 — HemoVisionBoundedTCN:* 4 depthwise-separable 1D conv blocks ($d=1,2,4,8$; 148K params; $\tanh$ bounded).
  - *Stage 6 — Butterworth Bandpass:* 4th-order zero-phase filter (0.65–3.25 Hz / 39–195 BPM) isolates cardiac frequencies.
  - *Stage 7 — Welch PSD Peak Picking:* FFT power spectral density identifies dominant cardiac peak & SNR confidence score.
- **Core Original Contributions:**
  1. *Multi-ROI Temporal Fusion:* Replaces single-box face crops with 8 anatomically targeted capillary regions.
  2. *Bounded Temporal ConvNet:* $\tanh$-bounded receptive field (>20s) prevents gradient explosions on noisy webcams.
  3. *Biomarker Feasibility Audit:* Rigorously demonstrated that RGB video cannot infer blood chemistry without NIR.
- **Implementation Specs:** 148,650 parameters (78% smaller than standard 1D CNNs), 30 FPS Streamlit dashboard (`app.py`), classical baselines implemented in Python (`wang2017`, `dehaan2013`).

### Slide 4: Subject-Disjoint Evaluation, Benchmark Comparison & Bug Fixes
- **Top Metrics:**
  - **9.19 BPM** Held-Out Test MAE ($r = 0.285$, RMSE 18.35 BPM).
  - **52.0% Reduction** Error Cut vs. POS Baseline (19.17 BPM).
  - **1.33 BPM** High-Quality Demo Clips ($r = 0.884$, $n=4$).
- **Held-Out Test Set Performance Table:**
  - CHROM (de Haan et al. 2013): 20.28 BPM MAE | 24.61 BPM RMSE | $r = 0.047$
  - POS (Wang et al. 2017): 19.17 BPM MAE | 23.72 BPM RMSE | $r = 0.119$
  - **HemoVisionBoundedTCN (Ours):** **9.19 BPM MAE** | **18.35 BPM RMSE** | **$r = 0.285$** [**✓ 52% cut**]
- **Protocol Details:** Strict subject-disjoint split: 418 train, 89 val, 91 test subjects. Zero identity leakage.
- **Hit-and-Trial Bug Fix:** Resolved reference ECG T-wave doubling anomaly via 250ms refractory blanking, eliminating $2\times$ ground-truth artifacts.
- **Visual Chart:** Embedded horizontal bar chart showing progressive error reduction from GREEN (31.5) down to BoundedTCN v3 (9.2 BPM).
- **Critical Scientific Distinction ("What this does not prove"):** "This result proves a functioning optical pulse recovery signal under retrospective benchmark conditions on recorded data. It is NOT yet a prospective clinical trial. Real-world performance across motion artifacts, extreme skin tones (Fitzpatrick V–VI), low-light environments, and diverse mobile camera sensors requires formal prospective clinical validation."

### Slide 5: Quality Gates, Limitations, Biomarker Audit & Safety Boundaries
- **5 Automated Quality Gates:**
  1. *Lighting Adequacy:* Rejects underexposed (<40 lux) or severely back-lit frames.
  2. *Face Mesh Stability:* Requires MediaPipe landmark detection confidence >0.75.
  3. *Motion Jitter Guard:* Bounds inter-frame displacement; pauses during sudden yaw/pitch.
  4. *Temporal Window Buffer:* Requires minimum 20s (600 frames at 30 FPS) for cardiac stability.
  5. *Spectral SNR Confidence:* Welch PSD peak-to-noise ratio must exceed threshold before output.
- **Validated vs. Not Claimed:**
  - **✓ VALIDATED NOW (RGB Video):** Heart-rate estimation from facial video via optical capillary blood volume pulse (BVP).
  - **✗ STRICTLY NOT CLAIMED:** SpO2, Blood Pressure, Hemoglobin, Respiratory Rate, Stress, HbA1c, or Disease Diagnosis.
- **The Biomarker Audit Differentiator:**
  - Tested 10 biomarkers on 598 subjects. Deep models failed to beat demographic baselines (e.g. Hemoglobin video $r=0.44$ beaten by age/sex/BMI OLS $r=0.70$). The network learned subject ID shortcuts, not blood chemistry.
  - *Optical Physics Reality:* Blood chemistry requires narrow-band dual-wavelength NIR (660/940 nm). RGB Bayer filters conflate melanin with absorption.
- **Future Validation Pathway (4 Steps):**
  - *Step 1: Diverse Cohort Trial:* Prospective multi-center study across skin tones (Fitzpatrick I–VI), age brackets, and lighting environments.
  - *Step 2: Concurrent Clinical Reference:* Simultaneous validation against hospital-grade ECG and calibrated contact pulse oximeters.
  - *Step 3: Rigorous Statistical Analysis:* Bland-Altman 95% limits of agreement, mean bias profiling, and comprehensive failure taxonomy.
  - *Step 4: CDSCO SaMD Assessment:* Formal Software as a Medical Device (SaMD) regulatory filing and patient data privacy certification.

### Slide 6: Impact, Scalability, 3-Stage Roadmap & Team Conclusion
- **Winning Differentiator:** A working, evidence-backed heart-rate prototype (9.19 BPM held-out MAE) combined with an intellectually honest boundary around what RGB video cannot yet measure.
- **Healthcare Impact Hypothesis:**
  - *Accessible Pre-Consultation Screening:* Enables patients to provide an objective vital sign prior to a teleconsult without purchasing dedicated hardware.
  - *Rural Primary Health Centre (PHC) Potential:* Low-friction pre-screening tool for community health workers (ASHAs/ANMs) across 1.5 lakh Ayushman Arogya Mandirs.
  - *Zero Peripheral Hardware Cost:* Utilizes ubiquitous commodity smartphone and laptop cameras already available across India.
- **Three-Stage Development Roadmap:**
  - **[COMPLETED] Phase 1: Prototype & Audit:** Real-time webcam pipeline (30 FPS, 8 ROIs, 24 channels); HemoVisionBoundedTCN evaluated on MCD-rPPG (9.19 BPM MAE); systematic 10-biomarker audit exposing demographic confounds.
  - **[NEXT MILESTONE] Phase 2: Prospective Validation:** Multi-center prospective clinical study across Fitzpatrick skin tones; hardened automated quality gates & ambient light calibration; containerized web application for research evaluation.
  - **[FUTURE RESEARCH] Phase 3: Multi-Wavelength Optics:** Dual-wavelength NIR active illumination hardware research for SpO2; CDSCO Software as a Medical Device (SaMD) regulatory evaluation.
- **Research Disclaimer & Compliance Notice:** Explicit notice that HemoVision is an academic research prototype, does not provide medical diagnosis, and requires formal clinical trial verification and CDSCO clearance prior to real-world healthcare deployment.
- **Official References:** MoHFW eSanjeevani Overview, NHA ABDM Framework, CDSCO SaMD Guidance (2025), HemoVision Project Repository & Technical Report.

---

## 4. Prohibited Claims Check (Audited & Excluded)

| Potential Marketing / Inflated Claim | Actual Treatment in Deck & Submission Package | Empirical Justification |
|:---|:---:|:---|
| *"Hospital-grade ICU diagnostic monitor"* | **REJECTED / EXCLUDED** | Positioned strictly as pre-consultation tele-triage screening. |
| *"Measures SpO2, Blood Pressure, Hemoglobin"* | **REJECTED / EXCLUDED** | Proved to be identity memorization ($r=0.44$ vs $0.70$ OLS); requires NIR. |
| *"Live integration with eSanjeevani and ABDM"* | **FRAMED AS FUTURE OPPORTUNITY** | No API integration currently exists; framed as architectural interoperability. |
| *"Reduces hospital wait times by 80%"* | **REJECTED / EXCLUDED** | Zero empirical healthcare operations data in workspace. |
| *"Runs on Raspberry Pi / Edge AI hardware"* | **REJECTED / EXCLUDED** | No edge latency benchmarks run; tested on standard PC/laptop. |
| *"Evaluated on UBFC-rPPG / PURE"* | **CORRECTED TO MCD-rPPG** | Actual trained weights and splits are from 598-subject MCD-rPPG. |

---

## 5. Summary of Final Deliverables

1. **Final SIH Presentation (PPTX):** `HEMOVISION_SIH_2026_FINAL.pptx` (16:9 widescreen, 6 slides, SIH template compliant).
2. **Final SIH Submission (PDF):** `HEMOVISION_SIH_2026_FINAL.pdf` (Clean vector/raster export via PowerPoint COM).
3. **Rendered Slide Preview Images:** `sih_final_preview/slide_1.png` through `slide_6.png` (High-resolution 150 DPI).
4. **Evidence Audit Document:** `evidence_audit.md` (Complete verification table and 15-header traceability matrix).
5. **Grand Finale Pitch Script:** `grand_finale_demo_script.md` (Part 2: 3-minute slide-by-slide jury pitch).
6. **Live-Demo Script:** `grand_finale_demo_script.md` (Part 1: 90-second live demonstration script with intentional failure test).
7. **Judge Q&A Document:** `judge_qna.md` (7 tough judge questions + technical appendix covering privacy, bias, optics, and CDSCO SaMD).
