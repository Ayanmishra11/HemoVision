# HemoVision — Grand Finale Presentation & Live Demo Package

This package equips the team for the **Smart India Hackathon 2026 Grand Finale Jury Presentation** in the Student Innovation (Software / MedTech) category.

---

## Part 1: 90-Second Live Demonstration Script

**Setup Prior to Entering the Room:**
- Laptop running `python hemovision_demo.py` with an external or built-in webcam.
- Test clips pre-loaded via `python hemovision_demo.py --test` as an instant offline fallback.
- Live contact pulse oximeter placed visibly on the demonstration table.

---

### Phase 1: Initiation, Consent & Calibration (0:00 – 0:25)

> **Presenter Action:** Stand comfortably in front of the camera, looking directly forward.
>
> **Spoken Script:**
> *"Respected Jury, before you is HemoVision running live on this standard laptop webcam. Notice what happened when I sat down:*
> 1. *First, the system requested explicit camera access with zero background data transmission.*
> 2. *Second, Google MediaPipe locked onto 468 facial landmark coordinates, segmenting 8 distinct capillary skin regions across my forehead, upper cheeks, lower cheeks, nose bridge, and chin.*
> 3. *Third, before computing a single heart beat, our automated quality gates evaluated ambient lighting adequacy and rigid head stability.*
> *This is not a mock UI or an aspirational render. It is a live, running PyTorch pipeline operating at 30 frames per second."*

---

### Phase 2: Live Optical Extraction & Verification (0:25 – 0:55)

> **Presenter Action:** Point to the live red BVP waveform rendering smoothly on the right side of the screen as the BPM counter settles.
>
> **Spoken Script:**
> *"Watch the waveform panel. The 24-channel RGB time-series is passed through our lightweight Bounded Temporal Convolutional Network—a 148,000-parameter architecture with dilated causal depthwise convolutions. It isolates the subtle pulsatile micro-expansions of facial capillaries.*
> *The output is filtered through a 4th-order Butterworth bandpass filter from 0.65 to 3.25 Hz, and Welch's Power Spectral Density extracts the dominant cardiac frequency.*
> *The system currently estimates my pulse at [e.g., 74 BPM]. Simultaneously, my contact pulse oximeter reads [e.g., 73 BPM]. Across high-quality, stable benchmark clips, our prototype reaches a 1.33 BPM Mean Absolute Error."*

---

### Phase 3: The Intentional Negative / Quality-Gate Test (0:55 – 1:15)

> **Presenter Action:** Intentionally turn head sharply left and right, or partially cover the webcam/block lighting.
>
> **Spoken Script:**
> *"Now, let me show you what truly separates a responsible medical AI innovation from a student hackathon toy: **our refusal to guess.***
> *If I turn my head sharply or block the light... watch the display:*
> *The system immediately pauses estimation and flags a 'Motion / Low Quality' gate warning. It refuses to invent numbers when optical signal-to-noise ratio is corrupted.*
> *In healthcare, returning 'Measurement Suspended: Quality Low' is infinitely safer than presenting a hallucinated vital sign."*

---

### Phase 4: Clinical Scope & Closing (1:15 – 1:30)

> **Presenter Action:** Step back to the podium and conclude with calm authority.
>
> **Spoken Script:**
> *"To conclude: HemoVision is a research prototype designed as a low-friction pre-consultation screening tool before telemedicine calls. It is not an emergency monitor, and it does not replace a clinical physician. We honestly measure what optical physics permits—heart rate—while rigorously rejecting unsupported claims. Thank you."*

---

## Part 2: 3-Minute Grand Finale Pitch Script (Aligned to the 6 Slides)

*Timing budget: ~30 seconds per slide.*

---

### Slide 1: Title & Positioning (0:00 – 0:30)
**Slide on Screen:** *Slide 1 — HemoVision: Contactless Heart-Rate Screening from Facial Video*

> *"Respected Members of the Jury, good morning.*
> *We are Team [Team Name], presenting **HemoVision** under the Student Innovation Software category.*
> *Our core thesis is simple, precise, and scientifically grounded:*
> ***We deliver a quality-gated heart-rate estimate from an ordinary camera before a remote teleconsult, while honestly identifying which physiological signals RGB video cannot yet measure.***
> *Unlike typical hackathon projects that showcase aspirational mockups, HemoVision is an end-to-end working prototype built in PyTorch. It tracks 8 anatomical facial regions using a 468-point mesh and achieves a 9.19 BPM Mean Absolute Error on a strictly held-out test split of 91 subjects.*
> *Every claim in this deck has an empirical number behind it."*

---

### Slide 2: The Real Problem & Workflow (0:30 – 1:00)
**Slide on Screen:** *Slide 2 — A Low-Friction First Step Before Remote Care*

> *"Consider the primary friction in telemedicine today. When a patient in rural India connects to a doctor on a platform like eSanjeevani, the consultation frequently starts without a single objective vital sign. Why? Because the patient does not own a digital pulse oximeter or ECG patch at home.*
> *HemoVision provides a low-friction first step before that consultation begins.*
> *The patient records a 15-to-20 second facial video on their existing phone or computer. Our automated pipeline verifies lighting and head stability. If—and only if—quality criteria pass, it estimates resting heart rate to give the attending doctor vital clinical context.*
> *If quality fails or an abnormality is detected, the protocol immediately escalates the patient to standard in-person medical evaluation.*
> *This creates an architectural opportunity for future interoperability with national digital health platforms, with zero initial hardware procurement."*

---

### Slide 3: Technical Innovation (1:00 – 1:30)
**Slide on Screen:** *Slide 3 — From Facial Video to a Quality-Gated Heart-Rate Estimate*

> *"How does the optical physics work?*
> *With each heartbeat, blood ejected from the left ventricle changes local hemoglobin concentration in facial capillary beds, creating minute, sub-pixel chromatic fluctuations.*
> *Instead of naive whole-face cropping, we track 468 3D landmarks to isolate 8 anatomically stable regions across the forehead, cheeks, and nose bridge, producing a synchronized 24-channel temporal signal.*
> *To reconstruct the blood volume pulse without gradient explosions on noisy webcams, we engineered **HemoVisionBoundedTCN**: a lightweight 148,000-parameter Temporal Convolutional Network using depthwise-separable convolutions and a tanh-bounded output.*
> *The signal is filtered through a 4th-order Butterworth bandpass filter spanning 0.65 to 3.25 Hz—matching physiological cardiac limits—and Welch's Power Spectral Density extracts the dominant heart rate peak."*

---

### Slide 4: Subject-Disjoint Evaluation (1:30 – 2:00)
**Slide on Screen:** *Slide 4 — Subject-Disjoint Evaluation Shows a Working Heart-Rate Signal*

> *"Now to the data. Many computer vision models claim high accuracy because their test sets leak subjects from training.*
> *We evaluated HemoVision on the **MCD-rPPG Clinical Dataset**—comprising 3,427 video clips across 598 subjects with synchronized ECG ground truth.*
> *Our test set is strictly **subject-disjoint**: 91 human beings who never appeared during model training.*
> *On this challenging held-out split, classical signal processing methods like POS and CHROM achieved 19.2 and 20.3 BPM MAE.*
> *Our model achieves **9.19 BPM MAE**—a **52% error reduction**.*
> *On stable, high-quality test clips, our MAE reaches **1.33 BPM**.*
> *Crucially, we state plainly what this does not prove: this is a retrospective benchmark on recorded data, not yet a prospective clinical trial across uncontrolled field environments."*

---

### Slide 5: Quality Gates & Biomarker Limits (2:00 – 2:30)
**Slide on Screen:** *Slide 5 — Quality Gates and Honest Limits Make the Prototype Safer*

> *"This brings us to HemoVision's defining strength: **intellectual and scientific honesty**.*
> *In published literature, you will see bold claims of measuring blood pressure, SpO2, and hemoglobin from standard smartphone cameras. We systematically audited these claims across all 11 biomarkers in the dataset.*
> *Our findings: for complex blood chemistry like hemoglobin, our deep model showed an apparent correlation of 0.44. But a simple baseline using only age, sex, and BMI achieved 0.70! The neural network was merely memorizing demographic identity proxies, not reading blood chemistry.*
> *Why? Because optical absorption of hemoglobin requires narrow-band dual-wavelength near-infrared optics. Standard RGB Bayer filters conflate melanin with absorption.*
> *Therefore, we explicitly state: **HemoVision validates heart rate only.** We do not claim blood pressure, SpO2, or disease diagnosis.*
> *Our roadmap defines the exact prospective trial and CDSCO SaMD path required before broader deployment."*

---

### Slide 6: Scale Opportunity & Roadmap (2:30 – 3:00)
**Slide on Screen:** *Slide 6 — A Credible Path From Prototype to Tele-Triage Support*

> *"Looking ahead, India has over 1.5 lakh Ayushman Bharat Health and Wellness Centres. While procuring physical ECG monitors for every village is capital-intensive, smartphones and webcams are already in the hands of healthcare workers.*
> *Our 3-stage roadmap is disciplined:*
> - *Phase 1 is complete: a working real-time prototype, 9.19 BPM held-out MAE, and a published biomarker audit.*
> - *Phase 2 is our next milestone: a multi-center prospective validation study across Fitzpatrick skin tones I through VI, automated quality-gate hardening, and pilot testing.*
> - *Phase 3: dual-wavelength NIR hardware research for validated SpO2, and formal CDSCO medical-device software compliance.*
> *HemoVision proves that student innovation can be technically deep, medically responsible, and ready for the national stage. Thank you, and we welcome your questions."*
