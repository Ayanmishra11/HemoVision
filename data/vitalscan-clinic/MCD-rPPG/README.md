---
license: cc-by-4.0
size_categories:
- 1K<n<10K
tags:
- medical
- video
- ecg
- ppg
task_categories:
- other
---

# MCD-rPPG: Multi-Camera Dataset for Remote Photoplethysmography

This repository contains the dataset from the paper ["Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation"](https://dl.acm.org/doi/10.1145/3746027.3758255).

The MCD-rPPG dataset is available on the Hugging Face Hub: [**MCD-rPPG Dataset**](https://huggingface.co/datasets/kyegorov/mcd_rppg)

The presented large-scale multimodal MCD-rPPG dataset is designed for remote photoplethysmography (rPPG) and health biomarker estimation from video. The dataset includes synchronized video recordings from three cameras at different angles, PPG and ECG signals, and extended health metrics (arterial blood pressure, oxygen saturation, stress level, etc.) for 600 subjects in both resting and post-exercise states.

We also provide an efficient multi-task neural network model that estimates the pulse wave signal and other biomarkers from facial video in real-time, even on a CPU.

## The MCD-rPPG Dataset

The dataset contains:
*   **3600 video recordings** (600 subjects × 2 states × 3 cameras)
*   **Synchronized PPG** (100 Hz) and ECG signals
*   **13 health biomarkers**: systolic/diastolic pressure, oxygen saturation, temperature, glucose, glycated hemoglobin, cholesterol, respiratory rate, arterial stiffness, stress level (PSM-25), age, sex, BMI.
*   **Multi-view videos**: frontal webcam, FullHD camcorder, mobile phone camera.
*   
## Fast Baseline Model

We propose an efficient multi-task model that:
*   Processes video in **real-time on a CPU** (up to 13% faster than leading models).
*   Estimates the **PPG signal** and **10+ health biomarkers** simultaneously.
*   Is lightweight (~4 MB) and uses domain-specific preprocessing suitable for low-power devices.

The model architecture combines domain-specific preprocessing (ROI selection on the face) with a convolutional network (1D Feature Pyramid Network).

## Code and Sample Usage

See GitHub repository [https://github.com/ksyegorov/mcd_rppg](https://github.com/ksyegorov/mcd_rppg)

To get started with the code and reproduce experiments, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ksyegorov/mcd_rppg.git
    cd mcd_rppg/
    ```

2.  **Install dependencies.** Using a virtual environment is recommended.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the notebooks** you are interested in (e.g., `train_SCNN_8roi_mcd_rppg.ipynb`) for training or reproducing experiments. Remember to download the MCD-rPPG dataset first.

## Results and Comparison

The tables below show key results of our model (Ours) compared to state-of-the-art (SOTA) alternatives. MAE (Mean Absolute Error) is calculated for the PPG signal and Heart Rate (HR).

**Table: Model performance comparison (MAE) in cross-dataset scenarios**
*(Summary of results from the paper)*

| Model          | ... | MCD-rPPG (HR MAE) | ... |
|----------------|-----|-------------------|-----|
| PBV            | ... | 15.37             | ... |
| OMIT           | ... | 4.78              | ... |
| POS            | ... | 3.80              | ... |
| PhysFormer     | ... | 4.08              | ... |
| **Ours**       | ... | **4.86**          | ... |

**Table: Performance for different camera views and inference speed**

| Model          | CPU Inference (s) | Size (Mb) | Frontal PPG MAE | Side PPG MAE |
|----------------|-------------------|-----------|-----------------|--------------|
| POS            | 0.26              | 0         | 0.87            | 1.25         |
| PhysFormer     | 0.93              | 28.4      | 0.46            | 0.97         |
| **Ours**       | **0.15**          | **3.9**   | 0.68            | 1.10         |

Complete results, including biomarker evaluation, are presented in the paper.

## Citation

If you use the MCD-rPPG dataset or code from this repository, please cite our work:

```bibtex
@inproceedings{10.1145/3746027.3758255,
author = {Egorov, Konstantin and Botman, Stepan and Blinov, Pavel and Zubkova, Galina and Ivaschenko, Anton and Kolsanov, Alexander and Savchenko, Andrey},
title = {Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746027.3758255},
doi = {10.1145/3746027.3758255},
abstract = {Progress in remote PhotoPlethysmoGraphy (rPPG) is limited by the critical issues of existing publicly available datasets: small size, privacy concerns with facial videos, and lack of diversity in conditions. The paper introduces a novel, comprehensive, large-scale multi-view video dataset for rPPG and health biomarkers estimation. Our dataset comprises 3600 synchronized video recordings from 600 subjects, captured under varied conditions (resting and post-exercise) using multiple consumer-grade cameras at different angles. To enable multimodal analysis of physiological states, each recording is paired with a 100 Hz PPG signal and extended health metrics, such as electrocardiogram, arterial blood pressure, biomarkers, temperature, oxygen saturation, respiratory rate, and stress level. Using this data, we train an efficient rPPG model and compare its quality with existing approaches in cross-dataset scenarios. The public release of our dataset and model should significantly speed up the progress in the development of AI medical assistants.},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {13053–13059},
numpages = {7},
keywords = {biosignals, rppg, telemedicine, video},
location = {Dublin, Ireland},
series = {MM '25}
}
```