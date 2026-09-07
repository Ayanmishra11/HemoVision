import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Ordered output convention shared across V3 pipeline
TASK_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("respiratory", "respiratory", "established"),
    ("saturation", "saturation", "established"),
    ("upper_ap", "upper_ap", "moderate"),
    ("lower_ap", "lower_ap", "moderate"),
    ("hemoglobin", "hemoglobin", "exploratory"),
    ("bmi", "bmi", "exploratory"),
)

TASK_NAMES = tuple(task[0] for task in TASK_SPECS)
TASK_COLUMNS = tuple(task[1] for task in TASK_SPECS)
TASK_TIERS = {name: tier for name, _, tier in TASK_SPECS}

NUM_ROIS = 8
CHANNELS_PER_ROI = 3
TOTAL_CHANNELS = NUM_ROIS * CHANNELS_PER_ROI

CAMERA_TYPES = ["FullHDwebcam", "USBVideo", "IriunWebcam"]

@dataclass(frozen=True)
class HemoTrainingConfig:
    window_frames: int = 900  # Extended window (Phase 5)
    batch_size: int = 2       # Reduced batch size for 4GB VRAM
    grad_accumulation_steps: int = 8 # Effective batch size 16 (Phase 4)
    num_workers: int = 0
    epochs: int = 80          # Extended training (Phase 4)
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    
    # Model architecture (Phase 2)
    tcn_width: int = 64
    tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    dropout: float = 0.15
    
    # Tier-weighted CCC Loss (Phase 3)
    lambda_pearson: float = 1.0
    lambda_snr: float = 0.02
    lambda_biomarker: float = 1.0
    
    established_weight: float = 2.0
    moderate_weight: float = 1.0
    exploratory_weight: float = 0.3  # Reduced for exploratory/BMI
    
    checkpoint_name: str = "hemovision_v3_best.pt"

    def __post_init__(self):
        if self.window_frames < 2:
            raise ValueError("window_frames must be at least 2.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.tcn_width < 1:
            raise ValueError("tcn_width must be positive.")

TRAIN_CFG = HemoTrainingConfig()
