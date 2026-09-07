# %%
# ==========================================
# CELL 13: HELD-OUT TEST SET EVALUATION
# ==========================================

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

def _masked_biomarker_mae(prediction_raw: torch.Tensor, target_raw: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return ((prediction_raw - target_raw).abs() * mask).sum(dim=0), mask.sum(dim=0)

def evaluate_held_out_test(model: HemoVisionBoundedTCN, loader: DataLoader, biomarker_mean: torch.Tensor, biomarker_std: torch.Tensor) -> Dict[str, Any]:
    model.eval()
    waveform_r_values, hr_absolute_errors, hr_squared_errors = [], [], []
    biomarker_error_sum = torch.zeros(len(TASK_NAMES), device=DEVICE)
    biomarker_valid_count = torch.zeros(len(TASK_NAMES), device=DEVICE)

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE, non_blocking=True)
            x_raw = batch["x_raw"].to(DEVICE, non_blocking=True) # EXTRACT x_raw
            bvp_target = batch["bvp_target"].to(DEVICE, non_blocking=True)
            biomarkers_raw = batch["biomarkers"].to(DEVICE, non_blocking=True)
            biomarker_mask = batch["biomarker_mask"].to(DEVICE, non_blocking=True)
            fps = batch["video_fps"].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                prediction = model(x, x_raw) # PASS x_raw TO MODEL

            waveform_r_values.append(batch_pearson_correlation(prediction["bvp"], bvp_target).detach())
            predicted_hr = fft_hr_bpm(prediction["bvp"], fps)
            target_hr = fft_hr_bpm(bvp_target, fps)
            valid_hr = torch.isfinite(predicted_hr) & torch.isfinite(target_hr)

            if valid_hr.any():
                hr_error = predicted_hr[valid_hr] - target_hr[valid_hr]
                hr_absolute_errors.append(hr_error.abs().detach())
                hr_squared_errors.append(hr_error.square().detach())

            biomarker_prediction_raw = (prediction["biomarkers"].float() * biomarker_std + biomarker_mean)
            error_sum, valid_count = _masked_biomarker_mae(biomarker_prediction_raw, biomarkers_raw.nan_to_num(), biomarker_mask)
            biomarker_error_sum += error_sum
            biomarker_valid_count += valid_count

    per_task_mae = biomarker_error_sum / biomarker_valid_count.clamp_min(1.0)
    valid_tasks = biomarker_valid_count > 0

    tier_values = defaultdict(list)
    for task_index, task_name in enumerate(TASK_NAMES):
        if bool(valid_tasks[task_index].item()):
            tier_values[TASK_TIERS[task_name]].append(float(per_task_mae[task_index].item()))

    avg_r = float(torch.cat(waveform_r_values).mean().item())
    mae_hr = float(torch.cat(hr_absolute_errors).mean().item()) if hr_absolute_errors else float("nan")
    rmse_hr = float(torch.sqrt(torch.cat(hr_squared_errors).mean()).item()) if hr_squared_errors else float("nan")

    return {
        "pearson_r": avg_r,
        "hr_mae_bpm": mae_hr,
        "hr_rmse_bpm": rmse_hr,
        "biomarker_mae": float(per_task_mae[valid_tasks].mean().item()) if valid_tasks.any() else float("nan"),
        "per_task_mae": {n: float(per_task_mae[idx].item()) if bool(valid_tasks[idx].item()) else float("nan") for idx, n in enumerate(TASK_NAMES)},
        "tier_mae": {tier: float(np.mean(vals)) for tier, vals in tier_values.items()},
    }


if CHECKPOINT_PATH.exists():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint from Epoch {checkpoint['epoch']}")

    test_results = evaluate_held_out_test(model, test_loader, BIOMARKER_MEAN, BIOMARKER_STD)

    print("\n" + "=" * 50)
    print("HELD-OUT TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Waveform Pearson r : {test_results['pearson_r']:.4f}")
    print(f"Heart Rate MAE     : {test_results['hr_mae_bpm']:.2f} BPM")
    print(f"Heart Rate RMSE    : {test_results['hr_rmse_bpm']:.2f} BPM")
    print(f"Biomarker Mean MAE : {test_results['biomarker_mae']:.4f}")
    print("\nTier-wise MAE Breakdown:")
    for tier, mae_val in test_results["tier_mae"].items():
        print(f"  {tier:12s} : {mae_val:.4f}")