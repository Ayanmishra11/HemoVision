# ==========================================
# CELL 14: HELD-OUT TEST SET EVALUATION
# Requires Cells 1--12.
# ==========================================

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def _masked_biomarker_mae(
    prediction_raw: torch.Tensor,
    target_raw: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-task absolute-error sums and valid-label counts."""
    if prediction_raw.shape != target_raw.shape or mask.shape != target_raw.shape:
        raise ValueError("Biomarker prediction, target, and mask shapes must match.")
    return ((prediction_raw - target_raw).abs() * mask).sum(dim=0), mask.sum(dim=0)


def _run_cell14_self_test() -> None:
    """Known-answer test for masked biomarker aggregation."""
    predicted = torch.tensor([[12.0, 4.0], [8.0, 99.0]])
    target = torch.tensor([[10.0, 1.0], [6.0, float("nan")]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    error_sum, count = _masked_biomarker_mae(predicted, target.nan_to_num(), mask)
    assert torch.equal(error_sum, torch.tensor([4.0, 3.0]))
    assert torch.equal(count, torch.tensor([2.0, 1.0]))
    assert torch.allclose(error_sum / count, torch.tensor([2.0, 3.0]))


_run_cell14_self_test()


def evaluate_held_out_test(
    model: HemoVisionBoundedTCN,
    loader: DataLoader,
    biomarker_mean: torch.Tensor,
    biomarker_std: torch.Tensor,
) -> Dict[str, Any]:
    """Evaluate the held-out test split using the same metrics as Cell 12."""
    model.eval()

    waveform_r_values = []
    hr_absolute_errors = []
    hr_squared_errors = []
    biomarker_error_sum = torch.zeros(len(TASK_NAMES), device=DEVICE)
    biomarker_valid_count = torch.zeros(len(TASK_NAMES), device=DEVICE)

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE, non_blocking=True)
            bvp_target = batch["bvp_target"].to(DEVICE, non_blocking=True)
            biomarkers_raw = batch["biomarkers"].to(DEVICE, non_blocking=True)
            biomarker_mask = batch["biomarker_mask"].to(DEVICE, non_blocking=True)
            fps = batch["video_fps"].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                prediction = model(x)

            waveform_r_values.append(
                batch_pearson_correlation(prediction["bvp"], bvp_target).detach()
            )

            predicted_hr = fft_hr_bpm(prediction["bvp"], fps)
            target_hr = fft_hr_bpm(bvp_target, fps)
            valid_hr = torch.isfinite(predicted_hr) & torch.isfinite(target_hr)

            if valid_hr.any():
                hr_error = predicted_hr[valid_hr] - target_hr[valid_hr]
                hr_absolute_errors.append(hr_error.abs().detach())
                hr_squared_errors.append(hr_error.square().detach())

            biomarker_prediction_raw = (
                prediction["biomarkers"].float() * biomarker_std + biomarker_mean
            )
            error_sum, valid_count = _masked_biomarker_mae(
                prediction_raw=biomarker_prediction_raw,
                target_raw=biomarkers_raw.nan_to_num(),
                mask=biomarker_mask,
            )
            biomarker_error_sum += error_sum
            biomarker_valid_count += valid_count

    if not waveform_r_values:
        raise RuntimeError("Test loader produced no batches.")

    per_task_mae = biomarker_error_sum / biomarker_valid_count.clamp_min(1.0)
    valid_tasks = biomarker_valid_count > 0
    tier_mae = defaultdict(list)

    for task_index, task_name in enumerate(TASK_NAMES):
        if bool(valid_tasks[task_index].item()):
            tier_mae[TASK_TIERS[task_name]].append(float(per_task_mae[task_index].item()))

    return {
        "pearson_r": float(torch.cat(waveform_r_values).mean().item()),
        "hr_mae_bpm": float(torch.cat(hr_absolute_errors).mean().item()),
        "hr_rmse_bpm": float(torch.cat(hr_squared_errors).mean().sqrt().item()),
        "per_task_mae": {
            task_name: (
                float(per_task_mae[index].item())
                if bool(valid_tasks[index].item())
                else float("nan")
            )
            for index, task_name in enumerate(TASK_NAMES)
        },
        "per_task_valid_count": {
            task_name: int(biomarker_valid_count[index].item())
            for index, task_name in enumerate(TASK_NAMES)
        },
        "tier_mae": {
            tier: float(np.mean(values)) for tier, values in tier_mae.items()
        },
    }


print("=" * 70)
print("HELD-OUT TEST SET EVALUATION")
print("=" * 70)

required_globals = {
    "HemoWaveformDataset", "HemoVisionBoundedTCN", "TRAIN_CFG", "TASK_NAMES",
    "TASK_TIERS", "DEVICE", "fft_hr_bpm", "batch_pearson_correlation",
    "load_manifest_split", "CFG",
}
missing_globals = sorted(name for name in required_globals if name not in globals())
if missing_globals:
    raise RuntimeError(
        "Run Cells 1--12 in order before Cell 14. Missing: " + ", ".join(missing_globals)
    )

checkpoint_path = Path(CFG.BASE_DATA_DIR) / TRAIN_CFG.checkpoint_name
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
for required_key in ("model_state_dict", "biomarker_mean", "biomarker_std", "task_names"):
    if required_key not in checkpoint:
        raise KeyError(f"Checkpoint missing required key: {required_key}")

if tuple(checkpoint["task_names"]) != tuple(TASK_NAMES):
    raise ValueError("Checkpoint task order differs from Cell 10 TASK_NAMES.")

with open(CFG.MANIFEST_PATH, "r", encoding="utf-8") as file:
    manifest_df = pd.DataFrame(json.load(file))

_, _, test_manifest = load_manifest_split(manifest_df)
test_dataset = HemoWaveformDataset(
    manifest_df=test_manifest,
    db_csv_path=Path(CFG.BASE_DATA_DIR) / "db.csv",
    window_frames=TRAIN_CFG.window_frames,
    random_window=False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=TRAIN_CFG.batch_size,
    shuffle=False,
    num_workers=TRAIN_CFG.num_workers,
    pin_memory=DEVICE.type == "cuda",
    persistent_workers=TRAIN_CFG.num_workers > 0,
)

expected_test_subjects = set(checkpoint.get("test_subject_ids", []))
observed_test_subjects = set(test_manifest["subject_id"].astype(str))
if expected_test_subjects and expected_test_subjects != observed_test_subjects:
    raise ValueError("Checkpoint test subjects do not match the manifest test split.")

model = HemoVisionBoundedTCN().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)

biomarker_mean = checkpoint["biomarker_mean"].float().to(DEVICE)
biomarker_std = checkpoint["biomarker_std"].float().to(DEVICE)

test_metrics = evaluate_held_out_test(
    model=model,
    loader=test_loader,
    biomarker_mean=biomarker_mean,
    biomarker_std=biomarker_std,
)

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Held-out test rows / subjects: {len(test_manifest)} / {test_manifest['subject_id'].nunique()}")
print(f"Waveform Pearson r: {test_metrics['pearson_r']:.4f}")
print(f"Heart-rate MAE:     {test_metrics['hr_mae_bpm']:.2f} BPM")
print(f"Heart-rate RMSE:    {test_metrics['hr_rmse_bpm']:.2f} BPM")

for tier in ("established", "moderate", "exploratory"):
    tier_tasks = [task for task in TASK_NAMES if TASK_TIERS[task] == tier]
    print(f"\n{tier.upper()} BIOMARKERS")
    print(f"Tier MAE: {test_metrics['tier_mae'].get(tier, float('nan')):.4f}")
    for task_name in tier_tasks:
        print(
            f"  {task_name:<22} MAE={test_metrics['per_task_mae'][task_name]:.4f} "
            f"(n={test_metrics['per_task_valid_count'][task_name]})"
        )

results_path = Path(CFG.BASE_DATA_DIR) / "hemovision_heldout_test_metrics.json"
with open(results_path, "w", encoding="utf-8") as file:
    json.dump(test_metrics, file, indent=2, allow_nan=False)

print(f"\nMetrics saved: {results_path}")
