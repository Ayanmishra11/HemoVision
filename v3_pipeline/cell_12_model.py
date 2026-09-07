# %%
# ==========================================
# CELL 12: SUBJECT-DISJOINT TRAINING LOOP
# ==========================================

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

SEED = 42
PROGRESS_EVERY_N_BATCHES = 25

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

print(f"Training device: {DEVICE} | AMP enabled: {USE_AMP}", flush=True)

def find_dataset_paths() -> Tuple[Path, Path, Path]:
    cwd = Path.cwd().resolve()
    root_candidates = [cwd / "data" / "vitalscan-clinic" / "MCD-rPPG", cwd / "MCD-rPPG", cwd]

    for root in root_candidates:
        db_path = root / "db.csv"
        if not db_path.exists(): continue

        manifest_candidates = [root / "manifest.json", root / "rois_v2" / "manifest.json"] + list(root.rglob("manifest.json"))
        seen_paths = set()
        
        for manifest_path in manifest_candidates:
            manifest_path = manifest_path.resolve()
            if manifest_path in seen_paths or not manifest_path.exists(): continue
            seen_paths.add(manifest_path)

            try:
                with open(manifest_path, "r", encoding="utf-8") as file:
                    manifest_content = json.load(file)
                if not isinstance(manifest_content, list) or not manifest_content: continue
                if {"roi_cache_path", "subject_id", "num_frames", "video_fps"}.issubset(set(manifest_content[0].keys())):
                    return (root.resolve(), manifest_path.resolve(), db_path.resolve())
            except (json.JSONDecodeError, OSError, TypeError):
                continue

    raise FileNotFoundError("Could not find db.csv plus a valid manifest.")

def load_manifest_split(manifest_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    test_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df

def fft_hr_bpm(waveform: torch.Tensor, fps: torch.Tensor, low_hz: float = BVP_BAND_LOW_HZ, high_hz: float = BVP_BAND_HIGH_HZ) -> torch.Tensor:
    waveform = waveform.squeeze(1)
    bpm_values = []
    for index in range(waveform.shape[0]):
        sample = waveform[index].float()
        sample_fps = float(fps[index].detach().float().item())
        sample = sample - sample.mean()
        spectrum = torch.fft.rfft(sample)
        power = spectrum.abs().square()
        frequencies = torch.fft.rfftfreq(sample.numel(), d=1.0 / sample_fps, device=sample.device)
        band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)

        if not band_mask.any():
            bpm_values.append(torch.tensor(float("nan"), device=sample.device))
            continue

        band_indices = torch.where(band_mask)[0]
        peak_index = band_indices[torch.argmax(power[band_indices])]
        bpm_values.append(frequencies[peak_index] * 60.0)

    return torch.stack(bpm_values)

def build_loader(dataset: HemoWaveformDataset, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=TRAIN_CFG.batch_size, shuffle=shuffle, num_workers=TRAIN_CFG.num_workers, pin_memory=USE_AMP, drop_last=shuffle)

def standardize_biomarkers(biomarkers_raw: torch.Tensor, biomarker_mask: torch.Tensor, biomarker_mean: torch.Tensor, biomarker_std: torch.Tensor) -> torch.Tensor:
    standardized = (biomarkers_raw - biomarker_mean) / biomarker_std
    return torch.where(biomarker_mask.bool(), standardized, torch.zeros_like(standardized))


def training_epoch(model, criterion, loader, optimizer, scaler, biomarker_mean, biomarker_std) -> Dict[str, float]:
    model.train()
    total_loss_sum, sample_count = 0.0, 0
    pearson_values = []
    total_batches = len(loader)

    for batch_index, batch in enumerate(loader, start=1):
        x = batch["x"].to(DEVICE, non_blocking=True)
        x_raw = batch["x_raw"].to(DEVICE, non_blocking=True)  # DC BYPASS
        bvp_target = batch["bvp_target"].to(DEVICE, non_blocking=True)
        biomarkers_raw = batch["biomarkers"].to(DEVICE, non_blocking=True)
        biomarker_mask = batch["biomarker_mask"].to(DEVICE, non_blocking=True)
        fps = batch["video_fps"].to(DEVICE, non_blocking=True)

        biomarker_target = standardize_biomarkers(biomarkers_raw, biomarker_mask, biomarker_mean, biomarker_std)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            prediction = model(x, x_raw)  # PASS DC BYPASS
            losses = criterion(prediction, bvp_target, biomarker_target, biomarker_mask, fps)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = x.shape[0]
        total_loss_sum += float(losses["total"].item()) * batch_size
        sample_count += batch_size
        pearson_values.append(losses["batch_pearson_r"].reshape(1))

        if batch_index % PROGRESS_EVERY_N_BATCHES == 0 or batch_index == total_batches:
            print(f"  Train batch {batch_index:04d}/{total_batches:04d} | loss={losses['total'].item():.5f} | r={losses['batch_pearson_r'].item():.4f}", flush=True)

    return {"loss": total_loss_sum / max(sample_count, 1), "pearson_r": float(torch.cat(pearson_values).mean().item())}


@torch.no_grad()
def validation_epoch(model, criterion, loader, biomarker_mean, biomarker_std) -> Dict[str, Any]:
    model.eval()
    total_loss_sum, sample_count = 0.0, 0
    pearson_values, hr_errors = [], []
    biomarker_absolute_error = torch.zeros(len(TASK_NAMES), device=DEVICE)
    biomarker_valid_count = torch.zeros(len(TASK_NAMES), device=DEVICE)
    total_batches = len(loader)

    for batch_index, batch in enumerate(loader, start=1):
        x = batch["x"].to(DEVICE, non_blocking=True)
        x_raw = batch["x_raw"].to(DEVICE, non_blocking=True) # DC BYPASS
        bvp_target = batch["bvp_target"].to(DEVICE, non_blocking=True)
        biomarkers_raw = batch["biomarkers"].to(DEVICE, non_blocking=True)
        biomarker_mask = batch["biomarker_mask"].to(DEVICE, non_blocking=True)
        fps = batch["video_fps"].to(DEVICE, non_blocking=True)

        biomarker_target = standardize_biomarkers(biomarkers_raw, biomarker_mask, biomarker_mean, biomarker_std)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            prediction = model(x, x_raw) # PASS DC BYPASS
            losses = criterion(prediction, bvp_target, biomarker_target, biomarker_mask, fps)

        batch_size = x.shape[0]
        total_loss_sum += float(losses["total"].item()) * batch_size
        sample_count += batch_size

        pearson_values.append(batch_pearson_correlation(prediction["bvp"], bvp_target).detach())
        predicted_hr = fft_hr_bpm(prediction["bvp"], fps)
        target_hr = fft_hr_bpm(bvp_target, fps)
        valid_hr = torch.isfinite(predicted_hr) & torch.isfinite(target_hr)
        if valid_hr.any():
            hr_errors.append((predicted_hr[valid_hr] - target_hr[valid_hr]).abs().detach())

        predicted_biomarkers_raw = prediction["biomarkers"].float() * biomarker_std + biomarker_mean
        absolute_error = (predicted_biomarkers_raw - biomarkers_raw).abs() * biomarker_mask
        biomarker_absolute_error += absolute_error.sum(dim=0)
        biomarker_valid_count += biomarker_mask.sum(dim=0)

        if batch_index % PROGRESS_EVERY_N_BATCHES == 0 or batch_index == total_batches:
            print(f"  Val   batch {batch_index:04d}/{total_batches:04d} | loss={losses['total'].item():.5f}", flush=True)

    per_task_mae = biomarker_absolute_error / biomarker_valid_count.clamp_min(1.0)
    valid_tasks = biomarker_valid_count > 0
    tier_values = defaultdict(list)
    for task_index, task_name in enumerate(TASK_NAMES):
        if bool(valid_tasks[task_index].item()):
            tier_values[TASK_TIERS[task_name]].append(float(per_task_mae[task_index].item()))

    return {
        "loss": total_loss_sum / max(sample_count, 1),
        "pearson_r": float(torch.cat(pearson_values).mean().item()),
        "hr_mae_bpm": float(torch.cat(hr_errors).mean().item()) if hr_errors else float("nan"),
        "biomarker_mae": float(per_task_mae[valid_tasks].mean().item()) if valid_tasks.any() else float("nan"),
        "per_task_mae": {t: float(per_task_mae[idx].item()) if bool(valid_tasks[idx].item()) else float("nan") for idx, t in enumerate(TASK_NAMES)},
        "tier_mae": {tier: float(np.mean(values)) for tier, values in tier_values.items()},
    }

DATASET_ROOT, MANIFEST_PATH, DB_CSV_PATH = find_dataset_paths()

with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
    manifest_df = pd.DataFrame(json.load(file))

train_manifest, validation_manifest, test_manifest = load_manifest_split(manifest_df)

print(f"Dataset root: {DATASET_ROOT}\nManifest: {MANIFEST_PATH}")

train_dataset = HemoWaveformDataset(manifest_df=train_manifest, db_csv_path=DB_CSV_PATH, window_frames=TRAIN_CFG.window_frames, random_window=True)
validation_dataset = HemoWaveformDataset(manifest_df=validation_manifest, db_csv_path=DB_CSV_PATH, window_frames=TRAIN_CFG.window_frames, random_window=False)
test_dataset = HemoWaveformDataset(manifest_df=test_manifest, db_csv_path=DB_CSV_PATH, window_frames=TRAIN_CFG.window_frames, random_window=False)

train_loader = build_loader(train_dataset, shuffle=True)
validation_loader = build_loader(validation_dataset, shuffle=False)
test_loader = build_loader(test_dataset, shuffle=False)

BIOMARKER_MEAN, BIOMARKER_STD = train_dataset.biomarker_normalisation()
BIOMARKER_MEAN, BIOMARKER_STD = BIOMARKER_MEAN.to(DEVICE), BIOMARKER_STD.to(DEVICE)

model = HemoVisionBoundedTCN().to(DEVICE)
criterion = HemoMultiTaskLoss().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=TRAIN_CFG.learning_rate, weight_decay=TRAIN_CFG.weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

CHECKPOINT_PATH = DATASET_ROOT / TRAIN_CFG.checkpoint_name
best_val_loss = float("inf")
training_history: List[Dict[str, Any]] = []

for epoch in range(1, TRAIN_CFG.epochs + 1):
    print(f"\n{'=' * 70}\nEpoch {epoch:03d}/{TRAIN_CFG.epochs:03d} starting\n{'=' * 70}", flush=True)

    train_metrics = training_epoch(model, criterion, train_loader, optimizer, scaler, BIOMARKER_MEAN, BIOMARKER_STD)
    validation_metrics = validation_epoch(model, criterion, validation_loader, BIOMARKER_MEAN, BIOMARKER_STD)
    training_history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})

    print(
        f"\nEpoch {epoch:03d}/{TRAIN_CFG.epochs:03d} complete | "
        f"train loss={train_metrics['loss']:.5f}, r={train_metrics['pearson_r']:.4f} | "
        f"val loss={validation_metrics['loss']:.5f}, r={validation_metrics['pearson_r']:.4f}, "
        f"HR-MAE={validation_metrics['hr_mae_bpm']:.2f} BPM, biomarker-MAE={validation_metrics['biomarker_mae']:.4f}",
        flush=True,
    )

    if validation_metrics["loss"] < best_val_loss:
        best_val_loss = validation_metrics["loss"]
        torch.save(
            {
                "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss, "train_config": TRAIN_CFG.__dict__, "task_names": TASK_NAMES,
                "biomarker_mean": BIOMARKER_MEAN.detach().cpu(), "biomarker_std": BIOMARKER_STD.detach().cpu(),
                "dataset_root": str(DATASET_ROOT), "manifest_path": str(MANIFEST_PATH), "validation_metrics": validation_metrics,
                "test_subject_ids": sorted(test_manifest["subject_id"].astype(str).unique().tolist()),
            },
            CHECKPOINT_PATH,
        )
        print(f"✓ Improved checkpoint saved: {CHECKPOINT_PATH} | val_loss={best_val_loss:.5f}", flush=True)

print(f"\nTraining complete. Best validation loss={best_val_loss:.5f}\nCheckpoint: {CHECKPOINT_PATH}", flush=True)