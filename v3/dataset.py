import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
import random

from v3.config import (
    TRAIN_CFG, TASK_COLUMNS, TASK_NAMES, TOTAL_CHANNELS, NUM_ROIS, CHANNELS_PER_ROI, CAMERA_TYPES
)

log = logging.getLogger("HemoVisionV3")

def _normalise_roi_channels(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != TOTAL_CHANNELS:
        raise ValueError(f"Expected ROI tensor [{TOTAL_CHANNELS}, T], received {tuple(x.shape)}.")
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
    return (x - mean) / std

def _normalise_bvp(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if y.ndim != 1:
        raise ValueError(f"Expected BVP vector [T], received {tuple(y.shape)}.")
    y = y - y.mean()
    return y / y.abs().amax().clamp_min(eps)

def _edge_pad_1d(y: torch.Tensor, target_length: int) -> torch.Tensor:
    if y.numel() >= target_length: return y[:target_length]
    pad_count = target_length - y.numel()
    return torch.cat((y, y[-1:].expand(pad_count)), dim=0)

def _edge_pad_2d(x: torch.Tensor, target_length: int) -> torch.Tensor:
    if x.shape[-1] >= target_length: return x[:, :target_length]
    pad_count = target_length - x.shape[-1]
    return torch.cat((x, x[:, -1:].expand(-1, pad_count)), dim=1)

class SubjectCameraSampler(Sampler):
    """
    Samples one random camera per (subject, step) group to prevent label duplication collapse.
    """
    def __init__(self, dataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        
        # Group indices by (subject_id, step)
        self.groups = defaultdict(list)
        for idx, record in enumerate(dataset.records):
            # Infer step from video key (e.g. '7_FullHDwebcam_before')
            step = "before" if "before" in record["video_key"] else "after"
            group_key = (record["subject_id"], step)
            self.groups[group_key].append(idx)
            
    def __iter__(self):
        indices = []
        for group_key, idxs in self.groups.items():
            # Pick exactly one camera for this subject/step
            indices.append(random.choice(idxs))
            
        if self.shuffle:
            random.shuffle(indices)
            
        return iter(indices)
        
    def __len__(self):
        return len(self.groups)

class HemoWaveformDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, db_csv_path: str | Path, window_frames: int = TRAIN_CFG.window_frames, random_window: bool = True):
        db_csv_path = Path(db_csv_path).resolve()
        
        self.window_frames = int(window_frames)
        self.random_window = bool(random_window)
        self._warned_reconciliations: set[str] = set()

        db = pd.read_csv(db_csv_path)
        db["video_key"] = db["video"].map(lambda value: Path(str(value)).stem)
        db_by_video = db.set_index("video_key", drop=False)
        dataset_root = db_csv_path.parent

        self.records: List[Dict] = []
        for manifest_row in manifest_df.to_dict(orient="records"):
            roi_cache_path = Path(manifest_row["roi_cache_path"]).resolve()
            video_key = roi_cache_path.stem
            if video_key not in db_by_video.index: continue
            
            db_row = db_by_video.loc[video_key]
            raw_sync_path = Path(str(db_row["ppg_sync"]))
            sync_path = raw_sync_path if raw_sync_path.is_absolute() else dataset_root / raw_sync_path
            sync_path = sync_path.resolve()

            if not roi_cache_path.exists() or not sync_path.exists(): continue

            biomarker_values = np.asarray([pd.to_numeric(db_row[column], errors="coerce") for column in TASK_COLUMNS], dtype=np.float32)

            # Determine camera type for one-hot encoding
            cam_type = 0  # default
            for i, ct in enumerate(CAMERA_TYPES):
                if ct in video_key:
                    cam_type = i
                    break

            self.records.append({
                "roi_cache_path": str(roi_cache_path),
                "ppg_sync_path": str(sync_path),
                "subject_id": str(manifest_row["subject_id"]),
                "split": str(manifest_row["split"]),
                "video_fps": float(manifest_row["video_fps"]),
                "biomarkers": biomarker_values,
                "video_key": video_key,
                "camera_type": cam_type
            })

        if not self.records:
            raise ValueError("Dataset contains zero valid joined records.")

        log.info("HemoWaveformDataset V3 ready: rows=%d, window=%d", len(self.records), self.window_frames)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_ppg_sync(path: str) -> torch.Tensor:
        raw = np.loadtxt(path, dtype=np.float32)
        raw = np.atleast_2d(raw)
        amplitudes = raw[:, 0]
        return torch.from_numpy(amplitudes.copy()).float()

    def _warn_once(self, video_key: str, message: str) -> None:
        if video_key not in self._warned_reconciliations:
            log.warning("%s | %s", video_key, message)
            self._warned_reconciliations.add(video_key)

    def _align_full_clip(self, x: torch.Tensor, bvp: torch.Tensor, video_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        source_frames = x.shape[-1]
        if bvp.numel() > source_frames:
            self._warn_once(video_key, f"ppg_sync rows={bvp.numel()} > ROI frames={source_frames}; truncating.")
            bvp = bvp[:source_frames]
        elif bvp.numel() < source_frames:
            self._warn_once(video_key, f"ppg_sync rows={bvp.numel()} < ROI frames={source_frames}; edge-padding.")
            bvp = _edge_pad_1d(bvp, source_frames)
        return x, bvp

    def _select_window(self, x: torch.Tensor, bvp: torch.Tensor, video_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_frames = x.shape[-1]
        if clip_frames >= self.window_frames:
            if self.random_window:
                start = int(torch.randint(low=0, high=clip_frames - self.window_frames + 1, size=(1,)).item())
            else:
                start = (clip_frames - self.window_frames) // 2
            end = start + self.window_frames
            return x[:, start:end], bvp[start:end]
        return _edge_pad_2d(x, self.window_frames), _edge_pad_1d(bvp, self.window_frames)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        cache = torch.load(record["roi_cache_path"], map_location="cpu")
        x = cache.get("signal") if isinstance(cache, dict) else None

        if x.ndim == 3 and tuple(x.shape[:2]) == (NUM_ROIS, CHANNELS_PER_ROI):
            x = x.reshape(TOTAL_CHANNELS, x.shape[-1])
        x = x.float()

        bvp = self._load_ppg_sync(record["ppg_sync_path"])
        x, bvp = self._align_full_clip(x, bvp, record["video_key"])
        x, bvp = self._select_window(x, bvp, record["video_key"])

        biomarkers = torch.from_numpy(record["biomarkers"].copy()).float()
        biomarker_mask = torch.isfinite(biomarkers).float()

        x_raw = x.clone() / 255.0  # Preserve DC absolute color scales

        # Phase 1: Camera one-hot encoding
        cam_onehot = torch.zeros(len(CAMERA_TYPES), dtype=torch.float32)
        cam_onehot[record["camera_type"]] = 1.0

        return {
            "x": _normalise_roi_channels(x),                 # Normalized for TCN (AC stream)
            "x_raw": x_raw,                                  # Raw absolute values (DC stream)
            "bvp_target": _normalise_bvp(bvp).unsqueeze(0),
            "biomarkers": biomarkers,
            "biomarker_mask": biomarker_mask,
            "video_fps": torch.tensor(record["video_fps"], dtype=torch.float32),
            "camera_onehot": cam_onehot,
            "subject_id": record["subject_id"],
            "video_key": record["video_key"],
        }

    def biomarker_normalisation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        values = np.stack([record["biomarkers"] for record in self.records], axis=0)
        means, stds = [], []
        for task_index, task_name in enumerate(TASK_NAMES):
            valid = values[:, task_index][np.isfinite(values[:, task_index])]
            means.append(float(valid.mean()))
            stds.append(float(max(valid.std(), 1e-6)))
        return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)
