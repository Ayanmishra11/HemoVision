# %% [markdown]
# # HemoVision V3 — AC/DC-disentangled rPPG and biomarker experiment
#
# This notebook implements a falsifiable V3 experiment. It uses `ppg_sync` for
# camera-aligned waveform supervision, selects one camera per `(subject, state)`
# during training, preserves an explicit DC/static feature branch, and evaluates
# all biomarker claims against collapse and train-mean baselines.  It does not
# claim clinical validity; subject-disjoint held-out evaluation remains required.

# %%
# ==========================================
# CELL 1: IMPORTS, CONFIGURATION, AND CONSTANTS
# ==========================================
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, Sampler


def make_grad_scaler(enabled: bool):
    """Return an AMP scaler on both old and new PyTorch releases."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@dataclass(frozen=True)
class V3Config:
    data_root: Path = Path(r"C:\Users\VICTUS\HemoVision\data\vitalscan-clinic\MCD-rPPG")
    num_rois: int = 8
    channels_per_roi: int = 3
    band_low_hz: float = 0.65
    band_high_hz: float = 3.25
    window_frames: int = 900
    tcn_width: int = 64
    tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    attention_heads: int = 4
    temporal_stride: int = 4
    batch_size: int = 2
    accumulation_steps: int = 8
    epochs: int = 80
    early_stopping_patience: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    lambda_snr: float = 0.02
    gradient_diagnostic_batches: int = 4
    tier_prior_weights: Mapping[str, float] = field(
        default_factory=lambda: {"established": 2.0, "moderate": 1.0, "exploratory": 0.5}
    )
    seed: int = 42
    num_workers: int = 0

    @property
    def total_channels(self) -> int:
        return self.num_rois * self.channels_per_roi

    @property
    def db_csv(self) -> Path:
        return self.data_root / "db.csv"

    @property
    def manifest_path(self) -> Path:
        return self.data_root / "manifest.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_root / "checkpoints_v2" / "best_hemovision_v3.pt"


CFG = V3Config()
TASK_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("pulse", "pulse", "established"),
    ("respiratory", "respiratory", "established"),
    ("saturation", "saturation", "established"),
    ("upper_ap", "upper_ap", "moderate"),
    ("lower_ap", "lower_ap", "moderate"),
    ("hemoglobin", "hemoglobin", "exploratory"),
    ("glycated_hemoglobin", "glycated_hemoglobin", "exploratory"),
    ("cholesterol", "cholesterol", "exploratory"),
    ("rigidity", "rigidity", "exploratory"),
    ("stress", "stress", "exploratory"),
    ("bmi", "bmi", "exploratory"),
)
TASK_NAMES = tuple(name for name, _, _ in TASK_SPECS)
TASK_COLUMNS = tuple(column for _, column, _ in TASK_SPECS)
TASK_TIERS = {name: tier for name, _, tier in TASK_SPECS}
CAMERA_NAMES = ("FullHDwebcam", "USBVideo", "IriunWebcam")
CAMERA_TO_INDEX = {name: index for index, name in enumerate(CAMERA_NAMES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.seed)
    torch.backends.cudnn.benchmark = True

assert CFG.band_low_hz < CFG.band_high_hz
assert CFG.total_channels == 24
assert CFG.tcn_width % CFG.attention_heads == 0
print(f"Device={DEVICE}; V3 config={asdict(CFG)}")

# %%
# ==========================================
# CELL 2: MANIFEST JOIN AND SPLIT INTEGRITY
# ==========================================
def video_key(value: object) -> str:
    return Path(str(value)).stem


def load_subject_disjoint_splits(cfg: V3Config = CFG) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join Cell-6 manifest to db.csv and preserve its existing split column."""
    db = pd.read_csv(cfg.db_csv).copy()
    with open(cfg.manifest_path, "r", encoding="utf-8") as handle:
        manifest = pd.DataFrame(json.load(handle))
    required_manifest = {"roi_cache_path", "subject_id", "condition", "split", "video_fps"}
    required_db = {"video", "ppg_sync", "patient_id", "camera", "step", *TASK_COLUMNS}
    if missing := required_manifest - set(manifest.columns):
        raise KeyError(f"Manifest missing {sorted(missing)}")
    if missing := required_db - set(db.columns):
        raise KeyError(f"db.csv missing {sorted(missing)}")
    manifest["video_key"] = manifest["roi_cache_path"].map(video_key)
    db["video_key"] = db["video"].map(video_key)
    joined = manifest.merge(db, on="video_key", how="left", validate="one_to_one", suffixes=("_manifest", "_db"))
    if joined["ppg_sync"].isna().any():
        raise ValueError("Manifest contains cache files that have no db.csv/ppg_sync row.")
    if not (joined["subject_id"].astype(str) == joined["patient_id"].astype(str)).all():
        raise ValueError("Manifest and db.csv subject IDs disagree.")
    joined["camera_type"] = joined["camera"].astype(str)
    frames = tuple(joined[joined["split"] == split].reset_index(drop=True) for split in ("train", "val", "test"))
    if any(frame.empty for frame in frames):
        raise ValueError("train, val, and test must all be non-empty in the Cell-6 manifest.")
    subject_sets = [set(frame["subject_id"].astype(str)) for frame in frames]
    assert subject_sets[0].isdisjoint(subject_sets[1])
    assert subject_sets[0].isdisjoint(subject_sets[2])
    assert subject_sets[1].isdisjoint(subject_sets[2])
    return frames


def _run_split_self_test() -> None:
    manifest = pd.DataFrame({"roi_cache_path": ["a.pt"], "subject_id": ["1"], "condition": ["before"], "split": ["train"], "video_fps": [30.]})
    assert video_key(manifest.iloc[0]["roi_cache_path"]) == "a"


_run_split_self_test()
train_manifest, val_manifest, test_manifest = load_subject_disjoint_splits()
print("Rows / subjects:", [(name, len(frame), frame.subject_id.nunique()) for name, frame in zip(("train", "val", "test"), (train_manifest, val_manifest, test_manifest))])

# %%
# ==========================================
# CELL 3: V3 DATASET — ALIGNED PPG, AC + DC FEATURES
# ==========================================
def _load_ppg_sync(path: Path) -> torch.Tensor:
    raw = np.atleast_2d(np.loadtxt(path, dtype=np.float32))
    if raw.shape[0] == 0 or raw.shape[1] < 1 or not np.isfinite(raw[:, 0]).all():
        raise ValueError(f"Invalid ppg_sync file: {path}")
    return torch.from_numpy(raw[:, 0].copy())


def _normalise_ac(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)


def _normalise_bvp(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y = y - y.mean()
    return y / y.abs().amax().clamp_min(eps)


def _dc_features(raw_roi: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return 80 static descriptors: DC, log-ratios, AC/DC, and luminance."""
    if raw_roi.ndim != 3 or raw_roi.shape[:2] != (CFG.num_rois, CFG.channels_per_roi):
        raise ValueError(f"Expected [8,3,T], received {tuple(raw_roi.shape)}")
    dc = raw_roi.mean(dim=-1).clamp_min(eps)                 # [8,3]
    ac_std = raw_roi.std(dim=-1, unbiased=False)             # [8,3]
    r, g, b = dc[:, 0], dc[:, 1], dc[:, 2]
    log_ratios = torch.stack((torch.log(r / g), torch.log(r / b), torch.log(g / b)), dim=1)
    acdc = ac_std / dc
    luminance = (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(1)
    return torch.cat((dc.flatten(), log_ratios.flatten(), acdc.flatten(), luminance.flatten()))


class V3WaveformDataset(Dataset):
    """Loads one camera recording with matched ppg_sync and static DC descriptors."""
    def __init__(self, frame: pd.DataFrame, cfg: V3Config = CFG, random_window: bool = False):
        self.cfg, self.random_window = cfg, random_window
        self.records = frame.to_dict(orient="records")
        if not self.records:
            raise ValueError("Dataset has no records.")

    def __len__(self) -> int:
        return len(self.records)

    def session_key(self, index: int) -> Tuple[str, str]:
        record = self.records[index]
        return str(record["subject_id"]), str(record["condition"])

    def biomarker_normalisation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = np.asarray([[pd.to_numeric(record[column], errors="coerce") for column in TASK_COLUMNS] for record in self.records], dtype=np.float32)
        return torch.tensor(np.nanmean(labels, axis=0)), torch.tensor(np.maximum(np.nanstd(labels, axis=0), 1e-6))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        cache = torch.load(Path(record["roi_cache_path"]), map_location="cpu", weights_only=True)
        raw = cache.get("signal")
        if not isinstance(raw, torch.Tensor) or raw.ndim != 3 or tuple(raw.shape[:2]) != (self.cfg.num_rois, self.cfg.channels_per_roi):
            raise ValueError(f"Invalid V2 ROI cache: {record['roi_cache_path']}")
        raw = raw.float()
        bvp_path = self.cfg.data_root / str(record["ppg_sync"])
        bvp = _load_ppg_sync(bvp_path)
        frames = raw.shape[-1]
        bvp = bvp[:frames] if bvp.numel() >= frames else torch.cat((bvp, bvp[-1:].expand(frames - bvp.numel())))
        if frames >= self.cfg.window_frames:
            start = int(torch.randint(0, frames - self.cfg.window_frames + 1, (1,)).item()) if self.random_window else (frames - self.cfg.window_frames) // 2
            end = start + self.cfg.window_frames
            raw, bvp = raw[:, :, start:end], bvp[start:end]
        else:
            pad = self.cfg.window_frames - frames
            raw = torch.cat((raw, raw[:, :, -1:].expand(-1, -1, pad)), dim=-1)
            bvp = torch.cat((bvp, bvp[-1:].expand(pad)))
        biomarkers = torch.tensor([pd.to_numeric(record[column], errors="coerce") for column in TASK_COLUMNS], dtype=torch.float32)
        camera = torch.zeros(len(CAMERA_NAMES), dtype=torch.float32)
        if str(record["camera_type"]) not in CAMERA_TO_INDEX:
            raise ValueError(f"Unknown camera type {record['camera_type']!r}")
        camera[CAMERA_TO_INDEX[str(record["camera_type"])]] = 1.0
        return {"x_ac": _normalise_ac(raw.flatten(0, 1)), "x_dc": _dc_features(raw), "camera": camera,
                "bvp_target": _normalise_bvp(bvp).unsqueeze(0), "biomarkers": biomarkers,
                "biomarker_mask": torch.isfinite(biomarkers).float(), "video_fps": torch.tensor(float(record["video_fps"])),
                "subject_id": str(record["subject_id"]), "condition": str(record["condition"])}


def _run_feature_self_test() -> None:
    raw = torch.full((8, 3, 10), 10.0)
    raw[:, 0] = 20.0
    features = _dc_features(raw)
    assert features.shape == (80,) and torch.isfinite(features).all()
    assert _normalise_ac(raw.flatten(0, 1)).abs().max() < 1e-5


_run_feature_self_test()


class OneCameraPerSessionSampler(Sampler[int]):
    """At each epoch, draw exactly one camera for every `(subject, condition)` group."""
    def __init__(self, dataset: V3WaveformDataset, seed: int = CFG.seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0
        self.groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for index in range(len(dataset)):
            self.groups[dataset.session_key(index)].append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        selected = [rng.choice(indices) for _, indices in sorted(self.groups.items())]
        rng.shuffle(selected)
        return iter(selected)

    def __len__(self) -> int:
        return len(self.groups)


def _run_sampler_self_test() -> None:
    class FakeDataset:
        def __len__(self): return 4
        def session_key(self, index): return [("1", "before"), ("1", "before"), ("2", "after"), ("2", "after")][index]
    sampler = OneCameraPerSessionSampler(FakeDataset(), seed=1)
    indices = list(sampler)
    assert len(indices) == 2 and len({sampler.dataset.session_key(i) for i in indices}) == 2


_run_sampler_self_test()

# %%
# ==========================================
# CELL 4: V3 MODEL — AC TCN + DC STATIC BYPASS
# ==========================================
class DepthwiseBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels, bias=False),
                                 nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout))
        self.activation = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class AttentionPool(nn.Module):
    """Query-to-temporal attention preserves sequence content without full T² attention."""
    def __init__(self, channels: int, heads: int, stride: int):
        super().__init__()
        self.downsample = nn.Conv1d(channels, channels, kernel_size=stride, stride=stride, bias=False)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, channels) * 0.02)
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.downsample(features).transpose(1, 2)
        query = self.query.expand(tokens.shape[0], -1, -1)
        return self.attention(query, tokens, tokens, need_weights=False)[0].squeeze(1)


class V3BiomarkerHead(nn.Module):
    def __init__(self, width: int, dc_dim: int, camera_dim: int, attention_heads: int, temporal_stride: int, dropout: float = 0.15):
        super().__init__()
        self.attention_pool = AttentionPool(width, attention_heads, temporal_stride)
        self.respiratory_branch = nn.Sequential(nn.Conv1d(width, width // 2, kernel_size=31, padding=15, bias=False), nn.BatchNorm1d(width // 2), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.dc_encoder = nn.Sequential(nn.LayerNorm(dc_dim), nn.Linear(dc_dim, width), nn.GELU(), nn.Dropout(dropout))
        fused_dim = width + width // 2 + width + camera_dim
        self.heads = nn.ModuleDict({name: nn.Sequential(nn.Linear(fused_dim, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1)) for name in TASK_NAMES})
    def forward(self, temporal: torch.Tensor, dc: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
        fused = torch.cat((self.attention_pool(temporal), self.respiratory_branch(temporal), self.dc_encoder(dc), camera), dim=1)
        return torch.cat([self.heads[name](fused) for name in TASK_NAMES], dim=1)


class HemoVisionV3(nn.Module):
    def __init__(self, cfg: V3Config = CFG, dc_dim: int = 80):
        super().__init__()
        self.cfg = cfg
        self.input_projection = nn.Sequential(nn.Conv1d(cfg.total_channels, cfg.tcn_width, 1, bias=False), nn.BatchNorm1d(cfg.tcn_width), nn.GELU())
        self.temporal_blocks = nn.Sequential(*[DepthwiseBlock(cfg.tcn_width, dilation) for dilation in cfg.tcn_dilations])
        self.waveform_head = nn.Sequential(nn.Conv1d(cfg.tcn_width, cfg.tcn_width, 3, padding=1, bias=False), nn.BatchNorm1d(cfg.tcn_width), nn.GELU(), nn.Conv1d(cfg.tcn_width, 1, 1), nn.Tanh())
        self.biomarker_head = V3BiomarkerHead(cfg.tcn_width, dc_dim, len(CAMERA_NAMES), cfg.attention_heads, cfg.temporal_stride)
    def forward(self, x_ac: torch.Tensor, x_dc: torch.Tensor, camera: torch.Tensor) -> Dict[str, torch.Tensor]:
        temporal = self.temporal_blocks(self.input_projection(x_ac))
        return {"bvp": self.waveform_head(temporal), "biomarkers": self.biomarker_head(temporal, x_dc, camera), "temporal": temporal}


def _run_model_self_test() -> None:
    model = HemoVisionV3()
    output = model(torch.randn(2, CFG.total_channels, 128), torch.randn(2, 80), torch.eye(3)[:2])
    assert output["bvp"].shape == (2, 1, 128) and output["biomarkers"].shape == (2, len(TASK_NAMES))


_run_model_self_test()

# %%
# ==========================================
# CELL 5: CCC LOSS, DYNAMIC WEIGHTING, GRADIENT DIAGNOSTIC
# ==========================================
def batch_pearson(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p, t = prediction.float().flatten(1), target.float().flatten(1)
    p, t = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
    return (p * t).sum(1) / torch.sqrt(p.square().sum(1).clamp_min(eps) * t.square().sum(1).clamp_min(eps)).clamp_min(eps)


def frequency_concentration_loss(prediction: torch.Tensor, target: torch.Tensor, fps: torch.Tensor, cfg: V3Config = CFG) -> torch.Tensor:
    losses = []
    for index, (p, t) in enumerate(zip(prediction.squeeze(1), target.squeeze(1))):
        p, t = p.float() - p.float().mean(), t.float() - t.float().mean()
        frequencies = torch.fft.rfftfreq(t.numel(), d=1.0 / float(fps[index]), device=t.device)
        band = (frequencies >= cfg.band_low_hz) & (frequencies <= cfg.band_high_hz)
        if band.sum() < 3: continue
        pp, tp = torch.fft.rfft(p).abs().square(), torch.fft.rfft(t).abs().square()
        peak = torch.where(band)[0][torch.argmax(tp[band])]
        lo, hi = max(0, int(peak) - 1), min(pp.numel(), int(peak) + 2)
        losses.append(-torch.log((pp[lo:hi].sum() + 1e-8) / (pp[band].sum() + 1e-8)))
    return torch.stack(losses).mean() if losses else prediction.new_zeros((), dtype=torch.float32)


def masked_ccc_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    task_losses, valid_tasks = [], []
    for index in range(target.shape[1]):
        valid = mask[:, index].bool()
        if valid.sum() < 2: continue
        p, t = prediction[valid, index].float(), target[valid, index].float()
        covariance = ((p - p.mean()) * (t - t.mean())).mean()
        ccc = 2 * covariance / (p.var(unbiased=False) + t.var(unbiased=False) + (p.mean() - t.mean()).square() + eps)
        task_losses.append(1 - ccc)
        valid_tasks.append(index)
    if not task_losses: return prediction.new_zeros((), dtype=torch.float32), torch.empty(0, dtype=torch.long, device=prediction.device)
    return torch.stack(task_losses).mean(), torch.tensor(valid_tasks, device=prediction.device)


class DynamicHemoMultiTaskLoss(nn.Module):
    """Kendall weighting across waveform and three biomarker tiers; masks stay explicit."""
    def __init__(self, cfg: V3Config = CFG):
        super().__init__()
        self.cfg = cfg
        self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in ("waveform", "established", "moderate", "exploratory")})
    def _weighted(self, key: str, loss: torch.Tensor, prior: float = 1.0) -> torch.Tensor:
        log_var = self.log_vars[key].clamp(-5, 5)
        return prior * (torch.exp(-log_var) * loss + log_var)
    def forward(self, prediction: Dict[str, torch.Tensor], bvp_target: torch.Tensor, biomarker_target: torch.Tensor, biomarker_mask: torch.Tensor, fps: torch.Tensor) -> Dict[str, torch.Tensor]:
        waveform = (1 - batch_pearson(prediction["bvp"], bvp_target)).mean() + self.cfg.lambda_snr * frequency_concentration_loss(prediction["bvp"], bvp_target, fps, self.cfg)
        total, details = self._weighted("waveform", waveform), {"waveform": waveform.detach(), "waveform_objective": waveform}
        for tier in ("established", "moderate", "exploratory"):
            indices = [i for i, name in enumerate(TASK_NAMES) if TASK_TIERS[name] == tier]
            tier_loss, _ = masked_ccc_loss(prediction["biomarkers"][:, indices], biomarker_target[:, indices], biomarker_mask[:, indices])
            # A small masked MSE term keeps CCC stable for small effective batches.
            valid = biomarker_mask[:, indices]
            mse = (((prediction["biomarkers"][:, indices] - biomarker_target[:, indices]).float().square() * valid).sum() / valid.sum().clamp_min(1.0))
            tier_loss = tier_loss + 0.1 * mse
            total = total + self._weighted(tier, tier_loss, self.cfg.tier_prior_weights[tier])
            details[tier] = tier_loss.detach()
            details[f"{tier}_objective"] = tier_loss
        details["total"] = total
        details["selection_score"] = (
            waveform
            + sum(self.cfg.tier_prior_weights[tier] * details[f"{tier}_objective"] for tier in ("established", "moderate", "exploratory"))
        ).detach()
        return details


def objective_gradient_norms(model: HemoVisionV3, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Separate autograd gradients entering the shared temporal trunk; no accumulation."""
    parameters = tuple(model.input_projection.parameters()) + tuple(model.temporal_blocks.parameters())
    result = {}
    for name in ("waveform", "established", "moderate", "exploratory"):
        gradients = torch.autograd.grad(losses[f"{name}_objective"], parameters, retain_graph=True, allow_unused=True)
        result[name] = math.sqrt(sum(float(g.detach().float().square().sum()) for g in gradients if g is not None))
    return result


def _run_loss_self_test() -> None:
    target = torch.tensor([[0., 1., 2.], [1., 2., 3.], [2., 3., 4.]])
    loss_equal, _ = masked_ccc_loss(target, target, torch.ones_like(target))
    loss_flat, _ = masked_ccc_loss(torch.zeros_like(target), target, torch.ones_like(target))
    assert loss_equal < 1e-6 and loss_flat > 0.9


_run_loss_self_test()

# %%
# ==========================================
# CELL 6: TRAINING AND HELD-OUT EVALUATION
# ==========================================
def standardize(labels: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.bool(), (labels - mean) / std, torch.zeros_like(labels))


def fft_hr_bpm(waveform: torch.Tensor, fps: torch.Tensor, cfg: V3Config = CFG) -> torch.Tensor:
    result = []
    for index, wave in enumerate(waveform.squeeze(1)):
        wave = wave.float() - wave.float().mean()
        frequency = torch.fft.rfftfreq(wave.numel(), d=1 / float(fps[index]), device=wave.device)
        band = (frequency >= cfg.band_low_hz) & (frequency <= cfg.band_high_hz)
        power = torch.fft.rfft(wave).abs().square()
        result.append(frequency[torch.where(band)[0][power[band].argmax()]] * 60)
    return torch.stack(result)


def make_loader(dataset: V3WaveformDataset, sampler: Sampler[int] | None = None) -> DataLoader:
    return DataLoader(dataset, batch_size=CFG.batch_size, sampler=sampler, shuffle=sampler is None and dataset.random_window,
                      num_workers=CFG.num_workers, pin_memory=USE_AMP, persistent_workers=CFG.num_workers > 0)


def run_epoch(model: HemoVisionV3, criterion: DynamicHemoMultiTaskLoss, loader: DataLoader, mean: torch.Tensor, std: torch.Tensor, optimizer: AdamW | None = None, scaler: Any | None = None) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    totals, gradient_totals, n, diagnostic_batches = defaultdict(float), defaultdict(float), 0, 0
    for batch_index, batch in enumerate(loader):
        x_ac, x_dc, camera = (batch[key].to(DEVICE, non_blocking=True) for key in ("x_ac", "x_dc", "camera"))
        bvp, labels, mask, fps = (batch[key].to(DEVICE, non_blocking=True) for key in ("bvp_target", "biomarkers", "biomarker_mask", "video_fps"))
        target = standardize(labels, mask, mean, std)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            output = model(x_ac, x_dc, camera)
            losses = criterion(output, bvp, target, mask, fps)
            loss = losses["total"] / CFG.accumulation_steps
        if training:
            if batch_index < CFG.gradient_diagnostic_batches:
                for key, value in objective_gradient_norms(model, losses).items():
                    gradient_totals[key] += value
                diagnostic_batches += 1
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            if (batch_index + 1) % CFG.accumulation_steps == 0 or batch_index + 1 == len(loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
        size = x_ac.shape[0]
        n += size
        for key, value in losses.items(): totals[key] += float(value.detach()) * size
    result = {key: value / max(n, 1) for key, value in totals.items()}
    if diagnostic_batches:
        result.update({f"trunk_grad_{key}": value / diagnostic_batches for key, value in gradient_totals.items()})
    return result


@torch.no_grad()
def evaluate(model: HemoVisionV3, loader: DataLoader, mean: torch.Tensor, std: torch.Tensor) -> Dict[str, Any]:
    model.eval(); predicted_bio, target_bio, masks, r_values, hr_error, identities = [], [], [], [], [], []
    for batch in loader:
        x_ac, x_dc, camera = (batch[key].to(DEVICE) for key in ("x_ac", "x_dc", "camera"))
        output = model(x_ac, x_dc, camera)
        r_values.extend(batch_pearson(output["bvp"], batch["bvp_target"].to(DEVICE)).cpu().tolist())
        hr_error.extend((fft_hr_bpm(output["bvp"], batch["video_fps"].to(DEVICE)) - fft_hr_bpm(batch["bvp_target"].to(DEVICE), batch["video_fps"].to(DEVICE))).abs().cpu().tolist())
        predicted_bio.append((output["biomarkers"].float() * std + mean).cpu())
        target_bio.append(batch["biomarkers"]); masks.append(batch["biomarker_mask"])
        identities.extend(zip(batch["subject_id"], batch["condition"]))
    p, t, m = torch.cat(predicted_bio), torch.cat(target_bio), torch.cat(masks).bool()

    def task_metrics(prediction: torch.Tensor, target: torch.Tensor, task_index: int) -> Dict[str, float | bool]:
        baseline_mae = float((target - mean[task_index].cpu()).abs().mean())
        mae = float((prediction - target).abs().mean())
        return {
            "mae": mae,
            "train_mean_baseline_mae": baseline_mae,
            "beats_train_mean_by_15pct": mae < baseline_mae * 0.85,
            "pearson_r": float(torch.corrcoef(torch.stack((prediction, target)))[0, 1]) if prediction.std() > 1e-8 and target.std() > 1e-8 else float("nan"),
            "variance_ratio": float(prediction.std(unbiased=False) / target.std(unbiased=False).clamp_min(1e-8)),
        }

    per_task_video, per_task_session = {}, {}
    for index, name in enumerate(TASK_NAMES):
        valid = m[:, index]
        pv, tv = p[valid, index], t[valid, index]
        per_task_video[name] = task_metrics(pv, tv, index)

        task_frame = pd.DataFrame({
            "subject_id": [identities[i][0] for i in torch.where(valid)[0].tolist()],
            "condition": [identities[i][1] for i in torch.where(valid)[0].tolist()],
            "prediction": pv.numpy(), "target": tv.numpy(),
        })
        grouped = task_frame.groupby(["subject_id", "condition"], as_index=False).agg(prediction=("prediction", "mean"), target=("target", "first"), unique_targets=("target", "nunique"))
        if (grouped["unique_targets"] > 1).any():
            raise ValueError(f"{name}: inconsistent labels across camera views within a subject/session.")
        per_task_session[name] = task_metrics(torch.tensor(grouped["prediction"].to_numpy()), torch.tensor(grouped["target"].to_numpy()), index)

    return {
        "waveform_pearson_r": float(np.mean(r_values)), "hr_mae_bpm": float(np.mean(hr_error)),
        "per_task": per_task_session, "per_task_session": per_task_session,
        "per_task_video": per_task_video,
        "biomarker_primary_unit": "subject-session (mean prediction across available camera views)",
    }


RUN_TRAINING = False  # Set True only after confirming the self-tests and dataset summary above.
train_dataset, val_dataset, test_dataset = V3WaveformDataset(train_manifest, random_window=True), V3WaveformDataset(val_manifest), V3WaveformDataset(test_manifest)
train_sampler = OneCameraPerSessionSampler(train_dataset)
train_loader, val_loader, test_loader = make_loader(train_dataset, train_sampler), make_loader(val_dataset), make_loader(test_dataset)
mean, std = (value.to(DEVICE) for value in train_dataset.biomarker_normalisation())
model, criterion = HemoVisionV3().to(DEVICE), DynamicHemoMultiTaskLoss().to(DEVICE)
optimizer = AdamW(list(model.parameters()) + list(criterion.parameters()), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
scaler = make_grad_scaler(enabled=USE_AMP)

if RUN_TRAINING:
    best, stale = float("inf"), 0
    for epoch in range(CFG.epochs):
        train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(model, criterion, train_loader, mean, std, optimizer, scaler)
        val_metrics = run_epoch(model, criterion, val_loader, mean, std)
        scheduler.step(epoch + 1)
        # Dynamic Kendall terms can decrease by changing learned uncertainty; select on raw objectives instead.
        score = val_metrics["selection_score"]
        gradient_report = " | ".join(f"{key[11:]}={value:.3g}" for key, value in train_metrics.items() if key.startswith("trunk_grad_"))
        print(f"epoch={epoch+1:03d} train={train_metrics['total']:.4f} val={score:.4f} | trunk-grad {gradient_report}")
        if score < best:
            best, stale = score, 0
            CFG.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "loss_state_dict": criterion.state_dict(), "biomarker_mean": mean.cpu(), "biomarker_std": std.cpu(), "config": asdict(CFG), "task_names": TASK_NAMES}, CFG.checkpoint_path)
        else:
            stale += 1
            if stale >= CFG.early_stopping_patience: break
    print("Validation:", evaluate(model, val_loader, mean, std))
    print("Held-out test (run once after validation selection):", evaluate(model, test_loader, mean, std))
else:
    print("V3 notebook built and self-tested. Set RUN_TRAINING=True to launch the controlled experiment.")
