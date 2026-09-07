# %%
# ==========================================
# CELL 11: BOUNDED TCN + DC BYPASS BIOMARKER HEAD (WITH GENTLE VARIANCE REGULARIZATION)
# ==========================================

from typing import Dict, Tuple
import torch
import torch.nn as nn

BVP_BAND_LOW_HZ = 0.65
BVP_BAND_HIGH_HZ = 3.25
EPS = 1e-8

class DepthwiseTemporalBlock(nn.Module):
    """Residual depthwise-separable temporal convolution block."""
    def __init__(self, channels: int, dilation: int, dropout: float, kernel_size: int = 3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.activation(x + residual)


class RichBiomarkerHead(nn.Module):
    """
    Dual-stream pooling:
    1. AC Stream: Takes TCN output (pulse dynamics) via Avg/Max.
    2. DC Stream: Takes RAW input (absolute skin tone/color) via Avg/Std.
    """
    def __init__(self, tcn_channels: int, raw_channels: int = TOTAL_CHANNELS, num_biomarkers: int = len(TASK_NAMES), dropout: float = 0.15):
        super().__init__()
        
        # 2 x TCN features (avg, max) + 2 x Raw features (avg, std)
        combined_dim = (tcn_channels * 2) + (raw_channels * 2)

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_biomarkers),
        )

    def forward(self, tcn_features: torch.Tensor, raw_input: torch.Tensor) -> torch.Tensor:
        # TCN AC Features (Batch, C, T)
        ac_avg = tcn_features.mean(dim=-1)
        ac_max = tcn_features.max(dim=-1)[0]
        
        # RAW DC Features (Batch, 24, T)
        dc_avg = raw_input.mean(dim=-1)
        dc_std = raw_input.std(dim=-1, unbiased=False)

        pooled = torch.cat([ac_avg, ac_max, dc_avg, dc_std], dim=1)
        return self.mlp(pooled)


class HemoVisionBoundedTCN(nn.Module):
    def __init__(
        self,
        in_channels: int = TOTAL_CHANNELS,
        width: int = TRAIN_CFG.tcn_width,
        dilations: Tuple[int, ...] = TRAIN_CFG.tcn_dilations,
        dropout: float = TRAIN_CFG.dropout,
        num_biomarkers: int = len(TASK_NAMES),
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )

        self.temporal_blocks = nn.Sequential(
            *[DepthwiseTemporalBlock(channels=width, dilation=int(d), dropout=dropout) for d in dilations]
        )

        self.waveform_head = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, 1, kernel_size=1),
            nn.Tanh(),
        )

        self.biomarker_head = RichBiomarkerHead(
            tcn_channels=width,
            raw_channels=in_channels,
            num_biomarkers=num_biomarkers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, x_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.input_projection(x)
        feat = self.temporal_blocks(feat)

        return {
            "bvp": self.waveform_head(feat),
            # CRITICAL: Pass BOTH the TCN features and the RAW features to the head
            "biomarkers": self.biomarker_head(feat, x_raw), 
        }

def batch_pearson_correlation(prediction: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    prediction = prediction.float().reshape(prediction.shape[0], -1)
    target = target.float().reshape(target.shape[0], -1)
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (prediction * target).sum(dim=1)
    denominator = torch.sqrt(prediction.square().sum(dim=1).clamp_min(eps) * target.square().sum(dim=1).clamp_min(eps))
    return numerator / denominator.clamp_min(eps)


def frequency_snr_loss(prediction: torch.Tensor, target: torch.Tensor, fps: torch.Tensor, low_hz: float = BVP_BAND_LOW_HZ, high_hz: float = BVP_BAND_HIGH_HZ, eps: float = EPS) -> torch.Tensor:
    prediction = prediction.squeeze(1)
    target = target.squeeze(1)
    losses = []
    
    for i in range(prediction.shape[0]):
        pred_w = prediction[i].float()
        tgt_w = target[i].float()
        s_fps = float(fps[i].detach().float().item())

        pred_w = pred_w - pred_w.mean()
        tgt_w = tgt_w - tgt_w.mean()

        pred_spec = torch.fft.rfft(pred_w)
        tgt_spec = torch.fft.rfft(tgt_w)

        freqs = torch.fft.rfftfreq(tgt_w.numel(), d=1.0 / s_fps, device=tgt_w.device)
        mask = (freqs >= low_hz) & (freqs <= high_hz)

        if int(mask.sum().item()) < 3: 
            continue

        pred_pwr = pred_spec.abs().square()
        tgt_pwr = tgt_spec.abs().square()

        tgt_idx = torch.where(mask)[0]
        peak_idx = tgt_idx[torch.argmax(tgt_pwr[tgt_idx])]

        left = max(0, int(peak_idx.item()) - 1)
        right = min(pred_pwr.numel(), int(peak_idx.item()) + 2)

        tgt_freq_pwr = pred_pwr[left:right].sum()
        total_pwr = pred_pwr[mask].sum()

        losses.append(-torch.log((tgt_freq_pwr + eps) / (total_pwr + eps)))

    if not losses: 
        return prediction.new_zeros((), dtype=torch.float32)
    return torch.stack(losses).mean()


class HemoMultiTaskLoss(nn.Module):
    def __init__(self, lambda_pearson: float = TRAIN_CFG.lambda_pearson, lambda_snr: float = TRAIN_CFG.lambda_snr, lambda_biomarker: float = TRAIN_CFG.lambda_biomarker):
        super().__init__()
        self.lambda_pearson = float(lambda_pearson)
        self.lambda_snr = float(lambda_snr)
        self.lambda_biomarker = float(lambda_biomarker)
        self.lambda_var = 0.25  # REDUCED: Gentle variance penalty to prevent hedging without blowing up MAE

    def forward(self, prediction: Dict[str, torch.Tensor], bvp_target: torch.Tensor, biomarker_target: torch.Tensor, biomarker_mask: torch.Tensor, fps: torch.Tensor) -> Dict[str, torch.Tensor]:
        bvp_pred = prediction["bvp"]
        bio_pred = prediction["biomarkers"]
        
        pearson_vals = batch_pearson_correlation(bvp_pred, bvp_target)
        pearson_loss = (1.0 - pearson_vals).mean()
        snr_loss = frequency_snr_loss(bvp_pred, bvp_target, fps)

        bio_pred = bio_pred.float()
        bio_tgt = biomarker_target.float()
        bio_mask = biomarker_mask.float()
        
        sq_err = (bio_pred - bio_tgt).square() * bio_mask
        valid_per_task = bio_mask.sum(dim=0)
        task_mse = sq_err.sum(dim=0) / valid_per_task.clamp_min(1.0)
        
        valid_task_mask = valid_per_task > 1
        
        if valid_task_mask.any():
            bio_loss = task_mse[valid_task_mask].mean()
            
            # --- Variance Regularization ---
            # Penalizes the model if the standard deviation of its predictions drops below 1.0
            # (since the targets are standardized to have std=1.0)
            pred_std = torch.zeros_like(task_mse)
            for i in range(bio_pred.shape[1]):
                valid_preds = bio_pred[:, i][bio_mask[:, i] > 0]
                if len(valid_preds) > 1:
                    pred_std[i] = valid_preds.std()
                    
            var_loss = ((pred_std[valid_task_mask] - 1.0) ** 2).mean()
        else:
            bio_loss = bvp_pred.new_zeros((), dtype=torch.float32)
            var_loss = bvp_pred.new_zeros((), dtype=torch.float32)

        total = (self.lambda_pearson * pearson_loss) + (self.lambda_snr * snr_loss) + (self.lambda_biomarker * bio_loss) + (self.lambda_var * var_loss)
        
        return {
            "total": total,
            "pearson": pearson_loss.detach(),
            "snr": snr_loss.detach(),
            "biomarker": bio_loss.detach(),
            "batch_pearson_r": pearson_vals.detach().mean(),
        }


# Model shape smoke test.
with torch.no_grad():
    _test_model = HemoVisionBoundedTCN()
    _test_input = torch.randn(2, TOTAL_CHANNELS, TRAIN_CFG.window_frames)
    _test_raw = torch.rand(2, TOTAL_CHANNELS, TRAIN_CFG.window_frames)
    _test_output = _test_model(_test_input, _test_raw)
    
    assert _test_output["bvp"].shape == (2, 1, TRAIN_CFG.window_frames)
    assert _test_output["biomarkers"].shape == (2, len(TASK_NAMES))

log.info("Cell 11 updated: HemoVisionBoundedTCN + DC Bypass + Gentle Variance Regularization.")