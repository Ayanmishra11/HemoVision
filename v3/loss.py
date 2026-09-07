from typing import Dict
import torch
import torch.nn as nn
from v3.config import (
    TRAIN_CFG, TASK_TIERS, TASK_NAMES
)

EPS = 1e-8
BVP_BAND_LOW_HZ = 0.65
BVP_BAND_HIGH_HZ = 3.25

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

def ccc_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Computes 1 - Concordance Correlation Coefficient (CCC) for each biomarker task.
    Requires batch variance to penalize collapse.
    pred, target, mask: [B, num_tasks]
    """
    losses = torch.zeros(pred.shape[1], device=pred.device)
    valid_counts = mask.sum(dim=0)
    
    for i in range(pred.shape[1]):
        valid_idx = mask[:, i] > 0
        if valid_idx.sum() > 1: # need at least 2 samples for variance
            p = pred[valid_idx, i]
            t = target[valid_idx, i]
            
            p_mean = p.mean()
            t_mean = t.mean()
            
            p_var = p.var(unbiased=False)
            t_var = t.var(unbiased=False)
            
            covar = ((p - p_mean) * (t - t_mean)).mean()
            
            ccc = (2 * covar) / (p_var + t_var + (p_mean - t_mean)**2 + eps)
            losses[i] = 1.0 - ccc
    
    return losses, valid_counts

class HemoMultiTaskLoss_V3(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_pearson = TRAIN_CFG.lambda_pearson
        self.lambda_snr = TRAIN_CFG.lambda_snr
        self.lambda_biomarker = TRAIN_CFG.lambda_biomarker
        
        # Mapping weights to tasks
        self.task_weights = torch.zeros(len(TASK_NAMES))
        for i, name in enumerate(TASK_NAMES):
            tier = TASK_TIERS[name]
            if tier == "established":
                self.task_weights[i] = TRAIN_CFG.established_weight
            elif tier == "moderate":
                self.task_weights[i] = TRAIN_CFG.moderate_weight
            elif tier == "exploratory":
                self.task_weights[i] = TRAIN_CFG.exploratory_weight

    def forward(self, prediction: Dict[str, torch.Tensor], bvp_target: torch.Tensor, biomarker_target: torch.Tensor, biomarker_mask: torch.Tensor, fps: torch.Tensor) -> Dict[str, torch.Tensor]:
        bvp_pred = prediction["bvp"]
        bio_pred = prediction["biomarkers"]
        
        pearson_vals = batch_pearson_correlation(bvp_pred, bvp_target)
        pearson_loss = (1.0 - pearson_vals).mean()
        snr_loss = frequency_snr_loss(bvp_pred, bvp_target, fps)

        bio_pred = bio_pred.float()
        bio_tgt = biomarker_target.float()
        bio_mask = biomarker_mask.float()
        
        # Phase 3: CCC Loss
        ccc_losses, valid_counts = ccc_loss(bio_pred, bio_tgt, bio_mask)
        
        # Tier-weighted aggregation
        self.task_weights = self.task_weights.to(bio_pred.device)
        valid_task_mask = valid_counts > 1
        
        if valid_task_mask.any():
            # Weighted average of valid CCC losses
            active_weights = self.task_weights[valid_task_mask]
            active_losses = ccc_losses[valid_task_mask]
            
            bio_loss = (active_losses * active_weights).sum() / active_weights.sum().clamp_min(1e-8)
        else:
            bio_loss = bvp_pred.new_zeros((), dtype=torch.float32)

        total = (self.lambda_pearson * pearson_loss) + (self.lambda_snr * snr_loss) + (self.lambda_biomarker * bio_loss)
        
        return {
            "total": total,
            "pearson": pearson_loss.detach(),
            "snr": snr_loss.detach(),
            "biomarker": bio_loss.detach(),
            "batch_pearson_r": pearson_vals.detach().mean(),
        }
