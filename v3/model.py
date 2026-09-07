from typing import Dict, Tuple
import torch
import torch.nn as nn
from v3.config import (
    TRAIN_CFG, TASK_NAMES, TOTAL_CHANNELS, NUM_ROIS, CHANNELS_PER_ROI, CAMERA_TYPES
)

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

class TemporalAttentionBiomarkerHead(nn.Module):
    def __init__(self, tcn_channels: int, raw_channels: int = TOTAL_CHANNELS, num_biomarkers: int = len(TASK_NAMES), dropout: float = TRAIN_CFG.dropout):
        super().__init__()
        
        # AC Stream: Multi-Head Self-Attention on TCN features
        self.ac_attn = nn.MultiheadAttention(embed_dim=tcn_channels, num_heads=4, batch_first=True, dropout=dropout)
        
        # DC Stream: Ratio extraction and temporal pooling
        # Raw is [B, 24, T]. We can learn combination of raw channels.
        self.dc_conv = nn.Sequential(
            nn.Conv1d(raw_channels, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Respiratory Stream: Learn slow varying envelope
        self.resp_conv = nn.Sequential(
            nn.Conv1d(tcn_channels, tcn_channels, kernel_size=15, padding=7, groups=tcn_channels), # Large kernel to smooth
            nn.Conv1d(tcn_channels, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        cam_dim = len(CAMERA_TYPES)
        # Combined dim: AC (64) + DC (32) + Resp (16) + Cam (3) = 115
        combined_dim = tcn_channels + 32 + 16 + cam_dim
        
        # Task specific output heads (Tiered)
        # Established tier: RR, SpO2 (Uses everything)
        self.head_established = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        # Moderate tier: BP (Sys, Dia) (Uses AC + Cam)
        self.head_moderate = nn.Sequential(
            nn.Linear(tcn_channels + cam_dim, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        # Exploratory tier: Hb, BMI (Uses DC + Cam)
        self.head_exploratory = nn.Sequential(
            nn.Linear(32 + cam_dim, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, tcn_features: torch.Tensor, raw_input: torch.Tensor, camera_onehot: torch.Tensor) -> torch.Tensor:
        # 1. AC Stream [B, C, T] -> [B, T, C] for attention
        x_ac = tcn_features.transpose(1, 2)
        attn_out, _ = self.ac_attn(x_ac, x_ac, x_ac)
        ac_feat = attn_out.mean(dim=1) # [B, C]
        
        # 2. DC Stream [B, 24, T] -> [B, 32, 1] -> [B, 32]
        dc_feat = self.dc_conv(raw_input).squeeze(-1)
        
        # 3. Respiratory Stream [B, C, T] -> [B, 16, 1] -> [B, 16]
        resp_feat = self.resp_conv(tcn_features).squeeze(-1)
        
        # Combine everything
        combined_feat = torch.cat([ac_feat, dc_feat, resp_feat, camera_onehot], dim=-1)
        
        # Task specific heads
        # established: respiratory, saturation
        out_established = self.head_established(combined_feat)
        
        # moderate: upper_ap, lower_ap
        out_moderate = self.head_moderate(torch.cat([ac_feat, camera_onehot], dim=-1))
        
        # exploratory: hemoglobin, bmi
        out_exploratory = self.head_exploratory(torch.cat([dc_feat, camera_onehot], dim=-1))
        
        return torch.cat([out_established, out_moderate, out_exploratory], dim=-1)

class HemoVisionBoundedTCN_V3(nn.Module):
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

        self.biomarker_head = TemporalAttentionBiomarkerHead(
            tcn_channels=width,
            raw_channels=in_channels,
            num_biomarkers=num_biomarkers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, x_raw: torch.Tensor, camera_onehot: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.input_projection(x)
        feat = self.temporal_blocks(feat)

        return {
            "bvp": self.waveform_head(feat),
            "biomarkers": self.biomarker_head(feat, x_raw, camera_onehot), 
        }
