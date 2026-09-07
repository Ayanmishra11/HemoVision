"""
HemoVision Live Demo — Heart Rate from Facial Video
====================================================
Self-contained demo script. Uses the trained HemoVisionBoundedTCN checkpoint
(hemovision_bounded_tcn_best.pt) to estimate heart rate from a video file.

Usage:
    python hemovision_demo.py                           # webcam
    python hemovision_demo.py path/to/video.avi         # specific video
    python hemovision_demo.py --test                    # run on 3 test-set clips

Requirements (use .venv310):
    mediapipe, opencv-python, torch, numpy, scipy, matplotlib
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# SECTION 1: MODEL ARCHITECTURE (exact copy from v3_pipeline)
# ============================================================
# These must match the checkpoint's architecture EXACTLY.
# Source of truth: v3_pipeline/cell_11_dataset.py
# Checkpoint: hemovision_bounded_tcn_best.pt (width=48, 4 dilations, 6 biomarkers)

NUM_ROIS = 8
CHANNELS_PER_ROI = 3
TOTAL_CHANNELS = NUM_ROIS * CHANNELS_PER_ROI  # 24
WINDOW_FRAMES = 600  # from HemoTrainingConfig in cell_10_dataset.py
BVP_BAND_LOW_HZ = 0.65
BVP_BAND_HIGH_HZ = 3.25


class DepthwiseTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float, kernel_size: int = 3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                   padding=padding, dilation=dilation,
                                   groups=channels, bias=False)
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
    def __init__(self, tcn_channels: int, raw_channels: int = TOTAL_CHANNELS,
                 num_biomarkers: int = 6, dropout: float = 0.15):
        super().__init__()
        combined_dim = (tcn_channels * 2) + (raw_channels * 2)
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_biomarkers),
        )

    def forward(self, tcn_features: torch.Tensor, raw_input: torch.Tensor) -> torch.Tensor:
        ac_avg = tcn_features.mean(dim=-1)
        ac_max = tcn_features.max(dim=-1)[0]
        dc_avg = raw_input.mean(dim=-1)
        dc_std = raw_input.std(dim=-1, unbiased=False)
        pooled = torch.cat([ac_avg, ac_max, dc_avg, dc_std], dim=1)
        return self.mlp(pooled)


class HemoVisionBoundedTCN(nn.Module):
    def __init__(self, in_channels: int = TOTAL_CHANNELS, width: int = 48,
                 dilations: Tuple[int, ...] = (1, 2, 4, 8), dropout: float = 0.15,
                 num_biomarkers: int = 6):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width), nn.GELU(),
        )
        self.temporal_blocks = nn.Sequential(
            *[DepthwiseTemporalBlock(channels=width, dilation=int(d), dropout=dropout)
              for d in dilations]
        )
        self.waveform_head = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, 1, kernel_size=1), nn.Tanh(),
        )
        self.biomarker_head = RichBiomarkerHead(
            tcn_channels=width, raw_channels=in_channels,
            num_biomarkers=num_biomarkers, dropout=dropout,
        )

    def forward(self, x: torch.Tensor, x_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.input_projection(x)
        feat = self.temporal_blocks(feat)
        return {
            "bvp": self.waveform_head(feat),
            "biomarkers": self.biomarker_head(feat, x_raw),
        }


# ============================================================
# SECTION 2: FACE ROI EXTRACTION (from HemoVision_v2_Pipeline)
# ============================================================
# Source of truth: HemoVision_v2_Pipeline.ipynb Cell 2 (config) + Cell 3 (extractor)

ROI_INDICES = (
    (10, 338, 297, 332),    # Forehead Left
    (10, 109, 67, 103),     # Forehead Right
    (117, 118, 101, 205),   # Left Cheek Upper
    (138, 135, 210, 211),   # Left Cheek Lower
    (346, 347, 330, 425),   # Right Cheek Upper
    (367, 364, 430, 431),   # Right Cheek Lower
    (1, 2, 98, 327),        # Nose / Mid-face
    (152, 148, 176, 377),   # Chin Area
)

ROI_LABELS = [
    "Forehead L", "Forehead R", "L Cheek Up", "L Cheek Low",
    "R Cheek Up", "R Cheek Low", "Nose/Mid", "Chin",
]

# Nice colors for ROI overlay
ROI_COLORS = [
    (255, 100, 100), (100, 100, 255), (100, 255, 100), (255, 255, 100),
    (255, 100, 255), (100, 255, 255), (200, 200, 200), (255, 180, 100),
]


class FaceROIExtractor:
    """MediaPipe Face Mesh ROI extractor — matches training preprocessing exactly."""

    def __init__(self):
        try:
            from mediapipe.python.solutions import face_mesh as mp_face_mesh
            self.mp_face_mesh = mp_face_mesh
        except (ImportError, AttributeError):
            try:
                from mediapipe.solutions import face_mesh as mp_face_mesh
                self.mp_face_mesh = mp_face_mesh
            except (ImportError, AttributeError):
                import mediapipe as mp
                solutions = getattr(mp, "solutions", getattr(getattr(mp, "python", None), "solutions", None))
                self.mp_face_mesh = solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.roi_indices = [list(indices) for indices in ROI_INDICES]

    def extract_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """
        Returns (roi_means[24], roi_polygons) or (None, None) if no face.
        roi_means is flat [8*3] = [24] in RGB order (cv2.mean on RGB image).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame_rgb.shape
        roi_means = np.zeros((NUM_ROIS, 3), dtype=np.float32)
        roi_polygons = []

        for roi_idx, indices in enumerate(self.roi_indices):
            pts = np.array(
                [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
                dtype=np.int32,
            )
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, 255)
            mean_val = cv2.mean(frame_rgb, mask=mask)[:3]
            roi_means[roi_idx] = np.array(mean_val, dtype=np.float32)
            roi_polygons.append(pts)

        return roi_means.reshape(-1), roi_polygons

    def close(self):
        self.face_mesh.close()


# ============================================================
# SECTION 3: PREPROCESSING (exact match to cell_10_dataset.py)
# ============================================================

def normalise_roi_channels(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-channel z-score along time. Input: [24, T]. Output: [24, T]."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def make_x_raw(x: torch.Tensor) -> torch.Tensor:
    """DC-preserving raw features. Input: [24, T]. Output: [24, T]."""
    return x.clone() / 255.0


# ============================================================
# SECTION 4: HR ESTIMATION (exact match to cell_12_model.py)
# ============================================================

def fft_hr_bpm(waveform: torch.Tensor, fps: float,
               low_hz: float = BVP_BAND_LOW_HZ,
               high_hz: float = BVP_BAND_HIGH_HZ) -> float:
    """FFT peak-picking HR from a 1D waveform tensor. Returns BPM."""
    wave = waveform.squeeze().float()
    wave = wave - wave.mean()
    spectrum = torch.fft.rfft(wave)
    power = spectrum.abs().square()
    frequencies = torch.fft.rfftfreq(wave.numel(), d=1.0 / fps)
    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not band_mask.any():
        return float("nan")
    band_indices = torch.where(band_mask)[0]
    peak_index = band_indices[torch.argmax(power[band_indices])]
    return float(frequencies[peak_index].item() * 60.0)


def compute_ground_truth_hr(ppg_sync_path: str, fps: float) -> float:
    """Compute HR from ground-truth ppg_sync file using same FFT method."""
    raw = np.atleast_2d(np.loadtxt(ppg_sync_path, dtype=np.float32))
    amplitudes = raw[:, 0]
    bvp = torch.from_numpy(amplitudes).float()
    bvp = bvp - bvp.mean()
    bvp = bvp / bvp.abs().amax().clamp_min(1e-8)
    return fft_hr_bpm(bvp, fps)


# ============================================================
# SECTION 5: MODEL LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "data" / "vitalscan-clinic" / "MCD-rPPG" / "hemovision_bounded_tcn_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> HemoVisionBoundedTCN:
    """Load the trained model from the checkpoint."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model = HemoVisionBoundedTCN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_metrics = checkpoint.get("validation_metrics", {})
    print(f"Model loaded: epoch={epoch}, "
          f"val_r={val_metrics.get('pearson_r', '?'):.4f}, "
          f"val_hr_mae={val_metrics.get('hr_mae_bpm', '?'):.2f} BPM")
    return model


# ============================================================
# SECTION 6: LIVE DEMO ENGINE
# ============================================================

class HemoVisionDemo:
    """Processes video frames, runs inference, displays live results."""

    def __init__(self, model: HemoVisionBoundedTCN, device: torch.device = DEVICE):
        self.model = model
        self.device = device
        self.extractor = FaceROIExtractor()
        self.frame_buffer: List[np.ndarray] = []  # list of [24] arrays
        self.bvp_history: List[float] = []
        self.hr_history: List[float] = []
        self.fps = 30.0
        self.last_polygons = None

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        Process one frame. Returns (hr_bpm, predicted_bvp_segment) once
        enough frames are buffered, else (None, None).
        """
        roi_means, polygons = self.extractor.extract_frame(frame_bgr)

        if roi_means is not None:
            self.frame_buffer.append(roi_means)
            self.last_polygons = polygons
        elif len(self.frame_buffer) > 0:
            # Forward-fill on missed face
            self.frame_buffer.append(self.frame_buffer[-1].copy())
        else:
            self.frame_buffer.append(np.zeros(TOTAL_CHANNELS, dtype=np.float32))

        if len(self.frame_buffer) < WINDOW_FRAMES:
            return None, None

        # Take the last WINDOW_FRAMES frames
        window = np.stack(self.frame_buffer[-WINDOW_FRAMES:], axis=0)  # [T, 24]
        signal = torch.from_numpy(window).T.float()  # [24, T]

        # Preprocessing — exact match to cell_10_dataset.py
        x = normalise_roi_channels(signal)        # AC-normalized
        x_raw = make_x_raw(signal)                # DC-preserved / 255

        # Run model
        x_batch = x.unsqueeze(0).to(self.device)          # [1, 24, 600]
        x_raw_batch = x_raw.unsqueeze(0).to(self.device)  # [1, 24, 600]

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            output = self.model(x_batch, x_raw_batch)

        bvp_pred = output["bvp"].squeeze().cpu().numpy()  # [600]
        hr = fft_hr_bpm(output["bvp"].squeeze().cpu(), self.fps)

        self.bvp_history.extend(bvp_pred.tolist())
        if not np.isnan(hr):
            self.hr_history.append(hr)

        return hr, bvp_pred

    def draw_roi_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw ROI polygons on the frame."""
        overlay = frame_bgr.copy()
        if self.last_polygons is not None:
            for idx, pts in enumerate(self.last_polygons):
                color_bgr = ROI_COLORS[idx][::-1]  # RGB -> BGR
                cv2.polylines(overlay, [pts], True, color_bgr, 2)
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.putText(overlay, ROI_LABELS[idx], (cx - 30, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_bgr, 1)
        return overlay

    def close(self):
        self.extractor.close()


# ============================================================
# SECTION 7: MAIN DEMO MODES
# ============================================================

def run_video_demo(video_path: str, save_plot: Optional[str] = None, save_video: Optional[str] = None):
    """Process a video file with live opencv display."""
    print(f"\n{'='*60}")
    print(f"HemoVision Demo — Processing: {Path(video_path).name}")
    print(f"{'='*60}")

    model = load_model()
    demo = HemoVisionDemo(model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    demo.fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames @ {demo.fps:.1f} FPS "
          f"({total_frames/demo.fps:.1f}s)")

    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    h_vid, w_vid, _ = first_frame.shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    plot_w = max(640, w_vid)
    canvas_h = max(h_vid, 480)
    canvas_w = w_vid + plot_w
    
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_video, fourcc, demo.fps, (canvas_w, canvas_h))

    frame_idx = 0
    current_hr = None
    inference_count = 0
    start_time = time.time()

    # Stride for inference: run every WINDOW_FRAMES//4 new frames after the first window
    stride = max(1, WINDOW_FRAMES // 4)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Always extract ROIs
            roi_means, polygons = demo.extractor.extract_frame(frame)
            if roi_means is not None:
                demo.frame_buffer.append(roi_means)
                demo.last_polygons = polygons
            elif len(demo.frame_buffer) > 0:
                demo.frame_buffer.append(demo.frame_buffer[-1].copy())
            else:
                demo.frame_buffer.append(np.zeros(TOTAL_CHANNELS, dtype=np.float32))

            # Run inference when we have enough frames and at stride intervals
            should_infer = (
                len(demo.frame_buffer) >= WINDOW_FRAMES and
                (len(demo.frame_buffer) == WINDOW_FRAMES or
                 (len(demo.frame_buffer) - WINDOW_FRAMES) % stride == 0)
            )

            if should_infer:
                window = np.stack(demo.frame_buffer[-WINDOW_FRAMES:], axis=0)
                signal = torch.from_numpy(window).T.float()

                x = normalise_roi_channels(signal)
                x_raw = make_x_raw(signal)

                x_batch = x.unsqueeze(0).to(demo.device)
                x_raw_batch = x_raw.unsqueeze(0).to(demo.device)

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=demo.device.type == "cuda"):
                    output = demo.model(x_batch, x_raw_batch)

                bvp_pred = output["bvp"].squeeze().cpu().numpy()
                hr = fft_hr_bpm(output["bvp"].squeeze().cpu(), demo.fps)

                # Append BVP: on first run add all, after that add only the stride portion
                if inference_count == 0:
                    demo.bvp_history.extend(bvp_pred.tolist())
                else:
                    demo.bvp_history.extend(bvp_pred[-stride:].tolist())

                if not np.isnan(hr):
                    demo.hr_history.append(hr)
                    current_hr = hr
                inference_count += 1

            # Show video with ROI overlay
            overlay = demo.draw_roi_overlay(frame)
            
            # Create combined canvas
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            canvas[0:h_vid, 0:w_vid] = overlay

            # 1. Big BPM number
            if current_hr is not None:
                cv2.putText(canvas, f"{current_hr:.0f} BPM",
                            (w_vid + 50, 100), cv2.FONT_HERSHEY_DUPLEX, 2.5,
                            (0, 255, 0), 3)
            else:
                cv2.putText(canvas, "Estimating...",
                            (w_vid + 50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5,
                            (0, 165, 255), 2)

            # 2. Draw Waveform line plot
            if demo.bvp_history:
                plot_samples = min(len(demo.bvp_history), int(demo.fps * 10)) # last 10 seconds
                display_bvp = demo.bvp_history[-plot_samples:]
                
                min_val = min(display_bvp) if min(display_bvp) < -0.1 else -1.0
                max_val = max(display_bvp) if max(display_bvp) > 0.1 else 1.0
                val_range = max(1e-5, max_val - min_val)
                
                plot_x_start = w_vid + 20
                plot_x_end = canvas_w - 20
                plot_y_start = 200
                plot_y_end = canvas_h - 50
                
                # Draw box & axes
                cv2.rectangle(canvas, (plot_x_start, plot_y_start), (plot_x_end, plot_y_end), (50, 50, 50), 1)
                cv2.putText(canvas, "Predicted BVP Waveform", (plot_x_start, plot_y_start - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                pts = []
                for i, val in enumerate(display_bvp):
                    x = int(plot_x_start + (i / max(1, len(display_bvp) - 1)) * (plot_x_end - plot_x_start))
                    y = int(plot_y_end - ((val - min_val) / val_range) * (plot_y_end - plot_y_start))
                    pts.append((x, y))
                    
                if len(pts) > 1:
                    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 0, 255), 2)

            # 3. Progress bar at bottom of video
            cv2.putText(canvas, f"Frame {frame_idx}/{total_frames}",
                        (20, h_vid - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)
            progress = frame_idx / max(total_frames, 1)
            bar_w = w_vid - 40
            cv2.rectangle(canvas, (20, h_vid - 20),
                          (20 + int(bar_w * progress), h_vid - 10),
                          (0, 255, 0), -1)

            cv2.imshow("HemoVision Live Demo", canvas)
            if video_writer is not None:
                video_writer.write(canvas)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\nUser interrupted.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        demo.close()
        if video_writer is not None:
            video_writer.release()

    elapsed = time.time() - start_time
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx/elapsed:.1f} FPS)")

    if demo.hr_history:
        median_hr = float(np.median(demo.hr_history))
        mean_hr = float(np.mean(demo.hr_history))
        print(f"Final HR estimate: {median_hr:.1f} BPM (median), "
              f"{mean_hr:.1f} BPM (mean)")

    if save_plot:
        cv2.imwrite(save_plot, canvas)
        print(f"Plot saved: {save_plot}")

    return demo.hr_history


def run_test_comparison():
    """Run on 3 held-out test clips and compare predicted vs ground-truth HR."""
    print(f"\n{'='*60}")
    print("HemoVision Test Set Comparison")
    print(f"{'='*60}\n")

    DATA_ROOT = BASE_DIR / "data" / "vitalscan-clinic" / "MCD-rPPG"
    manifest = json.load(open(DATA_ROOT / "manifest.json"))
    db = {
        Path(str(row["video"])).stem: row
        for _, row in __import__("pandas").read_csv(DATA_ROOT / "db.csv").iterrows()
    }

    # Explicitly pick the requested 4 clips (excluded 9825_after due to ground truth mismatch)
    target_keys = {
        "1107_FullHDwebcam_before",
        "1314_FullHDwebcam_before",
        "1362_FullHDwebcam_before",
        "1107_FullHDwebcam_after"
    }
    
    test_clips = [
        r for r in manifest
        if r["split"] == "test"
        and Path(r.get("roi_cache_path", "")).stem in target_keys
    ]
    
    # Sort them to process in a predictable order
    test_clips.sort(key=lambda r: Path(r.get("roi_cache_path", "")).stem)

    if not test_clips:
        print("ERROR: No test clips found!")
        return

    model = load_model()

    results = []
    for clip in test_clips:
        video_key = Path(clip["roi_cache_path"]).stem
        video_path = DATA_ROOT / "video" / f"{video_key}.avi"
        ppg_path = DATA_ROOT / "ppg_sync" / f"{video_key}.txt"
        fps = clip["video_fps"]
        subject = clip["subject_id"]

        if not video_path.exists():
            print(f"  SKIP {video_key}: video not found")
            continue
        if not ppg_path.exists():
            print(f"  SKIP {video_key}: ppg_sync not found")
            continue

        print(f"\nProcessing: {video_key} (subject {subject}, {fps:.1f} FPS)")

        # Ground truth HR
        gt_hr = compute_ground_truth_hr(str(ppg_path), fps)
        print(f"  Ground truth HR: {gt_hr:.1f} BPM")

        # Run inference
        demo = HemoVisionDemo(model)
        demo.fps = fps

        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            roi_means, polygons = demo.extractor.extract_frame(frame)
            if roi_means is not None:
                demo.frame_buffer.append(roi_means)
                demo.last_polygons = polygons
            elif len(demo.frame_buffer) > 0:
                demo.frame_buffer.append(demo.frame_buffer[-1].copy())
            else:
                demo.frame_buffer.append(np.zeros(TOTAL_CHANNELS, dtype=np.float32))

            # Run inference at windows
            if (len(demo.frame_buffer) >= WINDOW_FRAMES and
                    (len(demo.frame_buffer) == WINDOW_FRAMES or
                     (len(demo.frame_buffer) - WINDOW_FRAMES) % 150 == 0)):
                window = np.stack(demo.frame_buffer[-WINDOW_FRAMES:], axis=0)
                signal = torch.from_numpy(window).T.float()
                x = normalise_roi_channels(signal)
                x_raw = make_x_raw(signal)
                with torch.no_grad():
                    output = model(x.unsqueeze(0).to(demo.device),
                                   x_raw.unsqueeze(0).to(demo.device))
                hr = fft_hr_bpm(output["bvp"].squeeze().cpu(), fps)
                if not np.isnan(hr):
                    demo.hr_history.append(hr)

        cap.release()
        demo.close()

        if demo.hr_history:
            pred_hr = float(np.median(demo.hr_history))
            error = abs(pred_hr - gt_hr)
            print(f"  Predicted HR:    {pred_hr:.1f} BPM (median of {len(demo.hr_history)} windows)")
            print(f"  Error:           {error:.1f} BPM")
            results.append({
                "video": video_key, "subject": subject,
                "gt_hr": gt_hr, "pred_hr": pred_hr, "error": error,
                "n_windows": len(demo.hr_history), "frames": frame_count,
            })
        else:
            print(f"  ERROR: No predictions generated (only {len(demo.frame_buffer)} frames)")

    # Summary table
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY: Predicted vs Ground Truth HR")
        print(f"{'='*60}")
        print(f"{'Video':<35} {'GT HR':>7} {'Pred HR':>8} {'Error':>7}")
        print("-" * 60)
        for r in results:
            print(f"{r['video']:<35} {r['gt_hr']:>6.1f} {r['pred_hr']:>7.1f} {r['error']:>6.1f}")
        mean_error = np.mean([r["error"] for r in results])
        print("-" * 60)
        print(f"{'Mean Absolute Error':<35} {'':>7} {'':>8} {mean_error:>6.1f}")
    else:
        print("\nNo results to summarize.")


def main():
    parser = argparse.ArgumentParser(
        description="HemoVision Demo — Heart Rate from Facial Video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hemovision_demo.py video.avi          Process a video file
  python hemovision_demo.py --test             Run on 4 test-set clips
  python hemovision_demo.py 0                  Use webcam (device 0)
  python hemovision_demo.py video.avi --save-video backup_demo.mp4
        """,
    )
    parser.add_argument("video", nargs="?", default=None,
                        help="Path to video file, or camera index (e.g. 0)")
    parser.add_argument("--test", action="store_true",
                        help="Run comparison on held-out test clips")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Save the final BVP/HR plot to this path")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Save the live demo UI to an MP4 video file")

    args = parser.parse_args()

    if args.test:
        run_test_comparison()
    elif args.video is not None:
        # Check if it's a camera index
        try:
            cam_idx = int(args.video)
            run_video_demo(cam_idx, args.save_plot, args.save_video)
        except ValueError:
            run_video_demo(args.video, args.save_plot, args.save_video)
    else:
        # Default: webcam
        print("No video specified. Use --test for test comparison or provide a video path.")
        print("Starting webcam demo (device 0)...")
        run_video_demo(0, args.save_plot, args.save_video)


if __name__ == "__main__":
    main()
