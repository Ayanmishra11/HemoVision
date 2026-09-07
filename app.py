"""
HemoVision — Presentation Dashboard
=====================================
Streamlit web app for live rPPG heart-rate demo.
Imports ALL inference logic from hemovision_demo.py — zero duplication.
"""

import json
import sys
import time
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Bootstrap path so we can import from hemovision_demo.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
try:
    from hemovision_demo import (
        load_model, HemoVisionDemo, compute_ground_truth_hr,
        TOTAL_CHANNELS, WINDOW_FRAMES,
        normalise_roi_channels, make_x_raw, fft_hr_bpm,
    )
    import torch
except ImportError as e:
    st.error(f"Import failed: {e}\nMake sure hemovision_demo.py is in the same folder.")
    st.stop()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HemoVision — rPPG Vital Signs",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
DATA_ROOT  = BASE_DIR / "data" / "vitalscan-clinic" / "MCD-rPPG"
VIDEO_DIR  = DATA_ROOT / "video"
PPG_DIR    = DATA_ROOT / "ppg_sync"
AMBIGUOUS_DIFF_THRESHOLD = 15.0   # BPM — clips where manifest vs db disagree more are excluded


# ---------------------------------------------------------------------------
# Model (cached — loads once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading HemoVisionBoundedTCN checkpoint…")
def get_model():
    return load_model()


# ---------------------------------------------------------------------------
# Clip catalogue (cached — reads manifest once)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading dataset manifest…")
def load_test_catalogue():
    """Return a DataFrame of all test-split clips with clean ground truth."""
    manifest_path = DATA_ROOT / "manifest.json"
    db_path       = DATA_ROOT / "db.csv"

    if not manifest_path.exists() or not db_path.exists():
        return pd.DataFrame()

    manifest = json.load(open(manifest_path))
    db       = pd.read_csv(db_path)

    # Build db lookup: ppg_sync filename stem → db.csv 'pulse' value
    db_lookup: dict = {}
    for _, row in db.iterrows():
        if pd.notna(row.get("ppg_sync")):
            key = Path(str(row["ppg_sync"])).stem
            db_lookup[key] = row.get("pulse")

    rows = []
    for r in manifest:
        if r.get("split") != "test":
            continue

        stem       = Path(r["roi_cache_path"]).stem
        video_path = VIDEO_DIR / f"{stem}.avi"
        ppg_path   = PPG_DIR   / f"{stem}.txt"

        if not video_path.exists() or not ppg_path.exists():
            continue

        manifest_hr = r.get("hr_bpm")
        db_hr       = db_lookup.get(stem)

        # Exclude clips with conflicting ground-truth sources
        if manifest_hr is not None and db_hr is not None:
            if abs(manifest_hr - db_hr) > AMBIGUOUS_DIFF_THRESHOLD:
                continue

        rows.append({
            "stem":        stem,
            "label":       f"{r['subject_id']} · {r['camera_type']} · {r['condition']}",
            "subject_id":  r["subject_id"],
            "camera_type": r["camera_type"],
            "condition":   r["condition"],
            "fps":         float(r["video_fps"]),
            "manifest_hr": manifest_hr,
            "video_path":  str(video_path),
            "ppg_path":    str(ppg_path),
        })

    return pd.DataFrame(rows).sort_values(["subject_id", "camera_type", "condition"])


# ===========================================================================
# MAIN UI
# ===========================================================================
model    = get_model()
catalogue = load_test_catalogue()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_demo, tab_about = st.tabs(["🎥 Live Demo", "ℹ️ About HemoVision"])

# ===========================================================================
# TAB 2 — About (static, no computation)
# ===========================================================================
with tab_about:
    st.header("About HemoVision")

    st.markdown("""
**HemoVision** is a remote photoplethysmography (rPPG) system that estimates heart rate
contactlessly from ordinary facial video. It was trained and evaluated on the
**MCD-rPPG** clinical dataset (~600 subjects, 3 camera types, before/after exercise conditions).

The pipeline extracts 8 facial skin-colour ROIs per frame using MediaPipe Face Mesh,
reconstructs a Blood Volume Pulse (BVP) waveform with a dilated temporal convolutional network,
and reads heart rate via FFT peak-picking in the physiological band (0.65–3.25 Hz / 39–195 BPM).
""")

    col1, col2, col3 = st.columns(3)
    col1.metric("Architecture", "HemoVisionBoundedTCN")
    col2.metric("Full test-set MAE", "~9 BPM")
    col3.metric("Pearson r", "~0.27")

    st.info(
        "**10 additional biomarkers** (SpO₂, blood pressure, hemoglobin, HbA1c, cholesterol, "
        "stress, BMI, respiratory rate, glycated hemoglobin, rigidity) were rigorously investigated "
        "and found not to have reliable signal in RGB-only video. "
        "See the presentation for detailed findings."
    )

    st.markdown("""
#### Why Only Heart Rate Works in RGB Video

The BVP signal is detectable because oxygenated blood absorbs green light differently than surrounding tissue —
a subtle colour oscillation at the heartbeat frequency that cameras can measure.
Deeper biomarkers like SpO₂ require the ratio of absorption at **two separate wavelengths** (660 nm red + 940 nm NIR),
which standard RGB cameras cannot provide.
""")

# ===========================================================================
# TAB 1 — Live Demo
# ===========================================================================
with tab_demo:

    if catalogue.empty:
        st.error("Clip catalogue is empty — check that DATA_ROOT is correct.")
        st.stop()

    # -----------------------------------------------------------------------
    # Sidebar — clip selection
    # -----------------------------------------------------------------------
    st.sidebar.header("❤️ HemoVision Demo")
    st.sidebar.caption(
        "Choose any video below — **none of these were seen during training**."
    )
    st.sidebar.markdown("---")

    # Filter controls
    cam_types   = ["All"] + sorted(catalogue["camera_type"].unique().tolist())
    conditions  = ["All"] + sorted(catalogue["condition"].unique().tolist())
    subject_ids = ["All"] + sorted(catalogue["subject_id"].unique().tolist())

    sel_cam  = st.sidebar.selectbox("Camera type", cam_types)
    sel_cond = st.sidebar.selectbox("Condition", conditions)
    sel_subj = st.sidebar.selectbox("Subject ID", subject_ids)
    
    filtered = catalogue.copy()
    if sel_cam  != "All": filtered = filtered[filtered["camera_type"] == sel_cam]
    if sel_cond != "All": filtered = filtered[filtered["condition"]   == sel_cond]
    if sel_subj != "All": filtered = filtered[filtered["subject_id"]  == sel_subj]

    if filtered.empty:
        st.sidebar.warning("No clips match your filters.")
        st.stop()

    clip_labels = filtered["label"].tolist()
    selected_label = st.sidebar.selectbox(
        f"Select clip ({len(filtered)} available)",
        clip_labels,
    )

    row = filtered[filtered["label"] == selected_label].iloc[0]
    clip_id    = row["stem"]
    video_path = Path(row["video_path"])
    ppg_path   = Path(row["ppg_path"])
    fps_hint   = row["fps"]

    # -----------------------------------------------------------------------
    # Main area — hero header
    # -----------------------------------------------------------------------
    st.title("HemoVision: Contactless Vital Signs (rPPG)")
    st.caption(
        f"📽️ **{clip_id}** · {row['camera_type']} · {row['condition']} "
        f"· None of these clips were seen during training"
    )

    # -----------------------------------------------------------------------
    # Run button
    # -----------------------------------------------------------------------
    run_btn = st.sidebar.button("▶ Run Inference", type="primary", use_container_width=True)

    if not run_btn:
        st.info("👈 Select a test clip in the sidebar, then click **▶ Run Inference** to start.")

    if run_btn:
        if not video_path.exists():
            st.error(f"Video file not found: {video_path}")
            st.stop()

        # -------------------------------------------------------------------
        # Stage 1 — video load & face extraction
        # -------------------------------------------------------------------
        stage_status = st.empty()
        stage_status.info("**Stage 1 / 4 — Loading video & extracting facial ROIs…**  "
                          "MediaPipe is detecting 8 skin-colour regions per frame.")

        cap         = cv2.VideoCapture(str(video_path))
        fps         = cap.get(cv2.CAP_PROP_FPS) or fps_hint
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        gt_hr = compute_ground_truth_hr(str(ppg_path), fps) if ppg_path.exists() else None

        # Layout — two equal columns
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("📷 Face Feed with ROI Overlay")
            video_placeholder = st.empty()

        with right_col:
            st.subheader("📊 Results")
            results_placeholder = st.empty()

        # Progress bar + stage label below video
        progress_bar   = st.progress(0.0)
        stage_label    = st.empty()

        # -------------------------------------------------------------------
        # Inference loop
        # -------------------------------------------------------------------
        demo          = HemoVisionDemo(model)
        demo.fps      = fps
        stride        = max(1, WINDOW_FRAMES // 4)
        frame_idx     = 0
        inference_count = 0
        current_hr    = None
        stage_shown   = 1
        UI_EVERY      = 6   # update UI every N frames to stay responsive

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            progress    = frame_idx / max(total_frames, 1)

            # ----- Stage label transitions -----
            if frame_idx == 1:
                stage_label.caption("Stage 1 / 4 — Extracting ROIs…")
            if len(demo.frame_buffer) >= WINDOW_FRAMES and stage_shown == 1:
                stage_status.info("**Stage 2 / 4 — Reconstructing BVP waveform…**  "
                                  "TCN is reconstructing the blood volume pulse signal.")
                stage_shown = 2
            if inference_count >= 1 and stage_shown == 2:
                stage_status.info("**Stage 3 / 4 — Estimating heart rate via FFT…**  "
                                  "Peak-picking the dominant frequency in the physiological band.")
                stage_shown = 3

            # ----- ROI extraction -----
            roi_means, polygons = demo.extractor.extract_frame(frame)
            if roi_means is not None:
                demo.frame_buffer.append(roi_means)
                demo.last_polygons = polygons
            elif demo.frame_buffer:
                demo.frame_buffer.append(demo.frame_buffer[-1].copy())
            else:
                demo.frame_buffer.append(np.zeros(TOTAL_CHANNELS, dtype=np.float32))

            # ----- Model inference -----
            should_infer = (
                len(demo.frame_buffer) >= WINDOW_FRAMES and
                (len(demo.frame_buffer) == WINDOW_FRAMES or
                 (len(demo.frame_buffer) - WINDOW_FRAMES) % stride == 0)
            )

            if should_infer:
                window = np.stack(demo.frame_buffer[-WINDOW_FRAMES:], axis=0)
                signal = torch.from_numpy(window).T.float()
                x      = normalise_roi_channels(signal)
                x_raw  = make_x_raw(signal)

                x_b     = x.unsqueeze(0).to(demo.device)
                x_raw_b = x_raw.unsqueeze(0).to(demo.device)

                with torch.no_grad():
                    output = demo.model(x_b, x_raw_b)

                bvp_pred = output["bvp"].squeeze().cpu().numpy()
                hr       = fft_hr_bpm(output["bvp"].squeeze().cpu(), demo.fps)

                if inference_count == 0:
                    demo.bvp_history.extend(bvp_pred.tolist())
                else:
                    demo.bvp_history.extend(bvp_pred[-stride:].tolist())

                if not np.isnan(hr):
                    demo.hr_history.append(hr)
                    current_hr = float(np.median(demo.hr_history))
                inference_count += 1

            # ----- UI update -----
            if frame_idx % UI_EVERY == 0:
                # Video frame
                overlay = demo.draw_roi_overlay(frame)
                ok, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok:
                    video_placeholder.image(buf.tobytes())

                # Results panel
                with results_placeholder.container():
                    if current_hr is not None and stage_shown >= 3:
                        # Transition to final stage
                        if stage_shown == 3:
                            stage_status.success("**Stage 4 / 4 — Results ready!** ✅")
                            stage_shown = 4

                        if gt_hr is not None:
                            err = abs(current_hr - gt_hr)
                            r1, r2, r3 = st.columns(3)
                            r1.metric("🫀 Predicted HR (rPPG)", f"{current_hr:.1f} BPM")
                            r2.metric("📋 Clinical Ground Truth", f"{gt_hr:.1f} BPM")
                            r3.metric("📐 Error", f"{err:.1f} BPM",
                                      delta=f"{'✅' if err < 5 else '⚠️'} {'Good' if err < 5 else 'High'}",
                                      delta_color="off")
                        else:
                            st.metric("🫀 Predicted HR", f"{current_hr:.1f} BPM")

                        # BVP waveform
                        st.markdown("**Predicted BVP Waveform** *(last 10 s)*")
                        plot_n = min(len(demo.bvp_history), int(fps * 10))
                        st.line_chart(demo.bvp_history[-plot_n:])

                        # Clinical proof
                        if ppg_path.exists():
                            with st.expander("🔬 Clinical proof — raw PPG sensor data", expanded=False):
                                raw_lines = ppg_path.read_text().splitlines()[:60]
                                st.code("\n".join(raw_lines) + "\n...", language="text")
                                st.caption(
                                    "Ground truth sourced from synchronized clinical PPG sensor "
                                    f"(ppg_sync/{clip_id}.txt). "
                                    "Each row is one hardware sample from the pulse oximeter worn "
                                    "by the subject during filming."
                                )
                    else:
                        st.info(f"Buffering… {frame_idx}/{total_frames} frames processed. "
                                "HR will appear once enough signal is collected.")

                progress_bar.progress(min(progress, 1.0))
                stage_label.caption(
                    f"Frame {frame_idx}/{total_frames} · "
                    f"Windows processed: {inference_count} · "
                    f"{'HR locked ✅' if current_hr else 'Buffering…'}"
                )

            # Pace to real-time so WebSocket doesn't flood
            time.sleep(1.0 / max(fps, 30.0))

        cap.release()
        progress_bar.progress(1.0)

        # -------------------------------------------------------------------
        # Final summary
        # -------------------------------------------------------------------
        if demo.hr_history:
            final_hr = float(np.median(demo.hr_history))
            if gt_hr is not None:
                err = abs(final_hr - gt_hr)
                st.success(
                    f"✅ **Processing complete!**  "
                    f"Predicted: **{final_hr:.1f} BPM** · "
                    f"Ground Truth: **{gt_hr:.1f} BPM** · "
                    f"Error: **{err:.1f} BPM**"
                )
            else:
                st.success(f"✅ Estimated HR: **{final_hr:.1f} BPM**")
        else:
            st.warning("Processing finished but no HR estimate produced (insufficient signal).")
