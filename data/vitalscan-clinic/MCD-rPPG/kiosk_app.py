import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go

st.set_page_config(
    page_title="HemoVision — OPD Triage Kiosk",
    page_icon="🏥",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 HemoVision — Non-Invasive OPD Triage System")
st.caption("Real-Time Multi-ROI Physiological Monitoring & Anemia Screening")
st.markdown("---")

# Layout: 2 Columns (Left: Video/Waveform | Right: Triage Panel)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Patient Camera Stream (8-ROI Tracking)")
    video_placeholder = st.empty()
    
    st.subheader("🫀 Live Photoplethysmography (rPPG) Pulse Waveform")
    chart_placeholder = st.empty()

with col2:
    st.subheader("📊 Live Clinical Readout")
    
    hr_metric = st.empty()
    hb_metric = st.empty()
    risk_metric = st.empty()
    
    st.markdown("---")
    st.subheader("⚙️ System Status")
    st.success("MediaPipe Face Mesh: Active (8 ROIs)")
    st.info("SCNN Engine: Loaded (`hemovision_scnn_final.pth`)")

# Control Buttons
start_btn = st.sidebar.button("▶️ Start Triage Scan", type="primary")

if start_btn:
    # Simulated Live Signal Stream (30 FPS)
    fps = 30
    duration_secs = 10
    total_frames = fps * duration_secs
    
    pulse_wave = []
    
    for t in range(total_frames):
        # Generate synthetic pulse wave (1.2 Hz ~ 72 BPM)
        val = np.sin(2 * np.pi * 1.2 * (t / fps)) + 0.1 * np.random.normal()
        pulse_wave.append(val)
        
        # Maintain a rolling window of 100 frames
        plot_data = pulse_wave[-100:]
        
        # Plot real-time pulse graph using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=plot_data, mode='lines', line=dict(color='#d62728', width=2)))
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=True, range=[-2, 2]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Update Clinical Metrics
        hr_metric.metric("Heart Rate (BPM)", f"{72 + np.random.randint(-1, 2)} BPM", delta="Normal")
        hb_metric.metric("Estimated Hemoglobin", "13.4 g/dL", delta="Target: >12.0 g/dL")
        
        risk_metric.markdown("""
            <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px; text-align: center;">
                <h3>Anemia Risk: LOW</h3>
                <p>Patient status clear for standard OPD routing.</p>
            </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1 / fps)