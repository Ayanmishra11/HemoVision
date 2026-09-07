# ==========================================
# CELL 7: CLASSICAL SIGNAL PROCESSING TOOLKIT (POS / CHROM)
# FIX: estimate_hr_fft now zero-pads the Welch PSD via nfft, removing the
# rigid 7.5 BPM quantization grid that was forcing every estimate onto a
# multiple of 7.5 (nperseg=fs*8=240 samples at fs=30 -> 30/240=0.125Hz=7.5BPM
# bins). Confirmed via the self-test itself landing exactly on 75.0 BPM
# (a 7.5-multiple) against a true 72.0 BPM signal. Self-test tolerance also
# tightened from 5.0 to 2.0 BPM — the looser tolerance was wide enough to
# hide this exact bug as a passing test.
# ==========================================
import numpy as np
from scipy.signal import butter, filtfilt, detrend, welch

# ---------- BGR / RGB Auto-Correction ----------

def verify_and_fix_channel_order(roi_signal: np.ndarray) -> np.ndarray:
    """
    Ensures RGB ordering [3, T] or [ROIs, 3, T].
    Human skin under standard lighting shows Red > Blue mean intensity.
    """
    sig = roi_signal.copy()
    if sig.ndim == 3:
        r_mean = np.mean(sig[:, 0, :])
        b_mean = np.mean(sig[:, 2, :])
        if r_mean < b_mean:  # Indicates BGR format [Blue, Green, Red]
            sig = sig[:, ::-1, :]
    elif sig.ndim == 2:
        r_mean = np.mean(sig[0, :])
        b_mean = np.mean(sig[2, :])
        if r_mean < b_mean:
            sig = sig[::-1, :]
    return sig

# ---------- Short-Time Overlap-Add Projections ----------

def pos_projection(rgb_signal: np.ndarray, fs: float = 30.0, win_sec: float = 1.6, overlap: float = 0.75) -> np.ndarray:
    """
    Plane-Orthogonal-to-Skin (Wang et al. 2016) with temporal overlap-add sub-windowing.
    rgb_signal shape: [3, T]
    """
    channels, T = rgb_signal.shape
    win_len = int(win_sec * fs)
    stride = max(1, int(win_len * (1.0 - overlap)))

    if T < win_len:
        mean_val = np.mean(rgb_signal, axis=-1, keepdims=True) + 1e-8
        norm = rgb_signal / mean_val
        R, G, B = norm[0], norm[1], norm[2]
        S1, S2 = G - B, G + B - 2.0 * R
        alpha = (np.std(S1) + 1e-8) / (np.std(S2) + 1e-8)
        return S1 + alpha * S2

    H = np.zeros(T)
    W = np.zeros(T)
    hanning = np.hanning(win_len)

    for start in range(0, T - win_len + 1, stride):
        end = start + win_len
        sub = rgb_signal[:, start:end]

        dc = np.mean(sub, axis=1, keepdims=True) + 1e-8
        norm = sub / dc
        R, G, B = norm[0], norm[1], norm[2]

        S1 = G - B
        S2 = G + B - 2.0 * R

        alpha = (np.std(S1) + 1e-8) / (np.std(S2) + 1e-8)
        S = S1 + alpha * S2

        H[start:end] += S * hanning
        W[start:end] += hanning

    W[W == 0] = 1e-6
    return H / W

def chrom_projection(rgb_signal: np.ndarray, fs: float = 30.0, win_sec: float = 1.6, overlap: float = 0.75) -> np.ndarray:
    """
    Chrominance-based rPPG (de Haan & Jeanne 2013) with temporal overlap-add sub-windowing.
    rgb_signal shape: [3, T]
    """
    channels, T = rgb_signal.shape
    win_len = int(win_sec * fs)
    stride = max(1, int(win_len * (1.0 - overlap)))

    if T < win_len:
        mean_val = np.mean(rgb_signal, axis=-1, keepdims=True) + 1e-8
        norm = (rgb_signal - mean_val) / mean_val
        R, G, B = norm[0], norm[1], norm[2]
        Xs = 3.0 * R - 2.0 * G
        Ys = 1.5 * R + G - 1.5 * B
        alpha = (np.std(Xs) + 1e-8) / (np.std(Ys) + 1e-8)
        return Xs - alpha * Ys

    H = np.zeros(T)
    W = np.zeros(T)
    hanning = np.hanning(win_len)

    for start in range(0, T - win_len + 1, stride):
        end = start + win_len
        sub = rgb_signal[:, start:end]

        dc = np.mean(sub, axis=1, keepdims=True) + 1e-8
        norm = (sub - dc) / dc
        R, G, B = norm[0], norm[1], norm[2]

        Xs = 3.0 * R - 2.0 * G
        Ys = 1.5 * R + G - 1.5 * B

        alpha = (np.std(Xs) + 1e-8) / (np.std(Ys) + 1e-8)
        S = Xs - alpha * Ys

        H[start:end] += S * hanning
        W[start:end] += hanning

    W[W == 0] = 1e-6
    return H / W

# ---------- AC/DC Normalization (Retained for Compatibility) ----------

def ac_dc_normalize(rgb_signal: np.ndarray) -> np.ndarray:
    mean_val = np.mean(rgb_signal, axis=-1, keepdims=True) + 1e-8
    return (rgb_signal - mean_val) / mean_val

# ---------- Bandpass Filter & Welch PSD Peak Estimation ----------

def _get_band_limits(cfg=None):
    if cfg is not None:
        return cfg.BAND_LOW_HZ, cfg.BAND_HIGH_HZ
    if "CFG" in globals():
        global_cfg = globals()["CFG"]
        return getattr(global_cfg, "BAND_LOW_HZ", 0.65), getattr(global_cfg, "BAND_HIGH_HZ", 3.25)
    return 0.65, 3.25  # Fallback to standard physiological band

def bandpass_filter(sig: np.ndarray, fs: float, cfg=None, order: int = 3) -> np.ndarray:
    low_hz, high_hz = _get_band_limits(cfg)
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = min(high_hz, nyq - 0.1) / nyq
    if not (0 < low < high < 1.0):
        raise ValueError(f"Invalid filter band for fs={fs}: low={low}, high={high} (must be in (0,1))")
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, sig)

def estimate_hr_fft(sig: np.ndarray, fs: float, cfg=None, n_fft: int = 8192) -> float:
    """
    FIX: nfft now explicitly passed to welch(), zero-padding the FFT beyond
    nperseg. nperseg (8s window -> 240 samples at fs=30) still bounds the
    TRUE distinguishable frequency resolution -- that's a separate, harder
    limit that would need longer windows to improve. What nfft fixes is the
    ARTIFICIAL quantization: without it, welch() only returns bin centers
    spaced at fs/nperseg = 7.5 BPM apart at fs=30, forcing every estimate
    onto that rigid grid regardless of the true underlying peak location.
    With nfft >> nperseg, the same underlying spectral estimate is
    interpolated across many more bins, so the reported peak can land
    anywhere, not just on multiples of 7.5.
    """
    low_hz, high_hz = _get_band_limits(cfg)
    sig_detrend = detrend(sig)
    nperseg = min(len(sig_detrend), int(fs * 8))
    freqs, psd = welch(
        sig_detrend,
        fs=fs,
        nperseg=nperseg,
        nfft=max(n_fft, nperseg),
    )
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        raise ValueError(f"No frequency bins fall in cardiac band [{low_hz}, {high_hz}] Hz for fs={fs}.")
    peak_freq = freqs[mask][np.argmax(psd[mask])]
    return float(peak_freq * 60.0)

# ---------- End-to-End Classical Pipeline ----------

def classical_hr_estimate(
    roi_signal: np.ndarray,
    fs: float,
    method: str = "POS",
    cfg=None
) -> float:
    """
    Base estimator with auto channel correction, spatial ROI mean, short-time overlap-add projection, and Welch PSD peak estimation.
    """
    assert roi_signal.ndim == 3 and roi_signal.shape[1] == 3, f"Expected [ROI,3,T], got {roi_signal.shape}"

    # 1. Verify/Fix BGR -> RGB
    roi_signal_rgb = verify_and_fix_channel_order(roi_signal)

    # 2. Spatial mean across ROIs
    global_rgb = roi_signal_rgb.mean(axis=0)  # [3, T]

    # 3. Short-time overlap-add projection
    if method.upper() == "POS":
        pulse = pos_projection(global_rgb, fs=fs)
    elif method.upper() == "CHROM":
        pulse = chrom_projection(global_rgb, fs=fs)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 4. Bandpass and Spectral HR estimation
    filtered = bandpass_filter(pulse, fs, cfg)
    return estimate_hr_fft(filtered, fs, cfg)

def classical_hr_estimate_windowed(
    roi_signal: np.ndarray,
    fs: float,
    method: str = "POS",
    cfg=None,
    window_sec: float = 10.0,
    step_sec: float = 2.0
) -> float:
    """
    Sliding-window POS/CHROM + median-vote across windows.
    """
    T = roi_signal.shape[-1]
    win_len = int(window_sec * fs)
    step = int(step_sec * fs)

    if T < win_len:
        return classical_hr_estimate(roi_signal, fs=fs, method=method, cfg=cfg)

    estimates = []
    for start in range(0, T - win_len + 1, step):
        chunk = roi_signal[:, :, start:start + win_len]
        try:
            est = classical_hr_estimate(chunk, fs=fs, method=method, cfg=cfg)
            estimates.append(est)
        except Exception:
            continue

    if not estimates:
        return classical_hr_estimate(roi_signal, fs=fs, method=method, cfg=cfg)

    return float(np.median(estimates))

# ==========================================
# MANDATORY SELF-TEST
# FIX: tolerance tightened from 5.0 to 2.0 BPM. The looser 5.0 tolerance
# was wide enough to let a quantized 75.0 BPM estimate (vs a true 72.0 BPM
# signal) pass silently -- that's the exact bug this test exists to catch.
# ==========================================
def _run_classical_toolkit_self_test():
    log_func = getattr(globals().get("log"), "info", print)
    log_func("Running updated classical toolkit self-test...")
    fs_test = 30.0
    duration_s = 20.0
    t = np.arange(0, duration_s, 1.0 / fs_test)
    true_bpm = 72.0
    true_freq_hz = true_bpm / 60.0

    seed = getattr(globals().get("CFG"), "SEED", 42)
    rng = np.random.default_rng(seed)
    pulse_shape = np.sin(2 * np.pi * true_freq_hz * t)

    channel_pulse_strength = {"R": 0.010, "G": 0.020, "B": 0.006}
    num_rois = getattr(globals().get("CFG"), "NUM_ROIS", 8)
    signal = np.zeros((num_rois, 3, len(t)), dtype=np.float32)
    base_rgb = np.array([180.0, 140.0, 120.0])
    strengths = np.array([channel_pulse_strength["R"], channel_pulse_strength["G"], channel_pulse_strength["B"]])

    for roi in range(num_rois):
        noise = rng.normal(0, 0.5, size=(3, len(t)))
        for ch in range(3):
            signal[roi, ch] = base_rgb[ch] * (1.0 + strengths[ch] * pulse_shape) + noise[ch]

    for method in ("POS", "CHROM"):
        est_bpm = classical_hr_estimate_windowed(signal, fs=fs_test, method=method)
        error = abs(est_bpm - true_bpm)
        assert error < 2.0, (
            f"{method} self-test FAILED: estimated {est_bpm:.1f} BPM, "
            f"expected ~{true_bpm} BPM (error {error:.1f} exceeds 2.0 BPM tolerance)"
        )
        log_func(f"  {method} self-test PASSED: estimated {est_bpm:.1f} BPM (true={true_bpm}, error={error:.2f})")

    log_func("Classical toolkit (Windowed POS/CHROM) verified. Ready for Cell 8.")

_run_classical_toolkit_self_test()