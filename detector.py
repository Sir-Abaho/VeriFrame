"""
detector.py — AI Video Detection Engine
========================================
Detects whether a video was AI-generated or filmed by a human using:
  1. FFT Frequency Analysis     — AI video has unnaturally smooth frequency falloff
  2. Noise Floor Analysis       — Real cameras have organic sensor noise; AI video doesn't
  3. Temporal Consistency       — AI frames have unnatural frame-to-frame delta patterns
  4. DCT Coefficient Analysis   — AI video DCT histograms cluster differently
  5. Edge Coherence             — AI video edges are unnaturally sharp or unnaturally smooth
"""

import cv2
import numpy as np
from scipy.fft import fft2, fftshift
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MAX_FRAMES        = 40      # frames sampled per video
FRAME_RESIZE      = (256, 256)  # resize for consistent analysis
TEMPORAL_PAIRS    = 15      # consecutive frame pairs for motion analysis


# ─────────────────────────────────────────────
#  FRAME SAMPLER
# ─────────────────────────────────────────────
def extract_frames(video_path: str, max_frames: int = MAX_FRAMES) -> list[np.ndarray]:
    """Uniformly sample frames from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError("Video has no readable frames.")

    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, FRAME_RESIZE)
            frames.append(frame)

    cap.release()
    return frames


# ─────────────────────────────────────────────
#  ANALYSIS 1 — FFT FREQUENCY SPECTRUM
# ─────────────────────────────────────────────
def analyze_fft(frames: list[np.ndarray]) -> dict:
    """
    Real camera footage has a natural 1/f power spectrum with organic high-freq noise.
    AI-generated video has an unnaturally smooth falloff — the high frequencies
    are either missing entirely or eerily regular.

    We measure:
      - High-frequency energy ratio (AI video = lower)
      - Spectral flatness (AI video = less flat, more concentrated)
      - Radial power slope (AI video = steeper dropoff)
    """
    hf_ratios = []
    spectral_flatness_scores = []
    radial_slopes = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 2D FFT
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude = np.abs(fshift)
        power = magnitude ** 2

        h, w = power.shape
        cy, cx = h // 2, w // 2

        # High-frequency energy ratio: outer 40% of spectrum
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)

        low_mask  = dist < (max_dist * 0.3)
        high_mask = dist > (max_dist * 0.6)

        low_energy  = power[low_mask].sum()
        high_energy = power[high_mask].sum()
        total_energy = power.sum() + 1e-10

        hf_ratios.append(high_energy / total_energy)

        # Spectral flatness (geometric mean / arithmetic mean of spectrum)
        flat_power = power.flatten() + 1e-10
        geo_mean = np.exp(np.mean(np.log(flat_power)))
        arith_mean = np.mean(flat_power)
        spectral_flatness_scores.append(geo_mean / arith_mean)

        # Radial power slope — fit line to log-log radial profile
        radial_bins = 30
        bin_edges = np.linspace(1, max_dist, radial_bins + 1)
        radial_power = []
        for i in range(radial_bins):
            mask = (dist >= bin_edges[i]) & (dist < bin_edges[i+1])
            if mask.sum() > 0:
                radial_power.append(power[mask].mean())

        if len(radial_power) > 5:
            x_log = np.log(np.arange(1, len(radial_power) + 1))
            y_log = np.log(np.array(radial_power) + 1e-10)
            slope = np.polyfit(x_log, y_log, 1)[0]
            radial_slopes.append(slope)

    return {
        "hf_ratio":          float(np.mean(hf_ratios)),
        "spectral_flatness": float(np.mean(spectral_flatness_scores)),
        "radial_slope":      float(np.mean(radial_slopes)) if radial_slopes else 0.0,
    }


# ─────────────────────────────────────────────
#  ANALYSIS 2 — NOISE FLOOR
# ─────────────────────────────────────────────
def analyze_noise(frames: list[np.ndarray]) -> dict:
    """
    Real cameras introduce photon shot noise + sensor read noise.
    AI-generated video is either too clean (near-zero noise) or has
    structured/patterned noise that differs from organic camera noise.

    We measure:
      - Mean local noise magnitude
      - Noise kurtosis (AI noise is more Gaussian-regular; real noise is spikier)
      - Noise spatial variance (how uniform is the noise across the frame)
    """
    noise_magnitudes = []
    noise_kurtoses   = []
    noise_variances  = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Estimate noise by subtracting a slightly blurred version
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise   = gray - blurred

        noise_magnitudes.append(float(np.std(noise)))
        noise_kurtoses.append(float(kurtosis(noise.flatten())))

        # Spatial variance of noise in 8x8 blocks
        h, w = noise.shape
        block_stds = []
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = noise[y:y+8, x:x+8]
                block_stds.append(np.std(block))
        noise_variances.append(float(np.var(block_stds)))

    return {
        "noise_magnitude": float(np.mean(noise_magnitudes)),
        "noise_kurtosis":  float(np.mean(noise_kurtoses)),
        "noise_variance":  float(np.mean(noise_variances)),
    }


# ─────────────────────────────────────────────
#  ANALYSIS 3 — TEMPORAL CONSISTENCY
# ─────────────────────────────────────────────
def analyze_temporal(frames: list[np.ndarray]) -> dict:
    """
    In real video, motion blur and camera shake create natural inter-frame differences.
    AI video often has either too-smooth or unnaturally abrupt frame transitions.

    We measure:
      - Mean absolute frame delta (how much changes per frame)
      - Delta variance (how consistent/inconsistent the motion is)
      - Optical flow irregularity (skewness of flow magnitude distribution)
    """
    if len(frames) < 2:
        return {"mean_delta": 0.0, "delta_variance": 0.0, "flow_skew": 0.0}

    pairs = min(TEMPORAL_PAIRS, len(frames) - 1)
    step  = max(1, (len(frames) - 1) // pairs)

    deltas     = []
    flow_skews = []

    for i in range(0, len(frames) - step, step):
        f1 = cv2.cvtColor(frames[i],      cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(frames[i+step], cv2.COLOR_BGR2GRAY)

        # Absolute pixel delta
        diff = cv2.absdiff(f1, f2).astype(np.float32)
        deltas.append(float(diff.mean()))

        # Optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            f1, f2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).flatten()
        flow_skews.append(float(skew(magnitude)))

    delta_arr = np.array(deltas)
    return {
        "mean_delta":    float(delta_arr.mean()),
        "delta_variance":float(delta_arr.var()),
        "flow_skew":     float(np.mean(flow_skews)),
    }


# ─────────────────────────────────────────────
#  ANALYSIS 4 — DCT COEFFICIENT DISTRIBUTION
# ─────────────────────────────────────────────
def analyze_dct(frames: list[np.ndarray]) -> dict:
    """
    DCT (Discrete Cosine Transform) is used in JPEG/H.264 compression.
    AI-generated video tends to have DCT coefficients that cluster too cleanly
    around zero — real footage has a heavier-tailed distribution.

    We measure:
      - DCT coefficient kurtosis (AI = lower, too Gaussian)
      - High-AC energy ratio (AI = unusually low)
    """
    dct_kurtoses = []
    ac_ratios    = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Process in 8x8 blocks (like JPEG)
        h, w = gray.shape
        coeffs_all = []
        ac_energies = []
        dc_energies = []

        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = gray[y:y+8, x:x+8]
                dct_block = cv2.dct(block)
                coeffs_all.extend(dct_block.flatten().tolist())
                dc_energies.append(float(dct_block[0, 0]**2))
                ac_energies.append(float((dct_block**2).sum() - dct_block[0,0]**2))

        if coeffs_all:
            dct_kurtoses.append(float(kurtosis(coeffs_all)))
            total_e = sum(ac_energies) + sum(dc_energies) + 1e-10
            ac_ratios.append(sum(ac_energies) / total_e)

    return {
        "dct_kurtosis": float(np.mean(dct_kurtoses)) if dct_kurtoses else 0.0,
        "ac_ratio":     float(np.mean(ac_ratios))    if ac_ratios    else 0.0,
    }


# ─────────────────────────────────────────────
#  ANALYSIS 5 — EDGE COHERENCE
# ─────────────────────────────────────────────
def analyze_edges(frames: list[np.ndarray]) -> dict:
    """
    AI video edges are often unnaturally crisp (over-sharpened by the model)
    or unnaturally smooth (model avoids high-frequency detail).
    Real footage has organic edge distributions with natural aliasing.

    We measure:
      - Edge density (Canny edges / total pixels)
      - Edge gradient variance (spread of edge strengths)
      - Edge regularity (how periodic/structured the edges are)
    """
    edge_densities  = []
    gradient_vars   = []
    edge_regularities = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Canny edge map
        edges = cv2.Canny(gray, 50, 150)
        edge_densities.append(float(edges.sum() / (255 * edges.size)))

        # Sobel gradient variance
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_vars.append(float(grad_magnitude.var()))

        # Edge regularity via FFT of edge map
        edge_f = fftshift(fft2(edges.astype(np.float32)))
        edge_power = np.abs(edge_f)**2
        flat = edge_power.flatten() + 1e-10
        regularity = float(np.exp(np.mean(np.log(flat))) / np.mean(flat))
        edge_regularities.append(regularity)

    return {
        "edge_density":    float(np.mean(edge_densities)),
        "gradient_var":    float(np.mean(gradient_vars)),
        "edge_regularity": float(np.mean(edge_regularities)),
    }


# ─────────────────────────────────────────────
#  SCORING ENGINE
# ─────────────────────────────────────────────
def compute_score(fft_stats: dict, noise_stats: dict,
                  temporal_stats: dict, dct_stats: dict,
                  edge_stats: dict) -> dict:
    """
    Combine all signals into a single AI probability score (0–100).
    Each signal is normalized and weighted based on its discriminative power.

    Score > 70  → likely AI-generated
    Score 40–70 → uncertain
    Score < 40  → likely human-filmed
    """
    signals = {}

    # --- FFT signals ---
    # Low HF ratio → more AI-like (AI video lacks high-freq content)
    signals["fft_hf"]       = 1.0 - min(fft_stats["hf_ratio"] / 0.15, 1.0)

    # Low spectral flatness → more AI-like
    signals["fft_flatness"] = 1.0 - min(fft_stats["spectral_flatness"] / 0.05, 1.0)

    # Very steep radial slope → more AI-like (energy drops off fast)
    slope = fft_stats["radial_slope"]
    signals["fft_slope"]    = min(abs(slope) / 6.0, 1.0) if slope < -1 else 0.0

    # --- Noise signals ---
    # Very low noise → more AI-like
    signals["noise_mag"]    = 1.0 - min(noise_stats["noise_magnitude"] / 8.0, 1.0)

    # Low kurtosis noise → more AI-like (AI noise is too regular)
    nk = noise_stats["noise_kurtosis"]
    signals["noise_kurt"]   = 1.0 - min(max(nk, 0) / 5.0, 1.0)

    # Very low noise spatial variance → AI-like (uniform noise = synthetic)
    signals["noise_var"]    = 1.0 - min(noise_stats["noise_variance"] / 2.0, 1.0)

    # --- Temporal signals ---
    # Very low or very high delta variance → AI-like (too smooth or glitchy)
    dv = temporal_stats["delta_variance"]
    signals["temp_delta"]   = 1.0 - min(dv / 50.0, 1.0) if dv < 20 else min(dv / 300.0, 1.0)

    # Abnormal flow skew → AI-like
    fs = abs(temporal_stats["flow_skew"])
    signals["temp_flow"]    = min(fs / 3.0, 1.0)

    # --- DCT signals ---
    # Low DCT kurtosis → AI-like (too Gaussian, not spiky enough)
    dk = dct_stats["dct_kurtosis"]
    signals["dct_kurt"]     = 1.0 - min(max(dk, 0) / 10.0, 1.0)

    # Very low AC ratio → AI-like (no high-frequency DCT energy)
    signals["dct_ac"]       = 1.0 - min(dct_stats["ac_ratio"] / 0.85, 1.0)

    # --- Edge signals ---
    # Edge density either too low or too high → AI-like
    ed = edge_stats["edge_density"]
    signals["edge_density"] = abs(ed - 0.08) / 0.08  # natural is ~8%
    signals["edge_density"] = min(signals["edge_density"], 1.0)

    # Very low gradient variance → AI over-smoothed
    gv = edge_stats["gradient_var"]
    signals["edge_grad"]    = 1.0 - min(gv / 3000.0, 1.0)

    # --- Weighted combination ---
    weights = {
        "fft_hf":       0.12,
        "fft_flatness": 0.10,
        "fft_slope":    0.08,
        "noise_mag":    0.10,
        "noise_kurt":   0.07,
        "noise_var":    0.06,
        "temp_delta":   0.09,
        "temp_flow":    0.08,
        "dct_kurt":     0.09,
        "dct_ac":       0.08,
        "edge_density": 0.07,
        "edge_grad":    0.06,
    }

    total_weight = sum(weights.values())
    raw_score = sum(signals[k] * weights[k] for k in signals) / total_weight
    ai_score  = round(raw_score * 100, 1)

    if ai_score >= 68:
        verdict = "AI-Generated"
        confidence = "High" if ai_score >= 80 else "Moderate"
    elif ai_score >= 42:
        verdict = "Uncertain"
        confidence = "Low"
    else:
        verdict = "Human-Filmed"
        confidence = "High" if ai_score <= 28 else "Moderate"

    return {
        "ai_score":    ai_score,
        "verdict":     verdict,
        "confidence":  confidence,
        "signals":     {k: round(v * 100, 1) for k, v in signals.items()},
    }


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────
def analyze_video(video_path: str, progress_callback=None) -> dict:
    """
    Full pipeline. Returns a result dict with score, verdict, and per-signal breakdown.

    progress_callback(step: int, total: int, message: str) — optional
    """
    def progress(step, msg):
        if progress_callback:
            progress_callback(step, 6, msg)

    progress(1, "Extracting frames...")
    frames = extract_frames(video_path)
    if not frames:
        raise ValueError("Could not extract any frames from the video.")

    progress(2, "Running FFT frequency analysis...")
    fft_stats = analyze_fft(frames)

    progress(3, "Analyzing noise floor...")
    noise_stats = analyze_noise(frames)

    progress(4, "Checking temporal consistency...")
    temporal_stats = analyze_temporal(frames)

    progress(5, "Running DCT coefficient analysis...")
    dct_stats = analyze_dct(frames)

    progress(6, "Measuring edge coherence...")
    edge_stats = analyze_edges(frames)

    result = compute_score(fft_stats, noise_stats, temporal_stats, dct_stats, edge_stats)

    result["details"] = {
        "frames_analyzed": len(frames),
        "fft":             fft_stats,
        "noise":           noise_stats,
        "temporal":        temporal_stats,
        "dct":             dct_stats,
        "edges":           edge_stats,
    }

    return result


# ─────────────────────────────────────────────
#  CLI TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python detector.py <video_file>")
        sys.exit(1)

    def cli_progress(step, total, msg):
        print(f"  [{step}/{total}] {msg}")

    print(f"\n🔍 Analyzing: {sys.argv[1]}\n")
    result = analyze_video(sys.argv[1], progress_callback=cli_progress)

    print(f"\n{'='*45}")
    print(f"  VERDICT:    {result['verdict']}")
    print(f"  AI SCORE:   {result['ai_score']} / 100")
    print(f"  CONFIDENCE: {result['confidence']}")
    print(f"{'='*45}\n")
    print("Signal breakdown:")
    for sig, val in result["signals"].items():
        bar = "█" * int(val / 5)
        print(f"  {sig:<16} {val:>5.1f}  {bar}")
    print()
