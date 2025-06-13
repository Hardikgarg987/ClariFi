# utils/metrics.py

import numpy as np
from pesq import pesq
from pystoi.stoi import stoi

def segmental_snr(clean, enhanced, frame_len=512, overlap=256):
    eps = 1e-10
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    seg_snr = []
    for i in range(0, min_len - frame_len, frame_len - overlap):
        clean_seg = clean[i:i + frame_len]
        enhanced_seg = enhanced[i:i + frame_len]
        noise = clean_seg - enhanced_seg

        signal_energy = np.sum(clean_seg ** 2)
        noise_energy = np.sum(noise ** 2) + eps

        if signal_energy > eps:
            seg_snr.append(10 * np.log10(signal_energy / noise_energy))

    return np.mean(seg_snr) if seg_snr else 0.0

def compute_pesq(clean, enhanced, sr=16000):
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    try:
        return pesq(sr, clean, enhanced, 'wb')  # Wideband PESQ
    except Exception as e:
        print(f"[Warning] PESQ computation failed: {e}")
        return 0.0

def compute_stoi(clean, enhanced, sr=16000):
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    try:
        return stoi(clean, enhanced, sr, extended=False)
    except Exception as e:
        print(f"[Warning] STOI computation failed: {e}")
        return 0.0
