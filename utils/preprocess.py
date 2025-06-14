# utils/preprocess.py

import os
import numpy as np
import librosa

from .audio_utils import N_FFT, HOP_LENGTH, WINDOW_TYPE, SR

def extract_features(file_path, sr=SR):
    try:
        y, _ = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"[Error] Failed to load {file_path}: {e}")
        return None, None, None

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW_TYPE)
    mag = np.abs(stft)
    db = librosa.amplitude_to_db(mag)
    return db.T, y, stft

def prepare_data(clean_path, noisy_path):
    X_frames, Y_frames = [], []

    for fname in os.listdir(clean_path):
        clean_file = os.path.join(clean_path, fname)
        noisy_file = os.path.join(noisy_path, fname)

        if os.path.exists(noisy_file):
            clean_feats, _, _ = extract_features(clean_file)
            noisy_feats, _, _ = extract_features(noisy_file)

            if clean_feats is None or noisy_feats is None:
                continue

            min_len = min(clean_feats.shape[0], noisy_feats.shape[0])
            clean_feats = clean_feats[:min_len]
            noisy_feats = noisy_feats[:min_len]

            X_frames.append(noisy_feats)
            Y_frames.append(clean_feats)

    if not X_frames or not Y_frames:
        raise RuntimeError("No valid audio pairs found in the dataset.")

    X = np.vstack(X_frames)
    Y = np.vstack(Y_frames)

    mean = np.mean(X)
    std = np.std(X) + 1e-8  # prevent divide-by-zero
    X_norm = (X - mean) / std
    Y_norm = (Y - mean) / std

    return X_norm, Y_norm, mean, std
