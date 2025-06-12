# enhance.py

import os
import librosa
import numpy as np
import soundfile as sf
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from utils.preprocess import extract_features
from utils.audio_utils import apply_istft, butter_lowpass_filter, HOP_LENGTH, WINDOW_TYPE, SR
from utils.metrics import segmental_snr, compute_pesq, compute_stoi

MODEL_PATH = "models/frame_model.h5"
NORM_PATH = "models/norm.pkl"

def enhance_audio(noisy_file, output_path="outputs/enhanced.wav"):
    # Load model and normalization
    model = load_model(MODEL_PATH, compile=False)
    with open(NORM_PATH, 'rb') as f:
        norm_data = pickle.load(f)

    mean = norm_data['mean']
    std = norm_data['std']

    # File paths
    clean_file = noisy_file.replace("noisy", "clean")

    # Extract features
    noisy_feats, y_noisy, stft_noisy = extract_features(noisy_file)
    clean_feats, _, _ = extract_features(clean_file)

    # Normalize and predict
    norm_noisy = (noisy_feats - mean) / std
    enhanced_frames = model.predict(norm_noisy)
    enhanced_frames = (enhanced_frames * std) + mean
    mag = librosa.db_to_amplitude(enhanced_frames.T)

    phase = np.angle(stft_noisy[:, :mag.shape[1]])
    enhanced_audio = apply_istft(mag, phase)
    enhanced_audio = butter_lowpass_filter(enhanced_audio)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, enhanced_audio, SR)
    print(f"✅ Enhanced audio saved at: {output_path}")

    # Metrics
    seg_snr = segmental_snr(y_noisy, enhanced_audio)
    pesq_val = compute_pesq(y_noisy, enhanced_audio)
    stoi_val = compute_stoi(y_noisy, enhanced_audio)

    print(f"\n📊 Segmental SNR: {seg_snr:.2f} dB")
    print(f"🎧 PESQ: {pesq_val:.2f}")
    print(f"🗣️ STOI: {stoi_val:.2f}")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_noisy), ref=np.max),
                             sr=SR, hop_length=HOP_LENGTH, y_axis='log', x_axis='time')
    plt.title("Input Noisy Spectrogram")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    enhanced_stft = librosa.stft(enhanced_audio, n_fft=512, hop_length=HOP_LENGTH)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(enhanced_stft), ref=np.max),
                             sr=SR, hop_length=HOP_LENGTH, y_axis='log', x_axis='time')
    plt.title("Enhanced Spectrogram")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    enhance_audio("dataset/noisy/p232_023.wav")
