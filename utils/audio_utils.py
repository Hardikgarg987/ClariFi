# utils/audio_utils.py

import numpy as np
from scipy.signal import butter, lfilter
import librosa

# Constants
N_FFT = 512
HOP_LENGTH = 128
FRAME_LENGTH = N_FFT // 2 + 1
WINDOW_TYPE = 'hann'
SR = 16000

def butter_lowpass_filter(data, cutoff=4000, sr=SR, order=6):
    nyquist = 0.5 * sr
    normal_cutoff = min(0.99, cutoff / nyquist)  # safety clamp
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = lfilter(b, a, data)
    return np.clip(filtered, -1.0, 1.0)  # avoid overflow

def apply_istft(mag, phase):
    # Ensure mag and phase have same shape
    if mag.shape != phase.shape:
        min_t = min(mag.shape[1], phase.shape[1])
        mag = mag[:, :min_t]
        phase = phase[:, :min_t]

    stft_complex = mag * np.exp(1j * phase)
    signal = librosa.istft(stft_complex, hop_length=HOP_LENGTH, window=WINDOW_TYPE)
    return np.clip(signal, -1.0, 1.0)  # normalize signal range
