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
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def apply_istft(mag, phase):
    stft_complex = mag * np.exp(1j * phase)
    return librosa.istft(stft_complex, hop_length=HOP_LENGTH, window=WINDOW_TYPE)
