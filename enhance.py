import os
import librosa
import numpy as np
import soundfile as sf
import pickle
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session

from utils.preprocess import extract_features
from utils.audio_utils import apply_istft, butter_lowpass_filter, HOP_LENGTH, SR
# from utils.metrics import segmental_snr, compute_pesq, compute_stoi
# import matplotlib.pyplot as plt

# Reduce TensorFlow memory use
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

MODEL_PATH = "models/frame_model.h5"
NORM_PATH = "models/norm.pkl"

print("🔁 Loading model and normalization data...")
model = load_model(MODEL_PATH, compile=False)
with open(NORM_PATH, 'rb') as f:
    norm_data = pickle.load(f)
mean = norm_data['mean']
std = norm_data['std']

def enhance_audio(noisy_file, output_path="outputs/enhanced.wav"):
    # clean_file = noisy_file.replace("noisy", "clean")  # ❌ Not needed in Lite version

    noisy_feats, y_noisy, stft_noisy = extract_features(noisy_file)
    norm_noisy = (noisy_feats - mean) / std

    enhanced_frames = model.predict(norm_noisy, verbose=0)
    enhanced_frames = (enhanced_frames * std) + mean
    mag = librosa.db_to_amplitude(enhanced_frames.T)

    phase = np.angle(stft_noisy[:, :mag.shape[1]])
    enhanced_audio = apply_istft(mag, phase)
    enhanced_audio = butter_lowpass_filter(enhanced_audio)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, enhanced_audio, SR)
    print(f"✅ Enhanced audio saved at: {output_path}")

    # # ⚠️ Skip metrics
    # seg_snr = segmental_snr(y_noisy, enhanced_audio)
    # pesq_val = compute_pesq(y_noisy, enhanced_audio)
    # stoi_val = compute_stoi(y_noisy, enhanced_audio)

    # # ⚠️ Skip spectrogram plot
    # os.makedirs("static/spectrograms", exist_ok=True)
    # spectrogram_path = f"static/spectrograms/{os.path.basename(output_path).split('.')[0]}_spectrogram.png"
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_noisy), ref=np.max),
    #                         sr=SR, hop_length=HOP_LENGTH, y_axis='log', x_axis='time')
    # plt.title("Input Noisy Spectrogram")
    # plt.colorbar()

    # plt.subplot(1, 2, 2)
    # enhanced_stft = librosa.stft(enhanced_audio, n_fft=512, hop_length=HOP_LENGTH)
    # librosa.display.specshow(librosa.amplitude_to_db(np.abs(enhanced_stft), ref=np.max),
    #                         sr=SR, hop_length=HOP_LENGTH, y_axis='log', x_axis='time')
    # plt.title("Enhanced Spectrogram")
    # plt.colorbar()

    # plt.tight_layout()
    # plt.savefig(spectrogram_path)
    # plt.clf()
    # plt.close('all')

    clear_session()

    return  # No return values in Lite version
