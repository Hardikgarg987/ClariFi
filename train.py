# train.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from utils.preprocess import prepare_data

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "frame_model.h5")
NORM_PATH = os.path.join(MODEL_DIR, "norm.pkl")

def build_frame_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(1024, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(clean_path, noisy_path):
    X, Y, mean, std = prepare_data(clean_path, noisy_path)
    model = build_frame_model(X.shape[1])

    history = model.fit(X, Y, epochs=10, batch_size=32)

    # Save model and normalization
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH, include_optimizer=False)
    with open(NORM_PATH, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)

    # Plot training loss
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    print(f"✅ Model saved at: {MODEL_PATH}")
    print(f"✅ Normalization saved at: {NORM_PATH}")

if __name__ == "__main__":
    train_model("dataset/clean", "dataset/noisy")
