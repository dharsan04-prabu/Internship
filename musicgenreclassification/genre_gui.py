import tkinter as tk
from tkinter import filedialog, messagebox
import pygame
import librosa
import numpy as np
import joblib
from PIL import Image, ImageTk
import os

# Load the trained model and scaler
model = joblib.load("genre_model.pkl")
scaler = joblib.load("scaler.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    features = []
    features.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.rms(y=y)))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for mfcc in mfccs:
        features.append(np.mean(mfcc))
    return np.array(features).reshape(1, -1)

def predict_genre(file_path):
    features = extract_features(file_path)
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]
    return prediction, probabilities

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        try:
            predicted, probas = predict_genre(file_path)
            result_label.config(text=f"Predicted Genre: {predicted}")
            probas_text = "\n".join([f"{label}: {probas[i]*100:.2f}%" for i, label in enumerate(model.classes_)])
            probabilities_label.config(text=f"\nConfidence:\n{probas_text}")
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
        except Exception as e:
            messagebox.showerror("Error", str(e))

def stop_music():
    pygame.mixer.music.stop()

# Initialize pygame mixer
pygame.mixer.init()

# Create the main window
root = tk.Tk()
root.title("🎵 Music Genre Classifier")
root.geometry("600x600")
root.configure(bg="#1e1e2f")  # dark blue/purple background

# Fonts and styles
title_font = ("Helvetica", 20, "bold")
label_font = ("Helvetica", 12)
button_font = ("Helvetica", 12, "bold")

# Logo or banner (optional)
banner_frame = tk.Frame(root, bg="#1e1e2f")
banner_frame.pack(pady=10)
try:
    banner_img = Image.open("banner.png")
    banner_img = banner_img.resize((400, 100))
    banner_photo = ImageTk.PhotoImage(banner_img)
    banner_label = tk.Label(banner_frame, image=banner_photo, bg="#1e1e2f")
    banner_label.pack()
except:
    banner_label = tk.Label(banner_frame, text="🎶 Music Genre Classifier 🎶", font=title_font, fg="white", bg="#1e1e2f")
    banner_label.pack()

# Buttons
button_frame = tk.Frame(root, bg="#1e1e2f")
button_frame.pack(pady=20)

browse_button = tk.Button(button_frame, text="📂 Browse Audio", command=browse_file, bg="#3b82f6", fg="white", font=button_font, padx=10, pady=5)
browse_button.grid(row=0, column=0, padx=10)

stop_button = tk.Button(button_frame, text="⏹ Stop Music", command=stop_music, bg="#ef4444", fg="white", font=button_font, padx=10, pady=5)
stop_button.grid(row=0, column=1, padx=10)

# Prediction Results
result_label = tk.Label(root, text="", font=title_font, fg="#22c55e", bg="#1e1e2f")
result_label.pack(pady=20)

# Confidence Probabilities
probabilities_label = tk.Label(root, text="", font=label_font, fg="white", bg="#1e1e2f", justify="left")
probabilities_label.pack(pady=10)

# Run the GUI loop
root.mainloop()
