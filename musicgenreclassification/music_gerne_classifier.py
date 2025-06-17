import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        features = [
            np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            np.mean(librosa.feature.rms(y=y)),
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(y)),
        ]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        return np.array(features)
    except Exception as e:
        print(f"⚠️ Error processing {file_path}:\n{e}")
        return None

# Set your dataset path
data_dir = "genres"  # Folder with genre subfolders and .wav files
genres = os.listdir(data_dir)

print("🎵 Extracting features...")
X, y = [], []
for genre in genres:
    genre_path = os.path.join(data_dir, genre)
    for file in os.listdir(genre_path):
        if file.endswith('.wav'):
            file_path = os.path.join(genre_path, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(genre)

X = np.array(X)
y = np.array(y)
print(f"✅ Feature extraction completed.\n🔢 Total samples: {len(X)}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "genre_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("💾 Model and scaler saved.")
