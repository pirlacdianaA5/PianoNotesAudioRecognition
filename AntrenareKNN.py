import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf

# Path to the dataset
DATASET_PATH = 'dataset_keyboard'
data_dir = tf.keras.utils.get_file(origin=DATASET_PATH, untar=True)

# Load the dataset
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=64000,
    subset='both'
)

# Preprocess the dataset
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

def preprocess_dataset(dataset):
    spectrograms = []
    labels = []
    for waveform, label in dataset:
        spectrogram = get_spectrogram(waveform)
        spectrograms.append(tf.reshape(spectrogram, [-1]).numpy())
        labels.append(label.numpy())
    return np.array(spectrograms), np.array(labels)

# Extract spectrograms and labels for training
train_spectrograms, train_labels = preprocess_dataset(train_ds)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_spectrograms, train_labels)

print("KNN model training complete.")