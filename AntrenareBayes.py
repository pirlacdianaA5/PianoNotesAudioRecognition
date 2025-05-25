import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import tensorflow as tf
import pathlib

# Path to the dataset
DATASET_PATH = 'dataset_keyboard'
data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    raise FileNotFoundError(f"Dataset folder '{DATASET_PATH}' not found.")

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
    # Ensure the waveform has a minimum length of 255
    if tf.shape(waveform)[0] < 255:
        waveform = tf.pad(waveform, [[0, 255 - tf.shape(waveform)[0]]])
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
val_spectrograms, val_labels = preprocess_dataset(val_ds)

# Reduce dimensionality using PCA
pca = PCA(n_components=50)  # Adjust the number of components as needed
train_features_pca = pca.fit_transform(train_spectrograms)
val_features_pca = pca.transform(val_spectrograms)

# Train the Naive Bayes model
nb = GaussianNB()
nb.fit(train_features_pca, train_labels)

# Evaluate the model
val_predictions = nb.predict(val_features_pca)
accuracy = accuracy_score(val_labels, val_predictions)
print(f"Naive Bayes Validation Accuracy: {accuracy * 100:.2f}%")