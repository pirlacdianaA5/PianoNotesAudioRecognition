import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

output_folder = 'Diagrams'
os.makedirs(output_folder, exist_ok=True)

###
DATASET_PATH = 'dataset_keyboard'
data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    raise FileNotFoundError(f"Dataset folder '{DATASET_PATH}' not found in the project directory.")

###
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)

train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=64000,
    subset='both')

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

train_ds.element_spec

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

for example_audio, example_labels in train_ds.take(1):
  print(example_audio.shape)
  print(example_labels.shape)

#############################################################################################################################
label_names[[1, 1, 3, 0]]

plt.figure(figsize=(16, 10))
rows = 3
cols = 3
n = rows * cols
for i in range(n):
  plt.subplot(rows, cols, i+1)
  audio_signal = example_audio[i]
  plt.plot(audio_signal)
  plt.title(label_names[example_labels[i]])
  plt.yticks(np.arange(-1.2, 1.2, 0.2))
  plt.ylim([-1.1, 1.1])



##############################################################################################################################
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

for i in range(3):
  label = label_names[example_labels[i]]
  waveform = example_audio[i]
  spectrogram = get_spectrogram(waveform)

  print('Label:', label)
  print('Waveform shape:', waveform.shape)
  print('Spectrogram shape:', spectrogram.shape)
  print('Audio playback')
  display.display(display.Audio(waveform, rate=64000))


def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)

  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)


fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 64000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
plt.show()
output_path = os.path.join(output_folder, f'diagram_{i + 1}.png')
plt.savefig(output_path)
plt.close()


def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)


train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)


# Load the models
model1 = tf.keras.models.load_model("best_model_cnn.h5")
model2 = tf.saved_model.load("saved_200L2")


y_true = []
y_pred1_classes = []
for x_batch, y_batch in test_spectrogram_ds:
    preds = model1.predict(x_batch)
    y_true.extend(y_batch.numpy())
    y_pred1_classes.extend(np.argmax(preds, axis=1))


y_pred2_classes = []
for x_batch, _ in test_spectrogram_ds:

    x_batch_waveform = tf.squeeze(x_batch, axis=-1)
    preds = model2(x_batch_waveform)["class_ids"].numpy()
    y_pred2_classes.extend(preds)

# Function to calculate metrics
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro'),
        "recall": recall_score(y_true, y_pred, average='macro'),
        "f1": f1_score(y_true, y_pred, average='macro'),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

# Evaluate both models
metrics1 = evaluate(y_true, y_pred1_classes)
metrics2 = evaluate(y_true, y_pred2_classes)

# Print results
print("Model 1 Metrics (best_model_cnn.h5):")
for k, v in metrics1.items():
    print(f"{k}: {v}")

print("\nModel 2 Metrics (saved_model.pb):")
for k, v in metrics2.items():
    print(f"{k}: {v}")