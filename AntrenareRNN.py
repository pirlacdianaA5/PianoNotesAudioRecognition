# ANTRENARE CU Rețele neuronale convoluționale (CNN)

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
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
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
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
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
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

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)
    ax.set_title(label_names[example_spect_labels[i].numpy()])


train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
plt.show()
output_path = os.path.join(output_folder, f'diagram_{i + 1}.png')
plt.savefig(output_path)
plt.close()


input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(32, 32),
    layers.Normalization(),
    layers.Reshape((32, 32)),  # Reshape to (timesteps, features)
    layers.SimpleRNN(64, activation='tanh', return_sequences=True),
    layers.SimpleRNN(64, activation='tanh'),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

metrics = history.history
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1,2,2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
output_path = os.path.join(output_folder, f'diagram_{i + 1}.png')
plt.savefig(output_path)
plt.close()


model.evaluate(test_spectrogram_ds, return_dict=True)


y_pred = model.predict(test_spectrogram_ds)
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()
output_path = os.path.join(output_folder, f'diagram_{i + 1}.png')
plt.savefig(output_path)
plt.close()

#################################### TESTING ##########################################################
x = data_dir/'Do#0/keyboard_electronic_005-013-025.wav'
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=64000,)
x = tf.squeeze(x, axis=-1)
waveform = x
x = get_spectrogram(x)
x = x[tf.newaxis,...]

prediction = model(x)
x_labels =  ['Do#0' 'Do#1' 'Do#2' 'Do#3' 'Do#4' 'Do#5' 'Do#6' 'Do#7' 'Do#8' 'Do0'
 'Do1' 'Do2' 'Do3' 'Do4' 'Do5' 'Do6' 'Do7' 'Do8' 'Fa#0' 'Fa#1' 'Fa#2'
 'Fa#3' 'Fa#4' 'Fa#5' 'Fa#6' 'Fa#7' 'Fa0' 'Fa1' 'Fa2' 'Fa3' 'Fa4' 'Fa5'
 'Fa6' 'Fa7' 'La#-1' 'La#0' 'La#1' 'La#2' 'La#3' 'La#4' 'La#5' 'La#6'
 'La#7' 'La-1' 'La0' 'La1' 'La2' 'La3' 'La4' 'La5' 'La6' 'La7' 'Mi0' 'Mi1'
 'Mi2' 'Mi3' 'Mi4' 'Mi5' 'Mi6' 'Mi7' 'Re#0' 'Re#1' 'Re#2' 'Re#3' 'Re#4'
 'Re#5' 'Re#6' 'Re#7' 'Re0' 'Re1' 'Re2' 'Re3' 'Re4' 'Re5' 'Re6' 'Re7'
 'Re8' 'Si-1' 'Si0' 'Si1' 'Si2' 'Si3' 'Si4' 'Si5' 'Si6' 'Si7' 'Sol#0'
 'Sol#1' 'Sol#2' 'Sol#3' 'Sol#4' 'Sol#5' 'Sol#6' 'Sol#7' 'Sol0' 'Sol1'
 'Sol2' 'Sol3' 'Sol4' 'Sol5' 'Sol6' 'Sol7']
plt.bar(x_labels, tf.nn.softmax(prediction[0]))
plt.title('DO#0')
plt.show()
output_path = os.path.join(output_folder, f'diagram_{i + 1}.png')
plt.savefig(output_path)
plt.close()

###################################

class ExportModel(tf.Module):
    def __init__(self, model):
        self.model = model

        # Accept either a string-filename or a batch of waveforms.
        # You could add additional signatures for a single wave, or a ragged-batch.
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=(), dtype=tf.string))
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=[None, 64000], dtype=tf.float32))

    @tf.function
    def __call__(self, x):
        # If they pass a string, load the file and decode it.
        if x.dtype == tf.string:
            x = tf.io.read_file(x)
            x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=64000, )
            x = tf.squeeze(x, axis=-1)
            x = x[tf.newaxis, :]

        x = get_spectrogram(x)
        result = self.model(x, training=False)

        class_ids = tf.argmax(result, axis=-1)
        class_names = tf.gather(label_names, class_ids)
        return {'predictions': result,
                'class_ids': class_ids,
                'class_names': class_names}

export = ExportModel(model)
tf.saved_model.save(export, "savedRNN")
imported = tf.saved_model.load("savedRNN")
imported(waveform[tf.newaxis, :])