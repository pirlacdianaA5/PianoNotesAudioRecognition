# ANTRENARE CU Rețele neuronale convoluționale (CNN)

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

class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data
        self.f1_scores = []  # Store F1 scores for all epochs

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred = []

        # Iterate through the validation dataset
        for x_batch, y_batch in self.val_data:
            preds = self.model.predict(x_batch)
            y_true.extend(y_batch.numpy())
            y_pred.extend(tf.argmax(preds, axis=1).numpy())

        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average='macro')
        self.f1_scores.append(f1)
        print(f"Epoch {epoch + 1}: F1 Score = {f1:.4f}")



class PrintBestModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best_epoch = None
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch + 1
            print(f"Best model saved at epoch {self.best_epoch}: {self.filepath}")

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

norm_layer = layers.Normalization()
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

# Regularizare L2

l2_reg = regularizers.l2(1e-4)

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(32, 32),
    norm_layer,

    layers.Conv2D(64, 3, activation='relu', kernel_regularizer=l2_reg),
    layers.Conv2D(128, 3, activation='relu', kernel_regularizer=l2_reg),
    layers.Dropout(0.3),

    layers.Conv2D(128, 3, activation='relu', kernel_regularizer=l2_reg),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=l2_reg),
    layers.Dropout(0.5),

    layers.Dense(num_labels)
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath='best_model_cnn.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
print_best_model_callback = PrintBestModelCallback('best_model_cnn.h5')

f1_callback = F1ScoreCallback(val_spectrogram_ds)

history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=200,
    callbacks=[f1_callback, checkpoint, print_best_model_callback]
)

# After training, print all F1 scores
print("F1 Scores for all epochs:", f1_callback.f1_scores)

# After training
metrics = history.history
epochs = range(1, len(metrics['loss']) + 1)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics['loss'], label='Training Loss', color='blue')
plt.plot(epochs, metrics['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()



# Plot training and validation accuracy
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


# Obține predicțiile și etichetele reale
y_pred = model.predict(test_spectrogram_ds)
y_pred = tf.argmax(y_pred, axis=1).numpy()
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0).numpy()

# Calculează scorul F1###############################################################################################################
f1 = f1_score(y_true, y_pred, average='macro')
print(f"F1 Score: {f1}")

# Plot F1 scores for all epochs
plt.figure(figsize=(10, 6))
epochs = range(1, len(f1_callback.f1_scores) + 1)
plt.plot(epochs, f1_callback.f1_scores, label='F1 Score', color='green')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score per Epoch')
plt.legend()
plt.grid(True)
plt.show()
output_path = os.path.join(output_folder, 'f1_score_diagram.png')
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
tf.saved_model.save(export, "saved_200L2")
imported = tf.saved_model.load("saved_200L2")
imported(waveform[tf.newaxis, :])



if __name__ == '__main__':
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=200,
        callbacks=[f1_callback, checkpoint, print_best_model_callback]
    )

    # After training, print all F1 scores
    print("F1 Scores for all epochs:", f1_callback.f1_scores)