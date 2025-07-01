# ANTRENARE CU Rețele neuronale convoluționale (CNN) - Versiune finală ajustată

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from IPython import display

# Callback pentru F1
class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for x_batch, y_batch in self.val_data:
            preds = self.model.predict(x_batch, verbose=0)
            y_true.extend(y_batch.numpy())
            y_pred.extend(tf.argmax(preds, axis=1).numpy())
        f1 = f1_score(y_true, y_pred, average='macro')
        self.f1_scores.append(f1)
        print(f"Epoch {epoch+1}: F1 Score = {f1:.4f}")

# Fix seed
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Folder pentru diagrame
output_folder = 'Diagrams'
os.makedirs(output_folder, exist_ok=True)

# Încărcare dataset audio cu batch și split
DATASET_PATH = 'dataset_keyboard'
data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    raise FileNotFoundError(f"Dataset folder '{DATASET_PATH}' nu există.")

BATCH_SIZE = 32
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    seed=seed,
    output_sequence_length=64000,
    subset='both'
)
label_names = np.array(train_ds.class_names)
NUM_CLASSES = len(label_names)

# Squeeze doar dimensiunea audio (label-urile rămân scalare)
train_ds = train_ds.map(lambda x, y: (tf.squeeze(x, -1), y), tf.data.AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (tf.squeeze(x, -1), y), tf.data.AUTOTUNE)

# Split val în validare și test
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds  = val_ds.shard(num_shards=2, index=1)
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds  = val_ds.shard(num_shards=2, index=1)

# Funcție SpecAugment
def spec_augment(spectrogram, time_mask_param=30, freq_mask_param=13):
    shape = tf.shape(spectrogram)
    time_max = shape[1]
    freq_max = shape[2]
    # Time mask
    t = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, time_max - t + 1, dtype=tf.int32)
    mask_time = tf.concat([
        tf.ones([t0, freq_max, 1]),
        tf.zeros([t, freq_max, 1]),
        tf.ones([time_max - t0 - t, freq_max, 1])
    ], axis=0)
    # Frequency mask
    f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, freq_max - f + 1, dtype=tf.int32)
    mask_freq = tf.concat([
        tf.ones([time_max, f0, 1]),
        tf.zeros([time_max, f, 1]),
        tf.ones([time_max, freq_max - f0 - f, 1])
    ], axis=1)
    spec = spectrogram * mask_time
    spec = spec * mask_freq
    return spec

# Extracție spectrogramă din waveform
def get_spectrogram(waveform):
    stft = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(stft)[..., tf.newaxis]
    return spectrogram

# Pipeline spectrogramă
def make_spec_ds(ds, augment=False):
    ds = ds.map(lambda x, y: (get_spectrogram(x), y), tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda spec, y: (spec_augment(spec), y), tf.data.AUTOTUNE)
    return ds

train_spec_ds = make_spec_ds(train_ds, augment=True)
val_spec_ds   = make_spec_ds(val_ds)
test_spec_ds  = make_spec_ds(test_ds)


autotune = tf.data.AUTOTUNE
train_spec_ds = train_spec_ds.cache().shuffle(20000).prefetch(autotune)
val_spec_ds   = val_spec_ds.cache().prefetch(autotune)
test_spec_ds  = test_spec_ds.cache().prefetch(autotune)


labels = []
for x_batch, y_batch in train_ds:
    labels.extend(y_batch.numpy())
labels = np.array(labels)
counts = np.bincount(labels, minlength=NUM_CLASSES)
class_weight = {i: np.median(counts) / c if c>0 else 1.0 for i, c in enumerate(counts)}

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3), include_top=False, weights=None
)
inputs = layers.Input(shape=(None, None, 1))
# SpecAugment în model
x = layers.Lambda(lambda spec: spec_augment(spec))(inputs)
# Resize și replicare canale
x = layers.Resizing(224, 224)(x)
x = layers.Concatenate()([x, x, x])
# Transfer learning head
x = base_model(x, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES)(x)
model = models.Model(inputs, outputs)

# Compilare model
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
)
model.summary()

# Callback-uri
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
lr_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
f1_cb = F1ScoreCallback(val_spec_ds)

# Antrenare model
history = model.fit(
    train_spec_ds,
    validation_data=val_spec_ds,
    epochs=100,
    class_weight=class_weight,
    callbacks=[f1_cb, lr_plateau, early_stop]
)
print("F1 Scores pe epoci:", f1_cb.f1_scores)

# Evaluare finală
eval_res = model.evaluate(test_spec_ds, return_dict=True)
print("Eval results:", eval_res)

# Matrice de confuzie
y_pred = tf.argmax(model.predict(test_spec_ds), axis=1).numpy()
y_true = np.concatenate([y.numpy() for _, y in test_spec_ds.unbatch()])
cm = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, xticklabels=label_names, yticklabels=label_names, annot=False, fmt='g')
plt.xlabel('Predicție')
plt.ylabel('Adevărat')
plt.show()

# Salvare model
model.save('saved_model_enet_b0')
