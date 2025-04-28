import sys
import tensorflow as tf

# Încarcă modelul salvat
imported_model = tf.saved_model.load(r"saved")

# Listează clasele
class_names = ['Do#0', 'Do#1', 'Do#2', 'Do#3', 'Do#4', 'Do#5', 'Do#6', 'Do#7', 'Do#8',
               'Do0', 'Do1', 'Do2', 'Do3', 'Do4', 'Do5', 'Do6', 'Do7', 'Do8',
               'Fa#0', 'Fa#1', 'Fa#2', 'Fa#3', 'Fa#4', 'Fa#5', 'Fa#6', 'Fa#7',
               'Fa0', 'Fa1', 'Fa2', 'Fa3', 'Fa4', 'Fa5', 'Fa6', 'Fa7',
               'La#-1', 'La#0', 'La#1', 'La#2', 'La#3', 'La#4', 'La#5', 'La#6',
               'La#7', 'La-1', 'La0', 'La1', 'La2', 'La3', 'La4', 'La5', 'La6', 'La7',
               'Mi0', 'Mi1', 'Mi2', 'Mi3', 'Mi4', 'Mi5', 'Mi6', 'Mi7',
               'Re#0', 'Re#1', 'Re#2', 'Re#3', 'Re#4', 'Re#5', 'Re#6', 'Re#7',
               'Re0', 'Re1', 'Re2', 'Re3', 'Re4', 'Re5', 'Re6', 'Re7', 'Re8',
               'Si-1', 'Si0', 'Si1', 'Si2', 'Si3', 'Si4', 'Si5', 'Si6', 'Si7',
               'Sol#0', 'Sol#1', 'Sol#2', 'Sol#3', 'Sol#4', 'Sol#5', 'Sol#6', 'Sol#7',
               'Sol0', 'Sol1', 'Sol2', 'Sol3', 'Sol4', 'Sol5', 'Sol6', 'Sol7']

# Încarcă și decodează WAV în waveform 1-D
def preprocess_audio(file_path):
    audio = tf.io.read_file(file_path)
    audio_tensor, _ = tf.audio.decode_wav(
        audio,
        desired_channels=1,
        desired_samples=-1   # ia toate sample-urile
    )
    waveform = tf.squeeze(audio_tensor, axis=-1)
    return waveform  # shape (n_samples,)

# Suprapunere sau segmentare fixă
SAMPLE_RATE = 64000
WINDOW_SIZE = SAMPLE_RATE  # 1 sec
HOP_SIZE    = SAMPLE_RATE  # pentru ferestre fără suprapunere
# HOP_SIZE = SAMPLE_RATE // 2  # pentru 50% overlap

def split_into_windows(waveform):
    n = tf.shape(waveform)[0]
    #in caz ca avem audio < 4 sec
    if n < WINDOW_SIZE:
        # adaugam zero padding
       padding = WINDOW_SIZE - n
       waveform = tf.pad(waveform, [[0, padding]], mode='CONSTANT')
       n = WINDOW_SIZE

    # reduc la multiplu de WINDOW_SIZE
    n_windows = n // WINDOW_SIZE
    trimmed = waveform[:n_windows * WINDOW_SIZE]
    # reshape în (n_windows, WINDOW_SIZE)
    windows = tf.reshape(trimmed, (n_windows, WINDOW_SIZE))
    return windows  # Tensor shape (n_windows, 64000)

def get_spectrogram(waveform_1d):
    # input: (64000,) sau (WINDOW_SIZE,)
    spectrogram = tf.signal.stft(
        waveform_1d,
        frame_length=255,
        frame_step=128
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram  # shape (time, freq, 1)

if __name__ == "__main__":
    wav_path = sys.argv[1]
    # 1) Încarcă și decodează tot wav-ul
    waveform = preprocess_audio(wav_path)

    # 2) Împarte în ferestre de câte 1 sec
    windows = split_into_windows(waveform)  # (n_windows, 64000)

    # 3) Rulează modelul pe fiecare fereastră și colectează probabilități
    all_probs = []
    for i in range(windows.shape[0]):
        seg = windows[i]  # shape (64000,)
        seg_batch = seg[tf.newaxis, ...]  # Add batch dimension: (1, 64000)
        out = imported_model(seg_batch)  # Pass raw waveform to the model
        probs = tf.nn.softmax(out['predictions'][0])  # (n_classes,)
        all_probs.append(probs)
    all_probs = tf.stack(all_probs, axis=0)  # (n_windows, n_classes)

    # 4) Agregare: media probabilităților pe toate ferestrele
    mean_probs = tf.reduce_mean(all_probs, axis=0)  # (n_classes,)
    pred_id = tf.argmax(mean_probs).numpy()
    # Check pred_id validity
    if pred_id < 0 or pred_id >= len(class_names):
        raise ValueError(f"Invalid prediction ID: {pred_id}. Check the model output and class_names.")

    predicted_class = class_names[pred_id]

    # 5) Afișează rezultatul
    print(predicted_class)

