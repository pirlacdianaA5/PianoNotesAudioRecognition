print("hello")
# evaluate_server.py
from fastapi import FastAPI, WebSocket
import uvicorn, tensorflow as tf, asyncio, io
import numpy as np
from pydub import AudioSegment
from starlette.websockets import WebSocketDisconnect
from numpy.lib.stride_tricks import sliding_window_view

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Server is running. Use WebSocket at /ws/predict"}

# Load the model once
model = tf.saved_model.load("saved")
class_names = ['no', 'yes', 'down', 'go', 'left', 'up', 'right', 'stop']


SAMPLE_RATE = 16000
WINDOW_SIZE = SAMPLE_RATE            # 1 s → 16 000 samples
HOP_SAMPLES = WINDOW_SIZE // 2       # 0.5 s → 8 000 samples
bytes_per_window = WINDOW_SIZE * 2   # 32 000 bytes
bytes_per_hop = HOP_SAMPLES * 2      # 16 000 bytes
min_buffer = (WINDOW_SIZE + HOP_SAMPLES) * 2  # 48 000 bytes


def split_into_windows(waveform: tf.Tensor):
    n = tf.shape(waveform)[0]
    n_windows = n // WINDOW_SIZE
    trimmed = waveform[:n_windows * WINDOW_SIZE]
    windows = tf.reshape(trimmed, (n_windows, WINDOW_SIZE))
    return windows  # (n_windows, 16000)

def get_spectrogram(waveform_1d: tf.Tensor):
    spec = tf.signal.stft(
        waveform_1d,
        frame_length=255,
        frame_step=128
    )
    spec = tf.abs(spec)[..., tf.newaxis]
    return spec  # (time, freq, 1)

def predict_from_waveform(waveform: tf.Tensor):
    # 1) Split into windows
    windows = split_into_windows(waveform)
    # 2) Run inference on each segment
    probs_list = []
    for i in range(windows.shape[0]):
        seg = windows[i]
        seg_batch = seg[tf.newaxis, ...]
        out = model(seg_batch)
        probs = tf.nn.softmax(out['predictions'][0])
        probs_list.append(probs)
    all_probs = tf.stack(probs_list, axis=0)
    mean_probs = tf.reduce_mean(all_probs, axis=0)
    pred_id = tf.argmax(mean_probs).numpy()
    return class_names[pred_id]

def create_wav_header(sample_rate, num_channels, bits_per_sample, num_samples):
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    header = b'RIFF'
    header += (36 + data_size).to_bytes(4, 'little')  # Chunk size
    header += b'WAVE'
    header += b'fmt ' + (16).to_bytes(4, 'little')  # Subchunk1 size
    header += (1).to_bytes(2, 'little')  # Audio format (1 = PCM)
    header += num_channels.to_bytes(2, 'little')
    header += sample_rate.to_bytes(4, 'little')
    header += byte_rate.to_bytes(4, 'little')
    header += block_align.to_bytes(2, 'little')
    header += bits_per_sample.to_bytes(2, 'little')
    header += b'data'
    header += data_size.to_bytes(4, 'little')  # Subchunk2 size

    return header

import os
import time


def preprocess_from_bytes(audio_bytes: bytes) -> tf.Tensor:
    """
    Procesare directă a PCM 16-bit mono la 16 kHz și salvare ca fișier .wav.
    Returnează un tf.Tensor normalizat în intervalul [-1, 1].
    """
    # Define the absolute path to the 'assets' folder
    assets_path = r"C:\Users\pirla\Desktop\Licenta2024\TensorWordsRecognition\words\assets"

    # Create the folder if it doesn't exist
    os.makedirs(assets_path, exist_ok=True)

    # Check if audio_bytes is valid
    if not audio_bytes or len(audio_bytes) < 2:
        print("Invalid or empty audio bytes!")
        return tf.zeros([WINDOW_SIZE], dtype=tf.float32)

    # Generate WAV header
    num_samples = len(audio_bytes) // 2  # Each sample is 2 bytes (16-bit PCM)
    wav_header = create_wav_header(SAMPLE_RATE, 1, 16, num_samples)

    # Save the audio file in the 'assets' folder
    timestamp = int(time.time() * 1000)  # Current time in milliseconds
    file_path = os.path.join(assets_path, f"audio_{timestamp}.wav")
    with open(file_path, "wb") as f:
        f.write(wav_header + audio_bytes)
    print(f"Audio saved: {file_path}")

    # Convert raw PCM to TensorFlow tensor
    waveform = tf.constant(np.frombuffer(audio_bytes, dtype=np.int16), tf.float32) / 32768.0

    # Pad to at least one window
    length = tf.shape(waveform)[0]
    print(f"Waveform length: {length}")
   # if length < WINDOW_SIZE:
   #    padding_amount = WINDOW_SIZE - length
   #   waveform = tf.pad(waveform, [[0, padding_amount]])
    return waveform


@app.websocket("/ws/predict")
async def ws_predict(ws: WebSocket):
    await ws.accept()
    pcm_buffer = bytearray()  #Un buffer pentru a stoca datele audio brute primite de la client.

    try:
        while True:
            # 1) Read raw PCM chunk
            try:
                raw = await ws.receive_bytes()
                print(f"Received audio bytes: {len(raw)} bytes")

                # Check if the length is a multiple of 2 (16-bit PCM)
                if len(raw) % 2 != 0:
                    print("Warning: Audio data length is not a multiple of 2 (16-bit PCM)")

                # Analyze the first few samples
                samples = np.frombuffer(raw[:16], dtype=np.int16)
                print(f"First samples: {samples}")
            except WebSocketDisconnect:
                print("Client disconnected, exiting loop")
                break

            pcm_buffer.extend(raw)

            # 2) Only proceed once we have at least WINDOW + HOP worth of data
            #if len(pcm_buffer) < min_buffer:
            #    continue

            print(f"pcm_buffer length: {len(pcm_buffer)} bytes")

            # 3) Ensure even length (int16 = 2 bytes/sample)
            if len(pcm_buffer) % 2 == 1:
                pcm_buffer.pop()  # drop last odd byte

            # 4) Convert to immutable bytes
            audio_bytes = bytes(pcm_buffer)

            # 5) Preprocess audio bytes into a normalized waveform tensor
            waveform = preprocess_from_bytes(audio_bytes)

            # 6) Sliding windows with hop
            windows = sliding_window_view(waveform.numpy(), WINDOW_SIZE)[::HOP_SAMPLES]

            # 7) Predict each window
            preds = []
            for w in windows:
                wf = tf.constant(w, tf.float32)  # Already normalized
                preds.append(predict_from_waveform(wf))

            # 8) Aggregate (majority vote)
            final = max(set(preds), key=preds.count)
            await ws.send_json({"predicted": final})
            print(f"Overlap prediction: {final}")

            # 9) Slide buffer forward by one hop (in bytes)
            del pcm_buffer[:bytes_per_hop]

    finally:
        print("Cleanup in finally (no explicit ws.close() needed)")

if __name__ == "__main__":
    print("Starting Python WebSocket server...")
    uvicorn.run("evaluate_server:app", host="127.0.0.1", port=8000)