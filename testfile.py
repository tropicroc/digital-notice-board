import sounddevice as sd
import numpy as np
import threading
import time
from scipy.io.wavfile import write
import whisper
from vad import EnergyVAD

# =============================
# Configuration Parameters
# =============================
sample_rate = 16000
chunk_duration = 5  # seconds
gain = 2.0
merged_filename = 'final_clean_output.wav'

# =============================
# Detect Input Device Index
# =============================
def get_input_device_index():
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"Input Device Found - Index {idx}: {device['name']}")
    try:
        default = sd.default.device[0]  # Default input device index
        print(f"Using default input device index: {default}")
        return default
    except Exception as e:
        print("Error detecting input device. Please specify manually.")
        raise e

mic_device_index = get_input_device_index()

# =============================
# Initialize VAD
# =============================
vad = EnergyVAD(
    sample_rate=sample_rate,
    frame_length=25,
    frame_shift=20,
    energy_threshold=0.05,
    pre_emphasis=0.95
)

speech_detected = False
stop_recording = False
first_chunk_with_speech = None

# =============================
# VAD Monitor Thread
# =============================
def vad_monitor():
    global speech_detected, stop_recording, first_chunk_with_speech
    print("🔍 Waiting for speech to start recording...")

    while not speech_detected:
        audio = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate,
                      channels=1, dtype='float32', device=mic_device_index)
        sd.wait()
        audio *= gain
        audio = np.clip(audio, -1.0, 1.0)
        if np.any(vad(audio.flatten())):
            speech_detected = True
            first_chunk_with_speech = audio
            print("✅ Speech detected! Starting actual recording...")
        else:
            print("⏳ No speech yet, checking again...")

    # Monitor for silence
    while not stop_recording:
        audio = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate,
                      channels=1, dtype='float32', device=mic_device_index)
        sd.wait()
        audio *= gain
        audio = np.clip(audio, -1.0, 1.0)
        if not np.any(vad(audio.flatten())):
            print("🛑 No speech detected for 5 seconds. Stopping recording...")
            stop_recording = True
        else:
            print("🎙️ Still hearing speech... continuing.")

# Start VAD thread
vad_thread = threading.Thread(target=vad_monitor)
vad_thread.start()

# Wait until speech is detected
while not speech_detected:
    time.sleep(0.1)

# =============================
# Recording Actual Audio
# =============================
recorded_audio = [first_chunk_with_speech]
stream = sd.InputStream(samplerate=sample_rate, channels=1,
                        dtype='float32', device=mic_device_index)
stream.start()
print("🎧 Recording started...")

while not stop_recording:
    audio_chunk, _ = stream.read(int(0.5 * sample_rate))  # 0.5 sec chunks
    audio_chunk *= gain
    audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
    recorded_audio.append(audio_chunk)

stream.stop()
stream.close()
print("✅ Recording stopped.")

# =============================
# Save Final Audio
# =============================
final_audio = np.concatenate(recorded_audio, axis=0)
write(merged_filename, sample_rate, final_audio.astype(np.float32))
print(f"💾 Final audio saved as {merged_filename}")

# =============================
# Transcribe with Whisper
# =============================
model = whisper.load_model("tiny")
result = model.transcribe(merged_filename)
print("📝 Transcription:")
print(result['text'])
