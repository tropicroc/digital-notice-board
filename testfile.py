import sounddevice as sd
import numpy as np
import threading
import time
from scipy.io.wavfile import write
import whisper
from vad import EnergyVAD

# 🧠 Detect default sample rate for current input device
default_input = sd.query_devices(kind='input')
sample_rate = int(default_input['default_samplerate'])  # Auto-adjust to Pi-supported rate
print(f"🎙️ Using sample rate: {sample_rate} Hz")

# Parameters
gain = 2.0
chunk_duration = 5  # seconds
merged_filename = 'final_clean_output.wav'

# Initialize VAD with dynamic sample rate
vad = EnergyVAD(
    sample_rate=sample_rate,
    frame_length=25,
    frame_shift=20,
    energy_threshold=0.05,
    pre_emphasis=0.95
)

# Shared flags and data
speech_detected = False
stop_recording = False
first_chunk_with_speech = None

# 🛎️ VAD Monitor Thread
def vad_monitor():
    global speech_detected, stop_recording, first_chunk_with_speech
    print("🔍 Waiting for speech to start recording...")

    # Wait for first speech
    while not speech_detected:
        audio = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio *= gain
        audio = np.clip(audio, -1.0, 1.0)
        if np.any(vad(audio.flatten())):
            speech_detected = True
            first_chunk_with_speech = audio
            print("✅ Speech detected! Starting actual recording...")
        else:
            print("⏳ No speech yet, checking again...")

    # Continue monitoring for silence
    while not stop_recording:
        audio = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio *= gain
        audio = np.clip(audio, -1.0, 1.0)
        if not np.any(vad(audio.flatten())):
            print("🛑 No speech detected for 5 seconds. Stopping recording...")
            stop_recording = True
        else:
            print("🎙️ Still hearing speech... continuing.")

# 🚀 Start VAD thread
vad_thread = threading.Thread(target=vad_monitor)
vad_thread.start()

# Wait until speech is detected
while not speech_detected:
    time.sleep(0.1)

# 🎧 Start continuous recording
recorded_audio = [first_chunk_with_speech]  # Include the initial 5s with speech
stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32')
stream.start()
print("🎤 Recording...")

while not stop_recording:
    audio_chunk, _ = stream.read(int(0.5 * sample_rate))  # Read 0.5s chunks
    audio_chunk *= gain
    audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
    recorded_audio.append(audio_chunk)

stream.stop()
stream.close()
print("✅ Recording stopped.")

# 💾 Save final audio
final_audio = np.concatenate(recorded_audio, axis=0)
write(merged_filename, sample_rate, final_audio.astype(np.float32))
print(f"📁 Final audio saved as {merged_filename}")

# 📝 Transcribe with Whisper
print("🔍 Transcribing...")
model = whisper.load_model("tiny")
result = model.transcribe(merged_filename)
print("📝 Transcription:")
print(result['text'])
