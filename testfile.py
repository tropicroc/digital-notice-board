import sounddevice as sd
import numpy as np
import threading
import time
from scipy.io.wavfile import write

import whisper
from vad import EnergyVAD

# Parameters
sample_rate = 44100
chunk_duration = 5  # seconds
merged_filename = 'final_clean_output.wav'

# Initialize VAD
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
first_chunk_with_speech = None  # This will store the first detected speech chunk

# VAD Monitor Thread
def vad_monitor():
    global speech_detected, stop_recording, first_chunk_with_speech
    print("🔍 Waiting for speech to start recording...")

    # Wait until speech is first detected
    while not speech_detected:
        audio = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = np.clip(audio, -1.0, 1.0)
        if np.any(vad(audio.flatten())):
            speech_detected = True
            first_chunk_with_speech = audio  # Save the first chunk with speech
            print("✅ Speech detected! Starting actual recording...")
        else:
            print("⏳ No speech yet, checking again...")

    # Keep monitoring for silence while recording
    while not stop_recording:
        audio = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = np.clip(audio, -1.0, 1.0)
        if not np.any(vad(audio.flatten())):
            print("🛑 No speech detected for 5 seconds. Stopping recording...")
            stop_recording = True
        else:
            print("🎙️ Still hearing speech... continuing.")

# Start VAD monitor thread
vad_thread = threading.Thread(target=vad_monitor)
vad_thread.start()

# Wait for speech to be detected
while not speech_detected:
    time.sleep(0.1)

# Start continuous recording
recorded_audio = [first_chunk_with_speech]  # Include the first 5-sec detected chunk
stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32')
stream.start()
print("🎧 Recording...")

while not stop_recording:
    audio_chunk, _ = stream.read(int(0.5 * sample_rate))  # 0.5 sec chunks
    audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
    recorded_audio.append(audio_chunk)

stream.stop()
stream.close()
print("✅ Recording stopped.")

# Merge and save final audio
final_audio = np.concatenate(recorded_audio, axis=0)
write(merged_filename, sample_rate, final_audio.astype(np.float32))
print(f"💾 Final audio saved as {merged_filename}")

# Transcribe with Whisper
model = whisper.load_model("tiny")
result = model.transcribe(merged_filename)
print("📝 Transcription:")
print(result['text'])