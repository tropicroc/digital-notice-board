import sounddevice as sd
import numpy as np
import threading
import time
from scipy.io.wavfile import write
import whisper
from vad import EnergyVAD

# 🎙️ Step 1: Auto-detect valid mic device and sample rate
devices = sd.query_devices()
input_devices = [i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]

mic_device_index = None
sample_rate = 16000  # Preferred sample rate

for index in input_devices:
    try:
        sd.check_input_settings(device=index, samplerate=sample_rate)
        mic_device_index = index
        print(f"✅ Using device: {devices[index]['name']} (index {index}) with {sample_rate} Hz")
        break
    except Exception as e:
        print(f"⚠️ Device {devices[index]['name']} (index {index}) doesn't support {sample_rate} Hz: {e}")

if mic_device_index is None:
    mic_device_index = sd.default.device[0]
    sample_rate = int(devices[mic_device_index]['default_samplerate'])
    print(f"🔄 Falling back to device: {devices[mic_device_index]['name']} (index {mic_device_index}) at {sample_rate} Hz")

# Parameters
gain = 2.0
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
first_chunk_with_speech = None  # Store the first 5s audio with speech

# 🔍 VAD Monitor Thread
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

# Start VAD monitor thread
vad_thread = threading.Thread(target=vad_monitor)
vad_thread.start()

# Wait for speech to be detected
while not speech_detected:
    time.sleep(0.1)

# 🎧 Start continuous recording
recorded_audio = [first_chunk_with_speech]  # Include the first 5s with speech
stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', device=mic_device_index)
stream.start()
print("🎧 Recording...")

while not stop_recording:
    audio_chunk, _ = stream.read(int(0.5 * sample_rate))  # Record in 0.5s chunks
    audio_chunk *= gain
    audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
    recorded_audio.append(audio_chunk)

stream.stop()
stream.close()
print("✅ Recording stopped.")

# 💾 Merge and save final audio
final_audio = np.concatenate(recorded_audio, axis=0)
write(merged_filename, sample_rate, final_audio.astype(np.float32))
print(f"💾 Final audio saved as {merged_filename}")

# 📝 Transcribe with Whisper
model = whisper.load_model("tiny")
result = model.transcribe(merged_filename)
print("📝 Transcription:")
print(result['text'])
