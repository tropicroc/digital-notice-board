import sounddevice as sd
import numpy as np
import threading
import time
from scipy.io.wavfile import write
import whisper
from vad import EnergyVAD

# Parameters
rate = 44100
dur = 5  # seconds
totranscribe = 'transcribex.wav'

vad = EnergyVAD(
    rate=rate,
    frame_length=25,
    frame_shift=20,
    energy_threshold=0.05,
    pre_emphasis=0.95
)

speech_detected = False
stop_recording = False
first_chunk_with_speech = None  # This will store the first detected speech chunk

def vad_monitor():
    global speech_detected, stop_recording, first_chunk_with_speech
    print("Waiting for speech to start recording...")

    while not speech_detected:
        audio = sd.rec(int(dur * rate), samplerate=rate, channels=1, dtype='float32')
        sd.wait()
        audio = np.clip(audio, -1.0, 1.0)
        if np.any(vad(audio.flatten())):
            speech_detected = True
            first_chunk_with_speech = audio  # Save the first chunk with speech
            print("Speech detected! Starting actual recording...")
        else:
            print("No speech yet, checking again...")

    while not stop_recording:
        audio = sd.rec(int(dur * rate), samplerate=rate, channels=1, dtype='float32')
        sd.wait()
        audio = np.clip(audio, -1.0, 1.0)
        if not np.any(vad(audio.flatten())):
            print("No speech detected for 5 seconds. Stopping recording...")
            stop_recording = True
        else:
            print("Still hearing speech... continuing.")

vad_thread = threading.Thread(target=vad_monitor)
vad_thread.start()

while not speech_detected:
    time.sleep(0.1)

recorded_audio = [first_chunk_with_speech]  # Include the first 5-sec detected chunk
stream = sd.InputStream(samplerate=rate, channels=1, dtype='float32')
stream.start()
print("🎧 Recording...")

while not stop_recording:
    audio_chunk, _ = stream.read(int(0.5 * rate))  # 0.5 sec chunks
    audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
    recorded_audio.append(audio_chunk)

stream.stop()
stream.close()
print("✅ Recording stopped.")

final_audio = np.concatenate(recorded_audio, axis=0)
write(totranscribe, rate, final_audio.astype(np.float32))
print(f"💾 Final audio saved as {totranscribe}")

model = whisper.load_model("tiny")
result = model.transcribe(totranscribe)
print("Transcription:")
print(result['text'])