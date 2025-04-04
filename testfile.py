import sounddevice as sd
import numpy as np
import threading
import time
from scipy.io.wavfile import write

import whisper
from vad import EnergyVAD
import os

# Parameters
sample_rate = 44100
chunk_duration = 5  # seconds
merged_filename = 'final_clean_output.wav'
transcript_file = 'transcript.txt'

# Initialize VAD
vad = EnergyVAD(
    sample_rate=sample_rate,
    frame_length=25,
    frame_shift=20,
    energy_threshold=0.05,
    pre_emphasis=0.95
)

# Load Whisper model once
model = whisper.load_model("tiny")

# Track count
if not os.path.exists(transcript_file):
    with open(transcript_file, 'w') as f:
        pass  # create empty file

def get_transcript_index():
    with open(transcript_file, 'r') as f:
        lines = f.readlines()
    return len(lines) + 1

def record_and_transcribe():
    global speech_detected, stop_recording, first_chunk_with_speech

    # Flags
    speech_detected = False
    stop_recording = False
    first_chunk_with_speech = None

    # VAD Monitor Thread
    def vad_monitor():
        nonlocal speech_detected, stop_recording, first_chunk_with_speech
        print("🔍 Waiting for speech to start recording...")

        while not speech_detected:
            audio = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            audio = np.clip(audio, -1.0, 1.0)
            if np.any(vad(audio.flatten())):
                speech_detected = True
                first_chunk_with_speech = audio
                print("✅ Speech detected! Starting actual recording...")
            else:
                print("⏳ No speech yet, checking again...")

        while not stop_recording:
            audio = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            audio = np.clip(audio, -1.0, 1.0)
            if not np.any(vad(audio.flatten())):
                print("🛑 No speech detected for 5 seconds. Stopping recording...")
                stop_recording = True
            else:
                print("🎙️ Still hearing speech... continuing.")

    # Start monitor
    vad_thread = threading.Thread(target=vad_monitor)
    vad_thread.start()

    while not speech_detected:
        time.sleep(0.1)

    # Recording chunks
    recorded_audio = [first_chunk_with_speech]
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32')
    stream.start()
    print("🎧 Recording...")

    while not stop_recording:
        audio_chunk, _ = stream.read(int(0.5 * sample_rate))
        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
        recorded_audio.append(audio_chunk)

    stream.stop()
    stream.close()
    print("✅ Recording stopped.")

    # Save and transcribe
    final_audio = np.concatenate(recorded_audio, axis=0)
    write(merged_filename, sample_rate, final_audio.astype(np.float32))
    print(f"💾 Final audio saved as {merged_filename}")

    result = model.transcribe(merged_filename)
    text = result['text'].strip()

    # Append to transcript file
    index = get_transcript_index()
    with open(transcript_file, 'a') as f:
        f.write(f"{index}. {text}\n")

    print("📝 Transcription saved.")
    print("⌛ Waiting for next input...\n")

# Main loop
try:
    while True:
        record_and_transcribe()
except KeyboardInterrupt:
    print("👋 Exiting...")
