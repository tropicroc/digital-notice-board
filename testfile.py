import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
import torch
import whisper
import os
from speechbrain.pretrained import VAD

# Constants
SAMPLE_RATE = 44100  # 44.1 kHz
CHANNELS = 1  # Mono
SILENCE_THRESHOLD = 5  # Stop after 5s of silence
OUTPUT_FILENAME = "recorded_audio.wav"
TEMP_VAD_FILE = "temp_vad.wav"

# Load SpeechBrain VAD model
vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")

# Load Whisper model
whisper_model = whisper.load_model("base")

def record_audio():
    """Records audio when speech is detected and stops after 5s of silence."""
    print("Listening for speech...")
    audio_buffer = []
    silent_time = 0
    recording = False

    while True:
        # Capture 1 second of audio
        audio_chunk = sd.rec(int(SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16)
        sd.wait()

        # Save temporary file for VAD detection
        wav.write(TEMP_VAD_FILE, SAMPLE_RATE, audio_chunk)

        # Check if speech is detected (using the saved file)
        speech_prob = vad.get_speech_prob_file(TEMP_VAD_FILE)

        if speech_prob > 0.5:  # Speech detected
            if not recording:
                print("Speech detected, recording started...")
                recording = True
            audio_buffer.append(audio_chunk)
            silent_time = 0  # Reset silence timer
        else:
            if recording:
                silent_time += 1  # Increment silent seconds
                if silent_time >= SILENCE_THRESHOLD:
                    print("No speech detected for 5s, stopping recording...")
                    break

    # Cleanup temp file
    if os.path.exists(TEMP_VAD_FILE):
        os.remove(TEMP_VAD_FILE)

    if not audio_buffer:
        print("No speech detected, exiting.")
        return None

    # Save recorded audio
    recorded_audio = np.concatenate(audio_buffer, axis=0)
    wav.write(OUTPUT_FILENAME, SAMPLE_RATE, recorded_audio)
    print(f"Audio saved as {OUTPUT_FILENAME}")

    return OUTPUT_FILENAME

def transcribe_audio(filename):
    """Transcribes recorded audio using Whisper."""
    print("Transcribing audio...")
    result = whisper_model.transcribe(filename)
    print("Transcription:", result["text"])
    return result["text"]

# Run the recording and transcription
audio_file = record_audio()
if audio_file:
    transcribe_audio(audio_file)
