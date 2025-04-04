import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import torchaudio
import torch
import os
import whisper
from speechbrain.pretrained import VAD

# Constants
SAMPLE_RATE = 44100  # Original recording sample rate
TARGET_SAMPLE_RATE = 16000  # SpeechBrain VAD expected sample rate
CHANNELS = 1  # Mono
SILENCE_THRESHOLD = 5  # Stop after 5s of silence
OUTPUT_FILENAME = "recorded_audio.wav"
TEMP_VAD_FILE = "temp_vad.wav"

# Load SpeechBrain VAD model
vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")

# Load Whisper model
whisper_model = whisper.load_model("base")

def resample_audio(audio, orig_sr, target_sr):
    """Resample audio to the target sample rate using torchaudio."""
    transform = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    audio_tensor = torch.from_numpy(audio.astype(np.float32) / 32768.0)  # Normalize
    resampled_audio = transform(audio_tensor)
    return (resampled_audio.numpy() * 32768.0).astype(np.int16)  # Convert back to int16

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

        # Resample to 16kHz for SpeechBrain VAD
        resampled_chunk = resample_audio(audio_chunk, SAMPLE_RATE, TARGET_SAMPLE_RATE)

        # Save temporary file for VAD detection
        wav.write(TEMP_VAD_FILE, TARGET_SAMPLE_RATE, resampled_chunk)

        # Check if speech is detected (using the resampled file)
        speech_prob = vad.get_speech_prob_file(TEMP_VAD_FILE)

        if speech_prob.mean() > 0.5:  # Fix: Extract mean probability
            if not recording:
                print("Speech detected, recording started...")
                recording = True
            audio_buffer.append(audio_chunk)  # Store original 44.1kHz audio
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

    # Save recorded audio (original 44.1kHz for Whisper)
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
