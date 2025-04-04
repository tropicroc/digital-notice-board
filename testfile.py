import sounddevice as sd
import numpy as np
import torch
import whisper
from speechbrain.pretrained import VAD
import time
import wave

# Load SpeechBrain VAD model
vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")

# Load Whisper model
model = whisper.load_model("tiny")

# Audio settings
SAMPLE_RATE = 44100
CHANNELS = 1
BUFFER_DURATION = 1  # Process audio every 1 second
SILENCE_THRESHOLD = 5  # Stop recording after 5 seconds of silence
OUTPUT_FILE = "live_audio.wav"

# Audio buffer
audio_buffer = []
last_speech_time = time.time()

def callback(indata, frames, time_info, status):
    """Callback function for audio stream"""
    global last_speech_time, audio_buffer

    if status:
        print(status, flush=True)

    # Convert audio to tensor for SpeechBrain VAD
    audio_tensor = torch.tensor(indata[:, 0], dtype=torch.float32)

    # Get speech probability
    speech_prob = vad.get_speech_probabilities(audio_tensor.unsqueeze(0))

    if max(speech_prob) > 0.5:  # Speech detected
        last_speech_time = time.time()
        print("🎤 Speech detected! Recording...")

        # Store audio data
        audio_buffer.append(indata.copy())

    elif time.time() - last_speech_time > SILENCE_THRESHOLD:
        print("⏳ No speech detected for 5 seconds. Stopping...")
        sd.stop()
        process_audio()
        exit()

def process_audio():
    """Save audio and transcribe using Whisper"""
    if not audio_buffer:
        print("No speech detected. Exiting...")
        return

    # Convert buffer to NumPy array
    audio_data = np.concatenate(audio_buffer, axis=0)

    # Save audio as WAV file
    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    print("🔍 Transcribing...")
    result = model.transcribe(OUTPUT_FILE)
    print("\n📝 Transcription:\n", result["text"])

# Start recording
print("🎙️ Listening...")
with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
    while True:
        time.sleep(0.1)  # Keep script running
