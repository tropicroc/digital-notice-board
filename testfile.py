import sounddevice as sd # for input and output purposes
import numpy as np # an array library in python
import threading # used for processing two input audio data simultaneously
import time # used to cause delays in the execution
from scipy.io.wavfile import write # writing the recorded audio to an external .wav file
import whisper # openai's whisper model, used for transcribed the obtained audio
from vad import EnergyVAD # used to detect speech in a audio file
sample_rate = 44100
chunk_duration = 5 
merged_filename = 'final_output.wav'
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
def vad_monitor():
    global speech_detected, stop_recording, first_chunk_with_speech
    print("Waiting for speech to start recording.")
    while not speech_detected:
        audio = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = np.clip(audio, -1.0, 1.0)
        if np.any(vad(audio.flatten())):
            speech_detected = True
            first_chunk_with_speech = audio
            print("Speech is detected!")
        else:
            print("No speech yet, checking again.")
    while not stop_recording:
        audio = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = np.clip(audio, -1.0, 1.0)
        if not np.any(vad(audio.flatten())):
            print("No speech ia detected for 5 seconds. Stopping the recording.")
            stop_recording = True
        else:
            print("Still hearing speech... continuing.")
vad_thread = threading.Thread(target=vad_monitor)
vad_thread.start()
while not speech_detected:
    time.sleep(0.1)
recorded_audio = [first_chunk_with_speech]
stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32')
stream.start()
print("Recording...")
while not stop_recording:
    audio_chunk, _ = stream.read(int(0.5 * sample_rate))
    audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
    recorded_audio.append(audio_chunk)
stream.stop()
stream.close()
print("Recording stopped.")
final_audio = np.concatenate(recorded_audio, axis=0)
write(merged_filename, sample_rate, final_audio.astype(np.float32))
print(f"Final audio saved as {merged_filename}")
model = whisper.load_model("tiny")
result = model.transcribe(merged_filename)
print("Transcription:")
print(result['text'])