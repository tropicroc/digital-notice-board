import sounddevice as sd
import numpy as np
import torch
import whisper
from speechbrain.pretrained import VAD
import time
import scipy.io.wavefile as wavefile
import sounddevice as sd

SAMPLE_RATE = 44100

print("Recording....")

recoring = sd.rec((int(5*SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()

print("Recording finished. Saving...")
wav.write("output.wav", SAMPLE_RATE, recording)
print("Audio saved as output.wav")
