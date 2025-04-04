import whisper

model = whisper.load_model("tiny")
result = model.transcribe("test-audio-lol.mp3")
print(result["text"])