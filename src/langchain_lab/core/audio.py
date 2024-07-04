import os

from openai import OpenAI


def openai_speech_to_text(audio_file_path):
    client = OpenAI(base_url=os.environ["OPENAI_API_BASE"], api_key=os.environ["OPENAI_API_KEY"])
    audio_file = open(audio_file_path, "rb")
    transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    print(transcription.text)
