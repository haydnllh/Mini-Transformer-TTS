import gradio as gr
import torch
from pathlib import Path
import soundfile as sf
from src.inference.infer import infer
from src.models.tts_model import TTS_model
import requests
from mediafiredl import MediafireDL as MF
import os
import nltk

model_url = "https://www.mediafire.com/file/k9bpk4qmkm6c7j0/refined_tts_model_weights.pth/file"

model_path = "./refined_tts_model_weights.pth"

nltk.download('averaged_perceptron_tagger_eng')

if not os.path.exists(model_path):
    file = MF.Download(model_url, output=".")

device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = TTS_model(device=device)
tts_model.load_state_dict(torch.load("refined_tts_model_weights.pth", weights_only=True, map_location="cpu"))

def demo(text):
    audio = infer(text, tts_model, device=device, display=False, play=False)
    sf.write("audio.wav", audio, samplerate=22050)
    return "audio.wav"

demo = gr.Interface(fn=demo, inputs="text", outputs=gr.Audio())
demo.launch()