import torch
from src.models.tts_model import TTS_model
import torch
from src.utils.get_unicode import phonemes_to_id
from src.datasets.tts_dataset import TTS_dataset
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TTS_model(n_encoder=6, n_decoder=6, d_model=512, dim_feedforward=2048, device=device).to(device)
    model.load_state_dict(torch.load("trained_models/6layers_tts_model.pth", weights_only=True))
    losses=torch.load("trained_models/6layers_losses.pth")
    print(losses)
    model.eval()

    with torch.inference_mode():
        dataset = TTS_dataset()
        ids = torch.tensor(phonemes_to_id(np.array(dataset[0][0].split(" ")))).to(device).unsqueeze(0)

        generated = (model.infer(ids).squeeze() * 100).to("cpu").numpy()
        librosa.display.specshow(dataset[0][1].numpy(), sr=16000, hop_length=256, y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        print(generated[:, 0])
        print(dataset[0][1].numpy())
        plt.show()