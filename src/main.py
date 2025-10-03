import torch
from src.models.tts_model import TTS_model
from src.inference.infer import infer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts_model = TTS_model(device=device)
    tts_model.load_state_dict(torch.load("trained_models/refined_tts_model_weights.pth", weights_only=True))

    losses = torch.load("trained_models/losses.pth").squeeze().numpy()

    text = input("Enter your sentence: \n")

    infer(text, tts_model, losses=losses, device=device)