import torch
from src.utils.get_unicode import phonemes_to_id
from src.utils.text_to_phonemes import text_to_phonemes
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np

def infer(text, model, losses, device="cpu"):
    phonemes = text_to_phonemes(text)
    print(f"Phonemes: {phonemes}")
    ids = phonemes_to_id(phonemes)
    model_input = torch.tensor(ids).unsqueeze(0).to(device)

    model.eval()
    model = model.to(device)

    with torch.inference_mode():
        generated = model.infer(model_input).squeeze() * 100
        generated_np = np.array(generated.squeeze().to("cpu")).T
        print(generated.shape)

    fig, ax = plt.subplots(1, 3, figsize=(12,5))

    mel_img = librosa.display.specshow(generated_np, sr=22050, hop_length=512, ax=ax[0])
    fig.colorbar(mel_img, ax=ax[0], format="%+2.0f dB")
    plt.title("Generated Mel Spectrogram")

    generated_spectrogram = librosa.feature.inverse.mel_to_stft(librosa.db_to_power(generated_np))
    generated_waveform = librosa.griffinlim(generated_spectrogram, hop_length=512, n_fft=2048) #An approximation from mel spectrograms

    librosa.display.waveshow(generated_waveform, ax=ax[1])
    plt.title("Generated Waveform")

    print(f"Final Loss: {losses[-1]}")
    plt.plot(losses)
    plt.title("Loss")

    sd.play(generated_waveform, 22050)

    plt.show()