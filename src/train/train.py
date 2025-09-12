from src.models.tts_model import TTS_model
from src.train.train_model import train_model
from src.datasets.tts_dataloader import create_dataloader
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.optim import Adam
import torch

if __name__ == "__main__":
    tts_model = TTS_model()
    dataloader = create_dataloader(shuffle=True)

    mels_loss = MSELoss()
    stop_loss = BCEWithLogitsLoss()

    optimiser = Adam(tts_model.parameters(), lr=1e-3, weight_decay=1e-6)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model.to(device)

    epochs = 1

    losses = train_model(model=tts_model, mel_loss_fn=mels_loss, stop_loss_fn=stop_loss, optimiser=optimiser, data_loader=dataloader, device=device, epochs=epochs, stop_weight=5)