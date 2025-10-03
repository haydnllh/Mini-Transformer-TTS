from src.models.tts_model import TTS_model
from src.train.train_model import train_model
from src.datasets.tts_dataloader import create_dataloader
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import torch

if __name__ == "__main__":
    torch.manual_seed(111)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts_model = TTS_model(device=device)
    tts_model.train()
    tts_model.load_state_dict(torch.load("trained_models/tts_model.pth", weights_only=True))
    dataloader = create_dataloader()

    mels_loss = MSELoss(reduction="sum")
    stop_loss = BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]).to(device), reduction="sum")

    optimiser = Adam(tts_model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = lr_scheduler.StepLR(optimiser, step_size=1000, gamma=0.9)

    tts_model.to(device)

    epochs = 5
    losses = train_model(model=tts_model, mel_loss_fn=mels_loss, stop_loss_fn=stop_loss, optimiser=optimiser, scheduler=scheduler, data_loader=dataloader, device=device, epochs=epochs)

    torch.save(tts_model.state_dict(), "trained_models/tts_model.pth")
    torch.save(torch.tensor(losses), ("trained_models/losses.pth"))