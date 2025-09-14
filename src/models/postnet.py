import torch
import torch.nn as nn
from torchsummary import summary

class Postnet(nn.Module):
    def __init__(self, d_model=256, n_mels=80, device="cpu"):
        super().__init__()

        self.mel_linear = nn.Linear(in_features=d_model, out_features=n_mels)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=n_mels, out_channels=d_model, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(in_channels=d_model, out_channels=n_mels, kernel_size=5, padding=2),
        )

        # Stop token, 1 if stop, 0 othewise
        self.stop = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model//2),
            nn.ReLU(),
            nn.Linear(in_features=d_model//2, out_features=d_model//4),
            nn.ReLU(),
            nn.Linear(in_features=d_model//4, out_features=1)
        )

    def forward(self, x, padding_mask=None):
        mel_linear = self.mel_linear(x).transpose(-2,-1)
        post_conv = self.conv_block(mel_linear)

        #Residual
        mel = (mel_linear + post_conv).transpose(-2,-1)
        if padding_mask is not None:
            mel = mel * padding_mask

        stop_vec = self.stop(x)

        return mel, stop_vec
    

if __name__ == "__main__":
    torch.manual_seed(42)
    postnet = Postnet(d_model=4, n_mels=2)

    input = torch.rand((1,2,4))
    mel, stop_vec = postnet(input)

    print(mel.shape, stop_vec.shape)
    print(summary(Postnet(), (10,256)))