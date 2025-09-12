import torch
import torch.nn as nn
from src.utils.get_unicode import phonemes_to_id
import numpy as np

class EncoderPrenet(nn.Module):
    def __init__(self, num_embeddings=80, d_model=256, padding_idx=None):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_model, padding_idx=padding_idx)

        ## 3 blocks of Conv1d -> batch normalisation -> ReLU -> Dropout
        self.prenet = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=d_model),
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=d_model),
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=d_model),
            nn.ReLU(),
            nn.Dropout()
        )

        ## Linear projection for 0 centered output
        self.projection = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, x, padding_mask=None):
        x = self.embedding(x)
        x = x.transpose(1,2)
        x = self.prenet(x)
        x = x.transpose(1,2)
        x = self.projection(x)
        if padding_mask is not None:
            x = x * padding_mask
        return x

class DecoderPrenet(nn.Module):
    def __init__(self, n_mels=80, hidden_units=256, d_model=256):
        super().__init__()

        self.prenet = nn.Sequential(
            nn.Linear(in_features=n_mels, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU()
        )

        ## Linear projection to obtain same dimension as triangle positional embeddings
        self.projection = nn.Linear(in_features=hidden_units, out_features=d_model)

    def forward(self, x):
        x = self.prenet(x)
        x = self.projection(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    encoder_prenet = EncoderPrenet()
    ids = torch.tensor(phonemes_to_id(["sos", "V", "eos", "F"])).unsqueeze(0)

    decoder_prenet = DecoderPrenet()
    mel = torch.tensor(np.load("data/processed/mels/LJ001-0001.npy")).unsqueeze(0).transpose(1,2)

    encoder_output = encoder_prenet(ids)
    decoder_output = decoder_prenet(mel)
    print(encoder_output.shape, decoder_output.shape)

