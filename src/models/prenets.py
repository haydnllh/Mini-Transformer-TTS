import torch
import torch.nn as nn
from src.utils.get_unicode import phonemes_to_id

class EncoderPrenet(nn.Module):
    def __init__(self, num_embeddings=80, embedding_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        ## 3 blocks of Conv1d -> batch normalisation -> ReLU -> Dropout
        self.prenet = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=embedding_dim),
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=embedding_dim),
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=embedding_dim),
            nn.ReLU(),
            nn.Dropout()
        )

        ## Linear projection for 0 centered output
        self.projection = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1,2)
        x = self.prenet(x)
        x = x.transpose(1,2)
        x = self.projection(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    encoder_prenet = EncoderPrenet()
    ids = torch.tensor(phonemes_to_id(["sos", "V", "eos", "F"])).unsqueeze(0)

    output = encoder_prenet(ids)
    print(output.shape)
    print(output)

