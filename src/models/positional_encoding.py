import torch
import torch.nn as nn


# Sinusoidal positional emcoding
class ScaledPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_num_frames):
        super().__init__()

        pe = torch.zeros(max_num_frames, d_model)
        position = torch.arange(0, max_num_frames, dtype=torch.float32).unsqueeze(1)

        exp_term = torch.reciprocal(
            torch.tensor(10000) ** torch.div(torch.arange(0, d_model, 2), d_model)
        ).unsqueeze(0)

        pe[:, 0::2] = torch.sin(torch.matmul(position, exp_term))
        pe[:, 1::2] = torch.cos(torch.matmul(position, exp_term))

        self.register_buffer("pe", pe, persistent=False)

        self.alpha = nn.Parameter(
            torch.tensor([1.0])
        )  # Adapt to different scales betweem encoder input and decoder output

    def forward(self, x):
        return x + self.alpha * self.pe[: x.size(1)]
