import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

class MiniTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, n_encoder=1, n_decoder=1, dim_feedforward=1024, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.device = device

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True) 
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_encoder)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_decoder)

    def forward(self, src, target, src_key_padding_mask=None, tgt_key_padding_mask=None):
        encoder_output = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        look_ahead_mask = torch.triu(torch.ones((target.size(1), target.size(1))), diagonal=1).bool().to(self.device)
        decoder_output = self.decoder(target, encoder_output, tgt_mask=look_ahead_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)

        if tgt_key_padding_mask is not None:
            decoder_output = decoder_output * ~tgt_key_padding_mask.unsqueeze(-1)

        return decoder_output


    
if __name__ == "__main__":
    torch.manual_seed(42)
    transformer = MiniTransformer(d_model=4, dim_feedforward=8, nhead=2)
    
    print(summary(transformer, [(2,4), (2,4)]))