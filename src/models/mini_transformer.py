import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

class MiniTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, n_encoder=1, n_decoder=1, dim_feedforward=1024):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True) 
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_encoder)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_decoder)

    def forward(self, src, target):
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(target, encoder_output)

        return decoder_output
    
    def infer(self, src, max_len=800):
        self.eval()

        encoder_output = self.encoder(src)

        mel = torch.tensor([[0,0,0,0]], dtype=torch.float32).unsqueeze(0)

        for _ in range(max_len):
            decoder_output = self.decoder(mel, encoder_output)
            next = decoder_output[:, -1, :].unsqueeze(0)

            mel = torch.concat((mel, next), dim=1)
        
        return mel


    
if __name__ == "__main__":
    torch.manual_seed(42)
    transformer = MiniTransformer(d_model=4, dim_feedforward=8, nhead=2)

    input = torch.tensor([[[1,2,3,4],[5,6,7,8]]], dtype=torch.float32)
    target = torch.tensor([[[1,2,3,4],[5,6,7,8]]], dtype=torch.float32)
    output = transformer.infer(input, max_len=2)
    print(output)
    
    print(summary(transformer, [(2,4), (2,4)]))