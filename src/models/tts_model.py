import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import sigmoid
from src.models.positional_encoding import ScaledPositionalEncoding
from src.models.prenets import EncoderPrenet, DecoderPrenet
from src.models.mini_transformer import MiniTransformer
from src.models.postnet import Postnet
from src.utils.get_unicode import phonemes_to_id
from src.datasets.tts_dataloader import create_dataloader
import numpy as np
import pandas as pd
from torchsummary import summary

class TTS_model(nn.Module):
    def __init__(self, d_model=256, max_n_phonemes=150, max_n_frames=650, vocab_size=80, n_mels=80, n_head=8, n_encoder=1, n_decoder=1, dim_feedforward=1024, stop_threshold=0.5):
        super().__init__()
        self.d_model = d_model
        self.n_mels = n_mels
        self.max_n_frames = max_n_frames
        self.stop_threshold = stop_threshold

        self.encoder_pe = ScaledPositionalEncoding(d_model=d_model, max_seq_len=max_n_phonemes)
        self.decoder_pe = ScaledPositionalEncoding(d_model=d_model, max_seq_len=max_n_frames)

        self.encoder_prenet = EncoderPrenet(num_embeddings=vocab_size, d_model=d_model, padding_idx=0)
        self.decoder_prenet = DecoderPrenet(n_mels=n_mels, d_model=d_model)

        self.transformer = MiniTransformer(d_model=d_model, nhead=n_head, n_encoder=n_encoder, n_decoder=n_decoder, dim_feedforward=dim_feedforward)
        
        self.postnet = Postnet(d_model=d_model, n_mels=n_mels)


    def forward(self, src, target, src_key_padding_mask=None, tgt_key_padding_mask=None):
        phonemes_encoded = self.encoder_pe(self.encoder_prenet(src), padding_mask=~src_key_padding_mask.unsqueeze(-1)) 

        batch = target.size(0)
        mel_start = torch.concat((torch.zeros((batch, 1, self.n_mels)), target[:, :-1, :]), dim=1)
        mel_encoded = self.decoder_pe(self.decoder_prenet(mel_start), padding_mask=~tgt_key_padding_mask.unsqueeze(-1))

        generated = self.transformer(phonemes_encoded, mel_encoded, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        mel, stop = self.postnet(generated, padding_mask=~tgt_key_padding_mask.unsqueeze(-1))

        return mel, stop
    
    def infer(self, src):
        phonemes_encoded = self.encoder_pe(self.encoder_prenet(src))
        mel = torch.zeros((1, 1, self.n_mels))

        for _ in range(self.max_n_frames):
            mel_encoded = self.decoder_pe(self.decoder_prenet(mel))
            logits = self.transformer(phonemes_encoded, mel_encoded)
            mel_next, stop_vec = self.postnet(logits)

            next = mel_next[:, -1:, :]
            mel = torch.concat((mel, next), dim=1)

            stop_token = stop_vec[:, -1, 0]
            stop_prob = sigmoid(stop_token)
            if stop_prob > self.stop_threshold:
                break

        return mel


if __name__ == "__main__":
    torch.manual_seed(42)
    tts_model = TTS_model()
    dataLoader = create_dataloader()

    for i, (phonemes_padded, phonemes_mask, mels_padded, mels_mask) in enumerate(dataLoader):
        if i > 0: break

        mel, stop = tts_model(phonemes_padded, mels_padded / 100, src_key_padding_mask=phonemes_mask, tgt_key_padding_mask=mels_mask)
        print(mel)
    #print(summary(tts_model, input_size=[(1, 113), (1, 622, 80)]))

    
            