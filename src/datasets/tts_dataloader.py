from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
from src.utils.get_unicode import phonemes_to_id
from src.datasets.tts_dataset import TTS_dataset
import numpy as np

### Collate function to make sure phonemes and mels are all padded
def collate_fn(batch, pad=0):
    phonemes = []
    mels = []

    for src, mel in batch:
        phonemes_list = np.array(src.split(" "))
        ids = torch.tensor(phonemes_to_id(phonemes_list))

        phonemes.append(ids)
        mels.append(mel)

    phonemes_padded = pad_sequence(phonemes, padding_value=pad, batch_first=True)
    phonemes_mask = phonemes_padded == pad

    mels_padded = pad_sequence(mels, padding_value=0, batch_first=True)
    mels_mask = mels_padded.abs().sum(dim=-1) == 0

    return phonemes_padded, phonemes_mask, mels_padded, mels_mask

def create_dataloader(batch_size=32, shuffle=False):
    dataset = TTS_dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader

if __name__ == "__main__":
    dataLoader = create_dataloader(batch_size=2)
    for i, (phonemes_padded, phonemes_mask, mels_padded, mels_mask) in enumerate(dataLoader):
        if i > 1: break
        print(phonemes_padded.shape, phonemes_mask.shape, mels_padded.shape, mels_mask.shape)
        print(mels_padded)
        print(mels_mask)
