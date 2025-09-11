from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import numpy as np

class TTS_dataset(Dataset):
    def __init__(self):
        super().__init__()
        
        self.meta = pd.read_csv("data/processed/metadata.csv")
        self.mels_dir = "data/processed/mels"

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        filepath = self._get_path_from_index(index)
        mel = torch.tensor(np.load(filepath), dtype=torch.float32)

        src = self.meta.loc[index, "phonemes"]

        return src, mel


    def _get_path_from_index(self, index):
        filename = self.meta.loc[index, "file_id"] + ".npy"
        path = os.path.join(self.mels_dir, filename)
        return path
    
