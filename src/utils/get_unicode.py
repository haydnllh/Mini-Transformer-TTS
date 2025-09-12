import torch
import numpy as np

vocab = sorted(set({'B', 'V', 'F', '?', 'ZH', 'ER0', 'NG', 'AA1', '..', 'CH', 'EH0', "'", 'UW2', 'P', 'EH2', 'OW0', 'S', 'AH0', ',', 'UH2', '!', 'EH1', 'R', 'AY2', 'OY1', 'N', 'SH', 'L', 'AH2', 'IY1', 'EY1', 'UW0', 'HH', 'AE0', 'G', 'AE2', 'UW1', 'UH0', 'AY0', 'AY1', 'Z', 'sos', 'M', 'AA2', 'IH2', 'IY2', 'IY0', 'AW0', 'OY0', 'OW2', 'AO1', 'eos', 'OW1', 'JH', 'AE1', 'D', 'ER2', 'IH1', 'AA0', 'EY2', '-', 'AO0', '.', 'AH1', 'UH1', 'T', 'AW2', 'AW1', 'W', 'EY0', 'IH0', 'Y', 'OY2', 'ER1', 'AO2', 'TH', 'K', 'DH'}), reverse=True)

vocab_id = {v: (i + 1) for i, v in enumerate(vocab)}

def phonemes_to_id(phonemes: np.ndarray) -> np.ndarray:
    f = np.vectorize(lambda x : vocab_id[x])

    return f(phonemes)