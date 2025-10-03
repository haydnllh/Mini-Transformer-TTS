from g2p_en import G2p
import os
from tqdm.auto import tqdm
import csv
import librosa
import matplotlib.pyplot as plt
import numpy as np


def text_to_phonemes(text):
    g2p = G2p()

    phonemes = [x.strip() if x.strip() != "" else "ss" for x in g2p(text)]
    sentence = ["sos"] + phonemes + ["eos"]

    return sentence