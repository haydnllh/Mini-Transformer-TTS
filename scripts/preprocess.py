from g2p_en import G2p
import os
from tqdm.auto import tqdm
import csv
import librosa
import matplotlib.pyplot as plt
import numpy as np


meta_input = "../data/raw/LJSpeech-1.1/metadata.csv"
meta_output = "../data/processed/metadata.csv"
audio_input = "../data/raw/LJSpeech-1.1/wavs/"
audio_output = "../data/processed/mels/"


def convert_phonemes():
    g2p = G2p()
    END_PUNCT = [".", "!", "?"]
    sos = True
    eos = True

    with open(meta_input, "r", encoding="utf-8") as fin, open(
        meta_output, "w", newline="", encoding="utf-8"
    ) as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_ALL)
        writer.writerow(["file_id", "phonemes"])

        for row in tqdm(fin):
            [file_id, _, text] = row.split("|")

            phonemes = [x.strip() if x.strip() != "" else "ss" for x in g2p(text)]
            eos = phonemes[-1] in END_PUNCT

            if sos:
                phonemes = ["sos"] + phonemes
            if eos:
                phonemes = phonemes + ["eos"]
                sos = True
            else:
                sos = False

            phonemes_text = " ".join(phonemes)

            writer.writerow([file_id, phonemes_text])


### Store raw wav files in log-mel spectrum
def convert_mels():
    for filename in tqdm(os.listdir(audio_input)):
        waveform, sr = librosa.load(audio_input + filename, sr=22050)
        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=waveform, sr=sr, n_fft=2048, hop_length=512, n_mels=80
            ))

        np.save(audio_output + filename.replace(".wav", ".npy"), mel.T)

convert_mels()