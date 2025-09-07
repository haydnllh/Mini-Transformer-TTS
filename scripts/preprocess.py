from g2p_en import G2p
import os
from tqdm.auto import tqdm
import csv


meta_input = "../data/raw/LJSpeech-1.1/metadata.csv"
meta_output = "../data/processed/metadata.csv"

g2p = G2p()


def convert_phonemes():
    with open(meta_input, "r", encoding="utf-8") as fin, open(
        meta_output, "w", newline="", encoding="utf-8"
    ) as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_ALL)
        writer.writerow(["file_id", "phonemes"])

        for row in tqdm(fin):
            [file_id, _, text] = row.split("|")
            phonemes = [x.strip() for x in g2p(text) if x.strip() != ""]
            phonemes_text = " ".join(phonemes)

            writer.writerow([file_id, phonemes_text])


convert_phonemes()
