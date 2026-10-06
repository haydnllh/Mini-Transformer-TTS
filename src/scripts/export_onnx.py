import torch
from src.utils.get_unicode import phonemes_to_id
from src.utils.text_to_phonemes import text_to_phonemes
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np

from src.models.tts_model import TTS_model
from src.inference.infer import infer

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts_model = TTS_model(device=device)
    tts_model.load_state_dict(torch.load("trained_models/refined_tts_model_weights.pth", map_location=device, weights_only=True))

    batch_size = 1
    src_len = 100
    tgt_len = 100

    dummy_input = torch.randint(1, 79, (batch_size, src_len), dtype=torch.int64).to(device)
    dummy_target = torch.zeros((batch_size, tgt_len, 80), dtype=torch.float32).to(device)
    dummy_src_mask = torch.zeros((batch_size, src_len), dtype=torch.bool).to(device)
    dummy_tgt_mask = torch.zeros((batch_size, tgt_len), dtype=torch.bool).to(device)

    torch.onnx.export(
        tts_model,
        (dummy_input, dummy_target, dummy_src_mask, dummy_tgt_mask),
        "trained_models/refined_tts_model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["src", "target", "src_key_padding_mask", "tgt_key_padding_mask"],
        output_names=["mel", "stop"],
        dynamic_axes={
            "src": {0: "batch_size", 1: "src_seq_len"},
            "target": {0: "batch_size", 1: "tgt_seq_len"},
            "src_key_padding_mask": {0: "batch_size", 1: "src_seq_len"},
            "tgt_key_padding_mask": {0: "batch_size", 1: "tgt_seq_len"},
            "mel": {0: "batch_size", 1: "tgt_seq_len"},
            "stop": {0: "batch_size", 1: "tgt_seq_len"}
        }
    )

    print("Model successfully exported to onnx!")