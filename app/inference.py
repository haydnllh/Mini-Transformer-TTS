import numpy as np
import onnxruntime as ort
import librosa
import io
import soundfile as sf

from app.fixed_length import prepare_fixed_chunks, MAX_LEN, PAD_TOKEN_ID

class TTSInferenceEngine:
    def __init__(self, model_path = "models/transformer_tts.onnx"):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.output_name = self.session.get_outputs()[0].name

    def run_inference_fixed(self, phoneme_ids):
        chunks_with_lengths = prepare_fixed_chunks(phoneme_ids, max_len=MAX_LEN)
        mel_outputs = []

        for chunk_tensor, original_len in chunks_with_lengths:
            batch_size, seq_len = chunk_tensor.shape

            src = chunk_tensor.astype(np.int64)

            target_len = seq_len
            mel_channels = 80
            target = np.zeros((batch_size, target_len, mel_channels), dtype=np.float32)

            src_key_padding_mask = (src == PAD_TOKEN_ID).astype(np.bool)
            tgt_key_padding_mask = np.zeros((batch_size, target_len), dtype=np.bool)

            input_feed = {
                "src": src,
                "target": target,
                "src_key_padding_mask": src_key_padding_mask,
                "tgt_key_padding_mask": tgt_key_padding_mask,
            }

            outputs = self.session.run([self.output_name], input_feed)
            mel_spec = outputs[0]

            if original_len < MAX_LEN:
                valid_time_steps = int(mel_spec.shape[1] * (original_len / MAX_LEN))
                mel_spec = mel_spec[:, :valid_time_steps, :]

            mel_outputs.append(mel_spec)

        full_mel_spectrogram = np.concatenate(mel_outputs, axis=1)
        return full_mel_spectrogram
    
    def mel_to_wav_bytes(self, mel_spec, sample_rate = 22050):
        if mel_spec.ndim == 3:
            mel_spec = mel_spec.squeeze(0)
        if mel_spec.shape[0] != 80 and mel_spec.shape[1] == 80:
            mel_spec = mel_spec.T

        stft = librosa.feature.inverse.mel_to_stft(
            librosa.db_to_power(mel_spec), sr=sample_rate, n_fft=2048
        )

        waveform = librosa.griffinlim(stft, hop_length=512, n_fft=2048)

        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format="wav", subtype="PCM_16")
        buffer.seek(0)

        return buffer.getvalue()