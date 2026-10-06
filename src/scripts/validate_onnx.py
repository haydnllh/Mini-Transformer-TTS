import onnx
import onnxruntime as ort
import numpy as np
import torch

if __name__ == "__main__":
    onnx_model = onnx.load("trained_models/refined_tts_model.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure is valid")

    session = ort.InferenceSession("trained_models/refined_tts_model.onnx", providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    batch_size = 1
    src_len = 10
    tgt_len = 15

    test_input = np.random.randint(1, 79, size=(batch_size, src_len), dtype=np.int64)
    test_target = np.zeros((batch_size, tgt_len, 80), dtype=np.float32)
    test_src_mask = np.zeros((batch_size, src_len), dtype=bool)
    test_tgt_mask = np.zeros((batch_size, tgt_len), dtype=bool)

    ort_inputs = {
        "src": test_input,
        "target": test_target,
        "src_key_padding_mask": test_src_mask,
        "tgt_key_padding_mask": test_tgt_mask,
    }

    outputs = session.run(None, ort_inputs)
    mel_np, stop_np = outputs[0], outputs[1]

    print("ONNX Output shape:", outputs[0].shape)