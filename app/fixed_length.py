import numpy as np

MAX_LEN = 100
PAD_TOKEN_ID = 52

def text_to_phoneme_ids(text: str) -> list[int]:
    return [ord(char) for char in text]

def prepare_fixed_chunks(phoneme_ids: list[int], max_len: int = MAX_LEN) -> list[tuple[np.ndarray, int]]:
    if isinstance(phoneme_ids, np.ndarray):
        phoneme_ids = phoneme_ids.tolist()
    elif hasattr(phoneme_ids, "tolist"):
        phoneme_ids = phoneme_ids.tolist()

    chunks = []
    
    for i in range(0, len(phoneme_ids), max_len):
        chunk = phoneme_ids[i:i + max_len]
        
        if isinstance(chunk, np.ndarray):
            chunk = chunk.tolist()
            
        original_len = len(chunk)
        
        if original_len < max_len:
            chunk = chunk + [PAD_TOKEN_ID] * (max_len - original_len)
            
        chunk_array = np.array([chunk], dtype=np.int64)
        chunks.append((chunk_array, original_len))
        
    return chunks