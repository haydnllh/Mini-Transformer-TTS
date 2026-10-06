from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import traceback

from app.inference import TTSInferenceEngine
from src.utils.get_unicode import phonemes_to_id
from src.utils.text_to_phonemes import text_to_phonemes

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["tts_engine"] = TTSInferenceEngine(model_path="trained_models/refined_tts_model.onnx")
    yield
    ml_models.clear()

app = FastAPI(
    title="Transformer TTS Inference API",
    description="Low-latency text-to-speech engine powered by ONNX Runtime and FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

class TTSRequest(BaseModel):
    text: str = Field(..., example="Hello world, this is a live text to speech test")

@app.get("/health")
async def health_check():
    if "tts_engine" not in ml_models:
        raise HTTPException(status_code=503, detail="Model session not initialized")
    return {"status": "healthy", "model_loaded": True}

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    start_time = time.perf_counter()
    engine: TTSInferenceEngine = ml_models["tts_engine"]

    try:
        phonemes = text_to_phonemes(request.text)
        phoneme_ids = phonemes_to_id(phonemes)

        mel_spectrogram = engine.run_inference_fixed(phoneme_ids)

        audio_bytes = engine.mel_to_wav_bytes(mel_spectrogram)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Latency-Ms": f"{latency_ms:.2f}",
                "X-Generated-Frames": str(mel_spectrogram.shape[1]),
                "Content-Disposition": "inline; filename=speech.wav"
            }
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")