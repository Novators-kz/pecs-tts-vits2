import io
import os
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Dict, Tuple
from app.tts import TTSService

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: str = "girl"

app = FastAPI()

# One TTS service instance per (language, voice)
tts_services: Dict[Tuple[str, str], TTSService] = {}

# Models are read from TTS_MODEL_DIR (default: current dir); configs sit next to the code.
MODEL_DIR = os.getenv("TTS_MODEL_DIR", ".")

@app.on_event("startup")
async def load_models():
    """Load all available TTS models on application startup."""
    global tts_services
    # (lang, voice) -> file stem: vits2_<stem>.onnx and config_<stem>.json
    model_configs = {
        ("en", "girl"): "eng_girl",
        ("kz", "girl"): "kaz_girl",
        ("ru", "girl"): "rus_girl",
        ("kz", "nao"): "kaz_nao",
        ("ru", "nao"): "rus_nao",
    }

    for (lang, voice), name in model_configs.items():
        model_path = os.path.join(MODEL_DIR, f"vits2_{name}.onnx")
        config_path = f"config_{name}.json"
        # Check if the files exist before loading
        if os.path.exists(model_path) and os.path.exists(config_path):
            tts_services[(lang, voice)] = TTSService(
                model_path=model_path,
                config_path=config_path,
                lang=lang
            )
        else:
            # Handle cases where model files might be missing
            print(f"Warning: Model files for '{lang}'/'{voice}' not found. Skipping.")


@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    """Synthesize speech from text using the selected language model."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    # Get the appropriate TTS service based on the requested language and voice
    tts_service = tts_services.get((req.lang, req.voice))
    if not tts_service:
        raise HTTPException(status_code=404, detail=f"Language '{req.lang}' with voice '{req.voice}' is not supported.")

    wav_bytes = tts_service.synthesize(req.text)

    # ─── Save a copy locally ─────────────────────────────────────────────────
    # 1. ensure your output folder exists
    save_dir = "tts_audios"
    os.makedirs(save_dir, exist_ok=True)

    # 2. pick a filename (e.g. include timestamp  lang)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    filename = f"tts_{req.lang}_{timestamp}.wav"
    filepath = os.path.join(save_dir, filename)

    # 3. write the bytes to disk
    with open(filepath, "wb") as f:
        f.write(wav_bytes)
    print(f"[saved] {filepath}")
    # ─────────────────────────────────────────────────────────────────────────

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f"inline; filename=tts_{req.lang}.wav"}
    )