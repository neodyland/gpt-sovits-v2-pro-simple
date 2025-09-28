from src.inference import get_tts_wav, load
from src.textcut import Strategy
from fastapi import FastAPI, Response, File, Form, UploadFile
from asyncio import Lock
from scipy.io import wavfile
from io import BytesIO
from typing import Literal, Optional, List
from typing_extensions import Annotated

load("v2proplus")

app = FastAPI()
lock = Lock()

type Language = Literal[
    "all_zh",
    "en",
    "all_ja",
    "all_yue",
    "all_ko",
    "zh",
    "ja",
    "yue",
    "ko",
    "auto",
    "auto_yue",
]


@app.post("/synthesize")
async def synthesize(
    ref_wavs: Annotated[Optional[List[UploadFile]], File()] = None,
    prompt_wav: Annotated[Optional[UploadFile], File()] = None,
    text: str = Form(...),
    prompt_text: Optional[str] = Form(None),
    prompt_language: Language = Form("auto"),
    text_language: Language = Form("auto"),
    how_to_cut: Strategy = Form("four_sentences"),
    top_k: int = Form(15, ge=1, le=100),
    top_p: float = Form(1.0, ge=0, le=1),
    temperature: float = Form(1.0, ge=0, le=1),
    speed: float = Form(1, ge=0.6, le=1.65),
    freeze: bool = Form(False),
    pause_second: float = Form(0.3, ge=0.1, le=0.5),
):
    prompt_wav = await prompt_wav.read() if prompt_wav else None
    ref_wavs = [BytesIO(await f.read()) for f in ref_wavs] if ref_wavs else None
    async with lock:
        sr, pcm = get_tts_wav(
            ref_wav_path=prompt_wav,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            text=text,
            text_language=text_language,
            how_to_cut=how_to_cut,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed,
            if_freeze=freeze,
            inp_refs=ref_wavs,
            pause_second=pause_second,
        )
        print(len(pcm) / sr)
    with BytesIO() as buf:
        wavfile.write(buf, sr, pcm)
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
