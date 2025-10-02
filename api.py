import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

from server.tts import TTS
from server.textcut import Strategy
from fastapi import FastAPI, Response, File, Form, UploadFile
from asyncio import Lock
from scipy.io import wavfile
from io import BytesIO
from typing import Literal, Optional, List
from typing_extensions import Annotated

tts = TTS("v2proplus")

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
    ref_wavs: Annotated[List[UploadFile], File()] = [],
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
):
    prompt_wav = await prompt_wav.read() if prompt_wav else None
    ref_wavs = [BytesIO(await f.read()) for f in ref_wavs]
    async with lock:
        sr, pcm = tts.synthesize(
            ref_wavs=ref_wavs,
            prompt_wav=prompt_wav,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            how_to_cut=how_to_cut,
            text=text,
            text_language=text_language,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed,
        )
        print(len(pcm) / sr)
    with BytesIO() as buf:
        wavfile.write(buf, sr, pcm)
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
