from asyncio import Lock
from io import BytesIO
from typing import List, Literal, Optional

import torch
from fastapi import FastAPI, File, Form, Response, UploadFile
from scipy.io import wavfile
from typing_extensions import Annotated

from server.textcut import Strategy
from server.tts import TTS

torch.backends.cuda.matmul.allow_tf32 = True

torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch.autograd.set_grad_enabled(False)

tts = TTS("v2proplus")
lock = Lock()

app = FastAPI()

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
    prompt_wav_bytes = await prompt_wav.read() if prompt_wav else None
    ref_wavs_bytesio = [BytesIO(await f.read()) for f in ref_wavs]
    async with lock:
        sr, pcm, timing = tts.synthesize(
            ref_wavs=ref_wavs_bytesio,
            prompt_wav=prompt_wav_bytes,
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
    print(f"Time taken: {timing}")
    print(f"Output length: {len(pcm) / sr:.2f} seconds")
    with BytesIO() as buf:
        wavfile.write(buf, sr, pcm)
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
