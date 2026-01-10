import time
from argparse import ArgumentParser
from io import BytesIO

import librosa
import requests
from scipy.io import wavfile

from server.utils import clip, transcribe


def synthesize(
    url: str, wav: bytes, text: str, text_language: str, prompt_language: str
):
    y, sr = librosa.load(BytesIO(wav))
    y = clip(y, sr)
    with BytesIO() as buf:
        wavfile.write(buf, sr, (y * 32767).astype("int16"))
        buf.seek(0)
        wav = buf.read()
    files = {
        "prompt_wav": (
            "prompt.wav",
            wav,
            "audio/wav",
        )
    }
    prompt_text = transcribe(wav, prompt_language)
    print(f"Transcribed prompt text: {prompt_text}")
    data = {
        "prompt_text": prompt_text,
        "text": text,
        "text_language": text_language,
        "prompt_language": prompt_language,
    }
    now = time.time()
    response = requests.post(url, data=data, files=files)
    response.raise_for_status()
    content = response.content
    print(f"Synthesis took {time.time() - now:.2f}s")
    print(f"Transcribed result wav: {transcribe(content)}")
    return content


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:7860/synthesize")
    parser.add_argument("--wav", type=str)
    parser.add_argument("--text", type=str, default="こんにちは、元気ですか？")
    parser.add_argument("--output", type=str, default="./wav/output.wav")
    args = parser.parse_args()
    content = synthesize(
        args.url,
        open(args.wav, "rb").read(),
        args.text,
        "auto",
        "ja",
    )
    with open(args.output, "wb") as f:
        f.write(content)
    print(f"Wrote output to {args.output}")
