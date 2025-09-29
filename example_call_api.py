import requests
import librosa
from scipy.io import wavfile
from io import BytesIO
from faster_whisper import WhisperModel
import numpy as np
from argparse import ArgumentParser

model = WhisperModel("kotoba-tech/kotoba-whisper-v2.0-faster")


def clip(y: np.ndarray, sr: int):
    intervals = librosa.effects.split(y, top_db=30)

    min_len = 3 * sr
    max_len = 10 * sr
    candidates = []

    for start, end in intervals:
        length = end - start
        if min_len <= length <= max_len:
            candidates.append((start, end))

    if not candidates:
        for start, end in intervals:
            length = end - start
            if length > max_len:
                mid = (start + end) // 2
                candidates.append((mid - max_len // 2, mid + max_len // 2))
                break

    if candidates:
        start, end = candidates[np.random.randint(len(candidates))]
        y_out = y[start:end]
    else:
        y_out = y[:max_len]
    return y_out


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
    segments, info = model.transcribe(
        BytesIO(wav), language="ja", chunk_length=15, condition_on_previous_text=False
    )
    data = {
        "prompt_text": "".join([segment.text for segment in segments]),
        "text": text,
        "text_language": text_language,
        "prompt_language": prompt_language,
    }
    response = requests.post(url, data=data, files=files)
    response.raise_for_status()
    return response.content


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:7860/synthesize")
    parser.add_argument("--wav", type=str)
    parser.add_argument("--text", type=str, default="こんにちは、元気ですか？")
    args = parser.parse_args()
    content = synthesize(
        args.url,
        open(args.wav, "rb").read(),
        args.text,
        "auto",
        "ja",
    )
    with open("test.wav", "wb") as f:
        f.write(content)
    print("Saved as test.wav")
