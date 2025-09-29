import requests
import librosa
from scipy.io import wavfile
from io import BytesIO
from faster_whisper import WhisperModel
import numpy as np
from argparse import ArgumentParser

model = WhisperModel("kotoba-tech/kotoba-whisper-v2.0-faster")


def clip(y: np.ndarray, sr: int):
    min_len = 3 * sr
    max_len = 10 * sr
    # if under 3 seconds, repeat
    if len(y) < min_len:
        n_repeat = int(np.ceil(min_len / len(y)))
        print(f"Input too short ({len(y) / sr:.2f}s), repeating {n_repeat} times")
        y = np.tile(y, n_repeat)
        return y
    intervals = librosa.effects.split(y, top_db=30)
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
    if len(y_out) != len(y):
        print(f"Clipped from {len(y) / sr:.2f}s to {len(y_out) / sr:.2f}s")
    return y_out


def transcribe(wav: bytes):
    segments, info = model.transcribe(
        BytesIO(wav), language="ja", chunk_length=15, condition_on_previous_text=False
    )
    return "".join([segment.text for segment in segments])


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
    prompt_text = transcribe(wav)
    print(f"Transcribed prompt text: {prompt_text}")
    data = {
        "prompt_text": prompt_text,
        "text": text,
        "text_language": text_language,
        "prompt_language": prompt_language,
    }
    response = requests.post(url, data=data, files=files)
    response.raise_for_status()
    content = response.content
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
