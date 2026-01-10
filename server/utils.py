from io import BytesIO
from typing import Optional

import librosa
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("large-v3-turbo")


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


def transcribe(wav: bytes, language: Optional[str] = None):
    segments, info = model.transcribe(
        BytesIO(wav),
        chunk_length=15,
        condition_on_previous_text=False,
        language=language,
    )
    return "".join([segment.text for segment in segments])
