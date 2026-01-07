from io import BytesIO
from typing import Optional

import gradio as gr
import librosa
import numpy as np
import torch
from scipy.io import wavfile

from server.tts import TTS
from server.utils import clip, transcribe

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch.autograd.set_grad_enabled(False)
tts = TTS("v2proplus")


def load_and_clip_wav(wav_path: str):
    wav = open(wav_path, "rb").read()
    y, sr = librosa.load(BytesIO(wav))
    y = clip(y, sr)
    with BytesIO() as buf:
        wavfile.write(buf, sr, (y * 32767).astype("int16"))
        buf.seek(0)
        wav = buf.read()
    return wav


def inference(
    prompt_wav_path: Optional[str],
    text: str,
    speed: float,
):
    if prompt_wav_path is None or text.strip() == "":
        return None, ""
    wav = load_and_clip_wav(prompt_wav_path)
    prompt_text = transcribe(wav)
    sr, pcm, timing = tts.synthesize(
        [], wav, prompt_text, "auto", "four_sentences", text, "auto", 15, 1, 1, speed
    )
    return (sr, (pcm * 32767).astype(np.int16)), "\n".join(str(timing).split(" | "))


with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False) as app:
    with gr.Group():
        with gr.Row():
            with gr.Column():
                prompt_wav = gr.Audio(
                    label="Please upload the reference audio within 3~10 seconds. If it exceeds the limit, an error will be reported!",
                    type="filepath",
                    scale=14,
                )
            with gr.Column():
                text = gr.Textbox(
                    label="Text that needs to be synthesized",
                    value="",
                    max_lines=10,
                    lines=10,
                )
                speed = gr.Slider(
                    minimum=0.6,
                    maximum=1.65,
                    step=0.05,
                    label="speed",
                    value=1,
                    scale=4,
                )
        with gr.Column():
            inference_button = gr.Button(value="synthesize", variant="primary", scale=8)
            with gr.Row():
                with gr.Column():
                    output_audio = gr.Audio(label="output speech", scale=14)
                with gr.Column():
                    timing = gr.Textbox(label="timing", lines=6, max_lines=6)

        inference_button.click(
            inference,
            [
                prompt_wav,
                text,
                speed,
            ],
            [output_audio, timing],
        )

if __name__ == "__main__":
    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=False,
        share=False,
        server_port=9872,
    )
