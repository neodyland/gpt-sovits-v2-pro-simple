from io import BytesIO
from typing import Optional

import gradio as gr
import librosa
import numpy as np
import torch
import torch._inductor.config as inductor_config
from scipy.io import wavfile

from server.tts import TTS, languages
from server.utils import clip, transcribe

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch.autograd.set_grad_enabled(False)
inductor_config.freezing = True
inductor_config.epilogue_fusion = True

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


def check_audio_length(wav_path: str):
    if wav_path is None:
        return True, ""
    wav = open(wav_path, "rb").read()
    y, sr = librosa.load(BytesIO(wav))
    duration = len(y) / sr
    if duration < 3.0:
        return (
            False,
            f"Audio is too short ({duration:.2f}s). Please use audio between 3~10 seconds.",
        )
    if duration > 10.0:
        return (
            False,
            f"Audio is too long ({duration:.2f}s). Please use audio between 3~10 seconds.",
        )
    return True, ""


def auto_transcribe(wav_path: str):
    if wav_path is None:
        return "", ""
    is_valid, error_msg = check_audio_length(wav_path)
    if not is_valid:
        return "", error_msg
    wav = load_and_clip_wav(wav_path)
    text = transcribe(wav)
    return text, ""


def inference(
    ref_wav_paths: Optional[list[str]],
    prompt_wav_path: Optional[str],
    prompt_text: Optional[str],
    *args,
):
    if prompt_wav_path is None and (ref_wav_paths is None or len(ref_wav_paths) == 0):
        return None, ""
    if prompt_wav_path is not None:
        is_valid, error_msg = check_audio_length(prompt_wav_path)
        if not is_valid:
            return None, error_msg
    ref_wav_paths = ref_wav_paths or []
    ref_wavs = [open(f, "rb").read() for f in ref_wav_paths]
    if prompt_wav_path is not None:
        prompt_wav = open(prompt_wav_path, "rb").read()
        prompt_text = prompt_text.strip()
    else:
        prompt_wav = None
        prompt_text = None
    sr, pcm, timing = tts.synthesize(ref_wavs, prompt_wav, prompt_text, *args)
    return (
        (
            sr,
            (pcm * 32767).astype(np.int16),
        ),
        "\n".join(str(timing).split(" | ")),
    )


with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False) as app:
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1):
                prompt_wav = gr.Audio(
                    label="Please upload the reference audio within 3~10 seconds. The text will be automatically transcribed.",
                    type="filepath",
                )
            with gr.Column(scale=1):
                prompt_text = gr.Textbox(
                    label="Reference audio text (auto-transcribed)",
                    value="",
                    lines=14,
                    max_lines=14,
                )
            with gr.Column(scale=1):
                prompt_language = gr.Dropdown(
                    label="Reference audio language",
                    choices=languages,
                    value="ja",
                )
                ref_wavs = gr.File(
                    label="Upload multiple reference audios (same gender recommended) by dragging and dropping multiple files to evenly blend their timbres. If this item is not filled in, the timbre is controlled by a single reference audio on the left. If you are fine-tuning the model, it is recommended that the reference audio is all within the fine-tuning training set timbre, and the base model is ignored.",
                    file_count="multiple",
                )
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(
                    label="Text that needs to be synthesized",
                    value="",
                    lines=22,
                    max_lines=22,
                )
            with gr.Column(scale=1):
                text_language = gr.Dropdown(
                    label="Languages that need to be synthesized. The smaller the restriction range, the better the discrimination effect.",
                    choices=languages,
                    value="auto",
                )
                how_to_cut = gr.Dropdown(
                    label="How to cut",
                    choices=[
                        "four_sentences",
                        "fifty_characters",
                        "chinese_period",
                        "english_period",
                        "punctuation",
                    ],
                    value="four_sentences",
                    interactive=True,
                )
                speed = gr.Slider(
                    minimum=0.6,
                    maximum=1.65,
                    step=0.05,
                    label="speed",
                    value=1,
                    interactive=True,
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    label="top_k",
                    value=15,
                    interactive=True,
                )
                top_p = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    label="top_p",
                    value=1,
                    interactive=True,
                )
                temperature = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    label="temperature",
                    value=1,
                    interactive=True,
                )
        with gr.Row(equal_height=True):
            inference_button = gr.Button(value="synthesize", variant="primary", scale=1)
            debug_info = gr.Textbox(
                label="Status / Error",
                value="",
                lines=6,
                max_lines=6,
                interactive=False,
                scale=2,
            )
            output = gr.Audio(label="output speech", scale=2)

        inference_button.click(
            inference,
            [
                ref_wavs,
                prompt_wav,
                prompt_text,
                prompt_language,
                how_to_cut,
                text,
                text_language,
                top_k,
                top_p,
                temperature,
                speed,
            ],
            [output, debug_info],
        )
        prompt_wav.change(
            auto_transcribe,
            [prompt_wav],
            [prompt_text, debug_info],
        )

if __name__ == "__main__":
    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=False,
        share=False,
        server_port=9872,
    )
