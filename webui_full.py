import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch.autograd.set_grad_enabled(False)

from server.tts import TTS, languages
import gradio as gr
import numpy as np
from typing import Optional


tts = TTS("v2proplus")


def inference(
    ref_wav_paths: Optional[list[str]],
    prompt_wav_path: Optional[str],
    *args,
):
    if prompt_wav_path is None and (ref_wav_paths is None or len(ref_wav_paths) == 0):
        return None
    ref_wav_paths = ref_wav_paths or []
    ref_wavs = [open(f, "rb").read() for f in ref_wav_paths]
    if prompt_wav_path is not None:
        prompt_wav = open(prompt_wav_path, "rb").read()
    sr, pcm, timing = tts.synthesize(ref_wavs, prompt_wav, *args)
    print(timing)
    return (sr, (pcm * 32767).astype(np.int16),)


with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False) as app:
    with gr.Group():
        with gr.Row():
            prompt_wav = gr.Audio(
                label="Please upload the reference audio within 3~10 seconds. If it exceeds the limit, an error will be reported!",
                type="filepath",
                scale=13,
            )
            with gr.Column(scale=13):
                prompt_text = gr.Textbox(
                    label="Reference audio text",
                    value="",
                    lines=5,
                    max_lines=5,
                    scale=1,
                )
            with gr.Column(scale=14):
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
            with gr.Column(scale=13):
                text = gr.Textbox(
                    label="Text that needs to be synthesized",
                    value="",
                    lines=26,
                    max_lines=26,
                )
            with gr.Column(scale=7):
                text_language = gr.Dropdown(
                    label="Languages that need to be synthesized. The smaller the restriction range, the better the discrimination effect.",
                    choices=languages,
                    value="auto",
                    scale=1,
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
                    scale=1,
                )
                with gr.Row():
                    speed = gr.Slider(
                        minimum=0.6,
                        maximum=1.65,
                        step=0.05,
                        label="speed",
                        value=1,
                        interactive=True,
                        scale=1,
                    )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    label="top_k",
                    value=15,
                    interactive=True,
                    scale=1,
                )
                top_p = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    label="top_p",
                    value=1,
                    interactive=True,
                    scale=1,
                )
                temperature = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    label="temperature",
                    value=1,
                    interactive=True,
                    scale=1,
                )
        with gr.Row():
            inference_button = gr.Button(
                value="synthesize", variant="primary", size="lg", scale=25
            )
            output = gr.Audio(label="output speech", scale=14)

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
            [output],
        )

if __name__ == "__main__":
    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=False,
        share=False,
        server_port=9872,
    )
