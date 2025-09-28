from src.inference import get_tts_wav, languages, load
import gradio as gr
import numpy as np

load("v2proplus")


def inference(
    *args,
):
    sr, pcm = get_tts_wav(*args)
    yield sr, (pcm * 32767).astype(np.int16)


with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False) as app:
    with gr.Group():
        with gr.Row():
            inp_ref = gr.Audio(
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
                inp_refs = gr.File(
                    label="Optional: Upload multiple reference audios (same gender recommended) by dragging and dropping multiple files to evenly blend their timbres. If this item is not filled in, the timbre is controlled by a single reference audio on the left. If you are fine-tuning the model, it is recommended that the reference audio is all within the fine-tuning training set timbre, and the base model is ignored.",
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
                    value="ja",
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
                if_freeze = gr.Checkbox(
                    label="Whether to directly adjust the speaking speed and timbre to the last synthesis result. Prevent randomness.",
                    value=False,
                    interactive=True,
                    show_label=True,
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
                    pause_second_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        step=0.01,
                        label="pause duration (seconds)",
                        value=0.3,
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
                inp_ref,
                prompt_text,
                prompt_language,
                text,
                text_language,
                how_to_cut,
                top_k,
                top_p,
                temperature,
                speed,
                if_freeze,
                inp_refs,
                pause_second_slider,
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
