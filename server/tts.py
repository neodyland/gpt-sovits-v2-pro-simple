from typing import List, Optional
import torch
import torchaudio
import librosa
from .textcut import preprocess_text, append_final_punctuation
from .lang_segmenter import segment
from .gpt_sovits.models import SynthesizerTrn
from .text.text_cleaner import clean_text
import safetensors.torch as st
import json
from io import BytesIO
from typing import Literal
from transformers import (
    HubertModel,
)
from torch.nn import functional as F
from .models import Bert, T2S, SV

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
ssl_model = (
    HubertModel.from_pretrained("./data/chinese-hubert-base", local_files_only=True)
    .to(dtype=dtype, device=device)
    .eval()
)

languages = [
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

sv_cn_model = SV(device, dtype)
t2s_model = T2S(device, dtype)
bert_model = Bert(dtype, device)


def get_bert(
    phones: List[int], word2ph: Optional[List[int]], norm_text: str, language: str
):
    if language.replace("all_", "") == "zh":
        bert = bert_model.get_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros((1024, len(phones)), dtype=dtype, device=device)

    return bert


def clip(audio: torch.Tensor, min_value: Optional[float] = None):
    max_audio = audio.abs().max()
    if max_audio > 1:
        audio /= max_audio if min_value is None else min(min_value, max_audio)
    return audio


def get_phones_and_bert(text: str, language: str, final=False):
    textlist, langlist = segment(text, language)
    phones_list = []
    bert_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text(textlist[i], lang.replace("all_", ""))
        bert = get_bert(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, final=True)

    return phones, bert.to(dtype)


class TTS:
    resample_transform_dict = {}

    def __init__(self, variant: Literal["v2pro", "v2proplus"]):
        hps = dict(json.load(open(f"./data/v2pro/{variant}.json", "r")))
        self.filter_length = hps["data"]["filter_length"]
        self.hop_length = hps["data"]["hop_length"]
        self.sampling_rate = hps["data"]["sampling_rate"]
        self.win_length = hps["data"]["win_length"]
        self.vq_model = (
            SynthesizerTrn(
                **hps["model"],
            )
            .to(dtype=dtype, device=device)
            .eval()
        )
        self.vq_model.load_state_dict(
            st.load_file(
                f"./data/v2pro/{variant}.safetensors",
                device=device,
            ),
            strict=True,
        )  #!TODO

        self.zero_wav_torch = torch.zeros(
            int(self.sampling_rate * 0.3), dtype=dtype, device=device
        )
        self.hann = torch.hann_window(self.win_length).to(dtype=dtype, device=device)

    def resample(self, audio: torch.Tensor, sr: int):
        key = "%s-%s" % (sr, str(audio.device))
        if key not in self.resample_transform_dict:
            self.resample_transform_dict[key] = torchaudio.transforms.Resample(
                self.sampling_rate, sr
            ).to(audio.device)
        return self.resample_transform_dict[key](audio)

    def spectrogram_torch(
        self,
        y: torch.Tensor,
    ):
        if torch.min(y) < -1.2:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.2:
            print("max value is ", torch.max(y))

        y = F.pad(
            y.unsqueeze(1),
            (
                int((self.filter_length - self.hop_length) / 2),
                int((self.filter_length - self.hop_length) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)
        spec = torch.view_as_real(
            torch.stft(
                y,
                self.filter_length,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.hann,
                center=False,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )

        spec = (spec.pow(2).sum(-1) + 1e-8).sqrt()
        return spec

    def get_spepc(self, audio: torch.Tensor):
        audio = clip(audio, 2.0)
        spec = self.spectrogram_torch(
            audio,
        ).to(dtype)
        audio = self.resample(audio, 16000).to(dtype)
        return spec, audio

    def load_audio(self, source: bytes, sr: Optional[int] = None):
        wav = librosa.load(BytesIO(source), sr=sr, mono=True)[0]
        wav = torch.from_numpy(wav)
        wav = wav.to(device=device)
        return wav

    def reference_prompt(self, source: bytes, prompt_text: str, prompt_language: str):
        wav16k = self.load_audio(source, sr=16000).to(dtype=dtype)
        if wav16k.shape[0] < 48000 or wav16k.shape[0] > 160000:
            raise ValueError("Please use a 3~10 seconds reference audio!")
        wav16k = torch.cat([wav16k, self.zero_wav_torch])
        ssl_content = ssl_model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(
            1, 2
        )
        codes = self.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)
        phones, bert = get_phones_and_bert(
            append_final_punctuation(prompt_text.strip("\n"), prompt_language),
            prompt_language,
        )
        return prompt, phones, bert

    def reference_audios(self, ref_wavs: List[bytes]):
        refers_sb_embs = []
        for ref_wav in ref_wavs:
            audio = (
                self.load_audio(ref_wav, sr=self.sampling_rate).squeeze(-1).unsqueeze(0)
            )
            refer, audio_tensor = self.get_spepc(
                audio,
            )
            refers_sb_embs.append((refer, sv_cn_model.compute_embedding3(audio_tensor)))
        ge = self.vq_model.get_ges(refers_sb_embs)
        return ge

    @torch.inference_mode()
    def synthesize(
        self,
        ref_wavs: list[bytes],
        prompt_wav: Optional[bytes],
        prompt_text: Optional[str],
        prompt_language: Optional[str],
        how_to_cut: str,
        text: str,
        text_language: str,
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        speed=1,
    ):
        if prompt_text is not None:
            prompt, phones1, bert1 = self.reference_prompt(
                prompt_wav, prompt_text, prompt_language
            )
            ref_wavs = [prompt_wav]
        ge = self.reference_audios(ref_wavs)
        audios = [self.zero_wav_torch]
        for text in preprocess_text(how_to_cut, text, text_language):
            print("Actual input target text (per sentence)", text)
            phones2, bert2 = get_phones_and_bert(text, text_language)
            if prompt_text is not None:
                bert = torch.cat([bert1, bert2], 1)
                all_phoneme_ids = (
                    torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
                )
            else:
                bert = bert2
                all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                prompt if prompt_text is not None else None,
                bert.to(device).unsqueeze(0),
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            audio = self.vq_model.decode(
                pred_semantic,
                torch.LongTensor(phones2).to(device).unsqueeze(0),
                ge,
                speed=speed,
            )[0][0]
            audios.append(clip(audio))
            audios.append(self.zero_wav_torch)
        return 32000, torch.cat(audios, dim=0).float().cpu().detach().numpy()
