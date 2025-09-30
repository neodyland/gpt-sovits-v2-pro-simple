import traceback
from typing import List, Optional
import torch
import torchaudio
from time import time as ttime
import librosa
from .textcut import preprocess_text, splits
from .mel_processing import spectrogram_torch
from .eres2net.sv import SV
from .lang_segmenter import segment
from transformers import AutoModelForMaskedLM, AutoTokenizer
from .gpt_sovits.models import SynthesizerTrn
from .ar.t2s_model import Text2SemanticDecoder
from .text.text_cleaner import clean_text_inf
import safetensors.torch as st
import json
from io import BytesIO
from typing import Literal
from transformers import (
    HubertModel,
)

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

bert_path = "./data/chinese-roberta-wwm-ext-large"
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = (
    AutoModelForMaskedLM.from_pretrained(bert_path)
    .to(dtype=dtype, device=device)
    .eval()
)
sv_cn_model = SV(device, dtype)


@torch.inference_mode()
def get_bert_feature(text: str, word2ph: Optional[List[int]]):
    inputs = tokenizer(text, return_tensors="pt")
    for i in inputs:
        inputs[i] = inputs[i].to(device, non_blocking=True)
    res = bert_model(**inputs, output_hidden_states=True)
    res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def get_bert_inf(
    phones: List[int], word2ph: Optional[List[int]], norm_text: str, language: str
):
    if language.replace("all_", "") == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros((1024, len(phones)), dtype=dtype, device=device)

    return bert


def load(type: Literal["v2pro", "v2proplus"]):
    global \
        filter_length, \
        sampling_rate, \
        hop_length, \
        win_length, \
        hz_x_max_sec, \
        vq_model, \
        t2s_model
    hps = dict(json.load(open(f"./data/v2pro/{type}.json", "r")))
    filter_length = hps["data"]["filter_length"]
    hop_length = hps["data"]["hop_length"]
    vq_model = (
        SynthesizerTrn(
            **hps["model"],
        )
        .to(dtype=dtype, device=device)
        .eval()
    )
    vq_model.load_state_dict(
        st.load_file(
            f"./data/v2pro/{type}.safetensors",
            device=device,
        ),
        strict=False,
    )
    sampling_rate = hps["data"]["sampling_rate"]
    win_length = hps["data"]["win_length"]

    config = json.load(open("./data/gsv/config.json", "r"))
    hz_x_max_sec = 50 * config["data"]["max_sec"]  # hz is 50
    t2s_model = (
        Text2SemanticDecoder(config=config, top_k=3)
        .to(dtype=dtype, device=device)
        .eval()
    )
    t2s_model.load_state_dict(
        st.load_file(
            "./data/gsv/s1bert25hz-5kh-longer-12-epoch-369668-step.safetensors",
            device=device,
        ),
        strict=False,
    )


resample_transform_dict = {}


def resample(audio_tensor, sr0, sr1, device):
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(
            device
        )
    return resample_transform_dict[key](audio_tensor)


def get_spepc(filter_length, sr1, hop_length, win_length, audio, dtype, device):
    if audio.shape[0] == 2:
        audio = audio.mean(0).unsqueeze(0)

    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    spec = spectrogram_torch(
        audio,
        filter_length,
        sampling_rate,
        hop_length,
        win_length,
        center=False,
    )
    spec = spec.to(dtype)
    audio = resample(audio, sr1, 16000, device).to(dtype)
    return spec, audio


def get_phones_and_bert(text: str, language: str, final=False):
    textlist, langlist = segment(text, language)
    print(textlist)
    print(langlist)
    phones_list = []
    bert_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(
            textlist[i], lang.replace("all_", "")
        )
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, final=True)

    return phones, bert.to(dtype), norm_text


cache = {}


@torch.inference_mode()
def get_tts_wav(
    ref_wav_path,  # required
    prompt_text,
    prompt_language,
    text: str,  # required
    text_language,
    how_to_cut,
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    speed=1,
    if_freeze=False,
    inp_refs=None,
    pause_second=0.3,
):
    global cache
    t = []
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    else:
        ref_free = False
    t0 = ttime()

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print("Actual reference text", prompt_text)

    zero_wav_torch = torch.zeros(
        int(sampling_rate * pause_second), dtype=dtype, device=device
    )
    if ref_wav_path is not None and isinstance(ref_wav_path, bytes):

        def get_path():
            return BytesIO(ref_wav_path)
    else:

        def get_path():
            return ref_wav_path

    if not ref_free:
        wav16k, _sr = librosa.load(get_path(), sr=16000)
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            raise ValueError("Please use a 3~10 seconds reference audio!")
        wav16k = torch.from_numpy(wav16k)
        wav16k = wav16k.to(dtype=dtype, device=device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1 - t0)
    texts = preprocess_text(text, how_to_cut)
    audio_opt = []
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language)

    for i_text, text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if len(text.strip()) == 0:
            continue
        if text[-1] not in splits:
            text += "." if text_language == "en" else "。"
        print("Actual input target text (per sentence)", text)
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language)
        print("Front-end processed text (each sentence):", norm_text2)
        if ref_free:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)
        else:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = (
                torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
            )

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()

        if i_text in cache and if_freeze == True:
            pred_semantic = cache[i_text]
        else:
            pred_semantic, idx = t2s_model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz_x_max_sec,
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            cache[i_text] = pred_semantic
        t3 = ttime()
        refers = []
        sv_emb = []
        if inp_refs:
            for path in inp_refs:
                try:
                    audio = librosa.load(path, sr=sampling_rate)[0]
                    audio = (
                        torch.from_numpy(audio)
                        .to(device=device)
                        .squeeze(-1)
                        .unsqueeze(0)
                    )
                    refer, audio_tensor = get_spepc(
                        filter_length,
                        sampling_rate,
                        hop_length,
                        win_length,
                        audio,
                        dtype,
                        device,
                    )
                    refers.append(refer)
                    sv_emb.append(sv_cn_model.compute_embedding3(audio_tensor))
                except:
                    traceback.print_exc()
        if len(refers) == 0:
            audio = librosa.load(get_path(), sr=sampling_rate)[0]
            audio = torch.from_numpy(audio).to(device=device).squeeze(-1).unsqueeze(0)
            refers, audio_tensor = get_spepc(
                filter_length,
                sampling_rate,
                hop_length,
                win_length,
                audio,
                dtype,
                device,
            )
            refers = [refers]
            sv_emb = [sv_cn_model.compute_embedding3(audio_tensor)]
        audio = vq_model.decode(
            pred_semantic,
            torch.LongTensor(phones2).to(device).unsqueeze(0),
            refers,
            speed=speed,
            sv_emb=sv_emb,
        )[0][0]
        max_audio = torch.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio = audio / max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav_torch)  # zero_wav
        t4 = ttime()
        t.extend([t2 - t1, t3 - t2, t4 - t3])
        t1 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))
    audio_opt = torch.cat(audio_opt, 0)  # np.concatenate
    audio_opt = audio_opt.float().cpu().detach().numpy()
    return 32000, audio_opt
