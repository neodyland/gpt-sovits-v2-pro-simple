import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from typing import List, Optional
from transformers import AutoModelForMaskedLM, AutoTokenizer
import safetensors.torch as st
import json
from .ar.t2s_model import Text2SemanticDecoder
from .eres2net.eres2netv2 import ERes2NetV2
from .eres2net.kaldi import fbank


class Bert:
    def __init__(self, dtype, device, bert_path="./data/chinese-roberta-wwm-ext-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = (
            AutoModelForMaskedLM.from_pretrained(bert_path)
            .to(dtype=dtype, device=device)
            .eval()
        )

    @torch.inference_mode()
    def get_feature(self, text: str, word2ph: Optional[List[int]]):
        inputs = self.tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(self.bert_model.device, non_blocking=True)
        res = self.bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T


class T2S:
    def __init__(
        self,
        device,
        dtype,
        config_path="./data/gsv/config.json",
        model_path="./data/gsv/model.safetensors",
    ):
        self.model = (
            Text2SemanticDecoder(config=json.load(open(config_path, "r")))
            .to(dtype=dtype, device=device)
            .eval()
        )
        self.model.load_state_dict(
            st.load_file(
                model_path,
                device=device,
            ),
        )


class SV:
    def __init__(
        self,
        device,
        dtype,
        model_path="./data/sv/model.safetensors",
    ):
        self.embedding_model = (
            ERes2NetV2(base_width=24, scale=4, expansion=4)
            .eval()
            .to(device=device, dtype=dtype)
        )
        self.embedding_model.load_state_dict(
            st.load_file(
                model_path,
                device=device,
            )
        )
        self.dtype = dtype

    @torch.inference_mode()
    def compute_embedding3(self, wav: torch.Tensor):
        feat = torch.stack(
            [
                fbank(
                    wav0.unsqueeze(0),
                    num_mel_bins=80,
                    sample_frequency=16000,
                    dither=0,
                )
                for wav0 in wav.to(dtype=self.dtype)
            ]
        )
        return self.embedding_model.forward3(feat)
