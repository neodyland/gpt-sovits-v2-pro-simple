# This code is modified from https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw
# This code is modified from https://github.com/GitYCC/g2pW

import msgpack
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import onnxruntime
from opencc import OpenCC
from pypinyin import Style, pinyin
from transformers.models.auto.tokenization_auto import AutoTokenizer

from ..zh_normalization.char_convert import tranditional_to_simplified
from .dataset import prepare_onnx_input


def predict(
    session, onnx_input: Dict[str, Any], labels: List[str]
) -> Tuple[List[str], List[float]]:
    all_preds = []
    all_confidences = []
    probs = session.run(
        [],
        {
            "input_ids": onnx_input["input_ids"],
            "token_type_ids": onnx_input["token_type_ids"],
            "attention_mask": onnx_input["attention_masks"],
            "phoneme_mask": onnx_input["phoneme_masks"],
            "char_ids": onnx_input["char_ids"],
            "position_ids": onnx_input["position_ids"],
        },
    )[0]

    preds = np.argmax(probs, axis=1).tolist()
    max_probs = []
    for index, arr in zip(preds, probs.tolist()):
        max_probs.append(arr[index])
    all_preds += [labels[pred] for pred in preds]
    all_confidences += max_probs

    return all_preds, all_confidences


class G2PWOnnxConverter:
    def __init__(
        self,
        model_dir: str = "G2PWModel/",
        style: str = "bopomofo",
        model_source: str = "bert-base-chinese",
        enable_non_tradional_chinese: bool = False,
    ):
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 2 if torch.cuda.is_available() else 0
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            self.session_g2pW = onnxruntime.InferenceSession(
                os.path.join(model_dir, "model.onnx"),
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        else:
            self.session_g2pW = onnxruntime.InferenceSession(
                os.path.join(model_dir, "model.onnx"),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

        self.enable_opencc = enable_non_tradional_chinese

        self.tokenizer = AutoTokenizer.from_pretrained(model_source)

        self.non_polyphonic = {
            "一",
            "不",
            "和",
            "咋",
            "嗲",
            "剖",
            "差",
            "攢",
            "倒",
            "難",
            "奔",
            "勁",
            "拗",
            "肖",
            "瘙",
            "誒",
            "泊",
            "听",
            "噢",
        }
        self.non_monophonic = {"似", "攢"}

        self.monophonic_chars_dict = msgpack.load(
            open("./data/g2pw_model/monophonic_chars_dict.msgpack", "rb")
        )
        self.labels = msgpack.load(open("./data/g2pw_model/labels.msgpack", "rb"))
        self.char2phonemes = msgpack.load(
            open("./data/g2pw_model/char2phonemes.msgpack", "rb")
        )

        self.chars = sorted(list(self.char2phonemes.keys()))

        self.polyphonic_chars_new = set(self.chars)
        for char in self.non_polyphonic:
            if char in self.polyphonic_chars_new:
                self.polyphonic_chars_new.remove(char)
        for char in self.non_monophonic:
            if char in self.monophonic_chars_dict:
                self.monophonic_chars_dict.pop(char)

        self.pos_tags = ["UNK", "A", "C", "D", "I", "N", "P", "T", "V", "DE", "SHI"]

        self.bopomofo_convert_dict = msgpack.load(
            open("./data/g2pw_model/bopomofo_to_pinyin_wo_tune_dict.msgpack", "rb")
        )
        self.style_convert_func = {
            "bopomofo": lambda x: x,
            "pinyin": self._convert_bopomofo_to_pinyin,
        }[style]

        self.char_bopomofo_dict = msgpack.load(
            open("./data/g2pw_model/char_bopomofo_dict.msgpack", "rb")
        )

        if self.enable_opencc:
            self.cc = OpenCC("s2tw")

    def _convert_bopomofo_to_pinyin(self, bopomofo: str) -> str:
        tone = bopomofo[-1]
        assert tone in "12345"
        component = self.bopomofo_convert_dict.get(bopomofo[:-1])
        if component:
            return component + tone
        else:
            print(f'Warning: "{bopomofo}" cannot convert to pinyin')
            return None

    def __call__(self, sentences: List[str]) -> List[List[str]]:
        if isinstance(sentences, str):
            sentences = [sentences]

        if self.enable_opencc:
            translated_sentences = []
            for sent in sentences:
                translated_sent = self.cc.convert(sent)
                assert len(translated_sent) == len(sent)
                translated_sentences.append(translated_sent)
            sentences = translated_sentences

        texts, query_ids, sent_ids, partial_results = self._prepare_data(
            sentences=sentences
        )
        if len(texts) == 0:
            # sentences no polyphonic words
            return partial_results

        onnx_input = prepare_onnx_input(
            tokenizer=self.tokenizer,
            labels=self.labels,
            char2phonemes=self.char2phonemes,
            chars=self.chars,
            texts=texts,
            query_ids=query_ids,
            use_mask=True,
            window_size=None,
        )

        preds, confidences = predict(
            session=self.session_g2pW, onnx_input=onnx_input, labels=self.labels
        )

        results = partial_results
        for sent_id, query_id, pred in zip(sent_ids, query_ids, preds):
            results[sent_id][query_id] = self.style_convert_func(pred)

        return results

    def _prepare_data(
        self, sentences: List[str]
    ) -> Tuple[List[str], List[int], List[int], List[List[str]]]:
        texts, query_ids, sent_ids, partial_results = [], [], [], []
        for sent_id, sent in enumerate(sentences):
            # pypinyin works well for Simplified Chinese than Traditional Chinese
            sent_s = tranditional_to_simplified(sent)
            pypinyin_result = pinyin(
                sent_s, neutral_tone_with_five=True, style=Style.TONE3
            )
            partial_result = [None] * len(sent)
            for i, char in enumerate(sent):
                if char in self.polyphonic_chars_new:
                    texts.append(sent)
                    query_ids.append(i)
                    sent_ids.append(sent_id)
                elif char in self.monophonic_chars_dict:
                    partial_result[i] = self.style_convert_func(
                        self.monophonic_chars_dict[char]
                    )
                elif char in self.char_bopomofo_dict:
                    partial_result[i] = pypinyin_result[i][0]
                    # partial_result[i] =  self.style_convert_func(self.char_bopomofo_dict[char][0])
                else:
                    partial_result[i] = pypinyin_result[i][0]

            partial_results.append(partial_result)
        return texts, query_ids, sent_ids, partial_results
