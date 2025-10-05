# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
from typing import Optional, List

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .sample import (
    sample,
)
from .embedding import SinePositionalEmbedding, TokenEmbedding


class T2SMLP(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class TS2Cache:
    def __init__(
        self,
        bsz: int,
        max_len: int,
        model_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.k_cache = torch.zeros(
            (bsz, max_len, model_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (bsz, max_len, model_dim),
            dtype=dtype,
            device=device,
        )

    def update(self, k: torch.Tensor, v: torch.Tensor, idx: int):
        pos = torch.tensor([idx], device=k.device)
        self.k_cache.index_copy_(1, pos, k)
        self.v_cache.index_copy_(1, pos, v)
        return self.k_cache, self.v_cache


class T2SBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.mlp = T2SMLP(
            hidden_dim,
            hidden_dim * 4,
        )
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(
            hidden_dim,
            hidden_dim,
        )
        self.norm_1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.norm_2 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)

        nn.init.constant_(self.qkv.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def process_prompt(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        cache: TS2Cache,
    ):
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        batch_size = q.size(0)
        q_len = q.size(1)
        kv_len = k.size(1)
        pos = torch.arange(0, x.size(1), device=k.device)
        cache.k_cache.index_copy_(1, pos, k)
        cache.v_cache.index_copy_(1, pos, v)

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = self.out_proj(attn)

        x = x + attn
        x = self.norm_1(x)
        x = x + self.mlp(x)
        x = self.norm_2(x)
        return x

    def decode_next_token(
        self,
        x: torch.Tensor,
        cache: TS2Cache,
        idx: int,
    ):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        k, v = cache.update(k, v, idx + 1)

        batch_size = q.size(0)
        q_len = q.size(1)

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = (
            k[:, : idx + 1, :]
            .view(batch_size, idx + 1, self.num_heads, -1)
            .transpose(1, 2)
        )
        v = (
            v[:, : idx + 1, :]
            .view(batch_size, idx + 1, self.num_heads, -1)
            .transpose(1, 2)
        )

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = self.out_proj(attn)

        x = x + attn
        x = self.norm_1(
            x,
        )
        x = x + self.mlp(x)
        x = self.norm_2(
            x,
        )
        return x


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.eos = config["model"]["EOS"]
        self.hz_x_max_sec = 50 * config["data"]["max_sec"]
        assert self.eos == self.vocab_size - 1
        # should be same as num of kmeans bin
        # assert self.eos == 1024
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim,
            self.phoneme_vocab_size,
            self.p_dropout,
        )
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim,
            self.vocab_size,
            self.p_dropout,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.blocks = nn.ModuleList(
            [
                T2SBlock(
                    self.num_head,
                    self.model_dim,
                )
                for _ in range(self.num_layers)
            ]
        )

    @torch.compile(fullgraph=True, dynamic=True)
    def process_prompt(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        cache: List[TS2Cache],
    ):
        for i, block in enumerate(self.blocks):
            x = block.process_prompt(x, attn_mask, cache[i])
        return self.ar_predict_layer(x[:, -1])

    @torch.compile(fullgraph=True)
    def decode_next_token(
        self,
        x: torch.Tensor,
        cache: List[TS2Cache],
        idx: int,
    ):
        for i, block in enumerate(self.blocks):
            x = block.decode_next_token(
                x,
                cache[i],
                idx,
            )
        return self.ar_predict_layer(x[:, -1])

    def should_stop(
        self,
        y: torch.Tensor,
        logits: torch.Tensor,
        prefix_len: int,
        samples: torch.Tensor,
    ):
        if self.hz_x_max_sec != -1 and (y.shape[1] - prefix_len) > self.hz_x_max_sec:
            print("use early stop num:", self.hz_x_max_sec)
            return True

        if torch.argmax(logits, dim=-1)[0] == self.eos or samples[0, 0] == self.eos:
            return True
        return False

    def postprocess(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        top_k: int,
        top_p: int,
        temperature: float,
        repetition_penalty: float,
        first_10: bool,
        prefix_len: int,
    ):
        if first_10:
            logits = logits[:, :-1]

        samples = sample(
            logits,
            y,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )[0]
        y = torch.concat([y, samples], dim=1)
        return y, samples, self.should_stop(y, logits, prefix_len, samples)

    def prepare_attn_mask(self, x_len: int, y_len: int, x: torch.Tensor):
        src_len = x_len + y_len
        bsz = x.size(0)
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        x_attn_mask_pad = F.pad(
            x_attn_mask,
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = (
            torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
            .unsqueeze(0)
            .expand(bsz * self.num_head, -1, -1)
            .view(bsz, self.num_head, src_len, src_len)
            .to(device=x.device, dtype=torch.bool)
        )
        return xy_attn_mask

    def encode_input(
        self, x: torch.LongTensor, bert_feature: torch.LongTensor, max_len=2048
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)
        return (
            x,
            x.size(1),
            [
                TS2Cache(x.size(0), max_len, self.model_dim, x.device, x.dtype)
                for _ in range(len(self.blocks))
            ],
        )

    def infer_panel(
        self,
        x: torch.LongTensor,
        y: Optional[torch.LongTensor],
        bert_feature: torch.LongTensor,
        top_k: int = -100,
        top_p: int = 100,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
    ):
        x, x_len, cache = self.encode_input(x, bert_feature)

        if y is None:
            y_len = 0
            prefix_len = 0
            y = torch.zeros(x.size(0), 0, dtype=torch.int, device=x.device)
        else:
            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.size(1)
            prefix_len = y.size(1)
            x = torch.concat([x, self.ar_audio_position(y_emb)], dim=1)
        pbar = tqdm(range(1500))
        for idx in pbar:
            if idx == 0:
                x = self.process_prompt(
                    x, self.prepare_attn_mask(x_len, y_len, x), cache
                )
                kv_cache_len = x.size(1)
            else:
                x = self.decode_next_token(x, cache, kv_cache_len)
                kv_cache_len += 1

            y, samples, stop = self.postprocess(
                x,
                y,
                top_k,
                top_p,
                temperature,
                repetition_penalty,
                idx < 11,
                prefix_len,
            )

            if stop:
                if y.size(1) == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    pbar.set_description("bad zero prediction")
                pbar.set_postfix({"T2S Decoding EOS": f"[{prefix_len} -> {y.size(1)}]"})
                break

            y_emb = self.ar_audio_embedding(y[:, -1:])
            x = self.ar_audio_position.update(y_emb, y_len + idx)

        return y[:, :-1], 0 if prefix_len == 0 else idx
