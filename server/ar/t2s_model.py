# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .sample import (
    sample,
)
from .modules.embedding import SinePositionalEmbedding, TokenEmbedding
from .modules.transformer import LayerNorm


class T2SMLP(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super(T2SMLP, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class T2SBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
    ):
        super(T2SBlock, self).__init__()
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
        self.norm_1 = LayerNorm(hidden_dim, eps=1e-5)
        self.norm_2 = LayerNorm(hidden_dim, eps=1e-5)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)

        nn.init.constant_(self.qkv.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def to_mask(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ):
        if padding_mask is None:
            return x

        if padding_mask.dtype == torch.bool:
            return x.masked_fill(padding_mask, 0)
        else:
            return x * padding_mask

    def process_prompt(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        q, k, v = self.qkv(self.to_mask(x, padding_mask)).chunk(3, dim=-1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k.shape[1]

        q = self.to_mask(q, padding_mask)
        k_cache = self.to_mask(k, padding_mask)
        v_cache = self.to_mask(v, padding_mask)

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = self.out_proj(attn)

        x = x + attn
        x = self.norm_1(x)
        x = x + self.mlp(x)
        x = self.norm_2(x)
        return x, k_cache, v_cache

    def decode_next_token(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ):
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        k_cache = torch.cat([k_cache.clone(), k], dim=1)
        v_cache = torch.cat([v_cache.clone(), v], dim=1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k_cache.shape[1]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v, (~attn_mask) if attn_mask is not None else None
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
        return x, k_cache, v_cache


class T2STransformer(nn.Module):
    def __init__(self, num_head: int, model_dim: int, num_layers: int):
        super(T2STransformer, self).__init__()
        self.blocks = nn.ModuleList(
            [
                T2SBlock(
                    num_head,
                    model_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def process_prompt(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        k_cache: List[torch.Tensor] = []
        v_cache: List[torch.Tensor] = []
        for block in self.blocks:
            x, k_cache_, v_cache_ = block.process_prompt(x, attn_mask, padding_mask)
            k_cache.append(k_cache_)
            v_cache.append(v_cache_)
        return x, k_cache, v_cache

    def decode_next_token(
        self,
        x: torch.Tensor,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        attn_mask: torch.Tensor = None,
    ):
        for i, block in enumerate(self.blocks):
            x, k_cache[i], v_cache[i] = block.decode_next_token(
                x, k_cache[i], v_cache[i], attn_mask
            )
        return x, k_cache, v_cache


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        # should be same as num of kmeans bin
        # assert self.EOS == 1024
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
        self.h = T2STransformer(self.num_head, self.model_dim, self.num_layers)

    def infer_panel_naive(
        self,
        x: torch.LongTensor,  #####全部文本token
        prompts: torch.LongTensor,  ####参考音频token
        bert_feature: torch.LongTensor,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts

        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        # print(1111111,self.num_layers)

        k_cache = None
        v_cache = None
        ###################  first step ##########################
        if y is not None:
            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.shape[1]
            prefix_len = y.shape[1]
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            ref_free = False
        else:
            y_emb = None
            y_len = 0
            prefix_len = 0
            y_pos = None
            xy_pos = x
            y = torch.zeros(x.shape[0], 0, dtype=torch.int, device=x.device)
            ref_free = True

        bsz = x.shape[0]
        src_len = x_len + y_len
        x_attn_mask_pad = F.pad(
            x_attn_mask,
            (0, y_len),  ###xx的纯0扩展到xx纯0+xy纯1，(x,x+y)
            value=True,
        )
        y_attn_mask = F.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
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

        for idx in tqdm(range(1500)):
            if xy_attn_mask is not None:
                xy_dec, k_cache, v_cache = self.h.process_prompt(
                    xy_pos, xy_attn_mask, None
                )
            else:
                xy_dec, k_cache, v_cache = self.h.decode_next_token(
                    xy_pos, k_cache, v_cache
                )

            logits = self.ar_predict_layer(xy_dec[:, -1])

            if idx == 0:
                xy_attn_mask = None
            if idx < 11:  ###至少预测出10个token不然不给停止（0.4s）
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

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break

            ####################### update next step ###################################
            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = (
                y_emb * self.ar_audio_position.x_scale
                + self.ar_audio_position.alpha
                * self.ar_audio_position.pe[:, y_len + idx].to(
                    dtype=y_emb.dtype, device=y_emb.device
                )
            )

        if ref_free:
            return y[:, :-1], 0
        return y[:, :-1], idx

    def infer_panel(
        self,
        x: torch.LongTensor,  #####全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  ####参考音频token
        bert_feature: torch.LongTensor,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
    ):
        return self.infer_panel_naive(
            x,
            prompts,
            bert_feature,
            top_k,
            top_p,
            early_stop_num,
            temperature,
            repetition_penalty,
        )
