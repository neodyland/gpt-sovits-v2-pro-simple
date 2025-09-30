import torch
from torch import nn
from .attentions import MultiHeadAttention


class MRTE(nn.Module):
    def __init__(
        self,
        content_enc_channels=192,
        hidden_size=512,
        out_channels=192,
        kernel_size=5,
        n_heads=4,
        ge_layer=2,
    ):
        super(MRTE, self).__init__()
        self.cross_attention = MultiHeadAttention(hidden_size, hidden_size, n_heads)
        self.c_pre = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.text_pre = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.c_post = nn.Conv1d(hidden_size, out_channels, 1)

    def forward(self, ssl_enc, ssl_mask, text, text_mask, ge, test=None):
        if ge is None:
            ge = 0
        attn_mask = text_mask.unsqueeze(2) * ssl_mask.unsqueeze(-1)

        ssl_enc = self.c_pre(ssl_enc * ssl_mask)
        text_enc = self.text_pre(text * text_mask)
        if test is not None:
            if test == 0:
                x = (
                    self.cross_attention(
                        ssl_enc * ssl_mask, text_enc * text_mask, attn_mask
                    )
                    + ssl_enc
                    + ge
                )
            elif test == 1:
                x = ssl_enc + ge
            elif test == 2:
                x = (
                    self.cross_attention(
                        ssl_enc * 0 * ssl_mask, text_enc * text_mask, attn_mask
                    )
                    + ge
                )
            else:
                raise ValueError("test should be 0,1,2")
        else:
            x = (
                self.cross_attention(
                    ssl_enc * ssl_mask, text_enc * text_mask, attn_mask
                )
                + ssl_enc
                + ge
            )
        x = self.c_post(x * ssl_mask)
        return x


if __name__ == "__main__":
    content_enc = torch.randn(3, 192, 100)
    content_mask = torch.ones(3, 1, 100)
    ref_mel = torch.randn(3, 128, 30)
    ref_mask = torch.ones(3, 1, 30)
    model = MRTE()
    out = model(content_enc, content_mask, ref_mel, ref_mask)
    print(out.shape)
