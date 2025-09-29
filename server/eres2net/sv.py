import torch
from .ERes2NetV2 import ERes2NetV2
from .kaldi import fbank
import safetensors.torch as st


class SV:
    def __init__(self, device, dtype):
        embedding_model = (
            ERes2NetV2(baseWidth=24, scale=4, expansion=4)
            .eval()
            .to(device=device, dtype=dtype)
        )
        embedding_model.load_state_dict(
            st.load_file(
                "./data/sv/pretrained_eres2netv2w24s4ep4.safetensors",
                device=device,
            )
        )
        self.dtype = dtype
        self.embedding_model = embedding_model

    def compute_embedding3(self, wav):
        with torch.no_grad():
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
