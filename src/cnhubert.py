from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

from torch import nn


class CNHubert(nn.Module):
    def __init__(self):
        super().__init__()
        base_path = "./data/chinese-hubert-base"
        self.model = HubertModel.from_pretrained(base_path, local_files_only=True)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            base_path, local_files_only=True
        )

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=16000
        ).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats
