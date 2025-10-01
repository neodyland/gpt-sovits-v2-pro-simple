import torch
from torch.nn import functional as F

hann_window = {}


def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center=False,
):
    if torch.min(y) < -1.2:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.2:
        print("max value is ", torch.max(y))

    global hann_window
    key = "%s-%s-%s-%s-%s-%s" % (
        str(y.dtype),
        str(y.device),
        n_fft,
        sampling_rate,
        hop_size,
        win_size,
    )
    if key not in hann_window:
        hann_window[key] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = F.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = (spec.pow(2).sum(-1) + 1e-8).sqrt()
    return spec
