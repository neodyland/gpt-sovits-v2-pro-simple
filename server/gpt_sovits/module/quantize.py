# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Residual vector quantizer implementation."""

import typing as tp

import torch
from torch import nn

from .core_vq import ResidualVectorQuantization


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
    ):
        super().__init__()
        self.n_q = n_q
        self.vq = ResidualVectorQuantization(
            dim=dimension,
            codebook_size=bins,
            num_quantizers=self.n_q,
        )

    def forward(
        self,
        x: torch.Tensor,
        n_q: tp.Optional[int] = None,
        layers: tp.Optional[list] = None,
    ):
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            n_q (int): Number of quantizer used to quantize. Default: All quantizers.
            layers (list): Layer that need to return quantized. Defalt: None.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated numbert quantizers and layer quantized required to return.
        """
        n_q = n_q if n_q else self.n_q
        if layers and max(layers) >= n_q:
            raise ValueError(
                f"Last layer index in layers: A {max(layers)}. Number of quantizers in RVQ: B {self.n_q}. A must less than B."
            )
        quantized, codes, quantized_list = self.vq(x, n_q=n_q, layers=layers)
        return quantized, codes, quantized_list

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        Args:
            codes (torch.Tensor): Input indices for each quantizer.
            st (int): Start to decode input codes from which layers. Default: 0.
        """
        quantized = self.vq.decode(codes)
        return quantized
