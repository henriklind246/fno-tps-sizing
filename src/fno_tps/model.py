from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from fno_tps.config import StudyConfig


Representation = Literal["summary", "temporal_global", "temporal_local"]
TRANSFER_SOURCE_REVISION = "143e506"


class SpectralConv2d(nn.Module):
    """2D Fourier convolution with real leaf parameters for DDP-safe state."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        spectral_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.spectral_dropout = float(spectral_dropout)
        scale = 1.0 / max(1, self.in_channels * self.out_channels)
        shape = (self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        self.weights1 = nn.Parameter(scale * torch.rand(*shape))
        self.weights2 = nn.Parameter(scale * torch.rand(*shape))

    @staticmethod
    def _complex_multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        nx, ny = inputs.shape[-2:]
        transformed = torch.fft.rfft2(inputs, dim=(-2, -1))
        nx_freq, ny_freq = transformed.shape[-2:]
        mx = min(self.modes1, nx_freq)
        my = min(self.modes2, ny_freq)
        if 2 * mx > nx_freq:
            raise ValueError(
                f"modes1={self.modes1} is too large for x frequency size {nx_freq}; "
                "positive and negative slices overlap."
            )
        weights1 = torch.view_as_complex(self.weights1)
        weights2 = torch.view_as_complex(self.weights2)
        output = torch.zeros(
            inputs.shape[0],
            self.out_channels,
            nx_freq,
            ny_freq,
            dtype=transformed.dtype,
            device=inputs.device,
        )
        output[:, :, :mx, :my] = self._complex_multiply(
            transformed[:, :, :mx, :my], weights1[:, :, :mx, :my]
        )
        output[:, :, -mx:, :my] = self._complex_multiply(
            transformed[:, :, -mx:, :my], weights2[:, :, :mx, :my]
        )
        if self.training and self.spectral_dropout > 0.0:
            keep_x = (torch.rand(mx, device=inputs.device) >= self.spectral_dropout).to(output.dtype)
            keep_y = (torch.rand(my, device=inputs.device) >= self.spectral_dropout).to(output.dtype)
            mask = keep_x[:, None] * keep_y[None, :]
            output[:, :, :mx, :my] *= mask
            output[:, :, -mx:, :my] *= mask
        return torch.fft.irfft2(output, s=(nx, ny), dim=(-2, -1))


class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.norm = nn.InstanceNorm2d(channels, affine=False, eps=eps)

    def forward(
        self,
        inputs: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        valid_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if valid_shape is None:
            normalized = self.norm(inputs)
        else:
            nx, ny = valid_shape
            valid = inputs[:, :, :nx, :ny]
            mean = valid.mean(dim=(-2, -1), keepdim=True)
            variance = valid.var(dim=(-2, -1), keepdim=True, unbiased=False)
            normalized = (inputs - mean) / torch.sqrt(variance + self.eps)
        return gamma[:, :, None, None] * normalized + beta[:, :, None, None]


class ConditioningMLP(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int, layers: int, width: int):
        super().__init__()
        self.layers = int(layers)
        self.width = int(width)
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden_dim, layers * 2 * width)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            bias = torch.zeros(layers, 2, width)
            bias[:, 0, :] = 1.0
            self.head.bias.copy_(bias.reshape(-1))

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        output = self.head(self.net(conditioning))
        return output.view(-1, self.layers, 2, self.width)


class TemporalForcingEncoder(nn.Module):
    """Shared row-wise Conv1D encoder transferred from the IHCP FNO."""

    def __init__(self, token_dim: int = 2, hidden: int = 64, embed_dim: int = 16):
        super().__init__()
        self.lift = nn.Linear(token_dim, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.projection = nn.Linear(2 * hidden, embed_dim)
        self.activation = nn.GELU()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.lift(tokens)).transpose(1, 2)
        hidden = self.activation(self.conv1(hidden))
        hidden = self.activation(self.conv2(hidden))
        pooled = torch.cat([hidden.mean(dim=-1), hidden.amax(dim=-1)], dim=-1)
        return self.activation(self.projection(pooled))


@dataclass(frozen=True)
class FNOConfig:
    representation: Representation
    modes1: int
    modes2: int
    width: int
    layers: int
    cond_hidden: int
    temporal_hidden: int
    forcing_embed_dim: int
    dropout: float = 0.0
    spectral_dropout: float = 0.0
    padding: int = 8
    padding_mode: str = "replicate"
    cin_exclude_padding: bool = True

    @property
    def spatial_channels(self) -> int:
        return 14 if self.representation == "summary" else 10

    @property
    def uses_temporal_encoder(self) -> bool:
        return self.representation != "summary"

    @classmethod
    def from_study(
        cls,
        study: StudyConfig,
        representation: Representation,
    ) -> "FNOConfig":
        training = study.training
        return cls(
            representation=representation,
            modes1=training.modes1,
            modes2=training.modes2,
            width=training.width,
            layers=training.layers,
            cond_hidden=training.cond_hidden,
            temporal_hidden=training.temporal_hidden,
            forcing_embed_dim=training.forcing_embed_dim,
            dropout=training.dropout,
        )


class TPSFNO(nn.Module):
    """Time-conditioned FNO for summary, global, and local forcing representations."""

    def __init__(self, config: FNOConfig):
        super().__init__()
        if config.representation not in ("summary", "temporal_global", "temporal_local"):
            raise ValueError(f"Unknown representation {config.representation!r}.")
        if config.padding_mode not in ("zeros", "replicate", "reflect"):
            raise ValueError("padding_mode must be zeros, replicate, or reflect.")
        self.config = config
        self.representation = config.representation
        self.spatial_channels = config.spatial_channels
        self.width = config.width
        self.padding = int(config.padding)
        self.padding_mode = config.padding_mode
        self.cin_exclude_padding = bool(config.cin_exclude_padding)

        temporal_extra = config.forcing_embed_dim if config.uses_temporal_encoder else 0
        self.lift = nn.Linear(self.spatial_channels + temporal_extra, config.width)
        if config.uses_temporal_encoder:
            self.temporal_encoder = TemporalForcingEncoder(
                token_dim=2,
                hidden=config.temporal_hidden,
                embed_dim=config.forcing_embed_dim,
            )
        cond_dim = 2 + temporal_extra
        self.conditioning = ConditioningMLP(
            cond_dim=cond_dim,
            hidden_dim=config.cond_hidden,
            layers=config.layers,
            width=config.width,
        )
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(
                config.width,
                config.width,
                config.modes1,
                config.modes2,
                config.spectral_dropout,
            )
            for _ in range(config.layers)
        ])
        self.local_layers = nn.ModuleList([
            nn.Conv2d(config.width, config.width, kernel_size=1)
            for _ in range(config.layers)
        ])
        self.norm_layers = nn.ModuleList([
            ConditionalInstanceNorm2d(config.width)
            for _ in range(config.layers)
        ])
        self.projection = nn.Linear(config.width, 128)
        self.output = nn.Linear(128, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def encode_forcing(self, forcing_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if forcing_seq.ndim != 4 or forcing_seq.shape[-1] != 2:
            raise ValueError(
                "Temporal forcing must have shape (B, Ny, M, 2); "
                f"got {tuple(forcing_seq.shape)}."
            )
        batch, ny, samples, token_dim = forcing_seq.shape
        rows = forcing_seq.reshape(batch * ny, samples, token_dim)
        local = self.temporal_encoder(rows).reshape(batch, ny, -1)
        return local, local.mean(dim=1)

    def forward(
        self,
        spatial: torch.Tensor,
        cond_static: torch.Tensor,
        forcing_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if spatial.shape[-1] != self.spatial_channels:
            raise ValueError(
                f"Expected {self.spatial_channels} spatial channels, "
                f"got {spatial.shape[-1]}."
            )
        if cond_static.ndim != 2 or cond_static.shape[-1] != 2:
            raise ValueError("cond_static must have shape (B, 2).")

        if self.config.uses_temporal_encoder:
            if forcing_seq is None:
                raise ValueError("Temporal representations require forcing_seq.")
            local_embedding, global_embedding = self.encode_forcing(forcing_seq)
            nx, ny = spatial.shape[1:3]
            if local_embedding.shape[1] != ny:
                raise ValueError("forcing_seq Ny does not match the spatial tensor.")
            if self.representation == "temporal_global":
                injection = global_embedding[:, None, None, :].expand(-1, nx, ny, -1)
            else:
                injection = local_embedding[:, None, :, :].expand(-1, nx, -1, -1)
            lifted_input = torch.cat([spatial, injection], dim=-1)
            conditioning = torch.cat([cond_static, global_embedding], dim=-1)
        else:
            lifted_input = spatial
            conditioning = cond_static

        affine = self.conditioning(conditioning)
        hidden = self.lift(lifted_input).permute(0, 3, 1, 2)
        nx, ny = hidden.shape[-2:]
        if self.padding_mode == "zeros":
            hidden = F.pad(hidden, (0, self.padding, 0, self.padding))
        else:
            hidden = F.pad(
                hidden,
                (0, self.padding, 0, self.padding),
                mode=self.padding_mode,
            )
        for index, (spectral, local, norm) in enumerate(
            zip(self.spectral_layers, self.local_layers, self.norm_layers)
        ):
            hidden = spectral(hidden) + local(hidden)
            valid_shape = (nx, ny) if self.cin_exclude_padding else None
            hidden = norm(
                hidden,
                affine[:, index, 0, :],
                affine[:, index, 1, :],
                valid_shape=valid_shape,
            )
            hidden = self.dropout(self.activation(hidden))
        hidden = hidden[:, :, :nx, :ny].permute(0, 2, 3, 1)
        hidden = self.dropout(self.activation(self.projection(hidden)))
        return self.output(hidden)


def build_model(study: StudyConfig, representation: Representation) -> TPSFNO:
    return TPSFNO(FNOConfig.from_study(study, representation))
