"""The temporal convolutional network, and why it is this small.

A stack of dilated 1-D convolutions over the feature channels, one prediction
per sample. The choice of architecture matters less here than two things about
how it is sized and padded, both of which would produce a plausible wrong number
if got wrong.

## The receptive field is reported in seconds

A dilation stack sees a fixed number of *samples*. What decides whether it can do
the job is how many *seconds* that is, against the thing being detected: the top
of a backswing is the instant the hands reverse, and an output that can only see
0.2 s either side cannot tell the top from the momentary slowing that happens
whenever a golfer changes direction. Six layers at 60 Hz is 2.1 s, which contains
a whole swing. `TrainingReport.receptive_field_s` carries it so that a poor score
can be read as "the model could not see enough" rather than left as a mystery.

## It is not causal, and says so

Each output sees equally far forward and back. That is correct for offline
analysis of a recorded clip -- the frames after an instant are as available as
the frames before it, and the top is *defined* by what happens next -- and it
means every number this model produces is unavailable to a live detector. A live
version would be a different model with a different score, and
`TCNConfig.causal` exists so that nobody has to reverse-engineer which one
produced a checkpoint.

## It is deliberately too small to memorise the set

Around twenty thousand parameters. A network able to memorise forty clips will,
and on a set that size memorisation is indistinguishable from learning until it
meets a golfer it has not seen -- which is precisely what the held-out split
exists to arrange, and precisely what a project without enough players cannot
arrange. The capacity is set against the data that exists.
"""

from __future__ import annotations

import torch
from torch import nn

from analyzer.contracts.ml import FRAME_CLASSES, TCNConfig


class ResidualBlock(nn.Module):
    """One dilated convolution, normalised, with a skip connection.

    `GroupNorm` with one group rather than `BatchNorm1d`. Batch statistics on a
    training set of tens of clips are noise, and worse, they make a clip's
    prediction depend on which other clips happened to share its batch -- which
    turns an evaluation into a statement about batching. GroupNorm normalises
    each clip against itself.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        *,
        causal: bool,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.pad = dilation * (kernel_size - 1)
        # Both forms preserve length, by putting the same total padding in
        # different places: half each side leaves an output aligned with its
        # input, and all of it on the left leaves an output that has seen only
        # the past. An odd kernel is required for the first to be exact, which
        # `TCNConfig.kernel_size` does not enforce and `SwingTCN` checks.
        padding = 0 if causal else self.pad // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.norm = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.causal:
            x = nn.functional.pad(x, (self.pad, 0))
        out = self.conv(x)
        out = self.dropout(self.activation(self.norm(out)))
        return residual + out


class SwingTCN(nn.Module):
    """Per-sample swing-phase classification from the feature channels."""

    def __init__(self, in_channels: int, config: TCNConfig | None = None) -> None:
        super().__init__()
        self.config = config or TCNConfig()
        if not self.config.causal and self.config.kernel_size % 2 == 0:
            raise ValueError(
                f"A non-causal stack needs an odd kernel to stay aligned with its input; "
                f"{self.config.kernel_size} would shift every prediction by half a tap, "
                "which reads downstream as a model with a consistent timing bias."
            )
        hidden = self.config.hidden_channels
        self.stem = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    hidden,
                    self.config.kernel_size,
                    dilation=2**layer,
                    dropout=self.config.dropout,
                    causal=self.config.causal,
                )
                for layer in range(self.config.layers)
            ]
        )
        self.head = nn.Conv1d(hidden, len(FRAME_CLASSES), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, channels, samples) in, (batch, classes, samples) out.

        Length is preserved, so an output sample corresponds to the input sample
        at the same index and nothing above has to track an offset. An offset
        that nobody tracks is a whole-clip shift in every event a model reports,
        and it looks exactly like a model with a consistent bias.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected (batch, channels, samples); got shape {tuple(x.shape)}.")
        out = self.stem(x)
        for block in self.blocks:
            out = block(out)
        return self.head(out)

    @property
    def receptive_field(self) -> int:
        return self.config.receptive_field

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
