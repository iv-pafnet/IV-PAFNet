"""ISA and MSFA building blocks used by IV-PAFNet."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfraredPurifier(nn.Module):
    """Learnable high-pass filtering and thermal-response gating (ISA stage 1)."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.shallow = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.highpass = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=True
        )
        self.gate = nn.Conv2d(channels * 2, channels, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.shallow.weight)
        nn.init.zeros_(self.shallow.bias)
        with torch.no_grad():
            for channel in range(self.channels):
                self.shallow.weight[channel, channel, 1, 1] = 1.0

        laplacian = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        )
        with torch.no_grad():
            self.highpass.weight.copy_(
                laplacian.view(1, 1, 3, 3).repeat(self.channels, 1, 1, 1)
            )
            self.highpass.bias.zero_()

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, infrared: torch.Tensor) -> torch.Tensor:
        f0 = self.shallow(infrared)
        fhp = self.highpass(f0)
        gir = torch.sigmoid(self.gate(torch.cat((f0, fhp), dim=1)))
        return gir * fhp + (1.0 - gir) * f0


class CrossModalAlignment(nn.Module):
    """Visible-anchored deformable resampling of infrared features (ISA stage 2).

    Multi-point bilinear sampling is implemented with a vectorized
    ``grid_sample``. Offsets and aggregation weights are predicted
    from concatenated RGB/IR features.  The initial state is close to identity.
    """

    def __init__(
        self,
        channels: int,
        num_points: int = 9,
        max_offset: float = 4.0,
    ) -> None:
        super().__init__()
        side = int(math.sqrt(num_points))
        if side * side != num_points or side % 2 == 0:
            raise ValueError("num_points must be an odd square, e.g. 1, 9, or 25")

        self.channels = channels
        self.num_points = num_points
        self.max_offset = float(max_offset)

        merged = channels * 2
        self.offset_features = nn.Sequential(
            nn.Conv2d(merged, merged, 3, padding=1, groups=merged, bias=False),
            nn.BatchNorm2d(merged),
            nn.SiLU(inplace=True),
            nn.Conv2d(merged, merged, 3, padding=1, groups=merged, bias=False),
            nn.BatchNorm2d(merged),
            nn.SiLU(inplace=True),
            nn.Conv2d(merged, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.offset_head = nn.Conv2d(channels, num_points * 2, 1)
        self.weight_head = nn.Conv2d(channels, num_points, 1)

        radius = side // 2
        regular = [
            (float(dx), float(dy))
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
        self.register_buffer(
            "regular_offsets", torch.tensor(regular), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.constant_(self.weight_head.bias, -4.0)
        with torch.no_grad():
            self.weight_head.bias[self.num_points // 2] = 4.0

    def forward(
        self, visible: torch.Tensor, infrared: torch.Tensor
    ) -> torch.Tensor:
        if visible.shape != infrared.shape:
            raise ValueError(
                f"RGB and IR feature shapes must match, got "
                f"{tuple(visible.shape)} and {tuple(infrared.shape)}"
            )

        batch, channels, height, width = infrared.shape
        features = self.offset_features(torch.cat((visible, infrared), dim=1))

        offsets = self.offset_head(features)
        offsets = offsets.view(batch, self.num_points, 2, height, width)
        offsets = torch.tanh(offsets) * self.max_offset
        offsets = offsets.permute(0, 1, 3, 4, 2)

        weights = torch.softmax(self.weight_head(features), dim=1).unsqueeze(2)

        y = torch.linspace(
            -1.0, 1.0, height, device=infrared.device, dtype=infrared.dtype
        )
        x = torch.linspace(
            -1.0, 1.0, width, device=infrared.device, dtype=infrared.dtype
        )
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.view(1, 1, height, width, 2)

        x_scale = 2.0 / max(width - 1, 1)
        y_scale = 2.0 / max(height - 1, 1)
        scale = infrared.new_tensor((x_scale, y_scale)).view(1, 1, 1, 1, 2)
        regular = self.regular_offsets.to(
            device=infrared.device, dtype=infrared.dtype
        ).view(1, self.num_points, 1, 1, 2)

        sampling_grid = base_grid + (regular + offsets) * scale
        sampling_grid = sampling_grid.reshape(
            batch * self.num_points, height, width, 2
        )
        repeated_ir = infrared[:, None].expand(
            batch, self.num_points, channels, height, width
        )
        repeated_ir = repeated_ir.reshape(
            batch * self.num_points, channels, height, width
        )
        sampled = F.grid_sample(
            repeated_ir,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.view(
            batch, self.num_points, channels, height, width
        )
        return (sampled * weights).sum(dim=1)


class MSFA(nn.Module):
    """Channel-then-spatial competitive RGB/IR fusion."""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        use_channel: bool = True,
        use_spatial: bool = True,
    ) -> None:
        super().__init__()
        self.use_channel = bool(use_channel)
        self.use_spatial = bool(use_spatial)
        hidden = max(channels // reduction, 4)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.rgb_fc = nn.Conv2d(hidden, channels, 1, bias=True)
        self.ir_fc = nn.Conv2d(hidden, channels, 1, bias=True)
        self.spatial = nn.Conv2d(channels * 2, 2, 7, padding=3, bias=True)

        nn.init.zeros_(self.rgb_fc.weight)
        nn.init.zeros_(self.rgb_fc.bias)
        nn.init.zeros_(self.ir_fc.weight)
        nn.init.zeros_(self.ir_fc.bias)
        nn.init.zeros_(self.spatial.weight)
        nn.init.zeros_(self.spatial.bias)

    def forward(
        self, visible: torch.Tensor, aligned_infrared: torch.Tensor
    ) -> torch.Tensor:
        if self.use_channel:
            descriptor = F.adaptive_avg_pool2d(
                visible + aligned_infrared, output_size=1
            )
            descriptor = self.shared_mlp(descriptor)
            channel_logits = torch.stack(
                (self.rgb_fc(descriptor), self.ir_fc(descriptor)), dim=1
            )
            channel_weights = torch.softmax(channel_logits, dim=1)
            visible_hat = channel_weights[:, 0] * visible
            infrared_hat = channel_weights[:, 1] * aligned_infrared
        else:
            visible_hat = visible
            infrared_hat = aligned_infrared

        if self.use_spatial:
            spatial_logits = self.spatial(
                torch.cat((visible_hat, infrared_hat), dim=1)
            )
            spatial_weights = torch.softmax(spatial_logits, dim=1)
            return (
                spatial_weights[:, 0:1] * visible_hat
                + spatial_weights[:, 1:2] * infrared_hat
            )

        if self.use_channel:
            return visible_hat + infrared_hat
        return 0.5 * (visible_hat + infrared_hat)
