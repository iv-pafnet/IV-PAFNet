"""IV-PAFNet dual-stream multimodal detector.

Extends RT-DETR into ``FusionDETR``: a paired RGB/IR detector that runs a shared
backbone on the purified infrared stream and on the visible stream, applies the
Infrared Signal purification and spatial Alignment module (ISA), and feeds the
aligned feature pyramid to the Hybrid Encoder for multi-scale fusion (MSFA)
before the detection head.
"""

import numpy as np
from src.core import register
from torch.nn import functional as F
import copy
import torch.nn as nn

# ISA: infrared purification and visible-guided spatial alignment.
from .iv_pafnet import InfraredPurifier, CrossModalAlignment

__all__ = ['FusionDETR', ]


@register
class FusionDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    # Backbone output channels for the R50VD pyramid levels used in this work.
    def __init__(
        self,
        backbone: nn.Module,
        encoder,
        decoder,
        multi_scale=None,
        backbone_out_channels=[512, 1024, 2048],
        use_purification=True,
        use_alignment=True,
    ):
        super().__init__()
        # Two backbone instances so the RGB and IR streams do not share weights.
        self.backbone_rgb = copy.deepcopy(backbone)
        self.backbone_ir = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.use_purification = use_purification
        self.use_alignment = use_alignment

        # 1. ISA stage 1: infrared purification, applied to the raw IR input,
        #    i.e. before the backbone.
        self.isa_purification = InfraredPurifier(channels=3)

        # 2. ISA stage 2: one independent alignment head per pyramid level.
        #    The ModuleList is built from the backbone output channels so that
        #    adding or removing a level stays consistent with the backbone.
        self.isa_alignments = nn.ModuleList([
            CrossModalAlignment(
                channels=c, num_points=9, max_offset=4.0
            ) for c in backbone_out_channels
        ])

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        # The dataset stacks the two modalities: channels 0-2 RGB, 3-5 IR.
        rgb = x[:, 0:3, ...]
        ir = x[:, 3:, ...]

        # --- ISA stage 1: purify the infrared stream -------------------------
        ir_purified = self.isa_purification(ir) if self.use_purification else ir

        # Extract multi-scale features (each backbone returns a list of tensors).
        feats_rgb = self.backbone_rgb(rgb)
        feats_ir = self.backbone_ir(ir_purified)

        # --- ISA stage 2: visible-guided spatial alignment -------------------
        if self.use_alignment:
            aligned_feats_ir = []
            for f_rgb, f_ir, align_module in zip(
                feats_rgb, feats_ir, self.isa_alignments
            ):
                f_ir_aligned = align_module(f_rgb, f_ir)
                aligned_feats_ir.append(f_ir_aligned)
        else:
            aligned_feats_ir = feats_ir

        # Multi-scale fusion (MSFA) is performed inside the hybrid encoder.
        x = self.encoder(feats_rgb, aligned_feats_ir)
        x = self.decoder(x, targets)
        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
