import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from conf.model import ModelParams, BackboneParams
from src.Backbones.Unet_cold_diffusion import Unet_Cold


def get_model(params: BackboneParams) -> nn.Module:
    if params.name == 'unet_cold':
        model = Unet_Cold(
            dim=params.dim_mlp,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=params.in_channels,
            with_time_emb=params.time_mlp,
            residual=True,
        )
        return model
    else:
        model = smp.create_model(
            arch           =params.arch,
            encoder_name   =params.encoder,
            in_channels    =params.in_channels + (1 if params.time_mlp else 0),
            encoder_weights=params.encoder_weights,
            classes        =params.classes,
        )
        return model
