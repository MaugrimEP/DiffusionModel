import torch
import torch.nn as nn
import torch.nn.functional as F

from conf.model import LossParams


def get_loss(loss_params: LossParams):
    return DiffusionLoss(loss_params)


class DiffusionLoss(nn.Module):
    def __init__(self, params: LossParams):
        super().__init__()
        self.params = params

    def forward(self, batch, batch_mixed, noise_infos, batch_recon, t):
        pred   = batch_recon
        target = noise_infos if self.params.predict_noise else batch

        if self.params.loss_name == 'l1':
            loss = F.l1_loss(input=pred, target=target, reduction=self.params.reduction)
        elif self.params.loss_name == 'l2':
            loss = F.mse_loss(input=pred, target=target, reduction=self.params.reduction)
        elif self.params.loss_name == 'huber':
            loss = F.smooth_l1_loss(input=pred, target=target, reduction=self.params.reduction)
        else:
            raise ValueError(f'Not implemented loss: {self.params.loss_name}')

        return loss
