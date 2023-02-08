import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from torchvision.utils import make_grid

from typing import Tuple, Union, List

from conf.model import LoggingParams
from utils.utils import normalize_value_range, undersample_list


class LogStrategy:
    def log_train(self, plMod: pl.LightningModule, stage: str, prompt_img: str, prediction, input_to_model, batch, x0_hat, ts, batch_idx) -> None:
        pass

    def log_generate(self, plMod: pl.LightningModule, stage: str, prompt_img: str, batch_idx:int, batch):
        pass

    @staticmethod
    def already_logged(plMod: pl.LightningModule, batch_idx: int, batch_size: int) -> int:
        """
        Return how much of the sample should be logged
        """
        max_quant = plMod.params.logging.log_steps.max_quantity
        already_logged = batch_idx * batch_size
        remaining = max(0, max_quant - already_logged)
        return remaining


class LogBlender1(LogStrategy):
    def log_train(self, plMod: pl.LightningModule, stage: str, prompt_img: str, prediction, input_to_model, batch, x0_hat, ts, batch_idx) -> None:
        t_list = ts.tolist()
        images = torch.cat([batch, input_to_model, prediction, x0_hat], dim=-1)

        remaining = self.already_logged(plMod, batch_idx, batch_size=images.shape[0])
        if remaining <= 0:
            return
        images = images[:remaining]
        images = normalize_value_range(images, plMod.params.logging.value_range, clip=True)

        wandb_images = wandb.Image(make_grid(images, nrow=2), caption=f'data, D(data), model(D(data)) rinsed, x0_hat')
        plMod.logger.experiment.log({prompt_img: wandb_images})

        # for image, t in zip(images, ts):
        #     wandb_image = wandb.Image(image, caption=f'{t=}')
        #     plMod.logger.experiment.log({prompt_img: wandb_image})

    def log_generate(self, plMod: pl.LightningModule, stage: str, prompt_img: str, batch_idx: int, batch):
        remaining = self.already_logged(plMod, batch_idx, batch_size=batch.shape[0])

        with torch.no_grad():
            _, generated_data, generated_x0 = plMod.generate_samples(remaining, return_samples=True, return_pred_x0=True)

        generated_data, generated_x0 = zip(*undersample_list(
            list(zip(generated_data, generated_x0)),
            plMod.params.logging.time_step_in_process,
            plMod.params.logging.strategy,
            plMod.params.logging.quad_factor,
        ))

        full_tensor    = torch.cat(generated_data, dim=-1)
        full_tensor_x0 = torch.cat(generated_x0  , dim=-1)
        # batch_size, channels, height, width * time_steps

        full_tensor = torch.cat([full_tensor, full_tensor_x0], dim=-2)  # concat them in height dimension

        full_tensor = normalize_value_range(full_tensor, plMod.params.logging.value_range, clip=True)

        wandb_images = wandb.Image(make_grid(full_tensor, nrow=1), caption=f'{stage} xt and x0_hat')
        plMod.logger.experiment.log({prompt_img: wandb_images})


def get_log_strategy(params: LoggingParams) -> LogStrategy:
    name = params.name.lower()
    if name == 'blender1':
        return LogBlender1()
    else:
        raise NotImplementedError(f'Logging strategy {name} not implemented')
