from dataclasses import dataclass, field
from typing import Any, Tuple, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from conf.dataset import DatasetParams
from conf.model import ModelParams, BackboneParams, DiffusionParams_DDPM, DiffusionParams_DDIM
from src.Backbones.utils import get_model
from utils.LogStrategy import get_log_strategy
from utils.Losses import get_loss
from utils.Metrics import Metrics
from utils.utils import is_logging_time


def extract(a, t, x_shape):
    a = a.to(t.device)
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DiffusionVariables_DDPM(nn.Module):
    def __init__(self, params: DiffusionParams_DDPM):
        super().__init__()
        self.p = params

        self.n_steps_training = params.n_steps_training
        self.beta_min = params.beta_min
        self.beta_max = params.beta_max

        self.beta             = torch.linspace(self.beta_min, self.beta_max, self.n_steps_training)  # linearly increasing variance schedule
        self.alpha            = (1. - self.beta)
        self.alpha_bar        = self.alpha.cumprod(dim=0)
        self.one_minus_alpha  = self.beta
        self.sqrt_recip_alpha = (1. / self.alpha).sqrt()

        self.sqrt_alpha_bar            = self.alpha_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1. - self.alpha_bar).sqrt()
        self.eps_coef                  = self.one_minus_alpha / self.sqrt_one_minus_alphas_bar

        self.sigma2      = self.beta
        self.sigma2_sqrt = self.beta.sqrt()

        # for x0
        self.sqrt_recip_alpha_bar    = (1. / self.alpha_bar).sqrt()
        self.sqrt_recip_m1_alpha_bar = (1. / self.alpha_bar - 1.).sqrt()


class DiffusionVariables_DDIM(nn.Module):
    def __init__(self, params: DiffusionParams_DDIM):
        super().__init__()
        self.p = params

        self.skip_steps   = params.skip_steps
        self.repeat_noise = params.repeat_noise
        self.temperature  = params.temperature

        self.n_steps_training = params.n_steps_training
        self.n_steps_generation = params.n_steps_generation

        if params.ddim_discretize == 'uniform':
            c = self.n_steps_training // self.n_steps_generation
            self.time_steps = torch.arange(0, self.n_steps_training, c).long() + 1
        elif params.ddim_discretize == 'quad':
            self.time_steps = torch.linspace(0, np.sqrt(self.n_steps_training * params.time_step_factor), self.n_steps_generation).pow(2).long() + 1
        else:
            raise ValueError(f'{params.ddim_discretize=} is not an available discretization method.')

        self.beta_min = params.beta_min
        self.beta_max = params.beta_max

        self.beta           = torch.linspace(self.beta_min, self.beta_max, self.n_steps_training)  # linearly increasing variance schedule
        self.alpha          = (1. - self.beta)
        self.alpha_bar      = self.alpha.cumprod(dim=0)
        self.sqrt_alpha_bar = self.alpha_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1. - self.alpha_bar).sqrt()

        self.ddim_alpha           = self.alpha_bar[self.time_steps].clone()
        self.ddim_alpha_sqrt      = self.ddim_alpha.sqrt()
        self.ddim_alpha_prev      = torch.cat([self.alpha_bar[:1], self.alpha_bar[self.time_steps[:-1]]])
        self.ddim_alpha_prev_sqrt = self.ddim_alpha_prev.sqrt()
        self.ddim_sigma = (
            params.ddim_eta *
            (
                    (1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                    (1 - self.ddim_alpha / self.ddim_alpha_prev)
            ).sqrt()
        )

        self.ddim_sqrt_one_minus_alpha = (1 - self.ddim_alpha).sqrt()

        self.dir_xt_coef = (1. - self.ddim_alpha_prev - self.ddim_sigma.pow(2)).sqrt()


class DiffusionModel(pl.LightningModule):
    def __init__(
            self,
            params: ModelParams,
            params_data: DatasetParams,
    ):
        super().__init__()
        self.params = params
        self.params_data = params_data
        self.model = self._get_epsilons_model(params.backbone)

        if params.diffusion.diffusion_name == 'ddpm':
            self.diffusion_variables: DiffusionVariables_DDPM = DiffusionVariables_DDPM(params=params.diffusion)
            self.generate_samples = self.generate_samples_ddpm
            self.get_x0_hat = self.get_x0_hat_ddpm
        elif params.diffusion.diffusion_name == 'ddim':
            self.diffusion_variables: DiffusionVariables_DDIM = DiffusionVariables_DDIM(params=params.diffusion)
            self.generate_samples = self.generate_samples_ddim
            self.get_x0_hat = self.get_x0_hat_ddim

        self.loss = get_loss(params.loss)

        self.train_metrics = Metrics(params.metrics)
        self.valid_metrics = Metrics(params.metrics)
        self.test_metrics  = Metrics(params.metrics)

        self.log_strategy = get_log_strategy(params.logging)

    def _get_epsilons_model(self, params: BackboneParams) -> nn.Module:
        return get_model(params)

    def configure_optimizers(self):

        optimizers_params = self.params.optimizer

        model_params = [{'params': self.model.parameters()}]

        if optimizers_params.optimizer == 'adam':
            opti = optim.Adam(
                model_params,
                lr=optimizers_params.learning_rate,
                betas=optimizers_params.betas,
                weight_decay=optimizers_params.weight_decay,
            )
        elif optimizers_params.optimizer == 'adamw':
            opti = optim.AdamW(
                model_params,
                lr=optimizers_params.learning_rate,
                betas=optimizers_params.betas,
                weight_decay=optimizers_params.weight_decay,
            )
        elif optimizers_params.optimizer == 'sgd':
            opti = optim.SGD(
                model_params,
                lr=optimizers_params.learning_rate,
                momentum=optimizers_params.momentum,
                weight_decay=optimizers_params.weight_decay,
            )
        else:
            raise ValueError(f'{optimizers_params.optimizer=} is not an available optimizer.')

        scheduler = CosineAnnealingLR(
            optimizer=opti,
            T_max=self.get_number_iterations(),
            eta_min=0. if optimizers_params.use_scheduler else optimizers_params.learning_rate,
        )

        return {
                'optimizer': opti,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                },
            }

    def get_number_iterations(self) -> int:
        max_epochs = self.params.optimizer.max_epochs
        max_steps = self.params.optimizer.max_steps
        assert max_epochs == -1 or max_steps == -1, 'At least one of max_epochs and max_steps must be -1 to remove ambiguity.'

        if max_epochs != -1:
            estimated_stepping_batches = self.trainer.estimated_stepping_batches  # use the epochs
            t_max = estimated_stepping_batches
        elif max_steps != -1:
            t_max = max_steps
        else:
            raise ValueError('At least one of max_epochs and max_steps must be -1 to remove ambiguity.')

        return t_max

    def log_g(self, train_stage: str, logged: str, value: Any, **kwargs):
        is_train = 'train' in train_stage
        self.log(f'{train_stage}/{logged}', value, **kwargs, on_epoch=True, on_step=is_train)

    def training_step(self, batch, batch_idx, optimizer_idx: int = -1):
        train_stage = 'train'
        loss = self._step(batch, train_stage, batch_idx=batch_idx, optimizer_idx=optimizer_idx)
        self.log_g(train_stage, 'lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        return loss

    def _step(self, batch, stage: str, batch_idx: int, optimizer_idx: int = -1) -> torch.Tensor:
        batch_size = batch.shape[0]
        t = torch.randint(0, self.diffusion_variables.n_steps_training, (batch_size,), device=self.device, dtype=torch.long)

        batch_mixed, noise_infos = self.deteriorate(batch, t)

        batch_recon = self.model(batch_mixed, t)

        with torch.no_grad():
            x0_hat = self.get_x0_hat(current_model_input=batch_mixed, model_prediction=batch_recon, t=t)

        loss = self.loss(batch=batch, batch_mixed=batch_mixed, noise_infos=noise_infos, batch_recon=batch_recon, t=t)

        # Log
        self.log_g(stage, 'loss', loss.item())

        # Metrics
        metrics = self.get_metric_object().get_dict(batch_recon, batch_mixed, batch)
        for metric_name, value in metrics.items():
            self.log_g(stage, metric_name, value)

        # Log images
        if is_logging_time(self.params.logging.log_steps, current_epoch=self.current_epoch, batch_idx=batch_idx, stage=stage):
            self.log_strategy.log_train(
                stage     =stage,
                prompt_img=f'{stage}/images_steps',
                prediction=batch_recon, input_to_model=batch_mixed, batch=batch, ts=t,
                x0_hat    =x0_hat,
                plMod     =self,
                batch_idx =batch_idx,
            )
        if is_logging_time(self.params.logging.log_generate, current_epoch=self.current_epoch, batch_idx=batch_idx, stage=stage):
            self.log_strategy.log_generate(
                stage     =stage,
                prompt_img=f'{stage}/images_generate',
                plMod     =self,
                batch     =batch,
                batch_idx =batch_idx,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        self._step(batch, 'valid', batch_idx=batch_idx)

    def test_step(self, batch, batch_idx):
        self._step(batch, stage='test', batch_idx=batch_idx)

    # region deterioration functions
    def deteriorate(self, batch: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(batch)

        sqrt_alpha_bar = extract(self.diffusion_variables.sqrt_alpha_bar, t, batch.shape)
        sqrt_one_minus_alphas_bar = extract(self.diffusion_variables.sqrt_one_minus_alphas_bar, t, batch.shape)

        batch_mixed = sqrt_alpha_bar * batch + sqrt_one_minus_alphas_bar * noise

        return batch_mixed, noise
    # endregion

    # region generation functions
    @torch.no_grad()
    def generate_samples_ddpm(
            self,
            n_samples: int,
            return_samples: bool = False,
            return_pred_x0: bool = False,
    ):
        samples = []
        x0s     = []
        df: DiffusionVariables_DDPM = self.diffusion_variables

        batch_s  = n_samples
        channels = self.params_data.data_params.channels
        height   = self.params_data.data_params.height
        width    = self.params_data.data_params.width

        x_t = torch.randn(batch_s, channels, height, width, device=self.device)
        t   = self.params.diffusion.n_steps_generation - 1
        while t >= 1:
            z = torch.randn_like(x_t, device=self.device) if t > 1 else 0.
            t_tensor = torch.full((batch_s,), fill_value=t, device=self.device).long()

            eps_theta = self.model(x_t, t_tensor)

            eps_coef         = extract(df.eps_coef, t_tensor, x_t.shape)
            sqrt_recip_alpha = extract(df.sqrt_recip_alpha, t_tensor, x_t.shape)

            mean     = sqrt_recip_alpha * (x_t - eps_coef * eps_theta)
            var_sqrt = extract(df.sigma2_sqrt, t_tensor, x_t.shape)

            # region compute x0
            sqrt_recip_alpha_bar    = extract(df.sqrt_recip_alpha_bar   , t_tensor, x_t.shape)
            sqrt_recip_m1_alpha_bar = extract(df.sqrt_recip_m1_alpha_bar, t_tensor, x_t.shape)
            x0 = sqrt_recip_alpha_bar * x_t - sqrt_recip_m1_alpha_bar * eps_theta
            # endregion

            x_t_minus_one = mean + var_sqrt * z
            x_t = x_t_minus_one

            t -= 1

            if return_samples:
                samples.append(x_t)
            if return_pred_x0:
                x0s.append(x0)

        res = [x_t]
        if return_samples:
            res.append(samples)
        if return_pred_x0:
            res.append(x0s)

        return res

    @torch.no_grad()
    def get_x0_hat_ddpm(self, current_model_input, model_prediction, t):
        """
        For logging only in the training loop.
        """
        df: DiffusionVariables_DDPM = self.diffusion_variables

        sqrt_recip_alpha_bar    = extract(df.sqrt_recip_alpha_bar   , t, model_prediction.shape)
        sqrt_recip_m1_alpha_bar = extract(df.sqrt_recip_m1_alpha_bar, t, model_prediction.shape)

        x0 = sqrt_recip_alpha_bar * current_model_input - sqrt_recip_m1_alpha_bar * model_prediction

        return x0

    @torch.no_grad()
    def generate_samples_ddim(
            self,
            n_samples     : int,
            return_samples: bool = False,
            return_pred_x0: bool = False,
    ):
        samples = []
        x0s     = []
        df: DiffusionVariables_DDIM = self.diffusion_variables

        batch_s  = n_samples
        channels = self.params_data.data_params.channels
        height   = self.params_data.data_params.height
        width    = self.params_data.data_params.width

        x_tau_i = torch.randn(batch_s, channels, height, width, device=self.device)  # x_tau_S
        time_steps = df.time_steps.flip(dims=[0])[df.skip_steps:]  # tau_S, ..., tau_1

        for i, step in enumerate(time_steps):
            index = len(time_steps) - i - 1  # index in the tau list
            tau_i = torch.full((batch_s,), fill_value=step, device=self.device).long()  # time steps tau_i

            x_tau_i_minus_1, pred_x0, e_t = self.ddim_p_sample(
                x_tau_i     =x_tau_i,
                tau_i       =tau_i,
                index       =index,
                repeat_noise=df.repeat_noise,
                temperature =df.temperature,
            )
            x_tau_i = x_tau_i_minus_1

            if return_samples:
                samples.append(x_tau_i_minus_1)
            if return_pred_x0:
                x0s.append(pred_x0)

        res = [x_tau_i]
        if return_samples:
            res.append(samples)
        if return_pred_x0:
            res.append(x0s)

        return res

    @torch.no_grad()
    def ddim_p_sample(
            self,
            x_tau_i: torch.Tensor,
            tau_i  : torch.Tensor,
            index  : int,
            *,
            repeat_noise: bool,
            temperature : float,
    ):
        df: DiffusionVariables_DDIM = self.diffusion_variables
        e_t = self.model(x_tau_i, tau_i)

        alpha_sqrt           = df.ddim_alpha_sqrt[index]
        alpha_prev_sqrt      = df.ddim_alpha_prev_sqrt[index]
        sigma                = df.ddim_sigma[index]
        sqrt_one_minus_alpha = df.ddim_sqrt_one_minus_alpha[index]
        dir_xt_coef          = df.dir_xt_coef[index]

        pred_x0 = (x_tau_i - sqrt_one_minus_alpha * e_t) / alpha_sqrt  # current prediction of x0
        dir_xt  = dir_xt_coef * e_t                                    # direction pointing to xt

        if sigma == 0:
            noise = 0.
        elif repeat_noise:
            noise = torch.rand([1, *x_tau_i.shape[1:]], device=self.device)
        else:
            noise = torch.randn_like(x_tau_i, device=self.device)

        noise = noise * temperature

        x_tau_i_minus_1 = alpha_prev_sqrt * pred_x0 + dir_xt + sigma * noise

        return x_tau_i_minus_1, pred_x0, e_t

    @torch.no_grad()
    def get_x0_hat_ddim(self, current_model_input, model_prediction, t):
        """
        For logging only in the training loop.
        """
        df: DiffusionVariables_DDIM = self.diffusion_variables

        # keep in mind that alpha from ddim is alpha_bar from ddpm
        alpha_sqrt           = extract(df.sqrt_alpha_bar, t, current_model_input.shape)
        sqrt_one_minus_alpha = extract(df.sqrt_one_minus_alphas_bar, t, current_model_input.shape)

        pred_x0 = (current_model_input - sqrt_one_minus_alpha * model_prediction) / alpha_sqrt

        return pred_x0

    # endregion

    def get_metric_object(self) -> Metrics:
        if self.trainer.training:
            return self.train_metrics
        elif self.trainer.validating or self.trainer.sanity_checking:
            return self.valid_metrics
        elif self.trainer.testing or self.trainer.predicting:
            return self.test_metrics
        else:
            raise Exception(f'Stage not supported.')
