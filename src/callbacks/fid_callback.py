import math
from os import path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from pytorch_lightning import Callback
import pytorch_lightning as pl

from torchmetrics.image.fid import FrechetInceptionDistance

from conf.fids import FIDParams
from utils.utils import normalize_value_range


class FrechetInceptionDistanceV2(FrechetInceptionDistance):
    def __init__(self, feature: int, reset_real_features: bool, normalize: bool):
        super().__init__(feature=feature, reset_real_features=reset_real_features, normalize=normalize)
        self.real_features_sum         = nn.Parameter(self.real_features_sum        , requires_grad=False)
        self.real_features_cov_sum     = nn.Parameter(self.real_features_cov_sum    , requires_grad=False)
        self.real_features_num_samples = nn.Parameter(self.real_features_num_samples, requires_grad=False)


class FIDCallback(Callback):
    def __init__(self, params: FIDParams, train_dataset , valid_dataset, test_dataset):
        super().__init__()
        self.p = params

        self.total_length = len(train_dataset) + len(valid_dataset) + len(test_dataset)
        self.full_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])

        self.fid = FrechetInceptionDistanceV2(
            feature=self.p.dims,
            reset_real_features=False,
            normalize=True,  # expect images in [0, 1] of type float
        )

        if self.p.load_initialization_path is not None and path.exists(self.p.load_initialization_path):
            print(f"FID Callback START Loading real stats from {self.p.load_initialization_path}")
            loaded = torch.load(self.p.load_initialization_path)
            self.fid.real_features_sum         = nn.Parameter(loaded['real_features_sum']        , requires_grad=False)
            self.fid.real_features_cov_sum     = nn.Parameter(loaded['real_features_cov_sum']    , requires_grad=False)
            self.fid.real_features_num_samples = nn.Parameter(loaded['real_features_num_samples'], requires_grad=False)
            print("END Loading real stats")
        elif self.p.init_fid:
            self.init_real_stats()

        self.validation_call = 0

    def init_real_stats(self):
        path_without_file = path.dirname(self.p.load_initialization_path)
        if not path.exists(path_without_file):
            raise ValueError(f"Path {path_without_file=} does not exist")

        full_dataset = self.full_dataset
        batch_size = self.p.batch_size

        dataloader = torch.utils.data.DataLoader(full_dataset, batch_size)

        print("START Computing real stats...")
        for i, batch in enumerate(dataloader, start=1):
            print(f'batch:{i} / {len(dataloader)}')
            self.fid.update(self.normalize_batch(batch), real=True)
        print("END Computing real stats")
        torch.save(self.fid.state_dict(), self.p.load_initialization_path)
        print(f"END Saving real stats at {self.p.load_initialization_path}")

    def _on_all_batch_end(self, pl_module) -> None:
        self.fid = self.fid.to(pl_module.device)

        size_to_generate = self.p.number_to_generate

        number_of_batch = math.ceil(size_to_generate / self.p.batch_size)
        print(f"START Computing fake stats for {number_of_batch} batches of size {self.p.batch_size}...")
        for i in range(number_of_batch):
            with torch.no_grad():
                fakes = pl_module.generate_samples(self.p.batch_size)
            self.fid.update(self.normalize_batch(fakes), real=False)
        print("END Computing fake stats")

        self.fid = self.fid.cpu()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.validation_call % self.p.check_frequency != 0:
            self.validation_call += 1
            return
        self.validation_call += 1

        self.fid.reset()
        self._on_all_batch_end(pl_module)
        res = self.fid.compute()
        pl_module.log(f"valid/FID_score", res)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.fid.reset()
        self._on_all_batch_end(pl_module)
        res = self.fid.compute()
        pl_module.log(f"test/FID_score", res)

    def normalize_batch(self, batch):
        """
        Fid callback expects images in [0,1] of type float if normalize is True
        """
        return normalize_value_range(batch, self.p.value_range, clip=True)
