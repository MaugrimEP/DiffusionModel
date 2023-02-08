import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Dict

from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import MulticlassJaccardIndex

from conf.model import MetricsParams


class Metrics(pl.LightningModule):
	def __init__(self, params: MetricsParams):
		super().__init__()
		self.params = params
		self.mae = MeanAbsoluteError()
		self.mse = MeanSquaredError()

	def get_dict(self, batch_reco, batch_mixed, batch) -> Dict:
		self.mae.update(preds=batch_reco, target=batch)
		self.mse.update(preds=batch_reco, target=batch)
		return {
			'mae': self.mae,
			'mse': self.mse,
		}
