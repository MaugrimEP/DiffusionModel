from dataclasses import dataclass
from typing import Optional

from omegaconf import SI
from pytorch_lightning.callbacks import EarlyStopping


@dataclass
class EarlyStopParams:
	#################################
	early_stop: bool = False  # if enable early stopping
	#################################
	monitor  : str = SI('valid/loss')
	min_delta: float = 0.0
	patience : int = 100
	verbose  : bool = True
	mode     : str = 'min'

	check_on_train_epoch_end: Optional[bool] = False
	# early_stopping__log_rank_zero_only      : bool = False


def getEarlyStopping(params: EarlyStopParams):
	return EarlyStopping(
		monitor  =params.monitor,
		min_delta=params.min_delta,
		patience =params.patience,
		verbose  =params.verbose,
		mode     =params.mode,

		check_on_train_epoch_end=params.check_on_train_epoch_end,
		# log      =conf['early_stopping__log_rank_zero_only'],
	)
