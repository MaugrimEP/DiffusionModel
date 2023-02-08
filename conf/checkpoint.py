from dataclasses import dataclass
from typing import Optional, List

from omegaconf import SI
from pytorch_lightning.callbacks import ModelCheckpoint


@dataclass
class CheckpointParams:
	#################################
	model_checkpoint_on_monitor : bool = True   # check the metric
	model_checkpoint_on_duration: bool = True   # save every N durations with top-k on the duration
	model_checkpoint_on_tick    : bool = True   # tick on the last every N duration (only here to update last with, so independent of top k)
	loading_for_test_mode: str = 'monitor'  # [ none | monitor | last ]

	# generals params
	dirpath: str = '_model_save/'
	auto_insert_metric_name: bool = False
	save_weights_only: bool = False
	verbose: bool = True
	save_on_train_epoch_end: Optional[bool] = True  # without this, does not save at all :think:
	save_last: Optional[bool] = True  # create last.ckpt (for each checkpoint: cp checkpoint.ckpt last.ckpt)

	#################################
	#################################

	# region ON MONITOR
	on_monitor__filename: Optional[str] = '{epoch}_best_model'
	monitor: Optional[str] = SI('valid/loss')  # if None save to the last epoch
	mode: str = 'min'
	on_monitor__every_n_epochs: Optional[int] = 1
	on_monitor__save_top_k: int = 1
	# endregion

	#################################

	# region ON DURATION
	on_duration__filename: Optional[str] = '{epoch}_duration_model'
	on_duration__save_top_k: int = 1
	on_duration__every_n_epochs: Optional[int] = 1
	# endregion

	#################################

	# region ON TICK
	on_tick__every_n_epochs: Optional[int] = 1
	# endregion

	#################################
	retrain_retrain_from_checkpoint: bool = False
	#################################
	retrain_saved_path: str = '_model_save/last.ckpt'


@dataclass
class CheckpointsCallbacks:
	on_monitor : Optional[ModelCheckpoint]
	on_duration: Optional[ModelCheckpoint]
	on_tick    : Optional[ModelCheckpoint]


def getModelCheckpoint(params: CheckpointParams) -> CheckpointsCallbacks:
	if params.model_checkpoint_on_monitor:
		on_monitor = ModelCheckpoint(
			dirpath=params.dirpath,
			filename=params.on_monitor__filename,
			monitor=params.monitor,
			verbose=params.verbose,
			save_last=params.save_last,
			save_top_k=params.on_monitor__save_top_k,
			mode=params.mode,
			auto_insert_metric_name=params.auto_insert_metric_name,
			save_weights_only=params.save_weights_only,
			every_n_epochs=params.on_monitor__every_n_epochs,
			save_on_train_epoch_end=params.save_on_train_epoch_end,
		)
	else: on_monitor = None

	if params.model_checkpoint_on_duration:
		on_duration = ModelCheckpoint(
			dirpath=params.dirpath,
			filename=params.on_duration__filename,
			auto_insert_metric_name=params.auto_insert_metric_name,
			save_weights_only=params.save_weights_only,
			monitor='epoch',
			mode='max',
			save_top_k=params.on_duration__save_top_k,
			verbose=params.verbose,
			save_last=params.save_last,
			every_n_epochs=params.on_duration__every_n_epochs,
			save_on_train_epoch_end=params.save_on_train_epoch_end,
		)
	else: on_duration = None

	if params.model_checkpoint_on_tick:
		on_tick = ModelCheckpoint(
			dirpath=params.dirpath,
			# filename=params.filename,
			verbose=params.verbose,
			save_last=True,
			save_top_k=0,
			auto_insert_metric_name=params.auto_insert_metric_name,
			save_weights_only=params.save_weights_only,
			every_n_epochs=params.on_monitor__every_n_epochs,
			save_on_train_epoch_end=params.save_on_train_epoch_end,
		)
	else: on_tick = None

	return CheckpointsCallbacks(on_monitor=on_monitor, on_duration=on_duration, on_tick=on_tick)
