import pytorch_lightning as pl
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainerParams:
	max_training_time: Optional[str] = None  # None  # DD:HH:MM:SS (days, hours, minutes seconds)

	accelerator: str = 'gpu'  # [ gpu | cpu ]

	strategy   : Optional[str] = None
	amp_backend: str = 'native'
	devices    : Optional[int] = None
	amp_level  : Optional[str] = None
	precision  : int = 32

	auto_select_gpus: bool = False
	gpus            : Optional[int] = None
	num_nodes       : int = 1

	plugins: List[str] = field(default_factory=lambda: [])

	check_val_every_n_epoch: Optional[int] = 1  # perform valid every N epoch, or set to None and
	# validation will be checked every val_check_interval batches
	log_every_n_steps: int = 1  # How often to log within steps. Default: 50.
	accumulate_grad_batches: Optional[int] = None

	max_epochs: int = 1000
	max_steps : int = -1

	auto_scale_batch_size: Optional[str] = None  # power | binsearch

	# parameters not related to pl trainer
	benchmark: Optional[bool] = True
	cudnn_benchmark: Optional[bool] = True

	num_sanity_val_steps: int = 2


def get_trainer(global_params, callbacks: List, logger):
	trainer_params = global_params.trainer_params
	trainer = pl.Trainer(
		max_epochs             =trainer_params.max_epochs,
		max_steps              =trainer_params.max_steps,
		max_time               =trainer_params.max_training_time,
		accelerator            =trainer_params.accelerator,
		devices                =trainer_params.devices,
		check_val_every_n_epoch=trainer_params.check_val_every_n_epoch,
		callbacks              =callbacks,
		logger                 =logger,
		enable_checkpointing   =True,
		log_every_n_steps      =trainer_params.log_every_n_steps,
		accumulate_grad_batches=trainer_params.accumulate_grad_batches,
		###
		strategy        =trainer_params.strategy,
		amp_backend     =trainer_params.amp_backend,
		amp_level       =trainer_params.amp_level,
		precision       =trainer_params.precision,
		auto_select_gpus=trainer_params.auto_select_gpus,
		gpus            =trainer_params.gpus,
		num_nodes       =trainer_params.num_nodes,
		plugins         =list(trainer_params.plugins),
		auto_scale_batch_size=trainer_params.auto_scale_batch_size,
		benchmark            =trainer_params.benchmark,
		num_sanity_val_steps =trainer_params.num_sanity_val_steps,
	)
	return trainer
