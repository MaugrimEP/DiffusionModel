from dataclasses import dataclass, field
from typing import Tuple, Optional, Any, List
from omegaconf import MISSING, SI

from conf._util import return_factory
from conf.dataset import ValueRange


@dataclass
class LossParams:
	loss_name: str = 'l2'  # [ l1 | l2 | huber | ce ]
	reduction: str = 'mean'  # [ mean | sum | none ]
	predict_noise: bool = True


@dataclass
class MetricsParams:
	n_class     : Optional[int] = SI('${dataset_params.data_params.n_class}')
	ignore_index: Optional[int] = SI('${dataset_params.data_params.ignore_index}')


@dataclass
class OptimizersParams:
	learning_rate: float = 2e-5
	optimizer: str = 'adam'
	betas: Tuple[float, float] = (0.9, 0.999)
	weight_decay: float = 0
	momentum: float = 0.9

	use_scheduler: bool = True

	max_epochs: int = SI('${trainer_params.max_epochs}')
	max_steps : int = SI('${trainer_params.max_steps}' )


@dataclass
class BackboneParams:
	name           : str = 'unet_cold'
	arch           : str = 'unet'
	encoder        : str = 'resnet18'
	in_channels    : int = 3
	encoder_weights: Optional[str] = None
	classes        : int = 3

	time_mlp: bool = True
	dim_mlp : int = 64


@dataclass
class DiffusionParams_DDPM:
	diffusion_name: str = 'ddpm'

	n_steps_training  : int = 1_000
	n_steps_generation: int = SI('${model_params.diffusion.n_steps_training}')
	beta_min: float = 0.0001
	beta_max: float = 0.02


@dataclass
class DiffusionParams_DDIM:
	diffusion_name: str = 'ddim'

	n_steps_training  : int = 1_000
	n_steps_generation: int = 100
	ddim_discretize: str = 'uniform'  # [ uniform | quad ]
	ddim_eta: float = 0.

	time_step_factor: float = 0.8

	beta_min: float = 0.0001
	beta_max: float = 0.02

	#
	repeat_noise: bool = False
	temperature : float = 1.   # temperature is the noise temperature (random noise gets multiplied by this)
	skip_steps: int = 0


@dataclass
class SubLoggingParams:
	logging_mode: Optional[str] = 'epoch'  # [ epoch | batch ]
	train_stage: List[str] = return_factory(['train', 'valid', 'test'])  # in which stage to log
	freq: int = 1
	max_quantity: int = 5


@dataclass
class LoggingParams:
	name: str = 'blender1'

	# Logging step params
	log_steps: SubLoggingParams = SubLoggingParams()

	# Logging generate params
	log_generate: SubLoggingParams = SubLoggingParams(
		train_stage=['train'],
		freq=1,
		max_quantity=5,
	)
	time_step_in_process: int = 10  # number of time steps logged in process
	strategy: str = 'quad'
	"""
		uniform: uniform time steps
		quad   : quadratic time steps: sample more at the end and less at the beginning
	"""
	quad_factor: float = 0.8
	value_range: ValueRange = SI('${dataset_params.data_params.value_range}')


@dataclass
class ModelParams:
	optimizer       : OptimizersParams      = OptimizersParams()
	backbone        : BackboneParams        = BackboneParams()
	logging         : LoggingParams         = LoggingParams()
	loss            : LossParams            = LossParams()
	diffusion       : Any                   = MISSING
	metrics         : MetricsParams         = MetricsParams
