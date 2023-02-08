from dataclasses import dataclass, field
from typing import List, Any, Optional

from hydra.core.config_store import ConfigStore

from conf.checkpoint import CheckpointParams
from conf.dataset import DatasetParams
from conf.earlystop import EarlyStopParams
from conf.fids import FIDParams
from conf.model import ModelParams, DiffusionParams_DDPM, DiffusionParams_DDIM
from conf.system_params import SystemParams
from conf.wandb_params import WandbParams
from conf.slurm import CfgSlurm
from conf.trainer import TrainerParams


@dataclass
class GlobalConfiguration:
	# region default values
	defaults: List[Any] = field(default_factory=lambda: [
		'_self_',

		{'model_params/diffusion': 'ddim'},
	])

	seed: Optional[int] = 42
	# endregion

	checkpoint_params: CheckpointParams = CheckpointParams()
	dataset_params   : DatasetParams    = DatasetParams()
	early_stop_params: EarlyStopParams  = EarlyStopParams()
	model_params     : ModelParams      = ModelParams()
	wandb_params     : WandbParams      = WandbParams()
	cfgSlurm_params  : CfgSlurm         = CfgSlurm()
	trainer_params   : TrainerParams    = TrainerParams()
	system_params    : SystemParams     = SystemParams()
	fid_params       : FIDParams        = FIDParams()


# region register config
cs = ConfigStore.instance()

cs.store(name='globalConfiguration', node=GlobalConfiguration)

cs.store(group='model_params/diffusion', name='ddpm', node=DiffusionParams_DDPM)
cs.store(group='model_params/diffusion', name='ddim', node=DiffusionParams_DDIM)
# endregion
