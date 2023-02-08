from dataclasses import dataclass, field
from typing import List, Optional

import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger


@dataclass
class WandbParams:
	project     : str = "DiffusionModel"
	group       : str = ''  # the dataset
	name        : Optional[str] = None
	notes       : Optional[str] = None
	mode        : str = 'offline'  # [ offline | online | disabled ]
	tags        : List[str] = field(default_factory=lambda: [])
	resume      : str = 'never'  # [ never | must ]
	id          : Optional[str] = None


def get_wandb(params: WandbParams, global_dict: dict, additional_conf: dict = None):
	return _get_wandb(
		constructor=wandb.init,
		params=params,
		global_dict=global_dict,
		additional_conf=additional_conf,
	)


def get_wandb_logger(params: WandbParams, global_dict: dict, additional_conf: dict = None):
	return _get_wandb(
		constructor=WandbLogger,
		params=params,
		global_dict=global_dict,
		additional_conf=additional_conf,
	)


def _get_wandb(constructor, params: WandbParams, global_dict: dict, additional_conf: dict = None):
	if additional_conf is None:
		additional_conf = dict()

	to_save_conf = global_dict | additional_conf
	run = constructor(
		project=params.project,
		group  =params.group,
		name   =params.name,
		notes  =params.notes,
		mode   =params.mode,
		tags   =params.tags,
		resume =params.resume,
		id     =params.id,
		config =to_save_conf,
	)
	return run
