from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TorchParams:
	hub_dir: Optional[str] = 'cwd'
	"""
	None: use default torch params
	'cwd': use cwd/torch_hub
	'path': use path
	"""


@dataclass
class SystemParams:
	torch_params: TorchParams = TorchParams()
