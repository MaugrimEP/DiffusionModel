from dataclasses import dataclass
from typing import Optional

from omegaconf import SI

from conf.dataset import ValueRange


@dataclass
class FIDParams:
    use_fid                 : bool = False
    dims                    : int = 2048
    batch_size              : int = 64
    init_fid                : bool = True
    load_initialization_path: Optional[str] = './_fids/fid_init.ckpt'
    number_to_generate      : int = 20_000
    check_frequency         : int = 1
    transform_batch         : Optional[str] = None

    value_range: ValueRange = SI('${dataset_params.data_params.value_range}')
