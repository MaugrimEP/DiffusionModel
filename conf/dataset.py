from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Any, Optional, Union, Set, List
from omegaconf import MISSING, SI


@dataclass
class SupervisionDatasetConfig:
    # region supervision
    supervision_mode: str = 'xy'
    """
    in each part, we still return (x,y), each model has to care what mode it is using
    xy  : only x,y pairs
    x   : only x (from x alone or (x,y))
    y   : only y (from y alone or (x,y))
    full: return the available (x,y) pairs as well as the x and y only.
    """

    # sum should be equal or less than one
    proportion_xy   : Union[int, str, float] = 1.  # (x,y) samples
    proportion_x    : Union[int, str, float] = 0.  # x only samples
    proportion_y    : Union[int, str, float] = 0.  # y only samples

    proportion_mode : str = 'frac'
    """
    how to interpret the proportions
        - frac: [float] values are proportions, eg 0.5 is 50%
        - perc: [int, float] values are percentages, eg 50 is 50%
        - abso: [int] values are absolutes, specify the number of samples for each parts
        - path: [str] values are paths to files containing the samples to use
    """

    return_supervision : bool = False
    random_supervision : bool = True  # if the token list is shuffled
    random_from_dataset: bool = True  # if we randomly sample from the source dataset:
    random_file        : Optional[str] = None  # if not None, pickle the random list to this file
                                               # if not None, random_supervision and random_from_dataset should be False
    # endregion


class ValueRange(Enum):
    Zero        = '01'
    ZeroUnbound = '01'
    One         = '11'
    OneUnbound  = '11unbound'


@dataclass
class BlenderParams:
    name: str = 'blender'
    root: str = "datasetpath/blender.lmdb"
    return_params: bool = False
    is_one: bool = True  # collapse all domains into one
    return_domain: Optional[List[int]] = None  # return only the specified domains if is_one is False
    height  : int = 64
    width   : int = 64
    channels: int = 3
    n_class : int = 0
    ignore_index: Optional[int] = None

    value_range: ValueRange = ValueRange.One


@dataclass
class DatasetParams:
    data_params: BlenderParams = BlenderParams()

    drop_last_train: bool = True
    drop_last_valid: bool = False
    drop_last_test : bool = False

    batch_size: int = 64
    batch_size_val : int = SI('${dataset_params.batch_size}')
    batch_size_test: int = SI('${dataset_params.batch_size}')
    use_min_for_batch_size: bool = True  # if drop_last_train is True and len(train)<batch size, set batch size to len(train)
    workers: int = 0
    pin_memory: bool = True

    limit_size: Optional[int] = None

    supervision_params: SupervisionDatasetConfig = SupervisionDatasetConfig()

    train_prop: Union[float, int, str] = 0.80
    valid_prop: Union[float, int, str] = 0.10
    test_prop : Union[float, int, str] = 0.10
    proportion_mode: str = 'frac'
    """
    how to interpret the proportions
        - frac: [float] values are proportions, eg 0.5 is 50%
        - perc: [int, float] values are percentages, eg 50 is 50%
        - abso: [int] values are absolutes, specify the number of samples for each parts
        - path: [str] values are paths to files containing the samples to use
    """
    limit_train: Optional[int] = None
    limit_valid: Optional[int] = None
    limit_test : Optional[int] = None
