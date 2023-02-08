import os
from os.path import isfile, join
from typing import List

import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from conf.dataset import ValueRange
from conf.model import LoggingParams, SubLoggingParams

colors = [   [  0,   0,   0],
             [128, 64, 128],
             [244, 35, 232],
             [70, 70, 70],
             [102, 102, 156],
             [190, 153, 153],
             [153, 153, 153],
             [250, 170, 30],
             [220, 220, 0],
             [107, 142, 35],
             [152, 251, 152],
             [0, 130, 180],
             [220, 20, 60],
             [255, 0, 0],
             [0, 0, 142],
             [0, 0, 70],
             [0, 60, 100],
             [0, 80, 100],
             [0, 0, 230],
             [119, 11, 32],
             ]

label_colours = dict(zip(range(len(colors)), colors))


def mask2rgb(mask, return_tensor: bool = True):
    """Mask should not be one-hot-encoded"""
    device = mask.device
    max_class = mask.max()

    mask = mask.cpu().numpy()
    r = mask.copy()
    g = mask.copy()
    b = mask.copy()

    for class_i in range(max_class):
        r[mask == class_i] = label_colours[class_i][0]
        g[mask == class_i] = label_colours[class_i][1]
        b[mask == class_i] = label_colours[class_i][2]

    bs, c, h, w = mask.shape
    rgb = np.zeros([bs, 3, h, w])
    rgb[:, 0] = r.reshape(bs, h, w) / 255.0
    rgb[:, 1] = g.reshape(bs, h, w) / 255.0
    rgb[:, 2] = b.reshape(bs, h, w) / 255.0

    if return_tensor:
        return torch.Tensor(rgb).to(device)
    return rgb


def is_logging_time(logging_params: SubLoggingParams, current_epoch, batch_idx, stage) -> bool:
    if stage not in logging_params.train_stage:
        return False

    if logging_params.logging_mode is None:
        return False
    elif logging_params.logging_mode == 'epoch':
        return current_epoch % logging_params.freq == 0 and batch_idx == 0
    elif logging_params.logging_mode == 'batch':
        return batch_idx % logging_params.freq == 0
    else:
        raise Exception(f'Unknown logging mode: {logging_params.logging_mode=}')


def get_file(path: str):
    """
    Return the checkpoint from the _model_save folder for testing
    """
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if isfile(join(path, f))]
        best_models = [f for f in files if 'best_model' in f]
        best_model = max(best_models, key=lambda file: int(file.split('_')[0]))
        return join(path, best_model)
    else:
        return path


def display_tensor(tensor: torch.Tensor, unnormalize: bool = False):
    """
    Debugging function to display tensor on screen
    """
    if unnormalize:
        tensor = (tensor + 1)/2
    if len(tensor.shape) == 4:  # there is the batch is the shape -> make a grid
        tensor = make_grid(tensor, padding=20)
    plt.imshow(tensor.permute(1, 2, 0).cpu())
    plt.show()


def display_mask(tensor: torch.Tensor):
    """
    Debugging function to display mask on screen
    """
    if 'FloatTensor' in tensor.type():
        tensor = torch.argmax(tensor, dim=1).unsqueeze(dim=1)
    tensor = mask2rgb(tensor)
    if len(tensor.shape) == 4:
        tensor = make_grid(tensor, padding=20)
    plt.imshow(tensor.permute(1, 2, 0).cpu())
    plt.show()


def normalize_value_range(tensor: torch.Tensor, value_range: ValueRange, clip: bool = False):
    if value_range == ValueRange.Zero:
        res = tensor
    elif value_range == ValueRange.ZeroUnbound:
        res = tensor
    elif value_range == ValueRange.One:
        res = (tensor + 1) / 2
    elif value_range == ValueRange.OneUnbound:
        res = (tensor + 1) / 2
    else:
        raise Exception(f'Unknown value range: {value_range=}')

    return res if not clip else torch.clamp(res, 0., 1.)


def undersample_list(src_list: List, nb_samples: int, strategy: str = 'uniform', quad_factor: float = 0.8, return_indices: bool = False):
    assert nb_samples >= 2
    if len(src_list) <= nb_samples:
        return src_list

    res = []
    first = src_list[0]
    last = src_list[-1]
    src_list = src_list[1:-1]
    nb_samples -= 2

    if strategy == 'uniform':
        step = len(src_list) // nb_samples
        time_steps = [i * step for i in range(0, nb_samples)]
    elif strategy == 'quad':
        time_steps = ((np.linspace(0, np.sqrt(len(src_list) * quad_factor), nb_samples)) ** 2).astype(int) + 1
        time_steps = [len(src_list) - i for i in time_steps][::-1]
    else:
        raise ValueError(f'{strategy=} is not an available discretization method.')

    for i in time_steps:
        res.append(src_list[i])

    res = [first] + res + [last]

    if return_indices:
        return res, [0] + time_steps + [len(src_list) + 1]
    else:
        return res
