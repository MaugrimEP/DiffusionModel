import os
import pickle
import random
from abc import ABC, abstractmethod

from PIL import Image
from os.path import join

import numpy as np
from typing import Callable, Optional, List, Tuple, Any
import pytorch_lightning as pl

import torch
from torch.utils import data
from torch.utils.data import Subset, random_split

from conf.dataset import SupervisionDatasetConfig, DatasetParams
from utils.utils import display_tensor, display_mask

_modes = {
	(0, 'x' , (1, 0, 0), torch.Tensor((1, 0, 0))),
	(1, 'y' , (0, 1, 0), torch.Tensor((0, 1, 0))),
	(2, 'xy', (1, 1, 1), torch.Tensor((1, 1, 1))),
}
_g_modes = dict()
for values in _modes:
	for v in values[:-1]:
		_g_modes[v] = values


class ModeMasks:
	"""
	Object responsible to reduce a tensor according to it's mask

	"""
	def __init__(self, mode: torch.Tensor):
		self.mask_x  = mode[:, 0]
		self.mask_y  = mode[:, 1]
		self.mask_xy = mode[:, 2]

		self.mask_only_x = self.mask_x * (1 - self.mask_xy)
		self.mask_only_y = self.mask_y * (1 - self.mask_xy)

		self.nb_x = self.mask_x.sum()
		self.nb_y = self.mask_y.sum()
		self.nb_xy = self.mask_xy.sum()
		self.nb_only_x = self.nb_x - self.nb_xy
		self.nb_only_y = self.nb_y - self.nb_xy

		self.mask_dict = {
			'x' : (self.mask_x , self.nb_x ),
			'y' : (self.mask_y , self.nb_y ),
			'xy': (self.mask_xy, self.nb_xy),

			'ox': (self.mask_only_x, self.nb_only_x),
			'oy': (self.mask_only_y, self.nb_only_y),
		}

	@classmethod
	def FullSupervision(cls, x: torch.Tensor):
		batch_size = x.shape[0]
		mode = torch.ones([batch_size, 3])
		return cls(mode)

	def get_mask(self, key: str) -> torch.Tensor:
		return self.mask_dict[key][0].reshape(-1, 1)

	def get_boolean_mask(self, key: str) -> torch.Tensor:
		return (self.get_mask(key) == 1).reshape(-1)

	def get_nb(self, key: str) -> int:
		return self.mask_dict[key][1]

	def __call__(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
		if len(tensor.shape) != 2 or tensor.shape[1] != 1:
			raise Exception(f'Wrong tensor shape, waiting [batch size, 1], got {tensor.shape=}')

		if key is None:
			return torch.tensor(0.)
		if key == 'mean':
			return tensor.mean()

		number = self.get_nb(key)
		if number == 0:
			return torch.tensor(0.)

		return (tensor * self.get_mask(key)).sum() / number

	def mask(self, tensor: torch.Tensor, key: str):
		"""
		return the lines of the batch according to the keys
		"""
		batch_size = tensor.shape[0]
		bool_mask = self.get_boolean_mask(key)  # batch size
		indices   = torch.arange(0, batch_size).to(bool_mask.device)
		mask = torch.masked_select(indices, bool_mask)

		return tensor[mask]


class Modes:
	modes = _g_modes

	@staticmethod
	def get(key):
		if isinstance(key, List):
			key = tuple(key)
		return Modes.modes[key]

	@staticmethod
	def get_int(key) -> int:
		return Modes.get(key)[0]

	@staticmethod
	def get_str(key) -> str:
		return Modes.get(key)[1]

	@staticmethod
	def get_tuple(key) -> Tuple:
		return Modes.get(key)[2]

	@staticmethod
	def get_tensor(key) -> torch.Tensor:
		return Modes.get(key)[3]

	@staticmethod
	def get_list(key) -> List:
		return list(Modes.get(key)[2])


def limit_dataset_size(dataset: data.Dataset, size: int):
	size = min(size, len(dataset))
	limited_dataset = Subset(dataset, range(size))
	return limited_dataset


def getFileNumber(filename: str):
	return int(filename.split('_')[0])


class NumpyDataset(data.Dataset):
	def __init__(
			self,
			root_folder: str,
	):
		super(NumpyDataset, self).__init__()
		self.root_folder = root_folder

		files = [filename for filename in os.listdir(root_folder)
		         if os.path.isfile(os.path.join(root_folder, filename))
		         and 'numpy' in filename]

		x_files = sorted([f for f in files if '_x' in f], key=getFileNumber)
		y_files = sorted([f for f in files if '_y' in f], key=getFileNumber)

		self.files = list(zip(x_files, y_files))

	def __len__(self) -> int:
		return len(self.files)

	def __getitem__(self, idx):
		x_file, y_file = self.files[idx]

		x = np.load(os.path.join(self.root_folder, x_file)).astype(np.float32)
		y = np.load(os.path.join(self.root_folder, y_file)).astype(np.float32)

		# we expect segmentation map here, so we reshape them with channel first
		x = np.expand_dims(x, axis=-1)
		y = np.expand_dims(y, axis=-1)

		return x, y


class FolderDataset(data.Dataset):
	def __init__(
			self,
			root_folder: str,
	):
		super(FolderDataset, self).__init__()
		self.root_folder = root_folder
		self.images_folder = join(self.root_folder, 'images')
		self.labels_folder = join(self.root_folder, 'labels')

		images_files = [filename for filename in os.listdir(self.images_folder)
		         if os.path.isfile(join(self.images_folder, filename))]

		labels_files = [filename for filename in os.listdir(self.labels_folder)
		                if os.path.isfile(join(self.labels_folder, filename))]

		images_files = sorted(images_files)
		labels_files = sorted(labels_files)

		assert len(images_files) == len(labels_files), f'Expect to have the same number of image({len(images_files)}) and labels({len(labels_files)})'
		self.files = list(zip(images_files, labels_files))

	def __len__(self) -> int:
		return len(self.files)

	def __getitem__(self, idx):
		x_file, y_file = self.files[idx]
		x_path = join(self.images_folder, x_file)
		y_path = join(self.labels_folder, y_file)

		x = Image.open(x_path)
		y = Image.open(y_path)

		return x, y


class SupervisionDataset(data.Dataset):
	def __init__(
		self,
		dataset: data.Dataset,
		supervision_mode: str = 'full',  # [full, xy, x, y]
		proportion_xy   : float = 1.,
		proportion_x    : float = 0.,
		proportion_y    : float = 0.,
		proportion_mode : str = 'frac',

		return_supervision: bool = False,
		random_supervision: bool = True,
		random_from_dataset: bool = True,
		random_file        : Optional[str] = None,
	):
		super().__init__()
		self.supervision_mode = supervision_mode

		self.proportion_xy = proportion_xy
		self.proportion_x  = proportion_x
		self.proportion_y  = proportion_y
		self.proportion_mode = proportion_mode
		self.return_supervision = return_supervision
		self.random_supervision = random_supervision
		self.random_from_dataset = random_from_dataset
		self.random_file = random_file

		if random_file is not None and (random_supervision or random_from_dataset):
			raise ValueError('random_file is incompatible with random_supervision or random_from_dataset')

		if proportion_mode == 'frac':
			nb_xy = int(len(dataset) * proportion_xy)
			nb_x  = int(len(dataset) * proportion_x )
			nb_y  = int(len(dataset) * proportion_y )
		elif proportion_mode == 'perc':
			nb_xy = int(len(dataset) * proportion_xy / 100)
			nb_x  = int(len(dataset) * proportion_x  / 100)
			nb_y  = int(len(dataset) * proportion_y  / 100)
		elif proportion_mode == 'abso':
			nb_xy = proportion_xy
			nb_x  = proportion_x
			nb_y  = proportion_y
		elif proportion_mode == 'path':
			indexes_xy = pickle.load(open(proportion_xy, 'rb')).tolist() if proportion_xy != "" else []
			indexes_x  = pickle.load(open(proportion_x , 'rb')).tolist() if proportion_x  != "" else []
			indexes_y  = pickle.load(open(proportion_y , 'rb')).tolist() if proportion_y  != "" else []
			nb_xy = len(indexes_xy)
			nb_x  = len(indexes_x )
			nb_y  = len(indexes_y )
			assert set(indexes_xy).isdisjoint(set(indexes_x)) and set(indexes_xy).isdisjoint(set(indexes_y)) and set(indexes_x).isdisjoint(set(indexes_y)), "The supervision file are not well separated"
		else:
			raise ValueError(f'{proportion_mode=}')

		self.nb_xy = nb_xy
		self.nb_x  = nb_x
		self.nb_y  = nb_y
		print(
		f"""
			SupervisionDataset: {nb_xy=} {nb_x=} {nb_y=}
		""")

		if proportion_mode != 'path':
			# create the list of tokens
			tokens = ['xy'] * nb_xy + ['x'] * nb_x + ['y'] * nb_y
			if random_supervision:
				random.shuffle(tokens)

			# get random indices from the samples we will query from the dataset
			nb_tokens = len(tokens)
			if random_from_dataset:
				# takes nb_tokens random indices from the dataset
				dataset_indices = range(len(dataset))
				indices_dataset = random.sample(dataset_indices, nb_tokens)
			elif random_file is not None:
				indices_dataset = pickle.load(open(random_file, 'rb')).tolist()
				# the random file length is the dataset length, need to take less than that:
				indices_dataset = indices_dataset[:nb_tokens]
			else:
				indices_dataset = range(nb_tokens)
		else:
			tokens = ['xy'] * nb_xy + ['x'] * nb_x + ['y'] * nb_y
			indices_dataset = indexes_xy + indexes_x + indexes_y

		# only keep the targeted part
		keep = {
			'full': {'xy', 'x', 'y'},
			'xy'  : {'xy'},
			'x'   : {'xy', 'x'},
			'y'   : {'xy', 'y'},
		}[supervision_mode]

		final_dataset_indices = []
		final_tokens          = []
		for dataset_indice_i, token_i in zip(indices_dataset, tokens):
			if token_i in keep:
				final_dataset_indices.append(dataset_indice_i)
				final_tokens.append(token_i)

		trimmed_dataset = data.Subset(dataset, final_dataset_indices)
		trimmed_tokens  = final_tokens

		self.dataset = trimmed_dataset
		self.tokens = trimmed_tokens

		assert len(self.dataset) == len(self.tokens)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, i: int):
		sample = self.dataset[i]
		token  = self.tokens[i]
		tensor_token = Modes.get_tensor(token)

		if self.return_supervision:
			return sample + (tensor_token, )  # we want it as the last element of the tuple, not creating nested type
		else:
			return sample

	@staticmethod
	def get_supervised_dataset(dataset: data.Dataset, supervision_params: SupervisionDatasetConfig):
		return SupervisionDataset(
			dataset=dataset,
			supervision_mode=supervision_params.supervision_mode,
			proportion_xy=supervision_params.proportion_xy,
			proportion_x =supervision_params.proportion_x,
			proportion_y =supervision_params.proportion_y,
			proportion_mode=supervision_params.proportion_mode,
			return_supervision=supervision_params.return_supervision,
			random_supervision=supervision_params.random_supervision,
			random_from_dataset=supervision_params.random_from_dataset,
			random_file=supervision_params.random_file,
		)


class CustomDataModule(pl.LightningDataModule, ABC):
	def __init__(self, params: DatasetParams):
		super().__init__()
		self.train_dataset = None
		self.valid_dataset = None
		self.test_dataset = None
		self.p = params

	@abstractmethod
	def _fetch_base_dataset(self) -> Tuple[data.Dataset, data.Dataset, data.Dataset]:
		"""
		Return train, valid and test dataset
		"""
		pass

	def setup(self, stage: Optional[str] = None) -> None:
		base_train_dataset, base_valid_dataset, base_test_dataset = self._fetch_base_dataset()

		train_dataset = base_train_dataset
		valid_dataset = base_valid_dataset
		test_dataset = base_valid_dataset

		# region add limit size on the dataset
		if self.p.limit_size is not None:
			train_dataset = limit_dataset_size(train_dataset, self.p.limit_size)
			valid_dataset = limit_dataset_size(valid_dataset, self.p.limit_size)
			test_dataset  = limit_dataset_size(test_dataset, self.p.limit_size)

		# region Add supervision wrapper
		supervision_params = self.p.supervision_params
		train_dataset = SupervisionDataset.get_supervised_dataset(train_dataset, supervision_params)

		self.train_dataset = train_dataset
		self.valid_dataset = valid_dataset
		self.test_dataset  = test_dataset

		print(f'{len(self.train_dataset)=}')
		print(f'{len(self.valid_dataset)=}')
		print(f'{len(self.test_dataset)=}')

	def train_dataloader(self):
		dataset = self.train_dataset
		batch_size = self.p.batch_size
		if self.p.use_min_for_batch_size and self.p.drop_last_train and batch_size > len(dataset):
			print(f'[DropLast + Train dataset size = {len(dataset)} < {batch_size=}] => set batch size to dataset size'
			      f'this ensure that we do not have an empty dataset with drop last = True')
			batch_size = len(dataset)

		return data.DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=self.p.workers,
			pin_memory=self.p.pin_memory,
			drop_last=self.p.drop_last_train,
		)

	def val_dataloader(self):
		dataset = self.valid_dataset
		return data.DataLoader(
			dataset,
			batch_size=self.p.batch_size_val,
			shuffle=False,
			num_workers=self.p.workers,
			pin_memory=self.p.pin_memory,
			drop_last=self.p.drop_last_valid,
		)

	def test_dataloader(self):
		dataset = self.test_dataset
		return data.DataLoader(
			dataset,
			batch_size=self.p.batch_size_test,
			shuffle=False,
			num_workers=self.p.workers,
			pin_memory=self.p.pin_memory,
			drop_last=self.p.drop_last_test,
		)

	def split_dataset(self, dataset: data.Dataset):
		"""
		Instantiate the datasets and split them into train, val, test
		"""
		len_d = len(dataset)
		proportion_mode = self.p.proportion_mode

		if proportion_mode == 'frac':
			train_size = int(len_d * self.p.train_prop)
			valid_size = int(len_d * self.p.valid_prop)
			test_size = len_d - train_size - valid_size
		elif proportion_mode == 'perc':
			train_size = int(len_d * self.p.train_prop / 100)
			valid_size = int(len_d * self.p.valid_prop / 100)
			test_size = len_d - train_size - valid_size
		elif proportion_mode == 'abso':
			train_size = self.p.train_prop
			valid_size = self.p.valid_prop
			test_size = self.p.test_prop
		elif proportion_mode == 'path':
			indexes_train = pickle.load(open(self.p.train_prop, 'rb')).tolist() if self.p.train_prop != "" else []
			indexes_valid = pickle.load(open(self.p.valid_prop, 'rb')).tolist() if self.p.valid_prop != "" else []
			indexes_test = pickle.load(open(self.p.test_prop  , 'rb')).tolist() if self.p.test_prop  != "" else []
			train_size = len(indexes_train)
			valid_size = len(indexes_valid)
			test_size = len(indexes_test)
			assert set(indexes_train).isdisjoint(set(indexes_valid)) and set(indexes_train).isdisjoint(
				set(indexes_test)) and set(
				indexes_valid).isdisjoint(set(indexes_test)), "Files are not well separated"
		else:
			raise ValueError(f'{self.p.proportion_mode=}')

		self.train_size = train_size
		self.valid_size = valid_size
		self.test_size = test_size
		print(
			f"""
		    Splitting size: {train_size=} {valid_size=} {test_size=}
		""")

		# split the dataset
		if proportion_mode != 'path':
			train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
		else:
			train_dataset = Subset(dataset, indexes_train)
			valid_dataset = Subset(dataset, indexes_valid)
			test_dataset  = Subset(dataset, indexes_test)

		if self.p.limit_train is not None:
			train_dataset = Subset(train_dataset, range(self.p.limit_train))
		if self.p.limit_valid is not None:
			valid_dataset = Subset(valid_dataset, range(self.p.limit_valid))
		if self.p.limit_test is not None:
			test_dataset = Subset(test_dataset, range(self.p.limit_test))

		print("split_dataset >")
		print(f'{len(train_dataset)=}')
		print(f'{len(valid_dataset)=}')
		print(f'{len(test_dataset)=}')

		return train_dataset, valid_dataset, test_dataset
