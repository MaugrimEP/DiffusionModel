import os
import os.path as osp
from typing import Set, Tuple
import pytorch_lightning as pl

import lmdb as lmdb
from torch.utils import data
from torch.utils.data import Dataset
import pyarrow as pa

from conf.dataset import BlenderParams
from data.BlenderDataset.blender_transforms import sample_transforms
from data.UtilsDataset import CustomDataModule
from utils.utils import display_tensor


def get_last_csv_line(csv_path: str, line_sep: str = '\n') -> str:
	line_sep = line_sep.encode()
	with open(csv_path, 'rb') as file:
		# Go to the end of the file before the last break-line
		file.seek(-2, os.SEEK_END)
		# Keep reading backward until the next break-line or the 2nd next if last_empty is True
		remaining_to_find = 1
		while remaining_to_find != 0:
			curr = file.read(1)
			if curr == line_sep:
				remaining_to_find -= 1
			else:
				file.seek(-2, os.SEEK_CUR)

		line = file.readline().decode()[:-1]
	return line


class BlenderLMDBDataset(Dataset):
	def __init__(
			self,
			db_path: str,
			return_params: bool = True,
			target_transform=None,
			return_domain: Set[int] = None,
			is_one: bool = True,
	):
		super(BlenderLMDBDataset, self).__init__()
		if return_domain is None and not is_one:
			return_domain = sorted({0, 1, 2})
		self.return_domain = return_domain

		self.db_path = db_path
		self.env = lmdb.open(
			path=db_path, subdir=osp.isdir(db_path),
			readonly=True, lock=False, readahead=False, meminit=False
		)
		with self.env.begin(write=False) as txn:
			self.length = pa.deserialize(txn.get(b'__len__'))
			self.keys   = pa.deserialize(txn.get(b'__keys__'))

		self.return_params = return_params
		self.target_transform = target_transform
		self.is_one = is_one

		self.transform = sample_transforms

		if is_one and return_domain is not None:
			raise ValueError('return_domain must be None if is_one is True')

	def __getitem__(self, index):
		if self.is_one:
			return self.get_data_one(index)
		else:
			return self.get_data_all(index)

	def get_data_one(self, index):

		fetch_in_db_index = index // 3
		in_row_inde = index % 3

		with self.env.begin(write=False) as txn:
			byteflow = txn.get(self.keys[fetch_in_db_index].encode())
		unpacked = pa.deserialize(byteflow)
		(cube_img, pyramid_img, cylinder_img), (cube_par, pyramid_par, cylinder_par) = unpacked

		if self.transform is not None:
			cube_img     = self.transform(cube_img)
			pyramid_img  = self.transform(pyramid_img)
			cylinder_img = self.transform(cylinder_img)

		if self.return_params and self.target_transform is not None:
			cube_par     = self.target_transform(cube_par)
			pyramid_par  = self.target_transform(pyramid_par)
			cylinder_par = self.target_transform(cylinder_par)

		imgs = [cube_img, pyramid_img, cylinder_img]
		params = [cube_par, pyramid_par, cylinder_par]

		img   = imgs[in_row_inde]
		param = params[in_row_inde]

		if self.return_params:
			return img, param
		else:
			return img

	def get_data_all(self, index):
		with self.env.begin(write=False) as txn:
			byteflow = txn.get(self.keys[index].encode())
		unpacked = pa.deserialize(byteflow)
		(cube_img, pyramid_img, cylinder_img), (cube_par, pyramid_par, cylinder_par) = unpacked

		if self.transform is not None:
			cube_img     = self.transform(cube_img)
			pyramid_img  = self.transform(pyramid_img)
			cylinder_img = self.transform(cylinder_img)

		if self.return_params and self.target_transform is not None:
			cube_par     = self.target_transform(cube_par)
			pyramid_par  = self.target_transform(pyramid_par)
			cylinder_par = self.target_transform(cylinder_par)

		imgs = [cube_img, pyramid_img, cylinder_img]
		params = [cube_par, pyramid_par, cylinder_par]

		imgs   = [imgs[i] for i in self.return_domain]
		params = [params[i] for i in self.return_domain]

		if self.return_params:
			return imgs, params
		else:
			return imgs

	def __len__(self):
		if self.is_one:
			return self.length * 3
		else:
			return self.length

	def __repr__(self):
		return self.__class__.__name__ + ' (' + self.db_path + ')'


class BlenderDataModule(CustomDataModule):
	def _fetch_base_dataset(self) -> Tuple[data.Dataset, data.Dataset, data.Dataset]:
		"""
		Return train, valid and test dataset
		"""
		blender_params: BlenderParams = self.p.data_params
		global_dataset = BlenderLMDBDataset(
			db_path         =blender_params.root,
			return_params   =blender_params.return_params,
			target_transform=None,
			return_domain   =blender_params.return_domain,
			is_one          =blender_params.is_one,
		)

		train_dataset, valid_dataset, test_dataset = self.split_dataset(global_dataset)

		return train_dataset, valid_dataset, test_dataset
