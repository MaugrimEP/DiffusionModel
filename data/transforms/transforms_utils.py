import torch
import torch.nn as nn

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()


class ZeroNormalize(nn.Module):
	"""Scale image to [-1, 1]"""
	def forward(self, x):
		return x * 2 - 1

	def reverse(self, x, clip: bool = True):
		x = (x + 1) / 2
		if clip:
			x = torch.clip(x, min=0, max=1)
		return x
