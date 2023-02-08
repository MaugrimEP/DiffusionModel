from conf.dataset import DatasetParams
from data.BlenderDataset.BlenderDataset import BlenderDataModule


def get_dm(params: DatasetParams):
	dataset_name = params.data_params.name
	if dataset_name in ['blender']:
		return BlenderDataModule(params)
	else:
		raise Exception(f'Dataset type not available: {dataset_name=}')
