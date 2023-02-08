import os

import hydra
import torch
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from omegaconf import OmegaConf

import wandb
from conf.checkpoint import CheckpointsCallbacks, getModelCheckpoint
from conf.main_config import GlobalConfiguration
from conf.trainer import get_trainer

from data.data import get_dm
from conf.earlystop import getEarlyStopping
from conf.wandb_params import get_wandb_logger
from src.DiffusionModel import DiffusionModel as Model
from src.callbacks.fid_callback import FIDCallback


@hydra.main(version_base=None, config_name='globalConfiguration')
def main(_cfg: GlobalConfiguration):
	print(OmegaConf.to_yaml(_cfg))
	cfg: GlobalConfiguration = OmegaConf.to_object(_cfg)

	pl.seed_everything(cfg.seed)
	if cfg.system_params.torch_params.hub_dir is not None:
		if cfg.system_params.torch_params.hub_dir == 'cwd':
			torch.hub.set_dir(os.path.join(os.getcwd(), 'torch_hub'))
		else:
			torch.hub.set_dir(cfg.system_params.torch_params.hub_dir)

	model_class = Model

	# wandb
	run_wandb = get_wandb_logger(params=cfg.wandb_params, global_dict=OmegaConf.to_container(_cfg))

	# Setup trainer
	dm = get_dm(cfg.dataset_params)
	model = model_class(cfg.model_params, cfg.dataset_params)

	if cfg.trainer_params.cudnn_benchmark is not None:
		cudnn.benchmark = True

	# region callbacks
	callbacks = []

	if cfg.fid_params.use_fid:
		dm.setup()
		fid_cb = FIDCallback(cfg.fid_params, train_dataset=dm.train_dataset, valid_dataset=dm.valid_dataset,
		                     test_dataset=dm.test_dataset)
		callbacks.append(fid_cb)

	modelCheckpoint: CheckpointsCallbacks = getModelCheckpoint(cfg.checkpoint_params)
	callbacks += [modelCheckpoint.on_monitor] if modelCheckpoint.on_monitor is not None else []
	callbacks += [modelCheckpoint.on_duration] if modelCheckpoint.on_duration is not None else []
	callbacks += [modelCheckpoint.on_tick] if modelCheckpoint.on_tick is not None else []

	early_stop = getEarlyStopping(cfg.early_stop_params)
	if cfg.early_stop_params.early_stop:
		callbacks.append(early_stop)
	# endregion

	trainer = get_trainer(cfg, callbacks, run_wandb)
	# Train
	trainer.tune(model, dm)
	trainer.fit(
		model,
		datamodule=dm,
		ckpt_path=cfg.checkpoint_params.retrain_saved_path if cfg.checkpoint_params.retrain_retrain_from_checkpoint else None)
	print("end fitting")

	if cfg.checkpoint_params.loading_for_test_mode == 'monitor':
		best_model = modelCheckpoint.on_monitor.best_model_path
		print(f'Load {best_model=} for testing')
		best_state_dict = torch.load(best_model)['state_dict']
		model.load_state_dict(best_state_dict)
	elif cfg.checkpoint_params.loading_for_test_mode == 'last':
		last_model = os.path.join(cfg.checkpoint_params.dirpath, 'last.ckpt')
		print(f'Load last {last_model=}')
		best_state_dict = torch.load(last_model)['state_dict']
		model.load_state_dict(best_state_dict)
	elif cfg.checkpoint_params.loading_for_test_mode == 'none':
		print(f'No modelCheckpoint callback, continue')
	else:
		raise Exception(f'Unknown {cfg.checkpoint_params.loading_for_test_mode}')
	# endregion

	print("start testing")
	trainer.test(model, datamodule=dm)
	# endregion

	print(f'<TERMINATE WANDB>')
	wandb.finish()
	print(f'<WANDB TERMINATED>')


if __name__ == '__main__':
	main()
