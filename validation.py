import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path
import shutil
import torch
from torch.backends import cuda, cudnn
from callbacks.custom import get_ckpt_callback, get_viz_callback
cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelSummary

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module


@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # GPU options
    # ---------------------
    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    logger = WandbLogger(project=config.wandb.project_name,name='testing', group=config.wandb.group_name)
    ckpt_path = Path(config.checkpoint)

    # ---------------------
    # Model
    # ---------------------

    module = fetch_model_module(config=config)
    module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    viz_callback = get_viz_callback(config=config)
    callbacks.append(viz_callback)
    callbacks.append(ModelSummary(max_depth=2))

    if os.path.exists(config.img_save_path) and os.path.isdir(config.img_save_path):
        for filename in os.listdir(config.img_save_path):
            file_path = os.path.join(config.img_save_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path) 
            except Exception as e:
                print(f"error {file_path}: {e}")
    else:
        os.makedirs(config.img_save_path)

    # ---------------------
    # Validation
    # ---------------------

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        default_root_dir=None,
        devices=gpus,
        logger=logger,
        log_every_n_steps=100,
        precision=config.training.precision,
        move_metrics_to_cpu=False,
    )
    with torch.inference_mode():
        if config.use_test_set:
            trainer.test(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))
        else:
            trainer.validate(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))


if __name__ == '__main__':
    main()
