"Adapted from https://github.com/SongweiGe/TATS"

import os
import sys

sys.path.append(os.getcwd())

# from train.callbacks import ImageLogger, VideoLogger
# from train.get_dataset import get_dataset
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, open_dict, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from training.callbacks import ImageLogger, VideoLogger, ImageLoggerMatrix
from lightning.pytorch.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DDPStrategy

from data.dataset import VoxelDataset
from vq_gan.vqgan import VQGAN


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    print(cfg)
    pl.seed_everything(cfg.model.seed)
    
    save_dir = os.path.join(cfg.model.default_root_dir, cfg.model.checkpoint_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    
    run = wandb.init(project=cfg.model.wandb_project_name, config=OmegaConf.to_container(cfg))
    wandb_logger = WandbLogger(project=cfg.model.wandb_project_name, save_dir=save_dir)
    
    transform = transforms.Compose([
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),  # Resize images if necessary
            # transforms.Lambda(lambda x: (x / 127.5) - 1),  # Normalize to [-1, 1]
        ])

    train_dataset = VoxelDataset(chunks_csv_path=os.path.join(cfg.model.default_root_dir, "2.5d_train.csv"), data_type="train", base_path=os.path.join(cfg.model.default_root_dir, cfg.model.voxel_dir), transform=transform, normalize=cfg.model.normalize)
    validation_dataset = VoxelDataset(chunks_csv_path=os.path.join(cfg.model.default_root_dir, "2.5d_val.csv"), data_type="val", base_path=os.path.join(cfg.model.default_root_dir, cfg.model.voxel_dir), transform=transform, normalize=cfg.model.normalize)
    
    print(f"\n\nTrain Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(validation_dataset)}\n\n")
    
    # train_dataset, val_dataset, sampler = dataset
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size, num_workers=cfg.model.num_workers, sampler=None, pin_memory=True)
    validation_dataloder = DataLoader(dataset=validation_dataset, batch_size=cfg.model.batch_size, num_workers=cfg.model.num_workers, sampler=None, pin_memory=True)
    print("\n\nDataLoader batch size:", train_dataloader.batch_size)
    # val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
    #                             shuffle=False, num_workers=cfg.model.num_workers)

    # # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus, cfg.model.accumulate_grad_batches

    with open_dict(cfg):
        cfg.model.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
        # cfg.model.default_root_dir = os.path.join(
        #     cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
        cfg.model.lr, accumulate, ngpu/8, bs/4, base_lr))

    model = VQGAN(cfg, run)

    callbacks = []
    # callbacks.append(ModelCheckpoint(monitor='val/recon_loss', save_top_k=3, mode='min', filename='latest_checkpoint'))
    # callbacks.append(ModelCheckpoint(every_n_train_steps=3000, save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    # callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1, filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    # Create a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model.checkpoint_dir,                    # Directory to save checkpoints
        filename='epoch_{epoch:02d}_step_{step}', # Unique filename format
        save_top_k=-1,                            # Save all checkpoints (no limit)
        save_last=True,                           # Also save the latest checkpoint
        every_n_epochs=10,                         # Save every epoch
        save_on_train_epoch_end=True              # Save at the end of each epoch
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(ImageLogger(batch_frequency=50, max_images=4, clamp=True))
    callbacks.append(VideoLogger(batch_frequency=10, max_videos=4, clamp=True))

    # # load the most recent checkpoint file
    # base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    # if os.path.exists(base_dir):
    #     log_folder = ckpt_file = ''
    #     version_id_used = step_used = 0
    #     for folder in os.listdir(base_dir):
    #         version_id = int(folder.split('_')[1])
    #         if version_id > version_id_used:
    #             version_id_used = version_id
    #             log_folder = folder
    #     if len(log_folder) > 0:
    #         ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
    #         for fn in os.listdir(ckpt_folder):
    #             if fn == 'latest_checkpoint.ckpt':
    #                 ckpt_file = 'latest_checkpoint_prev.ckpt'
    #                 os.rename(os.path.join(ckpt_folder, fn),
    #                           os.path.join(ckpt_folder, ckpt_file))
    #         if len(ckpt_file) > 0:
    #             cfg.model.resume_from_checkpoint = os.path.join(
    #                 ckpt_folder, ckpt_file)
    #             print('will start from the recent ckpt %s' %
    #                   cfg.model.resume_from_checkpoint)

    accelerator = None
    if cfg.model.gpus > 1:
        accelerator = 'ddp'
        
    # print("Running sanity check...")
    # sample_inputs, _ = next(iter(train_dataloader))  # Get a sample input from the dataloader
    # sample_inputs = sample_inputs.permute(0, 2, 1, 3, 4)  # Reshape from [B,T,C,H,W] to [B,C,T,H,W]
    # print("----type----", type(sample_inputs))
    # # sample_inputs = np.array(sample_inputs) # Convert directly to numpy array
    # # print(sample_inputs.shape, type(sample_inputs))
    # sample_outputs = model(sample_inputs)  # Forward pass through the model

    # # Print input and output shapes for sanity check
    # print("Sample Input Shape:", sample_inputs.shape)
    # print("Sample Output Shape:", sample_outputs.shape)
    # profiler = PyTorchProfiler(profile_memory=True)
    trainer = pl.Trainer(
        # gpus=cfg.model.gpus,
        accelerator="gpu",
        devices=cfg.model.gpus,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=os.path.join(cfg.model.default_root_dir, "checkpoints_normalized_output"),
        # ckpt_path=cfg.model.resume_from_checkpoint,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=wandb_logger
        # gradient_clip_val=cfg.model.gradient_clip_val,
        # profiler=profiler
    )

    trainer.fit(model, train_dataloader, validation_dataloder, ckpt_path=cfg.model.resume_from_checkpoint)


if __name__ == '__main__':
    run()
