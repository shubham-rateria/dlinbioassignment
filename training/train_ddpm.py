import os
import sys

sys.path.append(os.getcwd())

from re import I
# from ddpm.diffusion import Unet3D, GaussianDiffusion, Trainer
from ddpm.diffusion_without_latent import Unet3D, GaussianDiffusion, Trainer
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import os
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from data.dataset import VoxelDataset

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed as dist

# NCCL_P2P_DISABLE=1 accelerate launch train/train_ddpm.py

def setup_ddp(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def main_worker(rank, world_size, cfg):
  
    print(f"Main Worker: {rank}")   
    
    # setup_ddp(rank, world_size)
    
    print("Process group initialized.")
    
    results_folder = os.path.join(cfg.model.default_root_dir, "ddpm_results", cfg.model.results_folder_postfix)

    os.makedirs(results_folder, exist_ok=True)

    model = Unet3D(
        # dim=cfg.model.diffusion_img_size,
        cond_dim=cfg.model.cond_dim*cfg.model.num_features,
        dim=cfg.model.init_dim,
        dim_mults=cfg.model.dim_mults,
        attn_heads=cfg.model.attn_heads,
        init_kernel_size=cfg.model.init_kernel_size,
        # init_dim=cfg.model.init_dim,
        channels=cfg.model.diffusion_num_channels,
    ).to(rank)
    
    print("Num Unet3D Parameters", sum(p.numel() for p in model.parameters()))

    diffusion = GaussianDiffusion(
        model,
        # vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        # sampling_timesteps=cfg.model.sampling_timesteps,
        loss_type=cfg.model.loss_type,
        cond_dim=cfg.model.cond_dim,
        num_features=cfg.model.num_features,
        # objective=cfg.objective
    ).to(rank)

    # train_dataset, *_ = get_dataset(cfg)
    
    transform = transforms.Compose([
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),  # Resize images if necessary
            # transforms.Lambda(lambda x: (x / 127.5) - 1),  # Normalize to [-1, 1]
        ])

    train_dataset = VoxelDataset(chunks_csv_path=os.path.join(cfg.model.default_root_dir, "2.5d_train.csv"), data_type="train", base_path=os.path.join(cfg.model.default_root_dir, cfg.model.voxel_dir), transform=transform, return_meta=False, normalize=cfg.model.normalize)
    validation_dataset = VoxelDataset(chunks_csv_path=os.path.join(cfg.model.default_root_dir, "2.5d_val.csv"), data_type="val", base_path=os.path.join(cfg.model.default_root_dir, cfg.model.voxel_dir), transform=transform, return_meta=False, normalize=cfg.model.normalize)
    
    print(f"\n\nTrain Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(validation_dataset)}\n\n")
    
    if os.path.exists(cfg.model.load_milestone_path):
        model_files = [f for f in os.listdir(cfg.model.load_milestone_path) if f.startswith("model-") and f.endswith(".pt")]
        if model_files:
            latest_model = max(model_files, key=lambda x: int(x.split('-')[1].split('.')[0]))
            cfg.model.load_milestone = os.path.join(cfg.model.load_milestone_path, latest_model)
            print(f"Loading model from: {cfg.model.load_milestone}")


    trainer = Trainer(
        diffusion_model=diffusion,
        rank=rank,
        world_size=world_size,
        cfg=cfg,
        dataset=train_dataset,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=results_folder,
        num_workers=cfg.model.num_workers,
        sampling_csv="/home/sr2369/inpainting/2.5d_val.csv",
        load_path=cfg.model.load_milestone,
        denorm=cfg.model.normalize
        # logger=cfg.model.logger
    )

    # if cfg.model.load_milestone:
        # trainer.load(cfg.model.load_milestone)

    trainer.train()
    
    # cleanup_ddp()

if __name__ == '__main__':
    
    print("Starting Training...")
    
    # print("Cuda Visible Devices", os.environ['CUDA_VISIBLE_DEVICES'])
    # print("Master Addr", os.environ['MASTER_ADDR'])
    # print("Master Port", os.environ['MASTER_PORT'])
    
    hydra.initialize(config_path='../config', version_base=None)
    cfg = hydra.compose(config_name='base_cfg')
    
    world_size = cfg.model.gpus
    main_worker(0, world_size, cfg)
    # mp.spawn(main_worker, args=(world_size, cfg), nprocs=world_size, join=True) 
    # run()

    # wandb.finish()

    # Incorporate GAN loss in DDPM training?
    # Incorporate GAN loss in UNET segmentation?
    # Maybe better if I don't use ema updates?
    # Use with other vqgan latent space (the one with more channels?)
