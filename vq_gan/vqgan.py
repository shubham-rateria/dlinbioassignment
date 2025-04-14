"""Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import argparse
import numpy as np
import pickle as pkl

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from vq_gan.utils import shift_dim, adopt_weight, comp_getattr
from vq_gan.lpips import LPIPS
from vq_gan.codebook import Codebook

from torch.cuda.amp import GradScaler, autocast

def check_for_nan(parameters, name):
    for param in parameters:
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"NaN or Inf detected in gradients of {name}")
                return True
    return False

def print_model_memory(model):
    total_params = 0
    for name, param in model.named_parameters():
        param_size = param.numel() * param.element_size()  # numel: number of elements, element_size: size of each element in bytes
        total_params += param_size
        print(f"{name}: {param_size / 1024 ** 2:.2f} MB")
    
    print(f"Total model memory: {total_params / 1024 ** 2:.2f} MB")
    
def check_tensor_memory(tensor, name=""):
    size_in_bytes = tensor.numel() * tensor.element_size()  # Size of tensor in bytes
    print(f"{name}: {size_in_bytes / 1024 ** 2:.2f} MB")
    return size_in_bytes

def silu(x):
    return x*torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQGAN(pl.LightningModule):
    def __init__(self, cfg, wandb_run):
        super().__init__()
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.embedding_dim = cfg.model.embedding_dim
        self.n_codes = cfg.model.n_codes

        self.encoder = Encoder(cfg.model.n_hiddens, cfg.model.downsample,
                               cfg.dataset.image_channels, cfg.model.norm_type, cfg.model.padding_type,
                               cfg.model.num_groups,
                               )
        self.decoder = Decoder(
            cfg.model.n_hiddens*2, cfg.model.downsample, cfg.dataset.image_channels, cfg.model.norm_type, cfg.model.num_groups)
        
        print("Decoder", self.decoder)
        
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(
            self.enc_out_ch, cfg.model.embedding_dim, 1, padding_type=cfg.model.padding_type)
        self.post_vq_conv = SamePadConv3d(
            cfg.model.embedding_dim, self.enc_out_ch, 1)

        self.codebook = Codebook(cfg.model.n_codes, cfg.model.embedding_dim,
                                 no_random_restart=cfg.model.no_random_restart, restart_thres=cfg.model.restart_thres)

        self.gan_feat_weight = cfg.model.gan_feat_weight
        # TODO: Changed batchnorm from sync to normal
        self.image_discriminator = NLayerDiscriminator(
            cfg.dataset.image_channels, cfg.model.disc_channels, cfg.model.disc_layers, norm_layer=nn.BatchNorm2d)
        self.video_discriminator = NLayerDiscriminator3D(
            cfg.dataset.image_channels, cfg.model.disc_channels, cfg.model.disc_layers, norm_layer=nn.BatchNorm3d)

        if cfg.model.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif cfg.model.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = cfg.model.image_gan_weight
        self.video_gan_weight = cfg.model.video_gan_weight

        self.perceptual_weight = cfg.model.perceptual_weight

        self.l1_weight = cfg.model.l1_weight
        self.save_hyperparameters()
        
        # manual optimization
        self.automatic_optimization = False
        self.gradient_clip_val = cfg.model.gradient_clip_val
        
        
    def encode(self, x, include_embeddings=False, quantize=True):
        h = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        return h

    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x, optimizers = [None, None], log_image=False, validation=False):
        for param in self.encoder.parameters():
            if torch.isnan(param).any() or torch.isinf(x).any():
                print(f"NaN detected in parameter of encoder")

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or inf detected in input data")

        x = x.permute(0, 2, 1, 3, 4)

        # opt_ae = generator
        opt_ae, opt_disc = optimizers
        B, C, T, H, W = x.shape
        encoded = self.encoder(x)
        # print(f"encoded.shape: {encoded.shape}")
        z = self.pre_vq_conv(encoded)
        # print(f"vqgan: latent: {z.shape}")
        vq_output = self.codebook(z)
        for key, value in vq_output.items():
            # print(f"{key}: {value.shape}")
            pass
        post_vq = self.post_vq_conv(vq_output['embeddings'])
        # print(f"post_vq: {post_vq.shape}")
        # print("Original Min Max", torch.min(x), torch.max(x))
        # t = post_vq
        # for name, module in self.decoder.named_children():
        #     print(f"{name}: {t.shape}")
        #     t = module(t)
        #     print(f"{name}: Mean={t.mean()}, Std={t.std()}, Min={t.min()}, Mat={t.max()}")
        
        # x_recon = t
        x_recon = self.decoder(post_vq)
        # print(f"x_recon: {x_recon.shape}")
        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, T, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1,1, 1, 1, 1).repeat(1, C, 1, H, W).cuda()
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if log_image:
            return frames, frames_recon, x, x_recon

        if not validation:
            # if optimizer_idx == 0:
            # Autoencoder - train the "generator"
            
            # Perceptual loss
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight

            # Discriminator loss (turned on after a certain epoch)
            logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon)
            logits_video_fake, pred_video_fake = self.video_discriminator(x_recon)
            g_image_loss = -torch.mean(logits_image_fake)
            g_video_loss = -torch.mean(logits_video_fake)
            g_loss = self.image_gan_weight*g_image_loss + self.video_gan_weight*g_video_loss
            disc_factor = adopt_weight(self.global_step, threshold=self.cfg.model.discriminator_iter_start)
            aeloss = disc_factor * g_loss

            # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
            image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.image_gan_weight > 0:
                logits_image_real, pred_image_real = self.image_discriminator(frames)
                for i in range(len(pred_image_fake)-1):
                    image_gan_feat_loss += feat_weights * F.l1_loss(pred_image_fake[i],pred_image_real[i].detach()).item() * (self.image_gan_weight > 0)
            if self.video_gan_weight > 0:
                logits_video_real, pred_video_real = self.video_discriminator(x)
                for i in range(len(pred_video_fake)-1):
                    video_gan_feat_loss += feat_weights * F.l1_loss(pred_video_fake[i],pred_video_real[i].detach()).item() * (self.video_gan_weight > 0)
            gan_feat_loss = disc_factor * self.gan_feat_weight * (image_gan_feat_loss + video_gan_feat_loss)

            self.log("train/g_image_loss", g_image_loss, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/g_video_loss", g_video_loss,
                        logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/image_gan_feat_loss", image_gan_feat_loss,
                        logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/video_gan_feat_loss", video_gan_feat_loss,
                        logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/perceptual_loss", perceptual_loss,
                        prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/recon_loss", recon_loss, prog_bar=True,
                        logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/aeloss", aeloss, prog_bar=True,
                        logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/commitment_loss", vq_output['commitment_loss'],
                        prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/perplexity', vq_output['perplexity'],
                        prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            
            # recon_loss, _, vq_output, aeloss, perceptual_loss, gan_feat_loss = self.forward(
            #         x, optimizer_idx)
            #     commitment_loss = vq_output['commitment_loss']
            #     loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
            # if optimizer_idx == 1:
            commitment_loss = vq_output['commitment_loss']
            
            loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
            if opt_ae:     
                opt_ae.zero_grad()
                self.manual_backward(loss)
                all_parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.pre_vq_conv.parameters()) + list(self.post_vq_conv.parameters()) + list(self.codebook.parameters())
                
                # print(f"Losses: recon_loss={recon_loss}, commitment_loss={commitment_loss}, aeloss={aeloss}, perceptual_loss={perceptual_loss}, gan_feat_loss={gan_feat_loss}")
                
                # print("======BEFORE CLIPPING GRADIENTS======")
                
                # print(f"Max value in input x: {x.max().item()}")
                # print(f"Min value in input x: {x.min().item()}")
                
                max_grad = max(p.grad.abs().max().item() for p in all_parameters if p.grad is not None)
                # print(f"Max gradient Encoder: {max_grad}")
                self.log("train/max_grad_all_params", max_grad, prog_bar=True, sync_dist=True)
                
                min_grad = min(p.grad.abs().min().item() for p in all_parameters if p.grad is not None)
                # print(f"Min gradient Encoder: {min_grad}")
                self.log("train/min_grad_all_params", min_grad, prog_bar=True, sync_dist=True)
                
                
                # print("=====================================")
                
                self.clip_gradients(opt_ae, gradient_clip_val=self.gradient_clip_val)
                
                # print("======AFTER CLIPPING GRADIENTS======")
                
                # max_grad = max(p.grad.abs().max().item() for p in self.encoder.parameters() if p.grad is not None)
                # print(f"Max gradient Encoder: {max_grad}")
                
                # min_grad = min(p.grad.abs().min().item() for p in self.encoder.parameters() if p.grad is not None)
                # print(f"Min gradient Encoder: {min_grad}")
                
                # print("=====================================")
                
                # Initialize a variable to store the total norm
                total_norm = 0.0

                
                total_e_norm = 0.0
                
                for p in list(self.encoder.parameters()):
                    if p.grad is not None:
                        total_e_norm += p.grad.data.norm(2)
                
                # if check_for_nan(self.encoder.parameters(), 'encoder'):
                #     self.wandb_run.alert(title="NaN in encoder parameters", text="NaN found in encoder parameters")
                # if check_for_nan(self.decoder.parameters(), 'decoder'):
                #     self.wandb_run.alert(title="NaN in decoder parameters", text="NaN found in decoder parameters")
                # if check_for_nan(self.pre_vq_conv.parameters(), 'pre_vq_conv'):
                #     self.wandb_run.alert(title="NaN in pre_vq_conv parameters", text="NaN found in pre_vq_conv parameters")
                # if check_for_nan(self.post_vq_conv.parameters(), 'post_vq_conv'):
                #     self.wandb_run.alert(title="NaN in post_vq_conv parameters", text="NaN found in post_vq_conv parameters")
                # if check_for_nan(self.codebook.parameters(), 'codebook'):
                #     self.wandb_run.alert(title="NaN in codebook parameters", text="NaN found in codebook parameters")
                
                nan_or_inf_detected = False

                # Iterate over model parameters and accumulate the norm of gradients
                for p in all_parameters:
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            # print(f"NaN or Inf detected in gradients of parameter.")
                            nan_or_inf_detected = True
                            self.wandb_run.alert(title="NaN in gradient", text="nan in gradient")
                        param_norm = p.grad.data.norm(2)  # L2 norm
                        total_norm += param_norm.item() ** 2
                    else:
                        print(f"None gradient for {p}")

                # if not nan_or_inf_detected:
                total_norm = total_norm ** 0.5
                self.log("train/opt_ae_total_norm", total_norm, prog_bar=True, sync_dist=True)
                opt_ae.step()
            
            # return recon_loss, x_recon, vq_output, aeloss, perceptual_loss, gan_feat_loss

            # if optimizer_idx == 1:
            # Train discriminator
            logits_image_real, _ = self.image_discriminator(frames.detach())
            logits_video_real, _ = self.video_discriminator(x.detach())

            logits_image_fake, _ = self.image_discriminator(frames_recon.detach())
            logits_video_fake, _ = self.video_discriminator(x_recon.detach())

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
            disc_factor = adopt_weight(self.global_step, threshold=self.cfg.model.discriminator_iter_start)
            discloss = disc_factor * (self.image_gan_weight*d_image_loss + self.video_gan_weight*d_video_loss)

            self.log("train/logits_image_real", logits_image_real.mean().detach(),
                        logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/logits_image_fake", logits_image_fake.mean().detach(),
                        logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/logits_video_real", logits_video_real.mean().detach(),
                        logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/logits_video_fake", logits_video_fake.mean().detach(),
                        logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/d_image_loss", d_image_loss,
                        logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/d_video_loss", d_video_loss,
                        logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/discloss", discloss, prog_bar=True,
                        logger=True, on_step=True, on_epoch=True, sync_dist=True)
            # return discloss

            if opt_disc:
                opt_disc.zero_grad()
                self.manual_backward(discloss)
                self.clip_gradients(opt_disc, gradient_clip_val=self.gradient_clip_val)
                
                 # Initialize a variable to store the total norm
                total_norm = 0.0

                all_parameters = list(self.image_discriminator.parameters()) +list(self.video_discriminator.parameters())
                nan_or_inf_detected = False
                # Iterate over model parameters and accumulate the norm of gradients
                for p in all_parameters:
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            # print(f"NaN or Inf detected in gradients of parameter.")
                            nan_or_inf_detected = True
                            self.wandb_run.alert(title="NaN in gradient", text="nan in gradient")
                        param_norm = p.grad.data.norm(2)  # L2 norm
                        total_norm += param_norm.item() ** 2
                    else:
                        print(f"None gradient for {p}")

                # if not nan_or_inf_detected:
                total_norm = total_norm ** 0.5
                self.log("train/opt_disc_total_norm", total_norm, prog_bar=True)
                opt_disc.step()
            
            return
        
        # this is returned in validation 
        perceptual_loss = self.perceptual_model(
            frames, frames_recon) * self.perceptual_weight
        return recon_loss, x_recon, vq_output, perceptual_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        self.forward(x, self.optimizers(), validation=False)
        # if optimizer_idx == 0:
        #     recon_loss, _, vq_output, aeloss, perceptual_loss, gan_feat_loss = self.forward(
        #         x, validation=False)
        #     commitment_loss = vq_output['commitment_loss']
        #     loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
        # if optimizer_idx == 1:
        #     discloss = self.forward(x, optimizer_idx)
        #     loss = discloss
        # return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch  # TODO: batch['stft']
        recon_loss, _, vq_output, perceptual_loss = self.forward(x, validation=True)
        self.log('val/recon_loss', recon_loss, prog_bar=True, sync_dist=True)
        self.log('val/perceptual_loss', perceptual_loss.mean(), prog_bar=True, sync_dist=True)
        self.log('val/perplexity', vq_output['perplexity'], prog_bar=True, sync_dist=True)
        self.log('val/commitment_loss',
                 vq_output['commitment_loss'], prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.pre_vq_conv.parameters()) +
                                  list(self.post_vq_conv.parameters()) +
                                  list(self.codebook.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(list(self.image_discriminator.parameters()) +
                                    list(self.video_discriminator.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        log = dict()
        x, _ = batch
        x = x.to(self.device)
        frames, frames_rec, _, _ = self(x,log_image=True)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        #log['mean_org'] = batch['mean_org']
        #log['std_org'] = batch['std_org']
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        x, _ = batch
        _, _, x, x_rec = self(x, log_image=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        #log['mean_org'] = batch['mean_org']
        #log['std_org'] = batch['std_org']
        return log

def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        # TODO Changed num_groups from 32 to 8
        # print("In Normalize", num_groups, in_channels)
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)

class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        self.conv_first = SamePadConv3d(
            image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        kernels = [3, 3]
        stride = 2

        for i in range(2):
            print("Down step", i)
            block = nn.Module()
            in_channels = n_hiddens * 2**i
            out_channels = n_hiddens * 2**(i+1)
            # out_channels = n_hiddens
            # print(f"block up: {i}, {in_channels}, {out_channels}")
            # block.down = SamePadConv3d(in_channels, out_channels, kernels[i], stride=stride, padding_type=padding_type)
            block.down = nn.Conv3d(in_channels, out_channels, kernels[i], stride=stride, padding=1, dilation=1, bias=True)
            block.res = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        # print_model_memory(self.conv_first)
        # check_tensor_memory(h, "encoder:conv_first")
        # print(f"After conv_first shape: {h.shape} {h.dtype}")
        for i, block in enumerate(self.conv_blocks):
            h = block.down(h)
            # print(f"After conv_block {i} down shape: {h.shape}")
            # print(f"MEM: After conv_block {i} down: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
            h = block.res(h)
            # print(f"After conv_block {i} res shape: {h.shape}")
            # print(f"MEM: After conv_block {i} res: {torch.cuda.memory_allocated() / 1024 ** 3} GB, shape: {h.shape}")
        h = self.final_block(h)
        # print(f"After final_block shape: {h.shape}")
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group', num_groups=32):
        super().__init__()

        # n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        # max_us = n_times_upsample.max()

        in_channels = n_hiddens*2
        # in_channels = n_hiddens
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        
        kernels = [(1, 2, 2), (7, 2, 2)]
        strides = [(1, 2, 2), (1, 2, 2)]
        
        for i in range(2):
            block = nn.Module()
            in_channels = in_channels if i == 0 else in_channels//2**(i)
            out_channels = in_channels//2
            print(f"block down: {i}, {in_channels}, {out_channels}")
            us = tuple([2, 2, 2])
            # print(f"decoder up, {in_channels}, {out_channels}")
            block.up = nn.ConvTranspose3d(in_channels, out_channels, kernels[i], stride=strides[i], padding=0, dilation=1, bias=True)
            block.res1 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            block.res2 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            # n_times_upsample -= 1

        self.conv_last = SamePadConv3d(
            out_channels, image_channel, kernel_size=3)
        
        self.final_activation = nn.Tanh()

    def forward(self, x):
        h = self.final_block(x)
        # print(f"After final_block shape: {h.shape}")
        torch.cuda.empty_cache()
        
        for i, block in enumerate(self.conv_blocks):
            # h = F.pad(h, (1,1,1,1,0,0))
            # print(f"MEM: Before conv_block {i} up: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
            h = block.up(h)
            # torch.cuda.empty_cache()
            # print(f"After conv_block {i} up shape: {h.shape}")
            
            # print(f"MEM: Before conv_block {i} res1: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
            h = block.res1(h)
            # torch.cuda.empty_cache()
            # print(f"After conv_block {i} res1 shape: {h.shape}")
            
            # print(f"MEM: Before conv_block {i} res2: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
            h = block.res2(h)
            # torch.cuda.empty_cache()
            # print(f"After conv_block {i} res2 shape: {h.shape}")
            
        # print(f"MEM: Before conv_last: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        h = self.conv_last(h)
        # h = self.final_activation(h)
        # torch.cuda.empty_cache()
        # print(f"After conv_last shape: {h.shape}")
        return h


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = SamePadConv3d(
            in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv2 = SamePadConv3d(
            out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        # print(f"RESNET - MEM: After x: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        h = silu(self.norm1(h))
        # print(f"RESNET - MEM: After norm1: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        # h = silu(h)
        # print(f"RESNET - MEM: After silu: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        h = self.conv1(h)
        # print(f"RESNET - MEM: After conv1: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        h = silu(self.norm2(h))
        # print(f"RESNET - MEM: After norm2: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        # h = silu(h)
        # print(f"RESNET - MEM: After silu: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
        h = self.conv2(h)
        # print(f"RESNET - MEM: After conv2: {torch.cuda.memory_allocated() / 1024 ** 3} GB")

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _
