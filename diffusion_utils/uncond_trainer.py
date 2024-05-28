import os
import math
from copy import deepcopy
import yaml
import time

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
import wandb

from diffusion_utils.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule
from models.improved_diffusion_unet import Unet
from models.torch_utils import update_ema
from dataset.imagenet_64 import prepare_imagenet64_loader

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise ValueError(f"Error loading configuration file: {e}")

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, untrainable_params

def initialize_model(cfg):
    model = Unet(**cfg['model_cfg'])
    trainable_params, untrainable_params = count_parameters(model)
    print(f"Number of trainable parameters: {trainable_params}\nNumber of untrainable parameters: {untrainable_params}")
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg['training_cfg']['learning_rate'],
        betas=(cfg['training_cfg']['beta1'], cfg['training_cfg']['beta2']),
        eps=cfg['training_cfg']['epsilon'],
        weight_decay=cfg['training_cfg']['weight_decay']
    )
    def warmup_scheduler(step, total_steps):
        warmup_steps = int(total_steps * cfg['training_cfg']['warmup_prop'])
        if step < warmup_steps:
            return cfg['training_cfg']["min_lr"] / cfg['training_cfg']["learning_rate"] + (1 - cfg['training_cfg']["min_lr"] / cfg['training_cfg']["learning_rate"]) * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return ((cfg['training_cfg']["learning_rate"] - cfg['training_cfg']["min_lr"]) * cosine_decay + cfg['training_cfg']["min_lr"]) / cfg['training_cfg']["learning_rate"]
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_scheduler(step, cfg['training_cfg']['max_steps']))
    ema_model = deepcopy(model)
    return model, optimizer, scheduler, ema_model

def train(config_path):
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Load configuration
    cfg = load_config(config_path)

    if not cfg['training_cfg']['dry_run']:
        wandb.init(project=cfg['project_name'], name=cfg['run_name'], config=cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    betas = get_named_beta_schedule(cfg['training_cfg']['schedule_name'], cfg['model_cfg']['t_max'])

    model, optimizer, scheduler, ema_model = initialize_model(cfg)
    gaussian_diffusion = GaussianDiffusion(betas=betas, device=device)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Loading the information from the checkpoint file
    if os.path.exists(cfg['training_cfg']['checkpoint_pth']):
        print("Checkpoint found! Attemping to load checkpoint...")
        step = load_checkpoint(cfg['training_cfg']['checkpoint_pth'], model, ema_model, optimizer, scheduler)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        print("No checkpoint found, training from scratch...")
        step = 0

    # Compiling model
    if cfg['training_cfg']['compile']:
        print("Compiling model...")
        unoptimized_model = model
        model = torch.compile(unoptimized_model)

    # Preparing the data
    train_loader, val_loader = prepare_imagenet64_loader(batch_size=cfg['training_cfg']['batch_size'])
    model.to(device)
    print("Starting training...")
    loop_start = time.time()

    if not cfg['training_cfg']['dry_run']:
        wandb.watch(model)
        while True:
            for images, _ in train_loader:
                if step >= cfg['training_cfg']['max_steps']:
                    save_checkpoint(unoptimized_model, optimizer, scheduler, step, ema_model, cfg['training_cfg']['checkpoint_pth'])
                    print("Training Complete.")
                    wandb.save(cfg['training_cfg']['checkpoint_pth'])
                    wandb.finish()
                    return
                start_time = time.time()
                images = images.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    x_ts, ts, noise = gaussian_diffusion.sample_x_ts(images)
                    loss = gaussian_diffusion.calculate_loss(model, x_ts, ts, noise)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), cfg['training_cfg']['grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                loss /= cfg['training_cfg']['batch_size']
                update_ema(ema_model.parameters(), model.parameters())
                print(f"Step {step}: Time elapsed: {time.time() - loop_start}s, Loss: {loss}")

                if step % cfg['training_cfg']['log_interval'] == 0:
                    val_loss, ema_samples, samples = validation(
                        model if not cfg['training_cfg']['compile'] else unoptimized_model,
                        ema_model,
                        device,
                        gaussian_diffusion,
                        val_loader,
                    )
                    wandb.log({
                        "train_loss": loss.item(),
                        "step": step,
                        "step_time": time.time() - start_time,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "val_loss": val_loss,
                        "ema_samples": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_samples],
                        "samples": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in samples]
                    })
                else:
                    wandb.log({
                        "train_loss": loss.item(),
                        "step": step,
                        "step_time": time.time() - start_time,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                    })
                if step % cfg['training_cfg']['save_every'] == 0 and step != 0:
                    save_checkpoint(
                        model if not cfg['training_cfg']['compile'] else unoptimized_model,
                        optimizer,
                        scheduler,
                        step,
                        ema_model,
                        cfg['training_cfg']['checkpoint_pth']
                    )

                step += 1
    else:
        print("Dry run complete!")

def validation(model, ema_model, device, gaussian_diffusion, val_loader):
    print("Validating...")
    model.eval()
    original_model = model.state_dict()
    model.load_state_dict(ema_model.state_dict())
    model = model.to(device)

    val_losses = 0
    with torch.inference_mode():
        for images, _ in val_loader:
            images = images.to(device)
            with torch.cuda.amp.autocast():
                x_ts, ts, noise = gaussian_diffusion.sample_x_ts(images)
                val_loss = gaussian_diffusion.calculate_loss(model, x_ts, ts, noise)
            val_losses += val_loss.item() / images.shape[0]

    val_losses /= len(val_loader)

    # Sample
    ema_samples = gaussian_diffusion.sample_x0(model, 5, (64, 64)) # RESOLUTION FOR CIFAR10
    print("Range of samples before denormalization:", ema_samples.min().item(), ema_samples.max().item())
    ema_samples = torch.nan_to_num(ema_samples, nan=0.0)
    ema_samples = torch.clamp(ema_samples, 0, 1)
    print("Range of samples after denormalization:", ema_samples.min().item(), ema_samples.max().item())

    # Restore original model
    model.load_state_dict(original_model)
    samples = gaussian_diffusion.sample_x0(model, 5, (64, 64))
    samples = torch.nan_to_num(samples, nan=0.0)
    samples = torch.clamp(samples, 0, 1)
    model.train()
    print("Finished validating!")
    return val_losses, ema_samples, samples


def save_checkpoint(model, optimizer, scheduler, step, ema_model, path="checkpoint.pth"):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "ema_model": ema_model.state_dict(),
    }
    torch.save(state, path)

def load_checkpoint(
    load_pth: str,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler
):
    checkpoint = torch.load(load_pth)
    state_dict = checkpoint["model"]
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    ema_model.load_state_dict(checkpoint["ema_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    step = checkpoint["step"]
    print(f"Checkpoint loaded from {load_pth} at step {step}")
    return step

if __name__ == "__main__":
    train("uncond_config.yaml")