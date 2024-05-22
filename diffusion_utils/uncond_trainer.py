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
from datasets.cifar10 import prepare_cifar10_loader

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise ValueError(f"Error loading configuration file: {e}")

def initialize_model(cfg):
    model = Unet(**cfg['model_cfg'])
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
    ema_params = deepcopy(model.state_dict())
    return model, optimizer, scheduler, ema_params

def train(config_path):
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Load configuration
    cfg = load_config(config_path)

    # Initialize wandb
    wandb.init(project=cfg['project_name'], config=cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    betas = get_named_beta_schedule(cfg['training_cfg']['schedule_name'], cfg['model_cfg']['t_max'])

    model, optimizer, scheduler, ema_params = initialize_model(cfg)
    gaussian_diffusion = GaussianDiffusion(betas=betas)

    print("Compiling model...")
    model = torch.compile(model)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    wandb.watch(model)

    # Loading the information from the checkpoint file
    if os.path.exists(cfg['training_cfg']['checkpoint_pth']):
        step, ema_params = load_checkpoint(cfg['training_cfg']['checkpoint_pth'], model, optimizer, scheduler)
    else:
        step = 0

    # Preparing the data
    train_loader, val_loader = prepare_cifar10_loader(batch_size=cfg['training_cfg']['batch_size'])
    model.to(device)
    print("Starting training...")
    loop_start = time.time()

    while True:
        for images, _ in train_loader:
            if step >= cfg['training_cfg']['max_steps']:
                save_checkpoint(model, optimizer, scheduler, step, ema_params, cfg['training_cfg']['checkpoint_pth'])
                print("Training Complete.")
                return 0
            start_time = time.time()
            images = images.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x_ts, ts, noise = gaussian_diffusion.sample_x_ts(images)
                loss = gaussian_diffusion.calculate_loss(model, x_ts, ts, noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg['training_cfg']['grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            loss /= cfg['training_cfg']['batch_size']
            update_ema(ema_params, model.state_dict())
            print(f"Step {step}: Time elapsed: {loop_start-time.time()}s")

            wandb.log({"train_loss": loss.item(), "step": step, "step_time": start_time-time.time()})

            if step % cfg['training_cfg']['log_interval'] == 0:
                validate(model, val_loader, ema_params, gaussian_diffusion, device)

            step += 1

def validate(model, val_loader, ema_params, gaussian_diffusion, device):
    model.eval()
    original = model.state_dict()
    model.load_state_dict(ema_params).to(device)

    val_loss = 0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            with torch.cuda.amp.autocast():
                x_ts, ts, noise = gaussian_diffusion.sample_x_ts(images)
                loss = gaussian_diffusion.calculate_loss(model, x_ts, ts, noise)
            val_loss += loss.item() / images.shape[0]

    val_loss /= len(val_loader)

    # Log validation metrics
    wandb.log({"val_loss": val_loss})

    # Restore original model
    model.load_state_dict(original)
    model.train()

def save_checkpoint(model, optimizer, scheduler, step, ema_params, path="checkpoint.pth"):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "ema_params": ema_params
    }
    torch.save(state, path)
    # Save checkpoint to wandb
    wandb.save(path)

def load_checkpoint(
    load_pth: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler
):
    checkpoint = torch.load(load_pth)
    model.load_state_dict(checkpoint["model"])
    ema_dict = checkpoint["ema_params"]
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    step = checkpoint["step"]
    print(f"Checkpoint loaded from {load_pth} at step {step}")
    return step, ema_dict

if __name__ == "__main__":
    train("config.yaml")