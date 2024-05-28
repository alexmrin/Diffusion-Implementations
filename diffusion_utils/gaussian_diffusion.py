import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.improved_diffusion_unet import Unet

def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 1e-2
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplemented(f"unknown beta schedule: {schedule_name}")
    
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class GaussianDiffusion():
    """
        Diffusion methods for DDPM
    """
    def __init__(self, betas, device):
        betas = torch.tensor(betas, dtype=torch.float32)
        self.betas = betas.to(device)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <=1).all()

        self.num_timesteps = int(betas.shape[0])
        self.device = device

        self.alphas = 1.0 - betas.to(device)
        self.alpha_bars = torch.cumprod(self.alphas, axis=0).to(device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars).to(device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-self.alpha_bars).to(device)

    def sample_x_ts(self, x_0: torch.Tensor):
        """
            Given x_0s, it will return a list of x_ts where t ~ U(0, T) corresponding time steps and epsilons
        """
        x_0 = x_0.to(self.device)
        ts = torch.randint(low=1, high=self.num_timesteps, size=(x_0.shape[0],), device=self.device)
        epsilons = torch.randn_like(x_0, device=self.device)
        x_ts = self.sqrt_alpha_bars[ts][:, None, None, None] * x_0 + self.sqrt_one_minus_alpha_bar[ts][:, None, None, None] * epsilons
        return x_ts, ts, epsilons

    def denoise_step(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, var_simple: bool = True):
        """
            Given x_t, use the model to generate a sample x_t-1 from p(x_t-1 | x_t)
        """
        # Ensure t is on the same device as the model
        t = t.to(self.device)
        x_t = x_t.to(self.device)
        
        if var_simple:
            sigmas = torch.sqrt(self.betas[t][:, None, None, None]).to(self.device)
        else:
            sigmas = torch.sqrt(torch.where(
                t[:, None, None, None] == 0,
                (self.betas[t][:, None, None, None] / (1 - self.alpha_bars[t][:, None, None, None])).to(self.device),
                ((1 - self.alpha_bars[t - 1][:, None, None, None]) / (1 - self.alpha_bars[t][:, None, None, None]) * self.betas[t][:, None, None, None]).to(self.device)
            ))
        
        zs = torch.where(
            t[:, None, None, None] == 0, 
            torch.zeros_like(x_t, device=self.device), 
            torch.randn_like(x_t, device=self.device)
        )
        noise = model(x_t, t)
        mean = 1 / torch.sqrt(self.alphas[t][:, None, None, None]).to(self.device) * (x_t - (1 - self.alphas[t][:, None, None, None]).to(self.device) / self.sqrt_one_minus_alpha_bar[t][:, None, None, None].to(self.device) * noise)
        std = sigmas * zs
        x_t_minus_one = mean + std
        # print("Step:", t.item(), "Sigma:", sigmas.mean().item(), "change in x_t:", F.mse_loss(x_t_minus_one, x_t).item(), "mean:", mean.mean().item(), "std:", std.mean().item())
        
        return x_t_minus_one, t - 1
    def sample_diffusion_process(self, model: nn.Module, size: Tuple, var_simple: bool = True):
        x_ts = torch.randn(size=(1, 3, *size), device=self.device)
        ts = torch.full(fill_value=self.num_timesteps-1, size=(1,), device=self.device)
        intermediate = []

        # Calculate step intervals for capturing intermediate images
        capture_interval = self.num_timesteps // 5

        with torch.inference_mode():
            for step in range(self.num_timesteps):
                x_ts, ts = self.denoise_step(model, x_ts, ts, var_simple)
                if (step % capture_interval == 0 or step == self.num_timesteps-1):
                    intermediate.append(x_ts.clone())  # Clone to avoid modification in subsequent steps

        # Stack the intermediate outputs along a new dimension
        if intermediate:
            intermediate = torch.stack(intermediate, dim=0).squeeze(1)  # Shape will be [num_intermediate_steps, 3, h, w]
        
        return intermediate

    def sample_x0(self, model: nn.Module, num_samples, size: Tuple, var_simple: bool = True):
        x_ts = torch.randn(size=(num_samples, 3, *size), device=self.device)
        ts = torch.full(fill_value=self.num_timesteps-1, size=(1,), device=self.device)

        with torch.inference_mode():
            for _ in range(self.num_timesteps):
                x_ts, ts = self.denoise_step(model, x_ts, ts, var_simple)
                
        return x_ts
    
    def calculate_loss(self, model: nn.Module, x_ts: torch.Tensor, ts: torch.Tensor, epsilons: torch.Tensor):
        """
            Calculates the loss using MSE of the given x_ts, the corresponding time steps against the true noise.
        """
        x_ts, epsilons = x_ts.to(self.device), epsilons.to(self.device)
        output = model(x_ts, ts)
        loss = F.mse_loss(output, epsilons)
        return loss
    
def plot_alpha_bars(schedule_name: str, num_diffusion_timesteps: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    betas = get_named_beta_schedule(schedule_name, num_diffusion_timesteps)
    diffusion = GaussianDiffusion(betas, device)
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(num_diffusion_timesteps), diffusion.alpha_bars.cpu().detach().numpy())
    plt.title(f"Alpha Bars for {schedule_name.capitalize()} Schedule")
    plt.xlabel("Timesteps")
    plt.ylabel("Alpha Bars")
    plt.show()

def main():
    plt.figure(figsize=(5, 5))
    linear_betas = get_named_beta_schedule("linear", 1000)
    cos_betas = get_named_beta_schedule("cosine", 1000)
    plt.plot(np.linspace(0, 1, 1000), linear_betas)
    plt.plot(np.linspace(0, 1, 1000), cos_betas)
    plt.title("Beta schedules")
    plt.show()
    
    plot_alpha_bars("linear", 1000)
    plot_alpha_bars("cosine", 1000)

def test_sample_x_ts():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_0 = torch.randn((100, 3, 32, 32))
    betas = get_named_beta_schedule("cosine", 1000)
    diffusion = GaussianDiffusion(betas, device)
    xt, ts, eps = diffusion.sample_x_ts(x_0)
    print(xt.shape, ts.shape, eps.shape)

def test_denoise_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_0 = torch.randn((2, 3, 32, 32))
    model = Unet(in_channels=3, out_channels=3, channels=64, n_res_blocks=1, attention_levels=[1],
                  channel_multipliers=[1, 1, 2],
                  channels_per_head=1, tf_layers=1, t_max=1000).to(device)
    betas = get_named_beta_schedule("cosine", 1000)
    diffusion = GaussianDiffusion(betas, device)
    xt, ts, _ = diffusion.sample_x_ts(x_0)
    print(xt.shape)
    x, ts = diffusion.denoise_step(model, xt, ts, False)
    print(x.shape, ts.shape)

def test_sample_x0():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet(in_channels=3, out_channels=3, channels=64, n_res_blocks=1, attention_levels=[1],
                  channel_multipliers=[1, 1, 2],
                  channels_per_head=1, tf_layers=1, t_max=1000).to(device)
    betas = get_named_beta_schedule("cosine", 1000)
    diffusion = GaussianDiffusion(betas, device)
    x = diffusion.sample_x0(model, 2, (32, 32))
    print(x.shape)

def test_calculate_loss():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_0 = torch.randn((2, 3, 32, 32))
    model = Unet(in_channels=3, out_channels=3, channels=64, n_res_blocks=1, attention_levels=[1],
                  channel_multipliers=[1, 1, 2],
                  channels_per_head=1, tf_layers=1, t_max=1000).to(device)
    betas = get_named_beta_schedule("cosine", 1000)
    diffusion = GaussianDiffusion(betas, device)
    xt, ts, eps = diffusion.sample_x_ts(x_0)
    loss = diffusion.calculate_loss(model, xt, ts, eps)
    print(f"Loss: {loss}")

if __name__ == "__main__":
    # test_sample_x_ts()
    # test_denoise_step()
    # test_sample_x0()
    # test_calculate_loss()
    main()