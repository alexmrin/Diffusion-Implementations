import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 1e-2
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
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
    def __init__(self, betas):
        betas = torch.tensor(betas, dtype=torch.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <=1).all()

        self.num_timesteps = int(betas.shape[0])

        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-self.alpha_bars)

    def sample_x_ts(self, x_0: torch.Tensor):
        """
            Given x_0s, it will return a list of x_ts where t ~ U(0, T) corresponding time steps and epsilons
        """
        ts = torch.randint(low=0, high=self.num_timesteps, size=(x_0.shape[0]))
        epsilons = torch.randn_like(x_0)
        x_ts = self.sqrt_alpha_bars[ts] * x_0 + self.sqrt_one_minus_alpha_bar[ts] * epsilons
        return x_ts, ts, epsilons

    def denoise_step(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, var_simple: bool = True):
        """
            Given x_t, use the model to generate a sample x_t-1 from p(x_t-1 | x_t)
        """
        if var_simple:
            sigmas = self.betas[t]
        else:
            sigmas = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]

        zs = torch.where(t == 1, torch.zeros_like(x_t), torch.randn_like(x_t))
        x_t_minus_one = 1 / self.sqrt_alpha_bars[t] * (x_t - (1 - self.alphas[t]) / self.sqrt_one_minus_alpha_bar[t] * model(x_t, t)) + sigmas * zs
        return x_t_minus_one, t - 1
    
    def sample_x0(self, model: nn.Module, n_samples: int, size: Tuple, var_simple: bool = True):
        x_ts = torch.randn(size=(n_samples, *size))
        ts = torch.full(self.num_timesteps, (n_samples,))
        for _ in range(self.num_timesteps):
            x_t_minus_one, ts = self.denoise_step(model, x_ts, ts, var_simple)
        return x_t_minus_one
    
    def calculate_loss(self, model: nn.Module, x_ts: torch.Tensor, ts: torch.Tensor, epsilons: torch.Tensor):
        """
            Calculates the loss using MSE of the given x_ts, the corresponding time steps against the true noise.
        """
        device = model.device()
        x_ts, epsilons = x_ts.to(device), epsilons.to(device)
        output = model(x_ts, ts)
        loss = F.mse_loss(output, epsilons)
        return loss
    
def plot_alpha_bars(schedule_name: str, num_diffusion_timesteps: int):
    betas = get_named_beta_schedule(schedule_name, num_diffusion_timesteps)
    diffusion = GaussianDiffusion(betas)
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(num_diffusion_timesteps), diffusion.alpha_bars.numpy())
    plt.title(f"Alpha Bars for {schedule_name.capitalize()} Schedule")
    plt.xlabel("Timesteps")
    plt.ylabel("Alpha Bars")
    plt.show()

def main():
    # plt.figure(figsize=(5, 5))
    # linear_betas = get_named_beta_schedule("linear", 1000)
    # cos_betas = get_named_beta_schedule("cosine", 1000)
    # plt.plot(np.linspace(0, 1, 1000), linear_betas)
    # plt.plot(np.linspace(0, 1, 1000), cos_betas)
    # plt.title("Beta schedules")
    # plt.show()
    
    plot_alpha_bars("linear", 1000)
    plot_alpha_bars("cosine", 1000)

if __name__ == "__main__":
    main()