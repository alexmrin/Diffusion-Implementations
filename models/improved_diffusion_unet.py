from typing import List

import torch
import torch.nn as nn

from models.torch_utils import linear, conv_nd
from models.modules import (
    Upsample,
    Downsample,
    ConvTransformer,
    ResBlock,
    SwitchSequential,
)

class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        n_res_blocks: int,
        attention_levels: List[int],
        channel_multipliers: List[int],
        channels_per_head: int,
        tf_layers: int = 1,
        t_max: int = 1000,
    ):
        super().__init__()
        self.channels = channels
        levels = len(channel_multipliers)

        d_time_emb = channels * 4
        self.time_emb = nn.Sequential(
            linear(channels, d_time_emb),
            nn.SiLU(),
            linear(d_time_emb, d_time_emb)
        )
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(
            SwitchSequential(conv_nd(2, in_channels, channels, kernel_size=3, padding=1))
        )
        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]
        for i in range(levels):
            for _ in range(n_res_blocks):
                layers = [ResBlock(channels, d_time_emb, channels_list[i])]
                channels = channels_list[i]
                if i in attention_levels:
                    layers.append(ConvTransformer(channels, tf_layers, channels_per_head))
                self.down_blocks.append(SwitchSequential(*layers))
                input_block_channels.append(channels)
            if i != levels - 1:
                self.down_blocks.append(SwitchSequential(Downsample(channels)))
                input_block_channels.append(channels)

        self.middle_block = SwitchSequential(
            ResBlock(channels, d_time_emb),
            ConvTransformer(channels, tf_layers, channels_per_head),
            ResBlock(channels, d_time_emb)
        )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(levels)):
            for j in range(n_res_blocks + 1):
                layers = [ResBlock(channels + input_block_channels.pop(), d_time_emb, channels_list[i])]
                channels = channels_list[i]
                if i in attention_levels:
                    layers.append(ConvTransformer(channels, tf_layers, channels_per_head))
                if i != 0 and j == n_res_blocks:
                    layers.append(Upsample(channels))
                self.up_blocks.append(SwitchSequential(*layers))
        
        self.out = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            conv_nd(2, channels, out_channels, kernel_size=3, padding=1)
        )
        self._init_time_emb(channels, t_max)

    def _init_time_emb(self, d_model, t_max):
        assert d_model % 2 == 0

        pos = torch.arange(0, t_max)
        exp = torch.arange(0, d_model, step=2) / d_model
        freq = torch.exp(exp * -torch.log(torch.tensor(10000.0)))
        freq = torch.einsum("i, j -> ij", pos, freq) # (t_max, d_model//2)
        pe = torch.zeros((t_max, d_model))
        pe[:, 0::2] = torch.sin(freq)
        pe[:, 1::2] = torch.cos(freq) # (t_max, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x, time_steps):
        """
        time_steps is a tensor of the input time positions (b, )
        """
        x_input_blocks = []
        t_emb = self.pe[time_steps] # (b, channels)
        t_emb = self.time_emb(t_emb) # (b, d_time_emb)

        for module in self.down_blocks:
            x = module(x, t_emb)
            x_input_blocks.append(x)
        x = self.middle_block(x, t_emb)
        
        for module in self.up_blocks:
            x = torch.cat([x, x_input_blocks.pop()], dim=1)
            x = module(x, t_emb)
        
        return self.out(x)
    
def _test_time_embeddings():
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(15, 5))
    m = Unet(in_channels=1, out_channels=1, channels=320, n_res_blocks=1, attention_levels=[],
                  channel_multipliers=[1, 2, 2],
                  channels_per_head=1, tf_layers=1, t_max=1000)
    time = torch.arange(0, 1000)
    te = m.pe[time]
    plt.plot(np.arange(1000), te[:, [50, 100, 190, 260]].numpy())
    plt.legend(["dim %d" % p for p in [50, 100, 190, 260]])
    plt.title("Time embeddings")
    plt.show()

def _test_unet():
    m = Unet(in_channels=1, out_channels=1, channels=64, n_res_blocks=1, attention_levels=[1],
                  channel_multipliers=[1, 1, 2],
                  channels_per_head=1, tf_layers=1, t_max=1000)
    time = torch.randint(0, 1000, (10,))
    input = torch.randn(10, 1, 32, 32)
    output = m(input, time)
    print(output.shape)

if __name__ == "__main__":
    # _test_time_embeddings()
    _test_unet()