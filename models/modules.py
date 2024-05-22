import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import conv_nd, avg_pool_nd, linear

class Downsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        if use_conv:
            self.down = conv_nd(
                dims=2,
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1, 
                stride=2
            )
        else:
            self.down = avg_pool_nd(
                dims=2,
                kernel_size=2,
            )

    def forward(self, x):
        return self.down(x)
    
class Upsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = 2
        if use_conv:
            self.conv = conv_nd(2, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int):
        super().__init__()
        assert d_head * n_heads == d_model, "d_model must be divisible by n_heads."
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.q = linear(d_model, d_model, bias=False)
        self.k = linear(d_cond, d_model, bias=False)
        self.v = linear(d_cond, d_model, bias=False)
        self.o = linear(d_model, d_model)

    def forward(self, x, cond: torch.Tensor=None):
        """
        Performs cross attention on x: (b, h*w, c) -> (b, h*w, c) taking in cond: (b, n_cond, d_cond) if d_cond is not specified d_cond=x
        """
        if cond is None:
            cond = x
        b, spatial, _ = x.shape
        n_cond = cond.shape[1]
        q = self.q(x) # (b, h*w, d_model) -> (b, h*w, d_model)
        k = self.k(cond) # (b, n_cond, d_cond) -> (b, n_cond, d_model)
        v = self.v(cond) # (b, n_cond, d_cond) -> (b, n_cond, d_model)
        q = q.reshape(b, spatial, self.n_heads, self.head_dim).transpose(1, 2) # (b, h*w, d_model) -> (b, n_heads, h*w, head_dim)
        k = k.reshape(b, n_cond, self.n_heads, self.head_dim).transpose(1, 2) # (b, n_cond, d_model) -> (b, n_heads, n_cond, head_dim)
        v = v.reshape(b, n_cond, self.n_heads, self.head_dim).transpose(1, 2) # (b, n_cond, d_model) -> (b, n_heads, n_cond, head_dim)
        attn = F.scaled_dot_product_attention(q, k, v) # (b, n_heads, h*w, head_dim)
        attn = attn.transpose(1, 2).reshape(b, spatial, self.d_model)
        return self.o(attn)
 
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_multi=4):
        super().__init__()
        self.up = linear(d_model, d_model * d_multi)
        self.gate = linear(d_model, d_model * d_multi)
        self.down = linear(d_model * 4, d_model)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor):
        return self.down(self.activation(self.gate(x)) * self.up(x))
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int):
        super().__init__()
        self.self_attn = CrossAttention(d_model, d_model, n_heads, d_head)
        self.cross_attn = CrossAttention(d_model, d_cond, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model)

    def forward(self, x, cond: torch.Tensor=None):
        """
        x is a (b, h*w, c) and cond is (b, n_cond, d_cond)
        """
        x = self.self_attn(self.norm1(x)) + x
        x = self.cross_attn(self.norm2(x), cond) + x
        x = self.ffn(x)
        return self.norm3(x)

class SpatialTransformer(nn.Module):
    def __init__(self, num_channels, n_layers, d_cond, channels_per_head=64):
        super().__init__()
        assert num_channels % channels_per_head == 0, "Number of channels must be divisible by channels per head."
        n_heads = num_channels // channels_per_head
        self.norm = nn.GroupNorm(32, num_channels)
        self.proj_in = conv_nd(2, num_channels, num_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = conv_nd(2, num_channels, num_channels, kernel_size=1, stride=1, padding=0)
        self.attn_blocks = nn.ModuleList([
            CrossAttentionBlock(num_channels, d_cond, n_heads, channels_per_head) for _ in range(n_layers)
        ])
    
    def forward(self, x, cond: torch.Tensor):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        for block in self.attn_blocks:
            x = block(x, cond)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        return x + residual

class ConvAttentionBlock(nn.Module):
    def __init__(self, num_channels, channels_per_head=64):
        super().__init__()
        assert num_channels % channels_per_head == 0, "Number of channels must be divisible by channels per head."
        self.qkv = conv_nd(1, num_channels, 3 * num_channels, 1)
        self.norm = nn.GroupNorm(32, num_channels)
        self.channels_per_head = channels_per_head
        self.num_heads = num_channels // channels_per_head
        self.out_proj = conv_nd(1, num_channels, num_channels, 1)

    def forward(self, x):
        """
        Turns x: (b, c, h, w) -> (b, c, h, w) while performing self attention using 1x1 convolutions instead of linear layers.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        q, k, v = torch.chunk(qkv, chunks=3, dim=1) # shape (b, 3*c, h*w) -> (b, c, h*w)
        q = q.reshape(b, self.num_heads, self.channels_per_head, -1).transpose(-2, -1)
        k = k.reshape(b, self.num_heads, self.channels_per_head, -1).transpose(-2, -1)
        v = v.reshape(b, self.num_heads, self.channels_per_head, -1).transpose(-2, -1)
        attn = F.scaled_dot_product_attention(q, k, v) # (b, n, h*w, d) -> (b, n, h*w, d)
        attn = attn.transpose(-2, -1).reshape(b, c, -1)
        o = self.out_proj(attn)
        return (x+o).reshape(b, c, *spatial)
    
class ConvTransformer(nn.Module):
    def __init__(self, num_channels, n_layers, channels_per_head=64):
        super().__init__()
        assert num_channels % channels_per_head == 0, "Number of channels must be divisible by channels per head."

        self.attn_blocks = nn.ModuleList([
            ConvAttentionBlock(num_channels, channels_per_head) for _ in range(n_layers)
        ])
    
    def forward(self, x):
        for block in self.attn_blocks:
            x = block(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channels, d_t_emb, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels

        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            conv_nd(2, channels, channels, kernel_size=3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            conv_nd(2, channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            linear(d_t_emb, channels)
        )
        self.residual = nn.Identity() if channels == out_channels else conv_nd(2, channels, out_channels, kernel_size=1)

    def forward(self, x, t_emb: torch.Tensor):
        # x: (b, c, h, w), t_emb: (b, c)
        conv1 = self.conv1(x)
        t_emb = self.time_emb(t_emb).type(conv1.dtype)
        fuse = conv1 + t_emb[:, :, None, None]
        conv2 = self.conv2(fuse)
        return conv2 + self.residual(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x