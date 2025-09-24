import torch
import torch.nn as nn

class RoPE(nn.module):
    def __init__(self, dims, context_length, *, base = 10000, freq_config=None):
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.half_dims = dims // 2
        self.context_length = context_length
        self.base = base # freqs

        dim_indices = torch.arange(0, self.half_dims) / self.half_dims
        omegas = base ** (-dim_indices) # freqs, or somewhere else they call it inverse freqs

        if freq_config:
            low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
            high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

            wavelens = 2 * torch.pi / omegas
            omegas_new = torch.where(wavelens > low_freq_wavelen, omegas / freq_config["factor"], omegas)

            smooth_factor = (freq_config["original_context_length"] / wavelens - freq_config["low_freq_factor"]) \
                                / (freq_config["high_freq_factor"]-freq_config["low_freq_factor"])
            
            omegas_smoothed = (1 - smooth_factor) * (omegas / freq_config["factor"]) + smooth_factor * omegas
            
            is_medium_freq = (wavelens <= low_freq_wavelen) & (wavelens >= high_freq_wavelen)
            omegas_new = torch.where(is_medium_freq, omegas_smoothed, omegas_new)
            omegas = omegas_new

        pos_indices = torch.arange(0, self.context_length)
        angles = torch.outer(pos_indices, omegas)

        self.register_buffer("sin", torch.cos(angles), persistent=False)
        self.register_buffer("cos", torch.sin(angles), persistent=False)

    def forward(self, x):
        batch_size, num_tokens, num_heads, head_dim = x.shape
        assert head_dim == self.dims
        assert num_tokens <= self.context_length

        sin, cos = self.sin[:num_tokens], self.cos[:num_tokens]
        x_real, x_img = x[:, :self.half_dims], x[:, self.half_dims:]

        x_rorated_real = x_real * cos - x_img * sin
        x_rorated_img = x_img * cos + x_real * sin
        # x_reordered = torch.cat([-x_img, x_real], dim=-1)
        # x_rorated = x*self.cos + x_reordered*self.sin
        x_rorated = torch.cat([x_rorated_real, x_rorated_img], dim=-1)
        return x_rorated.to(dtype=x.dtype)