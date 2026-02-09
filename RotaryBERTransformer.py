import torch
import torch.nn as nn
from torch.nn import functional as F
# from BeamSearch import BeamHypotheses
from RoPE import RotaryEmbedding,rotate_half,apply_rotary_pos_emb
from visualizer import get_local
import math

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.rotary = config.rotary
        self.config = config
        if config.rotary:
            self.rotaryemb = RotaryEmbedding(config.n_embd // config.n_head)

        self.kv_cache = None

        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def set_bias(self, block_size):
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x, use_cache=False, return_cache=False, start_pos=0):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # print("QKV",q.shape,k.shape,v.shape)
        if self.rotary:
            # 使用 start_pos 调整 RoPE 的位置
            cos, sin = self.rotaryemb(q, start_pos=start_pos,seq_dim=2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV_cache 逻辑
        if use_cache and self.kv_cache is not None:
            cached_k, cached_v = self.kv_cache
            k = torch.cat([cached_k, k], dim=-2)
            v = torch.cat([cached_v, v], dim=-2)

        if use_cache or return_cache:
            self.kv_cache = (k, v)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att_map = self.attn_dropout(att)
            y = att_map @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# class Block(nn.Module):
#
#     def __init__(self, config):
#         super().__init__()
#         self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
#         self.attn = CausalSelfAttention(config)
#         self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
#         self.mlp = MLP(config)
#
#     def forward(self, x, use_cache=False, return_cache=False):
#         x = x + self.attn(self.ln_1(x),use_cache=use_cache, return_cache=return_cache)
#         x = x + self.mlp(self.ln_2(x))
#         return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, use_cache=False, return_cache=False, start_pos=0):
        x = x + self.attn(self.ln_1(x), use_cache=use_cache, return_cache=return_cache, start_pos=start_pos)
        x = x + self.mlp(self.ln_2(x))
        return x