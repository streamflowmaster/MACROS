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

        # 不再在 __init__ 里注册固定的 causal bias
        # 我们改成在 forward 里根据需要生成，或者直接用外部传入的 mask

    def forward(
            self,
            x,
            use_cache=False,
            return_cache=False,
            start_pos=0,
            attn_mask=None,  # ← 新增：外部传入的 attention mask
            is_causal=True,  # ← 新增：是否强制使用 causal（默认 True）
            return_attn=False,
    ):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.rotary:
            cos, sin = self.rotaryemb(q, start_pos=start_pos, seq_dim=2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache
        if use_cache and self.kv_cache is not None:
            cached_k, cached_v = self.kv_cache
            k = torch.cat([cached_k, k], dim=-2)
            v = torch.cat([cached_v, v], dim=-2)

        if use_cache or return_cache:
            self.kv_cache = (k, v)

        # ====================== 核心修改：attention mask 处理 ======================
        # 形状要求：
        #   attn_mask: (B, 1, T', T') 或 (B, T', T')，T' 是当前总序列长度（包括 cache）
        #   值为 0/-inf 或 bool（True=允许，False=禁止）

        if self.flash and not return_attn:
            # 使用 flash attention（最推荐）
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,  # ← 直接传入外部 mask
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal and attn_mask is None  # ← 只有没传 mask 时才用 causal
            )
        else:
            # 手动实现（兼容旧版本或需要 debug）
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # 1. 如果外部传了 mask，就用它
            if attn_mask is not None:
                # 确保形状兼容
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.view(B, 1, T, T)  # 扩展到 head 维度
                att = att.masked_fill(attn_mask == 0, float('-inf'))
            # 2. 否则，如果要求 causal，就用经典的下三角 mask
            elif is_causal:
                # 动态生成 causal mask（支持 KV cache 后长度变化）
                seq_len = k.size(-2)  # 当前总长度（包括 cache）
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=att.device))
                causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
                att = att.masked_fill(causal_mask == 0, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        if return_attn:
            return y, att  # 返回 (output, attn_probs)
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


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
            self,
            x,
            use_cache=False,
            return_cache=False,
            start_pos=0,
            attn_mask=None,
            is_causal=True,
            return_attn=False,
    ):
        attn_out = self.attn(
            self.ln_1(x),
            use_cache=use_cache,
            return_cache=return_cache,
            start_pos=start_pos,
            attn_mask=attn_mask,
            is_causal=is_causal,
            return_attn=return_attn
        )

        if return_attn:
            attn_out, layer_attn = attn_out
            x = x + attn_out
            return x, layer_attn
        else:
            x = x + attn_out
            return x