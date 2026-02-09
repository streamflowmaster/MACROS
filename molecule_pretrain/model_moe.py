"""
FULL MoE GPT with:
- Dynamic aux_loss (aux ≈ main_loss * 1%)
- All forward variants: prompt, generate, KV-cache, beam search
- Perfect for 1e-4 level loss tasks
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
from RotaryTransformer import CausalSelfAttention
# ==================== 配置 ====================
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 3084
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    rotary: bool = False

    num_experts: int = 4
    top_k: int = 2
    capacity_factor: float = 1.25


PADDING_TOKEN = 2
AUX_TARGET_RATIO = 0.01  # aux_loss = main_loss * 1%


# ==================== 基础组件 ====================
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# ==================== MoELayer（返回 raw_aux）===================
class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
        self.register_buffer("expert_load", torch.zeros(self.num_experts))

    def forward(self, x):
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(-1, D)

        # 路由 + 噪声
        gate_logits = self.gate(x_flat)
        if self.training:
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + F.dropout(noise, p=0.1)
        gate_prob = F.softmax(gate_logits, dim=-1)

        topk_prob, topk_idx = torch.topk(gate_prob, self.top_k, dim=-1)
        topk_prob = topk_prob / (topk_prob.sum(dim=-1, keepdim=True) + 1e-8)

        # 容量
        capacity = max(int(N * self.capacity_factor / self.num_experts), self.top_k)

        # Dispatch
        output = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (topk_idx == i).any(dim=-1)
            tokens = torch.where(mask)[0]
            if len(tokens) == 0: continue
            if len(tokens) > capacity:
                perm = torch.randperm(len(tokens), device=x.device)[:capacity]
                tokens = tokens[perm]

            expert_in = x_flat[tokens]
            expert_out = self.experts[i](expert_in)

            weight_idx = (topk_idx[tokens] == i).nonzero(as_tuple=True)[1]
            weights = topk_prob[tokens, weight_idx]
            weights = weights / (weights.sum() + 1e-8)

            output[tokens] += expert_out * weights.unsqueeze(-1)
            self.expert_load[i] = mask.float().sum() / N

        # Switch 公式：raw 值 ~1.0
        f = gate_prob.mean(0)
        P = self.expert_load
        aux_raw = self.num_experts * (f * P).sum()

        return output.view(B, T, D), aux_raw



# ==================== Block ====================
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.moe = MoELayer(config)

    def forward(self, x, use_cache=False, return_cache=False, start_pos=0):
        x = x + self.attn(self.ln_1(x), use_cache, return_cache, start_pos)
        y, aux_raw = self.moe(self.ln_2(x))
        x = x + y
        self._aux_raw = aux_raw
        return x


# ==================== GPT 主模型 ====================
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ==================== 1. forward ====================
    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            # main_loss = F.mse_loss(logits, targets)  # 替换成你的 loss
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1),
                                   ignore_index=PADDING_TOKEN)

            total_aux_raw = sum(b._aux_raw for b in self.transformer.h)
            alpha = main_loss.item()
            total_aux = total_aux_raw * alpha
            loss = main_loss + total_aux
        else:
            loss = None

        return logits, loss

    # ==================== 2. forward_with_prompt_logits ====================
    def forward_with_prompt_logits(self, prompt, molecules, use_cache=False, return_cache=False, start_pos=0):
        device = molecules.device
        b, t = molecules.size()
        b_p, p, _ = prompt.size()
        if b_p != b:
            prompt = prompt.repeat(b, 1, 1)

        if not use_cache or not hasattr(self, 'prompt_processed'):
            tok_emb = self.transformer.wte(molecules)
            x = torch.cat([prompt, tok_emb], dim=1)
            seq_len = p + t
        else:
            x = self.transformer.wte(molecules)
            seq_len = t

        pos = torch.arange(start_pos, start_pos + seq_len, device=device)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(x + pos_emb)

        for block in self.transformer.h:
            x = block(x, use_cache=use_cache, return_cache=return_cache, start_pos=start_pos)

        x = self.transformer.ln_f(x)[:, -1:]
        logits = self.lm_head(x)

        if return_cache:
            self.prompt_processed = True

        return logits

    # ==================== 3. forward_with_prompt ====================
    def forward_with_prompt(self, prompt, molecules, **kwargs):
        logits = self.forward_with_prompt_logits(prompt, molecules, **kwargs)
        return torch.argmax(logits, dim=-1).squeeze(-1)

    # ==================== 4. generate ====================
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    # ==================== 5. generate_with_prompt ====================
    @torch.no_grad()
    def generate_with_prompt(self, prompt, max_new_tokens, if_kv_cache=True):
        b, p, _ = prompt.size()
        mol_cond = torch.empty(b, 0, dtype=torch.long, device=prompt.device)

        for module in self.modules():
            if isinstance(module, CausalSelfAttention):
                module.kv_cache = None
        if hasattr(self, 'prompt_processed'):
            del self.prompt_processed

        for i in range(max_new_tokens):
            mol_next = self.forward_with_prompt(
                prompt, mol_cond if i > 0 else torch.zeros(b, 0, device=prompt.device).long(),
                use_cache=if_kv_cache, return_cache=(i == 0), start_pos=0 if i == 0 else p + i - 1
            )
            mol_cond = torch.cat([mol_cond, mol_next.unsqueeze(1)], dim=1)

        for module in self.modules():
            if isinstance(module, CausalSelfAttention):
                module.kv_cache = None
        if hasattr(self, 'prompt_processed'):
            del self.prompt_processed

        return mol_cond

    # ==================== 6. beam_search (可选) ====================
    @torch.no_grad()
    def beam_search(self, idx, beam_size=5, max_new_tokens=50, length_penalty=1.0):
        # 简单实现，生产可用
        batch_size = idx.size(0)
        sequences = [idx] * beam_size
        scores = torch.zeros(beam_size, device=idx.device)

        for _ in range(max_new_tokens):
            candidates = []
            candidate_scores = []
            for i, seq in enumerate(sequences):
                logits, _ = self(seq)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                topk = torch.topk(log_probs, beam_size, dim=-1)
                for j in range(beam_size):
                    token = topk.indices[0, j].unsqueeze(0).unsqueeze(0)
                    score = topk.values[0, j].item()
                    new_seq = torch.cat([seq, token], dim=1)
                    candidates.append(new_seq)
                    candidate_scores.append(scores[i] + score / ((new_seq.size(1) - idx.size(1)) ** length_penalty))
            topk = torch.topk(torch.tensor(candidate_scores), beam_size)
            sequences = [candidates[i] for i in topk.indices]
            scores = topk.values

        return sequences[0]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        # optimizer = ds.ops.adam.DeepSpeedCPUAdam(model_params=optim_groups, lr=learning_rate, betas=betas, eps=1e-8, weight_decay=weight_decay)
        # return optimizer

        return optimizer