import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List, Optional
from spectra_prediction_token.Weighted_CrossEntropyLoss import ToeplitzWeightedCrossEntropyLoss

# Assuming RotaryTransformer and Block are defined elsewhere
from RotaryTransformerMask import Block, LayerNorm

IGNORE_INDEX = 2

@dataclass
class GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    delta: int = 1025
    intensity: int = 150
    rotary: bool = False
    bias: bool = True
    prompt_pe: bool = True
    start_generate_token: bool = False
    use_wce_loss: bool = False
    delta_weight: float = 2.0
    intensity_weight: float = 0.2
    use_intensity: bool = False
    # New parameter to control intensity inclusion

class NMRFeatures:
    DELTA = 0
    INTENSITY = 1

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # print('cnmr_config', config)
        assert config.block_size is not None
        assert config.delta > IGNORE_INDEX, "Delta vocab size must exceed ignore index"
        if config.use_intensity:
            assert config.intensity > IGNORE_INDEX, "Intensity vocab size must exceed ignore index"

        self.transformer = nn.ModuleDict({
            'delta_embedding': nn.Embedding(config.delta, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        })
        if config.use_intensity:
            self.transformer['intensity_embedding'] = nn.Embedding(config.intensity, config.n_embd)

        self.heads = nn.ModuleDict({
            'delta': nn.Linear(config.n_embd, config.delta, bias=False),
        })
        if config.use_intensity:
            self.heads['intensity'] = nn.Linear(config.n_embd, config.intensity, bias=False)

        # Weight tying
        self.heads['delta'].weight = self.transformer['delta_embedding'].weight
        if config.use_intensity:
            self.heads['intensity'].weight = self.transformer['intensity_embedding'].weight

        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        if config.use_wce_loss:
            self.wce_loss = nn.ModuleDict({
                'delta': ToeplitzWeightedCrossEntropyLoss(num_classes=config.delta, W=config.delta, ignore_index=IGNORE_INDEX),
            })
            if config.use_intensity:
                self.wce_loss['intensity'] = ToeplitzWeightedCrossEntropyLoss(num_classes=config.intensity, W=config.intensity, ignore_index=IGNORE_INDEX)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            if self.config.use_intensity:
                n_params -= self.transformer.intensity_embedding.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def validate_nmr_idx(self, nmr_idx: Tuple[torch.Tensor, ...]):
        delta = nmr_idx[0]
        if self.config.use_intensity:
            intensity = nmr_idx[1]
            assert intensity.size() == delta.size(), "All sequences must have same shape"
            assert intensity.device == delta.device, "Device mismatch"
        assert delta.size(1) <= self.config.block_size, f"Sequence length {delta.size(1)} exceeds block size"

    def set_input_targets(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return seq[:, :-1], seq[:, 1:]

    # def forward_to_embeds(self, nmr_idx: Tuple[torch.Tensor, ...],only_embeds:bool=True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    #     self.validate_nmr_idx(nmr_idx)
    #     delta = nmr_idx[0]
    #     inputs_delta, targets_delta = self.set_input_targets(delta)
    #     targets = [targets_delta]
    #
    #     device = inputs_delta.device
    #     b, t = inputs_delta.size()
    #
    #     x = self.transformer.delta_embedding(inputs_delta)
    #     if self.config.use_intensity:
    #         intensity = nmr_idx[1]
    #         inputs_intensity, targets_intensity = self.set_input_targets(intensity)
    #         intensity_emb = self.transformer.intensity_embedding(inputs_intensity)
    #         x = x + intensity_emb
    #         targets.append(targets_intensity)
    #
    #     if not self.config.rotary:
    #         pos = torch.arange(0, t, dtype=torch.long, device=device)
    #         x = self.transformer.drop(x + self.transformer.wpe(pos))
    #     else:
    #         x = self.transformer.drop(x)
    #
    #     for block in self.transformer.h:
    #         x = block(x)
    #     x = self.transformer.ln_f(x)
    #     if only_embeds:
    #         return x
    #     else:
    #         return x, targets

    def forward_to_embeds(self, nmr_idx: Tuple[torch.Tensor, ...],only_embeds:bool=True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.validate_nmr_idx(nmr_idx)
        delta = nmr_idx[0]
        inputs_delta, targets_delta = self.set_input_targets(delta)
        targets = [targets_delta]
        device = inputs_delta.device
        b, t = inputs_delta.size()
        attn_mask = (inputs_delta != IGNORE_INDEX)[:, None, None, :]
        x = self.transformer.delta_embedding(inputs_delta)

        if self.config.use_intensity:
            intensity = nmr_idx[1]
            inputs_intensity, targets_intensity = self.set_input_targets(intensity)
            intensity_emb = self.transformer.intensity_embedding(inputs_intensity)
            x = x + intensity_emb
            targets.append(targets_intensity)

        if not self.config.rotary:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = self.transformer.drop(x + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, attn_mask=attn_mask)
        x = self.transformer.ln_f(x)
        if only_embeds:
            return x
        else:
            return x, targets

    def forward(self, nmr_idx: Tuple[torch.Tensor, ...], targets: Optional[List[torch.Tensor]] = None) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        x, targets = self.forward_to_embeds(nmr_idx,only_embeds=False)
        logits = [self.heads['delta'](x)]
        if self.config.use_intensity:
            logits.append(self.heads['intensity'](x))

        if targets is not None:
            losses = []
            keys = ['delta'] if not self.config.use_intensity else ['delta', 'intensity']
            for i, (key, logit) in enumerate(zip(keys, logits)):
                loss_fn = self.wce_loss[key] if self.config.use_wce_loss else self.ce_loss
                loss = loss_fn(logit.view(-1, logit.size(-1)), targets[i].contiguous().view(-1))
                losses.append(loss * getattr(self.config, f'{key}_weight'))
            loss = sum(losses)
            return logits, loss
        return logits, None

    def forward_with_prompt(self, nmr_idx: Tuple[torch.Tensor, ...], prompt: torch.Tensor, training: bool = False) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        self.validate_nmr_idx(nmr_idx)
        delta = nmr_idx[0]
        inputs_delta, targets_delta = self.set_input_targets(delta)
        targets = [targets_delta]

        if self.config.use_intensity:
            intensity = nmr_idx[1]
            # print(intensity)
            inputs_intensity, targets_intensity = self.set_input_targets(intensity)
            targets.append(targets_intensity)

        device = inputs_delta.device
        b, t = inputs_delta.size()
        b_p, p, _ = prompt.size()

        if b_p != b:
            prompt = prompt.repeat(b, 1, 1)

        if self.config.start_generate_token:
            start_token = torch.full((b, 1), self.config.block_size - 1, dtype=torch.long, device=device)
            start_emb = self.transformer.wpe(start_token)
            prompt = torch.cat([prompt, start_emb], dim=1)
            p += 1

        x = self.transformer.delta_embedding(inputs_delta)
        if self.config.use_intensity:
            intensity_emb = self.transformer.intensity_embedding(inputs_intensity)
            x = x + intensity_emb

        # print(1, x.shape)
        if self.config.prompt_pe:
            pos = torch.arange(0, t+p, dtype=torch.long, device=device) # shape (t)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)

        if not self.config.rotary:
            # print(2, x.shape, t, p, self.transformer.wpe(pos).shape)
            x = self.transformer.drop(x + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(x)

        if not self.config.prompt_pe:
            x = torch.cat([prompt, x], dim=1)

        for block in self.transformer.h:
            x = block(x)


        if training:
            x = self.transformer.ln_f(x)[:, p:]

            logits = [self.heads['delta'](x)]
            if self.config.use_intensity:
                logits.append(self.heads['intensity'](x))

            losses = []
            keys = ['delta'] if not self.config.use_intensity else ['delta', 'intensity']
            for i, (key, logit) in enumerate(zip(keys, logits)):
                loss_fn = self.wce_loss[key] if self.config.use_wce_loss else self.ce_loss
                loss = loss_fn(logit.view(-1, logit.size(-1)), targets[i].contiguous().view(-1))
                losses.append(loss * getattr(self.config, f'{key}_weight'))
            loss = sum(losses)
            return logits, loss
        else:
            x = self.transformer.ln_f(x)[:, -1:]
            logits = [self.heads['delta'](x)]
            if self.config.use_intensity:
                logits.append(self.heads['intensity'](x))
            predictions = [torch.argmax(logit, dim=-1).long() for logit in logits]
            return predictions, None


    @torch.no_grad()
    def generate(self, ids: Tuple[torch.Tensor, ...], max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> List[torch.Tensor]:
        self.validate_nmr_idx(ids)
        ids_next = [ids[0].clone()]
        if self.config.use_intensity:
            ids_next.append(ids[1].clone())
        device = ids[0].device

        for _ in range(max_new_tokens):
            ids_cond = [idx[:, -self.config.block_size:] for idx in ids_next]
            logits, _ = self.forward(ids_cond)
            idx_next = []
            for logit in logits:
                logit = logit[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logit, min(top_k, logit.size(-1)))
                    logit[logit < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logit, dim=-1)
                idx_next.append(torch.multinomial(probs, num_samples=1))
            ids_next = [torch.cat([ids_next[i], idx], dim=1) for i, idx in enumerate(idx_next)]
        return ids_next

    @torch.no_grad()
    def generate_with_prompt(self, prompt: torch.Tensor, max_new_tokens: int) -> List[torch.Tensor]:
        b, p, _ = prompt.size()
        device = prompt.device
        num_tensors = 2 if self.config.use_intensity else 1
        nmr_idx = torch.empty(num_tensors, b, 0, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            predictions, _ = self.forward_with_prompt(nmr_idx, prompt, training=False)

            nmr_idx = torch.cat([nmr_idx, torch.stack(predictions, dim=0)[:, :, -1:]], dim=2)
        return [nmr_idx[i] for i in range(num_tensors)]

    def crop_block_size(self, block_size: int):
        assert block_size <= self.config.block_size, "New block size must be smaller"
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[dict] = None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)

        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained GPT: {model_type}")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args.update({
            'block_size': 1024,
            'bias': True,
            'delta': 1025,
            'intensity': 150,
            'use_intensity': True,
        })
        if 'dropout' in override_args:
            print(f"Overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(f'gpt2-{model_type}')
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        assert len(sd_keys_hf) == len(sd_keys), f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str) -> torch.optim.Optimizer:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        print(f"Num decayed params: {len(decay_params)}, {sum(p.numel() for p in decay_params):,} parameters")
        print(f"Num non-decayed params: {len(nodecay_params)}, {sum(p.numel() for p in nodecay_params):,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        print(f"Using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 bfloat16 peak FLOPS
        return flops_achieved / flops_promised

if __name__ == '__main__':
    config = GPTConfig(use_intensity=False)  # Example with intensity disabled
    model = GPT(config)
    nmr_idx = (torch.randint(IGNORE_INDEX, config.delta, (2, 10), dtype=torch.long),)  # Only delta when use_intensity=False
    logits, loss = model(nmr_idx)
    print(f"Logits shapes: {[l.shape for l in logits]}")
    print(f"Loss: {loss.item() if loss is not None else None}")