import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List, Optional
from spectra_prediction_token.Weighted_CrossEntropyLoss import ToeplitzWeightedCrossEntropyLoss

# Assuming RotaryTransformer and Block are defined elsewhere
from MolRotaryTransformer import Block, LayerNorm

class HNMRFeatures:
    CENTROID = 0
    NH = 1
    CATEGORY = 2
    JVALUE = 3

PADDING_TOKEN = 2
IGNORE_INDEX = 2

@dataclass
class HNMR_GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    rotary: bool = False
    dropout: float = 0.0
    centroid: int = 200
    nh: int = 200
    category: int = 200
    jvalue: int = 200
    bias: bool = True
    prompt_pe: bool = True
    start_generate_token: bool = False
    use_wce_loss: bool = False
    centroid_weight: float = 2.0
    nh_weight: float = 2.0
    category_weight: float = 0.2
    jvalue_weight: float = 0.2

class HNMR_GPT(nn.Module):
    def __init__(self, config: HNMR_GPTConfig):
        super().__init__()
        self.config = config
        assert config.block_size is not None
        assert all(config.__dict__[k] > PADDING_TOKEN for k in ['centroid', 'nh', 'category', 'jvalue']), \
            "Vocab sizes must exceed padding token index"

        self.transformer = nn.ModuleDict({
            'centroid_embedding': nn.Embedding(config.centroid, config.n_embd),
            'nh_embedding': nn.Embedding(config.nh, config.n_embd),
            'category_embedding': nn.Embedding(config.category, config.n_embd),
            'jvalue_embedding': nn.Embedding(config.jvalue, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        })
        self.heads = nn.ModuleDict({
            'centroid': nn.Linear(config.n_embd, config.centroid, bias=False),
            'nh': nn.Linear(config.n_embd, config.nh, bias=False),
            'category': nn.Linear(config.n_embd, config.category, bias=False),
            'jvalue': nn.Linear(config.n_embd, config.jvalue, bias=False),
        })
        # Weight tying
        for key in ['centroid', 'nh', 'category', 'jvalue']:
            self.heads[key].weight = self.transformer[f'{key}_embedding'].weight

        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=PADDING_TOKEN)
        if config.use_wce_loss:
            self.wce_loss = nn.ModuleDict({
                'centroid': ToeplitzWeightedCrossEntropyLoss(num_classes=config.centroid, W=config.centroid, ignore_index=PADDING_TOKEN),
                'nh': ToeplitzWeightedCrossEntropyLoss(num_classes=config.nh, W=config.nh, ignore_index=PADDING_TOKEN),
                # category and jvalue use standard CE loss as they are replaceable
            })

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            for key in ['centroid', 'nh', 'category', 'jvalue']:
                n_params -= self.transformer[f'{key}_embedding'].weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def validate_hnmr_idx(self, hnmr_idx: Tuple[torch.Tensor, ...]):
        assert len(hnmr_idx) == 4, "hnmr_idx must contain centroid, nh, category, and jvalue"
        centroid, nh, category, jvalue = hnmr_idx
        assert all(s.size() == centroid.size() for s in [nh, category, jvalue]), "All sequences must have same shape"
        assert centroid.size(1) <= self.config.block_size, f"Sequence length {centroid.size(1)} exceeds block size"
        assert all(s.device == centroid.device for s in [nh, category, jvalue]), "Device mismatch"

    def set_input_targets(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return seq[:, :-1], seq[:, 1:]

    def forward_to_embeds(self, hnmr_idx: Tuple[torch.Tensor, ...],
                          only_embeds=True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.validate_hnmr_idx(hnmr_idx)
        centroid, nh, category, jvalue = hnmr_idx
        inputs_centroid, targets_centroid = self.set_input_targets(centroid)
        inputs_nh, targets_nh = self.set_input_targets(nh)
        inputs_category, targets_category = self.set_input_targets(category)
        inputs_jvalue, targets_jvalue = self.set_input_targets(jvalue)

        device = inputs_centroid.device
        b, t = inputs_centroid.size()

        centroid_emb = self.transformer.centroid_embedding(inputs_centroid)
        nh_emb = self.transformer.nh_embedding(inputs_nh)
        category_emb = self.transformer.category_embedding(inputs_category)
        jvalue_emb = self.transformer.jvalue_embedding(inputs_jvalue)
        x = centroid_emb + nh_emb + category_emb + jvalue_emb

        if not self.config.rotary:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = self.transformer.drop(x + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if only_embeds:
            return x
        else:
            return x, [targets_centroid, targets_nh, targets_category, targets_jvalue]

    def forward(self, hnmr_idx: Tuple[torch.Tensor, ...], targets: Optional[List[torch.Tensor]] = None) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        x, targets = self.forward_to_embeds(hnmr_idx, only_embeds=False)
        logits = [self.heads[key](x) for key in ['centroid', 'nh', 'category', 'jvalue']]

        if targets is not None:
            losses = []
            for i, (key, logit) in enumerate(zip(['centroid', 'nh', 'category', 'jvalue'], logits)):
                # Use weighted CE for centroid and nh, standard CE for category and jvalue
                loss_fn = self.wce_loss[key] if self.config.use_wce_loss and key in ['centroid', 'nh'] else self.ce_loss
                loss = loss_fn(logit.view(-1, logit.size(-1)), targets[i].contiguous().view(-1))
                losses.append(loss * getattr(self.config, f'{key}_weight'))
            loss = sum(losses)
            return logits, loss
        return logits, None

    def forward_with_prompt(self, hnmr_idx: Tuple[torch.Tensor, ...], prompt: torch.Tensor, training: bool = False) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        self.validate_hnmr_idx(hnmr_idx)
        centroid, nh, category, jvalue = hnmr_idx
        inputs_centroid, targets_centroid = self.set_input_targets(centroid)
        inputs_nh, targets_nh = self.set_input_targets(nh)
        inputs_category, targets_category = self.set_input_targets(category)
        inputs_jvalue, targets_jvalue = self.set_input_targets(jvalue)
        targets = [targets_centroid, targets_nh, targets_category, targets_jvalue]

        device = inputs_centroid.device
        b, t = inputs_centroid.size()
        b_p, p, _ = prompt.size()

        if b_p != b:
            prompt = prompt.repeat(b, 1, 1)

        if self.config.start_generate_token:
            start_token = torch.full((b, 1), self.config.block_size - 1, dtype=torch.long, device=device)
            start_emb = self.transformer.wpe(start_token)
            prompt = torch.cat([prompt, start_emb], dim=1)
            p += 1

        centroid_emb = self.transformer.centroid_embedding(inputs_centroid)
        nh_emb = self.transformer.nh_embedding(inputs_nh)
        category_emb = self.transformer.category_embedding(inputs_category)
        jvalue_emb = self.transformer.jvalue_embedding(inputs_jvalue)
        x = centroid_emb + nh_emb + category_emb + jvalue_emb

        if self.config.prompt_pe:
            pos = torch.arange(0, t+p, dtype=torch.long, device=device) # shape (t)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)

        if not self.config.rotary:
            x = self.transformer.drop(x + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(x)

        if not self.config.prompt_pe:
            x = torch.cat([prompt, x], dim=1)

        for block in self.transformer.h:
            x = block(x)
        # print(x.shape)
        if training:
            x = self.transformer.ln_f(x)[:, p:]

            logits = [self.heads[key](x) for key in ['centroid', 'nh', 'category', 'jvalue']]

            losses = []
            for i, (key, logit) in enumerate(zip(['centroid', 'nh', 'category', 'jvalue'], logits)):
                loss_fn = self.wce_loss[key] if self.config.use_wce_loss and key in ['centroid', 'nh'] else self.ce_loss
                loss = loss_fn(logit.view(-1, logit.size(-1)), targets[i].contiguous().view(-1))
                losses.append(loss * getattr(self.config, f'{key}_weight'))
            loss = sum(losses)
            return logits, loss
        else:
            x = self.transformer.ln_f(x)[:, -1:]

            logits = [self.heads[key](x) for key in ['centroid', 'nh', 'category', 'jvalue']]

            predictions = [torch.argmax(logit, dim=-1).long() for logit in logits]
            return predictions, None

    @torch.no_grad()
    def generate(self, ids: Tuple[torch.Tensor, ...], max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> List[torch.Tensor]:
        self.validate_hnmr_idx(ids)
        ids_next = [idx.clone() for idx in ids]
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
        nmr_idx = torch.empty(4, b, 0, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            predictions, _ = self.forward_with_prompt(nmr_idx, prompt, training=False)
            # print(predictions)
            nmr_idx = torch.cat([nmr_idx, torch.stack(predictions, dim=0)[:, :, -1:]], dim=2)

        return [nmr_idx[i] for i in range(4)]

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
            'centroid': 200,
            'nh': 200,
            'category': 200,
            'jvalue': 200,
        })
        if 'dropout' in override_args:
            print(f"Overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        config = HNMR_GPTConfig(**config_args)
        model = HNMR_GPT(config)
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

@dataclass
class CNMR_GPTConfig:
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

class CNMR_NMRFeatures:
    DELTA = 0
    INTENSITY = 1

class CNMR_GPT(nn.Module):
    def __init__(self, config: CNMR_GPTConfig):
        super().__init__()
        self.config = config
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

    def forward_to_embeds(self, nmr_idx: Tuple[torch.Tensor, ...],only_embeds:bool=True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.validate_nmr_idx(nmr_idx)
        delta = nmr_idx[0]
        inputs_delta, targets_delta = self.set_input_targets(delta)
        targets = [targets_delta]

        device = inputs_delta.device
        b, t = inputs_delta.size()

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
            x = block(x)
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

        if self.config.prompt_pe:
            pos = torch.arange(0, t+p, dtype=torch.long, device=device) # shape (t)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)

        if not self.config.rotary:
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

        config = CNMR_GPTConfig(**config_args)
        model = CNMR_GPT(config)
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



@dataclass
class HSQCNMR_GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    c13_centroid: int = 1025
    h1_centroid: int = 300
    nh: int = 300
    rotary: bool = False
    bias: bool = True
    prompt_pe: bool = True
    start_generate_token: bool = False
    use_wce_loss: bool = False
    c13_centroid_weight: float = 2.0
    h1_centroid_weight: float = 2.0
    nh_weight: float = 2.0

class HSQCFeatures:
    C13_CENTROID = 0
    H1_CENTROID = 1
    NH = 2

class HSQCNMR_GPT(nn.Module):
    def __init__(self, config: HSQCNMR_GPTConfig):
        super().__init__()
        self.config = config
        assert config.block_size is not None
        assert config.c13_centroid > PADDING_TOKEN, "C13_centroid vocab size must exceed padding token"
        assert config.h1_centroid > PADDING_TOKEN, "H1_centroid vocab size must exceed padding token"
        assert config.nh > PADDING_TOKEN, "nH vocab size must exceed padding token"

        self.transformer = nn.ModuleDict({
            'c13_centroid_embedding': nn.Embedding(config.c13_centroid, config.n_embd),
            'h1_centroid_embedding': nn.Embedding(config.h1_centroid, config.n_embd),
            'nh_embedding': nn.Embedding(config.nh, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        })

        self.heads = nn.ModuleDict({
            'c13_centroid': nn.Linear(config.n_embd, config.c13_centroid, bias=False),
            'h1_centroid': nn.Linear(config.n_embd, config.h1_centroid, bias=False),
            'nh': nn.Linear(config.n_embd, config.nh, bias=False),
        })

        # Weight tying
        self.heads['c13_centroid'].weight = self.transformer['c13_centroid_embedding'].weight
        self.heads['h1_centroid'].weight = self.transformer['h1_centroid_embedding'].weight
        self.heads['nh'].weight = self.transformer['nh_embedding'].weight

        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=PADDING_TOKEN)
        if config.use_wce_loss:
            self.wce_loss = nn.ModuleDict({
                'c13_centroid': ToeplitzWeightedCrossEntropyLoss(num_classes=config.c13_centroid, W=config.c13_centroid, ignore_index=PADDING_TOKEN),
                'h1_centroid': ToeplitzWeightedCrossEntropyLoss(num_classes=config.h1_centroid, W=config.h1_centroid, ignore_index=PADDING_TOKEN),
                'nh': ToeplitzWeightedCrossEntropyLoss(num_classes=config.nh, W=config.nh, ignore_index=PADDING_TOKEN),
            })

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            n_params -= self.transformer.c13_centroid_embedding.weight.numel()
            n_params -= self.transformer.h1_centroid_embedding.weight.numel()
            n_params -= self.transformer.nh_embedding.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def validate_hsqc_idx(self, hsqc_idx: Tuple[torch.Tensor, ...]):
        assert len(hsqc_idx) == 3, "hsqc_idx must contain 3 tensors (c13_centroid, h1_centroid, nh)"
        c13_centroid, h1_centroid, nh = hsqc_idx
        for s in [c13_centroid, h1_centroid, nh]:
            assert s.size() == c13_centroid.size(), "All sequences must have same shape"
            assert s.device == c13_centroid.device, "Device mismatch"
        assert c13_centroid.size(1) <= self.config.block_size, f"Sequence length {c13_centroid.size(1)} exceeds block size"

    def set_input_targets(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return seq[:, :-1], seq[:, 1:]

    def forward_to_embeds(self, hsqc_idx: Tuple[torch.Tensor, ...],
                          only_embeds = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.validate_hsqc_idx(hsqc_idx)
        c13_centroid, h1_centroid, nh = hsqc_idx

        inputs_c13_centroid, targets_c13_centroid = self.set_input_targets(c13_centroid)
        inputs_h1_centroid, targets_h1_centroid = self.set_input_targets(h1_centroid)
        inputs_nh, targets_nh = self.set_input_targets(nh)
        targets = [targets_c13_centroid, targets_h1_centroid, targets_nh]

        x = (self.transformer.c13_centroid_embedding(inputs_c13_centroid) +
             self.transformer.h1_centroid_embedding(inputs_h1_centroid) +
             self.transformer.nh_embedding(inputs_nh))

        device = inputs_c13_centroid.device
        b, t = inputs_c13_centroid.size()
        if not self.config.rotary:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = self.transformer.drop(x + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if only_embeds:
            return x
        else:
            return x, targets

    def forward(self, hsqc_idx: Tuple[torch.Tensor, ...], targets: Optional[List[torch.Tensor]] = None) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        x, targets = self.forward_to_embeds(hsqc_idx, only_embeds = False)
        logits = [self.heads['c13_centroid'](x), self.heads['h1_centroid'](x), self.heads['nh'](x)]
        keys = ['c13_centroid', 'h1_centroid', 'nh']

        if targets is not None:
            losses = []
            for i, (key, logit) in enumerate(zip(keys, logits)):
                loss_fn = self.wce_loss[key] if self.config.use_wce_loss else self.ce_loss
                loss = loss_fn(logit.view(-1, logit.size(-1)), targets[i].contiguous().view(-1))
                losses.append(loss * getattr(self.config, f'{key}_weight'))
            loss = sum(losses)
            return logits, loss
        return logits, None

    def forward_with_prompt(self, hsqc_idx: Tuple[torch.Tensor, ...], prompt: torch.Tensor, training: bool = False) -> \
    Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        self.validate_hsqc_idx(hsqc_idx)
        c13_centroid, h1_centroid, nh = hsqc_idx

        # Prepare inputs and targets
        inputs_c13_centroid, targets_c13_centroid = self.set_input_targets(c13_centroid)
        inputs_h1_centroid, targets_h1_centroid = self.set_input_targets(h1_centroid)
        inputs_nh, targets_nh = self.set_input_targets(nh)
        targets = [targets_c13_centroid, targets_h1_centroid, targets_nh]

        # Get device and shapes
        device = inputs_c13_centroid.device
        b, t = inputs_c13_centroid.size()
        b_p, p, _ = prompt.size()

        # Adjust prompt batch size if needed
        if b_p != b:
            prompt = prompt.repeat(b, 1, 1)

        # Add start token if enabled
        if self.config.start_generate_token:
            start_token = torch.full((b, 1), self.config.block_size - 1, dtype=torch.long, device=device)
            start_emb = self.transformer.wpe(start_token)
            prompt = torch.cat([prompt, start_emb], dim=1)
            p += 1

        # Compute embeddings
        x = (self.transformer.c13_centroid_embedding(inputs_c13_centroid) +
             self.transformer.h1_centroid_embedding(inputs_h1_centroid) +
             self.transformer.nh_embedding(inputs_nh))

        if self.config.prompt_pe:
            pos = torch.arange(0, t+p, dtype=torch.long, device=device) # shape (t)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)

        if not self.config.rotary:
            x = self.transformer.drop(x + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(x)

        if not self.config.prompt_pe:
            x = torch.cat([prompt, x], dim=1)

        # Transformer forward pass
        for block in self.transformer.h:
            x = block(x)

        # Training or inference
        if training:
            x = self.transformer.ln_f(x)[:, p:]
            # Compute logits
            logits = [
                self.heads['c13_centroid'](x),
                self.heads['h1_centroid'](x),
                self.heads['nh'](x)
            ]
            keys = ['c13_centroid', 'h1_centroid', 'nh']

            losses = []
            for i, (key, logit) in enumerate(zip(keys, logits)):
                loss_fn = self.wce_loss[key] if self.config.use_wce_loss else self.ce_loss
                loss = loss_fn(logit.view(-1, logit.size(-1)), targets[i].contiguous().view(-1))
                losses.append(loss * getattr(self.config, f'{key}_weight'))
            loss = sum(losses)
            return logits, loss
        else:
            x = self.transformer.ln_f(x)[:, -1:]
            # Compute logits
            logits = [
                self.heads['c13_centroid'](x),
                self.heads['h1_centroid'](x),
                self.heads['nh'](x)
            ]
            keys = ['c13_centroid', 'h1_centroid', 'nh']

            predictions = [torch.argmax(logit, dim=-1).long() for logit in logits]
            return predictions, None

    def forward_with_prompt_duringtrain(self, hsqc_idx: Tuple[torch.Tensor, ...], prompt: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        self.validate_hsqc_idx(hsqc_idx)
        c13_centroid, h1_centroid, nh = hsqc_idx

        inputs_c13_centroid, targets_c13_centroid = self.set_input_targets(c13_centroid)
        inputs_h1_centroid, targets_h1_centroid = self.set_input_targets(h1_centroid)
        inputs_nh, targets_nh = self.set_input_targets(nh)
        targets = [targets_c13_centroid, targets_h1_centroid, targets_nh]

        x = (self.transformer.c13_centroid_embedding(inputs_c13_centroid) +
             self.transformer.h1_centroid_embedding(inputs_h1_centroid) +
             self.transformer.nh_embedding(inputs_nh))

        device = inputs_c13_centroid.device
        b, t = inputs_c13_centroid.size()
        b_p, p, _ = prompt.size()
        if  b_p != b:
            prompt = prompt.repeat(b, 1, 1)

        if self.config.start_generate_token:
            start_token = torch.full((b, 1), self.config.block_size - 1, dtype=torch.long, device=device)
            start_emb = self.transformer.wpe(start_token)
            prompt = torch.cat((prompt, start_emb), dim=1)
            p += 1

        if self.config.prompt_pe:
            x = torch.cat([prompt, x], dim=1)
            pos = torch.arange(0, t + p, dtype=torch.long, device=device) if not self.config.rotary else None
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) if not self.config.rotary else None
            x = torch.cat([prompt, x], dim=1)

        if not self.config.rotary and pos is not None:
            x = self.transformer.drop(x + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)[:, p:]

        logits = [self.heads['c13_centroid'](x), self.heads['h1_centroid'](x), self.heads['nh'](x)]
        keys = ['c13_centroid', 'h1_centroid', 'nh']

        losses = []
        for i, (key, logit) in enumerate(zip(keys, logits)):
            loss_fn = self.wce_loss[key] if self.config.use_wce_loss else self.ce_loss
            loss = loss_fn(logit.view(-1, logit.size(-1)), targets[i].contiguous().view(-1))
            losses.append(loss * getattr(self.config, f'{key}_weight'))
        loss = sum(losses)
        return logits, loss

    # @torch.no_grad()
    # def forward_with_prompt(self, hsqc_idx: torch.Tensor, prompt: torch.Tensor) -> List[torch.Tensor]:
    #     b, p, _ = prompt.size()
    #     device = prompt.device
    #     b_idx, t = hsqc_idx.size()[1:3] if hsqc_idx.size(2) > 0 else (b, 0)
    #
    #     if b_idx != b:
    #         prompt = prompt.repeat(b_idx, 1, 1)
    #
    #     if self.config.start_generate_token:
    #         start_token = torch.full((b_idx, 1), self.config.block_size - 1, dtype=torch.long, device=device)
    #         start_emb = self.transformer.wpe(start_token)
    #         prompt = torch.cat((prompt, start_emb), dim=1)
    #         p += 1
    #
    #     if t == 0:
    #         x = prompt
    #     else:
    #         self.validate_hsqc_idx(hsqc_idx)
    #         c13_centroid, h1_centroid, nh = hsqc_idx
    #
    #         x = (self.transformer.c13_centroid_embedding(c13_centroid) +
    #              self.transformer.h1_centroid_embedding(h1_centroid) +
    #              self.transformer.nh_embedding(nh))
    #
    #         if self.config.prompt_pe:
    #             x = torch.cat([prompt, x], dim=1)
    #             pos = torch.arange(0, p + t, dtype=torch.long, device=device) if not self.config.rotary else None
    #         else:
    #             pos = torch.arange(0, t, dtype=torch.long, device=device) if not self.config.rotary else None
    #             x = torch.cat([prompt, x], dim=1)
    #
    #         if not self.config.rotary and pos is not None:
    #             x = x + self.transformer.wpe(pos)
    #
    #     x = self.transformer.drop(x)
    #     for block in self.transformer.h:
    #         x = block(x)
    #     x = self.transformer.ln_f(x)[:, -1:]
    #
    #     logits = [self.heads['c13_centroid'](x), self.heads['h1_centroid'](x), self.heads['nh'](x)]
    #     return [torch.argmax(logit, dim=-1) for logit in logits]

    @torch.no_grad()
    def generate(self, ids: Tuple[torch.Tensor, ...], max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> List[torch.Tensor]:
        self.validate_hsqc_idx(ids)
        ids_next = [idx.clone() for idx in ids]
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

    # @torch.no_grad()
    # def generate_with_prompt(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    #     b, p, _ = prompt.size()
    #     device = prompt.device
    #     hsqc_idx = torch.empty(3, b, 0, dtype=torch.long, device=device)
    #
    #     for _ in range(max_new_tokens):
    #         predictions = self.forward_with_prompt(hsqc_idx, prompt)
    #         hsqc_idx = torch.cat([hsqc_idx, torch.stack(predictions, dim=0)[:, :, -1:]], dim=2)
    #     return hsqc_idx


    @torch.no_grad()
    def generate_with_prompt(self, prompt: torch.Tensor, max_new_tokens: int) -> List[torch.Tensor]:
        b, p, _ = prompt.size()
        device = prompt.device
        nmr_idx = torch.empty(3, b, 0, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            predictions, _ = self.forward_with_prompt(nmr_idx, prompt, training=False)
            # print(predictions)
            nmr_idx = torch.cat([nmr_idx, torch.stack(predictions, dim=0)[:, :, -1:]], dim=2)

        return [nmr_idx[i] for i in range(3)]

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
        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args.update({
            'block_size': 1024,
            'bias': True,
            'c13_centroid': 1025,
            'h1_centroid': 300,
            'nh': 300,
        })
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        config = HSQCNMR_GPTConfig(**config_args)
        model = HSQCNMR_GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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
        print(f"num decayed params: {len(decay_params)}, {sum(p.numel() for p in decay_params):,} parameters")
        print(f"num non-decayed params: {len(nodecay_params)}, {sum(p.numel() for p in nodecay_params):,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        print(f"using fused AdamW: {use_fused}")
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

