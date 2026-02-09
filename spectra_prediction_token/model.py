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

class HNMRFeatures:
    CENTROID = 0
    NH = 1
    CATEGORY = 2
    JVALUE = 3

PADDING_TOKEN = 2

@dataclass
class GPTConfig:
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
    use_categories_jvalue: bool = True  # New flag to control category and jvalue usage

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert config.block_size is not None
        assert config.centroid > PADDING_TOKEN and config.nh > PADDING_TOKEN, \
            "Centroid and nh vocab sizes must exceed padding token index"
        if config.use_categories_jvalue:
            assert config.category > PADDING_TOKEN and config.jvalue > PADDING_TOKEN, \
                "Category and jvalue vocab sizes must exceed padding token index"

        # Initialize transformer components
        transformer_dict = {
            'centroid_embedding': nn.Embedding(config.centroid, config.n_embd),
            'nh_embedding': nn.Embedding(config.nh, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        }
        if config.use_categories_jvalue:
            transformer_dict.update({
                'category_embedding': nn.Embedding(config.category, config.n_embd),
                'jvalue_embedding': nn.Embedding(config.jvalue, config.n_embd),
            })
        self.transformer = nn.ModuleDict(transformer_dict)

        # Initialize heads
        heads_dict = {
            'centroid': nn.Linear(config.n_embd, config.centroid, bias=False),
            'nh': nn.Linear(config.n_embd, config.nh, bias=False),
        }
        if config.use_categories_jvalue:
            heads_dict.update({
                'category': nn.Linear(config.n_embd, config.category, bias=False),
                'jvalue': nn.Linear(config.n_embd, config.jvalue, bias=False),
            })
        self.heads = nn.ModuleDict(heads_dict)

        # Weight tying
        for key in ['centroid', 'nh'] + (['category', 'jvalue'] if config.use_categories_jvalue else []):
            self.heads[key].weight = self.transformer[f'{key}_embedding'].weight

        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=PADDING_TOKEN)
        if config.use_wce_loss:
            self.wce_loss = nn.ModuleDict({
                'centroid': ToeplitzWeightedCrossEntropyLoss(num_classes=config.centroid, W=config.centroid, ignore_index=PADDING_TOKEN),
                'nh': ToeplitzWeightedCrossEntropyLoss(num_classes=config.nh, W=config.nh, ignore_index=PADDING_TOKEN),
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
            for key in ['centroid', 'nh'] + (['category', 'jvalue'] if self.config.use_categories_jvalue else []):
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
        expected_len = 4 if self.config.use_categories_jvalue else 2
        # assert len(hnmr_idx) == expected_len, f"hnmr_idx must contain {expected_len} tensors"
        centroid, nh = hnmr_idx[:2]
        if self.config.use_categories_jvalue:
            category, jvalue = hnmr_idx[2:]
            tensors = [centroid, nh, category, jvalue]
        else:
            tensors = [centroid, nh]
        assert all(s.size() == centroid.size() for s in tensors), "All sequences must have same shape"
        assert centroid.size(1) <= self.config.block_size, f"Sequence length {centroid.size(1)} exceeds block size"
        assert all(s.device == centroid.device for s in tensors), "Device mismatch"

    def set_input_targets(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return seq[:, :-1], seq[:, 1:]

    # def forward_to_embeds(self, hnmr_idx: Tuple[torch.Tensor, ...], only_embeds=True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    #     self.validate_hnmr_idx(hnmr_idx)
    #     centroid, nh = hnmr_idx[:2]
    #     inputs_centroid, targets_centroid = self.set_input_targets(centroid)
    #     inputs_nh, targets_nh = self.set_input_targets(nh)
    #     targets = [targets_centroid, targets_nh]
    #
    #     if self.config.use_categories_jvalue:
    #         category, jvalue = hnmr_idx[2:]
    #         inputs_category, targets_category = self.set_input_targets(category)
    #         inputs_jvalue, targets_jvalue = self.set_input_targets(jvalue)
    #         targets.extend([targets_category, targets_jvalue])
    #
    #     device = inputs_centroid.device
    #     b, t = inputs_centroid.size()
    #
    #     centroid_emb = self.transformer.centroid_embedding(inputs_centroid)
    #     nh_emb = self.transformer.nh_embedding(inputs_nh)
    #     x = centroid_emb + nh_emb
    #     if self.config.use_categories_jvalue:
    #         category_emb = self.transformer.category_embedding(inputs_category)
    #         jvalue_emb = self.transformer.jvalue_embedding(inputs_jvalue)
    #         x = x + category_emb + jvalue_emb
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
    #
    #     if only_embeds:
    #         return x
    #     else:
    #         return x, targets

    def forward_to_embeds(self, hnmr_idx: Tuple[torch.Tensor, ...], only_embeds=True) -> Tuple[
        torch.Tensor, List[torch.Tensor]]:
        self.validate_hnmr_idx(hnmr_idx)
        centroid, nh = hnmr_idx[:2]
        inputs_centroid, targets_centroid = self.set_input_targets(centroid)
        inputs_nh, targets_nh = self.set_input_targets(nh)
        targets = [targets_centroid, targets_nh]
        attn_mask = (inputs_centroid != PADDING_TOKEN)[:, None, None, :]
        if self.config.use_categories_jvalue:
            category, jvalue = hnmr_idx[2:]
            inputs_category, targets_category = self.set_input_targets(category)
            inputs_jvalue, targets_jvalue = self.set_input_targets(jvalue)
            targets.extend([targets_category, targets_jvalue])

        device = inputs_centroid.device
        b, t = inputs_centroid.size()

        centroid_emb = self.transformer.centroid_embedding(inputs_centroid)
        nh_emb = self.transformer.nh_embedding(inputs_nh)
        x = centroid_emb + nh_emb
        if self.config.use_categories_jvalue:
            category_emb = self.transformer.category_embedding(inputs_category)
            jvalue_emb = self.transformer.jvalue_embedding(inputs_jvalue)
            x = x + category_emb + jvalue_emb

        if not self.config.rotary:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = self.transformer.drop(x + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x,attn_mask=attn_mask)
        x = self.transformer.ln_f(x)

        if only_embeds:
            return x
        else:
            return x, targets

    def forward(self, hnmr_idx: Tuple[torch.Tensor, ...], targets: Optional[List[torch.Tensor]] = None) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        x, targets = self.forward_to_embeds(hnmr_idx, only_embeds=False)
        keys = ['centroid', 'nh'] if not self.config.use_categories_jvalue else ['centroid', 'nh', 'category', 'jvalue']
        logits = [self.heads[key](x) for key in keys[:2]]  # Only centroid and nh logits

        if targets is not None:
            losses = []
            for i, (key, logit) in enumerate(zip(['centroid', 'nh'], logits)):
                loss_fn = self.wce_loss[key] if self.config.use_wce_loss and key in ['centroid', 'nh'] else self.ce_loss
                loss = loss_fn(logit.view(-1, logit.size(-1)), targets[i].contiguous().view(-1))
                losses.append(loss * getattr(self.config, f'{key}_weight'))
            loss = sum(losses)
            return logits, loss
        return logits, None

    def forward_with_prompt(self, hnmr_idx: Tuple[torch.Tensor, ...], prompt: torch.Tensor, training: bool = False) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        self.validate_hnmr_idx(hnmr_idx)
        centroid, nh = hnmr_idx[:2]
        inputs_centroid, targets_centroid = self.set_input_targets(centroid)
        inputs_nh, targets_nh = self.set_input_targets(nh)
        targets = [targets_centroid, targets_nh]
        if self.config.use_categories_jvalue:
            category, jvalue = hnmr_idx[2:]
            inputs_category, targets_category = self.set_input_targets(category)
            inputs_jvalue, targets_jvalue = self.set_input_targets(jvalue)
            targets.extend([targets_category, targets_jvalue])


        device = inputs_centroid.device
        b, t = inputs_centroid.size()
        b_p, p, _ = prompt.size()

        # print(centroid.shape,t,p)
        if b_p != b:
            prompt = prompt.repeat(b, 1, 1)

        if self.config.start_generate_token:
            start_token = torch.full((b, 1), self.config.block_size - 1, dtype=torch.long, device=device)
            start_emb = self.transformer.wpe(start_token)
            prompt = torch.cat([prompt, start_emb], dim=1)
            p += 1

        centroid_emb = self.transformer.centroid_embedding(inputs_centroid)
        nh_emb = self.transformer.nh_embedding(inputs_nh)
        x = centroid_emb + nh_emb
        if self.config.use_categories_jvalue:
            category_emb = self.transformer.category_embedding(inputs_category)
            jvalue_emb = self.transformer.jvalue_embedding(inputs_jvalue)
            x = x + category_emb + jvalue_emb

        # print(x.shape,t,p,self.config.prompt_pe)
        if self.config.prompt_pe:
            x = torch.cat([prompt, x], dim=1)
            pos = torch.arange(0, t+p, dtype=torch.long, device=device)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
        # print(1,x.shape)
        if not self.config.rotary:
            # print(2, x.shape)
            x = self.transformer.drop(x + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(x)

        if not self.config.prompt_pe:
            x = torch.cat([prompt, x], dim=1)

        for block in self.transformer.h:
            x = block(x)

        if training:
            x = self.transformer.ln_f(x)[:, p:]
            keys = ['centroid', 'nh'] if not self.config.use_categories_jvalue else ['centroid', 'nh', 'category', 'jvalue']
            logits = [self.heads[key](x) for key in keys]

            losses = []
            for i, (key, logit) in enumerate(zip(keys, logits)):
                loss_fn = self.wce_loss[key] if self.config.use_wce_loss and key in ['centroid', 'nh'] else self.ce_loss
                loss = loss_fn(logit.view(-1, logit.size(-1)), targets[i].contiguous().view(-1))
                losses.append(loss * getattr(self.config, f'{key}_weight'))
            loss = sum(losses)
            return logits, loss
        else:
            x = self.transformer.ln_f(x)[:, -1:]
            keys = ['centroid', 'nh'] if not self.config.use_categories_jvalue else ['centroid', 'nh', 'category', 'jvalue']
            logits = [self.heads[key](x) for key in keys]
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
        num_tensors = 4 if self.config.use_categories_jvalue else 2
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
            'centroid': 200,
            'nh': 200,
            'category': 200,
            'jvalue': 200,
            'use_categories_jvalue': True,
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
    config = GPTConfig(use_categories_jvalue=False)  # Example with categories and jvalue disabled
    model = GPT(config)
    hnmr_idx = tuple(torch.randint(PADDING_TOKEN, config.__dict__[key], (2, 10), dtype=torch.long)
                     for key in ['centroid', 'nh'])
    logits, loss = model(hnmr_idx)
    print(f"Logits shapes: {[l.shape for l in logits]}")
    print(f"Loss: {loss.item() if loss is not None else None}")