"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
# from  MolRotaryTransformer import *
# from RotaryTransformer import *
from RotaryTransformerAtten import *
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

# from BeamSearch import BeamHypotheses
from RoPE import RotaryEmbedding,rotate_half,apply_rotary_pos_emb
PADDING_TOKEN = 2


def compute_cnmr_attn_loss(attn_weights, cnmr_assig_token, cnmr_assig_spec, lambda_attn=0.05, temperature=0.5):
    """
    attn_weights: (b, heads, seq_mol, seq_spec)
    cnmr_assig_token: list[int]  # token positions
    cnmr_assig_spec: list[int]   # corresponding C_nmr peak indices
    """
    b, heads, seq_mol, seq_spec = attn_weights.shape

    # 1. mean over heads
    attn_mean = attn_weights.mean(dim=1)  # (b, seq_mol, seq_spec)

    # 2. 构建 gold attention map (b, seq_mol, seq_spec)
    gold = torch.zeros(b, seq_mol, seq_spec, device=attn_weights.device)

    for b_idx in range(b):
        # 每个 batch 独立处理
        token_list = cnmr_assig_token[b_idx].long()
        spec_list = cnmr_assig_spec[b_idx].long()
        token_list = torch.clip(token_list, 0, seq_mol - 1)
        spec_list = torch.clip(spec_list, 0, seq_spec - 1)
        # print(token_list,spec_list)
        gold[b_idx,token_list,spec_list] = 1
        gold[b_idx, 0, 0] = 0
        # for t_pos, s_idx in zip(token_list, spec_list):
        #     if 0 <= t_pos < seq_mol and 0 <= s_idx < seq_spec:
        #         gold[b_idx, t_pos, s_idx] = 1.0  # hard assignment

    # 3. 归一化（每行 sum=1）
    gold = gold / (gold.sum(dim=-1, keepdim=True) + 1e-8)

    # 4. KL loss
    attn_logp = F.log_softmax(attn_mean / temperature, dim=-1)
    attn_loss = F.kl_div(attn_logp, gold, reduction='batchmean')

    return lambda_attn * attn_loss

def compute_hnmr_attn_loss(attn_weights, hnmr_assig_token, hnmr_assig_spec, lambda_attn=0.05, temperature=0.5):
    # 氢的逻辑完全相同，只是换名字
    return compute_cnmr_attn_loss(attn_weights, hnmr_assig_token, hnmr_assig_spec, lambda_attn, temperature)

def pad_or_truncate_mol_prompt(mol_prompt, target_length, vocab_size, device):
    b, mp = mol_prompt.size()
    if mp < target_length:
        padding = torch.zeros(b, target_length - mp, dtype=torch.long, device=device)
        mol_prompt = torch.cat([mol_prompt, padding], dim=1)
    elif mp > target_length:
        mol_prompt = mol_prompt[:, :target_length]
    return mol_prompt

# class LayerNorm(nn.Module):
#     """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
#
#     def __init__(self, ndim, bias):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(ndim))
#         self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
#
#     def forward(self, input):
#         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
#
#
# class CausalSelfAttention(nn.Module):
#
#     def __init__(self, config):
#         super().__init__()
#         assert config.n_embd % config.n_head == 0
#         # key, query, value projections for all heads, but in a batch
#         self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
#         # output projection
#         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
#         # regularization
#         self.attn_dropout = nn.Dropout(config.dropout)
#         self.resid_dropout = nn.Dropout(config.dropout)
#         self.n_head = config.n_head
#         self.n_embd = config.n_embd
#         self.dropout = config.dropout
#         # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
#         self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
#         self.rotary = config.rotary
#
#         if config.rotary:
#             self.rotaryemb = RotaryEmbedding(config.n_embd//config.n_head)
#
#         if not self.flash:
#             # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
#             # causal mask to ensure that attention is only applied to the left in the input sequence
#             self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
#                                         .view(1, 1, config.block_size, config.block_size))
#
#     def forward(self, x):
#         B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
#
#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
#         k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#
#         if self.rotary:
#             # apply rotary pos emb before attention
#             cos,sin = self.rotaryemb(q)
#             # print(cos.shape,sin.shape,q.shape,k.shape)
#             q,k = apply_rotary_pos_emb(q,k,cos,sin)
#
#         # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
#         if self.flash:
#             # efficient attention using Flash Attention CUDA kernels
#             y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
#         else:
#             # manual implementation of attention
#             att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#             att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
#             att = F.softmax(att, dim=-1)
#             att = self.attn_dropout(att)
#             y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
#         y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
#
#         # output projection
#         y = self.resid_dropout(self.c_proj(y))
#         return y
#
# class MLP(nn.Module):
#
#     def __init__(self, config):
#         super().__init__()
#         self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
#         self.gelu    = nn.GELU()
#         self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
#         self.dropout = nn.Dropout(config.dropout)
#
#     def forward(self, x):
#         x = self.c_fc(x)
#         x = self.gelu(x)
#         x = self.c_proj(x)
#         x = self.dropout(x)
#         return x
#
# class Block(nn.Module):
#
#     def __init__(self, config):
#         super().__init__()
#         self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
#         self.attn = CausalSelfAttention(config)
#         self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
#         self.mlp = MLP(config)
#
#     def forward(self, x):
#         x = x + self.attn(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 3084
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    rotary: bool = False

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),

        ))

        self.lm_head = nn.Linear(self.config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.lm_head.weight = self.transformer.wte.weight # https://paperswithcode.com/method/weight-tying
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        self.ignore_id = -1
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # def forward(self, idx, targets=None):
    #     device = idx.device
    #     # print(idx.shape)
    #     b, t = idx.size()
    #     assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
    #     pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
    #
    #     # forward the GPT model itself
    #     tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
    #     # self.transformer.spec_wte.weight = self.wte2spec_wte(self.transformer.wte.weight)
    #     # tok_emb = self.spec_wte(idx) # token embeddings of shape (b, t, n_embd)
    #     if not self.config.rotary:
    #         pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
    #         x = self.transformer.drop(tok_emb + pos_emb)
    #     else:
    #         x = self.transformer.drop(tok_emb)
    #
    #     for block in self.transformer.h:
    #         x = block(x)
    #     x = self.transformer.ln_f(x)
    #
    #     if targets is not None:
    #         # if we are given some desired targets also calculate the loss
    #         logits = self.lm_head(x)
    #         # print(logits.shape,targets.shape)
    #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=PADDING_TOKEN)
    #     else:
    #         # inference-time mini-optimization: only forward the lm_head on the very last position
    #         logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
    #         loss = None
    #
    #     return logits, loss


    def forward(self, idx, targets=None):
        device = idx.device
        # print(idx.shape)
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        attn_mask = (idx != PADDING_TOKEN)[:, None, None, :]
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # self.transformer.spec_wte.weight = self.wte2spec_wte(self.transformer.wte.weight)
        # tok_emb = self.spec_wte(idx) # token embeddings of shape (b, t, n_embd)
        if not self.config.rotary:
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x,attn_mask=attn_mask)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # print(logits.shape,targets.shape)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=PADDING_TOKEN)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def forward_to_embeds(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # self.transformer.spec_wte.weight = self.wte2spec_wte(self.transformer.wte.weight)
        # tok_emb = self.spec_wte(idx) # token embeddings of shape (b, t, n_embd)
        if not self.config.rotary:
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        embed = self.transformer.ln_f(x)

        return embed

    def forward_with_prompt_duringtrain(self, molecules, prompt):
        device = molecules.device
        b, t = molecules.size()
        b_p, p, p_embedding = prompt.size()
        if b_p != b:
            prompt = prompt.repeat(b,1,1)
        pos = torch.arange(0, t + p - 1, dtype=torch.long, device=device)
        input_ids = molecules[:, :-1]
        targets = molecules[:, 1:]
        tok_emb = self.transformer.wte(input_ids)
        tok_emb = torch.cat([prompt,tok_emb],dim=1)
        if not self.config.rotary:
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)[:, p:] # cut off the prompt
        logits = self.lm_head(x)
        # focus more on the non-carbon atoms
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=PADDING_TOKEN)
               # F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=1) + \
               # 8*F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=4)

        return logits, loss


    def forward_with_prompt_attn(self, molecules, prompt, assigment,):
        h_assigment, c_assigment = assigment
        device = molecules.device
        b, t = molecules.size()
        b_p, p, p_embedding = prompt.size()
        if b_p != b:
            prompt = prompt.repeat(b,1,1)
        pos = torch.arange(0, t + p - 1, dtype=torch.long, device=device)
        input_ids = molecules[:, :-1]
        targets = molecules[:, 1:]
        tok_emb = self.transformer.wte(input_ids)
        tok_emb = torch.cat([prompt,tok_emb],dim=1)
        if not self.config.rotary:
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        for i, block in enumerate(self.transformer.h):
            is_last = (i == len(self.transformer.h) - 4)
        # for block in self.transformer.h:
            if is_last:
                x, attn_weights = block( x, return_attn= True and is_last)
            else:
                x = block( x, return_attn= False)


        '''
        attn_weights.shape: B, Num_head, Molecule, Spec
        weights = atten_matrix[layer_id][sample_id, :,
                          func_id, pos_id].max(1).mean(-1)
        '''
        # clip attention
        attn_weights = attn_weights[:,:,p:,:p]
        attn_mean = attn_weights.mean(dim=1)
        h_assigment = h_assigment[:,:,:-1]
        c_assigment = c_assigment[:,:,:-1]
        # print(h_assigment.shape)
        # print(c_assigment.shape)
        # print(attn_weights.shape)
        # print(prompt.shape)
        # print(input_ids.shape)
        # print(attn_weights.shape)

        c_token = c_assigment[:,0]  # list or list of list
        c_spec = c_assigment[:,1]
        len_c_token = c_token.shape[-1]
        h_token = h_assigment[:,0]
        h_spec = h_assigment[:,1]

        # 计算 loss
        # print(attn_weights)
        c_loss = compute_cnmr_attn_loss(attn_weights, c_token, c_spec, lambda_attn=0.01)
        h_loss = compute_hnmr_attn_loss(attn_weights, h_token, h_spec+len_c_token, lambda_attn=0.01)
        # print(c_loss, h_loss)
        x = self.transformer.ln_f(x)[:, p:] # cut off the prompt
        logits = self.lm_head(x)
        # focus more on the non-carbon atoms
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=PADDING_TOKEN)

        return logits, loss+c_loss+h_loss


    def refine_with_prompt_duringtrain(self, molecules, prompt,mol_prompt):
        # import numpy as np
        # torch.set_printoptions(threshold=np.inf)
        # print(molecules)
        device = molecules.device
        b, t = molecules.size()
        b_p, p, p_embedding = prompt.size()
        if b_p != b:
            prompt = prompt.repeat(b,1,1)

        b_mp, mp = mol_prompt.size()
        # random_tuning_length = random.randint(-5,10)
        # mol_prompt = pad_or_truncate_mol_prompt(mol_prompt, target_length=mp+random_tuning_length,
        #                                         vocab_size=self.config.vocab_size, device=device)
        # molecules = pad_or_truncate_mol_prompt(molecules, target_length=t-random_tuning_length,
        #                                         vocab_size=self.config.vocab_size, device=device)
        #
        #
        # b_mp, mp = mol_prompt.size()
        # b, t = molecules.size()
        pos = torch.arange(0, t + p + mp, dtype=torch.long, device=device)
        input_ids = molecules[:, :-1]
        targets = molecules[:, 1:]
        tok_emb = self.transformer.wte(input_ids)

        mol_prompt_emb = self.transformer.wte(mol_prompt)


        START_GENERATE = (self.config.block_size - 1) * torch.ones((b, 1)).to(prompt.device).long()
        START_GENERATE = self.transformer.wpe(START_GENERATE).to(prompt.device)

        tok_emb = torch.cat([prompt,mol_prompt_emb,START_GENERATE,tok_emb], dim=1)
        if not self.config.rotary:
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)[:, p+mp+1:] # cut off the prompt
        logits = self.lm_head(x)
        # focus more on the non-carbon atoms
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=PADDING_TOKEN)
               # F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=1) + \
               # 8*F.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=4)

        return logits, loss

    def refine_with_prompt_logits(self, molecules, prompt, mol_prompt):
        device = molecules.device
        b, t = molecules.size()
        b_p, p, p_embedding = prompt.size()
        if b_p != b:
            prompt = prompt.repeat(b, 1, 1)
        b_mp, mp = mol_prompt.size()

        pos = torch.arange(0, t + p + mp, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(molecules)

        mol_prompt_emb = self.transformer.wte(mol_prompt)

        START_GENERATE = (self.config.block_size - 1) * torch.ones((b, 1)).to(prompt.device).long()
        START_GENERATE = self.transformer.wpe(START_GENERATE).to(prompt.device)

        tok_emb = torch.cat([prompt, mol_prompt_emb, START_GENERATE, tok_emb], dim=1)
        if not self.config.rotary:
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)[:,-1:]  # cut off the prompt
        logits = self.lm_head(x)
        return logits


    # def forward_with_prompt_logits(self, prompt,molecules,use_cache=False,return_cache=False):
    #     device = molecules.device
    #     b, t = molecules.size()
    #     b_p, p, p_embedding = prompt.size()
    #     if b_p != b:
    #         prompt = prompt.repeat(b,1,1)
    #     pos = torch.arange(0, t + p, dtype=torch.long, device=device)
    #     input_ids = molecules[:]
    #     tok_emb = self.transformer.wte(input_ids)
    #     tok_emb = torch.cat([prompt,tok_emb],dim=1)
    #     if not self.config.rotary:
    #         pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
    #         x = self.transformer.drop(tok_emb + pos_emb)
    #     else:
    #         x = self.transformer.drop(tok_emb)
    #     for block in self.transformer.h:
    #         x = block(x,use_cache=use_cache,return_cache=return_cache)
    #     x = self.transformer.ln_f(x)[:,-1:] # cut off the prompt
    #     logits = self.lm_head(x)
    #     return logits

    def forward_with_prompt_logits(self, prompt, molecules, use_cache=False, return_cache=False, start_pos=0):
        device = molecules.device
        b, t = molecules.size()
        b_p, p, p_embedding = prompt.size()

        # 如果 batch 维度不匹配，重复 prompt
        if b_p != b:
            prompt = prompt.repeat(b, 1, 1)

        # 嵌入 molecules
        input_ids = molecules
        tok_emb = self.transformer.wte(input_ids)  # 形状 (b, t, n_embd)

        # 在非缓存模式或首次调用时，拼接 prompt
        if not use_cache or not hasattr(self, 'prompt_processed'):
            tok_emb = torch.cat([prompt, tok_emb], dim=1)  # 形状 (b, p+t, n_embd)
            seq_len = p + t
        else:
            # KV_cache 模式下，只处理 molecules
            seq_len = t

        # 位置编码
        if not self.config.rotary:
            pos = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # 形状 (p+t 或 t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)  # RoPE 由注意力层处理

        # 通过 transformer 层
        for block in self.transformer.h:
            x = block(x, use_cache=use_cache, return_cache=return_cache, start_pos=start_pos)

        # 取最后一个 token 的输出
        x = self.transformer.ln_f(x)[:, -1:]  # 形状 (b, 1, n_embd)
        logits = self.lm_head(x)  # 形状 (b, 1, vocab_size)

        # 标记 prompt 已处理
        if return_cache:
            self.prompt_processed = True

        return logits


    def forward_with_prompt_logits_all(self, prompt, molecules, use_cache=False, return_cache=False, start_pos=0):
        device = molecules.device
        b, t = molecules.size()
        b_p, p, p_embedding = prompt.size()

        # 如果 batch 维度不匹配，重复 prompt
        if b_p != b:
            prompt = prompt.repeat(b, 1, 1)

        # 嵌入 molecules
        input_ids = molecules
        tok_emb = self.transformer.wte(input_ids)  # 形状 (b, t, n_embd)

        # 在非缓存模式或首次调用时，拼接 prompt
        if not use_cache or not hasattr(self, 'prompt_processed'):
            tok_emb = torch.cat([prompt, tok_emb], dim=1)  # 形状 (b, p+t, n_embd)
            seq_len = p + t
        else:
            # KV_cache 模式下，只处理 molecules
            seq_len = t

        # 位置编码
        if not self.config.rotary:
            pos = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # 形状 (p+t 或 t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)  # RoPE 由注意力层处理

        # 通过 transformer 层
        for block in self.transformer.h:
            x = block(x, use_cache=use_cache, return_cache=return_cache, start_pos=start_pos)

        # 取最后一个 token 的输出
        x = self.transformer.ln_f(x)[:, p:]  # 形状 (b, 1, n_embd)
        logits = self.lm_head(x)  # 形状 (b, 1, vocab_size)



        return logits

    def refine_with_prompt(self, prompt,molecules, mol_prompt):
        logits = self.refine_with_prompt_logits(prompt=prompt,molecules=molecules, mol_prompt=mol_prompt)
        mol = torch.argmax(logits,dim=-1).long()
        return mol

    def forward_with_prompt(self, prompt,molecules,use_cache=False,return_cache=False,start_pos=0):
        logits = self.forward_with_prompt_logits(prompt,molecules,use_cache,return_cache,start_pos=start_pos)
        mol = torch.argmax(logits,dim=-1).long()
        return mol


    def forward_with_prompt_prob(self, prompt,molecules):
        logits = self.forward_with_prompt_logits(prompt,molecules)
        mol = torch.argmax(logits,dim=-1).long()
        # the probability of the generated mol token
        prob = F.softmax(logits,dim=-1).gather(2,mol.unsqueeze(-1)).squeeze(-1)
        return mol,prob

    def refine_with_prompt_prob(self, prompt,molecules, mol_prompt):
        logits = self.refine_with_prompt_logits(prompt=prompt,molecules=molecules,mol_prompt=mol_prompt)
        mol = torch.argmax(logits,dim=-1).long()
        # the probability of the generated mol token
        prob = F.softmax(logits,dim=-1).gather(2,mol.unsqueeze(-1)).squeeze(-1)
        return mol,prob

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)

        model = GPT(config)
        sd = model.state_dict()
        # init a huggingface/transformers model
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path = 'gpt-base-uncased')
        sd_hf = model_hf.state_dict()
        language_embedding = sd_hf['transformer.wte.weight']
        print(language_embedding.shape)

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # model.init_spec_tokens()
        return model

    def kmeans(self,x, ncluster, niter=10):
        '''
        x : torch.tensor(data_num,data_dim)
        ncluster : The number of clustering for data_num
        niter : Number of iterations for kmeans
        '''
        N, D = x.size()
        c = x[torch.randperm(N)[:ncluster]]  # init clusters at random
        for i in range(niter):
            # assign all pixels to the closest codebook element
            # .argmin(1) : 按列取最小值的下标,下面这行的意思是将x.size(0)个数据点归类到random选出的ncluster类
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            # 计算每一类的迭代中心，然后重新把第一轮随机选出的聚类中心移到这一类的中心处
            c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            print('done step %d/%d, re-initialized %d dead clusters' % (i + 1, niter, ndead))
            c[nanix] = x[torch.randperm(N)[:ndead]]  # re-init dead clusters
        return c

    def init_spec_tokens(self):
        self.transformer.requires_grad_(False)
        self.spec_wte = nn.Embedding(8192+256, self.config.n_embd)
        param = self.transformer.wte.weight.clone().detach()
        prototype = self.kmeans(param, 1000, 10)
        self.wte2spec_wte = nn.Linear(1000, 8192+256, bias=False)
        # self.wte2spec_wte = nn.Linear(self.config.vocab_size, 8192+256, bias=False)
        print(prototype.shape, self.wte2spec_wte.weight.shape)
        self.spec_wte.weight = torch.mm(self.wte2spec_wte.weight,prototype)

        self.lm_head = nn.Linear(self.config.n_embd, 8192+256, bias=False)

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

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    # @ torch.no_grad()
    # def generate_with_prompt(self,prompt, max_new_tokens,if_kv_cache):
    #     """
    #     Take a conditioning sequence of input spectra (LongTensor of shape (b,t,L)) and complete
    #     the sequence max_new_tokens times, feeding the predictions back into the model each time.
    #     Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    #     """
    #
    #     b, n_seq, n_feat = prompt.size()
    #     mol_cond = torch.empty(b,0,dtype=torch.long,device=prompt.device)
    #     for _ in range(max_new_tokens):
    #         # if the sequence context is growing too long we must crop it at block_size
    #         # forward the model to get the logits for the index in the sequence
    #         mol_next = self.forward_with_prompt(prompt,mol_cond)
    #         # append sampled index to the running sequence and continue
    #         mol_cond = torch.cat((mol_cond, mol_next), dim=1)
    #     return mol_cond

    @torch.no_grad()
    def generate_with_prompt(self, prompt, max_new_tokens, if_kv_cache):
        """
        Take a conditioning sequence of input spectra (LongTensor of shape (b,t,L)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Args:
            prompt: Tensor of shape (b, t, L), conditioning input spectra
            max_new_tokens: Number of new tokens to generate
            if_kv_cache: Boolean, whether to use KV_cache for efficient generation
        Returns:
            mol_cond: Generated sequence of shape (b, max_new_tokens)
        """
        b, p, p_embedding = prompt.size()
        mol_cond = torch.empty(b, 0, dtype=torch.long, device=prompt.device)

        if if_kv_cache:
            # 清理缓存和标记
            for module in self.modules():
                if isinstance(module, CausalSelfAttention):
                    module.kv_cache = None
            if hasattr(self, 'prompt_processed'):
                del self.prompt_processed

            # 首次处理完整序列
            mol_next = self.forward_with_prompt(
                prompt, mol_cond, use_cache=True, return_cache=True, start_pos=0
            )
            mol_cond = torch.cat((mol_cond, mol_next), dim=1)

            # 增量生成
            for t in range(max_new_tokens - 1):
                mol_cond_last = mol_cond[:, -1:]  # 形状 (b, 1)
                mol_next = self.forward_with_prompt(
                    prompt, mol_cond_last, use_cache=True, start_pos=p + t
                )
                mol_cond = torch.cat((mol_cond, mol_next), dim=1)

            # 清理缓存和标记
            for module in self.modules():
                if isinstance(module, CausalSelfAttention):
                    module.kv_cache = None
            if hasattr(self, 'prompt_processed'):
                del self.prompt_processed
        else:
            # 不使用 KV_cache
            for _ in range(max_new_tokens):
                mol_next = self.forward_with_prompt(prompt, mol_cond)
                mol_cond = torch.cat((mol_cond, mol_next), dim=1)

        return mol_cond



    @torch.no_grad()
    def generate_with_beam_search(self, prompt, max_new_tokens, beam_width=3, if_kv_cache=True):
        """
        Generate sequences using beam search, maintaining beam_width candidates at each step.

        Args:
            prompt: Tensor of shape (b, t, L), conditioning input spectra
            max_new_tokens: Number of new tokens to generate
            beam_width: Number of beams to maintain during search
            if_kv_cache: Boolean, whether to use KV_cache for efficient generation
        Returns:
            mol_cond: Generated sequence of shape (b, max_new_tokens) with highest score
        """
        b, p, p_embedding = prompt.size()
        device = prompt.device

        # Initialize beam: (batch_size, beam_width, sequence_length)
        mol_cond = torch.empty(b, beam_width, 0, dtype=torch.long, device=device)
        beam_scores = torch.zeros(b, beam_width, device=device)  # Track log-probability scores

        if if_kv_cache:
            # Clean up cache and flags
            for module in self.modules():
                if isinstance(module, CausalSelfAttention):
                    module.kv_cache = None
            if hasattr(self, 'prompt_processed'):
                del self.prompt_processed

        # Expand prompt for each beam
        prompt_expanded = prompt.unsqueeze(1).expand(-1, beam_width, -1, -1)  # (b, beam_width, t, L)
        prompt_expanded = prompt_expanded.reshape(b * beam_width, p, p_embedding)

        # First step
        if if_kv_cache:
            logits = self.forward_with_prompt_logits(
                prompt_expanded, mol_cond.view(b * beam_width, 0),
                use_cache=True, return_cache=True, start_pos=0
            )
        else:
            logits = self.forward_with_prompt_logits(prompt_expanded, mol_cond.view(b * beam_width, 0))

        # Ensure logits are float before applying log_softmax
        logits = logits.float()
        log_probs = F.log_softmax(logits, dim=-1)  # (b * beam_width, vocab_size)
        topk_log_probs, topk_indices = log_probs.topk(beam_width, dim=-1)  # (b * beam_width, beam_width)

        # Reshape for batch processing
        topk_log_probs = topk_log_probs.view(b, beam_width, beam_width)  # (b, beam_width, beam_width)
        topk_indices = topk_indices.view(b, beam_width, beam_width)

        # Select top beam_width candidates for each batch
        new_scores = beam_scores.unsqueeze(-1) + topk_log_probs  # (b, beam_width, beam_width)
        new_scores = new_scores.view(b, beam_width * beam_width)
        top_scores, top_indices = new_scores.topk(beam_width, dim=-1)  # (b, beam_width)

        # Update beams
        beam_idx = top_indices // beam_width  # Which beam the token came from
        token_idx = top_indices % beam_width  # Which token was selected
        # Expand token_idx to match topk_indices dimensions
        token_idx = token_idx.unsqueeze(-1)  # (b, beam_width, 1)
        mol_cond = torch.gather(topk_indices, dim=2, index=token_idx)  # (b, beam_width, 1)
        beam_scores = top_scores

        # Generate remaining tokens
        for t in range(max_new_tokens - 1):
            if if_kv_cache:
                # Process last token with KV cache
                mol_cond_last = mol_cond[:, :, -1:]  # (b, beam_width, 1)
                mol_cond_last = mol_cond_last.view(b * beam_width, 1)
                logits = self.forward_with_prompt_logits(
                    prompt_expanded, mol_cond_last,
                    use_cache=True, start_pos=p + t
                )
            else:
                # Process full sequence without cache
                mol_cond_expanded = mol_cond.view(b * beam_width, -1)
                logits = self.forward_with_prompt_logits(prompt_expanded, mol_cond_expanded)

            # Ensure logits are float before applying log_softmax
            logits = logits.float()
            log_probs = F.log_softmax(logits, dim=-1)  # (b * beam_width, vocab_size)
            topk_log_probs, topk_indices = log_probs.topk(beam_width, dim=-1)  # (b * beam_width, beam_width)

            # Reshape for batch processing
            topk_log_probs = topk_log_probs.view(b, beam_width, beam_width)
            topk_indices = topk_indices.view(b, beam_width, beam_width)

            # Update scores
            new_scores = beam_scores.unsqueeze(-1) + topk_log_probs  # (b, beam_width, beam_width)
            new_scores = new_scores.view(b, beam_width * beam_width)
            top_scores, top_indices = new_scores.topk(beam_width, dim=-1)

            # Update beams
            beam_idx = top_indices // beam_width
            token_idx = top_indices % beam_width

            # Expand token_idx for gather
            token_idx = token_idx.unsqueeze(-1)  # (b, beam_width, 1)
            new_tokens = torch.gather(topk_indices, dim=2, index=token_idx)  # (b, beam_width, 1)

            # Update sequences
            mol_cond = torch.cat([mol_cond, new_tokens], dim=-1)
            beam_scores = top_scores

            # Update beam indices for next iteration
            mol_cond = torch.gather(mol_cond, dim=1,
                                    index=beam_idx.unsqueeze(-1).expand(-1, -1, mol_cond.size(-1)))

        if if_kv_cache:
            # Clean up cache and flags
            for module in self.modules():
                if isinstance(module, CausalSelfAttention):
                    module.kv_cache = None
            if hasattr(self, 'prompt_processed'):
                del self.prompt_processed

        # Return the sequence with the highest score
        best_beam_idx = beam_scores.argmax(dim=-1)  # (b,)
        best_mol_cond = mol_cond[torch.arange(b), best_beam_idx]  # (b, max_new_tokens)

        return best_mol_cond

    import torch
    from torch import nn
    import torch.nn.functional as F

    @torch.no_grad()
    def generate_with_hybrid_search(self, prompt, max_new_tokens, beam_width=3, beam_steps=5, temperature=1.0,
                                    if_kv_cache=True):
        """
        Generate sequences using a hybrid search: beam search for the first beam_steps tokens,
        followed by greedy search for the remaining tokens.

        Args:
            prompt: Tensor of shape (b, t, L), conditioning input spectra
            max_new_tokens: Number of new tokens to generate
            beam_width: Number of beams to maintain during beam search
            beam_steps: Number of tokens to generate using beam search before switching to greedy search
            temperature: Temperature for controlling randomness (default: 1.0, no scaling)
            if_kv_cache: Boolean, whether to use KV_cache for efficient generation
        Returns:
            mol_cond: Generated sequence of shape (b, max_new_tokens)
        """
        b, p, p_embedding = prompt.size()
        device = prompt.device

        # Initialize beam: (batch_size, beam_width, sequence_length)
        mol_cond = torch.empty(b, beam_width, 0, dtype=torch.long, device=device)
        beam_scores = torch.zeros(b, beam_width, device=device)  # Track log-probability scores

        if if_kv_cache:
            # Clean up cache and flags
            for module in self.modules():
                if isinstance(module, CausalSelfAttention):
                    module.kv_cache = None
            if hasattr(self, 'prompt_processed'):
                del self.prompt_processed

        # Expand prompt for each beam
        prompt_expanded = prompt.unsqueeze(1).expand(-1, beam_width, -1, -1)  # (b, beam_width, t, L)
        prompt_expanded = prompt_expanded.reshape(b * beam_width, p, p_embedding)

        # Beam search for the first beam_steps tokens
        effective_beam_steps = min(beam_steps, max_new_tokens)  # Ensure beam_steps <= max_new_tokens
        for t in range(effective_beam_steps):
            if t == 0:
                # First step
                if if_kv_cache:
                    logits = self.forward_with_prompt_logits(
                        prompt_expanded, mol_cond.view(b * beam_width, 0),
                        use_cache=True, return_cache=True, start_pos=0
                    )
                else:
                    logits = self.forward_with_prompt_logits(prompt_expanded, mol_cond.view(b * beam_width, 0))
            else:
                # Subsequent steps
                mol_cond_last = mol_cond[:, :, -1:]  # (b, beam_width, 1)
                mol_cond_last = mol_cond_last.view(b * beam_width, 1)
                if if_kv_cache:
                    logits = self.forward_with_prompt_logits(
                        prompt_expanded, mol_cond_last,
                        use_cache=True, start_pos=p + t - 1
                    )
                else:
                    mol_cond_expanded = mol_cond.view(b * beam_width, -1)
                    logits = self.forward_with_prompt_logits(prompt_expanded, mol_cond_expanded)

            # Apply temperature scaling
            logits = logits.float() / temperature
            log_probs = F.log_softmax(logits, dim=-1)  # (b * beam_width, vocab_size)

            # Dynamically adjust beam_width based on vocab_size
            vocab_size = log_probs.size(-1)
            effective_beam_width = min(beam_width, vocab_size)
            if effective_beam_width < beam_width:
                print(
                    f"Warning: beam_width ({beam_width}) is larger than vocab_size ({vocab_size}). Using effective_beam_width={effective_beam_width}")

            topk_log_probs, topk_indices = log_probs.topk(effective_beam_width,
                                                          dim=-1)  # (b * beam_width, effective_beam_width)

            # Reshape for batch processing
            topk_log_probs = topk_log_probs.view(b, beam_width, effective_beam_width)
            topk_indices = topk_indices.view(b, beam_width, effective_beam_width)

            # Select top beam_width candidates for each batch
            new_scores = beam_scores.unsqueeze(-1) + topk_log_probs  # (b, beam_width, effective_beam_width)
            new_scores = new_scores.view(b, beam_width * effective_beam_width)
            top_scores, top_indices = new_scores.topk(beam_width, dim=-1)  # (b, beam_width)

            # Update beams
            beam_idx = top_indices // effective_beam_width
            token_idx = top_indices % effective_beam_width
            token_idx = token_idx.unsqueeze(-1)  # Expand to (b, beam_width, 1)
            new_tokens = torch.gather(topk_indices, dim=2, index=token_idx)  # (b, beam_width, 1)

            # Update sequences
            mol_cond = torch.cat([mol_cond, new_tokens], dim=-1)
            beam_scores = top_scores

            # Update beam indices for next iteration
            mol_cond = torch.gather(mol_cond, dim=1,
                                    index=beam_idx.unsqueeze(-1).expand(-1, -1, mol_cond.size(-1)))

        # Select the best beam for greedy search
        best_beam_idx = beam_scores.argmax(dim=-1)  # (b,)
        mol_cond = mol_cond[torch.arange(b), best_beam_idx]  # (b, beam_steps)

        # Greedy search for remaining tokens
        for t in range(effective_beam_steps, max_new_tokens):
            mol_cond_last = mol_cond[:, -1:]  # (b, 1)
            if if_kv_cache:
                logits = self.forward_with_prompt_logits(
                    prompt, mol_cond_last,
                    use_cache=True, start_pos=p + t - 1
                )
            else:
                logits = self.forward_with_prompt_logits(prompt, mol_cond)

            # Apply temperature scaling
            log_probs = F.log_softmax(logits, dim=-1)  # (b, vocab_size)

            # Greedy selection: take the token with the highest probability
            next_token = log_probs.argmax(dim=-1) # (b, 1)
            # print(mol_cond.shape,next_token.shape)
            mol_cond = torch.cat([mol_cond, next_token], dim=-1)

        if if_kv_cache:
            # Clean up cache and flags
            for module in self.modules():
                if isinstance(module, CausalSelfAttention):
                    module.kv_cache = None
            if hasattr(self, 'prompt_processed'):
                del self.prompt_processed

        return mol_cond

    @torch.no_grad()
    def generate_with_hybrid_search_with_prob(self, prompt, max_new_tokens, beam_width=3, beam_steps=5, temperature=1.0,
                                              if_kv_cache=True):
        """
        Generate sequences using a hybrid search: beam search for the first beam_steps tokens,
        followed by greedy search for the remaining tokens, returning both sequences and probabilities.

        Args:
            prompt: Tensor of shape (b, t, L), conditioning input spectra
            max_new_tokens: Number of new tokens to generate
            beam_width: Number of beams to maintain during beam search
            beam_steps: Number of tokens to generate using beam search before switching to greedy search
            temperature: Temperature for controlling randomness (default: 1.0, no scaling)
            if_kv_cache: Boolean, whether to use KV_cache for efficient generation

        Returns:
            mol_cond: Generated sequence of shape (b, max_new_tokens)
            probs: Probabilities for each generated token, shape (b, max_new_tokens)
        """
        b, p, p_embedding = prompt.size()
        device = prompt.device

        # Initialize beam: (batch_size, beam_width, sequence_length)
        mol_cond = torch.empty(b, beam_width, 0, dtype=torch.long, device=device)
        beam_scores = torch.zeros(b, beam_width, device=device)  # Track log-probability scores
        beam_probs = torch.empty(b, beam_width, 0, dtype=torch.float, device=device)  # Track probabilities

        if if_kv_cache:
            # Clean up cache and flags
            for module in self.modules():
                if isinstance(module, CausalSelfAttention):
                    module.kv_cache = None
            if hasattr(self, 'prompt_processed'):
                del self.prompt_processed

        # Expand prompt for each beam
        prompt_expanded = prompt.unsqueeze(1).expand(-1, beam_width, -1, -1)  # (b, beam_width, t, L)
        prompt_expanded = prompt_expanded.reshape(b * beam_width, p, p_embedding)

        # Beam search for the first beam_steps tokens
        effective_beam_steps = min(beam_steps, max_new_tokens)  # Ensure beam_steps <= max_new_tokens
        for t in range(effective_beam_steps):
            if t == 0:
                # First step
                if if_kv_cache:
                    logits = self.forward_with_prompt_logits(
                        prompt_expanded, mol_cond.view(b * beam_width, 0),
                        use_cache=True, return_cache=True, start_pos=0
                    )
                else:
                    logits = self.forward_with_prompt_logits(prompt_expanded, mol_cond.view(b * beam_width, 0))
            else:
                # Subsequent steps
                mol_cond_last = mol_cond[:, :, -1:]  # (b, beam_width, 1)
                mol_cond_last = mol_cond_last.view(b * beam_width, 1)
                if if_kv_cache:
                    logits = self.forward_with_prompt_logits(
                        prompt_expanded, mol_cond_last,
                        use_cache=True, start_pos=p + t - 1
                    )
                else:
                    mol_cond_expanded = mol_cond.view(b * beam_width, -1)
                    logits = self.forward_with_prompt_logits(prompt_expanded, mol_cond_expanded)

            # Apply temperature scaling
            logits = logits.float() / temperature
            log_probs = F.log_softmax(logits, dim=-1)  # (b * beam_width, vocab_size)
            probs = F.softmax(logits, dim=-1)  # (b * beam_width, vocab_size)

            # Dynamically adjust beam_width based on vocab_size
            vocab_size = log_probs.size(-1)
            effective_beam_width = min(beam_width, vocab_size)
            if effective_beam_width < beam_width:
                print(
                    f"Warning: beam_width ({beam_width}) is larger than vocab_size ({vocab_size}). Using effective_beam_width={effective_beam_width}"
                )

            topk_log_probs, topk_indices = log_probs.topk(effective_beam_width,
                                                          dim=-1)  # (b * beam_width, effective_beam_width)
            topk_probs = torch.gather(probs, dim=-1, index=topk_indices)  # (b * beam_width, effective_beam_width)

            # Reshape for batch processing
            topk_log_probs = topk_log_probs.view(b, beam_width, effective_beam_width)
            topk_probs = topk_probs.view(b, beam_width, effective_beam_width)
            topk_indices = topk_indices.view(b, beam_width, effective_beam_width)

            # Select top beam_width candidates for each batch
            new_scores = beam_scores.unsqueeze(-1) + topk_log_probs  # (b, beam_width, effective_beam_width)
            new_scores = new_scores.view(b, beam_width * effective_beam_width)
            top_scores, top_indices = new_scores.topk(beam_width, dim=-1)  # (b, beam_width)

            # Update beams
            beam_idx = top_indices // effective_beam_width
            token_idx = top_indices % effective_beam_width
            token_idx = token_idx.unsqueeze(-1)  # Expand to (b, beam_width, 1)
            new_tokens = torch.gather(topk_indices, dim=2, index=token_idx)  # (b, beam_width, 1)
            new_probs = torch.gather(topk_probs, dim=2, index=token_idx)  # (b, beam_width, 1)

            # Update sequences and probabilities
            mol_cond = torch.cat([mol_cond, new_tokens], dim=-1)
            beam_probs = torch.cat([beam_probs, new_probs], dim=-1)
            beam_scores = top_scores

            # Update beam indices for next iteration
            mol_cond = torch.gather(mol_cond, dim=1, index=beam_idx.unsqueeze(-1).expand(-1, -1, mol_cond.size(-1)))
            beam_probs = torch.gather(beam_probs, dim=1,
                                      index=beam_idx.unsqueeze(-1).expand(-1, -1, beam_probs.size(-1)))

        # Select the best beam for greedy search
        best_beam_idx = beam_scores.argmax(dim=-1)  # (b,)
        mol_cond = mol_cond[torch.arange(b), best_beam_idx]  # (b, beam_steps)
        probs = beam_probs[torch.arange(b), best_beam_idx]  # (b, beam_steps)

        # Greedy search for remaining tokens
        for t in range(effective_beam_steps, max_new_tokens):
            mol_cond_last = mol_cond[:, -1:]  # (b, 1)
            if if_kv_cache:
                logits = self.forward_with_prompt_logits(
                    prompt, mol_cond_last,
                    use_cache=True, start_pos=p + t - 1
                )
            else:
                logits = self.forward_with_prompt_logits(prompt, mol_cond)

            # Apply temperature scaling
            logits = logits.squeeze(1).float() / temperature
            log_probs = F.log_softmax(logits, dim=-1)  # (b, vocab_size)
            token_probs = F.softmax(logits, dim=-1)  # (b, vocab_size)

            # Greedy selection: take the token with the highest probability
            next_token = log_probs.argmax(dim=-1, keepdim=True)  # (b, 1)
            # print(next_token.shape,mol_cond.shape,token_probs.shape)
            next_prob = torch.gather(token_probs, dim=-1, index=next_token)  # (b, 1)

            # Append to sequence and probabilities
            mol_cond = torch.cat([mol_cond, next_token], dim=-1)
            probs = torch.cat([probs, next_prob], dim=-1)

        if if_kv_cache:
            # Clean up cache and flags
            for module in self.modules():
                if isinstance(module, CausalSelfAttention):
                    module.kv_cache = None
            if hasattr(self, 'prompt_processed'):
                del self.prompt_processed

        return mol_cond, probs

    @ torch.no_grad()
    def generate_with_prompt_prob(self,prompt, max_new_tokens):
        """
        Take a conditioning sequence of input spectra (LongTensor of shape (b,t,L)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        b, n_seq, n_feat = prompt.size()
        mol_cond = torch.empty(b,0,dtype=torch.long,device=prompt.device)
        probs = torch.empty(b,0,dtype=torch.float,device=prompt.device)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # forward the model to get the logits for the index in the sequence
            mol_next,prob = self.forward_with_prompt_prob(prompt,mol_cond)
            # append sampled index to the running sequence and continue
            mol_cond = torch.cat((mol_cond, mol_next), dim=1)
            probs = torch.cat((probs, prob), dim=1)
        return mol_cond,probs


    @ torch.no_grad()
    def refine_generate_with_prompt(self,prompt, max_new_tokens,mol_prompt):
        """
        Take a conditioning sequence of input spectra (LongTensor of shape (b,t,L)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        b, n_seq, n_feat = prompt.size()
        mol_cond = torch.empty(b,0,dtype=torch.long,device=prompt.device)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # forward the model to get the logits for the index in the sequence
            mol_next = self.refine_with_prompt(prompt,mol_cond,mol_prompt)
            # append sampled index to the running sequence and continue
            mol_cond = torch.cat((mol_cond, mol_next), dim=1)
        return mol_cond

    @ torch.no_grad()
    def refine_generate_with_prompt_prob(self,prompt, max_new_tokens,mol_prompt):
        """
        Take a conditioning sequence of input spectra (LongTensor of shape (b,t,L)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        b, n_seq, n_feat = prompt.size()
        mol_cond = torch.empty(b,0,dtype=torch.long,device=prompt.device)
        probs = torch.empty(b,0,dtype=torch.float,device=prompt.device)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # forward the model to get the logits for the index in the sequence
            mol_next,prob = self.refine_with_prompt_prob(prompt,mol_cond,mol_prompt=mol_prompt)
            # append sampled index to the running sequence and continue
            mol_cond = torch.cat((mol_cond, mol_next), dim=1)
            probs = torch.cat((probs, prob), dim=1)
        return mol_cond,probs

    @torch.no_grad()
    @torch.no_grad()
    def refine_generate_with_hybrid_search_with_prob(self, prompt, max_new_tokens, mol_prompt, beam_width=3,
                                                                 beam_steps=5, temperature=1.0, if_kv_cache=True):
        """
        Generate sequences using a hybrid search: beam search for the first beam_steps tokens,
        followed by greedy search for the remaining tokens, conditioned on input spectra and molecular prompt,
        returning both sequences and probabilities.

        Args:
            prompt: Tensor of shape (b, t, L), conditioning input spectra
            max_new_tokens: Number of new tokens to generate
            mol_prompt: Tensor of shape (mb, mp), molecular conditioning input
            beam_width: Number of beams to maintain during beam search
            beam_steps: Number of tokens to generate using beam search before switching to greedy search
            temperature: Temperature for controlling randomness (default: 1.0, no scaling)
            if_kv_cache: Boolean, whether to use KV_cache for efficient generation
        Returns:
            mol_cond: Generated sequence of shape (b, max_new_tokens)
            probs: Probabilities for each generated token, shape (b, max_new_tokens)
        """
        b, p, p_embedding = prompt.size()
        mb, mp = mol_prompt.size()
        device = prompt.device

        # Initialize beam: (batch_size, beam_width, sequence_length)
        mol_cond = torch.empty(b, beam_width, 0, dtype=torch.long, device=device)
        beam_scores = torch.zeros(b, beam_width, device=device)  # Track log-probability scores
        beam_probs = torch.empty(b, beam_width, 0, dtype=torch.float, device=device)  # Track probabilities

        if if_kv_cache:
            # Clean up cache and flags
            for module in self.modules():
                if isinstance(module, CausalSelfAttention):
                    module.kv_cache = None
            if hasattr(self, 'prompt_processed'):
                del self.prompt_processed

        # Expand prompt and mol_prompt for each beam
        prompt_expanded = prompt.unsqueeze(1).expand(-1, beam_width, -1, -1)  # (b, beam_width, t, L)
        prompt_expanded = prompt_expanded.reshape(b * beam_width, p, p_embedding)
        mol_prompt_expanded = mol_prompt.unsqueeze(1).expand(-1, beam_width, -1)  # (mb, beam_width, mp)
        mol_prompt_expanded = mol_prompt_expanded.reshape(mb * beam_width, mp)

        # Beam search for the first beam_steps tokens
        effective_beam_steps = min(beam_steps, max_new_tokens)  # Ensure beam_steps <= max_new_tokens
        for t in range(effective_beam_steps):
            if t == 0:
                # First step
                if if_kv_cache:
                    logits = self.refine_with_prompt_logits(
                        prompt=prompt_expanded, molecules=mol_cond.view(b * beam_width, 0),
                        mol_prompt=mol_prompt_expanded,
                        use_cache=True, return_cache=True, start_pos=0
                    )
                else:
                    logits = self.refine_with_prompt_logits(
                        prompt=prompt_expanded, molecules=mol_cond.view(b * beam_width, 0),
                        mol_prompt=mol_prompt_expanded
                    )
            else:
                # Subsequent steps
                mol_cond_last = mol_cond[:, :, -1:]  # (b, beam_width, 1)
                mol_cond_last = mol_cond_last.view(b * beam_width, 1)
                if if_kv_cache:
                    logits = self.refine_with_prompt_logits(
                        prompt=prompt_expanded, molecules=mol_cond_last, mol_prompt=mol_prompt_expanded,
                        use_cache=True, start_pos=p + t + mp - 1
                    )
                else:
                    mol_cond_expanded = mol_cond.view(b * beam_width, -1)
                    logits = self.refine_with_prompt_logits(
                        prompt=prompt_expanded, molecules=mol_cond_expanded, mol_prompt=mol_prompt_expanded
                    )

            # Apply temperature scaling
            logits = logits.float() / temperature
            log_probs = F.log_softmax(logits, dim=-1)  # (b * beam_width, vocab_size)
            probs = F.softmax(logits, dim=-1)  # (b * beam_width, vocab_size)
            # Dynamically adjust beam_width based on vocab_size
            vocab_size = log_probs.size(-1)
            effective_beam_width = min(beam_width, vocab_size)
            if effective_beam_width < beam_width:
                print(
                    f"Warning: beam_width ({beam_width}) is larger than vocab_size ({vocab_size}). Using effective_beam_width={effective_beam_width}")

            topk_log_probs, topk_indices = log_probs.topk(effective_beam_width,
                                                          dim=-1)  # (b * beam_width, effective_beam_width)
            topk_probs = torch.gather(probs, dim=-1, index=topk_indices)  # (b * beam_width, effective_beam_width)

            # Reshape for batch processing
            topk_log_probs = topk_log_probs.view(b, beam_width, effective_beam_width)
            topk_probs = topk_probs.view(b, beam_width, effective_beam_width)
            topk_indices = topk_indices.view(b, beam_width, effective_beam_width)

            # Select top beam_width candidates for each batch
            new_scores = beam_scores.unsqueeze(-1) + topk_log_probs  # (b, beam_width, effective_beam_width)
            new_scores = new_scores.view(b, beam_width * effective_beam_width)
            top_scores, top_indices = new_scores.topk(beam_width, dim=-1)  # (b, beam_width)

            # Update beams
            beam_idx = top_indices // effective_beam_width
            token_idx = top_indices % effective_beam_width
            token_idx = token_idx.unsqueeze(-1)  # Expand to (b, beam_width, 1)
            new_tokens = torch.gather(topk_indices, dim=2, index=token_idx)  # (b, beam_width, 1)
            new_probs = torch.gather(topk_probs, dim=2, index=token_idx)  # (b, beam_width, 1)

            # Update sequences and probabilities
            mol_cond = torch.cat([mol_cond, new_tokens], dim=-1)
            beam_probs = torch.cat([beam_probs, new_probs], dim=-1)
            beam_scores = top_scores

            # Update beam indices for next iteration
            mol_cond = torch.gather(mol_cond, dim=1, index=beam_idx.unsqueeze(-1).expand(-1, -1, mol_cond.size(-1)))
            beam_probs = torch.gather(beam_probs, dim=1,
                                      index=beam_idx.unsqueeze(-1).expand(-1, -1, beam_probs.size(-1)))

        # Select the best beam for greedy search
        best_beam_idx = beam_scores.argmax(dim=-1)  # (b,)
        mol_cond = mol_cond[torch.arange(b), best_beam_idx]  # (b, beam_steps)
        probs = beam_probs[torch.arange(b), best_beam_idx]  # (b, beam_steps)

        # Greedy search for remaining tokens
        for t in range(effective_beam_steps, max_new_tokens):
            mol_cond_last = mol_cond[:, -1:]  # (b, 1)
            if if_kv_cache:
                logits = self.refine_with_prompt_logits(
                    prompt=prompt, molecules=mol_cond_last, mol_prompt=mol_prompt,
                    use_cache=True, start_pos=p + t + mp - 1
                )
            else:
                logits = self.refine_with_prompt_logits(
                    prompt=prompt, molecules=mol_cond, mol_prompt=mol_prompt
                )


            # Apply temperature scaling
            logits = logits.squeeze(1).float() / temperature
            log_probs = F.log_softmax(logits, dim=-1)  # (b, vocab_size)
            token_probs = F.softmax(logits, dim=-1)  # (b, vocab_size)

            # Greedy selection: take the token with the highest probability
            next_token = log_probs.argmax(dim=-1, keepdim=True)  # (b, 1)
            next_prob = torch.gather(token_probs, dim=-1, index=next_token)  # (b, 1)

            # Append to sequence and probabilities
            mol_cond = torch.cat([mol_cond, next_token], dim=-1)
            probs = torch.cat([probs, next_prob], dim=-1)

        if if_kv_cache:
            # Clean up cache and flags
            for module in self.modules():
                if isinstance(module, CausalSelfAttention):
                    module.kv_cache = None
            if hasattr(self, 'prompt_processed'):
                del self.prompt_processed

        return mol_cond, probs
