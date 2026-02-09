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
from RotaryBERTransformer import *
import torch
import torch.nn as nn
from torch.nn import functional as F

mse = nn.MSELoss()
@dataclass
class BERTConfig:
    block_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    n_learnable_tokens: int = 5
    n_feat: int = 768
    rotary: bool = True
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class BERT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config
        # spectra 与 word input 有天然的区别，
        # spectra 是连续的输入，而 word input 是离散的输入，因而 spectra的embedding是连续的，而word input的embedding是离散的
        #
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(config.n_feat, config.n_embd,bias=False),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # init all weights
        self.apply(self._init_weights)
        self.learnable_tokens = nn.Parameter(torch.randn(config.n_learnable_tokens, config.n_embd))
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

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

    def forward(self, x):
        # input x of shape (b, n, l) is a arbitrary sequence of features
        # where n is the length of the sequence and l is the dims of features
        b, n, l = x.shape
        assert n <= self.config.block_size, f"length {n} of sequence is greater than block_size {self.config.block_size}"
        learnable_tokens = self.learnable_tokens.unsqueeze(0).expand(b, -1, -1)


        # input embedding
        h = self.transformer.wte(x)
        # concatenate learnable tokens to the input
        # x = torch.cat((learnable_tokens, h), dim=1)
        # position embedding
        h = h + self.transformer.wpe(torch.arange(self.config.block_size, device=x.device)[None, :n])
        # dropout
        h = self.transformer.drop(h)
        # transformer layers
        for block in self.transformer.h:
            h = block(h)
        # layer norm
        h = self.transformer.ln_f(h)
        # print(h.shape)
        # return the learnable tokens and the output of the transformer
        # return h[:, :self.config.n_learnable_tokens], h[:, self.config.n_learnable_tokens:]
        # print('prompt_szie',h.shape)
        return h,h


