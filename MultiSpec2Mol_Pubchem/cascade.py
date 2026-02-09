from typing import Mapping, Any, List, Optional
import torch
import torch.nn as nn
import logging
import time
import psutil
# from molecule_pretrain.model import GPTConfig as MolConfig, GPT as MolGPT
from molecule_pretrain.model import GPTConfig as MolConfig, GPT as MolGPT
from MultiSpec2Mol.model import BERT, BERTConfig
from spectra_prdiction.model import GPT as SpectraGPT, GPTConfig as SpectraConfig
from spectra_prediction_token.model import GPT as HNMRGPT, GPTConfig as HNMRConfig
from spectra_prediction_token.model_cnmr import GPT as CNMRGPT, GPTConfig as CNMRConfig
from spectra_prediction_token.model_hsqc import GPT as HSQCGPT, GPTConfig as HSQCConfig
import inspect

try:
    from muon import MuonWithAuxAdam
except ImportError:
    MuonWithAuxAdam = None

class MultiSpec2Mol(nn.Module):
    def __init__(self, head_gpts: dict[str, SpectraGPT], tail_gpt: MolGPT, cascade_bert: BERT,
                 frozen_model: List[str], allowed_modalities: List[str] = ['[hnmr]', '[cnmr]']):
        super(MultiSpec2Mol, self).__init__()
        self.head_gpts = nn.ModuleDict(head_gpts)
        self.tail_gpt = tail_gpt
        self.cascade_bert = cascade_bert
        self.frozen_model = frozen_model
        self.allowed_modalities = allowed_modalities
        # Validate allowed_modalities
        for modality in allowed_modalities:
            if modality not in head_gpts:
                raise ValueError(f"Allowed modality {modality} not in head_gpts: {list(head_gpts.keys())}")
        self.embed_dim = head_gpts[allowed_modalities[0]].config.n_embd if allowed_modalities else 768
        self.masked_tokens = nn.Embedding(1, self.embed_dim)
        self.tokens = {'[hnmr]': 20, '[cnmr]': 64, '[ir]': 31, '[hsqc]': 64}
        self.freeze_params()

    def freeze_params(self, num_layers_to_unfreeze=3):
        # Freeze non-allowed modalities
        for modality, head_gpt in self.head_gpts.items():
            if modality not in self.allowed_modalities:
                for param in head_gpt.parameters():
                    param.requires_grad = False

        # Freeze allowed modalities based on frozen_model
        if 'head_gpt' in self.frozen_model:
            for modality, head_gpt in self.head_gpts.items():
                if modality in self.allowed_modalities:
                    for param in head_gpt.parameters():
                        param.requires_grad = False

        if 'partial_head_gpt' in self.frozen_model:
            for modality, head_gpt in self.head_gpts.items():
                if modality in self.allowed_modalities:
                    for param in head_gpt.parameters():
                        param.requires_grad = False
                    total_layers = len(head_gpt.transformer.h)
                    start_layer = max(0, total_layers - num_layers_to_unfreeze)
                    for layer_idx in range(start_layer, total_layers):
                        for param in head_gpt.transformer.h[layer_idx].parameters():
                            param.requires_grad = True

        if 'tail_gpt' in self.frozen_model:
            for param in self.tail_gpt.parameters():
                param.requires_grad = False

        if 'partial_tail_gpt' in self.frozen_model:
            for param in self.tail_gpt.parameters():
                param.requires_grad = False
            total_layers = len(self.tail_gpt.transformer.h)
            start_layer = max(0, total_layers - num_layers_to_unfreeze)
            for layer_idx in range(start_layer, total_layers):
                for param in self.tail_gpt.transformer.h[layer_idx].parameters():
                    param.requires_grad = True

        if 'cascade_bert' in self.frozen_model:
            for param in self.cascade_bert.parameters():
                param.requires_grad = False

    def forward(self, molecules, spectra_dict, random_mask_rate=0.0,
                mask_modals=None):
        # the head gpt generate all the tokens
        prompt_dict = {}
        for keys in self.head_gpts.keys():
            if keys in self.allowed_modalities:
                prompt_dict[keys] = \
                    self.head_gpts[keys].forward_to_embeds(spectra_dict[keys])
            else:
                prompt_dict[keys] = \
                    self.masked_tokens(torch.zeros(molecules.shape[0],self.tokens[keys]).long().to(molecules.device))

        if random_mask_rate > 0:
            random_mask_num = int(random_mask_rate * molecules.shape[0])
            for keys in spectra_dict.keys() and mask_modals:
                if keys in mask_modals:
                    if keys in self.allowed_modalities:
                        masked_batch_ids = torch.randperm(molecules.shape[0])[0:random_mask_num]
                        seq_length = prompt_dict[keys].shape[1]
                        prompt_dict[keys][masked_batch_ids] = self.masked_tokens(torch.zeros(random_mask_num,seq_length).long().to(molecules.device))


        # prompt stack
        prompt = torch.concat(list(prompt_dict.values()),dim=1)
        prompt,_ = self.cascade_bert.forward(prompt)
        # the tail gpt generate the spectra
        molecules_idx,loss = self.tail_gpt.forward_with_prompt_duringtrain(molecules, prompt)
        return molecules_idx,loss

    def forward_fast(self, molecules, spectra_dict, random_mask_rate=0.0, mask_modals=None,
                       return_logits = True):
        """
        Forward pass for refining molecular representations with spectral prompts, optimized to avoid redundant embedding calculations.

        Args:
            molecules (torch.Tensor): Input molecular representations [batch_size, ...].
            spectra_dict (dict): Dictionary of spectral data for different modalities.
            random_mask_rate (float): Fraction of batch to mask (0 to 1).
            mask_modals: Modalities to apply masking to (list, set, or None).

        Returns:
            tuple: (molecules_idx, loss) - Refined molecule indices and training loss.
        """
        # Input validation
        assert molecules is not None and molecules.shape[0] > 0, "molecules must be a non-empty tensor"
        assert isinstance(spectra_dict, dict), "spectra_dict must be a dictionary"
        assert 0 <= random_mask_rate <= 1, "random_mask_rate must be between 0 and 1"
        assert mask_modals is None or isinstance(mask_modals, (list, set)), "mask_modals must be a list, set, or None"

        prompt_dict = {}
        batch_size = molecules.shape[0]

        # Determine masked indices if random_mask_rate > 0 and mask_modals is specified
        masked_batch_ids = None
        if random_mask_rate > 0 and mask_modals:
            random_mask_num = int(random_mask_rate * batch_size)
            masked_batch_ids = torch.randperm(batch_size, device=molecules.device)[:random_mask_num]
            unmasked_batch_ids = torch.ones(batch_size, dtype=torch.bool, device=molecules.device)
            unmasked_batch_ids[masked_batch_ids] = False
            unmasked_batch_ids = torch.arange(batch_size, device=molecules.device, dtype=torch.long)[unmasked_batch_ids]
        else:
            unmasked_batch_ids = torch.arange(batch_size, device=molecules.device, dtype=torch.long)

        for key in spectra_dict.keys():
            if key in self.head_gpts:

                embed_dim = self.embed_dim

                # Apply masking for specified modalities
                if random_mask_rate > 0 and mask_modals and key in mask_modals:
                    # Initialize output tensor for this modality
                    if key == '[ir]':
                        seq_length = spectra_dict[key].shape[-1] // self.head_gpts['[ir]'].config.n_patchsize - 1

                    elif key == '[hnmr]':
                        seq_length = spectra_dict[key][0].shape[-1] - 1

                    elif key == '[cnmr]':
                        seq_length = spectra_dict[key][0].shape[-1] - 1

                    elif key == '[hsqc]':
                        seq_length = spectra_dict[key][0].shape[-1] - 1

                    else:
                        seq_length = spectra_dict[key][0].shape[-1] - 1

                    prompt_dict[key] = torch.empty(batch_size, seq_length, embed_dim, device=molecules.device)
                    # Assign masked tokens for masked indices
                    if masked_batch_ids is not None:
                        prompt_dict[key][masked_batch_ids] = self.masked_tokens(
                            torch.zeros(len(masked_batch_ids), seq_length, device=molecules.device).long()
                        )
                    # Compute embeddings only for unmasked indices
                    # print(unmasked_batch_ids)
                    if len(unmasked_batch_ids) > 0:
                        if key == '[ir]':
                            unmasked_spectra = spectra_dict[key][unmasked_batch_ids]
                        elif key == '[cnmr]':
                            unmasked_spectra = [spectra_dict[key][i][unmasked_batch_ids] for i in range(2)]
                        elif key == '[hnmr]':
                            unmasked_spectra = [spectra_dict[key][i][unmasked_batch_ids] for i in range(4)]
                        else:
                            unmasked_spectra = [spectra_dict[key][i][unmasked_batch_ids] for i in range(3)]
                        # print(key,len(unmasked_spectra[0][0]))
                        prompt_dict[key][unmasked_batch_ids] = self.head_gpts[key].forward_to_embeds(unmasked_spectra)
                else:
                    # Compute embeddings for all data if no masking is applied
                    prompt_dict[key] = self.head_gpts[key].forward_to_embeds(spectra_dict[key])

        # Check if prompt_dict is empty
        if not prompt_dict:
            raise ValueError("prompt_dict is empty, cannot concatenate")

        # Concatenate prompts
        prompt = torch.concat(list(prompt_dict.values()), dim=1)
        prompt, _ = self.cascade_bert.forward(prompt)

        # Generate spectra with tail GPT
        molecules_idx, loss = self.tail_gpt.forward_with_prompt_duringtrain(
            molecules=molecules, prompt=prompt
        )
        return molecules_idx, loss

    def forward_per_modality(self, molecules, spectra_dict, random_mask_rate=0.0, mask_modals=None, return_logits=True):
        """
        Forward pass for refining molecular representations with spectral prompts, optimized to avoid redundant embedding calculations.
        Now with per-modality independent random masking: for each specified modality, independently mask a random subset of samples.

        Args:
            molecules (torch.Tensor): Input molecular representations [batch_size, ...].
            spectra_dict (dict): Dictionary of spectral data for different modalities.
            random_mask_rate (float): Fraction of samples to mask per modality (0 to 1).
            mask_modals: Modalities to apply random masking to (list, set, or None).

        Returns:
            tuple: (molecules_idx, loss) - Refined molecule indices and training loss.
        """
        # Input validation (unchanged)
        assert molecules is not None and molecules.shape[0] > 0, "molecules must be a non-empty tensor"
        assert isinstance(spectra_dict, dict), "spectra_dict must be a dictionary"
        assert 0 <= random_mask_rate <= 1, "random_mask_rate must be between 0 and 1"
        assert mask_modals is None or isinstance(mask_modals, (list, set)), "mask_modals must be a list, set, or None"

        prompt_dict = {}
        batch_size = molecules.shape[0]
        all_batch_ids = torch.arange(batch_size, device=molecules.device, dtype=torch.long)

        for key in self.head_gpts.keys():

            if key in self.allowed_modalities:

                # Determine seq_length (unchanged)
                if key == '[ir]':
                    seq_length = spectra_dict[key].shape[-1] // self.head_gpts['[ir]'].config.n_patchsize - 1
                elif key in ['[hnmr]', '[cnmr]', '[hsqc]']:
                    seq_length = spectra_dict[key][0].shape[-1] - 1
                else:
                    seq_length = spectra_dict[key][0].shape[-1] - 1

                # Initialize prompt tensor for this modality
                prompt_dict[key] = torch.empty(batch_size, seq_length, self.embed_dim, device=molecules.device)

                if random_mask_rate > 0 and mask_modals and key in mask_modals:
                    # Per-modality independent masking: randomly select samples to mask for this modality
                    mask_num = int(random_mask_rate * batch_size)
                    masked_batch_ids = torch.randperm(batch_size, device=molecules.device)[:mask_num]
                    unmasked_batch_ids = torch.ones(batch_size, dtype=torch.bool, device=molecules.device)
                    unmasked_batch_ids[masked_batch_ids] = False
                    unmasked_batch_ids = all_batch_ids[unmasked_batch_ids]

                    # Assign masked tokens for masked samples
                    if len(masked_batch_ids) > 0:
                        prompt_dict[key][masked_batch_ids] = self.masked_tokens(
                            torch.zeros(len(masked_batch_ids), seq_length, device=molecules.device).long()
                        )

                    # Compute embeddings only for unmasked samples
                    if len(unmasked_batch_ids) > 0:
                        # Extract spectra for unmasked samples (unchanged logic)
                        if key == '[ir]':
                            unmasked_spectra = spectra_dict[key][unmasked_batch_ids]
                        elif key == '[cnmr]':
                            unmasked_spectra = [spectra_dict[key][i][unmasked_batch_ids] for i in range(2)]
                        elif key == '[hnmr]':
                            unmasked_spectra = [spectra_dict[key][i][unmasked_batch_ids] for i in range(4)]
                        else:
                            unmasked_spectra = [spectra_dict[key][i][unmasked_batch_ids] for i in range(3)]

                        prompt_dict[key][unmasked_batch_ids] = self.head_gpts[key].forward_to_embeds(unmasked_spectra)
                else:
                    # No masking: compute embeddings for all samples
                    prompt_dict[key] = self.head_gpts[key].forward_to_embeds(spectra_dict[key])
            else:

                seq_length = self.tokens[key]

                prompt_dict[key] = self.masked_tokens(
                            torch.zeros(batch_size, seq_length, device=molecules.device).long()
                        )

        # Check if prompt_dict is empty (unchanged)
        if not prompt_dict:
            raise ValueError("prompt_dict is empty, cannot concatenate")

        # for key in prompt_dict.keys():
        #     print(key,prompt_dict[key].shape)

        # Concatenate prompts (unchanged)
        prompt = torch.concat(list(prompt_dict.values()), dim=1)

        prompt, _ = self.cascade_bert.forward(prompt)

        # Generate spectra with tail GPT (unchanged)
        molecules_idx, loss = self.tail_gpt.forward_with_prompt_duringtrain(molecules=molecules, prompt=prompt)
        return molecules_idx, loss


    def forward_per_modality_with_assign(self,
                                         molecules,
                                         spectra_dict,
                                         assignment,
                                         random_mask_rate=0.0,
                                         mask_modals=None,
                                         return_logits=True):
        """
        Forward pass for refining molecular representations with spectral prompts, optimized to avoid redundant embedding calculations.
        Now with per-modality independent random masking: for each specified modality, independently mask a random subset of samples.

        Args:
            molecules (torch.Tensor): Input molecular representations [batch_size, ...].
            spectra_dict (dict): Dictionary of spectral data for different modalities.
            random_mask_rate (float): Fraction of samples to mask per modality (0 to 1).
            mask_modals: Modalities to apply random masking to (list, set, or None).

        Returns:
            tuple: (molecules_idx, loss) - Refined molecule indices and training loss.
        """
        # Input validation (unchanged)
        assert molecules is not None and molecules.shape[0] > 0, "molecules must be a non-empty tensor"
        assert isinstance(spectra_dict, dict), "spectra_dict must be a dictionary"
        assert 0 <= random_mask_rate <= 1, "random_mask_rate must be between 0 and 1"
        assert mask_modals is None or isinstance(mask_modals, (list, set)), "mask_modals must be a list, set, or None"

        prompt_dict = {}
        batch_size = molecules.shape[0]
        all_batch_ids = torch.arange(batch_size, device=molecules.device, dtype=torch.long)

        for key in self.head_gpts.keys():

            if key in self.allowed_modalities:

                # Determine seq_length (unchanged)
                if key == '[ir]':
                    seq_length = spectra_dict[key].shape[-1] // self.head_gpts['[ir]'].config.n_patchsize - 1
                elif key in ['[hnmr]', '[cnmr]', '[hsqc]']:
                    seq_length = spectra_dict[key][0].shape[-1] - 1
                else:
                    seq_length = spectra_dict[key][0].shape[-1] - 1

                # Initialize prompt tensor for this modality
                prompt_dict[key] = torch.empty(batch_size, seq_length, self.embed_dim, device=molecules.device)

                if random_mask_rate > 0 and mask_modals and key in mask_modals:
                    # Per-modality independent masking: randomly select samples to mask for this modality
                    mask_num = int(random_mask_rate * batch_size)
                    masked_batch_ids = torch.randperm(batch_size, device=molecules.device)[:mask_num]
                    unmasked_batch_ids = torch.ones(batch_size, dtype=torch.bool, device=molecules.device)
                    unmasked_batch_ids[masked_batch_ids] = False
                    unmasked_batch_ids = all_batch_ids[unmasked_batch_ids]

                    # Assign masked tokens for masked samples
                    if len(masked_batch_ids) > 0:
                        prompt_dict[key][masked_batch_ids] = self.masked_tokens(
                            torch.zeros(len(masked_batch_ids), seq_length, device=molecules.device).long()
                        )

                    # Compute embeddings only for unmasked samples
                    if len(unmasked_batch_ids) > 0:
                        # Extract spectra for unmasked samples (unchanged logic)
                        if key == '[ir]':
                            unmasked_spectra = spectra_dict[key][unmasked_batch_ids]
                        elif key == '[cnmr]':
                            unmasked_spectra = [spectra_dict[key][i][unmasked_batch_ids] for i in range(2)]
                        elif key == '[hnmr]':
                            unmasked_spectra = [spectra_dict[key][i][unmasked_batch_ids] for i in range(4)]
                        else:
                            unmasked_spectra = [spectra_dict[key][i][unmasked_batch_ids] for i in range(3)]

                        prompt_dict[key][unmasked_batch_ids] = self.head_gpts[key].forward_to_embeds(unmasked_spectra)
                else:
                    # No masking: compute embeddings for all samples
                    prompt_dict[key] = self.head_gpts[key].forward_to_embeds(spectra_dict[key])
            else:

                seq_length = self.tokens[key]

                prompt_dict[key] = self.masked_tokens(
                            torch.zeros(batch_size, seq_length, device=molecules.device).long()
                        )

        # Check if prompt_dict is empty (unchanged)
        if not prompt_dict:
            raise ValueError("prompt_dict is empty, cannot concatenate")

        # for key in prompt_dict.keys():
        #     print(key,prompt_dict[key].shape)

        # Concatenate prompts (unchanged)
        prompt = torch.concat(list(prompt_dict.values()), dim=1)

        prompt, _ = self.cascade_bert.forward(prompt)

        # Generate spectra with tail GPT (unchanged)
        molecules_idx, loss = self.tail_gpt.forward_with_prompt_attn(molecules=molecules, prompt=prompt,assigment=assignment)
        return molecules_idx, loss

    def generate(self,spectra_dict,max_new_tokens=64, random_mask_rate=0.0,
                mask_modals=None,if_kv_cache=False,if_monte_carlo=True,search:str='greedy'):
        if if_monte_carlo:
            self.tail_gpt.train()
            for keys in self.head_gpts:
                self.head_gpts[keys].train()
            self.cascade_bert.train()
        else:
            self.tail_gpt.eval()
            for keys in self.head_gpts:
                self.head_gpts[keys].eval()
            self.cascade_bert.eval()

        # the head gpt generate all the tokens
        prompt_dict = {}
        for keys in spectra_dict.keys():
            if keys in self.head_gpts:
                prompt_dict[keys] = \
                    self.head_gpts[keys].forward_to_embeds(spectra_dict[keys])


        if random_mask_rate > 0:
            for keys in spectra_dict.keys():
                if keys in mask_modals:
                    # print(keys)
                    random_mask_num = int(random_mask_rate * prompt_dict[keys].shape[0])
                    masked_batch_ids = torch.randperm(prompt_dict[keys].shape[0])[0:random_mask_num]
                    seq_length = prompt_dict[keys].shape[1]
                    prompt_dict[keys][masked_batch_ids] = self.masked_tokens(torch.zeros(random_mask_num,seq_length).long().to(prompt_dict[keys].device))

        # print(spectra_dict,
        #       prompt_dict)
        # prompt stack
        # print(spectra_dict.keys())
        prompt = torch.concat(list(prompt_dict.values()),dim=1)
        prompt,_ = self.cascade_bert.forward(prompt)
        # print(prompt.shape, prompt)

        # the tail gpt generate the spectra
        if search.lower() == 'greedy':
            mol = self.tail_gpt.generate_with_prompt(prompt, max_new_tokens=max_new_tokens, if_kv_cache=if_kv_cache)
        elif search.lower() == 'beam':
            mol = self.tail_gpt.generate_with_beam_search(prompt, max_new_tokens=max_new_tokens,
                                                              if_kv_cache=if_kv_cache)
        elif search.lower() == 'hybrid':
            mol = self.tail_gpt.generate_with_hybrid_search(prompt, max_new_tokens=max_new_tokens,
                                                                if_kv_cache=if_kv_cache,beam_width=3,beam_steps=2)
        else: print('search params errors')
        return mol

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        self.masked_tokens.load_state_dict(state_dict['masked_tokens'])

    def save_state_dict(
            self,
            save_path
    ):
        state_dict = {"masked_tokens":self.masked_tokens.state_dict()}
        torch.save(state_dict, save_path)
        print("save the proj weights to {}".format(save_path))

    def generate_prob(self,spectra_dict,max_new_tokens=64,search ='hybrid',
                      if_monte_carlo=True):
        # the head gpt generate all the tokens
        if if_monte_carlo:
            self.tail_gpt.train()
            for keys in self.head_gpts:
                self.head_gpts[keys].train()
            self.cascade_bert.train()
        else:
            self.tail_gpt.eval()
            for keys in self.head_gpts:
                self.head_gpts[keys].eval()
            self.cascade_bert.eval()

        prompt_dict = {}
        for keys in spectra_dict.keys():
            prompt_dict[keys] = \
                self.head_gpts[keys].forward_to_embeds(spectra_dict[keys])
        # prompt stack
        prompt = torch.concat(list(prompt_dict.values()),dim=1)
        prompt,_ = self.cascade_bert.forward(prompt)
        # the tail gpt generate the spectra
        if search.lower() == 'hybrid':
            spectra, probs = self.tail_gpt.generate_with_hybrid_search_with_prob(prompt,
                                                                    max_new_tokens=max_new_tokens,
                                                                    beam_steps=3,beam_width=3,
                                                                                 if_kv_cache=False)
        if search.lower() == 'greedy':
            spectra,probs = self.tail_gpt.generate_with_prompt_prob(prompt,max_new_tokens=max_new_tokens)
        return spectra,probs

    def beam_search(self,spectra_dict,max_new_tokens=64,
                    beam_width=5,
                    pad_token_id=None,
                    eos_token_id=None,
                    ):
        # the head gpt generate all the tokens
        prompt_dict = {}
        for keys in spectra_dict.keys():
            prompt_dict[keys] = \
                self.head_gpts[keys].forward_to_embeds(spectra_dict[keys])
        # prompt stack
        prompt = torch.concat(list(prompt_dict.values()),dim=1)
        prompt,_ = self.cascade_bert.forward(prompt)
        # the tail gpt generate the spectra
        mol = self.tail_gpt.beam_search_with_prompt(prompt,max_new_tokens=max_new_tokens,
                                                    num_beams=beam_width,
                                                    pad_token_id=pad_token_id,
                                                    eos_token_id=eos_token_id)
        return mol

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

        return optimizer

    def embed_with_missing_modality(self, spectra_dict, missing_identification: str = '<missing>',
                                    tokens={'[ir]': 31, '[hnmr]': 22, '[cnmr]': 66, '[hsqc]': 66}):
        '''
        Generate embeddings for spectra_dict, handling missing modalities, and return encoded prompt.

        Args:
            spectra_dict (dict): Dictionary with modality keys (e.g., '[ir]', '[hnmr]', '[cnmr]', '[hsqc]') and values as lists
                               containing tensors, lists of tensors, or '<missing>' for each batch sample.
            missing_identification (str): Identifier for missing modality data. Defaults to '<missing>'.
            tokens (dict): Dictionary mapping modality keys to their sequence lengths. Defaults to predefined values.

        Returns:
            torch.Tensor: Encoded prompt after concatenating embeddings and processing through cascade BERT.
        '''
        prompt_dict = {}
        batch_size = None
        device = None

        # Validate batch_size consistency across modalities
        for key in spectra_dict.keys():
            if key in self.head_gpts:
                if batch_size is None:
                    batch_size = len(spectra_dict[key])
                elif len(spectra_dict[key]) != batch_size:
                    raise ValueError(
                        f"Inconsistent batch size for modality {key}: expected {batch_size}, got {len(spectra_dict[key])}")
        if batch_size is None:
            raise ValueError("No valid modalities found in spectra_dict.")

        # Process each modality
        for key in spectra_dict.keys():
            if key not in self.head_gpts or key not in tokens:
                continue

            # Infer device from first valid sample
            if device is None:
                for sample in spectra_dict[key]:
                    if sample != missing_identification:
                        device = sample.device if isinstance(sample, torch.Tensor) else sample[0].device
                        break
                device = device or next(self.head_gpts[key].parameters()).device

            # Get sequence length from tokens
            seq_length = tokens[key]
            embed_dim = self.embed_dim

            # Initialize embedding tensor
            if key in ['[hsqc]','[cnmr]','[hnmr]']:
                prompt_dict[key] = torch.empty(batch_size, seq_length-1, embed_dim, device=device)

            else:
                prompt_dict[key] = torch.empty(batch_size, seq_length, embed_dim, device=device)

            # Identify masked and unmasked indices
            masked_batch_ids = [i for i, sample in enumerate(spectra_dict[key]) if sample == missing_identification]
            unmasked_batch_ids = [i for i, sample in enumerate(spectra_dict[key]) if sample != missing_identification]

            # Apply masked tokens for missing samples
            if masked_batch_ids:
                if key in ['[hsqc]','[cnmr]','[hnmr]']:
                    prompt_dict[key][masked_batch_ids] = self.masked_tokens(
                        torch.zeros(len(masked_batch_ids), seq_length-1, device=device).long()
                    )
                else:
                    prompt_dict[key][masked_batch_ids] = self.masked_tokens(
                        torch.zeros(len(masked_batch_ids), seq_length, device=device).long()
                    )


            # Compute embeddings for valid samples
            if len(unmasked_batch_ids) != 0:
                if key == '[ir]':
                    # [ir] expects a list of tensors
                    unmasked_spectra = [spectra_dict[key][i] for i in unmasked_batch_ids]
                    unmasked_spectra = torch.stack(unmasked_spectra,dim=0).unsqueeze(1)
                elif key in ['[hnmr]']:
                    # [hnmr], [cnmr] expect a list of 4 sub-tensors per sample
                    unmasked_spectra = []
                    for i in range(4):
                        neo_spec = torch.zeros(len(unmasked_batch_ids), seq_length)
                        for j in range(len(unmasked_batch_ids)):
                            # print(spectra_dict[key])
                            neo_spec[j] = spectra_dict[key][j][i]
                        unmasked_spectra.append(neo_spec.long().to(device))

                elif key in ['[cnmr]']:
                    # [hnmr], [cnmr] expect a list of 4 sub-tensors per sample
                    unmasked_spectra = []
                    for i in range(2):
                        neo_spec = torch.zeros(len(unmasked_batch_ids), seq_length)
                        for j in range(len(unmasked_batch_ids)):
                            # print(spectra_dict[key])
                            neo_spec[j] = spectra_dict[key][j][i]
                        unmasked_spectra.append(neo_spec.long().to(device))

                else:  # [hsqc]
                    # [hsqc] expects a list of 5 sub-tensors per sample
                    # unmasked_spectra = [spectra_dict[key][i][unmasked_batch_ids] for i in range(5)]
                    unmasked_spectra = []
                    for i in range(3):
                        neo_spec = torch.zeros(len(unmasked_batch_ids), seq_length)
                        for j in range(len(unmasked_batch_ids)):
                            neo_spec[j] = spectra_dict[key][j][i]
                        unmasked_spectra.append(neo_spec.long().to(device))
                # Compute embedding
                prompt_dict[key][unmasked_batch_ids] = self.head_gpts[key].forward_to_embeds(unmasked_spectra)

        # Validate prompt_dict
        if not prompt_dict:
            raise ValueError("No valid modalities found in spectra_dict matching head_gpts.")

        # Concatenate embeddings along sequence dimension
        prompt = torch.concat(list(prompt_dict.values()), dim=1)

        # Process through cascade BERT
        prompt, _ = self.cascade_bert.forward(prompt)

        return prompt

    def generate_with_missing_modality_with_prob(self, spectra_dict, max_new_tokens=64, search='hybrid',
                                                 missing_identification: str = '<missing>',
                                                 tokens={'[ir]': 31, '[hnmr]': 22, '[cnmr]': 66, '[hsqc]': 66},
                                                 if_monte_carlo: bool = True):
        '''
        Generate spectra with probability output, handling missing modalities in spectra_dict.

        Args:
            spectra_dict (dict): Dictionary with modality keys (e.g., '[ir]', '[hnmr]', '[cnmr]', '[hsqc]') and values as lists
                               containing tensors, lists of tensors, or '<missing>' for each batch sample.
            max_new_tokens (int): Maximum number of tokens to generate. Defaults to 64.
            search (str): Search mode for generation ('hybrid' or 'greedy'). Defaults to 'hybrid'.
            missing_identification (str): Identifier for missing modality data. Defaults to '<missing>'.
            tokens (dict): Dictionary mapping modality keys to their sequence lengths. Defaults to predefined values.
            if_monte_carlo (bool): Whether to use Monte Carlo mode (train mode). Defaults to True.

        Returns:
            tuple: (spectra, probs) where spectra is the generated spectra tensor and probs is the probability output.
        '''
        # Set model modes based on if_monte_carlo
        if if_monte_carlo:
            self.tail_gpt.train()
            for key in self.head_gpts:
                self.head_gpts[key].train()
            self.cascade_bert.train()
        else:
            self.tail_gpt.eval()
            for key in self.head_gpts:
                self.head_gpts[key].eval()
            self.cascade_bert.eval()

        # Generate encoded prompt using shared embedding method
        prompt = self.embed_with_missing_modality(spectra_dict, missing_identification, tokens)

        # Generate spectra with probability output
        if search.lower() == 'hybrid':
            spectra, probs = self.tail_gpt.generate_with_hybrid_search_with_prob(
                prompt, max_new_tokens=max_new_tokens, beam_steps=3, beam_width=3, if_kv_cache=False
            )
        elif search.lower() == 'greedy':
            spectra, probs = self.tail_gpt.generate_with_prompt_prob(
                prompt, max_new_tokens=max_new_tokens
            )
        else:
            raise ValueError(f"Unsupported search mode: {search}. Expected 'hybrid' or 'greedy'.")

        return spectra, probs

if __name__ == '__main__':
    # test
    tail_gpt = MolGPT(MolConfig())
    spec_gpt = SpectraGPT(SpectraConfig())
    hnmr_gpt = HNMRGPT(HNMRConfig())
    cnmr_gpt = CNMRGPT(CNMRConfig())
    head_gpt = {'[ir]':spec_gpt,'[hnmr]':hnmr_gpt,'[cnmr]':cnmr_gpt}
    cascade_bert = BERT(BERTConfig())
    model = MultiSpec2Mol(head_gpt,tail_gpt,cascade_bert,['head_gpt','tail_gpt'])
    batch_size = 4
    sig_len = 512
    seq_len = 16
    ir_peaks = torch.randn([batch_size,1,sig_len])
    category = torch.randint(0, 10, [batch_size, seq_len])
    centroids = torch.randint(0, 10, [batch_size, seq_len])
    jvalue = torch.randint(0, 10, [batch_size, seq_len])
    nH = torch.randint(0, 10, [batch_size, seq_len])
    delta = torch.randint(0, 10, [batch_size, seq_len])
    width = torch.randint(0, 10, [batch_size, seq_len])
    integral = torch.randint(0, 10, [batch_size, seq_len])
    intensity = torch.randint(0, 10, [batch_size, seq_len])


    molecules = torch.randint(0, 10, [batch_size, 1])
    spectra_dict = {'[ir]': ir_peaks, '[hnmr]': [category, centroids, jvalue, nH],
        '[cnmr]': [delta, width, integral, intensity]}
    mol = model.beam_search(spectra_dict, max_new_tokens=64,
                            beam_width=5,
                            pad_token_id=2,
                            eos_token_id=1)
    print(mol.shape)
    print(mol)