import os
import math
import random
from contextlib import nullcontext

from click import prompt
import yaml
from dataset_peaks import dataset
# from dataset_toy import dataset
import numpy as np
import torch
from molecule_pretrain.model import GPTConfig as MolConfig, GPT as MolGPT
from Mol2SpectraLong.model import BERT, BERTConfig
from spectra_prdiction.model import GPT as SpectraGPT, GPTConfig as SpectraConfig
from Mol2SpectraLong.cascade import Mol2Spectra
import torch.utils.data as tud

# -----------------------------------------------------------------------------
def load_SpectraGPT(config: dict,relative_dir:str):
    if relative_dir == '':
        out_dir = config['out_dir']
    else:
        out_dir = os.path.join(relative_dir,config['out_dir'])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    eval_interval = config['eval_interval']
    log_interval = config['log_interval']
    eval_iters = config['eval_iters']
    eval_only = config['eval_only']
    always_save_checkpoint = config['always_save_checkpoint']
    init_from = config['init_from']
    wandb_log = config['wandb_log']
    wandb_project = config['wandb_project']
    wandb_run_name = config['wandb_run_name']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    batch_size = config['batch_size']
    block_size = config['block_size']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    n_patchsize = config['n_patchsize']
    signal_length = config['signal_length']
    dropout = config['dropout']
    bias = config['bias']
    learning_rate = config['learning_rate']
    max_iters = config['max_iters']
    weight_decay = config['weight_decay']
    beta1 = config['beta1']
    beta2 = config['beta2']
    grad_clip = config['grad_clip']
    decay_lr = config['decay_lr']
    warmup_iters = config['warmup_iters']
    lr_decay_iters = config['lr_decay_iters']
    min_lr = config['min_lr']
    backend = config['backend']
    device = 'cpu'
    dtype = config['dtype']
    compile = config['compile']
    config_keys = config['config_keys']
    gpu_ids = config['gpu_ids']
    device_type = config['device_type']
    num_workers = config['num_workers']
    dp = config['dp'] if len(gpu_ids) > 1 else False
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    data_path = config['data_path']
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    best_val_loss = 1e9

    prompt_pe = config['prompt_pe'] if 'prompt_pe' in config else True
    print('if add position embeddings on Prompt TOKENS', prompt_pe)

    start_generate_token = config['start_generate_token'] if 'start_generate_token' in config else False
    print('if use start_generate_token', start_generate_token)
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = SpectraConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            n_patchsize=n_patchsize, signal_length=signal_length,
                            dropout=dropout, bias=bias,
                            prompt_pe=prompt_pe,
                            start_generate_token=start_generate_token)

    model_args = dict(n_layer=n_layer, n_head=n_head,
                      n_embd=n_embd, block_size=block_size,
                      bias=bias,dropout=dropout,
                      n_patchsize=n_patchsize,
                      signal_length=signal_length,
                      prompt_pe = prompt_pe,
                      start_generate_token=start_generate_token)  # start with model_args from command line

    model = SpectraGPT(gpt_config)
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        gptconf = SpectraConfig(**model_args)
        model = SpectraGPT(gptconf)
        print(f"initializing model with vq-vae config: {gptconf}")


    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias','n_patchsize','signal_length','dropout']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
            else:
                model_args[k] = config[k]
        # create the model
        gptconf = SpectraConfig(**model_args)
        model = SpectraGPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        # for k, v in list(state_dict.items()):
        #     if k.startswith(unwanted_prefix):
        #         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    return model, model_args

def load_MolGPT(config: dict,relative_dir:str):
    if relative_dir == '':
        out_dir = config['out_dir']
    else:
        out_dir = os.path.join(relative_dir,config['out_dir'])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    eval_interval = config['eval_interval']
    log_interval = config['log_interval']
    eval_iters = config['eval_iters']
    eval_only = config['eval_only']
    always_save_checkpoint = config['always_save_checkpoint']
    init_from = config['init_from']
    wandb_log = config['wandb_log']
    wandb_project = config['wandb_project']
    wandb_run_name = config['wandb_run_name']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    batch_size = config['batch_size']
    block_size = config['block_size']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    n_patchsize = config['n_patchsize']
    signal_length = config['signal_length']
    dropout = config['dropout']
    bias = config['bias']
    learning_rate = config['learning_rate']
    max_iters = config['max_iters']
    weight_decay = config['weight_decay']
    beta1 = config['beta1']
    beta2 = config['beta2']
    grad_clip = config['grad_clip']
    decay_lr = config['decay_lr']
    warmup_iters = config['warmup_iters']
    lr_decay_iters = config['lr_decay_iters']
    min_lr = config['min_lr']
    backend = config['backend']
    device = 'cpu'
    dtype = config['dtype']
    vocab_size = config['vocab_size']
    compile = config['compile']
    config_keys = config['config_keys']
    gpu_ids = config['gpu_ids']
    device_type = config['device_type']
    num_workers = config['num_workers']
    dp = config['dp'] if len(gpu_ids) > 1 else False
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    data_path = config['data_path']
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    best_val_loss = 1e9


    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = MolConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            vocab_size=vocab_size, block_size=block_size,
                            dropout=dropout, bias=bias)

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias,dropout=dropout)  # start with model_args from command line
    model = MolGPT(gpt_config)
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        gptconf = MolConfig(**model_args)
        model = MolGPT(gptconf)
        print(f"initializing model with config: {gptconf}")
        iter_num = 0


    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # print(checkpoint_model_args)
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        # for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias','vocab_size','dropout']:
        #     if k in checkpoint_model_args:
        #         model_args[k] = checkpoint_model_args[k]
        #     else:
        #         model_args[k] = config[k]
        # # create the model
        # gptconf = MolConfig(**model_args)
        # model = MolGPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        # for k, v in list(state_dict.items()):
        #     if k.startswith(unwanted_prefix):
        #         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        iter_num = checkpoint['iter_num']
        # best_val_loss = checkpoint['best_val_loss']

    return model, model_args

def load_BERT(config: dict,out_dir=None):
    # task = 'mol_predict_ir'
    if out_dir == None:
        out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    eval_interval = config['eval_interval']
    log_interval = config['log_interval']
    eval_iters = config['eval_iters']
    eval_only = config['eval_only']
    always_save_checkpoint = config['always_save_checkpoint']
    init_from = config['init_from']
    wandb_log = config['wandb_log']
    wandb_project = config['wandb_project']
    wandb_run_name = config['wandb_run_name']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    batch_size = config['batch_size']
    block_size = config['block_size']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    n_learnable_tokens = config['n_learnable_tokens']
    dropout = config['dropout']
    bias = config['bias']
    learning_rate = config['learning_rate']
    max_iters = config['max_iters']
    weight_decay = config['weight_decay']
    beta1 = config['beta1']
    beta2 = config['beta2']
    grad_clip = config['grad_clip']
    decay_lr = config['decay_lr']
    warmup_iters = config['warmup_iters']
    lr_decay_iters = config['lr_decay_iters']
    min_lr = config['min_lr']
    backend = config['backend']
    device = config['device']
    dtype = config['dtype']
    gpu_ids = config['gpu_ids']
    device_type = config['device_type']
    num_workers = config['num_workers']
    dp = config['dp'] if len(gpu_ids) > 1 else False
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    data_path = config['data_path']
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    best_val_loss = 1e9
    frozen_model = config['frozen_model']

    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = BERTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            block_size=block_size, n_learnable_tokens=n_learnable_tokens,
                            dropout=dropout, bias=bias)

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      n_learnable_tokens=n_learnable_tokens,
                      bias=bias, dropout=dropout)  # start with model_args from command line
    model = BERT(gpt_config)

    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        gptconf = BERTConfig(**model_args)
        model = BERT(gptconf)
        print(f"initializing model with config: {gptconf}")
        iter_num = 0


    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
            else:
                model_args[k] = config[k]
        # create the model
        bertconf = BERTConfig(**model_args)
        model = BERT(bertconf)
        state_dict = checkpoint['model']
        # for k, v in list(state_dict.items()):
        #     print(k,v.shape)
        model.load_state_dict(state_dict, strict=False)
    return model, model_args

def load_spec_GPT(spectra_config,spectra_related_dir,task:str):
    task = task.lower()
    spectra_gpt, spectra_args = load_SpectraGPT(spectra_config,spectra_related_dir)
    return spectra_gpt,spectra_args

def load_backward(output_dir:str,
                  agent_path:str,
                  config_path:str= 'config_[all_unfroz]_NoPromptPE.yaml',
                  task:str='cnmr',
                  frozen_model:str=[],
                  device='cpu'):

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    mol_related_dir = os.path.join(output_dir, agent_path)
    spectra_related_dir = os.path.join(output_dir, agent_path)

    spectra_path = 'config_spectra_1.yaml'
    spectra_path = os.path.join(spectra_related_dir, spectra_path)
    spectra_config = yaml.load(open(spectra_path, 'r'), Loader=yaml.FullLoader)

    mol_path = 'config_mol_1.yaml'
    mol_path = os.path.join(mol_related_dir, mol_path)
    mol_config = yaml.load(open(mol_path, 'r'), Loader=yaml.FullLoader)

    mol_gpt, mol_args = load_MolGPT(config=mol_config, relative_dir=mol_related_dir)
    spec_gpt, spec_args = load_spec_GPT(spectra_config=spectra_config, spectra_related_dir=spectra_related_dir,
                                        task=task)
    output_dir = os.path.join(output_dir, config['out_dir'])
    cascade_bert, model_args = load_BERT(config=config,out_dir=output_dir)
    model = Mol2Spectra(head_gpt=mol_gpt, tail_gpt=spec_gpt, cascade_bert=cascade_bert, frozen_model=frozen_model)
    return model.to(device)