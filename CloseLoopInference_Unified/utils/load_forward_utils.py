import os
import math
from contextlib import nullcontext

from websockets import connect
import yaml
from dataset_peaks import dataset
import numpy as np
import torch
from molecule_pretrain.model import GPTConfig as MolConfig, GPT as MolGPT
from MultiSpec2Mol_Pubchem.model import BERT, BERTConfig
from MultiSpec2Mol_Pubchem.cascade import MultiSpec2Mol
from spectra_prdiction.model import GPT as SpectraGPT, GPTConfig as SpectraConfig
from spectra_prediction_token.model import GPTConfig as HNMRConfig, GPT as HNMRGPT
from spectra_prediction_token.model_cnmr import GPTConfig as CNMRConfig, GPT as CNMRGPT
from spectra_prediction_token.model_hsqc import GPTConfig as HSQCConfig, GPT as HSQCGPT
from molecule_pretrain.token_decode import token_decode
from mol_smilarity_metric import fingerprint_similarity_metric,equivalent_similarity_metric
import torch.utils.data as tud
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Optional, Union, Tuple

def load_HSQCNMR_GPT(config: dict,relative_dir:str):
    if relative_dir == '':
        out_dir = config['out_dir']
    else:
        out_dir = os.path.join(relative_dir, config['out_dir'])

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    init_from = config['init_from']
    block_size = config['block_size']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    dropout = config['dropout']
    bias = config['bias']
    device = config['device']

    # -----------------------------------------------------------------------------
    gpt_config = HSQCConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,

                            dropout=dropout, bias=bias)
    model_args = dict(n_layer=n_layer,
                      n_head=n_head,
                      n_embd=n_embd,
                      block_size=block_size,
                      bias=bias,dropout=dropout)  # start with model_args from command line
    model = HSQCGPT(gpt_config)
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        print(f"initializing model with vq-vae config: {gpt_config}")

    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
            else:
                model_args[k] = config[k]
        # create the model
        gptconf = HSQCConfig(**model_args)
        # print(gptconf.nH)
        # print(checkpoint['model'].keys())
        # print(checkpoint['model']['transformer.nH_embedding.weight'].shape)
        model = HSQCGPT(gptconf)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict, strict=False)

    return model,model_args

def load_IR_GPT(config: dict,relative_dir:str):
    if relative_dir == '':
        out_dir = config['out_dir']
    else:
        out_dir = os.path.join(relative_dir,config['out_dir'])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    init_from = config['init_from']
    block_size = config['block_size']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    n_patchsize = config['n_patchsize']
    signal_length = config['signal_length']
    dropout = config['dropout']
    bias = config['bias']
    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = SpectraConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            n_patchsize=n_patchsize, signal_length=signal_length,
                            dropout=dropout, bias=bias)
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias,dropout=dropout,n_patchsize=n_patchsize,signal_length=signal_length)  # start with model_args from command line
    model = SpectraGPT(gpt_config)
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        gptconf = SpectraConfig(**model_args)
        model = SpectraGPT(gptconf)
        print(f"initializing model with config: {gptconf}")


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
    return model,model_args

def load_HNMR_GPT(config: dict,relative_dir:str):
    if relative_dir == '':
        out_dir = config['out_dir']
    else:
        out_dir = os.path.join(relative_dir,config['out_dir'])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    init_from = config['init_from']
    block_size = config['block_size']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    dropout = config['dropout']
    bias = config['bias']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    beta1 = config['beta1']
    beta2 = config['beta2']
    device = config['device']
    dtype = config['dtype']
    gpu_ids = config['gpu_ids']
    device_type = config['device_type']
    dp = config['dp'] if len(gpu_ids) > 1 else False
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = HNMRConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,

                           dropout=dropout, bias=bias)
    model_args = dict(n_layer=n_layer,
                      n_head=n_head,
                      n_embd=n_embd,
                      block_size=block_size,
                      bias=bias, dropout=dropout)  # start with model_args from command line
    model = HNMRGPT(gpt_config)
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        gptconf = HNMRConfig(**model_args)
        model = HNMRGPT(gptconf).to(device)
        print(f"initializing model with vq-vae config: {gptconf}")
        iter_num = 0


    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
            else:
                model_args[k] = config[k]
        # create the model
        gptconf = HNMRConfig(**model_args)
        model = HNMRGPT(gptconf)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict, strict=False)

    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = HNMRGPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size  # so that the checkpoint will have the right value
    model.to(device)
    # -----------------------------------------------------------------------------
    model.to(device)
    return model,model_args

def load_CNMR_GPT(config: dict,relative_dir:str, use_cnmr_intensity:bool=True):
    if relative_dir == '':
        out_dir = config['out_dir']
    else:
        out_dir = os.path.join(relative_dir,config['out_dir'])
    os.makedirs(out_dir, exist_ok=True)
    init_from = config['init_from']
    block_size = config['block_size']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    dropout = config['dropout']
    bias = config['bias']
    device = config['device']
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = CNMRConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            use_intensity=use_cnmr_intensity,
                           dropout=dropout, bias=bias)
    model_args = dict(n_layer=n_layer,
                      n_head=n_head,
                      n_embd=n_embd,
                      block_size=block_size,
                      bias=bias, dropout=dropout,
                      use_intensity=use_cnmr_intensity
                      )  # start with model_args from command line
    model = CNMRGPT(gpt_config)
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        gptconf = CNMRConfig(**model_args)
        model = CNMRGPT(gptconf).to(device)
        print(f"initializing model with vq-vae config: {gptconf}")

    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
            else:
                model_args[k] = config[k]
        # create the model
        gptconf = CNMRConfig(**model_args)
        model = CNMRGPT(gptconf)
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
    return model,model_args


self_define_tokens = {'[predict]':0,'[functional_group]':1,}

def load_MolGPT(config: dict,relative_dir:str):
    if relative_dir == '':
        out_dir = config['out_dir']
    else:
        out_dir = os.path.join(relative_dir,config['out_dir'])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    init_from = config['init_from']
    block_size = config['block_size']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    dropout = config['dropout']
    bias = config['bias']
    vocab_size = config['vocab_size']
    if 'rotary' in config.keys():
        rotary = config['rotary']
    else: rotary = False

    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = MolConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            vocab_size=vocab_size, block_size=block_size,
                            dropout=dropout, bias=bias,rotary=rotary)

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias,dropout=dropout,vocab_size=vocab_size,
                      rotary=rotary)  # start with model_args from command line
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
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias','vocab_size','dropout']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
            else:
                model_args[k] = config[k]
        # create the model
        gptconf = MolConfig(**model_args)
        model = MolGPT(gptconf)
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

    return model,model_args

def load_BERT(config: dict,out_dir=None):
    if out_dir==None:
        out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    init_from = config['init_from']
    block_size = config['block_size']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    n_learnable_tokens = config['n_learnable_tokens']
    dropout = config['dropout']
    bias = config['bias']
    proj_path = os.path.join(out_dir, 'proj_weights.pt')

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
        checkpoint = torch.load(ckpt_path, map_location='cpu')
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
        model.load_state_dict(state_dict, strict=False)

    return model,proj_path,model_args


def load_forward(output_dir:str,
                 config_path:str,
                 agent_path:str='all_unfroz_[hsqcnmr_cnmr_hnmr_ir]_5',
                 connection_path:str=None,
                 frozen_model:list = [],
                 device:str='cpu'
                 ):
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    mol_related_dir = os.path.join(output_dir, agent_path)
    ir_spectra_related_dir = os.path.join(output_dir, agent_path)
    hnmr_spectra_related_dir = os.path.join(output_dir, agent_path)
    cnmr_spectra_related_dir = os.path.join(output_dir, agent_path)
    hsqc_nmr_spectra_related_dir = os.path.join(output_dir, agent_path)
    head_gpt = {}

    try:
        ir_spectra_path = 'config_ir_spectra_1.yaml'
        ir_spectra_path = os.path.join(ir_spectra_related_dir, ir_spectra_path)
        ir_spectra_config = yaml.load(open(ir_spectra_path, 'r'), Loader=yaml.FullLoader)
        ir_spectra_config['ir_related_dir'] = ir_spectra_related_dir
        spec_gpt, ir_args = load_IR_GPT(config=ir_spectra_config, relative_dir=ir_spectra_related_dir)
        spec_gpt = spec_gpt.to(device)
        head_gpt['[ir]'] = spec_gpt
    except:
        print("Failed to load IR spectra")

    try:
        hnmr_spectra_path = 'config_hnmr_spectra_1.yaml'
        hnmr_spectra_path = os.path.join(hnmr_spectra_related_dir, hnmr_spectra_path)
        hnmr_spectra_config = yaml.load(open(hnmr_spectra_path, 'r'), Loader=yaml.FullLoader)
        hnmr_spectra_config['hnmr_related_dir'] = hnmr_spectra_related_dir
        hnmr_gpt, hnmr_args = load_HNMR_GPT(config=hnmr_spectra_config, relative_dir=hnmr_spectra_related_dir)
        hnmr_gpt = hnmr_gpt.to(device)
        head_gpt['[hnmr]'] = hnmr_gpt
        print("Load HNMR spectra!")
    except:
        print("Failed to load HNMR spectra")

    try:

        cnmr_spectra_path = 'config_cnmr_spectra_1.yaml'
        cnmr_spectra_path = os.path.join(cnmr_spectra_related_dir, cnmr_spectra_path)
        cnmr_spectra_config = yaml.load(open(cnmr_spectra_path, 'r'), Loader=yaml.FullLoader)
        cnmr_spectra_config['cnmr_related_dir'] = cnmr_spectra_related_dir
        cnmr_gpt, cnmr_args = load_CNMR_GPT(config=cnmr_spectra_config, relative_dir=cnmr_spectra_related_dir,
                                            use_cnmr_intensity=False)
        cnmr_gpt = cnmr_gpt.to(device)
        head_gpt['[cnmr]'] = cnmr_gpt
        print("Load CNMR spectra!")
    except:
        print("Failed to load CNMR spectra")

    try:
        hsqc_spectra_path = 'config_hsqc_spectra_1.yaml'
        hsqc_spectra_path = os.path.join(hsqc_nmr_spectra_related_dir, hsqc_spectra_path)
        hsqc_spectra_config = yaml.load(open(hsqc_spectra_path, 'r'), Loader=yaml.FullLoader)
        hsqc_spectra_config['hsqc_related_dir'] = hsqc_nmr_spectra_related_dir
        hsqc_gpt, hsqc_args = load_HSQCNMR_GPT(config=hsqc_spectra_config, relative_dir=hsqc_nmr_spectra_related_dir)
        hsqc_gpt = hsqc_gpt.to(device)
        head_gpt['[hsqc]'] = hsqc_gpt
        print("Load HSQC_NMR spectra!")
    except:
        print("Failed to load HSQCNMR spectra")

    mol_path = 'config_mol_1.yaml'
    mol_path = os.path.join(mol_related_dir, mol_path)
    mol_config = yaml.load(open(mol_path, 'r'), Loader=yaml.FullLoader)
    mol_config['mol_related_dir'] = mol_related_dir
    tail_gpt, mol_args = load_MolGPT(config=mol_config, relative_dir=mol_related_dir)
    tail_gpt = tail_gpt.to(device)

    frozen_model = []

    # connection_path = os.path.join(output_dir,connection_path)
    cascade_bert, proj_path, model_args = load_BERT(config=config,out_dir=connection_path)
    model = MultiSpec2Mol(head_gpt, tail_gpt, cascade_bert, frozen_model=frozen_model,
                          )
    if os.path.exists(proj_path):
        print(f"loading projection weights from {proj_path}")
        model.load_state_dict(torch.load(proj_path))

    return model
