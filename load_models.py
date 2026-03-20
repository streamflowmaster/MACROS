
import os
import torch
from molecule_pretrain.model import GPT as MolGPT_SF, GPTConfig as MolConfig_SF
from MultiSpec2Mol_Unified.model import BERT,BERTConfig
from spectra_prdiction.model import GPT as SpectraGPT, GPTConfig as SpectraConfig
from spectra_prediction_token.model import GPTConfig as HNMRConfig, GPT as HNMRGPT
from spectra_prediction_token.model_cnmr import GPTConfig as CNMRConfig, GPT as CNMRGPT
from spectra_prediction_token.model_hsqc import GPTConfig as HSQCConfig, GPT as HSQCGPT
from molecule_pretrain.model import GPTConfig as MolConfig, GPT as MolGPT
from molecule_pretrain.model_attn import GPTConfig as MolConfig, GPT as MolGPT_Attn
# from HNMR_prediction.model import GPTConfig as HNMRSConfig, GPT as HNMRSGPT
# from CNMR_prediction.model import GPTConfig as CNMRSConfig, GPT as CNMRSGPT

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
    if 'rotary' in config:
        rotary = config['rotary']
    else:
        rotary = False

    # -----------------------------------------------------------------------------
    gpt_config = HSQCConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            rotary=rotary,
                            dropout=dropout, bias=bias)
    model_args = dict(n_layer=n_layer,
                      n_head=n_head,
                      n_embd=n_embd,
                      rotary=rotary,
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
    if 'rotary' in config:
        rotary = config['rotary']
    else:
        rotary = False
    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = SpectraConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            n_patchsize=n_patchsize, signal_length=signal_length,
                            dropout=dropout, bias=bias,rotary=rotary)
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias,dropout=dropout,n_patchsize=n_patchsize,signal_length=signal_length,
                      rotary=rotary)  # start with model_args from command line
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
    if 'rotary' in config:
        rotary = config['rotary']
    else:
        rotary = False
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # model settings && initialization
    # config['use_categories_jvalue'] = True
    gpt_config = HNMRConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            rotary=rotary,
                           dropout=dropout, bias=bias,
                            use_categories_jvalue=config.get('use_categories_jvalue', True),)
    model_args = dict(n_layer=n_layer,
                      n_head=n_head,
                      n_embd=n_embd,
                      block_size=block_size,
                      rotary=rotary,
                      use_categories_jvalue=config.get('use_categories_jvalue', True),
                      bias=bias, dropout=dropout)  # start with model_args from command line
    print('use_categories_jvalue', config.get('use_categories_jvalue', True))

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

    return model,model_args

def load_CNMR_GPT(config: dict,relative_dir:str,use_intensity:bool=False):
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
    if 'rotary' in config:
        rotary = config['rotary']
    else:
        rotary = False
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = CNMRConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            rotary=rotary,
                           dropout=dropout, bias=bias,use_intensity=use_intensity)
    model_args = dict(n_layer=n_layer,
                      n_head=n_head,
                      n_embd=n_embd,
                      block_size=block_size,
                      rotary=rotary,
                      bias=bias, dropout=dropout,
                      use_intensity=use_intensity)  # start with model_args from command line
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
        best_val_loss = checkpoint['best_val_loss']
    return model,model_args


self_define_tokens = {'[predict]':0,'[functional_group]':1,}

def load_MolGPT_SF(config: dict,relative_dir:str):
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
    dtype = config['dtype']
    vocab_size = config['vocab_size']
    rotary = config['rotary']
    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = MolConfig_SF(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           vocab_size=vocab_size, block_size=block_size,
                           dropout=dropout, bias=bias,
                           rotary=rotary,
                           bos_token_id=config['bos_id'],
                           eos_token_id=config['eos_id'],
                           pad_token_id=config['pad_id'])


    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, dropout=dropout,
                      rotary = rotary)  # start with model_args from command line
    model = MolGPT_SF(gpt_config)
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        gptconf = MolConfig_SF(**model_args)
        model = MolGPT_SF(gptconf)
        print(f"initializing model with config: {gptconf}")
        iter_num = 0


    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # create the model
        gptconf = MolConfig_SF(**model_args)
        model = MolGPT_SF(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        # for k, v in list(state_dict.items()):
        #     if k.startswith(unwanted_prefix):
        #         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    return model,gpt_config

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
    if 'rotary' in config:
        rotary = config['rotary']
    else:
        rotary = False
    print('if use rotary embeddings,', rotary)
    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = MolConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            vocab_size=vocab_size, block_size=block_size,
                            dropout=dropout, bias=bias,rotary=rotary)

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias,dropout=dropout,rotary=rotary)  # start with model_args from command line
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
        best_val_loss = checkpoint['best_val_loss']

    return model,model_args

def load_MolGPT_Attn(config: dict,relative_dir:str):
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
    if 'rotary' in config:
        rotary = config['rotary']
    else:
        rotary = False
    print('if use rotary embeddings,', rotary)
    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = MolConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            vocab_size=vocab_size, block_size=block_size,
                            dropout=dropout, bias=bias,rotary=rotary)

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias,dropout=dropout,rotary=rotary)  # start with model_args from command line
    model = MolGPT_Attn(gpt_config)
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
        best_val_loss = checkpoint['best_val_loss']

    return model,model_args

def load_BERT(config: dict):
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


    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = BERTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            block_size=block_size, n_learnable_tokens=n_learnable_tokens,
                            dropout=dropout, bias=bias, rotary=config.get('rotary', False))

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      n_learnable_tokens=n_learnable_tokens,
                      bias=bias, dropout=dropout,rotary=config.get('rotary', False))  # start with model_args from command line
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
    return model,model_args
