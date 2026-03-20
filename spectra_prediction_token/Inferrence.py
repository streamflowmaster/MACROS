import os
import math
from contextlib import nullcontext
from dataset_peaks import dataset
import numpy as np
import torch
from model import GPTConfig, GPT
import torch.utils.data as tud
import pickle
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O


def main(config: dict,num_samples=3,start_length=3,batch_size=1,max_length=10):
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
    device = config['device']
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

    # -----------------------------------------------------------------------------
    # model settings && initialization
    gpt_config = GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                            dropout=dropout, bias=bias)
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias,dropout=dropout)  # start with model_args from command line
    model = GPT(gpt_config)
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        print(f"initializing model with vq-vae config: {gptconf}")
        iter_num = 0


    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias','dropout']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
            else:
                model_args[k] = config[k]
        print(model_args)
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        model.load_state_dict(state_dict, strict=False)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size  # so that the checkpoint will have the right value
    model.to(device)

    # dataset settings

    test_set = dataset(data_path=config['data_path'], type='valid',
                          if_assemble=True, if_functional_group=False, tasks= config['tasks'],
                       device = device)
    test_loader = tud.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    def get_data(loader):
        peaks, _ = next(iter(loader))
        b, l = peaks.shape
        peaks = peaks.to(device).reshape(b, 4, -1)
        category = peaks[:, 0, :]
        centroids = peaks[:, 1, :]
        jvalue = peaks[:, 2, :]
        nH = peaks[:, 3, :]
        # print(category,centroids,jvalue,nH)
        return (category, centroids, jvalue, nH), _

    # -----------------------------------------------------------------------------
    while True:
        data, _ = get_data(test_loader)
        data_input = []
        for i, datum in enumerate(data):
            data_input.append(datum[:,:start_length])
        with torch.no_grad():
            with ctx:
                for k in range(num_samples):
                    peaks = model.generate(data_input, max_new_tokens=max_length-start_length,temperature=1e-5)

                    print('category')
                    print('predicted',peaks[0])
                    print('GT',data[0][:,start_length:])

                    print('centroids')
                    print('predicted',peaks[1])
                    print('GT',data[1][:,start_length:])

                    print('jvalue')
                    print('predicted',peaks[2])
                    print('GT',data[2][:,start_length:])

                    print('nH')
                    print('predicted',peaks[3])
                    print('GT',data[3][:,start_length:])
                    print('------------------------------------')