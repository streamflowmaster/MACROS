import os
import torch
from Mol2Spec_Unified.model import BERT, BERTConfig
from molecule_pretrain.model import GPT as MolGPT, GPTConfig as MolConfig
from spectra_prediction_token.model import GPT as HNMRSpectraGPT, GPTConfig as HNMRSpectraConfig
from spectra_prediction_token.model_cnmr import GPT as CNMRSpectraGPT, GPTConfig as CNMRSpectraConfig
from spectra_prediction_token.model_hsqc import GPT as HSQCSpectraGPT, GPTConfig as HSQCSpectraConfig

# 加载模型的函数（复用训练脚本中的函数）
def load_HNMRSpectraGPT(config: dict, relative_dir: str):
    # 与训练脚本中的 load_HNMRSpectraGPT 一致
    if relative_dir == '':
        out_dir = config['out_dir']
    else:
        out_dir = os.path.join(relative_dir, config['out_dir'])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    device = config['device']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    dropout = config['dropout']
    bias = config['bias']
    block_size = config['block_size']
    prompt_pe = config.get('prompt_pe', True)
    start_generate_token = config.get('start_generate_token', False)

    gpt_config = HNMRSpectraConfig(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=dropout, bias=bias, prompt_pe=prompt_pe,
        block_size=block_size, start_generate_token=start_generate_token
    )
    model = HNMRSpectraGPT(gpt_config)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model_args = checkpoint['model_args']
    return model, model_args


def load_CNMRSpectraGPT(config: dict, relative_dir: str):
    # 与训练脚本中的 load_CNMRSpectraGPT 一致
    if relative_dir == '':
        out_dir = config['out_dir']
    else:
        out_dir = os.path.join(relative_dir, config['out_dir'])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    device = config['device']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    dropout = config['dropout']
    bias = config['bias']
    block_size = config['block_size']
    prompt_pe = config.get('prompt_pe',False)
    start_generate_token = config.get('start_generate_token', True)
    use_intensity = config.get('use_intensity', False)
    print('if use_intensity', use_intensity)
    gpt_config = CNMRSpectraConfig(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=dropout, bias=bias, prompt_pe=prompt_pe,
        block_size=block_size, start_generate_token=start_generate_token,
        use_intensity=use_intensity
    )
    model = CNMRSpectraGPT(gpt_config)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model_args = checkpoint['model_args']
    return model, model_args


def load_HSQCSpectraGPT(config: dict, relative_dir: str):
    # 与训练脚本中的 load_HSQCSpectraGPT 一致
    if relative_dir == '':
        out_dir = config['out_dir']
    else:
        out_dir = os.path.join(relative_dir, config['out_dir'])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    device = config['device']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    dropout = config['dropout']
    bias = config['bias']
    block_size = config['block_size']
    prompt_pe = config.get('prompt_pe', True)
    start_generate_token = config.get('start_generate_token', False)

    gpt_config = HSQCSpectraConfig(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=dropout, bias=bias, prompt_pe=prompt_pe,
        block_size=block_size, start_generate_token=start_generate_token
    )
    model = HSQCSpectraGPT(gpt_config)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model_args = checkpoint['model_args']
    return model, model_args


def load_MolGPT(config: dict, relative_dir: str):
    # 与训练脚本中的 load_MolGPT 一致
    if relative_dir == '':
        out_dir = config['out_dir']
    else:
        out_dir = os.path.join(relative_dir, config['out_dir'])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    device = config['device']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    dropout = config['dropout']
    bias = config['bias']
    block_size = config['block_size']
    vocab_size = config['vocab_size']

    gpt_config = MolConfig(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        vocab_size=vocab_size, block_size=block_size,
        dropout=dropout, bias=bias
    )
    model = MolGPT(gpt_config)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model_args = checkpoint['model_args']
    return model, model_args


def load_BERT(config: dict):
    # 与训练脚本中的 load_BERT 一致
    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    device = config['device']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    block_size = config['block_size']
    n_learnable_tokens = config['n_learnable_tokens']
    dropout = config['dropout']
    bias = config['bias']

    gpt_config = BERTConfig(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        block_size=block_size, n_learnable_tokens=n_learnable_tokens,
        dropout=dropout, bias=bias
    )
    model = BERT(gpt_config)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model_args = checkpoint['model_args']
    return model, model_args



