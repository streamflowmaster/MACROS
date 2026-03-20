"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import math
import random
from contextlib import nullcontext

from click import prompt

from dataset_peaks_acc_NMRMIND import dataset
from dataset_peaks_acc_nmr_assign import dataset as nmr_dataset
# from dataset_toy import dataset
import numpy as np
import torch
from Mol2SpectraToken.model import BERT, BERTConfig
from Mol2Spec_Unified.cascade import Mol2Spectra
import torch.utils.data as tud
from get_data_duringtrain import *
from load_backward_model import *
from scipy.optimize import linear_sum_assignment
# -----------------------------------------------------------------------------

def spectra_similarity_metric(
        spectra_pred,
        spectra_gt,
        task: str,
        batch_size: int = 16,
        bos_token: int = 0,
        eos_token: int = 1
):
    """
    Compute similarity between predicted and ground truth spectra using Hungarian matching.
    Removes all content after the first EOS token in each sequence.

    Args:
        spectra_pred: Predicted spectra tensor
        spectra_gt: Ground truth spectra tensor
        task: Either 'hnmr' or 'cnmr'
        batch_size: Batch size
        bos_token: Beginning of sequence token value
        eos_token: End of sequence token value

    Returns:
        float: Average similarity score across batch
    """
    # Convert inputs to tensors if not already
    if isinstance(spectra_pred, list):
        spectra_pred = torch.stack(spectra_pred).cpu()
    if isinstance(spectra_gt, list):
        spectra_gt = torch.stack(spectra_gt).cpu()

    # Remove everything after first EOS token
    def remove_tokens(spectra, batch_size, bos_token=0, eos_token=1):
        """
        Remove BOS token and everything after the first EOS token for each sequence in a batch.

        Args:
            spectra: Tensor of shape [batch, seq_length] containing sequences with BOS/EOS tokens.
            batch_size: Number of sequences in the batch.
            bos_token: Beginning of sequence token value (default: 0).
            eos_token: End of sequence token value (default: 1).

        Returns:
            Tensor of shape [batch, max_valid_len] containing cleaned sequences, padded to the longest valid length.
        """
        cleaned_spectra = []

        for batch_idx in range(batch_size):
            sequence = spectra[batch_idx, :]  # Shape: [seq_length]

            # Find first EOS token position
            eos_positions = (sequence == eos_token).nonzero(as_tuple=True)[0]
            eos_pos = eos_positions[0].item() if eos_positions.numel() > 0 else sequence.shape[-1]

            # Find last BOS token position (if any)
            bos_positions = (sequence == bos_token).nonzero(as_tuple=True)[0]
            start_pos = bos_positions[-1].item() + 1 if bos_positions.numel() > 0 else 0

            # Truncate sequence: from after BOS (if any) to first EOS (if any)
            valid_sequence = sequence[start_pos:eos_pos] if eos_pos > start_pos else sequence[start_pos:]
            cleaned_spectra.append(valid_sequence)

        return cleaned_spectra

    if 'hnmr' in task:
        spectra_pred = spectra_pred[0,:,:]
        spectra_gt = spectra_gt[0,:,:]

        chemical_shift_pred = remove_tokens(spectra_pred, batch_size)
        chemical_shift_gt = remove_tokens(spectra_gt, batch_size)

    elif 'cnmr' in task:
        # print(spectra_pred.shape,spectra_gt.shape)
        spectra_pred = spectra_pred[0,:,:]
        spectra_gt = spectra_gt[0,:,:]

        chemical_shift_pred = remove_tokens(spectra_pred, batch_size)
        chemical_shift_gt = remove_tokens(spectra_gt, batch_size)

    elif 'hsqc' in task:
        spectra_pred = spectra_pred[0,:,:]
        spectra_gt = spectra_gt[0,:,:]
        chemical_shift_pred = remove_tokens(spectra_pred, batch_size)
        chemical_shift_gt = remove_tokens(spectra_gt, batch_size)

    else:
        raise ValueError("Task must be either 'hnmr' or 'cnmr'")

    # Compute similarity using Hungarian matching
    similarities = []
    for i in range(batch_size):
        pred = chemical_shift_pred[i].cpu().numpy()
        gt = chemical_shift_gt[i].cpu().numpy()

        # Remove padding zeros
        pred = pred[pred != 0]
        gt = gt[gt != 0]

        # Create cost matrix (using absolute differences)
        cost_matrix = np.abs(pred[:, None] - gt[None, :])
        # print(cost_matrix)
        # Apply Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)


        # Calculate similarity (lower cost = higher similarity)
        total_cost = cost_matrix[row_ind, col_ind].sum()
        max_possible_cost = np.abs(pred).sum() + np.abs(gt).sum()

        # Convert cost to similarity (normalize to [0,1])
        if max_possible_cost > 0:
            similarity = 1.0 - (total_cost / max_possible_cost)
        else:
            similarity = 1.0

        similarities.append(similarity)

    return np.mean(similarities)


def main(config: dict, spectra_config: dict, mol_config: dict,
         mol_related_dir:str, spectra_related_dir:str,
        random_smiles=False,
         MAX_NMR_PEAKS=20):
    task = config['task']
    print('The model is training for', task)
    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    eval_interval = config['eval_interval']
    log_interval = config['log_interval']
    eval_iters = config['eval_iters']
    eval_only = config['eval_only']
    always_save_checkpoint = config['always_save_checkpoint']
    init_from = config['init_from']
    print(init_from)
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
                            block_size=block_size,n_learnable_tokens=n_learnable_tokens,
                            dropout=dropout, bias=bias)

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                        n_learnable_tokens=n_learnable_tokens,
                      bias=bias,dropout=dropout)  # start with model_args from command line
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
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias','dropout']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
            else:
                model_args[k] = config[k]
        # create the model
        bertconf = BERTConfig(**model_args)
        model = BERT(bertconf)
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
    print(model.config.block_size)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size  # so that the checkpoint will have the right value
    model.to(device)

    if 'hnmr' in task:
        spectra_gpt,spectra_args = load_HNMRSpectraGPT(spectra_config,spectra_related_dir)
    elif 'cnmr' in task:
        spectra_gpt, spectra_args = load_CNMRSpectraGPT(spectra_config, spectra_related_dir)
    elif 'hsqc' in task:
        spectra_gpt, spectra_args = load_HSQCSpectraGPT(spectra_config, spectra_related_dir)

    spectra_gpt.to(device)

    mol_gpt,_ = load_MolGPT(mol_config,mol_related_dir)
    mol_gpt.to(device)

    mol2spectra = Mol2Spectra(head_gpt=mol_gpt,tail_gpt=spectra_gpt,
                              cascade_bert=model,frozen_model=frozen_model)
    # -----------------------------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    raw_model = mol2spectra.module if dp else mol2spectra
    # -----------------------------------------------------------------------------
    # optimizer settings
    optimizer = mol2spectra.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    # if init_from == 'resume':
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    # -----------------------------------------------------------------------------
    # print trainable parameters number xxx M， unfrozen parameters number xxx M
    num_params = sum(p.numel() for p in raw_model.parameters())
    num_unfrozen_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"number of  parameters: {num_params / 1e6} M",
            f"unfrozen parameters: {num_unfrozen_params / 1e6} M")

    mol2spectra.to(device)
    if dp:
        print("using DataParallel...")
        mol2spectra = torch.nn.DataParallel(mol2spectra, device_ids=gpu_ids)

    # -----------------------------------------------------------------------------
    available_modalities = ['[cnmr]','[hnmr]']
    train_configs = [
        {'cache_dir': '../cache_NMRMIND/', 'mol_prompt_dir': '../cache_NMRMIND/', 'random_smiles': False},
        {'cache_dir': '../cache/', 'mol_prompt_dir': '../cache/', 'random_smiles': True},  # 注意这里原代码强制 True
        {'cache_dir': '../cache_NPMRD/', 'mol_prompt_dir': '../cache_NPMRD/', 'random_smiles': False},
        {'cache_dir': '../cache_NMRBANK/', 'mol_prompt_dir': '../cache_NMRBANK/', 'random_smiles': False},
        {'cache_dir': '../cache_QM9/', 'mol_prompt_dir': '../cache_QM9/', 'random_smiles': False},  # 同上
        {'cache_dir': '../cache_NMREXP/', 'mol_prompt_dir': '../cache_NMREXP/', 'random_smiles': False},
    ]

    valid_configs = [
        {'cache_dir': '../cache_NMRMIND/', 'mol_prompt_dir': '../cache_NMRMIND/', 'random_smiles': False},
        {'cache_dir': '../cache_NMREXP/', 'mol_prompt_dir': '../cache_NMREXP/', 'random_smiles': False},
        {'cache_dir': '../cache/', 'mol_prompt_dir': '../cache/', 'random_smiles': True},  # 原代码这里是 True
        {'cache_dir': '../cache_QM9/', 'mol_prompt_dir': '../cache_QM9/', 'random_smiles': False},
        {'cache_dir': '../cache_NMRBANK/', 'mol_prompt_dir': '../cache_NMRBANK/', 'random_smiles': False},
        {'cache_dir': '../cache_NPMRD/', 'mol_prompt_dir': '../cache_NPMRD/', 'random_smiles': False},
    ]

    # 创建 training sets 和 loaders
    training_sets = []
    train_loaders = []

    for i, cfg in enumerate(train_configs):
        training_set = dataset(
            data_path=config['data_path'],
            type='train',
            if_assemble=True,
            if_smiles=True,
            tasks=task,
            if_random_smiles=cfg['random_smiles'],
            device=device,
            cache_dir=cfg['cache_dir'],
            if_mol_prompt=True,
            # mol_prompt_dir=cfg['mol_prompt_dir'],
        )
        training_sets.append(training_set)

        train_loader = tud.DataLoader(training_set, batch_size=batch_size // (len(train_configs)+1), shuffle=True)
        train_loaders.append(train_loader)

    training_set = nmr_dataset(
                           type='train',
                           tasks=task,
                           device='cpu',
                           set_device=device,
                           cache_dir='../SimPub/cache_with_assign/',
                           chunk_id_range=[0,1],
                           )
    training_sets.append(training_set)
    train_loader = tud.DataLoader(training_set, batch_size=batch_size // (len(train_configs)+1), shuffle=True)
    train_loaders.append(train_loader)
    # 创建 validation sets 和 loaders
    valid_sets = []
    valid_loaders = []

    for i, cfg in enumerate(valid_configs):
        valid_set = dataset(
            data_path=config['data_path'],
            type='valid',
            if_assemble=True,
            if_smiles=True,
            tasks=task,
            if_random_smiles=cfg['random_smiles'],
            device=device,
            cache_dir=cfg['cache_dir'],
            if_mol_prompt=True,
            # mol_prompt_dir=cfg['mol_prompt_dir'],
        )
        valid_sets.append(valid_set)

        valid_loader = tud.DataLoader(valid_set, batch_size=batch_size // (len(train_configs)+1), shuffle=False)
        valid_loaders.append(valid_loader)

    valid_set = nmr_dataset(
                           type='train',
                           tasks=task,
                           device='cpu',
                           set_device=device,
                           cache_dir='../SimPub/cache_with_assign/',
                           chunk_id_range=[2,3]
                           )
    valid_sets.append(valid_set)
    valid_loader = tud.DataLoader(valid_set, batch_size=batch_size // (len(train_configs)+1), shuffle=False)
    valid_loaders.append(valid_loader)
    training_set_0 = training_sets[0]


    # def get_data(loader, mol_block_size=90, mask_missing_ratio=0.5, noise_range=3):
    #     '''
    #     :param loader:
    #     :param mol_block_size:
    #     :param mask_missing_ratio:
    #     :param noise_range:
    #     :return:
    #                 {'[ir]': ir_peaks.to(device),
    #          '[hnmr]': [category, centroids, jvalue, nH],
    #          '[cnmr]': [delta, width, integral, intensity],
    #          '[hsqc]': [C13, H1, max_C13, min_C13, nH_1]}
    #     '''
    #     spectra, smiles = next(iter(loader))
    #     smiles = smiles.to(device)
    #
    #     if 'hnmr' in task:
    #         hnmr_peaks = spectra
    #         processed_hnmr, _ = get_hnmr_data(hnmr_peaks.to(device), test_set=training_set,
    #                                           mask_missing_ratio=mask_missing_ratio,
    #                                           noise_range=noise_range, device=device, batch_size=batch_size)
    #         spectra_dict = processed_hnmr
    #
    #     elif 'cnmr' in task:
    #         cnmr_peaks = spectra
    #         processed_cnmr, _ = get_cnmr_data(cnmr_peaks.to(device), test_set=training_set,
    #                                           mask_missing_ratio=mask_missing_ratio,
    #                                           noise_range=noise_range, device=device, batch_size=batch_size)
    #         spectra_dict = processed_cnmr
    #
    #     elif 'hsqc' in task:
    #         hsqc_peaks = spectra
    #         processed_hsqc, _ = get_hsqc_data(hsqc_peaks.to(device), test_set=training_set,
    #                                           mask_missing_ratio=mask_missing_ratio,
    #                                           noise_range=noise_range, device=device, batch_size=batch_size)
    #         spectra_dict = processed_hsqc
    #
    #     elif 'ir' in task:
    #         ir_peaks = spectra
    #         spectra_dict = ir_peaks.to(device)
    #
    #
    #     # print(smiles,spectra_dict)
    #     return smiles[:,:mol_block_size].to(device), spectra_dict


    # Data extraction function
    def get_data(loaders):

        batched_spectra = {
            '[hnmr]':[[] for i in range(4)],
            '[cnmr]':[[] for i in range(2)],
        }
        batched_tokens = []

        for loader in loaders:
            peaks, smiles = next(iter(loader))
            if torch.isnan(peaks).any():
                print(torch.isnan(peaks).any())
            if 'cnmr' in task:
                # print(peaks.shape)
                spectra_dict,_ =  get_cnmr_data(peaks,test_set=training_set_0,
                                     mask_missing_ratio=1,noise_range=5,device=device,batch_size=peaks.shape[0], )
                key = '[cnmr]'
            if 'hnmr' in task:
                spectra_dict,_ = get_hnmr_data(peaks,test_set=training_set_0,
                                     mask_missing_ratio=1,noise_range=3,device=device,batch_size=peaks.shape[0])
                key = '[hnmr]'


            batched_tokens.append(smiles.to(device).long()[:,:84])

            for attr_id in range(len(spectra_dict)):
                if None not in spectra_dict:
                    batched_spectra[key][attr_id].extend(spectra_dict[attr_id])


        for attr_id in range(len(batched_spectra[key])):
            batched_spectra[key][attr_id] = torch.stack(batched_spectra[key][attr_id], dim=0)
        # print(batched_spectra[key][attr_id].shape)
        return torch.cat(batched_tokens, dim=0), batched_spectra[key]

    def estimate_loss():
        out = {}
        mol2spectra.eval()
        loader = {'train': train_loaders, 'val': valid_loaders, 'test': valid_loaders}
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                batch_token, spectra = get_data(loader[split])
                if dp:
                    with ctx:
                        spectra, loss = mol2spectra(batch_token, spectra)
                        loss = loss.mean()  # average the loss over the micro-batch
                else:
                    with ctx:
                        spectra, loss = mol2spectra(batch_token, spectra)
                losses[k] = loss.item()
            out[split] = losses.mean()
        mol2spectra.train()
        return out

    def estimate_similarity():
        MAX_NMR_PEAKS = 20 if 'hnmr' in task else 60
        out = {}
        mol2spectra.eval()
        loader = {'train': train_loaders, 'val': valid_loaders, 'test': valid_loaders}
        for split in ['train', 'val']:
            similarities = torch.zeros(eval_iters)
            for k in range(eval_iters):
                batch_token, spectra = get_data(loader[split])
                with torch.no_grad():
                    with ctx:
                        spectra_pred = mol2spectra.generate(batch_token,
                                    max_new_tokens=MAX_NMR_PEAKS)
                similarities[k] = spectra_similarity_metric(
                    spectra_pred,
                    spectra,
                    task,
                    batch_size=batch_token.shape[0]
                )
            out[split] = similarities.mean()
        mol2spectra.train()
        return out

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    dataset_counting = 0


    while True:
        iter_num += 1
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if iter_num % eval_interval == 0:
            with torch.no_grad():
                losses = estimate_loss()
                similarities = estimate_similarity()
            print(f"iter {iter_num} train loss: {losses['train']} val loss: {losses['val']} "
                  f"train similarity: {similarities['train']} val similarity: {similarities['val']} lr: {lr}\n")
            if wandb_log:
                import wandb
                wandb.log({
                    'train_loss': losses['train'],
                    'val_loss': losses['val'],
                    'train_similarity': similarities['train'],
                    'val_similarity': similarities['val'],
                    'lr': lr
                })
            with open(os.path.join(out_dir, 'losses_log.txt'), 'a') as f:
                f.write(f"iter {iter_num} train loss: {losses['train']} val loss: {losses['val']} "
                        f"train similarity: {similarities['train']} val similarity: {similarities['val']} lr: {lr}\n")
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.cascade_bert.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, ckpt_path)
                    if 'tail_gpt' not in frozen_model:

                        checkpoint_of_tail = {
                            'model': raw_model.tail_gpt.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': spectra_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        spectra_out_dir = spectra_config['out_dir']
                        tail_ckpt_path = os.path.join(spectra_related_dir,spectra_out_dir, 'ckpt.pt')
                        print(f"saving spectra checkpoint to {tail_ckpt_path}")
                        torch.save(checkpoint_of_tail, tail_ckpt_path)

                    if 'head_gpt' not in frozen_model:
                        checkpoint_of_head = {
                            'model': raw_model.head_gpt.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': mol_config,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        mol_out_dir = mol_config['out_dir']
                        head_ckpt_path = os.path.join(mol_related_dir,mol_out_dir,'ckpt.pt')
                        print(f"saving mol checkpoint to {head_ckpt_path}")
                        torch.save(checkpoint_of_head, head_ckpt_path)

        if iter_num == 0 and eval_only:
            break

        # print(estimate_similarity())
        for micro_step in range(gradient_accumulation_steps):
            mol2spectra.train()
            mol2spectra.freeze_params()
            random_sq_length = int(random.randint(70,90))
            if dataset_counting>=training_sets[-1].__len__():
                training_sets[-1].refresh_cache_chunks()
                dataset_counting = 0
            batch_token, spectra = get_data(train_loaders)
            dataset_counting += batch_token.shape[0]//len(train_loaders)
            if dp:
                with ctx:
                    logits, loss = mol2spectra(batch_token, spectra)
                    loss = loss.mean()  # average the loss over the micro-batch
                    loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            else:
                with ctx:
                    logits, loss = mol2spectra(batch_token, spectra)
                    loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(mol2spectra.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

