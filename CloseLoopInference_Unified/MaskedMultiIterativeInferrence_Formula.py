
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from utils.load_forward_utils import load_forward, MultiSpec2Mol
from utils.load_refine_utils import load_forward as load_refine, MultiSpec2Mol as MultiSpec2Mol_refine
from utils.load_backward_utils import load_backward, Mol2Spectra
from utils.load_IR_utils import Mol2Spectra as Mol2IR, load_backward as load_ir
from utils.nmr_similarity import reward_cnmr_similarity, reward_hnmr_similarity, reward_ir_similarity
from MaskNMRTokenizer import NMRSpectrumTokenizer, ir_encode
import os.path
from scipy.optimize import linear_sum_assignment
import yaml

def read_yaml(yaml_path: str):
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def process_mol_prompts(mol_prompts, max_mol_prompt_length, eos_token, pad_token, bos_token):
    '''
    Process molecular prompts by adding BOS token at the beginning, replacing tokens after the first EOS token with PAD tokens,
    and padding/truncating to max_mol_prompt_length.

    :param mol_prompts: torch.tensor [b, l] - Input tensor of shape (batch_size, sequence_length)
    :param max_mol_prompt_length: int - Desired length of each sequence
    :param eos_token: int - End of sequence token ID
    :param pad_token: int - Padding token ID
    :param bos_token: int - Beginning of sequence token ID
    :return: torch.tensor [b, max_mol_prompt_length] - Processed tensor
    '''
    b = mol_prompts.shape[0]
    device = mol_prompts.device
    processed_prompts = torch.full((b, max_mol_prompt_length), pad_token, device=device, dtype=mol_prompts.dtype)

    for i in range(b):
        # Get current sequence
        seq = mol_prompts[i]
        # Find first EOS token position
        eos_pos = (seq == eos_token).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            # If EOS token exists, keep tokens up to and including the first EOS
            valid_length = min(eos_pos[0] + 1, max_mol_prompt_length - 1)  # -1 to account for BOS
        else:
            # If no EOS token, keep all tokens up to max length (minus BOS)
            valid_length = min(seq.shape[0], max_mol_prompt_length - 1)

        # Add BOS token at the beginning
        processed_prompts[i, 0] = bos_token
        # Copy valid tokens (excluding tokens after first EOS)
        processed_prompts[i, 1:valid_length + 1] = seq[:valid_length]
        # Remaining positions are already filled with pad_token

    return processed_prompts


def nmr_token_similarity_metric(
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
    """
    if isinstance(spectra_pred, list):
        spectra_pred = torch.stack(spectra_pred).cpu()
    if isinstance(spectra_gt, list):
        spectra_gt = torch.stack(spectra_gt).cpu()

    def remove_tokens(spectra, batch_size, bos_token=0, eos_token=1):
        cleaned_spectra = []
        for batch_idx in range(batch_size):
            sequence = spectra[batch_idx, :]
            eos_positions = (sequence == eos_token).nonzero(as_tuple=True)[0]
            eos_pos = eos_positions[0].item() if eos_positions.numel() > 0 else sequence.shape[-1]
            bos_positions = (sequence == bos_token).nonzero(as_tuple=True)[0]
            start_pos = bos_positions[-1].item() + 1 if bos_positions.numel() > 0 else 0
            valid_sequence = sequence[start_pos:eos_pos] if eos_pos > start_pos else sequence[start_pos:]
            cleaned_spectra.append(valid_sequence)
        return cleaned_spectra

    if 'hnmr' in task:
        spectra_pred = spectra_pred[0,:,:]
        spectra_gt = spectra_gt[0,:,:]
        chemical_shift_pred = remove_tokens(spectra_pred, batch_size)
        chemical_shift_gt = remove_tokens(spectra_gt, batch_size)
    elif 'cnmr' in task:
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

    similarities = []
    for i in range(batch_size):
        pred = chemical_shift_pred[i].cpu().numpy()
        gt = chemical_shift_gt[i].cpu().numpy()
        pred = pred[pred != 0]
        gt = gt[gt != 0]
        cost_matrix = np.abs(pred[:, None] - gt[None, :])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_ind, col_ind].sum()
        max_possible_cost = np.abs(pred).sum() + np.abs(gt).sum()
        similarity = 1.0 - (total_cost / max_possible_cost) if max_possible_cost > 0 else 1.0
        similarities.append(similarity)
    return similarities

def nmr_chemical_shift_loss(pred_nmr, gt_nmr, mode='cnmr', num_diff_weight=1):
    def cal_shift_error(pred, gt):
        try:
            pred = np.array(pred)
            gt = np.array(gt)
            cost_matrix = np.abs(pred[:, None] - gt[None, :])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_cost = cost_matrix[row_ind, col_ind].sum()
            max_possible_cost = np.abs(pred).sum() + np.abs(gt).sum()
            similarity = 1.0 - (total_cost / max_possible_cost) if max_possible_cost > 0 else 1.0
            # 加入谱峰数量差异惩罚
            num_diff = abs(len(pred) - len(gt))
            penalty = num_diff * num_diff_weight / max(len(pred), len(gt))
            return similarity*(1-penalty)  # 确保不返回负值
        except:
            return 0.0

    shift_errors = []
    if mode == 'cnmr':
        for pred, gt in zip(pred_nmr, gt_nmr):
            pred_shift = [peak['delta (ppm)'] for peak in pred]
            gt_shift = [peak['delta (ppm)'] for peak in gt]
            shift_error = cal_shift_error(pred_shift, gt_shift)
            shift_errors.append(shift_error)
    elif mode == 'hnmr':
        for pred, gt in zip(pred_nmr, gt_nmr):
            pred_shift = [peak['centroid'] for peak in pred]
            gt_shift = [peak['centroid'] for peak in gt]
            # pred_nH = [peak['nH'] for peak in pred]
            # gt_nH = [peak['nH'] for peak in gt]

            shift_error = cal_shift_error(pred_shift, gt_shift)
            shift_errors.append(shift_error)
    elif mode == 'hsqc':
        c_shift_errors = []
        h_shift_errors = []
        for pred, gt in zip(pred_nmr, gt_nmr):
            pred_c_shift = [peak['13C_centroid'] for peak in pred]
            gt_c_shift = [peak['13C_centroid'] for peak in gt]
            pred_h_shift = [peak['1H_centroid'] for peak in pred]
            gt_h_shift = [peak['1H_centroid'] for peak in gt]
            c_shift_error = cal_shift_error(pred_c_shift, gt_c_shift)
            h_shift_error = cal_shift_error(pred_h_shift, gt_h_shift)
            c_shift_errors.append(c_shift_error)
            h_shift_errors.append(h_shift_error)
        return c_shift_errors, h_shift_errors
    return shift_errors

class MoleculeInferencePipeline:
    """Molecular inference pipeline with draft, selection, and refinement steps."""
    def __init__(
            self,
            spec2mol_dir: str,
            spec2mol_dir_refine: Optional[str],  # 支持 None
            spec2mol_agent_path: str,
            spec2mol_agent_path_refine: Optional[str],  # 支持 None
            spec2mol_config_path: str,
            spec2mol_config_path_refine: Optional[str],  # 支持 None
            hnmr_output_dir: str,
            hnmr_agent_path: str,
            hnmr_config_path: str,
            cnmr_output_dir: str,
            cnmr_agent_path: str,
            cnmr_config_path: str,
            hsqc_output_dir: str,
            hsqc_agent_path: str,
            hsqc_config_path: str,
            ir_output_dir: str,
            ir_config_path: str,
            device: str = 'cuda:0',
            encoding_order =
                ['[ir]', '[hnmr]','[cnmr]','[hsqc]']
    ):
        self.device = device
        self.cnmr_tokenizer = NMRSpectrumTokenizer(NMR_category='cnmr')
        self.hnmr_tokenizer = NMRSpectrumTokenizer(NMR_category='hnmr')
        self.hsqc_tokenizer = NMRSpectrumTokenizer(NMR_category='hsqc')
        self.encoding_order = encoding_order
        # 加载草稿模型

        self.mol2hnmr = None
        if hnmr_output_dir is not None:
            self.mol2hnmr = load_backward(
                output_dir=hnmr_output_dir,
                agent_path=hnmr_agent_path,
                config_path=hnmr_config_path,
                task='hnmr',
                device=device
            ).to(device)

        self.mol2cnmr = None
        if cnmr_output_dir is not None:
            self.mol2cnmr = load_backward(
                output_dir=cnmr_output_dir,
                agent_path=cnmr_agent_path,
                config_path=cnmr_config_path,
                task='cnmr',
                device=device
            ).to(device)

        self.mol2hsqc = None
        if hsqc_output_dir is not None:
            self.mol2hsqc = load_backward(
                output_dir=hsqc_output_dir,
                agent_path=hsqc_agent_path,
                config_path=hsqc_config_path,
                task='hsqc',
                device=device
            ).to(device)

        self.mol2ir = None
        if ir_output_dir is not None:
            self.mol2ir = load_ir(
                output_dir=ir_output_dir,
                agent_path='all_unfroz_NPPE',
                config_path=ir_config_path,
                task='ir',
                device=device
            ).to(device)

        self.spec2mol = load_refine(
            output_dir=spec2mol_dir,
            agent_path=spec2mol_agent_path,
            connection_path=os.path.join(spec2mol_dir, f'connect_{spec2mol_agent_path}'),
            config_path=spec2mol_config_path,
            device=device
        ).to(device)

        # 检查是否启用精炼模型
        self.enable_refinement = (
                spec2mol_dir_refine is not None and
                spec2mol_agent_path_refine is not None and
                spec2mol_config_path_refine is not None
        )
        self.spec2mol_refine = None
        if self.enable_refinement:
            self.spec2mol_refine = load_refine(
                output_dir=spec2mol_dir_refine,
                agent_path=spec2mol_agent_path_refine,
                connection_path=os.path.join(spec2mol_dir_refine, f'connect_{spec2mol_agent_path_refine}'),
                config_path=spec2mol_config_path_refine,
                device=device
            ).to(device)
            print("BLOCKSIZE", self.spec2mol_refine.tail_gpt.config.block_size)


        self.pipeline_results = []

    def encode_spectra(self, spectra_dict: List[Dict[str, List[Dict]]]) -> Dict[str, torch.Tensor]:
        hnmr_gt = self.hnmr_tokenizer.hnmr_encode(spectra_dict, device=self.device)
        cnmr_gt = self.cnmr_tokenizer.cnmr_encode(spectra_dict, device=self.device)
        hsqc_gt = self.hsqc_tokenizer.hsqc_encode(spectra_dict, device=self.device)
        ir_gt = ir_encode(spectra_dict, device=self.device)
        spectra =  {
            '[ir]': ir_gt,
            '[cnmr]': [cnmr for cnmr in cnmr_gt],
            '[hnmr]': [hnmr for hnmr in hnmr_gt],
            '[hsqc]': [hsqc for hsqc in hsqc_gt]
        }
        spectra_dict = {}
        for key in self.encoding_order:
            spectra_dict[key] = spectra[key]
        return spectra_dict

    def backward_infer(
            self,
            smile_mols_id: torch.Tensor,
            spectra_dict: List[Dict[str, Union[List[Dict], str]]],
            tokenized_spectra_dict: Dict[str, List[Union[torch.Tensor, str]]],
            cnmr_max_new_token: int = 66,
            hnmr_max_new_token: int = 22,
            hsqc_max_new_token: int = 66,
            ir_patch_size: int = 16,
            missing_identification: str = '<missing>',
            tokens: Dict[str, int] = {'[ir]': 32},
            if_monte_carlo: bool = True,
            ir_length=512
    ) -> Tuple[Dict[str, torch.Tensor], List[List[Dict]], List[List[Dict]], List[List[Dict]], torch.Tensor]:
        batch_size = smile_mols_id.shape[0]
        # print(smile_mols_id)
        smile_mols_id = process_mol_prompts(mol_prompts=smile_mols_id,
                                            max_mol_prompt_length=84,
                                            eos_token=1,
                                            bos_token=0,
                                            pad_token=2)
        scores = {
            'cnmr': torch.zeros(batch_size, device=self.device),
            'hnmr': torch.zeros(batch_size, device=self.device),
            'hsqc_c': torch.zeros(batch_size, device=self.device),  # Split HSQC scores
            'hsqc_h': torch.zeros(batch_size, device=self.device),
            'ir': torch.zeros(batch_size, device=self.device)
        }
        cnmr_pred_peaks = [[] for _ in range(batch_size)]
        hnmr_pred_peaks = [[] for _ in range(batch_size)]
        hsqc_pred_peaks = [[] for _ in range(batch_size)]
        ir_pred_specs = torch.zeros(batch_size, ir_length or 0, device=self.device)

        cnmr_valid_indices = [i for i in range(batch_size) if spectra_dict[i]['c_nmr_peaks'] != missing_identification]
        hnmr_valid_indices = [i for i in range(batch_size) if spectra_dict[i]['h_nmr_peaks'] != missing_identification]
        hsqc_valid_indices = [i for i in range(batch_size) if spectra_dict[i]['hsqc_nmr_peaks'] != missing_identification]
        ir_valid_indices = [i for i in range(batch_size) if tokenized_spectra_dict['[ir]'][i] != missing_identification]

        if cnmr_valid_indices and self.mol2cnmr:
            cnmr_valid_ids = smile_mols_id[cnmr_valid_indices]
            cnmr_spec = self.mol2cnmr.generate(prompt_ids=cnmr_valid_ids, max_new_tokens=cnmr_max_new_token)
            if len(cnmr_spec) == 1:
                cnmr_spec.append(torch.ones_like(cnmr_spec[0]))
            cnmr_pred = self.cnmr_tokenizer.decode(cnmr_spec)

            for idx, pred in zip(cnmr_valid_indices, cnmr_pred):
                cnmr_pred_peaks[idx].append(pred)

        if hnmr_valid_indices and self.mol2hnmr:
            hnmr_valid_ids = smile_mols_id[hnmr_valid_indices]
            hnmr_spec = self.mol2hnmr.generate(prompt_ids=hnmr_valid_ids, max_new_tokens=hnmr_max_new_token)
            # print(len(hnmr_spec))
            hnmr_pred = self.hnmr_tokenizer.decode(hnmr_spec)
            for idx, pred in zip(hnmr_valid_indices, hnmr_pred):
                hnmr_pred_peaks[idx].append(pred)

        if hsqc_valid_indices and self.mol2hsqc:
            hsqc_valid_ids = smile_mols_id[hsqc_valid_indices]
            hsqc_spec = self.mol2hsqc.generate(prompt_ids=hsqc_valid_ids, max_new_tokens=hsqc_max_new_token)
            hsqc_pred = self.hsqc_tokenizer.decode(hsqc_spec)
            for idx, pred in zip(hsqc_valid_indices, hsqc_pred):
                hsqc_pred_peaks[idx].append(pred)

        if ir_valid_indices and self.mol2ir:
            ir_valid_ids = smile_mols_id[ir_valid_indices]
            ir_pred_spec = self.mol2ir.generate(
                prompt_ids=ir_valid_ids,
                max_new_tokens=ir_length // ir_patch_size
            ).reshape(-1, ir_length)
            for idx, pred in zip(ir_valid_indices, ir_pred_spec):
                ir_pred_specs[idx] = pred

        if cnmr_valid_indices and self.mol2cnmr:
            cnmr_peaks = [spectra_dict[i]['c_nmr_peaks'] for i in cnmr_valid_indices]
            pred_cnmr_peaks = [cnmr_pred_peaks[i][0] for i in cnmr_valid_indices]
            cnmr_score = nmr_chemical_shift_loss(pred_nmr=pred_cnmr_peaks, gt_nmr=cnmr_peaks, mode='cnmr')
            for idx, score in zip(cnmr_valid_indices, cnmr_score):
                scores['cnmr'][idx] = torch.tensor(score, device=self.device)

        if hnmr_valid_indices and self.mol2hnmr:
            hnmr_peaks = [spectra_dict[i]['h_nmr_peaks'] for i in hnmr_valid_indices]
            pred_hnmr_peaks = [hnmr_pred_peaks[i][0] for i in hnmr_valid_indices]
            hnmr_score = nmr_chemical_shift_loss(pred_nmr=pred_hnmr_peaks, gt_nmr=hnmr_peaks, mode='hnmr')
            for idx, score in zip(hnmr_valid_indices, hnmr_score):
                scores['hnmr'][idx] = torch.tensor(score, device=self.device)

        if hsqc_valid_indices and self.mol2hsqc:
            hsqc_peaks = [spectra_dict[i]['hsqc_nmr_peaks'] for i in hsqc_valid_indices]
            pred_hsqc_peaks = [hsqc_pred_peaks[i][0] for i in hsqc_valid_indices]
            hsqc_c_score, hsqc_h_score = nmr_chemical_shift_loss(pred_nmr=pred_hsqc_peaks, gt_nmr=hsqc_peaks, mode='hsqc')
            for idx, c_score, h_score in zip(hsqc_valid_indices, hsqc_c_score, hsqc_h_score):
                scores['hsqc_c'][idx] = torch.tensor(c_score, device=self.device)
                scores['hsqc_h'][idx] = torch.tensor(h_score, device=self.device)

        if ir_valid_indices and self.mol2ir:
            gt_ir_specs = torch.stack([tokenized_spectra_dict['[ir]'][i] for i in ir_valid_indices]).to(self.device)
            pred_ir_specs = ir_pred_specs[ir_valid_indices]
            ir_score = reward_ir_similarity(batch_pred_ir=pred_ir_specs.cpu(), batch_gt_ir=gt_ir_specs.cpu())
            for idx, score in zip(ir_valid_indices, ir_score):
                scores['ir'][idx] = torch.tensor(score, device=self.device)

        return scores, cnmr_pred_peaks, hnmr_pred_peaks, hsqc_pred_peaks, ir_pred_specs

    def draft_infer(
            self,
            spectra_dict: List[Dict[str, List[Dict]]],
            search: str = 'hybrid',
            mol_max_new_token: int = 90,
            cnmr_max_new_token: int = 60,
            hnmr_max_new_token: int = 12,
            hsqc_max_new_token: int = 12,
            sample_width: int = 64,
            eos_token_id: int = 1,
            ir_patch_size: int = 16,
            ir_length=512
    ) -> None:
        tokenized_spectra_dict = self.encode_spectra(spectra_dict)
        batch_size = len(tokenized_spectra_dict['[ir]'])

        summary_scores = {
            'cnmr': torch.zeros(batch_size, sample_width, device=self.device),
            'hnmr': torch.zeros(batch_size, sample_width, device=self.device),
            'hsqc_c': torch.zeros(batch_size, sample_width, device=self.device),  # Split HSQC scores
            'hsqc_h': torch.zeros(batch_size, sample_width, device=self.device),
            'ir': torch.zeros(batch_size, sample_width, device=self.device),
            'mol_prob': torch.zeros(batch_size, sample_width, device=self.device)
        }
        summary_smiles = torch.zeros(batch_size, sample_width, mol_max_new_token, dtype=torch.long, device=self.device)
        summary_smiles_logits = torch.zeros(batch_size, sample_width, mol_max_new_token, dtype=torch.float, device=self.device)
        cnmr_pred_peaks = [[] for _ in range(batch_size)]
        hnmr_pred_peaks = [[] for _ in range(batch_size)]
        hsqc_pred_peaks = [[] for _ in range(batch_size)]
        ir_pred_specs = torch.zeros(batch_size, sample_width, ir_length, device=self.device)

        for i in range(sample_width):
            smile_mols_id, smile_logits = self.spec2mol.generate_with_missing_modality_with_prob(
                tokenized_spectra_dict, max_new_tokens=mol_max_new_token, search=search, if_monte_carlo=True
            )
            summary_smiles_logits[:, i] = smile_logits
            batch_size, seq_len = smile_logits.shape
            smile_logits_sum = torch.zeros(batch_size, device=self.device)

            for b in range(batch_size):
                eos_positions = (smile_mols_id[b] == eos_token_id).nonzero(as_tuple=False)
                eos_pos = eos_positions[0, 0].item() + 1 if len(eos_positions) > 0 else seq_len
                smile_logits_sum[b] = smile_logits[b, 1:eos_pos].detach().log().mean()

            scores, cnmr_peaks, hnmr_peaks, hsqc_peaks, ir_specs = self.backward_infer(
                smile_mols_id=smile_mols_id,
                spectra_dict=spectra_dict,
                tokenized_spectra_dict=tokenized_spectra_dict,
                cnmr_max_new_token=cnmr_max_new_token,
                hnmr_max_new_token=hnmr_max_new_token,
                hsqc_max_new_token=hsqc_max_new_token,
                ir_patch_size=ir_patch_size
            )

            summary_scores['cnmr'][:, i] = scores['cnmr']
            summary_scores['hnmr'][:, i] = scores['hnmr']
            summary_scores['hsqc_c'][:, i] = scores['hsqc_c']  # Split HSQC scores
            summary_scores['hsqc_h'][:, i] = scores['hsqc_h']
            summary_scores['ir'][:, i] = scores['ir']
            summary_scores['mol_prob'][:, i] = smile_logits_sum
            summary_smiles[:, i] = smile_mols_id
            ir_pred_specs[:, i, :] = ir_specs

            for b in range(batch_size):
                cnmr_pred_peaks[b].extend(cnmr_peaks[b])
                hnmr_pred_peaks[b].extend(hnmr_peaks[b])
                hsqc_pred_peaks[b].extend(hsqc_peaks[b])

        self.pipeline_results.append({
            'stage': 'draft',
            'summary_scores': summary_scores,
            'summary_smiles': summary_smiles,
            'summary_smiles_logits': summary_smiles_logits,
            'cnmr_pred_peaks': cnmr_pred_peaks,
            'hnmr_pred_peaks': hnmr_pred_peaks,
            'hsqc_pred_peaks': hsqc_pred_peaks,
            'ir_pred_specs': ir_pred_specs
        })

    def draft_infer_from_Formula(
            self,
            formula_prompt:List[str],
            spectra_dict: List[Dict[str, List[Dict]]],
            search: str = 'hybrid',
            mol_max_new_token: int = 90,
            cnmr_max_new_token: int = 60,
            hnmr_max_new_token: int = 12,
            hsqc_max_new_token: int = 12,
            sample_width: int = 64,
            eos_token_id: int = 1,
            ir_patch_size: int = 16,
            ir_length=512
    ) -> None:
        tokenized_spectra_dict = self.encode_spectra(spectra_dict)
        batch_size = len(tokenized_spectra_dict['[ir]'])
        formula_prompt = torch.full((batch_size,8),fill_value=2,device=self.device)
        summary_scores = {
            'cnmr': torch.zeros(batch_size, sample_width, device=self.device),
            'hnmr': torch.zeros(batch_size, sample_width, device=self.device),
            'hsqc_c': torch.zeros(batch_size, sample_width, device=self.device),  # Split HSQC scores
            'hsqc_h': torch.zeros(batch_size, sample_width, device=self.device),
            'ir': torch.zeros(batch_size, sample_width, device=self.device),
            'mol_prob': torch.zeros(batch_size, sample_width, device=self.device)
        }
        summary_smiles = torch.zeros(batch_size, sample_width, mol_max_new_token, dtype=torch.long, device=self.device)
        summary_smiles_logits = torch.zeros(batch_size, sample_width, mol_max_new_token, dtype=torch.float, device=self.device)
        cnmr_pred_peaks = [[] for _ in range(batch_size)]
        hnmr_pred_peaks = [[] for _ in range(batch_size)]
        hsqc_pred_peaks = [[] for _ in range(batch_size)]
        ir_pred_specs = torch.zeros(batch_size, sample_width, ir_length, device=self.device)

        for i in range(sample_width):
            smile_mols_id, smile_logits = self.spec2mol.refine_with_missing_modality_with_prob(
                    spectra_dict=tokenized_spectra_dict,
                    mol_prompt=formula_prompt,
                    max_new_tokens=mol_max_new_token,
                    search=search,
                    if_monte_carlo=True
            )
            summary_smiles_logits[:, i] = smile_logits
            batch_size, seq_len = smile_logits.shape
            smile_logits_sum = torch.zeros(batch_size, device=self.device)

            for b in range(batch_size):
                eos_positions = (smile_mols_id[b] == eos_token_id).nonzero(as_tuple=False)
                eos_pos = eos_positions[0, 0].item() + 1 if len(eos_positions) > 0 else seq_len
                smile_logits_sum[b] = smile_logits[b, 1:eos_pos].detach().log().mean()

            scores, cnmr_peaks, hnmr_peaks, hsqc_peaks, ir_specs = self.backward_infer(
                smile_mols_id=smile_mols_id,
                spectra_dict=spectra_dict,
                tokenized_spectra_dict=tokenized_spectra_dict,
                cnmr_max_new_token=cnmr_max_new_token,
                hnmr_max_new_token=hnmr_max_new_token,
                hsqc_max_new_token=hsqc_max_new_token,
                ir_patch_size=ir_patch_size
            )

            summary_scores['cnmr'][:, i] = scores['cnmr']
            summary_scores['hnmr'][:, i] = scores['hnmr']
            summary_scores['hsqc_c'][:, i] = scores['hsqc_c']  # Split HSQC scores
            summary_scores['hsqc_h'][:, i] = scores['hsqc_h']
            summary_scores['ir'][:, i] = scores['ir']
            summary_scores['mol_prob'][:, i] = smile_logits_sum
            summary_smiles[:, i] = smile_mols_id
            ir_pred_specs[:, i, :] = ir_specs

            for b in range(batch_size):
                cnmr_pred_peaks[b].extend(cnmr_peaks[b])
                hnmr_pred_peaks[b].extend(hnmr_peaks[b])
                hsqc_pred_peaks[b].extend(hsqc_peaks[b])

        self.pipeline_results.append({
            'stage': 'draft',
            'summary_scores': summary_scores,
            'summary_smiles': summary_smiles,
            'summary_smiles_logits': summary_smiles_logits,
            'cnmr_pred_peaks': cnmr_pred_peaks,
            'hnmr_pred_peaks': hnmr_pred_peaks,
            'hsqc_pred_peaks': hsqc_pred_peaks,
            'ir_pred_specs': ir_pred_specs
        })

    def draft_infer_from_reaction(
            self,
            spectra_dict: List[Dict[str, List[Dict]]],
            tokenized_reaction_prompt,
            search: str = 'hybrid',
            mol_max_new_token: int = 90,
            cnmr_max_new_token: int = 60,
            hnmr_max_new_token: int = 12,
            hsqc_max_new_token: int = 12,
            sample_width: int = 64,
            eos_token_id: int = 1,
            ir_patch_size: int = 16,
            max_mol_prompt_length = 60,
            ir_length=512
    ) -> None:
        tokenized_spectra_dict = self.encode_spectra(spectra_dict)
        batch_size = len(tokenized_spectra_dict['[ir]'])
        tokenized_reaction_prompt = process_mol_prompts(tokenized_reaction_prompt,max_mol_prompt_length=max_mol_prompt_length,
                                                        eos_token=eos_token_id,
                                                        bos_token=0,
                                                        pad_token=2,)
        summary_scores = {
            'cnmr': torch.zeros(batch_size, sample_width, device=self.device),
            'hnmr': torch.zeros(batch_size, sample_width, device=self.device),
            'hsqc_c': torch.zeros(batch_size, sample_width, device=self.device),  # Split HSQC scores
            'hsqc_h': torch.zeros(batch_size, sample_width, device=self.device),
            'ir': torch.zeros(batch_size, sample_width, device=self.device),
            'mol_prob': torch.zeros(batch_size, sample_width, device=self.device)
        }
        summary_smiles = torch.zeros(batch_size, sample_width, mol_max_new_token, dtype=torch.long, device=self.device)
        summary_smiles_logits = torch.zeros(batch_size, sample_width, mol_max_new_token, dtype=torch.float, device=self.device)
        cnmr_pred_peaks = [[] for _ in range(batch_size)]
        hnmr_pred_peaks = [[] for _ in range(batch_size)]
        hsqc_pred_peaks = [[] for _ in range(batch_size)]
        ir_pred_specs = torch.zeros(batch_size, sample_width, ir_length, device=self.device)

        for i in range(sample_width):
            smile_mols_id, smile_logits = self.spec2mol_refine.refine_with_missing_modality_with_prob(
                tokenized_spectra_dict,
                mol_prompt = tokenized_reaction_prompt,
                max_new_tokens=mol_max_new_token, search=search, if_monte_carlo=True
            )
            summary_smiles_logits[:, i] = smile_logits
            batch_size, seq_len = smile_logits.shape
            smile_logits_sum = torch.zeros(batch_size, device=self.device)

            for b in range(batch_size):
                eos_positions = (smile_mols_id[b] == eos_token_id).nonzero(as_tuple=False)
                eos_pos = eos_positions[0, 0].item() + 1 if len(eos_positions) > 0 else seq_len
                smile_logits_sum[b] = smile_logits[b, 1:eos_pos].detach().log().mean()

            scores, cnmr_peaks, hnmr_peaks, hsqc_peaks, ir_specs = self.backward_infer(
                smile_mols_id=smile_mols_id,
                spectra_dict=spectra_dict,
                tokenized_spectra_dict=tokenized_spectra_dict,
                cnmr_max_new_token=cnmr_max_new_token,
                hnmr_max_new_token=hnmr_max_new_token,
                hsqc_max_new_token=hsqc_max_new_token,
                ir_patch_size=ir_patch_size
            )

            summary_scores['cnmr'][:, i] = scores['cnmr']
            summary_scores['hnmr'][:, i] = scores['hnmr']
            summary_scores['hsqc_c'][:, i] = scores['hsqc_c']  # Split HSQC scores
            summary_scores['hsqc_h'][:, i] = scores['hsqc_h']
            summary_scores['ir'][:, i] = scores['ir']
            summary_scores['mol_prob'][:, i] = smile_logits_sum
            summary_smiles[:, i] = smile_mols_id
            ir_pred_specs[:, i, :] = ir_specs

            for b in range(batch_size):
                cnmr_pred_peaks[b].extend(cnmr_peaks[b])
                hnmr_pred_peaks[b].extend(hnmr_peaks[b])
                hsqc_pred_peaks[b].extend(hsqc_peaks[b])

        self.pipeline_results.append({
            'stage': 'draft',
            'summary_scores': summary_scores,
            'summary_smiles': summary_smiles,
            'summary_smiles_logits': summary_smiles_logits,
            'cnmr_pred_peaks': cnmr_pred_peaks,
            'hnmr_pred_peaks': hnmr_pred_peaks,
            'hsqc_pred_peaks': hsqc_pred_peaks,
            'ir_pred_specs': ir_pred_specs
        })


    def select_molecules(
            self,
            select_ratio: float = 0.25,
            score_weights: Optional[Dict[str, float]] = None
    ) -> None:
        if score_weights is None:
            score_weights = {
                'cnmr': 0.05,
                'hnmr': 0.05,
                'hsqc_c': 0.025,  # Split HSQC weights
                'hsqc_h': 0.025,
                'ir': 0.25,
                'mol_prob': 1.25
            }

        latest_results = self.pipeline_results[-1]
        summary_scores = latest_results['summary_scores']
        summary_smiles = latest_results['summary_smiles']
        cnmr_scores = summary_scores['cnmr']
        hnmr_scores = summary_scores['hnmr']
        hsqc_c_scores = summary_scores['hsqc_c']  # Split HSQC scores
        hsqc_h_scores = summary_scores['hsqc_h']
        ir_scores = summary_scores['ir']
        mol_prob_scores = summary_scores['mol_prob']
        batch_size, sample_width = cnmr_scores.shape

        composite_scores = (
            score_weights['cnmr'] * cnmr_scores +
            score_weights['hnmr'] * hnmr_scores +
            score_weights['hsqc_c'] * hsqc_c_scores +  # Split HSQC scores
            score_weights['hsqc_h'] * hsqc_h_scores +
            score_weights['ir'] * ir_scores +
            score_weights['mol_prob'] * mol_prob_scores
        )

        top_k = max(1, int(sample_width * select_ratio))
        _, top_k_indices = torch.topk(composite_scores, k=top_k, dim=1)
        selected_smiles = torch.stack([summary_smiles[b].index_select(0, top_k_indices[b]) for b in range(batch_size)])

        self.pipeline_results.append({
            'stage': latest_results['stage'] + '_select',
            'selected_smiles': selected_smiles,
            'top_k': top_k
        })

    def refine_molecules(
            self,
            spectra_dict: List[Dict[str, List[Dict]]],
            max_new_tokens: int = 90,
            search: str = 'hybrid',
            eos_token_id: int = 1,
            ir_patch_size: int = 16,
            max_mol_prompt_length: int = 90,
            ir_length=512
    ) -> None:
        selected_smiles = self.pipeline_results[-1].get('selected_smiles') if self.pipeline_results else None
        if not self.pipeline_results or selected_smiles is None:
            raise ValueError("No selected molecules available. Run select_molecules first.")

        tokenized_spectra_dict = self.encode_spectra(spectra_dict)
        selected_smiles = self.pipeline_results[-1]['selected_smiles']
        top_k = self.pipeline_results[-1]['top_k']
        batch_size = selected_smiles.shape[0]
        sample_width = self.pipeline_results[0]['summary_scores']['cnmr'].shape[1]

        refine_width = sample_width // top_k
        if refine_width * top_k != sample_width:
            raise ValueError(f"sample_width ({sample_width}) must be divisible by top_k ({top_k})")

        refined_scores = {
            'cnmr': torch.zeros(batch_size, sample_width, device=self.device),
            'hnmr': torch.zeros(batch_size, sample_width, device=self.device),
            'hsqc_c': torch.zeros(batch_size, sample_width, device=self.device),  # Split HSQC scores
            'hsqc_h': torch.zeros(batch_size, sample_width, device=self.device),
            'ir': torch.zeros(batch_size, sample_width, device=self.device),
            'mol_prob': torch.zeros(batch_size, sample_width, device=self.device)
        }
        refined_smiles = torch.zeros(batch_size, sample_width, max_new_tokens, dtype=torch.long, device=self.device)
        refined_smiles_logits = torch.zeros(batch_size, sample_width, max_new_tokens, dtype=torch.float, device=self.device)
        refined_cnmr_peaks = [[] for _ in range(batch_size)]
        refined_hnmr_peaks = [[] for _ in range(batch_size)]
        refined_hsqc_peaks = [[] for _ in range(batch_size)]
        refined_ir_specs = torch.zeros(batch_size, sample_width, ir_length, device=self.device)

        for k in range(top_k):
            prompt_ids = selected_smiles[:, k]
            prompt_ids = process_mol_prompts(prompt_ids, max_mol_prompt_length,eos_token=eos_token_id,pad_token=2,bos_token=0).to(self.device)

            for w in range(refine_width):
                refined_mol_ids, refined_logits = self.spec2mol_refine.refine_with_missing_modality_with_prob(
                    spectra_dict=tokenized_spectra_dict,
                    mol_prompt=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    search=search,
                    if_monte_carlo=True
                )

                batch_eos_pos = []
                for b in range(batch_size):
                    eos_positions = (refined_mol_ids[b] == eos_token_id).nonzero(as_tuple=False)
                    eos_pos = eos_positions[0, 0].item() + 1 if len(eos_positions) > 0 else refined_mol_ids.shape[1]
                    batch_eos_pos.append(eos_pos)
                    refined_mol_ids[b, eos_pos:] = 0
                    refined_logits[b, eos_pos:] = 0

                mol_prob = torch.zeros(batch_size, device=self.device)
                for b in range(batch_size):
                    eos_pos = batch_eos_pos[b]
                    mol_prob[b] = refined_logits[b, 1:eos_pos].detach().log().mean()

                scores, cnmr_peaks, hnmr_peaks, hsqc_peaks, ir_specs = self.backward_infer(
                    smile_mols_id=refined_mol_ids,
                    spectra_dict=spectra_dict,
                    tokenized_spectra_dict=tokenized_spectra_dict,
                    cnmr_max_new_token=66,
                    hnmr_max_new_token=22,
                    hsqc_max_new_token=66,
                    ir_patch_size=ir_patch_size
                )

                idx = k * refine_width + w
                refined_scores['cnmr'][:, idx] = scores['cnmr']
                refined_scores['hnmr'][:, idx] = scores['hnmr']
                refined_scores['hsqc_c'][:, idx] = scores['hsqc_c']  # Split HSQC scores
                refined_scores['hsqc_h'][:, idx] = scores['hsqc_h']
                refined_scores['ir'][:, idx] = scores['ir']
                refined_scores['mol_prob'][:, idx] = mol_prob
                refined_smiles[:, idx, :refined_mol_ids.shape[1]] = refined_mol_ids
                refined_smiles_logits[:, idx, :refined_logits.shape[1]] = refined_logits
                refined_ir_specs[:, idx] = ir_specs

                for b in range(batch_size):
                    refined_cnmr_peaks[b].extend(cnmr_peaks[b])
                    refined_hnmr_peaks[b].extend(hnmr_peaks[b])
                    refined_hsqc_peaks[b].extend(hsqc_peaks[b])

        cycle_number = len([r for r in self.pipeline_results if r['stage'].startswith('cycle')]) // 2 + 1
        self.pipeline_results.append({
            'stage': f'cycle_{cycle_number}_refine',
            'summary_scores': refined_scores,
            'summary_smiles': refined_smiles,
            'summary_smiles_logits': refined_smiles_logits,
            'cnmr_pred_peaks': refined_cnmr_peaks,
            'hnmr_pred_peaks': refined_hnmr_peaks,
            'hsqc_pred_peaks': refined_hsqc_peaks,
            'ir_pred_specs': refined_ir_specs
        })

    def clear_results(self) -> None:
        self.pipeline_results = []
        torch.cuda.empty_cache()

    def run_pipeline_from_formula(
            self,
            formula_prompt: List[str],
            spectra_dict: List[Dict[str, List[Dict]]],
            select_ratio: float = 0.25,
            score_weights: Optional[Dict[str, float]] = None,
            draft_params: Optional[Dict] = None,
            refine_params: Optional[Dict] = None,
            num_refine_select_cycles: int = 1
    ) -> List[Dict]:
        if draft_params is None:
            draft_params = {}
        if refine_params is None:
            refine_params = {}

        # 检查精炼模型可用性与 num_refine_select_cycles 的兼容性
        if not self.enable_refinement and num_refine_select_cycles > 0:
            raise ValueError(
                "Refinement model is not available (spec2mol_dir_refine, spec2mol_agent_path_refine, or "
                "spec2mol_config_path_refine is None). Set num_refine_select_cycles to 0 to skip refinement."
            )
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            self.clear_results()
            print("Running draft inference")
            self.draft_infer_from_Formula(formula_prompt,spectra_dict, **draft_params)
            print("Running draft selection")
            self.select_molecules(select_ratio, score_weights)

            if self.enable_refinement:
                for cycle in range(num_refine_select_cycles):
                    print(f"Running refine-select cycle {cycle + 1}/{num_refine_select_cycles}")
                    self.refine_molecules(spectra_dict, **refine_params)
                    self.select_molecules(select_ratio, score_weights)
                    torch.cuda.empty_cache()
            else:
                print("Skipping refinement cycles: refinement model is not available.")

            return self.pipeline_results

    def run_pipeline(
            self,

            spectra_dict: List[Dict[str, List[Dict]]],
            select_ratio: float = 0.25,
            score_weights: Optional[Dict[str, float]] = None,
            draft_params: Optional[Dict] = None,
            refine_params: Optional[Dict] = None,
            num_refine_select_cycles: int = 1
    ) -> List[Dict]:
        if draft_params is None:
            draft_params = {}
        if refine_params is None:
            refine_params = {}

        # 检查精炼模型可用性与 num_refine_select_cycles 的兼容性
        if not self.enable_refinement and num_refine_select_cycles > 0:
            raise ValueError(
                "Refinement model is not available (spec2mol_dir_refine, spec2mol_agent_path_refine, or "
                "spec2mol_config_path_refine is None). Set num_refine_select_cycles to 0 to skip refinement."
            )

        self.clear_results()
        print("Running draft inference")
        self.draft_infer(spectra_dict, **draft_params)
        print("Running draft selection")
        self.select_molecules(select_ratio, score_weights)

        if self.enable_refinement:
            for cycle in range(num_refine_select_cycles):
                print(f"Running refine-select cycle {cycle + 1}/{num_refine_select_cycles}")
                self.refine_molecules(spectra_dict, **refine_params)
                self.select_molecules(select_ratio, score_weights)
                torch.cuda.empty_cache()
        else:
            print("Skipping refinement cycles: refinement model is not available.")

        return self.pipeline_results

    def run_pipeline_from_reaction(
            self,
            spectra_dict: List[Dict[str, List[Dict]]],
            tokenized_reaction_prompt,
            select_ratio: float = 0.25,
            score_weights: Optional[Dict[str, float]] = None,
            draft_params: Optional[Dict] = None,
            refine_params: Optional[Dict] = None,
            num_refine_select_cycles: int = 1
    ) -> List[Dict]:
        if draft_params is None:
            draft_params = {}
        if refine_params is None:
            refine_params = {}

        # 检查精炼模型可用性与 num_refine_select_cycles 的兼容性
        if not self.enable_refinement and num_refine_select_cycles > 0:
            raise ValueError(
                "Refinement model is not available (spec2mol_dir_refine, spec2mol_agent_path_refine, or "
                "spec2mol_config_path_refine is None). Set num_refine_select_cycles to 0 to skip refinement."
            )

        self.clear_results()
        print("Running draft inference")
        self.draft_infer_from_reaction(spectra_dict,tokenized_reaction_prompt=tokenized_reaction_prompt,
                                       **draft_params)
        print("Running draft selection")
        self.select_molecules(select_ratio, score_weights)

        if self.enable_refinement:
            for cycle in range(num_refine_select_cycles):
                print(f"Running refine-select cycle {cycle + 1}/{num_refine_select_cycles}")
                self.refine_molecules(spectra_dict, **refine_params)
                self.select_molecules(select_ratio, score_weights)
                torch.cuda.empty_cache()
        else:
            print("Skipping refinement cycles: refinement model is not available.")

        return self.pipeline_results
if __name__ == "__main__":
    spectra_dict = [
        {
            'c_nmr_peaks': [{'shift': 10.0, 'intensity': 1.0}],
            'h_nmr_peaks': [{'shift': 1.0, 'intensity': 1.0}],
            'hsqc': [{'c_shift': 10.0, 'h_shift': 1.0}],
            'ir': torch.rand(1, 512)
        }
    ]

    pipeline = MoleculeInferencePipeline(
        spec2mol_dir='SPEC2Mol',
        spec2mol_dir_refine=None,
        spec2mol_agent_path='all_unfroz_[hsqcnmr_cnmr_hnmr_ir]_5',
        spec2mol_config_path='config_[all_unfroz]_[cnmr_hnmr]_4.yaml',
        spec2mol_agent_path_refine='all_unfroz_[hsqcnmr_cnmr_hnmr_ir]_5',
        spec2mol_config_path_refine='config_[all_unfroz]_[hsqcnmr_cnmr_hnmr_ir]_5.yaml',
        hnmr_output_dir=None,
        hnmr_agent_path='all_unfroz_NPPE_hnmr',
        hnmr_config_path='config_[all_unfroz]_NoPromptPE_hnmr.yaml',
        cnmr_output_dir=None,
        cnmr_agent_path='all_unfroz_NPPE_cnmr',
        cnmr_config_path='config_[all_unfroz]_NoPromptPE_cnmr.yaml',
        hsqc_output_dir=None,
        hsqc_agent_path='all_unfroz_NPPE_hsqc',
        hsqc_config_path='config_[all_unfroz]_NoPromptPE_hsqc.yaml',
        ir_output_dir=None,
        ir_config_path='config_[all_unfroz]_NoPromptPE.yaml',
        device='cuda:3'
    )

    pipeline_results = pipeline.run_pipeline(
        spectra_dict,
        select_ratio=0.25,
        score_weights={
            'cnmr': 0.2,
            'hnmr': 0.2,
            'hsqc_c': 0.15,  # Split HSQC weights
            'hsqc_h': 0.15,
            'ir': 0.2,
            'mol_prob': 0.1
        },
        draft_params={
            'search': 'hybrid',
            'mol_max_new_token': 90,
            'cnmr_max_new_token': 66,
            'hnmr_max_new_token': 22,
            'hsqc_max_new_token': 66,
            'sample_width': 64
        },
        refine_params={
            'max_new_tokens': 90,
            'search': 'hybrid'
        },
        num_refine_select_cycles=3
    )

    for result in pipeline_results:
        print(f"Stage: {result['stage']}")
        if 'summary_scores' in result:
            print(f"  Scores: {result['summary_scores']}")
        if 'selected_smiles' in result:
            print(f"  Selected SMILES shape: {result['selected_smiles'].shape}")
            print(f"  Top K: {result['top_k']}")

