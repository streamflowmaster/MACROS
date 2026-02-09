# from TRL_GRPO_NMR.GRPO_Trainer import MultiModalGRPOTrainer
from rdkit import Chem,RDLogger
from rdkit.Chem import AllChem,rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np
from scipy.optimize import linear_sum_assignment
RDLogger.DisableLog('rdApp.*')


import torch
from typing import List, Dict, Union

def restructure_data(data: List[Union[List[Dict[str, float]], Dict[str, torch.Tensor]]]) -> List[Dict[str, torch.Tensor]]:
    """
    将数据从 [[{'delta (ppm)': float, 'width': float}, ...], ...]
    转换为 [{'delta (ppm)': 1d-tensor, 'width': 1d-tensor}, ...]。
    如果数据已经重构好，则直接返回。

    Args:
        data: 输入数据，可能是：
              - List[List[Dict[str, float]]]: 原始格式，形状为 (batch_size, peak_num, modality)
              - List[Dict[str, torch.Tensor]]: 已重构格式，batch_size 个字典，每个字典包含 modality 键和 1D 张量值

    Returns:
        List[Dict[str, torch.Tensor]]: 重构后的数据，外层为 batch_size，
                                      每项为 dict，键为 modality，值为 (peak_num,) 的 1D 张量
    """
    # 检查数据是否已经重构好
    if data and isinstance(data, list) and all(isinstance(item, dict) for item in data):
        all_values_are_tensors = True
        for item in data:
            if not item:  # 空字典允许
                continue
            for value in item.values():
                if not isinstance(value, torch.Tensor) or value.ndim != 1:
                    all_values_are_tensors = False
                    break
            if not all_values_are_tensors:
                break
        if all_values_are_tensors:
            print("Data is already restructured. Returning as is.")
            return data

    # 原始重构逻辑
    batch_size = len(data)
    result = []
    for i in range(batch_size):
        peak_num = len(data[i])
        if peak_num == 0:
            result.append({})
            print('empty prediction')
        else:
            modalities = data[i][0].keys()  # 提取 modality 名称

            batch_dict = {}
            for modality in modalities:
                if modality not in  ["category",'category','j_values'] :
                    # 提取当前 batch 的 modality 值，构造 1D 列表
                    values = [data[i][j][modality] for j in range(peak_num)]
                    # 转换为 1D tensor
                    batch_dict[modality] = torch.tensor(values)
            result.append(batch_dict)

    return result

def data_collator(features):
    batch = {
        "prompt": [],
        "completion": []
    }
    for feature in features:
        # Prompt: SMILES string
        batch["prompt"].append(feature["prompt"])

        # Completion: CNMR data
        completion = feature["completion"]
        batch["completion"].append(completion)

    # Convert completion list to tensor
    batch["completion"] = [x for x in batch["completion"] if x is not None]
    if batch["completion"]:
        batch["completion"] = torch.stack(batch["completion"])

    return batch


def calculate_nmr_similarity(predicted_peaks, true_peaks, predict_intensity, true_intensity, similarity_threshold=0.1):
    """
    Calculate matching rate, negative log peak MSE loss, and negative log intensity MSE loss
    between predicted and true NMR spectral peaks. Allows partial intensity data and unmatched peaks.

    Parameters:
    - predicted_peaks (list): List of predicted chemical shift values (e.g., [1.2, 2.3, 3.4])
    - true_peaks (list): List of true chemical shift values (e.g., [1.1, 2.4, 3.5])
    - predict_intensity (list): List of predicted intensity values corresponding to predicted_peaks
    - true_intensity (list): List of true intensity values corresponding to true_peaks
    - similarity_threshold (float): Not used in MSE calculation but kept for compatibility

    Returns:
    - matching_rate (float): Fraction of peaks successfully matched
    - neg_log_peak_mse (float): Negative log of mean squared error of matched peak differences
    - neg_log_intensity_mse (float): Negative log of mean squared error loss between matched intensities
    """
    predicted_peaks = np.array(predicted_peaks)
    true_peaks = np.array(true_peaks)
    predict_intensity = np.array(predict_intensity)
    true_intensity = np.array(true_intensity)

    n_pred = len(predicted_peaks)
    n_true = len(true_peaks)

    if n_pred == 0 or n_true == 0:
        return 0.0, 0.0, 0.0

    # Calculate cost matrix for peak matching
    cost_matrix = np.abs(predicted_peaks[:, np.newaxis] - true_peaks[np.newaxis, :])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Calculate matching rate
    n_matches = len(row_ind)
    max_possible_matches = min(n_pred, n_true)
    matching_rate = n_matches / max_possible_matches if max_possible_matches > 0 else 0.0

    # Calculate peak MSE and its negative log
    matched_differences = cost_matrix[row_ind, col_ind]
    peak_mse = np.mean(matched_differences ** 2) if n_matches > 0 else 0.0
    neg_log_peak_mse = -np.log(max(peak_mse, 1e-10)) if peak_mse > 0 else 0.0

    # Calculate intensity MSE loss for matched peaks with valid intensity data
    valid_intensity_pairs = []
    for i, j in zip(row_ind, col_ind):
        if i < len(predict_intensity) and j < len(true_intensity):
            valid_intensity_pairs.append((predict_intensity[i], true_intensity[j]))

    neg_log_intensity_mse = 0.0
    if valid_intensity_pairs:
        pred_intensities, true_intensities = zip(*valid_intensity_pairs)
        intensity_mse = np.mean((np.array(pred_intensities) - np.array(true_intensities)) ** 2)
        neg_log_intensity_mse = -np.log(max(intensity_mse, 1e-10)) if intensity_mse > 0 else 0.0

    return matching_rate, neg_log_peak_mse, neg_log_intensity_mse


def reward_hnmr_similarity(completions, prompts=None, gt_completions=None, tokenizer=None, **kwargs):
    """
    Reward function for CNMR predictions using NMR peak similarity.

    Parameters:
    - completions: List of predicted CNMR tensors (shape: [4 * c_nmr_max_num_peak])
    - prompts: List of SMILES prompts (unused here)
    - gt_completions: List of ground-truth CNMR tensors
    - tokenizer: MolTranBertTokenizer (unused here)
    - dataset: GRPODataset instance to access c_nmr_vaccume_token_idx and delta_discrete_reverse
    - kwargs: Additional arguments (e.g., num_generations)

    Returns:
    - rewards: List of reward values for each completion
    """
    # print(restructure_data(completions),restructure_data(gt_completions))
    completions = restructure_data(completions)
    gt_completions = restructure_data(gt_completions)
    if gt_completions is None:
        raise ValueError("gt_completions must be provided")


    rewards = []
    num_generations = len(completions) // len(gt_completions)
    similarity_threshold = 2.0
    w_matching = 10
    w_chemical_shift = 2
    w_intensity = 2
    for i, completion in enumerate(completions):
        try:
            gt_idx = i // num_generations
            gt_completion = gt_completions[gt_idx]

            if completion is None or gt_completion is None:
                rewards.append(-1.0)
                continue
            # 'centroid','category','nH','j_values'
            pred_delta = completion['centroid']
            gt_delta = gt_completion['centroid']

            pred_intensity = completion['nH']
            gt_intensity = gt_completion['nH']

            matching_rate, chemical_shift_similarity, intensity_similarity = calculate_nmr_similarity(predicted_peaks=pred_delta,
                                                                      predict_intensity=pred_intensity,
                                                                      true_peaks=gt_delta,
                                                                      true_intensity=gt_intensity)
            reward = w_matching * matching_rate + w_chemical_shift * chemical_shift_similarity + w_intensity*intensity_similarity


        except Exception as e:
            print(f"Error in reward calculation: {e}")
            reward = -1.0

        rewards.append(reward)

    return rewards

def reward_cnmr_similarity(completions, prompts=None, gt_completions=None, tokenizer=None, **kwargs):
    """
    Reward function for CNMR predictions using NMR peak similarity.

    Parameters:
    - completions: List of predicted CNMR tensors (shape: [4 * c_nmr_max_num_peak])
    - prompts: List of SMILES prompts (unused here)
    - gt_completions: List of ground-truth CNMR tensors
    - tokenizer: MolTranBertTokenizer (unused here)
    - dataset: GRPODataset instance to access c_nmr_vaccume_token_idx and delta_discrete_reverse
    - kwargs: Additional arguments (e.g., num_generations)

    Returns:
    - rewards: List of reward values for each completion
    """
    # print(restructure_data(completions),restructure_data(gt_completions))
    completions = restructure_data(completions)
    gt_completions = restructure_data(gt_completions)
    if gt_completions is None:
        raise ValueError("gt_completions must be provided")


    rewards = []
    num_generations = len(completions) // len(gt_completions)
    similarity_threshold = 2.0
    w_matching = 10
    w_chemical_shift = 2
    w_intensity = 2
    for i, completion in enumerate(completions):
        try:
            gt_idx = i // num_generations
            gt_completion = gt_completions[gt_idx]

            if completion is None or gt_completion is None:
                rewards.append(-1.0)
                continue

            pred_delta = completion['delta (ppm)']
            gt_delta = gt_completion['delta (ppm)']

            pred_intensity = completion['intensity']
            gt_intensity = gt_completion['intensity']

            matching_rate, chemical_shift_similarity, intensity_similarity = calculate_nmr_similarity(predicted_peaks=pred_delta,
                                                                      predict_intensity=pred_intensity,
                                                                      true_peaks=gt_delta,
                                                                      true_intensity=gt_intensity)
            reward = w_matching * matching_rate + w_chemical_shift * chemical_shift_similarity + w_intensity*intensity_similarity


        except Exception as e:
            print(f"Error in reward calculation: {e}")
            reward = -1.0

        rewards.append(reward)

    return rewards


def reward_ir_similarity(batch_gt_ir: torch.Tensor, batch_pred_ir: torch.Tensor) -> List[float]:
    """
    计算真实和预测 IR 光谱的皮尔逊相关系数。
    batch_gt_ir: 真实 IR 光谱，形状 [batch_size, ir_length]
    batch_pred_ir: 预测 IR 光谱，形状 [batch_size, ir_length]
    返回: 每个样本的皮尔逊相关系数列表
    """
    assert batch_gt_ir.shape == batch_pred_ir.shape, "Input tensors must have the same shape"
    batch_size = batch_gt_ir.shape[0]
    batch_gt_ir = batch_gt_ir.reshape(batch_size,-1)
    batch_pred_ir = batch_pred_ir.reshape(batch_size,-1)
    scores = []

    for i in range(batch_size):
        x = batch_gt_ir[i]  # [ir_length]
        y = batch_pred_ir[i]  # [ir_length]

        # 计算皮尔逊相关系数
        x_mean = x.mean()
        y_mean = y.mean()
        x_centered = x - x_mean
        y_centered = y - y_mean
        covariance = (x_centered * y_centered).sum()
        x_std = torch.sqrt((x_centered ** 2).sum())
        y_std = torch.sqrt((y_centered ** 2).sum())
        pearson = covariance / (x_std * y_std + 1e-8)  # 避免除零
        scores.append(pearson.item())

    return scores

def reward_nmr_similarity(task='cnmr', **kwargs,):
    if task.lower() == 'hnmr':
        return reward_hnmr_similarity
    elif task.lower() == 'cnmr':
        return reward_cnmr_similarity

# 奖励函数
def reward_validity_and_length(completions, prompts=None, gt_completions=None, tokenizer=None, **kwargs):
    rewards = []
    target_length = 20
    w_length = 0
    w_similarity = 4.0
    w_equivalent = 5.0
    if gt_completions is None:
        raise ValueError("gt_completions must be provided")

    num_generations = len(completions) // len(gt_completions)

    # Initialize MorganGenerator
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    gen_mols = []
    gen_fps = []
    gt_mols = []
    gt_fps = []

    # Process generated SMILES
    for completion in completions:
        try:
            mol = Chem.MolFromSmiles(completion) if completion else None
            gen_mols.append(mol)
            if mol is not None:
                fp = morgan_gen.GetFingerprint(mol)  # Use MorganGenerator
                gen_fps.append(fp)
            else:
                gen_fps.append(None)
        except:
            gen_mols.append(None)
            gen_fps.append(None)

    # Process ground-truth SMILES
    for gt in gt_completions:
        try:
            mol = Chem.MolFromSmiles(gt)
            gt_mols.append(mol)
            if mol is not None:
                fp = morgan_gen.GetFingerprint(mol)  # Use MorganGenerator
                gt_fps.append(fp)
            else:
                gt_fps.append(None)
        except:
            gt_mols.append(None)
            gt_fps.append(None)

    # Calculate rewards
    for i, (mol, fp, completion) in enumerate(zip(gen_mols, gen_fps, completions)):
        try:
            validity_reward = 1.0 if mol is not None else -1.0
            # length_reward = -abs(target_length - len(completion)) / target_length if completion else -1.0
            gt_idx = i // num_generations
            gt_fp = gt_fps[gt_idx]
            # Similarity reward
            if fp is not None and gt_fp is not None:
                similarity_reward = DataStructs.TanimotoSimilarity(fp, gt_fp)
            else:
                similarity_reward = 0.0
            # Equivalence reward: check if canonical SMILES are identical
            equivalence_reward = 1.0 if gen_canon is not None and gt_canon is not None and gen_canon == gt_canon else 0.0

            reward = validity_reward + w_similarity * similarity_reward + w_equivalent * equivalence_reward
        except:
            reward = -1.0
        rewards.append(reward)

    return rewards